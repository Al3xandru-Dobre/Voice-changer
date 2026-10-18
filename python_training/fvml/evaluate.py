"""Score converted audio the same way every time (eval_vc.py's engine).

The only inputs are WAV/FLAC files plus the frozen lists, so C++ output
(fv_vc_file, live recordings) and PyTorch output are scored identically.

Metrics (research §7):
  wer, cer   Whisper large-v3 on the output vs. the list transcript (or vs.
             Whisper on the source with ref_text="source"), Whisper's
             normaliser on both sides, corpus level
  tgt_cos    ECAPA cosine of each output to the target voice's reference centroid
  src_cos    ECAPA cosine of each output to its own source utterance (leakage)
  utmos      UTMOS22-strong
  f0_pcc     log-F0 Pearson correlation, source vs. output, frames voiced in
             both, after removing the output's delay (envelope cross-correlation)
  rtf        sum(compute seconds) / sum(output seconds), when timing is given

Baseline mode (no converted dir) scores the source audio itself and adds the
same- vs. different-speaker cosine check.
"""
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata

import numpy as np

from . import asr, audio, evallist, f0_ref, mos, paths, speaker

METRICS = ("wer", "spk", "utmos", "f0")
SUMMARY_KEYS = ("wer", "cer", "tgt_cos", "src_cos", "utmos", "f0_pcc", "rtf")


@dataclass
class EvalConfig:
    sources: object                 # eval list (test_clean_200.tsv)
    converted: object = None        # dir of <utt_id>.wav|.flac; None = baseline
    targets: object = None          # targets.tsv (or a private list)
    voice: str = None               # target voice name in `targets`
    ref_text: str = "list"          # "list" | "source"
    metrics: tuple = METRICS
    device: str = "cuda"
    batch_size: int = 8
    max_lag_ms: float = 200.0
    timing: object = None           # CSV utt_id,seconds; default <converted>/timing.csv
    limit: int = None               # first N rows (smoke runs)


def find_output(directory, utt_id):
    for ext in (".wav", ".flac"):
        p = paths.resolve(directory) / f"{utt_id}{ext}"
        if p.is_file():
            return p
    return None


def read_timing(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["utt_id"]: float(r["seconds"]) for r in csv.DictReader(f)}


def evaluate(cfg, out_dir, log=print):
    paths.configure_model_caches()
    t_start = time.monotonic()
    stage_s = {}
    rows = evallist.read_sources(cfg.sources)[: cfg.limit or None]
    baseline = cfg.converted is None
    log(f"[eval] {len(rows)} utterances from {cfg.sources} "
        f"({'baseline: scoring the sources' if baseline else f'converted: {cfg.converted}'})")

    t = time.monotonic()
    src_paths = [paths.resolve(r["path"]) for r in rows]
    if baseline:
        out_paths = src_paths
    else:
        out_paths = [find_output(cfg.converted, r["utt_id"]) for r in rows]
        missing = [r["utt_id"] for r, p in zip(rows, out_paths) if p is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} converted file(s) missing in {cfg.converted}, "
                                    f"e.g. {missing[0]}.wav")
    outs = [audio.load(p) for p in out_paths]
    need_src = not baseline and ({"spk", "f0"} & set(cfg.metrics) or cfg.ref_text == "source")
    srcs = [audio.load(p) for p in src_paths] if (need_src or baseline) else None
    stage_s["load"] = time.monotonic() - t

    per = [{"utt_id": r["utt_id"], "speaker": r["speaker"], "duration_out": len(o) / audio.SAMPLE_RATE}
           for r, o in zip(rows, outs)]
    summary = {k: None for k in SUMMARY_KEYS}
    extra = {}

    if "wer" in cfg.metrics:
        t = time.monotonic()
        w = asr.Whisper(cfg.device, cfg.batch_size)
        hyps = w.transcribe(outs)
        if cfg.ref_text == "source":
            refs_raw = w.transcribe(srcs) if not baseline else hyps
        else:
            refs_raw = [r["text"] for r in rows]
        refs = [w.normalize(x) for x in refs_raw]
        hyps_n = [w.normalize(x) for x in hyps]
        rates = asr.error_rates(refs, hyps_n)
        summary.update(wer=rates["wer"], cer=rates["cer"])
        extra["wer_skipped_empty_refs"] = rates["skipped"]
        for p, ref, hyp, pw, pc in zip(per, refs, hyps_n, rates["per_wer"], rates["per_cer"]):
            p.update(ref=ref, hyp=hyp, wer=pw, cer=pc)
        w.close()
        stage_s["wer"] = time.monotonic() - t

    if "spk" in cfg.metrics:
        t = time.monotonic()
        e = speaker.Ecapa(cfg.device, cfg.batch_size)
        emb_out = e.embed(outs)
        if baseline:
            extra["speaker_pairs"] = speaker.pair_stats(emb_out, [r["speaker"] for r in rows])
            extra["speaker_pairs"]["gap"] = (extra["speaker_pairs"]["same_spk_cos"]
                                             - extra["speaker_pairs"]["diff_spk_cos"])
        else:
            emb_src = e.embed(srcs)
            src_cos = np.sum(emb_out * emb_src, axis=1)
            summary["src_cos"] = float(src_cos.mean())
            for p, c in zip(per, src_cos):
                p["src_cos"] = float(c)
        if cfg.targets and cfg.voice:
            refs = evallist.read_targets(cfg.targets, cfg.voice)
            if not refs:
                raise ValueError(f"no reference clips for voice '{cfg.voice}' in {cfg.targets}")
            target = speaker.centroid(e.embed([audio.load(paths.resolve(r["path"])) for r in refs]))
            tgt_cos = emb_out @ target
            summary["tgt_cos"] = float(tgt_cos.mean())
            extra["target_refs"] = len(refs)
            for p, c in zip(per, tgt_cos):
                p["tgt_cos"] = float(c)
        e.close()
        stage_s["spk"] = time.monotonic() - t

    if "utmos" in cfg.metrics:
        t = time.monotonic()
        u = mos.Utmos(cfg.device)
        scores = u.score(outs)
        summary["utmos"] = float(np.mean(scores))
        for p, s in zip(per, scores):
            p["utmos"] = s
        u.close()
        stage_s["utmos"] = time.monotonic() - t

    if "f0" in cfg.metrics:
        t = time.monotonic()
        pccs = []
        for i, p in enumerate(per):
            f_out = f0_ref.track(outs[i])
            p["f0_med_out"], p["lf0_std_out"] = f0_ref.stats(f_out)
            if not baseline:
                f_src = f0_ref.track(srcs[i])
                lag = f0_ref.best_lag(srcs[i], outs[i], max_lag_s=cfg.max_lag_ms / 1000) \
                    if cfg.max_lag_ms > 0 else 0
                p["lag_ms"] = lag * f0_ref.HOP_S * 1000
                p["f0_med_src"], p["lf0_std_src"] = f0_ref.stats(f_src)
                p["f0_pcc"] = f0_ref.pcc(f_src, f_out, lag)
                pccs.append(p["f0_pcc"])
        if pccs:
            summary["f0_pcc"] = float(np.nanmean(pccs))
        stage_s["f0"] = time.monotonic() - t

    timing_path = cfg.timing or (paths.resolve(cfg.converted) / "timing.csv"
                                 if cfg.converted and (paths.resolve(cfg.converted) / "timing.csv").is_file()
                                 else None)
    if timing_path:
        timing = read_timing(timing_path)
        secs = [timing.get(p["utt_id"]) for p in per]
        if all(s is not None for s in secs):
            summary["rtf"] = sum(secs) / sum(p["duration_out"] for p in per)
            for p, s in zip(per, secs):
                p["seconds"] = s

    stage_s["total"] = time.monotonic() - t_start
    result = {**summary, "n": len(rows), "mode": "baseline" if baseline else "converted",
              "voice": cfg.voice, **extra, "eval_seconds": {k: round(v, 1) for k, v in stage_s.items()},
              "meta": run_metadata(cfg)}
    write_outputs(out_dir, per, result)
    log(f"[eval] done in {stage_s['total']:.0f} s -> {out_dir}")
    return result


def run_metadata(cfg):
    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=paths.REPO_ROOT, timeout=10).stdout.strip() or None
    except OSError:
        commit = None
    import torch
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": commit, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if str(cfg.device).startswith("cuda") else str(cfg.device),
        "sources": str(cfg.sources), "sources_sha256": _sha256(cfg.sources),
        "targets": str(cfg.targets) if cfg.targets else None,
        "asr": f"{asr.MODEL}@{asr.REVISION[:12]} fp16, Whisper EnglishTextNormalizer (transformers)",
        "speaker": f"{speaker.SOURCE}@{speaker.REVISION[:12]} (ECAPA-TDNN)",
        "mos": mos.MODEL, "f0": f0_ref.TRACKER,
        "packages": {p: version(p) for p in ("torch", "transformers", "speechbrain", "speechmos",
                                             "praat-parselmouth", "jiwer", "numpy")},
    }


def _sha256(path):
    return hashlib.sha256(paths.resolve(path).read_bytes()).hexdigest()


def _fmt(v, digits=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def write_outputs(out_dir, per, result):
    out_dir = paths.resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["utt_id", "speaker", "duration_out", "ref", "hyp", "wer", "cer", "tgt_cos", "src_cos",
               "utmos", "f0_pcc", "lag_ms", "f0_med_src", "f0_med_out", "lf0_std_src", "lf0_std_out",
               "seconds"]
    columns = [c for c in columns if any(c in p for p in per)]
    with open(out_dir / "per_utt.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for p in per:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in p.items()})
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n",
                                          encoding="utf-8")
    pct = lambda v: None if v is None else 100 * v  # noqa: E731
    lines = [f"# eval_vc: {result['mode']}" + (f", voice `{result['voice']}`" if result["voice"] else ""),
             "",
             f"{result['n']} utterances from `{result['meta']['sources']}`, {result['meta']['time']}.",
             "",
             "| WER % | CER % | Tgt cos | Src cos | UTMOS | F0 PCC | RTF |",
             "|---|---|---|---|---|---|---|",
             f"| {_fmt(pct(result['wer']), 2)} | {_fmt(pct(result['cer']), 2)} | {_fmt(result['tgt_cos'])} "
             f"| {_fmt(result['src_cos'])} | {_fmt(result['utmos'], 2)} | {_fmt(result['f0_pcc'])} "
             f"| {_fmt(result['rtf'])} |"]
    sp = result.get("speaker_pairs")
    if sp:
        lines += ["", f"Speaker check: same-speaker cosine {sp['same_spk_cos']:.3f} "
                      f"({sp['same_pairs']} pairs), different-speaker {sp['diff_spk_cos']:.3f} "
                      f"({sp['diff_pairs']} pairs), gap {sp['gap']:.3f}, EER {100 * sp['spk_eer']:.2f} %."]
    lines += ["", "Models: " + "; ".join(result["meta"][k] for k in ("asr", "speaker", "mos", "f0")) + ".",
              f"Evaluation time: {result['eval_seconds']['total']:.0f} s "
              f"({', '.join(f'{k} {v:.0f} s' for k, v in result['eval_seconds'].items() if k != 'total')})."]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
