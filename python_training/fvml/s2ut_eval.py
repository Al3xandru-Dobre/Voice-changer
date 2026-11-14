"""Simultaneous S2UT evaluation engine (Milestone 3 / Stage 01: eval_s2ut.py).

Scores any directory of translated WAV/FLAC files (PyTorch gate_s2ut.py, Python
ORT, C++ offline fv_s2ut_file, or live fv_probe recordings) identically:

  asr_bleu, asr_chrfpp   Whisper large-v3 (forced `language=tgt_lang`,
                         `task="transcribe"`), language-aware normalisation,
                         SacreBLEU (`13a` and `chrF++` word_order=2)
  onset_delay_ms         first active output frame timestamp minus first active
                         source frame timestamp (waveform RMS > -55 dBFS)
  al_ms, al_ca_ms        SimulEval Average Lagging and Computation-Aware Average
                         Lagging (Ma et al., EMNLP 2020) from step emission
                         traces (`[{step, t_src_ms, t_tgt_frames_emitted, ...}]`)
  tgt_cos, src_cos       ECAPA-TDNN cosine to target voice references and to
                         source utterance (leakage)
  utmos                  UTMOS22-strong
  trailing_non_silent_ms non-silent audio duration after expected utterance end
  hallucinated_words     Whisper words emitted on silence/noise-only clips (must
                         be 0)
"""
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import pathlib
import platform
import subprocess
import time

import numpy as np

from . import asr, audio, evaluate as vc_eval, evallist, mos, paths, speaker

S2UT_METRICS = ("bleu", "latency", "spk", "utmos", "flush")
SUMMARY_KEYS = (
    "asr_bleu", "asr_chrfpp", "onset_delay_ms", "al_ms", "al_ca_ms",
    "tgt_cos", "src_cos", "utmos", "trailing_non_silent_ms",
    "hallucinated_words", "noise_hallucination_rate", "rtf",
)
FRAME_MS = 20.0
DEFAULT_TAIL_ALLOWANCE_MS = 320.0   # <= 4 chunks (320 ms) flush tail tolerance


@dataclass
class S2utEvalConfig:
    eval_list: object
    converted: object = None        # dir of <utt_id>.wav|.flac; None with oracle=True
    oracle: bool = False            # score ground-truth tgt_path from the eval TSV
    trace: object = None            # combined trace.jsonl or dir of <utt_id>.trace.jsonl
    targets: object = None          # targets.tsv
    voice: str = None               # target voice name in `targets`
    metrics: tuple = S2UT_METRICS
    silence_dbfs: float = -55.0
    tail_allowance_ms: float = DEFAULT_TAIL_ALLOWANCE_MS
    device: str = "cuda"
    batch_size: int = 8
    timing: object = None           # CSV utt_id,seconds
    limit: int = None


# -- SimulEval Average Lagging (Ma et al., 2019 / 2020) -----------------------
def average_lagging_ms(src_duration_ms, steps):
    """Compute SimulEval speech Average Lagging (`al_ms`, `al_ca_ms`) for one
    utterance from its emission trace steps.

    Each entry in `steps` is a dict with:
      * `t_src_ms`: cumulative source audio read (ms) when this step ran
      * `t_tgt_frames_emitted`: integer >= 0 count of 20 ms target frames emitted
      * optional `t_wall_ms` (completion wall-clock ms since utterance start)
        or `elapsed_ms` (compute time of this step in ms, accumulated via the
        SimulEval real-time streaming clock `wall = max(wall, t_src_ms) + elapsed_ms`).

    Follows Ma et al. (EMNLP 2020, §2.2):
      gamma = |Y| / T_src
      tau   = min { j in 1..|Y| : d_j >= T_src }  (or |Y| if none)
      AL    = (1 / tau) * sum_{j=1..tau} (d_j    - (j - 1) / gamma)
      AL_CA = (1 / tau) * sum_{j=1..tau} (d_j_ca - (j - 1) / gamma)
    Note: AL_CA uses the *exact same algorithmic cutoff tau* as AL so slow
    compute on early steps cannot truncate tau to 1.
    """
    if src_duration_ms <= 0 or not steps:
        return np.nan, np.nan

    delays = []
    delays_ca = []
    has_ca = False
    sim_wall_ms = 0.0

    ordered = sorted(steps, key=lambda s: (s.get("step", 0), float(s["t_src_ms"])))
    for s in ordered:
        t_src = float(s["t_src_ms"])
        n_emit = int(s.get("t_tgt_frames_emitted", 0))
        if "t_wall_ms" in s and s["t_wall_ms"] is not None:
            has_ca = True
            sim_wall_ms = float(s["t_wall_ms"])
        elif "elapsed_ms" in s and s["elapsed_ms"] is not None:
            has_ca = True
            sim_wall_ms = max(sim_wall_ms, t_src) + float(s["elapsed_ms"])
        else:
            sim_wall_ms = max(sim_wall_ms, t_src)
        if n_emit > 0:
            delays.extend([t_src] * n_emit)
            delays_ca.extend([sim_wall_ms] * n_emit)

    y_len = len(delays)
    if y_len == 0:
        return np.nan, np.nan

    gamma = y_len / float(src_duration_ms)
    tau = y_len
    for idx, d in enumerate(delays):
        if d >= src_duration_ms - 1e-6:
            tau = idx + 1   # 1-based cutoff index
            break

    ideal = np.arange(tau, dtype=np.float64) / gamma
    al = float(np.mean(np.asarray(delays[:tau], dtype=np.float64) - ideal))
    al_ca = float(np.mean(np.asarray(delays_ca[:tau], dtype=np.float64) - ideal)) \
        if has_ca else np.nan
    return al, al_ca


def waveform_onset_ms(wave, threshold_dbfs=-55.0):
    """Timestamp (ms) of the first 20 ms frame exceeding `threshold_dbfs`, or
    NaN when the entire clip is silent."""
    mask = asr.active_frames_mask(wave, frame_ms=FRAME_MS, threshold_dbfs=threshold_dbfs)
    idx = np.flatnonzero(mask)
    return float(idx[0] * FRAME_MS) if len(idx) else np.nan


def first_audio_onset_delay_ms(src_wave, out_wave, threshold_dbfs=-55.0):
    """Output speech onset timestamp minus source speech onset timestamp (ms)."""
    t_src = waveform_onset_ms(src_wave, threshold_dbfs=threshold_dbfs)
    t_out = waveform_onset_ms(out_wave, threshold_dbfs=threshold_dbfs)
    if math.isnan(t_src) or math.isnan(t_out):
        return np.nan
    return float(t_out - t_src)


def trailing_non_silent_duration_ms(out_wave, expected_end_s, threshold_dbfs=-55.0):
    """Duration (ms) of active non-silent frames occurring after `expected_end_s`."""
    mask = asr.active_frames_mask(out_wave, frame_ms=FRAME_MS, threshold_dbfs=threshold_dbfs)
    if len(mask) == 0:
        return 0.0
    start_frame = max(0, int(math.ceil(max(0.0, float(expected_end_s)) * 1000.0 / FRAME_MS)))
    if start_frame >= len(mask):
        return 0.0
    return float(np.sum(mask[start_frame:]) * FRAME_MS)


def load_traces(trace_arg, converted_dir, utt_ids):
    """Load per-utterance emission traces `{utt_id: [step_dict, ...]}` from:
      * a single JSONL file (either `{utt_id, steps: [...]}` per line or one step
        per line carrying `utt_id`),
      * a directory containing `<utt_id>.trace.jsonl` (or `<utt_id>.jsonl`), or
      * `<converted_dir>/trace.jsonl` / `<converted_dir>/<utt_id>.trace.jsonl`."""
    by_utt = defaultdict(list)

    def _read_jsonl_file(p, default_utt=None):
        for line in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and default_utt is not None:
                by_utt[default_utt].extend(obj)
            elif "steps" in obj:
                uid = obj.get("utt_id", default_utt)
                if uid:
                    by_utt[uid].extend(obj["steps"])
            else:
                uid = obj.get("utt_id", default_utt)
                if uid:
                    by_utt[uid].append(obj)

    if trace_arg is not None:
        tp = paths.resolve(trace_arg)
        if tp.is_file():
            _read_jsonl_file(tp)
            return dict(by_utt)
        if tp.is_dir():
            for uid in utt_ids:
                for suffix in (".trace.jsonl", ".jsonl"):
                    cand = tp / f"{uid}{suffix}"
                    if cand.is_file():
                        _read_jsonl_file(cand, default_utt=uid)
                        break
            return dict(by_utt)
        raise FileNotFoundError(f"trace path not found: {trace_arg}")

    if converted_dir is not None:
        cp = paths.resolve(converted_dir)
        combined = cp / "trace.jsonl"
        if combined.is_file():
            _read_jsonl_file(combined)
        else:
            for uid in utt_ids:
                cand = cp / f"{uid}.trace.jsonl"
                if cand.is_file():
                    _read_jsonl_file(cand, default_utt=uid)
    return dict(by_utt)


# -- Main evaluation driver ---------------------------------------------------
def evaluate_s2ut(cfg, out_dir, log=print):
    paths.configure_model_caches()
    t_start = time.monotonic()
    stage_s = {}

    rows = evallist.read_s2ut(cfg.eval_list)[: cfg.limit or None]
    if not rows:
        raise ValueError(f"no evaluation rows in {cfg.eval_list}")
    if not cfg.oracle and cfg.converted is None:
        raise ValueError("either converted directory or oracle=True is required")

    mode = "oracle" if cfg.oracle else "translated"
    log(f"[eval_s2ut] {len(rows)} utterances from {cfg.eval_list} ({mode})")

    t = time.monotonic()
    src_paths = [paths.resolve(r["src_path"]) for r in rows]
    if cfg.oracle:
        missing_tgt = [r["utt_id"] for r in rows if not r.get("tgt_path")]
        if missing_tgt:
            raise ValueError(f"oracle mode requires tgt_path on every row (missing on {missing_tgt[0]})")
        out_paths = [paths.resolve(r["tgt_path"]) for r in rows]
    else:
        out_paths = [vc_eval.find_output(cfg.converted, r["utt_id"]) for r in rows]
        missing = [r["utt_id"] for r, p in zip(rows, out_paths) if p is None]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} translated file(s) missing in {cfg.converted}, e.g. {missing[0]}.wav")

    outs = [audio.load(p) for p in out_paths]
    need_src = ({"latency", "spk"} & set(cfg.metrics)) and not cfg.oracle
    srcs = [audio.load(p) for p in src_paths] if need_src else None
    traces = load_traces(cfg.trace, cfg.converted, [r["utt_id"] for r in rows])
    stage_s["load"] = time.monotonic() - t

    per = [
        {
            "utt_id": r["utt_id"],
            "pair": f"{r['src_lang']}->{r['tgt_lang']}",
            "src_lang": r["src_lang"],
            "tgt_lang": r["tgt_lang"],
            "speaker": r["speaker"],
            "kind": r.get("kind", "paired"),
            "duration_src": float(r["src_duration"]),
            "duration_out": len(o) / audio.SAMPLE_RATE,
        }
        for r, o in zip(rows, outs)
    ]
    summary = {k: None for k in SUMMARY_KEYS}
    extra = {}

    need_whisper = bool({"bleu", "flush"} & set(cfg.metrics))
    if need_whisper:
        t = time.monotonic()
        w = asr.Whisper(cfg.device, cfg.batch_size)
        hyps_raw = [""] * len(rows)
        by_lang = defaultdict(list)
        for idx, r in enumerate(rows):
            by_lang[r["tgt_lang"]].append(idx)
        for tgt_lang, idxs in sorted(by_lang.items()):
            lang_waves = [outs[i] for i in idxs]
            lang_hyps = w.transcribe(lang_waves, language=tgt_lang, silence_dbfs=cfg.silence_dbfs)
            for i, hyp in zip(idxs, lang_hyps):
                hyps_raw[i] = hyp

        refs_norm = [w.normalize(r["tgt_text"], language=r["tgt_lang"]) for r in rows]
        hyps_norm = [w.normalize(h, language=r["tgt_lang"]) for h, r in zip(hyps_raw, rows)]
        w.close()
        stage_s["whisper"] = time.monotonic() - t

        if "bleu" in cfg.metrics:
            scores = asr.translation_scores(refs_norm, hyps_norm)
            if not math.isnan(scores["asr_bleu"]):
                summary["asr_bleu"] = scores["asr_bleu"]
                summary["asr_chrfpp"] = scores["asr_chrfpp"]
            extra["bleu_skipped_empty_refs"] = scores["skipped"]
            for p, ref, hyp, pb, pc in zip(
                per, refs_norm, hyps_norm, scores["per_bleu"], scores["per_chrfpp"]
            ):
                p.update(ref=ref, hyp=hyp, asr_bleu=pb, asr_chrfpp=pc)

            # Per-direction breakdown when multiple language pairs are present.
            by_pair = defaultdict(list)
            for idx, p in enumerate(per):
                if refs_norm[idx].strip():
                    by_pair[p["pair"]].append(idx)
            if len(by_pair) > 1:
                extra["per_pair"] = {}
                for pair, idxs in sorted(by_pair.items()):
                    ps = asr.translation_scores(
                        [refs_norm[i] for i in idxs], [hyps_norm[i] for i in idxs]
                    )
                    extra["per_pair"][pair] = {
                        "n": len(idxs),
                        "asr_bleu": ps["asr_bleu"],
                        "asr_chrfpp": ps["asr_chrfpp"],
                    }

        if "flush" in cfg.metrics:
            noise_words = 0
            noise_clips = 0
            noise_hallucinated_clips = 0
            for idx, (r, p) in enumerate(zip(rows, per)):
                words = len(hyps_norm[idx].split()) if hyps_norm[idx] else 0
                is_noise = r.get("kind") == "noise_only" or not r["tgt_text"].strip()
                p["hyp_words"] = words
                if is_noise:
                    noise_clips += 1
                    noise_words += words
                    if words > 0:
                        noise_hallucinated_clips += 1
                    p["hallucinated_words"] = words
            if noise_clips > 0:
                summary["hallucinated_words"] = int(noise_words)
                summary["noise_hallucination_rate"] = noise_hallucinated_clips / noise_clips
                extra["noise_clips"] = noise_clips

    if "flush" in cfg.metrics:
        tails = []
        for r, o, p in zip(rows, outs, per):
            if r.get("kind") == "noise_only":
                exp_end = 0.0
            elif "speech_end_s" in r and r["speech_end_s"] < r["src_duration"]:
                exp_end = float(r["speech_end_s"]) + cfg.tail_allowance_ms / 1000.0
            else:
                exp_end = float(r.get("tgt_duration") or r["src_duration"]) + cfg.tail_allowance_ms / 1000.0
            tail_ms = trailing_non_silent_duration_ms(o, exp_end, threshold_dbfs=cfg.silence_dbfs)
            p["trailing_non_silent_ms"] = tail_ms
            tails.append(tail_ms)
        summary["trailing_non_silent_ms"] = float(np.mean(tails)) if tails else None

    if "latency" in cfg.metrics and not cfg.oracle:
        t = time.monotonic()
        onsets, als, al_cas = [], [], []
        for idx, (r, o, p) in enumerate(zip(rows, outs, per)):
            if r.get("kind") == "noise_only":
                continue
            onset = first_audio_onset_delay_ms(srcs[idx], o, threshold_dbfs=cfg.silence_dbfs)
            p["onset_delay_ms"] = onset
            if not math.isnan(onset):
                onsets.append(onset)
            utt_trace = traces.get(r["utt_id"])
            if utt_trace:
                src_ms = float(r.get("speech_end_s", r["src_duration"])) * 1000.0
                al, al_ca = average_lagging_ms(src_ms, utt_trace)
                p["al_ms"] = al
                p["al_ca_ms"] = al_ca
                if not math.isnan(al):
                    als.append(al)
                if not math.isnan(al_ca):
                    al_cas.append(al_ca)
        if onsets:
            summary["onset_delay_ms"] = float(np.mean(onsets))
        if als:
            summary["al_ms"] = float(np.mean(als))
        if al_cas:
            summary["al_ca_ms"] = float(np.mean(al_cas))
        stage_s["latency"] = time.monotonic() - t

    if "spk" in cfg.metrics:
        t = time.monotonic()
        non_silent_idx = [i for i, o in enumerate(outs) if not asr.is_silent(o, cfg.silence_dbfs)]
        if non_silent_idx:
            e = speaker.Ecapa(cfg.device, cfg.batch_size)
            emb_out = e.embed([outs[i] for i in non_silent_idx])
            if not cfg.oracle and srcs is not None:
                emb_src = e.embed([srcs[i] for i in non_silent_idx])
                src_cos = np.sum(emb_out * emb_src, axis=1)
                summary["src_cos"] = float(src_cos.mean())
                for i, c in zip(non_silent_idx, src_cos):
                    per[i]["src_cos"] = float(c)
            if cfg.targets and cfg.voice:
                refs = evallist.read_targets(cfg.targets, cfg.voice)
                if not refs:
                    raise ValueError(f"no reference clips for voice '{cfg.voice}' in {cfg.targets}")
                target = speaker.centroid(
                    e.embed([audio.load(paths.resolve(r["path"])) for r in refs]))
                tgt_cos = emb_out @ target
                summary["tgt_cos"] = float(tgt_cos.mean())
                extra["target_refs"] = len(refs)
                for i, c in zip(non_silent_idx, tgt_cos):
                    per[i]["tgt_cos"] = float(c)
            e.close()
        stage_s["spk"] = time.monotonic() - t

    if "utmos" in cfg.metrics:
        t = time.monotonic()
        non_silent_idx = [i for i, o in enumerate(outs)
                          if rows[i].get("kind") != "noise_only"
                          and not asr.is_silent(o, cfg.silence_dbfs)]
        if non_silent_idx:
            u = mos.Utmos(cfg.device)
            scores = u.score([outs[i] for i in non_silent_idx])
            summary["utmos"] = float(np.mean(scores))
            for i, s in zip(non_silent_idx, scores):
                per[i]["utmos"] = float(s)
            u.close()
        stage_s["utmos"] = time.monotonic() - t

    timing_path = cfg.timing or (
        paths.resolve(cfg.converted) / "timing.csv"
        if cfg.converted and (paths.resolve(cfg.converted) / "timing.csv").is_file()
        else None
    )
    if timing_path:
        timing = vc_eval.read_timing(timing_path)
        secs = [timing.get(p["utt_id"]) for p in per]
        total_dur = sum(p["duration_src"] for p in per)
        if all(s is not None for s in secs) and total_dur > 0:
            summary["rtf"] = sum(secs) / total_dur
            for p, s in zip(per, secs):
                p["seconds"] = s

    stage_s["total"] = time.monotonic() - t_start
    result = {
        **summary,
        "n": len(rows),
        "mode": mode,
        "voice": cfg.voice,
        **extra,
        "eval_seconds": {k: round(v, 1) for k, v in stage_s.items()},
        "meta": _run_metadata(cfg),
    }
    _write_outputs(out_dir, per, result)
    log(f"[eval_s2ut] done in {stage_s['total']:.1f} s -> {out_dir}")
    return result


def _run_metadata(cfg):
    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=paths.REPO_ROOT, timeout=10,
        ).stdout.strip() or None
    except OSError:
        commit = None
    try:
        import torch
        dev = (torch.cuda.get_device_name(0)
               if str(cfg.device).startswith("cuda") and torch.cuda.is_available()
               else str(cfg.device))
    except ImportError:
        dev = str(cfg.device)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": commit,
        "python": platform.python_version(),
        "device": dev,
        "eval_list": str(cfg.eval_list),
        "eval_list_sha256": hashlib.sha256(paths.resolve(cfg.eval_list).read_bytes()).hexdigest(),
        "targets": str(cfg.targets) if cfg.targets else None,
        "asr": f"{asr.MODEL}@{asr.REVISION[:12]} (forced tgt_lang, SacreBLEU 13a + chrF++)",
        "speaker": f"{speaker.SOURCE}@{speaker.REVISION[:12]} (ECAPA-TDNN)",
        "mos": mos.MODEL,
        "packages": {p: version(p) for p in (
            "torch", "transformers", "sacrebleu", "speechbrain", "speechmos", "numpy"
        )},
    }


def _fmt(v, digits=2):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def _write_outputs(out_dir, per, result):
    out_dir = paths.resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "utt_id", "pair", "src_lang", "tgt_lang", "speaker", "kind",
        "duration_src", "duration_out", "ref", "hyp", "asr_bleu", "asr_chrfpp",
        "onset_delay_ms", "al_ms", "al_ca_ms", "tgt_cos", "src_cos", "utmos",
        "trailing_non_silent_ms", "hyp_words", "hallucinated_words", "seconds",
    ]
    columns = [c for c in columns if any(c in p for p in per)]
    with open(out_dir / "per_utt.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for p in per:
            w.writerow({
                k: ("" if isinstance(v, float) and math.isnan(v)
                    else f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in p.items()
            })

    def _clean_json(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean_json(v) for v in obj]
        return obj

    (out_dir / "summary.json").write_text(
        json.dumps(_clean_json(result), indent=2, default=float) + "\n", encoding="utf-8"
    )

    lines = [
        f"# eval_s2ut: {result['mode']}" + (f", voice `{result['voice']}`" if result["voice"] else ""),
        "",
        f"{result['n']} utterances from `{result['meta']['eval_list']}`, {result['meta']['time']}.",
        "",
        "| ASR-BLEU | ASR-chrF++ | Onset (ms) | AL (ms) | AL_CA (ms) | Tgt cos | Src cos | UTMOS | Tail (ms) | Halluc words | RTF |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_fmt(result['asr_bleu'], 2)} | {_fmt(result['asr_chrfpp'], 2)} "
        f"| {_fmt(result['onset_delay_ms'], 1)} | {_fmt(result['al_ms'], 1)} "
        f"| {_fmt(result['al_ca_ms'], 1)} | {_fmt(result['tgt_cos'], 3)} "
        f"| {_fmt(result['src_cos'], 3)} | {_fmt(result['utmos'], 2)} "
        f"| {_fmt(result['trailing_non_silent_ms'], 1)} "
        f"| {'-' if result['hallucinated_words'] is None else result['hallucinated_words']} "
        f"| {_fmt(result['rtf'], 3)} |",
    ]
    if "per_pair" in result:
        lines += [
            "",
            "## Per-direction breakdown",
            "",
            "| Direction | N | ASR-BLEU | ASR-chrF++ |",
            "|---|---:|---:|---:|",
        ]
        for pair, ps in result["per_pair"].items():
            lines.append(
                f"| `{pair}` | {ps['n']} | {_fmt(ps['asr_bleu'], 2)} | {_fmt(ps['asr_chrfpp'], 2)} |"
            )
    lines += [
        "",
        f"Evaluation time: {result['eval_seconds']['total']:.1f} s.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
