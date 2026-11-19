#!/usr/bin/env python3
"""Check the multilingual live-mic recordings and write the Stage 11 S2UT lists.

Milestone 3 / Stage 10 (Documents/Milestone 3/10_cross_lingual_target_voice.md,
"live-mic multilingual source test set"), consumed by Stage 11
(11_verification_exit.md: "translate the 20-prompt per-language real-mic
recordings and score ASR-BLEU and UTMOS").

Prompts: eval_lists/s2ut/live_mic_prompts_{en,es,fr,de}.tsv (prompt_id, lang,
text), 20 short original everyday sentences per language. The four sets are
parallel — prompt NN says the same thing in every language — so
`--references prompts` uses the same-numbered prompt of the target language as
the reference translation (tgt_text, for ASR-BLEU).

Recordings: --input DIR holding <lang>_<NN>.wav|.flac (e.g. en_01.wav), read
into the everyday microphone in the everyday room — or, for a language the
user does not speak, held-out clean speech played through a reference monitor
into that mic — with ~0.5 s of silence before and after each prompt (the
noise floor is estimated from the quietest frames). Or --manifest, a TSV
(prompt_id, path) or JSONL ({"prompt_id", "path"}); relative paths are relative
to the manifest's directory.

Nothing is denoised or loudness-normalised: the set must sound like the live
input. Per recording:
  readable   soundfile opens it and it holds samples
  rate       >= 16 kHz (then resampled to 16 kHz)
  mono       channels are averaged; a downmix more than 6 dB below the
             loudest channel (phase-cancelling stereo) fails
  duration   --min-s .. --max-s (default 1-12 s)
  clipping   no raw sample at or above --clip-dbfs (default -0.1 dBFS)
  level      speech level (mean power of the 20 ms frames >= floor + 10 dB)
             >= --min-speech-dbfs (default -45 dBFS RMS) and >= --min-snr-db
             (default 15 dB) above the noise floor (5th percentile of the
             20 ms frame levels)
Over the set: every prompt of --langs recorded (missing), no recording
without a prompt or with an unexpected name (extra), no prompt recorded twice
(duplicate), and with --references every list row has its reference.

Outputs in --out (default <data root>/private/live_mic: private, git-ignored):
  clips/<prompt_id>.wav     16 kHz mono PCM_24 (the fv_s2ut_file input format)
  live_mic_<src>_<tgt>.tsv  Stage 11 S2UT lists (fvml.evallist.S2UT_COLUMNS) for
                            each En<->{Es,Fr,De} direction whose source language
                            is in --langs: src_text from the prompts, tgt_text
                            empty unless --references gives it; only recordings
                            that passed every check
  report.json, report.md
Score them with eval_s2ut.py (UTMOS always; ASR-BLEU needs --references) or
eval_cross_lingual_voice.py --lists-dir <out> --list-pattern live_mic_{direction}.tsv.

Exit status: 0 = every check passed, 1 = a check failed or inputs are missing.
"""
import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np
import soundfile as sf

from fvml import asr, audio, evallist, paths
from fvml.target_voice import frame_db

DIRECTIONS = ("es_en", "fr_en", "de_en", "en_es", "en_fr", "en_de")   # fvml.s2ut_data
LANGS = ("en", "es", "fr", "de")
PROMPT_COLUMNS = ("prompt_id", "lang", "text")
REFERENCE_COLUMNS = ("prompt_id", "tgt_lang", "text")
NAME_RE = re.compile(r"^([a-z]{2,8})_(\d{2})$")
AUDIO_SUFFIXES = (".wav", ".flac")
FLOOR_PERCENTILE = 5.0          # 0.5 s of silence at each end covers 5 % up to ~20 s
SPEECH_MARGIN_DB = 10.0
MAX_DOWNMIX_LOSS_DB = 6.0
DB_FLOOR = -100.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="directory of <lang>_<NN>.wav|.flac recordings")
    src.add_argument("--manifest", help="TSV (prompt_id, path) or JSONL of the recordings")
    p.add_argument("--prompts-dir", default=str(paths.EVAL_LISTS / "s2ut"),
                   help="directory of live_mic_prompts_<lang>.tsv")
    p.add_argument("--langs", default=",".join(LANGS), help="recorded source languages")
    p.add_argument("--references", default=None,
                   help="reference translations: 'prompts' (the parallel prompt sets) or a "
                        "TSV (prompt_id, tgt_lang, text); default none (empty tgt_text)")
    p.add_argument("--out", help="default <data root>/private/live_mic")
    p.add_argument("--speaker", default="live_mic", help="speaker column of the lists")
    p.add_argument("--gender", default="?", help="gender column of the lists (F, M, ?)")
    p.add_argument("--min-s", type=float, default=1.0)
    p.add_argument("--max-s", type=float, default=12.0)
    p.add_argument("--clip-dbfs", type=float, default=-0.1,
                   help="a raw sample at or above this peak level counts as clipping")
    p.add_argument("--min-snr-db", type=float, default=15.0,
                   help="minimum speech level above the noise floor")
    p.add_argument("--min-speech-dbfs", type=float, default=-45.0,
                   help="minimum speech level (RMS dBFS of the active frames)")
    return p.parse_args(argv)


def rel_or_abs(path):
    """Path relative to the data root when inside it, else absolute posix."""
    try:
        return paths.relative(path)
    except ValueError:
        return pathlib.Path(path).resolve().as_posix()


def parse_langs(spec):
    langs = [x.strip() for x in (spec or "").split(",") if x.strip()]
    bad = [x for x in langs if x not in LANGS]
    if bad or not langs:
        raise ValueError(f"--langs {spec!r}: expected a subset of {','.join(LANGS)}")
    return list(dict.fromkeys(langs))


def read_prompts(prompts_dir, lang):
    """{prompt_id: text} of live_mic_prompts_<lang>.tsv (validated)."""
    path = pathlib.Path(prompts_dir) / f"live_mic_prompts_{lang}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (prompts for '{lang}')")
    out = {}
    for r in evallist.read(path, PROMPT_COLUMNS):
        pid, text = r["prompt_id"], " ".join(str(r["text"] or "").split())
        m = NAME_RE.match(pid or "")
        if not m or m.group(1) != lang or r["lang"] != lang:
            raise ValueError(f"{path}: prompt_id {pid!r} / lang {r['lang']!r} must be "
                             f"{lang}_<NN> / {lang}")
        if not text:
            raise ValueError(f"{path}: prompt {pid} has no text")
        if pid in out:
            raise ValueError(f"{path}: prompt {pid} listed twice")
        out[pid] = text
    if not out:
        raise ValueError(f"{path}: no prompts")
    return out


def read_references(spec, prompts_dir, prompts):
    """{(prompt_id, tgt_lang): text}, or None without --references."""
    if not spec:
        return None
    refs = {}
    if spec == "prompts":
        # The committed prompt sets are parallel: <tgt>_<NN> translates <src>_<NN>.
        parallel = {lang: read_prompts(prompts_dir, lang) for lang in LANGS}
        for d in DIRECTIONS:
            src, tgt = d.split("_")
            for pid in prompts.get(src, {}):
                twin = f"{tgt}_{pid.split('_', 1)[1]}"
                if twin in parallel[tgt]:
                    refs[(pid, tgt)] = parallel[tgt][twin]
        return refs
    path = pathlib.Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"--references {path} not found")
    seen = set()
    for r in evallist.read(path, REFERENCE_COLUMNS):
        pid, tgt = r["prompt_id"], r["tgt_lang"]
        text = " ".join(str(r["text"] or "").split())
        m = NAME_RE.match(pid or "")
        if not m:
            raise ValueError(f"{path}: prompt_id {pid!r} must be <lang>_<NN>")
        if tgt not in LANGS or tgt == m.group(1):
            raise ValueError(f"{path}: {pid}: tgt_lang {tgt!r} must be another of {LANGS}")
        if not text:
            raise ValueError(f"{path}: {pid} -> {tgt} has no text")
        if (pid, tgt) in seen:
            raise ValueError(f"{path}: {pid} -> {tgt} given twice")
        seen.add((pid, tgt))
        if m.group(1) not in prompts:           # a source language outside --langs
            continue
        if pid not in prompts[m.group(1)]:
            raise ValueError(f"{path}: {pid} is not one of the prompts")
        refs[(pid, tgt)] = text
    return refs


def scan_input(input_dir):
    """[(stem, path)] of the audio files under `input_dir` (recursive, sorted)."""
    root = pathlib.Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--input {root} is not a directory")
    return [(f.stem, f) for f in sorted(root.rglob("*"))
            if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES]


def read_manifest(path):
    """[(prompt_id, path)] from a TSV (prompt_id, path) or JSONL manifest."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"--manifest {path} not found")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        rows = evallist.read(path, ("prompt_id", "path"))
    out = []
    for r in rows:
        if not r.get("prompt_id") or not r.get("path"):
            raise ValueError(f"{path}: every row needs prompt_id and path")
        p = pathlib.Path(r["path"])
        out.append((str(r["prompt_id"]), p if p.is_absolute() else path.parent / p))
    return out


def level_stats(x, sr=audio.SAMPLE_RATE):
    """(speech_dbfs, floor_dbfs) of a mono signal from its 20 ms frame levels:
    floor = 5th percentile, speech = mean power of the frames >= floor + 10 dB
    (the loudest frame when none is)."""
    db = frame_db(x, sr)
    if len(db) == 0:
        return DB_FLOOR, DB_FLOOR
    floor = max(float(np.percentile(db, FLOOR_PERCENTILE)), DB_FLOOR)
    active = db[db >= floor + SPEECH_MARGIN_DB]
    speech = (10.0 * math.log10(float(np.mean(10.0 ** (active / 10.0))))
              if len(active) else float(np.max(db)))
    return max(speech, DB_FLOOR), floor


def check_recording(path, args):
    """-> {"checks": {name: {"pass", "detail"}}, "ok", metrics..., "_x16"}."""
    res = {"source": str(path), "checks": {}, "ok": False}

    def put(name, ok, detail):
        res["checks"][name] = {"pass": bool(ok), "detail": detail}

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:                       # soundfile: LibsndfileError, TypeError, ...
        put("readable", False, f"cannot read: {e}")
        return res
    if x.size == 0:
        put("readable", False, "no samples")
        return res
    put("readable", True, f"{x.shape[0]} frames, {x.shape[1]} channel(s), {sr} Hz")
    res.update(channels=int(x.shape[1]), sample_rate=int(sr),
               duration=round(x.shape[0] / sr, 3))
    put("rate", sr >= audio.SAMPLE_RATE, f"{sr} Hz (>= {audio.SAMPLE_RATE})")

    mono = x.mean(axis=1)
    loss = 0.0
    if x.shape[1] > 1:
        ch_rms = float(np.sqrt(np.max(np.mean(x.astype(np.float64) ** 2, axis=0))))
        mono_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        loss = 20.0 * math.log10(max(ch_rms, 1e-12) / max(mono_rms, 1e-12))
    res["downmix_loss_db"] = round(loss, 2)
    put("mono", loss <= MAX_DOWNMIX_LOSS_DB,
        f"{x.shape[1]} channel(s) averaged" + (f", downmix {loss:.1f} dB below the loudest "
                                               f"channel (<= {MAX_DOWNMIX_LOSS_DB:g})"
                                               if x.shape[1] > 1 else ""))
    put("duration", args.min_s <= res["duration"] <= args.max_s,
        f"{res['duration']:.2f} s ({args.min_s:g}-{args.max_s:g} s)")

    peak = float(np.max(np.abs(x)))
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    res["peak_dbfs"] = round(peak_db, 2)
    put("clipping", peak < 10.0 ** (args.clip_dbfs / 20.0),
        f"peak {peak_db:.2f} dBFS (< {args.clip_dbfs:g})")

    x16 = audio.resample(mono, sr, audio.SAMPLE_RATE)
    speech, floor = level_stats(x16)
    res.update(speech_dbfs=round(speech, 1), floor_dbfs=round(floor, 1),
               snr_db=round(speech - floor, 1))
    put("level", speech >= args.min_speech_dbfs and speech - floor >= args.min_snr_db,
        f"speech {speech:.1f} dBFS (>= {args.min_speech_dbfs:g}), "
        f"{speech - floor:.1f} dB above the floor {floor:.1f} dBFS (>= {args.min_snr_db:g})")
    res["ok"] = all(c["pass"] for c in res["checks"].values())
    res["_x16"] = x16
    return res


def classify(recordings, prompts, langs):
    """Match recordings to prompts -> (by_id, extra, duplicates, ignored, missing)."""
    by_id, extra, duplicates, ignored = {}, [], [], []
    for pid, path in recordings:
        m = NAME_RE.match(pid)
        if not m:
            extra.append({"name": pid, "path": str(path),
                          "why": "unexpected name (want <lang>_<NN>)"})
        elif m.group(1) not in langs:
            ignored.append({"name": pid, "path": str(path), "why": "language not in --langs"})
        elif pid not in prompts[m.group(1)]:
            extra.append({"name": pid, "path": str(path), "why": "no such prompt"})
        elif pid in by_id:
            duplicates.append({"name": pid, "path": str(path), "first": str(by_id[pid])})
        else:
            by_id[pid] = path
    missing = [pid for lang in langs for pid in prompts[lang] if pid not in by_id]
    return by_id, extra, duplicates, ignored, missing


def build_lists(files, prompts, langs, refs, args):
    """{direction: rows} for every direction whose source language was recorded."""
    lists = {}
    for d in DIRECTIONS:
        src, tgt = d.split("_")
        if src not in langs:
            continue
        rows = []
        for pid in prompts[src]:
            r = files.get(pid)
            if not r or not r["ok"]:
                continue
            ref = (refs or {}).get((pid, tgt), "")
            rows.append({"utt_id": f"live_{pid}", "src_lang": src, "tgt_lang": tgt,
                         "speaker": args.speaker, "gender": args.gender,
                         "src_duration": r["duration_16k"], "tgt_duration": 0.0, "gamma": 0.0,
                         "src_path": r["clip"], "tgt_path": "",
                         "src_text": asr.normalize_multilingual(prompts[src][pid]),
                         "tgt_text": asr.normalize_multilingual(ref) if ref else ""})
        lists[d] = rows
    return lists


def write_report(out_dir, report):
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    sc = report["set_checks"]
    lines = ["# Live-mic multilingual source set (M3 Stage 10 -> Stage 11)", "",
             f"Languages {', '.join(report['langs'])}; references "
             f"{report['references'] or 'none'}; {report['passed']} of {report['recorded']} "
             f"recording(s) passed; verdict **{'PASS' if report['ok'] else 'FAIL'}**.", ""]
    for name in ("missing", "extra", "duplicates", "references"):
        c = sc[name]
        lines.append(f"- [{'x' if c['pass'] else ' '}] {name}: {c['detail']}")
    if sc["ignored"]:
        lines.append(f"- ignored (language not in --langs): {', '.join(sc['ignored'])}")
    lines += ["", "| prompt | s | Hz | ch | peak dBFS | speech dBFS | floor dBFS | SNR dB "
                  "| status | failed |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for pid, r in report["recordings"].items():
        failed = [n for n, c in r["checks"].items() if not c["pass"]]
        lines.append(f"| {pid} | {r.get('duration', '—')} | {r.get('sample_rate', '—')} "
                     f"| {r.get('channels', '—')} | {r.get('peak_dbfs', '—')} "
                     f"| {r.get('speech_dbfs', '—')} | {r.get('floor_dbfs', '—')} "
                     f"| {r.get('snr_db', '—')} | {'PASS' if r['ok'] else 'FAIL'} "
                     f"| {', '.join(failed)} |")
    lines += ["", "| list | rows | with reference |", "|---|---:|---:|"]
    for d, info in report["lists"].items():
        lines.append(f"| live_mic_{d}.tsv | {info['rows']} | {info['with_reference']} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    try:
        langs = parse_langs(args.langs)
        prompts = {lang: read_prompts(args.prompts_dir, lang) for lang in langs}
        refs = read_references(args.references, args.prompts_dir, prompts)
        recordings = scan_input(args.input) if args.input else read_manifest(args.manifest)
        out_dir = pathlib.Path(args.out) if args.out \
            else paths.data_root() / "private" / "live_mic"
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        by_id, extra, duplicates, ignored, missing = classify(recordings, prompts, langs)
        files = {}
        for pid in [p for lang in langs for p in prompts[lang] if p in by_id]:
            r = check_recording(by_id[pid], args)
            x16 = r.pop("_x16", None)
            if r["ok"]:
                clip = clips_dir / f"{pid}.wav"
                sf.write(str(clip), x16, audio.SAMPLE_RATE, subtype="PCM_24")
                r.update(clip=rel_or_abs(clip), duration_16k=round(len(x16) / audio.SAMPLE_RATE, 3))
            files[pid] = r
            status = "ok" if r["ok"] else "FAIL " + ", ".join(
                n for n, c in r["checks"].items() if not c["pass"])
            print(f"[live-mic] {pid}: {status}")

        lists = build_lists(files, prompts, langs, refs, args)
        for d, rows in lists.items():
            evallist.write(out_dir / f"live_mic_{d}.tsv", rows, evallist.S2UT_COLUMNS)
        no_ref = sorted({f"{r['utt_id']}->{r['tgt_lang']}" for rows in lists.values()
                         for r in rows if not r["tgt_text"]})
        set_checks = {
            "missing": {"pass": not missing,
                        "detail": f"{len(missing)} prompt(s) not recorded"
                                  + (f": {', '.join(missing)}" if missing else "")},
            "extra": {"pass": not extra,
                      "detail": f"{len(extra)} unexpected recording(s)"
                                + (": " + ", ".join(f"{e['name']} ({e['why']})" for e in extra)
                                   if extra else "")},
            "duplicates": {"pass": not duplicates,
                           "detail": f"{len(duplicates)} prompt(s) recorded twice"
                                     + (": " + ", ".join(e["name"] for e in duplicates)
                                        if duplicates else "")},
            "references": ({"pass": not no_ref,
                            "detail": f"{len(no_ref)} list row(s) without a reference"
                                      + (f": {', '.join(no_ref[:10])}" if no_ref else "")}
                           if refs is not None else
                           {"pass": True, "detail": "none requested (tgt_text empty: UTMOS "
                                                    "only, ASR-BLEU needs --references)"}),
            "ignored": [e["name"] for e in ignored],
        }
        ok = (all(c["pass"] for k, c in set_checks.items() if k != "ignored")
              and all(r["ok"] for r in files.values()))
        report = {"langs": langs, "prompts_dir": str(args.prompts_dir),
                  "references": args.references, "out": str(out_dir),
                  "recorded": len(files), "passed": sum(1 for r in files.values() if r["ok"]),
                  "set_checks": set_checks, "recordings": files,
                  "lists": {d: {"rows": len(rows),
                                "with_reference": sum(1 for r in rows if r["tgt_text"])}
                            for d, rows in lists.items()},
                  "ok": ok}
        write_report(out_dir, report)
        print(f"[live-mic] {report['passed']}/{len(files)} recording(s) passed, "
              f"{len(missing)} missing, {len(extra)} extra, {len(duplicates)} duplicate(s) "
              f"-> {out_dir}")
        if not ok:
            print("[live-mic] checks failed (see report.md); the lists hold only the "
                  "recordings that passed", file=sys.stderr)
            return 1
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"[prepare_live_mic_set] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
