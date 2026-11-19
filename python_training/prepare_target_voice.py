#!/usr/bin/env python3
"""Prepare the user's own recordings into a clean target-voice dataset.

Milestone 2 / Stage 10 (Documents/Milestone 2/10_target_voice_finetune.md).

Input: a directory of raw 48 kHz / 24-bit mono WAVs named
``<speaker>_sNN_TTTT.wav`` plus one ``<speaker>_sNN_roomtone.wav`` per session
(30 s of room tone recorded at the session start). Per take:

  read raw -> reject segments whose *raw* samples clip -> 4th-order 65 Hz
  high-pass -> integrated-loudness normalise the whole take (--lufs) -> energy
  VAD into --min-clip-s/--max-clip-s clips (floor from that session's room
  tone, processed with the same HPF and the same take gain) -> per clip:
  reject SNR < --min-snr-db, reject peaks >= --clip-threshold after
  normalisation (clipped_after_norm: re-record, never repair) -> resample to
  16 kHz -> clips/<utt_id>.flac (PCM_24).

Outputs in --out (default <data root>/private/<speaker>, git-ignored):
  clips/*.flac, train.jsonl, heldout.jsonl, rejected.jsonl,
  <speaker>_targets.tsv (held-out references), speaker_metadata.json,
  report.json + report.md.

--live-mic writes the Stage 11 live-mic source set instead: same cleaning,
no SNR rejection, no held-out split, live_mic.tsv with empty text (scored
later with eval_vc.py --ref-text source).

--lang (default en) is the language of the recordings — one per voice
(Milestone 3 / Stage 10). It tags every written row and goes into
speaker_metadata.json, so train_unit_vocoder.py and the Stage 06 export give
the voice's voices.json entry its `lang`. Optional takes in another language
(the Stage 10 5-minute target-language reads) keep the voice's own language:
prepare them with the same --lang.

Exit status: 0 = prepared and the DoD passed (or --allow-short / --live-mic),
1 = missing inputs or the DoD failed.
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import soundfile as sf

from fvml import audio, evallist, paths, target_voice as tv
from fvml.datasets import write_jsonl
from train_unit_vocoder import validate_speaker_name


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--input", required=True, help="directory of raw WAV takes (scanned recursively)")
    p.add_argument("--speaker", required=True, help="speaker name, lowercase [a-z0-9_-]+")
    p.add_argument("--out", help="default <data root>/private/<speaker>")
    p.add_argument("--hpf-hz", type=float, default=65.0)
    p.add_argument("--lufs", type=float, default=-23.0)
    p.add_argument("--min-snr-db", type=float, default=30.0)
    p.add_argument("--clip-threshold", type=float, default=0.999)
    p.add_argument("--min-clip-s", type=float, default=2.0)
    p.add_argument("--max-clip-s", type=float, default=12.0)
    p.add_argument("--heldout-s", type=float, default=300.0, help="held-out seconds (DoD: 300)")
    p.add_argument("--min-minutes", type=float, default=60.0, help="kept minutes (DoD: 60)")
    p.add_argument("--allow-short", action="store_true",
                   help="exit 0 even when the DoD (minutes / held-out) fails")
    p.add_argument("--noise-floor-db", type=float, default=None,
                   help="use this floor instead of a session room tone")
    p.add_argument("--display", default=None, help='display name, e.g. "Me (private)"')
    p.add_argument("--lang", default="en",
                   help="language of the recordings (the voice's voices.json `lang`; "
                        "lowercase code, default en)")
    p.add_argument("--live-mic", action="store_true",
                   help="Stage 11 live-mic source set: no SNR rejection, no held-out, "
                        "writes live_mic.tsv")
    return p.parse_args(argv)


def validate_lang(code):
    """The voices.json `lang` rule of fvml.contract.validate_voices:
    lowercase [a-z-]{2,8} (en, es, fr, de, ...)."""
    if not isinstance(code, str) or not re.fullmatch(r"[a-z-]{2,8}", code):
        raise ValueError(f"invalid --lang {code!r}: expected a lowercase language code "
                         "such as en, es, fr or de")
    return code


def rel_or_abs(path):
    """Path relative to the data root when inside it, else absolute posix."""
    try:
        return paths.relative(path)
    except ValueError:
        return pathlib.Path(path).resolve().as_posix()


def scan_input(input_dir, speaker):
    """-> ({session: [(take, path)]}, {session: roomtone path}, [skipped names])."""
    takes, tones, skipped = {}, {}, []
    for f in sorted(input_dir.rglob("*.wav")):
        try:
            session = tv.parse_room_tone(f, speaker)
            tones.setdefault(session, f)
            continue
        except ValueError:
            pass
        try:
            session, take = tv.parse_take(f, speaker)
            takes.setdefault(session, []).append((take, f))
        except ValueError:
            skipped.append(f.name)
    for session in takes:
        takes[session].sort()
    return takes, tones, skipped


def any_peak_at_or_over(peaks, clip_threshold):
    """The "0 clipped clips kept" DoD: compare the *unrounded* linear peak
    against the threshold — the 0.1 dB-rounded `peak_dbfs` field is display
    only (a peak in [0.9989, 0.999) must not count as clipped)."""
    return any(p >= clip_threshold for p in peaks)


def clip_fields(speaker, session, take, seg_i, a, b, sr_in, norm_seg, take_path,
                snr, take_lufs):
    peak = float(np.max(np.abs(norm_seg))) if len(norm_seg) else 0.0
    return {"utt_id": f"{speaker}_{session}_{take}_{seg_i:02d}",
            "speaker": speaker, "session": session, "take": take, "seg": seg_i,
            "duration": (b - a) / sr_in, "source": take_path.name,
            "snr_db": round(snr, 1),
            "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-12)), 1),
            "lufs": round(take_lufs, 1)}


def write_clips(kept, out_dir):
    """Resample kept segments to 16 kHz and write clips/<utt_id>.flac (PCM_24)."""
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in kept:
        x16 = audio.resample(c["_seg"], c["_sr"], audio.SAMPLE_RATE)
        path = clips_dir / f"{c['utt_id']}.flac"
        sf.write(str(path), x16, audio.SAMPLE_RATE, subtype="PCM_24")
        rows.append({k: v for k, v in c.items() if not k.startswith("_")}
                    | {"path": rel_or_abs(path),
                       "duration": round(len(x16) / audio.SAMPLE_RATE, 3)})
    return rows


def write_report(out_dir, speaker, kept_rows, rejected, dod, live_mic,
                 heldout_mode=None, lang=None):
    reasons = {}
    for r in rejected:
        reasons[r["reason"]] = reasons.get(r["reason"], 0.0) + r["duration"]
    per_session = {}
    for r in kept_rows:
        s = per_session.setdefault(r["session"], {"clips": 0, "seconds": 0.0})
        s["clips"] += 1
        s["seconds"] += r["duration"]
    report = {
        "speaker": speaker,
        "lang": lang,
        "live_mic": live_mic,
        "kept": {"clips": len(kept_rows),
                 "seconds": round(sum(r["duration"] for r in kept_rows), 1)},
        "rejected": {"clips": len(rejected),
                     "seconds": round(sum(r["duration"] for r in rejected), 1),
                     "by_reason_seconds": {k: round(v, 1) for k, v in sorted(reasons.items())}},
        "per_session": {k: {"clips": v["clips"], "seconds": round(v["seconds"], 1)}
                        for k, v in sorted(per_session.items())},
        "dod": [{"check": n, "pass": ok, "detail": d} for n, ok, d in dod],
    }
    if heldout_mode is not None:
        report["heldout_mode"] = heldout_mode
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    lines = [f"# target-voice prep report — {speaker}" + (" (live-mic)" if live_mic else ""),
             ""]
    if lang:
        lines.append(f"- language: {lang}")
    lines += [f"- kept {report['kept']['clips']} clips, "
              f"{report['kept']['seconds'] / 60:.1f} min",
              f"- rejected {report['rejected']['clips']} clips, "
              f"{report['rejected']['seconds'] / 60:.1f} min",
              ""]
    if reasons:
        lines += ["| reject reason | seconds |", "|---|---|"]
        lines += [f"| {k} | {v:.1f} |" for k, v in sorted(reasons.items())]
        lines += [""]
    if per_session:
        lines += ["| session | clips | seconds |", "|---|---|---|"]
        lines += [f"| {k} | {v['clips']} | {v['seconds']:.1f} |"
                  for k, v in sorted(per_session.items())]
        lines += [""]
    if heldout_mode is not None:
        lines.append(f"- held-out selection mode: {heldout_mode}")
    for name, ok, detail in dod:
        lines.append(f"- [{'x' if ok else ' '}] {name}: {detail}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    try:
        speaker = validate_speaker_name(args.speaker)
        lang = validate_lang(args.lang)
        input_dir = pathlib.Path(args.input)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input {input_dir} is not a directory")
        out_dir = pathlib.Path(args.out) if args.out \
            else paths.data_root() / "private" / speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        takes, tones, skipped = scan_input(input_dir, speaker)
        if skipped:
            print(f"[prepare] skipped {len(skipped)} file(s) with unexpected names: "
                  f"{', '.join(skipped)}")
        if not takes:
            raise FileNotFoundError(
                f"no takes named <speaker>_sNN_TTTT.wav under {input_dir} "
                f"(speaker '{speaker}')")

        kept, rejected = [], []
        for session, take_list in sorted(takes.items()):
            tone_hpf, tone_sr = None, None
            if args.noise_floor_db is None:
                if session not in tones:
                    raise ValueError(
                        f"session {session} has no room tone "
                        f"({speaker}_{session}_roomtone.wav); pass --noise-floor-db to override")
                traw, tone_sr = sf.read(str(tones[session]), dtype="float32",
                                        always_2d=True)
                tone_hpf = tv.highpass(traw.mean(axis=1), tone_sr, args.hpf_hz)
            for take_name, take_path in take_list:
                raw, sr_in = sf.read(str(take_path), dtype="float32", always_2d=True)
                raw_mono = raw.mean(axis=1)
                clean = tv.highpass(raw_mono, sr_in, args.hpf_hz)
                norm, take_lufs, gain_db = tv.loudness_normalize(clean, sr_in, args.lufs)
                if args.noise_floor_db is not None:
                    floor = float(args.noise_floor_db)
                else:
                    # Room tone gets the same HPF and the same take gain.
                    floor = tv.noise_floor_db(tone_hpf * (10.0 ** (gain_db / 20.0)),
                                              tone_sr)
                segs = tv.vad_segments(norm, sr_in, floor, min_s=args.min_clip_s,
                                       max_s=args.max_clip_s)
                n_kept_before = len(kept)
                for seg_i, (a, b) in enumerate(segs):
                    raw_seg, norm_seg = raw_mono[a:b], norm[a:b]
                    snr = tv.estimate_snr_db(norm_seg, sr_in, floor)
                    peak = float(np.max(np.abs(norm_seg))) if len(norm_seg) else 0.0
                    reject = None
                    if tv.is_clipped(raw_seg, args.clip_threshold):
                        reject = "clipped"
                    elif peak >= args.clip_threshold:
                        reject = "clipped_after_norm"
                    elif not args.live_mic and snr < args.min_snr_db:
                        reject = "low_snr"
                    base = clip_fields(speaker, session, take_name, seg_i, a, b,
                                       sr_in, norm_seg, take_path, snr, take_lufs)
                    if reject:
                        rejected.append({**base, "reason": reject,
                                         "start_s": a / sr_in, "end_s": b / sr_in})
                    else:
                        kept.append({**base, "_seg": norm_seg, "_sr": sr_in,
                                     "_peak": peak})
                print(f"[prepare] {take_path.name}: {len(segs)} segment(s), "
                      f"{len(kept) - n_kept_before} kept")

        rows = [{**r, "lang": lang} for r in write_clips(kept, out_dir)]

        dod = []
        heldout_mode = None
        if args.live_mic:
            write_jsonl(out_dir / "rejected.jsonl", rejected)
            live_rows = [{"utt_id": r["utt_id"], "speaker": speaker, "gender": "U",
                          "duration": r["duration"], "path": r["path"], "text": ""}
                         for r in rows]
            evallist.write(out_dir / "live_mic.tsv", live_rows, evallist.SOURCE_COLUMNS)
            write_jsonl(out_dir / "live_mic.jsonl",
                        [{**r, "dataset": "live_mic", "subset": "source"} for r in rows])
        else:
            sel = (tv.select_heldout(rows, args.heldout_s) if rows
                   else {"heldout": set(), "guard": set(), "mode": "take"})
            held_ids, guard_ids = sel["heldout"], sel["guard"]
            heldout_mode = sel["mode"]
            if heldout_mode == "take-tail":
                print("[prepare] warning: held-out selection fell back to "
                      "'take-tail' (whole-take holdout overshoots or starves "
                      "training) — record takes of <= ~5 min so 'take' mode applies")
            for r in rows:
                if r["utt_id"] in guard_ids:
                    rejected.append({**r, "reason": "heldout_guard"})
            write_jsonl(out_dir / "rejected.jsonl", rejected)
            train_rows = [{**r, "dataset": "own_voice", "subset": "train"}
                          for r in rows
                          if r["utt_id"] not in held_ids and r["utt_id"] not in guard_ids]
            held_rows = [{**r, "dataset": "own_voice", "subset": "heldout"}
                         for r in rows if r["utt_id"] in held_ids]
            write_jsonl(out_dir / "train.jsonl", train_rows)
            write_jsonl(out_dir / "heldout.jsonl", held_rows)
            evallist.write(out_dir / f"{speaker}_targets.tsv",
                           [{"voice": speaker, "clip_id": r["utt_id"], "speaker": speaker,
                             "gender": "U", "duration": r["duration"], "path": r["path"],
                             "dataset": "own_voice"} for r in held_rows],
                           evallist.TARGET_COLUMNS)
            kept_min = sum(r["duration"] for r in rows) / 60.0
            held_s = sum(r["duration"] for r in held_rows)
            dod = [
                (f"clean >= {args.min_minutes:g} min", kept_min >= args.min_minutes,
                 f"{kept_min:.1f} min kept"),
                ("0 clipped clips kept",
                 not any_peak_at_or_over([c["_peak"] for c in kept],
                                         args.clip_threshold),
                 "clipping is rejected, never repaired"),
                (f"held-out >= {args.heldout_s:g} s", held_s >= args.heldout_s,
                 f"{held_s:.0f} s held out"),
            ]
        metadata = {speaker: {"display": args.display or speaker,
                              "license": "private — do not distribute",
                              "source": "own recordings (M2 Stage 10)",
                              "lang": lang}}
        (out_dir / "speaker_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        report = write_report(out_dir, speaker, rows, rejected, dod, args.live_mic,
                              heldout_mode, lang=lang)
        ok = all(ok for _, ok, _ in dod)
        print(f"[prepare] {report['kept']['clips']} clips, "
              f"{report['kept']['seconds'] / 60:.1f} min kept, "
              f"{len(rejected)} rejected -> {out_dir}")
        if not ok and not args.allow_short:
            print("[prepare] DoD failed (see report.md); use --allow-short to accept",
                  file=sys.stderr)
            return 1
        return 0
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[prepare_target_voice] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
