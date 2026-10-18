#!/usr/bin/env python3
"""Build the frozen evaluation lists in eval_lists/ from the prepared manifests.

  test_clean_200.tsv  LibriSpeech test-clean: 5 utterances of 3-15 s from each
                      of its 40 speakers (20 F / 20 M), evenly spaced over the
                      speaker's utterances in id order. No RNG: rerunning on
                      the same corpus gives the same list.
  targets.tsv         20 held-out reference clips (3-20 s) per target voice:
                      LJSpeech chapter LJ050 (excluded from training) and the
                      official test split of each chosen Hi-Fi TTS reader.

The lists are committed; rerun only to change them (and say so in the Stage 01
file). Needs: prepare_data.py --datasets librispeech-test-clean,ljspeech and
--metadata-only hifitts (the Hi-Fi TTS audio itself is only needed to *score*
against those voices).

Usage:
    make_eval_lists.py            # write eval_lists/*.tsv
    make_eval_lists.py --check    # rebuild in memory and compare with the committed lists

Exit status: 0 = written / identical, 1 = --check found a difference or inputs are missing.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

from fvml import datasets as fd
from fvml import evallist, paths

PER_SPEAKER = 5
CLIPS_PER_VOICE = 20
MIN_S, MAX_S = 3.0, 15.0
TARGET_MAX_S = 20.0               # speaker references may be longer than ASR sources
LJ_HELD_OUT_CHAPTER = "LJ050"


def test_clean_rows(manifest):
    by_spk = defaultdict(list)
    for r in fd.read_jsonl(manifest):
        if MIN_S <= r["duration"] <= MAX_S:
            by_spk[r["speaker"]].append(r)
    rows = []
    for spk in sorted(by_spk, key=int):
        rows += evallist.evenly_spaced(sorted(by_spk[spk], key=lambda r: r["utt_id"]), PER_SPEAKER)
    genders = defaultdict(int)
    for r in rows:
        genders[r["gender"]] += 1
    if len(by_spk) < 20 or abs(genders["F"] - genders["M"]) > PER_SPEAKER:
        raise SystemExit(f"unbalanced selection: {len(by_spk)} speakers, {dict(genders)}")
    return rows


def lj_targets(manifest):
    pool = sorted((r for r in fd.read_jsonl(manifest)
                   if r["utt_id"].startswith(LJ_HELD_OUT_CHAPTER) and MIN_S <= r["duration"] <= TARGET_MAX_S),
                  key=lambda r: r["utt_id"])
    return [{"voice": "lj", "clip_id": r["utt_id"], "speaker": "lj", "gender": "F",
             "duration": r["duration"], "path": r["path"], "dataset": "ljspeech"}
            for r in evallist.evenly_spaced(pool, CLIPS_PER_VOICE)]


def hifitts_targets(raw, speakers):
    rows = []
    for spk in speakers:
        pool = sorted((e for s, _, split, e in fd.hifitts_manifest_rows(raw, [spk])
                       if split == "test" and MIN_S <= e["duration"] <= TARGET_MAX_S),
                      key=lambda e: e["audio_filepath"])
        for e in evallist.evenly_spaced(pool, CLIPS_PER_VOICE):
            rel = pathlib.PurePosixPath(e["audio_filepath"]).relative_to("audio").with_suffix(".flac")
            rows.append({"voice": f"hifitts_{spk}", "clip_id": fd.hifitts_utt_id(spk, e["audio_filepath"]),
                         "speaker": f"hifitts_{spk}", "gender": fd.HIFITTS_SPEAKERS[spk][0],
                         "duration": e["duration"], "path": f"16k/hifitts/{rel.as_posix()}",
                         "dataset": "hifitts"})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="compare with the committed lists; write nothing")
    p.add_argument("--hifitts-speakers", default=",".join(fd.HIFITTS_DEFAULT))
    args = p.parse_args()
    root = paths.data_root()
    manifests = root / "manifests"
    needed = [manifests / "librispeech-test-clean.jsonl", manifests / "ljspeech.jsonl",
              root / "raw" / "hi_fi_tts_v0"]
    missing = [str(n) for n in needed if not n.exists()]
    if missing:
        print("make_eval_lists: missing " + ", ".join(missing), file=sys.stderr)
        return 1

    speakers = tuple(s.strip() for s in args.hifitts_speakers.split(","))
    outputs = {
        "test_clean_200.tsv": evallist.to_text(test_clean_rows(needed[0]), evallist.SOURCE_COLUMNS),
        "targets.tsv": evallist.to_text(lj_targets(needed[1]) + hifitts_targets(needed[2], speakers),
                                        evallist.TARGET_COLUMNS),
    }
    status = 0
    for name, text in outputs.items():
        path = paths.EVAL_LISTS / name
        if args.check:
            same = path.is_file() and path.read_text(encoding="utf-8").replace("\r\n", "\n") == text
            print(f"{name}: {'identical' if same else 'DIFFERS'}")
            status |= 0 if same else 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path} ({text.count(chr(10)) - 1} rows)")
    return status


if __name__ == "__main__":
    sys.exit(main())
