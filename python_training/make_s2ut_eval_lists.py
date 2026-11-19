#!/usr/bin/env python3
"""Deterministically build the frozen Milestone 3 S2UT evaluation lists in
`python_training/eval_lists/s2ut/` (Stage 01):

  test_{es_en,fr_en,de_en,en_es,en_fr,en_de}_200.tsv
      200 held-out test utterances per direction (3-12 s, >= 20 speakers,
      balanced gender, normalized source & target text, duration ratio gamma).
  reordering_torture_60.tsv
      60 utterances (10 per direction) selected for cross-lingual syntactic
      reordering (German subordinate verb-final clauses, Spanish/French clitic
      pronouns and post-nominal adjective chains).
  silence_hallucination_20.tsv
      10 room-tone / noise-only clips (MUSAN noise) + 10 speech utterances
      followed by 5.0 s of trailing digital silence (tests `kFlush` termination
      and zero silence-babble hallucination).

Usage:
    make_s2ut_eval_lists.py [--only all|paired|silence]
    make_s2ut_eval_lists.py --only silence --check
"""
import argparse
from collections import defaultdict
import pathlib
import sys

import numpy as np

from fvml import asr, audio, datasets as fd, evallist, paths, s2ut_data as sd

PAIR_TARGET_COUNT = 200
TORTURE_PER_PAIR = 10
MIN_S, MAX_S = 3.0, 12.0
MIN_SPEAKERS = 20
SILENCE_TAIL_S = 5.0

DE_SUBORDINATORS = ("weil", "dass", "obwohl", "wenn", "während", "damit", "nachdem", "bevor")
ES_CLITICS = ("lo", "la", "los", "las", "le", "les", "se", "me", "nos")
FR_CLITICS = ("le", "la", "les", "lui", "leur", "en", "y", "ne")
COMMON_NOUNS_ES = ("casa", "coche", "libro", "hombre", "mujer", "ciudad", "mundo", "sistema", "gobierno")
COMMON_NOUNS_FR = ("maison", "voiture", "livre", "homme", "femme", "ville", "monde", "système", "gouvernement")


def reordering_score(src_lang, tgt_lang, src_text, tgt_text):
    """Heuristic syntactic-reordering score (>= 1 means strong reordering cues)
    used to deterministically rank held-out utterances for reordering_torture_60."""
    s_toks = asr.normalize_multilingual(src_text).split()
    t_toks = asr.normalize_multilingual(tgt_text).split()
    score = 0
    if "de" in (src_lang, tgt_lang):
        de_toks = s_toks if src_lang == "de" else t_toks
        hits = [i for i, w in enumerate(de_toks) if w in DE_SUBORDINATORS]
        if hits:
            score += 3 * len(hits)
            # Long distance between subordinator and clause-final verb increases reordering lag.
            if len(de_toks) - hits[0] >= 5:
                score += 2
    if "es" in (src_lang, tgt_lang):
        es_toks = s_toks if src_lang == "es" else t_toks
        score += sum(2 for w in es_toks if w in ES_CLITICS)
        score += sum(2 for i, w in enumerate(es_toks[:-1]) if w in COMMON_NOUNS_ES and len(es_toks[i + 1]) >= 4)
    if "fr" in (src_lang, tgt_lang):
        fr_toks = s_toks if src_lang == "fr" else t_toks
        score += sum(2 for w in fr_toks if w in FR_CLITICS)
        score += sum(2 for i, w in enumerate(fr_toks[:-1]) if w in COMMON_NOUNS_FR and len(fr_toks[i + 1]) >= 4)
    if len(s_toks) >= 8:
        score += 1
    return score


def select_pair_200(rows, n_total=PAIR_TARGET_COUNT, min_speakers=MIN_SPEAKERS):
    """Deterministic speaker- and gender-balanced selection of `n_total` test
    utterances in [3.0, 12.0] s."""
    valid = [
        r for r in rows
        if r.get("split", "test") == "test"
        and MIN_S <= float(r["src_duration_s"]) <= MAX_S
        and float(r["tgt_duration_s"]) > 0
        and r.get("src_text", "").strip()
        and r.get("tgt_text", "").strip()
    ]
    by_spk = defaultdict(list)
    spk_gender = {}
    for r in sorted(valid, key=lambda x: x["id"]):
        by_spk[r["speaker_id"]].append(r)
        if r.get("gender") in ("F", "M"):
            spk_gender[r["speaker_id"]] = r["gender"]

    if len(by_spk) < min_speakers or len(valid) < n_total:
        raise ValueError(
            f"need >= {n_total} valid test utterances across >= {min_speakers} speakers "
            f"(have {len(valid)} utterances, {len(by_spk)} speakers)"
        )

    f_spks = sorted(s for s in by_spk if spk_gender.get(s) == "F")
    m_spks = sorted(s for s in by_spk if spk_gender.get(s) == "M")
    u_spks = sorted(s for s in by_spk if s not in spk_gender)

    def _round_robin(speakers, quota):
        picked = []
        idx = {s: 0 for s in speakers}
        while len(picked) < quota:
            progressed = False
            for s in speakers:
                pool = by_spk[s]
                if idx[s] < len(pool):
                    picked.append(pool[idx[s]])
                    idx[s] += 1
                    progressed = True
                    if len(picked) == quota:
                        break
            if not progressed:
                break
        return picked

    if f_spks and m_spks:
        half = n_total // 2
        f_pick = _round_robin(f_spks, half)
        m_pick = _round_robin(m_spks, n_total - len(f_pick))
        chosen = f_pick + m_pick
        if len(chosen) < n_total:
            used_ids = {r["id"] for r in chosen}
            rem = [r for s in (f_spks + m_spks + u_spks) for r in by_spk[s] if r["id"] not in used_ids]
            chosen += rem[: n_total - len(chosen)]
    else:
        chosen = _round_robin(sorted(by_spk), n_total)

    if len(chosen) < n_total:
        raise ValueError(f"could only select {len(chosen)}/{n_total} utterances")

    return [_to_eval_row(r) for r in sorted(chosen, key=lambda x: x["id"])]


def select_reordering_torture(rows_by_pair, per_pair=TORTURE_PER_PAIR):
    """Select `per_pair` highest-scoring reordering utterances per direction."""
    out = []
    for pair in sd.DIRECTIONS:
        pool = [
            r for r in rows_by_pair[pair]
            if r.get("split", "test") == "test"
            and MIN_S <= float(r["src_duration_s"]) <= MAX_S
            and r.get("src_text", "").strip()
            and r.get("tgt_text", "").strip()
        ]
        if len(pool) < per_pair:
            raise ValueError(f"{pair}: need >= {per_pair} test utterances for torture set")
        ranked = sorted(
            pool,
            key=lambda r: (
                -reordering_score(r["src_lang"], r["tgt_lang"], r["src_text"], r["tgt_text"]),
                r["id"],
            ),
        )
        out.extend(_to_eval_row(r) for r in ranked[:per_pair])
    return out


def _to_eval_row(r):
    sdur = float(r["src_duration_s"])
    tdur = float(r["tgt_duration_s"])
    return {
        "utt_id": r["id"],
        "src_lang": r["src_lang"],
        "tgt_lang": r["tgt_lang"],
        "speaker": r["speaker_id"],
        "gender": r.get("gender") or "?",
        "src_duration": sdur,
        "tgt_duration": tdur,
        "gamma": tdur / sdur if sdur > 0 else 0.0,
        "src_path": r["src_audio_path"],
        "tgt_path": r["tgt_audio_path"],
        "src_text": asr.normalize_multilingual(r["src_text"]),
        "tgt_text": asr.normalize_multilingual(r["tgt_text"]),
    }


def build_silence_hallucination_rows(data_root, write_padded_audio=True):
    """Build the 20-clip `silence_hallucination_20.tsv` list:
      * 10 room-tone / noise-only clips (3-10 s) from `musan.jsonl` (`subset == "noise"`)
      * 10 speech utterances (3-8 s) from `librispeech-test-clean.jsonl` padded
        with 5.0 s of trailing digital silence in `16k/s2ut_eval/`.

    The `sil_tail_*` rows keep the English `src_text` but carry an empty
    `tgt_text` (BUG-030): the LibriSpeech clips were never translated, so there
    is no es/fr/de reference — scoring them as BLEU compared Whisper transcripts
    of the wrong language. eval_s2ut audits these rows on their post-allowance
    tail (`tail_hallucinated_words`) instead."""
    data_root = pathlib.Path(data_root)
    musan_mf = data_root / "manifests" / "musan.jsonl"
    ls_mf = data_root / "manifests" / "librispeech-test-clean.jsonl"
    if not (musan_mf.is_file() and ls_mf.is_file()):
        raise FileNotFoundError(
            f"need {musan_mf} and {ls_mf} to build silence_hallucination_20.tsv"
        )

    noise_pool = sorted(
        (r for r in fd.read_jsonl(musan_mf)
         if r.get("subset") == "noise" and 3.0 <= float(r["duration"]) <= 10.0),
        key=lambda r: r["utt_id"],
    )
    speech_pool = sorted(
        (r for r in fd.read_jsonl(ls_mf) if 3.0 <= float(r["duration"]) <= 8.0),
        key=lambda r: r["utt_id"],
    )
    picked_noise = evallist.evenly_spaced(noise_pool, 10)
    picked_speech = evallist.evenly_spaced(speech_pool, 10)

    out_rows = []
    # Distribute target languages across es/fr/de so all target decoder branches are audited.
    tgt_cycle = ("es", "fr", "de")
    for idx, r in enumerate(picked_noise):
        out_rows.append({
            "utt_id": f"sil_noise_{idx:02d}_{r['utt_id']}",
            "kind": "noise_only",
            "src_lang": "en",
            "tgt_lang": tgt_cycle[idx % len(tgt_cycle)],
            "src_duration": float(r["duration"]),
            "speech_end_s": 0.0,
            "src_path": r["path"],
            "src_text": "",
            "tgt_text": "",
        })

    eval_audio_dir = data_root / "16k" / "s2ut_eval"
    tail_zeros = np.zeros(int(round(SILENCE_TAIL_S * audio.SAMPLE_RATE)), dtype=np.float32)
    for idx, r in enumerate(picked_speech):
        uid = f"sil_tail_{idx:02d}_{r['utt_id']}"
        rel_dst = f"16k/s2ut_eval/{uid}.flac"
        dst = eval_audio_dir / f"{uid}.flac"
        speech_dur = float(r["duration"])
        if write_padded_audio and not dst.is_file():
            eval_audio_dir.mkdir(parents=True, exist_ok=True)
            wav = audio.load(data_root / r["path"])
            padded = np.concatenate([wav, tail_zeros])
            tmp = dst.with_name(dst.stem + ".tmp.flac")
            audio.save(tmp, padded, subtype="PCM_24")
            tmp.replace(dst)
            total_dur = audio.info(dst)[0]
        elif dst.is_file():
            total_dur = audio.info(dst)[0]
        else:
            total_dur = round(speech_dur + SILENCE_TAIL_S, 3)
        norm_txt = asr.normalize_multilingual(r.get("text", ""))
        out_rows.append({
            "utt_id": uid,
            "kind": "trailing_silence",
            "src_lang": "en",
            "tgt_lang": tgt_cycle[idx % len(tgt_cycle)],
            "src_duration": total_dur,
            "speech_end_s": speech_dur,
            "src_path": rel_dst,
            "src_text": norm_txt,
            "tgt_text": "",
        })
    return out_rows


def load_paired_rows_for_direction(paired_dir, direction):
    rows = []
    for mf in sorted(pathlib.Path(paired_dir).glob("*.jsonl")):
        for r in fd.read_jsonl(mf):
            if f"{r['src_lang']}_{r['tgt_lang']}" == direction:
                rows.append(r)
    return rows


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", choices=("all", "paired", "silence"), default="all")
    p.add_argument("--check", action="store_true",
                   help="rebuild in memory and compare against committed TSVs")
    p.add_argument("--data-root", help="default $FLUXVOICE_DATA or <repo>/data")
    p.add_argument("--out-dir", default=str(paths.EVAL_LISTS / "s2ut"))
    args = p.parse_args(argv)

    root = paths.resolve(args.data_root) if args.data_root else paths.data_root()
    out_dir = pathlib.Path(args.out_dir)
    outputs = {}

    try:
        if args.only in ("all", "paired"):
            paired_dir = root / "manifests" / "s2ut"
            rows_by_pair = {}
            for pair in sd.DIRECTIONS:
                prows = load_paired_rows_for_direction(paired_dir, pair)
                if not prows:
                    raise FileNotFoundError(
                        f"no prepared S2UT manifest rows for '{pair}' under {paired_dir}"
                    )
                rows_by_pair[pair] = prows
                outputs[f"test_{pair}_200.tsv"] = evallist.to_text(
                    select_pair_200(prows), evallist.S2UT_COLUMNS
                )
            outputs["reordering_torture_60.tsv"] = evallist.to_text(
                select_reordering_torture(rows_by_pair), evallist.S2UT_COLUMNS
            )

        if args.only in ("all", "silence"):
            sil_rows = build_silence_hallucination_rows(root, write_padded_audio=not args.check)
            outputs["silence_hallucination_20.tsv"] = evallist.to_text(
                sil_rows, evallist.SILENCE_COLUMNS
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"make_s2ut_eval_lists: {e}", file=sys.stderr)
        return 1

    status = 0
    for name, text in sorted(outputs.items()):
        target = out_dir / name
        if args.check:
            same = target.is_file() and target.read_text(encoding="utf-8").replace("\r\n", "\n") == text
            print(f"{name}: {'identical' if same else 'DIFFERS'}")
            status |= 0 if same else 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {target} ({text.count(chr(10)) - 1} rows)")
    return status


if __name__ == "__main__":
    sys.exit(main())
