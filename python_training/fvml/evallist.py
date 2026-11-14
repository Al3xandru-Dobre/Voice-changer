"""Frozen evaluation lists (python_training/eval_lists/*.tsv, eval_lists/s2ut/*.tsv).

Tab-separated with a header row. Paths are relative to the data root
($FLUXVOICE_DATA) unless absolute, so a private list (e.g. the "me" voice's
held-out clips, Stage 10) can point anywhere.

Mode A (eval_lists/):
  test_clean_200.tsv   utt_id  speaker  gender  duration  path  text
  targets.tsv          voice  clip_id  speaker  gender  duration  path  dataset

Mode B simultaneous S2UT (eval_lists/s2ut/):
  test_<pair>_200.tsv / reordering_torture_60.tsv
      utt_id  src_lang  tgt_lang  speaker  gender  src_duration  tgt_duration
      gamma  src_path  tgt_path  src_text  tgt_text
  silence_hallucination_20.tsv
      utt_id  kind  src_lang  tgt_lang  src_duration  speech_end_s  src_path
      src_text  tgt_text
"""
import csv

SOURCE_COLUMNS = ("utt_id", "speaker", "gender", "duration", "path", "text")
TARGET_COLUMNS = ("voice", "clip_id", "speaker", "gender", "duration", "path", "dataset")
S2UT_COLUMNS = ("utt_id", "src_lang", "tgt_lang", "speaker", "gender",
                "src_duration", "tgt_duration", "gamma", "src_path", "tgt_path",
                "src_text", "tgt_text")
SILENCE_COLUMNS = ("utt_id", "kind", "src_lang", "tgt_lang", "src_duration",
                   "speech_end_s", "src_path", "src_text", "tgt_text")
FLOAT_COLUMNS = ("duration", "src_duration", "tgt_duration", "gamma", "speech_end_s")


def read(path, columns):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in columns if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        rows = list(reader)
    for r in rows:
        for col in FLOAT_COLUMNS:
            if col in r and r[col] not in (None, ""):
                r[col] = float(r[col])
    return rows


def to_text(rows, columns):
    lines = ["\t".join(columns)]
    for r in rows:
        values = [f"{float(r[c]):.3f}" if c in FLOAT_COLUMNS else str(r[c]) for c in columns]
        if any("\t" in v or "\n" in v for v in values):
            raise ValueError(f"tab or newline in {values}")
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def write(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_text(rows, columns))


def read_sources(path):
    return read(path, SOURCE_COLUMNS)


def read_targets(path, voice=None):
    rows = read(path, TARGET_COLUMNS)
    return [r for r in rows if voice is None or r["voice"] == voice]


def read_s2ut(path):
    """Read either a paired S2UT list (`S2UT_COLUMNS`) or a silence/flush audit
    list (`SILENCE_COLUMNS`), normalising both to a unified row dict."""
    with open(path, encoding="utf-8", newline="") as f:
        header = tuple(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE).fieldnames or ())
    if all(c in header for c in S2UT_COLUMNS):
        rows = read(path, S2UT_COLUMNS)
        for r in rows:
            r.setdefault("kind", "paired")
            r.setdefault("speech_end_s", r["src_duration"])
        return rows
    if all(c in header for c in SILENCE_COLUMNS):
        rows = read(path, SILENCE_COLUMNS)
        for r in rows:
            r.setdefault("speaker", "none")
            r.setdefault("gender", "?")
            r.setdefault("tgt_duration", r["speech_end_s"])
            r.setdefault("gamma", 0.0 if r["src_duration"] <= 0 else r["speech_end_s"] / r["src_duration"])
            r.setdefault("tgt_path", "")
        return rows
    missing = [c for c in S2UT_COLUMNS if c not in header]
    raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")


def evenly_spaced(items, n):
    """n items spread over a sorted list (deterministic, no RNG)."""
    if len(items) < n:
        raise ValueError(f"need {n} items, have {len(items)}")
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]
