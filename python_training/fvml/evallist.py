"""Frozen evaluation lists (python_training/eval_lists/*.tsv).

Tab-separated with a header row. Paths are relative to the data root
($FLUXVOICE_DATA) unless absolute, so a private list (e.g. the "me" voice's
held-out clips, Stage 10) can point anywhere.

  test_clean_200.tsv   utt_id  speaker  gender  duration  path  text
      Source utterances to convert. Converted files are matched by utt_id
      (<converted_dir>/<utt_id>.wav).
  targets.tsv          voice  clip_id  speaker  gender  duration  path  dataset
      Held-out reference clips per target voice (target-similarity scoring).
      Training must exclude these clips (Stages 05 and 10).
"""
import csv

SOURCE_COLUMNS = ("utt_id", "speaker", "gender", "duration", "path", "text")
TARGET_COLUMNS = ("voice", "clip_id", "speaker", "gender", "duration", "path", "dataset")


def read(path, columns):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in columns if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        rows = list(reader)
    for r in rows:
        if r.get("duration"):
            r["duration"] = float(r["duration"])
    return rows


def to_text(rows, columns):
    lines = ["\t".join(columns)]
    for r in rows:
        values = [f"{r[c]:.3f}" if c == "duration" else str(r[c]) for c in columns]
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


def evenly_spaced(items, n):
    """n items spread over a sorted list (deterministic, no RNG)."""
    if len(items) < n:
        raise ValueError(f"need {n} items, have {len(items)}")
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]
