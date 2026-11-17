"""Language-ratio sampling weights (Milestone 3 / Stage 03).

Both multilingual trainers draw utterances with probability proportional to a
per-item weight (their length). language_weights() rescales those weights so
each language receives a fixed share of all draws, whatever the hours behind
it, while utterances inside a language keep their relative weights:

  student   balanced {en, es, fr, de}: 25% each (BALANCED)
  vocoder   40% en, 20% es, 20% fr, 20% de (VOCODER): English is oversampled
            so Mode A quality does not regress
"""
LANGS = ("en", "es", "fr", "de")
BALANCED = {lang: 1.0 / len(LANGS) for lang in LANGS}
VOCODER = {"en": 0.4, "es": 0.2, "fr": 0.2, "de": 0.2}
PRESETS = {"balanced": BALANCED, "m3-student": BALANCED, "m3-vocoder": VOCODER}


def parse_ratios(spec):
    """A preset name ('balanced', 'm3-vocoder'), a list of languages sharing
    equally ('en,es,fr,de') or explicit shares ('en=0.4,es=0.2,fr=0.2,de=0.2')
    -> {lang: share}, shares normalised to sum to 1. None/'' -> None."""
    if not spec:
        return None
    if isinstance(spec, dict):
        ratios = dict(spec)
    elif spec in PRESETS:
        ratios = dict(PRESETS[spec])
    else:
        ratios = {}
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            lang, sep, value = part.partition("=")
            lang = lang.strip()
            if not lang or lang in ratios:
                raise ValueError(f"language ratios {spec!r}: empty or repeated language")
            try:
                ratios[lang] = float(value) if sep else 1.0
            except ValueError:
                raise ValueError(f"language ratios {spec!r}: '{value}' is not a number") from None
    if not ratios or any(not v > 0.0 for v in ratios.values()):
        raise ValueError(f"language ratios {spec!r}: every share must be > 0")
    total = sum(ratios.values())
    return {lang: v / total for lang, v in ratios.items()}


def language_weights(weights, langs, ratios):
    """Per-item weights under which language l is drawn with probability
    ratios[l]. `weights` and `langs` are parallel sequences. Every item must
    belong to a language in `ratios`, and every language in `ratios` must have
    at least one item with positive weight: a silent fallback would hide a
    missing corpus. Returns (weights, {lang: {"items", "share", "weight"}})."""
    weights = [float(w) for w in weights]
    if len(weights) != len(langs):
        raise ValueError("weights and langs must have the same length")
    unknown = sorted({str(lang) for lang in langs if lang not in ratios})
    if unknown:
        raise ValueError(f"items in language(s) {', '.join(unknown)} have no share in the "
                         f"language ratios {sorted(ratios)} (untagged rows need --default-lang)")
    totals = dict.fromkeys(ratios, 0.0)
    counts = dict.fromkeys(ratios, 0)
    for w, lang in zip(weights, langs):
        totals[lang] += w
        counts[lang] += 1
    empty = sorted(lang for lang, t in totals.items() if not t > 0.0)
    if empty:
        raise ValueError(f"no training items for language(s) {', '.join(empty)}: prepare that "
                         "corpus or drop the language from the ratios")
    out = [w * ratios[lang] / totals[lang] for w, lang in zip(weights, langs)]
    summary = {lang: {"items": counts[lang], "share": ratios[lang], "weight": totals[lang]}
               for lang in ratios}
    return out, summary
