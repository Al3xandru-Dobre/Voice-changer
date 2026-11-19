#!/usr/bin/env python3
"""Stage 10 DoD: the user's "me" voice across the six En<->{Es,Fr,De} directions.

Milestone 3 / Stage 10 (Documents/Milestone 3/10_cross_lingual_target_voice.md).

Gates measure the shipped runtime (MEMORY.md rule 11): every utterance of each
direction's list is translated by the C++ fv_s2ut_file (fvml/s2utfile.py) and
scored by eval_s2ut.py's engine (fvml/s2ut_eval.py). Per direction X->Y:

  me          --bundle with --voice (default "me"): the fine-tuned row
  native      --bundle with the native voice of Y (--native-voices)
  zero_shot   --zero-shot-bundle with --zero-shot-voice (default --voice): the
              "me" row before the fine-tune, i.e. initialised from the nearest
              English row (export a `train_unit_vocoder.py --steps 0
              --new-speaker-init nearest` checkpoint), or the nearest public
              voice of the base bundle itself
  before_<v>  --before-bundle with public voice v (regression; optional)

me and zero_shot outputs are scored against the held-out "me" references
(--me-targets, ECAPA tgt_cos); native and public voices against their own
references in --targets when it has any. ASR-BLEU / ASR-chrF++ score the
list's tgt_text with Whisper forced to the target language.

DoD (cross_lingual_dod), for each En->X with X in {es, fr, de}:
  en_X_bleu         ASR-BLEU(me) >= ASR-BLEU(native X voice) - 1.5
  en_X_cos_gain     tgt_cos(me) >= tgt_cos(zero-shot) + 0.05
  en_X_cos_vs_X_en  |tgt_cos(me, En->X) - tgt_cos(me, X->En)| <= 0.05
  utmos_<lang>      UTMOS(me) >= 3.0 in every output language evaluated
                    (utterance-weighted over the directions into it)
  coverage          all three En->X directions were evaluated
  me_voice_private  the bundle's "me" entry is licensed "private — do not distribute"
with --before-bundle, per public voice v and direction into v's language:
  public_<v>_<dir>_cos   tgt_cos(before) - tgt_cos(after) <= 0.02
  public_<v>_<dir>_bleu  ASR-BLEU(before) - ASR-BLEU(after) <= 0.5

A check without data (metric not scored, no references, direction not run) is
SKIPPED, never PASS; a silent or unscorable "me" output FAILS. Writes
summary.json + summary.md under --out with the verdict PASS (every check
passed), INCOMPLETE (none failed, some skipped) or FAIL. Exit 0 iff no
evaluated check failed. --skip-ml converts only (every quality check SKIPPED).

The outputs contain a personal voice: keep --out outside the repository or
under a git-ignored path (runs/, data/) and do not distribute them.

Usage:
  eval_cross_lingual_voice.py --bundle models/m3_me_v1 \\
      --zero-shot-bundle models/m3_me_zeroshot --me-targets <data>/private/me/me_targets.tsv \\
      --native-voices en=lj,es=css10_es,fr=css10_fr,de=css10_de \\
      --before-bundle models/m3_v1 --targets eval_lists/resynth/targets_m3.tsv \\
      --out runs/eval/m3_stage10
  # Stage 11 live-mic lists (prepare_live_mic_set.py):
  eval_cross_lingual_voice.py ... --lists-dir <data>/private/live_mic \\
      --list-pattern live_mic_{direction}.tsv --out runs/eval/m3_live_mic

Exit status: 0 = no DoD check failed, 1 = a check failed or inputs are missing.
"""
import argparse
import json
import math
import pathlib
import sys

from fvml import evallist, paths, s2utfile

DIRECTIONS = ("es_en", "fr_en", "de_en", "en_es", "en_fr", "en_de")   # fvml.s2ut_data
EN_TO_X = ("en_es", "en_fr", "en_de")
LANGS = ("en", "es", "fr", "de")
METRICS = ("bleu", "spk", "utmos")
BLEU_MAX_DROP = 1.5             # ASR-BLEU(me) >= ASR-BLEU(native) - 1.5
COS_MIN_GAIN = 0.05             # tgt_cos(me) >= tgt_cos(zero-shot) + 0.05
COS_MAX_GAP = 0.05              # |cos(me, En->X) - cos(me, X->En)| <= 0.05
UTMOS_MIN = 3.0
PUBLIC_COS_MAX_DROP = 0.02
PUBLIC_BLEU_MAX_DROP = 0.5
EPS = 1e-9                      # thresholds are inclusive; absorb float rounding
SUMMARY_FIELDS = ("asr_bleu", "asr_chrfpp", "tgt_cos", "src_cos", "utmos", "rtf",
                  "bleu_skipped_empty_refs")
PRIVATE_LICENSE = "private — do not distribute"


def _num(x):
    """A finite float, or None (None, NaN and non-numbers alike)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _langs(direction):
    src, _, tgt = direction.partition("_")
    return src, tgt


def _check(name, status, detail, value=None, threshold=None):
    row = {"check": name, "status": status, "detail": detail}
    if value is not None:
        row["value"] = round(float(value), 6)
    if threshold is not None:
        row["threshold"] = threshold
    return row


def _verdict_check(name, ok, detail, value, threshold):
    return _check(name, "pass" if ok else "fail", detail, value, threshold)


def _metric(summary, key, metric):
    """(value, scored): `scored` is False when `metric` was not computed."""
    summary = summary or {}
    return _num(summary.get(key)), metric in (summary.get("metrics") or ())


# -- DoD rules (pure) -------------------------------------------------------------
def cross_lingual_dod(results):
    """[{check, status, detail, value?, threshold?}] from

    {"directions": {dir: {"me": S, "native": S, "zero_shot": S}},
     "public": {voice: {dir: {"before": S, "after": S}}},
     "me_license": str or None}

    where S is a scored summary (asr_bleu, tgt_cos, utmos, n, metrics: the
    metrics that were computed). Values are None when absent."""
    dirs = results.get("directions") or {}
    out = []
    missing = [d for d in EN_TO_X if d not in dirs]
    out.append(_check("coverage", "skipped" if missing else "pass",
                      f"not evaluated: {', '.join(missing)}" if missing
                      else "en->es, en->fr and en->de evaluated"))
    lic = results.get("me_license")
    if lic is None:
        out.append(_check("me_voice_private", "skipped", "bundle has no voices.json entry to read"))
    else:
        out.append(_verdict_check("me_voice_private", "private" in str(lic).lower(),
                                  f"license {lic!r} (want {PRIVATE_LICENSE!r})", None, None))

    for d in EN_TO_X:
        if d not in dirs:
            continue
        x = _langs(d)[1]
        r = dirs[d]
        me, native, zero_shot = r.get("me"), r.get("native"), r.get("zero_shot")

        b_me, _ = _metric(me, "asr_bleu", "bleu")
        b_nat, _ = _metric(native, "asr_bleu", "bleu")
        if b_me is None or b_nat is None:
            out.append(_check(f"{d}_bleu", "skipped",
                              "no ASR-BLEU for me and native (bleu not scored or the list has "
                              "no tgt_text references)"))
        else:
            delta = b_me - b_nat
            out.append(_verdict_check(
                f"{d}_bleu", delta >= -BLEU_MAX_DROP - EPS,
                f"ASR-BLEU me {b_me:.2f} vs native '{(native or {}).get('voice')}' {b_nat:.2f}: "
                f"{delta:+.2f} (>= -{BLEU_MAX_DROP})", delta, -BLEU_MAX_DROP))

        c_me, me_scored = _metric(me, "tgt_cos", "spk")
        c_zs, zs_scored = _metric(zero_shot, "tgt_cos", "spk")
        if c_me is None and me_scored:
            out.append(_check(f"{d}_cos_gain", "fail",
                              "no tgt_cos for the me output (silent or unscorable)"))
        elif c_me is None or c_zs is None:
            why = ("zero-shot output silent or unscorable" if zs_scored and c_zs is None
                   else "ECAPA (spk) not scored")
            out.append(_check(f"{d}_cos_gain", "skipped", why))
        else:
            gain = c_me - c_zs
            out.append(_verdict_check(
                f"{d}_cos_gain", gain >= COS_MIN_GAIN - EPS,
                f"tgt_cos me {c_me:.3f} vs zero-shot {c_zs:.3f}: {gain:+.3f} "
                f"(>= +{COS_MIN_GAIN})", gain, COS_MIN_GAIN))

        rev = f"{x}_en"
        c_rev, rev_scored = _metric((dirs.get(rev) or {}).get("me"), "tgt_cos", "spk")
        name = f"{d}_cos_vs_{rev}"
        if c_me is None and me_scored:
            out.append(_check(name, "fail", "no tgt_cos for the me output (silent or unscorable)"))
        elif rev not in dirs:
            out.append(_check(name, "skipped", f"{rev} not evaluated"))
        elif c_rev is None and rev_scored:
            out.append(_check(name, "fail", f"no tgt_cos for the me output of {rev} "
                                            "(silent or unscorable)"))
        elif c_me is None or c_rev is None:
            out.append(_check(name, "skipped", "ECAPA (spk) not scored"))
        else:
            gap = abs(c_me - c_rev)
            out.append(_verdict_check(
                name, gap <= COS_MAX_GAP + EPS,
                f"tgt_cos me {d} {c_me:.3f} vs {rev} {c_rev:.3f}: |gap| {gap:.3f} "
                f"(<= {COS_MAX_GAP})", gap, COS_MAX_GAP))

    for lang, row in utmos_by_language(results).items():
        name = f"utmos_{lang}"
        if row["utmos"] is not None:
            out.append(_verdict_check(
                name, row["utmos"] >= UTMOS_MIN - EPS,
                f"UTMOS me in {lang} {row['utmos']:.2f} over {', '.join(row['directions'])} "
                f"(>= {UTMOS_MIN})", row["utmos"], UTMOS_MIN))
        elif row["unscorable"]:
            out.append(_check(name, "fail", f"no UTMOS for the me output of "
                                            f"{', '.join(row['unscorable'])} (silent)"))
        else:
            out.append(_check(name, "skipped", "UTMOS not scored"))

    for voice, per_dir in sorted((results.get("public") or {}).items()):
        if not per_dir:
            out.append(_check(f"public_{voice}", "skipped",
                              f"no evaluated direction into the language of '{voice}'"))
        for d, ba in sorted(per_dir.items()):
            before, after = ba.get("before"), ba.get("after")
            cb, ca = _num((before or {}).get("tgt_cos")), _num((after or {}).get("tgt_cos"))
            if cb is None or ca is None:
                out.append(_check(f"public_{voice}_{d}_cos", "skipped",
                                  f"no tgt_cos for '{voice}' (no references in --targets, "
                                  "spk not scored, or silent output)"))
            else:
                drop = cb - ca
                out.append(_verdict_check(
                    f"public_{voice}_{d}_cos", drop <= PUBLIC_COS_MAX_DROP + EPS,
                    f"tgt_cos before {cb:.3f} after {ca:.3f}: drop {drop:+.3f} "
                    f"(<= {PUBLIC_COS_MAX_DROP})", drop, PUBLIC_COS_MAX_DROP))
            bb, ba_ = _num((before or {}).get("asr_bleu")), _num((after or {}).get("asr_bleu"))
            if bb is None or ba_ is None:
                out.append(_check(f"public_{voice}_{d}_bleu", "skipped",
                                  "no ASR-BLEU (bleu not scored or no tgt_text references)"))
            else:
                drop = bb - ba_
                out.append(_verdict_check(
                    f"public_{voice}_{d}_bleu", drop <= PUBLIC_BLEU_MAX_DROP + EPS,
                    f"ASR-BLEU before {bb:.2f} after {ba_:.2f}: drop {drop:+.2f} "
                    f"(<= {PUBLIC_BLEU_MAX_DROP})", drop, PUBLIC_BLEU_MAX_DROP))
    return out


def utmos_by_language(results):
    """{output language: {"utmos": mean or None, "n", "directions", "unscorable"}}
    for the me outputs, utterance-weighted over the directions into it."""
    by_lang = {}
    for d, r in sorted((results.get("directions") or {}).items()):
        me = r.get("me") or {}
        lang = _langs(d)[1]
        row = by_lang.setdefault(lang, {"sum": 0.0, "n": 0, "directions": [], "unscorable": []})
        u, scored = _metric(me, "utmos", "utmos")
        if u is not None:
            n = max(int(me.get("n") or 0), 1)
            row["sum"] += u * n
            row["n"] += n
            row["directions"].append(d)
        elif scored:
            row["unscorable"].append(d)
    order = {lang: i for i, lang in enumerate(LANGS)}
    return {lang: {"utmos": (row["sum"] / row["n"]) if row["n"] else None, "n": row["n"],
                   "directions": row["directions"], "unscorable": row["unscorable"]}
            for lang, row in sorted(by_lang.items(), key=lambda kv: order.get(kv[0], 99))}


def verdict(checks):
    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        return "FAIL"
    return "INCOMPLETE" if "skipped" in statuses or not checks else "PASS"


# -- inputs -------------------------------------------------------------------------
def parse_native_voices(spec):
    """'en=lj,es=css10_es,...' -> {lang: voice}."""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        lang, sep, voice = part.partition("=")
        lang, voice = lang.strip(), voice.strip()
        if not sep or not lang or not voice:
            raise ValueError(f"--native-voices wants LANG=VOICE pairs, got {part!r}")
        if lang in out:
            raise ValueError(f"--native-voices: language '{lang}' given twice")
        out[lang] = voice
    if not out:
        raise ValueError("--native-voices is empty "
                         "(e.g. en=lj,es=css10_es,fr=css10_fr,de=css10_de)")
    return out


def parse_directions(spec):
    if not spec:
        return list(DIRECTIONS)
    dirs = [d.strip().replace("->", "_") for d in spec.split(",") if d.strip()]
    bad = [d for d in dirs if d not in DIRECTIONS]
    if bad:
        raise ValueError(f"--directions: unknown direction(s) {bad} "
                         f"(known: {', '.join(DIRECTIONS)})")
    return list(dict.fromkeys(dirs))


def parse_metrics(spec):
    metrics = tuple(m.strip() for m in (spec or "").split(",") if m.strip())
    bad = sorted(set(metrics) - set(METRICS))
    if bad:
        raise ValueError(f"--metrics: unknown metric(s) {bad} (known: {', '.join(METRICS)})")
    return metrics


def bundle_voices(bundle):
    """{name: voices.json entry} of a bundle, or None when it has no voices.json."""
    p = pathlib.Path(bundle) / "voices.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return {v["name"]: v for v in data.get("voices", []) if isinstance(v, dict) and "name" in v}


def _require_voice(voices, name, bundle, what):
    if voices is not None and name not in voices:
        raise ValueError(f"{what} '{name}' is not in {bundle}/voices.json "
                         f"({', '.join(sorted(voices)) or 'no voices'})")


def direction_rows(list_path, direction, limit):
    """The paired src->tgt rows of an S2UT list (first `limit`)."""
    src, tgt = _langs(direction)
    rows = evallist.read_s2ut(list_path)
    if any(r.get("kind", "paired") != "paired" for r in rows):
        raise ValueError(f"{list_path}: not a paired S2UT list (score the silence/flush "
                         "audit list with eval_s2ut.py)")
    sel = [r for r in rows if r["src_lang"] == src and r["tgt_lang"] == tgt]
    if not sel:
        raise ValueError(f"{list_path}: no {src}->{tgt} rows")
    return sel[: limit or None]


def public_voice_list(spec, native, before_voices, me_voice):
    """--public-voices: default the native voices, 'all' = every voice of the
    before-bundle except the "me" voice, else a comma-separated list."""
    if not spec:
        return list(dict.fromkeys(native.values()))
    if spec.strip() == "all":
        if before_voices is None:
            raise ValueError("--public-voices all needs a --before-bundle with voices.json")
        return [v for v in before_voices if v != me_voice]
    return list(dict.fromkeys(v.strip() for v in spec.split(",") if v.strip()))


def public_directions(voice, native, voice_tables, directions):
    """Directions a public voice is checked on: into the language(s) it is the
    native voice of, else into its voices.json `lang`, else every direction."""
    langs = {lang for lang, v in native.items() if v == voice}
    if not langs:
        for table in voice_tables:
            lang = ((table or {}).get(voice) or {}).get("lang")
            if lang:
                langs = {lang}
                break
    if not langs:
        return list(directions)
    return [d for d in directions if _langs(d)[1] in langs]


def default_score(list_path, converted, out_dir, targets, voice, metrics, device, limit):
    """eval_s2ut.py's engine on one converted directory -> its summary."""
    from fvml import s2ut_eval
    cfg = s2ut_eval.S2utEvalConfig(eval_list=list_path, converted=converted, targets=targets,
                                   voice=voice, metrics=tuple(metrics), device=device,
                                   limit=limit)
    return s2ut_eval.evaluate_s2ut(cfg, out_dir, log=lambda *a: None)


# -- driver --------------------------------------------------------------------------
def run(args, runner=None, score_fn=None, log=print):
    """-> (results, checks). `runner` replaces fvml.s2utfile.convert (one
    translation, same signature) and `score_fn(list_path, converted, out_dir,
    targets, voice, metrics, device, limit)` replaces the eval_s2ut engine, so
    tests run without fv_s2ut_file, Whisper, ECAPA or UTMOS."""
    score_fn = score_fn or default_score
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    directions = parse_directions(args.directions)
    native = parse_native_voices(args.native_voices)
    metrics = () if args.skip_ml else parse_metrics(args.metrics)
    need = sorted({_langs(d)[1] for d in directions} - set(native))
    if need:
        raise ValueError(f"--native-voices has no voice for target language(s) {', '.join(need)}")

    me_targets = paths.resolve(args.me_targets)
    if not me_targets.is_file():
        raise FileNotFoundError(f"--me-targets {me_targets} not found")
    if not evallist.read_targets(me_targets, args.voice):
        raise ValueError(f"--me-targets {me_targets} has no reference clips for voice "
                         f"'{args.voice}'")
    public_targets = paths.resolve(args.targets) if args.targets else None
    ref_cache = {}

    def has_refs(voice):
        if voice not in ref_cache:
            ref_cache[voice] = bool(public_targets and public_targets.is_file()
                                    and evallist.read_targets(public_targets, voice))
        return ref_cache[voice]

    after_voices = bundle_voices(args.bundle)
    zs_voices = bundle_voices(args.zero_shot_bundle)
    zs_voice = args.zero_shot_voice or args.voice
    _require_voice(after_voices, args.voice, args.bundle, "--voice")
    for lang in sorted({_langs(d)[1] for d in directions}):
        _require_voice(after_voices, native[lang], args.bundle, f"native voice for {lang}")
    _require_voice(zs_voices, zs_voice, args.zero_shot_bundle, "--zero-shot-voice")
    before_voices = bundle_voices(args.before_bundle) if args.before_bundle else None

    lists_dir = paths.resolve(args.lists_dir)
    lists = {}
    for d in directions:
        src, tgt = _langs(d)
        p = lists_dir / args.list_pattern.format(direction=d, src=src, tgt=tgt)
        if not p.is_file():
            raise FileNotFoundError(f"{p}: no {src}->{tgt} list (build the S2UT lists with "
                                    "make_s2ut_eval_lists.py, or pass --directions / "
                                    "--list-pattern)")
        lists[d] = p

    tool = None
    if runner is None:
        tool = s2utfile.find_s2ut_file()
        if tool is None:
            raise FileNotFoundError(s2utfile.NOT_FOUND)
    extra = []
    if args.wait_k:
        extra += ["--wait-k", str(args.wait_k)]
    if args.frames_per_step:
        extra += ["--frames-per-step", str(args.frames_per_step)]

    me_entry = (after_voices or {}).get(args.voice)
    results = {"voice": args.voice, "bundle": str(args.bundle),
               "zero_shot_bundle": str(args.zero_shot_bundle), "zero_shot_voice": zs_voice,
               "before_bundle": str(args.before_bundle) if args.before_bundle else None,
               "native_voices": native, "metrics": list(metrics),
               "me_license": me_entry.get("license") if me_entry else None,
               "me_lang": me_entry.get("lang") if me_entry else None,
               "directions": {}, "public": {}}
    # The rows of each direction (first --limit), frozen into <out>/<dir>/list.tsv
    # so the scorer sees exactly the translated utterances, even for a list
    # that mixes directions (reordering_torture_60.tsv).
    rows_by_dir, scored_lists = {}, {}
    for d in directions:
        rows_by_dir[d] = direction_rows(lists[d], d, args.limit)
        scored_lists[d] = out_dir / d / "list.tsv"
        scored_lists[d].parent.mkdir(parents=True, exist_ok=True)
        evallist.write(scored_lists[d], rows_by_dir[d], evallist.S2UT_COLUMNS)

    def translate_and_score(d, name, bundle, voice, targets, ref_voice, wanted):
        rows = rows_by_dir[d]
        base = out_dir / d / name
        s2utfile.convert_list(bundle, rows, base / "wav", voice=voice, extra_args=extra,
                              runner=runner, s2ut_file=tool)
        scored = tuple(m for m in wanted if m in metrics)
        row = {"voice": voice, "bundle": str(bundle), "n": len(rows), "metrics": list(scored)}
        if scored:
            spk = "spk" in scored
            s = score_fn(scored_lists[d], base / "wav", base / "eval", targets if spk else None,
                         ref_voice if spk else None, scored, args.device, None)
            row.update({k: _num(s.get(k)) if k != "bleu_skipped_empty_refs" else s.get(k)
                        for k in SUMMARY_FIELDS if k in s})
        log(f"[cross-lingual] {d} {name} ({voice} @ {bundle}): {len(rows)} utterances"
            + "".join(f", {k} {row[k]:.3f}" for k in ("asr_bleu", "tgt_cos", "utmos")
                      if row.get(k) is not None))
        return row

    for d in directions:
        tgt = _langs(d)[1]
        nat = native[tgt]
        results["directions"][d] = {
            "me": translate_and_score(d, "me", args.bundle, args.voice, me_targets,
                                      args.voice, ("bleu", "spk", "utmos")),
            "native": translate_and_score(
                d, "native", args.bundle, nat, public_targets, nat,
                ("bleu", "utmos") + (("spk",) if has_refs(nat) else ())),
            "zero_shot": translate_and_score(d, "zero_shot", args.zero_shot_bundle, zs_voice,
                                             me_targets, args.voice, ("spk", "utmos")),
        }

    if args.before_bundle:
        for voice in public_voice_list(args.public_voices, native, before_voices, args.voice):
            _require_voice(before_voices, voice, args.before_bundle, "public voice")
            _require_voice(after_voices, voice, args.bundle, "public voice")
            wanted = ("bleu",) + (("spk",) if has_refs(voice) else ())
            per = {}
            for d in public_directions(voice, native, (after_voices, before_voices), directions):
                after = (results["directions"][d]["native"] if native[_langs(d)[1]] == voice
                         else translate_and_score(d, f"after_{voice}", args.bundle, voice,
                                                  public_targets, voice, wanted))
                per[d] = {"before": translate_and_score(d, f"before_{voice}", args.before_bundle,
                                                        voice, public_targets, voice, wanted),
                          "after": after}
            results["public"][voice] = per

    results["utmos_by_lang"] = utmos_by_language(results)
    checks = cross_lingual_dod(results)
    write_summary(results, checks, out_dir)
    for c in checks:
        log(f"[cross-lingual] {c['check']}: {c['status'].upper()} — {c['detail']}")
    return results, checks


# -- report ----------------------------------------------------------------------------
def _fmt(v, digits=2):
    v = _num(v)
    return "—" if v is None else f"{v:.{digits}f}"


def write_summary(results, checks, out_dir):
    out_dir = pathlib.Path(out_dir)
    v = verdict(checks)
    private = results.get("me_license") is None or "private" in str(results["me_license"]).lower()
    doc = {**results, "private": private, "checks": checks, "verdict": v,
           "thresholds": {"bleu_max_drop": BLEU_MAX_DROP, "cos_min_gain": COS_MIN_GAIN,
                          "cos_max_gap": COS_MAX_GAP, "utmos_min": UTMOS_MIN,
                          "public_cos_max_drop": PUBLIC_COS_MAX_DROP,
                          "public_bleu_max_drop": PUBLIC_BLEU_MAX_DROP}}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    lines = ["# M3 Stage 10 DoD — cross-lingual target voice", ""]
    if private:
        lines += [f"> **PRIVATE** — contains the personal voice `{results['voice']}`; "
                  "do not distribute these outputs.", ""]
    lines += [f"Voice `{results['voice']}` from `{results['bundle']}`; zero-shot baseline "
              f"`{results['zero_shot_voice']}` from `{results['zero_shot_bundle']}`; native voices "
              + ", ".join(f"{k} `{x}`" for k, x in results["native_voices"].items())
              + f"; metrics {', '.join(results['metrics']) or 'none (--skip-ml)'}.", "",
              "| Direction | Config | Voice | ASR-BLEU | ASR-chrF++ | tgt_cos | UTMOS | N |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for d, per in results["directions"].items():
        for name, r in per.items():
            lines.append(f"| `{d.replace('_', '->')}` | {name} | `{r['voice']}` "
                         f"| {_fmt(r.get('asr_bleu'))} | {_fmt(r.get('asr_chrfpp'))} "
                         f"| {_fmt(r.get('tgt_cos'), 3)} | {_fmt(r.get('utmos'))} | {r['n']} |")
    lines += ["", f"UTMOS of `{results['voice']}` per output language:", "",
              "| Language | UTMOS | N | Directions |", "|---|---:|---:|---|"]
    for lang, row in results.get("utmos_by_lang", {}).items():
        lines.append(f"| {lang} | {_fmt(row['utmos'])} | {row['n']} "
                     f"| {', '.join(row['directions']) or '—'} |")
    if results.get("public"):
        lines += ["", "Public-voice regression (before -> after):", "",
                  "| Voice | Direction | tgt_cos before | after | BLEU before | after |",
                  "|---|---|---:|---:|---:|---:|"]
        for voice, per in results["public"].items():
            for d, ba in per.items():
                b, a = ba["before"], ba["after"]
                lines.append(f"| `{voice}` | `{d.replace('_', '->')}` "
                             f"| {_fmt(b.get('tgt_cos'), 3)} | {_fmt(a.get('tgt_cos'), 3)} "
                             f"| {_fmt(b.get('asr_bleu'))} | {_fmt(a.get('asr_bleu'))} |")
    lines += ["", "| Check | Status | Detail |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {c['status'].upper()} | {c['detail']} |")
    lines += ["", f"**Verdict: {v}**"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--bundle", required=True,
                   help="Mode B bundle after the fine-tune (with --voice)")
    p.add_argument("--zero-shot-bundle", required=True,
                   help="bundle of the zero-shot baseline (the 'me' row before the fine-tune)")
    p.add_argument("--zero-shot-voice", help="voice of --zero-shot-bundle (default: --voice)")
    p.add_argument("--voice", default="me", help="the fine-tuned voice and its --me-targets name")
    p.add_argument("--me-targets", required=True,
                   help="held-out 'me' references (<speaker>_targets.tsv from "
                        "prepare_target_voice.py)")
    p.add_argument("--native-voices", required=True,
                   help="native public voice per target language: en=V,es=V,fr=V,de=V")
    p.add_argument("--lists-dir", default=str(paths.EVAL_LISTS / "s2ut"),
                   help="directory of the per-direction S2UT lists")
    p.add_argument("--list-pattern", default="test_{direction}_200.tsv",
                   help="paired S2UT list file name; {direction}, {src}, {tgt} are substituted "
                        "(live-mic: live_mic_{direction}.tsv; a list holding several "
                        "directions, e.g. reordering_torture_60.tsv, is split per direction)")
    p.add_argument("--directions", help=f"subset of {','.join(DIRECTIONS)} (default all)")
    p.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                   help="public voice references (tgt_cos of native/public voices)")
    p.add_argument("--before-bundle", help="bundle before the fine-tune: public-voice regression")
    p.add_argument("--public-voices",
                   help="voices for the regression (default: the native voices; 'all' = every "
                        "voice of --before-bundle except --voice)")
    p.add_argument("--metrics", default=",".join(METRICS),
                   help=f"subset of {','.join(METRICS)} to score")
    p.add_argument("--skip-ml", action="store_true",
                   help="convert only: no Whisper/ECAPA/UTMOS, every quality check SKIPPED")
    p.add_argument("--wait-k", type=int,
                   help="fv_s2ut_file --wait-k (default: the bundle's per-pair default)")
    p.add_argument("--frames-per-step", type=int, choices=(1, 2, 4),
                   help="fv_s2ut_file --frames-per-step (default 4)")
    p.add_argument("--limit", type=int, help="first N utterances per direction (smoke runs)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="output directory (private: keep it git-ignored)")
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    try:
        _, checks = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"[eval_cross_lingual_voice] error: {e}", file=sys.stderr)
        return 1
    v = verdict(checks)
    print(f"[cross-lingual] verdict {v} -> {pathlib.Path(args.out) / 'summary.md'}")
    return 1 if v == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
