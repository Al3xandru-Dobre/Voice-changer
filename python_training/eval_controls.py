#!/usr/bin/env python3
"""Stage 09 offline DoD: measure the voice controls on the shipped runtime.

Milestone 2 / Stage 09 (Documents/Milestone 2/09_voice_controls_presets.md).

Gates measure the shipped runtime, not PyTorch eager (MEMORY.md rule 11):
every conversion below goes through fv_vc_file (fvml/vcfile.py) with --param /
--preset / --change, on the eval list's source utterances.

Checks (--checks, default all):
  pitch       --param pitch_shift=N for N in {-7,-3,+3,+7} vs 0: per-utterance
              12*log2(median_f0_N / median_f0_0) (fvml.f0_ref), median over
              utterances within +-0.3 st of N
  intonation  --param intonation=s for s in {0, 0.5, 1.5} vs 1: voiced log-f0
              std ratio, median within s +- 10 % for 0.5/1.5; s = 0 flattens
              to < 0.1 st (0.1 * ln2/12 in natural-log units)
  blend       --param timbre_blend=a for a in {0, .25, .5, .75, 1} toward
              --voice-b: ECAPA tgt_cos to the voice A references (targets.tsv)
              decreases monotonically; UTMOS at 0.5 >= 3.0
  click       --change T:name=value at T = 50 % of each utterance for the MVP
              params vs the unchanged conversion: the click_metric bound of
              the C++ unit tests (max |d| in [t, t+0.3 s] <= max(p99.9 |d|
              of the reference, p99.9 |d| of the stepped signal after it))
  presets     every built-in preset: WER <= Neutral + 3 pts

--skip-ml skips the checks that need Whisper/ECAPA/UTMOS (blend, presets);
skipped checks are marked SKIPPED in the report, never PASS. Writes
report.json + report.md under --out; exit 0 only when every executed check
passes.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

from fvml import audio, evallist, paths, vcfile

PITCH_SHIFTS = (-7.0, -3.0, 3.0, 7.0)
INTONATIONS = (0.0, 0.5, 1.5)
BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CLICK_PARAMS = (("pitch_shift", "+7"), ("intonation", "0"), ("timbre_blend", "1"),
                ("gain", "+6"), ("gate_threshold", "-30"), ("limiter", "0"))
PITCH_TOL_ST = 0.3
INTONATION_TOL = 0.10
INTONATION_ZERO_ST = 0.1 * math.log(2.0) / 12.0   # 0.1 semitone in natural-log units
UTMOS_BLEND_MIN = 3.0
WER_PRESET_SLACK = 0.03


# -- pure metrics --------------------------------------------------------------
def semitone_shift(med_f0_ref, med_f0_out):
    """Semitones from reference median F0 to output median F0."""
    return 12.0 * math.log2(med_f0_out / med_f0_ref)


def std_ratio(std_ref, std_out):
    """Voiced log-f0 std ratio (intonation scale estimate)."""
    return std_out / std_ref


def _p999_diffs(x):
    if len(x) < 2:
        return 0.0
    return float(np.percentile(np.abs(np.diff(np.asarray(x, np.float64))), 99.9))


def click_metric(ref, stepped, t, sr, window_s=0.3):
    """(value, bound): max |delta| of `stepped` inside [t, t + window_s] vs the
    bound used by the C++ click tests — max(p99.9 |delta| of `ref`,
    p99.9 |delta| of `stepped` after the window)."""
    i0 = max(1, int(round(t * sr)))
    i1 = min(len(stepped), int(round((t + window_s) * sr)))
    win = np.abs(np.diff(np.asarray(stepped[i0 - 1:i1], np.float64)))
    value = float(win.max()) if len(win) else 0.0
    bound = max(_p999_diffs(ref), _p999_diffs(np.asarray(stepped)[i1:]))
    return value, bound


def is_monotonic_decreasing(seq, eps=1e-9):
    return all(b <= a + eps for a, b in zip(seq, seq[1:]))


def controls_dod(results):
    """[(check, ok, detail)] over the executed checks; SKIPPED never counts."""
    return [(name, r.get("status") == "pass", r.get("detail", ""))
            for name, r in results.items() if r.get("status") != "skipped"]


# -- check implementations ------------------------------------------------------
def _med_f0_and_std(path, track_fn):
    from fvml import f0_ref
    med, std = f0_ref.stats(track_fn(audio.load(path)))
    return med, std


def check_pitch(rows, baseline_dir, convert, out_dir, track_fn, log):
    per_shift = {n: [] for n in PITCH_SHIFTS}
    dirs = {}
    for n in PITCH_SHIFTS:
        tag = f"pitch_{n:+g}".replace("+", "p").replace("-", "m")
        dirs[n] = convert(f"--param pitch_shift={n:g}".split(), out_dir / tag)
    for r in rows:
        med0, _ = _med_f0_and_std(baseline_dir / f"{r['utt_id']}.wav", track_fn)
        for n in PITCH_SHIFTS:
            med, _ = _med_f0_and_std(dirs[n] / f"{r['utt_id']}.wav", track_fn)
            if med == med and med0 == med0 and med > 0 and med0 > 0:
                per_shift[n].append(semitone_shift(med0, med))
    detail = {f"{n:+g}": round(float(np.median(v)), 3) if v else None
              for n, v in per_shift.items()}
    ok = all(v and abs(float(np.median(v)) - n) <= PITCH_TOL_ST
             for n, v in per_shift.items())
    return {"status": "pass" if ok else "fail",
            "measured_st": detail,
            "detail": f"median measured shift per target (tol {PITCH_TOL_ST} st): {detail}"}


def check_intonation(rows, baseline_dir, convert, out_dir, track_fn, log):
    pairs = {s: [] for s in INTONATIONS}        # (std_baseline, std_s) per utt
    dirs = {}
    for s in INTONATIONS:
        tag = f"intonation_{s:g}"
        dirs[s] = convert(f"--param intonation={s:g}".split(), out_dir / tag)
    for r in rows:
        _, std0 = _med_f0_and_std(baseline_dir / f"{r['utt_id']}.wav", track_fn)
        for s in INTONATIONS:
            _, std = _med_f0_and_std(dirs[s] / f"{r['utt_id']}.wav", track_fn)
            if std == std and std0 == std0 and std0 > 0:
                pairs[s].append((std0, std))
    detail, ok_all = {}, True
    for s, v in pairs.items():
        if not v:
            detail[f"{s:g}"] = None
            ok_all = False
        elif s > 0:
            med = float(np.median([std_ratio(a, b) for a, b in v]))
            detail[f"{s:g}"] = round(med, 3)
            ok_all &= abs(med - s) <= INTONATION_TOL * s
        else:   # s = 0: flattened — absolute voiced log-f0 std < 0.1 st
            med = float(np.median([b for _, b in v]))
            detail[f"{s:g}"] = round(med, 4)
            ok_all &= med < INTONATION_ZERO_ST
    return {"status": "pass" if ok_all else "fail",
            "measured": detail,
            "detail": f"median std ratio (0.5/1.5 within +-{INTONATION_TOL:.0%}) "
                      f"/ absolute std for s=0 (< {INTONATION_ZERO_ST:.4f} nat-log): {detail}"}


def check_click(rows, baseline_dir, vc_file, bundle, voice, runner, out_dir, log):
    import tempfile
    failures, measured = [], {}
    # Resampled inputs live in a temp dir, never inside a converted dir, so a
    # later fvml.evaluate scan can never pick them up as model output.
    with tempfile.TemporaryDirectory(prefix="fv_click_in_") as scratch_s:
        scratch = pathlib.Path(scratch_s)
        for name, value in CLICK_PARAMS:
            tag = f"click_{name}"
            cdir = out_dir / tag
            cdir.mkdir(parents=True, exist_ok=True)
            worst = None
            for r in rows:
                src = paths.resolve(r["path"])
                in_wav = src
                if not vcfile._is_16k_wav(src):
                    in_wav = scratch / f"{r['utt_id']}.wav"
                    audio.save(in_wav, audio.load(src))
                t = float(r["duration"]) / 2.0
                dst = cdir / f"{r['utt_id']}.wav"
                runner(vc_file, bundle, in_wav, dst, voice,
                       ("--change", f"{t:.3f}:{name}={value}"))
                ref = audio.load(baseline_dir / f"{r['utt_id']}.wav")
                stepped = audio.load(dst)
                val, bound = click_metric(ref, stepped, t, audio.SAMPLE_RATE)
                if worst is None or val > worst[0]:
                    worst = (val, bound, r["utt_id"])
                if val > bound:
                    failures.append(f"{r['utt_id']} {name}={value}: {val:.4f} > {bound:.4f}")
            measured[name] = {"max_delta": worst[0], "bound": worst[1],
                              "worst_utt": worst[2]} if worst else None
    return {"status": "pass" if not failures else "fail",
            "measured": measured,
            "detail": f"{len(failures)} click(s) over bound" if failures
                      else "all stepped conversions within the click bound"}


def check_blend(rows, source_list, evaluate_fn, convert, out_dir, voice_b,
                targets, voice, device, log):
    cosines, utmos = {}, {}
    for a in BLEND_ALPHAS:
        tag = f"blend_{a:g}"
        cdir = convert(["--voice-b", voice_b, "--param", f"timbre_blend={a:g}"],
                       out_dir / tag)
        res = evaluate_fn(source_list, cdir, targets=targets, voice=voice,
                          metrics=("spk", "utmos"), device=device)
        cosines[a] = res["tgt_cos"]
        utmos[a] = res["utmos"]
    ordered = [cosines[a] for a in BLEND_ALPHAS]
    ok = is_monotonic_decreasing(ordered) and utmos[0.5] >= UTMOS_BLEND_MIN
    return {"status": "pass" if ok else "fail",
            "tgt_cos": {f"{a:g}": round(cosines[a], 4) for a in BLEND_ALPHAS},
            "utmos_0.5": round(utmos[0.5], 3),
            "detail": f"tgt_cos {' -> '.join(f'{c:.3f}' for c in ordered)}; "
                      f"UTMOS @ 0.5 = {utmos[0.5]:.2f} (>= {UTMOS_BLEND_MIN})"}


def check_presets(source_list, evaluate_fn, convert, out_dir, presets_dir, device, log):
    preset_files = sorted(pathlib.Path(presets_dir).glob("*.json"))
    if not preset_files:
        return {"status": "skipped", "detail": f"no presets under {presets_dir}"}
    wers = {}
    for pf in preset_files:
        cdir = convert(["--preset", pf.stem], out_dir / f"preset_{pf.stem}")
        res = evaluate_fn(source_list, cdir, targets=None, voice=None,
                          metrics=("wer",), device=device)
        wers[pf.stem] = res["wer"]
    ref = wers.get("neutral", next(iter(wers.values())))
    bad = {k: v for k, v in wers.items() if v > ref + WER_PRESET_SLACK}
    return {"status": "pass" if not bad else "fail",
            "wer": {k: round(v, 4) for k, v in wers.items()},
            "detail": f"WER <= Neutral + {WER_PRESET_SLACK:.0%}"
                      + (f"; failed: {bad}" if bad else "")}


def run_checks(args, runner_convert=None, runner_one=None, evaluate_fn=None,
               track_fn=None, log=print):
    """Execute the requested checks -> {check: {status, detail, ...}}."""
    from fvml import f0_ref
    track_fn = track_fn or f0_ref.track
    rows = evallist.read_sources(paths.resolve(args.list))[: args.limit or None]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runner_one is None:
        runner_one = vcfile.convert

    def convert(extra_args, dest):
        dest = pathlib.Path(dest)
        if runner_convert is not None:
            return runner_convert(dest, extra_args)
        return vcfile.convert_list(vc_file, args.bundle, rows, dest,
                                   voice=args.voice, args=extra_args) and dest

    def evaluate_default(src_list, cdir, targets, voice, metrics, device):
        from fvml import evaluate
        cfg = evaluate.EvalConfig(sources=src_list, converted=cdir,
                                  targets=targets, voice=voice,
                                  metrics=metrics, device=device)
        return evaluate.evaluate(cfg, pathlib.Path(cdir) / "_eval",
                                 log=lambda *a: None)

    if evaluate_fn is None:
        evaluate_fn = evaluate_default

    vc_file = vcfile.find_vc_file()
    if vc_file is None:
        raise FileNotFoundError("no fv_vc_file: set FLUXVOICE_VC_FILE or build an ORT tree")
    log(f"[controls] {len(rows)} utterances from {args.list}; fv_vc_file {vc_file}")
    baseline_dir = convert([], out_dir / "baseline")
    if runner_convert is None:
        baseline_dir = pathlib.Path(baseline_dir)

    wanted = set(args.checks.split(",")) if args.checks else set()
    checks = {"pitch", "intonation", "blend", "click", "presets"} if not wanted else wanted
    results = {}
    if "pitch" in checks:
        results["pitch"] = check_pitch(rows, pathlib.Path(baseline_dir), convert,
                                       out_dir, track_fn, log)
    if "intonation" in checks:
        results["intonation"] = check_intonation(rows, pathlib.Path(baseline_dir),
                                                 convert, out_dir, track_fn, log)
    if "click" in checks:
        results["click"] = check_click(rows, pathlib.Path(baseline_dir), vc_file,
                                       args.bundle, args.voice, runner_one, out_dir, log)
    if "blend" in checks:
        if args.skip_ml:
            results["blend"] = {"status": "skipped",
                                "detail": "--skip-ml: needs ECAPA + UTMOS"}
        elif not args.voice_b:
            results["blend"] = {"status": "skipped",
                                "detail": "needs --voice-b to blend toward"}
        else:
            results["blend"] = check_blend(
                rows, paths.resolve(args.list), evaluate_fn,
                convert, out_dir, args.voice_b, args.targets, args.voice, args.device, log)
    if "presets" in checks:
        if args.skip_ml:
            results["presets"] = {"status": "skipped",
                                  "detail": "--skip-ml: needs Whisper (WER)"}
        else:
            results["presets"] = check_presets(
                paths.resolve(args.list), evaluate_fn,
                convert, out_dir, args.presets_dir, args.device, log)
    return results


def write_report(results, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps({"checks": results,
                    "verdict": "PASS" if all(r.get("status") != "fail" for r in results.values())
                               else "FAIL"}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage 09 offline DoD — voice controls", "",
             "| check | status | detail |", "|---|---|---|"]
    for name, r in results.items():
        status = {"pass": "PASS", "fail": "FAIL"}.get(r.get("status"), "SKIPPED")
        lines.append(f"| {name} | {status} | {r.get('detail', '')} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--bundle", required=True, help="model bundle directory for fv_vc_file")
    p.add_argument("--voice", required=True, help="target voice A")
    p.add_argument("--voice-b", help="target voice B (required by the blend check)")
    p.add_argument("--list", default=str(paths.EVAL_LISTS / "test_clean_200.tsv"),
                   help="source eval list TSV")
    p.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                   help="target reference clips TSV (blend check)")
    p.add_argument("--presets-dir", default=str(paths.REPO_ROOT / "presets"))
    p.add_argument("--limit", type=int, help="first N list rows (smoke runs)")
    p.add_argument("--out", required=True)
    p.add_argument("--checks", default="pitch,intonation,blend,click,presets")
    p.add_argument("--skip-ml", action="store_true",
                   help="skip checks needing Whisper/ECAPA/UTMOS (blend, presets)")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    try:
        results = run_checks(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[eval_controls] error: {e}", file=sys.stderr)
        return 1
    write_report(results, args.out)
    for name, r in results.items():
        print(f"[controls] {name}: {r.get('status', '?').upper()} — {r.get('detail', '')}")
    return 0 if all(r.get("status") != "fail" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
