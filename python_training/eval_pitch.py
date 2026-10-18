#!/usr/bin/env python3
"""Stage 03 measurements for the causal YIN tracker (fvml/features.py).

  eval_pitch.py priors    log-f0 / log-energy population priors over a manifest
                          (the constants at the top of fvml/features.py)
  eval_pitch.py accuracy  gross pitch error, voicing decision accuracy and
                          voiced log-f0 RMSE in cents against the offline
                          reference tracker (fvml/f0_ref.py: Praat AC), per
                          lookahead variant

Each tracker frame is compared with the reference frame nearest to the centre
of its analysis window. GPE counts frames voiced in both whose f0 is > 20% off;
the fine RMSE is over the remaining frames voiced in both (an octave error is
1200 cents and belongs in GPE, not in the RMSE).

The accuracy set is 30 min of dev-clean from every 6th manifest row starting
at row 3, taken round-robin over the 40 speakers. The voiced-decision
constants were tuned on rows 0, 60, 120, ... (45 utterances), which this set
never contains.

Usage:
    eval_pitch.py priors   [--manifest librispeech-train-clean-100] [--hours 100]
    eval_pitch.py accuracy [--minutes 30] [--lookahead 0 1] [--out report.json]
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# One BLAS thread per worker process: the workers are the parallelism, and
# OpenBLAS's per-thread buffers otherwise exhaust memory on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from fvml import audio, f0_ref, paths  # noqa: E402
from fvml import features as F  # noqa: E402
from fvml.datasets import read_jsonl  # noqa: E402

TUNING_STRIDE = 60     # rows 0, 60, ... of dev-clean tuned the voiced decision
ACCURACY_STRIDE, ACCURACY_OFFSET = 6, 3


def manifest_rows(name):
    path = paths.data_root() / "manifests" / f"{name}.jsonl"
    if not path.is_file():
        sys.exit(f"missing manifest {path}: run prepare_data.py first")
    return list(read_jsonl(path))


def round_robin(rows):
    """Interleave speakers so a time budget covers all of them."""
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)
    queues = list(by_spk.values())
    out = []
    for i in range(max(map(len, queues), default=0)):
        out += [q[i] for q in queues if i < len(q)]
    return out


def take_seconds(rows, seconds):
    out, have = [], 0.0
    for r in rows:
        if have >= seconds:
            break
        out.append(r)
        have += r["duration"]
    return out, have


# --------------------------------------------------------------------------- priors

def _prior_job(path):
    o = F.extract(audio.load(paths.resolve(path)))
    voiced = o[:, 0] > 0
    lf = np.log(o[voiced, 0])
    e = o[:, 3]
    return len(lf), lf.sum(), (lf ** 2).sum(), len(e), e.sum(), (e ** 2).sum()


def priors(args):
    rows = manifest_rows(args.manifest)
    if args.hours:
        rows = rows[:: max(1, int(sum(r["duration"] for r in rows) / 3600 / args.hours))]
    with ProcessPoolExecutor(args.jobs) as ex:
        parts = list(ex.map(_prior_job, [r["path"] for r in rows], chunksize=8))
    s = np.sum(np.array(parts, dtype=np.float64), axis=0)

    def mean_std(n, s1, s2):
        m = s1 / n
        return m, math.sqrt(max(s2 / n - m * m, 0.0))

    lf_m, lf_s = mean_std(*s[:3])
    e_m, e_s = mean_std(*s[3:])
    result = {
        "manifest": args.manifest, "utterances": len(rows),
        "hours": sum(r["duration"] for r in rows) / 3600,
        "voiced_frames": int(s[0]), "frames": int(s[3]),
        "logf0_prior_mean": round(lf_m, 3), "logf0_prior_std": round(lf_s, 3),
        "energy_prior_mean": round(e_m, 3), "energy_prior_std": round(e_s, 3),
        "median_f0_hz_of_mean": round(math.exp(lf_m), 1),
    }
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- accuracy

def _reference(x):
    import parselmouth
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=audio.SAMPLE_RATE)
    pitch = snd.to_pitch_ac(time_step=f0_ref.HOP_S, pitch_floor=f0_ref.FLOOR_HZ,
                            pitch_ceiling=f0_ref.CEILING_HZ)
    return np.asarray(pitch.xs()), np.nan_to_num(pitch.selected_array["frequency"], nan=0.0)


def _accuracy_job(job):
    path, lookaheads = job
    x = audio.load(paths.resolve(path))
    xs, ref = _reference(x)
    out = {}
    for la in lookaheads:
        cfg = F.PitchConfig(lookahead_frames=la)
        o = F.extract(x, cfg)
        idx = np.round((F.window_centre_seconds(cfg, len(o)) - xs[0]) / f0_ref.HOP_S).astype(int)
        ok = (idx >= 0) & (idx < len(xs))
        out[la] = (o[ok, 0].astype(np.float32), ref[idx[ok]].astype(np.float32))
    return out


def scores(f0, ref):
    v, rv = f0 > 0, ref > 0
    both = v & rv
    rel = np.abs(f0[both] - ref[both]) / ref[both]
    gross = rel > 0.2
    cents = 1200.0 * np.log2(f0[both] / ref[both])
    return {
        "frames": int(len(f0)),
        "ref_voiced": round(float(rv.mean()), 4),
        "voicing_accuracy": round(float((v == rv).mean()), 4),
        "voicing_recall": round(float(both.sum() / max(rv.sum(), 1)), 4),
        "voicing_false_alarm": round(float((v & ~rv).sum() / max((~rv).sum(), 1)), 4),
        "gpe": round(float(gross.mean()), 4),
        "gpe_octave_down": round(float((f0[both] < 0.8 * ref[both]).mean()), 4),
        "gpe_octave_up": round(float((f0[both] > 1.2 * ref[both]).mean()), 4),
        "fine_rmse_cents": round(float(np.sqrt(np.mean(cents[~gross] ** 2))), 1),
        "rmse_cents_all": round(float(np.sqrt(np.mean(cents ** 2))), 1),
    }


DOD = {"gpe": 0.05, "voicing_accuracy": 0.90, "fine_rmse_cents": 50.0}


def accuracy(args):
    rows = manifest_rows("librispeech-dev-clean")
    held_out = [r for i, r in enumerate(rows)
                if i % ACCURACY_STRIDE == ACCURACY_OFFSET and i % TUNING_STRIDE != 0]
    rows, seconds = take_seconds(round_robin(held_out), args.minutes * 60)
    with ProcessPoolExecutor(args.jobs) as ex:
        parts = list(ex.map(_accuracy_job, [(r["path"], tuple(args.lookahead)) for r in rows],
                            chunksize=4))
    report = {"reference": f0_ref.TRACKER, "set": "librispeech dev-clean, held-out rows",
              "utterances": len(rows), "minutes": round(seconds / 60, 1),
              "speakers": len({r["speaker"] for r in rows}), "dod": DOD, "variants": {}}
    for la in args.lookahead:
        f0 = np.concatenate([p[la][0] for p in parts])
        ref = np.concatenate([p[la][1] for p in parts])
        s = scores(f0, ref)
        s["pass"] = (s["gpe"] <= DOD["gpe"] and s["voicing_accuracy"] >= DOD["voicing_accuracy"]
                     and s["fine_rmse_cents"] <= DOD["fine_rmse_cents"])
        report["variants"][f"L{la}"] = s
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("priors", help="population priors over a manifest")
    pp.add_argument("--manifest", default="librispeech-train-clean-100",
                    help="manifest name under $FLUXVOICE_DATA/manifests (default %(default)s)")
    pp.add_argument("--hours", type=float, help="subsample evenly to about this many hours")
    pa = sub.add_parser("accuracy", help="GPE / voicing / RMSE vs. the reference tracker")
    pa.add_argument("--minutes", type=float, default=30.0)
    pa.add_argument("--lookahead", type=int, nargs="+", default=[0, 1])
    pa.add_argument("--out", help="also write the JSON report here")
    for sp in (pp, pa):
        sp.add_argument("--jobs", type=int, default=min(6, max(1, (os.cpu_count() or 2) // 2)))
    args = p.parse_args()
    {"priors": priors, "accuracy": accuracy}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
