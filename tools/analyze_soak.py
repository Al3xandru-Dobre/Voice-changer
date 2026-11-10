#!/usr/bin/env python3
"""Check a fluxvoice soak-test log for glitches, latency creep and leaks.

Run fluxvoice with --stats and keep its stdout:

    fluxvoice --input I --output O --duration 1800 --stats 10 > soak.log
    analyze_soak.py soak.log

It reads the "[stats] t=... frames=... overruns=... underruns=... drops=...
queued=... rss_kb=..." lines and, ignoring the first --warmup seconds, checks
the Milestone 1 / Stage 06 soak criteria:

  * overrun, underrun and drop counters stay flat after startup
  * queued audio stays constant (in this duplex design it only changes when a
    counter does: queued = prefilled + underruns - overruns - drops - skipped)
  * resident memory does not grow by more than --rss-tolerance-kb

and reports the callback rate measured against the system clock.

With --vc, also require inference telemetry, no post-warm-up misses/errors,
p95 inference <= half a hop, and queued within target +/- one hop. Intentional
output skips may grow in VC mode; each must still conserve samples. Old M1 logs
without skipped/prefilled are interpreted as zero for those two counters.
Use a single-duplex stream without route restarts; independent native loopback
callbacks and queue discards require different accounting/baselines.

Note: with one miniaudio duplex device, clock drift between two different
capture/playback devices is absorbed inside miniaudio / the OS and never shows
in these counters; measure it end-to-end with `fv_probe latency` (Stage 06).

Usage:
    analyze_soak.py soak.log [--warmup 10] [--rate 16000] [--rss-tolerance-kb 1024]
    analyze_soak.py --self-test
    analyze_soak.py vc.log --vc --latency-target-ms 55 --hop-ms 20

Exit status: 0 = criteria met, 1 = violated or not enough data.
Standard library only.
"""
import argparse
import math
import re
import sys

STATS_RE = re.compile(r"^\[stats\] (.*)$")
CLIENT_RE = re.compile(r"^Client\s*:\s*(\d+) Hz")
LEGACY_FIELDS = ("t", "frames", "overruns", "underruns", "drops", "queued", "rss_kb")
FIELDS = LEGACY_FIELDS + ("skipped", "prefilled")
VC_FIELDS = ("inf_p50_us", "inf_p95_us", "inf_max_us", "misses", "errors", "lat_ms",
             "skipped", "prefilled")


def parse(lines):
    """Return (rows, client_rate or None)."""
    rows, rate = [], None
    for line in lines:
        line = line.strip()
        m = CLIENT_RE.match(line)
        if m:
            rate = int(m.group(1))
        m = STATS_RE.match(line)
        if not m:
            continue
        kv = dict(item.split("=", 1) for item in m.group(1).split())
        if not all(f in kv for f in LEGACY_FIELDS):
            raise ValueError("incomplete [stats] line")
        row = {f: float(kv[f]) if f in ("t", "lat_ms") else int(kv[f])
               for f in set(FIELDS + VC_FIELDS) if f in kv}
        if any((isinstance(v, float) and not math.isfinite(v)) or (f != "queued" and v < 0)
               for f, v in row.items()):
            raise ValueError("non-finite or negative [stats] field")
        rows.append(row)
    return rows, rate


def slope(xs, ys):
    """Least-squares dy/dx."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def analyze(rows, rate, warmup, rss_tol_kb, vc=False, target_ms=None, hop_ms=None):
    steady = [r for r in rows if r["t"] >= warmup]
    if len(steady) < 2:
        return {"ok": False, "reason": f"need >= 2 [stats] lines after {warmup} s warm-up"}
    if vc:
        if not rate or rate <= 0 or target_ms is None or hop_ms is None or target_ms <= 0 or hop_ms <= 0:
            return {"ok": False, "reason": "--vc requires a rate, positive target and hop"}
        if not all(all(f in r for f in VC_FIELDS) for r in steady):
            return {"ok": False, "reason": "--vc requires complete inference/skipped/prefilled telemetry"}
    a, b = steady[0], steady[-1]
    minutes = (b["t"] - a["t"]) / 60.0
    ts = [r["t"] for r in steady]

    res = {
        "steady_minutes": round(minutes, 2),
        "overruns_delta": b["overruns"] - a["overruns"],
        "underruns_delta": b["underruns"] - a["underruns"],
        "drops_delta": b["drops"] - a["drops"],
        "skipped_delta": b.get("skipped", 0) - a.get("skipped", 0),
        "queued_samples": f"{min(r['queued'] for r in steady)} .. {max(r['queued'] for r in steady)}",
        "rss_kb": f"{a['rss_kb']} -> {b['rss_kb']} (max {max(r['rss_kb'] for r in steady)})",
        "rss_slope_kb_per_min": round(slope([t / 60 for t in ts], [r["rss_kb"] for r in steady]), 2),
    }
    callback_rate = slope(ts, [r["frames"] for r in steady])
    res["callback_rate_hz"] = round(callback_rate, 2)
    if rate:
        res["queued_ms"] = round(1000.0 * b["queued"] / rate, 1)
        res["callback_rate_ppm"] = round(1e6 * (callback_rate - rate) / rate, 1)

    reasons = []
    flat_fields = ("overruns", "underruns", "drops", "prefilled") + (
        ("misses", "errors") if vc else ("skipped",))
    for field in flat_fields:
        values = [r.get(field, 0) for r in steady]
        if min(values) != max(values):
            reasons.append(f"{field} changed after warm-up")
    if any(r["queued"] != r.get("prefilled", 0) + r["underruns"] - r["overruns"] -
           r["drops"] - r.get("skipped", 0) for r in steady):
        reasons.append("queued violates sample conservation")
    if not vc and min(r["queued"] for r in steady) != max(r["queued"] for r in steady):
        reasons.append("queued changed after warm-up")
    if any(right["t"] < left["t"] for left, right in zip(steady, steady[1:])) or b["t"] <= a["t"]:
        reasons.append("timestamps do not advance monotonically")
    counter_fields = ("frames", "overruns", "underruns", "drops", "skipped", "prefilled",
                      "misses", "errors")
    if any(right.get(f, 0) < left.get(f, 0) for left, right in zip(steady, steady[1:])
           for f in counter_fields):
        reasons.append("counters decreased (stream reset or malformed log)")
    growth = max(r["rss_kb"] for r in steady) - a["rss_kb"]
    if growth > rss_tol_kb:
        reasons.append(f"RSS grew {growth} KB (> {rss_tol_kb} KB)")
    if b["frames"] <= a["frames"]:
        reasons.append("frames stopped advancing (stream stalled)")
    if vc:
        target, hop = target_ms * rate / 1000, hop_ms * rate / 1000
        if any(not target - hop <= r["queued"] <= target + hop for r in steady):
            reasons.append("queued outside target +/- one hop")
        if any(r["inf_p95_us"] > hop_ms * 1000 / 2 for r in steady):
            reasons.append("inference p95 exceeds 50% of hop")
        if any(not math.isfinite(r["lat_ms"]) or r["lat_ms"] < 0 for r in steady):
            reasons.append("invalid latency estimate")
        res["misses_delta"] = b["misses"] - a["misses"]
        res["errors_delta"] = b["errors"] - a["errors"]
    res["ok"] = not reasons
    res["reason"] = "; ".join(reasons) or "clean"
    return res


def fake_log(minutes, underrun_at=None, leak_kb_per_min=0, rate=16000):
    lines = [f"Client  : {rate} Hz mono f32, requested period 20 ms"]
    for i in range(int(minutes * 6) + 1):  # one line per 10 s
        t = 10.0 * i + 0.02
        under = 320 + (320 if underrun_at is not None and t >= underrun_at else 0)
        rss = 6000 + int(leak_kb_per_min * t / 60)
        lines.append(f"[stats] t={t:.1f} frames={int(t * rate)} overruns=0 underruns={under} "
                     f"drops=0 queued={under} rss_kb={rss}")
    return lines


def fake_vc_log():
    rows, _ = parse(fake_log(1))
    for i, r in enumerate(rows):
        r.update(prefilled=640, skipped=min(i * 80, 320), inf_p50_us=1000,
                 inf_p95_us=5000, inf_max_us=9000, misses=0, errors=0, lat_ms=75.5)
        r["queued"] = r["prefilled"] + r["underruns"] - r["skipped"]
    return ["Client  : 16000 Hz mono f32"] + [
        "[stats] " + " ".join(f"{k}={v}" for k, v in r.items()) for r in rows]


def self_test():
    cases = [
        ("clean 30 min", fake_log(30), True),
        ("underrun at 12 min", fake_log(30, underrun_at=720), False),
        ("leak 100 KB/min", fake_log(30, leak_kb_per_min=100), False),
        ("startup underrun only", fake_log(30, underrun_at=5), True),
        ("too short", fake_log(0.1), False),
    ]
    ok = True
    for name, lines, expect in cases:
        rows, rate = parse(lines)
        r = analyze(rows, rate, warmup=10, rss_tol_kb=1024)
        good = r["ok"] == expect
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:22s} -> {r['reason']}")
    rows, rate = parse(fake_log(30))
    r = analyze(rows, rate, 10, 1024)
    good = r["callback_rate_hz"] == 16000.0 and r["queued_ms"] == 20.0
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'rate + queued_ms':22s} -> "
          f"{r['callback_rate_hz']} Hz, {r['queued_ms']} ms")
    vc_rows, rate = parse(fake_vc_log())
    vc_cases = [("VC quiet skips", vc_rows, True),
                ("VC missing telemetry", rows, False)]
    for name, field, value, expect in [
        ("late miss", "misses", 1, False),
        ("late error", "errors", 1, False),
        ("wrong identity", "queued", 801, False),
        ("p95 at half hop", "inf_p95_us", 10000, True),
        ("p95 above half hop", "inf_p95_us", 10001, False),
    ]:
        changed = [dict(r) for r in vc_rows]
        changed[-1][field] = value
        vc_cases.append((name, changed, expect))
    for shift in (-640, 800):
        changed = [dict(r, prefilled=r["prefilled"] + shift, queued=r["queued"] + shift)
                   for r in vc_rows]
        vc_cases.append((f"VC queue shift {shift}", changed, False))
    changed = [dict(r) for r in vc_rows]
    changed[-1]["skipped"] = 0
    changed[-1]["queued"] = changed[-1]["prefilled"] + changed[-1]["underruns"]
    vc_cases.append(("VC counter reset", changed, False))
    for name, data, expect in vc_cases:
        r = analyze(data, rate, 10, 5120, vc=True, target_ms=50, hop_ms=20)
        good = r["ok"] == expect
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:22s} -> {r['reason']}")
    try:
        parse(["[stats] t=nan frames=1 overruns=0 underruns=0 drops=0 queued=0 rss_kb=1"])
        ok = False
    except ValueError:
        print("  PASS  malformed stats rejected")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", nargs="?")
    p.add_argument("--warmup", type=float, default=10.0, help="seconds to ignore (default 10)")
    p.add_argument("--rate", type=int, default=None,
                   help="client sample rate (default: read from the log's 'Client' line)")
    p.add_argument("--rss-tolerance-kb", type=int, default=None,
                   help="allowed RSS growth (default: 1024 KB, or 5120 KB with --vc)")
    p.add_argument("--vc", action="store_true", help="check Mode A inference and governor telemetry")
    p.add_argument("--latency-target-ms", type=float)
    p.add_argument("--hop-ms", type=float)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.log:
        p.error("log file required")
    if not math.isfinite(args.warmup) or args.warmup < 0:
        p.error("--warmup must be finite and non-negative")
    if args.rate is not None and args.rate <= 0:
        p.error("--rate must be positive")
    if args.rss_tolerance_kb is not None and args.rss_tolerance_kb < 0:
        p.error("--rss-tolerance-kb must be non-negative")
    for name in ("latency_target_ms", "hop_ms"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.vc and (args.latency_target_ms is None or args.hop_ms is None):
        p.error("--vc requires --latency-target-ms and --hop-ms")
    if not args.vc and (args.latency_target_ms is not None or args.hop_ms is not None):
        p.error("target/hop options require --vc")
    try:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            rows, rate = parse(f)
    except (OSError, ValueError) as exc:
        print(f"Invalid soak log: {exc}", file=sys.stderr)
        return 1
    tolerance = args.rss_tolerance_kb
    if tolerance is None:
        tolerance = 5120 if args.vc else 1024
    r = analyze(rows, args.rate or rate, args.warmup, tolerance, args.vc,
                args.latency_target_ms, args.hop_ms)
    for k, v in r.items():
        print(f"{k:22s}: {v}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
