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
    counter does: queued = underruns - overruns - drops)
  * resident memory does not grow by more than --rss-tolerance-kb

and reports the callback rate measured against the system clock.

Note: with one miniaudio duplex device, clock drift between two different
capture/playback devices is absorbed inside miniaudio / the OS and never shows
in these counters; measure it end-to-end with `fv_probe latency` (Stage 06).

Usage:
    analyze_soak.py soak.log [--warmup 10] [--rate 16000] [--rss-tolerance-kb 1024]
    analyze_soak.py --self-test

Exit status: 0 = criteria met, 1 = violated or not enough data.
Standard library only.
"""
import argparse
import re
import sys

STATS_RE = re.compile(r"^\[stats\] (.*)$")
CLIENT_RE = re.compile(r"^Client\s*:\s*(\d+) Hz")
FIELDS = ("t", "frames", "overruns", "underruns", "drops", "queued", "rss_kb")


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
        if all(f in kv for f in FIELDS):
            rows.append({f: float(kv[f]) if f == "t" else int(kv[f]) for f in FIELDS})
    return rows, rate


def slope(xs, ys):
    """Least-squares dy/dx."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def analyze(rows, rate, warmup, rss_tol_kb):
    steady = [r for r in rows if r["t"] >= warmup]
    if len(steady) < 2:
        return {"ok": False, "reason": f"need >= 2 [stats] lines after {warmup} s warm-up"}
    a, b = steady[0], steady[-1]
    minutes = (b["t"] - a["t"]) / 60.0
    ts = [r["t"] for r in steady]

    res = {
        "steady_minutes": round(minutes, 2),
        "overruns_delta": b["overruns"] - a["overruns"],
        "underruns_delta": b["underruns"] - a["underruns"],
        "drops_delta": b["drops"] - a["drops"],
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
    for k in ("overruns_delta", "underruns_delta", "drops_delta"):
        if res[k]:
            reasons.append(f"{k.split('_')[0]} grew by {res[k]} after warm-up")
    if b["rss_kb"] - a["rss_kb"] > rss_tol_kb:
        reasons.append(f"RSS grew {b['rss_kb'] - a['rss_kb']} KB (> {rss_tol_kb} KB)")
    if b["frames"] <= a["frames"]:
        reasons.append("frames stopped advancing (stream stalled)")
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
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", nargs="?")
    p.add_argument("--warmup", type=float, default=10.0, help="seconds to ignore (default 10)")
    p.add_argument("--rate", type=int, default=None,
                   help="client sample rate (default: read from the log's 'Client' line)")
    p.add_argument("--rss-tolerance-kb", type=int, default=1024)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.log:
        p.error("log file required")
    with open(args.log, encoding="utf-8", errors="replace") as f:
        rows, rate = parse(f)
    r = analyze(rows, args.rate or rate, args.warmup, args.rss_tolerance_kb)
    for k, v in r.items():
        print(f"{k:22s}: {v}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
