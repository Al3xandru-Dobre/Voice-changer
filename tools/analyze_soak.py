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

With --s2ut (Milestone 3 Mode B; mutually exclusive with --vc), every --vc check
applies except the queued band, which widens to target - hop <= queued <=
target + --burst-cushion-ms + hop (the governor raises its target by the burst
cushion while translating; default 240 ms = 3 x 80 ms chunks). Additionally the
[stats] lines must carry s2ut_r/s2ut_w/s2ut_f/lag_ms/s2ut_utt/s2ut_cap; errors
must be 0 on every post-warm-up row; the s2ut_* counters must not decrease;
s2ut_cap <= s2ut_utt; and lag_ms must return to 0 at least once after
warm-up whenever utterances completed. Prints counter deltas, max lag and the
cap-terminated fraction (cap/utt). lag_ms may be negative (tail flush emits up
to ceil(1.5 * t_src) target chunks); it must be finite.

With --check-format <log>, only the [stats] line shape is validated, not the
soak criteria above: every line must carry the legacy fields plus skipped and
prefilled; a line with any inference field must carry the whole VC set; a line
with any s2ut field the whole S2UT set; all values must be finite and t
non-decreasing; and the file must hold at least one [stats] line (other lines
are ignored, as in the analysis mode). Prints a one-line count summary
(legacy lines carry neither field set). The GUI's Copy [stats] line / stats
log writes the same format as the CLI (fluxvoice::format_stats_line,
src/pipeline/HostStats.hpp), so this accepts GUI-produced soak logs.

Note: with one miniaudio duplex device, clock drift between two different
capture/playback devices is absorbed inside miniaudio / the OS and never shows
in these counters; measure it end-to-end with `fv_probe latency` (Stage 06).

Usage:
    analyze_soak.py soak.log [--warmup 10] [--rate 16000] [--rss-tolerance-kb 1024]
    analyze_soak.py --self-test
    analyze_soak.py --check-format gui_stats.log
    analyze_soak.py vc.log --vc --latency-target-ms 55 --hop-ms 20
    analyze_soak.py s2ut.log --s2ut --latency-target-ms 130 --hop-ms 80 [--burst-cushion-ms 240]

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
S2UT_COUNTERS = ("s2ut_r", "s2ut_w", "s2ut_f", "s2ut_utt", "s2ut_cap")
S2UT_FIELDS = S2UT_COUNTERS[:3] + ("lag_ms",) + S2UT_COUNTERS[3:]
FLOAT_FIELDS = ("t", "lat_ms", "lag_ms")
# Inference fields that mark a line as VC (skipped/prefilled alone do not:
# the GUI's Bypass emits them on plain legacy lines too).
VC_TRIGGER_FIELDS = tuple(f for f in VC_FIELDS if f not in FIELDS)


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
        row = {f: float(kv[f]) if f in FLOAT_FIELDS else int(kv[f])
               for f in set(FIELDS + VC_FIELDS + S2UT_FIELDS) if f in kv}
        if any((isinstance(v, float) and not math.isfinite(v)) or (f not in ("queued", "lag_ms") and v < 0)
               for f, v in row.items()):
            raise ValueError("non-finite or negative [stats] field")
        rows.append(row)
    return rows, rate


def check_format(lines):
    """Validate only the [stats] line shape (not the soak criteria).

    Returns (summary, lineno, reason): reason is None when every line is
    well-formed and summary counts the [stats] lines (a line carrying both the
    VC and S2UT sets counts in both; legacy lines carry neither); otherwise
    lineno is the 1-based file line of the first offender (None when the file
    holds no [stats] line at all) and reason says why. Reuses STATS_RE/parse()
    per line for field extraction, finiteness and sign checks.
    """
    total = legacy = vc = s2ut = 0
    prev_t = None
    for lineno, raw in enumerate(lines, 1):
        if not STATS_RE.match(raw.strip()):
            continue  # non-[stats] lines are ignored, exactly as parse() does
        try:
            (row,), _ = parse([raw])
        except ValueError as exc:
            return None, lineno, str(exc)
        total += 1
        missing = [f for f in ("skipped", "prefilled") if f not in row]
        if missing:
            return None, lineno, "missing " + "/".join(missing)
        is_vc = any(f in row for f in VC_TRIGGER_FIELDS)
        if is_vc:
            missing = [f for f in VC_FIELDS if f not in row]
            if missing:
                return None, lineno, "VC line missing " + "/".join(missing)
            vc += 1
        is_s2ut = any(f in row for f in S2UT_FIELDS)
        if is_s2ut:
            missing = [f for f in S2UT_FIELDS if f not in row]
            if missing:
                return None, lineno, "S2UT line missing " + "/".join(missing)
            s2ut += 1
        if not is_vc and not is_s2ut:
            legacy += 1
        if prev_t is not None and row["t"] < prev_t:
            return None, lineno, f"t={row['t']:g} decreases (previous t={prev_t:g})"
        prev_t = row["t"]
    if not total:
        return None, None, "no [stats] lines"
    return {"lines": total, "legacy": legacy, "vc": vc, "s2ut": s2ut}, None, None


def slope(xs, ys):
    """Least-squares dy/dx."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def analyze(rows, rate, warmup, rss_tol_kb, vc=False, target_ms=None, hop_ms=None,
            s2ut=False, burst_cushion_ms=0.0):
    vc = vc or s2ut
    mode = "--s2ut" if s2ut else "--vc"
    steady = [r for r in rows if r["t"] >= warmup]
    if len(steady) < 2:
        return {"ok": False, "reason": f"need >= 2 [stats] lines after {warmup} s warm-up"}
    if vc:
        if not rate or rate <= 0 or target_ms is None or hop_ms is None or target_ms <= 0 or hop_ms <= 0:
            return {"ok": False, "reason": f"{mode} requires a rate, positive target and hop"}
        if s2ut and (not math.isfinite(burst_cushion_ms) or burst_cushion_ms < 0):
            return {"ok": False, "reason": "--burst-cushion-ms must be finite and non-negative"}
        if not all(all(f in r for f in VC_FIELDS) for r in steady):
            return {"ok": False, "reason": f"{mode} requires complete inference/skipped/prefilled telemetry"}
        if s2ut and not all(all(f in r for f in S2UT_FIELDS) for r in steady):
            return {"ok": False, "reason": "--s2ut requires complete s2ut_r/w/f/lag_ms/utt/cap telemetry"}
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
                      "misses", "errors") + (S2UT_COUNTERS if s2ut else ())
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
        if s2ut:
            cushion = burst_cushion_ms * rate / 1000
            if any(not target - hop <= r["queued"] <= target + cushion + hop for r in steady):
                reasons.append("queued outside target - hop .. target + burst cushion + hop")
        elif any(not target - hop <= r["queued"] <= target + hop for r in steady):
            reasons.append("queued outside target +/- one hop")
        if any(r["inf_p95_us"] > hop_ms * 1000 / 2 for r in steady):
            reasons.append("inference p95 exceeds 50% of hop")
        if any(not math.isfinite(r["lat_ms"]) or r["lat_ms"] < 0 for r in steady):
            reasons.append("invalid latency estimate")
        res["misses_delta"] = b["misses"] - a["misses"]
        res["errors_delta"] = b["errors"] - a["errors"]
    if s2ut:
        if any(r["errors"] != 0 for r in steady):
            reasons.append("errors nonzero after warm-up")
        if any(r["s2ut_cap"] > r["s2ut_utt"] for r in steady):
            reasons.append("s2ut_cap exceeds s2ut_utt")
        utts = b["s2ut_utt"] - a["s2ut_utt"]
        if utts > 0 and not any(r["lag_ms"] == 0 for r in steady):
            reasons.append(f"lag_ms never returned to 0 although {utts} utterance(s) completed")
        for f in S2UT_COUNTERS:
            res[f + "_delta"] = b[f] - a[f]
        res["lag_ms_max"] = max(r["lag_ms"] for r in steady)
        res["cap_fraction"] = (round(res["s2ut_cap_delta"] / utts, 3) if utts > 0 else "n/a")
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


def fake_s2ut_log():
    rows, _ = parse(fake_vc_log())
    for i, r in enumerate(rows):
        r.update(s2ut_r=10 * i, s2ut_w=5 * i, s2ut_f=2 * i, s2ut_utt=i, s2ut_cap=i // 3,
                 lag_ms=0.0 if i % 2 == 0 else 160.0)
    return ["Client  : 16000 Hz mono f32"] + [
        "[stats] " + " ".join(f"{k}={v}" for k, v in r.items()) for r in rows]


# Literal GUI-produced [stats] lines in exactly the shape
# fluxvoice::format_stats_line (src/pipeline/HostStats.hpp) emits them:
# std::ios::fixed with precision(1) doubles (t, lat_ms), plain integers
# elsewhere (rss_kb is the integer division s.rss / 1024). The GUI's Bypass
# never sets vc/s2ut, so those lines stop after rss_kb but still carry
# skipped/prefilled; Mode A adds the inference block.
GUI_BYPASS_LOG = [
    "Client  : 16000 Hz mono f32, requested period 20 ms",
    "[stats] t=1.0 frames=16000 overruns=0 underruns=320 drops=0 queued=320 "
    "skipped=0 prefilled=0 rss_kb=13312",
    "[stats] t=11.0 frames=176000 overruns=0 underruns=320 drops=0 queued=320 "
    "skipped=0 prefilled=0 rss_kb=13312",
    "[stats] t=21.0 frames=336000 overruns=0 underruns=320 drops=0 queued=320 "
    "skipped=0 prefilled=0 rss_kb=13344",
    "[stats] t=31.0 frames=496000 overruns=0 underruns=320 drops=0 queued=320 "
    "skipped=0 prefilled=0 rss_kb=13344",
]

GUI_VC_LOG = [
    "Client  : 16000 Hz mono f32, requested period 20 ms",
    "[stats] t=5.0 frames=80000 overruns=0 underruns=0 drops=0 queued=1120 "
    "skipped=0 prefilled=1120 rss_kb=23552 inf_p50_us=1000 inf_p95_us=5000 "
    "inf_max_us=9000 misses=0 errors=0 lat_ms=75.5",
    "[stats] t=15.0 frames=240000 overruns=0 underruns=0 drops=0 queued=960 "
    "skipped=160 prefilled=1120 rss_kb=23552 inf_p50_us=1000 inf_p95_us=5000 "
    "inf_max_us=9000 misses=0 errors=0 lat_ms=75.5",
    "[stats] t=25.0 frames=400000 overruns=0 underruns=0 drops=0 queued=800 "
    "skipped=320 prefilled=1120 rss_kb=23584 inf_p50_us=1050 inf_p95_us=5000 "
    "inf_max_us=9000 misses=0 errors=0 lat_ms=75.5",
    "[stats] t=35.0 frames=560000 overruns=0 underruns=0 drops=0 queued=800 "
    "skipped=320 prefilled=1120 rss_kb=23584 inf_p50_us=1000 inf_p95_us=5000 "
    "inf_max_us=9000 misses=0 errors=0 lat_ms=75.5",
]

# One GUI session switching Bypass -> Mode A: both line shapes in one file,
# with t still increasing across the switch (only --check-format reads this).
GUI_MIXED_LOG = [
    "Client  : 16000 Hz mono f32, requested period 20 ms",
    "[stats] t=1.0 frames=16000 overruns=0 underruns=320 drops=0 queued=320 "
    "skipped=0 prefilled=0 rss_kb=13312",
    "[stats] t=2.0 frames=32000 overruns=0 underruns=320 drops=0 queued=320 "
    "skipped=0 prefilled=0 rss_kb=13312",
    "[stats] t=41.0 frames=656000 overruns=0 underruns=0 drops=0 queued=960 "
    "skipped=160 prefilled=1120 rss_kb=23552 inf_p50_us=1000 inf_p95_us=5000 "
    "inf_max_us=9000 misses=0 errors=0 lat_ms=75.5",
    "[stats] t=42.0 frames=672000 overruns=0 underruns=0 drops=0 queued=800 "
    "skipped=320 prefilled=1120 rss_kb=23552 inf_p50_us=1000 inf_p95_us=5000 "
    "inf_max_us=9000 misses=0 errors=0 lat_ms=75.5",
]


def cli_error(vc, s2ut, burst_cushion_ms, latency_target_ms, hop_ms):
    """Return a CLI validation error message, or None when the combination is valid."""
    if vc and s2ut:
        return "--vc and --s2ut are mutually exclusive"
    if burst_cushion_ms is not None:
        if not s2ut:
            return "--burst-cushion-ms requires --s2ut"
        if not math.isfinite(burst_cushion_ms) or burst_cushion_ms < 0:
            return "--burst-cushion-ms must be finite and non-negative"
    if (vc or s2ut) and (latency_target_ms is None or hop_ms is None):
        return f"{'--s2ut' if s2ut else '--vc'} requires --latency-target-ms and --hop-ms"
    if not (vc or s2ut) and (latency_target_ms is not None or hop_ms is not None):
        return "target/hop options require --vc or --s2ut"
    return None


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
    def s2ut_rows():
        return parse(fake_s2ut_log())[0]

    def shifted(shift):
        return [dict(r, prefilled=r["prefilled"] + shift, queued=r["queued"] + shift)
                for r in s2ut_rows()]

    def mutate(**last):
        data = s2ut_rows()
        data[-1].update(last)
        return data

    all_errors = [dict(r, errors=1) for r in s2ut_rows()]
    never_zero = [dict(r, lag_ms=160.0) for r in s2ut_rows()]
    one_zero = [dict(r) for r in never_zero]
    one_zero[-1]["lag_ms"] = 0.0
    neg_flush = [dict(r, lag_ms=-80.0) for r in never_zero]
    neg_flush[-1]["lag_ms"] = 0.0
    missing = s2ut_rows()
    del missing[-1]["s2ut_f"]
    for name, data, expect, vc_mode in [
        ("S2UT clean", s2ut_rows(), True, False),
        ("S2UT queued in cushion", shifted(2000), True, False),
        ("same log fails under --vc", shifted(2000), False, True),
        ("S2UT queued above cushion", shifted(4200), False, False),
        ("S2UT queued below target-hop", shifted(-400), False, False),
        ("S2UT late miss", mutate(misses=1), False, False),
        ("S2UT any error (flat)", all_errors, False, False),
        ("S2UT late error", mutate(errors=1), False, False),
        ("S2UT counter decrease", mutate(s2ut_r=0), False, False),
        ("S2UT cap > utt", mutate(s2ut_cap=99), False, False),
        ("S2UT lag never 0", never_zero, False, False),
        ("S2UT lag 0 once", one_zero, True, False),
        ("S2UT negative lag flush", neg_flush, True, False),
        ("S2UT missing fields", missing, False, False),
    ]:
        if vc_mode:
            r = analyze(data, rate, 10, 5120, vc=True, target_ms=50, hop_ms=20)
        else:
            r = analyze(data, rate, 10, 5120, s2ut=True, target_ms=50, hop_ms=20,
                        burst_cushion_ms=240.0)
        good = r["ok"] == expect
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:28s} -> {r['reason']}")
    r = analyze(s2ut_rows(), rate, 10, 5120, s2ut=True, target_ms=50, hop_ms=20,
                burst_cushion_ms=-1.0)
    good = not r["ok"] and "burst-cushion" in r["reason"]
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'S2UT negative cushion':28s} -> {r['reason']}")
    r = analyze(s2ut_rows(), rate, 10, 5120, s2ut=True, target_ms=50, hop_ms=20,
                burst_cushion_ms=240.0)
    good = (r["s2ut_utt_delta"] == 5 and r["s2ut_cap_delta"] == 2 and r["lag_ms_max"] == 160.0
            and r["cap_fraction"] == 0.4)
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'S2UT summary deltas':28s} -> "
          f"utt {r['s2ut_utt_delta']} cap {r['s2ut_cap_delta']} frac {r['cap_fraction']}")
    for name, args, bad in [
        ("CLI --s2ut --vc rejected", (True, True, None, 50, 20), True),
        ("CLI negative cushion rejected", (False, True, -1.0, 50, 20), True),
        ("CLI nan cushion rejected", (False, True, float("nan"), 50, 20), True),
        ("CLI cushion without --s2ut", (True, False, 240.0, 50, 20), True),
        ("CLI --s2ut needs target/hop", (False, True, None, None, 20), True),
        ("CLI --s2ut valid", (False, True, 240.0, 400, 80), False),
    ]:
        err = cli_error(*args)
        good = (err is not None) == bad
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:28s} -> {err}")
    base = "[stats] t=1 frames=1 overruns=0 underruns=0 drops=0 queued=0 rss_kb=1"
    try:
        parse([base + " lag_ms=-80.0"])
        print("  PASS  negative lag_ms accepted")
    except ValueError:
        ok = False
        print("  FAIL  negative lag_ms accepted")
    for bad_line in (base + " lag_ms=nan", base.replace("rss_kb=1", "rss_kb=-1")):
        try:
            parse([bad_line])
            ok = False
            print("  FAIL  nan lag / negative rss rejected")
        except ValueError:
            print("  PASS  nan lag / negative rss rejected")
    try:
        parse(["[stats] t=nan frames=1 overruns=0 underruns=0 drops=0 queued=0 rss_kb=1"])
        ok = False
    except ValueError:
        print("  PASS  malformed stats rejected")
    print("  -- GUI [stats] lines (format_stats_line shape) via analyze()/--check-format")
    rows, rate = parse(GUI_BYPASS_LOG)
    r = analyze(rows, rate, 10, 1024)
    good = r["ok"] and r["queued_ms"] == 20.0
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'GUI bypass analyze':26s} -> {r['reason']}")
    rows, rate = parse(GUI_VC_LOG)
    r = analyze(rows, rate, 10, 5120, vc=True, target_ms=50, hop_ms=20)
    good = r["ok"]
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'GUI Mode A analyze':26s} -> {r['reason']}")
    for name, log, lines, legacy, vc_n, s2ut_n in [
        ("GUI bypass format", GUI_BYPASS_LOG, 4, 4, 0, 0),
        ("GUI Mode A format", GUI_VC_LOG, 4, 0, 4, 0),
        ("GUI switch format", GUI_MIXED_LOG, 4, 2, 2, 0),
        ("S2UT fake format", fake_s2ut_log(), 7, 0, 7, 7),
    ]:
        summary, lineno, reason = check_format(log)
        good = (reason is None and summary["lines"] == lines and summary["legacy"] == legacy
                and summary["vc"] == vc_n and summary["s2ut"] == s2ut_n)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:26s} -> {reason or summary}")
    no_legacy = GUI_BYPASS_LOG[1:3]
    no_legacy[0] = no_legacy[0].replace(" rss_kb=13312", "", 1)
    bad_vc = GUI_VC_LOG[1:4]
    bad_vc[2] = bad_vc[2].replace(" lat_ms=75.5", "", 1)
    nonfinite = GUI_VC_LOG[1:4]
    nonfinite[2] = nonfinite[2].replace("t=25.0", "t=inf", 1)
    decreasing = GUI_BYPASS_LOG[1:4]
    decreasing[1] = decreasing[1].replace("t=11.0", "t=0.5", 1)
    no_s2ut_f = [fake_s2ut_log()[1].replace(" s2ut_f=0", "", 1)]
    for name, lines, want_line, want_reason in [
        ("missing legacy field", no_legacy, 1, "incomplete"),
        ("missing skipped/prefilled", [GUI_BYPASS_LOG[1].replace(" skipped=0 prefilled=0", "", 1)],
         1, "missing skipped/prefilled"),
        ("VC line missing lat_ms", bad_vc, 3, "VC line missing lat_ms"),
        ("non-finite value", nonfinite, 3, "non-finite"),
        ("decreasing t", decreasing, 2, "decreases"),
        ("empty file", ["Client  : 16000 Hz mono f32"], None, "no [stats] lines"),
        ("S2UT line missing s2ut_f", no_s2ut_f, 1, "S2UT line missing s2ut_f"),
    ]:
        summary, lineno, reason = check_format(lines)
        good = reason is not None and want_reason in reason and lineno == want_line
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:26s} -> line {lineno}: {reason}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", nargs="?")
    p.add_argument("--warmup", type=float, default=10.0, help="seconds to ignore (default 10)")
    p.add_argument("--rate", type=int, default=None,
                   help="client sample rate (default: read from the log's 'Client' line)")
    p.add_argument("--rss-tolerance-kb", type=int, default=None,
                   help="allowed RSS growth (default: 1024 KB, or 5120 KB with --vc/--s2ut)")
    p.add_argument("--vc", action="store_true", help="check Mode A inference and governor telemetry")
    p.add_argument("--s2ut", action="store_true",
                   help="check Mode B telemetry (VC checks + s2ut counters, lag, widened queued band)")
    p.add_argument("--burst-cushion-ms", type=float, default=None,
                   help="Mode B governor burst cushion (default 240 = 3 x 80 ms; --s2ut only)")
    p.add_argument("--latency-target-ms", type=float)
    p.add_argument("--hop-ms", type=float)
    p.add_argument("--check-format", action="store_true",
                   help="only validate the [stats] line shape (GUI or CLI logs); no soak criteria")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.log:
        p.error("log file required")
    if args.check_format:
        try:
            with open(args.log, encoding="utf-8", errors="replace") as f:
                summary, lineno, reason = check_format(f)
        except OSError as exc:
            print(f"Invalid soak log: {exc}", file=sys.stderr)
            return 1
        if reason is not None:
            where = f"line {lineno}: " if lineno is not None else ""
            print(f"Invalid [stats] format: {where}{reason}", file=sys.stderr)
            return 1
        print(f"format OK: {summary['lines']} [stats] lines "
              f"(legacy {summary['legacy']}, vc {summary['vc']}, s2ut {summary['s2ut']})")
        return 0
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
    err = cli_error(args.vc, args.s2ut, args.burst_cushion_ms, args.latency_target_ms, args.hop_ms)
    if err:
        p.error(err)
    cushion = 240.0 if args.burst_cushion_ms is None else args.burst_cushion_ms
    try:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            rows, rate = parse(f)
    except (OSError, ValueError) as exc:
        print(f"Invalid soak log: {exc}", file=sys.stderr)
        return 1
    tolerance = args.rss_tolerance_kb
    if tolerance is None:
        tolerance = 5120 if args.vc or args.s2ut else 1024
    r = analyze(rows, args.rate or rate, args.warmup, tolerance, args.vc,
                args.latency_target_ms, args.hop_ms, args.s2ut, cushion)
    for k, v in r.items():
        print(f"{k:22s}: {v}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
