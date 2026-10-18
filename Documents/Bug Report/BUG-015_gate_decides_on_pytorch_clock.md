# BUG-015 — The decision gate applies its compute rule to PyTorch-eager timings, not to the runtime that ships

> [Bug index](00_index.md) · Review: [REVIEW-M2-S05](REVIEW-M2-S05_2026-10-18.md)

| | |
|---|---|
| **Tag** | `BUG-015` |
| **Name** | *Wrong clock* — eligibility and projected T−B come from Python/PyTorch dispatch time, while the stage file's table and conclusions are written from ONNX Runtime time |
| **Reported** | 2026-10-18 |
| **Severity** | High for the Stage 05 decision (the gate, once the long runs finish, would pick a different winner than the stage file says it should); no effect on any model or on the real-time path |
| **Status** | ◐ diagnosed (code path traced; decision flip reproduced deterministically with the stage file's own numbers) |
| **Component** | `python_training/gate_report.py` — `run_gate` (line 689), `measure_candidate_cost`, `apply_gate_rules`, `format_gate_markdown` |
| **Blocks** | [Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md) "Decision" task and open decisions 4 and 7 in the [master plan](../Milestone%202/00_master.md); the Stage 06 `--gate-report` hand-off and the Stage 11 exit report |

## Environment

Windows 11, Python 3.12.13 (`python_training\.venv`), torch 2.11.0+cu128,
onnxruntime 1.31.0, tree as of 2026-10-18 (uncommitted `gate_report.py`).
Read from code and re-run with the stage file's recorded numbers as input, so
the result does not depend on how loaded this machine is.

## Summary

The Stage 05 rule is "encoder + vocoder compute p95 ≤ 50 % of the hop on one
core", and the product runs in ONNX Runtime (C++), not in PyTorch. The stage
file records both clocks and concludes from the ORT one:

| Candidate | PyTorch eager p95 | ORT p95 | 50 % of hop |
|---|---|---|---|
| student-L0 s1 | 15.4 ms | 4.5 ms | 10 ms |
| student-L1 s1 | 16.7 ms | 4.9 ms | 10 ms |
| student-L2 s1 | 16.5 ms | 4.6 ms | 10 ms |
| student-L1 s2 | 18.7 ms | 8.2 ms | 20 ms |

> "All `s1` and `s2` student candidates pass the ≤ 50 %-of-hop compute budget
> comfortably under ORT." — Stage 05, Verification

`gate_report.py` does not use the ORT figure for anything that decides:

```python
# run_gate(), line 686-690
smoke = export_smoke_test(cand, enc=enc, voc=voc)      # measures ORT p95 -> smoke["ort_total_p95_ms"]
cost  = measure_candidate_cost(cand, enc=enc, voc=voc, ...)   # PyTorch eager + Python tracker
p95_ms  = cost["total_p95_ms"]                         # <- the only number that is used
proj_tb = projected_t_minus_b_ms(cand, p95_ms, ...)
row["compute_p95_ms"] = p95_ms                         # <- what apply_gate_rules() tests
```

`smoke["ort_total_p95_ms"]` is stored under `row["export_smoke"]` in
`gate_report.json` and never read again. `format_gate_markdown` prints one
"Compute p95" column and one "Projected T−B" column (both PyTorch), while the
stage file's hand-written table has "PyTorch / ORT" pairs. Stage 06's
`export_stateful_onnx.py` uses ORT chain timings for the same rule, so the two
sibling tools now disagree on the same question.

## Steps to reproduce

Deterministic, no GPU or data needed. Feed the stage file's own PyTorch and ORT
columns, with identical passing quality metrics, through the real
`apply_gate_rules`:

```powershell
cd python_training
.venv\Scripts\python.exe -c "
import gate_report as g
pt  = {'student-L0 s1':15.4,'student-L1 s1':16.7,'student-L2 s1':16.5,'student-L1 s2':18.7}
ort = {'student-L0 s1':4.5, 'student-L1 s1':4.9, 'student-L2 s1':4.6, 'student-L1 s2':8.2}
for tag, src in (('PyTorch column', pt), ('ORT column', ort)):
    rows = [dict(candidate=c.name, hop_ms=c.hop_ms, lookahead_frames=c.lookahead_frames,
                 alg_latency_ms=c.alg_latency_ms, compute_p95_ms=src[c.name],
                 projected_t_minus_b_ms=g.projected_t_minus_b_ms(c, src[c.name]),
                 wer=.05, cer=.02, tgt_cos=.75, src_cos=.1, utmos=3.6, f0_pcc=.8, export_ok=True)
            for c in g.DEFAULT_CANDIDATES]
    ann, dec = g.apply_gate_rules(rows)
    print(tag, '->', dec['winner'], dec['projected_t_minus_b_ms'])
"
```

## Expected

Same winner as the stage file's reasoning: with equal quality, the lowest
projected T−B among eligible rows under the **shipping runtime's** compute —
`student-L0 s1` at 54.5 ms.

## Actual

| Compute source | s1 L0 / L1 / L2 | s2 L1 | Winner | Projected T−B |
|---|---|---|---|---|
| PyTorch column (what `run_gate` feeds the rules) | all **ineligible** (15.4 / 16.7 / 16.5 ms > 10 ms) | eligible (18.7 ≤ 20 ms) | **student-L1 s2** | **108.7 ms** — over the master plan's 100 ms *must* |
| ORT column (what the stage file concludes from) | all eligible | eligible | **student-L0 s1** | 54.5 ms |

So the gate, as coded, rejects every 20 ms-hop candidate for a reason that does
not exist in the product, and hands Stage 06 and Stage 08 a 40 ms-hop design
that misses the latency target. Worse, nothing in the output says the
PyTorch clock was used: the Markdown has no "runtime" label.

Secondary effects of the same line:

- The PyTorch number includes the Python `PitchTracker` (0.19 ms p50, 0.8 ms
  p95 per frame here) while the ORT number does not, so the two columns are not
  like-for-like.
- p95 of a single-core timing on a shared dev laptop is load-sensitive. On this
  machine during the review (≈25 % CPU from other sessions) the same code gave
  PyTorch 43–62 ms and ORT 12–25 ms at s1, so the figures in the stage file
  could **not** be reproduced; the report records neither host load nor
  thread affinity, so a noisy run silently flips eligibility.

## Root cause

`measure_candidate_cost` was written first (PyTorch path) and the ORT timing
was added later inside `export_smoke_test` as a by-product. `run_gate` kept
the original wiring, and the only test of the rules
(`test_gate_rules_and_markdown_table`) hand-writes `compute_p95_ms` rows, so no
test sees which clock `run_gate` selects. The stage file's tick box "Cost:
single-core … (PyTorch eager + ORT …)" was ticked on the strength of both
numbers existing, not of the rule using the right one.

## Fix

1. Take the compute number for the rule from ORT (`export_smoke["ort_total_p95_ms"]`,
   or better a dedicated multi-hop **stateful** loop — see below), and keep the
   PyTorch figure as an informational column.
2. Print both in `format_gate_markdown` (the stage table already has the
   columns) and label the rule's clock in the decision line.
3. Time a carried-state loop of ≥ 200 hops rather than 50 identical zero-state
   calls (`export_smoke_test` re-feeds the same initial state every run).
   Record `os.cpu_count()`, 1-minute load, and warn when p99 > 3 × p50.
4. Add a `run_gate`-level test with a fake cost/ORT provider where the two
   clocks disagree, asserting that the ORT one decides.

## Workaround

Until fixed, read `gate_report.json` → `candidates[].export_smoke.ort_total_p95_ms`
and `cost.total_p95_ms` side by side, and take the decision by hand from the
ORT column. Do not paste `gate_report.md` into the stage file unedited.
