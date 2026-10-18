# BUG-016 — The gate's export smoke test swallows a failed dynamo export and passes on the legacy TorchScript exporter

> [Bug index](00_index.md) · Review: [REVIEW-M2-S05](REVIEW-M2-S05_2026-10-18.md)

| | |
|---|---|
| **Tag** | `BUG-016` |
| **Name** | *Silent fallback* — `export_ok = ✓` no longer means "this design exports with `dynamo=True`" |
| **Reported** | 2026-10-18 |
| **Severity** | Medium (the gate's "a design that can't export can't win" check can pass for the wrong reason; the measured ORT numbers then belong to a different graph than Stage 06 ships) |
| **Status** | ◐ workaround (`PYTHONIOENCODING=utf-8`, already set for CTest; the swallow itself is still there) |
| **Component** | `python_training/gate_report.py` — `_export_module_to_onnx` (lines 150-184), `export_smoke_test` (`except Exception`, line 317), `format_gate_markdown` |
| **Blocks** | [Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md) export-smoke task and its Verification claim ("export cleanly via `torch.onnx.export(..., dynamo=True)`"); trust in the "Export OK" column of the gate table |

## Environment

Windows 11, Python 3.12.13, torch 2.11.0+cu128, onnxscript 0.7.x, onnx 1.23.2.
Output piped (as under CTest, a CI step or an agent shell), i.e. stdout encoding
`cp1252` — Windows' default for a redirected stream.

## Summary

```python
for use_dynamo in (True, False):
    try:
        if use_dynamo:
            torch.onnx.export(wrapper, args_tuple, ..., dynamo=True, opset_version=18, ...)
            exporter_used = "dynamo"
        else:
            torch.onnx.export(wrapper, args_tuple, ..., dynamo=False, opset_version=17, ...)
            exporter_used = "torchscript"
        break
    except Exception as e:          # <- any failure of the first attempt, whatever its cause
        last_err = e
```

`torch.onnx.export(dynamo=True)` prints a `✅` progress line. With a `cp1252`
stdout that raises `UnicodeEncodeError` *inside the exporter*, after the graph
was already traced. The broad `except` treats it as "dynamo can't export this
model", discards the exception (`last_err` is only used if both attempts fail)
and re-exports with the legacy TorchScript exporter at opset 17. The returned
`exporter_used` is stored in the JSON but never in the table or the decision,
and `export_ok` stays true.

The CTest entry already carries the workaround without saying why
(`ENVIRONMENT "PYTHONIOENCODING=utf-8"` and the comment "The dynamo exporter
prints emoji; keep a cp1252 console pipe from crashing it"), which confirms the
failure mode is known; only the swallowing was not fixed.

## Steps to reproduce

```powershell
cd python_training
# piped => cp1252 stdout on Windows
.venv\Scripts\python.exe -I -c "
import sys, torch, tempfile, pathlib
sys.path.insert(0, '.')
import gate_report as g
from fvml import vocoder
v = vocoder.UnitVocoder().eval(); init = v.init_state(1); st = [n for n,_,_ in v.state_layout()]
ex = (torch.zeros(1,1,dtype=torch.int64), torch.full((1,1),5.), torch.ones(1,1), torch.full((1,1),.1),
      torch.full((1,1),-5.), v.speaker_table.weight[:1].detach().clone(), *[init[n] for n in st])
p = pathlib.Path(tempfile.gettempdir()) / 'b016.onnx'
try:
    torch.onnx.export(g._VocoderStepWrapper(v).eval(), ex, str(p), dynamo=True, opset_version=18, external_data=False)
    print('DYNAMO OK')
except BaseException as e:
    print('DYNAMO FAILED', type(e).__name__, str(e)[:90])
" | findstr DYNAMO
# now the real thing, piped (cp1252):
.venv\Scripts\python.exe gate_report.py --skip-eval --out $env:TEMP\b016 --candidates student-L0:s1 | Out-Null
$LASTEXITCODE                      # 1, "[gate_report] error: 'charmap' codec can't encode character '−'"
(Get-Content $env:TEMP\b016\gate_report.json | ConvertFrom-Json).candidates[0].export_smoke |
    Select-Object encoder_exporter, vocoder_exporter      # torchscript, torchscript
# and with the workaround:
$env:PYTHONIOENCODING = 'utf-8'    # (repeat the gate_report command) -> exit 0, dynamo, dynamo
```

## Expected

Either `dynamo=True` succeeds, or the smoke test reports `export_ok = ✗` with the
real reason (the Stage 05 task text: "export … (`dynamo=True`) and run one step
in ONNX Runtime CPU. A design that can't export can't win"), or at minimum the
fallback is visible in the gate table.

## Actual

Measured on this machine (piped, no workaround):

| Run | stdout encoding | `exporter_used` (enc / voc) | Result |
|---|---|---|---|
| bare dynamo export of the vocoder step graph | `cp1252` | — | `UnicodeEncodeError: 'charmap' codec can't encode character '✅'` |
| same, `PYTHONIOENCODING=utf-8` | `utf-8` | — | exports |
| `gate_report.py --skip-eval --candidates student-L0:s1`, piped | `cp1252` | `torchscript` / `torchscript` | `export_ok = True`; then **exit 1** (see below) |
| same, `PYTHONIOENCODING=utf-8` | `utf-8` | `dynamo` / `dynamo` | `export_ok = True` |

Consequences:

- The same encoding fault also breaks the script's own ending: after
  `gate_report.json` and `gate_report.md` are written, `print(md)` raises
  `UnicodeEncodeError` on the `−` in "Projected T−B" (and on `✓`/`✗`), which
  `main()` catches as a `ValueError` and reports as
  `[gate_report] error: 'charmap' codec can't encode character '−'` with
  **exit 1** — the same code the tool uses for "bad inputs". A successful run
  is indistinguishable from a failed one for anything that checks the exit
  status. (Run on 2026-10-18: piped → exit 1, files complete; with
  `PYTHONIOENCODING=utf-8` → exit 0.)
- A genuine dynamo-only failure (an op the new exporter rejects, a state
  tensor it cannot trace) is hidden whenever the legacy exporter happens to
  accept the model. Stage 06 prefers `dynamo=True`, so Stage 05 would have
  cleared a design Stage 06 may then have to fall back on.
- The ORT p95 in the smoke result belongs to whichever graph was produced. On
  the loaded review machine the TorchScript/opset-17 graph measured
  17.7 ms vs 25.4 ms for the dynamo graph in two noisy runs, i.e. the number
  is not even attributable to a single exporter (see [BUG-015](BUG-015_gate_decides_on_pytorch_clock.md)).
- `test_export_smoke_and_streaming_chain` passes with either exporter, so the
  suite cannot detect the swallow (it asserts `smoke["ok"]` only).
- Every other failure of the first attempt (wrong dtype, shape, `onnxscript`
  missing) is swallowed the same way.

Stage 06's `fvml/export.py::export_graph` handles this correctly: it records
the fallback reason, deduplicates warnings, and `--exporter dynamo` refuses to
fall back. `gate_report.py` re-implemented a weaker copy of that logic
(including a different opset/IR clamp: ≤ 18 / ≤ 10 vs Stage 06's 26 / 13).

## Root cause

Duplicate export logic in `gate_report.py` with a catch-all `except Exception`
whose only purpose was "try the other exporter", combined with a progress
message that is not safe to print on a legacy-codepage console. The fallback is
recorded in a field that nothing reads.

## Fix

1. Reuse `fvml.export.export_graph` (or its building blocks) instead of the
   private copy, so both stages share one exporter policy, opset cap and
   fallback record.
2. Make the smoke test require `dynamo` by default (`--exporter auto` to allow
   the fallback explicitly), and include the exporter, and the fallback reason
   when there is one, in the gate table / decision text.
3. `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` next to the
   existing `reconfigure(line_buffering=True)` in `main()` for the gate script,
   `stream_check.py` and `export_stateful_onnx.py`, so the workaround does not
   depend on the caller's environment.
4. Test: a fake `torch.onnx.export` that raises on `dynamo=True` must give
   `export_ok = False` (or a visible "fallback" marker) — not a pass.

## Workaround

Run every Stage 05/06 Python tool with `PYTHONIOENCODING=utf-8` (CTest already
does) and check `export_smoke.encoder_exporter` / `vocoder_exporter` in
`gate_report.json` reads `dynamo` before trusting a row.
