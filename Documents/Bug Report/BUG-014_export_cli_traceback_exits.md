# BUG-014 — Exporter CLI leaks non-`ExportError` exceptions: tracebacks and a wrong exit code

> [Bug index](00_index.md) · Review: [REVIEW-M2-S06](REVIEW-M2-S06_2026-10-18.md)

| | |
|---|---|
| **Tag** | `BUG-014` |
| **Name** | *Traceback exits* — bad inputs and ORT failures crash the CLI with exit 1, the code reserved for "a gate failed" |
| **Reported** | 2026-10-18 (static review; see Environment) |
| **Severity** | Low–medium (automation misreads failures; documented exit-status contract broken; no wrong model is produced) |
| **Status** | ☐ open |
| **Component** | `python_training/export_stateful_onnx.py` — `main()` exception boundary; also `fvml/export.py` boundary calls (`audio.load`, `ort.InferenceSession`, `torch.load`) |
| **Blocks** | [Stage 06](../Milestone%202/06_stateful_onnx_export.md) CLI contract ("Exit status: 0 = …; 1 = a gate failed …; 2 = bad arguments or an export/contract error"); scripting/CI around the exporter |

## Environment

Found by code inspection during the Stage 06 review (no shell in the session;
the exception types cited are from the pinned libraries' documented behavior:
soundfile ≥ 0.13, onnxruntime 1.31, torch 2.11 — all raise `RuntimeError`
subclasses or `ValueError`s that are not `ContractError`). The same defect
shape was [BUG-004](BUG-004_contract_py_leaks_non_contracterror.md) for
`fvml/contract.py`, fixed 2026-10-18.

## Steps to reproduce

Any of these against the pinned venv:

```powershell
cd python_training
# 1. parity audio file that does not exist / is not audio
.venv\Scripts\python.exe export_stateful_onnx.py --role encoder `
    --checkpoint ..\checkpoints\student\best.pt --head student-L1 `
    --license x --bundle ..\models\demo --frames-per-step 1 `
    --exporter torchscript --parity-audio Z:\nope.wav
# 2. --gate-report pointing at a non-JSON file
#    (… --gate-report ..\data\reports\gate.txt)
# 3. an .onnx that ORT refuses to load during parity (e.g. hand-corrupted bytes)
# 4. a truncated/corrupt export_report.json or checkpoint in an existing bundle
```

## Expected

Per the module docstring: `2 = bad arguments or an export/contract error`,
printed as one line (`[export] error: …`), no traceback. Exit 1 must mean
"files and report written, a measured gate failed".

## Actual (traced from `main()`)

```python
try:
    return run(args)
except (fx.ExportError, fc.ContractError, FileNotFoundError) as e:
    print(f"[export] error: {e}", file=sys.stderr)
    return 2
```

Only three exception types are classified. Everything else escapes:

| Input | Raised by | Type | Result |
|---|---|---|---|
| missing/corrupt `--parity-audio` file, or a manifest row whose audio is gone | `fvml.audio.load` → `sf.read` | `soundfile.LibsndfileError` (`RuntimeError`) | traceback, **exit 1** |
| `--gate-report` file that is not JSON | `json.loads` | `json.JSONDecodeError` (`ValueError`, not `ContractError`) | traceback, exit 1 |
| exported graph ORT refuses at parity time | `ort.InferenceSession` | `onnxruntime.capi.onnxruntime_pybind11_state.*` (`RuntimeError`) | traceback, exit 1 — although this is exactly an "export error" |
| corrupt `export_report.json` in the bundle | `load_report` → `json.loads` | `JSONDecodeError` | traceback, exit 1 |
| corrupt/truncated checkpoint | `torch.load` | `RuntimeError` / `zipfile.BadZipFile` / `pickle.UnpicklingError` | traceback, exit 1 |
| checkpoint missing expected keys (`config`/`model`/`speakers`) | `from_checkpoint` / `voices_json` callers | `KeyError` | traceback, exit 1 |

Exit 1 from an unhandled exception is indistinguishable from "a gate
failed", which per the contract implies the report and files were written for
inspection — here they may not be.

## Root cause

The CLI wraps its own `ExportError`s but calls library boundaries
(soundfile, ORT, torch, json) that raise their own exception types; unlike
`fvml/contract.py` after BUG-004, nothing converts them at the boundary.

## Fix

- Preferably at the boundary, mirroring the BUG-004 fix: wrap
  `audio.load`/`concat` in `parity_audio`, the `json.loads` of report/gate,
  and `OrtStream`/`ort_session` construction so they raise `ExportError`
  with the offending path/name.
- Or pragmatically at the CLI edge: add `RuntimeError`, `ValueError`,
  `KeyError`, `OSError`, `zipfile.BadZipFile` to the `except` tuple (ordered
  specific-first), keeping the one-line `[export] error: {type: message}`
  and exit 2.
- Pin with a test: `--parity-audio <missing>` must exit 2 with
  `[export] error:` on stderr and no traceback (`test_cli_rejects_bad_input`
  is the natural home).

## Workaround

Treat any traceback (or exit 1 without a `verdict: … FAILED` log line) as an
error, not a gate failure, when scripting the exporter.
