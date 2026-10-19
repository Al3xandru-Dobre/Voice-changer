# BUG-013 — `golden/` survives a model change, so the bundle can mix models after all

> [Bug index](00_index.md) · Review: [REVIEW-M2-S06](REVIEW-M2-S06_2026-10-18.md)

| | |
|---|---|
| **Tag** | `BUG-013` |
| **Name** | *Stale golden* — re-export from a different checkpoint leaves `golden/` (and `report["golden"]`) pointing at deleted or overwritten files |
| **Reported** | 2026-10-18 (static review; see Environment) |
| **Severity** | Medium (Stage 07's C++-vs-golden parity check silently verifies the wrong model) |
| **Status** | ☑ fixed — verified 2026-10-19, see [Resolution](#resolution-verified-2026-10-19) |
| **Component** | `python_training/export_stateful_onnx.py` — `run()` bundle-lifecycle branch, `write_golden()` |
| **Blocks** | [Stage 07](../Milestone%202/07_ort_cpp_runtime.md) golden check (`fv_vc_file` vs `models/<bundle>/golden/`); contradicts [Stage 06](../Milestone%202/06_stateful_onnx_export.md)'s design decision *"a run with a different checkpoint deletes that role's variants …, and the chain results, so the bundle never mixes models"* |

## Environment

Found by code inspection during the Stage 06 review (this ACP session had no
shell, so the repro below is a traced scenario, not an executed log; the
existing test `test_cli_builds_a_bundle` already performs the exact CLI
sequence, it just never inspects `golden/` afterwards). Pinned env per
`pyproject.toml`: Python 3.12, torch 2.11.0, onnx 1.23.2, onnxruntime 1.31.0.

## Steps to reproduce

Any two-run bundle build followed by a model change, e.g. (as in
`test_cli_builds_a_bundle`):

```powershell
cd python_training
.venv\Scripts\python.exe export_stateful_onnx.py --role encoder `
    --checkpoint ..\checkpoints\student_a\best.pt --head student-L1 `
    --license "..." --bundle ..\models\demo --frames-per-step 1 2
.venv\Scripts\python.exe export_stateful_onnx.py --role vocoder `
    --checkpoint ..\checkpoints\vocoder\best.pt --bundle ..\models\demo `
    --frames-per-step 1 2            # golden/ now holds s1+s2 units, cond, audio
# ... later: a NEW student checkpoint, quick re-export without re-running parity
.venv\Scripts\python.exe export_stateful_onnx.py --role encoder `
    --checkpoint ..\checkpoints\student_b\best.pt --head student-L1 `
    --license "..." --bundle ..\models\demo --frames-per-step 1 --skip-parity
```

Then inspect `..\models\demo\golden\manifest.json` and the bundle directory.

## Expected

The bundle never mixes models (Stage 06 design decision): the re-export from
a different checkpoint deletes what the old model produced **and everything
derived from it** — variant files, `chain`/`chain_cost` report entries, and
`golden/` (+ `report["golden"]`), since golden is derived output of the exact
model files. After the run above, `golden/manifest.json` must not exist (or
must not reference `encoder_s2.onnx`, which was deleted, nor an
`encoder_sha256` that no longer matches `encoder_s1.onnx`).

## Actual (traced from `run()` / `write_golden()`)

The different-model branch cleans up only two of the three derived artifacts:

```python
else:                           # a different model: drop the old one's variants
    for v in sorted(set(old.get("files") or {}) - set(files)):
        stale = bundle / old["files"][v]["file"]
        stale.unlink(missing_ok=True)
    for key in ("chain", "chain_cost"):
        report.pop(key, None)
```

`golden/` is written only inside `if not args.skip_parity:` …
`report["golden"] = write_golden(...)`, so with `--skip-parity` nothing
touches it, and `report["golden"]` loaded from the previous run survives
into the new `export_report.json` and `MODEL_CARD.md` ("## Golden I/O").

After the three commands above:

- `encoder_s2.onnx` is deleted, but `golden/manifest.json` still lists
  variant `s2` with `"encoder": "encoder_s2.onnx"` — a reference to a file
  that no longer exists.
- `encoder_s1.onnx` is overwritten by student B's export, but variant `s1`'s
  `"encoder_sha256"` (and `vocoder` shas are fine, but the manifest as a
  whole describes the old pair) still holds student A's hash.
- `golden/units_s1.npy`, `cond.npy`, `audio_s1.wav` are still student A's
  outputs; Stage 07's `fv_vc_file` check against them compares C++ output of
  student B against references from student A.

A milder variant without `--skip-parity`: re-exporting with a *smaller*
`--frames-per-step` set regenerates the manifest correctly but leaves the
dropped variants' `units_s2.npy` / `audio_s2.wav` orphaned on disk.

## Root cause

`write_golden()` is treated as a parity-time step instead of a bundle
artifact with the same freshness rules as `chain`/`chain_cost`. The
different-model cleanup branch (and the shrink case) was written for the
`.onnx` variants and the chain results only.

## Fix

1. In the different-model branch: `shutil.rmtree(bundle / GOLDEN,
   ignore_errors=True)` (or unlink the manifest + variant files) and
   `report.pop("golden", None)`.
2. In `write_golden()`: after regenerating, delete `units_s*.npy` /
   `audio_s*.wav` files not referenced by the new manifest, so a shrinking
   variant set cannot leave orphans.
3. Regression test: extend the final block of `test_cli_builds_a_bundle`
   (which already runs the different-checkpoint `--skip-parity` re-export) to
   assert that `golden/manifest.json` is gone (or references only existing
   files whose sha256 matches the bundle).

## Workaround

Until fixed, delete `models/<bundle>/golden/` by hand whenever a role is
re-exported from a different checkpoint, and re-run the full export (without
`--skip-parity`) for both roles before Stage 07 consumes the bundle.

## Resolution (verified 2026-10-19)

1. In `export_stateful_onnx.py::run()`, the different-model (`not same_source`)
   branch now calls `_purge_golden(bundle, report, ...)`, deleting `bundle / "golden"`
   and popping `report["golden"]` alongside `"chain"` and `"chain_cost"`.
2. On `--skip-parity` runs, `_golden_is_current(bundle, report.get("golden"))`
   verifies that every variant's `.onnx` file referenced in `golden/manifest.json`
   still exists and matches its recorded SHA-256, purging `golden/` if not; on
   `same_source` `--skip-parity` runs, prior `parity`/`cost` entries for surviving
   variants are preserved.
3. `write_golden()` unlinks any orphaned `units_s*.npy` / `audio_s*.wav` files in
   `golden/` not referenced by the newly written manifest.
4. `test_cli_builds_a_bundle` in `python_training/tests/test_export.py` pins all
   three cases (same-checkpoint `--skip-parity` preservation, different-checkpoint
   `--skip-parity` golden removal, and shrink-without-orphans regeneration).
