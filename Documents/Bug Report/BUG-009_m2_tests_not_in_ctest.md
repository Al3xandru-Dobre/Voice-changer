# BUG-009 — Milestone 2 tests exist but nothing in `ctest` or CI runs them

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-009` |
| **Name** | *Silent gate* — the build-and-test gate passes without touching any Milestone 2 code |
| **Reported** | 2026-10-18 |
| **Severity** | Medium (a regression in the contract rules would pass the gate) |
| **Status** | ☑ fixed — verified 2026-10-18 15:30, see [Resolution](#resolution-verified-2026-10-18); CI pytest job added 17:00 (not yet run on GitHub) |
| **Component** | `CMakeLists.txt`, `.github/workflows/build.yml`, `tools/make_fixture_models.py` |
| **Blocks** | [Stage 02](../Milestone%202/02_model_contract_fixtures.md) definition of done ("tests run … and pass in both languages"); [`AGENTS.md`](../../AGENTS.md) Part 1 rules 4 and 5 |

## Summary

Stage 02 added a Python test suite (`python_training/tests/`), a C++ twin
(`tests/test_model_contract.cpp` + `src/pipeline/ModelContract.hpp`) and a
fixture generator with a `--check` mode. None is registered with CTest, and
the CI workflow runs only CTest. The project's gate
(`ctest --test-dir build -C Release`) therefore reports green whatever the
Milestone 2 code does.

## Steps to reproduce

From the repo root, after configuring a build directory:

```powershell
ctest --test-dir build -N                                  # list the registered tests
ctest --test-dir build -N | Select-String "contract|fixture"   # no output
Select-String -Path CMakeLists.txt,.github/workflows/build.yml -Pattern "contract|fixture|pytest"   # no output
Select-String -Path tools/make_fixture_models.py -Pattern "self-test|^import onnx"   # only the onnx import
```

## Expected

- `test_model_contract` (ORT-free, C++) and the Python contract tests run under
  `ctest`, as Stage 02 specifies ("added to the CTest loop in `CMakeLists.txt`").
- Every `tools/` Python utility is standard-library only, has `--self-test`,
  and is registered ([`AGENTS.md`](../../AGENTS.md) Part 1, rule 5).

## Actual

| Check | Result on 2026-10-18 (isolated build of the current tree) |
|---|---|
| `ctest` | 34/34 pass, **0** of them contract / fixture tests |
| `CMakeLists.txt`, `build.yml` | no mention of `model_contract`, nlohmann/json, `make_fixture_models`, or `pytest` |
| `tests/test_model_contract.cpp` | **passes when compiled by hand** (`cl /std:c++20 /W4 /WX`, nlohmann 3.12.0 from `tools/_dl/m2/json`): 8 valid + 59 invalid cases, 4 fixture pairs at 320/640/1280/640 samples |
| `python_training/tests/` | 16/16 pass under `uv`'s Python 3.12; `test_contract.py` also passes as a plain script on 3.14 (what CTest's `find_package(Python3)` picks) |
| `tools/make_fixture_models.py` | imports `onnx` at top level (not stdlib), has no `--self-test`, is not registered |
| `src/pipeline/ModelContract.hpp` | includes `<nlohmann/json.hpp>`, which no CMake target provides yet, so it is compiled by nothing |

So the code is healthy; the gate just can't see it.

## Root cause

Stage 02's Tasks list the CTest registration and the nlohmann/json
`FetchContent` as separate checkboxes, and they are the last to be ticked.
Separately, the stage spec requires the generator to use `onnx.helper`, which
conflicts with the repo rule that `tools/` scripts be standard-library only.

## Fix

1. Add nlohmann/json 3.12.0 by `FetchContent` (release tarball + `URL_HASH
   SHA256`, `SYSTEM` include) and register `model_contract` in the existing
   `foreach(_t …)` loop, passing `-DFLUXVOICE_TEST_DATA_DIR=${CMAKE_SOURCE_DIR}/tests/data`
   or the directory as `argv[1]` (the test already accepts it).
2. Register `python_training/tests/test_contract.py` in script mode:
   `add_test(NAME contract_py COMMAND Python3::Interpreter -I … test_contract.py)`.
   It is standard-library only and was verified on 3.12 and 3.14.
3. For the generator, either (a) add a documented exception to `AGENTS.md`
   rule 5 for `make_fixture_models.py` (needs `onnx`), or (b) register
   `make_fixture_models.py --check` with `SKIP_RETURN_CODE` so it skips when
   `onnx` is missing and runs in a CI job that does `uv sync --frozen`.
   Give it a `--self-test` (it already has `check()` to build on).
4. Add a CI job that runs `uv sync --frozen` and `pytest` in `python_training/`
   so `test_fixtures.py` (ORT parity) is gated too.

## Workaround

Run the three checks by hand before committing:

```powershell
cd python_training; uv run pytest
uv run python ..\tools\make_fixture_models.py --check
# C++ twin: compile tests/test_model_contract.cpp with nlohmann/json on the include path, run with tests\data
```

## Resolution (verified 2026-10-18)

`CMakeLists.txt` now fetches nlohmann/json 3.12.0 by release tarball with a
`URL_HASH` (the configure step downloaded it and the hash matched) and registers:

| Test | What it runs |
|---|---|
| `model_contract` | the C++ twin on `tests/data` (11 valid + 73 invalid shared cases, fixture manifest, pair latencies) |
| `model_contract_py` | the same cases in Python, standard library only |
| `fixture_models_selftest` | new `make_fixture_models.py --self-test`; per its author it checks file hashes, that each embedded contract string appears in the `.onnx` bytes, verdicts, pairs, voices and stale files. I ran it and saw `PASS` lines for the pairs, `identity/voices.json` and `no stale files`; I did not audit the other checks |
| `python_training_stage01_02` | pytest in the uv env; registered only when `python_training/.venv` exists |

`make_fixture_models.py` now imports `onnx` lazily, so `--help` and
`--self-test` need only the standard library, which satisfies `AGENTS.md` rule 5.

| Check | Result |
|---|---|
| Fresh configure, build, `ctest` in a private build dir, `/W4 /WX` | **42/42 pass**; `ctest -N` lists all four tests above |
| `python -I tools/make_fixture_models.py --help` and `--self-test` on system Python 3.14, which has no `onnx` | both work (`ModuleNotFoundError: onnx` confirmed for a plain import) |
| CI | `.github/workflows/build.yml` is unchanged and runs `ctest` on linux-gcc, linux-clang, windows-msvc and macos-clang, so the first three tests are now gated on all four. I could not run GCC, Clang or macOS |

Remaining: the workflow has no `uv sync --frozen` + `pytest` step, so
`test_fixtures.py` (ONNX Runtime bit-exactness of the fixtures) and
`python_training_stage01_02` run only on a machine that has the venv. That was
item 4 of my proposed fix; it is the one part not done.

*[Done, 2026-10-18 17:00:* `build.yml` has a `python-m2-contract` job
(ubuntu, Python 3.12). It does not run `uv sync --frozen`, which would pull
the CUDA torch wheels. Instead it reads `uv.lock` with `tomllib` and installs
`numpy`, `onnx`, `onnxruntime` and `pytest` at their locked versions, plus
their unconditional dependencies, `protobuf` included. Then it runs
`pytest tests/test_contract.py tests/test_fixtures.py`. The extraction
script was dry-run locally and gives the lock's pins (`onnx==1.23.2`,
`onnxruntime==1.31.0`, `protobuf==7.36.2`, …). **The job itself has not run
yet**: it will run on the next push, and that run is also the first time
byte-identical fixture regeneration is checked on Linux. Stage 03–04 tests
and `test_prepare_data.py` / `test_eval_tools.py` stay venv-only.*]*
