# Stage 01 — Build system & dependencies

> **Milestone 1** · [Master plan](00_master.md) · Next: [Stage 02 →](02_duplex_audio_io.md)

**Status:** ☑ done (2026-10-17)
**Depends on:** —
**Estimated scope:** build only, no audio logic

## Objective

A reproducible `cmake -S . -B build && cmake --build build` that compiles
`fluxvoice` on Linux/macOS/Windows with pinned dependencies — no global
installs, no floating `master` checkouts.

## Context & references

- `CMakeLists.txt` (repo root) — already implements FetchContent pins.
- RESEARCH_GUIDE §4 — the two M1 deps are header-only: miniaudio (audio I/O),
  readerwriterqueue (SPSC queue). ONNX Runtime is deliberately *excluded* until
  M2 to keep the M1 build self-contained.

## Tasks

- [x] `CMakeLists.txt`: C++20, `FetchContent` for miniaudio `0.11.25` +
      readerwriterqueue `v1.0.7` (`SOURCE_SUBDIR` trick — headers only, skip
      their own CMake projects)
- [x] `ONNXRUNTIME_ROOT` cache var plumbing (no-op until M2)
- [x] Strict warnings: `-Wall -Wextra -Wpedantic` / `/W4`
- [x] `.gitignore`: `build/`, `models/*.onnx`, vendored `third_party` contents
- [x] Verify build on a clean clone (`git clean` or fresh dir)
- [x] Confirm binary runs: `./build/fluxvoice --help`
- [x] `FLUXVOICE_WARNINGS_AS_ERRORS` option (`-Werror` / `/WX`) for CI
- [x] CTest smoke tests for the CLI (`ctest --test-dir build`)

## Implementation notes

- **Pin `GIT_TAG` to real tags/commits** — `master`/`latest` makes builds
  non-reproducible and is a supply-chain risk.
- Keep header-only deps as `INTERFACE` targets with `SYSTEM` includes so their
  warnings don't pollute our build.
- `MINIAUDIO_IMPLEMENTATION` must be defined in exactly one TU — `src/main.cpp`
  owns it (commented there). Never define it in a header.
- Platform audio backends: Linux needs `dl pthread m`; macOS needs CoreAudio /
  AudioToolbox / AVFoundation frameworks; Windows needs `ole32 winmm`.
- `FLUXVOICE_WARNINGS_AS_ERRORS` uses explicit flags rather than the
  `COMPILE_WARNING_AS_ERROR` property, which needs CMake ≥ 3.24 (we require 3.22).
- CLI numeric options are parsed with `std::from_chars` and range-checked
  (`--input`/`--output` ≥ 0, `--sample-rate` in [8000, 192000]). Missing or
  malformed values exit 1 *before* any audio device is opened — previously
  `std::stoi` aborted on `--input abc` and a missing value silently fell back
  to the default device.
- CLI tests use `WILL_FAIL` for the rejection cases; CTest still reports a
  crash (signal/abort) as a failure under `WILL_FAIL`, so they catch regressions
  to the old `std::stoi` abort.

## Verification (2026-10-17)

Fresh copy of the tree (no `build/`, no `.git`), Ubuntu 24.04, CMake 3.28.3:

| Compiler | Configure + build | Warnings (ours) | `ctest` |
|---|---|---|---|
| GCC 13.3.0 | ✓ (with `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`) | 0 | 8/8 pass |
| Clang 18.1.3 | ✓ (with `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`) | 0 | 8/8 pass |

- `fluxvoice --help` / `-h` print usage and exit 0.
- `--bogus`, `--input`, `--input abc`, `--output -1`, `--sample-rate 16000hz`,
  `--sample-rate 0` each exit 1 with a diagnostic.
- macOS and Windows builds not yet verified (no hardware in this pass) — the
  platform link lines are in place; re-check when Stage 05 is exercised on
  those platforms.

## Files touched

- `CMakeLists.txt`, `.gitignore`, `src/main.cpp`, `src/audio/OverlapAdd.hpp`
  (missing `<algorithm>` for `std::fill`), `README.md` (§6 test/CI notes)

## Definition of done

- [x] Clean-configure + clean-build succeeds with zero warnings from *our* sources.
- [x] `fluxvoice --help` prints usage and exits 0.
