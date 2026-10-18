# Stage 01 — Cross-platform CI build matrix

> **Milestone 1.5** · [Master plan](00_master.md) · Next: [Stage 02 →](02_audio_types_contract.md)

**Status:** ◐ in progress — workflow + README landed (2026-10-18); first green run
pending a push to GitHub
**Depends on:** Milestone 1 / Stage 01 (CMake + CTest)
**Estimated scope:** CI configuration only, no audio logic

## Objective

Every push builds `fluxvoice` and runs `ctest` on Linux, Windows and macOS, so
portability regressions are caught before any device-management code lands.
Milestone 1 / Stage 01 was only verified on Linux (GCC 13, Clang 18) — and
once, manually, on Windows (MSVC 19.50, M1 / Stage 05).

## Context & references

- Research §11: *"Build on Linux/GCC or Clang, Windows/MSVC, and macOS/Apple
  Clang using the pinned dependencies. CI compilation and fake tests establish
  portability of the code, not hardware or permission correctness."*
- `CMakeLists.txt` already has per-OS link lines, `FLUXVOICE_WARNINGS_AS_ERRORS`
  and CTest CLI smoke tests (no audio device needed).
- Repository is on GitHub (`origin`) → GitHub Actions.

## Tasks

- [x] `.github/workflows/build.yml`: matrix
      `ubuntu-latest` (GCC + Clang) · `windows-latest` (MSVC) · `macos-latest` (Apple Clang)
- [x] Configure with `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`, build, `ctest --output-on-failure`
- [x] Cache the FetchContent `_deps` directory keyed on `CMakeLists.txt` hash
      (`deps-<job>-<hash>` with a job-prefixed `restore-keys` fallback)
- [ ] Fix whatever MSVC / Apple Clang surface. MSVC is known-clean at `/W4 /WX`
      locally (19.50, M1 / Stage 05: `NOMINMAX` on `miniaudio_headers`, C4324
      silenced around `DuplexRouter`'s queues). Apple Clang compiles this
      project for the first time in CI — expect the first run to be the judge
- [ ] Confirm the CLI tests pass on CI runners **without audio hardware** —
      they must never open a device (argument errors are rejected in the parse
      loop, before `AudioEngine` is even constructed; `--help` exits earlier
      still; the first green CI run is the proof and is still pending)
- [x] Record supported OS versions (master open decision 1) in README §6:
      Windows 10+ · macOS 13+ · current Ubuntu LTS (24.04)

## Implementation notes

- CI runners have no sound cards; miniaudio may fall back to the null backend.
  Tests that need a device belong to stage 06's manual matrix, not CI.
- Pin action versions (`actions/checkout@<tag>`), same reasoning as pinning
  `GIT_TAG`: reproducibility and supply-chain hygiene. Both actions are pinned
  to `v5` (the node-24 runtime generation; node-20 actions are being retired
  from runners as of late 2026).
- Windows: the Visual Studio generator is multi-config → build with
  `--config Release` and `ctest -C Release` (both flags are harmless no-ops on
  the single-config generators).
- Compilers are selected per matrix entry via `CC`/`CXX` exported with
  `$GITHUB_ENV` — but only when the entry defines them, so `windows-msvc`
  never sees an empty `CC` variable.
- No extra apt packages are needed: miniaudio 0.11.25 `dlopen`s ALSA /
  PulseAudio at runtime (that is why `fluxvoice_link_audio` links only
  `dl`/`m` on Linux), so a headerless runner still builds.
- Runner labels as of 2026-10: `ubuntu-latest` = 24.04, `windows-latest` =
  Server 2025 with VS 2026, `macos-latest` = macOS 26 arm64. The MSVC
  verification from M1 / Stage 05 was VS 18 / cl 19.50, i.e. the same
  toolchain line CI uses.
- The 2026-10-18 session that landed this workflow had no shell available on
  the dev machine, so nothing was executed locally — the first push runs it
  for real.
- 2026-10-19 portability audit (local MSVC only): first-party headers made
  self-contained (owning standard includes, `std::`-qualified
  size/fixed-width types); MSVC-style vs GNU-style warning flags now chosen
  via `CMAKE_CXX_COMPILER_FRONTEND_VARIANT` so clang-cl gets `/W4 /WX`;
  explicit `psapi` link replaces any reliance on `#pragma comment(lib)`;
  ALL-built `fluxvoice_header_checks` OBJECT target compiles every
  `src/*.hpp`, `tools/LatencyProbe.hpp` and `tests/TestSupport.hpp` in
  isolation; new `process_memory` CTest covers the `<windows.h>`/`<psapi.h>`
  NOMINMAX guard and link; macOS legs configure with
  `-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0` and every job logs `cmake --version`
  (non-Windows also `"$CXX" --version`). Verified locally on VS BuildTools
  2026 (cl 19.51); Linux GCC/Clang and AppleClang stay unverified until the
  first green CI push.

## Files touched

- `.github/workflows/build.yml` (new), `README.md` (§6 platforms + §5 tree +
  CI note)

## Definition of done

- Green CI on all four compiler/OS combinations with warnings-as-errors.
- `ctest` runs (not skipped) on all of them.
