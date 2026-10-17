# Stage 01 — Cross-platform CI build matrix

> **Milestone 1.5** · [Master plan](00_master.md) · Next: [Stage 02 →](02_audio_types_contract.md)

**Status:** ☐ not started
**Depends on:** Milestone 1 / Stage 01 (CMake + CTest)
**Estimated scope:** CI configuration only, no audio logic

## Objective

Every push builds `fluxvoice` and runs `ctest` on Linux, Windows and macOS, so
portability regressions are caught before any device-management code lands.
Milestone 1 / Stage 01 was only verified on Linux (GCC 13, Clang 18).

## Context & references

- Research §11: *"Build on Linux/GCC or Clang, Windows/MSVC, and macOS/Apple
  Clang using the pinned dependencies. CI compilation and fake tests establish
  portability of the code, not hardware or permission correctness."*
- `CMakeLists.txt` already has per-OS link lines, `FLUXVOICE_WARNINGS_AS_ERRORS`
  and CTest CLI smoke tests (no audio device needed).
- Repository is on GitHub (`origin`) → GitHub Actions.

## Tasks

- [ ] `.github/workflows/build.yml`: matrix
      `ubuntu-latest` (GCC + Clang) · `windows-latest` (MSVC) · `macos-latest` (Apple Clang)
- [ ] Configure with `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`, build, `ctest --output-on-failure`
- [ ] Cache the FetchContent `_deps` directory keyed on `CMakeLists.txt` hash
- [ ] Fix whatever MSVC / Apple Clang surface (expect: `/W4` warnings in our
      sources, missing includes that libstdc++ provided transitively).
      Already seen with MSVC 19.50 (2026-10-17) and fixed in M1 / Stage 05:
      `windows.h` (via miniaudio) defines `min`/`max` macros that broke
      `std::min`/`std::max` (`NOMINMAX` on `miniaudio_headers`), and C4324
      "structure was padded" on `DuplexRouter`'s queues failed `/WX`
      (silenced in that header). Everything builds locally with `/W4 /WX`;
      CI still to come
- [ ] Confirm the CLI tests pass on CI runners **without audio hardware** —
      they must never open a device
- [ ] Record supported OS versions (master open decision 1) in README §6

## Implementation notes

- CI runners have no sound cards; miniaudio may fall back to the null backend.
  Tests that need a device belong to stage 06's manual matrix, not CI.
- Pin action versions (`actions/checkout@<tag>`), same reasoning as pinning
  `GIT_TAG`: reproducibility and supply-chain hygiene.
- Windows: the Visual Studio generator is multi-config → `ctest -C Release`.

## Files touched

- `.github/workflows/build.yml`, `CMakeLists.txt` (fixes only), `README.md`

## Definition of done

- Green CI on all four compiler/OS combinations with warnings-as-errors.
- `ctest` runs (not skipped) on all of them.
