# Stage 03 — Device registry

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 02](02_audio_types_contract.md) · [Stage 04 →](04_selection_policy_cli.md)

**Status:** ◐ in progress — registry, fake-backend tests and `--list-devices`
diagnostics landed (2026-10-18); verification pending (session had no shell;
first `ctest` run is the next push, stage 01's CI matrix covers the rest)
**Depends on:** Stage 02
**Research:** §5 "Enumeration is a snapshot", backend matrix · §7 refresh algorithm steps 1–2 · §10 stage 2

## Objective

An `AudioDeviceRegistry` that owns one miniaudio context, enumerates both
directions in a single serialized call, and publishes **owned, immutable
snapshots** — with failures, empty catalogs and null-backend operation clearly
distinguishable.

## Tasks

- [x] `AudioDeviceRegistry` over exactly one backend/context, whose address is
      stable for its lifetime. Ownership note: `AudioEngine` owns the
      `MiniaudioBackend` today and exposes it via `audio_backend()`; the
      registry holds the seam reference (`AudioBackend&`), so catalog keys and
      stream opens resolve against the same epoch. Stage 05's `AudioController`
      takes over owning the whole lifecycle
- [x] Init failure detail preserved: the context is created by
      `MiniaudioBackend`'s constructor (once, address-stable — a lazy
      re-`init()` would violate that), and the **exact** `ma_context_init`
      result plus the selected backend name surface through
      `BackendStatus{Unavailable, backend, native_result}` /
      `AudioError::native_result`. This is the stage plan's "init() preserves
      the exact ma_result", just at construction time
- [x] `refresh()` → `Result<DeviceSnapshot>`: one `ma_context_get_devices()`
      call for both directions; everything copied before returning
- [x] Publish a new `generation` only when keys, names or default flags
      actually change — **a reordered catalog is not a change** (order-
      insensitive `same_catalog` bijection by key; keys are unique per
      snapshot)
- [x] Detect `ma_backend_null` → `refresh()` returns `NullBackendOnly` /
      offline mode, never a working empty catalog; `--list-devices` prints
      "offline/test mode" explicitly
- [x] Backend diagnostics: `--list-devices` prints the selected backend name
      (`ma_get_backend_name`) and the snapshot generation
- [x] On refresh failure the previous snapshot is retained and marked
      **stale** (`registry.stale()`); recovery with unchanged content keeps
      the same generation and clears the flag
- [x] `capabilities(key)` → `Result<EndpointInfo>` via
      `ma_context_get_device_info()` (`nativeDataFormats[]` envelope); on
      demand for a selected device only — never started capture, never used to
      populate a list (pinned by a probe-call-count test)
- [x] Fake backend + tests (`tests/test_device_registry.cpp`, CTest
      `device_registry`): 0/1/many devices per direction; duplicate display
      names with distinct keys; empty-success vs. enumeration failure vs.
      null backend vs. unavailable backend (four distinguishable outcomes);
      stale-on-failure and recovery

## Implementation notes

- `ma_context_get_devices()` arrays are context-owned and invalidated by the
  next call; the API warns against concurrent use. **Serialized on the
  control thread and copied immediately** (research §5; BEST_PRACTICES §2.4).
- `ma_context_init(nullptr, …)` picks *one* backend in priority order; it does
  not merge inventories. On Linux 0.11.25 prefers PulseAudio over ALSA;
  PipeWire is reached via `pipewire-pulse` (no native `ma_backend_pipewire`).
- Successful context init with the null backend is **not** proof of hardware
  (the backend search list includes null). The registry refuses to publish a
  catalog from it and reports `NullBackendOnly`; the CLI stream path stops
  there instead of opening null devices.
- WASAPI default endpoints use the `eConsole` role — not labeled "default
  communications device" anywhere.
- The fake backend's identity model: tokens minted per (direction, name,
  occurrence), stable across enumerations — mirroring how the real backend
  keys by native id (two same-named devices get distinct tokens).
- Resolving selections against a **stale** snapshot is allowed (the retained
  catalog is the best information available); the stage 05 controller decides
  re-resolution policy.

## Files touched

- `src/audio/AudioDeviceRegistry.{hpp,cpp}` (new), `src/audio/AudioBackend.hpp`
  (seam: `status()` + `probe()`), `src/audio/MiniaudioBackend.{hpp,cpp}`
  (exact init result, `probe()`), `src/audio/AudioEngine.hpp`
  (`audio_backend()`), `src/audio/AudioTypes.hpp` (`BackendStatus`,
  `EndpointInfo`, `NativeFormat`), `src/main.cpp` (`--list-devices`),
  `tests/test_device_registry.cpp` (new), `tests/test_audio_types.cpp`,
  `CMakeLists.txt`

## Definition of done

- Duplicate names, empty catalogs, null backend and enumeration failures
  produce four distinguishable, tested outcomes. ☑ tested (fake backend);
  ☐ verified in CI
- `--list-devices` output includes backend name; behaves the same on all CI
  OSes. ☐ CI run pending
