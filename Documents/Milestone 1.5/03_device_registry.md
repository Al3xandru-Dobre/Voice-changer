# Stage 03 — Device registry

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 02](02_audio_types_contract.md) · [Stage 04 →](04_selection_policy_cli.md)

**Status:** ☐ not started
**Depends on:** Stage 02
**Research:** §5 "Enumeration is a snapshot", backend matrix · §7 refresh algorithm steps 1–2 · §10 stage 2

## Objective

An `AudioDeviceRegistry` that owns one miniaudio context, enumerates both
directions in a single serialized call, and publishes **owned, immutable
snapshots** — with failures, empty catalogs and null-backend operation clearly
distinguishable.

## Tasks

- [ ] `AudioDeviceRegistry` owning one `ma_context` (address stable for its lifetime)
- [ ] `init()` preserves the exact `ma_result` and selected backend on failure
- [ ] `refresh()` → `Result<DeviceSnapshot>`: one `ma_context_get_devices()`
      call for both directions; copy everything before returning
- [ ] Publish a new `generation` only when IDs, directions, names or default
      flags actually change
- [ ] Detect `ma_backend_null` → report as `NullBackendOnly` / offline mode,
      **not** a working microphone
- [ ] Backend diagnostics: `--list-devices` prints selected backend name
      (`ma_get_backend_name`) and the snapshot generation
- [ ] On refresh failure keep the previous snapshot, marked **stale**
- [ ] `capabilities(key)` via `ma_context_get_device_info()` — on demand for a
      selected device only; never start capture to populate a list
- [ ] Fake backend + tests: 0 / 1 / many devices per direction; duplicate
      display names; empty-success vs. enumeration failure; null backend;
      stale-on-failure

## Implementation notes

- `ma_context_get_devices()` arrays are context-owned and invalidated by the
  next call; the API warns against concurrent use. **Serialize on the control
  thread and copy immediately.** (research §5)
- `ma_context_init(nullptr, …)` picks *one* backend in priority order; it does
  not merge inventories. On Linux 0.11.25 prefers PulseAudio over ALSA; PipeWire
  is reached via `pipewire-pulse` (there is no native `ma_backend_pipewire`).
- Successful context init with the null backend is **not** proof of hardware.
- WASAPI default endpoints use the `eConsole` role — don't label them
  "default communications device".

## Files touched

- `src/audio/AudioDeviceRegistry.{hpp,cpp}` (new), `src/audio/MiniaudioBackend.*`,
  `src/main.cpp` (`--list-devices`), tests

## Definition of done

- Duplicate names, empty catalogs, null backend and enumeration failures produce
  four distinguishable, tested outcomes.
- `--list-devices` output includes backend name; behaves the same on all CI OSes.
