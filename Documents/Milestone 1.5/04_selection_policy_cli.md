# Stage 04 — Selection policy & CLI

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 03](03_device_registry.md) · [Stage 05 →](05_refresh_recovery_controller.md)

**Status:** ☐ not started
**Depends on:** Stage 03
**Research:** §6 "Identity and persistence rules" 1–6 · §10 stage 3 · §12 decisions 3–4

## Objective

Selecting a device means selecting a **policy** (follow the OS default, or one
explicit endpoint) resolved against a **specific snapshot** — so a reordered or
changed catalog can never redirect audio to a different device.

## Tasks

- [ ] `DeviceSelection` resolution: `FollowSystemDefault` → open with a null
      device ID; `ExplicitEndpoint{key}` → copied ID from the *current* snapshot
- [ ] Selections carry the snapshot `generation` they were made against; a
      stale key returns `DeviceChanged` (or re-resolves under policy) — **never**
      "whatever is now at that index"
- [ ] CLI: keep `--input <index>` / `--output <index>` for manual smoke tests,
      resolved through the snapshot printed by the same run
- [ ] CLI: add `--input default` / `--output default` (explicit follow-default)
- [ ] `StreamStatus` tracks requested policy separately from the endpoint
      actually opened
- [ ] Decide & document master open decisions 3 (fallback) and 4 (persisted tokens)
- [ ] Fake-backend tests: list reorder between listing and opening; device
      disappears between listing and opening; fixed selection vs. follow-default;
      old keys after context recreation / backend change are rejected

## Implementation notes

- **Indices are presentation only.** A future UI submits key + generation,
  never a row number.
- **Default is not a concrete endpoint.** Opening "default" passes `nullptr`
  so the backend can auto-route (WASAPI enables auto-routing only for
  null/default selections; PulseAudio sets `DONT_MOVE` for named endpoints).
- Persist "follow default" freely. Persist explicit endpoints only as a
  versioned, backend-namespaced token once tested on that backend; if it can't
  be resolved unambiguously, ask the user to choose again.
- Recommended fallback rule: **never silently replace an explicitly chosen
  microphone or virtual output.**

## Files touched

- `src/audio/AudioDeviceRegistry.*`, `src/audio/AudioTypes.hpp`, `src/main.cpp`,
  CLI CTest cases, tests

## Definition of done

- Tested: a reordered catalog cannot redirect an explicit selection; stale keys
  fail safely with `DeviceChanged`.
- `--input default` and `--input <index>` both work; `--help` documents them.
