# Stage 02 — Audio types & contract

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 01](01_ci_build_matrix.md) · [Stage 03 →](03_device_registry.md)

**Status:** ☐ not started
**Depends on:** Stage 01 (CI), Milestone 1 / Stage 02 (`AudioEngine`)
**Research:** §6 "Application-facing data contract", "Identity and persistence rules" · §10 stage 1

## Objective

Define the application-facing vocabulary for devices, selections, errors and
stream status, so CLI, future UI and tests can describe devices **without
including `<miniaudio.h>` or touching `ma_device_id`**.

## Tasks

- [ ] New `src/audio/AudioTypes.hpp` (no miniaudio include):

  | Type | Purpose |
  |---|---|
  | `EndpointDirection` | `Capture` / `Playback` — never inferred from a name |
  | `EndpointKey` | Opaque session key, scoped to backend/context epoch + direction |
  | `EndpointDescriptor` | Key, display name, direction, default flag, optional persistent token |
  | `DeviceSnapshot` | Owned capture + playback descriptors, backend identity, monotonic `generation` |
  | `DeviceSelection` | `FollowSystemDefault` or `ExplicitEndpoint{key}` |
  | `AudioCapabilities` | Ordinary I/O, native loopback, monitor capture, virtual routing — each `Supported / Unsupported / Unknown` |
  | `AudioError` | Operation, category, original `ma_result` (as int), backend, user-facing message |
  | `StreamStatus` | Controller state, requested selection, resolved endpoint, client vs. backend formats |

- [ ] `Result<T>` = `std::variant<T, AudioError>` (C++20; no `std::expected`)
- [ ] `AudioError` categories: `BackendUnavailable`, `NullBackendOnly`,
      `EnumerationFailed`, `DeviceNotFound`, `DeviceChanged`, `DeviceBusy`,
      `PermissionDenied`, `FormatUnsupported`, `Unknown`
- [ ] Private mapping layer (`src/audio/MiniaudioBackend.*`) owns
      `EndpointKey ↔ ma_device_id`; keys compare with `ma_device_id_equal()`
- [ ] Refactor `AudioEngine.hpp` so `AudioDeviceInfo::id` (a raw `ma_device_id`)
      is no longer public
- [ ] Unit tests: descriptors constructible without miniaudio; key equality is
      direction- and epoch-scoped

## Implementation notes

- **Names are labels, not keys** — two identical USB mics must yield two keys.
- **Never serialize or hash the raw `ma_device_id` union** as a permanent ID.
  Persistent tokens are a later, per-backend, best-effort addition (stage 04).
- Represent "not probed" as `Unknown`, never as `Unsupported`.
- Keep the backend seam small (an abstract enumerate/open interface) — only so
  stage 03 tests can inject a fake. No plugin system.
- `MINIAUDIO_IMPLEMENTATION` stays in `src/main.cpp` (or moves to exactly one
  new `.cpp` — never two).

## Files touched

- `src/audio/AudioTypes.hpp` (new), `src/audio/MiniaudioBackend.{hpp,cpp}` (new),
  `src/audio/AudioEngine.hpp`, `CMakeLists.txt` (test target)

## Definition of done

- `grep -rl 'miniaudio.h\|ma_device_id' src/` lists only `src/audio/` backend
  implementation files.
- Type/contract tests pass in CI on all three OSes.
