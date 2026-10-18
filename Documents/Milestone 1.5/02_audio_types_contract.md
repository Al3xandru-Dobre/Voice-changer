# Stage 02 — Audio types & contract

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 01](01_ci_build_matrix.md) · [Stage 03 →](03_device_registry.md)

**Status:** ◐ in progress — types, backend seam, `MiniaudioBackend`,
`AudioEngine`/`main.cpp` refactor and type tests landed (2026-10-18);
verification pending (the session had no shell, so no local build; first
`ctest` run is the next push, stage 01's CI matrix covers the rest)
**Depends on:** Stage 01 (CI), Milestone 1 / Stage 02 (`AudioEngine`)
**Research:** §6 "Application-facing data contract", "Identity and persistence rules" · §10 stage 1

## Objective

Define the application-facing vocabulary for devices, selections, errors and
stream status, so CLI, future UI and tests can describe devices **without
including `<miniaudio.h>` or touching `ma_device_id`**.

## Tasks

- [x] New `src/audio/AudioTypes.hpp` (no miniaudio include):

  | Type | Purpose |
  |---|---|
  | `EndpointDirection` | `Capture` / `Playback` — never inferred from a name |
  | `EndpointKey` | Opaque session key `{direction, epoch, token}`, epoch- and direction-scoped; defaulted `==` + total `<` |
  | `EndpointDescriptor` | Key, display name, default flag, optional persistent token (direction lives in the key) |
  | `DeviceSnapshot` | Owned capture + playback descriptors, backend identity, epoch, monotonic `generation`; `endpoints(dir)` accessor |
  | `DeviceSelection` | `variant<FollowSystemDefault, ExplicitEndpoint{key}>` |
  | `AudioCapabilities` | Ordinary I/O, native loopback, monitor capture, virtual routing — each `Unknown / Supported / Unsupported` |
  | `AudioError` | Category, operation, original `ma_result` (as int), backend, user-facing message |
  | `StreamStatus` | Controller state, requested policy per direction, resolved endpoints, client vs. backend rates |

- [x] `Result<T>` = `std::variant<T, AudioError>` + `is_ok()` helper (C++20; no `std::expected`)
- [x] `AudioError` categories: `BackendUnavailable`, `NullBackendOnly`,
      `EnumerationFailed`, `DeviceNotFound`, `DeviceChanged`, `DeviceBusy`,
      `PermissionDenied`, `FormatUnsupported`, `Unknown` — each with a stable
      `to_string()` pinned by a test
- [x] Private mapping layer (`src/audio/MiniaudioBackend.{hpp,cpp}`) owns
      `EndpointKey ↔ ma_device_id`; ids are compared with
      `ma_device_id_equal()`, never hashed or serialized. The id table
      persists across `enumerate()` calls within one context, so the same
      physical device keeps the same key when snapshots reorder
- [x] Refactored `AudioEngine.hpp`: `AudioDeviceInfo` is gone (its public
      `ma_device_id` field with it). `open()` takes `const EndpointKey*`
      (nullptr = OS default), resolves through the backend, and enumeration
      returns `EndpointDescriptor`s. Behavior is otherwise M1-identical
      (empty list on failure — typed errors are stage 03's registry)
- [x] Unit tests (`tests/test_audio_types.cpp`, CTest `audio_types`): the TU
      includes only `AudioTypes.hpp` and links no audio dependency, so it
      cannot compile if the contract ever grows a miniaudio include; key
      equality is direction- and epoch-scoped; ordering agrees with equality;
      tri-state capabilities keep `Unknown ≠ Unsupported`; every error
      category has a distinct name; `Result` carries value-or-error

## Implementation notes

- **Names are labels, not keys** — two identical USB mics must yield two keys
  (distinct native ids → distinct tokens; a backend reporting duplicate ids
  for two devices is a backend-level identity gap, surfaced as-is).
- **Never serialize or hash the raw `ma_device_id` union** as a permanent ID.
  The key's `token` is a backend-assigned integer meaningful only inside the
  process. Persistent tokens are a later, per-backend, best-effort addition
  (stage 04).
- Epochs are drawn from a process-wide atomic counter starting at 1, so a
  recreated context (new `MiniaudioBackend`) rejects every old key by epoch
  alone; epoch 0 is "never minted" (the default-constructed key matches
  nothing).
- Represent "not probed" as `Unknown`, never as `Unsupported`
  (`MiniaudioBackend::capabilities()` sets only what miniaudio can actually
  see: ordinary I/O and native loopback; monitor capture and virtual routing
  stay `Unknown` — server configuration is invisible to the backend).
- The backend seam is [`AudioBackend.hpp`](../../src/audio/AudioBackend.hpp)
  (abstract `enumerate()` / `capabilities()` / `backend_name()`, in
  `AudioTypes` vocabulary only) — the injection point for stage 03's fake.
  Stream open stays on `AudioEngine` until the controller work (stage 05).
- `MINIAUDIO_IMPLEMENTATION` moved from `src/main.cpp` to
  `src/audio/MiniaudioBackend.cpp` — still exactly one TU of this binary.
  `tools/fv_probe.cpp` keeps its own (separate executable).
- `main.cpp` no longer includes `<miniaudio.h>` at all: device indices are
  resolved to `EndpointKey`s and passed to `AudioEngine::open()`.

## Files touched

- `src/audio/AudioTypes.hpp` (new), `src/audio/AudioBackend.hpp` (new — the
  seam), `src/audio/MiniaudioBackend.{hpp,cpp}` (new),
  `src/audio/AudioEngine.hpp` (keyed API, backend-owned context),
  `src/main.cpp` (miniaudio-free), `tests/test_audio_types.cpp` (new),
  `CMakeLists.txt` (`MiniaudioBackend.cpp` in the fluxvoice target,
  `audio_types` test), `README.md` §5 tree

## Definition of done

- `grep -rl 'miniaudio.h\|ma_device_id' src/` lists only `src/audio/`
  implementation files — after this stage:
  `src/audio/MiniaudioBackend.hpp`, `src/audio/MiniaudioBackend.cpp` (the
  mapping layer) and `src/audio/AudioEngine.hpp` (the stream half, which
  still owns the `ma_device`). `src/main.cpp` and everything outside
  `src/audio/` are clean. The stream's move behind the seam is stage 05.
- Type/contract tests pass in CI on all three OSes.
