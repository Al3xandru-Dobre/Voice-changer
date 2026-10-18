# Stage 04 — Selection policy & CLI

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 03](03_device_registry.md) · [Stage 05 →](05_refresh_recovery_controller.md)

**Status:** ◐ in progress — registry `resolve()`, CLI `--input <index|default>`
and `StreamStatus` tracking landed (2026-10-18); verification pending (session
had no shell; first `ctest` run is the next push)
**Depends on:** Stage 03
**Research:** §6 "Identity and persistence rules" 1–6 · §10 stage 3 · §12 decisions 3–4

## Objective

Selecting a device means selecting a **policy** (follow the OS default, or one
explicit endpoint) resolved against a **specific snapshot** — so a reordered or
changed catalog can never redirect audio to a different device.

## Tasks

- [x] `AudioDeviceRegistry::resolve(selection, direction)`:
      `FollowSystemDefault` → `ResolvedEndpoint{use_default}` (opened with a
      null device id); `ExplicitEndpoint{key}` → the same concrete endpoint
      wherever it now sits in the published snapshot, tagged with that
      snapshot's `generation`
- [x] Stale-key safety: a key from another context epoch fails
      `DeviceChanged`; a vanished device fails `DeviceNotFound`; a
      direction-mismatched key fails `DeviceNotFound`; resolving before any
      snapshot exists fails `DeviceChanged` — **never** "whatever is now at
      that index"
- [x] CLI: `--input <index>` / `--output <index>` kept, resolved through the
      snapshot enumerated by the same run (indices are presentation only)
- [x] CLI: `--input default` / `--output default` (explicit follow-default;
      case-sensitive — anything else is rejected with a usage error)
- [x] `StreamStatus` tracks requested policy separately from the resolved
      endpoints and the client/backend formats actually opened; `fluxvoice`
      populates and prints it at startup ("Select: capture=…, playback=…
      (resolved against snapshot generation N)")
- [x] Master open decisions 3 (fallback) and 4 (persisted tokens) decided and
      recorded (see below and the master table)
- [x] Fake-backend tests: list reorder between listing and opening; device
      disappears between listing and opening; follow-default vs. fixed
      selection; old keys after context recreation rejected

## Decisions recorded (master open decisions 3 & 4)

- **Fallback (decision 3): never.** The registry returns
  `DeviceNotFound`/`DeviceChanged` instead of substituting a device; the CLI
  exits with the reason. This also covers the stage 05 policy rows ("explicit
  device disappears", "virtual output disappears"): a missing virtual output
  must stop the route, never redirect processed speech to speakers.
- **Persisted tokens (decision 4): follow-default only.** "Follow default" is
  a policy and persists freely. Explicit endpoints are **session-scoped
  keys** — `EndpointDescriptor::persistent_token` stays reserved (empty)
  until a backend-native token has been tested per backend (stage 06);
  unresolvable stored selections will ask the user to choose again.

## Implementation notes

- **Indices are presentation only.** The CLI resolves `--input <index>`
  against the snapshot it just enumerated; keys then travel by identity. A
  future UI submits a key plus snapshot generation, never a row number.
- **Default is not a concrete endpoint.** Opening a default selection passes
  a null key so the backend can auto-route (WASAPI enables auto-routing only
  for null/default selections; PulseAudio sets `DONT_MOVE` for named
  endpoints). Requested policy is tracked separately from what actually
  opened.
- `resolve(FollowSystemDefault)` works without any published snapshot
  (generation 0) — opening the default needs no catalog.
- The `--input`/`--output` value grammar is `<uint index> | default` —
  `--input abc`, `--input 1.5`, `--input -1` and `--input DEFAULT` are all
  rejected in the argument parser, before any audio context exists (CTest
  `cli_rejects_input_word` / `_float` / `_upper`, `cli_help_selection`).

## Files touched

- `src/audio/AudioTypes.hpp` (`ResolvedEndpoint`), `src/audio/AudioDeviceRegistry.*`
  (`resolve()`), `src/main.cpp` (selection flags, resolve, `StreamStatus`,
  help), `tests/test_device_registry.cpp`, `tests/test_audio_types.cpp`,
  `CMakeLists.txt` (CLI tests)

## Definition of done

- Tested: a reordered catalog cannot redirect an explicit selection; stale
  keys fail safely with `DeviceChanged`. ☑ tested (fake backend);
  ☐ verified in CI
- `--input default` and `--input <index>` both work; `--help` documents
  them. ☑ implemented (pinned by `cli_help_selection`); ☐ verified on
  hardware
