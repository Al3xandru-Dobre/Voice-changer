# Milestone 1.5 — Cross-platform audio devices

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md) ·
> Research basis: [Cross-platform audio I/O research](../Research_documents/CROSS_PLATFORM_AUDIO_IO_RESEARCH.md)

**Status:** ☐ not started
**Goal:** *One application-facing audio-device API on Windows, macOS and Linux:
discover capture/playback endpoints and OS defaults, select them reliably
without depending on list position or name, refresh after hardware/default
changes, recover safely from device loss, and report actionable errors.*

"OS-agnostic" here means **portable application code and consistent
behavior** — not identical audio capabilities on every OS, and not the absence
of platform-specific setup (virtual cables, permissions).

## Why this milestone

Milestone 1 proves the real-time substrate with a thin `AudioEngine` and
index-based CLI selection. That is enough for a loopback smoke test but not for
a product someone runs daily:

- **Indices are unstable.** Plugging in a headset reorders the list; `--input 2`
  can silently become a different microphone.
- **Failures are invisible.** An empty device list and a failed enumeration both
  return `{}`; a null backend "succeeds" with no real hardware.
- **No lifecycle.** A device unplugged mid-stream leaves `dev_ok_ == true` and
  the app reporting "running".
- **Only Linux/GCC/Clang has ever been built.**

The research concluded that **miniaudio 0.11.25 stays** — the work is safe
identity, snapshots, selection policy and lifecycle management on top of it,
not an audio backend per OS (research §1, §4).

## Placement & dependencies

```text
M1 stages 01–03 (build, duplex, SPSC) ──► M1.5 ──► M4 UI device selectors
                                    │
                                    └──► M2 (neural) proceeds in parallel
```

- **Depends on:** Milestone 1 stages 01–03 (build, `AudioEngine`, SPSC queues).
  M1.5 stage 06 extends M1 stages 05–06 (virtual cable, verification) to the
  full OS matrix — M1 may close on Linux alone first.
- **Does not block** Milestone 2/3: the neural pipeline only sees the SPSC
  queues, whose contract is unchanged.
- **Blocks** Milestone 4's device dropdowns, which must consume the same
  registry and selection rules as the CLI (research stage 5).

## Scope

**In scope**
- CI build matrix: Linux GCC/Clang, Windows MSVC, macOS Apple Clang
- Application-facing audio types; native/miniaudio IDs kept private
- `AudioDeviceRegistry`: owned context, copied snapshots, typed errors, backend diagnostics
- Selection policy: follow-default vs. explicit endpoint; stale-key safety; CLI on top
- `AudioController`: refresh, notification hints, bounded retries, safe stop/reopen
- Fake-backend unit tests (no hardware) + per-OS hardware validation matrix
- Optional, capability-gated Windows WASAPI system-playback capture

**Out of scope**
- ImGui device selectors → Milestone 4 (consumes this milestone's API)
- Native macOS system capture (Core Audio process taps / ScreenCaptureKit) — separate investigation
- Installing drivers or creating virtual devices automatically
- Replacing miniaudio (no measured problem justifies it)
- Clock-drift resampling — measure first (M1 stage 06 / M1.5 stage 06); add only if needed

## Stages

| # | Stage | Research §10 | Output | Status |
|---|---|---|---|---|
| 01 | [Cross-platform CI build matrix](01_ci_build_matrix.md) | §11 build matrix | Linux/Windows/macOS builds + `ctest` on every push | ☐ |
| 02 | [Audio types & contract](02_audio_types_contract.md) | Stage 1 | `AudioTypes.hpp`; no `<miniaudio.h>` outside `src/audio/` impl | ☐ |
| 03 | [Device registry](03_device_registry.md) | Stage 2 | Snapshots; empty vs. failed vs. null-backend distinguishable | ☐ |
| 04 | [Selection policy & CLI](04_selection_policy_cli.md) | Stage 3 | Reordered catalogs cannot redirect an explicit selection | ☐ |
| 05 | [Refresh & recovery controller](05_refresh_recovery_controller.md) | Stage 4 | Plug/unplug/default changes follow the policy table | ☐ |
| 06 | [Routing & OS validation](06_routing_os_validation.md) | Stage 6 | Hardware matrix passed per OS; optional WASAPI loopback | ☐ |
| — | UI device selectors | Stage 5 | → [Milestone 4](../Milestone%204/00_master.md) | — |

## Acceptance criteria

- [ ] `fluxvoice` builds warning-free and passes `ctest` on Linux, Windows and macOS in CI
- [ ] No file outside the audio implementation includes `<miniaudio.h>` or handles `ma_device_id`
- [ ] Fake-backend tests cover every case in research §11 "Deterministic tests without hardware"
- [ ] An explicitly selected device is **never** silently replaced (mic or virtual output)
- [ ] Unplug/replug and default-device changes behave per the stage 05 policy table on each OS
- [ ] Each advertised routing mode is recorded by a second application on its target OS
- [ ] ≥ 30 min streaming between different capture/playback devices without sustained queue growth

## Key constraints

1. **Real-time rules from Milestone 1 still hold.** Callbacks never enumerate,
   open, close, log or allocate. Notification callbacks only set lock-free
   atomic flags; all lifecycle work happens on the control thread.
2. **One control thread** owns enumeration, capability queries, open/close and
   retries. UI/CLI read immutable snapshot copies.
3. **Never retain miniaudio's enumeration arrays** — they are invalidated by the
   next call. Copy into owned snapshots.
4. **Never treat "not probed" as "unsupported"**, nor "no devices" as "failure".
5. **`MINIAUDIO_IMPLEMENTATION` stays in exactly one TU** if audio code moves
   out of `main.cpp`.

## Open decisions (from research §12)

| # | Decision | Recommended default | Needed by |
|---|---|---|---|
| 1 | Supported OS versions / distribution targets | Win 10+, macOS 13+, current Ubuntu LTS (confirm) | Stage 01 |
| 2 | Is system-playback capture a release requirement? | No — mic → virtual output first; WASAPI loopback optional | Stage 06 |
| 3 | Is automatic fallback ever allowed? | **Never** silently replace an explicit mic or virtual output | Stage 04 |
| 4 | Which backend-native tokens are persisted? | Persist "follow default" only until tokens are tested per backend | Stage 04 |
| 5 | Refresh interval, routing, drift behavior | 1 s polling; validate on real machines | Stages 05–06 |
| 6 | Online verification of miniaudio/OS docs & driver terms | Do before stage 06 / any release | Stage 06 |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Over-engineering into a plugin system | Concrete classes + one small backend seam for the fake; no native adapters until a verified gap |
| Refresh/open race selects the wrong device | Selections carry snapshot generation; stale keys fail with "device changed" |
| Notification coverage differs per backend | Notifications are hints only; periodic re-enumeration is the portable baseline |
| Silent reroute of processed speech to speakers | Missing virtual output stops the route; never falls back |
| Independent device clocks drift | Measure queue occupancy in soak tests; bound queued latency; resample only if measured |
| CI passes but hardware/permissions fail | Keep manual OS evidence (stage 06) separate from automated results |

## References

- [Cross-platform audio I/O research](../Research_documents/CROSS_PLATFORM_AUDIO_IO_RESEARCH.md) — full rationale (§5 verified miniaudio boundaries, §6 architecture, §7 policy, §11 tests).
- `miniaudio.h` 0.11.25 — `ma_context_get_devices`, `ma_context_get_device_info`, `ma_device_id_equal`, `notificationCallback`, `ma_context_is_loopback_supported`.
- [Milestone 1](../Milestone%201/00_master.md) — real-time rules this milestone must preserve.
