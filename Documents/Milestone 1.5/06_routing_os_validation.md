# Stage 06 — Routing & OS validation

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 05](05_refresh_recovery_controller.md)

**Status:** ☐ not started
**Depends on:** Stage 05, Milestone 1 / Stages 05–06 (virtual cable + latency method)
**Research:** §5 backend matrix · §9 · §10 stage 6 · §11 hardware/OS matrix

## Objective

Prove on real machines that every routing mode we advertise works on its
target OS — and that unsupported modes say so explicitly. Milestone 1 / Stage 05
validates the virtual-cable route on the dev machine; this stage extends it to
Windows/WASAPI, macOS/Core Audio and Linux/PipeWire-Pulse (+ PulseAudio / ALSA
where advertised).

## Routing modes

| Mode | Windows / WASAPI | macOS / Core Audio | Linux / PipeWire-Pulse |
|---|---|---|---|
| Mic → FluxVoice → virtual mic | VB-Cable playback endpoint; receiver picks its capture endpoint | BlackHole (or configured virtual device) | Null sink; receiver picks `.monitor` or a configured virtual source |
| System-playback capture (optional) | Native `ma_device_type_loopback`, gated by `ma_context_is_loopback_supported()` | **Not** via miniaudio 0.11.25 — unsupported, or a configured virtual route | Sink monitor source opened as ordinary capture |

## Tasks

- [ ] Run the hardware matrix below on each OS; record evidence (OS version,
      backend, devices, date) in the results table — separate from CI results
- [ ] Optional **Windows "system output" capture mode**: separate operation from
      mic selection; `capture.pDeviceID` is a **playback** endpoint ID; gated by
      `ma_context_is_loopback_supported()`
- [ ] macOS: report system-playback capture as unsupported unless a virtual
      route is configured (native process taps / ScreenCaptureKit = separate proposal)
- [ ] Linux: monitor-source classification only via native server metadata in
      an optional adapter — **not** by name suffix
- [ ] Feedback guard: never auto-select FluxVoice's own virtual output (or its
      monitor) as its input
- [ ] Permissions: enumeration never triggers capture; permission failures are
      reported as `PermissionDenied`, not "no devices". Verify macOS microphone
      usage description / consent flow and Windows privacy settings against
      official docs for the supported OS versions
- [ ] Fill per-OS commands into [M1 stage 05's cheatsheet](../Milestone%201/05_virtual_cable_loopback.md)

## Hardware / OS matrix

| Scenario | Acceptance | Win | macOS | Linux |
|---|---|---|---|---|
| List devices without capture | Separate lists; correct default flags; no recording begins | ☐ | ☐ | ☐ |
| USB plug/unplug | Catalog updates within 2 polling intervals; removed explicit selection not replaced | ☐ | ☐ | ☐ |
| Change OS default | Follow-default follows; explicit stays fixed or fails clearly | ☐ | ☐ | ☐ |
| 44.1/48 kHz device, 16 kHz client | Correct speed/pitch; distinct client/backend diagnostics | ☐ | ☐ | ☐ |
| Different capture/playback devices | ≥ 30 min; no sustained latency/queue growth (clock drift) | ☐ | ☐ | ☐ |
| Bluetooth profile change, sleep/resume | Recovery or actionable error; never stale "running" | ☐ | ☐ | ☐ |
| Denied mic / busy device / server restart | Correct category; bounded retries; no unauthorized fallback | ☐ | ☐ | ☐ |
| Virtual route | Second app records processed output | ☐ | ☐ | ☐ |
| Missing virtual output | No fallback to speakers; route stops safely | ☐ | ☐ | ☐ |
| System-playback mode (optional) | Supported path records; unsupported path says so | ☐ | ☐ | ☐ |

## Results *(fill in)*

| OS / version | Backend | Devices tested | Date | Notes |
|---|---|---|---|---|
| — | — | — | — | |

## Implementation notes

- Capturing one render endpoint ≠ capturing every app, protected content or
  per-app audio. Don't promise more.
- Routing helpers (creating sinks, etc.) are separate, opt-in operations: no
  driver installs, default changes, persistent server config or privileged
  commands during enumeration.
- Third-party virtual drivers (VB-Cable, BlackHole) have their own install,
  signing and license terms — the miniaudio license doesn't cover them.
- Run audible tests with feedback-safe routing and levels.

## Files touched

- `src/audio/*` (loopback mode, optional adapters), this doc (results),
  `Documents/Milestone 1/05_virtual_cable_loopback.md` (cheatsheet)

## Definition of done

- Every non-optional matrix row ticked on all three OSes, with evidence recorded.
- Each advertised routing mode is recorded by a second application on its
  target OS; each unsupported mode is reported as such by the app.
