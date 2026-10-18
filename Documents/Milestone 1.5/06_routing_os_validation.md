# Stage 06 — Routing & OS validation

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 05](05_refresh_recovery_controller.md)

**Status:** ◐ in progress — code, tests & Windows 11 WASAPI validation landed (2026-10-18); macOS and Linux hardware matrix runs pending
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
| System-playback capture (optional) | Native `ma_device_type_loopback` (`fluxvoice --loopback-input <index\|default>`, `fv_probe record --loopback`), gated by `ma_context_is_loopback_supported()` | **Not** via miniaudio 0.11.25 — unsupported, or a configured virtual route | Sink monitor source opened as ordinary capture |

## Tasks

- [x] Run the hardware matrix below on Windows 11 WASAPI; record evidence (OS
      version, backend, devices, date) in the results table — separate from CI
      results (macOS & Linux hardware runs pending)
- [x] Optional **Windows "system output" capture mode**: separate operation from
      mic selection (`CaptureMode::SystemPlaybackLoopback`, CLI
      `--loopback-input <index|default>` in `fluxvoice` and
      `record --loopback` in `fv_probe`); `capture.pDeviceID` is a **playback**
      endpoint ID; gated by `ma_context_is_loopback_supported()`
- [x] macOS: report system-playback capture as unsupported unless a virtual
      route is configured (native process taps / ScreenCaptureKit = separate proposal)
- [x] Linux: monitor-source classification only via native server metadata in
      an optional adapter (`EndpointDescriptor::is_monitor` / `monitor_of` in
      `AudioTypes.hpp` + metadata-first `fluxvoice::is_feedback_route` in
      `RouteGuard.hpp`) — **not** by name suffix in descriptor classification
- [x] Feedback guard: never auto-select FluxVoice's own virtual output (or its
      monitor) as its input (`AudioController` checks both pre-open metadata
      and post-open device names, plus default-change reroutes and loopback
      self-capture)
- [x] Permissions: enumeration never triggers capture; permission failures are
      reported as `PermissionDenied`, not "no devices" (`MiniaudioBackend::make_error`
      maps `MA_ACCESS_DENIED` to `AudioErrorCategory::PermissionDenied`)
- [x] Fill per-OS commands into [M1 stage 05's cheatsheet](../Milestone%201/05_virtual_cable_loopback.md)

## Hardware / OS matrix

| Scenario | Acceptance | Win | macOS | Linux |
|---|---|---|---|---|
| List devices without capture | Separate lists; correct default flags; no recording begins | ☑ | ☐ | ☐ |
| USB plug/unplug | Catalog updates within 2 polling intervals; removed explicit selection not replaced | ☑ (fake + policy) | ☐ | ☐ |
| Change OS default | Follow-default follows; explicit stays fixed or fails clearly | ☑ (fake + WASAPI auto-route) | ☐ | ☐ |
| 44.1/48 kHz device, 16 kHz client | Correct speed/pitch; distinct client/backend diagnostics | ☑ | ☐ | ☐ |
| Different capture/playback devices | ≥ 30 min; no sustained latency/queue growth (clock drift) | ☐ (short run verified) | ☐ | ☐ |
| Bluetooth profile change, sleep/resume | Recovery or actionable error; never stale "running" | ☑ (fake + policy) | ☐ | ☐ |
| Denied mic / busy device / server restart | Correct category; bounded retries; no unauthorized fallback | ☑ (fake + policy) | ☐ | ☐ |
| Virtual route | Second app records processed output | ☑ | ☐ | ☐ |
| Missing virtual output | No fallback to speakers; route stops safely | ☑ (fake + policy) | ☐ | ☐ |
| System-playback mode (optional) | Supported path records; unsupported path says so | ☑ | ☐ | ☐ |

## Results

| OS / version | Backend | Devices tested | Date | Notes |
|---|---|---|---|---|
| Windows 11 Pro (build 26300), MSVC 19.50 | WASAPI (`native loopback: supported`) | `CABLE Input` / `CABLE Output` (VB-Audio Virtual Cable), `Voicemeeter Input` / `Voicemeeter Out B1` (Voicemeeter 1.1.2.2, Medium integrity), `Microphone Array (Intel Smart Sound)`, `Headphones (LinkBuds S)` | 2026-10-18 | Virtual route (`CABLE Output` → `fluxvoice` 16 kHz → `Voicemeeter Input` recorded on `Voicemeeter Out B1` @ 48 kHz): `analyze_loopback.py` **PASS** (`440.0 Hz`, `-12.0 dBFS`, `0` dropouts, `0` clicks). Native WASAPI loopback (`fv_probe record --loopback --input-name "CABLE Input"`): **PASS** (`440.0 Hz`, `-12.0 dBFS`, `0` dropouts, `0` clicks). `fluxvoice --loopback-input 11 --output 8`: **PASS** (`0` steady-state overruns/underruns/drops, `16,000 Hz` callback rate). Live feedback guard rejected both `CABLE Output <- CABLE Input` and loopback `CABLE Input <- CABLE Input` with exit code `1`. |

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
