# Stage 02 — Duplex audio I/O (miniaudio)

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 01](01_build_and_dependencies.md) · [Stage 03 →](03_spsc_buffering_worker.md)

**Status:** ☑ done (2026-10-17) — objective verification; a headphone listening check is optional
**Depends on:** Stage 01

## Objective

Enumerate capture/playback endpoints and run a full-duplex stream where the
callback `memcpy`s input → output. Success = hearing your own mic through
headphones with no framework in between.

## Context & references

- RESEARCH_GUIDE §4: *"Read the Device Enumeration and Full-Duplex Callback
  examples in the single `miniaudio.h` header."*
- `src/audio/AudioEngine.hpp` — skeleton already implements the items below;
  this stage is about **verifying** them on real hardware.
- miniaudio API used: `ma_context_init`, `ma_context_get_devices`,
  `ma_device_config_init(ma_device_type_duplex)`, `ma_device_init`,
  `ma_device_start`, `ma_device_uninit`.

## Tasks

- [x] `AudioEngine` class: RAII context/device lifetime, non-copyable
- [x] Device enumeration → `AudioDeviceInfo { id, name, is_default }`
- [x] Duplex config: `ma_device_type_duplex`, `ma_format_f32`, 1 channel,
      16 kHz, `periodSizeInMilliseconds = 20`
- [x] `--passthrough` path: callback `memcpy` (or zero-fill when no input)
- [x] CLI: `--list-devices`, `--input`, `--output`, `--sample-rate`
- [x] CLI: `--duration <s>`; clean stop on Enter, `Ctrl+C` (SIGINT) and SIGTERM
      (previously `Ctrl+C` killed the process without closing the device)
- [x] `--list-devices` prints the selected backend; `start()` reports what was
      actually opened (`StreamInfo`: device names, client vs. backend rate,
      backend period sizes)
- [x] Real-time callback body moved into `DuplexRouter` (no miniaudio
      dependency) so it is unit-tested without a device — see Stage 03
- [x] Hardware check: enumerate on the dev machine; confirm default + named
      devices open
- [x] Passthrough test — **objective** instead of audible: a 440 Hz tone is
      routed through fluxvoice between two PipeWire null sinks, recorded, and
      analysed for dropouts / clicks / pitch (`tools/pw_loopback_check.sh`)
- [ ] *(Optional, human)* listen through headphones at feedback-safe levels:
      `./build/fluxvoice --passthrough`

## Implementation notes

- Keep capture and playback formats identical so the callback is a pure
  `memcpy` — format conversion inside the callback is an M4 concern.
- `periodSizeInMilliseconds = 20` *requests* 320 samples/callback @ 16 kHz,
  but miniaudio documents period sizes as **hints** — the callback must handle
  whatever `frame_count` it receives. Never count callbacks to form chunks;
  the worker assembles exact 1,280-sample (80 ms) chunks from accumulated
  samples (Stage 03). *(Cross-platform research §8.)*
- Pass `nullptr` device ids for OS-default endpoints; only copy `ma_device_id`
  **values**, never retain `ma_device_info` pointers.
- Index-based `--input`/`--output` selection is a **smoke-test convenience**:
  indices shift when devices are plugged in. Stable selection, typed errors and
  null-backend detection are
  [Milestone 1.5](../Milestone%201.5/00_master.md) work — don't extend the
  index scheme here.
- `pUserData` carries `this` into the static `data_callback` — set it in the
  config, not after init.

## Verification (2026-10-17)

Dev machine: Ubuntu 24.04, kernel 6.8, PipeWire via `pipewire-pulse`
(miniaudio backend: **PulseAudio**), Intel SOF HDA (`sof-hda-dsp`).

| Check | Result |
|---|---|
| `--list-devices` | Backend printed; 3 capture (incl. speaker monitor), 1 playback; default flags correct |
| Named devices open | Null-sink endpoints opened by index (capture "Monitor of …", playback sink) |
| Default playback opens | `sof-hda-dsp Speaker + Headphones` (silent input — nothing audible) |
| Default capture opens | `sof-hda-dsp Digital Microphone` (routed to a null sink, not recorded) |
| Passthrough, 20 s tone loopback | 0 dropouts, 0 clicks, 440.00 Hz, −12.0 dBFS, counters 0/0 |
| Stop via Enter / SIGINT / SIGTERM | exit 0, stats printed, nothing on stderr |

Observed backend periods: capture **320** frames, playback **240** frames for a
requested 20 ms (= 320) — direct evidence that period sizes are hints.

Re-run: `tools/pw_loopback_check.sh --mode passthrough` (Linux, PipeWire or
PulseAudio; creates and removes its own null sinks; restores your defaults).

## Files touched

- `src/audio/AudioEngine.hpp`, `src/audio/DuplexRouter.hpp` (new), `src/main.cpp`,
  `tools/pw_loopback_check.sh`, `tools/analyze_loopback.py` (new)

## Definition of done

- `--list-devices` prints capture + playback tables with default flags.
- `--passthrough` produces clear looped audio; `Ctrl+C`/`Enter` stops cleanly
  with no miniaudio errors.

Both met — see Verification.
