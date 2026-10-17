# Stage 02 — Duplex audio I/O (miniaudio)

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 01](01_build_and_dependencies.md) · [Stage 03 →](03_spsc_buffering_worker.md)

**Status:** ◐ skeleton landed
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
- [ ] Hardware check: enumerate on the dev machine; confirm default + named
      devices open
- [ ] Audible passthrough test through headphones (feedback-safe levels!)

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

## Files touched

- `src/audio/AudioEngine.hpp`, `src/main.cpp`

## Definition of done

- `--list-devices` prints capture + playback tables with default flags.
- `--passthrough` produces clear looped audio; `Ctrl+C`/`Enter` stops cleanly
  with no miniaudio errors.
