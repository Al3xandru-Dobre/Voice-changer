# Stage 05 — Virtual audio cable loopback

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 04](04_overlap_add_dsp.md) · [Stage 06 →](06_latency_verification.md)

**Status:** ☐ not started
**Depends on:** Stages 02–03 (duplex + queue paths working)

## Objective

Prove the app's output can act as a *virtual microphone* for other software:
physical mic → fluxvoice → virtual audio cable → any recorder/voice app.
This is the real product routing — everything before was self-test.

## Context & references

- README §6 prerequisites: VB-Cable (Windows), BlackHole (macOS), PipeWire
  virtual sink (Linux).
- On Linux/PipeWire the "cable" is a virtual sink + its monitor source; the
  recorder selects the monitor as its mic.

## Tasks

- [ ] Install/create the virtual cable for your OS:
  - Linux (PipeWire): `pw-cli`/`pactl load-module module-null-sink` or a
    persistent virtual sink named e.g. `FluxVoice-Out`
  - Windows: VB-Cable driver; macOS: BlackHole 2ch
- [ ] `--output <index>` selects the cable/sink as playback endpoint
- [ ] Verify end-to-end: speak → fluxvoice queue loopback → cable → record in
      a second app (Audacity/OBS/`pw-record`); playback recording matches input
- [ ] Document the exact per-OS commands in this file (fill in below once done)

## Implementation notes

- Virtual sinks often run at 44.1/48 kHz while we open 16 kHz — miniaudio
  handles device-side rate conversion; still, log the negotiated rate
  (`ma_device.sampleRate`) since mismatch here is the #1 source of
  "sounds chipmunk'd" bugs.
- Monitor sources can add latency; measure it in Stage 06 rather than tuning
  blind.
- Keep a `--passthrough` vs queue-mode A/B: if the cable path glitches but
  headphones don't, the sink — not our pipeline — is the bottleneck.

## Per-OS routing cheatsheet *(fill in during implementation)*

```bash
# Linux / PipeWire (example)
pactl load-module module-null-sink sink_name=fluxvoice_out \
    sink_properties=device.description=FluxVoice-Out
./build/fluxvoice --output <index-of-fluxvoice_out>
pw-record --target fluxvoice_out.monitor out.wav   # or pick it in your DAW
```

## Files touched

- `src/main.cpp` (CLI), this doc (cheatsheet), possibly `tools/`

## Definition of done

- A third-party app records fluxvoice's output cleanly; documented routing
  commands above work from a fresh shell.
