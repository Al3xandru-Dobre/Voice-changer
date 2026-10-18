# Stage 05 — Virtual audio cable loopback

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 04](04_overlap_add_dsp.md) · [Stage 06 →](06_latency_verification.md)

**Status:** ◐ in progress — feedback guard and `fv_probe` landed (2026-10-17);
the end-to-end cable check still needs a machine with a virtual cable
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
  - *The Windows dev machine has no virtual cable yet (2026-10-17). Linux null
    sinks were already used by `tools/pw_loopback_check.sh` in Stages 02–03.*
- [x] `--output <index>` selects the cable/sink as playback endpoint (since
      Stage 02; opened by index into a null sink on Linux)
- [x] Feedback guard: fluxvoice refuses to start when its capture device is the
      far end of its own playback device (`src/audio/RouteGuard.hpp`, CTest
      `route_guard`). See notes.
- [x] A second application for every OS: `fv_probe record` writes any capture
      device to a WAV (Windows has no `parecord`); `fv_probe tone` drives a
      cable with a test tone for the objective check
- [ ] Verify end-to-end: speak → fluxvoice queue loopback → cable → record in
      a second app (Audacity/OBS/`pw-record`/`fv_probe record`); playback
      recording matches input
- [ ] Document the exact per-OS commands in this file — written below; the
      Windows and macOS lines are still unverified on hardware

## Implementation notes

- Virtual sinks often run at 44.1/48 kHz while we open 16 kHz — miniaudio
  handles device-side rate conversion; still, log **both** rates since a
  mismatch here is the #1 source of "sounds chipmunk'd" bugs:
  - `ma_device.sampleRate` — the **client** rate we asked for (16 kHz), *not*
    the hardware rate;
  - `ma_device.capture.internalSampleRate` / `playback.internalSampleRate` —
    the backend-side rates (the OS audio server may still convert further).
  *(Cross-platform research §8.)* fluxvoice prints both at startup
  (`StreamInfo`, Stage 02); `fv_probe` prints the same for its own stream.
- **Feedback guard.** `AudioEngine::start()` is now `open()` + `start()`; main
  checks `is_feedback_route(capture_name, playback_name)` on the endpoints
  actually opened, before any audio flows. So an OS-default input that is
  the cable's far end (a common Windows setup: "CABLE Output" as default
  recording device) is refused too, with exit code 1. Recognised pairs:
  - `Monitor of X` ← `X` (PulseAudio / PipeWire sink monitor)
  - `… Output …` ← `… Input …` with the rest of the name equal (VB-Cable,
    CABLE-A/B, Hi-Fi Cable)
  - `BlackHole …` ← the same name (other same-name pairs are allowed: USB
    headsets use one name for both directions)

  It is a name heuristic and cannot see routes it has no name for (e.g. a
  PipeWire virtual source layered on the sink). Backend metadata replaces it
  in [Milestone 1.5 / Stage 06](../Milestone%201.5/06_routing_os_validation.md).
- `fv_probe` (`tools/fv_probe.cpp`) is a separate process with its own
  miniaudio stream — a real second application. Device indices match
  `fluxvoice --list-devices` on the same backend. Run fluxvoice first, then
  fv_probe, so fluxvoice's start-up underrun doesn't land in the recording.
- **Voicemeeter as the second cable pair (Windows).** Banana's virtual
  strips are `Strip[3]` (VAIO, "Voicemeeter Input") and `Strip[4]` (AUX,
  "Voicemeeter AUX Input") — *not* Strip[2]/[3] (see
  [BUG-001](../Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)).
  Never script the Remote API by hand: use `tools/vm_route.py` (applies the
  routing and verifies every write by read-back; `--check` inspects without
  writing; refuses mismatched integrity levels, so the BUG-002 silent-drop
  fails loudly; `--self-test` runs offline) and keep Voicemeeter
  **non-elevated**.
- **Fresh machines:** `tools/setup_test_rig.ps1` is the one-command rig
  bootstrap — toolchain check, build + ctest, VB-CABLE/Voicemeeter present
  or installed (UAC prompts), VAIO routing + end-to-end tone check.
- fv_probe accepts `--input-name`/`--output-name` (unique substring match)
  in addition to indices; scripts should prefer names — indices shift
  whenever audio drivers are installed or removed (BUG-001).
- This stage validates the route on the **dev machine**. The full
  Windows / macOS / Linux matrix, optional WASAPI system-playback capture and
  "missing virtual output must not fall back to speakers" are
  [Milestone 1.5 / Stage 06](../Milestone%201.5/06_routing_os_validation.md).
- Monitor sources can add latency; measure it in Stage 06 rather than tuning
  blind.
- `tools/pw_loopback_check.sh` (added in Stage 02/03) already exercises the
  Linux null-sink → monitor route with `parecord` as the second application,
  using a test tone and automated glitch analysis. It does not replace this
  stage's check with a real microphone and a real receiving app (Audacity, OBS,
  a voice-chat client).
- Keep a `--passthrough` vs queue-mode A/B: if the cable path glitches but
  headphones don't, the sink — not our pipeline — is the bottleneck.

## Per-OS routing cheatsheet

Linux commands are the ones `tools/pw_loopback_check.sh` runs (verified in
Stages 02–03). Windows and macOS lines are **not yet verified on hardware**;
confirm the exact names with `--list-devices`.

```bash
# Linux / PipeWire or PulseAudio
pactl load-module module-null-sink sink_name=fluxvoice_out \
    sink_properties=device.description=FluxVoice-Out     # prints a module id
./build/fluxvoice --list-devices      # playback "FluxVoice-Out"; capture = your mic
./build/fluxvoice --input <mic> --output <FluxVoice-Out>
pw-record --target fluxvoice_out.monitor out.wav         # or pick it in your DAW, or:
./build/fv_probe record --input <Monitor of FluxVoice-Out> --seconds 20 out.wav
pactl unload-module <module id>
# Some apps hide monitor sources — then also create a virtual source
# (pipewire-pulse-module-virtual-source(7)) on top of the sink.
```

```powershell
# Windows / WASAPI + VB-Cable (https://vb-audio.com/Cable/, installer needs admin)
# Fresh machine? tools\setup_test_rig.ps1 checks/installs everything first.
.\build\fluxvoice.exe --list-devices  # playback "CABLE Input (VB-Audio Virtual Cable)"
.\build\fluxvoice.exe --input <mic> --output <CABLE Input>
# Receiver: pick "CABLE Output (VB-Audio Virtual Cable)" as the microphone in
# Audacity / OBS / a voice-chat app, or record it from a second terminal
# (name selection survives driver installs; indices do not — BUG-001):
.\build\fv_probe.exe record --input-name "CABLE Output" --seconds 20 voice.wav

# Second cable pair via Voicemeeter Banana (Stage 06 two-cable method):
#   run Voicemeeter NON-elevated (BUG-002), then:
python tools\vm_route.py          # VAIO ("Voicemeeter Input") -> B1, AUX -> B2; write-by-write verified
python tools\vm_route.py --check  # integrity levels + current routing; writes nothing
#   tone round-trip (silence = dead route; also run by setup_test_rig.ps1):
.\build\fv_probe.exe tone --output-name "Voicemeeter Input" --input-name "Voicemeeter Out B1" --seconds 4 tone.wav

# Optional Windows WASAPI system-playback capture (M1.5 Stage 06):
#   capture a playback endpoint directly via native WASAPI loopback and route
#   processed audio into a different virtual cable playback endpoint:
.\build\fluxvoice.exe --loopback-input <playback index|default> --output <CABLE Input index>
.\build\fv_probe.exe record --loopback --input-name "CABLE Input" --seconds 10 loopback.wav
```

```bash
# macOS / Core Audio + BlackHole 2ch
./build/fluxvoice --input <mic> --output <BlackHole 2ch>
./build/fv_probe record --input <BlackHole 2ch> --seconds 20 voice.wav
```

Objective variant (no microphone, any OS): needs **two** cables — free null
sinks on Linux; on Windows e.g. VB-Cable + VB-Cable A. Cable 1 feeds
fluxvoice, cable 2 receives it:

```bash
fluxvoice --input <cable 1 capture end> --output <cable 2 playback end> --duration 40
fv_probe tone --output <cable 1 playback end> --input <cable 2 capture end> --seconds 30 tone.wav
python3 tools/analyze_loopback.py tone.wav               # needs numpy
```

## Verification (2026-10-17, Windows — partial)

Windows 11 Pro (build 26300), MSVC 19.50 (VS 18), backend **WASAPI**. Devices:
"Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"
and "Speaker (Realtek(R) Audio)" only; **no virtual cable installed**.

| Check | Result |
|---|---|
| `fluxvoice`, `fv_probe` build with `/W4 /WX` | ✓ — first Windows build of `fluxvoice` (MSVC fixes below) |
| `ctest` | 25/25 pass (incl. `route_guard`, `latency_probe`, `analyze_soak_selftest`, `fv_probe` CLI rejects) |
| `fv_probe --list-devices` vs `fluxvoice --list-devices` | same backend, same indices |
| `fv_probe record --input 0 --seconds 3` | 144,000 samples at 48 kHz (backend 48 kHz, period 480 frames); mic noise floor −76 dBFS |
| Feedback guard | unit tests only; every rule is pinned by a test (a broken monitor rule, VB-Audio rule, same-name rule or "any Output" rule each fails the suite) |
| Mic → fluxvoice → cable → second app | **not run** — no virtual cable |

Nothing that plays sound was run: the only output is the laptop speaker and the
only input the microphone beside it (acoustic feedback risk).

MSVC fixes needed for the first Windows build of `fluxvoice` (they were
listed under [M1.5 / Stage 01](../Milestone%201.5/01_ci_build_matrix.md)):
`NOMINMAX` on the `miniaudio_headers` target (miniaudio's `<windows.h>`
defines `min`/`max` macros), and C4324 ("structure was padded") silenced around
`DuplexRouter`, whose queues are cache-line aligned on purpose.

**To finish this stage:** install VB-Cable (or use the Linux machine), run the
Windows/Linux lines above with a real microphone and a real receiving app, and
listen to / analyse the recording.

## Files touched

- `src/main.cpp` (guard, open/start), `src/audio/AudioEngine.hpp`
  (`open()` + `start()`), `src/audio/RouteGuard.hpp` (new),
  `src/audio/DuplexRouter.hpp` (C4324), `tools/fv_probe.cpp` (new),
  `tests/test_route_guard.cpp` (new), `CMakeLists.txt`, this doc
- Rig hardening after
  [BUG-001](../Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)
  (2026-10-17): `tools/vm_route.py` (new — verified-write Voicemeeter
  routing + offline self-test), `tools/setup_test_rig.ps1` (new — rig
  bootstrap), `tools/fv_probe.cpp` (`--input-name`/`--output-name`),
  `CMakeLists.txt` (`vm_route_help`, `vm_route_selftest`,
  `probe_rejects_name_and_index` tests)

## Definition of done

- A third-party app records fluxvoice's output cleanly; documented routing
  commands above work from a fresh shell.

Not met yet — no virtual cable on the Windows dev machine.
