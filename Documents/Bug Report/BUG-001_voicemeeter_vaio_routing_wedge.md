# BUG-001 — Voicemeeter VAIO routing silently dies mid-session

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-001` |
| **Name** | *Dead pipes* — VAIO input endpoints go silent while config reads correct |
| **Reported** | 2026-10-17 |
| **Status** | ☐ open — workaround exists, root cause unconfirmed |
| **Component** | External dependency: VB-Audio Voicemeeter Banana 2.1.2.2 (`0x2010202`), Windows 11 Pro build 26300, WASAPI |
| **Blocks** | [Milestone 1 / Stage 05](../Milestone%201/05_virtual_cable_loopback.md), [Stage 06](../Milestone%201/06_latency_verification.md) — Voicemeeter pairs as second virtual cable |

## Summary

Audio played into Voicemeeter's virtual playback endpoints (`Voicemeeter
Input`, `Voicemeeter AUX Input`) stops reaching the mixer entirely after two
events in the same session: (a) a third-party audio driver (VB-CABLE) being
installed while the engine was running, and (b) strip→bus routing changes made
in the GUI. All parameter reads via the Remote API report the intended
routing; no levels, mutes, or device assignments are wrong. `Command.Restart`
does not recover the path. A full application restart is expected to restore
it (not yet verifiable — see BUG-002: the elevated process cannot be
terminated or written to from a non-elevated shell).

## Steps to reproduce

1. Launch `voicemeeterpro_x64.exe` elevated.
2. Bind the A1 bus to a hardware output
   (`Bus[0].device.name = "wdm: Speaker (2- Realtek(R) Audio)"`). Engine runs;
   VAIO→B1 loop verified working (measured **189.31 ms** with the
   `fv_probe latency` chirp/cross-correlation probe).
3. Install a different audio driver (VB-CABLE pack 45) while the engine is
   running.
4. In the GUI, change strip routing: `Strip[2]` (Voicemeeter VAIO) → B1 only,
   `Strip[3]` (Voicemeeter AUX) → B2 only, `Strip[4]` (VAIO3) → no buses.
5. Play audio into `Voicemeeter Input` (WASAPI) and record
   `Voicemeeter Out B1`.

## Expected

`Voicemeeter Out B1` captures what was played into `Voicemeeter Input`, and
`Voicemeeter Out B2` captures `Voicemeeter AUX Input`.

## Actual

Complete silence on every capture endpoint. Sweeping all 8 `Voicemeeter Out`
capture devices while playing into `Voicemeeter Input` or
`Voicemeeter AUX Input` produces no signal — including the previously working
VAIO→B1 path.

## Evidence collected

| Check | Result |
|---|---|
| `Strip[2]`: A1=0, B1=1; `Strip[3]`: A1=0, B2=1; `Strip[4]`: all off | correct routing persisted, readable via `VBVMR_GetParameterFloat` |
| All 8 `Bus[i].Mute` / `Strip[i].Mute` / `Strip[i].Solo` | 0 (not muted) |
| `Bus[0].device.name` | bound: `"wdm: Speaker (2- Realtek(R) Audio)"` |
| `Command.Restart` (remote) | returns 0, no recovery |
| `VBVMR_GetLevel`, all channel indices, during playback | all 0.0 |
| WASAPI endpoint enumeration | all `Voicemeeter In/Out` endpoints present and openable at 48 kHz |

## Likely root cause

The audio engine appears to wedge its VAIO input pipes when the Windows
device graph changes mid-run (VB-CABLE driver install), and
`Command.Restart` does not rebuild that plumbing — consistent with
Voicemeeter's documented sensitivity to device changes requiring an
engine/app restart. The GUI routing change is a co-occurrence, not proven to
be causal.

## Workaround

- Do not install/uninstall audio drivers while the Voicemeeter engine is
  running; restart the app afterwards.
- Prefer the standalone VB-CABLE pair (kernel-level loopback, no mixer
  engine) for test routes: `CABLE Input`/`CABLE Output`, measured
  113.67 ms round-trip at 48 kHz including probe buffers.

## Context

Found while building the two-cable latency-measurement route for
[Milestone 1 / Stage 06](../Milestone%201/06_latency_verification.md).
Measurements quoted are real `fv_probe` runs; scripts used for remote
control live in `tools/_dl/vm_route.py` (session scratch, not committed).

## Open questions

- Does a full app restart (close window, relaunch) restore the path? Blocked
  by BUG-002 until the elevated instance is closed.
- Does the wedge also happen without a concurrent driver install (routing
  change alone)?
