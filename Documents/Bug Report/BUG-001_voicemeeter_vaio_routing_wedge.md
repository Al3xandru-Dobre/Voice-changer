# BUG-001 — Voicemeeter VAIO routing silently dies mid-session

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-001` |
| **Name** | *Dead pipes* — VAIO input endpoints go silent while config reads correct |
| **Reported** | 2026-10-17 |
| **Status** | ◐ diagnosed — root cause found in the test rig (misrouting + stale device indices), not a Voicemeeter engine defect; hardware re-verification pending |
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

## Root cause (diagnosed 2026-10-17)

The engine-wedge hypothesis from the original report is superseded: every
observation is explained by the test rig itself, with no engine defect
required.

**1. Wrong strip-index model.** The repro (and the scratch script
`tools/_dl/vm_route.py`) assumed `Strip[2]` = VAIO, `Strip[3]` = AUX,
`Strip[4]` = VAIO3 — a Potato-style layout. Banana's actual Remote API
layout, confirmed by the saved configs (`tools/_dl/vm_*.xml`:
`LabelStrip1-3` + `LabelVirtualStrip1-2`):

| API strip | Banana strip | Fed by |
|---|---|---|
| `Strip[0..2]` | hardware inputs 1–3 | physical devices (`InputDev 3` was `-`: unbound) |
| `Strip[3]` | **Voicemeeter VAIO** | `Voicemeeter Input` playback endpoint |
| `Strip[4]` | **Voicemeeter AUX** | `Voicemeeter AUX Input` playback endpoint |

So "Strip[2] → B1, Strip[3] → B2" actually configured **unbound hardware
input #3 → B1** and **VAIO → B2**; AUX got no buses. The persisted state
(`vm_routes.xml`) proves it. `Voicemeeter Out B1` was silent *by
configuration* — its only feeder had no device.

**2. "Config reads correct" was circular.** The read-back in the evidence
table was compared against the same wrong mapping used for the writes. The
values persisted exactly; they just never meant what we thought.

**3. Stale device indices after the mid-session driver install.** The two
observations misrouting alone cannot explain (all 8 out-endpoints silent
while feeding `Voicemeeter Input`; all `GetLevel` 0.0) match `fv_probe`
selecting devices by **index** from a pre-install enumeration: installing
VB-CABLE shifts every later index, so playback and the sweep likely touched
the wrong endpoints entirely. This is precisely the "stable device identity /
hot-plug" hazard deferred to [M1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md).

**Why `Command.Restart` didn't recover**: nothing was wedged. Restart
rebuilds the audio engine, not routing parameters — and the wrong routing
was saved state (`vm_routes.xml`), so a full app restart would not have
fixed it either. **BUG-002 interplay**: writes against the elevated instance
silently no-op'd during the session, scrambling state further and blocking
the restart test.

Residual open item: a genuine VAIO-pipe wedge after a device-graph change
remains *possible* but is not needed to explain anything observed. The tone
round-trip in `tools/setup_test_rig.ps1` (fv_probe into `Voicemeeter Input`,
recorded from `Voicemeeter Out B1`, after `vm_route.py` verified the
routing) settles it on the next rig session.

## Workaround

- Do not install/uninstall audio drivers while the Voicemeeter engine is
  running; restart the app afterwards.
- Prefer the standalone VB-CABLE pair (kernel-level loopback, no mixer
  engine) for test routes: `CABLE Input`/`CABLE Output`, measured
  113.67 ms round-trip at 48 kHz including probe buffers.

## Fix landed in the rig (2026-10-17)

- `tools/vm_route.py` (committed) replaces the scratch script: correct
  per-edition strip layout, refuses to write across mismatched integrity
  levels, every write verified by read-back (fails loudly on the BUG-002
  silent-drop), offline `--self-test` (CTest `vm_route_selftest`).
- `fv_probe --input-name/--output-name`: scripts select devices by name
  substring, immune to index shifts. Index selection still works.
- `tools/setup_test_rig.ps1`: one-command rig bootstrap — toolchain check,
  build + ctest, VB-CABLE/Voicemeeter check or install (UAC), verified
  routing and a tone round-trip check that fails on silence. Run it again
  after any driver change.

## Context

Found while building the two-cable latency-measurement route for
[Milestone 1 / Stage 06](../Milestone%201/06_latency_verification.md).
Measurements quoted are real `fv_probe` runs; the scratch script that caused
the misrouting (`tools/_dl/vm_route.py`, git-ignored) is replaced by the
committed, verified-write [`tools/vm_route.py`](../../tools/vm_route.py).

## Open questions

- ~~Does a full app restart restore the path?~~ Moot under the diagnosis:
  the wrong routing was saved state, so a restart would reload it. (If the
  residual wedge hypothesis ever fires, this becomes relevant again.)
- ~~Does the wedge also happen without a concurrent driver install?~~ The
  silence is fully explained by misrouting alone; no wedge required.
