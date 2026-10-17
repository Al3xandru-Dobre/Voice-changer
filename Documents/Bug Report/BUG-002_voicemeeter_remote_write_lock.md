# BUG-002 — Voicemeeter Remote API silently drops writes from non-elevated clients

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-002` |
| **Name** | *Read-only handshake* — elevated Voicemeeter accepts remote writes, returns success, applies nothing |
| **Reported** | 2026-10-17 |
| **Status** | ☐ open — behavior confirmed; needs an elevated client to rule out intended design |
| **Component** | External dependency: `VoicemeeterRemote64.dll` (Voicemeeter Banana 2.1.2.2), Windows 11 Pro build 26300 |
| **Blocks** | Scripted test-rig setup for [Milestone 1 / Stage 05](../Milestone%201/05_virtual_cable_loopback.md) & [Stage 06](../Milestone%201/06_latency_verification.md) |

## Summary

While `voicemeeterpro_x64.exe` runs **elevated**, a non-elevated
`VoicemeeterRemote64.dll` client can `VBVMR_Login()` and **read** every
parameter correctly, but every **write** —
`VBVMR_SetParameterFloat`, `VBVMR_SetParameterStringA`, and
`Command.Load` — returns `0` (success) and is never applied. Combined with
the OS-level fact that a non-elevated `taskkill` cannot terminate the
elevated process, a machine in this state has *no* programmatic path to
reconfigure or restart Voicemeeter.

## Steps to reproduce

1. Launch `voicemeeterpro_x64.exe` elevated (`Start-Process -Verb RunAs`).
2. From a non-elevated process, `VBVMR_Login()` (returns 0).
3. `VBVMR_SetParameterFloat("Strip[3].B2", 1.0)` → returns 0.
4. Poll `VBVMR_IsParametersDirty()` until 0, then
   `VBVMR_GetParameterFloat("Strip[3].B2")`.

## Expected

`Strip[3].B2` reads back `1.0` (or the write call returns an error code).

## Actual

`Strip[3].B2` still reads its old value. Reproduced for strip routing
(`Strip[i].A1/B1/B2`), `Bus[0].device.name` (string assign), and
`Command.Load` (full settings file) — all return success, none apply.

## Evidence collected

| Call | Elevated target | Non-elevated target |
|---|---|---|
| `VBVMR_Login` | 0 | 0 |
| `VBVMR_GetParameterFloat` | works | works |
| `VBVMR_SetParameterFloat` (strip sends) | returns 0, **no effect** | returns 0, applies |
| `VBVMR_SetParameterFloat("Command.Restart")` | returns 0, no observable effect | returns 0, applies |
| `VBVMR_SetParameterStringA("Command.Save", …)` | — (untested) | returns 0, writes file |
| `VBVMR_SetParameterStringA("Command.Load", …)` | returns 0, **no effect** | returns 0, applies |

`taskkill /PID <elevated voicemeeter> /F` from the same shell:
`ERROR: Access is denied.`

## Likely root cause

Windows integrity-level isolation on the Remote API's shared-memory/IPC
channel: a lower-integrity client gets a read-only view. If that is the
intended security design, the write calls should return an error (`-1` or a
dedicated code) instead of `0` — the silent drop is the bug.

## Workaround

- Run Voicemeeter **non-elevated** whenever the Remote API must configure
  it (routing, device assignment, `Command.Load`/`Save` all verified working
  from an equal-integrity client).
- Otherwise drive the GUI directly; API reads still work for verification.

## Context

Hit while scripting the Voicemeeter strip→bus routing needed to expose two
virtual-cable pairs (`Voicemeeter Input`→`Out B1`, `AUX Input`→`Out B2`) for
the Stage 06 two-cable latency method. Session scripts:
`tools/_dl/vm_route.py` (scratch, not committed).
