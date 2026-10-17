# BUG-002 — Voicemeeter Remote API silently drops writes from non-elevated clients

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-002` |
| **Name** | *Read-only handshake* — elevated Voicemeeter accepts remote writes, returns success, applies nothing |
| **Reported** | 2026-10-17 |
| **Status** | ◐ workaround — `tools/vm_route.py` refuses an elevation mismatch and verifies every write; mechanism probe still to run on the rig ([below](#confirmation-still-to-run)) |
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

Any parameter works: the drop doesn't depend on which strip is written. (The
strip indices used in this session were off by one for Banana; see
[BUG-001](BUG-001_voicemeeter_vaio_routing_wedge.md).)

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

## Root cause (diagnosed 2026-10-17)

The trigger is an **integrity-level mismatch**: Voicemeeter runs at High
integrity (elevated), the client at Medium. The defect is that
`VoicemeeterRemote64.dll` reports success for a write that never reached
Voicemeeter.

- **Not UIPI.** UIPI is Windows' block on window messages sent to a more
  privileged app, and the Remote API sends none. Scanning the DLL's
  imports and strings shows it talks to Voicemeeter only through **named file
  mappings and named events**: `OpenFileMappingA`, `CreateFileMappingA`,
  `MapViewOfFile`, `CreateEventA`, with objects named `VBVMRClient{90CFE18D-…}`
  and `VBVMCClientV2{…}`. There is no `PostMessage`/`SendMessage`.
- **No explicit permissions.** The DLL imports no security-descriptor APIs,
  so these objects get the creating process's default permissions (DACL).
  Created by an elevated Voicemeeter, they let a same-user, non-elevated
  process **read** the parameter area but not **write** to the request
  channel. That explains the read/write split in the evidence table.
- **Silent success.** The DLL doesn't propagate the failed open or write, so
  every Set call still returns 0.
- **`taskkill` is not part of the bug.** Windows' integrity policy stops a
  Medium process from opening a High one for termination. That is expected OS
  behaviour.
- **How the machine got there.** Repro step 1 itself started the elevated
  instance. On 2026-10-17 this machine had no RUNASADMIN compatibility flag,
  Run key or scheduled task for Voicemeeter, and UAC was on. The other way in
  is an elevated script calling `VBVMR_RunVoicemeeter`: the mixer inherits
  the elevation and locks out every later non-elevated client.

### Confirmation still to run

Two checks need an elevated Voicemeeter, so a UAC prompt on the rig:

| Check | Expected | Result |
|---|---|---|
| Non-elevated `OpenFileMappingA(FILE_MAP_READ)` on each `VBVMRClient{…}` / `VBVMCClientV2{…}` name from the DLL | handle | — |
| Same with `FILE_MAP_WRITE`, and `OpenEventA(EVENT_MODIFY_STATE)` | `ERROR_ACCESS_DENIED` (5) | — |
| Routing writes from an **elevated** client | applied, so the cause is the mismatch and not a write lock on elevated instances | — |

If the probe shows something else, only the mechanism bullets above change.
The tooling fix below doesn't depend on the exact mechanism.

## Fix on our side (2026-10-17)

We can't change the vendor DLL. Instead, the rig tooling never relies on a
write's return code. `tools/vm_route.py` replaces the scratch script
`tools/_dl/vm_route.py`; usage is in its docstring.

- **Preflight.** Before any write it compares Voicemeeter's process integrity
  level with its own (Toolhelp + `TokenIntegrityLevel`). If they differ, or
  Voicemeeter's token can't be read (which from a Medium process means it runs
  elevated), it exits 2 and says how to fix it.
- **Read-back.** Every write is read back (`IsParametersDirty` +
  `GetParameterFloat`, ≤ 2 s). The first write that doesn't land exits 1
  naming BUG-002. So even with the preflight skipped, the silent drop is now
  loud.
- **No elevated launch.** It calls `VBVMR_RunVoicemeeter` only from a
  non-elevated process.
- **Self-test.** `--self-test` covers this with a fake Remote API that drops
  writes but returns 0, plus fake integrity levels.

Verified live on 2026-10-18 with Voicemeeter **non-elevated**. The elevated
side is the [confirmation still to run](#confirmation-still-to-run).

| Run | Result |
|---|---|
| `ctest` | 28/28 pass, incl. `vm_route_selftest`, `vm_route_help` |
| `--check` (Voicemeeter not running) | exit 2, "not running"; reports this script at Medium |
| Launch path (Voicemeeter not running) | `VBVMR_RunVoicemeeter(5)` started `voicemeeterpro_x64.exe` at Medium |
| `--check` with saved state loaded | showed `Strip[3]` → B2, `Strip[4]` → none (the scratch script's off-by-one, [BUG-001](BUG-001_voicemeeter_vaio_routing_wedge.md)), exit 0 |
| Route | 3 writes, each read back; `Strip[3]` → B1, `Strip[4]` → B2; exit 0 |
| Route again | no writes ("already correct"), exit 0; routing also survives a Voicemeeter restart |

One API quirk found on the way: a session that logged in **before**
Voicemeeter started never connects to it, so after launching, the script logs
in again.

## Workaround

- Start Voicemeeter normally, **never "Run as administrator"**. If an elevated
  instance is running, exit it from its own window or tray icon (a
  non-elevated `taskkill` is denied) or from an elevated shell, then start it
  again normally.
- Configure it with `tools/vm_route.py` from a non-elevated shell. It refuses
  the mismatch instead of silently doing nothing.
- Otherwise drive the GUI directly; API reads still work for verification.

## Upstream report (draft, not sent)

> **VoicemeeterRemote64.dll: Set calls return 0 but are dropped when
> Voicemeeter runs elevated and the client does not.**
> Voicemeeter Banana 2.1.2.2, Windows 11 Pro 26300. Start Voicemeeter with
> "Run as administrator". From a non-elevated process, `VBVMR_Login()` returns
> 0 and every `VBVMR_GetParameterFloat` works, but `VBVMR_SetParameterFloat`,
> `VBVMR_SetParameterStringA` (incl. `Command.Load`) and `Command.Restart`
> return 0 and change nothing. With both processes at the same elevation, the
> same calls apply.
> Expected: a negative return code when the request can't reach Voicemeeter
> (as `-2` "no server" does today), or a documented requirement that the
> client run at Voicemeeter's elevation.

## Context

Hit while scripting the Voicemeeter strip→bus routing needed to expose two
virtual-cable pairs (`Voicemeeter Input`→`Out B1`, `AUX Input`→`Out B2`) for
the Stage 06 two-cable latency method. The session's scratch script was
`tools/_dl/vm_route.py`. It ignored return codes and never compared its
read-back, so the drop surfaced only by inspection. The committed
`tools/vm_route.py` replaces it.
