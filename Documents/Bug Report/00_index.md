# Bug reports — index

> Parent: [Documents index](../README.md)

Bugs in FluxVoice itself, its test rig, or its external dependencies —
one file per bug, tagged `BUG-NNN` with a short name. External-dependency
bugs are tracked here because they constrain our milestone work; file them
upstream separately when reproducible.

| Tag | Name | File | Component | Status | First seen |
|---|---|---|---|---|---|
| BUG-001 | Dead pipes | [BUG-001_voicemeeter_vaio_routing_wedge.md](BUG-001_voicemeeter_vaio_routing_wedge.md) | Voicemeeter Banana 2.1.2.2 (external) | ◐ diagnosed | 2026-10-17 |
| BUG-002 | Read-only handshake | [BUG-002_voicemeeter_remote_write_lock.md](BUG-002_voicemeeter_remote_write_lock.md) | VoicemeeterRemote64.dll (external) | ◐ workaround | 2026-10-17 |

## Conventions

- **Tag:** `BUG-NNN`, assigned in order; never reused.
- **Name:** two-or-three-word handle for talking about the bug.
- **File:** `BUG-NNN_<snake_case_slug>.md` in this directory.
- **Status:** `☐ open` → `◐ diagnosed` → `◐ workaround` → `☑ fixed` /
  `☑ wontfix (external)` → closed.
- Every bug file carries: environment, repro steps, expected vs actual,
  evidence, workaround, and which milestone stage it blocks.
