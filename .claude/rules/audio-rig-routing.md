---
paths:
  - "tools/fv_probe.cpp"
  - "tools/LatencyProbe.hpp"
  - "tools/vm_route.py"
  - "tools/setup_test_rig.ps1"
  - "tools/pw_*.sh"
  - "tools/analyze_*.py"
---

# Audio test rig, virtual cables & OS routing rules (`@audio-rig-routing`)

Scope: VB-CABLE, Voicemeeter Banana/Potato Remote API, PipeWire null sinks, macOS BlackHole, `fv_probe`, `vm_route.py`, `setup_test_rig.ps1`, Stage 05/06 loopback and latency measurements.
Rationale, by section: [`Documents/BEST_PRACTICES.md`](../../Documents/BEST_PRACTICES.md) §1 (IPC, integrity levels, read-backs), §2.1 (device indices), §3.4 (clock drift), §5.2 (`--self-test`). Rig commands: [`MEMORY.md`](../../MEMORY.md) §5.

1. **`BUG-001` Defenses (Edition Strip Layout & Index Shift):**
   - Never hard-code Voicemeeter strip indices. Compute virtual strip indices from `VBVMR_GetVoicemeeterType`: Standard (`type 1`) -> `Strip[2]`, Banana (`type 2`) -> `Strip[3..4]`, Potato (`type 3`) -> `Strip[5..7]`.
   - Installing or removing audio drivers shifts WASAPI device indices. Always select endpoints by name substring (`--input-name` / `--output-name`) in scripts, and never install drivers while an audio engine is running.
   - Never rely on circular parameter read-backs alone; verify physical/virtual audio continuity with an end-to-end signal test (`fv_probe tone` + peak/RMS check).
2. **`BUG-002` Defenses (Integrity Mismatch & Silent Write Drops):**
   - `VoicemeeterRemote64.dll` silently returns `0` and drops writes when Voicemeeter runs elevated (High integrity, `0x3000`) and the client runs non-elevated (Medium integrity, `0x2000`).
   - Always run `TokenIntegrityLevel` preflight checks before writing, verify every parameter write by polling `IsParametersDirty()` + `GetParameterFloat()`, and never call `VBVMR_RunVoicemeeter` from an elevated process.
   - If Voicemeeter is launched after `VBVMR_Login()`, call `Logout()` + `Login()` until `GetVoicemeeterType()` succeeds.
3. **Two-Cable Latency & Drift Protocol:** Measure baseline `B` on Cable 1 alone, then `T` through `fluxvoice` (Cable 1 -> `fluxvoice` -> Cable 2) using `fv_probe latency`. Measure hardware clock drift via long `fv_probe latency` runs (>= 30 min), never via `out_q_` occupancy.
4. **Rig scripts are self-tested (AGENTS.md rule 5):** standard library only, offline `--self-test` with in-memory fakes that exercise the failure modes (e.g. `FakeRemote(drop=True)`), registered in `CMakeLists.txt`. Never commit scratch scripts under `tools/_dl/`.
