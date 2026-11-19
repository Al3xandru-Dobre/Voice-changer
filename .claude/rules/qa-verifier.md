---
paths:
  - "tests/**"
  - "CMakeLists.txt"
  - "CMakePresets.json"
  - "cmake/**"
  - "Documents/Bug Report/**"
  - "Documents/BEST_PRACTICES.md"
  - "MEMORY.md"
---

# Verification, soak/latency audit & bug-report rules (`@qa-verifier`)

Scope: C++ unit tests (`tests/test_*.cpp`), CTest registration, `--self-test` modes, audits against BEST_PRACTICES, soak/latency log analysis, bug reports.
Rationale, by section: [`Documents/BEST_PRACTICES.md`](../../Documents/BEST_PRACTICES.md) §4.7 (tests that can fail), §5.2 (hardware-free tests), §5.3–5.6 (portability and the default workflow), §6 (pre-flight checklist).

1. **Deterministic Hardware-Free Testing:** Every C++ component (`DuplexRouter`, `ChunkAssembler`, `OverlapAddProcessor`, `RouteGuard`, `LatencyProbe`, `AudioDeviceRegistry`, `WaitKPolicy`, `KVStateCache`, `ModelContract`, causal YIN) and every CLI error path must have deterministic unit tests in `tests/` that run in `ctest` without audio hardware or drivers.
2. **Mutation-Proof Assertions:** Pin each business rule with a test that fails if that specific rule is broken (e.g., breaking window bookkeeping, sub-sample interpolation sign, p95 rank, route-guard pattern, or dropped-write detection in `FakeRemote`).
3. **Soak & Latency Verification:** Audit live runs with `tools/analyze_soak.py` (0 overrun/underrun/drop delta after 10 s warm-up, RSS growth <= 1 MB in M1 / <= 5 MB in M2 over 30 min) and `fv_probe latency`.
4. **Bug Report & Best-Practice Maintenance:** Format new bugs in `Documents/Bug Report/BUG-NNN_<slug>.md` with tag, status (`☐ open` -> `◐ diagnosed` -> `◐ workaround` -> `☑ fixed`), environment, repro steps, expected vs. actual, evidence table, root cause, workaround, and blocked milestone stage; add the row to `00_index.md`. Distill every new root cause **once**, following "Recording a lesson" in `AGENTS.md` Part 3: the full rule + rationale in the matching `BEST_PRACTICES.md` section, and one imperative line in the owning `.claude/rules/<handle>.md`. Add a numbered `MEMORY.md` rule only for a new cross-domain invariant — append it; rule numbers are cited from code and are never renumbered.
5. **Cross-Platform Verification:** Enforce always-on rules 4, 6 and 7 for sources, tests and build files. Check owning headers and independent header compilation, intentional aggregate defaults, dead internal helpers, OS-guarded APIs and per-target platform links. Review raw compiler/test evidence for all four native matrix legs; distinguish frontend-only checks and unavailable/pending legs from a native build plus CTest pass. Do not mark portability verified from MSVC alone.
6. **Stale-binary guard:** chain build and test (`cmake --build … && ctest …`, or one `cmake --workflow --preset …`) so a failed build can never be followed by a green run of old binaries; test only in the preset trees.
