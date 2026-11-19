# FluxVoice-S2UT — Project Rules & Specialist Agents (`AGENTS.md`)

> This file is loaded into every agent session and every subagent — keep it short. Domain rules: [`.claude/rules/`](.claude/rules/) · Condensed invariants: [`MEMORY.md`](MEMORY.md) · Rationale & post-mortems: [`Documents/BEST_PRACTICES.md`](Documents/BEST_PRACTICES.md) · Milestones: [`Documents/README.md`](Documents/README.md) · Bug reports: [`Documents/Bug Report/00_index.md`](Documents/Bug%20Report/00_index.md)

---

## Part 1 — Always-On Project Rules

Rule numbers are cited from code and stage files: append new rules, never renumber.

1. **Load the matching domain rules first — and only those:** Before modifying audio I/O, test-rig scripts, DSP, ML/ONNX or UI code, find the path in the [Part 2 roster](#part-2--specialist-roster) and read its `.claude/rules/<handle>.md` (Claude Code loads it automatically when a matching file is opened). That file names the [`BEST_PRACTICES.md`](Documents/BEST_PRACTICES.md) sections for the domain: open those sections by heading, not the whole 46 KB file. [`MEMORY.md`](MEMORY.md) holds the cross-domain layer: architecture truths (§1), equations (§2), numbered rules (§4), commands (§5).
2. **Authority hierarchy of documentation:**
   - Numbered milestone stage files (`Documents/Milestone */NN_*.md`) are **authoritative**.
   - Research documents (`Documents/Research_documents/*.md`) are preserved dated snapshots.
   - Where [`Documents/Milestone 2/00_master.md`](Documents/Milestone%202/00_master.md) disagrees with early claims in `README.md` (e.g., 50 Hz StreamVC-style causal student vs. 12.5 Hz Mimi default, causal YIN vs. non-causal SwiftF0/RMVPE, OLA default off for causal HiFi-GAN, elevated MMCSS worker priority), follow the Milestone 2 stage files.
3. **Real-time callback sanctity:** Never introduce heap allocations, mutexes, I/O (`std::cout`/`stderr`), blocking calls, or ONNX inference inside `AudioEngine::data_callback` or `DuplexRouter::process()`.
4. **Build & test gate:** Every C++, CMake or tool change must build **all targets** (including `fluxvoice_header_checks`) with warnings-as-errors and pass the hardware-free CTest suite; a targeted build alone is not the final gate.
   - **Linux** — one command per toolchain, each in its own fixed tree `build-<preset>/` ([`CMakePresets.json`](CMakePresets.json)):
     ```bash
     cmake --workflow --preset linux-gcc      # configure + build all + ctest; also: linux-clang
     cmake --workflow --preset linux-gcc-ort  # with ONNX Runtime; also: linux-clang-ort
     ctest --preset linux-gcc-training        # python_training_* suites only (slow, need python_training/.venv)
     ```
     The workflow presets run what the CI C++ legs run. Add the `-ort` presets when the change touches `src/pipeline/**` or `FLUXVOICE_WITH_ORT` code, and the `-training` preset when it touches `python_training/**`, a Python↔C++ twin, a golden or a contract.
   - **Windows / macOS** — a separate build directory per compiler/generator:
     ```powershell
     cmake -S . -B build -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_BUILD_TESTS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON
     cmake --build build --config Release
     ctest --test-dir build -C Release --output-on-failure
     ```
   - Native Windows/MSVC, Linux/GCC, Linux/Clang and macOS/AppleClang CI results are required before claiming cross-platform verification. If a native toolchain is unavailable, report that leg as NOT RUN/pending, not passed; clang-tidy with Windows headers is only a frontend check.
5. **Self-tested rig scripts:** Every Python utility in `tools/` must use the Python standard library only, support `--self-test` with offline fakes, and be registered in `CMakeLists.txt`.
6. **Portable headers & toolchain selection:** First-party headers must be self-contained — own the standard include for every name used and qualify size/fixed-width types (`std::size_t`, `std::[u]intN_t`); the ALL-built `fluxvoice_header_checks` target checks independent header inclusion (one TU per header). Guard OS headers behind `_WIN32`/`__APPLE__`/`__linux__`, distinguish OS macros from compiler/frontend macros (clang-cl defines both `__clang__` and `_MSC_VER`), select MSVC-style vs GNU-style warning flags via `CMAKE_CXX_COMPILER_FRONTEND_VARIANT`, and link platform libraries per-OS in CMake — never `#pragma comment(lib)`. Details: BEST_PRACTICES §5.3–5.4.
7. **Cross-platform-first workflow:** Apply [BEST_PRACTICES section 5.6](Documents/BEST_PRACTICES.md#56-default-development-workflow) to every new or changed C++ header, source, test and CMake target. Own each standard include, use standard-qualified types in new code, give aggregate members explicit defaults only when those defaults are semantically correct, and add regression coverage for compiler failures. Fix first-party warnings at their source; do not weaken the shared warning gate or blanket-suppress diagnostics to make CI green. Record commands, compiler/platform identity and pass/fail/NOT RUN evidence for each affected native leg.

---

## Part 2 — Specialist Roster

One handle per domain. Its rules live in `.claude/rules/<handle>.md` (path-scoped: loaded when a matching file is opened) and its Claude Code subagent in `.claude/agents/<handle>.md`. Adopt the handle's rules whenever you work in its paths, whether directly or as a subagent.

| # | Handle | Use for | Paths | Milestones |
|---|---|---|---|---|
| 1 | `@explorer` | Read-only tracing of data flow, cross-milestone dependencies, code-vs-stage-spec audits, bug symptom investigation | reads everything; edits nothing | All |
| 2 | `@rt-audio-dsp` | `AudioEngine`, `DuplexRouter`, `ChunkAssembler`, `OverlapAddProcessor`, `RouteGuard`, `AudioDeviceRegistry` / `AudioController`, C++ causal YIN + whitening | `src/audio/**`, `src/dsp/**`, `src/util/**`, `src/main.cpp` and their tests | M1, M1.5, M2 (03/08) |
| 3 | `@audio-rig-routing` | Virtual cables (VB-CABLE, Voicemeeter Remote API, PipeWire null sinks, BlackHole), `fv_probe`, loopback and latency measurements | `tools/fv_probe.cpp`, `tools/LatencyProbe.hpp`, `tools/vm_route.py`, `tools/setup_test_rig.ps1`, `tools/pw_*.sh`, `tools/analyze_*.py` | M1 (05–06), M1.5 (06) |
| 4 | `@ml-speech-trainer` | Python 3.12 training & evaluation: causal features, HuBERT teacher, causal student, Unit-HiFi-GAN vocoder, target-voice fine-tune, Wait-*k* S2UT | `python_training/**`, `models/**`, `Documents/Milestone 2/{01,03,04,05,10}_*.md`, `Documents/Milestone 3/**` | M2 (01, 03–06, 10), M3 |
| 5 | `@onnx-ort-runtime` | `ModelContract`, stateful ONNX export, Python↔C++ parity, `OrtModel` / `KVStateCache` / `InferencePipeline` / `WaitKPolicy` / `LatencyGovernor`, execution providers, quantization | `src/pipeline/**`, `python_training/export_stateful_onnx.py`, `third_party/onnxruntime/**`, `Documents/Milestone 2/{02,06,07,08}_*.md` | M2 (02, 06–08), M3, M4 |
| 6 | `@qa-verifier` | C++ unit tests, CTest cases, `--self-test` modes, BEST_PRACTICES audits, soak/latency analysis, bug reports | `tests/**`, `CMakeLists.txt`, `Documents/Bug Report/**`, `Documents/BEST_PRACTICES.md`, `MEMORY.md` | All |
| 7 | `@ui-telemetry` | `VoiceParams` descriptor table, presets, CLI/console voice controls, the `frontend/` desktop UI | `frontend/**`, `src/voice/**`, `presets/**`, `third_party/imgui/**`, `third_party/SDL3/**`, `Documents/Milestone 2/09_*.md`, `Documents/Milestone 4/**` | M2 (09), M4 |

**Models (Claude Code):** `explorer` runs on Haiku, the six coding specialists on Sonnet; the lead plans, keeps the semantic/core logic and reviews every delegated diff. Override the model per call when a task needs more.
**Other harnesses (`run_subagent`):** `@explorer` -> `subagent_explore` (strictly read-only: `read`, `grep`, `find_file_by_name`, `code_search`, `web_search`; a fast model such as SWE-2 High, GPT-6 Luna Max or Deepseek v4.1-flash Max), every other handle -> `subagent_general` with that handle's rules file as its system instructions.

---

## Part 3 — Working Efficiently

**Search.** Use `rg` (or the Grep tool), never `grep -r`/`find` from the repo root: `rg` skips the build trees, `data/` and `.claude/worktrees/`, which otherwise return stale duplicates of every file. Read line ranges, not whole files.

**Build trees.** Use only the preset trees from rule 4. Do not create ad-hoc `build-*` directories: a new tree costs a full rebuild, and a tree whose configure fails keeps serving stale binaries to `ctest`. Inner loop: `cmake --build --preset linux-gcc --target <target> && ctest --preset linux-gcc -R <regex>`; run the full workflow once at the end. Always chain build and test with `&&`.

**Delegation.**
- *Lead:* plan first. Give each coding subagent exclusive file ownership, the exact preset tree it may build in (two agents never build in the same tree at once), its own scratch directory (`/tmp/fv-<handle>/`; agents sharing a scratch path have run each other's mutation scripts), and the acceptance check. Register new test files in `CMakeLists.txt` yourself before delegating. Review every returned diff and spot-check its tests by mutation before reporting them as passing.
- *Subagent:* stay inside the files your brief names; do not edit `CMakeLists.txt`, `AGENTS.md`, `MEMORY.md` or `Documents/BEST_PRACTICES.md` unless the brief assigns them. Final report: files changed, commands run with PASS / FAIL / NOT RUN each, and open questions — no file dumps.

**Recording a lesson (one home each).**
1. Bug: `Documents/Bug Report/BUG-NNN_<slug>.md` + its row in `00_index.md`.
2. Rule with its rationale: the matching `Documents/BEST_PRACTICES.md` section.
3. One imperative line in the owning `.claude/rules/<handle>.md`.
4. `MEMORY.md` only when a cross-domain invariant or architecture truth changes; this file only when an always-on rule changes.

Do not restate the same rule in several files — link to its home.
