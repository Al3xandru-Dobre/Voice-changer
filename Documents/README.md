# Documents — milestone plans

Staged implementation plans for FluxVoice-S2UT. Each milestone directory
contains a `00_master.md` index (goal, scope, acceptance criteria, stage
table) followed by numbered stage files `NN_<stage_name>.md`.

| Directory | Milestone | Status |
|---|---|---|
| [Milestone 1/](Milestone%201/00_master.md) | DSP loopback & audio engine | In progress |
| [Milestone 1.5/](Milestone%201.5/00_master.md) | Cross-platform audio devices (parallel to M2; blocks M4 device UI) | In progress (stages 01–02) |
| [Milestone 2/](Milestone%202/00_master.md) | Textless VC mode (encoder + vocoder, voice controls) | Planned (stage files written) |
| [Milestone 3/](Milestone%203/00_master.md) | Wait-k S2UT mode (bidirectional `En↔{Es,Fr,De}`, shared latents) | Planned (stage files written) |
| [Milestone 4/](Milestone%204/00_master.md) | UI & optimization | Draft |

## Research documents

Dated research reports live in [`Research_documents/`](Research_documents/).
They are **preserved snapshots**: once a report's recommendations are scheduled
into milestone stages, the stage files become authoritative and the report
gets a tracking note at the top rather than being rewritten.

| Report | Date | Scheduled in |
|---|---|---|
| [Cross-platform audio I/O](Research_documents/CROSS_PLATFORM_AUDIO_IO_RESEARCH.md) | 2026-10-17 | Milestone 1.5 (stages 01–06); M1 stages 02, 03, 05, 06; M4 device selectors |
| [Textless VC (Mode A)](Research_documents/M2_TEXTLESS_VC_RESEARCH.md) | 2026-10-18 | Milestone 2 (stages 01–11); notes in M1.5 stage 05, M3 and M4 masters |
| [Simultaneous S2UT (Mode B)](Research_documents/M3_WAITK_S2UT_RESEARCH.md) | 2026-10-18 | Milestone 3 (stages 01–11); handoff notes in M4 master |

The project-wide bibliography (papers, repos, tools) remains in the root
[`RESEARCH_GUIDE.md`](../RESEARCH_GUIDE.md).

## Bug reports & best practices

One file per bug (tagged `BUG-NNN`, with a short name) in
[`Bug Report/`](Bug%20Report/00_index.md) — covering both FluxVoice bugs and
external-dependency bugs that block milestone work.

Engineering rules and mistake-prevention patterns distilled from these bug
reports and milestone findings live in [`BEST_PRACTICES.md`](BEST_PRACTICES.md)
(with a condensed quick-reference layer in the root [`MEMORY.md`](../MEMORY.md)
and specialist agent workflows in [`AGENTS.md`](../AGENTS.md)).

## Default development workflow

All future C++ changes follow the [cross-platform-first workflow](BEST_PRACTICES.md#56-default-development-workflow): explicit owning includes, intentional aggregate defaults, guarded native APIs, upstream warning fixes, regression coverage and an all-target warnings-as-errors build plus hardware-free CTest. [AGENTS.md](../AGENTS.md) makes this an always-on rule; [MEMORY.md](../MEMORY.md) keeps the condensed invariants.

Before claiming cross-platform verification, retain native Windows/MSVC, Linux/GCC, Linux/Clang and macOS/AppleClang build/test evidence. Report unavailable toolchains as NOT RUN/pending; local MSVC or frontend-only parsing does not substitute for the native matrix. Live audio routing still requires the separate manual hardware checks in the milestone stages.

## Conventions


- Update the **Status** field inside each file (`☐ not started` →
  `◐ in progress` → `☑ done`) and the master table as work lands.
- Stage files are executed in order; each lists its dependencies.
- Keep each stage's *Definition of done* objective and measurable.
