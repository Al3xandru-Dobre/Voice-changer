# Documents — milestone plans

Staged implementation plans for FluxVoice-S2UT. Each milestone directory
contains a `00_master.md` index (goal, scope, acceptance criteria, stage
table) followed by numbered stage files `NN_<stage_name>.md`.

| Directory | Milestone | Status |
|---|---|---|
| [Milestone 1/](Milestone%201/00_master.md) | DSP loopback & audio engine | In progress |
| [Milestone 1.5/](Milestone%201.5/00_master.md) | Cross-platform audio devices (parallel to M2; blocks M4 device UI) | Planned |
| [Milestone 2/](Milestone%202/00_master.md) | Textless VC mode (encoder + vocoder) | Draft |
| [Milestone 3/](Milestone%203/00_master.md) | Wait-k S2UT mode | Draft |
| [Milestone 4/](Milestone%204/00_master.md) | UI & optimization | Draft |

## Research documents

Dated research reports live in [`Research_documents/`](Research_documents/).
They are **preserved snapshots**: once a report's recommendations are scheduled
into milestone stages, the stage files become authoritative and the report
gets a tracking note at the top rather than being rewritten.

| Report | Date | Scheduled in |
|---|---|---|
| [Cross-platform audio I/O](Research_documents/CROSS_PLATFORM_AUDIO_IO_RESEARCH.md) | 2026-10-17 | Milestone 1.5 (stages 01–06); M1 stages 02, 03, 05, 06; M4 device selectors |

The project-wide bibliography (papers, repos, tools) remains in the root
[`RESEARCH_GUIDE.md`](../RESEARCH_GUIDE.md).

## Conventions


- Update the **Status** field inside each file (`☐ not started` →
  `◐ in progress` → `☑ done`) and the master table as work lands.
- Stage files are executed in order; each lists its dependencies.
- Keep each stage's *Definition of done* objective and measurable.
