# Documents — milestone plans

Staged implementation plans for FluxVoice-S2UT. Each milestone directory
contains a `00_master.md` index (goal, scope, acceptance criteria, stage
table) followed by numbered stage files `NN_<stage_name>.md`.

| Directory | Milestone | Status |
|---|---|---|
| [Milestone 1/](Milestone%201/00_master.md) | DSP loopback & audio engine | In progress |
| [Milestone 2/](Milestone%202/00_master.md) | Textless VC mode (encoder + vocoder) | Draft |
| [Milestone 3/](Milestone%203/00_master.md) | Wait-k S2UT mode | Draft |
| [Milestone 4/](Milestone%204/00_master.md) | UI & optimization | Draft |

Conventions:

- Update the **Status** field inside each file (`☐ not started` →
  `◐ in progress` → `☑ done`) and the master table as work lands.
- Stage files are executed in order; each lists its dependencies.
- Keep each stage's *Definition of done* objective and measurable.
