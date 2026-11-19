# BUG-021 — Voice Studio saves a draft on every keystroke into an unvalidated path

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-021` |
| **Name** | *Prefix folders* — the draft directory is the half-typed voice name |
| **Reported** | 2026-11-14 |
| **Severity** | High (silent loss of a recording project's draft; writes outside the Studio directory) |
| **Status** | ◐ fixed in tree 2026-11-15 — native legs pending, see [Resolution](#resolution); `project_dir` residual fixed 2026-11-18, see [Fix (2026-11-18)](#fix-2026-11-18) |
| **Component** | `frontend/imgui/ViewStudio.cpp`, `frontend/core/AppModel.cpp` (`studio_save_draft`), `frontend/core/VoiceStudio.cpp` (`project_dir`, `save_draft`) |
| **Blocks** | [Milestone 4 / Stage 03](../Milestone%204/03_voice_studio_translate_catalogue_panels.md), [Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md); [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) uses the Studio's output |

## Summary

Three facts combine:

1. The "Voice name" field saves the draft on every edit
   (`ViewStudio.cpp:116`: `if (g_name.draw(…)) app.studio_save_draft();`).
2. `studio_save_draft()` only checks that the name is not empty
   (`AppModel.cpp:629-634`). It does not call `name_problem()`.
3. The draft directory is `studio_root / voice_name`
   (`VoiceStudio.cpp:418-420`), created with `create_directories`
   (`VoiceStudio.cpp:639-645`).

## Failure scenarios (by reading; not executed)

- **Prefix directories.** Typing `alex` creates `studio/a/`, `studio/al/`,
  `studio/ale/` and `studio/alex/`, each with a `draft.json`. At the next
  launch the draft is restored only when exactly one draft directory exists
  (`AppModel.cpp:123-132`), so nothing is restored. The README promises that
  "a multi-session recording project survives restarts".
- **Overwrite.** With project `alex` on disk (takes list, checklist), start a
  new session and type `alex` into the empty wizard. The keystroke that
  completes the name writes the empty draft over `studio/alex/draft.json`.
- **Path escape.** A name such as `..\..\x` or an absolute path makes
  `studio_root / voice_name` leave the Studio directory.
  `create_directories` and the write then happen there. The UI shows a
  warning for an invalid name but saves anyway.

## Expected

A draft is written only for a name that passes the `voices.json` name rule
(`[a-z0-9_-]+`), into a directory under the Studio root, and never over an
existing project without loading it first.

## Root cause

Persistence was attached to the edit event of the one field that also
defines the storage location, and the validation that exists
(`name_problem`) is used for display only.

## Suggested fix

- `studio_save_draft()` returns early unless `name_problem()` is empty.
  `project_dir()` asserts the same rule, so no caller can build a path from an
  unvalidated name.
- Save the name-step draft when the field loses focus or the user presses
  Next, not per keystroke.
- When the typed name matches an existing project, offer "Open existing
  project" and load it instead of overwriting.
- Startup: restore the most recently modified draft, or list the projects,
  instead of "exactly one".

## Regression tests (`tests/test_frontend_model.cpp`)

- Simulated edits `a`, `al`, `ale`, `alex` produce one directory.
- `voice_name = "../x"` and an absolute path: `studio_save_draft()` returns
  false and nothing is created outside the root.
- An existing project is not modified by typing its name into an empty
  wizard.
- Two projects on disk: startup restores the newer one.

## Workaround

Paste the complete name in one action, and keep a copy of
`studio/<voice>/draft.json`.

## Resolution

**2026-11-15: fix in tree, native legs pending.**

- **Name rule.** `is_safe_project_name()` (`VoiceStudio`) applies the
  `voices.json` rule and also rejects the Windows device names (`con`, `prn`,
  `aux`, `nul`, `com0-9`, `lpt0-9`).
  - `project_dir()` never builds a path from an unsafe name; it falls back to
    `untitled`.
  - `save_draft()`, `write_job()` and `load_draft()` refuse an unsafe name.
  - `name_problem()` and `can_advance()` report the reserved-name case.
- **Save gate.** `AppModel::studio_save_draft()` returns early unless
  `name_problem()` is empty. Only the format rule is checked here, not the
  installed-voice collision, so the Install step can still save. The model
  tracks the directory that owns the draft in `studio_project_`.
- **Rename and prefix typing.** After the draft is written under the new
  name, the old directory is removed if it held only `draft.json`. Typing
  `a`, `al`, `ale`, `alex` leaves one directory. A directory with job or
  status files is left intact.
- **Overwrite.** When the typed name belongs to another existing project,
  `studio_name_conflict()` is set and nothing is written. The same applies to
  `studio_write_job()`, which now saves through this check first.
  `studio_open_project()` loads that project instead.
- **Startup.** The most recently modified `draft.json` is reopened (ties
  broken by name), instead of requiring exactly one draft.
- **`ViewStudio`.** The name field saves when focus leaves it
  (`IsItemDeactivatedAfterEdit`), not on every keystroke. A name conflict
  shows a notice with an "Open existing project" button and blocks Next.

Regression test `test_studio_drafts`:
- Typing the prefixes leaves one directory.
- `../x`, `..\..\x`, an absolute path, `con`, `Alex` and `""` neither save nor
  write a job, and nothing is created outside the root.
- Raw `save_draft("../x")` fails, and `project_dir` falls back to `untitled`.
- With two projects on disk, startup restores the newer one (`zed`).
- Typing `alex` leaves `alex/draft.json` byte-identical and writes no job.
- "Open" loads `alex`.
- Renaming `alex` after its job is written keeps `alex/` and its job files.

| Leg | Evidence |
|---|---|
| Windows / zig 0.15.2 (Clang 20), strict + `-Werror`, `FLUXVOICE_BUILD_GUI=OFF` | Build pass. CTest 82/82 incl. `frontend_model`. |
| `frontend/imgui/ViewStudio.cpp` | **NOT COMPILED** locally (SDL3 / Dear ImGui not available offline). Needs a GUI CI leg. |
| MSVC, Linux GCC/Clang, macOS | NOT RUN |

Not done: the Studio still has no project list and no "New project" command.
Only the newest draft is reopened. To reach another project, type its name and
choose "Open existing project".

## Fix (2026-11-18)

Residual: `VoiceStudioModel::project_dir()` fell back to `untitled` for an unsafe
name, so a caller that skipped the name check could still write into an
`untitled` project. It now returns an empty path for an unsafe name.

Regression coverage: `test_project_dir` in `frontend_studio` (`tests/test_frontend_studio.cpp`) and
`frontend_model` (`test_studio_drafts`; the Resolution section above still says
`project_dir` falls back to `untitled`, which this fix replaces). Related 2026-11-18 Studio changes in the
same area: take analysis runs off the UI thread and the draft is saved per
batch (drained in `AppModel::tick`).

Evidence: Linux GCC 13.3.0 and Clang 18.1.3, strict + warnings-as-errors,
`ctest -E python_training` 105/105 PASS (2026-11-18, local). Windows MSVC and
macOS: NOT RUN. The status stays "native legs pending".
