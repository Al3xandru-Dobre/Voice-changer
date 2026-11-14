# BUG-021 — Voice Studio saves a draft on every keystroke into an unvalidated path

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-021` |
| **Name** | *Prefix folders* — the draft directory is the half-typed voice name |
| **Reported** | 2026-11-14 |
| **Severity** | High (silent loss of a recording project's draft; writes outside the Studio directory) |
| **Status** | ☐ open |
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
