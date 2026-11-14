# BUG-022 — A wrong-typed JSON field in `status.json` or `draft.json` terminates the GUI

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-022` |
| **Name** | *Null ETA* — `json::value()` throws and nothing catches |
| **Reported** | 2026-11-14 |
| **Severity** | High (process termination while audio is running; a bad `draft.json` terminates on every launch) |
| **Status** | ☐ open |
| **Component** | `frontend/core/VoiceStudio.cpp` (`poll_status`, `load_draft_json`, `take_from_json`); the same pattern should be checked in `Settings.cpp`, `VoiceCatalog.cpp`, `TranslationModel.cpp` |
| **Blocks** | [Milestone 4 / Stage 03](../Milestone%204/03_voice_studio_translate_catalogue_panels.md), [Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md) |

## Summary

The Studio parses files with `json::parse(text, nullptr, false)` (no
exception on a syntax error) and then reads fields with
`j.value("key", default)`. nlohmann's `value()` returns the default only when
the key is **absent**. When the key is present with another type, the
conversion throws `type_error.302`.

No `try` / `catch` exists anywhere under `frontend/`, and the build does not
define `JSON_NOEXCEPTION`. The exception leaves `AppModel::tick` and `main`,
and the process terminates. The executable is a `WIN32`-subsystem program, so
on Windows there is no console message either.

## Failure scenarios (by reading; not executed — library semantics)

| Input | Code | Effect |
|---|---|---|
| `status.json` = `{"stage":"finetune","progress":0.4,"eta_s":null}` (written by a training script that does not know the ETA yet) | `VoiceStudio.cpp:666` `j.value("eta_s", -1.0)`; polled once per second while the Studio section is visible (`AppModel.cpp:189-192`) | GUI terminates within 1 s of opening Studio; the running stream is cut |
| `status.json` with `"progress": "40%"` | `VoiceStudio.cpp:664` | same |
| `draft.json` with `"schema": "1"` or `"step": "2"` (hand edit, or a newer build's format) | `VoiceStudio.cpp:609`, `:633`; loaded from `initialise()` when one draft exists (`AppModel.cpp:128-131`) | GUI terminates at every launch until the file is deleted |
| `draft.json` with `"consent": "yes"` | `VoiceStudio.cpp:617` | same |

`status.json` is written by another process, possibly while the GUI reads it.
A partially written file is handled (parse fails, `nullopt`); a complete file
with an unexpected type is not.

## Expected

A malformed or unexpected file is reported ("status.json is not readable:
…") and ignored. The GUI and the audio stream keep running.

## Root cause

`value()` was treated as "default on any problem". It is "default when
missing".

## Suggested fix

- Add typed readers in one place (`frontend/core/JsonRead.hpp`):
  `read_string(j, key, def)`, `read_number(j, key, def)`,
  `read_bool(j, key, def)`, `read_int(j, key, def)`. Each checks
  `is_string()` / `is_number()` / `is_boolean()` before `get<>()` and returns
  the default otherwise, optionally appending a warning.
- Replace every `value()` call on external input in `frontend/core/`.
  `Settings::from_json` already promises "type mismatches produce warnings and
  keep the default"; verify it holds with the same tests.
- Add one top-level guard in the GUI main loop that reports an unexpected
  exception and shuts the stream down in order. It is a last line of defence,
  not a replacement for the typed readers.

## Regression tests (`tests/test_frontend_model.cpp`)

- `poll_status` on files with `null`, string and object values for each field
  returns a status with defaults (or `nullopt`) and does not throw.
- `load_draft_json` with a wrong type for each field returns false or keeps
  the default, and does not throw.
- The same table for `Settings::from_json`, `voices.json`, `langs.json` and
  the catalogue `index.json` loaders.

## Workaround

Delete or correct the offending file. For training scripts: omit unknown
fields from `status.json` rather than writing `null`.
