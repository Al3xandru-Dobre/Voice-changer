# Stage 10 — GUI: Quick Voice flow & Privacy centre

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 09](09_synthetic_audio_marking_disclosure.md) · [Stage 11 →](11_verification_privacy_audit_exit.md)

**Status:** ☐ not started
**Depends on:** [Stage 07](07_cpp_enrolment_engine_cli.md) (`EnrollJob`), [Stage 08](08_runtime_profile_integration.md) (profile commands),
[Stage 09](09_synthetic_audio_marking_disclosure.md) (disclosure specification), [Stage 01](01_privacy_consent_misuse_architecture.md) (texts),
[M4 / Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md) (green build gate),
[M4 / Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md) (the GUI runs Mode A)
**Owner persona:** `@ui-telemetry`
**Specification:** this file; `frontend/README.md` §4.4 gains a "Quick voice" subsection when it lands

## Objective

Let a user create, try, inspect, export and delete a personal voice from the
desktop application, with consent asked plainly before anything is read, and
with one place that shows every piece of voice data the application holds.

## Context & verified starting point (2026-11-18)

- The Voice Studio is a six-step wizard for the M2 Stage 10 protocol
  (`frontend/core/VoiceStudio.*`, `frontend/imgui/ViewStudio.cpp`). It
  analyses WAV takes, keeps a draft, and writes a job for the Python
  scripts. It does not enrol, record or train. Three defects are open
  (BUG-021, BUG-022, BUG-025) and belong to M4 Stage 04.
- Consent is a checkbox ("I have the right to use this voice") and a note.
- `AppModel` is the only place that mutates state; views read it and call
  commands. Native file dialogs return on an SDL thread, are queued, and are
  drained on the UI thread (`GuiApp::process_dialog_results`).
- No command deletes a voice. Studio drafts sit under the config directory.
- The tray already has icon states and a voice submenu.

## Design

### Entry

The Studio section opens on a choice:

- **Quick voice** — "From a few recordings, in about a minute. Approximates
  the timbre and pitch range."
- **Studio voice** — "From 1–3 hours of studio recordings and a GPU
  fine-tune. Closest result." (the existing wizard, unchanged)

### Quick voice steps

1. **Whose voice, and consent.** "My own voice" is the default. Name and
   display name. The consent text for the chosen subject is shown in full,
   with separate unticked boxes for each scope. "Another person's voice"
   appears only if open decision 1 allows it, requires an evidence file and
   shows the user's responsibilities. Nothing is read from disk before this
   step is complete.
2. **Add recordings.** Native file dialog, multiple selection (and file drop
   onto the window as a stretch). A short guide: one person, quiet room, at
   least 30 seconds of speech, two to five minutes is better. Supported
   formats are stated.
3. **Check.** One row per file: duration, speech seconds, SNR, clipping,
   verdict, consistency. Totals against the minimum and the recommended
   amount. Warnings in plain words: too short, too noisy, clipped, "these
   recordings may contain more than one voice". The user can remove files
   and continue with warnings, but not below the minimum.
4. **Confirm it is you** (own voice, optional). The user reads a short
   prompted sentence into the microphone; the app compares it with the
   recordings and shows the result. Passing grants the *verified* badge.
   Skipping keeps the profile unverified. The sample is discarded
   immediately. Shown only if open decision 10 ships the badge.
5. **Create.** Progress bar and Cancel. If the Refined tier was accepted in
   Stage 05, a "Refine (takes a few minutes)" action appears afterwards,
   not before.
6. **Result.** The quality numbers, what the profile does and does not
   reproduce, "Use as Voice A", "Use as Voice B", and where the data is
   stored. If the profile is `marking_required`, that is stated.

### Privacy centre (Settings → Privacy & voice data)

- **A statement backed by the audit:** "FluxVoice processes your voice on
  this computer and has no network code", with the date of the last audit.
- **Everything the application holds:** personal profiles; Studio drafts
  (which contain take paths and a consent note); settings that name a voice.
- **Per profile:** name, created date, subject, consent text version (with
  "show text"), verified state, speech seconds used, number of source files,
  compatibility with the active bundle, marking state, review date.
- **Actions:** Export…, Disable / Enable, Re-enrol…, Delete. "Delete all
  voice data" removes every profile and Studio draft after one confirmation
  that lists what will go.
- **Open data folder**, the privacy notice (offline text), and "Check a
  recording for the FluxVoice mark".

### Disclosure

Built from the Stage 09 specification: the "AI voice active" indicator in
the window, Mini panel and tray icon; the first-use notice; the copyable
disclosure sentence.

## Tasks

- [ ] `frontend/core/QuickVoiceModel.{hpp,cpp}` (toolkit-free): step state
      machine, consent state, file list with `AudioQc` results, job handle,
      result. `can_advance(why)` as in `VoiceStudioModel`.
- [ ] `AppModel` commands: `quick_voice_begin`, `quick_voice_set_consent`,
      `quick_voice_add_files`, `quick_voice_remove_file`,
      `quick_voice_start`, `quick_voice_cancel`, `quick_voice_verify`,
      `profile_export`, `profile_set_disabled`, `profile_erase`,
      `profile_reenroll`, `erase_all_voice_data`. Views call only these.
- [ ] **Threads.** File checks and `EnrollJob` run off the UI thread;
      results are queued and drained in `AppModel::tick`. Closing the window
      or the application cancels a running job and writes nothing.
- [ ] **Live sample for verification.** Taken from the worker's input into a
      preallocated buffer on request, when the stream is running; the wizard
      asks the user to start the stream otherwise. Nothing is added to the
      audio callback.
- [ ] `frontend/imgui/ViewStudio.cpp` (entry choice, Quick voice steps),
      `ViewSettings.cpp` or a new `ViewPrivacy.cpp` (Privacy centre),
      `ViewVoices.cpp` and the tray voice submenu (profiles with *private*,
      *verified* and *marked* badges), `ViewLive.cpp` and `TrayModel`
      (indicator).
- [ ] Erase flow: an in-use profile shows "in use: conversion will stop"
      before confirmation; afterwards the `EraseReport` is shown, including
      presets that still name the voice.
- [ ] Consent and notice texts loaded from `frontend/assets/privacy/` by
      version; a missing or altered text (hash mismatch) blocks enrolment.
- [ ] Move Studio drafts to the non-roaming data directory, with a one-time
      migration, and list them in the Privacy centre (inventory row 12).
- [ ] Wording review against
      [Stage 01](01_privacy_consent_misuse_architecture.md): no pre-ticked
      boxes, no bundled scopes, withdrawal as easy as consent,
      "approximates" not "copies".
- [ ] Tests (`tests/test_frontend_model.cpp`, new
      `tests/test_quick_voice_model.cpp`), headless over the fakes:
  - [ ] no file is opened before consent is complete;
  - [ ] enrolment cannot start without the `enroll` scope; a third-party
        subject needs evidence and is refused when decision 1 forbids it;
  - [ ] below the minimum speech the flow cannot advance; warnings do not
        block;
  - [ ] cancel and application close leave the store unchanged;
  - [ ] the job result is applied on `tick`, never from the job thread;
  - [ ] erase removes the profile from every voice list; erasing the active
        voice stops the stream with the message;
  - [ ] export writes `profile.json` and `README.txt` identical to the
        store's;
  - [ ] a damaged `profile.json` appears as *invalid* with its reason and
        does not terminate the GUI;
  - [ ] the indicator state follows conversion on and off;
  - [ ] "Delete all voice data" leaves the store and the drafts empty.
- [ ] `frontend/README.md`: §4.4 Quick voice, new §4.x Privacy centre, §9
      profile format, §10 verification rows.
- [ ] Headless smoke run of the new sections (`--smoke-frames`,
      `--screenshot`), 0 `[imgui-error]` lines.

## Implementation notes

- **Consent comes before the file dialog.** Reading a recording to check it
  is already processing.
- **Two flows, one consent model.** The Studio wizard's checkbox is replaced
  by the same versioned record, so a Studio voice and a Quick voice are
  described the same way in the Privacy centre.
- **Layering.** `QuickVoiceModel` knows nothing of SDL or ImGui. The tray
  and the Mini panel use the same commands as the main window.
- **Texts are assets, not string literals.** English first. A release in
  another language needs the consent text and the notice in that language.
- **Honest result screen.** It shows what was measured (speech used,
  quality warnings) and states the limits. It does not show an invented
  "similarity percentage".

## Files touched

- `frontend/core/QuickVoiceModel.{hpp,cpp}` (new), `AppModel.{hpp,cpp}`,
  `VoiceStudio.{hpp,cpp}`, `VoiceCatalog.{hpp,cpp}`, `TrayModel.{hpp,cpp}`,
  `Settings.{hpp,cpp}`, `Paths.{hpp,cpp}`, `frontend/imgui/ViewStudio.cpp`,
  `ViewPrivacy.cpp` (new), `ViewVoices.cpp`, `ViewLive.cpp`,
  `ViewMiniPanel.cpp`, `Views.hpp`, `frontend/assets/privacy/**`,
  `frontend/CMakeLists.txt`, `tests/test_quick_voice_model.cpp` (new),
  `tests/test_frontend_model.cpp`, `frontend/README.md`

## Definition of done

- A user completes Quick voice with their own recordings and converts live
  speech into the result, without leaving the application.
- Export, disable, re-enrol, delete and "delete all" work from the Privacy
  centre; delete is at most two actions from the profile.
- Every listed test passes headless and fails when its rule is removed.
- The texts shown are the reviewed, versioned assets from Stage 01.
- Warnings-as-errors build and CTest pass on the four native legs, recorded
  as PASS / FAIL / NOT RUN; the real file dialogs and tray are manual rows.
