# Stage 08 — Runtime integration: profiles as selectable voices

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 07](07_cpp_enrolment_engine_cli.md) · [Stage 09 →](09_synthetic_audio_marking_disclosure.md)

**Status:** ☐ not started
**Depends on:** [Stage 02](02_voice_profile_contract_store.md) (`ProfileStore`, `speaker_space`),
[Stage 07](07_cpp_enrolment_engine_cli.md) (profiles exist),
[M2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) and
[M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) (`ModeAWorker`, voice selection, timbre blend).
Works against fixture bundles and hand-written profiles before Stage 04 produces weights.
**Owner persona:** `@onnx-ort-runtime` with `@rt-audio-dsp` for the controller hooks

## Objective

Make a compatible personal profile selectable exactly like a bundle voice in
`fluxvoice --vc`, `fv_vc_file` and, through the shared host, the GUI: by
name, with every voice control working, and with profiles added or erased
while the application runs. No change to the real-time path.

## Context & verified starting point (2026-11-18)

- `ModelBundle::load` reads `voices.json` into `std::vector<VoiceEntry>`
  (`name`, `display`, `embedding`, `stats`, `license`, `source`).
- `InferencePipeline` takes `std::span<const PipelineVoice>` in its
  constructor and **copies** the statistics and embeddings into a flat table.
  Voice A and B are indices in `VoiceParams`; timbre blend interpolates
  between two table entries. There is no call that changes the table later.
- Names are resolved to indices on the control thread (`voice_index` in
  `frontend/core/AppModel.cpp` and in `tools/fv_vc_file.cpp`; `find_voice`
  in `VoiceCatalog.hpp`). An unknown name is rejected and the previous voice
  stays.
- `AudioController::set_worker_hooks` gives the control thread a serialised
  quiesce → reset → resume sequence, used today for device switches.
- [M4 / Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md) plans one
  shared `ModeAHost` for the CLI and the GUI. This stage is written so that
  it works with the composition in `src/main.cpp` today and moves into
  `ModeAHost` when that lands.

## Design

- **One merged table.** `load_voice_table(bundle_voices, profile_store,
  speaker_space)` returns bundle voices first, in file order (their indices
  never change), then enabled, compatible profiles sorted by name.
- **`VoiceEntry` gains** `origin` (`Bundle` or `Profile`), `profile_id`,
  `verified`, and `marking_required`.
- **A profile that cannot be used is still listed**, with its reason:
  `speaker_space` mismatch ("needs re-enrolment for this bundle"), bundle
  without a speaker encoder, disabled, invalid file, or name collision with
  a bundle voice (the bundle voice keeps the name; the profile must be
  renamed). None of these is skipped silently, and none is selectable.
- **Changing the table while running.** A new
  `InferencePipeline::replace_voices(span)` is legal only while the worker
  is quiesced. The control thread runs: quiesce → `replace_voices` →
  re-resolve voice A and B by *name* → reset → resume. The worker never
  parses JSON or touches a file.
- **Erasing the voice in use.** The erase command says so first ("in use:
  conversion will stop"). After confirmation the stream stops with a visible
  message. It does not fall back to another voice.
- **Marking.** When voice A or B (or a blend that includes one) has
  `marking_required`, the host requires an active marker (Stage 09). Without
  one the selection is refused with a message. Until Stage 09 lands, this is
  a stub that refuses such profiles unless the developer flag
  `--allow-unmarked-profiles` is given; the flag does not exist in release
  builds.
- **Presets** keep naming voices by string. A preset that names a profile
  absent on this machine behaves like any unknown voice: rejected with a
  message, previous voice kept.

## Tasks

- [ ] `VoiceCatalog.hpp`: the `VoiceEntry` fields and `load_voice_table`;
      collision, mismatch and disabled handling with typed reasons.
- [ ] `InferencePipeline::replace_voices`, asserting the quiesced
      precondition in debug builds; same validation as the constructor
      (dimension, finiteness, index range).
- [ ] `src/main.cpp`: `--profile-dir`, `--no-profiles`; `--list-voices`
      prints origin, verified and marking state, and the reason for each
      unusable profile; `--voice` and `--voice-b` accept profile names; the
      console `voices` command shows the same.
- [ ] `tools/fv_vc_file.cpp`: the same two options, so the evaluation
      harness converts into profiles on the shipped runtime.
- [ ] Control-thread commands `add_profile`, `erase_profile`,
      `set_profile_disabled` that perform the quiesced table swap; wired
      into the CLI console now and into `ModeAHost` / `AppModel` when
      [M4 / Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md) lands.
- [ ] Mode B: the same merge in `S2utBundle::load`. A profile is usable only
      if the bundle declares a matching `speaker_space` (stretch; open
      decision 12).
- [ ] Log and `[stats]` hygiene: a voice appears by name and origin only.
- [ ] Tests (`tests/test_voice_catalog.cpp`, `test_inference_pipeline.cpp`,
      `test_mode_a_worker.cpp`, `check_voice_cli.cmake`), all ORT-free:
  - [ ] merge order and stable bundle indices;
  - [ ] each unusable reason is reported and not selectable;
  - [ ] after a table swap voice A and B resolve to the same *names*, and
        an erased active voice stops the stream with the message;
  - [ ] timbre blend between a bundle voice and a profile interpolates the
        embedding and the statistics as it does between two bundle voices;
  - [ ] `pump()` still allocates 0 bytes with a profile selected
        (`AllocationCounter`);
  - [ ] a `marking_required` profile is refused without a marker;
  - [ ] `--no-profiles` yields exactly the bundle's voices.
- [ ] Update `frontend/README.md` §9 with the profile format and the merged
      table.

## Implementation notes

- **Bundle voices first, always.** Presets, tests and the tray menu refer to
  bundle voices; their indices must not move when a profile appears.
- **Names, not indices, cross the swap.** The same rule the device selectors
  follow with endpoint keys, for the same reason.
- **Nothing here is on the hot path.** The per-hop code already reads a
  statistics block and an embedding from a flat table; a profile is one more
  row.
- **Selecting a profile changes no latency number.** Stage 11 still measures
  it, because that is the claim.

## Files touched

- `src/pipeline/VoiceCatalog.hpp`, `ModelBundle.{hpp,cpp}`,
  `InferencePipeline.{hpp,cpp}`, `src/main.cpp`, `tools/fv_vc_file.cpp`,
  `tests/test_voice_catalog.cpp`, `test_inference_pipeline.cpp`,
  `test_mode_a_worker.cpp`, `tests/check_voice_cli.cmake`,
  `frontend/README.md`; later `src/pipeline/ModeAHost.*` and
  `frontend/core/AppModel.*`

## Definition of done

- `fluxvoice --vc --model-dir <bundle> --voice <profile>` and
  `fv_vc_file --voice <profile>` convert with a profile from the store, on a
  fixture bundle in CTest and on a trained bundle when one exists.
- Adding, disabling and erasing a profile while converting works through the
  quiesced swap; the listed tests pass and each fails when its rule is
  removed.
- The existing voice, preset and worker tests pass unchanged.
- Warnings-as-errors build and CTest pass, recorded per native leg.
