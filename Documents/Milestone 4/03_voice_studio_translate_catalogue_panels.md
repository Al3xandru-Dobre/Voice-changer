# Stage 03 — Voice controls, presets, Voice Studio, Translate & Catalogue panels

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 02](02_device_selectors.md) · [Stage 04 →](04_build_gate_frontend_hardening.md)

**Status:** ◐ UI and model layer landed 2026-11-02; residual code tasks fixed in the working
tree 2026-11-18 and tested on Linux GCC and Clang only. BUG-021 and BUG-025 are fixed in tree
(native legs pending); [BUG-022](../Bug%20Report/BUG-022_wrong_typed_json_field_terminates_gui.md)
is still open; the controls do not reach GUI audio yet (Stage 05)
([BUG-021](../Bug%20Report/BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md),
[BUG-025](../Bug%20Report/BUG-025_studio_commands_not_runnable.md))
**Depends on:** [M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) (`kParams`, presets),
[M3 / Stage 09](../Milestone%203/09_s2ut_controls_language_switching.md) (S2UT descriptors),
[M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) (recording protocol, fine-tune tooling)
**Specification:** [`frontend/README.md`](../../frontend/README.md) §4.3–§4.7, §9
**Review:** [REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md)

## Objective

Panels for voices and presets, the Voice Studio wizard, translation controls
and the voice catalogue, all driven from the shared descriptor table and the
bundle's data files. This file is the as-built record and the list of gaps
between the specification and the code.

## As built (read on 2026-11-14, updated 2026-11-18 for the working tree)

Rows marked "fixed in tree" are covered by `frontend_model`, `frontend_shell`
or `frontend_studio` on Linux GCC and Clang (2026-11-18); Windows MSVC and
macOS are NOT RUN. Locations are file and function names, not line numbers.

### Voices and presets (§4.3)

| Claim | State | Where |
|---|---|---|
| Sliders generated from `kParams`; no ranges in UI code | ☑ (Wait-*k* now included, below) | `Widgets.cpp` |
| No dry/wet slider | ☑ | — |
| Voice A / B from `voices.json` | ◐ one active bundle, not "grouped by bundle"; a licence pill instead of a lock badge | `ViewVoices.cpp` |
| Preset bar, built-in + user presets, loader warnings inline | ☑ | `ViewVoices.cpp`, `AppModel.cpp` |
| "Save as…" writes a voice preset | ◐ fixed in tree: writes voice parameters only (pitch, timbre, output, voices); the translation group (`translate`, `wait_k`, `adaptive_policy`) and the language pair only with the new "Include translation state (mode, Wait-k, languages)" checkbox | `AppModel` (save preset); `Presets.cpp` |
| A user preset named like a built-in | ◐ fixed in tree: built-in names are refused at save time (case-insensitive) and the Save button is disabled with the reason; two names mapping to one file name (for example "My voice"/"My_voice") are refused; a user preset file named like a built-in is hidden with a warning in the preset bar | `AppModel`; `ViewVoices.cpp` |
| "Applied" indicator | ◐ unchanged, Stage 04 and later: it compares delivery generations; it reads "applied" for controls the worker ignores and after a dropped snapshot ([BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md)) | `ViewVoices.cpp` |
| Controls act on GUI audio | ☐ gain, limiter and mute only | `EngineHost.cpp` → [Stage 05](05_gui_engine_host_mode_a.md) |
| Last preset restored at start | ◐ fixed in tree: the preset's voice parameters are re-applied (voices, languages, Wait-*k*, mode and gain come from Settings, which is newer); a preset that no longer exists clears the label with an Info notice | `AppModel.cpp` (startup) |

### Voice Studio (§4.4)

| Claim | State | Where |
|---|---|---|
| Six-step wizard, consent, setup and material checklists | ☑ | `VoiceStudio.*`, `ViewStudio.cpp` |
| WAV analysis (duration, rate, depth, channels, peak, RMS, clipping, SNR) | ◐ fixed in tree: the parser bounds every header-derived size (fmt read capped at 40 bytes, an fmt chunk larger than the file rejected without allocating, channels 1..8, rate 8000..384000, 16/24/32-bit PCM or 32-bit float, block-align check, data clamped to the file, RIFF pad byte, zero-size chunks cannot loop). NaN/Inf samples give the new verdict `NonFinite` ("contains non-finite samples (NaN/Inf)") and the take is rejected. Analysis runs on a per-model worker thread (`request_take_analysis`, `analysis_pending`); results are applied on the UI thread in `AppModel::tick` (any section) and the draft is saved per batch; the view shows "analysing N file(s)…" | `VoiceStudio.cpp`; `AppModel::tick`; `tests/test_frontend_studio.cpp` |
| Draft persists across restarts | ◐ BUG-021 fixed in tree (native legs pending); residual fixed 2026-11-18: `project_dir()` returns an empty path for an unsafe name instead of `untitled` | `VoiceStudio.cpp` |
| Room tone: 30 s per session | ◐ fixed in tree: the minimum is 30 s in one constant and the message is built from it | `VoiceStudio.hpp`; `VoiceStudio.cpp` |
| "Exact commands", Copy, Write job | ◐ BUG-025 fixed in tree: steps that cannot run are labelled unavailable instead of printed. Prepare is `prepare_target_voice.py --input <dir> --speaker=<voice> --out <project>/prepared [--display <name>]`, available only when every take and room-tone file is in one folder named `<voice>_sNN_TTTT.wav` / `<voice>_sNN_roomtone.wav` (otherwise unavailable with the failing rule). Fine-tune is always "unavailable until M2 Stage 10 lands" (needs the Stage 05 encoder checkpoint, base vocoder checkpoint, training manifest and unit directories, none collected by the Studio; no option names a base bundle). Export is unavailable (it exports the fine-tune's checkpoint; the recipe's `--encoder-ckpt` / `--vocoder-ckpt` do not exist in `export_stateful_onnx.py`). Unavailable steps are left out of `job.json` `"commands"` (a separate `"unavailable"` object) and the run scripts (comment only); Copy is disabled and the reason shown. POSIX script: `#!/bin/sh` + `set -eu`; PowerShell: `& …` + `if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`, single-quote quoting. `takes.tsv` / `room_tone.tsv` are still written but nothing consumes them; `base_bundle` / `target_bundle` only reach `job.json`. A test checks every emitted `--option` against the script's `add_argument` strings | `VoiceStudio.cpp` (`commands`, `write_job`); `tests/test_frontend_studio.cpp` |
| `status.json` polling | ☒ BUG-022, still open: a wrong-typed field terminates the GUI; the top-level guard still shuts down rather than keeping the GUI running | `VoiceStudio.cpp` (status parsing) |
| Install step (voice appears when the bundle lists it) | ☑ | `VoiceStudio.cpp` |

### Translate (§4.5)

| Claim | State | Where |
|---|---|---|
| Language dropdowns from `langs.json`, default-table fallback, Swap | ☑ | `TranslationModel.*`, `ViewTranslate.cpp` |
| Same-language guard "disables Apply" | ☑ spec corrected: there is no Apply button; a change applies at once and an invalid or same-language pair is rejected with a message (`frontend/README.md` §4.5) | `AppModel.cpp` |
| Wait-*k* slider from the descriptor | ◐ fixed in tree: the slider, the clamp, `langs.json` `default_wait_k` validation, Settings validation and the unknown-pair default all come from `param_desc(ParamId::WaitK)` (range 1..10, default 3 today) | `ViewTranslate.cpp`; `TranslationModel.cpp`; `Settings.cpp` |
| Chunk length and adaptive capability from the bundle | ◐ new `TranslatorCaps{known, chunk_ms, adaptive_policy}` in `TranslationModel`, set through `AppModel::set_translator_caps()` (from the translator contract: `chunk_samples / sample_rate`, `has_readiness_output`). The contract is embedded in the `.onnx` files and the GUI links no ONNX Runtime yet, so caps stay unknown until Stage 05/07 loads the bundle: the onset text shows no ms figure and the adaptive toggle is hidden. A bundle change resets caps to unknown; caps without readiness switch `adaptive_policy` off. Handoff to Stages 05/07: call `set_translator_caps` after loading the bundle | `TranslationModel.*`; `AppModel` |
| Adaptive policy hidden unless the bundle declares it | ◐ fixed in tree (hidden while caps are unknown or without readiness) | `ViewTranslate.cpp` |
| Whole panel disabled when the bundle has no translator | ◐ fixed in tree: every card (Languages, Wait-*k*, Mode) is disabled but readable | `ViewTranslate.cpp` |
| Lag telemetry | ☐ placeholder text | `ViewTranslate.cpp` → [Stage 07](07_mode_b_translation_wiring.md) |
| Saved Wait-*k* restored at start | ◐ fixed in tree: the saved Wait-*k* survives a restart for every pair; a user-chosen pair change applies the pair default; a preset's own `wait_k` wins over the pair default; a pair-only preset gets the pair default | `AppModel.cpp` |
| Tray "Languages": pairs with *k*, recent pairs | ☑ spec corrected: every ordered pair of the table, no *k*, no recents | `AppModel.cpp` |

### Catalogue and Telemetry (§4.6, §4.7)

| Claim | State | Where |
|---|---|---|
| Installed bundles + `index.json` entries, search, filters, sort, licence class | ☑ | `VoiceCatalog.*`, `ViewCatalogue.cpp` |
| "Open page" opens the system browser | ◐ fixed in tree: "Open page" / "Open download link" accept only `http://` / `https://` with a host (new header-only `frontend/core/UrlPolicy.hpp`, `is_web_url`) and are disabled with a tooltip otherwise; `GuiApp::open_url` refuses others with a Warning notice (local folders use `open_path`, `file://` on purpose) | `UrlPolicy.hpp`; `ViewCatalogue.cpp`; `GuiApp.cpp` |
| Telemetry counters, RSS, meters | ◐ meters fixed in tree: `widgets::level_meter(id, peak_dbfs, rms_dbfs, w, h)` draws an RMS bar with a peak tick (red above −3 dBFS) in the Live and Mini panels; histories for underruns and queued ms only | `Widgets.cpp`; `ViewLive.cpp`; `ViewMiniPanel.cpp`; `ViewTelemetry.cpp` |
| Queue identity check | ◐ unchanged: the formula is printed as a label; nothing is checked; the value is wrong after a restart ([BUG-024](../Bug%20Report/BUG-024_gui_queued_latency_not_rebaselined.md)) | `ViewTelemetry.cpp` |
| `[stats]` line in the CLI's format | ◐ unchanged: no `skipped=` / `prefilled=` (the analyzer accepts the short form) | `Telemetry.cpp` |

## Residual tasks

Defect fixes are scheduled in [Stage 04](04_build_gate_frontend_hardening.md).
Scope items (state on 2026-11-18; all `[x]` items are verified on Linux GCC and
Clang only, MSVC and macOS NOT RUN):

- [x] "Save as…" writes voice parameters only; translation keys are written
      only on an explicit "include translation state" choice.
- [x] Built-in preset names are refused at save time (a user file with a
      built-in's name is hidden with a warning).
- [x] Restoring the last preset re-applies its parameters, or the label is
      cleared with an Info notice.
- [x] Wait-*k* range and default read from `param_desc(ParamId::WaitK)`.
- [ ] Chunk length from the bundle contract: `TranslatorCaps` and
      `set_translator_caps()` exist, but the caps stay unknown until Stage 05/07
      loads the bundle (the GUI links no ONNX Runtime yet).
- [x] Adaptive-policy toggle hidden unless the bundle declares the capability
      (hidden while caps are unknown).
- [x] Translate panel fully disabled (readable) for a bundle without a
      translator.
- [x] Saved Wait-*k* survives a restart for every pair; a preset's `wait_k`
      wins over the pair default when both are present.
- [x] WAV analysis: every header-derived size bounded, non-finite samples
      rejected, analysis off the UI thread and drained in `AppModel::tick`.
- [x] Catalogue URLs: `http://` and `https://` with a host only.
- [x] RMS meters drawn; `frontend/README.md` §4 corrected for sparkline length
      (60 s), no Apply button, tray Languages, keyboard, preset rules, room
      tone, step 5 availability, Wait-*k*, URLs, settings `.bad` and
      `mini_position`.
- [ ] Open, later stages: BUG-022 (wrong-typed JSON field), the "applied"
      indicator (BUG-023), UTF-8 conversion of SDL dialog strings on Windows
      (the Studio passes `std::string` paths from native dialogs into
      `std::filesystem::path`), `src/ui/AppWindow.cpp` deletion, CI jobs.

## Definition of done

- BUG-021, BUG-022 and BUG-025 are fixed or, for BUG-025, the step is
  labelled as unavailable until M2 Stage 10 lands. State: BUG-021 and BUG-025
  (unavailable steps labelled) fixed in tree, native legs pending; BUG-022 fixed
  in tree (typed readers), but its top-level guard still shuts the GUI down
  instead of keeping it running (Stage 04).
- Every ◐ and ☒ row above is ☑, or the specification row is changed with a
  reason.
- Each fixed rule has a test in `tests/test_frontend_model.cpp` that fails
  when the rule is broken.
