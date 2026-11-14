# Stage 03 — Voice controls, presets, Voice Studio, Translate & Catalogue panels

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 02](02_device_selectors.md) · [Stage 04 →](04_build_gate_frontend_hardening.md)

**Status:** ◐ UI and model layer landed 2026-11-02; three defects open in the
Studio ([BUG-021](../Bug%20Report/BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md),
[BUG-022](../Bug%20Report/BUG-022_wrong_typed_json_field_terminates_gui.md),
[BUG-025](../Bug%20Report/BUG-025_studio_commands_not_runnable.md)); the controls do not reach GUI audio yet
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

## As built (verified by reading on 2026-11-14)

### Voices and presets (§4.3)

| Claim | State | Where |
|---|---|---|
| Sliders generated from `kParams`; no ranges in UI code | ☑ (exception: Wait-*k*, below) | `Widgets.cpp:128-174` |
| No dry/wet slider | ☑ | — |
| Voice A / B from `voices.json` | ◐ one active bundle, not "grouped by bundle"; a licence pill instead of a lock badge | `ViewVoices.cpp:32-42` |
| Preset bar, built-in + user presets, loader warnings inline | ☑ | `ViewVoices.cpp`, `AppModel.cpp:487-539` |
| "Save as…" writes a voice preset | ☒ it writes every `kParams` value, including `translate`, `wait_k` and `adaptive_policy`; loading it later changes the translation state, against the M3 Stage 09 rule and the panel's own help text | `AppModel.cpp:524` → `Presets.cpp:196` |
| A user preset named like a built-in | ☒ unreachable: the first name match wins | `AppModel.cpp:487-488` |
| "Applied" indicator | ◐ shown, but it compares delivery generations; it reads "applied" for controls the worker ignores and after a dropped snapshot ([BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md)) | `ViewVoices.cpp:141-143` |
| Controls act on GUI audio | ☐ gain, limiter and mute only | `EngineHost.cpp:113-117` → [Stage 05](05_gui_engine_host_mode_a.md) |
| Last preset restored at start | ☒ only the label: the UI shows "Deep" while every parameter except gain is at its default | `AppModel.cpp:116-119` |

### Voice Studio (§4.4)

| Claim | State | Where |
|---|---|---|
| Six-step wizard, consent, setup and material checklists | ☑ | `VoiceStudio.*`, `ViewStudio.cpp` |
| WAV analysis (duration, rate, depth, channels, peak, RMS, clipping, SNR) | ◐ works for well-formed files; a hostile or damaged header can request a multi-gigabyte allocation, NaN samples pass as "clean", and analysis runs on the UI/control thread | `VoiceStudio.cpp:212-275` |
| Draft persists across restarts | ☒ BUG-021 | — |
| Room tone: 30 s per session | ◐ the threshold in code is 25 s; the message says 30 | `VoiceStudio.hpp:106`; `VoiceStudio.cpp:306-307` |
| "Exact commands", Copy, Write job | ☒ BUG-025: one script and five options do not exist; the `.ps1` does not parse | `VoiceStudio.cpp:435-446` |
| `status.json` polling | ☒ BUG-022: a wrong-typed field terminates the GUI | `VoiceStudio.cpp:655-670` |
| Install step (voice appears when the bundle lists it) | ☑ | `VoiceStudio.cpp:672-677` |

### Translate (§4.5)

| Claim | State | Where |
|---|---|---|
| Language dropdowns from `langs.json`, default-table fallback, Swap | ☑ | `TranslationModel.*`, `ViewTranslate.cpp` |
| Same-language guard "disables Apply" | ◐ there is no Apply button; a change applies at once and a bad pair is rejected with a message | `AppModel.cpp:567-572` |
| Wait-*k* slider from the descriptor | ☒ range 1..10, default 3 and the 80 ms chunk are hard-coded in four places | `ViewTranslate.cpp:70`; `TranslationModel.cpp:148`; `Settings.cpp:145-148`; `TranslationModel.hpp:65` |
| Adaptive policy hidden unless the bundle declares it | ☒ always shown | `ViewTranslate.cpp:77-79` |
| Whole panel disabled when the bundle has no translator | ◐ only the mode toggle is disabled | `ViewTranslate.cpp:86` |
| Lag telemetry | ☐ placeholder text | `ViewTranslate.cpp:98-102` → [Stage 07](07_mode_b_translation_wiring.md) |
| Saved Wait-*k* restored at start | ☒ replaced by the pair default whenever the saved pair is not en→es | `AppModel.cpp:107-109`, `:573-581` |
| Tray "Languages": pairs with *k*, recent pairs | ◐ every ordered pair, no *k*, no recents | `AppModel.cpp:589-596` |

### Catalogue and Telemetry (§4.6, §4.7)

| Claim | State | Where |
|---|---|---|
| Installed bundles + `index.json` entries, search, filters, sort, licence class | ☑ | `VoiceCatalog.*`, `ViewCatalogue.cpp` |
| "Open page" opens the system browser | ◐ `page` / `url` from a loaded index go to `SDL_OpenURL` without a scheme check | `ViewCatalogue.cpp:95-98`; `GuiApp.cpp:437-439` |
| Telemetry counters, RSS, meters | ◐ peak bars only (RMS is computed but not drawn); histories for underruns and queued ms only | `ViewLive.cpp:130-145`; `ViewTelemetry.cpp` |
| Queue identity check | ◐ the formula is printed as a label; nothing is checked; the value is wrong after a restart ([BUG-024](../Bug%20Report/BUG-024_gui_queued_latency_not_rebaselined.md)) | `ViewTelemetry.cpp:30` |
| `[stats]` line in the CLI's format | ◐ no `skipped=` / `prefilled=` (the analyzer accepts the short form) | `Telemetry.cpp:14-16` |

## Residual tasks

Defect fixes are scheduled in [Stage 04](04_build_gate_frontend_hardening.md).
Scope items that remain in this stage:

- [ ] "Save as…" writes voice parameters only; translation keys are written
      only on an explicit "include translation state" choice.
- [ ] User presets take precedence over built-ins of the same name, or the
      name is refused at save time.
- [ ] Restoring the last preset re-applies its parameters, or the label is
      cleared at start.
- [ ] Wait-*k* range and default read from `param_desc(ParamId::WaitK)`; the
      chunk length from the bundle contract.
- [ ] Adaptive-policy toggle hidden unless the bundle declares the capability.
- [ ] Translate panel fully disabled (readable) for a Mode A only bundle.
- [ ] Saved Wait-*k* survives a restart for every pair; a preset's `wait_k`
      wins over the pair default when both are present.
- [ ] WAV analysis: bound every header-derived size before allocating, treat
      non-finite samples as a rejected take, and run the analysis off the
      UI thread with the result drained in `AppModel::tick`.
- [ ] Catalogue URLs: accept `https://` (and `http://`) only.
- [ ] RMS meters drawn; `frontend/README.md` §4 corrected wherever the code is
      the intended behaviour (sparkline length, Apply button, tray Languages).

## Definition of done

- BUG-021, BUG-022 and BUG-025 are fixed or, for BUG-025, the step is
  labelled as unavailable until M2 Stage 10 lands.
- Every ◐ and ☒ row above is ☑, or the specification row is changed with a
  reason.
- Each fixed rule has a test in `tests/test_frontend_model.cpp` that fails
  when the rule is broken.
