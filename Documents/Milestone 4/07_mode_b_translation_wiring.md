# Stage 07 — Mode B in the GUI: translation controls wired to the engine

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 06](06_live_latency_telemetry.md) · [Stage 08 →](08_execution_providers.md)

**Status:** ☐ not started — blocked on Milestone 3
**Depends on:** [Stage 05](05_gui_engine_host_mode_a.md),
[M3 / Stage 08](../Milestone%203/08_realtime_mode_b_pipeline.md) (live Mode B pipeline),
[M3 / Stage 09](../Milestone%203/09_s2ut_controls_language_switching.md) (S2UT controls, `reset_translator_only()`)
**Owner persona:** `@ui-telemetry`

## Objective

Make the Translate panel, the Live mode switch and the tray's Mode and
Languages submenus drive a real Mode B pipeline: translation on/off, source
and target language, Wait-*k*, and lag telemetry.

## Context & verified starting point (2026-11-14)

- The UI half exists: `TranslationModel` (`langs.json`, same-language guard,
  pair default *k*, translator availability), `ViewTranslate.cpp`, and the
  `translate` / `wait_k` / `adaptive_policy` descriptors in `kParams`.
- No engine consumes them. Milestone 3 is not started (all eleven stages ☐ in
  its master). `WaitKPolicy.hpp` is the pre-M3 state machine that M3 Stage 04
  upgrades, and there is no `translator_enc.onnx` / `translator_dec.onnx`
  runtime.
- `VoiceParams` already carries `translate`, `wait_k` and `adaptive_policy` as
  descriptor values plus `src_lang_id` / `tgt_lang_id` (`VoiceParams.hpp`).
  The worker-side transitions that act on them are M3 Stage 09 work.

This stage cannot start before M3 Stages 08–09 land. It is written now so the
contract between the two milestones is fixed.

## Tasks

- [ ] **Host:** extend `ModeAHost` into a host that also owns the Mode B
      pipeline when the bundle has translator models (or add `ModeBHost` over
      the M3 worker). The bundle decides which modes are available.
- [ ] **Controls:**
  - [ ] Live mode switch and tray Mode radio items set `translate`. A v1
        (Mode A only) bundle rejects `translate = 1` on the control thread
        with the M3 Stage 09 message; the UI stays in Mode A.
  - [ ] Translate panel Apply pushes `src_lang_id`, `tgt_lang_id` and `wait_k`
        in one snapshot. The same-language guard stays a control-thread rule.
  - [ ] Wait-*k* slider changes apply on the next chunk with no state reset.
  - [ ] Language IDs come from the *bundle's* `langs.json`. The shipped
        default table is display fallback only; it must never supply IDs for
        a bundle that has its own table.
- [ ] **Telemetry:** reads, writes, flushes, `lag_ms` and Average Lagging
      estimate, published by the Mode B worker through atomics and shown in
      the Translate and Telemetry panels.
- [ ] **Presets:** voice-only presets leave `translate`, the language pair and
      `wait_k` unchanged in Mode B (M3 Stage 09 rule), verified through the
      GUI command path, not only the loader.
- [ ] **Tests (ORT-free, fake translator steps):**
  - [ ] Mode A → B → A through `AppModel` commands resets translator state
        only, and the output step stays under the click bound.
  - [ ] `lang es en` → `en de` from the tray action queue reaches the worker
        as one snapshot (no intermediate `es → de` pair is ever applied).
  - [ ] `translate = 1` on a v1 bundle is rejected and reported.
  - [ ] Settings restore: a saved pair that the loaded bundle does not support
        falls back to the bundle's first valid pair with a visible message.

## Implementation notes

- Language switches and mode switches are a crossfade plus a translator-only
  reset in the worker. The GUI must not stop and restart the stream for them.
- Mode B onset latency is about `k × 80 ms` above Mode A. The latency card
  from Stage 06 adds the Wait-*k* term when `translate = 1`.

## Files touched

- `src/pipeline/ModeAHost.*` (or new `ModeBHost.*`), `frontend/core/EngineHost.*`,
  `frontend/core/AppModel.*`, `frontend/core/TranslationModel.*`,
  `frontend/core/Telemetry.*`, `frontend/imgui/ViewTranslate.cpp`,
  `frontend/imgui/ViewLive.cpp`, `frontend/core/TrayModel.cpp`,
  `tests/test_frontend_model.cpp`

## Definition of done

- With an S2UT bundle, the GUI switches Mode A ↔ B, changes the language pair
  and Wait-*k* live, with 0 deadline misses and no audible click.
- Lag telemetry is visible while Mode B runs.
- All new ORT-free tests pass; the four native legs are recorded.
