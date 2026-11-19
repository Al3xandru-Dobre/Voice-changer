# Stage 09 — S2UT controls, language switching & Mode A↔B toggle

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 08](08_realtime_mode_b_pipeline.md) · [Stage 10 →](10_cross_lingual_target_voice.md)

**Status:** ◐ implemented and unit-tested; live console DoD pending the rig
**Depends on:** Stage 08 (live Mode B pipeline), [Milestone 2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) (`VoiceParams`, presets, console command queue)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §9 C++ runtime & controls

## Objective

Let the user toggle between **Mode A (Voice Changer)** and **Mode B (Simultaneous
Translation + VC)**, adjust the **Wait-*k* latency/quality slider** (`k ∈
[1..10]`), and switch **source and target languages** (`en`, `es`, `fr`, `de`)
live without restarting the audio stream or causing clicks, allocations, or stale
KV-state leaks.

Expose these controls through the same `VoiceParams` descriptor table, preset JSON
schema, CLI flags, and interactive console commands that Milestone 4's ImGui
dashboard will consume.

## Context & references

- [M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) created
  `src/voice/VoiceParams.hpp` (`kParams[]` descriptor table), lock-free SPSC
  `ReaderWriterQueue<VoiceParams>` delivery from the control thread to the
  worker, `presets/*.json`, and interactive console commands (`set`, `get`,
  `preset`, `save`, `voices`, `help`).
- [Milestone 4](../Milestone%204/00_master.md) specifies an ImGui dashboard with:
  - translation ON/OFF toggle;
  - Wait-*k* slider (`k ∈ [1..10]`);
  - language selectors;
  - voice sliders driven by `fluxvoice --list-params --json`.
- **Why live mode/language transitions need care:**
  - Toggling **Mode A $\leftrightarrow$ Mode B** (`translate: 0 ↔ 1`) or changing
    **`src_lang` / `tgt_lang`** invalidates the translator's `mem_k_*`/`mem_v_*`
    and `dec_kv_*` ring buffers and `UnitProsodyFifo`.
  - However, Stage 1 (`encoder`) and Stage 3 (`vocoder`) are language-universal!
    Their conv left-context states do **not** need to be zeroed on a mode or
    language switch; instead, the worker crossfades the output envelope over one
    20 ms frame while resetting only `translator_enc`, `translator_dec`,
    `WaitKPolicy`, and `UnitProsodyFifo`.
  - Changing **`wait_k`** (`1..10`) mid-stream does **not** reset any KV cache —
    it simply calls `waitk_.set_k(new_k)`, immediately shifting the
    `t_src - t_tgt >= k` threshold on the next chunk!

## Parameters added to `VoiceParams`

| Name | Unit | Range / Domain | Default | Acts on | Transition behaviour | Tier |
|---|---|---|---|---|---|---|
| `translate` | bool (`0`/`1`) | `0` (Mode A VC), `1` (Mode B S2UT) | `0` (`1` under `--s2ut`) | Pipeline routing after Stage 1 | 20 ms output ramp + reset translator KV & `WaitKPolicy` | MVP |
| `wait_k` | chunks (80 ms) | `1 … 10` (integer) | Pair default in `langs.json` (`3`) | `WaitKPolicy::set_k(k)` | Immediate on next chunk; **no** KV reset | MVP |
| `src_lang` | lang code / ID | `langs.json` (`en`,`es`,`fr`,`de`) | `en` | `translator_enc` `src_lang` tensor | Reset translator KV, `WaitKPolicy`, `UnitProsodyFifo` | MVP |
| `tgt_lang` | lang code / ID | `langs.json` (`en`,`es`,`fr`,`de`) | `es` | `translator_dec` `tgt_lang` tensor | Reset translator KV, `WaitKPolicy`, `UnitProsodyFifo` | MVP |
| `adaptive_policy` | bool (`0`/`1`) | `0` (Strict Wait-*k*), `1` (CTC-Adaptive) | Gate winner | `WaitKPolicy` readiness mode | Immediate on next chunk | Stretch |

## Tasks

- [x] **Extend `src/voice/VoiceParams.hpp` and `src/voice/Presets.{hpp,cpp}`:**
  - [x] Add `translate`, `wait_k`, and `adaptive_policy` to `kParams[]` alongside
        `src_lang_id` and `tgt_lang_id` fields on `VoiceParams`.
  - [x] Extend `fluxvoice --list-params --json` to include the S2UT descriptors
        and the loaded `langs.json` table so M4's UI can build the toggle,
        Wait-*k* slider, and language dropdowns from one JSON dump.
  - [x] Allow optional `"translate"`, `"wait_k"`, `"src_lang"`, `"tgt_lang"` keys
        in `presets/*.json` (omitted keys leave the current mode/languages
        unchanged when switching voice-only presets like **Deep** or **Bright**!).
- [x] **Worker transition logic in `InferencePipeline`:**
  - [x] Detect changes in `params.translate`, `params.src_lang_id`, or
        `params.tgt_lang_id` at hop start:
    - apply a 20 ms cosine fade-out/fade-in across the switch boundary;
    - call `reset_translator_only()` (`translator_enc.reset()`,
      `translator_dec.reset()`, `waitk_.reset()`, `unit_fifo_.clear()`,
      `prosody_ctx_.reset()`) while keeping Stage 1 `encoder`, Stage 3 `vocoder`,
      and `PitchTracker` states continuous.
  - [x] Detect changes in `params.wait_k`: call `waitk_.set_k(...)` in-place with
        zero state reset.
  - [x] If the loaded bundle is a v1 (Mode A-only) bundle and the user requests
        `translate = 1`, reject the command on the control thread with `"bundle
        has no translator models (translator_enc.onnx / translator_dec.onnx)"`
        and stay in Mode A.
- [x] **Interactive console commands in `src/main.cpp`:**
  - [x] `mode vc` / `mode s2ut` (shortcut for `set translate 0|1`)
  - [x] `lang <src> <tgt>` (validates both codes in `langs.json` and `src != tgt`,
        then pushes updated `VoiceParams`)
  - [x] `waitk <1..10>` (shortcut for `set wait_k <k>`)
  - [x] `langs` (lists supported languages and current `src → tgt (k=...)` status)
  - [x] Update `help` output.
- [x] **Extend `tests/test_voice_params.cpp` and `tests/test_inference_pipeline.cpp`
      (ORT-free):**
  - [x] Switching a voice-only preset (**Neutral** → **Deep**) while in Mode B
        preserves `translate == 1`, `src_lang`, `tgt_lang`, and `wait_k`.
  - [x] Toggling `translate` `0 → 1 → 0` and switching `lang es en → en de` on the
        fake pipeline resets translator states cleanly, keeps sample-to-sample step
        deltas below the click bound, and allocates 0 bytes.
  - [x] Changing `wait_k` from `5` to `2` mid-utterance during `kRead` (when
        `t_src = 3, t_tgt = 0`) immediately transitions the very next chunk to
        `kWrite` without losing the 3 buffered source chunks.

## Implementation notes

- **Voice presets vs. translation state.** A user translating `Es→En` who clicks
  the **Deep** voice preset expects their voice to become deeper — not for
  translation to turn off or switch languages! Making `"translate"`, `"src_lang"`,
  and `"tgt_lang"` optional in `presets/*.json` (and omitted in the voice-timbre
  presets) keeps voice styling orthogonal to translation mode.
- **Same-language guard (`src_lang == tgt_lang`).** If a user selects the same
  source and target language in the console (`lang en en`), warn and suggest
  `mode vc` (Mode A gives ~60–100 ms latency vs. Mode B's ~330 ms).

## Files touched

- `src/voice/VoiceParams.hpp`, `src/voice/Presets.{hpp,cpp}`,
  `src/pipeline/InferencePipeline.{hpp,cpp}`, `src/main.cpp`,
  `tools/fv_s2ut_file.cpp`, `tests/test_voice_params.cpp`,
  `tests/test_inference_pipeline.cpp`,
  `Documents/Milestone 4/00_master.md` (handoff note for S2UT UI controls)

## Definition of done

- All new ORT-free unit tests in `test_voice_params` and `test_inference_pipeline`
  pass (orthogonal voice presets, click-free Mode A↔B and language switches,
  in-place `wait_k` hot adjustment, zero heap allocations).
- All Stage 09 voice controls (`pitch_shift`, `intonation`, `timbre_blend`,
  `gain`, `limiter`) pass their M2 Stage 09 DoD checks when applied to Mode B
  translated output via `fv_s2ut_file`.
- Live console commands `mode s2ut`, `mode vc`, `lang fr en`, and `waitk 4` switch
  within one hop with 0 `misses` and no audible click.

## As implemented (2026-11-18)

**Status: ◐ implemented and unit-tested; live console DoD pending the rig.** Open: live `mode s2ut` / `mode vc` / `lang fr en` / `waitk 4` switching within one hop with 0 `misses` and no audible click (NOT RUN, no trained bundle / no rig); the M2 Stage 09 control DoD on real Mode B output via `fv_s2ut_file` (NOT RUN, fixture bundle only).

### Files and APIs

- `src/voice/VoiceParams.hpp` (`translate`, `wait_k`, `adaptive_policy` descriptors with `src_lang_id` / `tgt_lang_id` fields), `src/voice/Presets.{hpp,cpp}` (optional `translate` / `wait_k` / `src_lang` / `tgt_lang` keys; `--list-params --json` adds `langs` and `default_wait_k` from a loaded `langs.json`; voice-only presets leave translation state untouched).
- `src/pipeline/InferencePipeline.{hpp,cpp}`: `set_params()` transition rules (below); `mode()`; `src/main.cpp`: console `mode vc|s2ut`, `lang <src> <tgt>`, `waitk <k>`, `langs`, updated `help`; CTest `cli_list_langs`, `cli_list_params_langs`.
- Tests: `tests/test_inference_pipeline.cpp` (`test_s2ut_live_switching`, `test_s2ut_wait_k_live`, `test_s2ut_adaptive_toggle_live`, `test_s2ut_param_rejections`, `test_s2ut_voice_preset_keeps_translation`), `tests/test_voice_params.cpp`.

### Deviations from the stage text

| Stage text | As implemented | Reason |
|---|---|---|
| 20 ms cosine fade-out then fade-in across the switch | A single **20 ms raised-cosine crossfade from the last emitted sample** into the new stream; only the translator side is reset (`translator_enc/dec`, `WaitKPolicy`, `UnitProsodyFifo`, `prosody_ctx`; encoder, vocoder and tracker keep state) | No separate fade-out/fade-in stage is needed and there is no gap or click; the starting level is the actual last sample |
| `lang` pushes new `VoiceParams` | `lang` also **resets `wait_k` to the pair default** from `langs.json` | The default depends on the pair (`de->en` 4, `es->en` 3) |
| Rejection on the control thread | `set_params()` itself refuses `translate = 1` without translators and, while translating, unknown or equal language ids (previous params kept); on the Mode A route language ids are stored, not checked. A v1 bundle with `translate = 1` is rejected with `"bundle has no translator models (translator_enc.onnx / translator_dec.onnx)"` | Two layers: the worker never accepts an impossible state |
| (not specified) | `mode()` is a relaxed atomic readable from the UI thread; `waitk()` / `params()` stay worker-only | M4 UI reads the live route without a lock |
| (not specified) | The worker's `set_params()` result is dropped by `ModeAWorker`, so the **control thread must pre-validate** (language ids, `src != tgt`, translators present) before pushing | A refusal on the worker would be silent |
| (not specified) | `adaptive_policy` calls `WaitKPolicy::set_adaptive_mode()` in place, no state reset | Stretch tier: delay-only until BUG-029 is fixed |
| Same-language guard suggests `mode vc` | Console `lang en en` is refused with that hint | As specified |

Stage 08 interplay: a switch hop is latency-governor protected (see Stage 08), so a queue-drift skip cannot cut the crossfade.

## Verification (AGENTS rule 7; recorded 2026-11-18)

Commands: `cmake -S . -B build-<leg> -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_BUILD_TESTS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON`, `cmake --build build-<leg> --config Release` (all targets, including `fluxvoice_header_checks`), `ctest --test-dir build-<leg> -C Release --output-on-failure`; the venv leg adds `python_training_m3_stage07_golden`.

| Leg | Result |
|---|---|
| Linux GCC 13.3 + ORT 1.31.0 | PASS: warning-free `-Werror` build, hardware-free CTest (non-venv 104/104 at the lead's last gate; final total: see lead's final gate) |
| Linux Clang 18.1.3 + ORT 1.31.0 | PASS (same gate) |
| Linux GCC, no ORT | PASS (103/103 at the lead's last gate; final: see lead's final gate) |
| Linux Clang, no ORT | PASS (same gate) |
| Venv `python_training_m3_stage07_golden` | PASS: 27 tests incl. 9 C++ parity cases, bit-exact |
| Windows / MSVC, macOS / AppleClang | NOT RUN (CI pending) |
| Live console switch timing (one hop, 0 `misses`, no click) | NOT RUN (no trained bundle / no rig) |
