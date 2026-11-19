# Stage 05 — GUI engine host: Mode A inference in `fluxvoice_gui`

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 04](04_build_gate_frontend_hardening.md) · [Stage 06 →](06_live_latency_telemetry.md)

**Status:** ◐ code complete and Linux-verified 2026-11-19 (GCC + Clang, with
and without ORT, headless GUI smoke incl. a fixture-bundle load); Windows
MSVC / macOS AppleClang NOT RUN; live microphone conversion NOT RUN (needs
audio hardware and a trained bundle) — see [evidence](#evidence-2026-11-19)
**Depends on:** [Stage 04](04_build_gate_frontend_hardening.md) (green build gate),
[M2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) (`ModeAWorker`, `LatencyGovernor`, warm-up),
[M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) (`ParamChannel`, `InferencePipeline::set_params`, `VoiceOutputChain`)
**Owner persona:** `@ui-telemetry` with `@onnx-ort-runtime` for the shared host

## Objective

Make the GUI run the same Mode A pipeline as `fluxvoice --vc`. Today the GUI
worker is a pass-through: the sliders reach the worker, and the worker applies
gain, limiter and mute only. After this stage the GUI loads a model bundle,
warms it up, converts live speech and applies every `VoiceParams` control, with
one composition shared by the CLI and the GUI.

## Context & verified starting point (2026-11-14)

- `frontend/core/EngineHost.cpp` builds the legacy `InferencePipeline()` shell
  and an 80 ms `ChunkAssembler` (`PipelineTiming::kChunkSamples`). The hop is
  copied through `frontend/core/OutputChain.hpp` unchanged
  (`EngineHost.cpp:136-142`). Only `Gain` and `Limiter` are read from a
  parameter snapshot (`EngineHost.cpp:113-114`).
- `frontend/CMakeLists.txt` links neither ONNX Runtime nor
  `ModelBundle.cpp` / `OrtModel.cpp`. `fluxvoice_gui` cannot load a model.
- The complete Mode A composition lives inline in `src/main.cpp:477-700`:
  bundle load, voice table, `PipelineOptions`, `warm_up(20)`, queue-capacity
  check, `ModeAWorker` + `ParamChannel`, the `set_prepare_hook` rate check,
  `fade_underruns`, `max_queued_samples = 0`, and `WorkerPriority` on the
  worker thread. None of it is reusable from the GUI.
- The GUI keeps a second parameter queue (`ReaderWriterQueue<ParamMessage>` in
  `EngineHost.hpp:92`) and a second output chain. `src/voice/ParamChannel.hpp`
  and `src/voice/VoiceOutputChain.hpp` are the engine versions.
- `ControllerConfig::sample_rate` is fixed when `EngineHost` is constructed.
  A bundle's contract rate is only known after the bundle loads.

## Tasks

- [x] **Extract a shared host** `src/pipeline/ModeAHost.{hpp,cpp}`:
  - [x] `load(model_dir, frames_per_step, error)`: `ModelBundle::load`, voice
        table → `PipelineVoice`, `PipelineOptions` (latency target, initial
        params), `InferencePipeline`, `warm_up(20)`, queue-capacity check.
        Returns a typed error; never throws across the boundary.
  - [x] `attach(AudioController&)`: creates `ModeAWorker` + `ParamChannel`,
        installs the prepare hook (client rate equals contract rate, then
        `ModeAWorker::prepare()`), and the worker hooks (quiesce, reset,
        resume) with `WorkerPriority` taken on the worker thread.
  - [x] `push_params`, `flush`, `applied_generation`, `snapshot()`
        (`WorkerSnapshot`), `timing()`, `failed()`, `error()`.
  - [x] `src/main.cpp` uses `ModeAHost` for `--vc`. No behaviour change:
        `check_vc_cli.cmake`, `check_voice_cli.cmake`, `test_mode_a_worker`
        and the golden `fv_vc_file` tests stay green.
- [x] **Build wiring** (`frontend/CMakeLists.txt`, root `CMakeLists.txt`):
  - [x] `fluxvoice_frontend_core` compiles `ModelBundle.cpp` and `OrtModel.cpp`.
  - [x] `fluxvoice_link_ort(fluxvoice_gui)` and `test_frontend_model` when
        `FLUXVOICE_HAS_ORT`; `avrt` on Windows for `WorkerPriority`.
  - [x] The ORT-less GUI build still links and runs. Starting Mode A then
        fails on the control thread with `kBuiltWithoutOrtMessage`.
- [x] **`EngineHost` becomes a two-mode host:**
  - [x] *Bypass* (no bundle loaded): today's pass-through, labelled "Bypass —
        no model loaded" in Live, the tray tooltip and the status bar. The UI
        must not show the sliders as "applied" in this mode.
  - [x] *Mode A*: `ModeAHost` drives the worker. `fade_underruns = true`,
        `max_queued_samples = 0` (the governor owns intentional skips).
  - [x] Hop size comes from the bundle geometry, not from `PipelineTiming`.
  - [x] The controller's sample rate follows the bundle's contract rate.
        Rebuild or reconfigure the controller only while the stream is Idle.
- [x] **Bundle lifecycle on the control thread:**
  - [x] Loading and warm-up take from hundreds of milliseconds to seconds, so
        they run on a short-lived loader thread while the stream is stopped.
        The result is queued and drained in `AppModel::tick` (same pattern as
        the native file dialogs).
  - [x] Switching bundles while Running: stop → load → warm up → restart with
        the same device policy. A failed load leaves the previous bundle
        active and reports the error. It never falls back to Bypass silently.
  - [x] Settings: `active_bundle`, `frames_per_step` (s1/s2/s4, default from
        the bundle's gate variant), `latency_target_ms` (0 = master-budget
        default).
- [x] **Parameters:**
  - [x] Replace the GUI's private queue with `ParamChannel`. `AppModel::tick`
        calls `flush()` each tick.
  - [x] Voice A/B names map to bundle voice indices on the control thread
        (`find_voice`). `voice_b` follows `voice_a` until chosen explicitly.
        An unknown voice is rejected with a message; the previous voice stays.
  - [x] Delete `frontend/core/OutputChain.hpp`. Mute, gain, gate, brightness
        and the limiter run in `VoiceOutputChain` inside the pipeline. Bypass
        mode uses the same class.
  - [x] `translate = 1` is rejected on the control thread until Stage 07
        ("bundle has no translator models").
  - [x] "Applied" means *accepted*. Today `ModeAWorker::pump` discards the
        result of `set_params()` after `ParamChannel::drain` has advanced
        `applied_generation` (`ModeAWorker.hpp:74-75`). Publish a separate
        accepted generation (or a rejected counter) and drive the UI
        indicator from it. Fixes
        [BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md)
        together with the `ParamChannel` switch.
- [x] **Failure surfacing:** when `ModeAWorker::failed()` becomes true, the
      host stops the stream, and Live shows the pipeline's `error()` text with
      "Retry" and "Choose another bundle". No automatic restart loop.
- [x] **Tests (`tests/test_frontend_model.cpp`, `tests/test_mode_a_host.cpp`,
      both ORT-free over `FakeModelStep` + `FakeAudioBackend`):**
  - [x] Start in Mode A: hops reach the fake models, and the conservation
        identity holds with `prefilled` and `skipped` included.
  - [x] A parameter push reaches `InferencePipeline::params()` within one hop;
        `applied_generation` equals `pushed_generation` afterwards.
  - [x] Voice switch by name maps to the right index; unknown name is rejected.
  - [x] Bundle swap while Running quiesces the worker before the old models
        are destroyed (use-after-free guard: the fake counts calls after
        release).
  - [x] Device loss and recovery keep the bundle loaded and re-run `prepare()`.
  - [x] A fatal pipeline error surfaces as a UI message and a stopped stream.
  - [x] `pump()` allocates 0 bytes through the host (`AllocationCounter`).

## Implementation notes

- **One composition.** The reason to extract `ModeAHost` rather than copy the
  CLI code is the list of ordering rules that are easy to get wrong: warm-up
  before `engine.start()`, prepare after the queues drain and before callbacks
  start, the stats snapshot before the worker stops, and SPSC roles changing
  only while quiesced.
- **Lifetime.** `InferencePipeline` holds references to the bundle's
  `ModelStep`s. `ModeAHost` owns the bundle, the pipeline and the worker in
  that order, so destruction runs worker → pipeline → bundle.
- **The UI thread is the control thread.** A blocking bundle load on it would
  stall `AudioController::tick` and freeze the window. That is why loading
  uses a loader thread and only while no stream is open.
- **No row indices for voices either.** The Voices and Catalogue panels submit
  a voice *name*; the host resolves it against the loaded bundle.

## Files touched

- `src/pipeline/ModeAHost.{hpp,cpp}` (new), `src/main.cpp`,
  `frontend/core/EngineHost.{hpp,cpp}`, `frontend/core/AppModel.{hpp,cpp}`,
  `frontend/core/Settings.{hpp,cpp}`, `frontend/core/OutputChain.hpp` (deleted),
  `frontend/imgui/ViewLive.cpp`, `ViewVoices.cpp`, `ViewSettings.cpp`,
  `frontend/CMakeLists.txt`, `CMakeLists.txt`, `tests/test_mode_a_host.cpp` (new),
  `tests/test_frontend_model.cpp`, `frontend/README.md` (§1, §8, §11)

## Definition of done

- With an exported bundle, `fluxvoice_gui` converts live microphone speech to
  the selected voice. Pitch shift, intonation, timbre blend, gain, gate and
  the limiter act on GUI audio and pass their M2 Stage 09 DoD checks.
- `src/main.cpp` and `EngineHost` contain no duplicated worker loop; both use
  `ModeAHost`.
- All new tests pass in the ORT-less build. The ORT build passes
  `test_ort_model` plus a GUI smoke run that loads the fixture bundle.
- Warnings-as-errors build and CTest pass on the four native legs, recorded
  as PASS / FAIL / NOT RUN per leg.

## As implemented (2026-11-19)

- **One host for both routes.** `ModeAHost` loads either `ModelBundle` or
  `S2utBundle` (`TranslatorPolicy` kNever / kIfPresent / kRequired), including
  the CLI's "translator unusable → Mode A" fallback and the "< 2 languages"
  note, because `fluxvoice --vc` already loads translator models through the
  same composition. The GUI requests `kNever` until Stage 07.
- **Model seam.** `HostModelLoader` / `HostModels`: `ort_bundle_loader()`
  (`src/pipeline/OrtBundleLoader.cpp`) in the shipped binaries, fakes in the
  tests. `ModeAHost.cpp` has no ORT dependency, so `test_mode_a_host` links
  without it.
- **Accepted vs delivered.** `ModeAWorker` publishes the accepted generation
  and a rejected count (`set_params()` result); the Voices pill compares
  accepted with pushed; Bypass never shows "applied".
- **Mute** is a host flag, not a `kParams` field: `VoiceOutputChain::set_muted`
  ramps over 5 ms inside the pipeline (Mode A and Mode B routes); the GUI's
  Bypass runs the same class on 20 ms blocks.
- **GUI lifecycle.** `LoaderThread` (one job, result drained in
  `AppModel::tick`); a failed load keeps the previous host attached; the
  stream restarts with the remembered device policy; `--bundle <dir>` loads at
  start and prints `[bundle] loaded …` / `[bundle] error: …`.
- Header additions beyond the first draft: `voice_index`,
  `HostBundleInfo::translator_unavailable`, `onset_latency_samples` (CLI
  banner).

## Evidence (2026-11-19)

Same machine and base as [Stage 04](04_build_gate_frontend_hardening.md#evidence-2026-11-19);
`cmake --workflow --preset <p>` (all targets, strict warnings as errors).

| Leg | Result |
|---|---|
| Linux GCC 13.3.0 / Clang 18.1.3, ORT-less | PASS, 122/122 each (incl. `mode_a_host`, `cli_vc_without_ort`) |
| Linux GCC 13.3.0 / Clang 18.1.3, ORT 1.31.0 | PASS, 124/124 each (incl. `gui_smoke_bundle`: the GUI loads and warms up `tests/data/fixtures/identity`) |
| `fv_vc_file` goldens (`ctest --preset linux-gcc-ort-training -R "stage07_golden\|stage09_controls"`) | PASS, 3/3 |
| CLI output parity: old vs new `fluxvoice` (ORT) on ~20 non-live paths — list/param/lang dumps, bad voice/rate/fps, queue-capacity error, `--vc`/`--s2ut` on identity, translator and mismatch fixtures (rc-2 fallback) | identical (only `argv[0]` in `--help` differs) |
| ORT-less GUI with `--bundle` | prints `[bundle] error: built without ONNX Runtime …`, exit 0 |
| Windows MSVC, macOS AppleClang | NOT RUN (no toolchain; tree not pushed) |
| Live microphone conversion with a trained bundle; M2 Stage 09 DoD on GUI audio | NOT RUN (audio hardware; no trained bundle in `models/`) |

Tests that can fail (rule broken → failing check; each restored):

| Rule broken | Failing test |
|---|---|
| `set_params` result discarded again | `mode_a_host`, `mode_a_worker` (accepted / rejected) |
| quiesce hook skipped before models are destroyed | `mode_a_host` (calls after release > 0) |
| `prepare()` only on the first start | `mode_a_host` (prefill after recovery) |
| prefill dropped from the identity | `mode_a_host`, `mode_a_worker` |
| contract rate not applied in `configure` | `mode_a_host` |
| controller config not applied by `install_host` | `frontend_model` |
| indicator driven by delivered instead of accepted | `frontend_model` |
| failed load falls back to Bypass / no restart after it | `frontend_model` |
| `failed()` does not stop the stream | `frontend_model` |
| unknown voice name accepted; voice B always follows A | `frontend_model` |
| Mode A queued figure without `prefilled` / `skipped` | `frontend_model` |
| instant mute instead of the ramp | `voice_params` (monotonic ramp, click bound) |
