# Stage 05 — GUI engine host: Mode A inference in `fluxvoice_gui`

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 04](04_build_gate_frontend_hardening.md) · [Stage 06 →](06_live_latency_telemetry.md)

**Status:** ☐ not started
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

- [ ] **Extract a shared host** `src/pipeline/ModeAHost.{hpp,cpp}`:
  - [ ] `load(model_dir, frames_per_step, error)`: `ModelBundle::load`, voice
        table → `PipelineVoice`, `PipelineOptions` (latency target, initial
        params), `InferencePipeline`, `warm_up(20)`, queue-capacity check.
        Returns a typed error; never throws across the boundary.
  - [ ] `attach(AudioController&)`: creates `ModeAWorker` + `ParamChannel`,
        installs the prepare hook (client rate equals contract rate, then
        `ModeAWorker::prepare()`), and the worker hooks (quiesce, reset,
        resume) with `WorkerPriority` taken on the worker thread.
  - [ ] `push_params`, `flush`, `applied_generation`, `snapshot()`
        (`WorkerSnapshot`), `timing()`, `failed()`, `error()`.
  - [ ] `src/main.cpp` uses `ModeAHost` for `--vc`. No behaviour change:
        `check_vc_cli.cmake`, `check_voice_cli.cmake`, `test_mode_a_worker`
        and the golden `fv_vc_file` tests stay green.
- [ ] **Build wiring** (`frontend/CMakeLists.txt`, root `CMakeLists.txt`):
  - [ ] `fluxvoice_frontend_core` compiles `ModelBundle.cpp` and `OrtModel.cpp`.
  - [ ] `fluxvoice_link_ort(fluxvoice_gui)` and `test_frontend_model` when
        `FLUXVOICE_HAS_ORT`; `avrt` on Windows for `WorkerPriority`.
  - [ ] The ORT-less GUI build still links and runs. Starting Mode A then
        fails on the control thread with `kBuiltWithoutOrtMessage`.
- [ ] **`EngineHost` becomes a two-mode host:**
  - [ ] *Bypass* (no bundle loaded): today's pass-through, labelled "Bypass —
        no model loaded" in Live, the tray tooltip and the status bar. The UI
        must not show the sliders as "applied" in this mode.
  - [ ] *Mode A*: `ModeAHost` drives the worker. `fade_underruns = true`,
        `max_queued_samples = 0` (the governor owns intentional skips).
  - [ ] Hop size comes from the bundle geometry, not from `PipelineTiming`.
  - [ ] The controller's sample rate follows the bundle's contract rate.
        Rebuild or reconfigure the controller only while the stream is Idle.
- [ ] **Bundle lifecycle on the control thread:**
  - [ ] Loading and warm-up take from hundreds of milliseconds to seconds, so
        they run on a short-lived loader thread while the stream is stopped.
        The result is queued and drained in `AppModel::tick` (same pattern as
        the native file dialogs).
  - [ ] Switching bundles while Running: stop → load → warm up → restart with
        the same device policy. A failed load leaves the previous bundle
        active and reports the error. It never falls back to Bypass silently.
  - [ ] Settings: `active_bundle`, `frames_per_step` (s1/s2/s4, default from
        the bundle's gate variant), `latency_target_ms` (0 = master-budget
        default).
- [ ] **Parameters:**
  - [ ] Replace the GUI's private queue with `ParamChannel`. `AppModel::tick`
        calls `flush()` each tick.
  - [ ] Voice A/B names map to bundle voice indices on the control thread
        (`find_voice`). `voice_b` follows `voice_a` until chosen explicitly.
        An unknown voice is rejected with a message; the previous voice stays.
  - [ ] Delete `frontend/core/OutputChain.hpp`. Mute, gain, gate, brightness
        and the limiter run in `VoiceOutputChain` inside the pipeline. Bypass
        mode uses the same class.
  - [ ] `translate = 1` is rejected on the control thread until Stage 07
        ("bundle has no translator models").
  - [ ] "Applied" means *accepted*. Today `ModeAWorker::pump` discards the
        result of `set_params()` after `ParamChannel::drain` has advanced
        `applied_generation` (`ModeAWorker.hpp:74-75`). Publish a separate
        accepted generation (or a rejected counter) and drive the UI
        indicator from it. Fixes
        [BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md)
        together with the `ParamChannel` switch.
- [ ] **Failure surfacing:** when `ModeAWorker::failed()` becomes true, the
      host stops the stream, and Live shows the pipeline's `error()` text with
      "Retry" and "Choose another bundle". No automatic restart loop.
- [ ] **Tests (`tests/test_frontend_model.cpp`, `tests/test_mode_a_host.cpp`,
      both ORT-free over `FakeModelStep` + `FakeAudioBackend`):**
  - [ ] Start in Mode A: hops reach the fake models, and the conservation
        identity holds with `prefilled` and `skipped` included.
  - [ ] A parameter push reaches `InferencePipeline::params()` within one hop;
        `applied_generation` equals `pushed_generation` afterwards.
  - [ ] Voice switch by name maps to the right index; unknown name is rejected.
  - [ ] Bundle swap while Running quiesces the worker before the old models
        are destroyed (use-after-free guard: the fake counts calls after
        release).
  - [ ] Device loss and recovery keep the bundle loaded and re-run `prepare()`.
  - [ ] A fatal pipeline error surfaces as a UI message and a stopped stream.
  - [ ] `pump()` allocates 0 bytes through the host (`AllocationCounter`).

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
