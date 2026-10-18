# Engineering Best Practices & Mistake Prevention

> Parent: [Documents index](README.md) · [Bug reports](Bug%20Report/00_index.md) · Condensed reference: [`MEMORY.md`](../MEMORY.md)

This document collects hard-won lessons from diagnosed bugs ([`BUG-001`](Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md), [`BUG-002`](Bug%20Report/BUG-002_voicemeeter_remote_write_lock.md)), Milestone 1/1.5 audio-engine discoveries, and Milestone 2 ML/ONNX audits. Every rule below traces to a specific failure mode or architectural hazard encountered in FluxVoice-S2UT.

---

## Contents

1. [External APIs, IPC & OS Integrity Boundaries](#1-external-apis-ipc--os-integrity-boundaries)
2. [Audio Device Identity, Drivers & Routing Safety](#2-audio-device-identity-drivers--routing-safety)
3. [Real-Time C++ Audio Threading, Queues & Latency Accounting](#3-real-time-c-audio-threading-queues--latency-accounting)
4. [Causal ML Pipeline & Stateful ONNX Runtime C++](#4-causal-ml-pipeline--stateful-onnx-runtime-c)
5. [Cross-Platform Build, Testing & Rig Tooling Discipline](#5-cross-platform-build-testing--rig-tooling-discipline)
6. [Quick Pre-Flight Checklist](#6-quick-pre-flight-checklist)

---

## 1. External APIs, IPC & OS Integrity Boundaries

### 1.1 Never trust return code `0` on IPC/driver writes without read-back
- **Past mistake ([`BUG-002`](Bug%20Report/BUG-002_voicemeeter_remote_write_lock.md)):** A scratch script called `VBVMR_SetParameterFloat`, `VBVMR_SetParameterStringA`, and `Command.Load` against an elevated Voicemeeter instance from a non-elevated shell. Every call returned `0` (success), yet zero changes took effect.
- **Root cause:** `VoicemeeterRemote64.dll` communicates via named file mappings (`VBVMRClient{...}`, `VBVMCClientV2{...}`) and named events created with default process DACLs (no explicit security descriptors). When Voicemeeter runs at High integrity (elevated) and the client runs at Medium integrity, the OS allows the client to map the parameter view for **read** (`VBVMR_GetParameterFloat` works) but denies **write** access to the request channel, while the DLL silently returns `0`.
- **Rule:** Treat `0` from an IPC/driver setter as "request queued locally", never "applied on server". Every state-mutating call must poll `IsParametersDirty()` and read back the target parameter (`GetParameterFloat`/`GetParameterStringA`) within a bounded timeout (e.g., 2.0 s), failing loudly if the value does not converge (`set_verified()` in [`tools/vm_route.py`](../tools/vm_route.py)).

### 1.2 Enforce process integrity-level preflights & never spawn mixers elevated
- **Past mistake ([`BUG-002`](Bug%20Report/BUG-002_voicemeeter_remote_write_lock.md)):** Launching `voicemeeterpro_x64.exe` with `Start-Process -Verb RunAs` (or calling `VBVMR_RunVoicemeeter` from an elevated installer script) starts the mixer at High integrity. From a normal Medium shell, `taskkill` is denied (`Access is denied`) and Remote API writes silently drop, leaving the machine with no non-elevated programmatic way to reconfigure or restart the mixer.
- **Rule:**
  1. Before writing to an external process over shared memory/IPC, inspect both processes' Windows Mandatory Integrity Levels (`OpenProcessToken` + `GetTokenInformation(TokenIntegrityLevel)`). Refuse to proceed if integrity RIDs differ (`Medium = 0x2000` vs. `High = 0x3000`) or if the target's token is unreadable.
  2. Never launch long-lived user-space audio daemons or mixers (`VBVMR_RunVoicemeeter`) when the calling script is elevated (`own_il >= HIGH`).

### 1.3 Never validate hardware/routing state with circular read-backs
- **Past mistake ([`BUG-001`](Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)):** A script wrote `Strip[2].B1 = 1` and `Strip[3].B2 = 1` on Voicemeeter Banana, then read back `Strip[2].B1 == 1` and `Strip[3].B2 == 1` and concluded "routing is verified correct; the VAIO audio pipe must be wedged."
- **Root cause:** The read-back used the same wrong index assumption as the write. On Voicemeeter Banana, `Strip[0..2]` are physical hardware inputs 1–3 (`Strip[2]` was unbound hardware input #3), whereas the virtual inputs are `Strip[3]` (`Voicemeeter VAIO`) and `Strip[4]` (`Voicemeeter AUX`). The configuration was persisted accurately, so `Command.Restart` and full app restarts faithfully reloaded the silent route.
- **Rule:**
  1. Derive strip/channel indices programmatically from queried edition metadata (`VBVMR_GetVoicemeeterType`), never from hard-coded constants:
     - **Standard (type 1):** 2 hardware strips (`0..1`), 1 virtual strip (`Strip[2]` = VAIO -> `B1`).
     - **Banana (type 2):** 3 hardware strips (`0..2`), 2 virtual strips (`Strip[3]` = VAIO -> `B1`, `Strip[4]` = AUX -> `B2`).
     - **Potato (type 3):** 5 hardware strips (`0..4`), 3 virtual strips (`Strip[5]` = VAIO -> `B1`, `Strip[6]` = AUX -> `B2`, `Strip[7]` = VAIO3 -> `B3`).
  2. Always pair API read-back verification with an independent, end-to-end **signal check** (e.g., playing a tone into the playback endpoint and checking peak/RMS level on the capture endpoint via `fv_probe tone`, as in [`tools/setup_test_rig.ps1`](../tools/setup_test_rig.ps1)).

### 1.4 Re-authenticate when a server starts after `Login()`
- **Past mistake ([`BUG-002`](Bug%20Report/BUG-002_voicemeeter_remote_write_lock.md)):** Calling `VBVMR_Login()` when Voicemeeter is not yet running, then launching Voicemeeter via `VBVMR_RunVoicemeeter()`, leaves the session disconnected.
- **Rule:** After starting the server process, loop `VBVMR_Logout()` followed by `VBVMR_Login()` until `VBVMR_GetVoicemeeterType()` returns `0` and `VBVMR_IsParametersDirty() >= 0` (`wait_ready()` in [`tools/vm_route.py`](../tools/vm_route.py)).

---

## 2. Audio Device Identity, Drivers & Routing Safety

### 2.1 Never rely on device enumeration indices across driver changes or sessions
- **Past mistake ([`BUG-001`](Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)):** Installing the VB-CABLE driver while an audio session was active shifted the WASAPI enumeration indices of all subsequent endpoints. `fv_probe` runs using pre-install integer indices played into and recorded from the wrong devices.
- **Rule:**
  - In CLI scripts and test harnesses, always select endpoints by **name substring** (`fv_probe --input-name "..." --output-name "..."`), never by remembered integer index.
  - In application code ([Milestone 1.5](Milestone%201.5/00_master.md)), bind selections to `AudioDeviceRegistry` endpoint keys paired with a **snapshot generation token** so a catalog reorder can never silently redirect an explicit microphone or virtual cable selection.
  - Never install or uninstall audio drivers while `fluxvoice` or Voicemeeter is running; always re-run [`tools/setup_test_rig.ps1`](../tools/setup_test_rig.ps1) (or `--list-devices`) after any device-graph change.

### 2.2 Check feedback routes on the *resolved* opened endpoints before `start()`
- **Hazard ([Milestone 1 / Stage 05](Milestone%201/05_virtual_cable_loopback.md)):** If the user omits `--input` and the OS default recording device happens to be `CABLE Output` while `--output` targets `CABLE Input`, checking only explicit CLI arguments misses the feedback loop.
- **Rule:** Separate `AudioEngine::open()` from `AudioEngine::start()`. Inspect the resolved device names in `engine.stream_info()` with `is_feedback_route(capture_name, playback_name)` ([`src/audio/RouteGuard.hpp`](../src/audio/RouteGuard.hpp)) *before* starting the audio callback. Furthermore, if an explicitly selected virtual output disappears, **stop the stream** — never fall back to physical speakers (which would broadcast processed/delayed mic audio into the room).

### 2.3 Always log both client and backend sample rates & period sizes
- **Hazard ([Milestone 1 / Stage 05](Milestone%201/05_virtual_cable_loopback.md)):** Virtual cables typically run at 44.1/48 kHz while FluxVoice requests 16 kHz. Confusing `ma_device.sampleRate` (the client rate seen by the callback) with `ma_device.capture.internalSampleRate` / `playback.internalSampleRate` (the backend rate) is the primary cause of pitch-shifted ("chipmunk") audio and miscalculated buffer delays.
- **Rule:** Always populate and print `StreamInfo` (`client_sample_rate`, `capture_internal_rate`, `playback_internal_rate`, `capture_internal_period`, `playback_internal_period`) on stream startup in both `fluxvoice` and `fv_probe`.

### 2.4 Copy `ma_device_id` and enumeration snapshots immediately
- **Hazard ([Cross-platform audio research §5](Research_documents/CROSS_PLATFORM_AUDIO_IO_RESEARCH.md)):** `ma_context_get_devices()` returns pointers to internal context buffers that are invalidated on the next enumeration call or context teardown. Also, `ma_context_init()` includes the `null` backend in its default fallback list.
- **Rule:**
  - Copy `ma_device_id` and device names into application-owned structs (`EndpointDescriptor` / `DeviceSnapshot` in [`src/audio/AudioTypes.hpp`](../src/audio/AudioTypes.hpp)) immediately inside a single combined `ma_context_get_devices(&ctx_, &playback, &n_playback, &capture, &n_capture)` call; never store `ma_device_info*` or expose `<miniaudio.h>` outside `src/audio/`.
  - Mint opaque `EndpointKey{direction, epoch, token}` values scoped to a `ContextEpoch` so keys from a dead context can never alias keys from a recreated one.
  - Serialize all enumeration, capability queries, and open/close operations on a single control thread.
  - Distinguish **empty device list**, **enumeration failure** (`AudioErrorCategory::EnumerationFailed`), **null backend** (`AudioErrorCategory::NullBackendOnly`), and **unprobed capabilities** (`Capability::Unknown` vs. `Capability::Unsupported`).
  - **Every** `refresh()` failure exit — `Unavailable`, `NullOnly`, enumerate-fail alike — must retain the previous snapshot marked `stale` ([`BUG-012`](Bug%20Report/BUG-012_unavailable_refresh_left_snapshot_fresh.md)); do not hand-roll the marking per branch.

### 2.5 Liveness = state + notifications + backend state + *moving* callback progress
- **Past mistake ([`BUG-010`](Bug%20Report/BUG-010_liveness_ignores_callback_progress.md)):** `is_stream_live()` checked the controller state, the pending-notification bits, and `ma_device_get_state == started` — but never the callback progress counter it was already publishing. A backend that still reports "started" while its real-time thread is dead (driver stall, USB suspend, backend-internal failure miniaudio cannot observe) looked like a live stream: no recovery, silence out.
- **Rule:**
  - While `Running`, `tick()` must keep watching `router_.frames_processed()`: a counter frozen longer than `callback_stall_timeout` is an interruption and takes the bounded-recovery path.
  - Never use "samples are non-zero" as a liveness signal — silence is a valid stream; the *counter moving* is what proves the RT thread is alive.
  - Re-arm the progress baseline on every successful open so retries cannot inherit a stale watermark.

### 2.6 One failure classification for every path that can fail before or during a stream
- **Past mistake ([`BUG-011`](Bug%20Report/BUG-011_preopen_retryable_failure_bypasses_recovery.md)):** The pre-open catalog-refresh sites classified failures with their own `is_non_retryable ? AwaitingUser : Error` ternary, so `DeviceBusy`/`EnumerationFailed` at open died instantly — while the *same* categories raised one call later inside `open_stream` correctly entered `Recovering`. The retry policy existed but was unreachable for that failure class, and `Recovering` could not self-heal a generation-0 catalog anyway (`resolve` failed instantly with `DeviceChanged`).
- **Rule:**
  - Route **all** "could not open" failures — catalog refresh, resolve, feedback refusal, `open_stream`, `start_stream` — through one classification point (today `enter_failure_state` / the `fail()` helper inside `open_resolved_internal`).
  - When a retry depends on state the failure destroyed (the catalog), the recovery path must rebuild that state inside the retry attempt — otherwise the retry is guaranteed to fail and the attempt count is wasted.

---

## 3. Real-Time C++ Audio Threading, Queues & Latency Accounting

### 3.1 Strict real-time audio callback discipline
- **Hazard ([Milestone 1](Milestone%201/00_master.md), `RESEARCH_GUIDE.md` §4):** Priority inversion or unbounded execution time inside `AudioEngine::data_callback` causes immediate OS audio dropouts.
- **Rule:** Code reachable from `data_callback` / `DuplexRouter::process()` ([`src/audio/DuplexRouter.hpp`](../src/audio/DuplexRouter.hpp)) must obey:
  - Zero heap allocations (`new`, `malloc`, `std::vector::push_back`, `std::string` construction).
  - Zero locks (`std::mutex`, `std::lock_guard`, OS locks).
  - Zero I/O or formatting (`std::cout`, `std::cerr`, `printf`, file writes).
  - Zero device lifecycle or ONNX inference calls.
  - Inter-thread data transfer uses only preallocated `moodycamel::ReaderWriterQueue<float>` (strictly single-producer, single-consumer per queue) and lock-free atomics verified at compile time with `static_assert(std::atomic<T>::is_always_lock_free)`.

### 3.2 Never assume callback `frame_count` equals the requested period
- **Hazard ([Milestone 1 / Stage 03](Milestone%201/03_spsc_buffering_worker.md)):** `cfg.periodSizeInMilliseconds` is only a hint to miniaudio and the OS audio server. WASAPI, Core Audio, PipeWire, and ALSA may deliver arbitrary or varying `frame_count` values per callback.
- **Rule:** Never size model steps by counting callbacks. Always accumulate samples through `ChunkAssembler` ([`src/audio/ChunkAssembler.hpp`](../src/audio/ChunkAssembler.hpp)) until `ready()` reports a complete `[context | hop]` frame.

### 3.3 Underruns are permanent latency in a single duplex stream
- **Discovery ([Milestone 1 / Stage 06](Milestone%201/06_latency_verification.md)):** In a single miniaudio duplex stream, every callback enqueues `n` captured samples and dequeues `n` playback samples. By conservation of samples:
  ```text
  queued = output_underruns - input_overruns - output_drops
  ```
  Because the callback zero-fills missing samples on an underrun (`output_underruns += missing`) and the worker later pushes the delayed samples anyway, **every underrun permanently increases `queued` (and thus end-to-end latency) for the rest of the session** unless samples are explicitly dropped (`output_drops`).
- **Rule:**
  1. **Warm up all ONNX sessions before `engine.start()`** so first-run kernel/arena allocations do not trigger a startup underrun spike.
  2. Implement a `LatencyGovernor` ([Milestone 2 / Stage 08](Milestone%202/08_realtime_mode_a_pipeline.md)) that drains excess `queued` samples during silent frames (incrementing `output_drops` so the conservation equation stays exact).
  3. Ramp out/in over ~2 ms on underrun edges (using a few scalar floats in `DuplexRouter`) to prevent step discontinuities (clicks).

### 3.4 `out_q_` occupancy cannot measure hardware clock drift
- **Past mistake ([Milestone 1 / Stage 06](Milestone%201/06_latency_verification.md)):** Early plans proposed monitoring `out_q_` occupancy during a soak test to measure clock drift between a USB microphone and a virtual cable.
- **Root cause:** Because `queued = output_underruns - input_overruns - output_drops` is invariant between glitch events on a single miniaudio duplex device, hardware clock drift between capture and playback endpoints is absorbed **inside miniaudio's internal duplex ring buffer or the OS audio server** (e.g., WASAPI buffers), completely invisible to `out_q_`.
- **Rule:** Measure clock drift end-to-end across the audio route using `fv_probe latency` over a ≥ 30-minute run (`--count 1800 --interval-ms 1000 --csv drift.csv`), inspecting the linear regression drift slope (`ms/min`, `ppm`) and watching for periodic resync sawtooths.

### 3.5 Snapshot glitch counters before worker teardown
- **Past mistake ([`src/main.cpp`](../src/main.cpp)):** Reading `engine.output_underruns()` *after* joining the worker thread and stopping the engine includes underruns from audio callbacks that fired while the worker was already stopped.
- **Rule:** Capture the final `Snapshot` (`frames`, `overruns`, `underruns`, `drops`, `queued`, `rss`) **before** setting `worker_run = false` and calling `engine.stop()`.

---

## 4. Causal ML Pipeline & Stateful ONNX Runtime C++

### 4.1 Audit every ML/DSP component for strict causality before adoption
- **Past mistake ([Milestone 2 Research §2–§4](Research_documents/M2_TEXTLESS_VC_RESEARCH.md)):** The initial architecture assumed Kyutai Mimi (12.5 Hz = 80 ms frame) as the default Mode A encoder and SwiftF0 or RMVPE as the pitch tracker. Research audit revealed:
  - **SwiftF0** requires **176 ms of future lookahead**; **RMVPE** and **FCPE** are non-causal.
  - **Mimi's** 80 ms frame alone consumes >75% of the 60–100 ms Mode A latency budget, and published VC benchmarks show its first semantic codebook (`q0`) alone yields >100% WER because linguistic content leaks into higher RVQ codebooks.
- **Rule:**
  - Verify lookahead frames (`L`) and frame rate (`Hz`) from primary sources/code before scheduling any model. Every lookahead frame adds `L * frame_ms` directly to algorithmic latency.
  - Use **causal YIN** on a past-sample window for F0/voicing/energy extraction.
  - Gate content encoders via an objective latency-first decision gate ([Milestone 2 / Stage 05](Milestone%202/05_causal_vocoder_decision_gate.md)), pitting a 50 Hz (20 ms frame) StreamVC-style causal student (`L in {0, 1, 2}`) against a time-boxed Mimi `q0` kill test.
  - Ensure all vocoder convolutions use causal left-padding `(k - 1) * d` and transposed convolutions trim right excess (`trim_right_ratio = 1.0`), making stateful chunked output identical to offline output without requiring OLA lookahead.

### 4.2 Enforce Python-to-C++ DSP parity with golden vectors
- **Hazard ([Milestone 2 / Stage 03](Milestone%202/03_causal_pitch_energy.md)):** Subtle differences between Python training feature extraction and C++ runtime feature extraction (high-pass pre-filter, YIN difference function, EMA whitening priors/floors, or target-domain log-F0 mapping) degrade vocoder quality and cause pitch drift.
- **Rule:** Implement the exact same algorithm in Python and C++ and pin both against committed golden input/output test vectors in CTest. Map source pitch into the target speaker's domain via:
  ```text
  logf0_tgt = mu_tgt + scale * sigma_tgt * z + shift_st * ln(2) / 12
  ```
  so the source speaker's absolute F0 range never leaks into the vocoder.

### 4.3 Apply voice controls to model conditioning, never dry/wet audio mixing
- **Hazard ([Milestone 2 Research §11](Research_documents/M2_TEXTLESS_VC_RESEARCH.md)):** Mixing dry input audio with wet converted audio that is 60–100 ms delayed creates severe comb filtering and exposes the user's real voice.
- **Rule:** Exclude dry/wet mixing from the VC pipeline (use a delay-aligned monitor mix only in M4 if needed). Drive pitch shift, intonation scale, whisper, and timbre blending directly through the vocoder's conditioning inputs (`logf0_tgt`, voicing, FiLM speaker embedding `e_spk`), which costs zero extra latency.

### 4.4 Avoid stateful ONNX export traps
- **Hazards seen in streaming exports ([Milestone 2 Research §8](Research_documents/M2_TEXTLESS_VC_RESEARCH.md)):**
  1. **Frozen Python scalars:** Using a Python `int` for a KV-cache offset or step counter freezes that integer as a constant during `torch.onnx.export`, producing NaNs or garbled audio after step 1.
  2. **Ring-buffer wrap drift:** Streaming attention KV caches that wrap around a sliding window (e.g., 10 s / 250 frames) can silently diverge from PyTorch right at the wrap boundary if masks or positions are mishandled.
  3. **Dynamic shape overhead:** Dynamic sequence/batch axes prevent ORT memory-pattern planning and can trigger per-hop allocations.
- **Rule:**
  - Expose every recurrent tensor (conv left-context, KV cache, position counters) as an explicit named graph input/output tensor pair (`state_in_*` / `state_out_*`).
  - Run chunked ONNX Runtime vs. offline PyTorch parity tests over durations **> 2x the longest attention window** (e.g., >= 25 s for a 10 s window in M2; >= 60 s / 6x window for M3's 128-chunk ring-KV translator).
  - Export fixed-shape graphs per hop variant (`s1`, `s2`, `s4`) and embed the `ModelContract` JSON inside `ModelProto.metadata_props`.

### 4.6 Simultaneous S2UT (Mode B) training & streaming pitfalls ([Milestone 3](Milestone%203/00_master.md))
- **Hazards ([M3 Research §2–§8](Research_documents/M3_WAITK_S2UT_RESEARCH.md)):**
  1. **Single-pass unit-only S2UT collapse:** Direct S2UT trained only with unit cross-entropy collapses on high word-order reordering (`De->En` drops to 2.46 ASR-BLEU on CVSS-C).
  2. **Hard source unit quantization loss:** Feeding discrete source unit IDs into the translator discards cross-lingual phonetic detail.
  3. **Tail-flush gating (`kFlush` mute bug):** During `kFlush` (> 400 ms after the user stops speaking), the live microphone is silent while the translator emits the end of the translated sentence; gating output on live mic energy clips or mutes the translated tail.
  4. **Wasted compute on `kRead`:** Running the decoder and vocoder during initial Wait-*k* `kRead` buffering steps wastes CPU budget to synthesize silence.
- **Rule:**
  - Feed the Stage 2 translator with Stage 1's continuous pre-quantization `latents` (`[1, frames_per_step, D]` f32), and train with StreamSpeech-style multi-task auxiliary CTC supervision (source ASR CTC + target text S2TT CTC), dropping the text heads at ONNX export so runtime stays 100% textless.
  - Split the exported translator into `translator_enc.onnx` and `translator_dec.onnx` stepping on 80 ms chunks ($C = 4 \times 20\text{ ms}$ frames at 50 Hz) with static ring-buffer KV caches ($W_{\text{src}} = W_{\text{tgt}} = 128$ chunks). On `kRead`, run only Stage 1 + `translator_enc` and skip `translator_dec` and Stage 3 (`vocoder`).
  - Predict target-frame prosody (`f0_z, voicing, aperiodicity, energy_z`) from `translator_dec` and drive the Mode B output noise gate from `WaitKPolicy` active output + predicted target energy rather than live microphone energy.

### 4.5 Configure ONNX Runtime C++ for deterministic real-time execution
- **Hazards ([Milestone 2 Research §9–§10](Research_documents/M2_TEXTLESS_VC_RESEARCH.md)):**
  1. **Default ORT threading:** By default, ORT spawns intra-op threads equal to physical cores with spin-waiting enabled, pegging CPU cores and causing scheduling jitter against the audio thread.
  2. **Windows `System32\onnxruntime.dll` shadowing:** Windows 11 ships an older system `onnxruntime.dll` in `C:\Windows\System32`. If the SDK DLL is not placed beside the executable, the loader silently binds the older system DLL or fails at startup.
  3. **Exceptions on the worker thread:** ORT's C++ API throws `Ort::Exception` on errors.
- **Rule:**
  - Set `intra_op_num_threads = 1`, `inter_op_num_threads = 1`, sequential execution, spin-wait disabled (`session.intra_op.allow_spinning = "0"`), global thread pool with `SetGlobalDenormalAsZero()`, and elevate the worker thread (Windows MMCSS `"Pro Audio"`, Linux `SCHED_FIFO`/nice, macOS `QOS_CLASS_USER_INTERACTIVE`).
  - Preallocate all tensor backing memory in `KVStateCache` and ping-pong two prebuilt `Ort::IoBinding` instances (state A->B on even steps, B->A on odd steps) so steady-state hops perform zero allocations.
  - Copy the pinned `onnxruntime.dll` next to every executable in CMake post-build, log `Ort::GetVersionString()`, and refuse to run on a version mismatch.
  - Guard all `<onnxruntime_cxx_api.h>` includes and calls behind `#ifdef FLUXVOICE_WITH_ORT` so the default ORT-less CI build always compiles and passes `ctest`.

---

## 5. Cross-Platform Build, Testing & Rig Tooling Discipline

### 5.1 Windows/MSVC & cross-compiler port hygiene
- **Past mistakes ([Milestone 1 / Stage 05](Milestone%201/05_virtual_cable_loopback.md)):**
  - Including `<miniaudio.h>` on Windows pulls in `<windows.h>`, whose `min` and `max` macros break `std::min` / `std::max` in subsequent headers.
  - `moodycamel::ReaderWriterQueue` aligns internal structures to cache lines to prevent false sharing, triggering MSVC warning `C4324` (`structure was padded due to alignment specifier`), which breaks `/W4 /WX` builds.
- **Rule:**
  - Keep `NOMINMAX` on the `miniaudio_headers` CMake interface target on Windows.
  - Locally bracket `DuplexRouter` with `#pragma warning(disable : 4324)` under `_MSC_VER`.
  - Keep `#define MINIAUDIO_IMPLEMENTATION` in **exactly one** translation unit per executable (`src/audio/MiniaudioBackend.cpp` for `fluxvoice`, `tools/fv_probe.cpp` for `fv_probe`); never include `<miniaudio.h>` outside the audio implementation layer (`src/audio/`).
  - Maintain zero warnings across MSVC (`/W4 /WX`), GCC, and Clang (`-Wall -Wextra -Wpedantic -Werror`).

### 5.2 Every tool and module must have a hardware-free CTest / `--self-test`
- **Pattern ([`CMakeLists.txt`](../CMakeLists.txt), [`tools/vm_route.py`](../tools/vm_route.py), [`tools/analyze_soak.py`](../tools/analyze_soak.py)):** Bugs in test-rig scripts (`BUG-001` off-by-one strip indices, `BUG-002` ignored dropped writes) wasted hardware debugging time because the scratch script had no offline unit tests.
- **Rule:**
  - Every Python rig/analysis script must use the Python standard library only (unless isolated inside `python_training/`) and implement an offline `--self-test` mode registered in `CMakeLists.txt` via `add_test()`.
  - Use in-memory fakes (such as `FakeRemote(drop=True)` in `tools/vm_route.py`) to explicitly test failure modes (dropped writes, integrity mismatches, late underruns, memory leaks, missing markers).
  - Never commit scratch scripts under `tools/_dl/`; promote verified, self-tested scripts to `tools/` and register them in CTest.

---

## 6. Quick Pre-Flight Checklist

Before marking any audio, rig, DSP, or ML change complete:

- [ ] **No RT violations:** Does `data_callback` / `DuplexRouter::process` remain 100% free of allocations, locks, logging, and blocking calls?
- [ ] **No stale indices:** Do scripts/tests select audio devices by name (`--input-name`/`--output-name`) or snapshot-generation keys rather than raw indices?
- [ ] **Verified IPC writes:** Does any external API/driver configuration script check process integrity levels, read back every write, and run an end-to-end tone/signal check?
- [ ] **Exact latency accounting:** Are ONNX sessions warmed up before `engine.start()`, and is the final stats snapshot taken *before* worker teardown?
- [ ] **Strict causality & parity:** Are all ML/DSP features strictly causal, guarded by `#ifdef FLUXVOICE_WITH_ORT` where appropriate, and tested for parity (> 2x window for sliding caches)?
- [ ] **Green hardware-free test suite:** Does `ctest --test-dir build -C Release --output-on-failure` pass 100% of unit, CLI-rejection, and `--self-test` cases without requiring an audio device?
