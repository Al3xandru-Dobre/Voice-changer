---
paths:
  - "src/audio/**"
  - "src/dsp/**"
  - "src/util/**"
  - "src/main.cpp"
  - "tests/test_duplex_router.cpp"
  - "tests/test_chunk_assembler.cpp"
  - "tests/test_overlap_add.cpp"
  - "tests/test_route_guard.cpp"
  - "tests/test_audio_controller.cpp"
  - "tests/test_device_registry.cpp"
  - "tests/test_audio_types.cpp"
  - "tests/test_pitch_tracker.cpp"
---

# Real-time C++ audio & DSP rules (`@rt-audio-dsp`)

Scope: `AudioEngine`, `DuplexRouter`, `ChunkAssembler`, `OverlapAddProcessor`, `RouteGuard`, `AudioDeviceRegistry` / `AudioController` (M1.5), C++ causal YIN + whitening.
Rationale, by section: [`Documents/BEST_PRACTICES.md`](../../Documents/BEST_PRACTICES.md) §2 (device identity, routing, liveness, recovery), §3 (threading, queues, latency accounting), §5.1 (MSVC hygiene). Equations: [`MEMORY.md`](../../MEMORY.md) §2.

1. **Real-Time Callback Law:** Code inside `AudioEngine::data_callback` and `DuplexRouter::process()` must never allocate heap memory, lock a mutex, write to `stdout`/`stderr`, or block. Use `moodycamel::ReaderWriterQueue<float>` (strictly SPSC) and `std::atomic` with `static_assert(std::atomic<T>::is_always_lock_free)`.
2. **Variable Callback Sizes:** Never assume `frame_count` matches `periodSizeInMilliseconds`. Always regroup samples by count via `ChunkAssembler`.
3. **Exact Queue Conservation:** Preserve the per-stream single-duplex identity `queued = prefilled + output_underruns - input_overruns - output_drops - skipped` (M1 diagnostics have `prefilled = skipped = 0`). Rebaseline after queue discards; independently clocked native loopback callbacks are not single duplex. Underruns permanently add delay until output is discarded; `out_q_` occupancy cannot show hardware clock drift. Take shutdown snapshots *before* stopping the worker thread.
4. **Miniaudio & Platform Hygiene:** Keep `#define MINIAUDIO_IMPLEMENTATION` in a single TU per binary (`src/audio/MiniaudioBackend.cpp` for `fluxvoice`, `tools/fv_probe.cpp` for `fv_probe`) and never leak `<miniaudio.h>` outside `src/audio/` (use `src/audio/AudioTypes.hpp`). Copy `ma_device_id` and device names immediately into `DeviceSnapshot` / `EndpointKey{direction, epoch, token}` (never retain `ma_device_info*`). Log both `client_sample_rate` and `capture/playback_internal_rate`. Keep `NOMINMAX` on Windows and suppress MSVC `C4324` around cache-line aligned SPSC queues.
5. **Device Selection Safety (M1.5):** Never silently replace an explicitly selected microphone or virtual output with a default speaker/mic on hot-plug or failure.
6. **Feedback route guard:** Check `is_feedback_route(capture_name, playback_name)` (`src/audio/RouteGuard.hpp`) on the **opened** `StreamInfo` names before `engine.start()`, and never fall back from a missing virtual cable to physical speakers.
7. **Controller liveness, recovery and selection state:** before changing them, read BEST_PRACTICES §2.5 (liveness needs *moving* callback progress, BUG-010), §2.6 (one failure classification point, BUG-011/012) and §2.7 (an unresolved explicit selection is a state, never a substitute value, BUG-020).
