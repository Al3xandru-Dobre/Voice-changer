# FluxVoice-S2UT — Project Memory (`MEMORY.md`)

> Condensed, high-signal engineering memory for agents and developers.
> Full post-mortems & rationale: [`Documents/BEST_PRACTICES.md`](Documents/BEST_PRACTICES.md) · Bug index: [`Documents/Bug Report/00_index.md`](Documents/Bug%20Report/00_index.md) · Specialist agents: [`AGENTS.md`](AGENTS.md)

---

## 1. Architecture & Milestone Truths

- **Product:** Ultra-low-latency C++20 textless voice changer (**Mode A**, target **<= 80–100 ms** added `T - B` latency) and simultaneous speech-to-unit translator (**Mode B**, Wait-*k* S2UT, target **~300–500 ms**).
- **Where stage files supersede early README drafts ([M1.5](Documents/Milestone%201.5/00_master.md), [M2](Documents/Milestone%202/00_master.md), [M3](Documents/Milestone%203/00_master.md)):**
  - **Audio isolation (M1.5):** `<miniaudio.h>` is isolated inside `src/audio/MiniaudioBackend.*`; application/CLI/UI code uses `src/audio/AudioTypes.hpp` (`EndpointKey{direction, epoch, token}`, `DeviceSnapshot`, `Result<T>`); `AudioDeviceRegistry` publishes content-addressed snapshot generations (a reorder is not a change), retains failed refreshes as **stale**, reports the null backend as `NullBackendOnly` (offline), probes selected devices on demand only, and resolves `DeviceSelection` policies so an explicit selection is **never** redirected or silently replaced (`DeviceNotFound`/`DeviceChanged` instead).
  - **Encoder (M2/M3):** Mimi's 12.5 Hz (80 ms) `q0` codebook uses >75% of the Mode A latency budget and loses content on its own (WER > 100%). M2 defaults to a **50 Hz (20 ms frame) StreamVC-style causal student** (`L in {0, 1, 2}` distilled from HuBERT km100; upgraded in M3 to emit both discrete `units` for Mode A and continuous `latents` f32 for Mode B).
  - **Pitch tracker (M2):** SwiftF0 (176 ms lookahead), RMVPE, and FCPE are **non-causal**. M2 uses a **causal YIN** tracker on past samples + EMA whitening.
  - **Simultaneous S2UT (M3):** Bidirectional `En<->{Es,Fr,De}` Conformer-Transformer trained with multi-task ASR/S2TT CTC supervision (dropped at export; `De->En` collapses without CTC). Split into `translator_enc.onnx` + `translator_dec.onnx` stepping on 80 ms chunks ($C = 4 \times 20\text{ ms}$ frames) with 128-chunk ring-KV caches. `kRead` skips decoder + vocoder; `translator_dec` predicts target prosody and overrides the output noise gate during `kFlush` so translated tails are never muted when the user stops speaking.
  - **Overlap-Add (OLA):** Default **OFF** for stateful causal HiFi-GAN (left-padded convs + right-trimmed transposed convs already emit continuous phase without lookahead). Kept for stateless fallback models.
  - **Worker priority:** Elevated (`MMCSS "Pro Audio"` on Windows, `SCHED_FIFO`/nice on Linux, `QOS_CLASS_USER_INTERACTIVE` on macOS) with ORT `intra_op_num_threads = 1` — not "Normal" priority.

---

## 2. Core Equations & Invariants

| Invariant | Formula / Rule | Why it matters |
|---|---|---|
| **Single-duplex queue conservation** | `queued = output_underruns - input_overruns - output_drops` | Exact sample count held between capture and playback. **Every underrun adds permanent latency** until drained by `LatencyGovernor` (`output_drops`) during silence. |
| **Clock drift visibility** | `out_q_` occupancy **cannot** show hardware clock drift | Single miniaudio duplex device absorbs drift internally/in OS buffers. Measure drift end-to-end with `fv_probe latency` over >= 30 min (chirp slope/sawtooth). |
| **Added latency (`T - B`)** | `T - B = IO_floor + hop + lookahead + compute_p99 + governor_margin` | Measure `B` (cable baseline) and `T` (through `fluxvoice`) with `fv_probe latency` across two virtual cables. |
| **Target-domain F0 mapping** | `logf0_tgt = mu_tgt + scale * sigma_tgt * z + shift_st * ln(2) / 12` | Maps whitened source pitch `z` to target speaker stats so source F0 never leaks into the vocoder. |

---

## 3. Active Bug Defenses (`BUG-001`, `BUG-002`)

- **[`BUG-001`](Documents/Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md) — Voicemeeter strip layout & index shift:**
  - Virtual inputs come *after* hardware strips and vary by edition:
    - **Standard (type 1):** `Strip[2]` (`Voicemeeter Input` -> `B1`)
    - **Banana (type 2):** `Strip[3]` (`Voicemeeter Input` -> `B1`), `Strip[4]` (`Voicemeeter AUX Input` -> `B2`) — *never `Strip[2..3]`!*
    - **Potato (type 3):** `Strip[5..7]` (`VAIO` -> `B1`, `AUX` -> `B2`, `VAIO3` -> `B3`)
  - Installing/removing audio drivers shifts WASAPI device indices mid-session. Always use `--input-name` / `--output-name` in scripts and re-run `--list-devices` or `tools/setup_test_rig.ps1` after driver changes.
  - Never rely on circular API read-backs alone; confirm routes with an acoustic/signal round-trip (`fv_probe tone`).
- **[`BUG-002`](Documents/Bug%20Report/BUG-002_voicemeeter_remote_write_lock.md) — Voicemeeter Remote API silent write drop:**
  - When Voicemeeter runs **elevated** (High integrity) and a client runs **non-elevated** (Medium integrity), `VBVMR_Login()` and `VBVMR_GetParameter*` succeed, but `VBVMR_SetParameter*` and `Command.Load` **return `0` and silently drop** (shared memory DACL mismatch).
  - **Defense:** Never run Voicemeeter as Administrator. Always configure via `tools/vm_route.py` (enforces `TokenIntegrityLevel` preflight, refuses to launch Voicemeeter from an elevated shell, and verifies every write by read-back).

---

## 4. Top 12 "Never / Always" Engineering Rules

1. **NEVER** allocate, lock a mutex, log (`std::cout`/`stderr`), or block inside `AudioEngine::data_callback` or `DuplexRouter::process()`.
2. **NEVER** assume callback `frame_count` equals `periodSizeInMilliseconds`; always assemble model hops by sample count via `ChunkAssembler`.
3. **NEVER** retain `ma_device_info*` pointers across `ma_context_get_devices()` calls; copy `ma_device_id` and names immediately.
4. **ALWAYS** check `is_feedback_route(capture_name, playback_name)` (`src/audio/RouteGuard.hpp`) on the **opened** `StreamInfo` names before `engine.start()`, and never fall back from a missing virtual cable to physical speakers.
5. **ALWAYS** warm up `Ort::Session` before `engine.start()`, and take the final stats `Snapshot` **before** stopping the worker thread.
6. **NEVER** pass Python `int` cache offsets/positions into `torch.onnx.export` (they freeze as trace constants); expose all state/positions as explicit named input/output tensors and test parity for **> 2x the sliding attention window**.
7. **ALWAYS** configure ORT C++ with `intra_op_num_threads = 1`, `inter_op = 1`, spin-wait off, denormals-as-zero, preallocated `KVStateCache`, and ping-pong `Ort::IoBinding`.
8. **ALWAYS** guard ONNX Runtime C++ code with `#ifdef FLUXVOICE_WITH_ORT`, copy `onnxruntime.dll` next to executables on Windows, and check `Ort::GetVersionString()` to prevent `System32\onnxruntime.dll` shadowing.
9. **ALWAYS** keep `NOMINMAX` on Windows `miniaudio_headers`, suppress MSVC `C4324` on cache-aligned SPSC queues, keep `MINIAUDIO_IMPLEMENTATION` in a single TU per binary, and compile cleanly with `/W4 /WX` / `-Wall -Wextra -Wpedantic -Werror`.
10. **ALWAYS** ship an offline `--self-test` (using stdlib + in-memory fakes) for every Python rig/analysis tool and register it in `CMakeLists.txt` (`ctest`).
11. **ALWAYS** make decision gates and parity checks measure the shipped runtime (ORT stateful timings, not PyTorch eager), use a relative tolerance on a non-silent output, report any exporter fallback, and keep the exit code independent of the console codepage ([`BUG-015`](Documents/Bug%20Report/BUG-015_gate_decides_on_pytorch_clock.md), [`BUG-016`](Documents/Bug%20Report/BUG-016_gate_export_silent_exporter_fallback.md), [`BUG-017`](Documents/Bug%20Report/BUG-017_vocoder_checks_below_noise_floor.md)).
12. **ALWAYS** keep first-party headers self-contained and portable: own the standard include for every name (`<cstddef>`/`<cstdint>`/`<string>`/`<string_view>`/`<algorithm>`/`<utility>`/`<system_error>`), write size/fixed-width types as `std::size_t`/`std::[u]intN_t`, guard OS headers behind `_WIN32`/`__APPLE__`/`__linux__`, pick MSVC-style vs GNU-style warning flags by compiler *frontend* (`CMAKE_CXX_COMPILER_FRONTEND_VARIANT` — clang-cl defines both `__clang__` and `_MSC_VER`), link platform libraries per-OS in CMake (never `#pragma comment(lib)`), and let the ALL-built `fluxvoice_header_checks` target (one TU per header, no PCH/unity) prove it.

---

## 5. Canonical Commands

```powershell
# Configure, build & run hardware-free unit/CLI/self-tests (28+ tests)
cmake -S . -B build -DFLUXVOICE_WARNINGS_AS_ERRORS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

# Windows test-rig bootstrap (checks/installs VB-CABLE + Voicemeeter, routes VAIO, runs tone check)
powershell -ExecutionPolicy Bypass -File tools\setup_test_rig.ps1

# Voicemeeter routing (preflight + verified writes + offline self-test)
python tools\vm_route.py --check
python tools\vm_route.py
python tools\vm_route.py --self-test

# Inspect devices, run loopback & measure route latency (prefer name selection in scripts)
.\build\Release\fluxvoice.exe --list-devices
.\build\Release\fv_probe.exe tone --output-name "Voicemeeter Input" --input-name "Voicemeeter Out B1" --seconds 4 tone.wav
.\build\Release\fv_probe.exe latency --output-name "CABLE Input" --input-name "CABLE Output" --count 60
```
