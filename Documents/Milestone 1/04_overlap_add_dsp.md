# Stage 04 — Overlap-Add DSP processor

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 03](03_spsc_buffering_worker.md) · [Stage 05 →](05_virtual_cable_loopback.md)

**Status:** ☑ done (2026-10-17)
**Depends on:** Stage 01 (build)

## Objective

Implement and unit-test `OverlapAddProcessor` — the raised-cosine (Hann)
crossfader that splices independent vocoder chunks without clicks. This is
pure DSP: no audio device needed, fully testable offline.

## Context & references

- README §4 — reference implementation this stage canonizes in
  `src/audio/OverlapAdd.hpp`.
- Why it exists: chunked neural vocoders emit slightly phase/amplitude-
  mismatched frames; naive concatenation clicks every 80 ms. Vocoder emits
  `chunk + overlap` (e.g. 80+20 ms = 1600 samples); we crossfade the leading
  overlap against the previous tail.

## Tasks

- [x] `OverlapAddProcessor`: precomputed Hann `fade_in_`/`fade_out_` curves,
      `prev_tail_` state, `process(in[chunk+overlap], out[chunk])`, `reset()`
- [x] Unit tests (`tests/test_overlap_add.cpp`, CTest `overlap_add`):
  - [x] `fade_in[i] + fade_out[i] == 1.0 ± ε` for all i (constant amplitude);
        `fade_in` starts at 0, rises strictly and stays below 1
  - [x] DC / constant-signal input → identical output (splice is transparent)
        once the first chunk's fade-in from silence is over
  - [x] Continuous sine input through chunked `process` → no discontinuity at
        boundaries (max sample-to-sample delta within tolerance of the sine's
        max slope); 50 Hz – 7 kHz, and the output equals the input
  - [x] `reset()` zeroes the tail; afterwards the processor matches a new one
  - [x] Added: frames with a different gain and phase each (what a chunked
        vocoder emits) → every step stays under an analytic bound that
        butt-joining the same frames exceeds by more than 2×
  - [x] Added: `overlap == 0` is a plain copy; `process` writes exactly
        `chunk_size()` samples
- [x] (Optional) `tools/ola_sine_test.cpp` writing a WAV of chunked-vs-direct
      sine for ear inspection: `ola_reference.wav`, `ola_naive.wav`
      (butt-joined frames), `ola_crossfaded.wav`. Built with the tests and run
      by CTest (`ola_sine_wavs`), which writes the WAVs into the build directory

## Implementation notes

- `phase = i / overlap_size` (not `overlap_size - 1`) — matches the README
  reference and keeps the last tap slightly < 1, which is intentional: the
  next chunk's fade-in starts exactly where this fade-out ends.
- The processor is **stateful** (`prev_tail_`): one instance per audio stream;
  `reset()` on voice-model swap or stream restart.
- Timeline: step k's input covers stream samples `[k*chunk, k*chunk + chunk +
  overlap)`. The crossfade adds no delay, and the first chunk after
  construction or `reset()` fades in from silence.
- Requires `overlap_size <= chunk_size` (asserted in the constructor): the
  crossfade is written into the head of the output chunk.
- Boundary bound used by the mismatched-frames test: blending tail `a` into
  head `b` with Hann weight `w` gives
  `Δy = w'·Δb + (1 − w')·Δa + Δw·(b − a)`, and the largest Hann weight step is
  `sin(π / 2N)`. So a step is never larger than the steeper signal's slope plus
  `max|b − a| · sin(π / 2N)` (≈ 0.005·`max|b − a|` at N = 320).
- Keep it header-only and dependency-free — it's the most reusable DSP block
  in the project.
- Test harness: simplest is a `tests/` dir with `assert`-based binaries added
  to CMake under `option(FLUXVOICE_BUILD_TESTS)`. Don't pull GoogleTest for
  three asserts; add it later if test volume grows.

## Verification (2026-10-17)

Windows 11, MSVC 19.50 (VS 18), `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON` (`/W4 /WX`):
`overlap_add`, `chunk_assembler` and `ola_sine_wavs` pass. `test_overlap_add`
also passes as a Debug build (asserts on) under `/fsanitize=address`.
GCC and Clang were not run for this stage.

`ola_sine_test` output (440 Hz, amplitude 0.25, per-frame gain 0.8–1.0 and
phase ±0.4 rad), checked with `tools/analyze_loopback.py`:

| File | Max step | `analyze_loopback.py` |
|---|---|---|
| `ola_reference.wav` | 0.0431 | clean |
| `ola_naive.wav` | 0.1188 | 86 click events |
| `ola_crossfaded.wav` | 0.0431 | clean |

How the tests were checked: each of these changes to `OverlapAdd.hpp` fails
the suite — no crossfade (copy the head), swapped fade curves,
`fade_out = fade_in`, a no-op `reset()`, the tail saved one sample early,
`phase = i / (overlap − 1)`, and writing one sample past the chunk.

Notes:
- At the time, the `fluxvoice` target itself did not build under MSVC
  (`windows.h` `min`/`max` macros broke `std::min`/`std::max` in `main.cpp` and
  `ChunkAssembler.hpp`; C4324 in `DuplexRouter.hpp` under `/WX`), so only the
  targets above were built. Fixed in
  [Stage 05](05_virtual_cable_loopback.md#verification-2026-10-17-windows--partial).
- Nothing calls the processor outside the tests yet. `InferencePipeline` holds
  one for the M2 vocoder path.

## Files touched

- `src/audio/OverlapAdd.hpp` (curve accessors, precondition assert, timeline
  comment), `tests/test_overlap_add.cpp` (new), `tools/ola_sine_test.cpp`
  (new), `CMakeLists.txt`

## Definition of done

- Synthetic sine chunked through OLA shows no audible/measureable boundary
  artefact; the three unit tests above pass.

Met — see Verification. The "no audible artefact" half rests on the analyzer
and the step bound; nobody has listened to the WAVs yet.
