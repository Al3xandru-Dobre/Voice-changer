# Stage 04 — Overlap-Add DSP processor

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 03](03_spsc_buffering_worker.md) · [Stage 05 →](05_virtual_cable_loopback.md)

**Status:** ◐ skeleton landed
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
- [ ] Unit tests (plain asserts or a minimal test file — see notes):
  - [ ] `fade_in[i] + fade_out[i] == 1.0 ± ε` for all i (constant amplitude)
  - [ ] DC / constant-signal input → identical output (splice is transparent)
  - [ ] Continuous sine input through chunked `process` → no discontinuity at
        boundaries (max sample-to-sample delta within tolerance of the sine's
        max slope)
  - [ ] `reset()` zeroes the tail
- [ ] (Optional) `tools/ola_sine_test.cpp` writing a WAV of chunked-vs-direct
      sine for ear inspection

## Implementation notes

- `phase = i / overlap_size` (not `overlap_size - 1`) — matches the README
  reference and keeps the last tap slightly < 1, which is intentional: the
  next chunk's fade-in starts exactly where this fade-out ends.
- The processor is **stateful** (`prev_tail_`): one instance per audio stream;
  `reset()` on voice-model swap or stream restart.
- Keep it header-only and dependency-free — it's the most reusable DSP block
  in the project.
- Test harness: simplest is a `tests/` dir with `assert`-based binaries added
  to CMake under `option(FLUXVOICE_BUILD_TESTS)`. Don't pull GoogleTest for
  three asserts; add it later if test volume grows.

## Files touched

- `src/audio/OverlapAdd.hpp`, new `tests/` (optional)

## Definition of done

- Synthetic sine chunked through OLA shows no audible/measureable boundary
  artefact; the three unit tests above pass.
