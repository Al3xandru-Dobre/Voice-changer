# Stage 04 — `WaitKPolicy` & streaming unit/prosody FIFO

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 03](03_multilingual_encoder_vocoder_upgrade.md) · [Stage 05 →](05_causal_s2ut_translator_policy_gate.md)

**Status:** ◐ implemented & verified on one leg (2026-11-15) — `WaitKPolicy.hpp`, `UnitProsodyFifo.hpp`, the Python twin, golden trajectories and both test suites are done; Windows/zig-Clang all-target warnings-as-errors build, `ctest -R waitk_policy` and the live 1,000-trajectory C++/Python parity run PASS. Native MSVC, Linux GCC/Clang and macOS AppleClang legs are NOT RUN (pending CI, see Verification)
**Depends on:** Stage 02 (contract v2 chunk geometry & special tokens)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §4.2 80 ms chunks at 50 Hz · §9.1 `WaitKPolicy` enhancements
**Estimated scope:** C++ header + unit tests + Python mirror (no ORT required)

## Objective

Harden and complete `src/pipeline/WaitKPolicy.hpp` (and a bit-exact Python
reference `python_training/fvml/waitk_policy.py`) so it handles every real-world
streaming condition at 50 Hz:
- **50 Hz frame-to-chunk aggregation** (`chunk_frames = 4`, so `k ∈ [1..10]`
  counts 80 ms chunks);
- **bounded tail flush** (`max_flush_steps` + `<EOS>` early termination + automatic
  post-flush `reset()`, fixing the current unbounded `flush_pending()` loop);
- **mid-flush speech resumption** (if the speaker resumes talking while a tail
  flush is still draining, transition cleanly back to `kWrite` without losing
  context);
- **CTC-adaptive readiness hook** (allowing 0, 1, or 2 target chunks per step
  when adaptive mode is enabled);
- a preallocated, lock-free **`UnitProsodyFifo`** that decouples chunk-rate
  translator emissions (`4` frames per step) from the vocoder's hop size (`s1`,
  `s2`, or `s4`).

## Context & references

- `src/pipeline/WaitKPolicy.hpp` (lines 1–67) currently implements:
  - `decide()` returning `kRead`, `kWrite`, or `kFlush`;
  - `notify_source_unit()`, `notify_target_unit()`, `on_chunk_energy(bool,
    uint32_t)`;
  - `flush_pending()`: `silence_ms_ > kTailFlushMs && (t_src_ > 0 || t_tgt_ > 0)`.
- **Bugs/gaps in the current 67-line stub:**
  1. **No flush termination:** during `kFlush`, `t_tgt_` increments on each
     emitted target chunk, so `t_src_ > 0 || t_tgt_ > 0` remains true forever!
     Nothing tells the caller when the tail flush is finished and `reset()`
     should fire.
  2. **Leading silence accumulation:** if the user is silent for 500 ms *before*
     saying the first word, `silence_ms_` is $> 400\text{ ms}$, and as soon as the
     first voiced chunk increments `t_src_ = 1` (if `on_chunk_energy` isn't
     ordered carefully or if a single unvoiced frame follows), `flush_pending()`
     can misfire.
  3. **Fixed 1:1 assumption:** `decide()` assumes one encoder chunk produces at
     most one decoder chunk, with no hook for `<EOS>` or adaptive duration ratios
     ($\gamma \in [0.5, 1.5]$).

## Tasks

- [x] **Upgrade `src/pipeline/WaitKPolicy.hpp`:**
  - [x] Keep the existing public API (`Decision::kRead`, `kWrite`, `kFlush`,
        `kMinK = 1`, `kMaxK = 10`, `kTailFlushMs = 400`, `decide()`, `k()`,
        `set_k()`, `t_src()`, `t_tgt()`, `reset()`) working so existing callers
        compile unchanged.
  - [x] Add `Config` struct:
    - `k` (`1..10`, default `3`)
    - `tail_flush_ms` (default `400`)
    - `max_flush_ratio` (default `1.5f`: flush stops once
      `t_tgt_ >= ceil(max_flush_ratio * t_src_)` or `flush_steps_ >=
      max_extra_flush_steps`)
    - `adaptive_mode` (`bool`, default `false`)
    - `readiness_threshold` (`float`, default `0.5f`)
  - [x] Add `<EOS>` & flush completion handling:
    - `notify_target_unit(bool is_eos = false)`: increments `t_tgt_`; if in
      `kFlush` and (`is_eos` or `t_tgt_` reaches the flush bound), marks the
      utterance drained and calls `reset()`, incrementing `flushes_completed_`.
    - If `is_eos` arrives during `kWrite` while the user is already silent, mark
      caught-up (`t_tgt_ = t_src_`) so no extra flush steps run after `<EOS>`.
  - [x] **Mid-flush speech resumption:** if `on_chunk_energy(true, elapsed_ms)`
        is called while `Decision::kFlush` is active, clear `silence_ms_ = 0` and
        resume `kWrite` / `kRead` from the current `(t_src_, t_tgt_)` lag without
        resetting the KV caches.
  - [x] **Adaptive readiness hook:** `on_source_readiness(float prob)` accumulates
        expected target chunks; in adaptive mode, `decide()` returns `kWrite`
        when `(t_src_ - t_tgt_ >= k_min_) && (readiness_accum_ >= 1.0f ||
        t_src_ - t_tgt_ >= k_max_)`, and `max_writes_this_step()` returns `1` or
        `2` if backlog exceeds target lag.
- [x] **`src/pipeline/UnitProsodyFifo.hpp` (new, header-only):**
  - [x] Fixed-capacity ring buffer (e.g. 64 frames = 1.28 s @ 50 Hz, preallocated
        at construction, zero heap allocations in steady state).
  - [x] Stores `Frame { int64_t unit; float logf0_z, voicing, aperiodicity,
        energy_z; }`.
  - [x] `push_chunk(std::span<const int64_t> units, std::span<const float>
        prosody_4xC)` ( skips trailing `<PAD>`/`<EOS>` frames).
  - [x] `pop_hop(size_t n_frames, int64_t* units_out, PitchFrame* prosody_out) ->
        bool`: returns `true` and fills `n_frames` when at least `n_frames` are
        queued; returns `false` (meaning emit silence / `kRead`) when fewer than
        `n_frames` are available.
- [x] **Python mirror (`python_training/fvml/waitk_policy.py`):**
  - [x] Bit-exact Python port of `WaitKPolicy` and `UnitProsodyFifo` so Stage 05's
        SimulEval runner and Stage 06's parity tests execute the identical policy
        state machine as C++.
- [x] **Unit tests (`tests/test_waitk_policy.cpp`, registered in `CMakeLists.txt`):**
  - [x] **Classic Wait-*k* sequence:** for $k \in \{1, 3, 5, 10\}$, feed 20
        voiced chunks → verify first $k-1$ steps return `kRead` and remaining
        steps return `kWrite`.
  - [x] **Short pause vs. tail flush:** a 300 ms unvoiced gap inside an utterance
        does *not* trigger `kFlush`; a 450 ms unvoiced gap triggers `kFlush`,
        drains remaining chunks up to the bound or `<EOS>`, and auto-resets
        `t_src() == 0 && t_tgt() == 0`.
  - [x] **Early `<EOS>` termination:** `notify_target_unit(/*is_eos=*/true)` stops
        `kFlush` immediately on the next step.
  - [x] **Speech resumption during flush:** speaking again after 1 flush step
        cancels `kFlush` and continues `kWrite` without zeroing `t_src_`.
  - [x] **Hot `set_k()` clamping:** changing $k$ mid-utterance clamps to `[1..10]`
        and transitions cleanly between `kRead` and `kWrite`.
  - [x] **`UnitProsodyFifo` rate bridging:** pushing 4-frame translator chunks and
        popping 1-, 2-, or 4-frame vocoder hops preserves exact frame order and
        allocates 0 bytes (`operator new` counter check).

## As implemented (2026-11-14)

- **Phases.** The policy is an explicit three-phase machine: *idle* (no
  utterance), *active* (`kRead`/`kWrite`) and *flushing* (`kFlush`). Silence is
  only counted while an utterance is active, so leading silence can never arm the
  tail flush (gap 2). The flush starts when a pause exceeds `tail_flush_ms`
  (strictly greater) and ends on `<EOS>`, on the ratio bound
  `t_tgt >= ceil(max_flush_ratio * t_src)` or after `max_flush_steps`
  decoder-only steps, whichever comes first (gap 1); completion calls `reset()`
  and increments `flushes_completed()`, which is the caller's signal to zero the
  translator state.
- **Per-chunk call order** (documented at the top of both files):
  `on_chunk_energy()` → if `source_wanted()`: run `translator_enc`,
  `notify_source_unit()`, `on_source_readiness()` → `decide()` →
  `max_writes_this_step()` × (`translator_dec`, `notify_target_unit(is_eos)`).
  `source_wanted()` is false while idle and while flushing, so the encoder is
  skipped on leading silence and during `kFlush`; stray notifications in those
  phases are ignored.
- **Config additions beyond the task list:** `max_flush_steps` (default `32`
  chunks = 2.56 s, the hard cap the objective calls for),
  `adaptive_extra_wait` (default `2`: adaptive mode holds a write back until
  the lag reaches `k + 2` at most, i.e. `k_min = k`, `k_max = k + 2`),
  `chunk_frames` (`4`) and `frame_ms` (`20`). Out-of-range values are clamped
  (`max_flush_ratio` to `[1, 4]`, NaN → `1`).
- **Frame aggregation.** `on_frame_energy(bool)` regroups 50 Hz frame verdicts
  into chunks (speech if any frame had speech) for `--frames-per-step 1|2`.
- **Adaptive readiness.** `on_source_readiness(x)` takes the expected number of
  target chunks the newest source chunk completed, `x` in `[0, 2]` (larger
  values are clamped). Values below `readiness_threshold` count as blank; at
  most two target chunks are banked; each write spends one. A step writes twice
  only when the lag exceeds `k` and two chunks are banked (or the lag is beyond
  `k_max`), so every write outside a flush happens at a lag of at least `k`.
  The first version clamped `x` to `[0, 1]`; the Stage 05 review showed that a
  step could then only hold a write back and never write twice at a constant
  `k` (0 two-write steps in 4M random steps), so the range was widened on
  2026-11-15. A caller that feeds a plain probability still gets the
  delay-only behaviour: the readiness head has to be mapped to an expected
  count for the catch-up path to be used. Strict mode never writes twice: after
  lowering `k` mid-utterance the extra lag is drained by the next tail flush.
- **`<EOS>` during a pause** marks the utterance caught up; `decide()` then
  returns `kRead` until either speech resumes (the next phrase waits `k` chunks
  afresh, translator state kept) or the tail timer fires (immediate completion,
  zero flush steps). `<EOS>` while the user is still speaking is an ordinary
  target chunk.
- **Backward compatibility.** A caller that never reports energy has no VAD, so
  `notify_source_unit()` opens the utterance itself — the old
  notify/decide-only call pattern behaves as before. `flush_pending()` remains
  as an alias of `flushing()`.
- **`UnitProsodyFifo`.** `push_chunk()` keeps the prefix of regular units and
  drops everything from the first special unit on, reports whether the chunk
  contained `<EOS>`, and is all-or-nothing on overflow (`overflows()` counts
  them, `free_frames()` lets the caller apply back-pressure before an adaptive
  double write). `pop_hop()` has the `PitchFrame` form from the task list
  (`z_logf0`, `voicing`, `aperiodicity`, `z_energy` set) and a raw `Frame`
  form. It is single-threaded by design: both ends run on the inference worker.
- **Parity mechanism.** `python_training/make_waitk_golden.py` writes 256
  seeded random trajectories (both modes, pauses on both sides of the flush
  threshold, `<EOS>` events, hot `set_k`) with the Python twin's decisions to
  `tests/data/waitk/golden_trajectories.txt`. `tests/test_waitk_policy.cpp`
  replays that file in every CTest build (no Python needed), and
  `python_training/tests/test_waitk_policy.py` replays it too and checks it is
  current. With `FLUXVOICE_WAITK_TEST` set to the built test binary (CMake does
  this for `python_training_m3_stage04`), pytest additionally replays 1,000
  fresh random trajectories through `test_waitk_policy --replay` and compares
  them decision for decision.

## Verification (2026-11-15)

| Check | Command | Result |
|---|---|---|
| Python twin: scenarios, golden replay, FIFO, live C++ parity | `ctest --test-dir build-zig-m3s04 -R python_training_m3_stage04` (= `pytest tests/test_waitk_policy.py` with `FLUXVOICE_WAITK_TEST` set to the built binary) | **PASS** — 36 passed, 0 skipped (Python 3.12 venv), including decision-for-decision parity with `test_waitk_policy --replay` on 1,000 random trajectories |
| Golden file is reproducible | `.venv/Scripts/python.exe make_waitk_golden.py --check` | **PASS** — 256 trajectories, 50,092 bytes, up to date |
| C++ build, Windows 11 x64, `zig c++` 0.15.2 (Clang 20.1.2 frontend), Ninja, CMake 4.5.0, Release | `cmake -S . -B build-zig-m3s04 -G Ninja -DCMAKE_CXX_COMPILER=zigcxx.cmd -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_BUILD_TESTS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON -DFLUXVOICE_BUILD_GUI=OFF; cmake --build build-zig-m3s04` | **PASS** — all 138 build steps with `-Wall -Wextra -Wpedantic -Werror`, including the `fluxvoice_header_checks` TUs for `WaitKPolicy.hpp` and `UnitProsodyFifo.hpp` |
| `waitk_policy` CTest (same leg) | `ctest --test-dir build-zig-m3s04 -R waitk_policy --output-on-failure` | **PASS** — all checks, 256 golden trajectories replayed, zero allocations in the policy and FIFO loops |
| Full CTest (same leg, GUI and ORT off) | `ctest --test-dir build-zig-m3s04 --output-on-failure` | **PASS** — 82 of 82, including `waitk_policy`, `gate_s2ut_selftest` and `python_training_m3_stage03` / `04` / `05`. (A first run, taken while Stage 03 was still editing `extract_units.py`, had one unrelated failure in `python_training_stage09_10`; it passes now.) |
| Native Windows/MSVC, Linux/GCC, Linux/Clang, macOS/AppleClang | `.github/workflows/build.yml` | **NOT RUN** — pending a push; the zig build above is a Clang-frontend Windows check, not the MSVC leg |

`python_training/tests/test_waitk_policy.py` mirrors the C++ scenarios one for
one with the same numbers, so the expectations in `tests/test_waitk_policy.cpp`
are checked against both implementations.

## Implementation notes

- **Keep `WaitKPolicy` 100% ORT-free.** It is pure integer/float state logic and
  runs in every CTest build on Linux, Windows, and macOS.
- **Why `UnitProsodyFifo` matters.** Even when `translator_dec` emits 4 frames
  (80 ms) per step, the user may run the audio pipeline with `--frames-per-step 1`
  (20 ms vocoder hops) or `--frames-per-step 2` (40 ms hops) for lower I/O jitter,
  or an adaptive step may emit 2 chunks (8 frames) when catching up on a long
  Spanish/French phrase. The FIFO decouples translator emission granularity from
  vocoder hop consumption cleanly.

## Files touched

- `src/pipeline/WaitKPolicy.hpp`, `src/pipeline/UnitProsodyFifo.hpp` (new),
  `python_training/fvml/waitk_policy.py` (new),
  `python_training/make_waitk_golden.py` (new),
  `python_training/tests/test_waitk_policy.py` (new),
  `tests/test_waitk_policy.cpp` (new),
  `tests/data/waitk/golden_trajectories.txt` (new), `CMakeLists.txt`,
  `.github/workflows/build.yml`, `.gitattributes`

## Definition of done

- `ctest -R waitk_policy` passes all state-machine and FIFO test cases with
  warnings-as-errors enabled and zero heap allocations after construction.
- Python `pytest` confirms 100% decision-for-decision parity between
  `fvml/waitk_policy.py` and `WaitKPolicy.hpp` across 1,000 randomized speech/pause
  trajectories.
