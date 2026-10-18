# Stage 04 — `WaitKPolicy` & streaming unit/prosody FIFO

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 03](03_multilingual_encoder_vocoder_upgrade.md) · [Stage 05 →](05_causal_s2ut_translator_policy_gate.md)

**Status:** ☐ not started
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

- [ ] **Upgrade `src/pipeline/WaitKPolicy.hpp`:**
  - [ ] Keep the existing public API (`Decision::kRead`, `kWrite`, `kFlush`,
        `kMinK = 1`, `kMaxK = 10`, `kTailFlushMs = 400`, `decide()`, `k()`,
        `set_k()`, `t_src()`, `t_tgt()`, `reset()`) working so existing callers
        compile unchanged.
  - [ ] Add `Config` struct:
    - `k` (`1..10`, default `3`)
    - `tail_flush_ms` (default `400`)
    - `max_flush_ratio` (default `1.5f`: flush stops once
      `t_tgt_ >= ceil(max_flush_ratio * t_src_)` or `flush_steps_ >=
      max_extra_flush_steps`)
    - `adaptive_mode` (`bool`, default `false`)
    - `readiness_threshold` (`float`, default `0.5f`)
  - [ ] Add `<EOS>` & flush completion handling:
    - `notify_target_unit(bool is_eos = false)`: increments `t_tgt_`; if in
      `kFlush` and (`is_eos` or `t_tgt_` reaches the flush bound), marks the
      utterance drained and calls `reset()`, incrementing `flushes_completed_`.
    - If `is_eos` arrives during `kWrite` while the user is already silent, mark
      caught-up (`t_tgt_ = t_src_`) so no extra flush steps run after `<EOS>`.
  - [ ] **Mid-flush speech resumption:** if `on_chunk_energy(true, elapsed_ms)`
        is called while `Decision::kFlush` is active, clear `silence_ms_ = 0` and
        resume `kWrite` / `kRead` from the current `(t_src_, t_tgt_)` lag without
        resetting the KV caches.
  - [ ] **Adaptive readiness hook:** `on_source_readiness(float prob)` accumulates
        expected target chunks; in adaptive mode, `decide()` returns `kWrite`
        when `(t_src_ - t_tgt_ >= k_min_) && (readiness_accum_ >= 1.0f ||
        t_src_ - t_tgt_ >= k_max_)`, and `max_writes_this_step()` returns `1` or
        `2` if backlog exceeds target lag.
- [ ] **`src/pipeline/UnitProsodyFifo.hpp` (new, header-only):**
  - [ ] Fixed-capacity ring buffer (e.g. 64 frames = 1.28 s @ 50 Hz, preallocated
        at construction, zero heap allocations in steady state).
  - [ ] Stores `Frame { int64_t unit; float logf0_z, voicing, aperiodicity,
        energy_z; }`.
  - [ ] `push_chunk(std::span<const int64_t> units, std::span<const float>
        prosody_4xC)` ( skips trailing `<PAD>`/`<EOS>` frames).
  - [ ] `pop_hop(size_t n_frames, int64_t* units_out, PitchFrame* prosody_out) ->
        bool`: returns `true` and fills `n_frames` when at least `n_frames` are
        queued; returns `false` (meaning emit silence / `kRead`) when fewer than
        `n_frames` are available.
- [ ] **Python mirror (`python_training/fvml/waitk_policy.py`):**
  - [ ] Bit-exact Python port of `WaitKPolicy` and `UnitProsodyFifo` so Stage 05's
        SimulEval runner and Stage 06's parity tests execute the identical policy
        state machine as C++.
- [ ] **Unit tests (`tests/test_waitk_policy.cpp`, registered in `CMakeLists.txt`):**
  - [ ] **Classic Wait-*k* sequence:** for $k \in \{1, 3, 5, 10\}$, feed 20
        voiced chunks → verify first $k-1$ steps return `kRead` and remaining
        steps return `kWrite`.
  - [ ] **Short pause vs. tail flush:** a 300 ms unvoiced gap inside an utterance
        does *not* trigger `kFlush`; a 450 ms unvoiced gap triggers `kFlush`,
        drains remaining chunks up to the bound or `<EOS>`, and auto-resets
        `t_src() == 0 && t_tgt() == 0`.
  - [ ] **Early `<EOS>` termination:** `notify_target_unit(/*is_eos=*/true)` stops
        `kFlush` immediately on the next step.
  - [ ] **Speech resumption during flush:** speaking again after 1 flush step
        cancels `kFlush` and continues `kWrite` without zeroing `t_src_`.
  - [ ] **Hot `set_k()` clamping:** changing $k$ mid-utterance clamps to `[1..10]`
        and transitions cleanly between `kRead` and `kWrite`.
  - [ ] **`UnitProsodyFifo` rate bridging:** pushing 4-frame translator chunks and
        popping 1-, 2-, or 4-frame vocoder hops preserves exact frame order and
        allocates 0 bytes (`operator new` counter check).

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
  `tests/test_waitk_policy.cpp` (new), `CMakeLists.txt`

## Definition of done

- `ctest -R waitk_policy` passes all state-machine and FIFO test cases with
  warnings-as-errors enabled and zero heap allocations after construction.
- Python `pytest` confirms 100% decision-for-decision parity between
  `fvml/waitk_policy.py` and `WaitKPolicy.hpp` across 1,000 randomized speech/pause
  trajectories.
