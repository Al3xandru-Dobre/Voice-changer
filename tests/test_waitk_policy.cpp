// WaitKPolicy + UnitProsodyFifo — state-machine and rate-bridging tests
// (Milestone 3 / Stage 04,
// Documents/Milestone 3/04_waitk_policy_streaming_state_machine.md).
//
// ORT-free and hardware-free: runs in every CI build.
//
//   test_waitk_policy <dir>             unit tests + the golden trajectories in
//                                       <dir>/golden_trajectories.txt, written
//                                       by the Python twin
//                                       (python_training/make_waitk_golden.py)
//   test_waitk_policy --replay <file>   replay trajectory headers, one per
//                                       line, and print one result line each
//                                       (python_training/tests/test_waitk_policy.py
//                                       compares them with fvml/waitk_policy.py)
#include "pipeline/UnitProsodyFifo.hpp"
#include "pipeline/WaitKPolicy.hpp"

#include "AllocationCounter.hpp"
#include "TestSupport.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Decision = WaitKPolicy::Decision;

constexpr std::uint32_t kChunkMs = 80;

struct StepOut {
    Decision decision = Decision::kRead;
    int writes = 0;
};

// The canonical per-chunk call order documented in WaitKPolicy.hpp.
StepOut drive(WaitKPolicy& p, bool speech, std::uint32_t elapsed_ms = kChunkMs,
              float readiness = 0.0f, bool eos = false) {
    p.on_chunk_energy(speech, elapsed_ms);
    if (p.source_wanted()) {
        p.notify_source_unit();
        p.on_source_readiness(readiness);
    }
    StepOut out;
    out.decision = p.decide();
    const int n = p.max_writes_this_step();
    const std::uint64_t before = p.flushes_completed();
    for (int w = 0; w < n; ++w) {
        p.notify_target_unit(eos && w == 0);
        ++out.writes;
        if (p.flushes_completed() != before) break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// WaitKPolicy
// ---------------------------------------------------------------------------

void test_constants_and_defaults() {
    CHECK(WaitKPolicy::kMinK == 1);
    CHECK(WaitKPolicy::kMaxK == 10);
    CHECK(WaitKPolicy::kTailFlushMs == 400);
    CHECK(WaitKPolicy::kChunkFrames == 4);
    CHECK(WaitKPolicy::kFrameMs == 20);

    const WaitKPolicy p;
    CHECK(p.k() == 3);
    CHECK(p.chunk_ms() == 80);            // 4 frames x 20 ms
    CHECK(p.wait_ms() == 240);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
    CHECK(p.decide() == Decision::kRead);
    CHECK(p.max_writes_this_step() == 0);
    CHECK(!p.utterance_active() && !p.flushing() && !p.flush_pending());

    const WaitKPolicy::Config c{};
    CHECK(c.k == 3);
    CHECK(c.tail_flush_ms == 400);
    CHECK(c.max_flush_ratio == 1.5f);
    CHECK(c.max_flush_steps == 32);
    CHECK(!c.adaptive_mode);
    CHECK(c.readiness_threshold == 0.5f);
    CHECK(c.adaptive_extra_wait == 2);
}

void test_config_is_sanitized() {
    WaitKPolicy::Config c;
    c.k = 99;
    c.max_flush_ratio = 0.25f;
    c.max_flush_steps = 0;
    c.readiness_threshold = 7.0f;
    c.adaptive_extra_wait = -3;
    c.chunk_frames = 0;
    c.frame_ms = 0;
    const WaitKPolicy p(c);
    CHECK(p.k() == 10);
    CHECK(p.config().max_flush_ratio == 1.0f);
    CHECK(p.config().max_flush_steps == 1);
    CHECK(p.config().readiness_threshold == 1.0f);
    CHECK(p.config().adaptive_extra_wait == 0);
    CHECK(p.config().chunk_frames == 1);
    CHECK(p.config().frame_ms == 1);

    c.max_flush_ratio = std::numeric_limits<float>::quiet_NaN();
    c.readiness_threshold = std::numeric_limits<float>::quiet_NaN();
    const WaitKPolicy q(c);
    CHECK(q.config().max_flush_ratio == 1.0f);
    CHECK(q.config().readiness_threshold == 0.0f);

    CHECK(WaitKPolicy(-5).k() == 1);
    CHECK(WaitKPolicy(0).k() == 1);
    CHECK(WaitKPolicy(11).k() == 10);
}

// Classic Wait-k: the first k-1 steps read, every later step writes exactly one
// target chunk, and the lag settles at k-1 after each write.
void test_classic_waitk_sequence() {
    for (const int k : {1, 3, 5, 10}) {
        WaitKPolicy p(k);
        for (int step = 0; step < 20; ++step) {
            const StepOut out = drive(p, true);
            if (step < k - 1) {
                CHECK(out.decision == Decision::kRead);
                CHECK(out.writes == 0);
            } else {
                CHECK(out.decision == Decision::kWrite);
                CHECK(out.writes == 1);
                CHECK(p.t_src() - p.t_tgt() == k - 1);
            }
        }
        CHECK(p.t_src() == 20);
        CHECK(p.t_tgt() == 20 - (k - 1));
        CHECK(p.flushes_completed() == 0);
    }
}

// The pre-Stage-04 call pattern (no energy reports) still works: source
// notifications open the utterance and no flush can ever start.
void test_legacy_calls_without_vad() {
    WaitKPolicy p(3);
    CHECK(p.source_wanted());
    p.notify_source_unit();
    p.notify_source_unit();
    CHECK(p.decide() == Decision::kRead);
    p.notify_source_unit();
    CHECK(p.decide() == Decision::kWrite);
    p.notify_target_unit();
    CHECK(p.decide() == Decision::kRead);
    CHECK(p.t_src() == 3 && p.t_tgt() == 1);
    CHECK(!p.flush_pending());
    p.reset();
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
    CHECK(p.source_wanted());
}

// Stage 04 bug 2: silence before the first word must not arm the tail flush.
void test_leading_silence_is_not_counted() {
    WaitKPolicy p(3);
    for (int i = 0; i < 25; ++i) {           // 2 s of leading silence
        const StepOut out = drive(p, false);
        CHECK(out.decision == Decision::kRead);
        CHECK(!p.source_wanted());
    }
    CHECK(p.silence_ms() == 0);
    CHECK(p.t_src() == 0);
    CHECK(!p.utterance_active());

    CHECK(drive(p, true).decision == Decision::kRead);     // first word
    CHECK(p.t_src() == 1);
    CHECK(drive(p, false).decision == Decision::kRead);    // one unvoiced chunk
    CHECK(!p.flushing());
    CHECK(p.silence_ms() == 80);
    CHECK(drive(p, true).decision == Decision::kWrite);
    CHECK(p.flushes_completed() == 0);
}

// A 300 ms pause stays inside the utterance; a 450 ms pause starts the flush,
// which drains to the bound and auto-resets.
void test_short_pause_vs_tail_flush() {
    WaitKPolicy p(3);
    for (int i = 0; i < 10; ++i) drive(p, true);
    CHECK(p.t_src() == 10 && p.t_tgt() == 8);

    drive(p, false, 150);
    const StepOut at_300 = drive(p, false, 150);            // 300 ms
    CHECK(at_300.decision == Decision::kWrite);
    CHECK(!p.flushing());
    CHECK(p.silence_ms() == 300);
    drive(p, true);                                         // speech resumes
    CHECK(p.silence_ms() == 0);
    CHECK(p.flushes_completed() == 0);
    CHECK(p.t_src() == 13 && p.t_tgt() == 11);

    drive(p, false, 150);
    drive(p, false, 150);
    CHECK(!p.flushing());                                   // 300 ms
    CHECK(p.t_src() == 15 && p.t_tgt() == 13);
    const StepOut first_flush = drive(p, false, 150);       // 450 ms > 400 ms
    CHECK(first_flush.decision == Decision::kFlush);
    CHECK(first_flush.writes == 1);
    CHECK(p.flushing() && p.flush_pending());
    CHECK(!p.source_wanted());                              // encoder is skipped
    CHECK(p.t_src() == 15);                                 // no source during the flush
    CHECK(p.t_tgt() == 14);

    // Bound: ceil(1.5 * 15) = 23 target chunks, i.e. 10 flush steps in total.
    int flush_steps = 1;
    while (p.flushing()) {
        CHECK(drive(p, false).decision == Decision::kFlush);
        ++flush_steps;
        CHECK(flush_steps <= 10);
        if (flush_steps > 10) break;
    }
    CHECK(flush_steps == 10);
    CHECK(p.flushes_completed() == 1);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
    CHECK(!p.utterance_active());
    CHECK(drive(p, false).decision == Decision::kRead);     // silence after the flush
    CHECK(p.flushes_completed() == 1);
}

// Exactly tail_flush_ms of silence is not yet a flush (strictly greater).
void test_tail_flush_threshold_is_strict() {
    WaitKPolicy p(2);
    for (int i = 0; i < 6; ++i) drive(p, true);
    for (int i = 0; i < 5; ++i) drive(p, false);            // 400 ms
    CHECK(p.silence_ms() == 400);
    CHECK(!p.flushing());
    CHECK(drive(p, false, 1).decision == Decision::kFlush); // 401 ms
}

// Stage 04 bug 1: without <EOS> the flush still ends, at max_flush_steps.
void test_flush_is_bounded_by_max_flush_steps() {
    WaitKPolicy::Config c;
    c.k = 3;
    c.max_flush_ratio = 4.0f;
    c.max_flush_steps = 3;
    WaitKPolicy p(c);
    for (int i = 0; i < 30; ++i) drive(p, true);
    for (int i = 0; i < 5; ++i) drive(p, false);
    int flush_steps = 0;
    for (int i = 0; i < 50; ++i) {
        const StepOut out = drive(p, false);
        if (out.decision == Decision::kFlush) ++flush_steps;
    }
    CHECK(flush_steps == 3);
    CHECK(p.flushes_completed() == 1);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
}

void test_flush_ratio_bound() {
    WaitKPolicy::Config c;
    c.k = 5;
    c.max_flush_ratio = 1.0f;      // stop once every source chunk has a target chunk
    WaitKPolicy p(c);
    for (int i = 0; i < 12; ++i) drive(p, true);
    for (int i = 0; i < 5; ++i) drive(p, false);
    CHECK(p.t_src() == 17 && p.t_tgt() == 13);
    int flush_steps = 0;
    while (drive(p, false).decision == Decision::kFlush) ++flush_steps;
    CHECK(flush_steps == 4);       // 17 - 13
    CHECK(p.flushes_completed() == 1);
}

// <EOS> during the flush ends it at once: the next step is no longer kFlush.
void test_early_eos_terminates_flush() {
    WaitKPolicy p(3);
    for (int i = 0; i < 20; ++i) drive(p, true);
    for (int i = 0; i < 5; ++i) drive(p, false);
    CHECK(drive(p, false).decision == Decision::kFlush);
    const StepOut eos_step = drive(p, false, kChunkMs, 0.0f, /*eos=*/true);
    CHECK(eos_step.decision == Decision::kFlush);
    CHECK(p.flushes_completed() == 1);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
    CHECK(!p.flushing());
    CHECK(drive(p, false).decision == Decision::kRead);
}

// <EOS> during kWrite while the user is already silent: caught up, so the tail
// flush completes without a single decoder-only step.
void test_eos_while_silent_skips_the_flush() {
    WaitKPolicy p(3);
    for (int i = 0; i < 10; ++i) drive(p, true);
    drive(p, false);
    const StepOut eos_step = drive(p, false, kChunkMs, 0.0f, /*eos=*/true);
    CHECK(eos_step.decision == Decision::kWrite);
    CHECK(p.t_tgt() == p.t_src());
    for (int i = 0; i < 3; ++i) {                           // 240..400 ms
        CHECK(drive(p, false).decision == Decision::kRead);
        CHECK(p.flushes_completed() == 0);
    }
    const StepOut done = drive(p, false);                   // 480 ms
    CHECK(done.decision == Decision::kRead);
    CHECK(done.writes == 0);
    CHECK(p.flushes_completed() == 1);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
}

// <EOS> while the user is still talking is an ordinary target chunk.
void test_eos_while_speaking_is_ignored() {
    WaitKPolicy p(2);
    for (int i = 0; i < 5; ++i) drive(p, true);
    const StepOut out = drive(p, true, kChunkMs, 0.0f, /*eos=*/true);
    CHECK(out.decision == Decision::kWrite);
    CHECK(p.t_src() == 6 && p.t_tgt() == 5);
    CHECK(drive(p, true).decision == Decision::kWrite);
}

// <EOS> in a short pause, then speech again: the new phrase waits k chunks
// afresh and the utterance is not reset.
void test_speech_after_eos_in_short_pause() {
    WaitKPolicy p(3);
    for (int i = 0; i < 10; ++i) drive(p, true);
    drive(p, false);
    drive(p, false, kChunkMs, 0.0f, /*eos=*/true);
    CHECK(drive(p, false).decision == Decision::kRead);
    CHECK(p.t_src() == 13);
    CHECK(drive(p, true).decision == Decision::kRead);      // lag restarts at 1
    CHECK(p.t_src() == 14 && p.t_tgt() == 13);
    CHECK(drive(p, true).decision == Decision::kRead);
    CHECK(drive(p, true).decision == Decision::kWrite);
    CHECK(p.flushes_completed() == 0);
}

// Speaking again after one flush step cancels the flush and keeps the counters.
void test_speech_resumption_during_flush() {
    WaitKPolicy p(3);
    for (int i = 0; i < 12; ++i) drive(p, true);
    for (int i = 0; i < 5; ++i) drive(p, false);
    CHECK(p.t_src() == 17 && p.t_tgt() == 15);
    CHECK(drive(p, false).decision == Decision::kFlush);    // one flush step
    CHECK(p.t_src() == 17 && p.t_tgt() == 16);
    CHECK(p.flush_steps() == 1);

    const StepOut resumed = drive(p, true);
    CHECK(resumed.decision == Decision::kRead);             // lag 18 - 16 = 2 < k
    CHECK(!p.flushing());
    CHECK(p.silence_ms() == 0);
    CHECK(p.t_src() == 18 && p.t_tgt() == 16);              // nothing was zeroed
    CHECK(p.flush_resumes() == 1);
    CHECK(p.flushes_completed() == 0);
    CHECK(drive(p, true).decision == Decision::kWrite);
    CHECK(p.t_src() == 19 && p.t_tgt() == 17);

    // The flush overshot the source (t_tgt > t_src): resumption reads until the
    // lag is back at k.
    WaitKPolicy q(2);
    for (int i = 0; i < 4; ++i) drive(q, true);
    for (int i = 0; i < 5; ++i) drive(q, false);
    for (int i = 0; i < 3; ++i) CHECK(drive(q, false).decision == Decision::kFlush);
    CHECK(q.t_src() == 9 && q.t_tgt() == 11);
    for (int i = 0; i < 3; ++i) CHECK(drive(q, true).decision == Decision::kRead);
    CHECK(drive(q, true).decision == Decision::kWrite);
    CHECK(q.t_src() - q.t_tgt() == 1);
}

void test_hot_set_k_clamping() {
    WaitKPolicy p(3);
    p.set_k(0);
    CHECK(p.k() == 1);
    p.set_k(-7);
    CHECK(p.k() == 1);
    p.set_k(42);
    CHECK(p.k() == 10);
    CHECK(p.wait_ms() == 800);
    p.set_k(3);

    for (int i = 0; i < 8; ++i) drive(p, true);
    CHECK(p.t_src() == 8 && p.t_tgt() == 6);

    p.set_k(6);                                 // raise mid-utterance: read until lag 6
    for (int i = 0; i < 3; ++i) CHECK(drive(p, true).decision == Decision::kRead);
    CHECK(drive(p, true).decision == Decision::kWrite);
    CHECK(p.t_src() == 12 && p.t_tgt() == 7);

    p.set_k(2);                                 // lower mid-utterance: keeps writing
    for (int i = 0; i < 4; ++i) {
        const StepOut out = drive(p, true);
        CHECK(out.decision == Decision::kWrite);
        CHECK(out.writes == 1);                 // strict mode never bursts
    }
    CHECK(p.t_src() == 16 && p.t_tgt() == 11);
    CHECK(p.flushes_completed() == 0);
}

void test_reset_keeps_k_and_counters() {
    WaitKPolicy p(4);
    for (int i = 0; i < 9; ++i) drive(p, true);
    for (int i = 0; i < 6; ++i) drive(p, false);
    while (p.flushing()) drive(p, false);
    CHECK(p.flushes_completed() == 1);
    for (int i = 0; i < 5; ++i) drive(p, true);
    p.reset();
    CHECK(p.k() == 4);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0 && p.silence_ms() == 0);
    CHECK(p.flushes_completed() == 1);
    CHECK(!p.source_wanted());                  // a VAD is attached: idle until speech
    for (int i = 0; i < 3; ++i) CHECK(drive(p, true).decision == Decision::kRead);
    CHECK(drive(p, true).decision == Decision::kWrite);
}

void test_stray_notifications_are_ignored() {
    WaitKPolicy p(3);
    p.on_chunk_energy(false, kChunkMs);         // VAD attached, idle
    p.notify_source_unit();
    p.notify_target_unit(true);
    p.on_source_readiness(1.0f);
    CHECK(p.t_src() == 0 && p.t_tgt() == 0);
    CHECK(p.flushes_completed() == 0);

    for (int i = 0; i < 8; ++i) drive(p, true);
    for (int i = 0; i < 6; ++i) drive(p, false);
    CHECK(p.flushing());
    const std::int64_t t_src = p.t_src();
    p.notify_source_unit();                     // encoder must not run in a flush
    CHECK(p.t_src() == t_src);
}

void test_silence_counter_saturates() {
    WaitKPolicy::Config c;
    c.tail_flush_ms = std::numeric_limits<std::uint32_t>::max();
    WaitKPolicy p(c);
    drive(p, true);
    p.on_chunk_energy(false, std::numeric_limits<std::uint32_t>::max() - 5);
    p.on_chunk_energy(false, 100);
    CHECK(p.silence_ms() == std::numeric_limits<std::uint32_t>::max());
    CHECK(!p.flushing());                       // never strictly greater
}

// 50 Hz frames are regrouped into 80 ms chunks; any speech frame makes the
// chunk speech.
void test_frame_to_chunk_aggregation() {
    WaitKPolicy p(1);
    CHECK(!p.on_frame_energy(false));
    CHECK(!p.on_frame_energy(true));
    CHECK(!p.on_frame_energy(false));
    CHECK(!p.utterance_active());               // nothing forwarded yet
    CHECK(p.on_frame_energy(false));            // 4th frame closes the chunk
    CHECK(p.utterance_active());
    CHECK(p.silence_ms() == 0);

    for (int chunk = 0; chunk < 5; ++chunk) {   // 20 silent frames = 400 ms
        for (int f = 0; f < 4; ++f) CHECK(p.on_frame_energy(false) == (f == 3));
    }
    CHECK(p.silence_ms() == 400);
    for (int f = 0; f < 3; ++f) p.on_frame_energy(false);
    CHECK(p.silence_ms() == 400);
    p.on_frame_energy(false);
    CHECK(p.silence_ms() == 0);                 // nothing fed: the flush completed at once
    CHECK(p.flushes_completed() == 1);

    WaitKPolicy::Config c;
    c.chunk_frames = 2;
    c.frame_ms = 10;
    WaitKPolicy q(c);
    CHECK(q.chunk_ms() == 20);
    CHECK(!q.on_frame_energy(true));
    CHECK(q.on_frame_energy(false));
}

WaitKPolicy::Config adaptive_config(int k) {
    WaitKPolicy::Config c;
    c.k = k;
    c.adaptive_mode = true;
    return c;
}

// Adaptive mode: 0 writes while the source is not ready (up to k + extra), 1
// when a chunk is ready, 2 when two are banked and the backlog allows it.
void test_adaptive_readiness() {
    {
        WaitKPolicy p(adaptive_config(2));      // k_min 2, k_max 4
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 0);  // lag 1
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 0);  // lag 2, not ready
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 0);  // lag 3, not ready
        const StepOut forced = drive(p, true, kChunkMs, 0.0f);   // lag 4 = k_max
        CHECK(forced.decision == Decision::kWrite);
        CHECK(forced.writes == 1);
        CHECK(p.t_src() == 4 && p.t_tgt() == 1);
    }
    {
        WaitKPolicy p(adaptive_config(2));
        CHECK(drive(p, true, kChunkMs, 1.0f).writes == 0);  // ready, but lag 1 < k
        const StepOut out = drive(p, true, kChunkMs, 0.0f);
        CHECK(out.decision == Decision::kWrite);            // banked readiness
        CHECK(out.writes == 1);
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 0);  // bank is spent
    }
    {
        // Readiness below the threshold is "blank" and never accumulates.
        WaitKPolicy::Config c = adaptive_config(1);
        c.adaptive_extra_wait = 10;                         // k_max 11: nothing is forced
        WaitKPolicy p(c);
        for (int i = 0; i < 3; ++i) CHECK(drive(p, true, kChunkMs, 0.49f).writes == 0);
        CHECK(drive(p, true, kChunkMs, 0.5f).writes == 0);  // 0.5 banked
        CHECK(drive(p, true, kChunkMs, 0.5f).writes == 1);  // 1.0 banked
        CHECK(p.t_src() == 5 && p.t_tgt() == 1);
    }
    {
        // Two banked chunks + backlog above k: a 2-write catch-up step, and the
        // second write still happens at a lag of at least k.
        WaitKPolicy p(adaptive_config(3));
        drive(p, true, kChunkMs, 1.0f);
        drive(p, true, kChunkMs, 1.0f);
        CHECK(p.t_tgt() == 0);
        drive(p, true, kChunkMs, 0.0f);                     // lag 3, bank 2 -> 1 write
        CHECK(p.t_src() == 3 && p.t_tgt() == 1);
        drive(p, true, kChunkMs, 1.0f);                     // lag 3, bank 2 -> 1 write
        CHECK(p.t_src() == 4 && p.t_tgt() == 2);
        p.set_k(1);                                         // backlog now exceeds k
        const StepOut burst = drive(p, true, kChunkMs, 1.0f);
        CHECK(burst.decision == Decision::kWrite);
        CHECK(burst.writes == 2);
        CHECK(p.t_src() == 5 && p.t_tgt() == 4);
    }
    {
        // NaN readiness is discarded; values above 2 are clamped to 2.
        WaitKPolicy p(adaptive_config(1));
        CHECK(drive(p, true, kChunkMs, std::numeric_limits<float>::quiet_NaN()).writes == 0);
        CHECK(drive(p, true, kChunkMs, 9.0f).writes == 2);  // lag 2 > k, bank 2
        CHECK(p.t_src() == 2 && p.t_tgt() == 2);
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 0);  // bank is spent
    }
    {
        // Constant k: a held-back step followed by a source chunk worth two
        // target chunks catches up with a 2-write step (longer target phrase).
        WaitKPolicy p(adaptive_config(2));
        drive(p, true, kChunkMs, 1.0f);                     // lag 1
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 1);  // lag 2, bank 1
        CHECK(drive(p, true, kChunkMs, 0.0f).writes == 0);  // lag 2, not ready
        const StepOut burst = drive(p, true, kChunkMs, 2.0f);   // lag 3, bank 2
        CHECK(burst.decision == Decision::kWrite);
        CHECK(burst.writes == 2);
        CHECK(p.t_src() == 4 && p.t_tgt() == 3);
        CHECK(p.k() == 2);
        // At lag == k two banked chunks still give a single write.
        CHECK(drive(p, true, kChunkMs, 2.0f).writes == 1);
        CHECK(p.t_src() - p.t_tgt() == 1);
    }
}

// Every write outside a flush happens at a lag >= k, in both modes.
void test_write_lag_invariant() {
    std::uint32_t rng = 12345;
    const auto next = [&rng] {
        rng = rng * 1664525u + 1013904223u;
        return rng >> 8;
    };
    for (int trial = 0; trial < 200; ++trial) {
        WaitKPolicy::Config c;
        c.k = 1 + static_cast<int>(next() % 10);
        c.adaptive_mode = (next() & 1u) != 0;
        c.adaptive_extra_wait = static_cast<int>(next() % 5);
        WaitKPolicy p(c);
        for (int step = 0; step < 150; ++step) {
            const bool speech = (next() % 8) != 0;
            p.on_chunk_energy(speech, kChunkMs);
            if (p.source_wanted()) {
                p.notify_source_unit();
                p.on_source_readiness(2.0f * static_cast<float>(next() % 8) / 7.0f);
            }
            const Decision d = p.decide();
            const int n = p.max_writes_this_step();
            CHECK(n >= 0 && n <= 2);
            CHECK((d == Decision::kRead) == (n == 0));
            if (d == Decision::kFlush) CHECK(n == 1);
            if (!c.adaptive_mode) CHECK(n <= 1);
            const std::uint64_t before = p.flushes_completed();
            for (int w = 0; w < n; ++w) {
                if (d == Decision::kWrite) CHECK(p.t_src() - p.t_tgt() >= p.k());
                p.notify_target_unit(false);
                if (p.flushes_completed() != before) break;
            }
        }
    }
}

void test_policy_never_allocates() {
    const long before = test::allocation_count();
    WaitKPolicy::Config c;
    c.adaptive_mode = true;
    WaitKPolicy p(c);
    for (int i = 0; i < 2000; ++i) {
        drive(p, (i / 13) % 3 != 0, kChunkMs, 0.7f, i % 97 == 0);
        if (i % 251 == 0) p.set_k(i % 12);
    }
    WaitKPolicy frames(c);
    for (int i = 0; i < 2000; ++i) frames.on_frame_energy((i / 40) % 2 == 0);
    if constexpr (test::kAllocationCounting) {
        CHECK(test::allocation_count() == before);
    }
    CHECK(p.flushes_completed() > 0);
    CHECK(frames.utterance_active() || frames.flushes_completed() > 0);
}

// ---------------------------------------------------------------------------
// Golden trajectories / replay
// ---------------------------------------------------------------------------

struct Trajectory {
    WaitKPolicy::Config config;
    std::uint32_t elapsed_ms = kChunkMs;
    std::string steps;
    std::vector<std::pair<std::size_t, int>> set_k;
};

// Header fields: k adaptive extra tail_ms ratio max_flush_steps threshold
// elapsed_ms steps set_k ("-" or "i=k,i=k").
bool parse_header(std::istringstream& in, Trajectory& t) {
    int adaptive = 0;
    std::string set_k;
    if (!(in >> t.config.k >> adaptive >> t.config.adaptive_extra_wait >> t.config.tail_flush_ms
              >> t.config.max_flush_ratio >> t.config.max_flush_steps
              >> t.config.readiness_threshold >> t.elapsed_ms >> t.steps >> set_k)) {
        return false;
    }
    t.config.adaptive_mode = adaptive != 0;
    t.set_k.clear();
    if (set_k == "-") return true;
    std::istringstream pairs(set_k);
    std::string item;
    while (std::getline(pairs, item, ',')) {
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos) return false;
        t.set_k.emplace_back(static_cast<std::size_t>(std::strtoull(item.c_str(), nullptr, 10)),
                             static_cast<int>(std::strtol(item.c_str() + eq + 1, nullptr, 10)));
    }
    return true;
}

// Mirrors replay() in python_training/fvml/waitk_policy.py.
std::string replay(const Trajectory& t) {
    WaitKPolicy p(t.config);
    std::string decisions;
    decisions.reserve(t.steps.size());
    std::uint64_t h = 0;
    for (std::size_t i = 0; i < t.steps.size(); ++i) {
        const unsigned bits = static_cast<unsigned>(t.steps[i] - '0');
        const bool speech = (bits & 1u) != 0;
        const unsigned q = (bits >> 1) & 7u;
        const bool eos = (bits & 16u) != 0;
        for (const auto& [index, k] : t.set_k) {
            if (index == i) p.set_k(k);
        }
        p.on_chunk_energy(speech, t.elapsed_ms);
        if (p.source_wanted()) {
            p.notify_source_unit();
            p.on_source_readiness(2.0f * static_cast<float>(q) / 7.0f);
        }
        const Decision d = p.decide();
        const int n = p.max_writes_this_step();
        const std::uint64_t before = p.flushes_completed();
        for (int w = 0; w < n; ++w) {
            p.notify_target_unit(eos && w == 0);
            if (p.flushes_completed() != before) break;
        }
        decisions.push_back(d == Decision::kFlush ? 'F' : "RWX"[n]);
        h = (h * 1000003u + (static_cast<std::uint64_t>(p.t_src()) & 0xFFFFu) * 257u
             + (static_cast<std::uint64_t>(p.t_tgt()) & 0xFFFFu) * 17u + p.flushes_completed())
            & 0xFFFFFFFFu;
    }
    std::ostringstream out;
    out << decisions << ' ' << p.t_src() << ' ' << p.t_tgt() << ' ' << p.flushes_completed() << ' '
        << p.flush_resumes() << ' ' << h;
    return out.str();
}

std::string rest_of(std::istringstream& in) {
    std::string rest;
    std::string word;
    while (in >> word) {
        if (!rest.empty()) rest.push_back(' ');
        rest += word;
    }
    return rest;
}

void test_golden_trajectories(const std::string& dir) {
    const std::string path = dir + "/golden_trajectories.txt";
    std::ifstream file(path);
    CHECK(file.is_open());
    if (!file.is_open()) {
        std::printf("cannot open %s\n", path.c_str());
        return;
    }
    std::string line;
    int cases = 0;
    int flushes = 0;
    int mismatches = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        Trajectory t;
        const bool parsed = parse_header(in, t);
        CHECK(parsed);
        if (!parsed) continue;
        const std::string expected = rest_of(in);
        const std::string actual = replay(t);
        if (actual != expected) {
            if (++mismatches <= 3) {
                std::printf("golden case %d differs:\n  expected %s\n  actual   %s\n", cases,
                            expected.c_str(), actual.c_str());
            }
        }
        if (actual.find('F') != std::string::npos) ++flushes;
        ++cases;
    }
    CHECK(mismatches == 0);
    CHECK(cases >= 200);
    CHECK(flushes >= 20);       // the file exercises the flush path
    std::printf("[waitk_policy] %d golden trajectories replayed\n", cases);
}

int replay_file(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        Trajectory t;
        if (!parse_header(in, t)) {
            std::fprintf(stderr, "malformed trajectory: %s\n", line.c_str());
            return 2;
        }
        std::printf("%s\n", replay(t).c_str());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// UnitProsodyFifo
// ---------------------------------------------------------------------------

constexpr std::int64_t kVocab = 1000;
constexpr std::int64_t kBos = 1000;
constexpr std::int64_t kEos = 1001;
constexpr std::int64_t kPad = 1002;

// Frame i carries unit i % kVocab and prosody derived from i, so order and
// pairing are both checkable after any push/pop pattern.
void make_chunk(std::int64_t first, std::array<std::int64_t, 4>& units,
                std::array<float, 16>& prosody) {
    for (std::size_t j = 0; j < 4; ++j) {
        const std::int64_t i = first + static_cast<std::int64_t>(j);
        units[j] = i % kVocab;
        const float base = static_cast<float>(i);
        prosody[4 * j + 0] = base + 0.25f;     // z_f0
        prosody[4 * j + 1] = base + 0.5f;      // voicing
        prosody[4 * j + 2] = base + 0.75f;     // aperiodicity
        prosody[4 * j + 3] = -base;            // z_energy
    }
}

bool frame_matches(std::int64_t i, std::int64_t unit, const PitchFrame& p) {
    const float base = static_cast<float>(i);
    return unit == i % kVocab && p.z_logf0 == base + 0.25f && p.voicing == base + 0.5f
           && p.aperiodicity == base + 0.75f && p.z_energy == -base;
}

void test_fifo_defaults() {
    const UnitProsodyFifo fifo;
    CHECK(fifo.capacity() == 64);              // 1.28 s at 50 Hz
    CHECK(fifo.size() == 0 && fifo.empty());
    CHECK(fifo.free_frames() == 64);
    CHECK(fifo.overflows() == 0);
    CHECK(UnitProsodyFifo(0).capacity() == 1);
}

// 4-frame translator chunks in, 1-/2-/4-frame vocoder hops out: exact frame
// order, exact unit/prosody pairing, zero allocations after construction.
void test_fifo_rate_bridging() {
    for (const std::size_t hop : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        UnitProsodyFifo fifo;
        std::array<std::int64_t, 4> units{};
        std::array<float, 16> prosody{};
        std::array<std::int64_t, 4> units_out{};
        std::array<PitchFrame, 4> prosody_out{};
        const long before = test::allocation_count();
        std::int64_t pushed = 0;
        std::int64_t popped = 0;
        bool ok = true;
        for (int step = 0; step < 500; ++step) {            // wraps the ring > 30 times
            // An adaptive step may emit two chunks; do so every third step.
            const int chunks = (step % 3 == 0) ? 2 : 1;
            for (int c = 0; c < chunks; ++c) {
                make_chunk(pushed, units, prosody);
                const auto r = fifo.push_chunk(units, prosody);
                ok = ok && !r.overflow && r.pushed == 4 && !r.eos;
                pushed += 4;
            }
            // The vocoder drains one chunk of audio per step, plus the backlog.
            while (fifo.size() > 8) {
                ok = ok && fifo.pop_hop(hop, units_out.data(), prosody_out.data());
                for (std::size_t j = 0; j < hop; ++j) {
                    ok = ok && frame_matches(popped, units_out[j], prosody_out[j]);
                    ++popped;
                }
            }
        }
        while (fifo.pop_hop(hop, units_out.data(), prosody_out.data())) {
            for (std::size_t j = 0; j < hop; ++j) {
                ok = ok && frame_matches(popped, units_out[j], prosody_out[j]);
                ++popped;
            }
        }
        if constexpr (test::kAllocationCounting) {
            CHECK(test::allocation_count() == before);
        }
        CHECK(ok);
        CHECK(popped == pushed);
        CHECK(fifo.empty());
        CHECK(fifo.overflows() == 0);
    }
}

void test_fifo_underflow_returns_false() {
    UnitProsodyFifo fifo;
    std::array<std::int64_t, 4> units{};
    std::array<float, 16> prosody{};
    std::array<std::int64_t, 4> units_out{-1, -1, -1, -1};
    std::array<PitchFrame, 4> prosody_out{};
    CHECK(!fifo.pop_hop(1, units_out.data(), prosody_out.data()));
    CHECK(fifo.pop_hop(0, units_out.data(), prosody_out.data()));   // empty hop
    make_chunk(0, units, prosody);
    fifo.push_chunk(units, prosody);
    CHECK(fifo.pop_hop(2, units_out.data(), prosody_out.data()));
    CHECK(fifo.size() == 2);
    units_out = {-1, -1, -1, -1};
    CHECK(!fifo.pop_hop(4, units_out.data(), prosody_out.data())); // 2 queued < 4
    CHECK(units_out[0] == -1);                                     // nothing written
    CHECK(fifo.size() == 2);                                       // nothing consumed
    CHECK(fifo.pop_hop(2, units_out.data(), prosody_out.data()));
    CHECK(frame_matches(2, units_out[0], prosody_out[0]));
    CHECK(frame_matches(3, units_out[1], prosody_out[1]));
}

void test_fifo_skips_trailing_special_units() {
    UnitProsodyFifo fifo(64, kVocab, kEos);
    std::array<std::int64_t, 4> units{};
    std::array<float, 16> prosody{};
    make_chunk(0, units, prosody);

    units = {10, 11, kEos, kPad};
    auto r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 2 && r.eos && !r.overflow);
    CHECK(fifo.size() == 2);

    units = {12, kPad, kPad, kPad};
    r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 1 && !r.eos);

    units = {kEos, kPad, kPad, kPad};
    r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 0 && r.eos && !r.overflow);

    // A regular unit after a special one is still dropped: only a prefix is kept.
    units = {13, kBos, 14, 15};
    r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 1 && !r.eos);

    units = {-1, 16, 17, 18};
    r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 0);

    units = {999, 0, 1, 2};                    // vocab - 1 is a regular unit
    r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 4);

    std::array<UnitProsodyFifo::Frame, 8> out{};
    CHECK(fifo.size() == 8);
    CHECK(fifo.pop_hop(8, out.data()));
    const std::array<std::int64_t, 8> expected{10, 11, 12, 13, 999, 0, 1, 2};
    for (std::size_t i = 0; i < expected.size(); ++i) CHECK(out[i].unit == expected[i]);
    CHECK(out[1].logf0_z == 1.25f);            // prosody row 1 travelled with unit 11
    CHECK(out[2].logf0_z == 0.25f);            // row 0 of the next chunk
}

void test_fifo_overflow_is_all_or_nothing() {
    UnitProsodyFifo fifo(6, kVocab);
    std::array<std::int64_t, 4> units{};
    std::array<float, 16> prosody{};
    make_chunk(0, units, prosody);
    CHECK(fifo.push_chunk(units, prosody).pushed == 4);
    make_chunk(4, units, prosody);
    const auto r = fifo.push_chunk(units, prosody);         // needs 4, 2 free
    CHECK(r.overflow && r.pushed == 0);
    CHECK(fifo.size() == 4);
    CHECK(fifo.overflows() == 1);

    std::array<std::int64_t, 4> units_out{};
    std::array<PitchFrame, 4> prosody_out{};
    CHECK(fifo.pop_hop(2, units_out.data(), prosody_out.data()));
    CHECK(fifo.push_chunk(units, prosody).pushed == 4);     // fits now, wraps the ring
    CHECK(fifo.size() == 6 && fifo.free_frames() == 0);
    for (std::int64_t i = 2; i < 8; ++i) {
        CHECK(fifo.pop_hop(1, units_out.data(), prosody_out.data()));
        CHECK(frame_matches(i, units_out[0], prosody_out[0]));
    }
    CHECK(fifo.empty());

    make_chunk(0, units, prosody);
    fifo.push_chunk(units, prosody);
    fifo.clear();
    CHECK(fifo.empty() && fifo.free_frames() == 6);
    CHECK(!fifo.pop_hop(1, units_out.data(), prosody_out.data()));
}

void test_fifo_short_prosody_span() {
    UnitProsodyFifo fifo;
    const std::array<std::int64_t, 4> units{1, 2, 3, 4};
    const std::array<float, 9> prosody{};                   // only 2 whole rows
    const auto r = fifo.push_chunk(units, prosody);
    CHECK(r.pushed == 2);
}

} // namespace

// BUG-028: pad_to_multiple() keeps the queue aligned to the vocoder hop.
void test_fifo_pad_to_multiple() {
    std::array<UnitProsodyFifo::Frame, 4> frames{};
    {   // already aligned, empty, and n <= 1: nothing appended
        UnitProsodyFifo fifo(64, kVocab, kEos);
        CHECK(fifo.pad_to_multiple(4) == 0 && fifo.empty());
        std::array<std::int64_t, 4> units{};
        std::array<float, 16> prosody{};
        make_chunk(0, units, prosody);
        CHECK(fifo.push_chunk(units, prosody).pushed == 4);
        CHECK(fifo.pad_to_multiple(4) == 0 && fifo.size() == 4);
        CHECK(fifo.pad_to_multiple(2) == 0 && fifo.size() == 4);
        CHECK(fifo.pad_to_multiple(1) == 0 && fifo.pad_to_multiple(0) == 0 && fifo.size() == 4);
        const std::array<std::int64_t, 4> partial{5, 6, kEos, kPad};
        CHECK(fifo.push_chunk(partial, prosody).pushed == 2);   // size 6
        CHECK(fifo.pad_to_multiple(1) == 0 && fifo.size() == 6); // n <= 1 never pads
        CHECK(fifo.pad_to_multiple(0) == 0 && fifo.size() == 6);
    }
    {   // [u0, u1] -> 2 pads copying u1's unit / z_f0 / z_energy
        UnitProsodyFifo fifo(64, kVocab, kEos);
        std::array<std::int64_t, 4> units{7, 9, kEos, kPad};
        std::array<float, 16> prosody{};
        for (std::size_t j = 0; j < 4; ++j) {
            prosody[4 * j + 0] = 10.0f + static_cast<float>(j);     // z_f0
            prosody[4 * j + 1] = 0.8f;                              // voicing
            prosody[4 * j + 2] = 0.2f;                              // aperiodicity
            prosody[4 * j + 3] = 20.0f + static_cast<float>(j);     // z_energy
        }
        const auto r = fifo.push_chunk(units, prosody);
        CHECK(r.pushed == 2 && r.eos);
        CHECK(fifo.pad_to_multiple(4) == 2 && fifo.size() == 4);
        CHECK(fifo.pop_hop(4, frames.data()));
        CHECK(frames[0].unit == 7 && !frames[0].pad && frames[0].voicing == 0.8f);
        CHECK(frames[1].unit == 9 && !frames[1].pad && frames[1].aperiodicity == 0.2f);
        for (std::size_t i = 2; i < 4; ++i) {
            CHECK(frames[i].unit == 9);                             // newest real unit
            CHECK(frames[i].logf0_z == 11.0f);                      // newest real z_f0
            CHECK(frames[i].energy_z == 21.0f);                     // newest real z_energy
            CHECK(frames[i].voicing == 0.0f);
            CHECK(frames[i].aperiodicity == 1.0f);
            CHECK(frames[i].pad);
        }
        CHECK(fifo.empty());
    }
    {   // n = 4 pads 3 (rem 1) and 1 (rem 3), counting what is already queued
        UnitProsodyFifo fifo(64, kVocab, kEos);
        std::array<float, 16> prosody{};
        const std::array<std::int64_t, 4> one{3, kEos, kPad, kPad};
        const std::array<std::int64_t, 4> three{4, 5, 6, kEos};
        CHECK(fifo.push_chunk(one, prosody).pushed == 1);
        CHECK(fifo.pad_to_multiple(4) == 3 && fifo.size() == 4);
        CHECK(fifo.push_chunk(three, prosody).pushed == 3);
        CHECK(fifo.pad_to_multiple(4) == 1 && fifo.size() == 8);
        CHECK(fifo.pad_to_multiple(4) == 0);                        // idempotent once aligned
    }
}

void test_fifo_pad_overflow_and_pop_flags() {
    {
        UnitProsodyFifo fifo(6, kVocab, kEos);
        std::array<std::int64_t, 4> units{};
        std::array<float, 16> prosody{};
        make_chunk(0, units, prosody);
        CHECK(fifo.push_chunk(units, prosody).pushed == 4);
        const std::array<std::int64_t, 4> more{1, 2, 3, 4};
        CHECK(fifo.push_chunk(more, prosody).overflow);             // needs 4, 2 free
        CHECK(fifo.overflows() == 1);
        // capacity 6, 5 queued would need 3 pads for n = 4: build 5 queued.
        UnitProsodyFifo tight(6, kVocab, kEos);
        const std::array<std::int64_t, 4> five{1, 2, 3, 4};
        CHECK(tight.push_chunk(five, prosody).pushed == 4);
        const std::array<std::int64_t, 4> one{8, kEos, kPad, kPad};
        CHECK(tight.push_chunk(one, prosody).pushed == 1);          // 5 queued, 1 free
        CHECK(tight.overflows() == 0);
        CHECK(tight.pad_to_multiple(4) == 0);                       // needs 3 pads, 1 free
        CHECK(tight.overflows() == 1 && tight.size() == 5);
    }
    {   // pop_hop(4, frames) reports the pad flags {false, false, true, true}
        UnitProsodyFifo fifo(64, kVocab, kEos);
        std::array<float, 16> prosody{};
        const std::array<std::int64_t, 4> units{1, 2, kEos, kPad};
        CHECK(fifo.push_chunk(units, prosody).pushed == 2);
        CHECK(fifo.pad_to_multiple(4) == 2);
        std::array<UnitProsodyFifo::Frame, 4> frames{};
        CHECK(fifo.pop_hop(4, frames.data()));
        CHECK(!frames[0].pad && !frames[1].pad && frames[2].pad && frames[3].pad);
        // The pad flag and the copied frame also survive a ring wrap.
        UnitProsodyFifo ring(6, kVocab, kEos);
        const std::array<std::int64_t, 4> full{1, 2, 3, 4};
        CHECK(ring.push_chunk(full, prosody).pushed == 4);
        CHECK(ring.pop_hop(4, frames.data()));                      // head = 4, empty
        CHECK(ring.push_chunk(units, prosody).pushed == 2);         // slots 4, 5
        CHECK(ring.pad_to_multiple(4) == 2);                        // pads wrap to slots 0, 1
        CHECK(ring.pop_hop(4, frames.data()));
        CHECK(frames[0].unit == 1 && frames[1].unit == 2);
        CHECK(!frames[0].pad && !frames[1].pad && frames[2].pad && frames[3].pad);
        CHECK(frames[2].unit == 2 && frames[3].unit == 2);
    }
}

// BUG-028 sequence: a partial <EOS> chunk is completed with pads, popped as
// one hop, and nothing leaks into the next utterance's first hop.
void test_fifo_bug028_sequence() {
    UnitProsodyFifo fifo(64, kVocab, kEos);
    std::array<float, 16> prosody{};
    for (std::size_t j = 0; j < 4; ++j) prosody[4 * j + 1] = 1.0f;
    const std::array<std::int64_t, 4> first{10, 11, kEos, kPad};
    CHECK(fifo.push_chunk(first, prosody).pushed == 2);
    CHECK(fifo.pad_to_multiple(4) == 2);
    std::array<UnitProsodyFifo::Frame, 4> hop{};
    CHECK(fifo.pop_hop(4, hop.data()));
    CHECK(hop[0].unit == 10 && hop[1].unit == 11);
    CHECK(fifo.empty());                                            // no stranded frame

    const std::array<std::int64_t, 4> second{20, 21, 22, 23};
    CHECK(fifo.push_chunk(second, prosody).pushed == 4);
    CHECK(fifo.pad_to_multiple(4) == 0);
    CHECK(fifo.pop_hop(4, hop.data()));
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(hop[i].unit == 20 + static_cast<std::int64_t>(i));
        CHECK(!hop[i].pad);
    }
    CHECK(fifo.empty());
}

// Stage 09 `adaptive_policy` live switch: keeps the utterance (t_src, t_tgt,
// phase) and restarts the readiness bank from 0, so a stale bank cannot
// release a write. Mirrors python_training/tests/test_waitk_policy.py::
// test_set_adaptive_mode_live_switch.
// Kills: set_adaptive_mode() not zeroing readiness_ (the banked 1.5 would
// write at lag k), resetting the utterance state / phase / counters on a
// switch, or ignoring adaptive_extra_wait after the switch.
void test_set_adaptive_mode_live_switch() {
    const auto banked = [] {
        WaitKPolicy p(adaptive_config(3));
        // 0.75 per chunk banks 1.5 (readiness is private: checked by behaviour below)
        CHECK(drive(p, true, kChunkMs, 0.75f).decision == Decision::kRead);
        CHECK(drive(p, true, kChunkMs, 0.75f).decision == Decision::kRead);
        CHECK(p.t_src() == 2 && p.t_tgt() == 0);
        return p;
    };

    {   // control: the untouched bank writes at lag k
        WaitKPolicy ref = banked();
        const StepOut out = drive(ref, true);
        CHECK(out.decision == Decision::kWrite && out.writes == 1);
    }
    {   // off -> on: bank zeroed, utterance kept, hold back until lag >= k + extra (3 + 2)
        WaitKPolicy p = banked();
        CHECK(p.utterance_active() && !p.flushing());
        p.set_adaptive_mode(false);
        CHECK(!p.config().adaptive_mode);
        CHECK(p.t_src() == 2 && p.t_tgt() == 0 && p.utterance_active() && !p.flushing());
        p.set_adaptive_mode(true);
        CHECK(p.config().adaptive_mode);
        CHECK(p.t_src() == 2 && p.t_tgt() == 0 && p.utterance_active() && !p.flushing());
        CHECK(p.flushes_completed() == 0 && p.flush_resumes() == 0 && p.k() == 3);
        StepOut a = drive(p, true);                          // lag 3, empty bank
        StepOut b = drive(p, true);                          // lag 4
        StepOut c = drive(p, true);                          // lag 5 = k + adaptive_extra_wait
        CHECK(a.decision == Decision::kRead && a.writes == 0);
        CHECK(b.decision == Decision::kRead && b.writes == 0);
        CHECK(c.decision == Decision::kWrite && c.writes == 1);
        CHECK(p.t_src() == 5 && p.t_tgt() == 1);
    }
    {   // adaptive -> strict: the stale bank is gone and lag >= k writes at once
        WaitKPolicy q = banked();
        q.set_adaptive_mode(false);
        const StepOut out = drive(q, true);
        CHECK(out.decision == Decision::kWrite && out.writes == 1);
        CHECK(q.t_src() == 3 && q.t_tgt() == 1);
        // and strict mode never banks: a later switch back starts empty again
        q.set_adaptive_mode(true);
        CHECK(drive(q, true).writes == 0);                   // t_src 4, t_tgt 1: lag 3 < k + extra (5), empty bank
    }
    {   // a flush in progress survives the switch untouched
        WaitKPolicy f(adaptive_config(2));
        for (int i = 0; i < 3; ++i) drive(f, true, kChunkMs, 1.0f);
        f.on_chunk_energy(false, 500);                       // pause > tail_flush_ms: flushing
        CHECK(f.flushing());
        const std::int64_t src = f.t_src(), tgt = f.t_tgt();
        f.set_adaptive_mode(false);
        CHECK(f.flushing() && f.t_src() == src && f.t_tgt() == tgt && f.flushes_completed() == 0);
    }
}

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--replay") == 0) return replay_file(argv[2]);
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_waitk_policy <tests/data/waitk> | --replay <file>\n");
        return 2;
    }

    test_constants_and_defaults();
    test_config_is_sanitized();
    test_classic_waitk_sequence();
    test_legacy_calls_without_vad();
    test_leading_silence_is_not_counted();
    test_short_pause_vs_tail_flush();
    test_tail_flush_threshold_is_strict();
    test_flush_is_bounded_by_max_flush_steps();
    test_flush_ratio_bound();
    test_early_eos_terminates_flush();
    test_eos_while_silent_skips_the_flush();
    test_eos_while_speaking_is_ignored();
    test_speech_after_eos_in_short_pause();
    test_speech_resumption_during_flush();
    test_hot_set_k_clamping();
    test_reset_keeps_k_and_counters();
    test_stray_notifications_are_ignored();
    test_silence_counter_saturates();
    test_frame_to_chunk_aggregation();
    test_adaptive_readiness();
    test_write_lag_invariant();
    test_policy_never_allocates();
    test_golden_trajectories(argv[1]);

    test_fifo_defaults();
    test_fifo_rate_bridging();
    test_fifo_underflow_returns_false();
    test_fifo_skips_trailing_special_units();
    test_fifo_overflow_is_all_or_nothing();
    test_fifo_short_prosody_span();
    test_fifo_pad_to_multiple();
    test_fifo_pad_overflow_and_pop_flags();
    test_fifo_bug028_sequence();
    test_set_adaptive_mode_live_switch();
    return test::finish("waitk_policy");
}
