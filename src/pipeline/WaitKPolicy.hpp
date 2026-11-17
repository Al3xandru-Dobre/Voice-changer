// WaitKPolicy — simultaneous-translation state machine (Mode B, Milestone 3 /
// Stage 04, Documents/Milestone 3/04_waitk_policy_streaming_state_machine.md).
//
// Pure integer/float logic: no ONNX, no allocation, no locks. Owned by the
// inference worker thread. python_training/fvml/waitk_policy.py is its
// bit-exact twin (same states, same order of operations, same float32
// arithmetic); tests/data/waitk/golden_trajectories.txt pins the two. Change a
// rule in both files at once.
//
// One policy step is one translation chunk (chunk_frames = 4 frames of 20 ms =
// 80 ms), so k in [1..10] is 80..800 ms of initial wait. The caller's order
// inside a step:
//
//   policy.on_chunk_energy(speech, elapsed_ms);      // VAD verdict for the chunk
//   if (policy.source_wanted()) {                    // not idle, not flushing
//       run translator_enc;
//       policy.notify_source_unit();
//       policy.on_source_readiness(expected_target_chunks);      // adaptive only, [0, 2]
//   }
//   const auto decision = policy.decide();
//   const auto before = policy.flushes_completed();
//   for (int i = policy.max_writes_this_step(); i > 0; --i) {   // 0 on kRead
//       run translator_dec (+ vocoder);
//       policy.notify_target_unit(is_eos);
//       if (policy.flushes_completed() != before) break;        // drained: zero the KV caches
//   }
//
// Phases:
//   idle      No utterance. Leading silence is not counted (it can never
//             trigger a flush) and source chunks are not wanted.
//   active    kRead while t_src - t_tgt < k, else kWrite. Silence accumulates;
//             a pause longer than tail_flush_ms starts the tail flush.
//   flushing  Decoder-only kFlush steps until <EOS>, the flush bound
//             (t_tgt >= ceil(max_flush_ratio * t_src)) or max_flush_steps, then
//             reset() to idle and ++flushes_completed(). Speech during the
//             flush returns to active with (t_src, t_tgt) untouched, so the
//             caller keeps its KV caches.
//
// A caller that never reports energy has no VAD: notify_source_unit() then
// opens the utterance itself (the pre-Stage-04 behaviour).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

class WaitKPolicy {
public:
    enum class Decision { kRead, kWrite, kFlush };

    static constexpr std::uint32_t kTailFlushMs = 400;
    static constexpr int kMinK = 1;
    static constexpr int kMaxK = 10;
    static constexpr int kChunkFrames = 4;   // 20 ms frames per translation chunk
    static constexpr int kFrameMs = 20;      // 50 Hz

    struct Config {
        int k = 3;                            // chunks of lag before the first write, 1..10
        std::uint32_t tail_flush_ms = kTailFlushMs;
        float max_flush_ratio = 1.5f;         // flush bound: t_tgt >= ceil(ratio * t_src); 1..4
        int max_flush_steps = 32;             // hard cap on decoder-only flush steps; 1..1024
        bool adaptive_mode = false;           // CTC-adaptive readiness gating
        float readiness_threshold = 0.5f;     // readiness below this counts as "blank"
        int adaptive_extra_wait = 2;          // adaptive holds back at most k + this; 0..10
        int chunk_frames = kChunkFrames;      // 1..16
        int frame_ms = kFrameMs;              // 1..1000
    };

    explicit WaitKPolicy(int k = 3) : WaitKPolicy(config_with_k(k)) {}

    explicit WaitKPolicy(const Config& config) : cfg_(sanitized(config)), k_(cfg_.k) {}

    // ---- inputs --------------------------------------------------------

    /// Feed the chunk's speech/silence verdict, once per step, before decide().
    void on_chunk_energy(bool speech_detected, std::uint32_t elapsed_ms) {
        vad_seen_ = true;
        if (speech_detected) {
            silence_ms_ = 0;
            if (phase_ == Phase::kIdle) {
                phase_ = Phase::kActive;
            } else if (phase_ == Phase::kFlushing) {
                // Mid-flush resumption: the lag carries over.
                phase_ = Phase::kActive;
                flush_steps_ = 0;
                ++flush_resumes_;
            }
            if (eos_seen_) {
                eos_seen_ = false;
                t_tgt_ = std::max(t_tgt_, t_src_);
            }
            return;
        }
        if (phase_ == Phase::kIdle) return;   // leading silence never counts
        const std::uint32_t room = std::numeric_limits<std::uint32_t>::max() - silence_ms_;
        silence_ms_ += std::min(elapsed_ms, room);
        if (phase_ == Phase::kActive && silence_ms_ > cfg_.tail_flush_ms) begin_flush();
    }

    /// 50 Hz frame -> chunk aggregation for callers that run 20 ms hops.
    /// Returns true on the frame that closes a chunk, after forwarding the
    /// chunk verdict (speech if any of its frames had speech).
    bool on_frame_energy(bool speech_detected) {
        frame_speech_ = frame_speech_ || speech_detected;
        if (++frame_count_ < cfg_.chunk_frames) return false;
        const bool speech = frame_speech_;
        frame_count_ = 0;
        frame_speech_ = false;
        on_chunk_energy(speech, chunk_ms());
        return true;
    }

    /// One source chunk was consumed by translator_enc. Ignored while idle
    /// (with a VAD) and while flushing: check source_wanted() first.
    void notify_source_unit() {
        if (phase_ == Phase::kIdle) {
            if (vad_seen_) return;
            phase_ = Phase::kActive;
        } else if (phase_ == Phase::kFlushing) {
            return;
        }
        ++t_src_;
    }

    /// Adaptive mode: expected number of target chunks the newest source
    /// chunk completed, in [0, 2] (a plain probability in [0, 1] is fine, but
    /// then a step can only hold a write back, never write twice).
    /// Sub-threshold values are discarded; at most two chunks are banked.
    void on_source_readiness(float expected_chunks) {
        if (phase_ != Phase::kActive) return;
        if (!(expected_chunks >= cfg_.readiness_threshold)) return;   // NaN-safe
        readiness_ = std::min(readiness_ + std::min(expected_chunks, 2.0f), 2.0f);
    }

    /// One target chunk was emitted by translator_dec. `is_eos` ends a flush
    /// at once; during a pause it marks the utterance caught up so no flush
    /// steps run after it.
    void notify_target_unit(bool is_eos = false) {
        if (phase_ == Phase::kIdle) return;
        ++t_tgt_;
        if (cfg_.adaptive_mode) readiness_ = std::max(readiness_ - 1.0f, 0.0f);
        if (phase_ == Phase::kFlushing) {
            ++flush_steps_;
            if (is_eos || t_tgt_ >= flush_target_ || flush_steps_ >= cfg_.max_flush_steps) {
                complete();
            }
            return;
        }
        if (is_eos && silence_ms_ > 0) {
            eos_seen_ = true;
            t_tgt_ = std::max(t_tgt_, t_src_);
        }
    }

    // ---- decisions -----------------------------------------------------

    /// Decide the action for the current chunk step.
    Decision decide() const {
        if (phase_ == Phase::kFlushing) return Decision::kFlush;
        if (phase_ == Phase::kIdle || eos_seen_) return Decision::kRead;
        const std::int64_t lag = t_src_ - t_tgt_;
        if (lag < k_) return Decision::kRead;
        if (!cfg_.adaptive_mode) return Decision::kWrite;
        if (readiness_ >= 1.0f || lag >= k_ + cfg_.adaptive_extra_wait) return Decision::kWrite;
        return Decision::kRead;
    }

    /// Target chunks to decode this step: 0 on kRead, 1 on kWrite/kFlush, and
    /// 2 in adaptive mode when the backlog exceeds the target lag. Every write
    /// outside a flush happens at a lag of at least k.
    int max_writes_this_step() const {
        const Decision d = decide();
        if (d == Decision::kRead) return 0;
        if (d == Decision::kFlush || !cfg_.adaptive_mode) return 1;
        const std::int64_t lag = t_src_ - t_tgt_;
        if (lag > k_ && (readiness_ >= 2.0f || lag > k_ + cfg_.adaptive_extra_wait)) return 2;
        return 1;
    }

    // ---- observers -----------------------------------------------------

    /// True when translator_enc should consume this step's source chunk.
    bool source_wanted() const {
        return phase_ == Phase::kActive || (phase_ == Phase::kIdle && !vad_seen_);
    }

    bool utterance_active() const { return phase_ != Phase::kIdle; }
    bool flushing() const { return phase_ == Phase::kFlushing; }
    /// Pre-Stage-04 name for flushing().
    bool flush_pending() const { return flushing(); }

    /// Counts completed tail flushes; a change means the utterance was drained
    /// and the caller must zero its translator state.
    std::uint64_t flushes_completed() const { return flushes_completed_; }
    std::uint64_t flush_resumes() const { return flush_resumes_; }
    int flush_steps() const { return flush_steps_; }
    std::uint32_t silence_ms() const { return silence_ms_; }

    /// Reset the utterance state after a flush / stream restart. Keeps k, the
    /// config and the telemetry counters.
    void reset() {
        phase_ = Phase::kIdle;
        t_src_ = 0;
        t_tgt_ = 0;
        silence_ms_ = 0;
        flush_steps_ = 0;
        flush_target_ = 0;
        eos_seen_ = false;
        readiness_ = 0.0f;
        frame_count_ = 0;
        frame_speech_ = false;
    }

    int k() const { return k_; }
    void set_k(int k) { k_ = std::clamp(k, kMinK, kMaxK); }

    const Config& config() const { return cfg_; }
    std::uint32_t chunk_ms() const {
        return static_cast<std::uint32_t>(cfg_.chunk_frames * cfg_.frame_ms);
    }
    std::uint32_t wait_ms() const { return static_cast<std::uint32_t>(k_) * chunk_ms(); }

    std::int64_t t_src() const { return t_src_; }
    std::int64_t t_tgt() const { return t_tgt_; }

private:
    enum class Phase { kIdle, kActive, kFlushing };

    static Config config_with_k(int k) {
        Config c;
        c.k = k;
        return c;
    }

    static Config sanitized(Config c) {
        c.k = std::clamp(c.k, kMinK, kMaxK);
        if (!(c.max_flush_ratio >= 1.0f)) c.max_flush_ratio = 1.0f;   // also NaN
        c.max_flush_ratio = std::min(c.max_flush_ratio, 4.0f);
        c.max_flush_steps = std::clamp(c.max_flush_steps, 1, 1024);
        if (!(c.readiness_threshold >= 0.0f)) c.readiness_threshold = 0.0f;
        c.readiness_threshold = std::min(c.readiness_threshold, 1.0f);
        c.adaptive_extra_wait = std::clamp(c.adaptive_extra_wait, 0, kMaxK);
        c.chunk_frames = std::clamp(c.chunk_frames, 1, 16);
        c.frame_ms = std::clamp(c.frame_ms, 1, 1000);
        return c;
    }

    void begin_flush() {
        if (eos_seen_ || t_src_ == 0) {
            complete();
            return;
        }
        flush_target_ = static_cast<std::int64_t>(
            std::ceil(static_cast<double>(cfg_.max_flush_ratio) * static_cast<double>(t_src_)));
        if (t_tgt_ >= flush_target_) {
            complete();
            return;
        }
        phase_ = Phase::kFlushing;
        flush_steps_ = 0;
    }

    void complete() {
        reset();
        ++flushes_completed_;
    }

    Config cfg_;
    int k_;
    Phase phase_ = Phase::kIdle;
    std::int64_t t_src_ = 0;
    std::int64_t t_tgt_ = 0;
    std::int64_t flush_target_ = 0;
    std::uint32_t silence_ms_ = 0;
    int flush_steps_ = 0;
    int frame_count_ = 0;
    float readiness_ = 0.0f;
    bool frame_speech_ = false;
    bool eos_seen_ = false;
    bool vad_seen_ = false;
    std::uint64_t flushes_completed_ = 0;
    std::uint64_t flush_resumes_ = 0;
};
