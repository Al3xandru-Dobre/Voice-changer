// WaitKPolicy — simultaneous-translation state machine (Mode B, Milestone 3).
//
// Pure, dependency-free logic — unit-testable without ONNX. Semantics per
// README §3 (Stage 2):
//
//   READ : while (t_src - t_tgt) < k  → buffer source units in the encoder
//          KV-cache, emit <SILENCE/WAIT> downstream.
//   WRITE: once  (t_src - t_tgt) >= k → decode target units autoregressively.
//   FLUSH: energy below noise floor for > 400 ms → drain remaining target
//          units, then reset() for the next utterance.
#pragma once

#include <algorithm>
#include <cstdint>

class WaitKPolicy {
public:
    enum class Decision { kRead, kWrite, kFlush };

    static constexpr std::uint32_t kTailFlushMs = 400;
    static constexpr int      kMinK = 1;
    static constexpr int      kMaxK = 10;

    explicit WaitKPolicy(int k = 3) : k_(std::clamp(k, kMinK, kMaxK)) {}

    /// Decide the action for the current 80 ms step.
    Decision decide() const {
        if (flush_pending()) return Decision::kFlush;
        return (t_src_ - t_tgt_ >= k_) ? Decision::kWrite : Decision::kRead;
    }

    /// One source semantic-unit chunk arrived from the encoder.
    void notify_source_unit() { ++t_src_; }

    /// One target unit chunk was emitted by the translator.
    void notify_target_unit() { ++t_tgt_; }

    /// Feed chunk energy classification each step.
    void on_chunk_energy(bool speech_detected, std::uint32_t elapsed_ms) {
        silence_ms_ = speech_detected ? 0 : silence_ms_ + elapsed_ms;
    }

    /// True when the tail-flush condition holds: sustained low energy and
    /// there is pending work worth draining.
    bool flush_pending() const {
        return silence_ms_ > kTailFlushMs && (t_src_ > 0 || t_tgt_ > 0);
    }

    /// Reset counters after a flush / stream restart.
    void reset() {
        t_src_ = 0;
        t_tgt_ = 0;
        silence_ms_ = 0;
    }

    int k() const { return k_; }
    void set_k(int k) { k_ = std::clamp(k, kMinK, kMaxK); }

    std::int64_t t_src() const { return t_src_; }
    std::int64_t t_tgt() const { return t_tgt_; }

private:
    int k_;
    std::int64_t  t_src_ = 0;
    std::int64_t  t_tgt_ = 0;
    std::uint32_t silence_ms_ = 0;
};
