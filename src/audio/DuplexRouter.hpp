// DuplexRouter — the real-time half of AudioEngine.
//
// Owns the two SPSC queues and the glitch counters, and implements the body of
// the miniaudio data callback. It has no miniaudio dependency, so the
// real-time path can be unit-tested (allocation-free, zero-fill on underrun,
// arbitrary frame counts) without an audio device.
//
// Thread roles (SPSC means SPSC):
//   in_q_  — producer: audio callback (process)   consumer: worker (read_input)
//   out_q_ — producer: worker (write_output)      consumer: audio callback (process)
#pragma once

#include <readerwriterqueue.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

// The queues are cache-line aligned on purpose (no false sharing between
// producer and consumer); MSVC's C4324 "structure was padded" reports exactly
// that padding.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

class DuplexRouter {
public:
    // ~4 s of mono 16 kHz audio — a safety limit, not a target delay.
    static constexpr std::size_t kQueueCapacity = 1 << 16;

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "RT-thread counters must be lock-free on this target");
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "RT-thread flags must be lock-free on this target");

    explicit DuplexRouter(std::size_t queue_capacity = kQueueCapacity)
        : in_q_(queue_capacity), out_q_(queue_capacity) {}

    DuplexRouter(const DuplexRouter&)            = delete;
    DuplexRouter& operator=(const DuplexRouter&) = delete;

    void set_passthrough(bool on) { passthrough_.store(on, std::memory_order_relaxed); }
    bool passthrough() const      { return passthrough_.load(std::memory_order_relaxed); }

    // ------------------------------------------------------------------
    // Real-time side — called from the audio callback only.
    // No allocation, no locks, no logging, no blocking.
    // ------------------------------------------------------------------

    /// Handle one callback's worth of mono samples. `in` may be null (no
    /// capture data, or playback-only callback); `out` may be null
    /// (capture-only or native loopback callback, M1.5 stages 05–06). `n` is
    /// whatever the backend delivered — never assume it equals the requested
    /// period.
    void process(float* out, const float* in, std::size_t n) {
        if (!out) {
            if (!in) return;
            capture_frames_processed_.fetch_add(n, std::memory_order_relaxed);
            if (!passthrough_.load(std::memory_order_relaxed)) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!in_q_.try_enqueue(in[i])) {
                        input_overruns_.fetch_add(n - i, std::memory_order_relaxed);
                        break;
                    }
                }
            }
            return;
        }

        frames_processed_.fetch_add(n, std::memory_order_relaxed);

        if (passthrough_.load(std::memory_order_relaxed)) {
            if (in) std::memcpy(out, in, n * sizeof(float));
            else    std::memset(out, 0, n * sizeof(float));
            return;
        }

        // Capture → input queue (drop + count on overflow).
        if (in) {
            for (std::size_t i = 0; i < n; ++i) {
                if (!in_q_.try_enqueue(in[i])) {
                    input_overruns_.fetch_add(n - i, std::memory_order_relaxed);
                    break;
                }
            }
        }

        // Output queue → device (zero-fill on underrun — never block).
        std::size_t got = 0;
        while (got < n && out_q_.try_dequeue(out[got])) ++got;
        if (got < n) {
            std::memset(out + got, 0, (n - got) * sizeof(float));
            output_underruns_.fetch_add(n - got, std::memory_order_relaxed);
        }
    }

    // ------------------------------------------------------------------
    // Worker side — call from the (single) inference/worker thread only.
    // ------------------------------------------------------------------

    /// Drain up to `max_samples` captured samples. Returns count read.
    std::size_t read_input(float* dst, std::size_t max_samples) {
        std::size_t n = 0;
        while (n < max_samples && in_q_.try_dequeue(dst[n])) ++n;
        return n;
    }

    /// Push processed samples toward the output device. Returns how many
    /// fit; the rest are dropped by the caller's choice (never blocks).
    std::size_t write_output(const float* src, std::size_t num_samples) {
        std::size_t n = 0;
        while (n < num_samples && out_q_.try_enqueue(src[n])) ++n;
        return n;
    }

    /// Push processed samples while bounding queued-audio latency (M1.5 stage
    /// 05). When `max_queued_samples > 0`, enqueues only up to the remaining
    /// headroom so `out_q_` occupancy never grows past `max_queued_samples`;
    /// the caller counts `num_samples - written` in `output_drops` so the
    /// conservation identity remains exact.
    std::size_t write_output_bounded(const float* src, std::size_t num_samples, std::size_t max_queued_samples) {
        if (max_queued_samples == 0) return write_output(src, num_samples);
        const std::size_t approx = out_q_.size_approx();
        if (approx >= max_queued_samples) return 0;
        const std::size_t limit = std::min(num_samples, max_queued_samples - approx);
        return write_output(src, limit);
    }

    /// Drain and discard all buffered samples from both SPSC queues (M1.5
    /// stage 05 switch transition).
    /// Control-thread only: MUST be called only after the audio callback and
    /// worker thread are both quiesced/stopped (preserves SPSC single-consumer
    /// discipline).
    std::size_t discard_queues() {
        std::size_t discarded = 0;
        float dummy = 0.0f;
        while (in_q_.try_dequeue(dummy)) ++discarded;
        while (out_q_.try_dequeue(dummy)) ++discarded;
        return discarded;
    }

    // ------------------------------------------------------------------
    // Telemetry — atomic, safe to read from any thread.
    // ------------------------------------------------------------------
    std::uint64_t input_overruns()   const { return input_overruns_.load(std::memory_order_relaxed); }
    std::uint64_t output_underruns() const { return output_underruns_.load(std::memory_order_relaxed); }
    std::uint64_t frames_processed() const {
        const std::uint64_t out_frames = frames_processed_.load(std::memory_order_relaxed);
        return out_frames > 0 ? out_frames
                              : capture_frames_processed_.load(std::memory_order_relaxed);
    }
    std::uint64_t capture_frames_processed() const {
        return capture_frames_processed_.load(std::memory_order_relaxed);
    }
    std::size_t   input_queued_approx()  const { return in_q_.size_approx(); }
    std::size_t   output_queued_approx() const { return out_q_.size_approx(); }

private:
    std::atomic<bool>     passthrough_{false};
    std::atomic<std::uint64_t> input_overruns_{0};
    std::atomic<std::uint64_t> output_underruns_{0};
    std::atomic<std::uint64_t> frames_processed_{0};
    std::atomic<std::uint64_t> capture_frames_processed_{0};

    moodycamel::ReaderWriterQueue<float> in_q_;
    moodycamel::ReaderWriterQueue<float> out_q_;
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
