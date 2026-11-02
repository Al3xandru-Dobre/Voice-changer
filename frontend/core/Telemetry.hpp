// Telemetry — the lock-free counters the UI reads, plus a small fixed ring for
// sparklines. Everything here is plain data; the worker and audio callback
// publish through atomics owned by EngineHost / DuplexRouter.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace fluxvoice::frontend {

struct TelemetrySnapshot {
    std::uint64_t frames           = 0;
    std::uint64_t input_overruns   = 0;
    std::uint64_t output_underruns = 0;
    std::uint64_t output_drops     = 0;
    /// Samples held between capture and playback by conservation:
    /// underruns - overruns - drops (MEMORY.md §2). Exact, never estimated
    /// from queue occupancy.
    std::int64_t  queued_samples   = 0;
    std::uint32_t sample_rate      = 0;
    std::size_t   rss_bytes        = 0;

    // Worker-side meters (dBFS; -120 means silence / not running).
    float input_peak_dbfs  = -120.0f;
    float input_rms_dbfs   = -120.0f;
    float output_peak_dbfs = -120.0f;
    float output_rms_dbfs  = -120.0f;

    /// Parameter snapshots pushed by the control thread vs. consumed by the
    /// worker; equal when the worker has caught up.
    std::uint64_t params_pushed  = 0;
    std::uint64_t params_applied = 0;

    double queued_ms() const {
        return sample_rate == 0 ? 0.0 : 1000.0 * static_cast<double>(queued_samples) / sample_rate;
    }

    /// Same format as src/main.cpp's `[stats]` line, parsed by tools/analyze_soak.py.
    std::string stats_line(double t_seconds) const;
};

/// Fixed-capacity ring of floats for sparklines (newest last).
template <std::size_t N>
class History {
public:
    static constexpr std::size_t capacity = N;

    void push(float v) {
        buf_[head_] = v;
        head_       = (head_ + 1) % N;
        if (count_ < N) ++count_;
    }

    std::size_t size() const { return count_; }

    /// Oldest-first copy, contiguous for plotting.
    std::array<float, N> ordered() const {
        std::array<float, N> out{};
        const std::size_t start = count_ < N ? 0 : head_;
        for (std::size_t i = 0; i < count_; ++i) out[i] = buf_[(start + i) % N];
        return out;
    }

    float latest() const { return count_ == 0 ? 0.0f : buf_[(head_ + N - 1) % N]; }
    float max() const {
        float m = 0.0f;
        for (std::size_t i = 0; i < count_; ++i) m = buf_[i] > m ? buf_[i] : m;
        return m;
    }

    void clear() { head_ = 0; count_ = 0; }

private:
    std::array<float, N> buf_{};
    std::size_t head_  = 0;
    std::size_t count_ = 0;
};

} // namespace fluxvoice::frontend
