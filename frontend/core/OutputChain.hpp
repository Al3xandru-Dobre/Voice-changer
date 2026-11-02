// OutputChain — the interim output stage run on the worker thread: smoothed
// gain, mute and a soft safety limiter, plus the peak/RMS meters the UI shows.
//
// M2 / Stage 09 places the full chain (gain -> gate -> limiter -> brightness)
// inside InferencePipeline. Until that stage lands, the frontend applies the
// two controls a user needs while the engine passes audio through — mute from
// the tray must be real, and gain must not click. The gate is deliberately
// absent: it needs the pipeline's energy tracker.
//
// Allocation-free; not real-time (worker thread), so std::tanh is acceptable.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace fluxvoice::frontend {

class OutputChain {
public:
    /// ~20 ms one-pole time constant at `sample_rate` (M2 / Stage 09 "gain:
    /// per-sample one-pole (~20 ms)").
    void configure(std::uint32_t sample_rate) {
        const double sr = sample_rate == 0 ? 16000.0 : static_cast<double>(sample_rate);
        coef_ = static_cast<float>(1.0 - std::exp(-1.0 / (0.020 * sr)));
        gain_ = target();
    }

    void set_gain_db(float db) { gain_db_ = db; }
    void set_muted(bool m) { muted_ = m; }
    void set_limiter(bool on) { limiter_ = on; }

    bool muted() const { return muted_; }
    float gain_db() const { return gain_db_; }

    /// Snap the ramp to its target (stream restart: no fade from stale state).
    void reset() { gain_ = target(); }

    /// In place. Also measures the block's output peak/RMS.
    void process(float* buf, std::size_t n) {
        const float tgt = target();
        float peak = 0.0f;
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            gain_ += coef_ * (tgt - gain_);
            float y = buf[i] * gain_;
            if (limiter_) y = soft_limit(y);
            buf[i] = y;
            const float a = std::fabs(y);
            peak = a > peak ? a : peak;
            sum_sq += static_cast<double>(y) * static_cast<double>(y);
        }
        last_peak_ = peak;
        last_rms_  = n == 0 ? 0.0f : static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
    }

    float last_peak() const { return last_peak_; }
    float last_rms() const { return last_rms_; }

    static float to_dbfs(float lin) {
        return lin <= 1e-6f ? -120.0f : 20.0f * std::log10(lin);
    }

    /// Linear peak/RMS of a block without modifying it (input meters).
    static void measure(const float* buf, std::size_t n, float& peak, float& rms) {
        peak = 0.0f;
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const float a = std::fabs(buf[i]);
            peak = a > peak ? a : peak;
            sum_sq += static_cast<double>(buf[i]) * static_cast<double>(buf[i]);
        }
        rms = n == 0 ? 0.0f : static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
    }

    /// Unity below the knee (70 % of the ceiling); above it a tanh curve that
    /// is C1-continuous at the knee and asymptotic to the -1 dBFS ceiling, so
    /// nominal levels pass untouched and nothing exceeds the ceiling.
    static float soft_limit(float x) {
        const float a = std::fabs(x);
        if (a <= kKnee) return x;
        const float y = kKnee + kRange * std::tanh((a - kKnee) / kRange);
        return x < 0.0f ? -y : y;
    }

private:
    static constexpr float kCeiling = 0.891251f;  // -1 dBFS
    static constexpr float kKnee    = 0.7f * kCeiling;
    static constexpr float kRange   = kCeiling - kKnee;

    float target() const { return muted_ ? 0.0f : std::pow(10.0f, gain_db_ / 20.0f); }

    float gain_db_   = 0.0f;
    bool  muted_     = false;
    bool  limiter_   = true;
    float coef_      = 0.0031f;  // 20 ms @ 16 kHz until configure()
    float gain_      = 1.0f;
    float last_peak_ = 0.0f;
    float last_rms_  = 0.0f;
};

} // namespace fluxvoice::frontend
