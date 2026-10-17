// LatencyProbe — marker signal and detector for measuring an audio route's
// delay (Milestone 1 / Stage 06). Used by tools/fv_probe.cpp.
//
// The probe plays a short chirp every `interval` samples and records what comes
// back on the same duplex stream, so playback and recording share one sample
// clock. Marker k starts at sample first + k*interval; the detector looks for
// it in the recording's window [first + k*interval, first + (k+1)*interval)
// and reports the lag of the best normalized cross-correlation, refined to a
// fraction of a sample by parabolic interpolation.
//
// So the route's delay must stay below `interval - marker length`, or a marker
// lands in the next window. The chirp (500 Hz – 3.5 kHz) passes FluxVoice's
// 16 kHz path; the resampler's filter delay is then part of the measured
// latency, as it should be.
//
// Pure DSP, no audio device: unit-tested in tests/test_latency_probe.cpp.
// Allocation-free after construction; MarkerDetector runs on the consumer
// thread, never in an audio callback.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace probe {

constexpr double kPi = 3.141592653589793;

/// When the markers are played, in samples of the probe's stream.
struct MarkerSchedule {
    uint64_t first    = 0;  // start of marker 0
    uint64_t interval = 0;  // between marker starts; also the search window
    size_t   count    = 0;

    uint64_t start(size_t k) const { return first + k * interval; }
    /// Samples the probe must play and record to cover every window.
    uint64_t total() const { return first + count * interval; }
};

/// Hann-windowed linear chirp from f0 to f1 Hz, `len` samples at `rate`.
inline std::vector<float> make_chirp(double rate, size_t len, double f0, double f1, double amp) {
    std::vector<float> c(len);
    const double dur = static_cast<double>(len) / rate;
    for (size_t i = 0; i < len; ++i) {
        const double t = static_cast<double>(i) / rate;
        const double phase = 2.0 * kPi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t);
        const double w = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                               static_cast<double>(len - 1)));
        c[i] = static_cast<float>(amp * w * std::sin(phase));
    }
    return c;
}

/// The marker stream at sample `n`: the chirp inside a marker, 0 elsewhere.
/// Real-time safe (no allocation, no locks).
inline float marker_sample(const std::vector<float>& marker, const MarkerSchedule& s, uint64_t n) {
    if (n < s.first) return 0.0f;
    const uint64_t k = (n - s.first) / s.interval;
    const uint64_t off = (n - s.first) % s.interval;
    return (k < s.count && off < marker.size()) ? marker[static_cast<size_t>(off)] : 0.0f;
}

struct Detection {
    bool   found = false;
    double lag   = 0.0;  // samples after the marker's start (fractional)
    double score = 0.0;  // normalized cross-correlation at the lag, -1 … 1
};

/// Finds each scheduled marker in a recording fed in arbitrary-sized pieces.
class MarkerDetector {
public:
    /// `min_score`: weakest correlation accepted as the marker. Noise alone
    /// stays far below 0.5 for a 20 ms chirp.
    MarkerDetector(std::vector<float> marker, MarkerSchedule sched, double min_score = 0.5)
        : marker_(std::move(marker)), sched_(sched), min_score_(min_score),
          win_(static_cast<size_t>(sched.interval)),
          score_(static_cast<size_t>(sched.interval) - marker_.size() + 1),
          detections_(sched.count) {
        for (float m : marker_) marker_energy_ += static_cast<double>(m) * m;
    }

    /// Recorded samples, consecutive from sample 0 of the stream.
    void push(const float* x, size_t n) {
        for (size_t i = 0; i < n; ++i, ++pos_) {
            if (next_ >= sched_.count) return;
            const uint64_t ws = sched_.start(next_);
            if (pos_ < ws) continue;
            win_[static_cast<size_t>(pos_ - ws)] = x[i];
            if (pos_ - ws + 1 == sched_.interval) detections_[next_++] = analyse();
        }
    }

    /// True once every marker's window has been analysed.
    bool done() const { return next_ >= sched_.count; }
    size_t analysed() const { return next_; }
    const std::vector<Detection>& detections() const { return detections_; }

private:
    Detection analyse() {
        const size_t len = marker_.size();
        const size_t lags = score_.size();

        double energy = 0.0;  // of win_[d, d + len), slid along with d
        for (size_t j = 0; j < len; ++j) energy += static_cast<double>(win_[j]) * win_[j];

        size_t best = 0;
        for (size_t d = 0; d < lags; ++d) {
            double c = 0.0;
            for (size_t j = 0; j < len; ++j) c += static_cast<double>(win_[d + j]) * marker_[j];
            score_[d] = energy > 1e-12 ? c / std::sqrt(energy * marker_energy_) : 0.0;
            if (score_[d] > score_[best]) best = d;
            if (d + 1 < lags) {
                energy += static_cast<double>(win_[d + len]) * win_[d + len] -
                          static_cast<double>(win_[d]) * win_[d];
                energy = std::max(energy, 0.0);
            }
        }

        Detection r;
        r.score = score_[best];
        r.found = r.score >= min_score_;
        r.lag = static_cast<double>(best);
        if (best > 0 && best + 1 < lags) {
            const double y0 = score_[best - 1], y1 = score_[best], y2 = score_[best + 1];
            const double denom = y0 - 2.0 * y1 + y2;
            if (denom < 0.0) r.lag += 0.5 * (y0 - y2) / denom;
        }
        return r;
    }

    std::vector<float> marker_;
    double             marker_energy_ = 0.0;
    MarkerSchedule     sched_;
    double             min_score_;
    std::vector<float> win_;    // current marker's window
    std::vector<double> score_; // correlation per lag
    std::vector<Detection> detections_;
    uint64_t pos_  = 0;         // stream index of the next pushed sample
    size_t   next_ = 0;         // marker whose window is being filled
};

/// Statistics over the found markers, in samples.
struct Summary {
    size_t found = 0, missed = 0;
    double min = 0, median = 0, p95 = 0, max = 0;
    /// Least-squares change of the lag per minute of stream time: a route
    /// whose clocks drift apart shows a steady slope (or a sawtooth).
    double slope_per_min = 0;
};

inline Summary summarize(const std::vector<Detection>& dets, const MarkerSchedule& s, double rate) {
    Summary r;
    std::vector<double> lags, t;
    for (size_t k = 0; k < dets.size(); ++k) {
        if (!dets[k].found) { ++r.missed; continue; }
        lags.push_back(dets[k].lag);
        t.push_back(static_cast<double>(s.start(k)) / rate / 60.0);
    }
    r.found = lags.size();
    if (lags.empty()) return r;

    if (lags.size() >= 2) {
        double mt = 0, ml = 0;
        for (size_t i = 0; i < lags.size(); ++i) { mt += t[i]; ml += lags[i]; }
        mt /= static_cast<double>(lags.size());
        ml /= static_cast<double>(lags.size());
        double num = 0, den = 0;
        for (size_t i = 0; i < lags.size(); ++i) {
            num += (t[i] - mt) * (lags[i] - ml);
            den += (t[i] - mt) * (t[i] - mt);
        }
        r.slope_per_min = den > 0 ? num / den : 0;
    }

    std::sort(lags.begin(), lags.end());
    const size_t n = lags.size();
    r.min    = lags.front();
    r.max    = lags.back();
    r.median = n % 2 ? lags[n / 2] : 0.5 * (lags[n / 2 - 1] + lags[n / 2]);
    // Nearest-rank percentile.
    r.p95    = lags[static_cast<size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1];
    return r;
}

} // namespace probe
