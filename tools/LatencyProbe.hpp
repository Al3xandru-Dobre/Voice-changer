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
#include <stdexcept>
#include <utility>
#include <vector>

namespace probe {

constexpr double kPi = 3.141592653589793;
constexpr double kSpeechBurstSeconds = 0.150;

enum class MarkerKind {
    Chirp,   // 20 ms linear chirp matched by waveform cross-correlation
    Speech,  // 150 ms plosive speech burst matched by log-energy envelope correlation
};

namespace detail {

struct Resonator {
    double b0 = 0.0, a1 = 0.0, a2 = 0.0, y1 = 0.0, y2 = 0.0;

    static Resonator make(double rate, double freq_hz, double bw_hz) noexcept {
        const double f = std::clamp(freq_hz, 60.0, 0.45 * rate);
        const double bw = std::clamp(bw_hz, 30.0, 0.25 * rate);
        const double r = std::exp(-kPi * bw / rate);
        Resonator res;
        res.a1 = -2.0 * r * std::cos(2.0 * kPi * f / rate);
        res.a2 = r * r;
        res.b0 = 1.0 - r;
        return res;
    }

    double step(double x) noexcept {
        const double y = b0 * x - a1 * y1 - a2 * y2;
        y2 = y1;
        y1 = y;
        return y;
    }

    void reset() noexcept { y1 = y2 = 0.0; }
};

inline double raised_gate(double t, double t0, double t1, double rise, double fall) noexcept {
    if (t <= t0 || t >= t1) return 0.0;
    if (rise > 0.0 && t < t0 + rise)
        return 0.5 * (1.0 - std::cos(kPi * (t - t0) / rise));
    if (fall > 0.0 && t > t1 - fall)
        return 0.5 * (1.0 - std::cos(kPi * (t1 - t) / fall));
    return 1.0;
}

} // namespace detail

/// When the markers are played, in samples of the probe's stream.
struct MarkerSchedule {
    std::uint64_t first    = 0;  // start of marker 0
    std::uint64_t interval = 0;  // between marker starts; also the search window
    std::size_t   count    = 0;

    std::uint64_t start(std::size_t k) const { return first + k * interval; }
    /// Samples the probe must play and record to cover every window.
    std::uint64_t total() const { return first + count * interval; }
};

/// Hann-windowed linear chirp from f0 to f1 Hz, `len` samples at `rate`.
inline std::vector<float> make_chirp(double rate, std::size_t len, double f0, double f1, double amp) {
    std::vector<float> c(len);
    const double dur = static_cast<double>(len) / rate;
    for (std::size_t i = 0; i < len; ++i) {
        const double t = static_cast<double>(i) / rate;
        const double phase = 2.0 * kPi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t);
        const double w = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                               static_cast<double>(len - 1)));
        c[i] = static_cast<float>(amp * w * std::sin(phase));
    }
    return c;
}

/// Deterministic ~150 ms plosive-vowel speech burst (/pa/) with a 20 ms silent
/// pre-onset margin, crisp 8 ms aspirated plosive attack, 100 ms voiced 3-formant
/// vowel body, and 25 ms silent post-margin so the silence-to-speech onset sits
/// cleanly inside [0, len) without sub-20-ms gaps that a 50 Hz VC model smears.
inline std::vector<float> make_speech_burst(double rate, double amp = 0.5) {
    const std::size_t len = std::max<std::size_t>(
        64, static_cast<std::size_t>(std::llround(rate * kSpeechBurstSeconds)));
    std::vector<float> out(len, 0.0f);

    auto burst1 = detail::Resonator::make(rate, 1850.0, 750.0);
    auto f1 = detail::Resonator::make(rate, 730.0, 90.0);
    auto f2 = detail::Resonator::make(rate, 1150.0, 115.0);
    auto f3 = detail::Resonator::make(rate, 2480.0, 170.0);

    std::uint32_t rng = 0x5A17C3E9u;
    auto white = [&]() noexcept {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<double>(rng >> 8) / 8388608.0 - 1.0;
    };

    const double max_harm_hz = std::min(3800.0, 0.44 * rate);
    double phase = 0.0;
    double peak = 0.0;

    for (std::size_t i = 0; i < len; ++i) {
        const double t = static_cast<double>(i) / rate;
        double sample = 0.0;

        if (t >= 0.020 && t < 0.030) {
            const double env = detail::raised_gate(t, 0.020, 0.030, 0.001, 0.004) *
                               std::exp(-(t - 0.020) / 0.0035);
            sample += 0.85 * env * burst1.step(white());
        }
        if (t >= 0.023 && t < 0.125) {
            const double u = (t - 0.023) / (0.125 - 0.023);
            const double f0 = 132.0 + 18.0 * std::sin(kPi * u);
            phase = std::fmod(phase + 2.0 * kPi * f0 / rate, 2.0 * kPi);
            const int n_harm = std::min(24, std::max(1, static_cast<int>(max_harm_hz / f0)));
            double glottal = 0.0;
            for (int h = 1; h <= n_harm; ++h)
                glottal += std::sin(static_cast<double>(h) * phase) / (1.0 + 0.10 * h * h);
            const double voiced = 1.00 * f1.step(glottal) +
                                  0.72 * f2.step(glottal) +
                                  0.36 * f3.step(glottal);
            const double env = detail::raised_gate(t, 0.023, 0.125, 0.004, 0.014) *
                               (1.0 - 0.35 * u);
            sample += env * voiced;
        }

        out[i] = static_cast<float>(sample);
        peak = std::max(peak, std::abs(sample));
    }

    const double scale = peak > 1e-12 ? amp / peak : 0.0;
    for (float& v : out) v = static_cast<float>(static_cast<double>(v) * scale);
    out.front() = 0.0f;
    out.back() = 0.0f;
    return out;
}

/// The marker stream at sample `n`: the chirp inside a marker, 0 elsewhere.
/// Real-time safe (no allocation, no locks).
inline float marker_sample(const std::vector<float>& marker, const MarkerSchedule& s, std::uint64_t n) {
    if (n < s.first) return 0.0f;
    const std::uint64_t k = (n - s.first) / s.interval;
    const std::uint64_t off = (n - s.first) % s.interval;
    return (k < s.count && off < marker.size()) ? marker[static_cast<std::size_t>(off)] : 0.0f;
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
    /// stays far below 0.5 for a 20 ms chirp and below 0.2 for the regularized
    /// speech log-energy envelope.
    MarkerDetector(std::vector<float> marker, MarkerSchedule sched, double min_score = 0.5,
                   MarkerKind kind = MarkerKind::Chirp, double rate = 48000.0)
        : marker_(std::move(marker)), sched_(sched), min_score_(min_score), kind_(kind),
          win_(static_cast<std::size_t>(sched.interval)),
          detections_(sched.count) {
        if (marker_.empty() || sched_.interval < marker_.size() || rate <= 0.0)
            throw std::invalid_argument("invalid MarkerDetector marker/schedule/rate");

        if (kind_ == MarkerKind::Chirp) {
            score_.resize(static_cast<std::size_t>(sched.interval) - marker_.size() + 1);
            for (float m : marker_) marker_energy_ += static_cast<double>(m) * m;
        } else {
            env_step_ = std::max<std::size_t>(
                1, static_cast<std::size_t>(std::llround(rate * 0.001)));
            env_win_ = std::clamp<std::size_t>(
                static_cast<std::size_t>(std::llround(rate * 0.020)),
                4, std::max<std::size_t>(4, marker_.size() / 4));
            if (marker_.size() < env_win_)
                throw std::invalid_argument("speech marker shorter than envelope window");

            hann_.resize(env_win_);
            double wsum = 0.0;
            for (std::size_t j = 0; j < env_win_; ++j) {
                hann_[j] = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(j) /
                                                 static_cast<double>(env_win_ - 1)));
                wsum += hann_[j];
            }
            for (double& w : hann_) w /= wsum;

            ref_frames_ = (marker_.size() - env_win_) / env_step_ + 1;
            const std::size_t win_frames = (win_.size() - env_win_) / env_step_ + 1;
            env_diff_ = std::clamp<std::size_t>(10, 1, std::max<std::size_t>(1, ref_frames_ / 4));
            if (win_frames < ref_frames_ || ref_frames_ <= env_diff_)
                throw std::invalid_argument("search interval too short for speech envelope");

            slice_log_.resize(ref_frames_);
            ref_env_.resize(ref_frames_ - env_diff_);
            win_energy_.resize(win_frames);
            score_.resize(win_frames - ref_frames_ + 1);

            double ref_max = 0.0;
            for (std::size_t f = 0; f < ref_frames_; ++f) {
                slice_log_[f] = frame_energy(marker_.data() + f * env_step_);
                ref_max = std::max(ref_max, slice_log_[f]);
            }
            const double floor_e = std::max(1e-7, ref_max * 0.02);
            for (double& e : slice_log_) e = std::log(std::max(e, floor_e));
            for (std::size_t f = 0; f < ref_env_.size(); ++f) {
                const double flux = std::max(0.0, slice_log_[f + env_diff_] - slice_log_[f]);
                ref_env_[f] = flux;
                marker_energy_ += flux * flux;
            }
        }
    }

    /// Recorded samples, consecutive from sample 0 of the stream.
    void push(const float* x, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i, ++pos_) {
            if (next_ >= sched_.count) return;
            const std::uint64_t ws = sched_.start(next_);
            if (pos_ < ws) continue;
            win_[static_cast<std::size_t>(pos_ - ws)] = x[i];
            if (pos_ - ws + 1 == sched_.interval)
                detections_[next_++] =
                    (kind_ == MarkerKind::Chirp) ? analyse_waveform() : analyse_envelope();
        }
    }

    /// True once every marker's window has been analysed.
    bool done() const { return next_ >= sched_.count; }
    std::size_t analysed() const { return next_; }
    const std::vector<Detection>& detections() const { return detections_; }

private:
    double frame_energy(const float* p) const noexcept {
        double acc = 0.0;
        for (std::size_t j = 0; j < env_win_; ++j) {
            const double v = static_cast<double>(p[j]);
            acc += hann_[j] * v * v;
        }
        return acc;
    }

    Detection analyse_waveform() {
        const std::size_t len = marker_.size();
        const std::size_t lags = score_.size();

        double energy = 0.0;  // of win_[d, d + len), slid along with d
        for (std::size_t j = 0; j < len; ++j) energy += static_cast<double>(win_[j]) * win_[j];

        std::size_t best = 0;
        for (std::size_t d = 0; d < lags; ++d) {
            double c = 0.0;
            for (std::size_t j = 0; j < len; ++j) c += static_cast<double>(win_[d + j]) * marker_[j];
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

    Detection analyse_envelope() {
        const std::size_t win_frames = win_energy_.size();
        for (std::size_t f = 0; f < win_frames; ++f)
            win_energy_[f] = frame_energy(win_.data() + f * env_step_);

        const std::size_t lags = score_.size();
        const std::size_t flux_len = ref_env_.size();
        const std::size_t pre_end = std::max<std::size_t>(1, std::min<std::size_t>(8, ref_frames_ / 10));
        const std::size_t body_a  = std::min<std::size_t>(28, ref_frames_ - 1);
        const std::size_t body_b  = std::max<std::size_t>(body_a + 1, std::min<std::size_t>(95, ref_frames_));
        // Positive log-energy onset flux correlates the rising speech onset
        // while ignoring trailing noise-gate hold or causal vocoder ring-out;
        // the sustained-body check rejects isolated 1 ms transient clicks.
        const double mod_reg = static_cast<double>(flux_len) * 0.015;
        std::size_t best = 0;

        for (std::size_t d = 0; d < lags; ++d) {
            double slice_max = 0.0;
            for (std::size_t f = 0; f < ref_frames_; ++f)
                slice_max = std::max(slice_max, win_energy_[d + f]);
            if (slice_max < 1e-7 || marker_energy_ <= 1e-12) {
                score_[d] = 0.0;
                continue;
            }
            const double floor_e = std::max(1e-7, slice_max * 0.02);
            for (std::size_t f = 0; f < ref_frames_; ++f)
                slice_log_[f] = std::log(std::max(win_energy_[d + f], floor_e));

            double pre_mean = 0.0, body_mean = 0.0;
            for (std::size_t f = 0; f < pre_end; ++f) pre_mean += slice_log_[f];
            pre_mean /= static_cast<double>(pre_end);
            for (std::size_t f = body_a; f < body_b; ++f) body_mean += slice_log_[f];
            body_mean /= static_cast<double>(body_b - body_a);
            if (body_mean - pre_mean < 1.5) {  // require sustained vowel body above -17 dB clamp
                score_[d] = 0.0;
                continue;
            }

            double dot = 0.0, w_energy = 0.0;
            for (std::size_t f = 0; f < flux_len; ++f) {
                const double wf = std::max(0.0, slice_log_[f + env_diff_] - slice_log_[f]);
                dot += wf * ref_env_[f];
                w_energy += wf * wf;
            }
            const double denom = std::sqrt(marker_energy_ * (w_energy + mod_reg));
            score_[d] = denom > 1e-12 ? dot / denom : 0.0;
            if (score_[d] > score_[best]) best = d;
        }

        Detection r;
        r.score = score_[best];
        r.found = r.score >= min_score_;
        double best_frame = static_cast<double>(best);
        if (best > 0 && best + 1 < lags) {
            const double y0 = score_[best - 1], y1 = score_[best], y2 = score_[best + 1];
            const double denom = y0 - 2.0 * y1 + y2;
            if (denom < 0.0) best_frame += 0.5 * (y0 - y2) / denom;
        }
        r.lag = best_frame * static_cast<double>(env_step_);
        return r;
    }

    std::vector<float> marker_;
    double             marker_energy_ = 0.0;
    MarkerSchedule     sched_;
    double             min_score_;
    MarkerKind         kind_ = MarkerKind::Chirp;
    std::size_t        env_step_ = 1;
    std::size_t        env_win_ = 4;
    std::size_t        env_diff_ = 1;
    std::size_t        ref_frames_ = 0;
    std::vector<double> hann_;
    std::vector<double> ref_env_;
    std::vector<double> slice_log_;
    std::vector<double> win_energy_;
    std::vector<float> win_;    // current marker's window
    std::vector<double> score_; // correlation per lag
    std::vector<Detection> detections_;
    std::uint64_t pos_  = 0;         // stream index of the next pushed sample
    std::size_t   next_ = 0;         // marker whose window is being filled
};

/// Statistics over the found markers, in samples.
struct Summary {
    std::size_t found = 0, missed = 0;
    double min = 0, median = 0, p95 = 0, max = 0;
    /// Least-squares change of the lag per minute of stream time: a route
    /// whose clocks drift apart shows a steady slope (or a sawtooth).
    double slope_per_min = 0;
};

inline Summary summarize(const std::vector<Detection>& dets, const MarkerSchedule& s, double rate) {
    Summary r;
    std::vector<double> lags, t;
    for (std::size_t k = 0; k < dets.size(); ++k) {
        if (!dets[k].found) { ++r.missed; continue; }
        lags.push_back(dets[k].lag);
        t.push_back(static_cast<double>(s.start(k)) / rate / 60.0);
    }
    r.found = lags.size();
    if (lags.empty()) return r;

    if (lags.size() >= 2) {
        double mt = 0, ml = 0;
        for (std::size_t i = 0; i < lags.size(); ++i) { mt += t[i]; ml += lags[i]; }
        mt /= static_cast<double>(lags.size());
        ml /= static_cast<double>(lags.size());
        double num = 0, den = 0;
        for (std::size_t i = 0; i < lags.size(); ++i) {
            num += (t[i] - mt) * (lags[i] - ml);
            den += (t[i] - mt) * (t[i] - mt);
        }
        r.slope_per_min = den > 0 ? num / den : 0;
    }

    std::sort(lags.begin(), lags.end());
    const std::size_t n = lags.size();
    r.min    = lags.front();
    r.max    = lags.back();
    r.median = n % 2 ? lags[n / 2] : 0.5 * (lags[n / 2 - 1] + lags[n / 2]);
    // Nearest-rank percentile.
    r.p95    = lags[static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1];
    return r;
}

} // namespace probe
