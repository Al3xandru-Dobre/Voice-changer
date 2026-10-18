// PitchTracker — causal YIN pitch, aperiodicity, voicing and energy per
// 20 ms frame (Milestone 2 / Stage 03).
//
// Port of PitchTracker in python_training/fvml/features.py, which the vocoder's
// training features come from: same constants, same order of operations,
// double precision inside. The golden vectors in tests/data/pitch/ pin the two
// (tests/test_pitch_tracker.cpp); change a step in both files at once.
//
// Per push(frame):
//   1. 70 Hz high-pass (biquad) into a history of filtered samples
//   2. log-energy of the frame, whitened by an EMA (RunningNorm)
//   3. YIN on a 40 ms window: the cumulative-mean-normalised difference (CMND),
//      aperiodicity = its minimum over the 55-1000 Hz lag range, f0 from the
//      first dip below 0.10 (else the global minimum) + parabolic interpolation
//   4. soft voicing = share of {0.05, 0.10, 0.15} the aperiodicity is below
//   5. voiced decision (aperiodicity < 0.40 and z_energy > -0.5) -> f0_hz, the
//      held log-f0 (relaxes towards the running mean on unvoiced frames) and
//      its whitening
//
// With lookahead L, push(frame t) returns the features of frame t - L, and the
// window is centred on that frame (no extra latency: the encoder already waits
// L frames). The first L outputs describe the zero history before the stream.
//
// Worker-thread component: every buffer is allocated in the constructor, and
// push() never allocates, locks or blocks.
#pragma once

#include "dsp/Biquad.hpp"
#include "dsp/RunningNorm.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

struct PitchTrackerConfig {
    // Defaults equal the Python reference; test_pitch_tracker checks them
    // against tests/data/pitch/header.json.
    int    sample_rate = 16000;
    int    frame_samples = 320;
    int    window_samples = 640;
    double f0_min_hz = 55.0;
    double f0_max_hz = 1000.0;
    std::array<double, 3> thresholds{0.05, 0.10, 0.15};
    double f0_threshold = 0.10;
    double voiced_max_aperiodicity = 0.40;
    double voiced_min_z_energy = -0.5;
    double hpf_hz = 70.0;
    double hpf_q = 0.70710678118654752440;
    double energy_eps = 1e-10;
    double silence_mean_square = 1e-8;
    double norm_tau_s = 4.0;
    double logf0_std_floor = 0.05;
    double energy_std_floor = 0.5;
    double hold_relax_s = 0.25;
    // Priors from LibriSpeech train-clean-100; see fvml/features.py.
    double logf0_prior_mean = 5.028;
    double logf0_prior_std = 0.421;
    double energy_prior_mean = -8.486;
    double energy_prior_std = 3.817;
    int    lookahead_frames = 0;

    int tau_min() const { return static_cast<int>(std::ceil(sample_rate / f0_max_hz)); }
    int tau_max() const { return static_cast<int>(std::floor(sample_rate / f0_min_hz)); }
    int integration_samples() const { return window_samples - tau_max(); }
    int window_offset() const { return lookahead_frames >= 1 ? frame_samples / 2 : 0; }
    int history_samples() const {
        return window_samples + lookahead_frames * frame_samples - window_offset();
    }
    double frame_seconds() const { return static_cast<double>(frame_samples) / sample_rate; }
    double norm_alpha() const { return 1.0 - std::exp(-frame_seconds() / norm_tau_s); }
    double hold_decay() const { return std::exp(-frame_seconds() / hold_relax_s); }

    bool valid() const {
        return sample_rate > 0 && frame_samples > 0 && lookahead_frames >= 0 &&
               f0_min_hz > 0.0 && f0_max_hz > f0_min_hz && tau_min() >= 2 &&
               tau_max() > tau_min() && integration_samples() >= 1 &&
               frame_samples <= window_samples &&
               std::find(thresholds.begin(), thresholds.end(), f0_threshold) != thresholds.end();
    }
};

struct PitchFrame {
    float f0_hz = 0.0f;          // 0 on unvoiced frames
    float aperiodicity = 1.0f;   // min CMND over the search range (1 for silence)
    float voicing = 0.0f;        // {0, 1/3, 2/3, 1}
    float log_energy = 0.0f;     // ln(mean square + eps) of the filtered frame
    float logf0_held = 0.0f;
    float z_logf0 = 0.0f;
    float z_energy = 0.0f;
};

/// A target voice's log-f0 / log-energy statistics (voices.json).
struct TargetVoiceStats {
    double logf0_mean = 0.0;
    double logf0_std = 1.0;
    double energy_mean = 0.0;
    double energy_std = 1.0;
};

/// Whitened source log-f0 -> the target's domain:
/// mu + scale * sigma * z + shift_st * ln2 / 12. `scale` (intonation) and
/// `shift_st` (semitones) are the Stage 09 controls; defaults 1 and 0.
inline double map_logf0_to_target(double z_logf0, const TargetVoiceStats& v, double scale = 1.0,
                                  double shift_st = 0.0) {
    constexpr double kLn2 = 0.69314718055994530942;
    return v.logf0_mean + scale * v.logf0_std * z_logf0 + shift_st * kLn2 / 12.0;
}

inline double map_energy_to_target(double z_energy, const TargetVoiceStats& v) {
    return v.energy_mean + v.energy_std * z_energy;
}

class PitchTracker {
public:
    explicit PitchTracker(const PitchTrackerConfig& cfg = {})
        : cfg_(cfg),
          tau_min_(cfg.tau_min()),
          tau_max_(cfg.tau_max()),
          integration_(static_cast<std::size_t>(cfg.integration_samples())),
          hop_(static_cast<std::size_t>(cfg.frame_samples)),
          window_(static_cast<std::size_t>(cfg.window_samples)),
          hold_decay_(cfg.hold_decay()),
          hpf_(BiquadCoefficients::high_pass(cfg.hpf_hz, cfg.sample_rate, cfg.hpf_q)),
          history_(static_cast<std::size_t>(cfg.history_samples()), 0.0),
          cmnd_(static_cast<std::size_t>(tau_max_) + 1, 1.0) {
        assert(cfg_.valid());
        reset();
    }

    /// One frame of frame_samples() input samples -> the features of frame
    /// (pushes - 1 - lookahead_frames).
    PitchFrame push(const float* frame) {
        // 1. Filter into the history (oldest first).
        std::copy(history_.begin() + static_cast<std::ptrdiff_t>(hop_), history_.end(),
                  history_.begin());
        double* tail = history_.data() + history_.size() - hop_;
        for (std::size_t i = 0; i < hop_; ++i) tail[i] = hpf_.process(static_cast<double>(frame[i]));

        const std::size_t frame_end = history_.size() - static_cast<std::size_t>(cfg_.lookahead_frames) * hop_;
        const double* cur = history_.data() + frame_end - hop_;
        const double* win = history_.data() + frame_end + static_cast<std::size_t>(cfg_.window_offset()) - window_;

        // 2. Energy.
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < hop_; ++i) sum_sq += cur[i] * cur[i];
        const double log_energy = std::log(sum_sq / static_cast<double>(hop_) + cfg_.energy_eps);
        energy_norm_.update(log_energy);
        const double z_energy = energy_norm_.z(log_energy);

        // 3-4. YIN and soft voicing.
        double f0 = 0.0;
        const double aperiodicity = yin(win, f0);
        int below = 0;
        for (double t : cfg_.thresholds) below += aperiodicity < t ? 1 : 0;
        const double voicing = static_cast<double>(below) / static_cast<double>(cfg_.thresholds.size());

        // 5. Voiced decision, held log-f0 and its whitening.
        const bool voiced = f0 > 0.0 && aperiodicity < cfg_.voiced_max_aperiodicity &&
                            z_energy > cfg_.voiced_min_z_energy;
        if (voiced) {
            logf0_held_ = std::log(f0);
            logf0_norm_.update(logf0_held_);
        } else {
            f0 = 0.0;
            const double m = logf0_norm_.mean();
            logf0_held_ = m + (logf0_held_ - m) * hold_decay_;
        }

        PitchFrame out;
        out.f0_hz = static_cast<float>(f0);
        out.aperiodicity = static_cast<float>(aperiodicity);
        out.voicing = static_cast<float>(voicing);
        out.log_energy = static_cast<float>(log_energy);
        out.logf0_held = static_cast<float>(logf0_held_);
        out.z_logf0 = static_cast<float>(logf0_norm_.z(logf0_held_));
        out.z_energy = static_cast<float>(z_energy);
        return out;
    }

    /// Silence history, filter state and whitening priors (stream restart).
    void reset() {
        hpf_.reset();
        std::fill(history_.begin(), history_.end(), 0.0);
        const double a = cfg_.norm_alpha();
        logf0_norm_ = RunningNorm(cfg_.logf0_prior_mean, cfg_.logf0_prior_std, a, cfg_.logf0_std_floor);
        energy_norm_ = RunningNorm(cfg_.energy_prior_mean, cfg_.energy_prior_std, a, cfg_.energy_std_floor);
        logf0_held_ = cfg_.logf0_prior_mean;
    }

    const PitchTrackerConfig& config() const { return cfg_; }
    std::size_t frame_samples() const { return hop_; }
    const BiquadCoefficients& hpf_coefficients() const { return hpf_.coefficients(); }

private:
    /// Aperiodicity of one window; `f0` gets the estimate (0 only for silence).
    double yin(const double* x, double& f0) {
        double energy = 0.0;
        for (std::size_t i = 0; i < window_; ++i) energy += x[i] * x[i];
        if (energy < cfg_.silence_mean_square * static_cast<double>(window_)) {
            f0 = 0.0;
            return 1.0;
        }
        // Difference function, then the CMND: d'(0) = 1,
        // d'(tau) = d(tau) * tau / sum(d(1..tau)).
        cmnd_[0] = 1.0;
        double cum = 0.0;
        for (int tau = 1; tau <= tau_max_; ++tau) {
            const double* y = x + tau;
            double s = 0.0;
            for (std::size_t j = 0; j < integration_; ++j) {
                const double diff = y[j] - x[j];
                s += diff * diff;
            }
            cum += s;
            cmnd_[static_cast<std::size_t>(tau)] = cum > 0.0 ? s * tau / cum : 1.0;
        }

        int best = tau_min_;
        for (int tau = tau_min_ + 1; tau <= tau_max_; ++tau)
            if (cmnd_[static_cast<std::size_t>(tau)] < cmnd_[static_cast<std::size_t>(best)]) best = tau;
        const double aperiodicity = cmnd_[static_cast<std::size_t>(best)];

        int tau = best;   // YIN's fallback: the global minimum
        if (aperiodicity < cfg_.f0_threshold) {
            // First dip below the threshold, then down to its local minimum.
            tau = tau_min_;
            while (cmnd_[static_cast<std::size_t>(tau)] >= cfg_.f0_threshold) ++tau;
            while (tau + 1 <= tau_max_ &&
                   cmnd_[static_cast<std::size_t>(tau) + 1] < cmnd_[static_cast<std::size_t>(tau)])
                ++tau;
        }
        double shift = 0.0;
        if (tau + 1 <= tau_max_) {
            const double a = cmnd_[static_cast<std::size_t>(tau) - 1];
            const double b = cmnd_[static_cast<std::size_t>(tau)];
            const double c = cmnd_[static_cast<std::size_t>(tau) + 1];
            const double den = a - 2.0 * b + c;
            if (den > 0.0) shift = 0.5 * (a - c) / den;
        }
        f0 = cfg_.sample_rate / (tau + shift);
        return aperiodicity;
    }

    PitchTrackerConfig cfg_;
    int tau_min_;
    int tau_max_;
    std::size_t integration_;
    std::size_t hop_;
    std::size_t window_;
    double hold_decay_;
    Biquad hpf_;
    std::vector<double> history_;
    std::vector<double> cmnd_;
    RunningNorm logf0_norm_;
    RunningNorm energy_norm_;
    double logf0_held_ = 0.0;
};
