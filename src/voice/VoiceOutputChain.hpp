// VoiceOutputChain — the Stage 09 output stage run on the inference worker
// thread inside InferencePipeline (Milestone 2 / Stage 09).
//
// Order: gain -> gate -> mute -> brightness -> limiter. This is a deliberate deviation
// from the stage text's "gain -> gate -> limiter (-> brightness)": the limiter
// stays last so the -1 dBFS ceiling still holds after a +6 dB brightness tilt
// (and after any future output stage). Documented in
// Documents/Milestone 2/09_voice_controls_presets.md.
//
//   gain       per-sample one-pole, tau = 20 ms, linear target 10^(dB/20)
//   gate       per-frame decision on the *input* energy (dBFS per 20 ms
//              conditioning frame, recorded by the pipeline's pitch tracker):
//              opens at `threshold`, stays open down to `threshold` - 6 dB
//              (hysteresis), then a 5-frame (100 ms) hold before closing;
//              per-sample envelope: 2 ms linear attack, 50 ms linear release,
//              floor 0 (full mute)
//   mute       linear per-sample ramp of a 0..1 multiplier, 5 ms each way
//              (no click on mute or unmute); set_muted() is a level, kept
//              across reset() (a muted stream restarts silent). It is part
//              of the chain, so PipelineOptions::output_chain = false
//              (raw vocoder PCM, parity runs only) ignores it
//   brightness first-order tilt: one-pole lowpass `lp` at a 1 kHz pivot,
//              hp = x - lp, y = lp * 10^(-t/40) + hp * 10^(+t/40) with t the
//              smoothed tilt dB (per-sample one-pole, tau 20 ms); exactly 0 dB
//              passes samples through untouched
//   limiter    soft clip at -1 dBFS, no lookahead; the on/off toggle
//              crossfades the limited signal in/out with a 20 ms one-pole so
//              toggling on a loud signal does not click
//
// Allocation-free after construction and noexcept: every value it needs is a
// scalar member. Worker thread (not the audio callback), so std::pow/tanh are
// acceptable.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace fluxvoice {

class VoiceOutputChain {
public:
    void configure(std::uint32_t sample_rate) noexcept {
        const double sr = sample_rate == 0 ? 16000.0 : static_cast<double>(sample_rate);
        pole_coef_ = static_cast<float>(1.0 - std::exp(-1.0 / (0.020 * sr)));
        tilt_lp_coef_ =
            static_cast<float>(1.0 - std::exp(-2.0 * 3.14159265358979323846 * kTiltPivotHz / sr));
        attack_step_ = static_cast<float>(1.0 / (0.002 * sr));   // 2 ms attack
        release_step_ = static_cast<float>(1.0 / (0.050 * sr));  // 50 ms release
        mute_step_ = static_cast<float>(1.0 / (0.005 * sr));     // 5 ms mute ramp
        reset();
    }

    void set_targets(float gain_db, float gate_threshold_dbfs, bool limiter_on,
                     float brightness_db) noexcept {
        gain_target_ = std::pow(10.0f, gain_db / 20.0f);
        threshold_ = gate_threshold_dbfs;
        limiter_target_ = limiter_on ? 1.0f : 0.0f;
        tilt_target_ = brightness_db;
    }

    /// Ramp the output to silence (true) or back to unity (false) over ~5 ms.
    /// Worker thread, like set_targets().
    void set_muted(bool muted) noexcept { mute_target_ = muted ? 0.0f : 1.0f; }
    bool muted() const noexcept { return mute_target_ == 0.0f; }

    /// Snap gain/tilt/limiter-mix/mute to their targets, close the gate
    /// (envelope 0, hold 0) and clear the tilt filter state.
    void reset() noexcept {
        gain_ = gain_target_;
        tilt_ = tilt_target_;
        mix_ = limiter_target_;
        mute_ = mute_target_;
        env_ = 0.0f;
        open_ = false;
        hold_ = 0;
        lp_ = 0.0f;
    }

    /// In place. `frame_level_dbfs[f]` is the input level of conditioning
    /// frame f (one per `frame_samples`); a short span holds its last value,
    /// an empty span closes the gate.
    void process(std::span<float> out, std::span<const float> frame_level_dbfs,
                 std::size_t frame_samples) noexcept {
        std::size_t i = 0, frame = 0;
        while (i < out.size()) {
            const float level =
                frame < frame_level_dbfs.size()
                    ? frame_level_dbfs[frame]
                    : (frame_level_dbfs.empty() ? -std::numeric_limits<float>::infinity()
                                                : frame_level_dbfs.back());
            update_gate(level);
            const float target = open_ ? 1.0f : 0.0f;
            const std::size_t n = frame_samples == 0
                                      ? out.size() - i
                                      : std::min(frame_samples, out.size() - i);
            for (std::size_t k = 0; k < n; ++k, ++i) {
                gain_ += pole_coef_ * (gain_target_ - gain_);
                env_ += env_ < target ? std::min(attack_step_, target - env_)
                                      : -std::min(release_step_, env_ - target);
                mute_ += mute_ < mute_target_ ? std::min(mute_step_, mute_target_ - mute_)
                                              : -std::min(mute_step_, mute_ - mute_target_);
                float y = out[i] * gain_ * env_ * mute_;
                tilt_ += pole_coef_ * (tilt_target_ - tilt_);
                // Snap within 1e-6: returns to the exact fast path (no pow)
                // instead of tailing off through denormals forever.
                if (std::fabs(tilt_target_ - tilt_) < 1e-6f) tilt_ = tilt_target_;
                lp_ += tilt_lp_coef_ * (y - lp_);
                if (tilt_ != 0.0f) {
                    const float hi = std::pow(10.0f, tilt_ / 40.0f);
                    y = lp_ / hi + (y - lp_) * hi;
                }
                mix_ += pole_coef_ * (limiter_target_ - mix_);
                if (std::fabs(limiter_target_ - mix_) < 1e-6f) mix_ = limiter_target_;
                if (mix_ != 0.0f) y += mix_ * (soft_limit(y) - y);
                out[i] = y;
            }
            ++frame;
        }
    }

    bool gate_open() const noexcept { return open_; }

    /// Unity below the knee (70 % of the ceiling); above it a tanh curve that
    /// is C1-continuous at the knee and asymptotic to the -1 dBFS ceiling, so
    /// nominal levels pass untouched and nothing exceeds the ceiling. The one
    /// definition of the curve (the GUI's Bypass mode runs this class too).
    static float soft_limit(float x) noexcept {
        const float a = std::fabs(x);
        if (a <= kKnee) return x;
        const float y = kKnee + kRange * std::tanh((a - kKnee) / kRange);
        return x < 0.0f ? -y : y;
    }

private:
    static constexpr float kCeiling = 0.891251f;  // -1 dBFS
    static constexpr float kKnee = 0.7f * kCeiling;
    static constexpr float kRange = kCeiling - kKnee;
    static constexpr double kTiltPivotHz = 1000.0;
    static constexpr float kGateHysteresisDb = 6.0f;
    static constexpr std::uint32_t kGateHoldFrames = 5;  // 100 ms at 20 ms frames

    void update_gate(float level_dbfs) noexcept {
        if (open_) {
            if (level_dbfs >= threshold_ - kGateHysteresisDb) {
                hold_ = kGateHoldFrames;
            } else if (hold_ > 0) {
                --hold_;
            } else {
                open_ = false;
            }
        } else if (level_dbfs >= threshold_) {
            open_ = true;
            hold_ = kGateHoldFrames;
        }
    }

    float pole_coef_ = 0.0031f;      // 20 ms @ 16 kHz until configure()
    float tilt_lp_coef_ = 0.3249f;   // 1 kHz @ 16 kHz until configure()
    float attack_step_ = 0.03125f;   // 2 ms @ 16 kHz until configure()
    float release_step_ = 0.00125f;  // 50 ms @ 16 kHz until configure()
    float mute_step_ = 0.0125f;      // 5 ms @ 16 kHz until configure()

    float gain_target_ = 1.0f;
    float threshold_ = -80.0f;
    float limiter_target_ = 1.0f;
    float tilt_target_ = 0.0f;
    float mute_target_ = 1.0f;  // multiplier: 1 = audible, 0 = muted

    float gain_ = 1.0f;
    float env_ = 0.0f;
    float tilt_ = 0.0f;
    float lp_ = 0.0f;
    float mix_ = 1.0f;
    float mute_ = 1.0f;
    bool open_ = false;
    std::uint32_t hold_ = 0;
};

} // namespace fluxvoice
