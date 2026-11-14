// ParamSmoothing — worker-side transition helpers for VoiceParams
// (Milestone 2 / Stage 09).
//
// FrameRamp is a linear ramp over a fixed number of 20 ms conditioning
// frames: a control change sweeps in over ~100 ms instead of stepping at a
// frame boundary (click). set_target() recomputes the per-frame step from the
// *current* value, so retargeting mid-ramp stays continuous; advance() lands
// exactly on the target after N frames (no overshoot, no denormal tail).
//
// Lengths (stage-file "Smoothing" column):
//   kParamRampFrames      — every Smoothing::PerFrameRamp conditioning param
//   kVoiceCrossfadeFrames — the internal blend fade when voice_a / voice_b
//                           changes (voice switch at timbre_blend 0 must not
//                           jump either)
#pragma once

#include <cstdint>

namespace fluxvoice {

inline constexpr std::uint32_t kParamRampFrames = 5;       // 100 ms at 20 ms frames
inline constexpr std::uint32_t kVoiceCrossfadeFrames = 5;  // voice-switch fade

class FrameRamp {
public:
    explicit FrameRamp(std::uint32_t frames = kParamRampFrames) : frames_(frames) {}

    /// Jump to `v` immediately (construction / stream reset).
    void snap(float v) noexcept {
        current_ = target_ = v;
        step_ = 0.0f;
        remaining_ = 0;
    }

    /// New target: step size recomputed from the current value.
    void set_target(float v) noexcept {
        target_ = v;
        step_ = (v - current_) / static_cast<float>(frames_);
        remaining_ = frames_;
    }

    /// One conditioning frame. Lands exactly on the target after `frames`
    /// calls; afterwards advance() is a no-op.
    float advance() noexcept {
        if (remaining_ == 0) return current_;
        if (--remaining_ == 0) current_ = target_;
        else current_ += step_;
        return current_;
    }

    /// Finish an in-progress ramp in place (stream reset keeps the target).
    void finish() noexcept { snap(target_); }

    float value() const noexcept { return current_; }
    float target() const noexcept { return target_; }
    bool ramping() const noexcept { return remaining_ != 0; }

private:
    std::uint32_t frames_;
    float current_ = 0.0f;
    float target_ = 0.0f;
    float step_ = 0.0f;
    std::uint32_t remaining_ = 0;
};

} // namespace fluxvoice
