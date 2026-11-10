#pragma once

#include "pipeline/ModelStep.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace test {

inline fluxvoice::ModelContract fake_contract(fluxvoice::ModelRole role, int fps = 1,
                                               int lookahead = 0, int frame_samples = 320,
                                               int overlap = 0) {
    fluxvoice::ModelContract c;
    c.role = role;
    c.sample_rate = 16000;
    c.frame_samples = frame_samples;
    c.frames_per_step = fps;
    c.lookahead_frames = lookahead;
    c.content_kind = fluxvoice::ContentKind::kSoft;
    c.content_dim = frame_samples;
    c.license = "CC0-1.0";
    if (role == fluxvoice::ModelRole::kVocoder) {
        c.cond_frame_samples = 320;
        c.cond_inputs = {"logf0", "voicing", "aperiodicity", "energy", "spk_emb"};
        c.spk_emb_dim = 4;
        c.overlap_samples = overlap;
    }
    return c;
}

class FakeModelStep final : public fluxvoice::ModelStep {
public:
    explicit FakeModelStep(fluxvoice::ModelContract c)
        : c_(std::move(c)), audio_(static_cast<std::size_t>(c_.hop())),
          content_(static_cast<std::size_t>(c_.frames_per_step * c_.content_dim)),
          units_(static_cast<std::size_t>(c_.frames_per_step)),
          output_(static_cast<std::size_t>(c_.output_samples())),
          delay_(static_cast<std::size_t>(c_.lookahead_frames * c_.frame_samples)),
          speaker_(static_cast<std::size_t>(c_.spk_emb_dim)) {
        for (auto& v : cond_) v.resize(static_cast<std::size_t>(c_.cond_frames()));
    }

    const fluxvoice::ModelContract& contract() const noexcept override { return c_; }
    std::span<float> input_f32(std::string_view name) noexcept override {
        if (c_.role == fluxvoice::ModelRole::kEncoder)
            return name == "audio" ? std::span<float>(audio_).subspan(bad_audio_view ? 1 : 0) :
                                     std::span<float>{};
        if (name == "content") return content_;
        if (name == "spk_emb") return speaker_;
        constexpr std::array<std::string_view, 4> names{"logf0", "voicing", "aperiodicity", "energy"};
        for (std::size_t i = 0; i < names.size(); ++i)
            if (name == names[i]) return cond_[i];
        return {};
    }
    std::span<std::int64_t> input_i64(std::string_view name) noexcept override {
        return name == "units" && c_.role == fluxvoice::ModelRole::kVocoder ?
            std::span<std::int64_t>(units_) : std::span<std::int64_t>{};
    }
    std::span<const float> output_f32(std::string_view name) const noexcept override {
        if (c_.role == fluxvoice::ModelRole::kVocoder)
            return name == "audio" ? std::span<const float>(output_) : std::span<const float>{};
        return name == "content" ? std::span<const float>(content_) : std::span<const float>{};
    }
    std::span<const std::int64_t> output_i64(std::string_view name) const noexcept override {
        return name == "units" && c_.role == fluxvoice::ModelRole::kEncoder ?
            std::span<const std::int64_t>(units_) : std::span<const std::int64_t>{};
    }

    bool step() noexcept override {
        ++steps;
        if (fail_next || always_fail) {
            fail_next = false;
            return false;
        }
        if (c_.role == fluxvoice::ModelRole::kEncoder) {
            if (c_.content_kind == fluxvoice::ContentKind::kDiscrete) {
                for (std::size_t i = 0; i < units_.size(); ++i)
                    units_[i] = static_cast<std::int64_t>(audio_[i * 320]);
            } else {
                for (std::size_t i = 0; i < content_.size(); ++i)
                    content_[i] = delayed(audio_[i]);
            }
        } else {
            for (std::size_t i = 0; i < output_.size(); ++i) {
                const float x = c_.content_kind == fluxvoice::ContentKind::kDiscrete ?
                    static_cast<float>(units_[(i / 320) % units_.size()]) :
                    content_[i % content_.size()];
                output_[i] = delayed(x);
            }
        }
        return true;
    }

    void reset() noexcept override {
        ++resets;
        std::fill(delay_.begin(), delay_.end(), 0.0f);
        cursor_ = 0;
    }
    std::string_view last_error() const noexcept override { return "fake Run failure"; }

    bool fail_next = false;
    bool always_fail = false;
    bool bad_audio_view = false;
    std::size_t steps = 0, resets = 0;

private:
    float delayed(float x) noexcept {
        if (delay_.empty()) return x;
        const float y = delay_[cursor_];
        delay_[cursor_] = x;
        cursor_ = (cursor_ + 1) % delay_.size();
        return y;
    }

    fluxvoice::ModelContract c_;
    std::vector<float> audio_, content_;
    std::vector<std::int64_t> units_;
    std::vector<float> output_, delay_, speaker_;
    std::array<std::vector<float>, 4> cond_;
    std::size_t cursor_ = 0;
};

} // namespace test
