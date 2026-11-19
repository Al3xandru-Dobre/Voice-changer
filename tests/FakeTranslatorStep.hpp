// Hardware-free Mode B (S2UT) ModelStep fakes, the offline twins of the
// generated ONNX fixtures in tests/data/fixtures/s2ut (tools/make_fixture_models.py):
//
//   FakeDualEncoder    audio -> latents (mean of each 40-sample bin) + units
//   FakeTranslatorEnc  4-frame chunk -> src_pos counter + K/V ring writes
//   FakeTranslatorDec  reads translator_enc's ring (the zero-copy shared
//                      binding) -> units, prosody, eos_prob
//   FakeUnitVocoder    units + conditioning -> (unit / vocab) * voicing PCM
//
// Every contract is built with parse_contract() from JSON mirroring the real
// fixtures, so the fakes are guaranteed valid; a bad JSON edit throws
// std::logic_error at construction. Buffers are sized once; step() and
// reset() never allocate (the zero-allocation steady-state test relies on it).
#pragma once

#include "FakeModelStep.hpp"   // FakeClock (the step_us members below)
#include "pipeline/ModelContract.hpp"
#include "pipeline/ModelStep.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace test {

namespace detail {

inline fluxvoice::ModelContract must_parse(const std::string& json) {
    fluxvoice::ModelContract c;
    std::string error;
    if (!fluxvoice::parse_contract(json, c, error) || !fluxvoice::validate_contract(c, error))
        throw std::logic_error("fake contract rejected: " + error);
    return c;
}

constexpr const char* kLicense = "CC0-1.0";
constexpr const char* kLangs = R"({"de":3,"en":0,"es":1,"fr":2})";

} // namespace detail

inline fluxvoice::ModelContract dual_encoder_contract(int fps) {
    return detail::must_parse(
        R"({"content_kind":"discrete","contract_version":2,"frame_samples":320,"frames_per_step":)" +
        std::to_string(fps) +
        R"(,"latent_dim":8,"license":"CC0-1.0","lookahead_frames":0,"role":"encoder","sample_rate":16000,"states":[],"vocab":16})");
}

inline fluxvoice::ModelContract translator_enc_contract() {
    return detail::must_parse(
        std::string(R"({"chunk_frames":4,"contract_version":2,"frame_samples":320,"has_readiness_output":true,"langs":)") +
        detail::kLangs +
        R"(,"latent_dim":8,"license":"CC0-1.0","max_src_steps":4,"role":"translator_enc","sample_rate":16000,"shared_mem_states":["mem_k_0","mem_v_0"],"states":["src_pos","mem_k_0","mem_v_0"]})");
}

inline fluxvoice::ModelContract translator_dec_contract() {
    return detail::must_parse(
        std::string(R"({"chunk_frames":4,"contract_version":2,"frame_samples":320,"has_prosody_output":true,"langs":)") +
        detail::kLangs +
        R"(,"license":"CC0-1.0","max_src_steps":4,"max_tgt_steps":4,"role":"translator_dec","sample_rate":16000,"shared_mem_states":["mem_k_0","mem_v_0"],"special_units":{"bos":16,"eos":17,"pad":18},"states":["tgt_pos","dec_kv_0"],"vocab":16})");
}

inline fluxvoice::ModelContract unit_vocoder_contract(int fps, int overlap = 0) {
    return detail::must_parse(
        R"({"cond_frame_samples":320,"cond_inputs":["logf0","voicing","aperiodicity","energy","spk_emb"],"content_kind":"discrete","contract_version":2,"frame_samples":320,"frames_per_step":)" +
        std::to_string(fps) +
        R"(,"license":"CC0-1.0","lookahead_frames":0,"overlap_samples":)" + std::to_string(overlap) +
        R"(,"role":"vocoder","sample_rate":16000,"spk_emb_dim":4,"states":[],"vocab":16})");
}

// ---------------------------------------------------------------------------

class FakeDualEncoder final : public fluxvoice::ModelStep {
public:
    explicit FakeDualEncoder(int fps) : FakeDualEncoder(dual_encoder_contract(fps)) {}
    // For contract-mismatch tests: any validated encoder contract.
    explicit FakeDualEncoder(fluxvoice::ModelContract c)
        : c_(std::move(c)), audio_(static_cast<std::size_t>(c_.hop())),
          latents_(static_cast<std::size_t>(c_.frames_per_step * c_.latent_dim)),
          units_(static_cast<std::size_t>(c_.frames_per_step)) {}

    const fluxvoice::ModelContract& contract() const noexcept override { return c_; }
    std::span<float> input_f32(std::string_view name) noexcept override {
        return name == "audio" ? std::span<float>(audio_) : std::span<float>{};
    }
    std::span<std::int64_t> input_i64(std::string_view) noexcept override { return {}; }
    std::span<const float> output_f32(std::string_view name) const noexcept override {
        return name == "latents" ? std::span<const float>(latents_) : std::span<const float>{};
    }
    std::span<const std::int64_t> output_i64(std::string_view name) const noexcept override {
        return name == "units" ? std::span<const std::int64_t>(units_) :
                                 std::span<const std::int64_t>{};
    }

    bool step() noexcept override {
        ++steps;
        FakeClock::now += step_us;
        if (fail_next) {
            fail_next = false;
            return false;
        }
        const std::size_t frame = static_cast<std::size_t>(c_.frame_samples);
        const std::size_t dim = static_cast<std::size_t>(c_.latent_dim);
        const std::size_t bin = std::max<std::size_t>(1, frame / std::max<std::size_t>(dim, 1));
        for (std::size_t f = 0; f < units_.size(); ++f) {
            for (std::size_t d = 0; d < dim; ++d) {
                float sum = 0.0f;
                for (std::size_t j = 0; j < bin; ++j)
                    sum += audio_[f * frame + std::min(d * bin + j, frame - 1)];
                latents_[f * dim + d] = sum / static_cast<float>(bin);
            }
            units_[f] = static_cast<std::int64_t>((steps + f) % 16);
        }
        return true;
    }
    void reset() noexcept override { ++resets; }
    std::string_view last_error() const noexcept override { return "fake encoder failure"; }

    bool fail_next = false;
    std::size_t steps = 0, resets = 0;
    std::uint64_t step_us = 0;   // added to FakeClock::now by every step()

private:
    fluxvoice::ModelContract c_;
    std::vector<float> audio_, latents_;
    std::vector<std::int64_t> units_;
};

// ---------------------------------------------------------------------------

class FakeTranslatorEnc final : public fluxvoice::ModelStep {
public:
    FakeTranslatorEnc() : FakeTranslatorEnc(translator_enc_contract()) {}
    explicit FakeTranslatorEnc(fluxvoice::ModelContract c)
        : c_(std::move(c)), dim_(static_cast<std::size_t>(c_.latent_dim)),
          chunk_(static_cast<std::size_t>(c_.chunk_frames)),
          ring_(static_cast<std::size_t>(c_.max_src_steps)), latents_(chunk_ * dim_),
          k_(ring_ * dim_, 0.0f), v_(ring_ * dim_, 0.0f) {}

    const fluxvoice::ModelContract& contract() const noexcept override { return c_; }
    std::span<float> input_f32(std::string_view name) noexcept override {
        return name == "latents" ? std::span<float>(latents_) : std::span<float>{};
    }
    std::span<std::int64_t> input_i64(std::string_view name) noexcept override {
        return name == "src_lang" ? std::span<std::int64_t>(&src_lang_, 1) :
                                    std::span<std::int64_t>{};
    }
    std::span<const float> output_f32(std::string_view name) const noexcept override {
        return name == "readiness_logit" ? std::span<const float>(&readiness_, 1) :
                                           std::span<const float>{};
    }
    std::span<const std::int64_t> output_i64(std::string_view) const noexcept override { return {}; }

    bool step() noexcept override {
        ++steps;
        FakeClock::now += step_us;
        if (fail_next) {
            fail_next = false;
            return false;
        }
        const std::size_t slot = static_cast<std::size_t>(src_pos_) % ring_;
        for (std::size_t d = 0; d < dim_; ++d) {
            float sum = 0.0f;
            for (std::size_t f = 0; f < chunk_; ++f) sum += latents_[f * dim_ + d];
            const float v = sum / static_cast<float>(chunk_);
            v_[slot * dim_ + d] = v;
            k_[slot * dim_ + d] = v + static_cast<float>(src_lang_);
        }
        ++src_pos_;
        readiness_ = readiness_logit_value;
        return true;
    }
    void reset() noexcept override {
        ++resets;
        src_pos_ = 0;
        readiness_ = 0.0f;
        std::fill(k_.begin(), k_.end(), 0.0f);
        std::fill(v_.begin(), v_.end(), 0.0f);
    }
    std::string_view last_error() const noexcept override { return "fake translator_enc failure"; }

    std::int64_t src_pos() const noexcept { return src_pos_; }
    std::span<const float> k_ring() const noexcept { return k_; }
    std::span<const float> v_ring() const noexcept { return v_; }
    std::size_t ring_slots() const noexcept { return ring_; }
    std::size_t dim() const noexcept { return dim_; }

    float readiness_logit_value = 0.0f;
    bool fail_next = false;
    std::size_t steps = 0, resets = 0;
    std::uint64_t step_us = 0;   // added to FakeClock::now by every step()

private:
    fluxvoice::ModelContract c_;
    std::size_t dim_, chunk_, ring_;
    std::vector<float> latents_, k_, v_;
    std::int64_t src_lang_ = 0;
    std::int64_t src_pos_ = 0;
    float readiness_ = 0.0f;
};

// ---------------------------------------------------------------------------

class FakeTranslatorDec final : public fluxvoice::ModelStep {
public:
    enum class EosRule {
        kFixture,   // post-increment tgt_pos >  src_pos (the ONNX fixture rule)
        kCatchUp,   // post-increment tgt_pos >= src_pos
        kNever,
    };

    explicit FakeTranslatorDec(const FakeTranslatorEnc& enc)
        : FakeTranslatorDec(enc, translator_dec_contract()) {}
    FakeTranslatorDec(const FakeTranslatorEnc& enc, fluxvoice::ModelContract c)
        : enc_(enc), c_(std::move(c)), chunk_(static_cast<std::size_t>(c_.chunk_frames)),
          prev_(chunk_), units_(chunk_), prosody_(chunk_ * 4), last_prev_(chunk_) {}

    const fluxvoice::ModelContract& contract() const noexcept override { return c_; }
    std::span<float> input_f32(std::string_view name) noexcept override {
        return name == "prosody_ctx" ? std::span<float>(ctx_) : std::span<float>{};
    }
    std::span<std::int64_t> input_i64(std::string_view name) noexcept override {
        if (name == "prev_units") return prev_;
        if (name == "tgt_lang") return std::span<std::int64_t>(&tgt_lang_, 1);
        return {};
    }
    std::span<const float> output_f32(std::string_view name) const noexcept override {
        if (name == "prosody") return prosody_;
        if (name == "eos_prob") return std::span<const float>(&eos_prob_, 1);
        return {};
    }
    std::span<const std::int64_t> output_i64(std::string_view name) const noexcept override {
        return name == "units" ? std::span<const std::int64_t>(units_) :
                                 std::span<const std::int64_t>{};
    }

    bool step() noexcept override {
        ++steps;
        FakeClock::now += step_us;
        if (fail_next) {
            fail_next = false;
            return false;
        }
        std::copy(prev_.begin(), prev_.end(), last_prev_.begin());
        last_tgt_lang_ = tgt_lang_;
        last_ctx_[0] = ctx_[0];
        last_ctx_[1] = ctx_[1];

        const std::int64_t src_pos = enc_.src_pos();
        const std::size_t ring = enc_.ring_slots(), dim = enc_.dim();
        const std::int64_t newest = std::max<std::int64_t>(src_pos - 1, 0);
        const std::size_t slot = static_cast<std::size_t>(std::min(tgt_pos_, newest)) % ring;
        float k_mean = 0.0f, v_mean = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            k_mean += enc_.k_ring()[slot * dim + d];
            v_mean += enc_.v_ring()[slot * dim + d];
        }
        k_mean /= static_cast<float>(dim);
        v_mean /= static_cast<float>(dim);

        for (std::size_t i = 0; i < chunk_; ++i) {
            units_[i] = (prev_[i] + tgt_lang_) % 16;
            float* row = &prosody_[i * 4];
            row[0] = k_mean + ctx_[0];
            row[1] = 0.9f;
            row[2] = 0.1f;
            row[3] = v_mean + ctx_[1];
        }
        ++tgt_pos_;
        if (steps == static_cast<std::size_t>(emit_eos_on_step) && emit_eos_at_frame >= 0) {
            const std::size_t at = static_cast<std::size_t>(emit_eos_at_frame);
            for (std::size_t i = at; i < chunk_; ++i)
                units_[i] = i == at ? c_.special_units.eos : c_.special_units.pad;
        }
        bool eos = false;
        if (rule == EosRule::kFixture) eos = tgt_pos_ > src_pos;
        else if (rule == EosRule::kCatchUp) eos = tgt_pos_ >= src_pos;
        eos_prob_ = eos ? 1.0f : 0.0f;
        return true;
    }
    void reset() noexcept override {
        ++resets;
        tgt_pos_ = 0;
        eos_prob_ = 0.0f;
    }
    std::string_view last_error() const noexcept override { return "fake translator_dec failure"; }

    // What the most recent step() saw on its inputs.
    std::span<const std::int64_t> last_prev_units() const noexcept { return last_prev_; }
    std::int64_t last_tgt_lang() const noexcept { return last_tgt_lang_; }
    float last_ctx(std::size_t i) const noexcept { return last_ctx_[i]; }
    std::int64_t tgt_pos() const noexcept { return tgt_pos_; }

    EosRule rule = EosRule::kFixture;
    // BUG-028: on the emit_eos_on_step-th step() (1-based, never reset) frame
    // emit_eos_at_frame becomes <EOS> and the rest of the chunk <PAD>.
    int emit_eos_on_step = -1;
    int emit_eos_at_frame = -1;
    bool fail_next = false;
    std::size_t steps = 0, resets = 0;
    std::uint64_t step_us = 0;   // added to FakeClock::now by every step()

private:
    const FakeTranslatorEnc& enc_;
    fluxvoice::ModelContract c_;
    std::size_t chunk_;
    std::vector<std::int64_t> prev_, units_;
    std::vector<float> prosody_;
    std::vector<std::int64_t> last_prev_;
    std::int64_t tgt_lang_ = 0, last_tgt_lang_ = 0;
    std::int64_t tgt_pos_ = 0;
    float ctx_[2] = {0.0f, 0.0f};
    float last_ctx_[2] = {0.0f, 0.0f};
    float eos_prob_ = 0.0f;
};

// ---------------------------------------------------------------------------

class FakeUnitVocoder final : public fluxvoice::ModelStep {
public:
    explicit FakeUnitVocoder(int fps, int overlap = 0)
        : FakeUnitVocoder(unit_vocoder_contract(fps, overlap)) {}
    explicit FakeUnitVocoder(fluxvoice::ModelContract c)
        : c_(std::move(c)), units_(static_cast<std::size_t>(c_.frames_per_step)),
          output_(static_cast<std::size_t>(c_.output_samples())),
          speaker_(static_cast<std::size_t>(c_.spk_emb_dim)) {
        for (auto& v : cond_) v.resize(static_cast<std::size_t>(c_.cond_frames()));
    }

    const fluxvoice::ModelContract& contract() const noexcept override { return c_; }
    std::span<float> input_f32(std::string_view name) noexcept override {
        if (name == "spk_emb") return speaker_;
        for (std::size_t i = 0; i < kCond.size(); ++i)
            if (name == kCond[i]) return cond_[i];
        return {};
    }
    std::span<std::int64_t> input_i64(std::string_view name) noexcept override {
        return name == "units" ? std::span<std::int64_t>(units_) : std::span<std::int64_t>{};
    }
    std::span<const float> output_f32(std::string_view name) const noexcept override {
        return name == "audio" ? std::span<const float>(output_) : std::span<const float>{};
    }
    std::span<const std::int64_t> output_i64(std::string_view) const noexcept override { return {}; }

    bool step() noexcept override {
        ++steps;
        FakeClock::now += step_us;
        if (fail_next) {
            fail_next = false;
            return false;
        }
        const std::size_t frame = static_cast<std::size_t>(c_.cond_frame_samples);
        std::fill(output_.begin(), output_.end(), 0.0f);
        for (std::size_t f = 0; f < units_.size(); ++f) {
            const float voicing = ignore_voicing ? 1.0f : cond_[1][f];
            const float x = static_cast<float>(units_[f]) / static_cast<float>(c_.vocab) * voicing;
            for (std::size_t j = 0; j < frame; ++j) output_[f * frame + j] = x;
        }
        return true;
    }
    void reset() noexcept override { ++resets; }
    std::string_view last_error() const noexcept override { return "fake vocoder failure"; }

    // The conditioning / unit inputs of the most recent step().
    std::span<const float> cond_input(std::size_t index) const noexcept { return cond_[index]; }
    std::span<const float> voicing_input() const noexcept { return cond_[1]; }
    std::span<const float> aperiodicity_input() const noexcept { return cond_[2]; }
    std::span<const std::int64_t> units_input() const noexcept { return units_; }

    // Render unit/vocab regardless of voicing (so a frame the pipeline marks
    // as pad, voicing 0, is still audible unless the pipeline silences it).
    bool ignore_voicing = false;
    bool fail_next = false;
    std::size_t steps = 0, resets = 0;
    std::uint64_t step_us = 0;   // added to FakeClock::now by every step()

private:
    static constexpr std::array<std::string_view, 4> kCond{"logf0", "voicing", "aperiodicity",
                                                           "energy"};
    fluxvoice::ModelContract c_;
    std::vector<std::int64_t> units_;
    std::vector<float> output_, speaker_;
    std::array<std::vector<float>, 4> cond_;
};

} // namespace test
