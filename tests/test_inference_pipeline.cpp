#include "pipeline/InferencePipeline.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "audio/OverlapAdd.hpp"
#include "AllocationCounter.hpp"
#include "FakeModelStep.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace fluxvoice;
using test::FakeModelStep;
using test::fake_contract;

constexpr std::array<float, 4> kEmbedding{0.1f, -0.2f, 0.3f, 0.4f};
const TargetVoiceStats kVoice{5.2, 0.25, -7.0, 2.0};

std::vector<float> signal(std::size_t samples) {
    std::vector<float> x(samples);
    for (std::size_t i = 0; i < samples; ++i)
        x[i] = static_cast<float>(0.2 * std::sin(static_cast<double>(i) * 0.079));
    return x;
}

template<class F> void rejects(F&& f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}

void test_identity_and_lookahead() {
    const auto input = signal(320 * 24);
    for (int fps : {1, 2, 4}) {
        for (int lookahead : {0, 1, 2}) {
            FakeModelStep enc(fake_contract(ModelRole::kEncoder, fps, lookahead));
            FakeModelStep voc(fake_contract(ModelRole::kVocoder, fps, 1));
            InferencePipeline p(enc, voc, kVoice, kEmbedding);
            const auto& g = p.geometry();
            CHECK(g.hop == static_cast<std::size_t>(320 * fps));
            CHECK(g.lookahead_frames == static_cast<std::size_t>(lookahead + 1));
            CHECK(g.algorithmic_samples() == g.hop + (lookahead + 1) * 320);
            std::vector<float> got(input.size());
            for (std::size_t i = 0; i < input.size(); i += g.hop)
                CHECK(p.process(std::span(input).subspan(i, g.hop),
                                std::span(got).subspan(i, g.hop)).ok);
            const std::size_t delay = static_cast<std::size_t>((lookahead + 1) * 320);
            for (std::size_t i = 0; i < got.size(); ++i)
                CHECK(got[i] == (i < delay ? 0.0f : input[i - delay]));
            std::vector<float> impulse(input.size(), 0.0f);
            impulse[7] = 1.0f;
            p.reset();
            for (std::size_t i = 0; i < impulse.size(); i += g.hop)
                CHECK(p.process(std::span(impulse).subspan(i, g.hop),
                                std::span(got).subspan(i, g.hop)).ok);
            CHECK(got[delay + 7] == 1.0f);
            CHECK(std::count(got.begin(), got.end(), 1.0f) == 1);
        }
        FakeModelStep enc(fake_contract(ModelRole::kEncoder, fps));
        FakeModelStep voc(fake_contract(ModelRole::kVocoder, fps));
        InferencePipeline p(enc, voc, kVoice, kEmbedding);
        std::vector<float> got(input.size());
        for (std::size_t i = 0; i < input.size(); i += p.geometry().hop)
            CHECK(p.process(std::span(input).subspan(i, p.geometry().hop),
                            std::span(got).subspan(i, p.geometry().hop)).ok);
        CHECK(got == input); // no HPF, extra delay, OLA or startup fade on PCM
        std::vector<float> impulse(p.geometry().hop, 0.0f), out(impulse.size());
        p.reset();
        impulse[7] = 1.0f;
        CHECK(p.process(impulse, out).ok);
        CHECK(out == impulse);
    }
}

void test_conditioning_and_reset() {
    for (int fps : {1, 2, 4}) {
        for (int frame : {320, 1280}) {
            FakeModelStep enc(fake_contract(ModelRole::kEncoder, fps, 1, frame));
            FakeModelStep voc(fake_contract(ModelRole::kVocoder, fps, 0, frame));
            PipelineOptions options;
            options.pitch_shift_st = 3.0;
            options.intonation_scale = 0.7;
            InferencePipeline p(enc, voc, kVoice, kEmbedding, options);
            PitchTrackerConfig cfg;
            cfg.lookahead_frames = frame / 320;
            PitchTracker reference(cfg);
            const auto input = signal(p.geometry().hop);
            std::vector<float> out(input.size()), first(input.size()), replay(input.size());
            for (int step = 0; step < 4; ++step) {
                CHECK(p.process(input, out).ok);
                if (step == 0) first = out;
                for (std::size_t i = 0; i < p.geometry().cond_frames; ++i) {
                    const auto f = reference.push(input.data() + i * 320);
                    CHECK(voc.input_f32("logf0")[i] == static_cast<float>(
                        map_logf0_to_target(f.z_logf0, kVoice, 0.7, 3.0)));
                    CHECK(voc.input_f32("energy")[i] == static_cast<float>(
                        map_energy_to_target(f.z_energy, kVoice)));
                    CHECK(voc.input_f32("voicing")[i] == f.voicing);
                    CHECK(voc.input_f32("aperiodicity")[i] == f.aperiodicity);
                }
                CHECK(std::equal(kEmbedding.begin(), kEmbedding.end(),
                                 voc.input_f32("spk_emb").begin()));
            }
            p.reset();
            CHECK(p.process(input, replay).ok);
            CHECK(replay == first);
            const std::vector<float> cond_before(voc.input_f32("energy").begin(),
                                                 voc.input_f32("energy").end());
            CHECK(p.warm_up(3));
            CHECK(p.timing().samples == 0 && p.errors() == 0);
            CHECK(p.process(input, replay).ok);
            CHECK(replay == first);
            CHECK(std::equal(cond_before.begin(), cond_before.end(),
                             voc.input_f32("energy").begin()));
        }
    }
}

void test_discrete_and_overlap() {
    auto ec = fake_contract(ModelRole::kEncoder, 4);
    auto vc = fake_contract(ModelRole::kVocoder, 4);
    ec.content_kind = vc.content_kind = ContentKind::kDiscrete;
    ec.content_dim = vc.content_dim = 0;
    ec.vocab = vc.vocab = 100;
    FakeModelStep enc(ec), voc(vc);
    InferencePipeline p(enc, voc, kVoice, kEmbedding);
    std::vector<float> in(1280), out(1280);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i / 320);
    CHECK(p.process(in, out).ok);
    CHECK(in == out);

    FakeModelStep soft_enc(fake_contract(ModelRole::kEncoder));
    FakeModelStep soft_voc(fake_contract(ModelRole::kVocoder, 1, 0, 320, 80));
    InferencePipeline ola_pipeline(soft_enc, soft_voc, kVoice, kEmbedding);
    OverlapAddProcessor reference(320, 80);
    auto audio = signal(320);
    std::vector<float> actual(320), expected(320);
    for (int i = 0; i < 3; ++i) {
        CHECK(ola_pipeline.process(audio, actual).ok);
        reference.process(soft_voc.output_f32("audio").data(), expected.data());
        CHECK(actual == expected);
    }
    ola_pipeline.reset();
    reference.reset();
    CHECK(ola_pipeline.process(audio, actual).ok);
    reference.process(soft_voc.output_f32("audio").data(), expected.data());
    CHECK(actual == expected);
}

void test_failures_and_gate() {
    FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
    PipelineOptions options;
    options.max_consecutive_errors = 3;
    InferencePipeline p(enc, voc, kVoice, kEmbedding, options);
    std::vector<float> input(320, 0.5f), out(320);
    CHECK(p.process(input, out).ok);
    enc.fail_next = true;
    auto result = p.process(input, out);
    CHECK(!result.ok && !result.stop && result.error == "fake Run failure");
    CHECK(p.errors() == 1 && voc.steps == 1);
    CHECK(out.front() < 0.5f && out.front() > 0.0f);
    CHECK(std::abs(out.front() - 0.5f) <= 0.5f / 32);
    CHECK(std::all_of(out.begin() + 32, out.end(), [](float x) { return x == 0.0f; }));
    CHECK(p.process(input, out).ok);
    CHECK(out.front() > 0.0f && out.front() <= 0.5f / 32);
    CHECK(out[31] == 0.5f && out.back() == 0.5f);
    voc.always_fail = true;
    for (int i = 0; i < 3; ++i) {
        result = p.process(input, out);
        CHECK(!result.ok && result.stop == (i == 2));
    }
    CHECK(p.errors() == 4 && p.timing().samples == 6);
    CHECK(!p.warm_up(2));
    CHECK(p.last_error() == "fake Run failure");
    voc.always_fail = false;
    CHECK(p.warm_up(2));
    CHECK(p.errors() == 0 && p.timing().samples == 0);
    CHECK(p.process(input, std::span(out).first(8)).stop);
    CHECK(p.process(input, out).ok);
    input[2] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!p.process(input, out).ok);
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }));

    options.output_gate = [](std::span<float> x, void*) noexcept {
        std::fill(x.begin(), x.end(), 0.0f);
    };
    InferencePipeline gated(enc, voc, kVoice, kEmbedding, options);
    std::fill(input.begin(), input.end(), 0.25f);
    CHECK(gated.process(input, out).quiet);
    CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == 0.0f; }));
}

void test_validation_and_no_allocations() {
    FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
    enc.bad_audio_view = true;
    rejects([&] { InferencePipeline p(enc, voc, kVoice, kEmbedding); });
    enc.bad_audio_view = false;
    rejects([&] { InferencePipeline p(enc, voc, kVoice, std::span(kEmbedding).first(2)); });
    TargetVoiceStats bad_voice = kVoice;
    bad_voice.logf0_std = 0.0;
    rejects([&] { InferencePipeline p(enc, voc, bad_voice, kEmbedding); });
    bad_voice = kVoice;
    bad_voice.energy_mean = std::numeric_limits<double>::max();
    {
        InferencePipeline p(enc, voc, bad_voice, kEmbedding);
        CHECK(!p.warm_up(1));
        CHECK(p.last_error() == "non-finite model conditioning");
    }
    FakeModelStep mismatch(fake_contract(ModelRole::kVocoder, 2));
    rejects([&] { InferencePipeline p(enc, mismatch, kVoice, kEmbedding); });
    auto bad_cond = fake_contract(ModelRole::kVocoder);
    bad_cond.cond_frame_samples = 160;
    FakeModelStep unsupported(bad_cond);
    rejects([&] { InferencePipeline p(enc, unsupported, kVoice, kEmbedding); });
    auto invalid = fake_contract(ModelRole::kEncoder);
    invalid.frames_per_step = 3;
    PipelineGeometry geometry;
    geometry.hop = 42;
    std::string error;
    CHECK(!make_pipeline_geometry(invalid, voc.contract(), geometry, error));
    CHECK(geometry.hop == 42);

    for (int fps : {1, 2, 4}) {
        FakeModelStep e(fake_contract(ModelRole::kEncoder, fps, 1));
        FakeModelStep v(fake_contract(ModelRole::kVocoder, fps, 0, 320, 80));
        InferencePipeline p(e, v, kVoice, kEmbedding);
        auto in = signal(p.geometry().hop);
        std::vector<float> out(in.size());
        CHECK(p.warm_up(2));
        if constexpr (test::kAllocationCounting) {
            const long before = test::allocation_count();
            for (int i = 0; i < 100; ++i) {
                CHECK(p.process(in, out).ok);
                p.governor().skip_output(1000, out);
                CHECK(p.timing().samples > 0);
                if (i % 10 == 0) {
                    v.fail_next = true;
                    CHECK(!p.process(in, out).ok);
                    p.reset();
                }
            }
            CHECK(test::allocation_count() == before);
        }
    }
}
} // namespace

int main() {
    test_identity_and_lookahead();
    test_conditioning_and_reset();
    test_discrete_and_overlap();
    test_failures_and_gate();
    test_validation_and_no_allocations();
    return test::finish("inference_pipeline");
}
