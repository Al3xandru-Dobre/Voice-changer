#include "pipeline/InferencePipeline.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "audio/OverlapAdd.hpp"
#include "AllocationCounter.hpp"
#include "FakeModelStep.hpp"
#include "FakeTranslatorStep.hpp"
#include "TestSupport.hpp"
#include "voice/Presets.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef FLUXVOICE_SOURCE_DIR
#define FLUXVOICE_SOURCE_DIR "."
#endif

namespace {
using namespace fluxvoice;
using test::FakeClock;
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

// output_chain = false: raw vocoder PCM, so these tests keep asserting
// bit-exact plumbing (the shipped chain is covered by test_voice_params).
PipelineOptions raw_options() {
    PipelineOptions o;
    o.output_chain = false;
    return o;
}

void test_identity_and_lookahead() {
    const auto input = signal(320 * 24);
    for (int fps : {1, 2, 4}) {
        for (int lookahead : {0, 1, 2}) {
            FakeModelStep enc(fake_contract(ModelRole::kEncoder, fps, lookahead));
            FakeModelStep voc(fake_contract(ModelRole::kVocoder, fps, 1));
            InferencePipeline p(enc, voc, kVoice, kEmbedding, raw_options());
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
        InferencePipeline p(enc, voc, kVoice, kEmbedding, raw_options());
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
            PipelineOptions options = raw_options();
            options.params.set(ParamId::PitchShift, 3.0f);
            options.params.set(ParamId::Intonation, 0.7f);
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
            for (const TimingSnapshot& t : p.stage_timings()) CHECK(t.samples == 0);
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
    InferencePipeline p(enc, voc, kVoice, kEmbedding, raw_options());
    std::vector<float> in(1280), out(1280);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i / 320);
    CHECK(p.process(in, out).ok);
    CHECK(in == out);

    FakeModelStep soft_enc(fake_contract(ModelRole::kEncoder));
    FakeModelStep soft_voc(fake_contract(ModelRole::kVocoder, 1, 0, 320, 80));
    InferencePipeline ola_pipeline(soft_enc, soft_voc, kVoice, kEmbedding, raw_options());
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
    // The Stage 09 gate/output chain replaced the old output_gate hook and is
    // covered in test_voice_params.
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
                // The any-thread telemetry readers allocate nothing either.
                const auto stages = p.stage_timings();
                CHECK(stages[static_cast<std::size_t>(PipelineStage::kFeatures)].samples > 0);
                CHECK(stages[static_cast<std::size_t>(PipelineStage::kOutputChain)].samples > 0);
                CHECK(p.governor_target_samples() == p.governor().effective_target_samples());
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

// ===========================================================================
// Mode B (S2UT) — Milestone 3 / Stage 08
//
// Expected decision strings, step counts and utterance ends come from the
// Python policy twin (python_training/fvml/waitk_policy.py) driven with the
// same fake decoder rules; they are literals here, not read back from the
// pipeline. One chunk = 4 frames = 1280 samples = 80 ms.
// ===========================================================================
using test::FakeDualEncoder;
using test::FakeTranslatorDec;
using test::FakeTranslatorEnc;
using test::FakeUnitVocoder;
using EosRule = FakeTranslatorDec::EosRule;

constexpr std::size_t kChunkSamples = 1280;
constexpr std::int64_t kBosUnit = 16;
const TargetVoiceStats kFixtureVoice{4.8, 0.25, -4.0, 1.5};
const std::array<float, 4> kZeroEmbedding{0.0f, 0.0f, 0.0f, 0.0f};

S2utOptions s2ut_opts(std::int64_t src = 1, std::int64_t tgt = 0, int k = 3) {
    S2utOptions o;
    o.policy.k = k;
    o.src_lang_id = src;
    o.tgt_lang_id = tgt;
    return o;
}

// The four fakes of one bundle plus the voice list. Members are declared in
// construction order: the decoder reads the translator_enc fake's rings.
struct Rig {
    explicit Rig(int fps_in)
        : fps(fps_in), enc(fps_in), tdec(tenc), voc(fps_in),
          voices{PipelineVoice{kFixtureVoice, kZeroEmbedding}} {}

    std::unique_ptr<InferencePipeline> make(const S2utOptions& o,
                                            PipelineOptions po = raw_options()) {
        return std::make_unique<InferencePipeline>(enc, tenc, tdec, voc,
                                                   std::span<const PipelineVoice>(voices), o, po);
    }

    int fps;
    FakeDualEncoder enc;
    FakeTranslatorEnc tenc;
    FakeTranslatorDec tdec;
    FakeUnitVocoder voc;
    std::array<PipelineVoice, 1> voices;
};

// 'S' = speech chunk, '.' = silent chunk. A pattern is `chunks` x 1280 samples
// of a 0.25-amplitude 200 Hz sine / zeros.
std::string chunks(std::size_t speech_then, std::size_t silent_then) {
    return std::string(speech_then, 'S') + std::string(silent_then, '.');
}

std::vector<float> make_input(std::string_view pattern) {
    std::vector<float> x(pattern.size() * kChunkSamples, 0.0f);
    for (std::size_t c = 0; c < pattern.size(); ++c) {
        if (pattern[c] == 'N') {   // unvoiced noise of similar level (deterministic LCG)
            std::uint32_t state = 12345u + static_cast<std::uint32_t>(c);
            for (std::size_t i = 0; i < kChunkSamples; ++i) {
                state = state * 1664525u + 1013904223u;
                x[c * kChunkSamples + i] =
                    0.25f * (static_cast<float>(state >> 8) / 8388608.0f - 1.0f);
            }
            continue;
        }
        if (pattern[c] != 'S') continue;
        for (std::size_t i = 0; i < kChunkSamples; ++i) {
            const std::size_t n = c * kChunkSamples + i;
            x[n] = static_cast<float>(0.25 * std::sin(2.0 * 3.14159265358979323846 * 200.0 *
                                                      static_cast<double>(n) / 16000.0));
        }
    }
    return x;
}

struct RunResult {
    std::vector<std::vector<float>> out;   // one entry per hop
    std::vector<char> voc_ran;             // the vocoder stepped during this hop
    std::size_t hops_per_chunk = 0;
};

// Drive `pattern` through `p` hop by hop. `after(hop_index)` runs after each
// process() call (governor / idle probes).
template <class After>
RunResult run(InferencePipeline& p, Rig& r, std::string_view pattern, After&& after) {
    const std::size_t hop = p.geometry().hop;
    const std::vector<float> in = make_input(pattern);
    RunResult res;
    res.hops_per_chunk = kChunkSamples / hop;
    for (std::size_t h = 0; h * hop < in.size(); ++h) {
        std::vector<float> out(hop);
        const std::size_t voc_before = r.voc.steps;
        const auto result = p.process(std::span<const float>(in).subspan(h * hop, hop), out);
        CHECK(result.ok);
        res.voc_ran.push_back(r.voc.steps != voc_before);
        res.out.push_back(std::move(out));
        after(h);
    }
    return res;
}

RunResult run(InferencePipeline& p, Rig& r, std::string_view pattern) {
    return run(p, r, pattern, [](std::size_t) {});
}

struct Event {
    S2utChunkEvent e;
    std::vector<std::int64_t> units;
};

class RecordingObserver final : public S2utObserver {
public:
    void on_chunk(const S2utChunkEvent& event) noexcept override {
        events.push_back({event, std::vector<std::int64_t>(event.units.begin(), event.units.end())});
    }
    std::string decisions() const {
        std::string s;
        for (const auto& ev : events) s += to_string(ev.e.decision)[0];
        return s;
    }
    std::vector<Event> events;
};

template <class F> void rejects_with(F&& f, std::string_view what) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument& e) {
        caught = std::string_view(e.what()).find(what) != std::string_view::npos;
        if (!caught) std::printf("unexpected message: %s\n", e.what());
    }
    CHECK(caught);
}

constexpr float kSpeechUnit(std::int64_t unit) { return static_cast<float>(unit) / 16.0f * 0.9f; }

// a. Construction.
void test_s2ut_construction_rejects() {
    Rig ok(4);
    CHECK(ok.make(s2ut_opts())->mode() == PipelineMode::kS2UT);   // control: valid rig builds
    {
        Rig r(4);
        auto p = r.make(s2ut_opts());
        CHECK(p->geometry().chunk_frames == 4 && p->geometry().hops_per_chunk() == 1);
        CHECK(p->waitk() != nullptr && !p->waitk()->utterance_active());
        CHECK(p->s2ut_telemetry().chunk_ms == 80);
    }
    {   // Mode A pipelines have no Mode B surface
        FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, kVoice, kEmbedding, raw_options());
        CHECK(p.mode() == PipelineMode::kVC && p.waitk() == nullptr && p.s2ut_idle());
        CHECK(p.s2ut_telemetry().dec_steps == 0);
    }
    {   // not four distinct models
        Rig r(4);
        rejects([&] {
            InferencePipeline p(r.enc, r.tenc, r.tenc, r.voc,
                                std::span<const PipelineVoice>(r.voices), s2ut_opts(), raw_options());
        });
    }
    {   // a vocoder with an OLA tail has no successor hop to land in
        Rig r(4);
        FakeUnitVocoder ola(4, 80);
        rejects_with([&] {
            InferencePipeline p(r.enc, r.tenc, r.tdec, ola,
                                std::span<const PipelineVoice>(r.voices), s2ut_opts(), raw_options());
        }, "overlap");
    }
    {   // contract v1 encoder
        Rig r(4);
        auto c = test::dual_encoder_contract(4);
        c.contract_version = 1;
        c.latent_dim = 0;
        FakeDualEncoder v1(c);
        rejects_with([&] {
            InferencePipeline p(v1, r.tenc, r.tdec, r.voc,
                                std::span<const PipelineVoice>(r.voices), s2ut_opts(), raw_options());
        }, "contract_version 2");
    }
    {   // encoder latent_dim 16 vs translator_enc latent_dim 8
        Rig r(4);
        auto c = test::dual_encoder_contract(4);
        c.latent_dim = 16;
        FakeDualEncoder wide(c);
        rejects_with([&] {
            InferencePipeline p(wide, r.tenc, r.tdec, r.voc,
                                std::span<const PipelineVoice>(r.voices), s2ut_opts(), raw_options());
        }, "latent_dim mismatch");
    }
    {   // language ids
        Rig r(4);
        rejects_with([&] { r.make(s2ut_opts(1, 1)); }, "must differ");
        rejects_with([&] { r.make(s2ut_opts(9, 0)); }, "not a translator language");
        rejects_with([&] { r.make(s2ut_opts(1, 9)); }, "not a translator language");
        rejects_with([&] { r.make(S2utOptions{}); }, "not a translator language");   // ids unset (-1)
        CHECK(r.make(s2ut_opts(3, 2)) != nullptr);          // any two distinct catalogue ids
    }
    {   // wait-k bounds 1..10
        Rig r(4);
        rejects_with([&] { r.make(s2ut_opts(1, 0, 0)); }, "wait-k");
        rejects_with([&] { r.make(s2ut_opts(1, 0, 11)); }, "wait-k");
        CHECK(r.make(s2ut_opts(1, 0, 1)) != nullptr);
        CHECK(r.make(s2ut_opts(1, 0, 10)) != nullptr);
    }
}

// b. Stage 08 doc scenario.
void test_s2ut_stage08_scenario() {
    Rig r(4);
    S2utOptions o = s2ut_opts();
    o.policy.tail_flush_ms = 0;
    r.tdec.rule = EosRule::kCatchUp;
    auto p = r.make(o);
    RecordingObserver obs;
    p->set_s2ut_observer(&obs);
    const std::size_t tenc_resets = r.tenc.resets, tdec_resets = r.tdec.resets;

    run(*p, r, chunks(10, 10));
    // Up to and including the utterance end: 2 reads, 8 writes, 2 flushes.
    CHECK(obs.events.size() == 20);
    CHECK(obs.decisions() == "RRWWWWWWWWFFRRRRRRRR");
    std::size_t reads = 0, writes = 0, flushes = 0;
    for (std::size_t i = 0; i < 12; ++i) {
        reads += obs.events[i].e.decision == WaitKPolicy::Decision::kRead;
        writes += obs.events[i].e.decision == WaitKPolicy::Decision::kWrite;
        flushes += obs.events[i].e.decision == WaitKPolicy::Decision::kFlush;
    }
    CHECK(reads == 2 && writes == 8 && flushes == 2);
    CHECK(writes + flushes == 10);                         // 10 translated chunks
    CHECK(r.tdec.steps == 10 && r.tenc.steps == 10);
    CHECK(obs.events[11].e.utterance_end && obs.events[11].e.eos && !obs.events[11].e.cap_end);
    CHECK(!obs.events[10].e.utterance_end);

    // The drain zeroed both translators and the policy.
    CHECK(p->waitk()->t_src() == 0 && p->waitk()->t_tgt() == 0);
    CHECK(!p->waitk()->utterance_active());
    CHECK(r.tenc.resets == tenc_resets + 1 && r.tdec.resets == tdec_resets + 1);
    CHECK(r.tenc.src_pos() == 0 && r.tdec.tgt_pos() == 0);
    CHECK(std::all_of(r.tenc.k_ring().begin(), r.tenc.k_ring().end(), [](float x) { return x == 0.0f; }));
    CHECK(std::all_of(r.tenc.v_ring().begin(), r.tenc.v_ring().end(), [](float x) { return x == 0.0f; }));
    const auto t = p->s2ut_telemetry();
    CHECK(t.utterances == 1 && t.eos_ends == 1 && t.cap_ends == 0);
    CHECK(t.dec_steps == 10 && t.lag_chunks == 0);

    // The next utterance starts from <BOS> with its own language ids.
    run(*p, r, "SSS");
    CHECK(r.tdec.steps == 11 && r.tdec.tgt_pos() == 1 && r.tenc.src_pos() == 3);
    CHECK(r.tdec.last_prev_units().size() == 4);
    for (std::int64_t u : r.tdec.last_prev_units()) CHECK(u == kBosUnit);
    CHECK(r.tdec.last_tgt_lang() == 0);
}

// c. Default policy (tail_flush_ms 400) with the fixture EOS rule.
void check_scenario(std::string_view pattern, const char* decisions, std::size_t enc_steps,
                    std::size_t dec_steps, std::vector<std::size_t> ends,
                    std::uint64_t resumes) {
    Rig r(4);
    auto p = r.make(s2ut_opts());
    RecordingObserver obs;
    p->set_s2ut_observer(&obs);
    run(*p, r, pattern);
    CHECK(obs.decisions() == decisions);
    CHECK(r.tenc.steps == enc_steps && r.tdec.steps == dec_steps);
    std::vector<std::size_t> got;
    for (std::size_t i = 0; i < obs.events.size(); ++i)
        if (obs.events[i].e.utterance_end) got.push_back(i);
    CHECK(got == ends);
    const auto t = p->s2ut_telemetry();
    CHECK(t.utterances == ends.size() && t.eos_ends == ends.size() && t.cap_ends == 0);
    CHECK(t.dec_steps == dec_steps);
    CHECK(p->waitk()->flush_resumes() == resumes);
    std::uint64_t r_n = 0, w_n = 0, f_n = 0;
    for (char c : std::string_view(decisions)) {
        r_n += c == 'R';
        w_n += c == 'W';
        f_n += c == 'F';
    }
    CHECK(t.reads == r_n && t.writes == w_n && t.flushes == f_n);
}

void test_s2ut_default_policy_scenarios() {
    // 10 speech + 10 silent: reads 4, writes 13, flushes 3; the tail flush
    // starts after > 400 ms of silence (chunk 15) and the fixture decoder
    // raises <EOS> on the 3rd flush step.
    check_scenario(chunks(10, 10), "RRWWWWWWWWWWWWWFFFRR", 15, 16, {17}, 0);
    {
        Rig r(4);
        auto p = r.make(s2ut_opts());
        run(*p, r, chunks(10, 10));
        const auto t = p->s2ut_telemetry();
        CHECK(t.reads == 4 && t.writes == 13 && t.flushes == 3);
    }
    // Speech resumes mid-flush: the lag carries over, no reset in between.
    check_scenario(std::string(10, 'S') + std::string(7, '.') + "SSS" + std::string(20, '.'),
                   "RRWWWWWWWWWWWWWFFRRWWWWWWFFFRRRRRRRRRRRR", 23, 24, {27}, 1);
    {
        Rig r(4);
        auto p = r.make(s2ut_opts());
        run(*p, r, std::string(10, 'S') + std::string(7, '.') + "SSS" + std::string(20, '.'));
        const auto t = p->s2ut_telemetry();
        CHECK(t.reads == 16 && t.writes == 19 && t.flushes == 5);
    }
    // Two utterances.
    check_scenario(std::string(10, 'S') + std::string(20, '.') + std::string(6, 'S') +
                       std::string(20, '.'),
                   "RRWWWWWWWWWWWWWFFFRRRRRRRRRRRRRRWWWWWWWWWFFFRRRRRRRRRRRR", 26, 28, {17, 43}, 0);
    {
        Rig r(4);
        auto p = r.make(s2ut_opts());
        run(*p, r, std::string(10, 'S') + std::string(20, '.') + std::string(6, 'S') +
                       std::string(20, '.'));
        const auto t = p->s2ut_telemetry();
        CHECK(t.reads == 28 && t.writes == 22 && t.flushes == 6);
    }
}

// d. The vocoder only runs on hops that hold translated frames.
void test_s2ut_vocoder_bypass() {
    for (int fps : {1, 2, 4}) {
        Rig r(fps);
        const std::size_t hpc = static_cast<std::size_t>(4 / fps);
        auto p = r.make(s2ut_opts(1, 3));                   // units 3, 6, 9, ... (3 * write mod 16)
        const auto res = run(*p, r, chunks(10, 14));
        CHECK(r.tdec.steps == 16);
        CHECK(r.voc.steps == r.tdec.steps * hpc);
        CHECK(p->s2ut_telemetry().voc_steps == r.voc.steps);
        // First write: chunk 2 (k = 3); the decoder runs on the chunk's last hop.
        const std::size_t first = 2 * hpc + (hpc - 1);
        const std::size_t count = 16 * hpc;
        CHECK(res.out.size() == 24 * hpc);
        std::size_t translated = 0;
        for (std::size_t h = 0; h < res.out.size(); ++h) {
            const bool expect_voc = h >= first && h < first + count;
            CHECK(static_cast<bool>(res.voc_ran[h]) == expect_voc);
            if (!expect_voc) {
                CHECK(std::all_of(res.out[h].begin(), res.out[h].end(), [](float x) { return x == 0.0f; }));
                continue;
            }
            for (std::size_t f = 0; f < static_cast<std::size_t>(fps); ++f) {
                const std::size_t frame = translated * static_cast<std::size_t>(fps) + f;
                const std::int64_t unit = static_cast<std::int64_t>((3 * (frame / 4 + 1)) % 16);
                const float want = kSpeechUnit(unit);
                for (std::size_t j = 0; j < 320; ++j)
                    CHECK(std::abs(res.out[h][f * 320 + j] - want) < 1e-6f);
            }
            ++translated;
        }
        CHECK(translated == count);
        CHECK(p->s2ut_idle());
    }
    {   // first translated unit = (BOS + tgt_lang) % 16 = tgt_lang
        Rig r(4);
        auto p = r.make(s2ut_opts(1, 2));
        run(*p, r, "SSS");
        CHECK(r.voc.units_input()[0] == 2 && r.voc.units_input()[3] == 2);
    }
}

// e. The output gate follows translated frames, not the live microphone.
void test_s2ut_tail_flush_not_gated() {
    for (int fps : {1, 4}) {
        const std::string pattern = chunks(10, 14);
        Rig raw_rig(fps), chain_rig(fps);
        auto raw = raw_rig.make(s2ut_opts(1, 3));
        PipelineOptions po;                                 // chain ON, made unity when open
        po.params.set(ParamId::Limiter, 0.0f);
        po.params.set(ParamId::Gain, 0.0f);
        po.params.set(ParamId::Brightness, 0.0f);
        auto chained = chain_rig.make(s2ut_opts(1, 3), po);
        const auto a = run(*raw, raw_rig, pattern);
        const auto b = run(*chained, chain_rig, pattern);
        const std::size_t hpc = static_cast<std::size_t>(4 / fps);
        const std::size_t flush_start = 15 * hpc;           // microphone is silent from chunk 10
        std::size_t flush_hops = 0, first_translated = a.out.size();
        for (std::size_t h = 0; h < a.out.size(); ++h)
            if (a.voc_ran[h]) { first_translated = h; break; }
        for (std::size_t h = 0; h < a.out.size(); ++h) {
            CHECK(a.voc_ran[h] == b.voc_ran[h]);
            if (!a.voc_ran[h]) continue;
            if (h >= flush_start) ++flush_hops;
            // skip the 2 ms speech onset ramp of the first translated hop
            const std::size_t from = h == first_translated ? 32 : 0;
            for (std::size_t i = from; i < a.out[h].size(); ++i)
                CHECK(std::abs(a.out[h][i] - b.out[h][i]) < 1e-6f);
        }
        CHECK(flush_hops >= 3 * hpc);                       // tail hops were actually compared
    }
}

// f. No heap allocation in the steady state.
void test_s2ut_no_allocations() {
    for (int fps : {1, 2, 4}) {
        Rig r(fps);
        auto p = r.make(s2ut_opts(1, 3), PipelineOptions{});
        CHECK(p->warm_up(2));
        const std::string cycle = chunks(10, 10);           // kRead -> kWrite -> kFlush -> reset
        std::string pattern;
        for (int i = 0; i < 4; ++i) pattern += cycle;
        const std::vector<float> in = make_input(pattern);
        const std::size_t hop = p->geometry().hop;
        std::vector<float> out(hop);
        const std::size_t cycle_hops = cycle.size() * kChunkSamples / hop;
        std::size_t h = 0;
        for (; h < cycle_hops; ++h)                         // one full cycle outside the window
            CHECK(p->process(std::span<const float>(in).subspan(h * hop, hop), out).ok);
        CHECK(p->s2ut_telemetry().utterances == 1);
        if constexpr (test::kAllocationCounting) {
            const long before = test::allocation_count();
            for (; h * hop < in.size(); ++h) {
                CHECK(p->process(std::span<const float>(in).subspan(h * hop, hop), out).ok);
                p->governor().skip_output(1000, out);
                const auto stages = p->stage_timings();
                CHECK(stages[static_cast<std::size_t>(PipelineStage::kFeatures)].samples > 0);
                CHECK(p->governor_hard_cap_samples() == p->governor().hard_cap_samples());
            }
            CHECK(test::allocation_count() == before);
        }
        CHECK(p->s2ut_telemetry().utterances == 4);
        CHECK(p->s2ut_telemetry().dec_steps == 64);
    }
}

// g. warm_up() in Mode B.
void test_s2ut_warm_up() {
    Rig a(4), b(4);
    auto warmed = a.make(s2ut_opts(1, 3));
    CHECK(warmed->warm_up(3));
    CHECK(a.tenc.steps == 3 && a.tdec.steps == 3 && a.voc.steps == 3);   // silence never reaches them
    CHECK(a.enc.steps == 3);                                // the encoder ran inside process()
    CHECK(!warmed->waitk()->utterance_active());
    CHECK(warmed->waitk()->t_src() == 0 && warmed->waitk()->t_tgt() == 0);
    const auto t = warmed->s2ut_telemetry();
    CHECK(t.reads == 0 && t.writes == 0 && t.flushes == 0 && t.utterances == 0);
    CHECK(t.eos_ends == 0 && t.cap_ends == 0 && t.dec_steps == 0 && t.voc_steps == 0);
    CHECK(t.fifo_stalls == 0 && t.pad_frames == 0 && t.lag_chunks == 0);
    CHECK(warmed->timing().samples == 0 && warmed->errors() == 0);
    for (auto stage : {PipelineStage::kEncoder, PipelineStage::kTranslatorEnc,
                       PipelineStage::kTranslatorDec, PipelineStage::kVocoder,
                       PipelineStage::kFeatures, PipelineStage::kOutputChain})
        CHECK(warmed->stage_timing(stage).samples == 0);
    for (const TimingSnapshot& t : warmed->stage_timings()) CHECK(t.samples == 0);
    CHECK(a.tenc.src_pos() == 0 && a.tdec.tgt_pos() == 0);  // warm-up state was reset

    auto fresh = b.make(s2ut_opts(1, 3));
    const auto ra = run(*warmed, a, chunks(10, 14));
    const auto rb = run(*fresh, b, chunks(10, 14));
    CHECK(ra.out == rb.out);
    CHECK(warmed->s2ut_telemetry().dec_steps == fresh->s2ut_telemetry().dec_steps);
    CHECK(warmed->s2ut_telemetry().dec_steps == 16);

    // A failing translator stage fails the warm-up.
    Rig c(4);
    auto p = c.make(s2ut_opts());
    c.tdec.fail_next = true;
    CHECK(!p->warm_up(1));
    CHECK(p->last_error() == "fake translator_dec failure");
    c.tdec.fail_next = false;
    c.voc.fail_next = true;
    CHECK(!p->warm_up(1));
    CHECK(p->warm_up(1));
}

// h. BUG-028: a partial <EOS> chunk keeps the vocoder hop aligned.
void test_s2ut_bug028_partial_eos() {
    {   // fps 4: [u, u, <EOS>, <PAD>] on the first flush step (decoder step 14)
        Rig r(4);
        r.voc.ignore_voicing = true;                        // make pad frames audible to the cut
        r.tdec.emit_eos_on_step = 14;
        r.tdec.emit_eos_at_frame = 2;
        auto p = r.make(s2ut_opts(1, 3));
        RecordingObserver obs;
        p->set_s2ut_observer(&obs);
        const auto res = run(*p, r, chunks(10, 10));
        const auto t = p->s2ut_telemetry();
        CHECK(t.pad_frames == 2);
        CHECK(t.utterances == 1 && t.eos_ends == 1 && t.cap_ends == 0);
        CHECK(obs.events[15].e.utterance_end && obs.events[15].e.eos);
        CHECK(obs.events[15].e.pad_frames == 2 && obs.events[15].e.frames_pushed == 2);
        CHECK(obs.events[15].units == std::vector<std::int64_t>({10, 10, 17, 18}));
        CHECK(res.voc_ran[15]);                             // the EOS hop is still vocoded
        CHECK(t.voc_steps == 14);                           // 13 writes + the EOS hop, nothing stranded
        // pad frames keep the newest unit/pitch, voicing 0, aperiodicity 1
        CHECK(r.voc.units_input()[2] == 10 && r.voc.units_input()[3] == 10);
        CHECK(r.voc.voicing_input()[1] == 0.9f);
        CHECK(r.voc.voicing_input()[2] == 0.0f && r.voc.voicing_input()[3] == 0.0f);
        CHECK(r.voc.aperiodicity_input()[2] == 1.0f && r.voc.aperiodicity_input()[3] == 1.0f);
        const auto& hop = res.out[15];
        CHECK(hop.size() == 1280);
        for (std::size_t i = 0; i < 640; ++i) CHECK(std::abs(hop[i] - 10.0f / 16.0f) < 1e-6f);
        for (std::size_t i = 640; i < 1280; ++i) CHECK(hop[i] == 0.0f);   // exact silence (raw mode)
        for (std::size_t h = 16; h < res.out.size(); ++h) CHECK(!res.voc_ran[h]);
        CHECK(p->s2ut_idle());

        // The next utterance's first translated hop starts with its own first
        // unit: no stale frame, no phase shift.
        const auto next = run(*p, r, "SSS");
        CHECK(next.voc_ran[2] && !next.voc_ran[0] && !next.voc_ran[1]);
        for (std::size_t i = 0; i < 4; ++i) CHECK(r.voc.units_input()[i] == 3);
        for (float x : next.out[2]) CHECK(std::abs(x - 3.0f / 16.0f) < 1e-6f);
        CHECK(p->s2ut_telemetry().pad_frames == 2);
    }
    {   // fps 1: a hop is one frame, nothing to pad
        Rig r(1);
        r.voc.ignore_voicing = true;
        r.tdec.emit_eos_on_step = 14;
        r.tdec.emit_eos_at_frame = 2;
        auto p = r.make(s2ut_opts(1, 3));
        const auto res = run(*p, r, chunks(10, 10));
        const auto t = p->s2ut_telemetry();
        CHECK(t.pad_frames == 0);
        CHECK(t.utterances == 1 && t.eos_ends == 1);
        CHECK(t.voc_steps == 54);                           // 13 * 4 + 2 valid frames
        CHECK(r.voc.steps == 54);
        std::size_t last_translated = 0;
        for (std::size_t h = 0; h < res.out.size(); ++h)
            if (res.voc_ran[h]) last_translated = h;
        for (float x : res.out[last_translated]) CHECK(std::abs(x - 10.0f / 16.0f) < 1e-6f);
        CHECK(p->s2ut_idle());
    }
}

// i. Observer.
void test_s2ut_observer() {
    Rig r(4);
    S2utOptions o = s2ut_opts();
    o.policy.tail_flush_ms = 0;
    r.tdec.rule = EosRule::kCatchUp;
    r.tenc.readiness_logit_value = 1.5f;
    auto p = r.make(o);
    RecordingObserver obs;
    p->set_s2ut_observer(&obs);
    const std::string pattern = chunks(10, 10);
    run(*p, r, pattern);
    CHECK(obs.events.size() == 20);                         // one event per chunk step
    CHECK(obs.decisions() == "RRWWWWWWWWFFRRRRRRRR");
    for (std::size_t i = 0; i < obs.events.size(); ++i) {
        const auto& e = obs.events[i].e;
        CHECK(e.chunk == i);
        CHECK(e.input_samples == (i + 1) * 1280);
        CHECK(e.speech == (pattern[i] == 'S'));
        CHECK(e.segment == (i <= 11 ? 0u : 1u));            // increments after the end event
        CHECK(e.utterance_end == (i == 11));
        CHECK(e.cap_end == false);
        const int writes = e.decision == WaitKPolicy::Decision::kRead ? 0 : 1;
        CHECK(e.writes == writes);
        CHECK(e.frames_pushed == static_cast<std::size_t>(4 * writes));
        CHECK(obs.events[i].units.size() == static_cast<std::size_t>(4 * writes));
        CHECK(e.pad_frames == 0);
        CHECK(e.encoded == (i < 10));                       // silence pause starts the flush, no encode
        CHECK(e.readiness_logit == (i < 10 ? 1.5f : 0.0f));
        CHECK(e.eos == (i == 11));
    }
    CHECK(obs.events[9].e.t_src == 10 && obs.events[9].e.t_tgt == 8);
    CHECK(obs.events[10].e.t_src == 10 && obs.events[10].e.t_tgt == 9);
    CHECK(obs.events[11].e.t_src == 0 && obs.events[11].e.t_tgt == 0);
    CHECK(obs.events[2].units == std::vector<std::int64_t>({0, 0, 0, 0}));   // (BOS + tgt 0) % 16

    // Detached observers receive nothing more.
    p->set_s2ut_observer(nullptr);
    run(*p, r, "SS");
    CHECK(obs.events.size() == 20);
}

// j. Adaptive readiness reaches the policy through the sigmoid.
void test_s2ut_readiness_plumbing() {
    for (const float logit : {-10.0f, 10.0f}) {
        Rig r(4);
        S2utOptions o = s2ut_opts();
        o.policy.adaptive_mode = true;
        r.tenc.readiness_logit_value = logit;
        auto p = r.make(o);
        RecordingObserver obs;
        p->set_s2ut_observer(&obs);
        run(*p, r, std::string(8, 'S'));
        const std::size_t first_write = obs.decisions().find('W');
        // sigmoid(-10) ~ 0 < threshold: wait until lag == k + adaptive_extra_wait (5 chunks);
        // sigmoid(+10) ~ 1: readiness banks and the first write comes at lag == k.
        CHECK(first_write == (logit < 0 ? 4u : 2u));
        CHECK(obs.decisions() == (logit < 0 ? "RRRRWWWW" : "RRWWWWWW"));
        CHECK(obs.events[first_write].e.readiness_logit == logit);
    }
    {   // non-adaptive ignores the readiness head entirely
        Rig r(4);
        r.tenc.readiness_logit_value = -10.0f;
        auto p = r.make(s2ut_opts());
        RecordingObserver obs;
        p->set_s2ut_observer(&obs);
        run(*p, r, std::string(8, 'S'));
        CHECK(obs.decisions() == "RRWWWWWW");
    }
}

// k. Governor hint: hold + cushion while translating.
void test_s2ut_governor_hint() {
    {   // k = 6: reads with the cushion active and no hold. Without the cushion
        // a quiet queue of target + hop + 1 would be trimmed on the 3rd hop.
        Rig r(4);
        auto p = r.make(s2ut_opts(1, 3, 6));
        auto& gov = p->governor();
        const std::size_t hop = p->geometry().hop;
        CHECK(gov.burst_cushion_samples() == 3 * 1280);
        CHECK(!gov.cushion_active() && !gov.hold_active());
        const std::vector<float> quiet(hop, 0.0f);
        run(*p, r, std::string(12, 'S'), [&](std::size_t h) {
            if (h > 4) return;                              // chunks 0..4 are kRead (t_src 1..5 < k = 6)
            CHECK(gov.cushion_active() && !gov.hold_active());
            CHECK(gov.skip_output(static_cast<std::int64_t>(gov.target_samples() + hop + 1), quiet) == 0);
        });
    }
    {
        Rig r(4);
        auto p = r.make(s2ut_opts(1, 3));
        auto& gov = p->governor();
        const std::size_t hop = p->geometry().hop;
        const std::size_t target = gov.target_samples(), cushion = gov.burst_cushion_samples();
        const std::vector<float> quiet(hop, 0.0f);
        const auto beyond = static_cast<std::int64_t>(target + cushion + hop + 1);
        // chunk c <-> hop c at fps 4. Translated hops: chunks 2..17.
        run(*p, r, chunks(10, 14), [&](std::size_t h) {
            if (h < 2) {
                CHECK(gov.cushion_active() && !gov.hold_active());      // reading, nothing decoded
            } else if (h <= 16) {
                CHECK(gov.cushion_active() && gov.hold_active());
                // Soft drops are held back even for quiet output past the cushion...
                CHECK(gov.skip_output(beyond, quiet) == 0);
                // ...and one hop of excess below target + cushion is never an excess.
                CHECK(gov.skip_output(static_cast<std::int64_t>(target + hop + 1), quiet) == 0);
            } else if (h == 17) {
                // The last tail hop is still audible although the policy is already
                // idle: hold AND cushion stay on, so a burst backlog just above the
                // plain (uncushioned) hard cap does not cut the translated tail.
                CHECK(gov.hold_active() && gov.cushion_active());
                const auto above_plain_cap = static_cast<std::int64_t>(target + 2400 + 1); // 150 ms
                CHECK(gov.skip_output(above_plain_cap, quiet) == 0);
            } else if (h == 18) {
                // The hop after the tail ramps down from its last translated sample:
                // still protected, so the ramp is never hard-capped into a click.
                CHECK(gov.hold_active() && gov.cushion_active());
                const auto above_plain_cap = static_cast<std::int64_t>(target + 2400 + 1); // 150 ms
                CHECK(gov.skip_output(above_plain_cap, quiet) == 0);
            } else {
                CHECK(!gov.cushion_active() && !gov.hold_active());     // drained and idle
            }
            if (h == 18) gov.skip_output(0, quiet);                     // clear the sustained history
            if (h > 18 && h < 22) {
                // back to the plain target: the same occupancy is trimmed once sustained (3 hops)
                const auto q = static_cast<std::int64_t>(target + hop + 1);
                const std::size_t got = gov.skip_output(q, quiet);
                CHECK(got == (h == 21 ? hop : 0));
            }
        });
        // A hop that vocodes while translated frames stay queued keeps the hint
        // (fps 1: 4 frames queued per write).
        Rig r1(1);
        auto p1 = r1.make(s2ut_opts(1, 3));
        run(*p1, r1, chunks(10, 14), [&](std::size_t h) {
            if (h == 11 || h == 12) CHECK(p1->governor().hold_active() && p1->governor().cushion_active());
        });
    }
    {   // a mid-stream reset() clears the hint
        Rig r(4);
        auto p = r.make(s2ut_opts());
        run(*p, r, "SSSS");
        CHECK(p->governor().hold_active() && p->governor().cushion_active());
        p->reset();
        CHECK(!p->governor().hold_active() && !p->governor().cushion_active());
    }
}

// l. s2ut_idle().
void test_s2ut_idle() {
    {
        Rig r(4);
        auto p = r.make(s2ut_opts());
        CHECK(p->s2ut_idle());
        run(*p, r, chunks(10, 14), [&](std::size_t h) {
            // speech pending / translating / flushing through chunk 16; the
            // chunk-17 hop ends the utterance and pops its last frames
            if (h <= 16) CHECK(!p->s2ut_idle());
            else CHECK(p->s2ut_idle());
        });
        CHECK(p->s2ut_idle());
    }
    {   // fps 1: a partial chunk with speech, and frames still queued after the drain
        Rig r(1);
        auto p = r.make(s2ut_opts());
        const std::vector<float> speech = make_input("S");
        std::vector<float> out(320);
        CHECK(p->process(std::span<const float>(speech).first(320), out).ok);
        CHECK(!p->s2ut_idle());                             // chunk 0 has speech and is incomplete
        Rig q(1);
        auto silent = q.make(s2ut_opts());
        std::vector<float> zeros(320, 0.0f);
        CHECK(silent->process(zeros, out).ok);
        CHECK(silent->s2ut_idle());                         // a silent partial chunk is idle

        run(*p, r, std::string(9, 'S'), [](std::size_t) {});
        Rig z(1);
        auto d = z.make(s2ut_opts());
        std::size_t not_idle_after_end = 0;
        std::uint64_t ends = 0;
        run(*d, z, chunks(10, 10), [&](std::size_t) {
            const auto now = d->s2ut_telemetry().utterances;
            if (now != ends) {                              // the hop that ended the utterance
                ends = now;
                // 3 of the last write's 4 frames are still queued: not idle yet
                CHECK(!d->s2ut_idle());
                ++not_idle_after_end;
            }
        });
        CHECK(not_idle_after_end == 1);
        CHECK(d->s2ut_idle());
    }
}

// ===========================================================================
// Mode B live translation controls — Milestone 3 / Stage 09
// ===========================================================================

// A Mode B bundle driven hop by hop with a continuous 0.25-amplitude sine,
// recording every output sample.
struct Live {
    Live(int fps, bool translate, std::int64_t src, std::int64_t tgt, int k = 3)
        : rig(fps) {
        S2utOptions o = s2ut_opts(src, tgt, k);
        o.translate = translate;
        p = rig.make(o);
        hop = p->geometry().hop;
        hpc = kChunkSamples / hop;
        in.assign(hop, 0.0f);
        out.assign(hop, 0.0f);
    }
    void hops(std::size_t count, bool record = true, bool speech = true) {
        for (std::size_t h = 0; h < count; ++h) {
            for (std::size_t i = 0; i < hop; ++i, ++n)
                in[i] = !speech ? 0.0f : static_cast<float>(0.25 * std::sin(2.0 * 3.14159265358979323846 * 200.0 *
                                                           static_cast<double>(n) / 16000.0));
            CHECK(p->process(in, out).ok);
            if (record) all.insert(all.end(), out.begin(), out.end());
        }
    }
    bool apply(bool translate, std::int64_t src, std::int64_t tgt) {
        VoiceParams v = p->params();
        v.set(ParamId::Translate, translate ? 1.0f : 0.0f);
        v.src_lang_id = static_cast<std::uint16_t>(src);
        v.tgt_lang_id = static_cast<std::uint16_t>(tgt);
        return p->set_params(v);
    }

    Rig rig;
    std::unique_ptr<InferencePipeline> p;
    std::size_t hop = 0, hpc = 0, n = 0;
    std::vector<float> in, out, all;
};

struct Resets {
    explicit Resets(const Rig& r) : tenc(r.tenc.resets), tdec(r.tdec.resets), enc(r.enc.resets), voc(r.voc.resets) {}
    std::size_t tenc, tdec, enc, voc;
};

// One switch: the translators reset exactly once, encoder/vocoder not at all,
// the policy is zeroed, and mode() follows `translate`. Returns the flat
// sample index of the first sample of the next (crossfaded) hop.
std::size_t do_switch(Live& L, bool translate, std::int64_t src, std::int64_t tgt) {
    const Resets before(L.rig);
    const std::size_t at = L.all.size();
    CHECK(L.apply(translate, src, tgt));
    CHECK(L.rig.tenc.resets == before.tenc + 1 && L.rig.tdec.resets == before.tdec + 1);
    CHECK(L.rig.enc.resets == before.enc && L.rig.voc.resets == before.voc);
    CHECK(L.p->waitk()->t_src() == 0 && L.p->waitk()->t_tgt() == 0);
    CHECK(L.rig.tenc.src_pos() == 0 && L.rig.tdec.tgt_pos() == 0);
    CHECK(L.p->mode() == (translate ? PipelineMode::kS2UT : PipelineMode::kVC));
    CHECK(L.p->has_translator());
    CHECK(L.p->params().get(ParamId::Translate) == (translate ? 1.0f : 0.0f));
    CHECK(L.p->params().src_lang_id == src && L.p->params().tgt_lang_id == tgt);
    return at;
}

// 0 -> 1 -> language switch -> 0. `verify` attaches the observer and checks
// decisions; the allocation run replays it with recording off.
void switch_sequence(Live& L, bool verify, std::vector<std::size_t>* switch_at) {
    const std::size_t off = L.hpc > 1 ? 2 : 0;       // fps 1: switch in the middle of a chunk
    RecordingObserver obs;
    if (verify) L.p->set_s2ut_observer(&obs);
    const bool rec = verify;

    L.hops(6 * L.hpc + off, rec);                    // Mode A route (voice changer)
    CHECK(L.p->mode() == PipelineMode::kVC);
    // 0 -> 1 with the initial language pair fr -> es.
    const std::size_t ev0 = obs.events.size();
    std::size_t at = do_switch(L, true, 2, 1);
    if (switch_at) switch_at->push_back(at);
    std::size_t v0 = L.rig.voc.steps;
    L.hops(3 * L.hpc - 1, rec);                      // k - 1 = 2 chunks of reads, nothing vocoded
    CHECK(L.rig.voc.steps == v0);
    L.hops(1, rec);                                  // chunk 2: first write
    CHECK(L.rig.voc.steps == v0 + 1);
    if (verify) {
        CHECK(obs.events.size() == ev0 + 3);
        CHECK(obs.decisions().substr(ev0, 3) == "RRW");   // k = 3: lag reaches k on the 3rd chunk
        CHECK(obs.events[ev0].e.t_src == 1 && obs.events[ev0 + 2].e.t_tgt == 1);
    }
    L.hops(8 * L.hpc + off, rec);                    // steady translation

    // es/fr -> en/de while translating, mid-chunk and with frames queued.
    at = do_switch(L, true, 0, 3);
    if (switch_at) switch_at->push_back(at);
    v0 = L.rig.voc.steps;
    const std::size_t dec0 = L.rig.tdec.steps;
    L.hops(3 * L.hpc - 1, rec);                      // queued frames and the partial chunk are gone
    CHECK(L.rig.voc.steps == v0 && L.rig.tdec.steps == dec0);
    L.hops(1, rec);
    CHECK(L.rig.voc.steps == v0 + 1 && L.rig.tdec.steps == dec0 + 1);
    CHECK(L.rig.tdec.last_tgt_lang() == 3);          // the new target language
    CHECK(L.rig.tdec.tgt_pos() == 1);
    for (std::int64_t u : L.rig.tdec.last_prev_units()) CHECK(u == kBosUnit);
    L.hops(8 * L.hpc + off, rec);

    // 1 -> 0.
    at = do_switch(L, false, 0, 3);
    if (switch_at) switch_at->push_back(at);
    v0 = L.rig.voc.steps;
    L.hops(2 * L.hpc, rec);
    CHECK(L.rig.voc.steps == v0 + 2 * L.hpc);        // the voice-changer route vocodes every hop
    CHECK(L.p->s2ut_idle());                         // nothing in flight on the Mode A route
    if (verify) L.p->set_s2ut_observer(nullptr);
}

void test_s2ut_live_switching() {
    for (int fps : {1, 4}) {
        Live L(fps, false, 2, 1);
        CHECK(L.p->mode() == PipelineMode::kVC && L.p->has_translator());
        CHECK(L.p->params().get(ParamId::Translate) == 0.0f);
        std::vector<std::size_t> at;
        switch_sequence(L, true, &at);
        CHECK(at.size() == 3);
        double maxabs = 0.0;
        for (float x : L.all) maxabs = std::max(maxabs, static_cast<double>(std::abs(x)));
        // |held - new| <= 2 * maxabs and the cosine slope is <= pi / (2 * 320).
        const double bound = 1.1 * maxabs * 3.14159265358979323846 / 320.0 + 1e-6;
        CHECK(maxabs > 0.05);
        for (std::size_t k = 0; k < at.size() && k < 3; ++k) {
            const std::size_t i = at[k];
            const double held = L.all[i - 1];
            CHECK(std::abs(held) > 0.01);                    // a real signal was playing
            CHECK(std::abs(static_cast<double>(L.all[i]) - held) < 1e-6);   // starts at the held sample
            // deltas fully inside the 320-sample fade; the step into the next
            // frame is the new route's own frame boundary, not a switch artefact
            for (std::size_t j = i; j + 1 < i + 320; ++j)
                CHECK(std::abs(static_cast<double>(L.all[j + 1]) - L.all[j]) <= bound);
        }
        if constexpr (test::kAllocationCounting) {
            // Warm replay, then count one more full 0 -> 1 -> lang -> 0 sequence.
            Live W(fps, false, 2, 1);
            switch_sequence(W, false, nullptr);
            const long before = test::allocation_count();
            switch_sequence(W, false, nullptr);
            CHECK(test::allocation_count() == before);
        }
    }
}

// 2. wait_k 5 -> 2 mid-utterance during kRead (t_src == 3, t_tgt == 0).
void test_s2ut_wait_k_live() {
    for (int fps : {1, 4}) {
        Live L(fps, true, 1, 0, 5);
        RecordingObserver obs;
        L.p->set_s2ut_observer(&obs);
        L.hops(3 * L.hpc);
        CHECK(obs.decisions() == "RRR");
        CHECK(L.p->waitk()->t_src() == 3 && L.p->waitk()->t_tgt() == 0 && L.p->waitk()->k() == 5);
        CHECK(L.rig.tenc.src_pos() == 3);
        const Resets before(L.rig);
        VoiceParams v = L.p->params();
        CHECK(v.get(ParamId::WaitK) == 5.0f);
        v.set(ParamId::WaitK, 2.0f);
        CHECK(L.p->set_params(v));
        CHECK(L.p->waitk()->k() == 2);
        CHECK(L.p->waitk()->t_src() == 3 && L.p->waitk()->t_tgt() == 0);   // no reset
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
        CHECK(L.rig.enc.resets == before.enc && L.rig.voc.resets == before.voc);
        CHECK(L.p->mode() == PipelineMode::kS2UT);
        // the 3 buffered chunks are still in the encoder ring, slot 3 still empty
        const std::size_t dim = L.rig.tenc.dim();
        for (std::size_t slot = 0; slot < 3; ++slot) {
            float mag = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) mag += std::abs(L.rig.tenc.k_ring()[slot * dim + d]);
            CHECK(mag > 0.5f);                                // k = v + src_lang(1)
        }
        for (std::size_t d = 0; d < dim; ++d) CHECK(L.rig.tenc.k_ring()[3 * dim + d] == 0.0f);

        L.hops(L.hpc);                                        // the very next chunk
        CHECK(obs.decisions() == "RRRW");
        CHECK(obs.events[3].e.t_src == 4 && obs.events[3].e.t_tgt == 1);
        CHECK(L.rig.tenc.src_pos() == 4 && L.rig.tdec.steps == 1);
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
    }
}

// 3. adaptive_policy toggled live: no reset, counters kept.
void test_s2ut_adaptive_toggle_live() {
    Live L(4, true, 1, 0, 3);
    L.hops(6);                                                // R R W W W W -> t_src 6, t_tgt 4
    CHECK(L.p->waitk()->t_src() == 6 && L.p->waitk()->t_tgt() == 4);
    CHECK(!L.p->waitk()->config().adaptive_mode);
    const Resets before(L.rig);
    for (const bool on : {true, false, true, false}) {
        VoiceParams v = L.p->params();
        v.set(ParamId::AdaptivePolicy, on ? 1.0f : 0.0f);
        CHECK(L.p->set_params(v));
        CHECK(L.p->waitk()->config().adaptive_mode == on);
        CHECK(L.p->waitk()->t_src() == 6 && L.p->waitk()->t_tgt() == 4);
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
        CHECK(L.rig.enc.resets == before.enc && L.rig.voc.resets == before.voc);
        CHECK(L.p->mode() == PipelineMode::kS2UT && L.rig.tenc.src_pos() == 6);
    }
    L.hops(2);                                                // translation continues
    CHECK(L.p->waitk()->t_src() == 8 && L.rig.tdec.steps == 6);
    // The constructor copies the initial adaptive flag into params().
    Rig r(4);
    S2utOptions o = s2ut_opts();
    o.policy.adaptive_mode = true;
    auto p = r.make(o);
    CHECK(p->params().get(ParamId::AdaptivePolicy) == 1.0f && p->waitk()->config().adaptive_mode);
    CHECK(p->params().get(ParamId::WaitK) == 3.0f && p->params().get(ParamId::Translate) == 1.0f);
}

// 4. Rejections.
void test_s2ut_param_rejections() {
    {   // translating: bad ids / src == tgt are refused, params and models untouched
        Live L(4, true, 1, 0);
        L.hops(4);
        const VoiceParams before_params = L.p->params();
        const Resets before(L.rig);
        const auto t_src = L.p->waitk()->t_src();
        auto expect_rejected = [&](std::int64_t src, std::int64_t tgt, bool translate) {
            CHECK(!L.apply(translate, src, tgt));
            CHECK(L.p->params() == before_params);
            CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
            CHECK(L.p->waitk()->t_src() == t_src && L.p->mode() == PipelineMode::kS2UT);
        };
        expect_rejected(9, 0, true);
        expect_rejected(1, 9, true);
        expect_rejected(2, 2, true);
        expect_rejected(1, 1, true);
        // an identical snapshot is accepted and is not a switch
        CHECK(L.p->set_params(before_params));
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
        CHECK(L.p->waitk()->t_src() == t_src);
        // a voice index out of range is refused, too
        VoiceParams bad = before_params;
        bad.voice_a = 5;
        CHECK(!L.p->set_params(bad) && L.p->params() == before_params);
    }
    {   // not translating: ids are stored unvalidated, no reset; translate = 1 then validates
        Live L(4, false, 1, 0);
        L.hops(3);
        const Resets before(L.rig);
        CHECK(L.apply(false, 9, 2) && L.p->params().src_lang_id == 9);     // unknown id
        CHECK(L.apply(false, 2, 2) && L.p->params().tgt_lang_id == 2);     // src == tgt
        CHECK(L.apply(false, 3, 0));                                       // plain change
        CHECK(L.p->params().src_lang_id == 3 && L.p->params().tgt_lang_id == 0);
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
        CHECK(L.p->mode() == PipelineMode::kVC);
        CHECK(L.apply(false, 9, 9));
        const VoiceParams stored = L.p->params();
        CHECK(!L.apply(true, 9, 9));                                       // stale ids refused
        CHECK(!L.apply(true, 2, 2));
        CHECK(!L.apply(true, 1, 9));
        CHECK(L.p->params() == stored && L.p->mode() == PipelineMode::kVC);
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
        CHECK(L.apply(true, 1, 3));                                        // valid and distinct
        CHECK(L.p->mode() == PipelineMode::kS2UT);
        CHECK(L.rig.tenc.resets == before.tenc + 1 && L.rig.tdec.resets == before.tdec + 1);
    }
    {   // Mode A pipeline (2-model constructor)
        FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, kVoice, kEmbedding, raw_options());
        CHECK(!p.has_translator() && p.mode() == PipelineMode::kVC);
        const VoiceParams before = p.params();
        VoiceParams v = before;
        v.set(ParamId::Translate, 1.0f);
        CHECK(!p.set_params(v));
        CHECK(p.params() == before && p.mode() == PipelineMode::kVC);
        v = before;
        v.set(ParamId::PitchShift, 2.0f);
        CHECK(p.set_params(v) && p.params().get(ParamId::PitchShift) == 2.0f);
    }
    {   // the 4-model constructor has translator models, even when starting as VC
        Live L(4, false, 1, 0);
        CHECK(L.p->has_translator());
    }
}

// 5. A voice-only preset keeps the translation controls and does not reset.
void test_s2ut_voice_preset_keeps_translation() {
    namespace fs = std::filesystem;
    const fs::path file = fs::path(FLUXVOICE_SOURCE_DIR) / "presets" / "deep.json";
    const auto loaded = load_preset_file(file);
    CHECK(loaded.ok());
    if (!loaded.ok()) return;
    const Preset& deep = *loaded.preset;
    CHECK(!deep.has(ParamId::Translate) && !deep.has(ParamId::WaitK) && !deep.has(ParamId::AdaptivePolicy));
    CHECK(!deep.src_lang && !deep.tgt_lang && deep.has(ParamId::PitchShift));

    for (int fps : {1, 4}) {
        Live L(fps, true, 2, 1, 4);
        L.hops(5 * L.hpc);                                   // translation in flight
        const auto t_src = L.p->waitk()->t_src();
        const auto t_tgt = L.p->waitk()->t_tgt();
        CHECK(t_src == 5 && t_tgt == 2);                     // k = 4: first write at the 4th chunk
        const Resets before(L.rig);
        VoiceParams v = L.p->params();
        apply_preset_params(deep, v);
        CHECK(L.p->set_params(v));
        const VoiceParams& now = L.p->params();
        CHECK(now.get(ParamId::PitchShift) == deep.params[param_index(ParamId::PitchShift)].value());
        CHECK(now.get(ParamId::PitchShift) < 0.0f);
        CHECK(now.get(ParamId::Translate) == 1.0f && L.p->mode() == PipelineMode::kS2UT);
        CHECK(now.src_lang_id == 2 && now.tgt_lang_id == 1);
        CHECK(now.get(ParamId::WaitK) == 4.0f && L.p->waitk()->k() == 4);
        CHECK(L.rig.tenc.resets == before.tenc && L.rig.tdec.resets == before.tdec);
        CHECK(L.rig.enc.resets == before.enc && L.rig.voc.resets == before.voc);
        CHECK(L.p->waitk()->t_src() == t_src && L.p->waitk()->t_tgt() == t_tgt);
        L.hops(L.hpc);
        CHECK(L.p->waitk()->t_src() == 6 && L.p->waitk()->t_tgt() == 3);
    }
}

// ===========================================================================
// Review follow-ups: rules the first Mode B tests left unpinned.
// Every test names the mutation it kills.
// ===========================================================================

// Process a pre-built signal hop by hop (no recording).
void feed(InferencePipeline& p, std::span<const float> sig) {
    const std::size_t hop = p.geometry().hop;
    std::vector<float> out(hop);
    for (std::size_t h = 0; h * hop < sig.size(); ++h)
        CHECK(p.process(sig.subspan(h * hop, hop), out).ok);
}

// a. Adaptive double write: the re-check after the first write.
// Kills: removing `if (w > 0 && p.decide() == kRead) break;` in chunk_step()
// (a 2nd decoder step would run after an <EOS> in a pause, t_tgt would pass
// t_src and the frames after the <EOS> would be pushed).
// Also pins f2: an <EOS> during a pause followed by a caught-up end counts eos_ends.
// Kills (f2): not setting utt_eos for an <EOS> emitted in a pause (eos_ends 0).
void test_s2ut_adaptive_double_write_recheck() {
    Rig r(4);
    r.tdec.rule = EosRule::kNever;
    r.tdec.emit_eos_on_step = 1;                            // the 1st write of the double-write step
    r.tdec.emit_eos_at_frame = 0;                           // [EOS, PAD, PAD, PAD]
    r.tenc.readiness_logit_value = -10.0f;                  // adaptive: never "ready"
    auto p = r.make(s2ut_opts(1, 3, 5));
    RecordingObserver obs;
    p->set_s2ut_observer(&obs);
    run(*p, r, "SSSS");                                     // k = 5: four reads, t_src = 4
    CHECK(obs.decisions() == "RRRR");
    CHECK(p->waitk()->t_src() == 4 && p->waitk()->t_tgt() == 0);
    VoiceParams v = p->params();
    v.set(ParamId::WaitK, 1.0f);
    v.set(ParamId::AdaptivePolicy, 1.0f);
    CHECK(p->set_params(v));
    CHECK(p->waitk()->config().adaptive_mode && p->waitk()->k() == 1);

    run(*p, r, ".");                                        // a pause chunk: lag 5 > k + 2 -> two writes allowed
    CHECK(obs.events.size() == 5);
    const auto& ev = obs.events[4];
    CHECK(ev.e.decision == WaitKPolicy::Decision::kWrite && !ev.e.speech && ev.e.encoded);
    CHECK(ev.e.writes == 1);                                // the second write was not run
    CHECK(r.tdec.steps == 1 && p->s2ut_telemetry().dec_steps == 1);
    CHECK(ev.units == std::vector<std::int64_t>({17, 18, 18, 18}));
    CHECK(ev.e.eos && ev.e.frames_pushed == 0 && ev.e.pad_frames == 0);
    CHECK(r.voc.steps == 0);                                // nothing after the <EOS> was vocoded
    CHECK(ev.e.t_src == 5 && ev.e.t_tgt == 5);              // caught up, never beyond t_src

    run(*p, r, "......");                                   // silence: no more writes, end after > 400 ms
    CHECK(r.tdec.steps == 1);
    CHECK(obs.events.size() == 11);
    for (std::size_t i = 5; i < 9; ++i) CHECK(!obs.events[i].e.utterance_end && obs.events[i].e.writes == 0);
    CHECK(obs.events[9].e.utterance_end && obs.events[9].e.writes == 0 && !obs.events[9].e.cap_end);   // 480 ms > 400 ms
    const auto t = p->s2ut_telemetry();
    CHECK(t.utterances == 1 && t.eos_ends == 1 && t.cap_ends == 0);
}

// f1. <EOS> during speech is ignored by the policy; a flush ending on the
// ratio bound is a cap end.
// Kills: deriving eos_ends from "any <EOS> this utterance" / ev.eos instead of
// the last flush write (eos_ends 1, cap_ends 0).
void test_s2ut_eos_during_speech_then_cap_end() {
    Rig r(4);
    r.tdec.rule = EosRule::kNever;
    r.tdec.emit_eos_on_step = 3;                            // chunk 4, still speaking
    r.tdec.emit_eos_at_frame = 0;
    auto p = r.make(s2ut_opts(1, 3));
    RecordingObserver obs;
    p->set_s2ut_observer(&obs);
    run(*p, r, chunks(10, 20));
    CHECK(obs.events[4].e.eos && obs.events[4].e.speech && !obs.events[4].e.utterance_end);
    CHECK(obs.events[4].e.frames_pushed == 0);
    std::size_t end = obs.events.size();
    for (std::size_t i = 0; i < obs.events.size(); ++i)
        if (obs.events[i].e.utterance_end) end = i;
    CHECK(end < obs.events.size());
    if (end < obs.events.size()) {
        CHECK(obs.events[end].e.cap_end && !obs.events[end].e.eos);
        CHECK(obs.events[end].e.decision == WaitKPolicy::Decision::kFlush);
    }
    const auto t = p->s2ut_telemetry();
    CHECK(t.utterances == 1 && t.eos_ends == 0 && t.cap_ends == 1);
}

// b. Governor hint on the edge hops.
void expect_hint(Live& L, bool on) {
    auto& g = L.p->governor();
    const std::vector<float> quiet(L.hop, 0.0f);
    CHECK(g.cushion_active() == on && g.hold_active() == on);
    // target + 150 ms + 1 sample is a hard drop unless the cushion raised the cap.
    const auto q = static_cast<std::int64_t>(g.target_samples() + 2400 + 1);
    const std::size_t got = g.skip_output(q, quiet);
    if (on) CHECK(got == 0);
    else CHECK(got > 0);
}

// Kills: dropping `vocode ||` from in_flight (final-frame hop: cushion off),
// dropping `was_vocoding ||` (the 2 ms ramp-down hop: hint off), dropping
// `switch_ramp ||` (language-switch hop), removing the B->A
// governor.set_translation_state(switch_ramp, switch_ramp) in process_vc(), or
// not clearing the hint on the hop after (stale hold would block trimming).
void test_s2ut_governor_edge_hops() {
    for (int fps : {1, 4}) {
        {   // end of an utterance
            Live L(fps, true, 2, 1);
            const std::size_t final_hop = 19 * L.hpc - 2;   // last frames of the chunk-17 write
            for (std::size_t h = 0; h < 24 * L.hpc; ++h) {
                L.hops(1, false, h < 10 * L.hpc);
                const std::size_t vs = L.rig.voc.steps;
                if (h + 1 == final_hop || h == final_hop || h == final_hop + 1) expect_hint(L, true);
                if (h == final_hop + 2) expect_hint(L, false);
                (void)vs;
            }
        }
        {   // B -> A switch, speech continues on the voice-changer route
            Live L(fps, true, 2, 1);
            L.hops(10 * L.hpc, false);
            CHECK(L.apply(false, 2, 1));
            L.hops(1, false);
            expect_hint(L, true);                           // crossfade hop from the last translated sample
            L.hops(1, false);
            expect_hint(L, false);                          // back to (false, false) one hop later
            L.hops(2, false);
            expect_hint(L, false);
        }
        {   // language switch while translating
            Live L(fps, true, 2, 1);
            L.hops(8 * L.hpc, false);
            CHECK(L.apply(true, 0, 3));
            L.hops(1, false, false);
            expect_hint(L, true);
            L.hops(1, false, false);
            expect_hint(L, false);
        }
    }
}

// c. Click ramps with the output chain on (unity when the gate is open).
PipelineOptions unity_chain() {
    PipelineOptions po;
    po.params.set(ParamId::Limiter, 0.0f);
    po.params.set(ParamId::Gain, 0.0f);
    po.params.set(ParamId::Brightness, 0.0f);
    return po;
}

// Ramped output mute on the Mode B route: the vocoder keeps running (mute is
// an output-stage level, not a model state), the output goes silent after the
// 5 ms ramp without a click, and output_chain = false ignores it.
void test_s2ut_mute() {
    const std::string pattern = chunks(10, 10);
    const auto prepare = [](Rig& r) {
        r.tdec.rule = EosRule::kCatchUp;
        r.voc.ignore_voicing = true;
    };
    S2utOptions o = s2ut_opts(1, 3);
    o.policy.tail_flush_ms = 0;

    Rig base_rig(4);
    prepare(base_rig);
    auto base = base_rig.make(o, unity_chain());
    const auto a = run(*base, base_rig, pattern);
    std::size_t first = a.out.size(), last = 0;
    for (std::size_t h = 0; h < a.out.size(); ++h)
        if (a.voc_ran[h]) {
            first = std::min(first, h);
            last = h;
        }
    CHECK(last >= first + 5);   // enough translated hops to mute in the middle

    Rig mute_rig(4);
    prepare(mute_rig);
    auto muted = mute_rig.make(o, unity_chain());
    const std::size_t at = first + 1;   // takes effect from hop `at + 1`
    const auto b = run(*muted, mute_rig, pattern, [&](std::size_t h) {
        if (h == at) muted->set_muted(true);
    });
    float prev = b.out[at].back();
    double max_step = 0.0, base_step = 0.0;
    for (const float x : b.out[at + 1]) {
        max_step = std::max(max_step, static_cast<double>(std::fabs(x - prev)));
        prev = x;
    }
    float base_prev = a.out[at].back();
    for (const float x : a.out[at + 1]) {
        base_step = std::max(base_step, static_cast<double>(std::fabs(x - base_prev)));
        base_prev = x;
    }
    CHECK(max_step <= base_step * 1.05 + 1e-4);   // the ramp adds no step beyond the signal's own
    for (std::size_t h = at + 2; h <= last; ++h) {
        CHECK(b.voc_ran[h]);   // the models were not paused
        for (const float x : b.out[h]) CHECK(std::fabs(x) < 1e-3f);
    }
    CHECK(std::fabs(a.out[last][0]) > 0.05f);   // the unmuted run does play there

    Rig raw_rig(4);
    prepare(raw_rig);
    auto raw = raw_rig.make(o);   // raw_options(): output_chain = false
    const auto c = run(*raw, raw_rig, pattern, [&](std::size_t h) {
        if (h == at) raw->set_muted(true);
    });
    Rig raw_ref_rig(4);
    prepare(raw_ref_rig);
    auto raw_ref = raw_ref_rig.make(o);
    const auto d = run(*raw_ref, raw_ref_rig, pattern);
    CHECK(c.out == d.out);   // documented: no chain, no mute
}

// Kills: removing ramp_up (the 2 ms onset), ramp_down on a speech->silence hop
// (a step to 0 after the last translated sample), ramp_down of the BUG-028 pad
// cut (step to 0 at the cut); and raw mode gaining ramps (exact zeros).
void test_s2ut_click_ramps() {
    for (int fps : {1, 4}) {
        Rig raw_rig(fps), chain_rig(fps);
        for (Rig* r : {&raw_rig, &chain_rig}) {
            r->tdec.rule = EosRule::kCatchUp;
            r->voc.ignore_voicing = true;
        }
        S2utOptions o = s2ut_opts(1, 3);
        o.policy.tail_flush_ms = 0;                         // 10 translated chunks, last unit 14
        auto raw = raw_rig.make(o);
        auto chained = chain_rig.make(o, unity_chain());
        const std::string pattern = chunks(10, 10);
        const auto a = run(*raw, raw_rig, pattern);
        const auto b = run(*chained, chain_rig, pattern);
        std::size_t first = a.out.size(), last = 0;
        for (std::size_t h = 0; h < a.out.size(); ++h)
            if (a.voc_ran[h]) {
                first = std::min(first, h);
                last = h;
            }
        const float x_first = a.out[first][0];
        CHECK(x_first > 0.1f);                              // 3/16
        // onset: silence -> speech scales up over 2 ms (32 samples) from ~0
        CHECK(a.out[first][0] == x_first);                  // raw keeps the exact plateau
        // (the chain's own gate attack multiplies the ramp, so the bound is "at most
        // the 2 ms linear ramp", starting near zero and monotone)
        for (std::size_t i = 0; i < 32; ++i) {
            CHECK(b.out[first][i] <= x_first * static_cast<float>(i + 1) / 32.0f + 1e-6f);
            if (i > 0) CHECK(b.out[first][i] >= b.out[first][i - 1]);
        }
        CHECK(b.out[first][0] < 0.01f * x_first);           // without ramp_up this is x/32
        CHECK(std::abs(b.out[first][40] - x_first) < 1e-5f);
        // offset: the hop after the last translated hop decays from the held sample
        const float held = a.out[last].back();
        CHECK(held > 0.5f);                                 // unit 14 -> 14/16
        CHECK(std::abs(b.out[last].back() - held) < 1e-5f);
        const auto& tail_raw = a.out[last + 1];
        const auto& tail = b.out[last + 1];
        CHECK(std::all_of(tail_raw.begin(), tail_raw.end(), [](float x) { return x == 0.0f; }));
        for (std::size_t i = 0; i < 32; ++i)
            CHECK(std::abs(tail[i] - held * static_cast<float>(31 - i) / 32.0f) < 1e-5f);
        CHECK(tail[0] > 0.9f * held * 31.0f / 32.0f);       // no step at the boundary
        for (std::size_t i = 32; i < tail.size(); ++i) CHECK(std::abs(tail[i]) < 1e-6f);
    }
    {   // BUG-028: the pad cut decays from the last real sample instead of stepping
        Rig raw_rig(4), chain_rig(4);
        for (Rig* r : {&raw_rig, &chain_rig}) {
            r->voc.ignore_voicing = true;
            r->tdec.emit_eos_on_step = 14;
            r->tdec.emit_eos_at_frame = 2;
        }
        auto raw = raw_rig.make(s2ut_opts(1, 3));
        auto chained = chain_rig.make(s2ut_opts(1, 3), unity_chain());
        const auto a = run(*raw, raw_rig, chunks(10, 10));
        const auto b = run(*chained, chain_rig, chunks(10, 10));
        const float real = 10.0f / 16.0f;
        CHECK(a.voc_ran[15] && b.voc_ran[15]);
        for (std::size_t i = 640; i < 1280; ++i) CHECK(a.out[15][i] == 0.0f);   // raw: exact zeros
        CHECK(std::abs(b.out[15][639] - real) < 1e-5f);
        for (std::size_t i = 0; i < 32; ++i)
            CHECK(std::abs(b.out[15][640 + i] - real * static_cast<float>(31 - i) / 32.0f) < 1e-5f);
        CHECK(b.out[15][640] > 0.9f * real * 31.0f / 32.0f);
        for (std::size_t i = 672; i < 1280; ++i) CHECK(std::abs(b.out[15][i]) < 1e-6f);
    }
}

// d. prosody_ctx = mean z of the VOICED tracker frames of the current utterance.
struct CtxMean {
    double f0 = 0.0, energy = 0.0;
    std::size_t n = 0;
};

std::vector<PitchFrame> reference_frames(const std::vector<float>& sig) {
    PitchTrackerConfig cfg;
    cfg.lookahead_frames = 0;
    PitchTracker tracker(cfg);
    std::vector<PitchFrame> frames;
    for (std::size_t i = 0; i + 320 <= sig.size(); i += 320) frames.push_back(tracker.push(sig.data() + i));
    return frames;
}

CtxMean mean_ctx(const std::vector<PitchFrame>& frames, std::size_t from, std::size_t to,
                 bool voiced_only = true) {
    CtxMean m;
    for (std::size_t i = from; i < to && i < frames.size(); ++i) {
        if (voiced_only && frames[i].voicing < 0.5) continue;
        m.f0 += frames[i].z_logf0;
        m.energy += frames[i].z_energy;
        ++m.n;
    }
    if (m.n) {
        m.f0 /= static_cast<double>(m.n);
        m.energy /= static_cast<double>(m.n);
    }
    return m;
}

void check_ctx(const FakeTranslatorDec& d, const CtxMean& want) {
    CHECK(std::abs(d.last_ctx(0) - static_cast<float>(want.f0)) < 1e-5f);
    CHECK(std::abs(d.last_ctx(1) - static_cast<float>(want.energy)) < 1e-5f);
}

// Kills: averaging unvoiced frames, accumulating across utterances (no clear
// at utterance end), no clear after a Stage 09 switch, prosody_ctx = false not
// zeroing the input, or a wrong denominator.
void test_s2ut_prosody_ctx() {
    {   // voiced sine: exactly the reference tracker's mean over the utterance so far
        const std::string pat = "SSSS";
        const auto sig = make_input(pat);
        const auto frames = reference_frames(sig);
        Rig r(4);
        auto p = r.make(s2ut_opts(1, 3));
        feed(*p, std::span<const float>(sig).first(3 * kChunkSamples));
        CHECK(r.tdec.steps == 1);
        const CtxMean want = mean_ctx(frames, 0, 12);
        CHECK(want.n > 6);
        CHECK(std::abs(want.f0) + std::abs(want.energy) > 1e-3);
        check_ctx(r.tdec, want);
        feed(*p, std::span<const float>(sig).subspan(3 * kChunkSamples));
        CHECK(r.tdec.steps == 2);
        check_ctx(r.tdec, mean_ctx(frames, 0, 16));
    }
    {   // prosody_ctx = false feeds [0, 0] (control: true feeds non-zero)
        const auto sig = make_input("SSS");
        Rig on(4), off(4);
        S2utOptions o = s2ut_opts(1, 3);
        auto p_on = on.make(o);
        o.prosody_ctx = false;
        auto p_off = off.make(o);
        feed(*p_on, sig);
        feed(*p_off, sig);
        CHECK(on.tdec.steps == 1 && off.tdec.steps == 1);
        CHECK(std::abs(on.tdec.last_ctx(0)) + std::abs(on.tdec.last_ctx(1)) > 1e-3f);
        CHECK(off.tdec.last_ctx(0) == 0.0f && off.tdec.last_ctx(1) == 0.0f);
    }
    {   // unvoiced noise of similar level is excluded; only the sine chunk counts
        const auto sig = make_input("NNNS");
        const auto frames = reference_frames(sig);
        Rig r(4);
        auto p = r.make(s2ut_opts(1, 3));
        feed(*p, std::span<const float>(sig).first(3 * kChunkSamples));
        CHECK(r.tdec.steps == 1 && mean_ctx(frames, 0, 12).n == 0);
        CHECK(r.tdec.last_ctx(0) == 0.0f && r.tdec.last_ctx(1) == 0.0f);
        feed(*p, std::span<const float>(sig).subspan(3 * kChunkSamples));
        CHECK(r.tdec.steps == 2);
        const CtxMean want = mean_ctx(frames, 0, 16);
        CHECK(want.n >= 1);
        const CtxMean all = mean_ctx(frames, 0, 16, false);          // what averaging every frame would give
        CHECK(std::abs(all.f0 - want.f0) + std::abs(all.energy - want.energy) > 1e-3);
        check_ctx(r.tdec, want);
    }
    {   // cleared after the utterance ends: the next one (noise) sees [0, 0]
        const std::string pat = chunks(10, 2) + "NNN";
        const auto sig = make_input(pat);
        const auto frames = reference_frames(sig);
        Rig r(4);
        r.tdec.rule = EosRule::kCatchUp;
        S2utOptions o = s2ut_opts(1, 3);
        o.policy.tail_flush_ms = 0;
        auto p = r.make(o);
        feed(*p, sig);
        CHECK(p->s2ut_telemetry().utterances == 1 && r.tdec.steps == 11);   // 10 + the first of utterance 2
        const CtxMean want = mean_ctx(frames, 48, 60);               // frames since the utterance end
        const CtxMean stale = mean_ctx(frames, 0, 60);
        CHECK(std::abs(stale.f0) + std::abs(stale.energy) > 1e-3);
        check_ctx(r.tdec, want);
    }
    {   // cleared by a Stage 09 switch: the first write after it ignores the older sine
        const auto sine = make_input(std::string(8, 'S'));
        const auto noise = make_input("NNN");
        std::vector<float> sig = sine;
        sig.insert(sig.end(), noise.begin(), noise.end());
        const auto frames = reference_frames(sig);
        Rig r(4);
        auto p = r.make(s2ut_opts(1, 3));
        feed(*p, sine);
        const std::size_t steps = r.tdec.steps;
        VoiceParams v = p->params();
        v.src_lang_id = 0;
        v.tgt_lang_id = 3;
        CHECK(p->set_params(v));
        feed(*p, noise);
        CHECK(r.tdec.steps == steps + 1);
        const CtxMean want = mean_ctx(frames, 32, 44);
        CHECK(std::abs(mean_ctx(frames, 0, 44).f0) + std::abs(mean_ctx(frames, 0, 44).energy) > 1e-3);
        check_ctx(r.tdec, want);
    }
}

// e. Readiness: non-saturated logits through sigmoid + threshold. Expected
// decisions come from python_training/fvml/waitk_policy.py (adaptive, k = 3,
// readiness = sigmoid(logit), 10 speech chunks).
// Kills: sigmoid removed (logit used as a probability), threshold ignored
// (0.378 would bank), the readiness not forwarded at all.
void test_s2ut_readiness_sigmoid() {
    struct Case { float logit; const char* decisions; std::size_t dec_steps; };
    const Case cases[] = {
        {0.0f, "RRWWRWRWWW", 6},     // 0.5 >= 0.5 banks 0.5 per chunk
        {-0.5f, "RRRRWWWWWW", 6},    // 0.378 < 0.5 discarded: strict k + extra wait
        {1.0f, "RRWWWWRWWW", 7},     // 0.731 banks
    };
    for (const Case& c : cases) {
        Rig r(4);
        S2utOptions o = s2ut_opts();
        o.policy.adaptive_mode = true;
        r.tenc.readiness_logit_value = c.logit;
        r.tdec.rule = EosRule::kNever;
        auto p = r.make(o);
        RecordingObserver obs;
        p->set_s2ut_observer(&obs);
        run(*p, r, std::string(10, 'S'));
        CHECK(obs.decisions() == c.decisions);
        CHECK(r.tdec.steps == c.dec_steps);
    }
}

// g. The stream clock counts Mode A hops too.
// Kills: advancing input_samples only on translating hops (event times would
// lag after a 1 -> 0 -> 1 toggle).
void test_s2ut_input_clock_across_toggles() {
    for (int fps : {1, 4}) {
        Live L(fps, true, 2, 1);
        RecordingObserver obs;
        L.p->set_s2ut_observer(&obs);
        L.hops(4 * L.hpc, false);
        CHECK(obs.events.size() == 4);
        CHECK(L.apply(false, 2, 1));
        L.hops(2 * L.hpc, false);                          // Mode A hops: no events
        CHECK(obs.events.size() == 4);
        CHECK(L.apply(true, 2, 1));
        L.hops(3 * L.hpc, false);
        CHECK(obs.events.size() == 7);
        for (std::size_t i = 0; i < obs.events.size(); ++i) {
            const std::size_t hops_done = i < 4 ? (i + 1) * L.hpc : (i + 3) * L.hpc;
            CHECK(obs.events[i].e.input_samples == hops_done * L.hop);
            CHECK(obs.events[i].e.chunk == i);             // chunk counter keeps going too
        }
        CHECK(obs.events[4].e.segment == 0);
        CHECK(obs.decisions().substr(4) == "RRW");         // restarted policy after the toggle
    }
}

// h. Constructor / set_params guards.
// Kills: a Mode A pipeline keeping translate = 1 in params() (set_params would
// then refuse its own snapshot); accepting adaptive mode without a readiness
// head; set_params() enabling adaptive_policy on a head-less translator.
void test_s2ut_mode_guards() {
    {
        FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
        PipelineOptions po = raw_options();
        po.params.set(ParamId::Translate, 1.0f);
        InferencePipeline p(enc, voc, kVoice, kEmbedding, po);
        CHECK(p.params().get(ParamId::Translate) == 0.0f);
        CHECK(p.set_params(p.params()));
        CHECK(!p.has_translator());
    }
    {
        auto c = test::translator_enc_contract();
        c.has_readiness_output = false;
        FakeDualEncoder enc(4);
        FakeTranslatorEnc blind(c);
        FakeTranslatorDec dec(blind);
        FakeUnitVocoder voc(4);
        const std::array<PipelineVoice, 1> voices{PipelineVoice{kFixtureVoice, kZeroEmbedding}};
        S2utOptions o = s2ut_opts();
        o.policy.adaptive_mode = true;
        rejects_with([&] {
            InferencePipeline p(enc, blind, dec, voc, std::span<const PipelineVoice>(voices), o, raw_options());
        }, "readiness");
        o.policy.adaptive_mode = false;
        InferencePipeline p(enc, blind, dec, voc, std::span<const PipelineVoice>(voices), o, raw_options());
        RecordingObserver obs;
        p.set_s2ut_observer(&obs);
        VoiceParams v = p.params();
        v.set(ParamId::AdaptivePolicy, 1.0f);
        CHECK(p.set_params(v));
        CHECK(p.params().get(ParamId::AdaptivePolicy) == 0.0f);
        CHECK(!p.waitk()->config().adaptive_mode);
        feed(p, make_input("SSSS"));
        CHECK(obs.decisions() == "RRWW");                  // strict Wait-k (adaptive would hold back)
    }
}
} // namespace

// ===========================================================================
// M4 Stage 06 — per-stage timing, injectable clock, governor mirror.
//
// Every expectation below is computed from the FakeClock script and the fakes'
// step_us, never read back from the pipeline. TimingHistogram bins are 50 us:
// a percentile is the upper bound of its bin, capped at the observed maximum,
// and a miss is a duration strictly above the hop.
// ===========================================================================

// Mode A, output chain on: the reads of one hop are, in order,
//   0 hop start | 1-2 features | 3-4 encoder | 5-6 vocoder | 7-8 chain | 9 hop end
// and FakeClock advances by deltas[k] after read k.
void test_stage_timing_mode_a_fake_clock() {
    constexpr std::size_t kReadsPerHop = 10;
    constexpr std::uint64_t kEncUs = 1000, kVocUs = 3000;
    // Two alternating hops. The chain is exactly one hop long (20000 us, not a
    // miss) on A and one us over on B (a miss). The other deltas are gaps
    // between brackets that no stage may include.
    const std::vector<std::uint64_t> a{11, 100, 13, 200, 17, 300, 19, 20000, 23, 7};
    const std::vector<std::uint64_t> b{11, 300, 13, 400, 17, 500, 19, 20001, 23, 7};
    std::vector<std::uint64_t> script = a;
    script.insert(script.end(), b.begin(), b.end());

    FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
    enc.step_us = kEncUs;
    voc.step_us = kVocUs;
    PipelineOptions options;
    options.clock_us = &FakeClock::read;
    InferencePipeline p(enc, voc, kVoice, kEmbedding, options);
    CHECK(p.geometry().hop == 320);   // hop_us 20000

    FakeClock::set(script);
    CHECK(p.warm_up(3));              // runs on the fake clock, then resets every histogram
    for (const TimingSnapshot& t : p.stage_timings()) CHECK(t.samples == 0);
    CHECK(p.timing().samples == 0);

    FakeClock::set(script);
    const auto in = signal(320);
    std::vector<float> out(320);
    constexpr std::size_t kHops = 10;   // 5 x A, 5 x B
    for (std::size_t h = 0; h < kHops; ++h) CHECK(p.process(in, out).ok);
    CHECK(FakeClock::reads == kHops * kReadsPerHop);   // no hidden clock reads

    const auto st = p.stage_timings();
    const auto at = [&](PipelineStage s) { return st[static_cast<std::size_t>(s)]; };
    // Feature stage: durations 100 (x5) and 300 (x5).
    CHECK(at(PipelineStage::kFeatures).samples == kHops);
    CHECK(at(PipelineStage::kFeatures).p50_us == 150 && at(PipelineStage::kFeatures).p95_us == 300);
    CHECK(at(PipelineStage::kFeatures).max_us == 300 && at(PipelineStage::kFeatures).misses == 0);
    // Encoder: delta + step_us = 1200 / 1400; vocoder: 3300 / 3500.
    CHECK(at(PipelineStage::kEncoder).samples == kHops);
    CHECK(at(PipelineStage::kEncoder).p50_us == 1250 && at(PipelineStage::kEncoder).p95_us == 1400);
    CHECK(at(PipelineStage::kEncoder).max_us == 1400 && at(PipelineStage::kEncoder).misses == 0);
    CHECK(at(PipelineStage::kVocoder).samples == kHops);
    CHECK(at(PipelineStage::kVocoder).p50_us == 3350 && at(PipelineStage::kVocoder).p95_us == 3500);
    CHECK(at(PipelineStage::kVocoder).max_us == 3500 && at(PipelineStage::kVocoder).misses == 0);
    // Chain: 20000 (equal to the hop: no miss) x5, 20001 (miss) x5.
    CHECK(at(PipelineStage::kOutputChain).samples == kHops);
    CHECK(at(PipelineStage::kOutputChain).max_us == 20001);
    CHECK(at(PipelineStage::kOutputChain).p50_us == 20001 && at(PipelineStage::kOutputChain).p95_us == 20001);
    CHECK(at(PipelineStage::kOutputChain).misses == kHops / 2);
    // The translator stages never run in Mode A.
    CHECK(at(PipelineStage::kTranslatorEnc).samples == 0 && at(PipelineStage::kTranslatorDec).samples == 0);
    // stage_timing(stage) agrees with the array.
    for (std::size_t i = 0; i < kPipelineStageCount; ++i) {
        const TimingSnapshot one = p.stage_timing(static_cast<PipelineStage>(i));
        CHECK(one.samples == st[i].samples && one.p50_us == st[i].p50_us &&
              one.p95_us == st[i].p95_us && one.max_us == st[i].max_us && one.misses == st[i].misses);
    }
    // The whole-hop histogram spans reads 0..9: every delta but the last, plus the model time.
    std::uint64_t hop_a = kEncUs + kVocUs, hop_b = kEncUs + kVocUs;
    for (std::size_t k = 0; k + 1 < kReadsPerHop; ++k) { hop_a += a[k]; hop_b += b[k]; }
    CHECK(hop_a == 24683 && hop_b == 25284);
    const TimingSnapshot hop = p.timing();
    CHECK(hop.samples == kHops && hop.max_us == hop_b && hop.misses == kHops);
    CHECK(hop.p50_us == 24700 && hop.p95_us == hop_b);

    // reset() keeps lifetime telemetry (like timing()); warm_up() zeroes it.
    p.reset();
    CHECK(p.stage_timings()[static_cast<std::size_t>(PipelineStage::kFeatures)].samples == kHops);
    CHECK(p.warm_up(1));
    for (const TimingSnapshot& t : p.stage_timings()) CHECK(t.samples == 0);

    // Without the output chain the chain stage records nothing and costs no reads.
    FakeModelStep enc2(fake_contract(ModelRole::kEncoder)), voc2(fake_contract(ModelRole::kVocoder));
    PipelineOptions raw = raw_options();
    raw.clock_us = &FakeClock::read;
    InferencePipeline q(enc2, voc2, kVoice, kEmbedding, raw);
    FakeClock::set({5});
    for (int h = 0; h < 4; ++h) CHECK(q.process(in, out).ok);
    CHECK(FakeClock::reads == 4 * 8);
    CHECK(q.stage_timing(PipelineStage::kOutputChain).samples == 0);
    CHECK(q.stage_timing(PipelineStage::kFeatures).samples == 4);
    CHECK(q.stage_timing(PipelineStage::kFeatures).max_us == 5);
    FakeClock::set({0});
}

// Mode B: the same stages on the translator route. A constant per-read cost r
// makes every duration exactly r + step_us (r alone for features and chain).
void test_stage_timing_mode_b_fake_clock() {
    constexpr std::uint64_t kRead = 40, kEnc = 1000, kTEnc = 2000, kTDec = 3000;
    constexpr std::uint64_t kHopUs = 80000;   // fps 4
    for (const bool over : {false, true}) {
        Rig r(4);
        r.enc.step_us = kEnc;
        r.tenc.step_us = kTEnc;
        r.tdec.step_us = kTDec;
        // Exactly one hop long (no miss), or one us over (a miss on every vocoder step).
        const std::uint64_t voc_us = kHopUs - kRead + (over ? 1 : 0);
        r.voc.step_us = voc_us;
        PipelineOptions po;   // output chain on
        po.clock_us = &FakeClock::read;
        auto p = r.make(s2ut_opts(), po);
        FakeClock::set({kRead});
        const std::string pattern = chunks(10, 10);
        const auto res = run(*p, r, pattern);
        const std::size_t hops = res.out.size();
        CHECK(hops == pattern.size());   // one hop per chunk at fps 4
        const auto st = p->stage_timings();
        const auto at = [&](PipelineStage s) { return st[static_cast<std::size_t>(s)]; };
        const auto exactly = [&](PipelineStage s, std::uint64_t samples, std::uint64_t us,
                                 std::uint64_t misses) {
            const TimingSnapshot t = at(s);
            CHECK(t.samples == samples);
            CHECK(t.p50_us == us && t.p95_us == us && t.max_us == us);
            CHECK(t.misses == misses);
        };
        // Sample counts come from the fakes' own step counters.
        exactly(PipelineStage::kFeatures, hops, kRead, 0);
        exactly(PipelineStage::kOutputChain, hops, kRead, 0);
        exactly(PipelineStage::kEncoder, r.enc.steps, kRead + kEnc, 0);
        exactly(PipelineStage::kTranslatorEnc, r.tenc.steps, kRead + kTEnc, 0);
        exactly(PipelineStage::kTranslatorDec, r.tdec.steps, kRead + kTDec, 0);
        exactly(PipelineStage::kVocoder, r.voc.steps, kRead + voc_us, over ? r.voc.steps : 0);
        CHECK(r.tenc.steps > 0 && r.tdec.steps > 0 && r.voc.steps > 0);   // the route really ran
        CHECK(p->mode() == PipelineMode::kS2UT);
        // kRead hops (no decoder, no vocoder) still run features and the chain.
        CHECK(r.voc.steps < hops);
    }
    FakeClock::set({0});
}

// The governor's target and hard cap are readable from any thread and follow
// the worker's state: the Mode B burst cushion raises both while a translation
// is in flight; a restart or a route switch clears it.
void test_governor_mirror() {
    // Mode A: fps 1 -> default target 320 + 160 + 80 = 560; hard cap + 150 ms.
    {
        FakeModelStep enc(fake_contract(ModelRole::kEncoder)), voc(fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, kVoice, kEmbedding);
        CHECK(p.governor_target_samples() == 560);               // before any hop
        CHECK(p.governor_hard_cap_samples() == 560 + 2400);
        const auto in = signal(320);
        std::vector<float> out(320);
        for (int i = 0; i < 5; ++i) CHECK(p.process(in, out).ok);
        CHECK(p.governor_target_samples() == 560 && p.governor_hard_cap_samples() == 2960);
        PipelineOptions custom;
        custom.latency_target_samples = 1000;
        FakeModelStep enc2(fake_contract(ModelRole::kEncoder)), voc2(fake_contract(ModelRole::kVocoder));
        InferencePipeline q(enc2, voc2, kVoice, kEmbedding, custom);
        CHECK(q.governor_target_samples() == 1000 && q.governor_hard_cap_samples() == 3400);
    }
    InferencePipeline legacy;   // no models: zeros, no throw
    CHECK(legacy.governor_target_samples() == 0 && legacy.governor_hard_cap_samples() == 0);

    // Mode B: fps 4 -> target 1280 + 640 + 160 = 2080; cushion 3 chunks = 3840.
    Rig r(4);
    auto p = r.make(s2ut_opts());
    constexpr std::size_t kBase = 2080, kCushion = 3840, kCap = 2400;
    CHECK(p->governor_target_samples() == kBase && p->governor_hard_cap_samples() == kBase + kCap);
    bool saw_base = false, saw_cushion = false;
    run(*p, r, chunks(6, 14), [&](std::size_t) {
        const std::size_t t = p->governor_target_samples();
        CHECK(t == p->governor().effective_target_samples());
        CHECK(p->governor_hard_cap_samples() == p->governor().hard_cap_samples());
        CHECK(p->governor_hard_cap_samples() == t + kCap);
        CHECK(t == kBase || t == kBase + kCushion);
        (t == kBase ? saw_base : saw_cushion) = true;
    });
    CHECK(saw_base && saw_cushion);
    CHECK(p->governor_target_samples() == kBase);   // utterance drained: cushion released

    // Speech in flight raises it; reset() restarts idle, set_params() publishes a switch at once.
    run(*p, r, "SSS");
    CHECK(p->governor_target_samples() == kBase + kCushion);
    p->reset();
    CHECK(p->governor_target_samples() == kBase && p->governor_hard_cap_samples() == kBase + kCap);
    run(*p, r, "SSS");
    CHECK(p->governor_target_samples() == kBase + kCushion);
    VoiceParams off = p->params();
    off.set(ParamId::Translate, 0.0f);
    CHECK(p->set_params(off));
    CHECK(p->governor_target_samples() == kBase && p->governor_hard_cap_samples() == kBase + kCap);
}

int main() {
    test_identity_and_lookahead();
    test_conditioning_and_reset();
    test_discrete_and_overlap();
    test_failures_and_gate();
    test_validation_and_no_allocations();
    test_s2ut_construction_rejects();
    test_s2ut_stage08_scenario();
    test_s2ut_default_policy_scenarios();
    test_s2ut_vocoder_bypass();
    test_s2ut_tail_flush_not_gated();
    test_s2ut_no_allocations();
    test_s2ut_warm_up();
    test_s2ut_bug028_partial_eos();
    test_s2ut_observer();
    test_s2ut_readiness_plumbing();
    test_s2ut_governor_hint();
    test_s2ut_idle();
    test_s2ut_live_switching();
    test_s2ut_wait_k_live();
    test_s2ut_adaptive_toggle_live();
    test_s2ut_param_rejections();
    test_s2ut_voice_preset_keeps_translation();
    test_s2ut_adaptive_double_write_recheck();
    test_s2ut_eos_during_speech_then_cap_end();
    test_s2ut_governor_edge_hops();
    test_s2ut_click_ramps();
    test_s2ut_mute();
    test_s2ut_prosody_ctx();
    test_s2ut_readiness_sigmoid();
    test_s2ut_input_clock_across_toggles();
    test_s2ut_mode_guards();
    test_stage_timing_mode_a_fake_clock();
    test_stage_timing_mode_b_fake_clock();
    test_governor_mirror();
    return test::finish("inference_pipeline");
}
