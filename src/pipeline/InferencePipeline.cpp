#include "pipeline/InferencePipeline.hpp"

#include "audio/OverlapAdd.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/ModelStep.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "voice/ParamSmoothing.hpp"
#include "voice/VoiceOutputChain.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice {
namespace {

constexpr double kLn2 = 0.69314718055994530942;
constexpr double kLn10 = 2.30258509299404568402;
constexpr double kTwoPi = 6.28318530717958647692;

PipelineGeometry checked_geometry(ModelStep& enc, ModelStep& voc) {
    PipelineGeometry g;
    std::string error;
    if (!make_pipeline_geometry(enc.contract(), voc.contract(), g, error))
        throw std::invalid_argument(error);
    // Stage 03 defines 20 ms conditioning, even for 80 ms content frames.
    if (g.sample_rate % 50 != 0 || g.cond_frame_samples != g.sample_rate / 50)
        throw std::invalid_argument("Mode A requires 20 ms pitch conditioning frames");
    return g;
}

PitchTrackerConfig tracker_config(const PipelineGeometry& g, const ModelContract& enc) {
    PitchTrackerConfig cfg;
    cfg.sample_rate = static_cast<int>(g.sample_rate);
    cfg.frame_samples = static_cast<int>(g.cond_frame_samples);
    cfg.window_samples = static_cast<int>(g.sample_rate / 25);
    cfg.lookahead_frames = static_cast<int>(
        static_cast<std::size_t>(enc.lookahead_frames) * g.frame_samples / g.cond_frame_samples);
    if (!cfg.valid()) throw std::invalid_argument("unsupported Mode A pitch geometry");
    return cfg;
}

void require_size(std::size_t got, std::size_t want, std::string_view name) {
    if (got != want)
        throw std::invalid_argument("model I/O size mismatch: " + std::string(name));
}

bool finite_voice(const TargetVoiceStats& v) {
    return std::isfinite(v.logf0_mean) && std::isfinite(v.logf0_std) && v.logf0_std > 0.0 &&
           std::isfinite(v.energy_mean) && std::isfinite(v.energy_std) && v.energy_std > 0.0;
}

// 50/50-style blend of every stats field — matches Stage 05's Python
// readiness check in gate_report.py, which blends stats with the embedding.
TargetVoiceStats lerp_stats(const TargetVoiceStats& a, const TargetVoiceStats& b,
                            double t) noexcept {
    return {a.logf0_mean + t * (b.logf0_mean - a.logf0_mean),
            a.logf0_std + t * (b.logf0_std - a.logf0_std),
            a.energy_mean + t * (b.energy_mean - a.energy_mean),
            a.energy_std + t * (b.energy_std - a.energy_std)};
}

const VoiceParams kNoParams{};

} // namespace

struct InferencePipeline::Legacy {
    OverlapAddProcessor ola{PipelineTiming::kChunkSamples, PipelineTiming::kOverlapSamples};
};

struct InferencePipeline::State {
    State(ModelStep& encoder, ModelStep& vocoder, std::span<const PipelineVoice> voices_in,
          PipelineOptions opts)
        : enc(encoder), voc(vocoder), g(checked_geometry(enc, voc)), options(opts),
          params(opts.params),
          tracker(tracker_config(g, enc.contract())), ola(g.hop, g.overlap_samples),
          governor(GovernorConfig{
              g.sample_rate, g.hop,
              opts.latency_target_samples ? opts.latency_target_samples :
                  LatencyGovernor::default_target(g.algorithmic_samples(), g.hop, g.sample_rate),
              3, opts.quiet_peak}),
          histogram(g.hop_us()),
          frame_levels(g.cond_frames, 0.0f),
          warm_input(g.hop, 0.0f), warm_output(g.hop, 0.0f) {
        if (&enc == &voc) throw std::invalid_argument("encoder and vocoder must be distinct models");
        if (voices_in.empty() || options.max_consecutive_errors == 0)
            throw std::invalid_argument("invalid Mode A voices or pipeline options");
        const std::size_t dim = static_cast<std::size_t>(voc.contract().spk_emb_dim);
        voice_stats.reserve(voices_in.size());
        embedding_table.reserve(voices_in.size() * dim);
        for (const PipelineVoice& v : voices_in) {
            if (!finite_voice(v.stats) || v.embedding.size() != dim)
                throw std::invalid_argument("invalid Mode A voice stats or embedding");
            for (float x : v.embedding)
                if (!std::isfinite(x)) throw std::invalid_argument("non-finite speaker embedding");
            voice_stats.push_back(v.stats);
            embedding_table.insert(embedding_table.end(), v.embedding.begin(), v.embedding.end());
        }
        params.clamp();
        if (static_cast<std::size_t>(params.voice_a) >= voice_stats.size() ||
            static_cast<std::size_t>(params.voice_b) >= voice_stats.size())
            throw std::invalid_argument("voice index out of range");
        enc_audio = enc.input_f32("audio");
        voc_audio = voc.output_f32("audio");
        require_size(enc_audio.size(), g.hop, "encoder audio");
        require_size(voc_audio.size(), g.hop + g.overlap_samples, "vocoder audio");
        if (enc.contract().content_kind == ContentKind::kDiscrete) {
            enc_units = enc.output_i64("units");
            voc_units = voc.input_i64("units");
            require_size(enc_units.size(), g.frames_per_step, "encoder units");
            require_size(voc_units.size(), enc_units.size(), "vocoder units");
        } else {
            enc_content = enc.output_f32("content");
            voc_content = voc.input_f32("content");
            require_size(enc_content.size(),
                         g.frames_per_step * static_cast<std::size_t>(enc.contract().content_dim),
                         "encoder content");
            require_size(voc_content.size(), enc_content.size(), "vocoder content");
        }
        constexpr std::array<std::string_view, 4> names{"logf0", "voicing", "aperiodicity", "energy"};
        const auto& inputs = voc.contract().cond_inputs;
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (std::find(inputs.begin(), inputs.end(), names[i]) != inputs.end()) {
                cond[i] = voc.input_f32(names[i]);
                require_size(cond[i].size(), g.cond_frames, names[i]);
            }
        }
        spk_input = voc.input_f32("spk_emb");
        require_size(spk_input.size(), dim, "vocoder spk_emb");
        eff_emb.assign(dim, 0.0f);
        xfade_emb.assign(dim, 0.0f);
        // Stage 09 initial state: controls snapped, no ramp at stream start.
        pitch_ramp.snap(params.get(ParamId::PitchShift));
        inton_ramp.snap(params.get(ParamId::Intonation));
        blend_ramp.snap(params.get(ParamId::TimbreBlend));
        vrate_ramp.snap(params.get(ParamId::VibratoRate));
        vdepth_ramp.snap(params.get(ParamId::VibratoDepth));
        whisper_ramp.snap(params.get(ParamId::Whisper));
        snapshot_target();
        xfade_ramp.snap(1.0f);
        chain.configure(g.sample_rate);
        reset_stream();
    }

    std::size_t emb_dim() const noexcept {
        return voice_stats.empty() ? 0 : embedding_table.size() / voice_stats.size();
    }
    const float* emb_of(std::size_t v) const noexcept {
        return embedding_table.data() + v * emb_dim();
    }
    // (1 - alpha) * e_A + alpha * e_B and the matching stats blend.
    void target_at(double alpha, float* emb, TargetVoiceStats& stats) const noexcept {
        const float* ea = emb_of(params.voice_a);
        const float* eb = emb_of(params.voice_b);
        for (std::size_t j = 0; j < emb_dim(); ++j)
            emb[j] = static_cast<float>((1.0 - alpha) * ea[j] + alpha * eb[j]);
        stats = lerp_stats(voice_stats[params.voice_a], voice_stats[params.voice_b], alpha);
    }
    void snapshot_target() noexcept {
        target_at(blend_ramp.value(), xfade_emb.data(), xfade_stats);
    }
    // Effective embedding + stats at the *current* ramp positions (used when a
    // voice switch snapshots the outgoing side of the crossfade). `emb` may
    // alias xfade_emb: each element is read before it is written.
    void current_effective(float* emb, TargetVoiceStats& stats) const noexcept {
        const double alpha = blend_ramp.value(), xf = xfade_ramp.value();
        const float* ea = emb_of(params.voice_a);
        const float* eb = emb_of(params.voice_b);
        for (std::size_t j = 0; j < emb_dim(); ++j)
            emb[j] = static_cast<float>((1.0 - xf) * xfade_emb[j] +
                                        xf * ((1.0 - alpha) * ea[j] + alpha * eb[j]));
        stats = lerp_stats(xfade_stats,
                           lerp_stats(voice_stats[params.voice_a], voice_stats[params.voice_b],
                                      alpha),
                           xf);
    }

    bool set_params(const VoiceParams& next) noexcept {
        VoiceParams p = next;
        p.clamp();
        if (static_cast<std::size_t>(p.voice_a) >= voice_stats.size() ||
            static_cast<std::size_t>(p.voice_b) >= voice_stats.size())
            return false;
        if (p.voice_a != params.voice_a || p.voice_b != params.voice_b) {
            // Voice switch: crossfade from the current effective state so even
            // switching voice_a at timbre_blend = 0 does not jump.
            current_effective(xfade_emb.data(), xfade_stats);
            xfade_ramp.snap(0.0f);
            xfade_ramp.set_target(1.0f);
        }
        params = p;
        pitch_ramp.set_target(p.get(ParamId::PitchShift));
        inton_ramp.set_target(p.get(ParamId::Intonation));
        blend_ramp.set_target(p.get(ParamId::TimbreBlend));
        vrate_ramp.set_target(p.get(ParamId::VibratoRate));
        vdepth_ramp.set_target(p.get(ParamId::VibratoDepth));
        whisper_ramp.set_target(p.get(ParamId::Whisper));
        return true;
    }

    void reset_stream() noexcept {
        enc.reset();
        voc.reset();
        tracker.reset();
        ola.reset();
        governor.reset();
        // Finish in-progress ramps at their targets (== params) so a reset is
        // indistinguishable from a freshly constructed pipeline, and finish any
        // voice crossfade by making the snapshot the current target.
        pitch_ramp.finish();
        inton_ramp.finish();
        blend_ramp.finish();
        vrate_ramp.finish();
        vdepth_ramp.finish();
        whisper_ramp.finish();
        snapshot_target();
        xfade_ramp.snap(1.0f);
        vibrato_phase = 0.0;
        chain.set_targets(params.get(ParamId::Gain), params.get(ParamId::GateThreshold),
                          params.enabled(ParamId::Limiter), params.get(ParamId::Brightness));
        chain.reset();
        last_output = 0.0f;
        recovering = false;
    }

    PipelineStepResult fail(std::string_view error, std::span<float> output, bool fatal) noexcept;
    PipelineStepResult process(std::span<const float> input, std::span<float> output) noexcept;

    ModelStep& enc;
    ModelStep& voc;
    const PipelineGeometry g;
    PipelineOptions options;
    VoiceParams params;
    PitchTracker tracker;
    OverlapAddProcessor ola;
    LatencyGovernor governor;
    TimingHistogram histogram;
    std::vector<TargetVoiceStats> voice_stats;
    std::vector<float> embedding_table;
    std::vector<float> eff_emb, xfade_emb;
    std::vector<float> frame_levels;
    std::vector<float> warm_input, warm_output;
    FrameRamp pitch_ramp{kParamRampFrames}, inton_ramp{kParamRampFrames},
        blend_ramp{kParamRampFrames}, vrate_ramp{kParamRampFrames},
        vdepth_ramp{kParamRampFrames}, whisper_ramp{kParamRampFrames};
    FrameRamp xfade_ramp{kVoiceCrossfadeFrames};
    TargetVoiceStats xfade_stats{};
    double vibrato_phase = 0.0;
    VoiceOutputChain chain;
    std::span<float> enc_audio, voc_content, spk_input;
    std::span<std::int64_t> voc_units;
    std::span<const float> enc_content, voc_audio;
    std::span<const std::int64_t> enc_units;
    std::array<std::span<float>, 4> cond{};
    std::array<char, 512> error_text{};
    std::size_t error_size = 0;
    std::atomic<std::uint64_t> errors{0};
    std::size_t consecutive_errors = 0;
    float last_output = 0.0f;
    bool recovering = false;
};

PipelineStepResult InferencePipeline::State::fail(std::string_view error,
                                                 std::span<float> output, bool fatal) noexcept {
    if (error.empty()) error = "model step failed without an error message";
    error_size = std::min(error.size(), error_text.size() - 1);
    std::copy_n(error.data(), error_size, error_text.data());
    error_text[error_size] = '\0';
    const std::size_t fade = std::min(output.size(), std::size_t{g.sample_rate} * 2 / 1000);
    const float previous = last_output;
    std::fill(output.begin(), output.end(), 0.0f);
    for (std::size_t i = 0; i < fade; ++i)
        output[i] = previous * static_cast<float>(fade - i - 1) / static_cast<float>(fade);
    // A failed Run may have partially advanced state. Restart both models so
    // the next hop cannot combine encoder and vocoder from different timelines.
    reset_stream();
    recovering = true;
    errors.fetch_add(1, std::memory_order_relaxed);
    ++consecutive_errors;
    return {false, fatal || consecutive_errors >= options.max_consecutive_errors,
            previous == 0.0f, {error_text.data(), error_size}};
}

PipelineStepResult InferencePipeline::State::process(std::span<const float> input,
                                                    std::span<float> output) noexcept {
    if (input.size() != g.hop || output.size() != g.hop)
        return fail("pipeline requires exactly one input/output hop", output, true);

    std::copy(input.begin(), input.end(), enc_audio.begin());
    for (std::size_t i = 0; i < g.cond_frames; ++i) {
        // The Stage 03 tracker owns HPF, EMA priors and lookahead delay. Keeping
        // encoder audio raw also makes identity bundles genuinely transparent.
        const PitchFrame f = tracker.push(enc_audio.data() + i * g.cond_frame_samples);
        // Input level for the gate, aligned with the content being vocoded
        // (the tracker applies the encoder lookahead).
        frame_levels[i] = static_cast<float>(f.log_energy * 10.0 / kLn10);
        // Advance the Stage 09 ramps once per conditioning frame.
        const double pitch = pitch_ramp.advance();
        const double intonation = inton_ramp.advance();
        const double alpha = blend_ramp.advance();
        const double vrate = vrate_ramp.advance();
        const double vdepth = vdepth_ramp.advance();
        const double whisper = whisper_ramp.advance();
        const double xf = xfade_ramp.advance();
        const TargetVoiceStats eff =
            lerp_stats(xfade_stats,
                       lerp_stats(voice_stats[params.voice_a], voice_stats[params.voice_b], alpha),
                       xf);
        // Scale the LFO by min(1, rate / 0.5 Hz): when vibrato_rate ramps to 0
        // the phase stops advancing and sin(phase) would otherwise freeze at a
        // static pitch offset; at rate 0 the contribution is exactly zero.
        const double vibrato = vdepth / 1200.0 * kLn2 * std::sin(vibrato_phase) *
                               std::min(1.0, vrate / 0.5);
        vibrato_phase += kTwoPi * vrate * 0.020;
        if (vibrato_phase >= kTwoPi) vibrato_phase = std::fmod(vibrato_phase, kTwoPi);
        const std::array<double, 4> mapped{
            map_logf0_to_target(f.z_logf0, eff, intonation, pitch) + vibrato,
            f.voicing * (1.0 - whisper), f.aperiodicity,
            map_energy_to_target(f.z_energy, eff)};
        for (std::size_t j = 0; j < cond.size(); ++j) {
            if (cond[j].empty()) continue;
            if (!std::isfinite(mapped[j]) || std::abs(mapped[j]) > std::numeric_limits<float>::max())
                return fail("non-finite model conditioning", output, false);
            cond[j][i] = static_cast<float>(mapped[j]);
        }
    }
    if (!enc.step()) return fail(enc.last_error(), output, false);
    if (!enc_content.empty()) std::copy(enc_content.begin(), enc_content.end(), voc_content.begin());
    if (!enc_units.empty()) std::copy(enc_units.begin(), enc_units.end(), voc_units.begin());
    if (!spk_input.empty()) {
        // The vocoder takes one vector per step: compute it once per hop from
        // the ramp state after the hop's last frame (per frame for s1, which
        // has one conditioning frame per hop anyway).
        const double alpha = blend_ramp.value(), xf = xfade_ramp.value();
        const float* ea = emb_of(params.voice_a);
        const float* eb = emb_of(params.voice_b);
        for (std::size_t j = 0; j < emb_dim(); ++j)
            eff_emb[j] = static_cast<float>((1.0 - xf) * xfade_emb[j] +
                                            xf * ((1.0 - alpha) * ea[j] + alpha * eb[j]));
        std::copy(eff_emb.begin(), eff_emb.end(), spk_input.begin());
    }
    if (!voc.step()) return fail(voc.last_error(), output, false);
    if (g.overlap_samples > 0) ola.process(voc_audio.data(), output.data());
    else std::copy(voc_audio.begin(), voc_audio.end(), output.begin());
    if (options.output_chain) {
        chain.set_targets(params.get(ParamId::Gain), params.get(ParamId::GateThreshold),
                          params.enabled(ParamId::Limiter), params.get(ParamId::Brightness));
        chain.process(output, frame_levels, g.cond_frame_samples);
    }

    if (recovering) {
        const std::size_t fade = std::min(output.size(), std::size_t{g.sample_rate} * 2 / 1000);
        for (std::size_t i = 0; i < fade; ++i)
            output[i] *= static_cast<float>(i + 1) / static_cast<float>(fade);
    }
    bool quiet = true;
    for (float x : output) {
        if (!std::isfinite(x)) return fail("vocoder/gate produced non-finite audio", output, false);
        quiet = quiet && std::abs(x) <= options.quiet_peak;
    }
    last_output = output.back();
    recovering = false;
    consecutive_errors = 0;
    error_size = 0;
    return {true, false, quiet, {}};
}

InferencePipeline::InferencePipeline() : legacy_(std::make_unique<Legacy>()) {}

InferencePipeline::InferencePipeline(ModelStep& encoder, ModelStep& vocoder,
                                     std::span<const PipelineVoice> voices, PipelineOptions options)
    : state_(std::make_unique<State>(encoder, vocoder, voices, options)) {}

InferencePipeline::InferencePipeline(ModelStep& encoder, ModelStep& vocoder,
                                     const TargetVoiceStats& voice,
                                     std::span<const float> embedding, PipelineOptions options)
    : InferencePipeline(encoder, vocoder,
                        std::span<const PipelineVoice>(
                            std::array<PipelineVoice, 1>{PipelineVoice{voice, embedding}}),
                        options) {}

InferencePipeline::~InferencePipeline() = default;

PipelineStepResult InferencePipeline::process(std::span<const float> input,
                                              std::span<float> output) noexcept {
    if (!state_) {
        std::fill(output.begin(), output.end(), 0.0f);
        return {false, true, true, "pipeline has no models"};
    }
    const auto start = std::chrono::steady_clock::now();
    const auto result = state_->process(input, output);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    state_->histogram.record(static_cast<std::uint64_t>(us));
    return result;
}

bool InferencePipeline::warm_up(std::size_t steps) {
    if (!state_ || steps == 0) return false;
    reset();
    bool ok = true;
    for (std::size_t i = 0; i < steps; ++i) {
        if (!process(state_->warm_input, state_->warm_output).ok) {
            ok = false;
            break;
        }
    }
    reset();
    state_->histogram.reset();
    state_->errors.store(0, std::memory_order_relaxed);
    return ok;
}

const PipelineGeometry& InferencePipeline::geometry() const {
    if (!state_) throw std::logic_error("pipeline has no models");
    return state_->g;
}

LatencyGovernor& InferencePipeline::governor() {
    if (!state_) throw std::logic_error("pipeline has no models");
    return state_->governor;
}

TimingSnapshot InferencePipeline::timing() const noexcept {
    return state_ ? state_->histogram.snapshot() : TimingSnapshot{};
}

std::uint64_t InferencePipeline::errors() const noexcept {
    return state_ ? state_->errors.load(std::memory_order_relaxed) : 0;
}

std::string_view InferencePipeline::last_error() const noexcept {
    return state_ ? std::string_view(state_->error_text.data(), state_->error_size) :
                    std::string_view("pipeline has no models");
}

bool InferencePipeline::set_params(const VoiceParams& next) noexcept {
    return state_ && state_->set_params(next);
}

const VoiceParams& InferencePipeline::params() const noexcept {
    return state_ ? state_->params : kNoParams;
}

std::size_t InferencePipeline::voice_count() const noexcept {
    return state_ ? state_->voice_stats.size() : 0;
}

void InferencePipeline::process_vocoder_frame(const float* in, float* out) {
    if (!legacy_) throw std::logic_error("use process() for Mode A models");
    legacy_->ola.process(in, out);
}

void InferencePipeline::reset() {
    if (legacy_) legacy_->ola.reset();
    if (state_) {
        state_->reset_stream();
        state_->consecutive_errors = 0;
    }
    ++resets_;
}

} // namespace fluxvoice
