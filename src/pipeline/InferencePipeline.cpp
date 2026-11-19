#include "pipeline/InferencePipeline.hpp"

#include "audio/OverlapAdd.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/ModelContract.hpp"
#include "pipeline/ModelStep.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "pipeline/UnitProsodyFifo.hpp"
#include "pipeline/WaitKPolicy.hpp"
#include "voice/ParamSmoothing.hpp"
#include "voice/VoiceOutputChain.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
// Mode B gate levels: a translated frame opens the gate at any user threshold
// (VoiceParams caps gate_threshold at -20 dBFS); silence and pad frames close it.
constexpr float kGateOpenDbfs = 0.0f;
constexpr float kGateClosedDbfs = -std::numeric_limits<float>::infinity();
constexpr std::size_t kMaxFifoFrames = 4096;

using Clock = std::chrono::steady_clock;

// The default PipelineOptions::clock_us.
std::uint64_t steady_now_us() noexcept {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        Clock::now().time_since_epoch()).count();
    return us > 0 ? static_cast<std::uint64_t>(us) : 0;
}

std::uint64_t elapsed_us(std::uint64_t t0, std::uint64_t t1) noexcept {
    return t1 > t0 ? t1 - t0 : 0;
}

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

// Mode B: the four-model bundle rules on top of the encoder/vocoder geometry.
PipelineGeometry checked_s2ut_geometry(ModelStep& enc, ModelStep& tenc, ModelStep& tdec,
                                       ModelStep& voc) {
    std::string error;
    for (const ModelStep* m : {&enc, &tenc, &tdec, &voc})
        if (!validate_contract(m->contract(), error)) throw std::invalid_argument(error);
    if (!validate_s2ut_bundle(enc.contract(), tenc.contract(), tdec.contract(), voc.contract(),
                              error))
        throw std::invalid_argument(error);
    PipelineGeometry g = checked_geometry(enc, voc);
    // Hops without translated frames skip the vocoder, so an OLA tail would
    // have no successor hop to land in.
    if (g.overlap_samples != 0)
        throw std::invalid_argument(
            "Mode B requires a stateful vocoder without overlap-add (overlap_samples = 0)");
    if (g.cond_frame_samples != g.frame_samples)
        throw std::invalid_argument("Mode B requires one conditioning frame per content frame");
    const int chunk = tenc.contract().chunk_frames;
    if (chunk < 1 || chunk > 16)
        throw std::invalid_argument("Mode B supports translator chunk_frames 1..16");
    g.chunk_frames = static_cast<std::size_t>(chunk);
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

// Stage 09 control values for one 20 ms conditioning frame.
struct FrameCtl {
    double pitch = 0.0;
    double intonation = 1.0;
    double whisper = 0.0;
    double vibrato = 0.0;   // log-f0 offset
    TargetVoiceStats eff{};
};

// Whitened frame -> vocoder conditioning (logf0, voicing, aperiodicity, energy).
std::array<double, 4> map_frame(const FrameCtl& c, double z_logf0, double voicing,
                                double aperiodicity, double z_energy) noexcept {
    return {map_logf0_to_target(z_logf0, c.eff, c.intonation, c.pitch) + c.vibrato,
            voicing * (1.0 - c.whisper), aperiodicity, map_energy_to_target(z_energy, c.eff)};
}

// 2 ms linear ramps (same length as the error fade): `from` -> 0, and 0 -> x.
void ramp_down(std::span<float> out, float from, std::uint32_t rate) noexcept {
    const std::size_t fade = std::min(out.size(), std::size_t{rate} * 2 / 1000);
    for (std::size_t i = 0; i < fade; ++i)
        out[i] = from * static_cast<float>(fade - i - 1) / static_cast<float>(fade);
}

void ramp_up(std::span<float> out, std::uint32_t rate) noexcept {
    const std::size_t fade = std::min(out.size(), std::size_t{rate} * 2 / 1000);
    for (std::size_t i = 0; i < fade; ++i)
        out[i] *= static_cast<float>(i + 1) / static_cast<float>(fade);
}

// Twin of fvml/s2ut_stream.py: the readiness logit enters WaitKPolicy as a
// probability in (0, 1). BUG-029: a [0, 2] expected-chunk mapping must change
// here and in the Python driver together, once the readiness head is
// retrained for it.
float readiness_from_logit(float logit) noexcept {
    return static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(logit))));
}

WaitKPolicy::Config policy_config(const S2utOptions& o, const PipelineGeometry& g) {
    if (o.policy.k < WaitKPolicy::kMinK || o.policy.k > WaitKPolicy::kMaxK)
        throw std::invalid_argument("wait-k must be in 1..10");
    WaitKPolicy::Config c = o.policy;
    c.chunk_frames = static_cast<int>(g.chunk_frames);
    c.frame_ms = static_cast<int>(std::uint64_t{g.frame_samples} * 1000 / g.sample_rate);
    return c;
}

std::size_t fifo_capacity(std::size_t requested, std::size_t chunk) {
    if (requested > kMaxFifoFrames)
        throw std::invalid_argument("Mode B FIFO capacity exceeds 4096 frames");
    const std::size_t want = std::max(requested, 2 * chunk);
    return (want + chunk - 1) / chunk * chunk;
}

const VoiceParams kNoParams{};

struct ModeBModels {
    ModelStep& tenc;
    ModelStep& tdec;
    const S2utOptions& opts;
};

// Mode B counters, written by the worker only (relaxed atomics, lock-free).
struct ModeBCounters {
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    static_assert(std::atomic<std::int64_t>::is_always_lock_free);
    static_assert(std::atomic<bool>::is_always_lock_free);

    std::atomic<std::uint64_t> reads{0}, writes{0}, flushes{0}, utterances{0}, eos_ends{0},
        cap_ends{0}, dec_steps{0}, voc_steps{0}, fifo_stalls{0}, pad_frames{0};
    std::atomic<std::int64_t> lag_chunks{0};

    static void bump(std::atomic<std::uint64_t>& c, std::uint64_t n = 1) noexcept {
        c.fetch_add(n, std::memory_order_relaxed);
    }
    void clear() noexcept {
        for (auto* c : {&reads, &writes, &flushes, &utterances, &eos_ends, &cap_ends,
                        &dec_steps, &voc_steps, &fifo_stalls, &pad_frames})
            c->store(0, std::memory_order_relaxed);
        lag_chunks.store(0, std::memory_order_relaxed);
    }
};

// Everything Mode B adds to the Mode A pipeline state. All buffers are sized
// here; the per-hop path only indexes into them.
struct ModeBState {
    ModeBState(ModelStep& enc, ModelStep& translator_enc, ModelStep& translator_dec,
               ModelStep& voc, const PipelineGeometry& g, const S2utOptions& o)
        : tenc(translator_enc), tdec(translator_dec), opts(o), policy(policy_config(o, g)),
          fifo(fifo_capacity(o.fifo_capacity_frames, g.chunk_frames), voc.contract().vocab,
               translator_dec.contract().special_units.eos),
          chunk(g.chunk_frames), fps(g.frames_per_step),
          latent_dim(static_cast<std::size_t>(enc.contract().latent_dim)),
          bos(translator_dec.contract().special_units.bos),
          chunk_ms(policy.chunk_ms()),
          chunk_latents(chunk * latent_dim, 0.0f), prev_units(chunk, bos),
          step_units(2 * chunk, 0), neutral_prosody(chunk * UnitProsodyFifo::kProsodyDims, 0.0f),
          hop_frames(fps), src_lang_id(o.src_lang_id), tgt_lang_id(o.tgt_lang_id) {
        const auto& tc = tenc.contract();
        lang_ids.reserve(tc.langs.size());
        for (const auto& kv : tc.langs) lang_ids.push_back(kv.second);
        if (!lang_ok(o.src_lang_id))
            throw std::invalid_argument("src_lang id " + std::to_string(o.src_lang_id) +
                                        " is not a translator language");
        if (!lang_ok(o.tgt_lang_id))
            throw std::invalid_argument("tgt_lang id " + std::to_string(o.tgt_lang_id) +
                                        " is not a translator language");
        if (o.src_lang_id == o.tgt_lang_id)
            throw std::invalid_argument("src_lang and tgt_lang must differ");
        if (!std::isfinite(o.speech_threshold_dbfs) ||
            !(o.voiced_threshold >= 0.0f && o.voiced_threshold <= 1.0f) ||
            !(o.eos_threshold >= 0.0f && o.eos_threshold <= 1.0f))
            throw std::invalid_argument("invalid Mode B VAD/voicing/eos thresholds");
        if (o.policy.adaptive_mode && !tc.has_readiness_output)
            throw std::invalid_argument(
                "adaptive policy needs a translator_enc readiness_logit output");

        enc_latents = enc.output_f32("latents");
        require_size(enc_latents.size(), fps * latent_dim, "encoder latents");
        tenc_latents = tenc.input_f32("latents");
        require_size(tenc_latents.size(), chunk * latent_dim, "translator_enc latents");
        tenc_src_lang = tenc.input_i64("src_lang");
        require_size(tenc_src_lang.size(), 1, "translator_enc src_lang");
        if (tc.has_readiness_output) {
            tenc_readiness = tenc.output_f32("readiness_logit");
            require_size(tenc_readiness.size(), 1, "translator_enc readiness_logit");
        }
        const auto& dc = tdec.contract();
        tdec_prev_units = tdec.input_i64("prev_units");
        require_size(tdec_prev_units.size(), chunk, "translator_dec prev_units");
        tdec_tgt_lang = tdec.input_i64("tgt_lang");
        require_size(tdec_tgt_lang.size(), 1, "translator_dec tgt_lang");
        tdec_prosody_ctx = tdec.input_f32("prosody_ctx");
        require_size(tdec_prosody_ctx.size(), 2, "translator_dec prosody_ctx");
        tdec_units = tdec.output_i64("units");
        require_size(tdec_units.size(), chunk, "translator_dec units");
        if (dc.has_prosody_output) {
            tdec_prosody = tdec.output_f32("prosody");
            require_size(tdec_prosody.size(), chunk * UnitProsodyFifo::kProsodyDims,
                         "translator_dec prosody");
        }
        tdec_eos = tdec.output_f32("eos_prob");
        require_size(tdec_eos.size(), 1, "translator_dec eos_prob");
        // A translator without a prosody head: mean pitch/energy, fully voiced.
        for (std::size_t i = 0; i < chunk; ++i) neutral_prosody[i * 4 + 1] = 1.0f;
    }

    // Utterance boundary: zero both translators' KV/ring state, back to <BOS>,
    // forget the utterance prosody summary. Encoder/vocoder/tracker continue.
    void reset_translator() noexcept {
        tenc.reset();
        tdec.reset();
        std::fill(prev_units.begin(), prev_units.end(), bos);
        ctx_sum_f0 = 0.0;
        ctx_sum_energy = 0.0;
        ctx_frames = 0;
        utt_eos = false;
    }

    bool lang_ok(std::int64_t id) const noexcept {
        return std::find(lang_ids.begin(), lang_ids.end(), id) != lang_ids.end();
    }

    // Stage 09 reset_translator_only(): a mode or language switch drops the
    // translation in flight (translators, policy, queued frames, partial
    // chunk) but keeps the stream clock and the shared encoder/vocoder/tracker.
    void reset_translation() noexcept {
        policy.reset();
        fifo.clear();
        reset_translator();
        std::fill(chunk_latents.begin(), chunk_latents.end(), 0.0f);
        chunk_fill = 0;
        chunk_speech = false;
        vocoded_last_hop = false;
        last_raw = 0.0f;
        last_decision = WaitKPolicy::Decision::kRead;
        counters.lag_chunks.store(0, std::memory_order_relaxed);
    }

    // Stream restart (the owning State resets encoder/vocoder/tracker).
    void reset_stream() noexcept {
        reset_translation();
        chunk_index = 0;
        segment = 0;
        input_samples = 0;
    }

    // prosody_ctx = [mean z_f0, mean z_energy] over the utterance's voiced frames.
    void write_prosody_ctx() noexcept {
        const bool use = opts.prosody_ctx && ctx_frames > 0;
        const double n = static_cast<double>(ctx_frames);
        tdec_prosody_ctx[0] = use ? static_cast<float>(ctx_sum_f0 / n) : 0.0f;
        tdec_prosody_ctx[1] = use ? static_cast<float>(ctx_sum_energy / n) : 0.0f;
    }

    ModelStep& tenc;
    ModelStep& tdec;
    const S2utOptions opts;
    WaitKPolicy policy;
    UnitProsodyFifo fifo;
    const std::size_t chunk;        // frames per translation chunk
    const std::size_t fps;          // frames per vocoder/encoder hop
    const std::size_t latent_dim;
    const std::int64_t bos;
    const std::uint32_t chunk_ms;

    std::span<const float> enc_latents;
    std::span<float> tenc_latents;
    std::span<std::int64_t> tenc_src_lang;
    std::span<const float> tenc_readiness;
    std::span<std::int64_t> tdec_prev_units;
    std::span<std::int64_t> tdec_tgt_lang;
    std::span<float> tdec_prosody_ctx;
    std::span<const std::int64_t> tdec_units;
    std::span<const float> tdec_prosody;
    std::span<const float> tdec_eos;

    std::vector<float> chunk_latents;
    std::vector<std::int64_t> prev_units;
    std::vector<std::int64_t> step_units;   // raw units of this step's writes (observer)
    std::vector<float> neutral_prosody;
    std::vector<UnitProsodyFifo::Frame> hop_frames;
    std::int64_t src_lang_id;              // live translator_enc src_lang (Stage 09)
    std::int64_t tgt_lang_id;              // live translator_dec tgt_lang
    std::vector<std::int64_t> lang_ids;    // the translator's language ids

    std::size_t chunk_fill = 0;
    bool chunk_speech = false;
    double ctx_sum_f0 = 0.0;
    double ctx_sum_energy = 0.0;
    std::uint64_t ctx_frames = 0;
    bool utt_eos = false;
    bool vocoded_last_hop = false;   // previous hop ended on translated audio
    float last_raw = 0.0f;           // its last pre-chain sample
    WaitKPolicy::Decision last_decision = WaitKPolicy::Decision::kRead;
    std::uint64_t chunk_index = 0;
    std::uint64_t segment = 0;
    std::uint64_t input_samples = 0;
    S2utObserver* observer = nullptr;
    ModeBCounters counters;
};

} // namespace

struct InferencePipeline::Legacy {
    OverlapAddProcessor ola{PipelineTiming::kChunkSamples, PipelineTiming::kOverlapSamples};
};

struct InferencePipeline::State {
    State(ModelStep& encoder, ModelStep& vocoder, std::span<const PipelineVoice> voices_in,
          PipelineOptions opts, const ModeBModels* mb = nullptr)
        : enc(encoder), voc(vocoder),
          g(mb ? checked_s2ut_geometry(encoder, mb->tenc, mb->tdec, vocoder)
               : checked_geometry(encoder, vocoder)),
          options(opts), clock(opts.clock_us ? opts.clock_us : &steady_now_us),
          params(opts.params),
          tracker(tracker_config(g, enc.contract())), ola(g.hop, g.overlap_samples),
          governor(GovernorConfig{
              g.sample_rate, g.hop,
              opts.latency_target_samples ? opts.latency_target_samples :
                  LatencyGovernor::default_target(g.queue_samples(), g.hop, g.sample_rate),
              3, opts.quiet_peak,
              mb ? (mb->opts.burst_cushion_samples ? mb->opts.burst_cushion_samples
                                                   : 3 * g.chunk_samples())
                 : 0}),
          histogram(g.hop_us()),
          stage_hist{TimingHistogram(g.hop_us()), TimingHistogram(g.hop_us()),
                     TimingHistogram(g.hop_us()), TimingHistogram(g.hop_us()),
                     TimingHistogram(g.hop_us()), TimingHistogram(g.hop_us())},
          frame_levels(g.cond_frames, 0.0f), frame_ctl(g.cond_frames),
          warm_input(g.hop, 0.0f), warm_output(g.hop, 0.0f) {
        publish_governor();
        if (&enc == &voc) throw std::invalid_argument("encoder and vocoder must be distinct models");
        if (mb) {
            const ModelStep* models[] = {&enc, &mb->tenc, &mb->tdec, &voc};
            for (std::size_t i = 0; i < 4; ++i)
                for (std::size_t j = i + 1; j < 4; ++j)
                    if (models[i] == models[j])
                        throw std::invalid_argument("Mode B models must be four distinct models");
        }
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
        // A Mode A pipeline cannot translate; a preset/--param asking for it
        // must not leave params() in a state set_params() would refuse.
        if (!mb) params.set(ParamId::Translate, 0.0f);
        if (static_cast<std::size_t>(params.voice_a) >= voice_stats.size() ||
            static_cast<std::size_t>(params.voice_b) >= voice_stats.size())
            throw std::invalid_argument("voice index out of range");
        enc_audio = enc.input_f32("audio");
        voc_audio = voc.output_f32("audio");
        require_size(enc_audio.size(), g.hop, "encoder audio");
        require_size(voc_audio.size(), g.hop + g.overlap_samples, "vocoder audio");
        if (mb || enc.contract().content_kind == ContentKind::kDiscrete) {
            // A Mode B bundle is discrete (validate_s2ut_bundle): vocoder units
            // come from the encoder on the Mode A route, from the translator
            // FIFO on the Mode B route.
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
        if (mb) {
            b = std::make_unique<ModeBState>(enc, mb->tenc, mb->tdec, voc, g, mb->opts);
            // The S2utOptions are the initial values of the live Stage 09
            // translation controls (language ids already validated above).
            params.set(ParamId::Translate, mb->opts.translate ? 1.0f : 0.0f);
            params.set(ParamId::WaitK, static_cast<float>(mb->opts.policy.k));
            params.set(ParamId::AdaptivePolicy, mb->opts.policy.adaptive_mode ? 1.0f : 0.0f);
            params.src_lang_id = static_cast<std::uint16_t>(mb->opts.src_lang_id);
            params.tgt_lang_id = static_cast<std::uint16_t>(mb->opts.tgt_lang_id);
            translating.store(mb->opts.translate, std::memory_order_relaxed);
        }
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
        const bool want_translate = p.enabled(ParamId::Translate);
        if (!b) {
            // No translator models (Mode A bundle): stay in Mode A. The control
            // thread rejects this first with kNoTranslatorModelsMessage.
            if (want_translate) return false;
        } else if (want_translate && (!b->lang_ok(p.src_lang_id) || !b->lang_ok(p.tgt_lang_id) ||
                                      p.src_lang_id == p.tgt_lang_id)) {
            // Language ids only matter while translating: a Mode A snapshot
            // carrying stale/default ids must not block unrelated slider moves.
            return false;
        }
        // No readiness head: strict Wait-k only (params() shows what runs).
        if (b && b->tenc_readiness.empty()) p.set(ParamId::AdaptivePolicy, 0.0f);
        const bool lang_changed =
            p.src_lang_id != params.src_lang_id || p.tgt_lang_id != params.tgt_lang_id;
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
        if (b) {
            // wait_k / adaptive_policy retune the policy in place (no reset).
            b->policy.set_k(static_cast<int>(p.get(ParamId::WaitK)));
            const bool adaptive = p.enabled(ParamId::AdaptivePolicy);
            if (b->policy.config().adaptive_mode != adaptive) b->policy.set_adaptive_mode(adaptive);
            if (want_translate) {   // validated above
                b->src_lang_id = p.src_lang_id;
                b->tgt_lang_id = p.tgt_lang_id;
            }
            const bool was_translating = translating.load(std::memory_order_relaxed);
            if (want_translate != was_translating || (want_translate && lang_changed)) {
                // Mode A <-> B or a new language pair: the translation in
                // flight is void; encoder, vocoder and tracker continue.
                translating.store(want_translate, std::memory_order_relaxed);
                b->reset_translation();
                governor.set_translation_state(false, false);
                switch_ramp = true;
            }
        }
        return true;
    }

    // Stage 09 OutputRamp: across a mode/language switch the first 20 ms
    // crossfade (raised cosine) from the last emitted sample into the new
    // route's output, so the boundary is continuous whatever either side plays.
    void switch_crossfade(std::span<float> out) const noexcept {
        const std::size_t n = std::min(out.size(), std::size_t{g.sample_rate} / 50);
        const double held = last_output;
        for (std::size_t i = 0; i < n; ++i) {
            const double c =
                0.5 * (1.0 + std::cos(kTwoPi * 0.5 * static_cast<double>(i) / static_cast<double>(n)));
            out[i] = static_cast<float>(held * c + static_cast<double>(out[i]) * (1.0 - c));
        }
    }

    void reset_stream() noexcept {
        enc.reset();
        voc.reset();
        if (b) b->reset_stream();
        switch_ramp = false;
        tracker.reset();
        ola.reset();
        governor.reset();
        publish_governor();
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

    // Advance every Stage 09 ramp (and the vibrato LFO) by one conditioning frame.
    FrameCtl advance_controls() noexcept {
        FrameCtl c;
        c.pitch = pitch_ramp.advance();
        c.intonation = inton_ramp.advance();
        const double alpha = blend_ramp.advance();
        const double vrate = vrate_ramp.advance();
        const double vdepth = vdepth_ramp.advance();
        c.whisper = whisper_ramp.advance();
        const double xf = xfade_ramp.advance();
        c.eff = lerp_stats(xfade_stats,
                           lerp_stats(voice_stats[params.voice_a], voice_stats[params.voice_b], alpha),
                           xf);
        // Scale the LFO by min(1, rate / 0.5 Hz): when vibrato_rate ramps to 0
        // the phase stops advancing and sin(phase) would otherwise freeze at a
        // static pitch offset; at rate 0 the contribution is exactly zero.
        c.vibrato = vdepth / 1200.0 * kLn2 * std::sin(vibrato_phase) * std::min(1.0, vrate / 0.5);
        vibrato_phase += kTwoPi * vrate * 0.020;
        if (vibrato_phase >= kTwoPi) vibrato_phase = std::fmod(vibrato_phase, kTwoPi);
        return c;
    }

    // Conditioning slot `i`; false on a non-finite/out-of-range value.
    bool write_cond(std::size_t i, const std::array<double, 4>& mapped) noexcept {
        for (std::size_t j = 0; j < cond.size(); ++j) {
            if (cond[j].empty()) continue;
            if (!std::isfinite(mapped[j]) || std::abs(mapped[j]) > std::numeric_limits<float>::max())
                return false;
            cond[j][i] = static_cast<float>(mapped[j]);
        }
        return true;
    }

    // The vocoder takes one speaker vector per step: compute it once per hop
    // from the ramp state after the hop's last frame (per frame for s1, which
    // has one conditioning frame per hop anyway).
    void write_speaker() noexcept {
        if (spk_input.empty()) return;
        const double alpha = blend_ramp.value(), xf = xfade_ramp.value();
        const float* ea = emb_of(params.voice_a);
        const float* eb = emb_of(params.voice_b);
        for (std::size_t j = 0; j < emb_dim(); ++j)
            eff_emb[j] = static_cast<float>((1.0 - xf) * xfade_emb[j] +
                                            xf * ((1.0 - alpha) * ea[j] + alpha * eb[j]));
        std::copy(eff_emb.begin(), eff_emb.end(), spk_input.begin());
    }

    void record_error(std::string_view error) noexcept {
        if (error.empty()) error = "model step failed without an error message";
        error_size = std::min(error.size(), error_text.size() - 1);
        std::copy_n(error.data(), error_size, error_text.data());
        error_text[error_size] = '\0';
    }

    bool timed_step(ModelStep& m, PipelineStage stage, std::uint64_t* us_out = nullptr) noexcept {
        const std::uint64_t t0 = clock();
        const bool ok = m.step();
        const std::uint64_t us = elapsed_us(t0, clock());
        stage_hist[static_cast<std::size_t>(stage)].record(us);
        if (us_out) *us_out += us;
        return ok;
    }

    // Closes a non-model stage opened with `t0 = clock()`.
    void record_stage(PipelineStage stage, std::uint64_t t0) noexcept {
        stage_hist[static_cast<std::size_t>(stage)].record(elapsed_us(t0, clock()));
    }

    // Worker only: mirror the governor's effective target / hard cap for the
    // UI (relaxed; a Mode B cushion changes both with the translation state).
    void publish_governor() noexcept {
        gov_target.store(governor.effective_target_samples(), std::memory_order_relaxed);
        gov_cap.store(governor.hard_cap_samples(), std::memory_order_relaxed);
    }

    PipelineStepResult fail(std::string_view error, std::span<float> output, bool fatal) noexcept;
    PipelineStepResult finish(std::span<float> output) noexcept;
    PipelineStepResult process(std::span<const float> input, std::span<float> output) noexcept;
    PipelineStepResult process_vc(std::span<float> output) noexcept;
    PipelineStepResult process_s2ut(std::span<float> output) noexcept;
    std::string_view chunk_step() noexcept;
    bool warm_up_translation(std::size_t steps) noexcept;

    ModelStep& enc;
    ModelStep& voc;
    const PipelineGeometry g;
    PipelineOptions options;
    std::uint64_t (*clock)() noexcept;   // options.clock_us or steady_now_us; worker thread only
    VoiceParams params;
    PitchTracker tracker;
    OverlapAddProcessor ola;
    LatencyGovernor governor;
    std::atomic<std::size_t> gov_target{0}, gov_cap{0};   // publish_governor()
    static_assert(std::atomic<std::size_t>::is_always_lock_free);
    TimingHistogram histogram;
    std::array<TimingHistogram, kPipelineStageCount> stage_hist;
    std::vector<TargetVoiceStats> voice_stats;
    std::vector<float> embedding_table;
    std::vector<float> eff_emb, xfade_emb;
    std::vector<float> frame_levels;
    std::vector<FrameCtl> frame_ctl;
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
    std::unique_ptr<ModeBState> b;   // translator models present (Mode B constructor)
    // Current route: Mode B (true) or Mode A. Written by the worker (set_params),
    // read by mode() from any thread, so a relaxed atomic.
    std::atomic<bool> translating{false};
    bool switch_ramp = false;        // crossfade the next hop (Stage 09 switch)
    std::array<char, 512> error_text{};
    std::size_t error_size = 0;
    std::atomic<std::uint64_t> errors{0};
    std::size_t consecutive_errors = 0;
    float last_output = 0.0f;
    bool recovering = false;
};

PipelineStepResult InferencePipeline::State::fail(std::string_view error,
                                                 std::span<float> output, bool fatal) noexcept {
    record_error(error);
    const std::size_t fade = std::min(output.size(), std::size_t{g.sample_rate} * 2 / 1000);
    const float previous = last_output;
    std::fill(output.begin(), output.end(), 0.0f);
    for (std::size_t i = 0; i < fade; ++i)
        output[i] = previous * static_cast<float>(fade - i - 1) / static_cast<float>(fade);
    // A failed Run may have partially advanced state. Restart every model so
    // the next hop cannot combine stages from different timelines.
    reset_stream();
    recovering = true;
    errors.fetch_add(1, std::memory_order_relaxed);
    ++consecutive_errors;
    return {false, fatal || consecutive_errors >= options.max_consecutive_errors,
            previous == 0.0f, {error_text.data(), error_size}};
}

PipelineStepResult InferencePipeline::State::finish(std::span<float> output) noexcept {
    if (switch_ramp) {
        switch_crossfade(output);
        switch_ramp = false;
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

PipelineStepResult InferencePipeline::State::process(std::span<const float> input,
                                                    std::span<float> output) noexcept {
    if (input.size() != g.hop || output.size() != g.hop)
        return fail("pipeline requires exactly one input/output hop", output, true);
    std::copy(input.begin(), input.end(), enc_audio.begin());
    // 11b: the stream clock (trace t_src_ms) advances on both routes.
    if (b) b->input_samples += g.hop;
    return b && translating.load(std::memory_order_relaxed) ? process_s2ut(output)
                                                            : process_vc(output);
}

PipelineStepResult InferencePipeline::State::process_vc(std::span<float> output) noexcept {
    const std::uint64_t features_t0 = clock();
    for (std::size_t i = 0; i < g.cond_frames; ++i) {
        // The Stage 03 tracker owns HPF, EMA priors and lookahead delay. Keeping
        // encoder audio raw also makes identity bundles genuinely transparent.
        const PitchFrame f = tracker.push(enc_audio.data() + i * g.cond_frame_samples);
        // Input level for the gate, aligned with the content being vocoded
        // (the tracker applies the encoder lookahead).
        frame_levels[i] = static_cast<float>(f.log_energy * 10.0 / kLn10);
        const FrameCtl c = advance_controls();
        if (!write_cond(i, map_frame(c, f.z_logf0, f.voicing, f.aperiodicity, f.z_energy)))
            return fail("non-finite model conditioning", output, false);
    }
    record_stage(PipelineStage::kFeatures, features_t0);
    if (!timed_step(enc, PipelineStage::kEncoder)) return fail(enc.last_error(), output, false);
    if (!enc_content.empty()) std::copy(enc_content.begin(), enc_content.end(), voc_content.begin());
    if (!enc_units.empty()) std::copy(enc_units.begin(), enc_units.end(), voc_units.begin());
    write_speaker();
    if (!timed_step(voc, PipelineStage::kVocoder)) return fail(voc.last_error(), output, false);
    if (g.overlap_samples > 0) ola.process(voc_audio.data(), output.data());
    else std::copy(voc_audio.begin(), voc_audio.end(), output.begin());
    if (options.output_chain) {
        const std::uint64_t chain_t0 = clock();
        chain.set_targets(params.get(ParamId::Gain), params.get(ParamId::GateThreshold),
                          params.enabled(ParamId::Limiter), params.get(ParamId::Brightness));
        chain.process(output, frame_levels, g.cond_frame_samples);
        record_stage(PipelineStage::kOutputChain, chain_t0);
    }
    // On a translator pipeline the Mode A route still owns the hint: hold and
    // cushion only for the B -> A switch hop (its crossfade starts from the
    // last translated sample and must not be hard-capped into a click).
    if (b) governor.set_translation_state(switch_ramp, switch_ramp);
    return finish(output);
}

PipelineStepResult InferencePipeline::State::process_s2ut(std::span<float> output) noexcept {
    ModeBState& s = *b;
    const std::size_t fs = g.cond_frame_samples;
    const std::uint64_t features_t0 = clock();
    for (std::size_t i = 0; i < g.cond_frames; ++i) {
        const float* x = enc_audio.data() + i * fs;
        // Raw-level VAD, the twin of fvml.asr.active_frames_mask (RMS dBFS).
        double ms = 0.0;
        for (std::size_t j = 0; j < fs; ++j) ms += static_cast<double>(x[j]) * x[j];
        ms /= static_cast<double>(fs);
        if (10.0 * std::log10(ms + 1e-16) > s.opts.speech_threshold_dbfs) s.chunk_speech = true;
        // The tracker summarises the speaker's voiced frames for prosody_ctx.
        const PitchFrame f = tracker.push(x);
        if (f.voicing >= s.opts.voiced_threshold && std::isfinite(f.z_logf0) &&
            std::isfinite(f.z_energy)) {
            s.ctx_sum_f0 += f.z_logf0;
            s.ctx_sum_energy += f.z_energy;
            ++s.ctx_frames;
        }
        // Ramps advance with wall-clock frames, whether or not we vocode.
        frame_ctl[i] = advance_controls();
    }
    record_stage(PipelineStage::kFeatures, features_t0);
    if (!timed_step(enc, PipelineStage::kEncoder)) return fail(enc.last_error(), output, false);
    std::copy(s.enc_latents.begin(), s.enc_latents.end(),
              s.chunk_latents.begin() + static_cast<std::ptrdiff_t>(s.chunk_fill * s.latent_dim));
    s.chunk_fill += s.fps;
    if (s.chunk_fill == s.chunk) {
        s.chunk_fill = 0;
        const std::string_view err = chunk_step();
        if (!err.empty()) return fail(err, output, false);
    }

    const bool was_vocoding = s.vocoded_last_hop;   // this hop ramps down from it
    const bool vocode = s.fifo.pop_hop(s.fps, s.hop_frames.data());
    if (vocode) {
        std::size_t real = s.fps;   // frames before the first pad frame
        for (std::size_t i = 0; i < s.fps; ++i) {
            const UnitProsodyFifo::Frame& fr = s.hop_frames[i];
            if (fr.pad && real == s.fps) real = i;
            voc_units[i] = fr.unit;
            frame_levels[i] = fr.pad ? kGateClosedDbfs : kGateOpenDbfs;
            if (!write_cond(i, map_frame(frame_ctl[i], fr.logf0_z, fr.voicing, fr.aperiodicity,
                                         fr.energy_z)))
                return fail("non-finite model conditioning", output, false);
        }
        write_speaker();
        if (!timed_step(voc, PipelineStage::kVocoder)) return fail(voc.last_error(), output, false);
        ModeBCounters::bump(s.counters.voc_steps);
        std::copy(voc_audio.begin(), voc_audio.end(), output.begin());
        // BUG-028: pad frames only keep the hop aligned; they render as silence.
        const std::size_t cut = real * fs;
        if (cut < output.size()) {
            const float from = cut > 0 ? output[cut - 1] : 0.0f;
            std::fill(output.begin() + static_cast<std::ptrdiff_t>(cut), output.end(), 0.0f);
            if (options.output_chain) ramp_down(output.subspan(cut), from, g.sample_rate);
        }
        // Silence -> speech: start from zero instead of jumping.
        if (options.output_chain && !s.vocoded_last_hop && cut > 0)
            ramp_up(output.first(cut), g.sample_rate);
        s.vocoded_last_hop = cut == output.size();
        s.last_raw = output.back();
    } else {
        // kRead / idle: no decoder, no vocoder — silence, ramped from the last
        // translated sample so a speech -> silence edge never clicks.
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(frame_levels.begin(), frame_levels.end(), kGateClosedDbfs);
        if (options.output_chain && s.vocoded_last_hop) ramp_down(output, s.last_raw, g.sample_rate);
        s.vocoded_last_hop = false;
        s.last_raw = 0.0f;
    }
    if (options.output_chain) {
        // The gate follows translated frames (frame_levels above), never the
        // live microphone: kFlush tails are spoken while the user is silent.
        const std::uint64_t chain_t0 = clock();
        chain.set_targets(params.get(ParamId::Gain), params.get(ParamId::GateThreshold),
                          params.enabled(ParamId::Limiter), params.get(ParamId::Brightness));
        chain.process(output, frame_levels, g.cond_frame_samples);
        record_stage(PipelineStage::kOutputChain, chain_t0);
    }
    // Cushion while a translation is in flight — including every hop whose
    // samples touch translated audio: the hop carrying the final frames (the
    // policy is already idle there), the 2 ms ramp-down hop after it and a
    // Stage 09 switch crossfade hop — so no tail or edge is hard-capped into
    // a click; excess is trimmed only from the plain silence after them
    // (research §9.2).
    const bool in_flight = vocode || was_vocoding || switch_ramp || !s.fifo.empty();
    governor.set_translation_state(
        in_flight || s.policy.t_src() > 0,
        in_flight || s.last_decision != WaitKPolicy::Decision::kRead);
    return finish(output);
}

// One translation chunk: the Stage 04 caller protocol, step for step the twin
// of fvml/s2ut_stream.py StreamingTranslator.push_chunk(). Returns the failing
// model's error, or an empty view.
std::string_view InferencePipeline::State::chunk_step() noexcept {
    ModeBState& s = *b;
    WaitKPolicy& p = s.policy;
    // Captured before on_chunk_energy(): a pause can complete the utterance
    // right there (caught up / <EOS> already seen) without any flush step.
    const std::uint64_t flushes_before = p.flushes_completed();
    const bool speech = s.chunk_speech;
    s.chunk_speech = false;
    // utt_eos mirrors the policy's "<EOS> seen during a pause": speech clears it.
    if (speech) s.utt_eos = false;
    p.on_chunk_energy(speech, s.chunk_ms);

    S2utChunkEvent ev;
    ev.chunk = s.chunk_index++;
    ev.segment = s.segment;
    ev.input_samples = s.input_samples;
    ev.speech = speech;
    if (p.source_wanted()) {
        std::copy(s.chunk_latents.begin(), s.chunk_latents.end(), s.tenc_latents.begin());
        s.tenc_src_lang[0] = s.src_lang_id;
        if (!timed_step(s.tenc, PipelineStage::kTranslatorEnc, &ev.trans_enc_us)) {
            const std::string_view e = s.tenc.last_error();
            return e.empty() ? std::string_view("translator_enc step failed") : e;
        }
        ev.encoded = true;
        p.notify_source_unit();
        if (!s.tenc_readiness.empty()) {
            ev.readiness_logit = s.tenc_readiness[0];
            if (p.config().adaptive_mode)
                p.on_source_readiness(readiness_from_logit(ev.readiness_logit));
        }
    }
    ev.decision = p.decide();
    switch (ev.decision) {
    case WaitKPolicy::Decision::kRead:  ModeBCounters::bump(s.counters.reads); break;
    case WaitKPolicy::Decision::kWrite: ModeBCounters::bump(s.counters.writes); break;
    case WaitKPolicy::Decision::kFlush: ModeBCounters::bump(s.counters.flushes); break;
    }

    bool completed_in_loop = false;
    bool last_eos = false;
    const int max_writes = p.max_writes_this_step();   // 0 on kRead, 1 on kFlush
    for (int w = 0; w < max_writes; ++w) {
        // An adaptive double write must still be wanted after the first one:
        // an <EOS> during a pause marks the utterance caught up (decide() ->
        // kRead), and decoding past it would only babble.
        if (w > 0 && p.decide() == WaitKPolicy::Decision::kRead) break;
        // Backpressure instead of loss: the write stays owed (lag unchanged).
        if (s.fifo.free_frames() < s.chunk) {
            ModeBCounters::bump(s.counters.fifo_stalls);
            break;
        }
        std::copy(s.prev_units.begin(), s.prev_units.end(), s.tdec_prev_units.begin());
        s.tdec_tgt_lang[0] = s.tgt_lang_id;
        s.write_prosody_ctx();
        if (!timed_step(s.tdec, PipelineStage::kTranslatorDec, &ev.trans_dec_us)) {
            const std::string_view e = s.tdec.last_error();
            return e.empty() ? std::string_view("translator_dec step failed") : e;
        }
        ModeBCounters::bump(s.counters.dec_steps);
        std::copy(s.tdec_units.begin(), s.tdec_units.end(),
                  s.step_units.begin() + static_cast<std::ptrdiff_t>(
                                             static_cast<std::size_t>(ev.writes) * s.chunk));
        const auto pushed = s.fifo.push_chunk(
            s.tdec_units, s.tdec_prosody.empty() ? std::span<const float>(s.neutral_prosody)
                                                 : s.tdec_prosody);
        const std::size_t pads = s.fifo.pad_to_multiple(s.fps);
        ev.frames_pushed += pushed.pushed;
        ev.pad_frames += pads;
        if (pads) ModeBCounters::bump(s.counters.pad_frames, pads);
        const bool eos = pushed.eos || s.tdec_eos[0] > s.opts.eos_threshold;
        // The raw emission (specials included) is the next step's input.
        std::copy(s.tdec_units.begin(), s.tdec_units.end(), s.prev_units.begin());
        ev.eos = ev.eos || eos;
        // Exactly when the policy will record "caught up" (eos in a pause).
        if (eos && !p.flushing() && p.utterance_active() && p.silence_ms() > 0) s.utt_eos = true;
        ++ev.writes;
        p.notify_target_unit(eos);
        if (p.flushes_completed() != flushes_before) {
            completed_in_loop = true;
            last_eos = eos;
            break;
        }
    }
    ev.units = std::span<const std::int64_t>(s.step_units)
                   .first(static_cast<std::size_t>(ev.writes) * s.chunk);
    ev.utterance_end = p.flushes_completed() != flushes_before;
    if (ev.utterance_end) {
        ModeBCounters::bump(s.counters.utterances);
        if (completed_in_loop ? last_eos : s.utt_eos) {
            ModeBCounters::bump(s.counters.eos_ends);
        } else if (completed_in_loop) {
            ev.cap_end = true;   // flush bound or max_flush_steps (BUG-030)
            ModeBCounters::bump(s.counters.cap_ends);
        }
    }
    ev.t_src = p.t_src();
    ev.t_tgt = p.t_tgt();
    s.counters.lag_chunks.store(ev.t_src - ev.t_tgt, std::memory_order_relaxed);
    s.last_decision = ev.decision;
    if (s.observer) s.observer->on_chunk(ev);
    if (ev.utterance_end) {
        s.reset_translator();
        ++s.segment;
    }
    return {};
}

// Silence never reaches the translator or the vocoder, so step them directly
// to initialise their kernels/arenas before the stream starts.
bool InferencePipeline::State::warm_up_translation(std::size_t steps) noexcept {
    for (std::size_t i = 0; i < steps; ++i) {
        for (ModelStep* m : {&b->tenc, &b->tdec, &voc}) {
            if (!m->step()) {
                record_error(m->last_error());
                return false;
            }
        }
    }
    return true;
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

InferencePipeline::InferencePipeline(ModelStep& encoder, ModelStep& translator_enc,
                                     ModelStep& translator_dec, ModelStep& vocoder,
                                     std::span<const PipelineVoice> voices,
                                     const S2utOptions& s2ut, PipelineOptions options) {
    const ModeBModels mb{translator_enc, translator_dec, s2ut};
    state_ = std::make_unique<State>(encoder, vocoder, voices, options, &mb);
}

InferencePipeline::~InferencePipeline() = default;

PipelineStepResult InferencePipeline::process(std::span<const float> input,
                                              std::span<float> output) noexcept {
    if (!state_) {
        std::fill(output.begin(), output.end(), 0.0f);
        return {false, true, true, "pipeline has no models"};
    }
    const std::uint64_t start = state_->clock();
    const auto result = state_->process(input, output);
    state_->histogram.record(elapsed_us(start, state_->clock()));
    state_->publish_governor();
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
    if (ok && state_->b) ok = state_->warm_up_translation(steps);
    reset();
    state_->histogram.reset();
    for (auto& h : state_->stage_hist) h.reset();
    state_->errors.store(0, std::memory_order_relaxed);
    if (state_->b) state_->b->counters.clear();
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

TimingSnapshot InferencePipeline::stage_timing(PipelineStage stage) const noexcept {
    const auto i = static_cast<std::size_t>(stage);
    return state_ && i < kPipelineStageCount ? state_->stage_hist[i].snapshot() : TimingSnapshot{};
}

std::array<TimingSnapshot, kPipelineStageCount> InferencePipeline::stage_timings() const noexcept {
    std::array<TimingSnapshot, kPipelineStageCount> all{};
    if (!state_) return all;
    for (std::size_t i = 0; i < kPipelineStageCount; ++i) all[i] = state_->stage_hist[i].snapshot();
    return all;
}

std::size_t InferencePipeline::governor_target_samples() const noexcept {
    return state_ ? state_->gov_target.load(std::memory_order_relaxed) : 0;
}

std::size_t InferencePipeline::governor_hard_cap_samples() const noexcept {
    return state_ ? state_->gov_cap.load(std::memory_order_relaxed) : 0;
}

std::uint64_t InferencePipeline::errors() const noexcept {
    return state_ ? state_->errors.load(std::memory_order_relaxed) : 0;
}

PipelineMode InferencePipeline::mode() const noexcept {
    return state_ && state_->b && state_->translating.load(std::memory_order_relaxed)
               ? PipelineMode::kS2UT
               : PipelineMode::kVC;
}

bool InferencePipeline::has_translator() const noexcept {
    return state_ && state_->b;
}

S2utTelemetry InferencePipeline::s2ut_telemetry() const noexcept {
    S2utTelemetry t;
    if (!state_ || !state_->b) return t;
    const ModeBCounters& c = state_->b->counters;
    constexpr auto r = std::memory_order_relaxed;
    t.reads = c.reads.load(r);
    t.writes = c.writes.load(r);
    t.flushes = c.flushes.load(r);
    t.utterances = c.utterances.load(r);
    t.eos_ends = c.eos_ends.load(r);
    t.cap_ends = c.cap_ends.load(r);
    t.dec_steps = c.dec_steps.load(r);
    t.voc_steps = c.voc_steps.load(r);
    t.fifo_stalls = c.fifo_stalls.load(r);
    t.pad_frames = c.pad_frames.load(r);
    t.lag_chunks = c.lag_chunks.load(r);
    t.chunk_ms = state_->b->chunk_ms;
    return t;
}

const WaitKPolicy* InferencePipeline::waitk() const noexcept {
    return state_ && state_->b ? &state_->b->policy : nullptr;
}

bool InferencePipeline::s2ut_idle() const noexcept {
    if (!state_ || !state_->b || !state_->translating.load(std::memory_order_relaxed)) return true;
    const ModeBState& s = *state_->b;
    return !s.policy.utterance_active() && s.fifo.empty() && !(s.chunk_fill > 0 && s.chunk_speech);
}

void InferencePipeline::set_s2ut_observer(S2utObserver* observer) noexcept {
    if (state_ && state_->b) state_->b->observer = observer;
}

std::string_view InferencePipeline::last_error() const noexcept {
    return state_ ? std::string_view(state_->error_text.data(), state_->error_size) :
                    std::string_view("pipeline has no models");
}

bool InferencePipeline::set_params(const VoiceParams& next) noexcept {
    if (!state_ || !state_->set_params(next)) return false;
    state_->publish_governor();   // a route switch clears the Mode B cushion
    return true;
}

void InferencePipeline::set_muted(bool muted) noexcept {
    if (state_) state_->chain.set_muted(muted);
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
