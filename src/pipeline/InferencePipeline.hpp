// Worker-thread orchestrator for Mode A (voice conversion, Milestone 2 /
// Stage 08) and Mode B (simultaneous S2UT, Milestone 3 / Stage 08). Models
// own stable I/O buffers and outlive this pipeline, with exclusive access; no
// ONNX Runtime dependency, so fake-model CI covers both modes.
//
// Mode A hop: tracker -> encoder -> F0/energy map -> vocoder -> [OLA] -> chain.
// Mode B hop (Documents/Milestone 3/08_realtime_mode_b_pipeline.md):
//   1. per 20 ms frame: raw-level VAD (any frame above speech_threshold_dbfs
//      makes the chunk "speech", the twin of gate_s2ut.py --vad-dbfs), the
//      pitch tracker (voiced frames feed the utterance prosody_ctx =
//      [mean z_f0, mean z_energy]) and the Stage 09 control ramps;
//   2. the encoder runs every hop (its causal state never skips time) and its
//      `latents` fill a chunk_frames accumulator;
//   3. on a full chunk, the Stage 04 protocol: on_chunk_energy ->
//      (source_wanted ? translator_enc + notify_source_unit [+ readiness]) ->
//      decide -> max_writes_this_step x (translator_dec -> UnitProsodyFifo ->
//      notify_target_unit(eos)); a completed flush zeroes both translators,
//      returns prev_units to <BOS> and clears prosody_ctx. translator_dec
//      runs at most once per chunk step during kFlush (the policy caps it),
//      so the tail drains at real-time pace;
//   4. the vocoder runs only when a whole hop of translated frames is queued
//      (never on kRead / idle silence); otherwise the hop is silence. The
//      output gate follows translated frames (open on every queued frame),
//      not the live microphone, so kFlush tails are never muted.
#pragma once

#include "dsp/PitchTracker.hpp"
#include "pipeline/TimingHistogram.hpp"
#include "pipeline/WaitKPolicy.hpp"
#include "voice/VoiceParams.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace fluxvoice {

class ModelStep;
class LatencyGovernor;
struct PipelineGeometry;

enum class PipelineMode { kVC, kS2UT };

inline const char* to_string(PipelineMode m) noexcept {
    return m == PipelineMode::kVC ? "vc" : "s2ut";
}

/// Stages timed per hop (stage_timing() / stage_timings()). The first four are
/// model Run() calls; kFeatures and kOutputChain (M4 Stage 06) bracket the
/// non-model spans of both routes. New stages are appended: indices are stable.
enum class PipelineStage : std::size_t {
    kEncoder = 0, kTranslatorEnc, kTranslatorDec, kVocoder,
    kFeatures,     // pitch tracker / YIN + whitening + conditioning loop (Mode B: VAD + tracker + control ramps)
    kOutputChain,  // VoiceOutputChain::process (absent with PipelineOptions::output_chain = false)
};
inline constexpr std::size_t kPipelineStageCount = 6;

inline const char* to_string(PipelineStage s) noexcept {
    switch (s) {
    case PipelineStage::kEncoder:       return "enc";
    case PipelineStage::kTranslatorEnc: return "trans_enc";
    case PipelineStage::kTranslatorDec: return "trans_dec";
    case PipelineStage::kVocoder:       return "voc";
    case PipelineStage::kFeatures:      return "feat";
    case PipelineStage::kOutputChain:   return "chain";
    }
    return "unknown";
}

inline const char* to_string(WaitKPolicy::Decision d) noexcept {
    switch (d) {
    case WaitKPolicy::Decision::kRead:  return "READ";
    case WaitKPolicy::Decision::kWrite: return "WRITE";
    case WaitKPolicy::Decision::kFlush: return "FLUSH";
    }
    return "unknown";
}

/// Mode B (S2UT) configuration. Language ids are translator token ids
/// (langs.json / contract `langs`), resolved by the caller (LangCatalog).
/// policy.k, policy.adaptive_mode, the language ids and `translate` are the
/// INITIAL values of the live Stage 09 controls: the constructor copies them
/// into params() (translate, wait_k, adaptive_policy, src/tgt_lang_id), and
/// set_params() changes them afterwards.
struct S2utOptions {
    // policy.k is the wait-k (1..10); chunk_frames and frame_ms are always
    // taken from the translator contracts.
    WaitKPolicy::Config policy{};
    std::int64_t src_lang_id = -1;
    std::int64_t tgt_lang_id = -1;
    // Initial route: true = Mode B (translating), false = start as a Mode A
    // voice changer on the same bundle and toggle live via set_params().
    bool translate = true;
    // Chunk VAD: speech when any 20 ms input frame's RMS level exceeds this
    // (twin of gate_s2ut.py --vad-dbfs / fvml.asr.active_frames_mask).
    float speech_threshold_dbfs = -55.0f;
    // Tracker voicing at or above this marks a frame as voiced for prosody_ctx.
    float voiced_threshold = 0.5f;
    // false feeds prosody_ctx = [0, 0] (what the Stage 05 Python driver
    // fvml/s2ut_stream.py OrtRuntime does) — for parity runs only.
    bool prosody_ctx = true;
    // <EOS> = the <EOS> unit was emitted or eos_prob > eos_threshold (twin of
    // fvml/s2ut_stream.py; Stage 08's text says >= 0.5, the twin decides).
    float eos_threshold = 0.5f;
    // Translated-frame FIFO, rounded up to a multiple of chunk_frames and to
    // at least two chunks (an adaptive step may write twice).
    std::size_t fifo_capacity_frames = 64;
    // Governor cushion while translating; 0 = 3 translation chunks (240 ms).
    std::size_t burst_cushion_samples = 0;
};

/// One Mode B chunk step, reported to an S2utObserver (offline tools).
struct S2utChunkEvent {
    std::uint64_t chunk = 0;            // chunk steps since stream start/reset
    std::uint64_t segment = 0;          // utterances completed before this step
    std::uint64_t input_samples = 0;    // input consumed since stream start, incl. this chunk
    WaitKPolicy::Decision decision = WaitKPolicy::Decision::kRead;
    bool speech = false;                // chunk VAD verdict
    bool encoded = false;               // translator_enc consumed the chunk
    int writes = 0;                     // translator_dec steps (0..2)
    std::size_t frames_pushed = 0;      // valid target frames queued (SimulEval t_tgt_frames_emitted)
    std::size_t pad_frames = 0;         // BUG-028 alignment frames appended
    bool eos = false;                   // a write reported <EOS>
    bool utterance_end = false;         // the flush completed: translator state is zeroed after this step
    bool cap_end = false;               // ...on the flush bound / max_flush_steps, without <EOS> (BUG-030)
    std::int64_t t_src = 0;             // policy counters after the step (0 after utterance_end)
    std::int64_t t_tgt = 0;
    float readiness_logit = 0.0f;       // translator_enc output (0 when absent or not run)
    std::uint64_t trans_enc_us = 0;
    std::uint64_t trans_dec_us = 0;     // summed over writes
    std::span<const std::int64_t> units; // raw units of every write (writes x chunk_frames); valid in the callback only
};

class S2utObserver {
public:
    virtual ~S2utObserver() = default;
    /// Worker thread, inside process(), after each chunk step. Must not call
    /// back into the pipeline. Live engines pass no observer.
    virtual void on_chunk(const S2utChunkEvent& event) noexcept = 0;
};

/// Mode B counters: relaxed atomics, readable from any thread. Cumulative
/// across reset(); warm_up() zeroes them with the other telemetry.
struct S2utTelemetry {
    std::uint64_t reads = 0, writes = 0, flushes = 0;   // chunk steps per decision
    std::uint64_t utterances = 0;   // completed utterances (policy flush completions)
    std::uint64_t eos_ends = 0;     // ...whose ending followed an <EOS>
    std::uint64_t cap_ends = 0;     // ...ended by the flush bound / max_flush_steps (BUG-030)
    std::uint64_t dec_steps = 0;    // translator_dec Run() calls
    std::uint64_t voc_steps = 0;    // vocoder Run() calls
    std::uint64_t fifo_stalls = 0;  // writes deferred: no FIFO room
    std::uint64_t pad_frames = 0;   // BUG-028 alignment frames rendered as silence
    std::int64_t lag_chunks = 0;    // t_src - t_tgt after the newest chunk step
    std::uint32_t chunk_ms = 0;
    double lag_ms() const noexcept { return static_cast<double>(lag_chunks) * chunk_ms; }
};

/// One bundle voice (stats + FiLM embedding) the pipeline can blend between.
/// `embedding` must have exactly `spk_emb_dim` finite values; contents are
/// copied at construction, so the span itself need not outlive the call.
struct PipelineVoice {
    TargetVoiceStats stats{};
    std::span<const float> embedding;
};

struct PipelineOptions {
    std::size_t max_consecutive_errors = 8;
    // Stage 09 controls. `params` is the initial snapshot (values are clamped
    // and snapped at construction — no ramp at stream start); later changes go
    // through set_params() and ramp per the descriptor's Smoothing kind.
    // `output_chain = false` leaves raw vocoder PCM (no gain/gate/brightness/
    // limiter, and in Mode B no 2 ms speech<->silence ramps: silent hops and
    // BUG-028 pad frames are exact zeros) — only for bit-exact plumbing/parity
    // checks, never for real-time use.
    VoiceParams params{};
    bool output_chain = true;
    float quiet_peak = 0.001f;
    std::size_t latency_target_samples = 0; // 0 = master-budget default
    // Microsecond clock for every timing histogram (whole hop and stages), read
    // on the worker thread only. nullptr = std::chrono::steady_clock. Tests
    // inject a fake clock so percentiles and misses match by construction.
    std::uint64_t (*clock_us)() noexcept = nullptr;
};

struct PipelineStepResult {
    bool ok = false;
    bool stop = false;
    bool quiet = true;
    std::string_view error;
};

class InferencePipeline {
public:
    // Legacy M1 diagnostic OLA/reset shell; does not provide model inference.
    InferencePipeline();
    InferencePipeline(ModelStep& encoder, ModelStep& vocoder,
                      std::span<const PipelineVoice> voices, PipelineOptions options = {});
    // Single-voice convenience form; delegates to the multi-voice ctor.
    InferencePipeline(ModelStep& encoder, ModelStep& vocoder,
                      const TargetVoiceStats& voice, std::span<const float> speaker_embedding,
                      PipelineOptions options = {});
    // Mode B (S2UT). The four models must satisfy validate_s2ut_bundle(), the
    // vocoder must be stateful (overlap_samples = 0) with conditioning frames
    // equal to content frames, and translator_dec's shared inputs (src_pos,
    // mem_*) must already read translator_enc's state
    // (OrtModel::bind_shared_inputs). Throws std::invalid_argument otherwise.
    InferencePipeline(ModelStep& encoder, ModelStep& translator_enc, ModelStep& translator_dec,
                      ModelStep& vocoder, std::span<const PipelineVoice> voices,
                      const S2utOptions& s2ut, PipelineOptions options = {});
    ~InferencePipeline();
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;

    // Exactly one contract hop in/out. Bad geometry is a fatal caller error;
    // model errors produce faded silence, reset stream state, and request stop
    // only after max_consecutive_errors. Error text is owned until next call.
    PipelineStepResult process(std::span<const float> input, std::span<float> output) noexcept;

    // Run zeros through the complete chain BEFORE audio starts, then reset all
    // stream state and telemetry. A failed warm-up must prevent stream startup.
    // Mode B also steps translator_enc, translator_dec and the vocoder
    // directly `steps` times (silence alone never reaches them).
    bool warm_up(std::size_t steps);
    const PipelineGeometry& geometry() const;
    LatencyGovernor& governor(); // worker only; use after process(), before enqueue
    // Any-thread view of the governor (relaxed atomics, refreshed once per
    // hop, on reset() and on set_params(); may lag the worker by a hop):
    // the target skip_output() works against NOW (the startup target plus the
    // Mode B burst cushion while a translation is in flight) and the hard cap
    // above it. 0 for a legacy shell.
    std::size_t governor_target_samples() const noexcept;
    std::size_t governor_hard_cap_samples() const noexcept;
    TimingSnapshot timing() const noexcept;
    // Per-stage Run() durations (any thread, like timing()).
    TimingSnapshot stage_timing(PipelineStage stage) const noexcept;
    // Every stage at once, indexed by PipelineStage (any thread, no allocation).
    // The snapshots are taken one after the other, so they may span adjacent hops.
    std::array<TimingSnapshot, kPipelineStageCount> stage_timings() const noexcept;
    std::uint64_t errors() const noexcept;
    // Current route: kS2UT while translating, kVC otherwise (also for a
    // translator bundle switched to Mode A). Any thread (relaxed atomic; a
    // UI may lag the worker by a hop).
    PipelineMode mode() const noexcept;
    // Built with translator models (the Mode B constructor), so translate = 1
    // is possible. Immutable after construction.
    bool has_translator() const noexcept;

    // ---- Mode B ---------------------------------------------------------
    // Lock-free counters (zeros in Mode A).
    S2utTelemetry s2ut_telemetry() const noexcept;
    // Worker/exclusive owner only; nullptr in Mode A.
    const WaitKPolicy* waitk() const noexcept;
    // Worker/exclusive owner only: nothing in flight (policy idle, no queued
    // translated frames, no partial chunk with speech). Mode A: always true.
    bool s2ut_idle() const noexcept;
    // Exclusive owner, between hops; nullptr detaches.
    void set_s2ut_observer(S2utObserver* observer) noexcept;
    // Worker/quiesced owner only; unlike timing()/errors(), not a UI snapshot.
    std::string_view last_error() const noexcept;

    // Stage 09 controls — worker/exclusive owner only, between hops. Clamps a
    // copy and rejects (false, previous params kept) an out-of-range voice
    // index. Takes effect from the next process(): conditioning ramps from
    // the current smoothed state, a voice change crossfades the embedding and
    // stats, and the output chain retargets its own smoothing. No allocations.
    // Milestone 3 / Stage 09 translation controls:
    //   * translate 0 <-> 1, or a src/tgt_lang_id change while translating:
    //     resets the translator side only (translator_enc/dec, WaitKPolicy,
    //     UnitProsodyFifo, prosody_ctx; encoder, vocoder and tracker keep
    //     their state) and crossfades the output over 20 ms (cosine) from
    //     the last emitted sample;
    //   * wait_k: WaitKPolicy::set_k() on the next chunk, no state reset;
    //   * adaptive_policy: WaitKPolicy::set_adaptive_mode(), no state reset.
    // Rejected: translate = 1 without translator models (a Mode A pipeline);
    // while translating, a language id the translator does not know or
    // src == tgt. On the Mode A route language ids are stored, not checked.
    bool set_params(const VoiceParams& next) noexcept;
    const VoiceParams& params() const noexcept;
    std::size_t voice_count() const noexcept;

    // Ramped output mute (~5 ms each way, VoiceOutputChain::set_muted) on the
    // Mode A and the Mode B route alike. Worker/exclusive owner only, between
    // hops — the same contract as set_params(). It is a level: it survives
    // reset() and is not part of VoiceParams. With
    // PipelineOptions::output_chain = false it is ignored (raw PCM), as are
    // gain/gate/limiter; a legacy shell has no chain and ignores it too.
    void set_muted(bool muted) noexcept;

    // Legacy stateless-vocoder diagnostic helper, not the Mode A entry point.
    void process_vocoder_frame(const float* in, float* out);

    // Exclusive owner only: worker between hops, or controller AFTER worker
    // join + callback stop. Restores models, tracker/priors, OLA and governor.
    // Lifetime telemetry stays cumulative across stream resets.
    void reset();
    std::uint64_t reset_count() const { return resets_; }

private:
    struct State;
    struct Legacy;
    std::unique_ptr<State> state_;
    std::unique_ptr<Legacy> legacy_;
    std::uint64_t resets_ = 0;
};

} // namespace fluxvoice
