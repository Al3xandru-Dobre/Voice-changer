// Mode A worker-thread orchestrator. Models own stable I/O buffers and outlive
// this pipeline, with exclusive access; no ONNX Runtime dependency in fake CI.
#pragma once

#include "dsp/PitchTracker.hpp"
#include "pipeline/TimingHistogram.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace fluxvoice {

class ModelStep;
class LatencyGovernor;
struct PipelineGeometry;

struct PipelineOptions {
    std::size_t max_consecutive_errors = 8;
    double pitch_shift_st = 0.0;
    double intonation_scale = 1.0;
    float quiet_peak = 0.001f;
    std::size_t latency_target_samples = 0; // 0 = master-budget default
    // Stage 09 hook; worker-owned context and callback, no allocations/throws.
    void (*output_gate)(std::span<float>, void*) noexcept = nullptr;
    void* output_gate_context = nullptr;
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
                      const TargetVoiceStats& voice, std::span<const float> speaker_embedding,
                      PipelineOptions options = {});
    ~InferencePipeline();
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;

    // Exactly one contract hop in/out. Bad geometry is a fatal caller error;
    // model errors produce faded silence, reset stream state, and request stop
    // only after max_consecutive_errors. Error text is owned until next call.
    PipelineStepResult process(std::span<const float> input, std::span<float> output) noexcept;

    // Run zeros through the complete chain BEFORE audio starts, then reset all
    // stream state and telemetry. A failed warm-up must prevent stream startup.
    bool warm_up(std::size_t steps);
    const PipelineGeometry& geometry() const;
    LatencyGovernor& governor(); // worker only; use after process(), before enqueue
    TimingSnapshot timing() const noexcept;
    std::uint64_t errors() const noexcept;
    // Worker/quiesced owner only; unlike timing()/errors(), not a UI snapshot.
    std::string_view last_error() const noexcept;

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
