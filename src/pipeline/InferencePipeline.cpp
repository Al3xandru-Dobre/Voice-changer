// InferencePipeline — per-80 ms-chunk orchestrator implementation.
#include "pipeline/InferencePipeline.hpp"

namespace fluxvoice {

InferencePipeline::InferencePipeline()
    : ola_(PipelineTiming::kChunkSamples, PipelineTiming::kOverlapSamples) {}

void InferencePipeline::process_vocoder_frame(const float* in, float* out) {
    ola_.process(in, out);
}

void InferencePipeline::reset() {
    ola_.reset();
    encoder_state_.reset();
    translator_state_.reset();
    waitk_.reset();
    ++resets_;
}

} // namespace fluxvoice
