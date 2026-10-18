// InferencePipeline — per-80 ms-chunk orchestrator:
//   encoder (ONNX) → [optional Wait-k S2UT translator] → unit vocoder → OLA.
//
// Milestone 1 / 1.5 provides the DSP shell (OverlapAddProcessor + KVStateCache
// + WaitKPolicy) and the stream-switch `reset()` hook called by
// AudioController after quiescing the worker and draining the SPSC queues.
#pragma once

#include "audio/OverlapAdd.hpp"
#include "pipeline/KVStateCache.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "pipeline/WaitKPolicy.hpp"

#include <cstddef>
#include <cstdint>

namespace fluxvoice {

/// Stateful per-stream inference context.
/// M1/M1.5: OLA + state cache shells. M2+: holds Ort::Session objects +
/// KVStateCache instances.
class InferencePipeline {
public:
    InferencePipeline();

    /// Consume one (chunk + overlap) vocoder frame → clean chunk of PCM.
    void process_vocoder_frame(const float* in, float* out);

    /// End-of-utterance / stream restart: drop OLA tail and model state.
    /// Contract (M1.5 stage 05 / M2 stage 08): call only when the worker is
    /// quiesced so no inference step is in flight.
    void reset();

    std::uint64_t reset_count() const { return resets_; }

private:
    OverlapAddProcessor ola_;
    KVStateCache encoder_state_;
    KVStateCache translator_state_;
    WaitKPolicy  waitk_{3};
    std::uint64_t resets_ = 0;
};

} // namespace fluxvoice
