// InferencePipeline — per-80 ms-chunk orchestrator:
//   encoder (ONNX) → [optional Wait-k S2UT translator] → unit vocoder → OLA.
//
// Milestone 1 provides the DSP shell only: an OverlapAddProcessor plus the
// fixed chunk geometry. The ONNX sessions and their KV-caches are wired in
// during Milestones 2 (Mode A: encoder → vocoder) and 3 (Mode B: Wait-k S2UT).
//
// TODO(M2): Ort::Session for causal encoder + unit vocoder; drive them with
//           KVStateCache-backed state tensors.
// TODO(M3): insert WaitKPolicy between encoder and vocoder.
#include "audio/OverlapAdd.hpp"
#include "pipeline/KVStateCache.hpp"
#include "pipeline/WaitKPolicy.hpp"

#include <cstddef>

namespace fluxvoice {

/// Timing geometry @ 16 kHz — one inference step per 80 ms hop.
struct PipelineTiming {
    static constexpr size_t kSampleRate     = 16000;
    static constexpr size_t kChunkSamples   = 1280;  // 80 ms hop
    static constexpr size_t kContextSamples = 320;   // 20 ms historical context
    static constexpr size_t kOverlapSamples = 320;   // 20 ms OLA tail
};

/// Stateful per-stream inference context.
/// M1: OLA only. M2+: holds Ort::Session objects + KVStateCache instances.
class InferencePipeline {
public:
    InferencePipeline()
        : ola_(PipelineTiming::kChunkSamples, PipelineTiming::kOverlapSamples) {}

    /// Consume one (chunk + overlap) vocoder frame → clean chunk of PCM.
    /// In M1 the input is the loopback/passthrough signal; in M2+ it is the
    /// unit-vocoder output.
    void process_vocoder_frame(const float* in, float* out) {
        ola_.process(in, out);
    }

    /// End-of-utterance / stream restart: drop OLA tail and model state.
    void reset() {
        ola_.reset();
        encoder_state_.reset();
        translator_state_.reset();
        waitk_.reset();
    }

private:
    OverlapAddProcessor ola_;

    // Populated in Milestones 2–3.
    KVStateCache encoder_state_;
    KVStateCache translator_state_;
    WaitKPolicy  waitk_{3};
};

} // namespace fluxvoice
