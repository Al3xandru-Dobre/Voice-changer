// PipelineTiming — fixed chunk geometry shared by the worker loop, the chunk
// assembler and the inference pipeline. One inference step per 80 ms hop.
#pragma once

#include <cstddef>

namespace fluxvoice {

/// Timing geometry @ 16 kHz.
struct PipelineTiming {
    static constexpr std::size_t kSampleRate     = 16000;
    static constexpr std::size_t kChunkSamples   = 1280;  // 80 ms hop
    static constexpr std::size_t kContextSamples = 320;   // 20 ms historical context
    static constexpr std::size_t kOverlapSamples = 320;   // 20 ms OLA tail
};

} // namespace fluxvoice
