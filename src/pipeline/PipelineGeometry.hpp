#pragma once

#include "pipeline/ModelContract.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace fluxvoice {

struct PipelineGeometry {
    std::uint32_t sample_rate = 0;
    std::size_t frame_samples = 0;
    std::size_t frames_per_step = 0;
    std::size_t hop = 0;
    std::size_t lookahead_frames = 0; // encoder + vocoder, in content frames
    std::size_t overlap_samples = 0;
    std::size_t cond_frame_samples = 0;
    std::size_t cond_frames = 0;
    // Mode B (S2UT) translation chunk in content frames (translator_enc
    // chunk_frames, a multiple of frames_per_step); 0 in Mode A.
    std::size_t chunk_frames = 0;

    std::size_t queue_samples() const noexcept {
        return hop;
    }
    std::size_t chunk_samples() const noexcept { return chunk_frames * frame_samples; }
    // Hops per translation chunk step (Mode B); 0 in Mode A.
    std::size_t hops_per_chunk() const noexcept {
        return frames_per_step ? chunk_frames / frames_per_step : 0;
    }
    std::size_t model_delay_samples() const noexcept {
        return lookahead_frames * frame_samples + overlap_samples;
    }
    std::size_t algorithmic_samples() const noexcept {
        return queue_samples() + model_delay_samples();
    }
    std::uint64_t hop_us() const noexcept {
        return sample_rate ? std::uint64_t{hop} * 1000000 / sample_rate : 0;
    }
};

// Load-time only. Leave `out` unchanged on failure.
inline bool make_pipeline_geometry(const ModelContract& enc, const ModelContract& voc,
                                   PipelineGeometry& out, std::string& error) {
    if (!validate_contract(enc, error) || !validate_contract(voc, error) ||
        !validate_pair(enc, voc, error)) return false;
    PipelineGeometry g;
    g.sample_rate = static_cast<std::uint32_t>(enc.sample_rate);
    g.frame_samples = static_cast<std::size_t>(enc.frame_samples);
    g.frames_per_step = static_cast<std::size_t>(enc.frames_per_step);
    g.hop = static_cast<std::size_t>(enc.hop());
    g.lookahead_frames = static_cast<std::size_t>(enc.lookahead_frames + voc.lookahead_frames);
    g.overlap_samples = static_cast<std::size_t>(voc.overlap_samples);
    g.cond_frame_samples = static_cast<std::size_t>(voc.cond_frame_samples);
    g.cond_frames = static_cast<std::size_t>(voc.cond_frames());
    out = g;
    error.clear();
    return true;
}

} // namespace fluxvoice
