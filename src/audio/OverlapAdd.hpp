// OverlapAddProcessor — raised-cosine (Hann) crossfader for chunked vocoder
// output.
//
// The vocoder emits CHUNK + OVERLAP samples per inference step (e.g. 80 ms +
// 20 ms @ 16 kHz = 1600 samples). The leading OVERLAP region is crossfaded
// with the previous chunk's saved tail, eliminating phase/amplitude clicks at
// chunk boundaries. The curves satisfy fade_in[i] + fade_out[i] == 1.0f so the
// splice is constant-amplitude.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

class OverlapAddProcessor {
public:
    OverlapAddProcessor(size_t chunk_size, size_t overlap_size)
        : chunk_size_(chunk_size),
          overlap_size_(overlap_size),
          prev_tail_(overlap_size, 0.0f),
          fade_in_(overlap_size),
          fade_out_(overlap_size) {
        for (size_t i = 0; i < overlap_size_; ++i) {
            const float phase = static_cast<float>(i) / static_cast<float>(overlap_size_);
            fade_in_[i]  = 0.5f * (1.0f - std::cos(phase * kPi));
            fade_out_[i] = 1.0f - fade_in_[i];
        }
    }

    /// in:  vocoder output of size (chunk_size() + overlap_size())
    /// out: clean chunk of size chunk_size(), ready for the output ring buffer.
    void process(const float* vocoder_out, float* clean_out) {
        for (size_t i = 0; i < overlap_size_; ++i)
            clean_out[i] = vocoder_out[i] * fade_in_[i] + prev_tail_[i] * fade_out_[i];
        for (size_t i = overlap_size_; i < chunk_size_; ++i)
            clean_out[i] = vocoder_out[i];
        for (size_t i = 0; i < overlap_size_; ++i)
            prev_tail_[i] = vocoder_out[chunk_size_ + i];
    }

    /// Drop the saved tail (stream restart / voice model swap).
    void reset() {
        std::fill(prev_tail_.begin(), prev_tail_.end(), 0.0f);
    }

    size_t chunk_size()   const { return chunk_size_; }
    size_t overlap_size() const { return overlap_size_; }

private:
    static constexpr float kPi = 3.1415926535f;

    size_t chunk_size_;
    size_t overlap_size_;
    std::vector<float> prev_tail_;
    std::vector<float> fade_in_;
    std::vector<float> fade_out_;
};
