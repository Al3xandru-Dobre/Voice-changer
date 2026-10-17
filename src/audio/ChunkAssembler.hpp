// ChunkAssembler — turns an arbitrary-sized sample stream into fixed
// neural-model frames.
//
// Each frame is [context | hop]: `context` samples of history carried over from
// the previous frame, followed by `hop` new samples (M2 geometry @ 16 kHz:
// 320 + 1280 = 20 ms + 80 ms). Consecutive frames advance by exactly `hop`.
// The first frame's context is zero (silence before the stream started).
//
// Frame boundaries depend only on the sample count, never on how the samples
// were delivered — miniaudio period sizes are hints, so callback sizes vary.
//
// Worker-thread component (not real-time), but still allocation-free after
// construction so it can live in the steady-state inference loop.
//
// Usage:
//     while (n > 0) {
//         size_t used = asm.push(p, n);  p += used;  n -= used;
//         if (asm.ready()) { consume(asm.frame(), asm.frame_size()); asm.advance(); }
//     }
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

class ChunkAssembler {
public:
    ChunkAssembler(size_t hop, size_t context)
        : hop_(hop), context_(context), buf_(context + hop, 0.0f), fill_(context) {}

    /// Copy up to `n` samples in; stops early once a frame is complete.
    /// Returns the number of samples consumed.
    size_t push(const float* in, size_t n) {
        const size_t take = std::min(n, buf_.size() - fill_);
        std::memcpy(buf_.data() + fill_, in, take * sizeof(float));
        fill_ += take;
        return take;
    }

    /// True when frame() holds a complete [context | hop] frame.
    bool ready() const { return fill_ == buf_.size(); }

    /// The current frame (valid while ready()); frame_size() samples.
    const float* frame() const { return buf_.data(); }

    /// The `hop` new samples of the current frame (after the context).
    const float* hop_samples() const { return buf_.data() + context_; }

    /// Release the current frame: its last `context` samples become the next
    /// frame's context. Call only when ready().
    void advance() {
        std::memmove(buf_.data(), buf_.data() + hop_, context_ * sizeof(float));
        fill_ = context_;
    }

    /// Stream restart: drop history and any partial frame.
    void reset() {
        std::fill(buf_.begin(), buf_.end(), 0.0f);
        fill_ = context_;
    }

    size_t hop()        const { return hop_; }
    size_t context()    const { return context_; }
    size_t frame_size() const { return buf_.size(); }
    /// New samples buffered toward the next frame (0 … hop).
    size_t pending()    const { return fill_ - context_; }

private:
    size_t hop_;
    size_t context_;
    std::vector<float> buf_;
    size_t fill_;
};
