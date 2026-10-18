// KVStateCache — named, reusable state buffers for stateful ONNX inference.
//
// Milestone 2 will export the encoder / translator / vocoder as *stateful*
// ONNX graphs whose `past_key_values` (and causal-conv recurrent states) are
// explicit graph inputs and outputs. This cache owns the backing memory for
// those tensors so the same buffers can be fed back as inputs on the next
// 80 ms chunk — zero allocation in the steady-state audio loop.
//
// For production examples of this pattern in C++, see sherpa-onnx's
// online-zipformer-transducer-model.cc (RESEARCH_GUIDE.md §4).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class KVStateCache {
public:
    /// fp32 backing store for one named state tensor.
    /// TODO(M2): bind each Buffer to an Ort::Value via
    /// Ort::Value::CreateTensor(..., data.data(), num_elements(), shape).
    struct Buffer {
        std::vector<std::int64_t> shape;
        std::vector<float>   data;

        std::size_t num_elements() const {
            return static_cast<std::size_t>(std::accumulate(
                shape.begin(), shape.end(), std::int64_t{1},
                [](std::int64_t a, std::int64_t b) { return a * b; }));
        }
    };

    /// Create (or return) the named buffer sized to `shape`.
    Buffer& register_buffer(const std::string& name, std::vector<std::int64_t> shape) {
        auto [it, inserted] = buffers_.try_emplace(name);
        if (inserted) {
            it->second.shape = std::move(shape);
            it->second.data.assign(it->second.num_elements(), 0.0f);
        }
        return it->second;
    }

    Buffer* find(const std::string& name) {
        auto it = buffers_.find(name);
        return it != buffers_.end() ? &it->second : nullptr;
    }
    const Buffer* find(const std::string& name) const {
        auto it = buffers_.find(name);
        return it != buffers_.end() ? &it->second : nullptr;
    }

    /// Zero every buffer — call on stream restart / end-of-utterance so
    /// stale context never leaks into the next segment.
    void reset() {
        for (auto& [_, b] : buffers_)
            std::fill(b.data.begin(), b.data.end(), 0.0f);
    }

    void clear() { buffers_.clear(); }
    std::size_t size() const { return buffers_.size(); }

private:
    std::unordered_map<std::string, Buffer> buffers_;
};
