// KVStateCache — named, reusable state buffers for stateful ONNX inference
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
//
// Stateful ONNX graphs expose every recurrent tensor (causal-conv context,
// attention KV cache, step/position counter) as an explicit `state_in.X` /
// `state_out.X` pair. KVStateCache owns the preallocated backing memory for a
// bank of tensors across all contract-supported dtypes (f32, i64, f16 raw
// bytes) while staying completely free of ONNX Runtime headers so ORT-less
// builds can compile and test it directly.
//
// OrtModel holds two KVStateCache banks (A and B) and binds their buffers to
// two prebuilt Ort::IoBinding instances (A -> B and B -> A) at load time, so
// steady-state hops perform zero heap allocations and zero state copies.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fluxvoice {

enum class TensorDtype { kF32, kI64, kF16, kOther };

inline const char* to_string(TensorDtype d) noexcept {
    switch (d) {
    case TensorDtype::kF32: return "f32";
    case TensorDtype::kI64: return "i64";
    case TensorDtype::kF16: return "f16";
    default:                return "other";
    }
}

inline TensorDtype dtype_from_string(std::string_view s) noexcept {
    if (s == "f32") return TensorDtype::kF32;
    if (s == "i64") return TensorDtype::kI64;
    if (s == "f16") return TensorDtype::kF16;
    return TensorDtype::kOther;
}

inline std::size_t dtype_element_size(TensorDtype d) noexcept {
    switch (d) {
    case TensorDtype::kF32: return sizeof(float);
    case TensorDtype::kI64: return sizeof(std::int64_t);
    case TensorDtype::kF16: return sizeof(std::uint16_t);
    default:                return 0;
    }
}

class KVStateCache {
public:
    /// Preallocated backing store for one named tensor (f32, i64, or f16 raw
    /// bytes). Backed by 64-bit words so raw_data() is always 8-byte aligned
    /// for float, int64_t, and uint16_t views.
    struct Buffer {
        std::vector<std::int64_t> shape;
        TensorDtype               dtype = TensorDtype::kF32;
        std::vector<std::uint64_t> storage;

        std::size_t num_elements() const noexcept {
            for (std::int64_t d : shape) {
                if (d <= 0) return 0;
            }
            return static_cast<std::size_t>(std::accumulate(
                shape.begin(), shape.end(), std::int64_t{1},
                [](std::int64_t a, std::int64_t b) { return a * b; }));
        }

        std::size_t element_size() const noexcept { return dtype_element_size(dtype); }

        std::size_t num_bytes() const noexcept { return num_elements() * element_size(); }

        void* raw_data() noexcept {
            return storage.empty() ? nullptr : static_cast<void*>(storage.data());
        }
        const void* raw_data() const noexcept {
            return storage.empty() ? nullptr : static_cast<const void*>(storage.data());
        }

        std::uint8_t* bytes() noexcept {
            return static_cast<std::uint8_t*>(raw_data());
        }
        const std::uint8_t* bytes() const noexcept {
            return static_cast<const std::uint8_t*>(raw_data());
        }

        float* as_f32() noexcept {
            return dtype == TensorDtype::kF32 ? static_cast<float*>(raw_data()) : nullptr;
        }
        const float* as_f32() const noexcept {
            return dtype == TensorDtype::kF32 ? static_cast<const float*>(raw_data()) : nullptr;
        }

        std::int64_t* as_i64() noexcept {
            return dtype == TensorDtype::kI64 ? static_cast<std::int64_t*>(raw_data()) : nullptr;
        }
        const std::int64_t* as_i64() const noexcept {
            return dtype == TensorDtype::kI64 ? static_cast<const std::int64_t*>(raw_data()) : nullptr;
        }

        std::uint16_t* as_f16() noexcept {
            return dtype == TensorDtype::kF16 ? static_cast<std::uint16_t*>(raw_data()) : nullptr;
        }
        const std::uint16_t* as_f16() const noexcept {
            return dtype == TensorDtype::kF16 ? static_cast<const std::uint16_t*>(raw_data()) : nullptr;
        }

        std::span<float> span_f32() noexcept {
            float* p = as_f32();
            return p ? std::span<float>(p, num_elements()) : std::span<float>{};
        }
        std::span<const float> span_f32() const noexcept {
            const float* p = as_f32();
            return p ? std::span<const float>(p, num_elements()) : std::span<const float>{};
        }

        std::span<std::int64_t> span_i64() noexcept {
            std::int64_t* p = as_i64();
            return p ? std::span<std::int64_t>(p, num_elements()) : std::span<std::int64_t>{};
        }
        std::span<const std::int64_t> span_i64() const noexcept {
            const std::int64_t* p = as_i64();
            return p ? std::span<const std::int64_t>(p, num_elements()) : std::span<const std::int64_t>{};
        }

        std::span<std::uint16_t> span_f16() noexcept {
            std::uint16_t* p = as_f16();
            return p ? std::span<std::uint16_t>(p, num_elements()) : std::span<std::uint16_t>{};
        }
        std::span<const std::uint16_t> span_f16() const noexcept {
            const std::uint16_t* p = as_f16();
            return p ? std::span<const std::uint16_t>(p, num_elements()) : std::span<const std::uint16_t>{};
        }

        std::span<std::uint8_t> span_bytes() noexcept {
            std::uint8_t* p = bytes();
            return p ? std::span<std::uint8_t>(p, num_bytes()) : std::span<std::uint8_t>{};
        }
        std::span<const std::uint8_t> span_bytes() const noexcept {
            const std::uint8_t* p = bytes();
            return p ? std::span<const std::uint8_t>(p, num_bytes()) : std::span<const std::uint8_t>{};
        }

        void zero() noexcept {
            std::fill(storage.begin(), storage.end(), std::uint64_t{0});
        }
    };

    /// Create (or return) the named buffer sized to `shape` and `dtype`.
    /// If the buffer already exists with a different shape or dtype, it is
    /// resized and zeroed; if shape and dtype match, existing contents are kept.
    Buffer& register_buffer(const std::string& name, std::vector<std::int64_t> shape,
                            TensorDtype dtype = TensorDtype::kF32) {
        auto [it, inserted] = buffers_.try_emplace(name);
        Buffer& b = it->second;
        if (inserted || b.shape != shape || b.dtype != dtype) {
            b.shape = std::move(shape);
            b.dtype = dtype;
            const std::size_t n_bytes = b.num_bytes();
            const std::size_t words = (n_bytes + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t);
            b.storage.assign(words, std::uint64_t{0});
        }
        return b;
    }

    Buffer* find(std::string_view name) noexcept {
        auto it = buffers_.find(name);
        return it != buffers_.end() ? &it->second : nullptr;
    }
    const Buffer* find(std::string_view name) const noexcept {
        auto it = buffers_.find(name);
        return it != buffers_.end() ? &it->second : nullptr;
    }

    /// Zero every buffer in place (zero allocations) — call on stream restart /
    /// end-of-utterance so stale context never leaks into the next segment.
    void reset() noexcept {
        for (auto& [_, b] : buffers_)
            b.zero();
    }

    void clear() { buffers_.clear(); }
    std::size_t size() const noexcept { return buffers_.size(); }

    std::size_t total_bytes() const noexcept {
        std::size_t total = 0;
        for (const auto& [_, b] : buffers_) total += b.num_bytes();
        return total;
    }

private:
    struct TransparentStringHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    std::unordered_map<std::string, Buffer, TransparentStringHash, std::equal_to<>> buffers_;
};

} // namespace fluxvoice

using fluxvoice::KVStateCache;

