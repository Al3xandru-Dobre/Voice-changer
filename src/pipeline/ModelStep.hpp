// Stable model-owned I/O views; an ORT adapter and offline fakes share this seam.
#pragma once

#include "pipeline/ModelContract.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace fluxvoice {

class ModelStep {
public:
    virtual ~ModelStep() = default;
    virtual const ModelContract& contract() const noexcept = 0;

    // Missing names/wrong dtypes return empty spans. Views remain valid across
    // step/reset (including state-bank flips); obtain them once at pipeline load.
    virtual std::span<float> input_f32(std::string_view name) noexcept = 0;
    virtual std::span<std::int64_t> input_i64(std::string_view name) noexcept = 0;
    virtual std::span<const float> output_f32(std::string_view name) const noexcept = 0;
    virtual std::span<const std::int64_t> output_i64(std::string_view name) const noexcept = 0;

    // No exceptions or first-party allocations on the steady-state worker path.
    virtual bool step() noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual std::string_view last_error() const noexcept = 0;
};

} // namespace fluxvoice
