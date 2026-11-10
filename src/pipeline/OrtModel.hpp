// OrtModel — stateful ONNX Runtime C++ model wrapper
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
//
// This header intentionally does NOT include <onnxruntime_cxx_api.h>: ORT
// types stay inside OrtModel.cpp (guarded by FLUXVOICE_WITH_ORT), keeping
// warnings-as-errors clean and allowing ORT-less builds to include this header
// and compile against the same interface.
//
// Design invariants (BEST_PRACTICES.md §4.5, MEMORY.md rules 5, 7, 8):
//   * One process-wide Ort::Env with global thread pools: intra_op = 1,
//     inter_op = 1, spin control disabled (0), denormals-as-zero enabled.
//   * Sessions disable per-session threads, run sequentially (ORT_SEQUENTIAL),
//     and enable all graph optimizations (ORT_ENABLE_ALL).
//   * Ort::GetVersionString() is checked against the pinned SDK version at load
//     time to prevent C:\Windows\System32\onnxruntime.dll shadowing.
//   * Every model's embedded `fluxvoice.contract` metadata is parsed and
//     cross-checked against the graph's inputs, outputs, and state pairs.
//   * All non-state inputs/outputs and two state banks (A and B) are
//     preallocated in KVStateCache at load time and bound to two prebuilt
//     Ort::IoBinding instances (bank A -> B on even steps, B -> A on odd
//     steps). step() performs zero allocations, zero copies, and never throws.
#pragma once

#include "pipeline/KVStateCache.hpp"
#include "pipeline/ModelContract.hpp"
#include "pipeline/ModelStep.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#ifndef FLUXVOICE_ORT_VERSION
#define FLUXVOICE_ORT_VERSION "1.31.0"
#endif

namespace fluxvoice {

inline constexpr const char* kPinnedOrtVersion = FLUXVOICE_ORT_VERSION;

inline constexpr const char* kBuiltWithoutOrtMessage =
    "built without ONNX Runtime (configure with -DONNXRUNTIME_ROOT=... or -DFLUXVOICE_FETCH_ORT=ON)";

enum class OrtStatus {
    kOk = 0,
    kBuiltWithoutOrt,
    kVersionMismatch,
    kContractError,
    kNotLoaded,
    kOrtError,
};

inline const char* to_string(OrtStatus s) noexcept {
    switch (s) {
    case OrtStatus::kOk:              return "ok";
    case OrtStatus::kBuiltWithoutOrt: return "built_without_ort";
    case OrtStatus::kVersionMismatch: return "version_mismatch";
    case OrtStatus::kContractError:   return "contract_error";
    case OrtStatus::kNotLoaded:       return "not_loaded";
    case OrtStatus::kOrtError:        return "ort_error";
    }
    return "unknown";
}

/// True when this binary was compiled with FLUXVOICE_WITH_ORT.
bool ort_available() noexcept;

/// Runtime version reported by Ort::GetVersionString(), or "" when built without ORT.
std::string ort_runtime_version();

/// Compare the runtime's loaded version string with the expected SDK version.
/// On mismatch, populates `error` with a diagnostic naming the Windows
/// System32\onnxruntime.dll shadowing problem and returns false.
bool check_ort_version(std::string_view actual_version,
                       std::string_view expected_version,
                       std::string& error);

class OrtModel : public ModelStep {
public:
    OrtModel();
    ~OrtModel() override;

    OrtModel(OrtModel&&) noexcept;
    OrtModel& operator=(OrtModel&&) noexcept;

    OrtModel(const OrtModel&) = delete;
    OrtModel& operator=(const OrtModel&) = delete;

    /// Load an ONNX model from disk, check the ORT runtime version against
    /// `expected_version`, validate the embedded `fluxvoice.contract` against
    /// the session's inputs/outputs, preallocate all I/O and A/B state banks,
    /// and prebuild both ping-pong Ort::IoBindings. Never throws.
    OrtStatus load(const std::filesystem::path& model_path,
                   std::string_view expected_version = kPinnedOrtVersion) noexcept;

    bool is_loaded() const noexcept { return loaded_; }
    const ModelContract& contract() const noexcept override { return contract_; }
    std::string_view last_error() const noexcept override { return last_error_; }
    const std::string& last_error_str() const noexcept { return last_error_; }

    /// Run one inference hop using the current prebuilt IoBinding (bank A -> B
    /// when active_bank() == 0, bank B -> A when active_bank() == 1) and flip
    /// the active state bank. Performs zero allocations in FluxVoice code and
    /// never throws.
    OrtStatus step_status() noexcept;
    bool step() noexcept override { return step_status() == OrtStatus::kOk; }

    /// Zero inputs/outputs/states, run `steps` inference passes on zeros to
    /// force ORT kernel/arena initialization before the audio stream starts,
    /// then reset() both state banks and parity back to 0.
    OrtStatus warm_up(std::size_t steps = 4) noexcept;

    /// Zero both state banks (A and B) and outputs in place and reset the
    /// ping-pong parity to bank 0 (A). Allocation-free.
    void reset() noexcept override;

    /// Index (0 = bank A, 1 = bank B) of the state bank that will be fed as
    /// `state_in.*` on the next step() (and that received `state_out.*` on the
    /// previous step()).
    std::size_t active_bank() const noexcept { return parity_; }

    const KVStateCache& state_bank(std::size_t bank) const noexcept { return state_banks_[bank & 1u]; }
    KVStateCache& state_bank(std::size_t bank) noexcept { return state_banks_[bank & 1u]; }
    const KVStateCache& active_state_bank() const noexcept { return state_banks_[parity_]; }

    // Generic named non-state input/output accessors (ModelStep interface).
    std::span<float> input_f32(std::string_view name) noexcept override;
    std::span<const float> input_f32(std::string_view name) const noexcept;
    std::span<std::int64_t> input_i64(std::string_view name) noexcept override;
    std::span<const std::int64_t> input_i64(std::string_view name) const noexcept;

    std::span<float> output_f32(std::string_view name) noexcept;
    std::span<const float> output_f32(std::string_view name) const noexcept override;
    std::span<std::int64_t> output_i64(std::string_view name) noexcept;
    std::span<const std::int64_t> output_i64(std::string_view name) const noexcept override;

    // Fast cached spans for standard contract v1 tensors (populated at load()).
    std::span<float> audio_in() noexcept { return audio_in_; }
    std::span<const float> audio_out() const noexcept { return audio_out_; }
    std::span<std::int64_t> units_in() noexcept { return units_in_; }
    std::span<const std::int64_t> units_out() const noexcept { return units_out_; }
    std::span<float> content_in() noexcept { return content_in_; }
    std::span<const float> content_out() const noexcept { return content_out_; }
    std::span<float> logf0_in() noexcept { return logf0_in_; }
    std::span<float> voicing_in() noexcept { return voicing_in_; }
    std::span<float> aperiodicity_in() noexcept { return aperiodicity_in_; }
    std::span<float> energy_in() noexcept { return energy_in_; }
    std::span<float> spk_emb_in() noexcept { return spk_emb_in_; }

private:
    void clear_cached_spans() noexcept;
    void cache_standard_spans() noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelContract contract_{};
    KVStateCache inputs_{};
    KVStateCache outputs_{};
    std::array<KVStateCache, 2> state_banks_{};
    std::size_t parity_ = 0;
    bool loaded_ = false;
    std::string last_error_;

    std::span<float> audio_in_{};
    std::span<const float> audio_out_{};
    std::span<std::int64_t> units_in_{};
    std::span<const std::int64_t> units_out_{};
    std::span<float> content_in_{};
    std::span<const float> content_out_{};
    std::span<float> logf0_in_{};
    std::span<float> voicing_in_{};
    std::span<float> aperiodicity_in_{};
    std::span<float> energy_in_{};
    std::span<float> spk_emb_in_{};
};

} // namespace fluxvoice
