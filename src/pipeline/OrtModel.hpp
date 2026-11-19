// OrtModel — stateful ONNX Runtime C++ model wrapper
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md;
// two-graph S2UT translator: Milestone 3 / Stage 07,
// Documents/Milestone 3/07_cpp_translator_runtime_offline_runner.md).
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
//
// Zero-copy cross-model inputs (M3 Stage 07). translator_enc keeps the
// cross-attention ring memory (`mem_k_*`, `mem_v_*`) and the `src_pos` counter
// as its own ping-pong states; translator_dec reads them as plain graph inputs.
// The two models step a different number of times (kRead runs only the
// encoder, kFlush only the decoder), so their parities drift apart.
// bind_shared_inputs() therefore prebuilds four IoBindings on the decoder,
// indexed [encoder bank][decoder parity], whose shared inputs point straight at
// the encoder's bank tensors: every step is an O(1) table lookup with no copy,
// no rebinding and no allocation.
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
#include <vector>

#ifndef FLUXVOICE_ORT_VERSION
#define FLUXVOICE_ORT_VERSION "1.31.0"
#endif

namespace fluxvoice {

inline constexpr const char* kPinnedOrtVersion = FLUXVOICE_ORT_VERSION;

inline constexpr const char* kBuiltWithoutOrtMessage =
    "built without ONNX Runtime (configure with -DONNXRUNTIME_ROOT=... or -DFLUXVOICE_FETCH_ORT=ON)";

/// step() error of a consumer whose producer was reloaded (or moved) after
/// bind_shared_inputs(): the bound tensor views are gone.
inline constexpr const char* kSharedProducerReloadedMessage =
    "shared producer was reloaded; call bind_shared_inputs() again";

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

class OrtModel;

/// Warm up a bound translator pair before the audio stream starts: zero both
/// models' inputs, reset() both, run `steps` x (trans_enc.step();
/// trans_dec.step()) so every ORT kernel/arena of both sessions initialises,
/// then zero the inputs and reset() both again (all four state banks zero,
/// both parities 0). `trans_dec` must be bound to `trans_enc`
/// (bind_shared_inputs()), else kContractError. Returns the first failing
/// status; that model's last_error() says why (both are reset either way).
OrtStatus warm_up_translator_pair(OrtModel& trans_enc, OrtModel& trans_dec,
                                  std::size_t steps = 4) noexcept;

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
    /// and prebuild both ping-pong Ort::IoBindings. Never throws. Any load()
    /// drops this model's shared-input binding and invalidates every binding
    /// other models made to this one (their next step() fails).
    OrtStatus load(const std::filesystem::path& model_path,
                   std::string_view expected_version = kPinnedOrtVersion) noexcept;

    bool is_loaded() const noexcept { return loaded_; }
    const ModelContract& contract() const noexcept override { return contract_; }
    std::string_view last_error() const noexcept override { return last_error_; }
    const std::string& last_error_str() const noexcept { return last_error_; }

    /// Graph inputs and outputs as ONNX Runtime reported them at the last
    /// successful load(): names, dtypes and shapes, including the state_in./
    /// state_out. tensors and translator_dec's read-only shared memory inputs
    /// (what validate_s2ut_shared_io() takes). Empty while not loaded.
    const std::vector<TensorSpec>& graph_inputs() const noexcept { return graph_inputs_; }
    const std::vector<TensorSpec>& graph_outputs() const noexcept { return graph_outputs_; }

    /// Run one inference hop using the current prebuilt IoBinding (bank A -> B
    /// when active_bank() == 0, bank B -> A when active_bank() == 1) and flip
    /// the active state bank. With shared inputs bound, the binding is
    /// table[shared_producer()->active_bank()][active_bank()]. A translator_dec
    /// that was never bound fails with kContractError, and so does a consumer
    /// whose producer was reloaded (kSharedProducerReloadedMessage). The parity
    /// flips only on success. Performs zero allocations in FluxVoice code and
    /// never throws.
    OrtStatus step_status() noexcept;
    bool step() noexcept override { return step_status() == OrtStatus::kOk; }

    /// Zero inputs/outputs/states, run `steps` inference passes on zeros to
    /// force ORT kernel/arena initialization before the audio stream starts,
    /// then reset() both state banks and parity back to 0. A bound consumer
    /// steps through its shared table and only reads the producer's banks; to
    /// warm up a translator pair use warm_up_translator_pair().
    OrtStatus warm_up(std::size_t steps = 4) noexcept;

    /// Zero both state banks (A and B) and outputs in place and reset the
    /// ping-pong parity to bank 0 (A). Allocation-free. Never touches a shared
    /// producer: reset() the producer as well to clear shared memory.
    void reset() noexcept override;

    /// Index (0 = bank A, 1 = bank B) of the state bank that will be fed as
    /// `state_in.*` on the next step() (and that received `state_out.*` on the
    /// previous step()).
    std::size_t active_bank() const noexcept { return parity_; }

    const KVStateCache& state_bank(std::size_t bank) const noexcept { return state_banks_[bank & 1u]; }
    KVStateCache& state_bank(std::size_t bank) noexcept { return state_banks_[bank & 1u]; }
    const KVStateCache& active_state_bank() const noexcept { return state_banks_[parity_]; }

    // -- Zero-copy shared inputs (M3 Stage 07) --------------------------------

    /// Bind plain graph inputs of this model (the consumer) straight to state
    /// tensors of `producer`, with zero copies. Typical call:
    ///     names = trans_enc.contract().shared_mem_states + {"src_pos"};
    ///     trans_dec.bind_shared_inputs(trans_enc, names);
    /// Requirements: both models are loaded; `producer` is not this model;
    /// `names` is non-empty without duplicates; every name is a graph input of
    /// this model that is not a state_in.* input, and a state of `producer`
    /// (listed in producer.contract().states) whose graph input
    /// `state_in.<name>` has the identical dtype and static shape; and every
    /// graph input this model has no buffer for (translator_dec's
    /// `shared_mem_states`) is among `names`.
    ///
    /// Builds a 2 x 2 table of prebuilt Ort::IoBindings indexed
    /// [producer bank e][own parity d]: own non-shared inputs and all outputs
    /// on this model's preallocated buffers, own states ping-ponging bank d ->
    /// bank d^1 as in load(), each shared name on the producer's bank-e tensor
    /// for that state (the same memory, read-only for this model).
    ///
    /// Afterwards input_f32()/input_i64() return an EMPTY span for a shared
    /// name: the tensor lives in the producer's state banks (write it there,
    /// through producer.state_bank()).
    ///
    /// Failure returns kContractError (a requirement above; the message in
    /// last_error() names it), kNotLoaded (this model is not loaded) or
    /// kOrtError (ONNX Runtime refused a binding) and leaves this model loaded
    /// but unbound — an earlier binding is dropped either way.
    ///
    /// Lifetime: the producer must stay loaded at the same address while
    /// bound; never destroy or move it (a bundle owns both models as members;
    /// call bind_shared_inputs() again after moving the bundle). A reload (or
    /// move) of the producer is detected through its load generation: the next
    /// step() fails with kSharedProducerReloadedMessage. Reloading this model
    /// drops its binding; moving it keeps the binding. Bind on the control
    /// thread, never while either model is stepping.
    OrtStatus bind_shared_inputs(const OrtModel& producer, std::span<const std::string> names) noexcept;

    /// True after a successful bind_shared_inputs() until this model is
    /// reloaded or a later bind_shared_inputs() fails. Stays true when the
    /// producer is reloaded (step() then fails until you bind again).
    bool has_shared_inputs() const noexcept { return shared_producer_ != nullptr; }
    const OrtModel* shared_producer() const noexcept { return shared_producer_; }

    /// Like step_status(), but read the shared inputs from producer bank
    /// `producer_bank & 1` instead of shared_producer()->active_bank() (tests
    /// and explicit scheduling): runs table[producer_bank & 1][active_bank()].
    /// Requires bind_shared_inputs() (else kContractError); flips only this
    /// model's parity, never the producer's.
    OrtStatus step_with_producer_bank(std::size_t producer_bank) noexcept;

    // Generic named non-state input/output accessors (ModelStep interface).
    // A missing name, a wrong dtype or a shared input (see
    // bind_shared_inputs()) returns an empty span.
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
    /// v2 dual-output encoder `latents` [1, frames_per_step, latent_dim] f32
    /// (empty for v1 encoders and other roles).
    std::span<const float> latents_out() const noexcept { return latents_out_; }

private:
    friend OrtStatus warm_up_translator_pair(OrtModel&, OrtModel&, std::size_t) noexcept;

    /// Identity of one loaded state of one object: a fresh process-wide value
    /// on construction, on every load() and on both sides of a move, so a
    /// consumer bound to this model notices a reload (or move) with one
    /// integer compare per step.
    class LoadGeneration {
    public:
        LoadGeneration() noexcept;
        LoadGeneration(LoadGeneration&& other) noexcept;
        LoadGeneration& operator=(LoadGeneration&& other) noexcept;
        LoadGeneration(const LoadGeneration&) = delete;
        LoadGeneration& operator=(const LoadGeneration&) = delete;
        ~LoadGeneration() = default;

        void advance() noexcept;
        std::uint64_t value() const noexcept { return value_; }

    private:
        std::uint64_t value_ = 0;
    };

    void clear_cached_spans() noexcept;
    void cache_standard_spans() noexcept;
    void clear_shared_binding() noexcept;
    bool is_shared_input(std::string_view name) const noexcept;
    void set_error(const char* message) noexcept;
    bool check_shared_request(const OrtModel& producer, std::span<const std::string> names,
                              std::vector<std::size_t>& producer_states, std::string& error) const;
    /// Runs the prebuilt binding for (producer bank, parity_) when `shared`,
    /// else bindings[parity_]; flips parity_ on success.
    OrtStatus run_prebuilt(bool shared, std::size_t producer_bank) noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelContract contract_{};
    KVStateCache inputs_{};
    KVStateCache outputs_{};
    std::array<KVStateCache, 2> state_banks_{};
    std::size_t parity_ = 0;
    bool loaded_ = false;
    std::string last_error_;
    std::vector<TensorSpec> graph_inputs_{};
    std::vector<TensorSpec> graph_outputs_{};
    LoadGeneration generation_{};

    // Consumer side of bind_shared_inputs() (empty / null while unbound).
    const OrtModel* shared_producer_ = nullptr;
    std::uint64_t shared_producer_generation_ = 0;  // producer generation_ at bind time
    std::vector<std::string> shared_names_{};
    bool requires_shared_inputs_ = false;  // translator_dec: step() needs a binding

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
    std::span<const float> latents_out_{};
};

} // namespace fluxvoice
