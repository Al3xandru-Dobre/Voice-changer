// OrtModel — stateful ONNX Runtime C++ model wrapper implementation
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md;
// zero-copy translator inputs: Milestone 3 / Stage 07,
// Documents/Milestone 3/07_cpp_translator_runtime_offline_runner.md).
#include "pipeline/OrtModel.hpp"

#ifdef FLUXVOICE_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <mutex>
#include <utility>
#include <vector>

namespace fluxvoice {

namespace {

/// Source of OrtModel load generations: process-wide, so a value is never
/// reused, not even by a new model constructed at a destroyed model's address.
std::atomic<std::uint64_t> g_load_generation{0};

std::uint64_t next_load_generation() noexcept {
    return g_load_generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

} // namespace

OrtModel::LoadGeneration::LoadGeneration() noexcept : value_(next_load_generation()) {}

OrtModel::LoadGeneration::LoadGeneration(LoadGeneration&& other) noexcept
    : value_(next_load_generation()) {
    other.advance();
}

OrtModel::LoadGeneration& OrtModel::LoadGeneration::operator=(LoadGeneration&& other) noexcept {
    if (this != &other) {
        advance();
        other.advance();
    }
    return *this;
}

void OrtModel::LoadGeneration::advance() noexcept { value_ = next_load_generation(); }

bool ort_available() noexcept {
#ifdef FLUXVOICE_WITH_ORT
    return true;
#else
    return false;
#endif
}

std::string ort_runtime_version() {
#ifdef FLUXVOICE_WITH_ORT
    try {
        return Ort::GetVersionString();
    } catch (...) {
        return {};
    }
#else
    return {};
#endif
}

bool check_ort_version(std::string_view actual_version,
                       std::string_view expected_version,
                       std::string& error) {
    if (actual_version == expected_version) return true;
    error = "ONNX Runtime version mismatch: loaded '" + std::string(actual_version) +
            "', expected '" + std::string(expected_version) +
            "' (ensure the SDK onnxruntime.dll/.so/.dylib sits next to the executable "
            "and is not shadowed by C:\\Windows\\System32\\onnxruntime.dll)";
    return false;
}

#ifdef FLUXVOICE_WITH_ORT
namespace {

/// Process-wide ONNX Runtime environment configured with global thread pools:
/// intra_op = 1, inter_op = 1, spin control off (0), denormals-as-zero on.
class OrtRuntime {
public:
    static OrtRuntime& instance() {
        static OrtRuntime rt;
        return rt;
    }

    Ort::Env& env() noexcept { return env_; }

    void log_version_once(std::string_view loaded_ver) {
        std::call_once(logged_flag_, [loaded_ver]() {
            std::fprintf(stderr,
                         "[ort] ONNX Runtime %.*s (pinned %s, global thread pool: "
                         "intra_op=1, inter_op=1, sequential, spin=0, denormal_as_zero=1)\n",
                         static_cast<int>(loaded_ver.size()), loaded_ver.data(),
                         kPinnedOrtVersion);
        });
    }

private:
    OrtRuntime() : env_( nullptr ) {
        Ort::ThreadingOptions tp_opts;
        tp_opts.SetGlobalIntraOpNumThreads(1);
        tp_opts.SetGlobalInterOpNumThreads(1);
        tp_opts.SetGlobalSpinControl(0);
        tp_opts.SetGlobalDenormalAsZero();
        env_ = Ort::Env(tp_opts, ORT_LOGGING_LEVEL_WARNING, "fluxvoice");
    }

    Ort::Env env_;
    std::once_flag logged_flag_;
};

TensorDtype map_ort_dtype(ONNXTensorElementDataType elem_type) noexcept {
    switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return TensorDtype::kF32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return TensorDtype::kI64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return TensorDtype::kF16;
    default:                                    return TensorDtype::kOther;
    }
}

ONNXTensorElementDataType to_ort_dtype(TensorDtype dtype) noexcept {
    switch (dtype) {
    case TensorDtype::kF32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case TensorDtype::kI64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TensorDtype::kF16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    default:                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

TensorSpec inspect_endpoint(std::string name, const Ort::TypeInfo& type_info) {
    TensorSpec spec;
    spec.name = std::move(name);
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        spec.dtype = TensorDtype::kOther;
        spec.has_shape = false;
        return spec;
    }
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    spec.dtype = map_ort_dtype(tensor_info.GetElementType());
    if (!tensor_info.HasShape()) {
        spec.has_shape = false;
        return spec;
    }
    spec.has_shape = true;
    spec.shape = tensor_info.GetShape();
    return spec;
}

Ort::Value create_tensor_view(const Ort::MemoryInfo& mem_info, KVStateCache::Buffer& buf) {
    return Ort::Value::CreateTensor(
        mem_info,
        buf.raw_data(),
        buf.num_bytes(),
        buf.shape.empty() ? nullptr : buf.shape.data(),
        buf.shape.size(),
        to_ort_dtype(buf.dtype));
}

/// Index into OrtModel::Impl::shared_bindings for [producer bank][own parity].
constexpr std::size_t shared_slot(std::size_t producer_bank, std::size_t parity) noexcept {
    return ((producer_bank & 1u) << 1) | (parity & 1u);
}

/// step() error of a model whose read-only graph inputs (translator_dec's
/// `shared_mem_states`) were never bound to a producer.
constexpr const char* kUnboundSharedInputsMessage =
    "shared inputs are not bound (translator_dec reads translator_enc's mem_* and src_pos); "
    "call bind_shared_inputs() first";

} // namespace

struct OrtModel::Impl {
    Ort::MemoryInfo mem_info{nullptr};
    Ort::Session session{nullptr};
    Ort::RunOptions run_options{nullptr};
    // Parallel name / tensor-view lists: inputs and outputs in contract order,
    // states in contract().states order (state_values[bank][i] <-> states[i]).
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::string> state_in_names;
    std::vector<std::string> state_out_names;
    std::vector<Ort::Value> input_values;
    std::vector<Ort::Value> output_values;
    std::array<std::vector<Ort::Value>, 2> state_values;
    std::array<Ort::IoBinding, 2> bindings{Ort::IoBinding{nullptr}, Ort::IoBinding{nullptr}};
    // bind_shared_inputs() table, indexed by shared_slot(producer bank, own parity).
    std::array<Ort::IoBinding, 4> shared_bindings{Ort::IoBinding{nullptr}, Ort::IoBinding{nullptr},
                                                  Ort::IoBinding{nullptr}, Ort::IoBinding{nullptr}};

    /// Bind this model's own tensors for ping-pong parity `parity`: every
    /// non-state input except `skip_inputs`, every output, state_in.* on bank
    /// `parity` and state_out.* on bank `parity ^ 1`.
    void bind_own(Ort::IoBinding& binding, std::size_t parity,
                  std::span<const std::string> skip_inputs) const {
        for (std::size_t i = 0; i < input_names.size(); ++i) {
            if (std::find(skip_inputs.begin(), skip_inputs.end(), input_names[i]) != skip_inputs.end())
                continue;
            binding.BindInput(input_names[i].c_str(), input_values[i]);
        }
        for (std::size_t i = 0; i < output_names.size(); ++i)
            binding.BindOutput(output_names[i].c_str(), output_values[i]);
        const std::vector<Ort::Value>& in_bank = state_values[parity & 1u];
        const std::vector<Ort::Value>& out_bank = state_values[(parity & 1u) ^ 1u];
        for (std::size_t i = 0; i < state_in_names.size(); ++i) {
            binding.BindInput(state_in_names[i].c_str(), in_bank[i]);
            binding.BindOutput(state_out_names[i].c_str(), out_bank[i]);
        }
    }
};

#else // !FLUXVOICE_WITH_ORT

struct OrtModel::Impl {};

#endif // FLUXVOICE_WITH_ORT

OrtModel::OrtModel() = default;
OrtModel::~OrtModel() = default;
OrtModel::OrtModel(OrtModel&&) noexcept = default;
OrtModel& OrtModel::operator=(OrtModel&&) noexcept = default;

void OrtModel::set_error(const char* message) noexcept {
    try {
        last_error_.assign(message ? message : "");
    } catch (...) {
        last_error_.clear();
    }
}

void OrtModel::clear_cached_spans() noexcept {
    audio_in_ = {};
    audio_out_ = {};
    units_in_ = {};
    units_out_ = {};
    content_in_ = {};
    content_out_ = {};
    logf0_in_ = {};
    voicing_in_ = {};
    aperiodicity_in_ = {};
    energy_in_ = {};
    spk_emb_in_ = {};
    latents_out_ = {};
}

void OrtModel::cache_standard_spans() noexcept {
    clear_cached_spans();
    audio_in_ = input_f32("audio");
    audio_out_ = static_cast<const OrtModel*>(this)->output_f32("audio");
    units_in_ = input_i64("units");
    units_out_ = static_cast<const OrtModel*>(this)->output_i64("units");
    content_in_ = input_f32("content");
    content_out_ = static_cast<const OrtModel*>(this)->output_f32("content");
    logf0_in_ = input_f32("logf0");
    voicing_in_ = input_f32("voicing");
    aperiodicity_in_ = input_f32("aperiodicity");
    energy_in_ = input_f32("energy");
    spk_emb_in_ = input_f32("spk_emb");
    latents_out_ = static_cast<const OrtModel*>(this)->output_f32("latents");
}

void OrtModel::clear_shared_binding() noexcept {
    shared_producer_ = nullptr;
    shared_producer_generation_ = 0;
    shared_names_.clear();
#ifdef FLUXVOICE_WITH_ORT
    if (impl_) {
        // Release the table so it stops referencing the producer's tensors.
        for (Ort::IoBinding& binding : impl_->shared_bindings) binding = Ort::IoBinding{nullptr};
    }
#endif
}

bool OrtModel::is_shared_input(std::string_view name) const noexcept {
    return std::find(shared_names_.begin(), shared_names_.end(), name) != shared_names_.end();
}

OrtStatus OrtModel::load(const std::filesystem::path& model_path,
                         std::string_view expected_version) noexcept {
    // A new generation first: every model bound to this one sees the change
    // before the tensors it bound are released below.
    generation_.advance();
    clear_shared_binding();
    loaded_ = false;
    parity_ = 0;
    requires_shared_inputs_ = false;
    contract_ = ModelContract{};
    inputs_.clear();
    outputs_.clear();
    state_banks_[0].clear();
    state_banks_[1].clear();
    graph_inputs_.clear();
    graph_outputs_.clear();
    clear_cached_spans();
    impl_.reset();
    last_error_.clear();

#ifndef FLUXVOICE_WITH_ORT
    (void)model_path;
    (void)expected_version;
    set_error(kBuiltWithoutOrtMessage);
    return OrtStatus::kBuiltWithoutOrt;
#else
    try {
        const std::string actual_ver = Ort::GetVersionString();
        if (!check_ort_version(actual_ver, expected_version, last_error_)) {
            return OrtStatus::kVersionMismatch;
        }
        OrtRuntime& rt = OrtRuntime::instance();
        rt.log_version_once(actual_ver);

        if (!std::filesystem::is_regular_file(model_path)) {
            last_error_ = "model file not found: " + model_path.string();
            return OrtStatus::kOrtError;
        }

        Ort::SessionOptions so;
        so.DisablePerSessionThreads();
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto impl = std::make_unique<Impl>();
        impl->mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        impl->run_options = Ort::RunOptions{};
        impl->session = Ort::Session(rt.env(), model_path.c_str(), so);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ModelMetadata meta = impl->session.GetModelMetadata();
        Ort::AllocatedStringPtr contract_alloc =
            meta.LookupCustomMetadataMapAllocated(kContractMetadataKey, allocator);
        std::string contract_text;
        const std::string* contract_ptr = nullptr;
        if (contract_alloc) {
            contract_text = contract_alloc.get();
            contract_ptr = &contract_text;
        }

        const std::size_t n_inputs = impl->session.GetInputCount();
        std::vector<TensorSpec> graph_inputs;
        graph_inputs.reserve(n_inputs);
        for (std::size_t i = 0; i < n_inputs; ++i) {
            auto name_alloc = impl->session.GetInputNameAllocated(i, allocator);
            std::string name = name_alloc ? name_alloc.get() : "";
            graph_inputs.push_back(inspect_endpoint(std::move(name), impl->session.GetInputTypeInfo(i)));
        }

        const std::size_t n_outputs = impl->session.GetOutputCount();
        std::vector<TensorSpec> graph_outputs;
        graph_outputs.reserve(n_outputs);
        for (std::size_t i = 0; i < n_outputs; ++i) {
            auto name_alloc = impl->session.GetOutputNameAllocated(i, allocator);
            std::string name = name_alloc ? name_alloc.get() : "";
            graph_outputs.push_back(inspect_endpoint(std::move(name), impl->session.GetOutputTypeInfo(i)));
        }

        ModelContract parsed;
        if (!check_model(contract_ptr, graph_inputs, graph_outputs, parsed, last_error_)) {
            return OrtStatus::kContractError;
        }

        // Preallocate non-state inputs and outputs.
        const auto expected_ins = parsed.expected_inputs();
        const auto expected_outs = parsed.expected_outputs();
        impl->input_names.reserve(expected_ins.size());
        impl->input_values.reserve(expected_ins.size());
        impl->output_names.reserve(expected_outs.size());
        impl->output_values.reserve(expected_outs.size());

        for (const TensorSpec& spec : expected_ins) {
            auto& buf = inputs_.register_buffer(spec.name, spec.shape, spec.dtype);
            impl->input_names.push_back(spec.name);
            impl->input_values.push_back(create_tensor_view(impl->mem_info, buf));
        }

        for (const TensorSpec& spec : expected_outs) {
            auto& buf = outputs_.register_buffer(spec.name, spec.shape, spec.dtype);
            impl->output_names.push_back(spec.name);
            impl->output_values.push_back(create_tensor_view(impl->mem_info, buf));
        }

        // Preallocate state banks A (0) and B (1).
        impl->state_in_names.reserve(parsed.states.size());
        impl->state_out_names.reserve(parsed.states.size());
        impl->state_values[0].reserve(parsed.states.size());
        impl->state_values[1].reserve(parsed.states.size());

        for (const std::string& state_name : parsed.states) {
            const std::string in_name = std::string(kStateIn) + state_name;
            const TensorSpec* state_spec = contract_detail::find_spec(graph_inputs, in_name);
            if (!state_spec) {
                last_error_ = "missing state input '" + in_name + "'";
                return OrtStatus::kContractError;
            }
            auto& bank_a = state_banks_[0].register_buffer(state_name, state_spec->shape, state_spec->dtype);
            auto& bank_b = state_banks_[1].register_buffer(state_name, state_spec->shape, state_spec->dtype);
            impl->state_in_names.push_back(in_name);
            impl->state_out_names.push_back(std::string(kStateOut) + state_name);
            impl->state_values[0].push_back(create_tensor_view(impl->mem_info, bank_a));
            impl->state_values[1].push_back(create_tensor_view(impl->mem_info, bank_b));
        }

        // Prebuild the ping-pong bindings:
        //   binding 0 (even parity): state_in.X -> bank A, state_out.X -> bank B
        //   binding 1 (odd parity):  state_in.X -> bank B, state_out.X -> bank A
        for (std::size_t parity = 0; parity < 2; ++parity) {
            impl->bindings[parity] = Ort::IoBinding(impl->session);
            impl->bind_own(impl->bindings[parity], parity, {});
        }

        contract_ = std::move(parsed);
        graph_inputs_ = std::move(graph_inputs);
        graph_outputs_ = std::move(graph_outputs);
        // Graph inputs without a buffer of their own (translator_dec's
        // read-only shared_mem_states) stay unbound until bind_shared_inputs().
        requires_shared_inputs_ =
            std::any_of(graph_inputs_.begin(), graph_inputs_.end(), [this](const TensorSpec& t) {
                return !std::string_view(t.name).starts_with(kStateIn) && inputs_.find(t.name) == nullptr;
            });
        impl_ = std::move(impl);
        loaded_ = true;
        cache_standard_spans();
        return OrtStatus::kOk;
    } catch (const Ort::Exception& e) {
        last_error_ = e.what();
        return OrtStatus::kOrtError;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        return OrtStatus::kOrtError;
    } catch (...) {
        last_error_ = "unknown exception in OrtModel::load";
        return OrtStatus::kOrtError;
    }
#endif
}

#ifdef FLUXVOICE_WITH_ORT
bool OrtModel::check_shared_request(const OrtModel& producer, std::span<const std::string> names,
                                    std::vector<std::size_t>& producer_states, std::string& error) const {
    using contract_detail::fail;
    using contract_detail::find_spec;
    const std::string pre = "bind_shared_inputs: ";
    if (&producer == this) return fail(error, pre + "a model cannot share inputs with itself");
    if (!producer.loaded_ || !producer.impl_) return fail(error, pre + "the producer model is not loaded");
    if (names.empty()) return fail(error, pre + "no input names given");

    const std::vector<std::string>& their_states = producer.contract_.states;
    producer_states.clear();
    for (std::size_t k = 0; k < names.size(); ++k) {
        const std::string& name = names[k];
        const std::string quoted = "'" + name + "'";
        const auto earlier = names.first(k);
        if (std::find(earlier.begin(), earlier.end(), name) != earlier.end())
            return fail(error, pre + "duplicate name " + quoted);
        if (std::string_view(name).starts_with(kStateIn) || std::string_view(name).starts_with(kStateOut))
            return fail(error, pre + quoted + " is a state tensor; only plain graph inputs can be shared");
        const TensorSpec* mine = find_spec(graph_inputs_, name);
        if (!mine)
            return fail(error, pre + quoted + " is not a graph input of this " + to_string(contract_.role) + " model");
        const auto state = std::find(their_states.begin(), their_states.end(), name);
        if (state == their_states.end()) {
            return fail(error, pre + quoted + " is not a state of the producer " + to_string(producer.contract_.role) +
                                   " model");
        }
        const std::string their_name = std::string(kStateIn) + name;
        const TensorSpec* theirs = find_spec(producer.graph_inputs_, their_name);
        if (!theirs) return fail(error, pre + "the producer has no graph input '" + their_name + "'");
        if (theirs->dtype != mine->dtype) {
            return fail(error, pre + quoted + " dtype mismatch: producer " + their_name + " is " +
                                   to_string(theirs->dtype) + ", this model's input is " + to_string(mine->dtype));
        }
        if (!theirs->has_shape || !mine->has_shape || theirs->shape != mine->shape) {
            return fail(error, pre + quoted + " shape mismatch: producer " + their_name + " is " +
                                   format_shape(*theirs) + ", this model's input is " + format_shape(*mine));
        }
        producer_states.push_back(static_cast<std::size_t>(state - their_states.begin()));
    }
    // Every graph input without a buffer of its own must come from the producer.
    for (const TensorSpec& g : graph_inputs_) {
        if (std::string_view(g.name).starts_with(kStateIn) || inputs_.find(g.name) != nullptr) continue;
        if (std::find(names.begin(), names.end(), g.name) == names.end())
            return fail(error, pre + "graph input '" + g.name + "' has no buffer of its own; add it to the shared names");
    }
    return true;
}
#endif

OrtStatus OrtModel::bind_shared_inputs(const OrtModel& producer, std::span<const std::string> names) noexcept {
#ifndef FLUXVOICE_WITH_ORT
    (void)producer;
    (void)names;
    set_error(kBuiltWithoutOrtMessage);
    return OrtStatus::kBuiltWithoutOrt;
#else
    if (!loaded_ || !impl_) {
        set_error("model is not loaded");
        return OrtStatus::kNotLoaded;
    }
    // Unbound from here on: a failed call leaves the model loaded but unbound.
    clear_shared_binding();
    cache_standard_spans();
    try {
        std::vector<std::size_t> producer_states;
        std::string error;
        if (!check_shared_request(producer, names, producer_states, error)) {
            last_error_ = std::move(error);
            return OrtStatus::kContractError;
        }

        // table[e][d]: own tensors with ping-pong parity d, plus each shared
        // name on the producer's bank-e tensor (the producer's memory itself).
        std::array<Ort::IoBinding, 4> table{Ort::IoBinding{nullptr}, Ort::IoBinding{nullptr},
                                            Ort::IoBinding{nullptr}, Ort::IoBinding{nullptr}};
        for (std::size_t bank = 0; bank < 2; ++bank) {
            const std::vector<Ort::Value>& producer_bank = producer.impl_->state_values[bank];
            for (std::size_t parity = 0; parity < 2; ++parity) {
                Ort::IoBinding& binding = table[shared_slot(bank, parity)];
                binding = Ort::IoBinding(impl_->session);
                impl_->bind_own(binding, parity, names);
                for (std::size_t k = 0; k < names.size(); ++k)
                    binding.BindInput(names[k].c_str(), producer_bank[producer_states[k]]);
            }
        }

        shared_names_.assign(names.begin(), names.end());
        impl_->shared_bindings = std::move(table);
        shared_producer_ = &producer;
        shared_producer_generation_ = producer.generation_.value();
        cache_standard_spans();  // shared names now read as empty spans
        return OrtStatus::kOk;
    } catch (const Ort::Exception& e) {
        clear_shared_binding();
        cache_standard_spans();
        set_error(e.what());
        return OrtStatus::kOrtError;
    } catch (const std::exception& e) {
        clear_shared_binding();
        cache_standard_spans();
        set_error(e.what());
        return OrtStatus::kOrtError;
    } catch (...) {
        clear_shared_binding();
        cache_standard_spans();
        set_error("unknown exception in OrtModel::bind_shared_inputs");
        return OrtStatus::kOrtError;
    }
#endif
}

#ifdef FLUXVOICE_WITH_ORT
OrtStatus OrtModel::run_prebuilt(bool shared, std::size_t producer_bank) noexcept {
    try {
        Ort::IoBinding& binding =
            shared ? impl_->shared_bindings[shared_slot(producer_bank, parity_)] : impl_->bindings[parity_];
        impl_->session.Run(impl_->run_options, binding);
        parity_ ^= 1u;
        return OrtStatus::kOk;
    } catch (const Ort::Exception& e) {
        set_error(e.what());
        return OrtStatus::kOrtError;
    } catch (const std::exception& e) {
        set_error(e.what());
        return OrtStatus::kOrtError;
    } catch (...) {
        set_error("unknown exception in OrtModel::step");
        return OrtStatus::kOrtError;
    }
}
#endif

OrtStatus OrtModel::step_status() noexcept {
#ifndef FLUXVOICE_WITH_ORT
    set_error(kBuiltWithoutOrtMessage);
    return OrtStatus::kBuiltWithoutOrt;
#else
    if (!loaded_ || !impl_) {
        set_error("model is not loaded");
        return OrtStatus::kNotLoaded;
    }
    if (shared_producer_) return step_with_producer_bank(shared_producer_->parity_);
    if (requires_shared_inputs_) {
        set_error(kUnboundSharedInputsMessage);
        return OrtStatus::kContractError;
    }
    return run_prebuilt(false, 0);
#endif
}

OrtStatus OrtModel::step_with_producer_bank(std::size_t producer_bank) noexcept {
#ifndef FLUXVOICE_WITH_ORT
    (void)producer_bank;
    set_error(kBuiltWithoutOrtMessage);
    return OrtStatus::kBuiltWithoutOrt;
#else
    if (!loaded_ || !impl_) {
        set_error("model is not loaded");
        return OrtStatus::kNotLoaded;
    }
    if (!shared_producer_) {
        set_error("step_with_producer_bank: no shared inputs are bound; call bind_shared_inputs() first");
        return OrtStatus::kContractError;
    }
    if (shared_producer_->generation_.value() != shared_producer_generation_) {
        set_error(kSharedProducerReloadedMessage);
        return OrtStatus::kContractError;
    }
    return run_prebuilt(true, producer_bank & 1u);
#endif
}

OrtStatus OrtModel::warm_up(std::size_t steps) noexcept {
#ifndef FLUXVOICE_WITH_ORT
    (void)steps;
    set_error(kBuiltWithoutOrtMessage);
    return OrtStatus::kBuiltWithoutOrt;
#else
    if (!loaded_ || !impl_) {
        set_error("model is not loaded");
        return OrtStatus::kNotLoaded;
    }
    inputs_.reset();
    reset();
    for (std::size_t i = 0; i < steps; ++i) {
        const OrtStatus st = step_status();
        if (st != OrtStatus::kOk) return st;
    }
    inputs_.reset();
    reset();
    return OrtStatus::kOk;
#endif
}

OrtStatus warm_up_translator_pair(OrtModel& trans_enc, OrtModel& trans_dec, std::size_t steps) noexcept {
#ifndef FLUXVOICE_WITH_ORT
    (void)steps;
    trans_enc.set_error(kBuiltWithoutOrtMessage);
    trans_dec.set_error(kBuiltWithoutOrtMessage);
    return OrtStatus::kBuiltWithoutOrt;
#else
    for (OrtModel* m : {&trans_enc, &trans_dec}) {
        if (!m->loaded_ || !m->impl_) {
            m->set_error("model is not loaded");
            return OrtStatus::kNotLoaded;
        }
    }
    if (trans_dec.shared_producer_ != &trans_enc) {
        trans_dec.set_error("warm_up_translator_pair: translator_dec is not bound to this translator_enc; "
                            "call bind_shared_inputs() first");
        return OrtStatus::kContractError;
    }
    for (OrtModel* m : {&trans_enc, &trans_dec}) {
        m->inputs_.reset();
        m->reset();
    }
    OrtStatus st = OrtStatus::kOk;
    for (std::size_t i = 0; i < steps && st == OrtStatus::kOk; ++i) {
        st = trans_enc.step_status();
        if (st == OrtStatus::kOk) st = trans_dec.step_status();
    }
    for (OrtModel* m : {&trans_enc, &trans_dec}) {
        m->inputs_.reset();
        m->reset();
    }
    return st;
#endif
}

void OrtModel::reset() noexcept {
    outputs_.reset();
    state_banks_[0].reset();
    state_banks_[1].reset();
    parity_ = 0;
}

std::span<float> OrtModel::input_f32(std::string_view name) noexcept {
    if (is_shared_input(name)) return {};
    auto* b = inputs_.find(name);
    return b ? b->span_f32() : std::span<float>{};
}

std::span<const float> OrtModel::input_f32(std::string_view name) const noexcept {
    if (is_shared_input(name)) return {};
    const auto* b = inputs_.find(name);
    return b ? b->span_f32() : std::span<const float>{};
}

std::span<std::int64_t> OrtModel::input_i64(std::string_view name) noexcept {
    if (is_shared_input(name)) return {};
    auto* b = inputs_.find(name);
    return b ? b->span_i64() : std::span<std::int64_t>{};
}

std::span<const std::int64_t> OrtModel::input_i64(std::string_view name) const noexcept {
    if (is_shared_input(name)) return {};
    const auto* b = inputs_.find(name);
    return b ? b->span_i64() : std::span<const std::int64_t>{};
}

std::span<float> OrtModel::output_f32(std::string_view name) noexcept {
    auto* b = outputs_.find(name);
    return b ? b->span_f32() : std::span<float>{};
}

std::span<const float> OrtModel::output_f32(std::string_view name) const noexcept {
    const auto* b = outputs_.find(name);
    return b ? b->span_f32() : std::span<const float>{};
}

std::span<std::int64_t> OrtModel::output_i64(std::string_view name) noexcept {
    auto* b = outputs_.find(name);
    return b ? b->span_i64() : std::span<std::int64_t>{};
}

std::span<const std::int64_t> OrtModel::output_i64(std::string_view name) const noexcept {
    const auto* b = outputs_.find(name);
    return b ? b->span_i64() : std::span<const std::int64_t>{};
}

} // namespace fluxvoice
