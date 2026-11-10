// OrtModel — stateful ONNX Runtime C++ model wrapper implementation
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
#include "pipeline/OrtModel.hpp"

#ifdef FLUXVOICE_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <cstdio>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>

namespace fluxvoice {

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

} // namespace

struct OrtModel::Impl {
    Ort::MemoryInfo mem_info{nullptr};
    Ort::Session session{nullptr};
    Ort::RunOptions run_options{nullptr};
    std::vector<Ort::Value> input_values;
    std::vector<Ort::Value> output_values;
    std::array<std::vector<Ort::Value>, 2> state_values;
    std::array<Ort::IoBinding, 2> bindings{Ort::IoBinding{nullptr}, Ort::IoBinding{nullptr}};
};

#else // !FLUXVOICE_WITH_ORT

struct OrtModel::Impl {};

#endif // FLUXVOICE_WITH_ORT

OrtModel::OrtModel() = default;
OrtModel::~OrtModel() = default;
OrtModel::OrtModel(OrtModel&&) noexcept = default;
OrtModel& OrtModel::operator=(OrtModel&&) noexcept = default;

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
}

OrtStatus OrtModel::load(const std::filesystem::path& model_path,
                         std::string_view expected_version) noexcept {
    loaded_ = false;
    parity_ = 0;
    contract_ = ModelContract{};
    inputs_.clear();
    outputs_.clear();
    state_banks_[0].clear();
    state_banks_[1].clear();
    clear_cached_spans();
    impl_.reset();
    last_error_.clear();

#ifndef FLUXVOICE_WITH_ORT
    (void)model_path;
    (void)expected_version;
    last_error_ = kBuiltWithoutOrtMessage;
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
        impl->input_values.reserve(expected_ins.size());
        impl->output_values.reserve(expected_outs.size());

        impl->bindings[0] = Ort::IoBinding(impl->session);
        impl->bindings[1] = Ort::IoBinding(impl->session);

        for (const TensorSpec& spec : expected_ins) {
            auto& buf = inputs_.register_buffer(spec.name, spec.shape, spec.dtype);
            impl->input_values.push_back(create_tensor_view(impl->mem_info, buf));
            impl->bindings[0].BindInput(spec.name.c_str(), impl->input_values.back());
            impl->bindings[1].BindInput(spec.name.c_str(), impl->input_values.back());
        }

        for (const TensorSpec& spec : expected_outs) {
            auto& buf = outputs_.register_buffer(spec.name, spec.shape, spec.dtype);
            impl->output_values.push_back(create_tensor_view(impl->mem_info, buf));
            impl->bindings[0].BindOutput(spec.name.c_str(), impl->output_values.back());
            impl->bindings[1].BindOutput(spec.name.c_str(), impl->output_values.back());
        }

        // Preallocate state banks A (0) and B (1) and bind ping-pong:
        //   binding 0 (even parity): state_in.X -> bank A, state_out.X -> bank B
        //   binding 1 (odd parity):  state_in.X -> bank B, state_out.X -> bank A
        impl->state_values[0].reserve(parsed.states.size());
        impl->state_values[1].reserve(parsed.states.size());

        for (const std::string& state_name : parsed.states) {
            const std::string in_name = std::string(kStateIn) + state_name;
            const std::string out_name = std::string(kStateOut) + state_name;
            const TensorSpec* state_spec = contract_detail::find_spec(graph_inputs, in_name);
            if (!state_spec) {
                last_error_ = "missing state input '" + in_name + "'";
                return OrtStatus::kContractError;
            }
            auto& bank_a = state_banks_[0].register_buffer(state_name, state_spec->shape, state_spec->dtype);
            auto& bank_b = state_banks_[1].register_buffer(state_name, state_spec->shape, state_spec->dtype);
            impl->state_values[0].push_back(create_tensor_view(impl->mem_info, bank_a));
            impl->state_values[1].push_back(create_tensor_view(impl->mem_info, bank_b));

            const Ort::Value& val_a = impl->state_values[0].back();
            const Ort::Value& val_b = impl->state_values[1].back();

            impl->bindings[0].BindInput(in_name.c_str(), val_a);
            impl->bindings[0].BindOutput(out_name.c_str(), val_b);

            impl->bindings[1].BindInput(in_name.c_str(), val_b);
            impl->bindings[1].BindOutput(out_name.c_str(), val_a);
        }

        contract_ = std::move(parsed);
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

OrtStatus OrtModel::step_status() noexcept {
#ifndef FLUXVOICE_WITH_ORT
    last_error_ = kBuiltWithoutOrtMessage;
    return OrtStatus::kBuiltWithoutOrt;
#else
    if (!loaded_ || !impl_) {
        last_error_ = "model is not loaded";
        return OrtStatus::kNotLoaded;
    }
    try {
        impl_->session.Run(impl_->run_options, impl_->bindings[parity_]);
        parity_ ^= 1u;
        return OrtStatus::kOk;
    } catch (const Ort::Exception& e) {
        last_error_ = e.what();
        return OrtStatus::kOrtError;
    } catch (const std::exception& e) {
        last_error_ = e.what();
        return OrtStatus::kOrtError;
    } catch (...) {
        last_error_ = "unknown exception in OrtModel::step";
        return OrtStatus::kOrtError;
    }
#endif
}

OrtStatus OrtModel::warm_up(std::size_t steps) noexcept {
#ifndef FLUXVOICE_WITH_ORT
    (void)steps;
    last_error_ = kBuiltWithoutOrtMessage;
    return OrtStatus::kBuiltWithoutOrt;
#else
    if (!loaded_ || !impl_) {
        last_error_ = "model is not loaded";
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

void OrtModel::reset() noexcept {
    outputs_.reset();
    state_banks_[0].reset();
    state_banks_[1].reset();
    parity_ = 0;
}

std::span<float> OrtModel::input_f32(std::string_view name) noexcept {
    auto* b = inputs_.find(name);
    return b ? b->span_f32() : std::span<float>{};
}

std::span<const float> OrtModel::input_f32(std::string_view name) const noexcept {
    const auto* b = inputs_.find(name);
    return b ? b->span_f32() : std::span<const float>{};
}

std::span<std::int64_t> OrtModel::input_i64(std::string_view name) noexcept {
    auto* b = inputs_.find(name);
    return b ? b->span_i64() : std::span<std::int64_t>{};
}

std::span<const std::int64_t> OrtModel::input_i64(std::string_view name) const noexcept {
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
