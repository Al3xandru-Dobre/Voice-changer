// ModelContract — what the engine needs to know about an ONNX model
// (Milestone 2 / Stage 02).
//
// Every encoder and vocoder carries one JSON string in its ONNX
// metadata_props["fluxvoice.contract"]: sample rate, frame geometry,
// lookahead, content kind, conditioning inputs, state pairs and licence. The
// ORT loader (Stage 07) reads that string, hands it to parse_contract(), then
// checks the session's real inputs/outputs with validate_io() — check_model()
// does both. A 20/40/80 ms hop is therefore a model property, not a recompile.
//
// No ONNX Runtime here, so the ORT-less build compiles and tests it. Nothing
// throws: functions return false and leave a message in `error`.
//
// python_training/fvml/contract.py implements the same rules in the same order
// with the same messages, and tests/data/contracts/ holds the cases both run.
// Change a rule or a message in both files at once.
//
// Spec: Documents/Milestone 2/02_model_contract_fixtures.md.
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fluxvoice {

inline constexpr int         kContractVersion     = 1;
inline constexpr const char* kContractMetadataKey = "fluxvoice.contract";

// Physical bounds. With these, every derived size (hop, cond_frames,
// output_samples, algorithmic latency) fits int32, and no contract can make the
// loader preallocate gigabytes.
inline constexpr int kMinSampleRate      = 8000;
inline constexpr int kMaxSampleRate      = 192000;
inline constexpr int kMaxLookaheadFrames = 64;
inline constexpr int kMaxVocab           = 1 << 20;
inline constexpr int kMaxContentDim      = 1 << 16;
inline constexpr int kMaxSpkEmbDim       = 1 << 12;

enum class ModelRole   { kEncoder, kVocoder };
enum class ContentKind { kDiscrete, kSoft };
enum class TensorDtype { kF32, kI64, kF16, kOther };

inline const char* to_string(ModelRole r)   { return r == ModelRole::kEncoder ? "encoder" : "vocoder"; }
inline const char* to_string(ContentKind k) { return k == ContentKind::kDiscrete ? "discrete" : "soft"; }
inline const char* to_string(TensorDtype d) {
    switch (d) {
    case TensorDtype::kF32: return "f32";
    case TensorDtype::kI64: return "i64";
    case TensorDtype::kF16: return "f16";
    default:                return "other";
    }
}

inline TensorDtype dtype_from_string(std::string_view s) {
    if (s == "f32") return TensorDtype::kF32;
    if (s == "i64") return TensorDtype::kI64;
    if (s == "f16") return TensorDtype::kF16;
    return TensorDtype::kOther;
}

/// One graph input or output. `shape` uses -1 for a dynamic dimension;
/// `has_shape` is false when the graph gives no shape at all (unknown rank —
/// an empty `shape` with has_shape = true is a scalar).
struct TensorSpec {
    std::string          name;
    TensorDtype          dtype = TensorDtype::kF32;
    std::vector<int64_t> shape;
    bool                 has_shape = true;
};

/// A parsed, validated v1 contract. Fields of the other role / content kind
/// stay at their defaults.
struct ModelContract {
    int                      contract_version = kContractVersion;
    ModelRole                role = ModelRole::kEncoder;
    int                      sample_rate = 0;
    int                      frame_samples = 0;       // samples per model frame
    int                      frames_per_step = 0;     // 1, 2 or 4 frames per Run
    int                      lookahead_frames = 0;    // output at step t describes frame t - L
    ContentKind              content_kind = ContentKind::kDiscrete;
    int                      vocab = 0;               // discrete only
    int                      content_dim = 0;         // soft only
    int                      cond_frame_samples = 0;  // vocoder only
    std::vector<std::string> cond_inputs;             // vocoder only
    int                      spk_emb_dim = 0;         // vocoder only
    int                      overlap_samples = 0;     // vocoder only; 0 for a stateful causal vocoder
    std::vector<std::string> states;                  // base names X of state_in.X / state_out.X
    std::string              license;

    /// Samples consumed (encoder) or produced (vocoder) per Run.
    int64_t hop() const { return int64_t{frames_per_step} * frame_samples; }
    int64_t cond_frames() const { return cond_frame_samples > 0 ? hop() / cond_frame_samples : 0; }
    /// Vocoder `audio` output length: hop plus the OLA tail.
    int64_t output_samples() const { return hop() + overlap_samples; }

    /// Encoder output = vocoder input: `units` (i64 ids) or `content` (f32).
    const char* content_name() const { return content_kind == ContentKind::kDiscrete ? "units" : "content"; }

    TensorSpec content_spec() const {
        if (content_kind == ContentKind::kDiscrete) return {"units", TensorDtype::kI64, {1, frames_per_step}};
        return {"content", TensorDtype::kF32, {1, frames_per_step, content_dim}};
    }

    /// Non-state graph inputs, in contract order (state shapes come from the graph).
    std::vector<TensorSpec> expected_inputs() const {
        if (role == ModelRole::kEncoder) return {{"audio", TensorDtype::kF32, {1, hop()}}};
        std::vector<TensorSpec> specs{content_spec()};
        for (const auto& name : cond_inputs)
            specs.push_back({name, TensorDtype::kF32, {1, name == "spk_emb" ? int64_t{spk_emb_dim} : cond_frames()}});
        return specs;
    }

    std::vector<TensorSpec> expected_outputs() const {
        if (role == ModelRole::kEncoder) return {content_spec()};
        return {{"audio", TensorDtype::kF32, {1, output_samples()}}};
    }
};

inline const char* const kCondInputNames[] = {"logf0", "voicing", "aperiodicity", "energy", "spk_emb"};
inline const char* const kVocoderKeys[]    = {"cond_frame_samples", "cond_inputs", "spk_emb_dim", "overlap_samples"};
inline constexpr std::string_view kStateIn  = "state_in.";
inline constexpr std::string_view kStateOut = "state_out.";

/// "[1,320]"; dynamic dimensions print as "?".
inline std::string format_shape(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ',';
        s += shape[i] < 0 ? std::string("?") : std::to_string(shape[i]);
    }
    return s + "]";
}

/// As above; a tensor without shape information prints as "unknown".
inline std::string format_shape(const TensorSpec& t) { return t.has_shape ? format_shape(t.shape) : "unknown"; }

/// The message a loader gives for a model without contract metadata.
inline std::string missing_contract_message() {
    return std::string("model has no '") + kContractMetadataKey + "' metadata";
}

namespace contract_detail {

using Json = nlohmann::json;

inline bool fail(std::string& error, std::string message) {
    error = std::move(message);
    return false;
}

inline std::string key_msg(std::string_view key, const char* what) {
    return "key '" + std::string(key) + "' " + what;
}

inline std::string version_message(int version) {
    return "unsupported contract_version " + std::to_string(version) + " (this build reads version " +
           std::to_string(kContractVersion) + ")";
}

inline bool is_vocoder_key(std::string_view key) {
    return std::find(std::begin(kVocoderKeys), std::end(kVocoderKeys), key) != std::end(kVocoderKeys);
}

inline const char* content_key(ContentKind kind) { return kind == ContentKind::kDiscrete ? "vocab" : "content_dim"; }

inline std::string misplaced_key_message(std::string_view key, ModelRole role, ContentKind kind) {
    if (is_vocoder_key(key) && role != ModelRole::kVocoder)
        return key_msg(key, "is not valid for role '") + to_string(role) + "'";
    if ((key == "vocab" || key == "content_dim") && key != content_key(kind))
        return key_msg(key, "is not valid for content_kind '") + to_string(kind) + "'";
    return "unknown key '" + std::string(key) + "'";
}

/// Keys a v1 contract must have, in the order they are checked.
inline std::vector<std::string_view> required_keys(ModelRole role, ContentKind kind) {
    std::vector<std::string_view> keys{"contract_version", "role", "sample_rate", "frame_samples",
                                       "frames_per_step", "lookahead_frames", "content_kind", content_key(kind)};
    if (role == ModelRole::kVocoder) keys.insert(keys.end(), std::begin(kVocoderKeys), std::end(kVocoderKeys));
    keys.insert(keys.end(), {"states", "license"});
    return keys;
}

inline bool get_int(const Json& obj, const char* key, int& out, std::string& error) {
    auto it = obj.find(key);
    if (it == obj.end()) return fail(error, "missing key '" + std::string(key) + "'");
    // Booleans and floats (320.0) are not integers. An integer literal beyond
    // 64 bits parses as a float here (Python says "out of range" instead).
    if (!it->is_number_integer()) return fail(error, key_msg(key, "must be an integer"));
    constexpr int64_t kMin = std::numeric_limits<int32_t>::min(), kMax = std::numeric_limits<int32_t>::max();
    int64_t v = 0;
    if (it->is_number_unsigned()) {
        const auto u = it->get<uint64_t>();
        if (u > static_cast<uint64_t>(kMax)) return fail(error, key_msg(key, "is out of range"));
        v = static_cast<int64_t>(u);
    } else {
        v = it->get<int64_t>();
        if (v < kMin || v > kMax) return fail(error, key_msg(key, "is out of range"));
    }
    out = static_cast<int>(v);
    return true;
}

inline bool get_str(const Json& obj, const char* key, std::string& out, std::string& error) {
    auto it = obj.find(key);
    if (it == obj.end()) return fail(error, "missing key '" + std::string(key) + "'");
    if (!it->is_string()) return fail(error, key_msg(key, "must be a string"));
    out = it->get<std::string>();
    return true;
}

inline bool get_str_list(const Json& obj, const char* key, std::vector<std::string>& out, std::string& error) {
    auto it = obj.find(key);
    if (it == obj.end()) return fail(error, "missing key '" + std::string(key) + "'");
    if (!it->is_array() || !std::all_of(it->begin(), it->end(), [](const Json& v) { return v.is_string(); }))
        return fail(error, key_msg(key, "must be a list of strings"));
    out.clear();
    for (const auto& v : *it) out.push_back(v.get<std::string>());
    return true;
}

inline bool positive(const char* key, int v, std::string& error) {
    if (v > 0) return true;
    return fail(error, std::string(key) + " must be > 0 (got " + std::to_string(v) + ")");
}

inline bool in_range(const char* key, int v, int lo, int hi, std::string& error) {
    if (v >= lo && v <= hi) return true;
    return fail(error, std::string(key) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi) +
                           " (got " + std::to_string(v) + ")");
}

inline bool contains(const std::vector<std::string>& v, std::string_view s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

inline const TensorSpec* find_spec(const std::vector<TensorSpec>& specs, std::string_view name) {
    for (const auto& t : specs)
        if (t.name == name) return &t;
    return nullptr;
}

inline bool check_side(const ModelContract& c, const char* kind, const std::vector<TensorSpec>& expected,
                       const std::vector<TensorSpec>& actual, std::string_view prefix, std::string& error) {
    const std::string graph_kind = std::string("graph ") + kind + " '";
    for (const auto& e : expected) {
        const TensorSpec* a = find_spec(actual, e.name);
        if (!a) return fail(error, "missing " + graph_kind + e.name + "'");
        if (a->dtype != e.dtype)
            return fail(error, graph_kind + e.name + "' has dtype " + to_string(a->dtype) + ", contract expects " +
                                   to_string(e.dtype));
        if (!a->has_shape || a->shape != e.shape)
            return fail(error, graph_kind + e.name + "' has shape " + format_shape(*a) + ", contract expects " +
                                   format_shape(e.shape));
    }
    for (const auto& a : actual) {
        if (find_spec(expected, a.name)) continue;
        if (std::string_view(a.name).starts_with(prefix)) {
            if (contains(c.states, std::string_view(a.name).substr(prefix.size()))) continue;
            return fail(error, graph_kind + a.name + "' is not a state listed in the contract");
        }
        return fail(error, "unexpected " + graph_kind + a.name + "'");
    }
    return true;
}

} // namespace contract_detail

/// Semantic checks on one contract (parse_contract() runs them too).
inline bool validate_contract(const ModelContract& c, std::string& error) {
    using namespace contract_detail;
    if (c.contract_version != kContractVersion) return fail(error, version_message(c.contract_version));
    if (c.content_kind == ContentKind::kDiscrete && c.content_dim != 0)
        return fail(error, misplaced_key_message("content_dim", c.role, c.content_kind));
    if (c.content_kind == ContentKind::kSoft && c.vocab != 0)
        return fail(error, misplaced_key_message("vocab", c.role, c.content_kind));
    if (c.role == ModelRole::kEncoder) {
        const bool set[] = {c.cond_frame_samples != 0, !c.cond_inputs.empty(), c.spk_emb_dim != 0,
                            c.overlap_samples != 0};
        for (size_t i = 0; i < std::size(set); ++i)
            if (set[i]) return fail(error, misplaced_key_message(kVocoderKeys[i], c.role, c.content_kind));
    }

    if (!in_range("sample_rate", c.sample_rate, kMinSampleRate, kMaxSampleRate, error) ||
        !in_range("frame_samples", c.frame_samples, 1, kMaxSampleRate, error))
        return false;
    if (c.frames_per_step != 1 && c.frames_per_step != 2 && c.frames_per_step != 4)
        return fail(error, "frames_per_step must be 1, 2 or 4 (got " + std::to_string(c.frames_per_step) + ")");
    if (c.hop() > c.sample_rate)
        return fail(error, "hop (" + std::to_string(c.hop()) + " samples) is longer than one second at sample_rate " +
                               std::to_string(c.sample_rate));
    if (!in_range("lookahead_frames", c.lookahead_frames, 0, kMaxLookaheadFrames, error)) return false;
    if (c.content_kind == ContentKind::kDiscrete ? !in_range("vocab", c.vocab, 1, kMaxVocab, error)
                                                 : !in_range("content_dim", c.content_dim, 1, kMaxContentDim, error))
        return false;

    if (c.role == ModelRole::kVocoder) {
        if (!positive("cond_frame_samples", c.cond_frame_samples, error)) return false;
        if (c.frame_samples % c.cond_frame_samples != 0)
            return fail(error, "frame_samples (" + std::to_string(c.frame_samples) +
                                   ") is not a multiple of cond_frame_samples (" +
                                   std::to_string(c.cond_frame_samples) + ")");
        std::vector<std::string> seen;
        for (const auto& name : c.cond_inputs) {
            if (std::find(std::begin(kCondInputNames), std::end(kCondInputNames), name) == std::end(kCondInputNames))
                return fail(error, "unknown cond input '" + name + "'");
            if (contains(seen, name)) return fail(error, "duplicate cond input '" + name + "'");
            seen.push_back(name);
        }
        if (!in_range("spk_emb_dim", c.spk_emb_dim, 0, kMaxSpkEmbDim, error)) return false;
        const bool has_spk = contains(seen, "spk_emb");
        if (has_spk && c.spk_emb_dim == 0)
            return fail(error, "spk_emb_dim must be > 0 when cond_inputs lists spk_emb");
        if (!has_spk && c.spk_emb_dim != 0)
            return fail(error, "spk_emb_dim must be 0 when cond_inputs does not list spk_emb");
        if (c.overlap_samples < 0)
            return fail(error, "overlap_samples must be >= 0 (got " + std::to_string(c.overlap_samples) + ")");
        if (c.overlap_samples > c.hop())
            return fail(error, "overlap_samples (" + std::to_string(c.overlap_samples) + ") exceeds hop (" +
                                   std::to_string(c.hop()) + ")");
    }

    std::vector<std::string> seen;
    for (const auto& name : c.states) {
        if (name.empty()) return fail(error, "state names must not be empty");
        if (std::string_view(name).starts_with(kStateIn) || std::string_view(name).starts_with(kStateOut))
            return fail(error, "state '" + name + "' must be a base name, without the state_in./state_out. prefix");
        if (contains(seen, name)) return fail(error, "duplicate state '" + name + "'");
        seen.push_back(name);
    }
    if (c.license.empty()) return fail(error, "license must not be empty");
    return true;
}

/// Parse and validate a contract string (the ONNX metadata value).
inline bool parse_contract(std::string_view text, ModelContract& out, std::string& error) {
    using namespace contract_detail;
    const Json obj = Json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
    if (obj.is_discarded()) return fail(error, "contract is not valid JSON");
    if (!obj.is_object()) return fail(error, "contract must be a JSON object");

    ModelContract c;
    if (!get_int(obj, "contract_version", c.contract_version, error)) return false;
    if (c.contract_version != kContractVersion) return fail(error, version_message(c.contract_version));
    std::string role, kind;
    if (!get_str(obj, "role", role, error)) return false;
    if (role == "encoder")      c.role = ModelRole::kEncoder;
    else if (role == "vocoder") c.role = ModelRole::kVocoder;
    else return fail(error, "unknown role '" + role + "' (expected encoder or vocoder)");
    if (!get_str(obj, "content_kind", kind, error)) return false;
    if (kind == "discrete")  c.content_kind = ContentKind::kDiscrete;
    else if (kind == "soft") c.content_kind = ContentKind::kSoft;
    else return fail(error, "unknown content_kind '" + kind + "' (expected discrete or soft)");

    // nlohmann::json objects iterate in sorted key order, like Python's sorted().
    const auto allowed = required_keys(c.role, c.content_kind);
    for (auto it = obj.begin(); it != obj.end(); ++it)
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end())
            return fail(error, misplaced_key_message(it.key(), c.role, c.content_kind));
    for (const auto key : allowed)
        if (!obj.contains(key)) return fail(error, "missing key '" + std::string(key) + "'");

    const bool vocoder = c.role == ModelRole::kVocoder;
    if (!get_int(obj, "sample_rate", c.sample_rate, error) ||
        !get_int(obj, "frame_samples", c.frame_samples, error) ||
        !get_int(obj, "frames_per_step", c.frames_per_step, error) ||
        !get_int(obj, "lookahead_frames", c.lookahead_frames, error))
        return false;
    if (c.content_kind == ContentKind::kDiscrete ? !get_int(obj, "vocab", c.vocab, error)
                                                 : !get_int(obj, "content_dim", c.content_dim, error))
        return false;
    if (vocoder && (!get_int(obj, "cond_frame_samples", c.cond_frame_samples, error) ||
                    !get_str_list(obj, "cond_inputs", c.cond_inputs, error) ||
                    !get_int(obj, "spk_emb_dim", c.spk_emb_dim, error) ||
                    !get_int(obj, "overlap_samples", c.overlap_samples, error)))
        return false;
    if (!get_str_list(obj, "states", c.states, error) || !get_str(obj, "license", c.license, error)) return false;

    if (!validate_contract(c, error)) return false;
    out = std::move(c);
    return true;
}

/// An encoder + vocoder that can run back to back (one duplex rate, same
/// frame geometry and content).
inline bool validate_pair(const ModelContract& enc, const ModelContract& voc, std::string& error) {
    using contract_detail::fail;
    if (enc.role != ModelRole::kEncoder)
        return fail(error, std::string("pair: first model must be an encoder (got ") + to_string(enc.role) + ")");
    if (voc.role != ModelRole::kVocoder)
        return fail(error, std::string("pair: second model must be a vocoder (got ") + to_string(voc.role) + ")");
    auto mismatch = [&](const char* key, const std::string& a, const std::string& b) {
        return fail(error, std::string(key) + " mismatch: encoder " + a + ", vocoder " + b);
    };
    if (enc.sample_rate != voc.sample_rate)
        return mismatch("sample_rate", std::to_string(enc.sample_rate), std::to_string(voc.sample_rate));
    if (enc.frame_samples != voc.frame_samples)
        return mismatch("frame_samples", std::to_string(enc.frame_samples), std::to_string(voc.frame_samples));
    if (enc.frames_per_step != voc.frames_per_step)
        return mismatch("frames_per_step", std::to_string(enc.frames_per_step), std::to_string(voc.frames_per_step));
    if (enc.content_kind != voc.content_kind)
        return mismatch("content_kind", to_string(enc.content_kind), to_string(voc.content_kind));
    if (enc.content_kind == ContentKind::kDiscrete && enc.vocab != voc.vocab)
        return mismatch("vocab", std::to_string(enc.vocab), std::to_string(voc.vocab));
    if (enc.content_kind == ContentKind::kSoft && enc.content_dim != voc.content_dim)
        return mismatch("content_dim", std::to_string(enc.content_dim), std::to_string(voc.content_dim));
    return true;
}

/// Samples from an input sample to the output that describes it, for a valid
/// pair: hop (framing) + (L_enc + L_voc) x frame_samples + overlap_samples.
/// Device I/O and compute come on top (master latency budget).
inline int64_t algorithmic_latency_samples(const ModelContract& enc, const ModelContract& voc) {
    return enc.hop() + (int64_t{enc.lookahead_frames} + voc.lookahead_frames) * enc.frame_samples +
           voc.overlap_samples;
}

/// Cross-check a graph's inputs/outputs against the contract: names, dtypes,
/// static shapes, and that every state is a same-shape, same-dtype pair.
inline bool validate_io(const ModelContract& c, const std::vector<TensorSpec>& inputs,
                        const std::vector<TensorSpec>& outputs, std::string& error) {
    using namespace contract_detail;
    if (!check_side(c, "input", c.expected_inputs(), inputs, kStateIn, error) ||
        !check_side(c, "output", c.expected_outputs(), outputs, kStateOut, error))
        return false;
    for (const auto& x : c.states) {
        const TensorSpec* i = find_spec(inputs, std::string(kStateIn) + x);
        const TensorSpec* o = find_spec(outputs, std::string(kStateOut) + x);
        if (!i && !o)
            return fail(error, "missing state '" + x + "': the graph has neither state_in." + x + " nor state_out." + x);
        if (!o) return fail(error, "unpaired state '" + x + "': state_in." + x + " has no state_out." + x);
        if (!i) return fail(error, "unpaired state '" + x + "': state_out." + x + " has no state_in." + x);
        if (i->dtype == TensorDtype::kOther)
            return fail(error, "state '" + x + "' has unsupported dtype " + to_string(i->dtype));
        if (i->dtype != o->dtype)
            return fail(error, "state '" + x + "' changes dtype: state_in." + x + " is " + to_string(i->dtype) +
                                   ", state_out." + x + " is " + to_string(o->dtype));
        // The loader sizes every state buffer once, from the graph.
        if (!i->has_shape) return fail(error, "state '" + x + "' has no shape in the graph");
        if (std::any_of(i->shape.begin(), i->shape.end(), [](int64_t d) { return d < 0; }))
            return fail(error, "state '" + x + "' has a dynamic shape " + format_shape(*i));
        if (std::any_of(i->shape.begin(), i->shape.end(), [](int64_t d) { return d == 0; }))
            return fail(error, "state '" + x + "' has an empty dimension " + format_shape(*i));
        if (!o->has_shape || i->shape != o->shape)
            return fail(error, "state '" + x + "' changes shape: state_in." + x + " is " + format_shape(*i) +
                                   ", state_out." + x + " is " + format_shape(*o));
    }
    return true;
}

/// What a loader does with a model: parse its metadata (nullptr = the key is
/// absent), then check the graph against it.
inline bool check_model(const std::string* contract_json, const std::vector<TensorSpec>& inputs,
                        const std::vector<TensorSpec>& outputs, ModelContract& out, std::string& error) {
    if (!contract_json) return contract_detail::fail(error, missing_contract_message());
    ModelContract c;
    if (!parse_contract(*contract_json, c, error) || !validate_io(c, inputs, outputs, error)) return false;
    out = std::move(c);
    return true;
}

} // namespace fluxvoice
