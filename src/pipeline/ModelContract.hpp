// ModelContract — what the engine needs to know about an ONNX model
// (Milestone 2 / Stage 02 contract v1, Milestone 3 / Stage 02 contract v2).
//
// Every encoder, translator_enc, translator_dec and vocoder carries one JSON
// string in its ONNX metadata_props["fluxvoice.contract"]: sample rate, frame
// geometry, lookahead, content kind, conditioning inputs, ring-KV states,
// language tags and licence. The ORT loader (Stage 07) reads that string,
// hands it to parse_contract(), then checks the session's real inputs/outputs
// with validate_io() — check_model() does both.
//
// No ONNX Runtime here, so the ORT-less build compiles and tests it. Nothing
// throws: functions return false and leave a message in `error`.
//
// python_training/fvml/contract.py implements the same rules in the same order
// with the same messages, and tests/data/contracts/ holds the cases both run.
// Change a rule or a message in both files at once.
//
// Specs:
//   * Contract v1: Documents/Milestone 2/02_model_contract_fixtures.md
//   * Contract v2: Documents/Milestone 3/02_translator_contract_v2_fixtures.md
#pragma once

#include "pipeline/KVStateCache.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fluxvoice {

inline constexpr int         kContractVersion     = 1;
inline constexpr int         kContractVersionV2   = 2;
inline constexpr const char* kContractMetadataKey = "fluxvoice.contract";

// Physical bounds. With these, every derived size (hop, chunk_samples,
// cond_frames, output_samples, algorithmic/onset latency) fits int32, and no
// contract can make the loader preallocate gigabytes.
inline constexpr int kMinSampleRate      = 8000;
inline constexpr int kMaxSampleRate      = 192000;
inline constexpr int kMaxLookaheadFrames = 64;
inline constexpr int kMaxVocab           = 1 << 20;
inline constexpr int kMaxContentDim      = 1 << 16;
inline constexpr int kMaxLatentDim       = 1 << 16;
inline constexpr int kMaxSpkEmbDim       = 1 << 12;
inline constexpr int kMaxChunkFrames     = 64;
inline constexpr int kMaxRingSteps       = 4096;
inline constexpr int kMaxLangId          = 65535;
inline constexpr int kMaxSpecialUnit     = kMaxVocab + 16;
inline constexpr int kMinWaitK           = 1;
inline constexpr int kMaxWaitK           = 10;

enum class ModelRole   { kEncoder, kTranslatorEnc, kTranslatorDec, kVocoder };
enum class ContentKind { kDiscrete, kSoft };

inline const char* to_string(ModelRole r) noexcept {
    switch (r) {
    case ModelRole::kEncoder:       return "encoder";
    case ModelRole::kTranslatorEnc: return "translator_enc";
    case ModelRole::kTranslatorDec: return "translator_dec";
    case ModelRole::kVocoder:       return "vocoder";
    }
    return "unknown";
}

inline const char* to_string(ContentKind k) noexcept {
    return k == ContentKind::kDiscrete ? "discrete" : "soft";
}

inline bool is_translator_role(ModelRole r) noexcept {
    return r == ModelRole::kTranslatorEnc || r == ModelRole::kTranslatorDec;
}

/// Special token IDs emitted/consumed by `translator_dec` (v2). Regular vocoder
/// units lie in `0 .. vocab - 1`, so every special unit ID must be `>= vocab`.
struct SpecialUnits {
    int bos = 0;
    int eos = 0;
    int pad = 0;

    bool empty() const noexcept { return bos == 0 && eos == 0 && pad == 0; }
    bool operator==(const SpecialUnits&) const = default;
};

/// One graph input or output. `shape` uses -1 for a dynamic dimension;
/// `has_shape` is false when the graph gives no shape at all (unknown rank —
/// an empty `shape` with has_shape = true is a scalar).
struct TensorSpec {
    std::string               name;
    TensorDtype               dtype = TensorDtype::kF32;
    std::vector<std::int64_t> shape;
    bool                      has_shape = true;
};

/// A parsed, validated v1 or v2 contract. Fields of other roles / content kinds
/// stay at their defaults.
struct ModelContract {
    int                        contract_version = kContractVersion;
    ModelRole                  role = ModelRole::kEncoder;
    int                        sample_rate = 0;
    int                        frame_samples = 0;       // samples per model frame
    int                        frames_per_step = 0;     // 1, 2 or 4 frames per Run (enc/voc)
    int                        lookahead_frames = 0;    // output at step t describes frame t - L (enc/voc)
    ContentKind                content_kind = ContentKind::kDiscrete;
    int                        vocab = 0;               // discrete enc/voc + translator_dec
    int                        content_dim = 0;         // soft enc/voc only
    int                        cond_frame_samples = 0;  // vocoder only
    std::vector<std::string>   cond_inputs;             // vocoder only
    int                        spk_emb_dim = 0;         // vocoder only
    int                        overlap_samples = 0;     // vocoder only; 0 for a stateful causal vocoder
    std::vector<std::string>   states;                  // base names X of state_in.X / state_out.X
    std::string                license;

    // Contract v2 fields (Mode B simultaneous S2UT):
    int                        latent_dim = 0;          // v2 encoder + translator_enc
    int                        chunk_frames = 0;        // translator_enc + translator_dec
    int                        max_src_steps = 0;       // translator_enc + translator_dec
    int                        max_tgt_steps = 0;       // translator_dec only
    std::vector<std::string>   shared_mem_states;       // translator_enc + translator_dec
    std::map<std::string, int> langs;                   // translator_enc + translator_dec
    SpecialUnits               special_units{};         // translator_dec only
    bool                       has_readiness_output = false;  // translator_enc only
    bool                       has_prosody_output = false;    // translator_dec only

    /// Samples consumed (encoder) or produced (vocoder) per Run.
    std::int64_t hop() const noexcept { return std::int64_t{frames_per_step} * frame_samples; }
    /// Samples per translation chunk (translator_enc / translator_dec).
    std::int64_t chunk_samples() const noexcept { return std::int64_t{chunk_frames} * frame_samples; }
    std::int64_t cond_frames() const noexcept { return cond_frame_samples > 0 ? hop() / cond_frame_samples : 0; }
    /// Vocoder `audio` output length: hop plus the OLA tail.
    std::int64_t output_samples() const noexcept { return hop() + overlap_samples; }

    /// Encoder output = vocoder input: `units` (i64 ids) or `content` (f32).
    const char* content_name() const noexcept {
        return content_kind == ContentKind::kDiscrete ? "units" : "content";
    }

    TensorSpec content_spec() const {
        if (content_kind == ContentKind::kDiscrete) return {"units", TensorDtype::kI64, {1, frames_per_step}};
        return {"content", TensorDtype::kF32, {1, frames_per_step, content_dim}};
    }

    /// Non-state graph inputs, in contract order (states and translator_dec
    /// shared_mem_states have graph-defined shapes checked in validate_io).
    std::vector<TensorSpec> expected_inputs() const {
        if (role == ModelRole::kEncoder) return {{"audio", TensorDtype::kF32, {1, hop()}}};
        if (role == ModelRole::kTranslatorEnc) {
            return {
                {"latents", TensorDtype::kF32, {1, chunk_frames, latent_dim}},
                {"src_lang", TensorDtype::kI64, {1}},
            };
        }
        if (role == ModelRole::kTranslatorDec) {
            return {
                {"prev_units", TensorDtype::kI64, {1, chunk_frames}},
                {"tgt_lang", TensorDtype::kI64, {1}},
                {"prosody_ctx", TensorDtype::kF32, {1, 2}},
                {"src_pos", TensorDtype::kI64, {1}},
            };
        }
        std::vector<TensorSpec> specs{content_spec()};
        for (const auto& name : cond_inputs)
            specs.push_back({name, TensorDtype::kF32, {1, name == "spk_emb" ? std::int64_t{spk_emb_dim} : cond_frames()}});
        return specs;
    }

    std::vector<TensorSpec> expected_outputs() const {
        if (role == ModelRole::kEncoder) {
            std::vector<TensorSpec> outs{content_spec()};
            if (contract_version == kContractVersionV2)
                outs.push_back({"latents", TensorDtype::kF32, {1, frames_per_step, latent_dim}});
            return outs;
        }
        if (role == ModelRole::kTranslatorEnc) {
            if (has_readiness_output) return {{"readiness_logit", TensorDtype::kF32, {1, 1}}};
            return {};
        }
        if (role == ModelRole::kTranslatorDec) {
            std::vector<TensorSpec> outs{{"units", TensorDtype::kI64, {1, chunk_frames}}};
            if (has_prosody_output) outs.push_back({"prosody", TensorDtype::kF32, {1, chunk_frames, 4}});
            outs.push_back({"eos_prob", TensorDtype::kF32, {1, 1}});
            return outs;
        }
        return {{"audio", TensorDtype::kF32, {1, output_samples()}}};
    }
};

inline const char* const kCondInputNames[]  = {"logf0", "voicing", "aperiodicity", "energy", "spk_emb"};
inline const char* const kVocoderKeys[]     = {"cond_frame_samples", "cond_inputs", "spk_emb_dim", "overlap_samples"};
inline const char* const kV2OnlyKeys[]      = {"latent_dim",        "chunk_frames",  "max_src_steps",
                                               "max_tgt_steps",     "shared_mem_states", "langs",
                                               "special_units",     "has_readiness_output", "has_prosody_output"};
inline const char* const kEncVocOnlyKeys[]  = {"frames_per_step", "lookahead_frames", "content_kind", "content_dim"};
inline const char* const kSpecialUnitKeys[] = {"bos", "eos", "pad"};
inline const char* const kDecFixedInputs[]  = {"prev_units", "tgt_lang", "prosody_ctx", "src_pos"};
inline constexpr std::string_view kStateIn  = "state_in.";
inline constexpr std::string_view kStateOut = "state_out.";

/// "[1,320]"; dynamic dimensions print as "?".
inline std::string format_shape(const std::vector<std::int64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
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

inline bool is_supported_version(int v) noexcept {
    return v == kContractVersion || v == kContractVersionV2;
}

inline std::string version_message(int version) {
    return "unsupported contract_version " + std::to_string(version) +
           " (this build reads versions 1 and 2)";
}

inline std::string role_message(std::string_view role, int version) {
    if (version == kContractVersion)
        return "unknown role '" + std::string(role) + "' (expected encoder or vocoder)";
    return "unknown role '" + std::string(role) +
           "' (expected encoder, translator_enc, translator_dec or vocoder)";
}

inline bool is_in_list(std::string_view s, const char* const* first, const char* const* last) {
    return std::find(first, last, s) != last;
}

inline bool is_vocoder_key(std::string_view key) {
    return is_in_list(key, std::begin(kVocoderKeys), std::end(kVocoderKeys));
}

inline bool is_v2_only_key(std::string_view key) {
    return is_in_list(key, std::begin(kV2OnlyKeys), std::end(kV2OnlyKeys));
}

inline bool is_enc_voc_only_key(std::string_view key) {
    return is_in_list(key, std::begin(kEncVocOnlyKeys), std::end(kEncVocOnlyKeys));
}

inline bool is_lang_code(std::string_view s) noexcept {
    if (s.size() < 2 || s.size() > 8) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return (c >= 'a' && c <= 'z') || c == '-'; });
}

inline const char* content_key(ContentKind kind) noexcept {
    return kind == ContentKind::kDiscrete ? "vocab" : "content_dim";
}

inline std::string misplaced_key_message(std::string_view key, int version, ModelRole role, ContentKind kind) {
    if (version == kContractVersion && is_v2_only_key(key))
        return key_msg(key, "requires contract_version 2");
    if (is_vocoder_key(key) && role != ModelRole::kVocoder)
        return key_msg(key, "is not valid for role '") + to_string(role) + "'";
    if (is_v2_only_key(key))
        return key_msg(key, "is not valid for role '") + to_string(role) + "'";
    if (is_translator_role(role) && (is_enc_voc_only_key(key) || key == "vocab"))
        return key_msg(key, "is not valid for role '") + to_string(role) + "'";
    if ((key == "vocab" || key == "content_dim") && key != content_key(kind))
        return key_msg(key, "is not valid for content_kind '") + to_string(kind) + "'";
    return "unknown key '" + std::string(key) + "'";
}

/// Keys a contract must have, in the order they are checked.
inline std::vector<std::string_view> required_keys(ModelRole role, ContentKind kind, int version) {
    if (version == kContractVersionV2 && role == ModelRole::kTranslatorEnc) {
        return {"contract_version",     "role",        "sample_rate", "frame_samples",
                "chunk_frames",         "latent_dim",  "max_src_steps", "shared_mem_states",
                "langs",                "has_readiness_output", "states", "license"};
    }
    if (version == kContractVersionV2 && role == ModelRole::kTranslatorDec) {
        return {"contract_version", "role",          "sample_rate",       "frame_samples",
                "chunk_frames",     "vocab",         "special_units",     "max_src_steps",
                "max_tgt_steps",    "shared_mem_states", "langs",         "has_prosody_output",
                "states",           "license"};
    }
    std::vector<std::string_view> keys{"contract_version", "role", "sample_rate", "frame_samples",
                                       "frames_per_step", "lookahead_frames", "content_kind", content_key(kind)};
    if (version == kContractVersionV2 && role == ModelRole::kEncoder) {
        keys.push_back("latent_dim");
    } else if (role == ModelRole::kVocoder) {
        keys.insert(keys.end(), std::begin(kVocoderKeys), std::end(kVocoderKeys));
    }
    keys.insert(keys.end(), {"states", "license"});
    return keys;
}

inline bool parse_int32_val(const Json& v, std::string_view label, int& out, std::string& error) {
    if (!v.is_number_integer()) return fail(error, std::string(label) + " must be an integer");
    constexpr std::int64_t kMin = std::numeric_limits<std::int32_t>::min();
    constexpr std::int64_t kMax = std::numeric_limits<std::int32_t>::max();
    std::int64_t n = 0;
    if (v.is_number_unsigned()) {
        const auto u = v.get<std::uint64_t>();
        if (u > static_cast<std::uint64_t>(kMax)) return fail(error, std::string(label) + " is out of range");
        n = static_cast<std::int64_t>(u);
    } else {
        n = v.get<std::int64_t>();
        if (n < kMin || n > kMax) return fail(error, std::string(label) + " is out of range");
    }
    out = static_cast<int>(n);
    return true;
}

inline bool get_int(const Json& obj, const char* key, int& out, std::string& error) {
    auto it = obj.find(key);
    if (it == obj.end()) return fail(error, "missing key '" + std::string(key) + "'");
    // Booleans and floats (320.0) are not integers. An integer literal beyond
    // 64 bits parses as a float here (Python says "out of range" instead).
    return parse_int32_val(*it, "key '" + std::string(key) + "'", out, error);
}

inline bool get_bool(const Json& obj, const char* key, bool& out, std::string& error) {
    auto it = obj.find(key);
    if (it == obj.end()) return fail(error, "missing key '" + std::string(key) + "'");
    if (!it->is_boolean()) return fail(error, key_msg(key, "must be a boolean"));
    out = it->get<bool>();
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

inline bool validate_langs_map(const std::map<std::string, int>& langs, std::string& error) {
    if (langs.empty()) return fail(error, "langs must not be empty");
    std::vector<int> seen_ids;
    for (const auto& [code, lid] : langs) {
        if (!is_lang_code(code))
            return fail(error, "langs: invalid language code '" + code + "' (expected lowercase [a-z-]{2,8})");
        if (lid < 0 || lid > kMaxLangId)
            return fail(error, "langs['" + code + "'] must be between 0 and " + std::to_string(kMaxLangId) +
                                   " (got " + std::to_string(lid) + ")");
        if (std::find(seen_ids.begin(), seen_ids.end(), lid) != seen_ids.end())
            return fail(error, "langs: duplicate language id " + std::to_string(lid) + " for '" + code + "'");
        seen_ids.push_back(lid);
    }
    return true;
}

inline bool get_langs(const Json& obj, std::map<std::string, int>& out, std::string& error) {
    auto it = obj.find("langs");
    if (it == obj.end()) return fail(error, "missing key 'langs'");
    if (!it->is_object()) return fail(error, key_msg("langs", "must be an object"));
    if (it->empty()) return fail(error, "langs must not be empty");
    out.clear();
    std::vector<int> seen_ids;
    for (auto m = it->begin(); m != it->end(); ++m) {
        const std::string& code = m.key();
        if (!is_lang_code(code))
            return fail(error, "langs: invalid language code '" + code + "' (expected lowercase [a-z-]{2,8})");
        int lid = 0;
        if (!parse_int32_val(m.value(), "langs['" + code + "']", lid, error)) return false;
        if (lid < 0 || lid > kMaxLangId)
            return fail(error, "langs['" + code + "'] must be between 0 and " + std::to_string(kMaxLangId) +
                                   " (got " + std::to_string(lid) + ")");
        if (std::find(seen_ids.begin(), seen_ids.end(), lid) != seen_ids.end())
            return fail(error, "langs: duplicate language id " + std::to_string(lid) + " for '" + code + "'");
        seen_ids.push_back(lid);
        out[code] = lid;
    }
    return true;
}

inline bool get_special_units(const Json& obj, SpecialUnits& out, std::string& error) {
    auto it = obj.find("special_units");
    if (it == obj.end()) return fail(error, "missing key 'special_units'");
    if (!it->is_object()) return fail(error, key_msg("special_units", "must be an object"));
    for (auto m = it->begin(); m != it->end(); ++m) {
        if (!is_in_list(m.key(), std::begin(kSpecialUnitKeys), std::end(kSpecialUnitKeys)))
            return fail(error, "special_units: unknown key '" + m.key() + "'");
    }
    for (const char* k : kSpecialUnitKeys) {
        if (!it->contains(k)) return fail(error, std::string("special_units: missing key '") + k + "'");
    }
    if (!parse_int32_val((*it)["bos"], "special_units['bos']", out.bos, error) ||
        !parse_int32_val((*it)["eos"], "special_units['eos']", out.eos, error) ||
        !parse_int32_val((*it)["pad"], "special_units['pad']", out.pad, error))
        return false;
    return true;
}

inline bool validate_special_units(const SpecialUnits& su, int vocab, std::string& error) {
    const std::pair<const char*, int> entries[] = {{"bos", su.bos}, {"eos", su.eos}, {"pad", su.pad}};
    for (const auto& [name, val] : entries) {
        if (val < vocab)
            return fail(error, std::string("special_units['") + name + "'] (" + std::to_string(val) +
                                   ") must be >= vocab (" + std::to_string(vocab) + ")");
        if (val > kMaxSpecialUnit)
            return fail(error, std::string("special_units['") + name + "'] (" + std::to_string(val) +
                                   ") exceeds max special unit " + std::to_string(kMaxSpecialUnit));
    }
    if (su.bos == su.eos || su.bos == su.pad || su.eos == su.pad) {
        return fail(error, "special_units values must be distinct (got bos=" + std::to_string(su.bos) +
                               ", eos=" + std::to_string(su.eos) + ", pad=" + std::to_string(su.pad) + ")");
    }
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

inline bool check_shared_mem_tensor(std::string_view name, const TensorSpec& spec, int max_src_steps,
                                    std::string& error) {
    if (spec.dtype != TensorDtype::kF32 && spec.dtype != TensorDtype::kF16)
        return fail(error, "shared_mem_state '" + std::string(name) + "' has unsupported dtype " +
                               std::string(to_string(spec.dtype)) + " (expected f32 or f16)");
    if (!spec.has_shape)
        return fail(error, "shared_mem_state '" + std::string(name) + "' has no shape in the graph");
    if (std::any_of(spec.shape.begin(), spec.shape.end(), [](std::int64_t d) { return d < 0; }))
        return fail(error, "shared_mem_state '" + std::string(name) + "' has a dynamic shape " + format_shape(spec));
    if (std::any_of(spec.shape.begin(), spec.shape.end(), [](std::int64_t d) { return d == 0; }))
        return fail(error, "shared_mem_state '" + std::string(name) + "' has an empty dimension " + format_shape(spec));
    if (std::find(spec.shape.begin(), spec.shape.end(), std::int64_t{max_src_steps}) == spec.shape.end()) {
        return fail(error, "shared_mem_state '" + std::string(name) + "' shape " + format_shape(spec) +
                               " does not contain max_src_steps (" + std::to_string(max_src_steps) + ")");
    }
    return true;
}

inline bool check_side(const ModelContract& c, const char* kind, const std::vector<TensorSpec>& expected,
                       const std::vector<TensorSpec>& actual, std::string_view prefix,
                       const std::vector<std::string>& readonly_mem, std::string& error) {
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
        if (find_spec(expected, a.name) || contains(readonly_mem, a.name)) continue;
        if (std::string_view(a.name).starts_with(prefix)) {
            if (contains(c.states, std::string_view(a.name).substr(prefix.size()))) continue;
            return fail(error, graph_kind + a.name + "' is not a state listed in the contract");
        }
        return fail(error, "unexpected " + graph_kind + a.name + "'");
    }
    return true;
}

inline bool check_misplaced_fields(const ModelContract& c, std::string& error) {
    const int v = c.contract_version;
    if (v == kContractVersion) {
        const std::pair<const char*, bool> v2_set[] = {
            {"latent_dim", c.latent_dim != 0},
            {"chunk_frames", c.chunk_frames != 0},
            {"max_src_steps", c.max_src_steps != 0},
            {"max_tgt_steps", c.max_tgt_steps != 0},
            {"shared_mem_states", !c.shared_mem_states.empty()},
            {"langs", !c.langs.empty()},
            {"special_units", !c.special_units.empty()},
            {"has_readiness_output", c.has_readiness_output},
            {"has_prosody_output", c.has_prosody_output},
        };
        for (const auto& [name, set] : v2_set)
            if (set) return fail(error, misplaced_key_message(name, v, c.role, c.content_kind));
    }
    if (c.role != ModelRole::kVocoder) {
        const bool voc_set[] = {c.cond_frame_samples != 0, !c.cond_inputs.empty(), c.spk_emb_dim != 0,
                                c.overlap_samples != 0};
        for (std::size_t i = 0; i < std::size(voc_set); ++i)
            if (voc_set[i]) return fail(error, misplaced_key_message(kVocoderKeys[i], v, c.role, c.content_kind));
    }
    if (is_translator_role(c.role)) {
        if (c.frames_per_step != 0)
            return fail(error, misplaced_key_message("frames_per_step", v, c.role, ContentKind::kDiscrete));
        if (c.lookahead_frames != 0)
            return fail(error, misplaced_key_message("lookahead_frames", v, c.role, ContentKind::kDiscrete));
        if (c.content_kind != ContentKind::kDiscrete)
            return fail(error, misplaced_key_message("content_kind", v, c.role, ContentKind::kDiscrete));
        if (c.content_dim != 0)
            return fail(error, misplaced_key_message("content_dim", v, c.role, ContentKind::kDiscrete));
        if (c.role == ModelRole::kTranslatorEnc) {
            const std::pair<const char*, bool> dec_set[] = {
                {"vocab", c.vocab != 0},
                {"special_units", !c.special_units.empty()},
                {"max_tgt_steps", c.max_tgt_steps != 0},
                {"has_prosody_output", c.has_prosody_output},
            };
            for (const auto& [name, set] : dec_set)
                if (set) return fail(error, misplaced_key_message(name, v, c.role, ContentKind::kDiscrete));
        } else {
            if (c.latent_dim != 0)
                return fail(error, misplaced_key_message("latent_dim", v, c.role, ContentKind::kDiscrete));
            if (c.has_readiness_output)
                return fail(error, misplaced_key_message("has_readiness_output", v, c.role, ContentKind::kDiscrete));
        }
    } else {
        if (c.content_kind == ContentKind::kDiscrete && c.content_dim != 0)
            return fail(error, misplaced_key_message("content_dim", v, c.role, c.content_kind));
        if (c.content_kind == ContentKind::kSoft && c.vocab != 0)
            return fail(error, misplaced_key_message("vocab", v, c.role, c.content_kind));
        const std::pair<const char*, bool> trans_set[] = {
            {"chunk_frames", c.chunk_frames != 0},
            {"max_src_steps", c.max_src_steps != 0},
            {"max_tgt_steps", c.max_tgt_steps != 0},
            {"shared_mem_states", !c.shared_mem_states.empty()},
            {"langs", !c.langs.empty()},
            {"special_units", !c.special_units.empty()},
            {"has_readiness_output", c.has_readiness_output},
            {"has_prosody_output", c.has_prosody_output},
        };
        for (const auto& [name, set] : trans_set)
            if (set) return fail(error, misplaced_key_message(name, v, c.role, c.content_kind));
        if (c.role == ModelRole::kVocoder && c.latent_dim != 0)
            return fail(error, misplaced_key_message("latent_dim", v, c.role, c.content_kind));
    }
    return true;
}

} // namespace contract_detail

/// Semantic checks on one contract (parse_contract() runs them too).
inline bool validate_contract(const ModelContract& c, std::string& error) {
    using namespace contract_detail;
    if (!is_supported_version(c.contract_version)) return fail(error, version_message(c.contract_version));
    if (c.contract_version == kContractVersion && is_translator_role(c.role))
        return fail(error, role_message(to_string(c.role), c.contract_version));
    if (c.contract_version == kContractVersionV2 && c.role == ModelRole::kEncoder &&
        c.content_kind != ContentKind::kDiscrete) {
        return fail(error, std::string("v2 encoder content_kind must be discrete (got ") + to_string(c.content_kind) +
                               ")");
    }
    if (!check_misplaced_fields(c, error)) return false;

    if (!in_range("sample_rate", c.sample_rate, kMinSampleRate, kMaxSampleRate, error) ||
        !in_range("frame_samples", c.frame_samples, 1, kMaxSampleRate, error))
        return false;

    if (is_translator_role(c.role)) {
        if (!in_range("chunk_frames", c.chunk_frames, 1, kMaxChunkFrames, error)) return false;
        if (c.chunk_samples() > c.sample_rate) {
            return fail(error, "chunk_samples (" + std::to_string(c.chunk_samples()) +
                                   " samples) is longer than one second at sample_rate " +
                                   std::to_string(c.sample_rate));
        }
        if (!in_range("max_src_steps", c.max_src_steps, 1, kMaxRingSteps, error)) return false;
        if (c.shared_mem_states.empty()) return fail(error, "shared_mem_states must not be empty");
        std::vector<std::string> seen_mem;
        for (const auto& m : c.shared_mem_states) {
            if (m.empty()) return fail(error, "shared_mem_states names must not be empty");
            if (std::string_view(m).starts_with(kStateIn) || std::string_view(m).starts_with(kStateOut))
                return fail(error, "shared_mem_state '" + m +
                                       "' must be a base name, without the state_in./state_out. prefix");
            if (contains(seen_mem, m)) return fail(error, "duplicate shared_mem_state '" + m + "'");
            seen_mem.push_back(m);
        }
        if (!validate_langs_map(c.langs, error)) return false;
        if (c.role == ModelRole::kTranslatorEnc) {
            if (!in_range("latent_dim", c.latent_dim, 1, kMaxLatentDim, error)) return false;
        } else {
            if (!in_range("vocab", c.vocab, 1, kMaxVocab, error) ||
                !in_range("max_tgt_steps", c.max_tgt_steps, 1, kMaxRingSteps, error) ||
                !validate_special_units(c.special_units, c.vocab, error))
                return false;
        }
    } else {
        if (c.frames_per_step != 1 && c.frames_per_step != 2 && c.frames_per_step != 4)
            return fail(error, "frames_per_step must be 1, 2 or 4 (got " + std::to_string(c.frames_per_step) + ")");
        if (c.hop() > c.sample_rate)
            return fail(error, "hop (" + std::to_string(c.hop()) +
                                   " samples) is longer than one second at sample_rate " +
                                   std::to_string(c.sample_rate));
        if (!in_range("lookahead_frames", c.lookahead_frames, 0, kMaxLookaheadFrames, error)) return false;
        if (c.content_kind == ContentKind::kDiscrete ? !in_range("vocab", c.vocab, 1, kMaxVocab, error)
                                                     : !in_range("content_dim", c.content_dim, 1, kMaxContentDim, error))
            return false;
        if (c.contract_version == kContractVersionV2 && c.role == ModelRole::kEncoder &&
            !in_range("latent_dim", c.latent_dim, 1, kMaxLatentDim, error))
            return false;
    }

    if (c.role == ModelRole::kVocoder) {
        if (!positive("cond_frame_samples", c.cond_frame_samples, error)) return false;
        if (c.frame_samples % c.cond_frame_samples != 0)
            return fail(error, "frame_samples (" + std::to_string(c.frame_samples) +
                                   ") is not a multiple of cond_frame_samples (" +
                                   std::to_string(c.cond_frame_samples) + ")");
        std::vector<std::string> seen;
        for (const auto& name : c.cond_inputs) {
            if (!is_in_list(name, std::begin(kCondInputNames), std::end(kCondInputNames)))
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

    if (c.role == ModelRole::kTranslatorEnc) {
        if (!contains(seen, "src_pos")) return fail(error, "translator_enc states must include 'src_pos'");
        for (const auto& m : c.shared_mem_states) {
            if (m == "src_pos") return fail(error, "shared_mem_state 'src_pos' conflicts with position counter");
            if (!contains(seen, m))
                return fail(error, "translator_enc states must include shared_mem_state '" + m + "'");
        }
    } else if (c.role == ModelRole::kTranslatorDec) {
        if (!contains(seen, "tgt_pos")) return fail(error, "translator_dec states must include 'tgt_pos'");
        for (const auto& m : c.shared_mem_states) {
            if (is_in_list(m, std::begin(kDecFixedInputs), std::end(kDecFixedInputs)))
                return fail(error, "shared_mem_state '" + m + "' conflicts with graph input");
            if (contains(seen, m))
                return fail(error, "translator_dec shared_mem_state '" + m + "' is read-only and must not be in states");
        }
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
    if (!is_supported_version(c.contract_version)) return fail(error, version_message(c.contract_version));

    std::string role, kind;
    if (!get_str(obj, "role", role, error)) return false;
    if (role == "encoder") c.role = ModelRole::kEncoder;
    else if (role == "vocoder") c.role = ModelRole::kVocoder;
    else if (c.contract_version == kContractVersionV2 && role == "translator_enc") c.role = ModelRole::kTranslatorEnc;
    else if (c.contract_version == kContractVersionV2 && role == "translator_dec") c.role = ModelRole::kTranslatorDec;
    else return fail(error, role_message(role, c.contract_version));

    if (!is_translator_role(c.role)) {
        if (!get_str(obj, "content_kind", kind, error)) return false;
        if (kind == "discrete") c.content_kind = ContentKind::kDiscrete;
        else if (kind == "soft") c.content_kind = ContentKind::kSoft;
        else return fail(error, "unknown content_kind '" + kind + "' (expected discrete or soft)");
        if (c.contract_version == kContractVersionV2 && c.role == ModelRole::kEncoder &&
            c.content_kind != ContentKind::kDiscrete) {
            return fail(error, "v2 encoder content_kind must be discrete (got " + kind + ")");
        }
    }

    // nlohmann::json objects iterate in sorted key order, like Python's sorted().
    const auto allowed = required_keys(c.role, c.content_kind, c.contract_version);
    for (auto it = obj.begin(); it != obj.end(); ++it)
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end())
            return fail(error, misplaced_key_message(it.key(), c.contract_version, c.role, c.content_kind));
    for (const auto key : allowed)
        if (!obj.contains(key)) return fail(error, "missing key '" + std::string(key) + "'");

    if (c.role == ModelRole::kTranslatorEnc) {
        if (!get_int(obj, "sample_rate", c.sample_rate, error) ||
            !get_int(obj, "frame_samples", c.frame_samples, error) ||
            !get_int(obj, "chunk_frames", c.chunk_frames, error) ||
            !get_int(obj, "latent_dim", c.latent_dim, error) ||
            !get_int(obj, "max_src_steps", c.max_src_steps, error) ||
            !get_str_list(obj, "shared_mem_states", c.shared_mem_states, error) ||
            !get_langs(obj, c.langs, error) ||
            !get_bool(obj, "has_readiness_output", c.has_readiness_output, error) ||
            !get_str_list(obj, "states", c.states, error) ||
            !get_str(obj, "license", c.license, error))
            return false;
    } else if (c.role == ModelRole::kTranslatorDec) {
        if (!get_int(obj, "sample_rate", c.sample_rate, error) ||
            !get_int(obj, "frame_samples", c.frame_samples, error) ||
            !get_int(obj, "chunk_frames", c.chunk_frames, error) ||
            !get_int(obj, "vocab", c.vocab, error) ||
            !get_special_units(obj, c.special_units, error) ||
            !get_int(obj, "max_src_steps", c.max_src_steps, error) ||
            !get_int(obj, "max_tgt_steps", c.max_tgt_steps, error) ||
            !get_str_list(obj, "shared_mem_states", c.shared_mem_states, error) ||
            !get_langs(obj, c.langs, error) ||
            !get_bool(obj, "has_prosody_output", c.has_prosody_output, error) ||
            !get_str_list(obj, "states", c.states, error) ||
            !get_str(obj, "license", c.license, error))
            return false;
    } else {
        const bool vocoder = c.role == ModelRole::kVocoder;
        if (!get_int(obj, "sample_rate", c.sample_rate, error) ||
            !get_int(obj, "frame_samples", c.frame_samples, error) ||
            !get_int(obj, "frames_per_step", c.frames_per_step, error) ||
            !get_int(obj, "lookahead_frames", c.lookahead_frames, error))
            return false;
        if (c.content_kind == ContentKind::kDiscrete ? !get_int(obj, "vocab", c.vocab, error)
                                                     : !get_int(obj, "content_dim", c.content_dim, error))
            return false;
        if (c.contract_version == kContractVersionV2 && c.role == ModelRole::kEncoder &&
            !get_int(obj, "latent_dim", c.latent_dim, error))
            return false;
        if (vocoder && (!get_int(obj, "cond_frame_samples", c.cond_frame_samples, error) ||
                        !get_str_list(obj, "cond_inputs", c.cond_inputs, error) ||
                        !get_int(obj, "spk_emb_dim", c.spk_emb_dim, error) ||
                        !get_int(obj, "overlap_samples", c.overlap_samples, error)))
            return false;
        if (!get_str_list(obj, "states", c.states, error) || !get_str(obj, "license", c.license, error))
            return false;
    }

    if (!validate_contract(c, error)) return false;
    out = std::move(c);
    return true;
}

/// An encoder + vocoder that can run back to back (Mode A: one duplex rate,
/// same frame geometry and content).
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
/// Mode A pair: hop (framing) + (L_enc + L_voc) x frame_samples + overlap_samples.
inline std::int64_t algorithmic_latency_samples(const ModelContract& enc, const ModelContract& voc) {
    return enc.hop() + (std::int64_t{enc.lookahead_frames} + voc.lookahead_frames) * enc.frame_samples +
           voc.overlap_samples;
}

/// Validate a Mode B 4-model S2UT bundle (`encoder`, `translator_enc`,
/// `translator_dec`, `vocoder`).
inline bool validate_s2ut_bundle(const ModelContract& enc, const ModelContract& trans_enc,
                                 const ModelContract& trans_dec, const ModelContract& voc,
                                 std::string& error) {
    using contract_detail::fail;
    if (enc.role != ModelRole::kEncoder)
        return fail(error, std::string("s2ut bundle: first model must be an encoder (got ") + to_string(enc.role) + ")");
    if (trans_enc.role != ModelRole::kTranslatorEnc)
        return fail(error, std::string("s2ut bundle: second model must be a translator_enc (got ") +
                               to_string(trans_enc.role) + ")");
    if (trans_dec.role != ModelRole::kTranslatorDec)
        return fail(error, std::string("s2ut bundle: third model must be a translator_dec (got ") +
                               to_string(trans_dec.role) + ")");
    if (voc.role != ModelRole::kVocoder)
        return fail(error, std::string("s2ut bundle: fourth model must be a vocoder (got ") + to_string(voc.role) + ")");
    if (enc.contract_version != kContractVersionV2) {
        return fail(error, "s2ut bundle: encoder must use contract_version 2 (got " +
                               std::to_string(enc.contract_version) + ")");
    }
    if (voc.content_kind != ContentKind::kDiscrete) {
        return fail(error, std::string("s2ut bundle: vocoder content_kind must be discrete (got ") +
                               to_string(voc.content_kind) + ")");
    }
    if (!validate_pair(enc, voc, error)) return false;

    const std::pair<const char*, const ModelContract*> trans_models[] = {
        {"translator_enc", &trans_enc}, {"translator_dec", &trans_dec}};
    for (const auto& [name, m] : trans_models) {
        if (enc.sample_rate != m->sample_rate) {
            return fail(error, "sample_rate mismatch: encoder " + std::to_string(enc.sample_rate) + ", " + name + " " +
                                   std::to_string(m->sample_rate));
        }
        if (enc.frame_samples != m->frame_samples) {
            return fail(error, "frame_samples mismatch: encoder " + std::to_string(enc.frame_samples) + ", " + name +
                                   " " + std::to_string(m->frame_samples));
        }
    }
    if (enc.latent_dim != trans_enc.latent_dim) {
        return fail(error, "latent_dim mismatch: encoder " + std::to_string(enc.latent_dim) + ", translator_enc " +
                               std::to_string(trans_enc.latent_dim));
    }
    if (trans_enc.chunk_frames != trans_dec.chunk_frames) {
        return fail(error, "chunk_frames mismatch: translator_enc " + std::to_string(trans_enc.chunk_frames) +
                               ", translator_dec " + std::to_string(trans_dec.chunk_frames));
    }
    if (trans_enc.chunk_frames % enc.frames_per_step != 0) {
        return fail(error, "chunk_frames (" + std::to_string(trans_enc.chunk_frames) +
                               ") is not a multiple of encoder frames_per_step (" +
                               std::to_string(enc.frames_per_step) + ")");
    }
    if (trans_enc.max_src_steps != trans_dec.max_src_steps) {
        return fail(error, "max_src_steps mismatch: translator_enc " + std::to_string(trans_enc.max_src_steps) +
                               ", translator_dec " + std::to_string(trans_dec.max_src_steps));
    }
    if (trans_enc.shared_mem_states != trans_dec.shared_mem_states)
        return fail(error, "shared_mem_states mismatch between translator_enc and translator_dec");
    if (trans_dec.vocab != voc.vocab) {
        return fail(error, "vocab mismatch: translator_dec " + std::to_string(trans_dec.vocab) + ", vocoder " +
                               std::to_string(voc.vocab));
    }
    if (trans_enc.langs != trans_dec.langs)
        return fail(error, "langs mismatch between translator_enc and translator_dec");
    return true;
}

/// Cross-check that `translator_enc`'s `state_out.mem_*` outputs match
/// `translator_dec`'s read-only `mem_*` inputs in dtype and shape.
inline bool validate_s2ut_shared_io(const ModelContract& trans_enc, const std::vector<TensorSpec>& enc_outputs,
                                    const ModelContract& trans_dec, const std::vector<TensorSpec>& dec_inputs,
                                    std::string& error) {
    using namespace contract_detail;
    (void)trans_dec;
    for (const auto& m : trans_enc.shared_mem_states) {
        const TensorSpec* o = find_spec(enc_outputs, std::string(kStateOut) + m);
        const TensorSpec* i = find_spec(dec_inputs, m);
        if (!o) return fail(error, "missing graph output '" + std::string(kStateOut) + m + "'");
        if (!i) return fail(error, "missing graph input '" + m + "'");
        if (o->dtype != i->dtype) {
            return fail(error, "shared_mem_state '" + m + "' dtype mismatch: translator_enc " + to_string(o->dtype) +
                                   ", translator_dec " + to_string(i->dtype));
        }
        if (!o->has_shape || !i->has_shape || o->shape != i->shape) {
            return fail(error, "shared_mem_state '" + m + "' shape mismatch: translator_enc " + format_shape(*o) +
                                   ", translator_dec " + format_shape(*i));
        }
    }
    return true;
}

/// Mode B algorithmic onset latency in samples (Milestone 3 / Stage 02):
/// wait_k * chunk_frames * frame_samples + (L_enc + L_voc) * frame_samples + overlap_samples.
inline std::int64_t s2ut_onset_latency_samples(const ModelContract& enc, const ModelContract& trans_enc,
                                               const ModelContract& trans_dec, const ModelContract& voc,
                                               int wait_k) noexcept {
    (void)trans_dec;
    return std::int64_t{wait_k} * trans_enc.chunk_frames * enc.frame_samples +
           (std::int64_t{enc.lookahead_frames} + voc.lookahead_frames) * enc.frame_samples +
           voc.overlap_samples;
}

/// Validate a bundle `langs.json` document (schema 1) and optionally check
/// that its language-code -> ID mapping matches a translator contract.
inline bool validate_langs_json(std::string_view text, const ModelContract* translator, std::string& error) {
    using namespace contract_detail;
    const Json doc = Json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded() || !doc.is_object()) return fail(error, "langs.json must be a JSON object");
    auto schema_it = doc.find("schema");
    int schema = 0;
    if (schema_it == doc.end() || !parse_int32_val(*schema_it, "schema", schema, error) || schema != 1)
        return fail(error, "unsupported langs.json schema (expected the integer 1)");
    for (auto it = doc.begin(); it != doc.end(); ++it) {
        if (it.key() != "schema" && it.key() != "langs" && it.key() != "default_wait_k")
            return fail(error, "langs.json: unknown key '" + it.key() + "'");
    }
    auto langs_it = doc.find("langs");
    if (langs_it == doc.end() || !langs_it->is_array() || langs_it->empty())
        return fail(error, "langs.json: 'langs' must be a non-empty list");

    std::map<std::string, int> codes;
    std::vector<int> ids;
    for (std::size_t idx = 0; idx < langs_it->size(); ++idx) {
        const Json& entry = (*langs_it)[idx];
        if (!entry.is_object()) return fail(error, "langs.json: every language must be an object");
        for (auto it = entry.begin(); it != entry.end(); ++it) {
            if (it.key() != "code" && it.key() != "id" && it.key() != "name")
                return fail(error, "langs[" + std::to_string(idx) + "]: unknown key '" + it.key() + "'");
        }
        auto code_it = entry.find("code");
        if (code_it == entry.end() || !code_it->is_string() || !is_lang_code(code_it->get_ref<const std::string&>()))
            return fail(error, "langs.json: 'code' must be a lowercase language tag");
        const std::string code = code_it->get<std::string>();
        auto id_it = entry.find("id");
        int lid = -1;
        if (id_it == entry.end() || !parse_int32_val(*id_it, "id", lid, error) || lid < 0 || lid > kMaxLangId)
            return fail(error, "langs.json: 'id' must be an integer in [0, " + std::to_string(kMaxLangId) + "] for '" +
                                   code + "'");
        auto name_it = entry.find("name");
        if (name_it == entry.end() || !name_it->is_string() || name_it->get_ref<const std::string&>().empty())
            return fail(error, "langs.json: 'name' must be a non-empty string for '" + code + "'");
        if (codes.find(code) != codes.end()) return fail(error, "langs.json: duplicate code '" + code + "'");
        if (std::find(ids.begin(), ids.end(), lid) != ids.end())
            return fail(error, "langs.json: duplicate id for '" + code + "'");
        codes[code] = lid;
        ids.push_back(lid);
    }

    auto dk_it = doc.find("default_wait_k");
    if (dk_it != doc.end()) {
        if (!dk_it->is_object()) return fail(error, "langs.json: 'default_wait_k' must be an object");
        for (auto it = dk_it->begin(); it != dk_it->end(); ++it) {
            const std::string& pair_key = it.key();
            const std::size_t arrow = pair_key.find("->");
            const bool bad_arrow = (arrow == std::string::npos) || (pair_key.find("->", arrow + 2) != std::string::npos);
            const std::string src = bad_arrow ? "" : pair_key.substr(0, arrow);
            const std::string tgt = bad_arrow ? "" : pair_key.substr(arrow + 2);
            if (bad_arrow || codes.find(src) == codes.end() || codes.find(tgt) == codes.end() || src == tgt)
                return fail(error, "langs.json: invalid language pair '" + pair_key + "' in default_wait_k");
            int kv = 0;
            if (!parse_int32_val(it.value(), "default_wait_k", kv, error) || kv < kMinWaitK || kv > kMaxWaitK) {
                return fail(error, "langs.json: default_wait_k['" + pair_key + "'] must be an integer in [" +
                                       std::to_string(kMinWaitK) + ", " + std::to_string(kMaxWaitK) + "]");
            }
        }
    }
    if (translator && translator->langs != codes)
        return fail(error, "langs.json does not match translator contract langs");
    return true;
}

/// Cross-check a graph's inputs/outputs against the contract: names, dtypes,
/// static shapes, shared cross-attention memory inputs, and state pairs.
inline bool validate_io(const ModelContract& c, const std::vector<TensorSpec>& inputs,
                        const std::vector<TensorSpec>& outputs, std::string& error) {
    using namespace contract_detail;
    const std::vector<std::string> empty_mem;
    const auto& readonly_mem = (c.role == ModelRole::kTranslatorDec) ? c.shared_mem_states : empty_mem;
    if (!check_side(c, "input", c.expected_inputs(), inputs, kStateIn, readonly_mem, error) ||
        !check_side(c, "output", c.expected_outputs(), outputs, kStateOut, empty_mem, error))
        return false;

    if (c.role == ModelRole::kTranslatorDec) {
        for (const auto& m : c.shared_mem_states) {
            const TensorSpec* t = find_spec(inputs, m);
            if (!t) return fail(error, "missing graph input '" + m + "'");
            if (!check_shared_mem_tensor(m, *t, c.max_src_steps, error)) return false;
        }
    }

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
        if (std::any_of(i->shape.begin(), i->shape.end(), [](std::int64_t d) { return d < 0; }))
            return fail(error, "state '" + x + "' has a dynamic shape " + format_shape(*i));
        if (std::any_of(i->shape.begin(), i->shape.end(), [](std::int64_t d) { return d == 0; }))
            return fail(error, "state '" + x + "' has an empty dimension " + format_shape(*i));
        if (!o->has_shape || i->shape != o->shape)
            return fail(error, "state '" + x + "' changes shape: state_in." + x + " is " + format_shape(*i) +
                                   ", state_out." + x + " is " + format_shape(*o));
        if ((c.role == ModelRole::kTranslatorEnc && x == "src_pos") ||
            (c.role == ModelRole::kTranslatorDec && x == "tgt_pos")) {
            if (i->dtype != TensorDtype::kI64)
                return fail(error, "position state '" + x + "' has dtype " + to_string(i->dtype) +
                                       ", contract expects i64");
            if (i->shape != std::vector<std::int64_t>{1})
                return fail(error, "position state '" + x + "' has shape " + format_shape(*i) +
                                       ", contract expects [1]");
        }
        if (c.role == ModelRole::kTranslatorEnc && contains(c.shared_mem_states, x)) {
            if (!check_shared_mem_tensor(x, *i, c.max_src_steps, error)) return false;
        }
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
