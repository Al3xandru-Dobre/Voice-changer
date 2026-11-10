// Presets — JSON preset loader/saver and the descriptor-table dump.
#include "voice/Presets.hpp"

#include "voice/VoiceParams.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fluxvoice {

namespace {

using nlohmann::json;
namespace fs = std::filesystem;

constexpr std::string_view kTopLevelKeys[] = {
    "schema", "name", "description", "voice_a", "voice_b", "params", "src_lang", "tgt_lang"};

bool is_known_top_level(std::string_view key) {
    for (std::string_view k : kTopLevelKeys)
        if (k == key) return true;
    return false;
}

std::string fmt_float(float v) {
    // Shortest round-trip-ish text without trailing zeros, for messages.
    std::ostringstream o;
    o << v;
    return o.str();
}

/// Optional non-empty string field; a wrong type is a warning, not an error.
std::optional<std::string> read_optional_string(const json& obj, const char* key,
                                                std::vector<std::string>& warnings) {
    auto it = obj.find(key);
    if (it == obj.end()) return std::nullopt;
    if (!it->is_string() || it->get_ref<const std::string&>().empty()) {
        warnings.push_back(std::string("'") + key + "' must be a non-empty string; ignored");
        return std::nullopt;
    }
    return it->get<std::string>();
}

} // namespace

PresetLoadResult parse_preset(std::string_view json_text) {
    PresetLoadResult r;
    const json doc = json::parse(json_text, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded()) {
        r.error = "preset is not valid JSON";
        return r;
    }
    if (!doc.is_object()) {
        r.error = "preset must be a JSON object";
        return r;
    }

    // Schema: the integer 1 only (nlohmann keeps bool/int/float distinct).
    auto schema = doc.find("schema");
    if (schema == doc.end()) {
        r.error = "preset has no 'schema' key";
        return r;
    }
    if (!schema->is_number_integer() || schema->get<long long>() != 1) {
        r.error = "unsupported preset schema " + schema->dump() + " (expected 1)";
        return r;
    }

    auto name = doc.find("name");
    if (name == doc.end() || !name->is_string() || name->get_ref<const std::string&>().empty()) {
        r.error = "preset needs a non-empty string 'name'";
        return r;
    }

    Preset p;
    p.name = name->get<std::string>();

    for (auto it = doc.begin(); it != doc.end(); ++it)
        if (!is_known_top_level(it.key())) r.warnings.push_back("unknown key '" + it.key() + "' ignored");

    if (auto d = read_optional_string(doc, "description", r.warnings)) p.description = *d;
    p.voice_a  = read_optional_string(doc, "voice_a", r.warnings);
    p.voice_b  = read_optional_string(doc, "voice_b", r.warnings);
    p.src_lang = read_optional_string(doc, "src_lang", r.warnings);
    p.tgt_lang = read_optional_string(doc, "tgt_lang", r.warnings);

    auto params = doc.find("params");
    if (params != doc.end()) {
        if (!params->is_object()) {
            r.warnings.push_back("'params' must be an object; ignored");
        } else {
            for (auto it = params->begin(); it != params->end(); ++it) {
                const ParamDesc* d = find_param(it.key());
                if (!d) {
                    r.warnings.push_back("params: unknown parameter '" + it.key() + "' ignored");
                    continue;
                }
                float v = 0.0f;
                if (it->is_boolean()) {
                    v = it->get<bool>() ? 1.0f : 0.0f;
                } else if (it->is_number()) {
                    v = it->get<float>();
                } else {
                    r.warnings.push_back("params: '" + it.key() + "' must be a number; ignored");
                    continue;
                }
                const float c = clamp_param(*d, v);
                if (c != v) {
                    r.warnings.push_back("params: '" + it.key() + "' = " + fmt_float(v) +
                                         " is outside [" + fmt_float(d->min) + ", " + fmt_float(d->max) +
                                         "]; using " + fmt_float(c));
                }
                p.params[param_index(d->id)] = c;
            }
        }
    }

    r.preset = std::move(p);
    return r;
}

PresetLoadResult load_preset_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        PresetLoadResult r;
        r.error = "cannot read " + path.string();
        return r;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    PresetLoadResult r = parse_preset(ss.str());
    if (r.preset) r.preset->path = path;
    return r;
}

std::string preset_to_json(const Preset& p) {
    json doc;
    doc["schema"] = 1;
    doc["name"]   = p.name;
    if (!p.description.empty()) doc["description"] = p.description;
    if (p.voice_a) doc["voice_a"] = *p.voice_a;
    if (p.voice_b) doc["voice_b"] = *p.voice_b;
    json params = json::object();
    for (std::size_t i = 0; i < kParamCount; ++i) {
        if (!p.params[i]) continue;
        const ParamDesc& d = kParams[i];
        const float v = *p.params[i];
        if (d.kind == ParamKind::Continuous) params[std::string(d.name)] = v;
        else                                 params[std::string(d.name)] = static_cast<long long>(std::lround(v));
    }
    doc["params"] = params;
    if (p.src_lang) doc["src_lang"] = *p.src_lang;
    if (p.tgt_lang) doc["tgt_lang"] = *p.tgt_lang;
    return doc.dump(2) + "\n";
}

bool save_preset_file(const fs::path& path, const Preset& p, std::string& error) {
    std::error_code ec;
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path(), ec);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        error = "cannot write " + path.string();
        return false;
    }
    f << preset_to_json(p);
    if (!f) {
        error = "write failed for " + path.string();
        return false;
    }
    return true;
}

Preset preset_from_params(std::string name, const VoiceParams& params, std::string_view voice_a,
                          std::string_view voice_b, std::string_view src_lang,
                          std::string_view tgt_lang) {
    Preset p;
    p.name = std::move(name);
    for (std::size_t i = 0; i < kParamCount; ++i) p.params[i] = params.values[i];
    if (!voice_a.empty())  p.voice_a  = std::string(voice_a);
    if (!voice_b.empty())  p.voice_b  = std::string(voice_b);
    if (!src_lang.empty()) p.src_lang = std::string(src_lang);
    if (!tgt_lang.empty()) p.tgt_lang = std::string(tgt_lang);
    return p;
}

void apply_preset_params(const Preset& p, VoiceParams& params) {
    for (std::size_t i = 0; i < kParamCount; ++i)
        if (p.params[i]) params.set(kParams[i].id, *p.params[i]);
}

std::vector<fs::path> list_preset_files(const fs::path& dir) {
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return files;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file(ec) && e.path().extension() == ".json") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) { return a.filename() < b.filename(); });
    return files;
}

const char* to_string(ParamKind k) {
    switch (k) {
        case ParamKind::Continuous: return "continuous";
        case ParamKind::Integer:    return "integer";
        case ParamKind::Toggle:     return "toggle";
    }
    return "?";
}

const char* to_string(ParamTier t) {
    switch (t) {
        case ParamTier::Mvp:     return "mvp";
        case ParamTier::Stretch: return "stretch";
    }
    return "?";
}

const char* to_string(ParamGroup g) {
    switch (g) {
        case ParamGroup::Pitch:       return "pitch";
        case ParamGroup::Timbre:      return "timbre";
        case ParamGroup::Output:      return "output";
        case ParamGroup::Translation: return "translation";
    }
    return "?";
}

const char* to_string(Smoothing s) {
    switch (s) {
        case Smoothing::None:             return "none";
        case Smoothing::PerFrameRamp:     return "per-frame ramp";
        case Smoothing::PerSampleOnePole: return "per-sample one-pole";
        case Smoothing::Envelope:         return "envelope";
        case Smoothing::OutputRamp:       return "20 ms output ramp + state reset";
    }
    return "?";
}

std::string descriptor_table_json() {
    json doc;
    doc["schema"] = 1;
    json arr = json::array();
    for (const ParamDesc& d : kParams) {
        json e;
        e["name"]      = std::string(d.name);
        e["label"]     = std::string(d.label);
        e["unit"]      = std::string(d.unit);
        e["min"]       = d.min;
        e["max"]       = d.max;
        e["default"]   = d.def;
        e["kind"]      = to_string(d.kind);
        e["smoothing"] = to_string(d.smoothing);
        e["tier"]      = to_string(d.tier);
        e["group"]     = to_string(d.group);
        e["help"]      = std::string(d.help);
        arr.push_back(std::move(e));
    }
    doc["params"] = std::move(arr);
    return doc.dump(2) + "\n";
}

} // namespace fluxvoice
