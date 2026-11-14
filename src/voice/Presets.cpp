// Presets — JSON preset loader/saver and the descriptor-table dump.
#include "voice/Presets.hpp"

#include "voice/VoiceParams.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef FLUXVOICE_SOURCE_DIR
#define FLUXVOICE_SOURCE_DIR ""
#endif

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

/// Does `s` end with `suffix`, ASCII case-insensitively?
bool ends_with_ci(std::string_view s, std::string_view suffix) {
    if (s.size() < suffix.size()) return false;
    const std::string_view tail = s.substr(s.size() - suffix.size());
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char a = static_cast<unsigned char>(tail[i]);
        const unsigned char b = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
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
                    if (d->kind != ParamKind::Toggle)
                        r.warnings.push_back("params: '" + it.key() +
                                             "' is not a toggle; JSON boolean read as " +
                                             fmt_float(v));
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

namespace {

std::string lower_copy(std::string_view s) {
    std::string out(s);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}

} // namespace

std::vector<fs::path> preset_search_dirs(const fs::path& user_dir, const fs::path& exe_dir) {
    std::vector<fs::path> dirs;
    const auto add = [&](const fs::path& dir) {
        if (dir.empty()) return;
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) return;
        fs::path key = fs::weakly_canonical(dir, ec);
        if (ec) key = dir;
        if (std::find(dirs.begin(), dirs.end(), key) == dirs.end()) dirs.push_back(key);
    };
    add(user_dir);
    add(exe_dir / "presets");
    const std::string src = FLUXVOICE_SOURCE_DIR;
    if (!src.empty()) add(fs::path(src) / "presets");
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec) add(cwd / "presets");
    return dirs;
}

PresetLoadResult resolve_preset(std::string_view name_or_path, std::span<const fs::path> dirs) {
    if (ends_with_ci(name_or_path, ".json") || name_or_path.find('/') != std::string_view::npos ||
        name_or_path.find('\\') != std::string_view::npos)
        return load_preset_file(fs::path(name_or_path));

    const std::string wanted = lower_copy(name_or_path);
    std::string available;
    for (const fs::path& dir : dirs) {
        for (const fs::path& file : list_preset_files(dir)) {
            if (lower_copy(file.stem().string()) == wanted) return load_preset_file(file);
            PresetLoadResult r = load_preset_file(file);
            if (!available.empty()) available += ", ";
            available += r.ok() ? r.preset->name : file.stem().string();
            if (r.ok() && lower_copy(r.preset->name) == wanted) return r;
        }
    }
    PresetLoadResult r;
    r.error = "unknown preset '" + std::string(name_or_path) + "' (available: " + available + ")";
    return r;
}

bool parse_param_assignment(std::string_view text, const ParamDesc*& desc, float& value,
                            std::string& warning, std::string& error) {
    desc = nullptr;
    value = 0.0f;
    warning.clear();
    error.clear();
    const std::size_t eq = text.find('=');
    if (eq == std::string_view::npos || eq == 0 || eq + 1 >= text.size()) {
        error = "expected name=value, got '" + std::string(text) + "'";
        return false;
    }
    const std::string_view name = text.substr(0, eq);
    const std::string_view rhs = text.substr(eq + 1);
    desc = find_param(name);
    if (!desc) {
        error = "unknown parameter '" + std::string(name) + "'";
        if (name == "voice_a" || name == "voice_b")
            error += " (use --voice/--voice-b to select voices)";
        return false;
    }
    float v = 0.0f;
    if (desc->kind == ParamKind::Toggle &&
        (rhs == "on" || rhs == "off" || rhs == "true" || rhs == "false")) {
        v = (rhs == "on" || rhs == "true") ? 1.0f : 0.0f;
    } else {
        double dv = 0.0;
        if (!parse_finite_float(rhs, dv)) {
            error = "expected name=value ('" + std::string(rhs) + "' is not a finite number)";
            return false;
        }
        v = static_cast<float>(dv);
    }
    const float clamped = clamp_param(*desc, v);
    if (v < desc->min || v > desc->max)
        warning = "'" + std::string(name) + "' = " + fmt_float(v) + " is outside [" +
                  fmt_float(desc->min) + ", " + fmt_float(desc->max) + "]; using " +
                  fmt_float(clamped);
    else if (clamped != v)
        warning = "'" + std::string(name) + "' = " + fmt_float(v) + " rounded to " +
                  format_param_value(*desc, clamped);
    value = clamped;
    return true;
}

bool parse_finite_float(std::string_view text, double& out) {
    // Floating-point std::from_chars is unavailable on Apple libc++ with our
    // macOS 13.0 deployment target, so parse with std::strtod instead. The
    // program never calls setlocale, so strtod always uses the C locale.
    out = 0.0;
    if (text.empty() || text.size() > 63) return false;
    // strtod would silently skip leading whitespace; reject it ourselves.
    if (std::isspace(static_cast<unsigned char>(text.front()))) return false;
    char buf[64];
    std::memcpy(buf, text.data(), text.size());
    buf[text.size()] = '\0';
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(buf, &end);
    if (end == buf || end != buf + text.size() || errno == ERANGE || !std::isfinite(v))
        return false;
    out = v;
    return true;
}

std::string format_param_value(const ParamDesc& d, float v) {
    std::ostringstream o;
    switch (d.kind) {
        case ParamKind::Toggle:     return v >= 0.5f ? "on" : "off";
        case ParamKind::Integer:    o << static_cast<long long>(std::lround(v)); break;
        case ParamKind::Continuous: o << v; break;
    }
    return o.str();
}

std::string params_table_text() {
    std::ostringstream o;
    o << "name            unit   range            default  smoothing                    tier     help\n";
    for (const ParamDesc& d : kParams) {
        char left[128];
        std::snprintf(left, sizeof(left), "%-15s %-6s [%5g, %5g]    %-8g %-26s %-8s ",
                      std::string(d.name).c_str(), std::string(d.unit).c_str(), d.min, d.max,
                      d.def, to_string(d.smoothing), to_string(d.tier));
        o << left << d.help << '\n';
    }
    return o.str();
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
