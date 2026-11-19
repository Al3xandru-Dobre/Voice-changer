// ToolCli — command-line helpers shared by the offline runners (fv_vc_file,
// fv_s2ut_file): bounded integer parsing, --param / --change parsing, preset
// and voice resolution (descriptor defaults -> preset params and voice names
// -> --voice / --voice-b -> --param assignments), percentiles and the
// effective-parameter summary lines. Offline tools only: allocates freely.
#pragma once

#include "pipeline/VoiceCatalog.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fluxvoice::tools {

struct ScheduledChange {
    double seconds = 0.0;
    const ParamDesc* desc = nullptr;
    float value = 0.0f;
};

using ParamAssignments = std::vector<std::pair<const ParamDesc*, float>>;

inline bool parse_uint(std::string_view s, std::uint32_t lo, std::uint32_t hi, std::uint32_t& out) {
    std::uint32_t v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || v < lo || v > hi)
        return false;
    out = v;
    return true;
}

inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(idx));
    const auto hi = static_cast<std::size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/// `--param name=value`; reports on stderr and returns false when rejected.
inline bool parse_param_option(std::string_view v, ParamAssignments& out) {
    const ParamDesc* d = nullptr;
    float value = 0.0f;
    std::string warning, error;
    if (!parse_param_assignment(v, d, value, warning, error)) {
        std::cerr << "--param: " << error << '\n';
        return false;
    }
    if (!warning.empty()) std::cerr << "--param: " << warning << '\n';
    out.emplace_back(d, value);
    return true;
}

/// `--change <seconds>:<name>=<value>`; reports on stderr and returns false when rejected.
inline bool parse_change_option(std::string_view v, std::vector<ScheduledChange>& out) {
    const std::size_t colon = v.find(':');
    if (colon == std::string_view::npos) {
        std::cerr << "Invalid --change '" << v << "' (expected seconds:name=value)\n";
        return false;
    }
    double seconds = 0.0;
    if (!parse_finite_float(v.substr(0, colon), seconds) || seconds < 0.0) {
        std::cerr << "Invalid --change '" << v << "' (bad seconds)\n";
        return false;
    }
    const ParamDesc* d = nullptr;
    float value = 0.0f;
    std::string warning, error;
    if (!parse_param_assignment(v.substr(colon + 1), d, value, warning, error)) {
        std::cerr << "--change: " << error << '\n';
        return false;
    }
    if (!warning.empty()) std::cerr << "--change: " << warning << '\n';
    out.push_back({seconds, d, value});
    return true;
}

/// Resolve `--preset` (searched in `preset_dir`, exe/presets, source presets/,
/// cwd/presets, or a .json path). An empty name leaves `res` empty (not ok()).
inline bool load_preset_option(const std::string& name, const std::filesystem::path& preset_dir,
                               const char* argv0, PresetLoadResult& res) {
    if (name.empty()) return true;
    std::error_code ec;
    const std::filesystem::path exe_dir =
        std::filesystem::weakly_canonical(std::filesystem::path(argv0), ec).parent_path();
    res = resolve_preset(name, preset_search_dirs(preset_dir, exe_dir));
    if (!res.ok()) {
        std::cerr << res.error << '\n';
        return false;
    }
    for (const std::string& w : res.warnings) std::cerr << "preset: " << w << '\n';
    return true;
}

/// Build the initial VoiceParams from the bundle's voices: descriptor defaults
/// -> preset params + voice names -> --voice / --voice-b -> --param
/// assignments. voice_b follows voice_a unless a preset or --voice-b chose one.
inline bool resolve_voice_params(const std::vector<VoiceEntry>& voices,
                                 const PresetLoadResult& preset_res, const std::string& voice,
                                 const std::string& voice_b, const ParamAssignments& assignments,
                                 VoiceParams& params) {
    std::string error;
    const auto voice_index = [&](std::string_view name, std::uint16_t& out) -> bool {
        const VoiceEntry* v = find_voice(voices, name, error);
        if (!v) {
            std::cerr << error << '\n';
            return false;
        }
        out = static_cast<std::uint16_t>(v - voices.data());
        return true;
    };
    bool voice_b_follows_a = true;
    if (preset_res.ok()) {
        apply_preset_params(*preset_res.preset, params);
        if (preset_res.preset->voice_a && !voice_index(*preset_res.preset->voice_a, params.voice_a))
            return false;
        if (preset_res.preset->voice_b && !voice_index(*preset_res.preset->voice_b, params.voice_b))
            return false;
        if (preset_res.preset->voice_b) voice_b_follows_a = false;
    }
    if (!voice.empty() && !voice_index(voice, params.voice_a)) return false;
    if (!voice_b.empty()) {
        if (!voice_index(voice_b, params.voice_b)) return false;
        voice_b_follows_a = false;
    }
    for (const auto& [d, v] : assignments) params.set(d->id, v);
    if (voice_b_follows_a) params.voice_b = params.voice_a;
    return true;
}

/// "Param" / "Scheduled change" summary lines (non-default controls only).
inline void print_param_summary(const VoiceParams& initial,
                                const std::vector<ScheduledChange>& changes) {
    for (const ParamDesc& d : kParams) {
        if (initial.get(d.id) == d.def) continue;
        std::printf("Param              : %s = %s\n", std::string(d.name).c_str(),
                    format_param_value(d, initial.get(d.id)).c_str());
    }
    for (const ScheduledChange& c : changes) {
        std::printf("Scheduled change   : %s = %s at %.3f s\n", std::string(c.desc->name).c_str(),
                    format_param_value(*c.desc, c.value).c_str(), c.seconds);
    }
}

} // namespace fluxvoice::tools
