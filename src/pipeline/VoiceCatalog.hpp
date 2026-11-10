// Shared offline/live voices.json validation. Load-time only, no ORT dependency.
#pragma once

#include "dsp/PitchTracker.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fluxvoice {

struct VoiceEntry {
    std::string name, display;
    std::vector<float> embedding;
    TargetVoiceStats stats{};
    std::string license, source;
};

inline bool valid_voice_name(std::string_view name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
    });
}

inline bool read_voice_catalog(const std::filesystem::path& path,
                                std::optional<std::size_t> embedding_dim,
                                std::vector<VoiceEntry>& out, std::string& error) {
    using nlohmann::json;
    const auto fail = [&](std::string message) { error = std::move(message); return false; };
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail("cannot open voices.json: " + path.string());
    std::stringstream text;
    text << f.rdbuf();
    const auto data = json::parse(text.str(), nullptr, false);
    if (!data.is_object()) return fail("voices.json must be a valid JSON object");
    if (!data.contains("schema") || !data["schema"].is_number_integer() || data["schema"] != 1)
        return fail("unsupported voices.json schema");
    for (const auto& item : data.items())
        if (item.key() != "schema" && item.key() != "voices")
            return fail("voices.json: unknown key '" + item.key() + "'");
    if (!data.contains("voices") || !data["voices"].is_array() || data["voices"].empty())
        return fail("voices.json: 'voices' must be a non-empty list");
    const auto finite = [](const json& x) {
        return x.is_number() && std::isfinite(x.get<double>());
    };
    std::vector<VoiceEntry> parsed;
    for (const auto& v : data["voices"]) {
        if (!v.is_object()) return fail("voices.json: each voice must be an object");
        for (const char* key : {"name", "display", "license", "source"})
            if (!v.contains(key) || !v[key].is_string() || v[key].get<std::string>().empty())
                return fail(std::string("voices.json: missing or empty string key '") + key + "'");
        VoiceEntry entry;
        entry.name = v["name"].get<std::string>();
        if (!valid_voice_name(entry.name)) return fail("voices.json: invalid voice name '" + entry.name + "'");
        for (const auto& previous : parsed)
            if (entry.name == previous.name) return fail("duplicate voice '" + entry.name + "'");
        for (const char* key : {"logf0_mean", "logf0_std", "energy_mean", "energy_std"})
            if (!v.contains(key) || !finite(v[key]))
                return fail("voice '" + entry.name + "': '" + key + "' must be a finite number");
        entry.stats = {v["logf0_mean"].get<double>(), v["logf0_std"].get<double>(),
                       v["energy_mean"].get<double>(), v["energy_std"].get<double>()};
        if (entry.stats.logf0_std <= 0.0 || entry.stats.energy_std <= 0.0)
            return fail("voice '" + entry.name + "': std must be > 0");
        if (!v.contains("embedding") || !v["embedding"].is_array())
            return fail("voice '" + entry.name + "': embedding must be a list of numbers");
        if (embedding_dim && v["embedding"].size() != *embedding_dim)
            return fail("voice '" + entry.name + "': embedding has " +
                std::to_string(v["embedding"].size()) + " values, vocoder expects spk_emb_dim " +
                std::to_string(*embedding_dim));
        for (const auto& x : v["embedding"]) {
            if (!finite(x) || std::abs(x.get<double>()) > std::numeric_limits<float>::max())
                return fail("voice '" + entry.name + "': embedding elements must be finite f32 numbers");
            entry.embedding.push_back(x.get<float>());
        }
        entry.display = v["display"].get<std::string>();
        entry.license = v["license"].get<std::string>();
        entry.source = v["source"].get<std::string>();
        parsed.push_back(std::move(entry));
    }
    out = std::move(parsed);
    error.clear();
    return true;
}

inline const VoiceEntry* find_voice(const std::vector<VoiceEntry>& voices,
                                    std::string_view name, std::string& error) {
    std::string available;
    for (const auto& voice : voices) {
        if (voice.name == name) { error.clear(); return &voice; }
        if (!available.empty()) available += ", ";
        available += voice.name;
    }
    error = "unknown voice '" + std::string(name) + "' (available: " + available + ")";
    return nullptr;
}

} // namespace fluxvoice
