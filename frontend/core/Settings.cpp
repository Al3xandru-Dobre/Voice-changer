// Settings — JSON persistence.
#include "core/Settings.hpp"

#include "core/JsonRead.hpp"
#include "core/Paths.hpp"
#include "voice/VoiceParams.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace fluxvoice::frontend {

namespace fs = std::filesystem;
using nlohmann::json;

const char* to_string(Theme t) {
    switch (t) {
        case Theme::Dark:   return "dark";
        case Theme::Light:  return "light";
        case Theme::System: return "system";
    }
    return "dark";
}

bool theme_from_string(std::string_view s, Theme& out) {
    if (s == "dark")   { out = Theme::Dark;   return true; }
    if (s == "light")  { out = Theme::Light;  return true; }
    if (s == "system") { out = Theme::System; return true; }
    return false;
}

namespace {

template <typename T>
void read(const json& doc, const char* key, T& out, std::vector<std::string>* warnings, bool (*check)(const json&)) {
    auto it = doc.find(key);
    if (it == doc.end()) return;
    if (!check(*it)) {
        if (warnings) warnings->push_back(std::string("settings: '") + key + "' has the wrong type; using default");
        return;
    }
    // Numbers narrow through the checked conversions: an out-of-range or
    // non-finite value keeps the default instead of casting (UB / wrong value).
    if constexpr (std::is_same_v<T, int>) {
        if (const auto v = json_to_integral<int>(*it)) { out = *v; return; }
    } else if constexpr (std::is_same_v<T, float>) {
        if (const auto v = json_to_float(*it)) { out = *v; return; }
    } else {
        out = it->get<T>();
        return;
    }
    if (warnings) warnings->push_back(std::string("settings: '") + key + "' is out of range; using default");
}

bool is_bool(const json& j)   { return j.is_boolean(); }
bool is_int(const json& j)    { return j.is_number_integer(); }
bool is_num(const json& j)    { return j.is_number(); }
bool is_string(const json& j) { return j.is_string(); }

// mini_x / mini_y are legacy keys: read for migration, never written, never kept in extras.
const char* const kKnownKeys[] = {
    "schema", "start_minimized", "close_to_tray", "start_engine_on_launch", "mini_panel_on_top",
    "theme", "ui_scale", "window_w", "window_h", "mini_position", "mini_x", "mini_y", "models_dir", "user_preset_dir",
    "studio_dir", "catalogue_index", "capture_policy", "playback_policy", "capture_loopback",
    "active_bundle", "voice_a", "voice_b", "last_preset", "src_lang", "tgt_lang", "wait_k",
    "translate", "gain_db", "frames_per_step", "latency_target_ms", "write_stats_log", "stats_log_path"};

bool known(const std::string& k) {
    for (const char* n : kKnownKeys)
        if (k == n) return true;
    return false;
}

} // namespace

std::string Settings::to_json() const {
    json d;
    d["schema"]                 = 1;
    d["start_minimized"]        = start_minimized;
    d["close_to_tray"]          = close_to_tray;
    d["start_engine_on_launch"] = start_engine_on_launch;
    d["mini_panel_on_top"]      = mini_panel_on_top;
    d["theme"]                  = to_string(theme);
    d["ui_scale"]               = ui_scale;
    d["window_w"]               = window_w;
    d["window_h"]               = window_h;
    if (mini_has_position) d["mini_position"] = json::array({mini_x, mini_y});
    d["models_dir"]             = models_dir;
    d["user_preset_dir"]        = user_preset_dir;
    d["studio_dir"]             = studio_dir;
    d["catalogue_index"]        = catalogue_index;
    d["capture_policy"]         = capture_policy;
    d["playback_policy"]        = playback_policy;
    d["capture_loopback"]       = capture_loopback;
    d["active_bundle"]          = active_bundle;
    d["voice_a"]                = voice_a;
    d["voice_b"]                = voice_b;
    d["last_preset"]            = last_preset;
    d["src_lang"]               = src_lang;
    d["tgt_lang"]               = tgt_lang;
    d["wait_k"]                 = wait_k;
    d["translate"]              = translate;
    d["gain_db"]                = gain_db;
    d["frames_per_step"]        = frames_per_step;
    d["latency_target_ms"]      = latency_target_ms;
    d["write_stats_log"]        = write_stats_log;
    d["stats_log_path"]         = stats_log_path;
    for (const auto& [k, text] : extras) {
        json v = json::parse(text, nullptr, false);
        if (!v.is_discarded()) d[k] = v;
    }
    return d.dump(2) + "\n";
}

Settings Settings::from_json(std::string_view text, std::vector<std::string>* warnings) {
    Settings s;
    const json doc = json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        if (warnings) warnings->push_back("settings: not a JSON object; using defaults");
        return s;
    }
    read(doc, "start_minimized", s.start_minimized, warnings, is_bool);
    read(doc, "close_to_tray", s.close_to_tray, warnings, is_bool);
    read(doc, "start_engine_on_launch", s.start_engine_on_launch, warnings, is_bool);
    read(doc, "mini_panel_on_top", s.mini_panel_on_top, warnings, is_bool);
    {
        std::string t;
        read(doc, "theme", t, warnings, is_string);
        if (!t.empty() && !theme_from_string(t, s.theme) && warnings)
            warnings->push_back("settings: unknown theme '" + t + "'; using dark");
    }
    read(doc, "ui_scale", s.ui_scale, warnings, is_num);
    if (s.ui_scale < 0.5f || s.ui_scale > 3.0f) {
        if (warnings) warnings->push_back("settings: ui_scale out of [0.5, 3]; using 1");
        s.ui_scale = 1.0f;
    }
    read(doc, "window_w", s.window_w, warnings, is_int);
    read(doc, "window_h", s.window_h, warnings, is_int);
    {
        auto mp = doc.find("mini_position");
        if (mp != doc.end()) {
            const auto mx = mp->is_array() && mp->size() == 2 ? json_to_integral<int>((*mp)[0]) : std::nullopt;
            const auto my = mp->is_array() && mp->size() == 2 ? json_to_integral<int>((*mp)[1]) : std::nullopt;
            if (mx && my && (*mp)[0].is_number_integer() && (*mp)[1].is_number_integer()) {
                s.mini_has_position = true;
                s.mini_x            = *mx;
                s.mini_y            = *my;
            } else if (warnings) {
                warnings->push_back("settings: 'mini_position' must be [x, y]; letting the window manager place the Mini panel");
            }
        } else {
            // Legacy pair (before mini_position): -1,-1 meant "no position".
            int lx = -1, ly = -1;
            const bool has_x = doc.contains("mini_x"), has_y = doc.contains("mini_y");
            read(doc, "mini_x", lx, warnings, is_int);
            read(doc, "mini_y", ly, warnings, is_int);
            // Both halves must be valid integers: one bad value must not place the panel at (-1, y).
            const auto valid = [&](const char* k) {
                const auto it = doc.find(k);
                return it != doc.end() && it->is_number_integer() && json_to_integral<int>(*it).has_value();
            };
            if (has_x && has_y && valid("mini_x") && valid("mini_y") && !(lx == -1 && ly == -1)) {
                s.mini_has_position = true;
                s.mini_x            = lx;
                s.mini_y            = ly;
            }
        }
    }
    read(doc, "models_dir", s.models_dir, warnings, is_string);
    read(doc, "user_preset_dir", s.user_preset_dir, warnings, is_string);
    read(doc, "studio_dir", s.studio_dir, warnings, is_string);
    read(doc, "catalogue_index", s.catalogue_index, warnings, is_string);
    read(doc, "capture_policy", s.capture_policy, warnings, is_string);
    read(doc, "playback_policy", s.playback_policy, warnings, is_string);
    read(doc, "capture_loopback", s.capture_loopback, warnings, is_bool);
    read(doc, "active_bundle", s.active_bundle, warnings, is_string);
    read(doc, "voice_a", s.voice_a, warnings, is_string);
    read(doc, "voice_b", s.voice_b, warnings, is_string);
    read(doc, "last_preset", s.last_preset, warnings, is_string);
    read(doc, "src_lang", s.src_lang, warnings, is_string);
    read(doc, "tgt_lang", s.tgt_lang, warnings, is_string);
    read(doc, "wait_k", s.wait_k, warnings, is_int);
    {
        const ParamDesc& wk = param_desc(ParamId::WaitK);
        const int lo = static_cast<int>(wk.min), hi = static_cast<int>(wk.max), def = static_cast<int>(wk.def);
        if (s.wait_k < lo || s.wait_k > hi) {
            if (warnings)
                warnings->push_back("settings: wait_k out of [" + std::to_string(lo) + ", " + std::to_string(hi) +
                                    "]; using " + std::to_string(def));
            s.wait_k = def;
        }
    }
    read(doc, "translate", s.translate, warnings, is_bool);
    read(doc, "gain_db", s.gain_db, warnings, is_num);
    {
        // Typed, checked readers: a string, a fraction, a negative or an out-of-range
        // number keeps the default (0) and says so; nothing is cast blindly.
        std::vector<std::string> local;
        const int fps = read_integral<int>(doc, "frames_per_step", 0, 0, 4, &local);
        if (valid_frames_per_step(fps)) {
            s.frames_per_step = fps;
        } else if (warnings) {
            warnings->push_back("settings: frames_per_step must be 0, 1, 2 or 4; using 0 (bundle default)");
        }
        s.latency_target_ms =
            read_integral<std::uint32_t>(doc, "latency_target_ms", 0u, 0u, kMaxLatencyTargetMs, &local);
        if (warnings)
            for (const std::string& w : local) warnings->push_back("settings: " + w);
    }
    read(doc, "write_stats_log", s.write_stats_log, warnings, is_bool);
    read(doc, "stats_log_path", s.stats_log_path, warnings, is_string);
    if (s.stats_log_path.find('\0') != std::string::npos) {  // an embedded NUL would truncate the path at open()
        if (warnings) warnings->push_back("settings: 'stats_log_path' contains a NUL character; using the default path");
        s.stats_log_path.clear();
    }
    for (auto it = doc.begin(); it != doc.end(); ++it)
        if (!known(it.key())) s.extras[it.key()] = it->dump();
    return s;
}

namespace {

bool read_file(const fs::path& file, std::string& out) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool is_json_object(const std::string& text) {
    const json doc = json::parse(text, nullptr, false);
    return !doc.is_discarded() && doc.is_object();
}

fs::path with_suffix(const fs::path& file, const char* suffix) {
    fs::path p = file;
    p += suffix;
    return p;
}

/// Move a damaged file to `<file>.bad`, replacing an older one.
bool move_aside(const fs::path& file, fs::path& bad) {
    bad = with_suffix(file, ".bad");
    std::error_code ec;
    fs::remove(bad, ec);
    ec.clear();
    fs::rename(file, bad, ec);
    return !ec;
}

} // namespace

bool Settings::save(const fs::path& file, std::string& error) const {
    std::error_code ec;
    if (!file.parent_path().empty()) fs::create_directories(file.parent_path(), ec);

    // Never overwrite a damaged file that could not be preserved.
    if (fs::exists(file, ec)) {
        std::string existing;
        if (read_file(file, existing) && !is_json_object(existing)) {
            fs::path bad;
            if (!move_aside(file, bad)) {
                error = "refusing to overwrite " + path_to_utf8(file) + ": it is not valid settings JSON and could not be moved to " +
                        path_to_utf8(bad);
                return false;
            }
        }
    }

    const fs::path tmp = with_suffix(file, ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            error = "cannot write " + path_to_utf8(tmp);
            return false;
        }
        f << to_json();
        f.flush();
        if (!f) {
            f.close();
            fs::remove(tmp, ec);
            error = "write failed for " + path_to_utf8(tmp);
            return false;
        }
    }
    fs::rename(tmp, file, ec);
    if (ec) {
        const std::string msg = ec.message();
        fs::remove(tmp, ec);
        error = "cannot replace " + path_to_utf8(file) + ": " + msg;
        return false;
    }
    return true;
}

Settings Settings::load(const fs::path& file, std::vector<std::string>* warnings) {
    std::string text;
    if (!read_file(file, text)) return Settings{};  // first run: defaults, no warning
    if (!is_json_object(text)) {
        fs::path bad;
        const bool moved = move_aside(file, bad);
        if (warnings)
            warnings->push_back(moved ? "settings: " + path_to_utf8(file) + " is not valid JSON; kept as " + path_to_utf8(bad) +
                                            " and using defaults"
                                      : "settings: " + path_to_utf8(file) + " is not valid JSON and could not be moved to " +
                                            path_to_utf8(bad) + "; using defaults and not overwriting it");
        return Settings{};
    }
    return from_json(text, warnings);
}

} // namespace fluxvoice::frontend
