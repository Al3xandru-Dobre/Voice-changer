// Settings — JSON persistence.
#include "core/Settings.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
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
    out = it->get<T>();
}

bool is_bool(const json& j)   { return j.is_boolean(); }
bool is_int(const json& j)    { return j.is_number_integer(); }
bool is_num(const json& j)    { return j.is_number(); }
bool is_string(const json& j) { return j.is_string(); }

const char* const kKnownKeys[] = {
    "schema", "start_minimized", "close_to_tray", "start_engine_on_launch", "mini_panel_on_top",
    "theme", "ui_scale", "window_w", "window_h", "mini_x", "mini_y", "models_dir", "user_preset_dir",
    "studio_dir", "catalogue_index", "capture_policy", "playback_policy", "capture_loopback",
    "active_bundle", "voice_a", "voice_b", "last_preset", "src_lang", "tgt_lang", "wait_k",
    "translate", "gain_db"};

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
    d["mini_x"]                 = mini_x;
    d["mini_y"]                 = mini_y;
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
    read(doc, "mini_x", s.mini_x, warnings, is_int);
    read(doc, "mini_y", s.mini_y, warnings, is_int);
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
    if (s.wait_k < 1 || s.wait_k > 10) {
        if (warnings) warnings->push_back("settings: wait_k out of [1, 10]; using 3");
        s.wait_k = 3;
    }
    read(doc, "translate", s.translate, warnings, is_bool);
    read(doc, "gain_db", s.gain_db, warnings, is_num);
    for (auto it = doc.begin(); it != doc.end(); ++it)
        if (!known(it.key())) s.extras[it.key()] = it->dump();
    return s;
}

bool Settings::save(const fs::path& file, std::string& error) const {
    std::error_code ec;
    if (!file.parent_path().empty()) fs::create_directories(file.parent_path(), ec);
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f) {
        error = "cannot write " + file.string();
        return false;
    }
    f << to_json();
    if (!f) {
        error = "write failed for " + file.string();
        return false;
    }
    return true;
}

Settings Settings::load(const fs::path& file, std::vector<std::string>* warnings) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return Settings{};  // first run: defaults, no warning
    std::stringstream ss;
    ss << f.rdbuf();
    return from_json(ss.str(), warnings);
}

} // namespace fluxvoice::frontend
