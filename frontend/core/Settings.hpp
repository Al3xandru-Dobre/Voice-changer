// Settings — persisted user preferences (frontend/README.md §9.5).
//
// Device policy persistence follows Milestone 1.5 decision 4: "Follow system
// default" persists as a policy; an explicit endpoint is remembered only as a
// display-name *hint* that is re-resolved against the current snapshot at
// start, and the user is asked again when it cannot be found. Unknown keys in
// the file are preserved across save so a newer build's settings survive an
// older build.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

enum class Theme : std::uint8_t { Dark, Light, System };

const char* to_string(Theme t);
bool theme_from_string(std::string_view s, Theme& out);

struct Settings {
    // Window & tray behaviour
    bool start_minimized        = false;
    bool close_to_tray          = true;
    bool start_engine_on_launch = false;
    bool mini_panel_on_top      = true;
    Theme theme                 = Theme::Dark;
    float ui_scale              = 1.0f;   // multiplies the display scale
    int window_w                = 1180;
    int window_h                = 760;
    int mini_x                  = -1;     // -1 = let the window manager place it
    int mini_y                  = -1;

    // Directories ("" = default discovered at start)
    std::string models_dir;
    std::string user_preset_dir;
    std::string studio_dir;
    std::string catalogue_index;          // extra index.json to load

    // Device policy (M1.5 decision 4): "default" or "name:<display name>".
    std::string capture_policy  = "default";
    std::string playback_policy = "default";
    bool capture_loopback       = false;

    // Last voice state (re-applied at start when still valid)
    std::string active_bundle;
    std::string voice_a;
    std::string voice_b;
    std::string last_preset;
    std::string src_lang = "en";
    std::string tgt_lang = "es";
    int  wait_k          = 3;
    bool translate       = false;
    float gain_db        = 0.0f;

    /// Keys this build does not know, kept verbatim (JSON text) for round-trip.
    std::map<std::string, std::string> extras;

    std::string to_json() const;

    /// Parse; unknown keys go to `extras`, type mismatches produce warnings
    /// and keep the default. Never throws; a malformed document yields
    /// defaults plus one warning.
    static Settings from_json(std::string_view text, std::vector<std::string>* warnings = nullptr);

    bool save(const std::filesystem::path& file, std::string& error) const;
    static Settings load(const std::filesystem::path& file, std::vector<std::string>* warnings = nullptr);

    friend bool operator==(const Settings&, const Settings&) = default;
};

} // namespace fluxvoice::frontend
