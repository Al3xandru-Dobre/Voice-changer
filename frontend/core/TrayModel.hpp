// TrayModel — the declarative system-tray state: icon state, tooltip and a
// menu tree derived from the application state, plus the procedural icon.
//
// The platform layer (platform/SdlTray.cpp) renders a TraySpec with SDL3 and
// reports clicks back as TrayActions. Keeping the spec declarative means the
// menu can be tested without a desktop and rebuilt only when it changes.
#pragma once

#include "audio/AudioTypes.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

enum class TrayState : std::uint8_t { Idle, Running, Muted, Recovering, Error };

const char* to_string(TrayState s);

struct TrayItem {
    enum class Kind : std::uint8_t { Button, Check, Radio, Separator, Submenu };

    std::string id;      // stable identifier, e.g. "engine", "preset:Deep"
    std::string label;
    Kind        kind    = Kind::Button;
    bool        checked = false;
    bool        enabled = true;
    std::vector<TrayItem> children;  // for Submenu
};

struct TraySpec {
    TrayState             state = TrayState::Idle;
    std::string           tooltip;
    std::vector<TrayItem> items;

    /// Content hash (state, tooltip, items).
    std::uint64_t hash() const;
    /// Items only: the platform layer rebuilds the native menu when this changes;
    /// the tooltip (queued ms) may change every second without a rebuild.
    std::uint64_t menu_hash() const;
};

/// Everything the tray needs to know, copied from the model.
struct TrayInputs {
    StreamState  stream        = StreamState::Idle;
    bool         engine_on     = false;  // user intent (Start pressed), not liveness
    bool         muted         = false;
    bool         translate     = false;
    bool         translator_available = false;
    bool         window_visible = true;
    std::string  preset;
    std::vector<std::string> presets;
    std::string  voice;
    std::vector<std::string> voices;
    std::string  language_pair;              // "es -> en"
    std::vector<std::string> language_pairs; // recent/known pairs
    double       queued_ms = 0.0;
    std::string  error_text;
    std::uint32_t retry_attempt = 0;
};

TrayState tray_state_for(const TrayInputs& in);
TraySpec  build_tray_spec(const TrayInputs& in);

struct TrayAction {
    enum class Type : std::uint8_t {
        ShowPanel, ToggleMiniPanel, ToggleEngine, ToggleMute,
        SetModeVoiceChanger, SetModeTranslate, LoadPreset, SetVoice, SetLanguagePair, Quit
    };
    Type        type = Type::ShowPanel;
    std::string arg;  // preset name, voice name or "src->tgt"
};

/// Item id -> action ("preset:Deep" -> LoadPreset "Deep"). Nullopt for
/// separators/submenus/unknown ids.
std::optional<TrayAction> action_for_item(std::string_view id);

/// 32x32 RGBA8 icon (R in the lowest byte), drawn procedurally: a rounded
/// tile in the state colour with a white five-bar waveform; Muted adds a
/// diagonal slash. No binary assets, so every build has an icon.
struct TrayIconPixels {
    static constexpr int kSize = 32;
    std::array<std::uint32_t, kSize * kSize> rgba{};
};

TrayIconPixels render_tray_icon(TrayState state);

} // namespace fluxvoice::frontend
