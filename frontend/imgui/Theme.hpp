// Theme — style tokens for the Dear ImGui views (frontend/README.md §7): one
// accent, semantic status colours, a sober dark theme and a light theme.
// Colours are chosen for WCAG AA contrast of body text on the window
// background in both themes.
#pragma once

#include "core/Settings.hpp"
#include "core/TrayModel.hpp"

#include <imgui.h>

namespace fluxvoice::frontend {

struct ThemeColors {
    ImVec4 accent;        // primary actions, selected rail item
    ImVec4 accent_hover;
    ImVec4 accent_text;   // text on accent
    ImVec4 success;       // running
    ImVec4 warning;       // recovering, muted
    ImVec4 danger;        // error / needs attention
    ImVec4 neutral;       // idle
    ImVec4 text_dim;      // secondary text
    ImVec4 card_bg;
    ImVec4 card_border;
    ImVec4 rail_bg;
    ImVec4 status_bg;
    ImVec4 meter_bg;
    ImVec4 meter_fill;
    ImVec4 meter_hot;
};

/// Apply ImGui style + colours. `dark` resolves Theme::System on the caller
/// side (SDL_GetSystemTheme). `scale` multiplies paddings/rounding once.
void apply_theme(bool dark, float scale);

const ThemeColors& theme_colors();
bool theme_is_dark();

/// Semantic colour for a tray/stream state.
ImVec4 state_color(TrayState state);

/// Size tokens (unscaled; multiply by the UI scale).
namespace tokens {
constexpr float kRailWidth    = 180.0f;
constexpr float kStatusHeight = 30.0f;
constexpr float kCardPadding  = 12.0f;
constexpr float kBaseFont     = 16.0f;
constexpr float kTitleFont    = 22.0f;
constexpr float kHeroFont     = 30.0f;
constexpr float kMiniWidth    = 360.0f;
constexpr float kMiniHeight   = 132.0f;
} // namespace tokens

} // namespace fluxvoice::frontend
