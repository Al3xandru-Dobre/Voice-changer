// Theme — ImGui style application.
#include "imgui/Theme.hpp"

#include "core/TrayModel.hpp"

#include <imgui.h>

namespace fluxvoice::frontend {

namespace {

ThemeColors g_colors{};
bool g_dark = true;

ImVec4 rgb(int r, int g, int b, float a = 1.0f) {
    return ImVec4(static_cast<float>(r) / 255.0f, static_cast<float>(g) / 255.0f, static_cast<float>(b) / 255.0f, a);
}

} // namespace

void apply_theme(bool dark, float scale) {
    g_dark = dark;
    ImGuiStyle& s = ImGui::GetStyle();
    s = ImGuiStyle();  // reset before scaling so repeated calls do not compound
    if (dark) ImGui::StyleColorsDark(&s); else ImGui::StyleColorsLight(&s);

    s.WindowPadding    = ImVec2(12.0f, 12.0f);
    s.FramePadding     = ImVec2(10.0f, 6.0f);
    s.CellPadding      = ImVec2(8.0f, 5.0f);
    s.ItemSpacing      = ImVec2(8.0f, 8.0f);
    s.ItemInnerSpacing = ImVec2(6.0f, 6.0f);
    s.IndentSpacing    = 18.0f;
    s.ScrollbarSize    = 12.0f;
    s.GrabMinSize      = 12.0f;
    s.WindowRounding   = 0.0f;
    s.ChildRounding    = 8.0f;
    s.FrameRounding    = 6.0f;
    s.PopupRounding    = 6.0f;
    s.ScrollbarRounding = 6.0f;
    s.GrabRounding     = 6.0f;
    s.TabRounding      = 6.0f;
    s.WindowBorderSize = 0.0f;
    s.ChildBorderSize  = 1.0f;
    s.FrameBorderSize  = dark ? 0.0f : 1.0f;
    s.PopupBorderSize  = 1.0f;

    ImVec4* c = s.Colors;
    const ImVec4 accent = rgb(0x2B, 0xB0, 0xA6);
    if (dark) {
        g_colors.accent       = accent;
        g_colors.accent_hover = rgb(0x3F, 0xC4, 0xBA);
        g_colors.accent_text  = rgb(0x0B, 0x1B, 0x1A);
        g_colors.success      = rgb(0x3F, 0xC4, 0x7C);
        g_colors.warning      = rgb(0xE3, 0xA0, 0x3A);
        g_colors.danger       = rgb(0xE5, 0x5A, 0x5A);
        g_colors.neutral      = rgb(0x8A, 0x93, 0x9E);
        g_colors.text_dim     = rgb(0xA3, 0xAB, 0xB5);
        g_colors.card_bg      = rgb(0x1E, 0x23, 0x2A);
        g_colors.card_border  = rgb(0x2C, 0x33, 0x3C);
        g_colors.rail_bg      = rgb(0x14, 0x18, 0x1E);
        g_colors.status_bg    = rgb(0x11, 0x15, 0x1A);
        g_colors.meter_bg     = rgb(0x12, 0x16, 0x1B);
        g_colors.meter_fill   = rgb(0x3F, 0xC4, 0x7C);
        g_colors.meter_hot    = rgb(0xE5, 0x5A, 0x5A);

        c[ImGuiCol_WindowBg]         = rgb(0x18, 0x1C, 0x22);
        c[ImGuiCol_ChildBg]          = rgb(0x18, 0x1C, 0x22, 0.0f);
        c[ImGuiCol_PopupBg]          = rgb(0x1E, 0x23, 0x2A, 0.98f);
        c[ImGuiCol_Text]             = rgb(0xEC, 0xEF, 0xF3);
        c[ImGuiCol_TextDisabled]     = rgb(0x7E, 0x87, 0x92);
        c[ImGuiCol_Border]           = g_colors.card_border;
        c[ImGuiCol_FrameBg]          = rgb(0x26, 0x2C, 0x35);
        c[ImGuiCol_FrameBgHovered]   = rgb(0x2E, 0x36, 0x40);
        c[ImGuiCol_FrameBgActive]    = rgb(0x35, 0x3E, 0x49);
        c[ImGuiCol_TitleBg]          = g_colors.rail_bg;
        c[ImGuiCol_TitleBgActive]    = g_colors.rail_bg;
        c[ImGuiCol_MenuBarBg]        = g_colors.rail_bg;
        c[ImGuiCol_ScrollbarBg]      = rgb(0x14, 0x18, 0x1E);
        c[ImGuiCol_ScrollbarGrab]    = rgb(0x3A, 0x43, 0x4E);
        c[ImGuiCol_CheckMark]        = accent;
        c[ImGuiCol_SliderGrab]       = accent;
        c[ImGuiCol_SliderGrabActive] = g_colors.accent_hover;
        c[ImGuiCol_Button]           = rgb(0x2C, 0x34, 0x3E);
        c[ImGuiCol_ButtonHovered]    = rgb(0x37, 0x41, 0x4D);
        c[ImGuiCol_ButtonActive]     = rgb(0x40, 0x4C, 0x59);
        c[ImGuiCol_Header]           = rgb(0x2B, 0xB0, 0xA6, 0.25f);
        c[ImGuiCol_HeaderHovered]    = rgb(0x2B, 0xB0, 0xA6, 0.40f);
        c[ImGuiCol_HeaderActive]     = rgb(0x2B, 0xB0, 0xA6, 0.55f);
        c[ImGuiCol_Separator]        = g_colors.card_border;
        c[ImGuiCol_Tab]              = rgb(0x22, 0x28, 0x30);
        c[ImGuiCol_TabHovered]       = rgb(0x2B, 0xB0, 0xA6, 0.6f);
        c[ImGuiCol_TabSelected]      = rgb(0x2B, 0xB0, 0xA6, 0.35f);
        c[ImGuiCol_TabSelectedOverline] = accent;
        c[ImGuiCol_TableHeaderBg]    = rgb(0x22, 0x28, 0x30);
        c[ImGuiCol_TableBorderStrong] = g_colors.card_border;
        c[ImGuiCol_TableBorderLight] = rgb(0x24, 0x2A, 0x32);
        c[ImGuiCol_TableRowBgAlt]    = rgb(0xFF, 0xFF, 0xFF, 0.03f);
        c[ImGuiCol_PlotLines]        = accent;
        c[ImGuiCol_PlotHistogram]    = accent;
        c[ImGuiCol_TextSelectedBg]   = rgb(0x2B, 0xB0, 0xA6, 0.35f);
        c[ImGuiCol_NavCursor]        = accent;
    } else {
        g_colors.accent       = rgb(0x12, 0x7E, 0x77);
        g_colors.accent_hover = rgb(0x1A, 0x94, 0x8C);
        g_colors.accent_text  = rgb(0xFF, 0xFF, 0xFF);
        g_colors.success      = rgb(0x1E, 0x8E, 0x4F);
        g_colors.warning      = rgb(0xB2, 0x6B, 0x0C);
        g_colors.danger       = rgb(0xC0, 0x32, 0x32);
        g_colors.neutral      = rgb(0x6B, 0x74, 0x7F);
        g_colors.text_dim     = rgb(0x55, 0x5E, 0x69);
        g_colors.card_bg      = rgb(0xFF, 0xFF, 0xFF);
        g_colors.card_border  = rgb(0xD7, 0xDC, 0xE2);
        g_colors.rail_bg      = rgb(0xEE, 0xF1, 0xF4);
        g_colors.status_bg    = rgb(0xE6, 0xEA, 0xEF);
        g_colors.meter_bg     = rgb(0xE3, 0xE7, 0xEC);
        g_colors.meter_fill   = rgb(0x1E, 0x8E, 0x4F);
        g_colors.meter_hot    = rgb(0xC0, 0x32, 0x32);

        c[ImGuiCol_WindowBg]         = rgb(0xF6, 0xF7, 0xF9);
        c[ImGuiCol_ChildBg]          = rgb(0xF6, 0xF7, 0xF9, 0.0f);
        c[ImGuiCol_PopupBg]          = rgb(0xFF, 0xFF, 0xFF, 0.98f);
        c[ImGuiCol_Text]             = rgb(0x16, 0x1B, 0x22);
        c[ImGuiCol_TextDisabled]     = rgb(0x8A, 0x93, 0x9E);
        c[ImGuiCol_Border]           = g_colors.card_border;
        c[ImGuiCol_FrameBg]          = rgb(0xFF, 0xFF, 0xFF);
        c[ImGuiCol_FrameBgHovered]   = rgb(0xF0, 0xF3, 0xF6);
        c[ImGuiCol_FrameBgActive]    = rgb(0xE6, 0xEB, 0xF0);
        c[ImGuiCol_MenuBarBg]        = g_colors.rail_bg;
        c[ImGuiCol_CheckMark]        = g_colors.accent;
        c[ImGuiCol_SliderGrab]       = g_colors.accent;
        c[ImGuiCol_SliderGrabActive] = g_colors.accent_hover;
        c[ImGuiCol_Button]           = rgb(0xE6, 0xEA, 0xEF);
        c[ImGuiCol_ButtonHovered]    = rgb(0xD9, 0xDF, 0xE6);
        c[ImGuiCol_ButtonActive]     = rgb(0xCB, 0xD3, 0xDC);
        c[ImGuiCol_Header]           = rgb(0x12, 0x7E, 0x77, 0.18f);
        c[ImGuiCol_HeaderHovered]    = rgb(0x12, 0x7E, 0x77, 0.30f);
        c[ImGuiCol_HeaderActive]     = rgb(0x12, 0x7E, 0x77, 0.45f);
        c[ImGuiCol_Tab]              = rgb(0xE6, 0xEA, 0xEF);
        c[ImGuiCol_TabSelected]      = rgb(0x12, 0x7E, 0x77, 0.25f);
        c[ImGuiCol_TabSelectedOverline] = g_colors.accent;
        c[ImGuiCol_TableHeaderBg]    = rgb(0xE6, 0xEA, 0xEF);
        c[ImGuiCol_TableRowBgAlt]    = rgb(0x00, 0x00, 0x00, 0.03f);
        c[ImGuiCol_PlotLines]        = g_colors.accent;
        c[ImGuiCol_PlotHistogram]    = g_colors.accent;
        c[ImGuiCol_NavCursor]        = g_colors.accent;
    }
    s.ScaleAllSizes(scale);
}

const ThemeColors& theme_colors() { return g_colors; }
bool theme_is_dark() { return g_dark; }

ImVec4 state_color(TrayState state) {
    switch (state) {
        case TrayState::Idle:       return g_colors.neutral;
        case TrayState::Running:    return g_colors.success;
        case TrayState::Muted:      return g_colors.warning;
        case TrayState::Recovering: return g_colors.warning;
        case TrayState::Error:      return g_colors.danger;
    }
    return g_colors.neutral;
}

} // namespace fluxvoice::frontend
