// Mini panel — compact always-on-top controls (frontend/README.md §4.9) and
// the section dispatcher.
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/TrayModel.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace {

int index_of(const std::vector<std::string>& items, const std::string& value) {
    for (std::size_t i = 0; i < items.size(); ++i)
        if (items[i] == value) return static_cast<int>(i);
    return -1;
}

} // namespace

void draw_mini_panel(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const ThemeColors& t = theme_colors();
    const TraySpec spec = app.tray_spec();

    widgets::status_dot(state_color(spec.state), 6.0f * ctx.scale);
    ImGui::SameLine();
    ImGui::TextUnformatted(app.status_summary().pill.c_str());
    ImGui::SameLine();
    char q[32];
    std::snprintf(q, sizeof q, "%.0f ms", app.telemetry().queued_ms());
    widgets::dim_text(q);
    ImGui::SameLine(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX() - 70.0f * ctx.scale);
    if (ImGui::SmallButton("Expand")) ctx.shell.toggle_mini_panel();

    const ImVec2 size(96.0f * ctx.scale, 30.0f * ctx.scale);
    ImGui::BeginDisabled(!app.backend_usable());
    if (widgets::toggle_button("Stop", "Start", app.engine_on(), size, t.success)) app.toggle_engine();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!app.engine_on());
    if (widgets::toggle_button("Muted", "Mute", app.muted(), size, t.warning)) app.toggle_mute();
    ImGui::EndDisabled();
    ImGui::SameLine();
    widgets::level_meter("mini-in", app.telemetry().input_peak_dbfs, ImGui::GetContentRegionAvail().x, size.y * 0.5f);

    const std::vector<std::string> voices = app.voice_names();
    int vi = index_of(voices, app.voice_a());
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f - 4.0f * ctx.scale);
    if (widgets::combo_strings("##mv", vi, voices, "(voice)")) static_cast<void>(app.set_voice_a(voices[static_cast<std::size_t>(vi)]));
    ImGui::SameLine();
    const std::vector<std::string> presets = app.preset_names();
    int pi = index_of(presets, app.current_preset());
    ImGui::SetNextItemWidth(-1.0f);
    if (widgets::combo_strings("##mp", pi, presets, "(preset)")) static_cast<void>(app.load_preset(presets[static_cast<std::size_t>(pi)]));
}

void draw_section(ViewContext& ctx, Section section) {
    switch (section) {
        case Section::Live:      draw_live(ctx); break;
        case Section::Devices:   draw_devices(ctx); break;
        case Section::Voices:    draw_voices(ctx); break;
        case Section::Studio:    draw_studio(ctx); break;
        case Section::Translate: draw_translate(ctx); break;
        case Section::Catalogue: draw_catalogue(ctx); break;
        case Section::Telemetry: draw_telemetry(ctx); break;
        case Section::Settings:  draw_settings(ctx); break;
        case Section::Count:     break;
    }
}

} // namespace fluxvoice::frontend
