// Settings — window/tray behaviour, appearance, directories, device policy,
// tray status, about (frontend/README.md §4.8).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/Settings.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <cstdio>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace {

struct DirField {
    char buf[512] = "";
    std::string mirror;
};
DirField g_models, g_presets, g_studio, g_index;

void dir_row(ViewContext& ctx, const char* label, std::string& value, DirField& f, bool& changed) {
    if (value != f.mirror) {
        std::snprintf(f.buf, sizeof f.buf, "%s", value.c_str());
        f.mirror = value;
    }
    ImGui::PushID(label);
    widgets::dim_text(label);
    ImGui::SetNextItemWidth(-110.0f * ctx.scale);
    if (ImGui::InputText("##dir", f.buf, sizeof f.buf)) {
        value    = f.buf;
        f.mirror = value;
        changed  = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Browse...")) {
        std::string* target = &value;
        AppModel* app = &ctx.app;
        ctx.shell.pick_folder([target, app](const std::vector<std::string>& paths) {
            if (paths.empty()) return;
            *target = paths.front();
            static_cast<void>(app->save_settings());
        });
    }
    ImGui::PopID();
}

} // namespace

void draw_settings(ViewContext& ctx) {
    AppModel& app = ctx.app;
    Settings& s = app.settings();
    const ThemeColors& t = theme_colors();
    bool changed = false;
    bool appearance_changed = false;
    widgets::heading("Settings");

    if (widgets::begin_card("tray", "Window & tray")) {
        const bool tray_ok = ctx.shell.tray_available();
        changed |= ImGui::Checkbox("Start minimised to the tray", &s.start_minimized);
        ImGui::BeginDisabled(!tray_ok);
        changed |= ImGui::Checkbox("Closing the window hides it to the tray (Quit from the tray or File menu)", &s.close_to_tray);
        ImGui::EndDisabled();
        changed |= ImGui::Checkbox("Start the engine on launch (with the remembered devices)", &s.start_engine_on_launch);
        changed |= ImGui::Checkbox("Mini panel stays on top of other windows", &s.mini_panel_on_top);
        if (tray_ok) widgets::pill("System tray: available", t.success);
        else widgets::notice(("System tray unavailable on this desktop: " + ctx.shell.tray_unavailable_reason() +
                              ". Closing the window quits; start minimised opens a minimised window.").c_str(), t.warning);
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("appearance", "Appearance")) {
        int theme = static_cast<int>(s.theme);
        const std::vector<std::string> themes = {"Dark", "Light", "Follow system"};
        ImGui::SetNextItemWidth(200.0f * ctx.scale);
        if (widgets::combo_strings("Theme", theme, themes)) {
            s.theme = static_cast<Theme>(theme);
            changed = appearance_changed = true;
        }
        ImGui::SetNextItemWidth(200.0f * ctx.scale);
        if (ImGui::SliderFloat("UI scale", &s.ui_scale, 0.75f, 2.0f, "%.2f x", ImGuiSliderFlags_AlwaysClamp)) changed = true;
        if (ImGui::IsItemDeactivatedAfterEdit()) appearance_changed = true;
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("dirs", "Directories")) {
        dir_row(ctx, "Models (bundles)", s.models_dir, g_models, changed);
        dir_row(ctx, "User presets", s.user_preset_dir, g_presets, changed);
        dir_row(ctx, "Voice Studio projects", s.studio_dir, g_studio, changed);
        dir_row(ctx, "Extra catalogue index.json", s.catalogue_index, g_index, changed);
        widgets::dim_text_wrapped(("Empty fields use the defaults: models " + app.paths().models_dir.string() + "; presets " +
                                   app.paths().user_preset_dir.string() + "; studio " + app.paths().studio_dir.string() +
                                   ". Directory changes take effect after a restart.").c_str());
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("policy", "Remembered device policy")) {
        if (widgets::kv_table_begin("pol")) {
            widgets::kv_row("Capture", s.capture_policy);
            widgets::kv_row("Playback", s.playback_policy);
            widgets::kv_row("Capture mode", s.capture_loopback ? "system-playback loopback" : "microphone");
            widgets::kv_table_end();
        }
        widgets::dim_text_wrapped("\"default\" follows the OS default endpoint. A named device is a hint that is re-resolved "
                                  "against the current device list at start; if it is missing or ambiguous you are asked to "
                                  "choose again. FluxVoice never substitutes a device (Milestone 1.5, decision 4).");
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("about", "About")) {
        ImGui::TextWrapped("%s", ctx.shell.about_text().c_str());
        if (widgets::kv_table_begin("about")) {
            widgets::kv_row("Settings file", app.paths().settings_file().string());
            widgets::kv_row("Audio backend", app.backend_status().backend.empty() ? "-" : app.backend_status().backend);
            widgets::kv_table_end();
        }
        widgets::end_card();
    }

    if (changed) static_cast<void>(app.save_settings());
    if (appearance_changed) ctx.shell.apply_appearance();
}

} // namespace fluxvoice::frontend
