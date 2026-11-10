// Live — the home panel: engine, mute, mode, quick pickers, device summary,
// meters, latency and the status card (frontend/README.md §4.1).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/TrayModel.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <array>
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

void draw_engine_card(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const ThemeColors& t = theme_colors();
    if (!widgets::begin_card("engine", "Engine")) { widgets::end_card(); return; }

    const ImVec2 big(200.0f * ctx.scale, 46.0f * ctx.scale);
    const bool usable = app.backend_usable();
    ImGui::BeginDisabled(!usable);
    if (widgets::toggle_button("Stop", "Start", app.engine_on(), big, t.success)) app.toggle_engine();
    ImGui::EndDisabled();
    if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", app.backend_status().state == BackendState::NullOnly
                                    ? "No audio hardware is reachable (null backend, offline mode)."
                                    : "No audio backend could be initialised.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!app.engine_on());
    if (widgets::toggle_button("Muted", "Mute", app.muted(), ImVec2(120.0f * ctx.scale, big.y), t.warning))
        app.toggle_mute();
    ImGui::EndDisabled();

    ImGui::SameLine(0.0f, 24.0f * ctx.scale);
    // Mode: segmented control.
    const bool translate = app.translate();
    if (widgets::toggle_button("Voice changer", "Voice changer", !translate, ImVec2(150.0f * ctx.scale, big.y), t.accent))
        static_cast<void>(app.set_translate(false));
    ImGui::SameLine(0.0f, 2.0f);
    const bool avail = app.translation().translator_available();
    ImGui::BeginDisabled(!avail);
    if (widgets::toggle_button("Translate", "Translate", translate, ImVec2(150.0f * ctx.scale, big.y), t.accent))
        static_cast<void>(app.set_translate(true));
    ImGui::EndDisabled();
    if (!avail && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", app.translation().unavailable_reason().c_str());

    ImGui::Spacing();
    widgets::dim_text_wrapped(app.engine_message().empty()
                                  ? "Start routes the selected microphone through FluxVoice to the selected output."
                                  : app.engine_message().c_str());
    widgets::end_card();
}

void draw_status_card(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const StatusSummary s = app.status_summary();
    if (s.state == StreamState::Running || s.state == StreamState::Idle) return;
    const ThemeColors& t = theme_colors();
    const ImVec4 colour = (s.state == StreamState::AwaitingUser || s.state == StreamState::Error) ? t.danger : t.warning;
    if (!widgets::begin_card("status", nullptr)) { widgets::end_card(); return; }
    widgets::pill(s.pill.c_str(), colour);
    ImGui::SameLine();
    ImGui::TextWrapped("%s", s.detail.c_str());
    if (s.state == StreamState::AwaitingUser || s.state == StreamState::Error) {
        if (ImGui::Button("Choose device")) app.navigate(Section::Devices);
        ImGui::SameLine();
        if (ImGui::Button("Retry")) static_cast<void>(app.start_engine());
    }
    widgets::end_card();
}

void draw_pickers_card(ViewContext& ctx) {
    AppModel& app = ctx.app;
    if (!widgets::begin_card("pickers", "Voice & preset")) { widgets::end_card(); return; }
    const std::vector<std::string> voices = app.voice_names();
    int vi = index_of(voices, app.voice_a());
    ImGui::SetNextItemWidth(260.0f * ctx.scale);
    if (widgets::combo_strings("Voice", vi, voices, app.active_bundle().empty() ? "(no bundle found)" : "(none)"))
        static_cast<void>(app.set_voice_a(voices[static_cast<std::size_t>(vi)]));
    ImGui::SameLine(0.0f, 24.0f * ctx.scale);
    const std::vector<std::string> presets = app.preset_names();
    int pi = index_of(presets, app.current_preset());
    ImGui::SetNextItemWidth(220.0f * ctx.scale);
    if (widgets::combo_strings("Preset", pi, presets, "(custom)"))
        static_cast<void>(app.load_preset(presets[static_cast<std::size_t>(pi)]));
    if (app.preset_modified()) {
        ImGui::SameLine();
        widgets::pill("modified", theme_colors().warning);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("All controls")) app.navigate(Section::Voices);
    widgets::end_card();
}

void draw_devices_card(ViewContext& ctx) {
    AppModel& app = ctx.app;
    if (!widgets::begin_card("devices", "Devices")) { widgets::end_card(); return; }
    if (widgets::kv_table_begin("devkv")) {
        const std::string cap_mode = app.capture_mode() == CaptureMode::SystemPlaybackLoopback ? " (system playback)" : "";
        widgets::kv_row("Capture", app.capture_name() + cap_mode);
        widgets::kv_row("Playback", app.playback_name());
        widgets::kv_table_end();
    }
    if (ImGui::Button("Change...")) app.navigate(Section::Devices);
    widgets::end_card();
}

void draw_meters_card(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const TelemetrySnapshot t = app.telemetry();
    if (!widgets::begin_card("meters", "Levels & latency")) { widgets::end_card(); return; }
    const float w = ImGui::GetContentRegionAvail().x - 90.0f * ctx.scale;
    ImGui::TextUnformatted("Input");
    ImGui::SameLine(80.0f * ctx.scale);
    widgets::level_meter("in", t.input_peak_dbfs, w, 14.0f * ctx.scale);
    ImGui::TextUnformatted("Output");
    ImGui::SameLine(80.0f * ctx.scale);
    widgets::level_meter("out", t.output_peak_dbfs, w, 14.0f * ctx.scale);
    ImGui::Spacing();

    char buf[96];
    std::snprintf(buf, sizeof buf, "Queued audio %.0f ms  |  %llu underruns  |  %llu overruns  |  %llu drops",
                  t.queued_ms(), static_cast<unsigned long long>(t.output_underruns),
                  static_cast<unsigned long long>(t.input_overruns), static_cast<unsigned long long>(t.output_drops));
    ImGui::TextUnformatted(buf);
    const std::array<float, 60> hist = app.underrun_history().ordered();
    const int n = static_cast<int>(app.underrun_history().size());
    ImGui::PlotLines("##underruns", hist.data(), n, 0, "underruns per second (last 60 s)", 0.0f,
                     app.underrun_history().max() > 0.0f ? app.underrun_history().max() : 1.0f,
                     ImVec2(ImGui::GetContentRegionAvail().x, 56.0f * ctx.scale));
    widgets::end_card();
}

} // namespace

void draw_live(ViewContext& ctx) {
    widgets::heading("Live");
    draw_engine_card(ctx);
    ImGui::Spacing();
    draw_status_card(ctx);
    draw_pickers_card(ctx);
    ImGui::Spacing();
    draw_devices_card(ctx);
    ImGui::Spacing();
    draw_meters_card(ctx);

    const auto& notices = ctx.app.notices();
    if (!notices.empty()) {
        ImGui::Spacing();
        widgets::dim_text("Recent messages");
        std::size_t shown = 0;
        for (auto it = notices.rbegin(); it != notices.rend() && shown < 3; ++it, ++shown) {
            const ImVec4 c = it->level == Notice::Level::Error ? theme_colors().danger
                             : it->level == Notice::Level::Warning ? theme_colors().warning : theme_colors().text_dim;
            ImGui::PushStyleColor(ImGuiCol_Text, c);
            ImGui::TextWrapped("%s", it->text.c_str());
            ImGui::PopStyleColor();
        }
    }
}

} // namespace fluxvoice::frontend
