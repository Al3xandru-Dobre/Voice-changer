// Devices — direction-specific selectors over immutable snapshots, capture
// mode, feedback preview, apply, probe and the stream status table
// (frontend/README.md §4.2; Milestone 1.5 rules).
#include "imgui/Views.hpp"

#include "audio/AudioTypes.hpp"
#include "core/AppModel.hpp"
#include "core/DeviceSelectorModel.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace fluxvoice::frontend {

namespace {

struct ProbeState {
    std::optional<Result<EndpointInfo>> capture;
    std::optional<Result<EndpointInfo>> playback;
};
ProbeState g_probe;

std::vector<std::string> labels(const DeviceRows& rows) {
    std::vector<std::string> out;
    for (const DeviceRow& r : rows.rows) out.push_back(r.label);
    return out;
}

void draw_probe(const Result<EndpointInfo>& r) {
    if (!is_ok(r)) {
        widgets::notice(explain_error(std::get<AudioError>(r)).c_str(), theme_colors().danger);
        return;
    }
    const EndpointInfo& info = std::get<EndpointInfo>(r);
    ImGui::Text("%s%s", info.name.c_str(), info.is_default ? "  (default)" : "");
    if (info.native_formats.empty()) {
        widgets::dim_text("The backend reports no native formats for this endpoint.");
        return;
    }
    if (ImGui::BeginTable("fmts", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Sample rate");
        ImGui::TableSetupColumn("Channels");
        ImGui::TableSetupColumn("Format id");
        ImGui::TableHeadersRow();
        for (const NativeFormat& f : info.native_formats) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (f.sample_rate == 0) ImGui::TextUnformatted("any"); else ImGui::Text("%u Hz", f.sample_rate);
            ImGui::TableNextColumn();
            if (f.channels == 0) ImGui::TextUnformatted("any"); else ImGui::Text("%u", f.channels);
            ImGui::TableNextColumn();
            ImGui::Text("%d", f.format);
        }
        ImGui::EndTable();
    }
}

const char* yes_no_unknown(Capability c) {
    switch (c) {
        case Capability::Supported:   return "supported";
        case Capability::Unsupported: return "unsupported";
        case Capability::Unknown:     return "unknown (not probed)";
    }
    return "?";
}

} // namespace

void draw_devices(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const ThemeColors& t = theme_colors();
    widgets::heading("Devices");

    // Catalogue header: backend, generation, stale flag, refresh.
    if (widgets::begin_card("catalog", nullptr)) {
        const DeviceSnapshot& snap = app.snapshot();
        const BackendStatus bs = app.backend_status();
        ImGui::Text("Backend: %s", snap.backend.empty() ? bs.backend.c_str() : snap.backend.c_str());
        ImGui::SameLine();
        ImGui::Text("| snapshot generation %llu", static_cast<unsigned long long>(snap.generation));
        if (app.snapshot_stale()) {
            ImGui::SameLine();
            widgets::pill("stale: last refresh failed, showing the previous list", t.warning);
        }
        if (bs.state == BackendState::NullOnly) {
            ImGui::SameLine();
            widgets::pill("offline: null backend", t.danger);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Refresh")) {
            app.refresh_devices();
            g_probe = ProbeState{};
        }
        ImGui::SameLine();
        widgets::help_marker("The list is an immutable snapshot. Selections are submitted as endpoint key + "
                             "snapshot generation, never as a row number, so a device plugged in meanwhile "
                             "can never redirect your choice.");
        const AudioCapabilities caps = app.capabilities();
        ImGui::Text("System-playback loopback capture: %s", yes_no_unknown(caps.native_loopback));
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("select", "Selection")) {
        // Capture mode.
        int mode = app.capture_mode() == CaptureMode::SystemPlaybackLoopback ? 1 : 0;
        if (ImGui::RadioButton("Microphone", mode == 0)) app.set_capture_mode(CaptureMode::OrdinaryMicrophone);
        ImGui::SameLine();
        ImGui::BeginDisabled(!app.loopback_supported());
        if (ImGui::RadioButton("System playback (loopback)", mode == 1)) app.set_capture_mode(CaptureMode::SystemPlaybackLoopback);
        ImGui::EndDisabled();
        if (!app.loopback_supported() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Native loopback is a WASAPI feature; this backend reports it as %s.",
                              yes_no_unknown(app.capabilities().native_loopback));
        ImGui::Spacing();

        const DeviceRows cap_rows  = app.capture_rows();
        const DeviceRows play_rows = app.playback_rows();
        const std::vector<std::string> cap_labels  = labels(cap_rows);
        const std::vector<std::string> play_labels = labels(play_rows);
        const auto cap_idx  = row_for_selection(cap_rows, app.capture_selection());
        const auto play_idx = row_for_selection(play_rows, app.playback_selection());
        int ci = cap_idx ? static_cast<int>(*cap_idx) : -1;
        int pi = play_idx ? static_cast<int>(*play_idx) : -1;

        ImGui::SetNextItemWidth(-160.0f * ctx.scale);
        if (widgets::combo_strings(app.capture_mode() == CaptureMode::SystemPlaybackLoopback ? "Capture (playback endpoint)" : "Capture",
                                   ci, cap_labels, "(device no longer listed - choose again)")) {
            app.select_capture_row(static_cast<std::size_t>(ci));
            g_probe.capture.reset();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Inspect##cap")) g_probe.capture = app.probe(app.capture_selection(), EndpointDirection::Capture);
        if (g_probe.capture) draw_probe(*g_probe.capture);

        ImGui::SetNextItemWidth(-160.0f * ctx.scale);
        if (widgets::combo_strings("Playback", pi, play_labels, "(device no longer listed - choose again)")) {
            app.select_playback_row(static_cast<std::size_t>(pi));
            g_probe.playback.reset();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Inspect##play")) g_probe.playback = app.probe(app.playback_selection(), EndpointDirection::Playback);
        if (g_probe.playback) draw_probe(*g_probe.playback);

        ImGui::Spacing();
        const std::optional<std::string> fb = app.feedback_warning();
        if (fb) widgets::notice(fb->c_str(), t.danger);
        if (!cap_idx || !play_idx)
            widgets::notice("A remembered device is no longer listed. Pick it again; FluxVoice never substitutes another device.", t.warning);

        ImGui::BeginDisabled(fb.has_value() || !app.backend_usable());
        if (ImGui::Button(app.engine_on() ? "Apply and restart stream" : "Apply and start", ImVec2(220.0f * ctx.scale, 0.0f)))
            static_cast<void>(app.apply_devices());
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!app.engine_on());
        if (ImGui::Button("Stop")) app.stop_engine();
        ImGui::EndDisabled();
        if (!app.engine_message().empty()) {
            ImGui::SameLine();
            widgets::dim_text(app.engine_message().c_str());
        }
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("status", "Stream status")) {
        const StreamStatus& st = app.engine().status();
        const StreamInfo& si   = app.engine().stream_info();
        const DeviceSnapshot& snap = app.snapshot();
        auto policy = [&](const DeviceSelection& sel, EndpointDirection dir) {
            if (std::holds_alternative<FollowSystemDefault>(sel)) return std::string("follow system default");
            return "explicit: " + describe_selection(sel, snap, dir, dir == EndpointDirection::Capture ? app.capture_mode()
                                                                                                        : CaptureMode::OrdinaryMicrophone);
        };
        auto rate = [](std::uint32_t hz) { return hz == 0 ? std::string("-") : std::to_string(hz) + " Hz"; };
        auto period = [](std::uint32_t frames) { return frames == 0 ? std::string("-") : std::to_string(frames) + " frames"; };
        if (widgets::kv_table_begin("stkv")) {
            widgets::kv_row("State", std::string(to_string(st.state)));
            widgets::kv_row("Backend", st.backend.empty() ? si.backend : st.backend);
            widgets::kv_row("Requested capture", policy(st.requested_capture, EndpointDirection::Capture));
            widgets::kv_row("Requested playback", policy(st.requested_playback, EndpointDirection::Playback));
            widgets::kv_row("Opened capture", si.capture_name.empty() ? "-" : si.capture_name);
            widgets::kv_row("Opened playback", si.playback_name.empty() ? "-" : si.playback_name);
            widgets::kv_row("Capture mode", std::string(to_string(st.capture_mode)));
            widgets::kv_row("Client sample rate", rate(si.client_sample_rate));
            widgets::kv_row("Backend rates", rate(si.capture_internal_rate) + " capture, " + rate(si.playback_internal_rate) + " playback");
            widgets::kv_row("Backend periods", period(si.capture_internal_period) + " capture, " + period(si.playback_internal_period) + " playback");
            widgets::kv_row("Resolved against generation", std::to_string(st.selection_generation));
            widgets::kv_row("Callback progress", std::to_string(st.callback_progress) + " callbacks");
            widgets::kv_row("Retry attempt", std::to_string(st.retry_attempt));
            if (st.last_error) widgets::kv_row("Last error", explain_error(*st.last_error));
            widgets::kv_table_end();
        }
        widgets::end_card();
    }
}

} // namespace fluxvoice::frontend
