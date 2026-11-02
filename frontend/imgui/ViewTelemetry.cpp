// Telemetry — counters, conservation identity, histories, [stats] line
// (frontend/README.md §4.7).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/Telemetry.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <array>
#include <cstdio>
#include <string>

namespace fluxvoice::frontend {

void draw_telemetry(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const TelemetrySnapshot t = app.telemetry();
    widgets::heading("Telemetry");

    if (widgets::begin_card("counters", "Counters (lock-free, read from the engine's atomics)")) {
        if (widgets::kv_table_begin("ctr")) {
            widgets::kv_row("Frames processed", std::to_string(t.frames));
            widgets::kv_row("Input overruns", std::to_string(t.input_overruns));
            widgets::kv_row("Output underruns", std::to_string(t.output_underruns));
            widgets::kv_row("Output drops", std::to_string(t.output_drops));
            char q[96];
            std::snprintf(q, sizeof q, "%lld samples = %.1f ms at %u Hz (underruns - overruns - drops)",
                          static_cast<long long>(t.queued_samples), t.queued_ms(), t.sample_rate);
            widgets::kv_row("Queued audio", q);
            widgets::kv_row("Resident memory", std::to_string(t.rss_bytes / 1024) + " KB");
            char p[64];
            std::snprintf(p, sizeof p, "%llu pushed, %llu applied", static_cast<unsigned long long>(t.params_pushed),
                          static_cast<unsigned long long>(t.params_applied));
            widgets::kv_row("Parameter snapshots", p);
            widgets::kv_row("Worker hops", std::to_string(app.engine().worker_hops()));
            char up[32];
            std::snprintf(up, sizeof up, "%.0f s", app.uptime_seconds());
            widgets::kv_row("UI uptime", up);
            widgets::kv_table_end();
        }
        widgets::dim_text_wrapped("Every underrun adds permanent delay until the latency governor drains it during silence; "
                                  "queue occupancy cannot show hardware clock drift (measure with fv_probe latency).");
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("hist", "Last 60 seconds")) {
        const std::array<float, 60> u = app.underrun_history().ordered();
        const std::array<float, 60> qh = app.queued_history().ordered();
        const int n = static_cast<int>(app.underrun_history().size());
        ImGui::PlotHistogram("##u", u.data(), n, 0, "underruns / s", 0.0f,
                             app.underrun_history().max() > 0.0f ? app.underrun_history().max() : 1.0f,
                             ImVec2(ImGui::GetContentRegionAvail().x, 70.0f * ctx.scale));
        ImGui::PlotLines("##q", qh.data(), static_cast<int>(app.queued_history().size()), 0, "queued ms", 0.0f,
                         app.queued_history().max() > 0.0f ? app.queued_history().max() * 1.2f : 1.0f,
                         ImVec2(ImGui::GetContentRegionAvail().x, 70.0f * ctx.scale));
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("stats", "[stats] line (tools/analyze_soak.py format)")) {
        const std::string line = t.stats_line(app.uptime_seconds());
        ImGui::TextWrapped("%s", line.c_str());
        if (ImGui::Button("Copy")) ctx.shell.set_clipboard(line);
        widgets::end_card();
    }
}

} // namespace fluxvoice::frontend
