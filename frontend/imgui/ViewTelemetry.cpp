// Telemetry — counters, conservation identity, histories, [stats] line
// (frontend/README.md §4.7).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/Paths.hpp"
#include "core/Telemetry.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace fluxvoice::frontend {

namespace {

std::string ms_cell(double ms) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f", ms);
    return b;
}

std::string samples_ms(std::int64_t samples, std::uint32_t rate) {
    char b[96];
    std::snprintf(b, sizeof b, "%lld samples = %.1f ms", static_cast<long long>(samples),
                  rate == 0 ? 0.0 : 1000.0 * static_cast<double>(samples) / rate);
    return b;
}

void draw_stage_table(ViewContext& ctx, const TelemetrySnapshot& t) {
    if (!widgets::begin_card("stages", "Compute time per hop (p50 / p95 / max)")) { widgets::end_card(); return; }
    if (!t.stats.vc) {
        widgets::dim_text_wrapped(t.mode_a ? "No timing yet: the engine is stopped or has not run a hop."
                                           : "Bypass - no model loaded: there are no model stages to time.");
        widgets::end_card();
        return;
    }
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("stage-table", 6, flags)) {
        ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 9.0f);
        ImGui::TableSetupColumn("hops");
        ImGui::TableSetupColumn("p50 ms");
        ImGui::TableSetupColumn("p95 ms");
        ImGui::TableSetupColumn("max ms");
        ImGui::TableSetupColumn("share of hop");
        ImGui::TableHeadersRow();
        for (const StageRow& r : t.stage_rows()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.name);
            const bool seen = r.timing.samples != 0;  // a stage that never ran (no translator) has no samples
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(std::to_string(r.timing.samples).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(seen ? ms_cell(r.p50_ms).c_str() : "-");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(seen ? ms_cell(r.p95_ms).c_str() : "-");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(seen ? ms_cell(r.max_ms).c_str() : "-");
            ImGui::TableNextColumn();
            char share[16];
            std::snprintf(share, sizeof share, "%.0f %%", r.share * 100.0);
            ImGui::TextUnformatted(seen ? share : "-");
        }
        ImGui::EndTable();
    }
    char hop[96];
    std::snprintf(hop, sizeof hop, "Hop = %.1f ms. Share = p95 / hop duration; the stages are parts of the hop row.",
                  t.hop_ms());
    widgets::dim_text_wrapped(hop);
    const std::array<float, 60> ch = ctx.app.compute_history().ordered();
    const float ceiling = std::max(static_cast<float>(t.hop_ms()), ctx.app.compute_history().max()) * 1.2f;
    ImGui::PlotLines("##compute", ch.data(), static_cast<int>(ctx.app.compute_history().size()), 0,
                     "whole-hop compute p95 (ms), last 60 s", 0.0f, ceiling > 0.0f ? ceiling : 1.0f,
                     ImVec2(ImGui::GetContentRegionAvail().x, 70.0f * ctx.scale));
    widgets::end_card();
}

void draw_identity_card(const TelemetrySnapshot& t) {
    if (!widgets::begin_card("identity", "Queue identity check")) { widgets::end_card(); return; }
    const IdentityCheck c = t.identity_check();
    if (widgets::kv_table_begin("identity-kv")) {
        widgets::kv_row("Identity (queued)", c.verdict == IdentityVerdict::NotApplicable && !t.queued_valid
                                                 ? std::string("n/a (independent clocks)")
                                                 : samples_ms(c.queued, t.sample_rate));
        widgets::kv_row("Occupancy (measured)", c.occupancy < 0 ? std::string("not measured")
                                                                : samples_ms(c.occupancy, t.sample_rate));
        if (c.verdict != IdentityVerdict::NotApplicable) {
            widgets::kv_row("Difference", std::to_string(c.difference) + " samples (tolerance +/- " +
                                              std::to_string(c.tolerance) + ")");
        }
        widgets::kv_table_end();
    }
    if (c.verdict == IdentityVerdict::NotApplicable) {
        widgets::pill((std::string("n/a - ") + c.why_na).c_str(), theme_colors().neutral);
    } else if (c.verdict == IdentityVerdict::Consistent) {
        widgets::pill("consistent", theme_colors().success);
    } else {
        widgets::pill("MISMATCH", theme_colors().danger);
    }
    widgets::dim_text_wrapped(t.mode_a ? "queued = prefilled + underruns - overruns - drops - skipped; occupancy = input queue + "
                                         "output queue + samples the worker holds. They agree at rest; a running stream may "
                                         "differ by one device period plus one hop."
                                       : "queued = underruns - overruns - drops; occupancy = input queue + output queue + samples "
                                         "the Bypass worker holds. A running stream may differ by one device period plus one "
                                         "block.");
    widgets::dim_text_wrapped("Queue occupancy cannot show hardware clock drift (measure it with fv_probe latency).");
    widgets::end_card();
}

} // namespace

void draw_telemetry(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const TelemetrySnapshot t = app.telemetry();
    widgets::heading("Telemetry");

    draw_latency_card(t, "telemetry-latency");
    ImGui::Spacing();
    draw_stage_table(ctx, t);
    ImGui::Spacing();

    if (widgets::begin_card("counters", "Counters (lock-free, read from the engine's atomics)")) {
        if (widgets::kv_table_begin("ctr")) {
            widgets::kv_row("Frames processed", std::to_string(t.frames));
            widgets::kv_row("Input overruns", std::to_string(t.input_overruns));
            widgets::kv_row("Output underruns", std::to_string(t.output_underruns));
            widgets::kv_row("Output drops", std::to_string(t.output_drops));
            widgets::kv_row("Overruns (lifetime)", std::to_string(t.lifetime_input_overruns));
            widgets::kv_row("Underruns (lifetime)", std::to_string(t.lifetime_output_underruns));
            widgets::kv_row("Drops (lifetime)", std::to_string(t.lifetime_output_drops));
            if (t.mode_a) {
                widgets::kv_row("Deadline misses", t.stats.vc ? std::to_string(t.stats.timing.misses) : std::string("-"));
                widgets::kv_row("Skipped samples (governor)", std::to_string(t.stats.skipped));
                widgets::kv_row("Prefilled samples (startup)", std::to_string(t.stats.prefilled));
                char g[96];
                std::snprintf(g, sizeof g, "target %.1f ms, hard cap %.1f ms", t.governor_target_ms(), t.governor_hard_cap_ms());
                widgets::kv_row("Latency governor", g);
            } else {
                widgets::kv_row("Skipped / prefilled", "n/a (Bypass has no governor)");
            }
            widgets::kv_row("Queued audio", t.queued_valid ? samples_ms(t.queued_samples, t.sample_rate) : std::string("n/a (independent clocks)"));
            widgets::kv_row("Resident memory", std::to_string(t.rss_bytes / 1024) + " KB");
            char p[128];
            if (t.mode_a)
                std::snprintf(p, sizeof p, "%llu pushed, %llu delivered, %llu accepted, %llu rejected",
                              static_cast<unsigned long long>(t.params_pushed), static_cast<unsigned long long>(t.params_applied),
                              static_cast<unsigned long long>(t.params_accepted), static_cast<unsigned long long>(t.params_rejected));
            else
                std::snprintf(p, sizeof p, "%llu pushed (Bypass: model controls inactive)",
                              static_cast<unsigned long long>(t.params_pushed));
            widgets::kv_row("Parameter snapshots", p);
            widgets::kv_row("Worker hops", std::to_string(app.engine().worker_hops()));
            char up[32];
            std::snprintf(up, sizeof up, "%.0f s", app.uptime_seconds());
            widgets::kv_row("UI uptime", up);
            widgets::kv_table_end();
        }
        widgets::dim_text_wrapped("Every underrun adds permanent delay until the latency governor drains it during silence.");
        widgets::end_card();
    }
    ImGui::Spacing();

    draw_identity_card(t);
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
        const std::string line = t.stats_line(app.stream_seconds());
        ImGui::TextWrapped("%s", line.c_str());
        if (ImGui::Button("Copy")) ctx.shell.set_clipboard(line);
        const std::string log = path_to_utf8(app.stats_log_file());
        if (app.stats_log_active())
            widgets::dim_text_wrapped(("Writing one line per second to " + log + " (" + std::to_string(app.stats_log_lines()) +
                                       " lines so far).").c_str());
        else if (app.settings().write_stats_log)
            widgets::dim_text_wrapped(("Stats log is on; it opens " + log + " when a stream runs.").c_str());
        else
            widgets::dim_text_wrapped("Stats log is off (Settings > Write stats log).");
        widgets::end_card();
    }
}

} // namespace fluxvoice::frontend
