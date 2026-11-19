// Telemetry — [stats] line, stage table, identity check and latency card
// (frontend/README.md §4.7).
#include "core/Telemetry.hpp"

#include "pipeline/HostStats.hpp"
#include "pipeline/InferencePipeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace fluxvoice::frontend {

namespace {

double us_to_ms(std::uint64_t us) { return static_cast<double>(us) / 1000.0; }

double samples_to_ms(std::size_t samples, std::uint32_t rate) {
    return rate == 0 ? 0.0 : 1000.0 * static_cast<double>(samples) / static_cast<double>(rate);
}

} // namespace

const char* to_string(IdentityVerdict v) {
    switch (v) {
        case IdentityVerdict::NotApplicable: return "n/a";
        case IdentityVerdict::Consistent:    return "consistent";
        case IdentityVerdict::Mismatch:      return "MISMATCH";
    }
    return "n/a";
}

std::string TelemetrySnapshot::queued_ms_text() const {
    if (!queued_valid) return "n/a (independent clocks)";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f ms", queued_ms());
    return buf;
}

std::string TelemetrySnapshot::stats_line(double t_seconds) const {
    return fluxvoice::format_stats_line(t_seconds, stats);
}

double TelemetrySnapshot::hop_ms() const {
    return samples_to_ms(stats.hop, stats.sample_rate != 0 ? stats.sample_rate : sample_rate);
}

std::array<StageRow, kStageRowCount> TelemetrySnapshot::stage_rows() const {
    const double hop = hop_ms();
    const auto fill = [hop](StageRow& r, const char* name, const fluxvoice::TimingSnapshot& t) {
        r.name   = name;
        r.timing = t;
        r.p50_ms = us_to_ms(t.p50_us);
        r.p95_ms = us_to_ms(t.p95_us);
        r.max_ms = us_to_ms(t.max_us);
        r.share  = hop > 0.0 ? r.p95_ms / hop : 0.0;
    };
    std::array<StageRow, kStageRowCount> rows{};
    for (std::size_t i = 0; i < fluxvoice::kPipelineStageCount; ++i)
        fill(rows[i], fluxvoice::to_string(static_cast<fluxvoice::PipelineStage>(i)), stats.stages[i]);
    fill(rows[fluxvoice::kPipelineStageCount], "hop", stats.timing);
    return rows;
}

double TelemetrySnapshot::governor_target_ms() const {
    return samples_to_ms(stats.governor_target, stats.sample_rate != 0 ? stats.sample_rate : sample_rate);
}

double TelemetrySnapshot::governor_hard_cap_ms() const {
    return samples_to_ms(stats.governor_hard_cap, stats.sample_rate != 0 ? stats.sample_rate : sample_rate);
}

IdentityCheck TelemetrySnapshot::identity_check() const {
    IdentityCheck c;
    c.queued    = stats.queued;
    c.occupancy = stats.occupancy;
    c.tolerance = stats.occupancy_tolerance;
    if (!queued_valid) {
        c.why_na = "independent clocks (system-playback loopback)";
        return c;
    }
    if (!stream_running) {
        c.why_na = "stream not running";
        return c;
    }
    if (stats.occupancy < 0) {
        c.why_na = "occupancy not measured";
        return c;
    }
    c.difference = stats.occupancy - stats.queued;
    const std::int64_t magnitude = c.difference < 0 ? -c.difference : c.difference;
    c.verdict = magnitude <= stats.occupancy_tolerance ? IdentityVerdict::Consistent : IdentityVerdict::Mismatch;
    return c;
}

LatencyCard TelemetrySnapshot::latency_card() const {
    LatencyCard c;
    c.bypass             = !mode_a;
    c.valid              = queued_valid;
    c.total_ms           = stats.latency.total_ms;  // the CLI's lat_ms, nothing added
    c.capture_device_ms  = stats.latency.capture_device_ms;
    c.queued_ms          = stats.latency.queued_ms;
    c.model_delay_ms     = stats.latency.model_delay_ms;
    c.playback_device_ms = stats.latency.playback_device_ms;
    c.compute_p95_ms     = us_to_ms(stats.timing.p95_us);
    c.algorithmic_ms     = samples_to_ms(stats.algorithmic_samples, stats.sample_rate != 0 ? stats.sample_rate : sample_rate);
    return c;
}

} // namespace fluxvoice::frontend
