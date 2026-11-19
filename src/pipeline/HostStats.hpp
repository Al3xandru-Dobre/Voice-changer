// HostStats — one `[stats]` snapshot, its machine-readable line and the
// end-to-end latency estimate (M4 Stage 06). The CLI (`fluxvoice`) and the GUI
// (`frontend/core/Telemetry`) both format through format_stats_line(), so a
// GUI soak log is the same format tools/analyze_soak.py already parses, and
// both show the same latency number.
#pragma once

#include "audio/AudioTypes.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "pipeline/TimingHistogram.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <sstream>
#include <string>

namespace fluxvoice {

/// End-to-end latency ESTIMATE (the BUG-026 formula): capture device period
/// + queued samples (queues and worker; the framing hop is paid here, once)
/// + model delay (lookahead + OLA tail, outside the queues) + playback device
/// period. Compute time is not a term: it runs while the queued audio plays.
/// The measured value comes only from `fv_probe latency` (M4 Stage 12).
struct LatencyEstimate {
    double capture_device_ms  = 0.0;
    double playback_device_ms = 0.0;
    double queued_ms          = 0.0;  // max(queued, 0)
    double model_delay_ms     = 0.0;
    /// The sum, evaluated exactly as the CLI's `lat_ms` always was, so the
    /// printed value is byte-identical (a different summation order may
    /// change the last printed digit).
    double total_ms           = 0.0;
};

inline LatencyEstimate estimate_latency(const PipelineGeometry& g, std::int64_t queued,
                                        const StreamInfo& stream) noexcept {
    const auto period_ms = [](std::uint32_t frames, std::uint32_t rate) {
        return rate ? 1000.0 * frames / rate : 0.0;
    };
    LatencyEstimate e;
    if (g.sample_rate == 0) return e;
    const double rate  = static_cast<double>(g.sample_rate);
    const double held  = static_cast<double>(std::max<std::int64_t>(queued, 0));
    const double model = static_cast<double>(g.model_delay_samples());
    e.capture_device_ms  = period_ms(stream.capture_internal_period, stream.capture_internal_rate);
    e.playback_device_ms = period_ms(stream.playback_internal_period, stream.playback_internal_rate);
    e.queued_ms          = 1000.0 * held / rate;
    e.model_delay_ms     = 1000.0 * model / rate;
    e.total_ms = 1000.0 * (model + held) / rate + e.capture_device_ms + e.playback_device_ms;
    return e;
}

/// Glitch counters and derived values at one instant. `vc` is set only by a
/// live neural host (ModeAHost::stats()); the diagnostic modes and the GUI's
/// Bypass leave it false and never fabricate inference telemetry.
struct HostStats {
    std::uint64_t frames = 0, overruns = 0, underruns = 0, drops = 0;
    std::uint64_t skipped = 0, prefilled = 0;
    // prefilled + underruns - overruns - drops - skipped (MEMORY.md §2).
    std::int64_t queued = 0;
    std::size_t rss = 0;  // bytes

    bool vc = false;
    TimingSnapshot timing{};  // whole hop
    std::uint64_t errors = 0;
    double lat_ms = 0.0;      // == latency.total_ms
    bool s2ut = false;        // translator models loaded (Mode B capable)
    S2utTelemetry tel{};

    // Not printed on the line: for the GUI cards and the CLI summary.
    std::array<TimingSnapshot, kPipelineStageCount> stages{};
    LatencyEstimate latency{};
    std::uint32_t sample_rate = 0;
    std::size_t hop = 0;                 // samples per hop
    std::size_t algorithmic_samples = 0; // framing hop + lookahead + OLA (contract)
    std::size_t governor_target = 0;     // samples
    std::size_t governor_hard_cap = 0;   // samples
    /// Measured occupancy for the identity check: input queue + output queue +
    /// samples the worker holds (read, not yet written, dropped or skipped);
    /// -1 when unknown. It equals `queued` at quiescence; while running the two
    /// are sampled at different instants, so they may differ by up to one
    /// device period plus one hop (`occupancy_tolerance`) — a larger
    /// difference is an accounting error (a missing term, a missed rebaseline).
    std::int64_t occupancy = -1;
    std::int64_t occupancy_tolerance = 0;
};

/// One machine-readable line; tools/analyze_soak.py parses these.
inline std::string format_stats_line(double t, const HostStats& s) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(1);
    o << "[stats] t=" << t << " frames=" << s.frames << " overruns=" << s.overruns
      << " underruns=" << s.underruns << " drops=" << s.drops << " queued=" << s.queued
      << " skipped=" << s.skipped << " prefilled=" << s.prefilled
      << " rss_kb=" << s.rss / 1024;
    if (s.vc)
        o << " inf_p50_us=" << s.timing.p50_us << " inf_p95_us=" << s.timing.p95_us
          << " inf_max_us=" << s.timing.max_us << " misses=" << s.timing.misses
          << " errors=" << s.errors << " lat_ms=" << s.lat_ms;
    // Same fields on every line of a run that has translator models (analyze_soak --s2ut).
    if (s.s2ut)
        o << " s2ut_r=" << s.tel.reads << " s2ut_w=" << s.tel.writes
          << " s2ut_f=" << s.tel.flushes << " lag_ms=" << s.tel.lag_ms()
          << " s2ut_utt=" << s.tel.utterances << " s2ut_cap=" << s.tel.cap_ends;
    return o.str();
}

} // namespace fluxvoice
