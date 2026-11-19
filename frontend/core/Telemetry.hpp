// Telemetry — the lock-free counters the UI reads, plus a small fixed ring for
// sparklines. Everything here is plain data; the worker and audio callback
// publish through atomics owned by EngineHost / DuplexRouter.
#pragma once

#include "pipeline/HostStats.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/TimingHistogram.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace fluxvoice::frontend {

/// One row of the Telemetry stage table: a model/DSP stage, or the whole hop
/// (the last row, "hop"). Times in ms; `share` is p95 / hop duration.
struct StageRow {
    const char*               name = "";
    fluxvoice::TimingSnapshot timing{};
    double                    p50_ms = 0.0, p95_ms = 0.0, max_ms = 0.0;
    double                    share  = 0.0;  // 0 when the hop duration is unknown
};

/// Rows of TelemetrySnapshot::stage_rows(): every PipelineStage, then "hop".
inline constexpr std::size_t kStageRowCount = fluxvoice::kPipelineStageCount + 1;

/// The conservation-identity check (MEMORY.md §2): the identity's `queued`
/// against the measured queue occupancy.
enum class IdentityVerdict : std::uint8_t { NotApplicable, Consistent, Mismatch };
const char* to_string(IdentityVerdict v);

struct IdentityCheck {
    IdentityVerdict verdict    = IdentityVerdict::NotApplicable;
    std::int64_t    queued     = 0;  // the identity
    std::int64_t    occupancy  = -1; // measured; -1 = unknown
    std::int64_t    difference = 0;  // occupancy - queued (0 when not applicable)
    std::int64_t    tolerance  = 0;  // samples
    /// Why the check does not apply ("" when it does).
    const char*     why_na     = "";
};

/// The "Estimated end-to-end latency" card. `total_ms` is the CLI's `lat_ms`
/// (the BUG-026 formula); compute time and the algorithmic latency are shown
/// beside it for information and are never part of the sum (HostStats.hpp).
struct LatencyCard {
    bool   bypass              = false;  // no model: queued + device terms only
    bool   valid               = true;   // false in system-playback loopback (independent clocks)
    double total_ms            = 0.0;
    double capture_device_ms   = 0.0;
    double queued_ms           = 0.0;
    double model_delay_ms      = 0.0;
    double playback_device_ms  = 0.0;
    double compute_p95_ms      = 0.0;    // information only
    double algorithmic_ms      = 0.0;    // contract: framing hop + lookahead + OLA; information only
    /// capture + queued + model delay + playback: equals total_ms.
    double terms_sum_ms() const { return capture_device_ms + queued_ms + model_delay_ms + playback_device_ms; }
};

struct TelemetrySnapshot {
    std::uint64_t frames           = 0;  // lifetime across streams
    /// Per-stream counters since the last queue discard (stop / reconfigure /
    /// recovery): the queued figure must not carry earlier streams' residue
    /// (BUG-024, MEMORY.md §2 "rebaseline after queue discards").
    std::uint64_t input_overruns   = 0;
    std::uint64_t output_underruns = 0;
    std::uint64_t output_drops     = 0;
    /// Samples held between capture and playback by conservation (MEMORY.md
    /// §2), equal to `stats.queued`: Mode A is the worker's full identity
    /// (prefilled + underruns - overruns - drops - skipped), Bypass is
    /// underruns - overruns - drops. Exact, never estimated from occupancy.
    std::int64_t  queued_samples   = 0;
    /// False in system-playback loopback mode: the capture and playback
    /// callbacks are independently clocked, so the identity does not hold
    /// there and the figure must be shown as unavailable.
    bool          queued_valid     = true;
    /// Lifetime totals across streams, for the Telemetry panel.
    std::uint64_t lifetime_input_overruns   = 0;
    std::uint64_t lifetime_output_underruns = 0;
    std::uint64_t lifetime_output_drops     = 0;
    std::uint32_t sample_rate      = 0;
    std::size_t   rss_bytes        = 0;

    // Worker-side meters (dBFS; -120 means silence / not running).
    float input_peak_dbfs  = -120.0f;
    float input_rms_dbfs   = -120.0f;
    float output_peak_dbfs = -120.0f;
    float output_rms_dbfs  = -120.0f;

    /// True while a model bundle (Mode A) drives the stream; false in Bypass.
    bool          mode_a           = false;

    /// Parameter snapshots pushed by the control thread, delivered to the
    /// worker, and accepted by the pipeline. "Applied" in the UI means
    /// `params_accepted == params_pushed`: delivery alone is not acceptance
    /// (`InferencePipeline::set_params` can refuse a snapshot). In Bypass the
    /// pipeline does not exist: accepted stays 0 and the UI says so.
    std::uint64_t params_pushed   = 0;
    std::uint64_t params_applied  = 0;  // delivered (drained by the worker)
    std::uint64_t params_accepted = 0;  // Mode A only
    std::uint64_t params_rejected = 0;  // Mode A only: snapshots the pipeline refused (lifetime)

    /// The `[stats]` snapshot: Mode A comes from ModeAHost::stats(); Bypass
    /// builds a legacy one (`vc` false, skipped = prefilled = 0, the
    /// rebaselined identity, BUG-024), never fabricating inference telemetry.
    /// Its counters equal the per-stream fields above.
    fluxvoice::HostStats stats{};
    /// The stream is Running (the identity check needs a live stream).
    bool stream_running = false;

    double queued_ms() const {
        return sample_rate == 0 ? 0.0 : 1000.0 * static_cast<double>(queued_samples) / sample_rate;
    }

    /// "42 ms", or "n/a (independent clocks)" when the identity does not hold.
    std::string queued_ms_text() const;

    /// Exactly the `[stats]` line src/main.cpp prints
    /// (fluxvoice::format_stats_line), parsed by tools/analyze_soak.py. Bypass
    /// prints the legacy line, as the CLI's diagnostic modes do.
    std::string stats_line(double t_seconds) const;

    /// Duration of one hop in ms (0 when unknown).
    double hop_ms() const;
    /// One row per PipelineStage (in enum order), then the whole hop ("hop").
    std::array<StageRow, kStageRowCount> stage_rows() const;
    double governor_target_ms() const;
    double governor_hard_cap_ms() const;
    /// `queued` against the measured occupancy; n/a in system-playback
    /// loopback, when the stream is not running, or when occupancy is unknown.
    IdentityCheck identity_check() const;
    LatencyCard latency_card() const;
};

/// Fixed-capacity ring of floats for sparklines (newest last).
template <std::size_t N>
class History {
public:
    static constexpr std::size_t capacity = N;

    void push(float v) {
        buf_[head_] = v;
        head_       = (head_ + 1) % N;
        if (count_ < N) ++count_;
    }

    std::size_t size() const { return count_; }

    /// Oldest-first copy, contiguous for plotting.
    std::array<float, N> ordered() const {
        std::array<float, N> out{};
        const std::size_t start = count_ < N ? 0 : head_;
        for (std::size_t i = 0; i < count_; ++i) out[i] = buf_[(start + i) % N];
        return out;
    }

    float latest() const { return count_ == 0 ? 0.0f : buf_[(head_ + N - 1) % N]; }
    float max() const {
        float m = 0.0f;
        for (std::size_t i = 0; i < count_; ++i) m = buf_[i] > m ? buf_[i] : m;
        return m;
    }

    void clear() { head_ = 0; count_ = 0; }

private:
    std::array<float, N> buf_{};
    std::size_t head_  = 0;
    std::size_t count_ = 0;
};

} // namespace fluxvoice::frontend
