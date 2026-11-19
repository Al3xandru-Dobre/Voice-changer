// Output-only latency recovery. The worker skips newly generated PCM, never
// input/model steps or samples already owned by the callback's SPSC consumer.
//
// Mode B (Milestone 3 / Stage 08, research §9.2): while a translation is in
// flight the pipeline reports it through set_translation_state() once per hop.
//   * cushion — `t_src > 0` or translated frames still queued: the effective
//     target (and the hard cap above it) rises by `burst_cushion_samples`, so
//     the compute burst of a translator chunk step or an adaptive/flush
//     double write is not mistaken for queue drift;
//   * hold — this chunk step decoded (kWrite/kFlush) or translated frames
//     still queued: soft (quiet-output) dropping is suppressed so buffered
//     translated speech is never discarded before it is heard. The hard cap
//     still applies.
// Excess silence is therefore trimmed only once the policy is back to idle.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace fluxvoice {

struct GovernorConfig {
    std::uint32_t sample_rate = 16000;
    std::size_t hop = 320;
    std::size_t target_samples = 560;
    std::size_t sustained_hops = 3;
    float quiet_peak = 0.001f;
    // Mode B only (0 = Mode A): extra queue allowance while translating.
    // InferencePipeline uses 3 translation chunks (240 ms at 80 ms chunks).
    std::size_t burst_cushion_samples = 0;
};

// Single-duplex conservation, including startup samples not captured by the
// callback. Callers must start a new baseline after discarding stream queues.
inline std::int64_t queued_samples(std::uint64_t underruns, std::uint64_t overruns,
                                   std::uint64_t drops, std::uint64_t skipped = 0,
                                   std::uint64_t prefilled = 0) noexcept {
    return static_cast<std::int64_t>(prefilled) + static_cast<std::int64_t>(underruns) -
           static_cast<std::int64_t>(overruns) - static_cast<std::int64_t>(drops) -
           static_cast<std::int64_t>(skipped);
}

class LatencyGovernor {
public:
    explicit LatencyGovernor(GovernorConfig cfg) : cfg_(cfg) {
        if (cfg.sample_rate < 8000 || cfg.sample_rate > 192000 ||
            cfg.hop == 0 || cfg.hop > cfg.sample_rate ||
            cfg.target_samples > 10 * cfg.sample_rate || cfg.sustained_hops == 0 ||
            cfg.burst_cushion_samples > 10 * cfg.sample_rate ||
            !std::isfinite(cfg.quiet_peak) || cfg.quiet_peak < 0.0f)
            throw std::invalid_argument("invalid latency governor configuration");
    }

    // Master budget queue occupancy: framing floor + 50%-hop compute + jitter
    // margin (5 ms for 20/40 ms hops, 10 ms for >=80 ms hops). Model lookahead,
    // OLA tail and device periods live outside the queues.
    static std::size_t default_target(std::size_t queue_floor_samples, std::size_t hop,
                                      std::uint32_t rate) noexcept {
        const std::size_t margin_ms = std::uint64_t{hop} * 1000 >= 80 * rate ? 10 : 5;
        return queue_floor_samples + (hop + 1) / 2 + (rate * margin_ms + 999) / 1000;
    }

    // Startup prefill / Mode A target; see effective_target_samples().
    std::size_t target_samples() const noexcept { return cfg_.target_samples; }
    std::size_t burst_cushion_samples() const noexcept { return cfg_.burst_cushion_samples; }
    // Target the next skip_output() call works against: target_samples()
    // plus the burst cushion while a Mode B translation is in flight.
    std::size_t effective_target_samples() const noexcept {
        return cfg_.target_samples + (cushion_ ? cfg_.burst_cushion_samples : 0);
    }
    std::size_t hard_cap_samples() const noexcept {
        return effective_target_samples() + (std::size_t{cfg_.sample_rate} * 150 + 999) / 1000;
    }

    // Mode B hint for the following skip_output() calls (see header comment).
    // InferencePipeline sets it once per hop; Mode A never calls it.
    void set_translation_state(bool cushion, bool hold) noexcept {
        cushion_ = cushion;
        hold_ = hold;
    }
    bool cushion_active() const noexcept { return cushion_; }
    bool hold_active() const noexcept { return hold_; }

    // `queued` includes queues and in-flight worker audio, BEFORE this output
    // is emitted. Return the prefix length the caller must NOT enqueue; count
    // exactly that many in `skipped`. At most one hop is removed per call, so
    // a large backlog drains over several hops, even above the hard cap.
    std::size_t skip_output(std::int64_t queued, std::span<const float> output,
                             bool gate_closed = false) noexcept {
        const std::size_t target = effective_target_samples();
        const bool hard = queued > static_cast<std::int64_t>(hard_cap_samples());
        if (!hard && queued <= static_cast<std::int64_t>(target + cfg_.hop)) {
            high_hops_ = 0;
            return 0;
        }
        if (high_hops_ < cfg_.sustained_hops) ++high_hops_;
        if (!hard && (hold_ || high_hops_ < cfg_.sustained_hops || !quiet(output, gate_closed)))
            return 0;
        const auto excess = static_cast<std::size_t>(queued - static_cast<std::int64_t>(target));
        return std::min({output.size(), cfg_.hop, excess});
    }

    // Also clears the Mode B hint (a restarted stream starts idle).
    void reset() noexcept {
        high_hops_ = 0;
        cushion_ = false;
        hold_ = false;
    }

private:
    bool quiet(std::span<const float> output, bool gate_closed) const noexcept {
        if (gate_closed) return true;
        for (float x : output)
            if (!std::isfinite(x) || std::abs(x) > cfg_.quiet_peak) return false;
        return true;
    }

    GovernorConfig cfg_;
    std::size_t high_hops_ = 0;
    bool cushion_ = false;
    bool hold_ = false;
};

} // namespace fluxvoice
