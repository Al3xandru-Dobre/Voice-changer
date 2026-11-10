// Output-only latency recovery. The worker skips newly generated PCM, never
// input/model steps or samples already owned by the callback's SPSC consumer.
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
            !std::isfinite(cfg.quiet_peak) || cfg.quiet_peak < 0.0f)
            throw std::invalid_argument("invalid latency governor configuration");
    }

    // Master budget: framing/lookahead/OLA + 50%-hop compute + jitter margin
    // (5 ms for 20/40 ms hops, 10 ms for >=80 ms hops). Device periods are separate.
    static std::size_t default_target(std::size_t algorithmic_samples, std::size_t hop,
                                      std::uint32_t rate) noexcept {
        const std::size_t margin_ms = std::uint64_t{hop} * 1000 >= 80 * rate ? 10 : 5;
        return algorithmic_samples + (hop + 1) / 2 + (rate * margin_ms + 999) / 1000;
    }

    std::size_t target_samples() const noexcept { return cfg_.target_samples; }
    std::size_t hard_cap_samples() const noexcept {
        return cfg_.target_samples + (std::size_t{cfg_.sample_rate} * 150 + 999) / 1000;
    }

    // `queued` includes queues and in-flight worker audio, BEFORE this output
    // is emitted. Return the prefix length the caller must NOT enqueue; count
    // exactly that many in `skipped`. At most one hop is removed per call, so
    // a large backlog drains over several hops, even above the hard cap.
    std::size_t skip_output(std::int64_t queued, std::span<const float> output,
                             bool gate_closed = false) noexcept {
        const bool hard = queued > static_cast<std::int64_t>(hard_cap_samples());
        if (!hard && queued <= static_cast<std::int64_t>(cfg_.target_samples + cfg_.hop)) {
            high_hops_ = 0;
            return 0;
        }
        if (high_hops_ < cfg_.sustained_hops) ++high_hops_;
        if (!hard && (high_hops_ < cfg_.sustained_hops || !quiet(output, gate_closed)))
            return 0;
        const auto excess = static_cast<std::size_t>(
            queued - static_cast<std::int64_t>(cfg_.target_samples));
        return std::min({output.size(), cfg_.hop, excess});
    }

    void reset() noexcept { high_hops_ = 0; }

private:
    bool quiet(std::span<const float> output, bool gate_closed) const noexcept {
        if (gate_closed) return true;
        for (float x : output)
            if (!std::isfinite(x) || std::abs(x) > cfg_.quiet_peak) return false;
        return true;
    }

    GovernorConfig cfg_;
    std::size_t high_hops_ = 0;
};

} // namespace fluxvoice
