// Single worker writer, lock-free control/UI readers. Never called in the callback.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fluxvoice {

struct TimingSnapshot {
    std::uint64_t samples = 0;
    std::uint64_t p50_us = 0;
    std::uint64_t p95_us = 0;
    std::uint64_t max_us = 0;
    std::uint64_t misses = 0;
};

class TimingHistogram {
public:
    static constexpr std::uint64_t kBinUs = 50;
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

    explicit TimingHistogram(std::uint64_t hop_us)
        : hop_us_(check_hop(hop_us)),
          bins_(static_cast<std::size_t>(2 * hop_us_ / kBinUs + 1)) {
        reset();
    }

    void record(std::uint64_t us) noexcept {
        const auto bin = static_cast<std::size_t>(
            std::min<std::uint64_t>(us / kBinUs, bins_.size() - 1));
        bins_[bin].fetch_add(1, std::memory_order_relaxed);
        if (us > max_us_.load(std::memory_order_relaxed))
            max_us_.store(us, std::memory_order_relaxed);
        if (us > hop_us_) misses_.fetch_add(1, std::memory_order_relaxed);
    }

    // Approximate while the worker is running; exact when quiesced. Percentiles
    // are conservative bin upper bounds, with an overflow bin beyond 2x hop.
    TimingSnapshot snapshot() const noexcept {
        TimingSnapshot s;
        for (const auto& b : bins_) s.samples += b.load(std::memory_order_relaxed);
        s.max_us = max_us_.load(std::memory_order_relaxed);
        s.misses = misses_.load(std::memory_order_relaxed);
        if (s.samples == 0) return s;
        s.p50_us = percentile(s.samples, 50, s.max_us);
        s.p95_us = percentile(s.samples, 95, s.max_us);
        return s;
    }

    // Exclusive ownership only (before stream start or after worker join).
    void reset() noexcept {
        for (auto& b : bins_) b.store(0, std::memory_order_relaxed);
        max_us_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
    }

private:
    static std::uint64_t check_hop(std::uint64_t us) {
        if (us == 0 || us > 1000000)
            throw std::invalid_argument("histogram hop must be in (0, 1000000] us");
        return us;
    }
    std::uint64_t percentile(std::uint64_t n, std::uint64_t pct,
                             std::uint64_t maximum) const noexcept {
        const std::uint64_t rank = (n / 100) * pct + ((n % 100) * pct + 99) / 100;
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < bins_.size(); ++i) {
            seen += bins_[i].load(std::memory_order_relaxed);
            if (seen >= rank) {
                if (i == bins_.size() - 1) return maximum;
                return std::min(maximum, (i + 1) * kBinUs);
            }
        }
        return maximum;
    }

    const std::uint64_t hop_us_;
    std::vector<std::atomic<std::uint64_t>> bins_;
    std::atomic<std::uint64_t> max_us_{0};
    std::atomic<std::uint64_t> misses_{0};
};

} // namespace fluxvoice
