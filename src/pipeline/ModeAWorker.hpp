// Hardware-free worker pump used by the live thread and deterministic tests.
#pragma once

#include "audio/ChunkAssembler.hpp"
#include "audio/DuplexRouter.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/PipelineGeometry.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fluxvoice {

struct WorkerSnapshot {
    std::uint64_t overruns = 0, underruns = 0, drops = 0, skipped = 0, prefilled = 0;
    std::int64_t queued = 0;
};

class ModeAWorker {
public:
    ModeAWorker(DuplexRouter& router, InferencePipeline& pipeline)
        : router_(router), pipeline_(pipeline), chunks_(pipeline.geometry().hop, 0),
          output_(pipeline.geometry().hop),
          silence_(pipeline.governor().target_samples(), 0.0f) {
        if (silence_.size() + output_.size() > DuplexRouter::kQueueCapacity)
            throw std::invalid_argument("latency target plus one hop exceeds the audio queue capacity");
    }

    // Control thread, AFTER queue drain and BEFORE callback start. SPSC roles
    // may transfer only while quiesced. Begin a new per-stream accounting epoch.
    bool prepare() {
        if (router_.input_queued_approx() != 0 || router_.output_queued_approx() != 0) return false;
        reset();
        fatal_.store(false, std::memory_order_relaxed);
        const auto inserted = router_.write_output(silence_.data(), silence_.size());
        prefilled_.store(inserted, std::memory_order_relaxed);
        return inserted == silence_.size();
    }

    // Called by the controller AFTER callbacks/worker stop and queue discard.
    void reset() {
        chunks_.reset();
        pipeline_.reset();
        underrun_base_ = router_.output_underruns();
        overrun_base_ = router_.input_overruns();
        drops_.store(0, std::memory_order_relaxed);
        skipped_.store(0, std::memory_order_relaxed);
        prefilled_.store(0, std::memory_order_relaxed);
    }

    // One worker only. Never skips input: models advance even during silence
    // and when the governor suppresses a whole output hop.
    bool pump() noexcept {
        if (failed()) return false;
        std::size_t remaining = router_.read_input(input_.data(), input_.size());
        const float* p = input_.data();
        while (remaining > 0) {
            const auto used = chunks_.push(p, remaining);
            p += used;
            remaining -= used;
            if (!chunks_.ready()) continue;
            const auto result = pipeline_.process(
                std::span(chunks_.hop_samples(), chunks_.hop()), output_);
            const auto skip = pipeline_.governor().skip_output(snapshot().queued, output_);
            skipped_.fetch_add(skip, std::memory_order_relaxed);
            const auto emit = std::span(output_).subspan(skip);
            drops_.fetch_add(emit.size() - router_.write_output(emit.data(), emit.size()),
                             std::memory_order_relaxed);
            chunks_.advance();
            if (result.stop) {
                // Publish the pipeline's stable last_error to the control
                // thread; there are no further process() calls after this.
                fatal_.store(true, std::memory_order_release);
                return false;
            }
        }
        return true;
    }

    WorkerSnapshot snapshot() const noexcept {
        WorkerSnapshot s;
        s.overruns = router_.input_overruns() - overrun_base_;
        s.underruns = router_.output_underruns() - underrun_base_;
        s.drops = drops_.load(std::memory_order_relaxed);
        s.skipped = skipped_.load(std::memory_order_relaxed);
        s.prefilled = prefilled_.load(std::memory_order_relaxed);
        s.queued = queued_samples(s.underruns, s.overruns, s.drops, s.skipped, s.prefilled);
        return s;
    }

    bool failed() const noexcept { return fatal_.load(std::memory_order_acquire); }
    std::string_view error() const noexcept { return failed() ? pipeline_.last_error() : std::string_view{}; }

private:
    DuplexRouter& router_;
    InferencePipeline& pipeline_;
    ChunkAssembler chunks_;
    std::array<float, 4096> input_{};
    std::vector<float> output_, silence_;
    // Written only while quiesced. Thread creation publishes these baselines
    // to the worker; prepare() never overlaps pump()/snapshot().
    std::uint64_t underrun_base_ = 0, overrun_base_ = 0;
    std::atomic<std::uint64_t> drops_{0}, skipped_{0}, prefilled_{0};
    std::atomic<bool> fatal_{false};
};

} // namespace fluxvoice
