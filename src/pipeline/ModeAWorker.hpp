// Hardware-free worker pump used by the live thread and deterministic tests.
#pragma once

#include "audio/ChunkAssembler.hpp"
#include "audio/DuplexRouter.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "voice/ParamChannel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fluxvoice {

struct WorkerSnapshot {
    std::uint64_t overruns = 0, underruns = 0, drops = 0, skipped = 0, prefilled = 0;
    std::int64_t queued = 0;
    // Samples the worker has read from the input queue that are not yet
    // written, dropped or skipped: the unprocessed rest of its read buffer,
    // the ChunkAssembler fill and the hop in progress. Added to the router's
    // queue sizes it gives the measured occupancy the identity check compares
    // with `queued` (M4 Stage 06). Relaxed and refreshed at hop boundaries, so
    // it may be off by up to one hop against the queue sizes read beside it.
    std::uint64_t held = 0;
};

/// Linear peak / RMS of the newest input read and output hop (display only).
struct WorkerMeters {
    float in_peak = 0.0f, in_rms = 0.0f, out_peak = 0.0f, out_rms = 0.0f;
};

class ModeAWorker {
public:
    ModeAWorker(DuplexRouter& router, InferencePipeline& pipeline,
                ParamChannel* params = nullptr)
        : router_(router), pipeline_(pipeline), params_(params),
          chunks_(pipeline.geometry().hop, 0),
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
        held_.store(0, std::memory_order_relaxed);
    }

    // One worker only. Never skips input: models advance even during silence
    // and when the governor suppresses a whole output hop.
    bool pump() noexcept {
        if (failed()) return false;
        const std::size_t read = router_.read_input(input_.data(), input_.size());
        std::size_t remaining = read;
        if (read > 0) {
            publish_input_meter(std::span<const float>(input_.data(), read));
            held_.store(chunks_.pending() + read, std::memory_order_relaxed);
        }
        const float* p = input_.data();
        while (remaining > 0) {
            const auto used = chunks_.push(p, remaining);
            p += used;
            remaining -= used;
            // Read but not yet written: the rest of this read, the partial
            // frame and, once ready, the hop about to be processed.
            held_.store(remaining + chunks_.pending(), std::memory_order_relaxed);
            if (!chunks_.ready()) continue;
            // Mute is a level, not an event: re-asserting it every hop keeps a
            // reset or a new pipeline in step with the control thread's choice.
            pipeline_.set_muted(muted_.load(std::memory_order_relaxed));
            // Stage 09: newest control snapshot wins, applied at hop start.
            // Delivery (drain) and acceptance (set_params) are separate facts:
            // a refused snapshot keeps the previous parameters and is counted.
            std::uint64_t drained = 0;
            if (params_ && params_->drain(params_buf_, drained)) {
                if (pipeline_.set_params(params_buf_))
                    accepted_generation_.store(drained, std::memory_order_relaxed);
                else
                    rejected_params_.fetch_add(1, std::memory_order_relaxed);
            }
            const auto result = pipeline_.process(
                std::span(chunks_.hop_samples(), chunks_.hop()), output_);
            publish_output_meter(output_);
            const auto skip = pipeline_.governor().skip_output(snapshot().queued, output_);
            skipped_.fetch_add(skip, std::memory_order_relaxed);
            const auto emit = std::span(output_).subspan(skip);
            drops_.fetch_add(emit.size() - router_.write_output(emit.data(), emit.size()),
                             std::memory_order_relaxed);
            chunks_.advance();
            // The hop left: written, dropped or skipped (all counted above).
            held_.store(remaining + chunks_.pending(), std::memory_order_relaxed);
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
        s.held = held_.load(std::memory_order_relaxed);
        return s;
    }

    /// Generation of the newest snapshot InferencePipeline::set_params()
    /// accepted (what "applied" means to a UI); any thread.
    std::uint64_t accepted_generation() const noexcept {
        return accepted_generation_.load(std::memory_order_relaxed);
    }
    /// Drained snapshots set_params() refused (previous parameters kept).
    std::uint64_t rejected_params() const noexcept {
        return rejected_params_.load(std::memory_order_relaxed);
    }

    /// Ramped output mute, applied by the worker at the next hop start
    /// (InferencePipeline::set_muted). Any thread.
    void set_muted(bool muted) noexcept { muted_.store(muted, std::memory_order_relaxed); }
    bool muted() const noexcept { return muted_.load(std::memory_order_relaxed); }

    /// Newest input read / emitted hop levels (display only); any thread.
    WorkerMeters meters() const noexcept {
        WorkerMeters m;
        m.in_peak = in_peak_.load(std::memory_order_relaxed);
        m.in_rms = in_rms_.load(std::memory_order_relaxed);
        m.out_peak = out_peak_.load(std::memory_order_relaxed);
        m.out_rms = out_rms_.load(std::memory_order_relaxed);
        return m;
    }

    bool failed() const noexcept { return fatal_.load(std::memory_order_acquire); }
    std::string_view error() const noexcept { return failed() ? pipeline_.last_error() : std::string_view{}; }

private:
    static void level_of(std::span<const float> x, float& peak, float& rms) noexcept {
        float pk = 0.0f;
        double sum = 0.0;
        for (const float v : x) {
            pk = std::max(pk, std::fabs(v));
            sum += static_cast<double>(v) * v;
        }
        peak = pk;
        rms = x.empty() ? 0.0f : static_cast<float>(std::sqrt(sum / static_cast<double>(x.size())));
    }
    void publish_input_meter(std::span<const float> x) noexcept {
        float pk = 0.0f, rms = 0.0f;
        level_of(x, pk, rms);
        in_peak_.store(pk, std::memory_order_relaxed);
        in_rms_.store(rms, std::memory_order_relaxed);
    }
    void publish_output_meter(std::span<const float> x) noexcept {
        float pk = 0.0f, rms = 0.0f;
        level_of(x, pk, rms);
        out_peak_.store(pk, std::memory_order_relaxed);
        out_rms_.store(rms, std::memory_order_relaxed);
    }

    DuplexRouter& router_;
    InferencePipeline& pipeline_;
    ParamChannel* params_;
    VoiceParams params_buf_{};
    ChunkAssembler chunks_;
    std::array<float, 4096> input_{};
    std::vector<float> output_, silence_;
    // Written only while quiesced. Thread creation publishes these baselines
    // to the worker; prepare() never overlaps pump()/snapshot().
    std::uint64_t underrun_base_ = 0, overrun_base_ = 0;
    std::atomic<std::uint64_t> drops_{0}, skipped_{0}, prefilled_{0};
    std::atomic<std::uint64_t> held_{0};   // see WorkerSnapshot::held
    std::atomic<bool> fatal_{false};
    // Control-thread-visible worker results (relaxed; may lag a hop).
    std::atomic<std::uint64_t> accepted_generation_{0}, rejected_params_{0};
    std::atomic<bool> muted_{false};
    std::atomic<float> in_peak_{0.0f}, in_rms_{0.0f}, out_peak_{0.0f}, out_rms_{0.0f};
    static_assert(std::atomic<float>::is_always_lock_free, "meters must be lock-free");
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "counters must be lock-free");
};

} // namespace fluxvoice
