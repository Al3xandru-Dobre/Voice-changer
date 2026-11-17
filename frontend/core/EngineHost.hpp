// EngineHost — composes the Milestone 1.5 engine for the GUI exactly as
// src/main.cpp does for the CLI: an AudioBackend, the AudioController (control
// thread = UI thread), and the worker thread that regroups audio into 80 ms
// frames (ChunkAssembler), runs the InferencePipeline shell and the interim
// OutputChain, and writes to the output SPSC queue with the controller's
// queued-latency cap.
//
// Thread contract:
//   * every public method runs on the UI/control thread;
//   * `push_params` feeds the ParamChannel (SPSC, newest snapshot wins): a
//     full queue parks the snapshot and `flush_params` — called from
//     AppModel::tick — retries it, so the newest snapshot always reaches the
//     worker (M2 / Stage 09, BUG-023);
//   * meters and drop counts are atomics;
//   * the audio callback stays DuplexRouter::process (no code here runs on it).
#pragma once

#include "audio/AudioBackend.hpp"
#include "audio/AudioController.hpp"
#include "audio/AudioTypes.hpp"
#include "audio/ChunkAssembler.hpp"
#include "core/OutputChain.hpp"
#include "core/Telemetry.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "voice/ParamChannel.hpp"
#include "voice/VoiceParams.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace fluxvoice::frontend {

class EngineHost {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static_assert(std::atomic<float>::is_always_lock_free, "meters must be lock-free");
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "counters must be lock-free");

    explicit EngineHost(AudioBackend& backend, ControllerConfig cfg = {});
    ~EngineHost();

    EngineHost(const EngineHost&)            = delete;
    EngineHost& operator=(const EngineHost&) = delete;

    AudioController&       controller()       { return controller_; }
    const AudioController& controller() const { return controller_; }

    /// Open + start, or a serialized reconfigure when a stream is already open.
    Result<StreamStatus> start(const DeviceSelection& capture, const DeviceSelection& playback,
                               CaptureMode capture_mode, TimePoint now = Clock::now());
    void stop();
    void tick(TimePoint now = Clock::now()) { controller_.tick(now); }

    const StreamStatus& status() const { return controller_.status(); }
    const StreamInfo&   stream_info() const { return controller_.stream_info(); }
    bool stream_live() const { return controller_.is_stream_live(); }

    /// Queue a full snapshot for the worker. Returns its generation number.
    std::uint64_t push_params(const VoiceParams& p);
    /// Retry a snapshot parked by a full queue (call each UI tick).
    void flush_params() { params_.flush(); }
    void set_muted(bool m) { muted_.store(m, std::memory_order_relaxed); }
    bool muted() const { return muted_.load(std::memory_order_relaxed); }

    TelemetrySnapshot telemetry() const;
    std::uint64_t worker_hops() const { return hops_.load(std::memory_order_relaxed); }

private:
    void worker_loop();
    void stop_worker();
    void start_worker();
    void reset_dsp();

    AudioBackend&    backend_;
    AudioController  controller_;
    ChunkAssembler   chunks_;
    InferencePipeline pipeline_;
    OutputChain      chain_;

    ParamChannel params_;

    // Per-stream conservation baselines (MEMORY.md §2): reset_dsp() stores
    // them right after the controller discarded the queues, so the queued
    // figure does not carry earlier streams' residue (BUG-024). Written only
    // while the worker is quiesced; read by telemetry() on the same thread.
    std::uint64_t underrun_base_ = 0;
    std::uint64_t overrun_base_  = 0;

    std::thread       worker_;
    std::atomic<bool> worker_run_{false};
    std::atomic<bool> muted_{false};
    // Per-stream drops (rebaselined in reset_dsp) and the lifetime total.
    std::atomic<std::uint64_t> output_drops_{0};
    std::atomic<std::uint64_t> lifetime_drops_{0};
    std::atomic<std::uint64_t> hops_{0};
    std::atomic<float> in_peak_{0.0f};
    std::atomic<float> in_rms_{0.0f};
    std::atomic<float> out_peak_{0.0f};
    std::atomic<float> out_rms_{0.0f};
};

} // namespace fluxvoice::frontend
