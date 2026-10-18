// AudioController — single-control-thread catalog refresh, notification-hint
// draining, policy-table recovery, and serialized stream switching
// (Milestone 1.5 / Stages 05–06).
//
// Invariants (Documents/Milestone 1.5/05_refresh_recovery_controller.md,
// BEST_PRACTICES.md §2–§3):
//   * All enumeration, capability queries, stream open/close, and retry
//     decisions run on a single control thread.
//   * Notification callbacks on backend threads only set bits in a lock-free
//     atomic bitset; the control thread drains and coalesces them in tick().
//   * Stream liveness is derived from controller state + notification bits +
//     backend stream state + callback progress counter — never from a boolean
//     init flag alone, and never from "samples are non-zero" (silent users
//     are still live).
//   * Explicitly selected microphones and virtual outputs are NEVER silently
//     replaced on unplug or error.
//   * Switch/stop transitions quiesce the worker and stop the backend stream
//     BEFORE discarding SPSC queues and resetting DSP/model state, so old
//     PCM never leaks into a new route and SPSC single-consumer discipline is
//     never violated.
#pragma once

#include "audio/AudioBackend.hpp"
#include "audio/AudioDeviceRegistry.hpp"
#include "audio/AudioTypes.hpp"
#include "audio/DuplexRouter.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace fluxvoice {

struct ControllerConfig {
    std::chrono::milliseconds refresh_interval{1000};
    std::chrono::milliseconds initial_retry_backoff{200};
    std::chrono::milliseconds max_retry_backoff{2000};
    std::uint32_t max_retries       = 3;
    std::uint32_t sample_rate       = 16000;
    std::uint32_t period_ms         = 20;
    bool passthrough                = false;
    /// Maximum queued output samples before the worker drops excess samples
    /// (4 x 80 ms hops = 320 ms @ 16 kHz by default; 0 = unbounded).
    std::size_t max_queued_samples  = 5120;
    /// A running stream whose callback progress counter has not advanced for
    /// this long is considered stalled (the backend reports "started" but the
    /// RT thread is dead). 0 disables the check.
    std::chrono::milliseconds callback_stall_timeout{2000};
};

class AudioController {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit AudioController(AudioBackend& backend, ControllerConfig cfg = {});
    ~AudioController();

    AudioController(const AudioController&)            = delete;
    AudioController& operator=(const AudioController&) = delete;

    // ------------------------------------------------------------------
    // Configuration & hooks (control thread only)
    // ------------------------------------------------------------------

    const ControllerConfig& config() const { return cfg_; }
    void set_config(const ControllerConfig& cfg) { cfg_ = cfg; }

    /// Register optional hooks for the worker thread and inference/DSP state
    /// during stream stop / switch / recovery transitions:
    ///   1. `on_quiesce`: pause or join the worker thread so it stops reading
    ///      `in_q_` and writing `out_q_`.
    ///   2. (controller stops the backend stream and drains both SPSC queues)
    ///   3. `on_reset`: reset ChunkAssembler, OverlapAddProcessor and
    ///      InferencePipeline::reset().
    ///   4. (controller opens and starts the new stream)
    ///   5. `on_resume`: resume or relaunch the worker thread.
    void set_worker_hooks(std::function<void()> on_quiesce,
                          std::function<void()> on_reset,
                          std::function<void()> on_resume = {});

    // ------------------------------------------------------------------
    // Catalog & status (control thread only)
    // ------------------------------------------------------------------

    AudioDeviceRegistry& registry() { return registry_; }
    const AudioDeviceRegistry& registry() const { return registry_; }

    DuplexRouter& router() { return router_; }
    const DuplexRouter& router() const { return router_; }

    /// On-demand catalog refresh (also updates the periodic timer).
    Result<DeviceSnapshot> refresh(TimePoint now = Clock::now());

    const DeviceSnapshot& snapshot() const { return registry_.snapshot(); }
    bool snapshot_stale() const { return registry_.stale(); }

    const StreamStatus& status() const { return status_; }
    const StreamInfo& stream_info() const { return info_; }

    /// True when the controller is in Running state, the backend stream is
    /// actively started, and no unhandled stop/interruption notification is
    /// pending. Does NOT require non-zero audio samples.
    bool is_stream_live() const;

    // ------------------------------------------------------------------
    // Stream lifecycle & policy table (control thread only)
    // ------------------------------------------------------------------

    /// Resolve `capture` and `playback` policies against the current snapshot,
    /// check the feedback guard, and open + start the stream.
    Result<StreamStatus> open_and_start(
        const DeviceSelection& capture,
        const DeviceSelection& playback,
        CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone,
        TimePoint now = Clock::now());

    /// Serialized switch transition: quiesce worker + stop callbacks ->
    /// discard both SPSC queues -> reset DSP/model state -> open + start the
    /// new selection. No old-route PCM can leak into the new route.
    Result<StreamStatus> reconfigure(
        const DeviceSelection& capture,
        const DeviceSelection& playback,
        CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone,
        TimePoint now = Clock::now());

    /// Stop the stream and transition to Idle (safe to call from any state,
    /// including Recovering).
    void stop();

    /// Handle backend context recreation: stop streams, invalidate old keys,
    /// rebuild the device inventory in the new epoch, and re-resolve the
    /// saved selection policy.
    Result<StreamStatus> on_context_recreated(TimePoint now = Clock::now());

    /// Run one control-thread poll iteration: drain notification hints,
    /// refresh the catalog when due or hinted (with burst coalescing and
    /// bounded error backoff), and apply the Stage 05 recovery policy table.
    void tick(TimePoint now = Clock::now());

    // ------------------------------------------------------------------
    // Worker-thread I/O helpers (delegate to DuplexRouter with latency cap)
    // ------------------------------------------------------------------

    std::size_t read_input(float* dst, std::size_t max_samples) {
        return router_.read_input(dst, max_samples);
    }

    /// Write processed output while enforcing `config().max_queued_samples`.
    /// Returns samples enqueued; `num_samples - returned` were dropped.
    std::size_t write_output(const float* src, std::size_t num_samples) {
        return router_.write_output_bounded(src, num_samples, cfg_.max_queued_samples);
    }

    // ------------------------------------------------------------------
    // Diagnostics for tests and telemetry
    // ------------------------------------------------------------------

    std::uint32_t last_notifications() const { return last_notifications_; }
    std::size_t   last_discarded_samples() const { return last_discarded_samples_; }
    std::uint64_t auto_reroutes() const { return auto_reroutes_; }
    std::uint32_t refresh_failures() const { return refresh_failures_; }

private:
    static void data_callback(void* user_data, float* output, const float* input,
                              std::size_t sample_count);

    void quiesce_and_stop_stream(bool reset_dsp);
    Result<StreamStatus> open_resolved_internal(TimePoint now, bool strict_generation);
    Result<DeviceSnapshot> refresh_catalog_internal(TimePoint now);
    void schedule_next_retry(TimePoint now);
    /// Classify a failure that prevented (re)opening: non-retryable ->
    /// AwaitingUser; retryable -> bounded Recovering (Error when retries are
    /// disabled or exhausted by the caller).
    void enter_failure_state(const AudioError& err, TimePoint now);

    AudioBackend&       backend_;
    ControllerConfig    cfg_;
    AudioDeviceRegistry registry_;
    DuplexRouter        router_;

    StreamStatus status_{};
    StreamInfo   info_{};
    ContextEpoch active_epoch_ = 0;
    bool         ever_opened_  = false;

    TimePoint     next_refresh_time_{};
    TimePoint     next_retry_time_{};
    std::uint32_t refresh_failures_       = 0;
    std::uint32_t last_notifications_     = NotificationBits::kNone;
    std::size_t   last_discarded_samples_ = 0;
    std::uint64_t auto_reroutes_          = 0;

    // Callback-progress liveness (stage 05): a Running stream whose progress
    // counter stops advancing is stalled even when the backend still reports
    // "started".
    std::uint64_t last_callback_progress_ = 0;
    TimePoint     progress_since_{};
    bool          callback_stalled_ = false;

    std::function<void()> on_quiesce_;
    std::function<void()> on_reset_;
    std::function<void()> on_resume_;
};

} // namespace fluxvoice
