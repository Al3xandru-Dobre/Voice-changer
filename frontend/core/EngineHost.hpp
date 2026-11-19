// EngineHost — composes the Milestone 1.5 engine for the GUI: an AudioBackend,
// the AudioController (control thread = UI thread) and one of two workers.
//
//   Bypass  no model bundle is loaded. The worker regroups the capture stream
//           into 20 ms blocks and runs them through fluxvoice::VoiceOutputChain
//           (gain, gate, brightness, limiter, ramped mute): the audio is the
//           microphone, processed by the output stage only. The pitch, timbre
//           and voice controls have nothing to act on, so the UI never shows
//           them as applied (TelemetrySnapshot::params_accepted stays 0).
//   Mode A  a fluxvoice::ModeAHost is installed (M4 Stage 05). The host owns
//           the worker, the parameter channel and the output chain exactly as
//           for `fluxvoice --vc`; this class only wires it to the controller
//           and forwards controls and telemetry. Nothing here duplicates a
//           worker loop for Mode A.
//
// Thread contract:
//   * every public method runs on the UI/control thread;
//   * `push_params` feeds a ParamChannel (SPSC, newest snapshot wins): a full
//     queue parks the snapshot and `flush_params` — called from
//     AppModel::tick — retries it, so the newest snapshot always reaches the
//     worker (M2 / Stage 09, BUG-023);
//   * meters and drop counts are atomics;
//   * the audio callback stays DuplexRouter::process (no code here runs on it).
#pragma once

#include "audio/AudioBackend.hpp"
#include "audio/AudioController.hpp"
#include "audio/AudioTypes.hpp"
#include "audio/ChunkAssembler.hpp"
#include "core/Telemetry.hpp"
#include "pipeline/ModeAHost.hpp"
#include "voice/ParamChannel.hpp"
#include "voice/VoiceOutputChain.hpp"
#include "voice/VoiceParams.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fluxvoice::frontend {

class EngineHost {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// Length of the Bypass processing block (the conditioning frame length).
    static constexpr std::uint32_t kBypassBlockMs = 20;

    static_assert(std::atomic<float>::is_always_lock_free, "meters must be lock-free");
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "counters must be lock-free");
    static_assert(std::atomic<const char*>::is_always_lock_free, "priority method must be lock-free");

    /// `loader` is the model source every host this engine (and the AppModel
    /// above it) builds uses; tests inject fakes, the default reads ONNX bundles.
    explicit EngineHost(AudioBackend& backend, ControllerConfig cfg = {},
                        std::shared_ptr<const fluxvoice::HostModelLoader> loader = fluxvoice::ort_bundle_loader());
    ~EngineHost();

    EngineHost(const EngineHost&)            = delete;
    EngineHost& operator=(const EngineHost&) = delete;

    AudioController&       controller()       { return controller_; }
    const AudioController& controller() const { return controller_; }

    // ---- mode ------------------------------------------------------------
    enum class Mode : std::uint8_t { Bypass, ModeA };
    Mode mode() const { return host_ ? Mode::ModeA : Mode::Bypass; }
    bool mode_a() const { return mode() == Mode::ModeA; }
    /// "Bypass - no model loaded" (ASCII on purpose: the UI font and the tray
    /// tooltip need not carry an em dash). Shown in Live, the tray and the status bar.
    /// In Mode A: "Mode A - <bundle name>".
    std::string mode_label() const;

    const std::shared_ptr<const fluxvoice::HostModelLoader>& loader() const { return loader_; }

    /// The installed Mode A host (null in Bypass). Telemetry readers only; the
    /// host's own threading rules (ModeAHost.hpp) apply to anything else.
    const fluxvoice::ModeAHost* host() const { return host_.get(); }
    fluxvoice::ModeAHost*       host()       { return host_.get(); }

    /// Make `host` (loaded, not attached) the engine's model, replacing the
    /// current one. Controller must be Idle: the contract rate, fade_underruns
    /// and max_queued_samples are applied with `set_config` only while no
    /// stream exists. The previous host is detached first and destroyed only
    /// after the new one is attached (or, on failure, restored: the previous
    /// bundle stays active and `error` says why). The host's hooks replace the
    /// Bypass ones.
    bool install_host(std::unique_ptr<fluxvoice::ModeAHost> host, std::string bundle_name, std::string& error);
    /// Back to Bypass (Idle only): detaches and destroys the host, restores the
    /// Bypass controller config and its hooks. Never called implicitly.
    bool remove_host(std::string& error);

    // ---- Mode A forwarding (control thread; Bypass answers are the "none" values) ----
    /// Index of the loaded bundle's voice `name`; nullopt and an error naming
    /// the available voices otherwise (always nullopt in Bypass).
    std::optional<std::uint16_t> voice_index(std::string_view name, std::string& error) const;
    /// True when the loaded host owns translator models (never in Bypass).
    bool host_has_translator() const;
    /// The host's pipeline reported a fatal error (worker stopped); false in Bypass.
    bool host_failed() const;
    /// The pipeline's stable last error, after host_failed().
    std::string host_error() const;
    /// Snapshots the pipeline refused (lifetime); 0 in Bypass.
    std::uint64_t rejected_params() const;

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
    void flush_params();
    /// Ramped mute: VoiceOutputChain in Bypass, the pipeline's chain in Mode A.
    void set_muted(bool m);
    bool muted() const { return muted_.load(std::memory_order_relaxed); }

    TelemetrySnapshot telemetry() const;
    /// Bypass: 20 ms blocks processed. Mode A: hops the pipeline timed.
    std::uint64_t worker_hops() const;

    /// Outcome of the worker's scheduling request (src/util/WorkerPriority.hpp).
    /// Published by the worker thread right after it starts (a few microseconds
    /// to milliseconds after start()); until then, and before the first start,
    /// it reports "not requested" / false.
    const char* priority_method() const;
    bool priority_elevated() const;

private:
    void bypass_loop();
    void stop_worker();
    void start_worker();
    void reset_dsp();
    void install_bypass_hooks();
    std::uint32_t stream_rate() const;
    void restore_bypass_config();

    AudioBackend&    backend_;
    AudioController  controller_;
    ControllerConfig bypass_cfg_;  // the config Bypass runs with (restored when a host is removed)
    std::shared_ptr<const fluxvoice::HostModelLoader> loader_;

    // Mode A: the shared host owns the worker, parameter channel and output
    // chain. Declared after controller_ so it is destroyed first (its hooks
    // point into the controller).
    std::unique_ptr<fluxvoice::ModeAHost> host_;
    std::string                           bundle_name_;
    std::uint64_t                         host_drops_closed_ = 0;  // drops of earlier Mode A streams (folded at stop())

    // ---- Bypass worker ---------------------------------------------------
    std::unique_ptr<ChunkAssembler> blocks_;  // 20 ms blocks at the stream rate (rebuilt in reset_dsp)
    VoiceOutputChain chain_;
    VoiceParams      bypass_params_{};  // last snapshot the worker applied (worker / quiesced only)
    ParamChannel     params_;

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
    // Samples the Bypass worker holds (read from the input queue, not yet
    // written, or dropped): the third term of the occupancy the identity check
    // measures. Published by the worker after each read; 0 while stopped.
    std::atomic<std::size_t> held_{0};
    std::atomic<float> in_peak_{0.0f};
    std::atomic<float> in_rms_{0.0f};
    std::atomic<float> out_peak_{0.0f};
    std::atomic<float> out_rms_{0.0f};
    // Published by the worker before its loop (method strings are literals).
    std::atomic<const char*> priority_method_{"not requested"};
    std::atomic<bool>        priority_elevated_{false};
};

} // namespace fluxvoice::frontend
