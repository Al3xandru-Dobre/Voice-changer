// AudioEngine — keyed endpoints + full-duplex/loopback stream wrapper.
//
// Design rules (see Documents/Milestone 1/ and Milestone 1.5/):
//   * The audio data callback runs on a real-time OS audio thread.
//     No heap allocation, no mutexes, no logging, no blocking — only
//     memcpy and lock-free SPSC queue operations (see DuplexRouter).
//   * The inference worker thread pulls samples via read_input() and pushes
//     results via write_output(). Underruns are zero-filled to keep the
//     output stream glitch-free.
//   * Native miniaudio types and ma_device_id stay inside MiniaudioBackend
//     (M1.5 stages 02–05); AudioEngine delegates stream open/start/stop
//     through the AudioBackend seam.
#pragma once

#include "audio/AudioTypes.hpp"
#include "audio/DuplexRouter.hpp"
#include "audio/MiniaudioBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using fluxvoice::StreamInfo;

class AudioEngine {
public:
    static constexpr std::uint32_t kDefaultSampleRate = 16000;
    static constexpr std::uint32_t kDefaultChannels   = 1;
    static constexpr std::uint32_t kDefaultPeriodMs   = 20;

    AudioEngine() = default;  // backend_ initialises (and owns) the context

    ~AudioEngine() { stop(); }

    AudioEngine(const AudioEngine&)            = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    // ------------------------------------------------------------------
    // Device enumeration — owned descriptor snapshots (M1 convenience;
    // M1.5 callers use AudioDeviceRegistry / AudioController).
    // ------------------------------------------------------------------
    std::vector<fluxvoice::EndpointDescriptor> playback_devices() {
        return devices(fluxvoice::EndpointDirection::Playback);
    }
    std::vector<fluxvoice::EndpointDescriptor> capture_devices() {
        return devices(fluxvoice::EndpointDirection::Capture);
    }

    bool context_ok() const { return backend_.ok(); }
    std::string backend_name() const { return backend_.backend_name(); }

    /// The shared backend (the one context) — hand it to AudioDeviceRegistry
    /// or AudioController so catalog keys and stream opens resolve against
    /// the same epoch (M1.5 stages 03–05). Must not outlive this engine.
    MiniaudioBackend& audio_backend() { return backend_; }

    /// Access the underlying SPSC router (for controller / worker integration).
    DuplexRouter& router() { return router_; }
    const DuplexRouter& router() const { return router_; }

    // ------------------------------------------------------------------
    // Stream control
    // ------------------------------------------------------------------

    /// Open a stream without starting it, so callers can inspect
    /// stream_info() first.
    /// nullptr keys select the OS default endpoints; explicit keys must have
    /// been minted by this engine's backend.
    bool open(const fluxvoice::EndpointKey* input  = nullptr,
              const fluxvoice::EndpointKey* output = nullptr,
              std::uint32_t sample_rate = kDefaultSampleRate,
              bool passthrough = false,
              fluxvoice::CaptureMode capture_mode = fluxvoice::CaptureMode::OrdinaryMicrophone) {
        if (!backend_.ok() || backend_.is_stream_open()) return false;

        router_.set_passthrough(passthrough);

        fluxvoice::StreamRequest req;
        req.capture_mode = capture_mode;
        req.capture_key  = input;
        req.playback_key = output;
        req.sample_rate  = sample_rate;
        req.period_ms    = kDefaultPeriodMs;
        req.passthrough  = passthrough;

        auto res = backend_.open_stream(req, &AudioEngine::data_callback, this);
        if (!fluxvoice::is_ok(res)) return false;

        info_ = std::move(std::get<StreamInfo>(res));
        return true;
    }

    /// Start the stream opened by open(); the data callback begins to run.
    bool start() {
        auto res = backend_.start_stream();
        return fluxvoice::is_ok(res);
    }

    /// Stop and close the stream (no-op when none is open).
    void stop() {
        backend_.stop_stream();
    }

    bool is_open() const { return backend_.is_stream_open(); }
    bool is_running() const { return backend_.is_stream_running(); }
    std::uint32_t sample_rate() const { return info_.client_sample_rate; }
    const StreamInfo& stream_info() const { return info_; }

    // ------------------------------------------------------------------
    // Worker-thread side (NOT real-time — call from the inference thread)
    // ------------------------------------------------------------------
    std::size_t read_input(float* dst, std::size_t max_samples) {
        return router_.read_input(dst, max_samples);
    }
    std::size_t write_output(const float* src, std::size_t num_samples) {
        return router_.write_output(src, num_samples);
    }
    std::size_t write_output_bounded(const float* src, std::size_t num_samples, std::size_t max_queued_samples) {
        return router_.write_output_bounded(src, num_samples, max_queued_samples);
    }

    // ------------------------------------------------------------------
    // Telemetry (atomic — safe to read from the UI thread)
    // ------------------------------------------------------------------
    std::uint64_t input_overruns()   const { return router_.input_overruns(); }
    std::uint64_t output_underruns() const { return router_.output_underruns(); }
    std::uint64_t frames_processed() const { return router_.frames_processed(); }

private:
    std::vector<fluxvoice::EndpointDescriptor> devices(fluxvoice::EndpointDirection dir) {
        auto snap = backend_.enumerate();
        if (!fluxvoice::is_ok(snap)) return {};
        auto& owned = std::get<fluxvoice::DeviceSnapshot>(snap);
        return dir == fluxvoice::EndpointDirection::Capture ? std::move(owned.capture)
                                                            : std::move(owned.playback);
    }

    /// Real-time audio callback. Keep it allocation-free and lock-free —
    /// all logic lives in DuplexRouter::process().
    static void data_callback(void* user_data, float* output, const float* input,
                              std::size_t sample_count) {
        auto* self = static_cast<AudioEngine*>(user_data);
        self->router_.process(output, input, sample_count);
    }

    MiniaudioBackend backend_;
    StreamInfo info_;

    DuplexRouter router_;
};
