// FakeAudioBackend — an AudioBackend with a scriptable catalog, open/start
// failures, notification bits and a pumpable data callback, for tests that
// drive AudioController-based code without hardware (M1.5 research §11
// "Deterministic tests without hardware"). The frontend tests use it; it
// mirrors FakeControllableBackend in tests/test_audio_controller.cpp.
#pragma once

#include "audio/AudioBackend.hpp"
#include "audio/AudioTypes.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fluxvoice::testing {

class FakeAudioBackend : public AudioBackend {
public:
    BackendStatus backend_status{BackendState::Ready, "wasapi", 0};
    ContextEpoch epoch = 1;
    std::vector<EndpointDescriptor> capture;
    std::vector<EndpointDescriptor> playback;
    AudioCapabilities caps{Capability::Supported, Capability::Supported, Capability::Unknown, Capability::Unknown};
    bool auto_routes = true;

    std::optional<AudioError> enumerate_error;
    std::optional<AudioError> next_open_error;
    std::optional<AudioError> next_start_error;

    bool stream_open    = false;
    bool stream_running = false;
    StreamRequest last_request{};
    bool last_capture_was_default  = false;
    bool last_playback_was_default = false;

    std::uint64_t enumerate_calls = 0;
    std::uint64_t open_calls      = 0;
    std::uint64_t start_calls     = 0;
    std::uint64_t stop_calls      = 0;

    AudioProcessFn process_fn = nullptr;
    void* process_user_data   = nullptr;
    std::atomic<std::uint32_t> notifications{NotificationBits::kNone};

    EndpointDescriptor make_ep(EndpointDirection dir, std::uint64_t token, const char* name,
                               bool is_def = false) const {
        EndpointDescriptor d;
        d.key        = EndpointKey{dir, epoch, token};
        d.name       = name;
        d.is_default = is_def;
        return d;
    }

    BackendStatus status() const override { return backend_status; }

    Result<DeviceSnapshot> enumerate() override {
        ++enumerate_calls;
        if (enumerate_error) return *enumerate_error;
        DeviceSnapshot s;
        s.backend    = backend_status.backend;
        s.epoch      = epoch;
        s.generation = enumerate_calls;
        s.capture    = capture;
        s.playback   = playback;
        return s;
    }

    Result<EndpointInfo> probe(const EndpointKey& key) const override {
        EndpointInfo info;
        info.key = key;
        NativeFormat f;
        f.sample_rate = 48000;
        f.channels    = 2;
        f.format      = 5;
        info.native_formats.push_back(f);
        for (const auto& d : capture)  if (d.key == key) info.name = d.name;
        for (const auto& d : playback) if (d.key == key) info.name = d.name;
        return info;
    }

    AudioCapabilities capabilities() const override { return caps; }

    Result<StreamInfo> open_stream(const StreamRequest& req, AudioProcessFn fn, void* user_data) override {
        ++open_calls;
        if (next_open_error) {
            AudioError err = *next_open_error;
            next_open_error.reset();
            return err;
        }
        if (req.capture_mode == CaptureMode::SystemPlaybackLoopback && caps.native_loopback != Capability::Supported) {
            return AudioError{AudioErrorCategory::FormatUnsupported, "open system-playback loopback", -201,
                              backend_status.backend, "Native loopback unsupported"};
        }
        last_request              = req;
        last_capture_was_default  = (req.capture_key == nullptr);
        last_playback_was_default = (req.playback_key == nullptr);
        process_fn                = fn;
        process_user_data         = user_data;
        stream_open               = true;

        StreamInfo info;
        info.backend                  = backend_status.backend;
        info.client_sample_rate       = req.sample_rate;
        info.capture_internal_rate    = 48000;
        info.playback_internal_rate   = 48000;
        info.capture_internal_period  = 480;
        info.playback_internal_period = 480;
        const auto& cap_list = req.capture_mode == CaptureMode::SystemPlaybackLoopback ? playback : capture;
        for (const auto& d : cap_list)
            if ((req.capture_key && d.key == *req.capture_key) || (!req.capture_key && d.is_default)) {
                info.capture_name = d.name;
                break;
            }
        for (const auto& d : playback)
            if ((req.playback_key && d.key == *req.playback_key) || (!req.playback_key && d.is_default)) {
                info.playback_name = d.name;
                break;
            }
        return info;
    }

    Result<bool> start_stream() override {
        ++start_calls;
        if (next_start_error) {
            AudioError err = *next_start_error;
            next_start_error.reset();
            return err;
        }
        stream_running = true;
        notifications.fetch_or(NotificationBits::kStarted, std::memory_order_relaxed);
        return true;
    }

    void stop_stream() override {
        if (stream_open || stream_running) ++stop_calls;
        stream_running    = false;
        stream_open       = false;
        process_fn        = nullptr;
        process_user_data = nullptr;
    }

    bool is_stream_open() const override { return stream_open; }
    bool is_stream_running() const override { return stream_running; }
    std::uint32_t drain_notifications() override {
        return notifications.exchange(NotificationBits::kNone, std::memory_order_acq_rel);
    }
    bool auto_routes_default() const override { return auto_routes; }

    void fire_notification(std::uint32_t bits) { notifications.fetch_or(bits, std::memory_order_relaxed); }

    /// Drive the registered data callback as the audio thread would.
    void pump_callback(const float* in, float* out, std::size_t n) {
        if (stream_running && process_fn) process_fn(process_user_data, out, in, n);
    }
};

} // namespace fluxvoice::testing
