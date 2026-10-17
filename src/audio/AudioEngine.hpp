// AudioEngine — miniaudio device enumeration + full-duplex stream.
//
// Design rules (see Documents/Milestone 1/):
//   * The miniaudio data callback runs on a real-time OS audio thread.
//     No heap allocation, no mutexes, no logging, no blocking — only
//     memcpy and lock-free SPSC queue operations (see DuplexRouter).
//   * The inference worker thread pulls samples via read_input() and pushes
//     results via write_output(). Underruns are zero-filled to keep the
//     output stream glitch-free.
#pragma once

#include <miniaudio.h>

#include "audio/DuplexRouter.hpp"

#include <cstdint>
#include <string>
#include <vector>

/// Human-readable wrapper around ma_device_info.
struct AudioDeviceInfo {
    ma_device_id id{};
    std::string name;
    bool is_default = false;
};

/// What was actually opened — filled by AudioEngine::start().
/// `client_sample_rate` is the rate our callback sees; the `*_internal_*`
/// fields are the backend-side values (the OS audio server may still convert
/// further). Mixing the two up is the classic "chipmunk voice" bug.
struct StreamInfo {
    std::string backend;
    std::string capture_name;
    std::string playback_name;
    uint32_t client_sample_rate        = 0;
    uint32_t capture_internal_rate     = 0;
    uint32_t playback_internal_rate    = 0;
    uint32_t capture_internal_period   = 0;  // frames
    uint32_t playback_internal_period  = 0;  // frames
};

class AudioEngine {
public:
    static constexpr uint32_t kDefaultSampleRate = 16000;
    static constexpr uint32_t kDefaultChannels   = 1;
    static constexpr uint32_t kDefaultPeriodMs   = 20;

    AudioEngine() {
        ctx_ok_ = (ma_context_init(nullptr, 0, nullptr, &ctx_) == MA_SUCCESS);
    }

    ~AudioEngine() {
        stop();
        if (ctx_ok_) ma_context_uninit(&ctx_);
    }

    AudioEngine(const AudioEngine&)            = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    // ------------------------------------------------------------------
    // Device enumeration
    // ------------------------------------------------------------------
    std::vector<AudioDeviceInfo> playback_devices()  { return enumerate(ma_device_type_playback); }
    std::vector<AudioDeviceInfo> capture_devices()   { return enumerate(ma_device_type_capture);  }

    bool context_ok() const { return ctx_ok_; }
    const char* backend_name() const {
        return ctx_ok_ ? ma_get_backend_name(ctx_.backend) : "none";
    }

    // ------------------------------------------------------------------
    // Stream control
    // ------------------------------------------------------------------

    /// Open and start a full-duplex stream (capture + playback, same format).
    /// nullptr device ids select the OS default endpoints.
    /// When `passthrough` is true the callback memcpy's input → output
    /// directly (Milestone 1 smoke test); otherwise audio flows through the
    /// SPSC queues and the worker thread.
    bool start(const ma_device_id* input_id  = nullptr,
               const ma_device_id* output_id = nullptr,
               uint32_t sample_rate = kDefaultSampleRate,
               bool passthrough = false) {
        if (!ctx_ok_ || dev_ok_) return false;

        router_.set_passthrough(passthrough);

        ma_device_config cfg = ma_device_config_init(ma_device_type_duplex);
        cfg.capture.pDeviceID   = input_id;
        cfg.capture.format      = ma_format_f32;
        cfg.capture.channels    = kDefaultChannels;
        cfg.playback.pDeviceID  = output_id;
        cfg.playback.format     = ma_format_f32;
        cfg.playback.channels   = kDefaultChannels;
        cfg.sampleRate          = sample_rate;
        cfg.periodSizeInMilliseconds = kDefaultPeriodMs;
        cfg.dataCallback        = &AudioEngine::data_callback;
        cfg.pUserData           = this;

        if (ma_device_init(&ctx_, &cfg, &dev_) != MA_SUCCESS) return false;

        info_.backend                  = backend_name();
        info_.capture_name             = dev_.capture.name;
        info_.playback_name            = dev_.playback.name;
        info_.client_sample_rate       = dev_.sampleRate;
        info_.capture_internal_rate    = dev_.capture.internalSampleRate;
        info_.playback_internal_rate   = dev_.playback.internalSampleRate;
        info_.capture_internal_period  = dev_.capture.internalPeriodSizeInFrames;
        info_.playback_internal_period = dev_.playback.internalPeriodSizeInFrames;

        if (ma_device_start(&dev_) != MA_SUCCESS) {
            ma_device_uninit(&dev_);
            return false;
        }
        dev_ok_ = true;
        return true;
    }

    void stop() {
        if (dev_ok_) {
            ma_device_uninit(&dev_);
            dev_ok_ = false;
        }
    }

    bool running() const { return dev_ok_; }
    uint32_t sample_rate() const { return info_.client_sample_rate; }
    const StreamInfo& stream_info() const { return info_; }

    // ------------------------------------------------------------------
    // Worker-thread side (NOT real-time — call from the inference thread)
    // ------------------------------------------------------------------
    size_t read_input(float* dst, size_t max_samples)       { return router_.read_input(dst, max_samples); }
    size_t write_output(const float* src, size_t num_samples) { return router_.write_output(src, num_samples); }

    // ------------------------------------------------------------------
    // Telemetry (atomic — safe to read from the UI thread)
    // ------------------------------------------------------------------
    uint64_t input_overruns()   const { return router_.input_overruns(); }
    uint64_t output_underruns() const { return router_.output_underruns(); }
    uint64_t frames_processed() const { return router_.frames_processed(); }

private:
    std::vector<AudioDeviceInfo> enumerate(ma_device_type type) {
        std::vector<AudioDeviceInfo> out;
        if (!ctx_ok_) return out;

        ma_device_info* infos = nullptr;
        ma_uint32 count = 0;
        if (ma_context_get_devices(&ctx_,
                                   type == ma_device_type_playback ? &infos : nullptr,
                                   type == ma_device_type_playback ? &count : nullptr,
                                   type == ma_device_type_capture  ? &infos : nullptr,
                                   type == ma_device_type_capture  ? &count : nullptr) != MA_SUCCESS)
            return out;

        out.reserve(count);
        for (ma_uint32 i = 0; i < count; ++i) {
            AudioDeviceInfo d;
            d.id         = infos[i].id;
            d.name       = infos[i].name;
            d.is_default = (infos[i].isDefault != 0);
            out.push_back(std::move(d));
        }
        return out;
    }

    /// Real-time audio callback. Keep it allocation-free and lock-free —
    /// all logic lives in DuplexRouter::process().
    static void data_callback(ma_device* dev, void* output, const void* input,
                              ma_uint32 frame_count) {
        auto* self = static_cast<AudioEngine*>(dev->pUserData);
        self->router_.process(static_cast<float*>(output),
                              static_cast<const float*>(input),
                              static_cast<size_t>(frame_count) * kDefaultChannels);
    }

    ma_context ctx_{};
    bool       ctx_ok_ = false;
    ma_device  dev_{};
    bool       dev_ok_ = false;
    StreamInfo info_;

    DuplexRouter router_;
};
