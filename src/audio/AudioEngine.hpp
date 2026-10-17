// AudioEngine — miniaudio device enumeration + full-duplex stream.
//
// Design rules (see Documents/Milestone 1/):
//   * The miniaudio data callback runs on a real-time OS audio thread.
//     No heap allocation, no mutexes, no logging, no blocking — only
//     memcpy and lock-free SPSC queue operations.
//   * The inference worker thread pulls samples via read_input() and pushes
//     results via write_output(). Underruns are zero-filled to keep the
//     output stream glitch-free.
#pragma once

#include <miniaudio.h>
#include <readerwriterqueue.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/// Human-readable wrapper around ma_device_info.
struct AudioDeviceInfo {
    ma_device_id id{};
    std::string name;
    bool is_default = false;
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

        passthrough_.store(passthrough, std::memory_order_relaxed);
        sample_rate_ = sample_rate;

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
    uint32_t sample_rate() const { return sample_rate_; }

    // ------------------------------------------------------------------
    // Worker-thread side (NOT real-time — call from the inference thread)
    // ------------------------------------------------------------------

    /// Drain up to `max_samples` captured samples. Returns count read.
    size_t read_input(float* dst, size_t max_samples) {
        size_t n = 0;
        while (n < max_samples && in_q_.try_dequeue(dst[n])) ++n;
        return n;
    }

    /// Push processed samples toward the output device. Samples that do not
    /// fit are dropped (counted as overruns on the input side of the cable).
    size_t write_output(const float* src, size_t num_samples) {
        size_t n = 0;
        while (n < num_samples && out_q_.try_enqueue(src[n])) ++n;
        return n;
    }

    // ------------------------------------------------------------------
    // Telemetry (atomic — safe to read from the UI thread)
    // ------------------------------------------------------------------
    uint64_t input_overruns()  const { return input_overruns_.load(std::memory_order_relaxed); }
    uint64_t output_underruns() const { return output_underruns_.load(std::memory_order_relaxed); }

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

    /// Real-time audio callback. Keep it allocation-free and lock-free.
    static void data_callback(ma_device* dev, void* output, const void* input,
                              ma_uint32 frame_count) {
        auto* self = static_cast<AudioEngine*>(dev->pUserData);
        const size_t n = static_cast<size_t>(frame_count) * kDefaultChannels;
        float* out = static_cast<float*>(output);
        const float* in = static_cast<const float*>(input);

        if (self->passthrough_.load(std::memory_order_relaxed)) {
            if (in) std::memcpy(out, in, n * sizeof(float));
            else    std::memset(out, 0, n * sizeof(float));
            return;
        }

        // Capture → input queue (drop + count on overflow).
        if (in) {
            for (size_t i = 0; i < n; ++i) {
                if (!self->in_q_.try_enqueue(in[i])) {
                    self->input_overruns_.fetch_add(n - i, std::memory_order_relaxed);
                    break;
                }
            }
        }

        // Output queue → device (zero-fill on underrun — never block).
        size_t got = 0;
        while (got < n && self->out_q_.try_dequeue(out[got])) ++got;
        if (got < n) {
            std::memset(out + got, 0, (n - got) * sizeof(float));
            self->output_underruns_.fetch_add(n - got, std::memory_order_relaxed);
        }
    }

    ma_context ctx_{};
    bool       ctx_ok_ = false;
    ma_device  dev_{};
    bool       dev_ok_ = false;
    uint32_t   sample_rate_ = kDefaultSampleRate;

    std::atomic<bool>     passthrough_{false};
    std::atomic<uint64_t> input_overruns_{0};
    std::atomic<uint64_t> output_underruns_{0};

    // ~4 s of mono 16 kHz audio — sized once at construction, never reallocs.
    static constexpr size_t kQueueCapacity = 1 << 16;
    moodycamel::ReaderWriterQueue<float> in_q_{kQueueCapacity};
    moodycamel::ReaderWriterQueue<float> out_q_{kQueueCapacity};
};
