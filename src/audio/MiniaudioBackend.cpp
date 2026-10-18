// MiniaudioBackend — implementation.
//
// This is the one translation unit of the fluxvoice binary that compiles
// miniaudio's implementation (MINIAUDIO_IMPLEMENTATION); main.cpp no longer
// includes <miniaudio.h> at all (Milestone 1.5 / Stage 02). fv_probe keeps
// its own copy in its own binary.

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "audio/MiniaudioBackend.hpp"

#include <utility>

namespace {

constexpr int as_int(ma_result r) { return static_cast<int>(r); }
constexpr ma_uint32 kStreamChannels = 1;

} // namespace

MiniaudioBackend::MiniaudioBackend() : epoch_(mint_epoch()) {
    init_result_ = ma_context_init(nullptr, 0, nullptr, &ctx_);
    ok_ = (init_result_ == MA_SUCCESS);
}

MiniaudioBackend::~MiniaudioBackend() {
    stop_stream();
    if (ok_) ma_context_uninit(&ctx_);
}

std::string MiniaudioBackend::backend_name() const {
    return ok_ ? ma_get_backend_name(ctx_.backend) : std::string("none");
}

fluxvoice::AudioError MiniaudioBackend::make_error(ma_result r, const char* operation) const {
    using fluxvoice::AudioError;
    using fluxvoice::AudioErrorCategory;

    const std::string bname = backend_name();
    switch (r) {
        case MA_ACCESS_DENIED:
            return AudioError{
                AudioErrorCategory::PermissionDenied, operation, as_int(r), bname,
                "Microphone or audio device permission was denied by the OS. Grant microphone "
                "access in system privacy settings and try again."};
        case MA_BUSY:
        case MA_ALREADY_IN_USE:
        case MA_UNAVAILABLE:
        case MA_NOT_CONNECTED:
        case MA_TIMEOUT:
        case MA_FAILED_TO_OPEN_BACKEND_DEVICE:
        case MA_FAILED_TO_START_BACKEND_DEVICE:
            return AudioError{
                AudioErrorCategory::DeviceBusy, operation, as_int(r), bname,
                "The audio device or server is busy or temporarily unavailable."};
        case MA_DOES_NOT_EXIST:
        case MA_NO_DEVICE:
            return AudioError{
                AudioErrorCategory::DeviceNotFound, operation, as_int(r), bname,
                "The requested audio device does not exist or was disconnected."};
        case MA_FORMAT_NOT_SUPPORTED:
        case MA_DEVICE_TYPE_NOT_SUPPORTED:
        case MA_SHARE_MODE_NOT_SUPPORTED:
        case MA_INVALID_DEVICE_CONFIG:
            return AudioError{
                AudioErrorCategory::FormatUnsupported, operation, as_int(r), bname,
                "The audio device rejected the requested stream format or mode."};
        case MA_NO_BACKEND:
        case MA_BACKEND_NOT_ENABLED:
        case MA_FAILED_TO_INIT_BACKEND:
        case MA_API_NOT_FOUND:
            return AudioError{
                AudioErrorCategory::BackendUnavailable, operation, as_int(r), bname,
                "No compatible audio backend is available."};
        default:
            return AudioError{
                AudioErrorCategory::Unknown, operation, as_int(r), bname,
                "The audio backend reported an unexpected error."};
    }
}

fluxvoice::BackendStatus MiniaudioBackend::status() const {
    if (!ok_)
        return fluxvoice::BackendStatus{fluxvoice::BackendState::Unavailable, "",
                                        as_int(init_result_)};
    return fluxvoice::BackendStatus{
        ctx_.backend == ma_backend_null ? fluxvoice::BackendState::NullOnly
                                        : fluxvoice::BackendState::Ready,
        ma_get_backend_name(ctx_.backend), 0};
}

fluxvoice::Result<fluxvoice::EndpointInfo> MiniaudioBackend::probe(
    const fluxvoice::EndpointKey& key) const {
    using fluxvoice::AudioError;
    using fluxvoice::AudioErrorCategory;
    using fluxvoice::EndpointInfo;
    using fluxvoice::NativeFormat;

    if (!ok_)
        return AudioError{AudioErrorCategory::BackendUnavailable, "probe device",
                          as_int(init_result_), "none",
                          "No audio backend could be initialised."};

    const std::optional<ma_device_id> id = resolve(key);
    if (!id)
        return AudioError{AudioErrorCategory::DeviceNotFound, "probe device", 0,
                          backend_name(),
                          "Unknown endpoint key (stale context or removed device)."};

    const ma_device_type type = key.direction == fluxvoice::EndpointDirection::Capture
                                    ? ma_device_type_capture
                                    : ma_device_type_playback;
    ma_device_info info{};
    // const_cast: miniaudio's C API isn't const-correct — the query only
    // reads the context.
    const ma_result r =
        ma_context_get_device_info(const_cast<ma_context*>(&ctx_), type, &*id, &info);
    if (r != MA_SUCCESS) {
        if (r == MA_ACCESS_DENIED || r == MA_DOES_NOT_EXIST || r == MA_NO_DEVICE)
            return make_error(r, "probe device");
        return AudioError{AudioErrorCategory::EnumerationFailed, "probe device",
                          as_int(r), backend_name(),
                          "The audio backend could not query the device."};
    }

    EndpointInfo out;
    out.key        = key;
    out.name       = info.name;
    out.is_default = (info.isDefault != 0);
    constexpr ma_uint32 kMaxNativeFormats =
        static_cast<ma_uint32>(sizeof(info.nativeDataFormats) / sizeof(info.nativeDataFormats[0]));
    const ma_uint32 count = info.nativeDataFormatCount < kMaxNativeFormats
                                ? info.nativeDataFormatCount
                                : kMaxNativeFormats;
    out.native_formats.reserve(count);
    for (ma_uint32 i = 0; i < count; ++i) {
        const auto& f = info.nativeDataFormats[i];
        out.native_formats.push_back(
            NativeFormat{f.sampleRate, f.channels, static_cast<int>(f.format)});
    }
    return out;
}

fluxvoice::AudioCapabilities MiniaudioBackend::capabilities() const {
    using fluxvoice::Capability;
    fluxvoice::AudioCapabilities caps;
    if (!ok_) return caps;  // no context to probe — everything stays Unknown

    caps.ordinary_io = Capability::Supported;
    // Native loopback = capturing a playback endpoint as the capture device
    // (WASAPI only in miniaudio 0.11.25). False does not mean a Linux monitor
    // source cannot be opened as ordinary capture — that is server config the
    // backend cannot see, so it stays Unknown.
    // const_cast: miniaudio's C API isn't const-correct — the function only
    // reads ctx_.backend.
    caps.native_loopback = ma_context_is_loopback_supported(const_cast<ma_context*>(&ctx_))
                               ? Capability::Supported
                               : Capability::Unsupported;
    return caps;
}

fluxvoice::Result<fluxvoice::DeviceSnapshot> MiniaudioBackend::enumerate() {
    using fluxvoice::AudioError;
    using fluxvoice::AudioErrorCategory;
    using fluxvoice::DeviceSnapshot;
    using fluxvoice::EndpointDescriptor;
    using fluxvoice::EndpointDirection;
    using fluxvoice::EndpointKey;

    if (!ok_)
        return AudioError{AudioErrorCategory::BackendUnavailable, "enumerate devices",
                          as_int(init_result_),
                          "none",
                          "No audio backend could be initialised (is an audio server running?)"};

    // One call for both directions: two separate calls could straddle a
    // hardware change and produce an inconsistent catalog (research §5).
    ma_device_info* playback = nullptr;
    ma_device_info* capture  = nullptr;
    ma_uint32 n_playback = 0, n_capture = 0;
    const ma_result r =
        ma_context_get_devices(&ctx_, &playback, &n_playback, &capture, &n_capture);
    if (r != MA_SUCCESS) {
        if (r == MA_ACCESS_DENIED) return make_error(r, "enumerate devices");
        return AudioError{AudioErrorCategory::EnumerationFailed, "enumerate devices",
                          as_int(r), backend_name(),
                          "The audio backend failed to list devices."};
    }

    DeviceSnapshot snap;
    snap.backend    = backend_name();
    snap.epoch      = epoch_;
    snap.generation = ++generation_;

    // Copy everything before returning; the arrays above die on the next call.
    const auto fill = [&](ma_device_info* infos, ma_uint32 count, IdTable& table,
                          EndpointDirection dir,
                          std::vector<EndpointDescriptor>& out) {
        out.reserve(count);
        for (ma_uint32 i = 0; i < count; ++i) {
            EndpointDescriptor d;
            d.key        = EndpointKey{dir, epoch_, intern(table, infos[i].id)};
            d.name       = infos[i].name;
            d.is_default = (infos[i].isDefault != 0);
            out.push_back(std::move(d));
        }
    };
    fill(playback, n_playback, playback_ids_, EndpointDirection::Playback, snap.playback);
    fill(capture, n_capture, capture_ids_, EndpointDirection::Capture, snap.capture);
    return snap;
}

fluxvoice::Result<fluxvoice::StreamInfo> MiniaudioBackend::open_stream(
    const fluxvoice::StreamRequest& req, AudioProcessFn process_fn, void* user_data) {
    using fluxvoice::AudioError;
    using fluxvoice::AudioErrorCategory;
    using fluxvoice::Capability;
    using fluxvoice::CaptureMode;
    using fluxvoice::EndpointDirection;
    using fluxvoice::StreamInfo;

    if (!ok_)
        return AudioError{AudioErrorCategory::BackendUnavailable, "open stream",
                          as_int(init_result_), "none",
                          "No audio backend could be initialised."};
    if (is_null_backend())
        return AudioError{AudioErrorCategory::NullBackendOnly, "open stream", 0,
                          backend_name(),
                          "Offline/test mode: the null backend has no audio hardware."};
    if (dev_ok_)
        return AudioError{AudioErrorCategory::DeviceBusy, "open stream",
                          as_int(MA_DEVICE_ALREADY_INITIALIZED), backend_name(),
                          "A stream is already open on this backend."};

    const EndpointDirection expected_cap_dir =
        req.capture_mode == CaptureMode::SystemPlaybackLoopback
            ? EndpointDirection::Playback
            : EndpointDirection::Capture;

    if (req.capture_mode == CaptureMode::SystemPlaybackLoopback &&
        capabilities().native_loopback != Capability::Supported) {
        std::string msg =
            "System-playback loopback capture is not supported by backend '" + backend_name() + "'.";
        if (ctx_.backend == ma_backend_coreaudio) {
            msg += " macOS Core Audio does not provide native loopback in miniaudio 0.11.25; "
                   "configure a virtual audio device (e.g. BlackHole) and select it with --input.";
        } else if (ctx_.backend == ma_backend_pulseaudio || ctx_.backend == ma_backend_alsa) {
            msg += " On Linux PulseAudio/PipeWire, select the sink's monitor source as an "
                   "ordinary capture device with --input.";
        }
        return AudioError{AudioErrorCategory::FormatUnsupported,
                          "open system-playback loopback",
                          as_int(MA_DEVICE_TYPE_NOT_SUPPORTED), backend_name(), std::move(msg)};
    }

    std::optional<ma_device_id> in_id;
    if (req.capture_key) {
        if (req.capture_key->epoch != epoch_)
            return AudioError{AudioErrorCategory::DeviceChanged, "open capture", 0,
                              backend_name(),
                              "Capture endpoint key belongs to an older audio context epoch."};
        if (req.capture_key->direction != expected_cap_dir)
            return AudioError{AudioErrorCategory::DeviceNotFound, "open capture", 0,
                              backend_name(),
                              "Capture endpoint key has the wrong endpoint direction."};
        in_id = resolve(*req.capture_key);
        if (!in_id)
            return AudioError{AudioErrorCategory::DeviceNotFound, "open capture", 0,
                              backend_name(), "Capture endpoint key is unknown."};
    }

    std::optional<ma_device_id> out_id;
    if (req.playback_key) {
        if (req.playback_key->epoch != epoch_)
            return AudioError{AudioErrorCategory::DeviceChanged, "open playback", 0,
                              backend_name(),
                              "Playback endpoint key belongs to an older audio context epoch."};
        if (req.playback_key->direction != EndpointDirection::Playback)
            return AudioError{AudioErrorCategory::DeviceNotFound, "open playback", 0,
                              backend_name(),
                              "Playback endpoint key has the wrong endpoint direction."};
        out_id = resolve(*req.playback_key);
        if (!out_id)
            return AudioError{AudioErrorCategory::DeviceNotFound, "open playback", 0,
                              backend_name(), "Playback endpoint key is unknown."};
    }

    process_fn_        = process_fn;
    process_user_data_ = user_data;
    pending_notifications_.store(fluxvoice::NotificationBits::kNone, std::memory_order_relaxed);

    if (req.capture_mode == CaptureMode::SystemPlaybackLoopback) {
        ma_device_config cap_cfg         = ma_device_config_init(ma_device_type_loopback);
        cap_cfg.capture.pDeviceID        = in_id ? &*in_id : nullptr;
        cap_cfg.capture.format           = ma_format_f32;
        cap_cfg.capture.channels         = kStreamChannels;
        cap_cfg.sampleRate               = req.sample_rate;
        cap_cfg.periodSizeInMilliseconds = req.period_ms;
        cap_cfg.dataCallback             = &MiniaudioBackend::on_data;
        cap_cfg.notificationCallback     = &MiniaudioBackend::on_notification;
        cap_cfg.pUserData                = this;

        const ma_result r_cap = ma_device_init(&ctx_, &cap_cfg, &dev_);
        if (r_cap != MA_SUCCESS) return make_error(r_cap, "open system-playback loopback");
        dev_ok_ = true;

        ma_device_config play_cfg         = ma_device_config_init(ma_device_type_playback);
        play_cfg.playback.pDeviceID       = out_id ? &*out_id : nullptr;
        play_cfg.playback.format          = ma_format_f32;
        play_cfg.playback.channels        = kStreamChannels;
        play_cfg.sampleRate               = req.sample_rate;
        play_cfg.periodSizeInMilliseconds = req.period_ms;
        play_cfg.dataCallback             = &MiniaudioBackend::on_data;
        play_cfg.notificationCallback     = &MiniaudioBackend::on_notification;
        play_cfg.pUserData                = this;

        const ma_result r_play = ma_device_init(&ctx_, &play_cfg, &play_dev_);
        if (r_play != MA_SUCCESS) {
            ma_device_uninit(&dev_);
            dev_ok_ = false;
            return make_error(r_play, "open playback");
        }
        play_dev_ok_ = true;

        StreamInfo info;
        info.backend                  = backend_name();
        info.capture_name             = dev_.capture.name;
        info.playback_name            = play_dev_.playback.name;
        info.client_sample_rate       = dev_.sampleRate;
        info.capture_internal_rate    = dev_.capture.internalSampleRate;
        info.playback_internal_rate   = play_dev_.playback.internalSampleRate;
        info.capture_internal_period  = dev_.capture.internalPeriodSizeInFrames;
        info.playback_internal_period = play_dev_.playback.internalPeriodSizeInFrames;
        return info;
    }

    ma_device_config cfg         = ma_device_config_init(ma_device_type_duplex);
    cfg.capture.pDeviceID        = in_id ? &*in_id : nullptr;
    cfg.playback.pDeviceID       = out_id ? &*out_id : nullptr;
    cfg.capture.format           = ma_format_f32;
    cfg.capture.channels         = kStreamChannels;
    cfg.playback.format          = ma_format_f32;
    cfg.playback.channels        = kStreamChannels;
    cfg.sampleRate               = req.sample_rate;
    cfg.periodSizeInMilliseconds = req.period_ms;
    cfg.dataCallback             = &MiniaudioBackend::on_data;
    cfg.notificationCallback     = &MiniaudioBackend::on_notification;
    cfg.pUserData                = this;

    const ma_result r = ma_device_init(&ctx_, &cfg, &dev_);
    if (r != MA_SUCCESS) return make_error(r, "open stream");
    dev_ok_ = true;

    StreamInfo info;
    info.backend                  = backend_name();
    info.capture_name             = dev_.capture.name;
    info.playback_name            = dev_.playback.name;
    info.client_sample_rate       = dev_.sampleRate;
    info.capture_internal_rate    = dev_.capture.internalSampleRate;
    info.playback_internal_rate   = dev_.playback.internalSampleRate;
    info.capture_internal_period  = dev_.capture.internalPeriodSizeInFrames;
    info.playback_internal_period = dev_.playback.internalPeriodSizeInFrames;
    return info;
}

fluxvoice::Result<bool> MiniaudioBackend::start_stream() {
    if (!dev_ok_)
        return fluxvoice::AudioError{fluxvoice::AudioErrorCategory::DeviceNotFound,
                                     "start stream", as_int(MA_DEVICE_NOT_INITIALIZED),
                                     backend_name(), "No stream is open to start."};
    const ma_result r1 = ma_device_start(&dev_);
    if (r1 != MA_SUCCESS) return make_error(r1, "start stream");
    if (play_dev_ok_) {
        const ma_result r2 = ma_device_start(&play_dev_);
        if (r2 != MA_SUCCESS) {
            ma_device_stop(&dev_);
            return make_error(r2, "start playback stream");
        }
    }
    return true;
}

void MiniaudioBackend::stop_stream() {
    if (play_dev_ok_) {
        ma_device_uninit(&play_dev_);
        play_dev_ok_ = false;
    }
    if (dev_ok_) {
        ma_device_uninit(&dev_);
        dev_ok_ = false;
    }
}

bool MiniaudioBackend::is_stream_running() const {
    if (!dev_ok_) return false;
    if (ma_device_get_state(&dev_) != ma_device_state_started) return false;
    if (play_dev_ok_ && ma_device_get_state(&play_dev_) != ma_device_state_started) return false;
    return true;
}

std::uint32_t MiniaudioBackend::drain_notifications() {
    return pending_notifications_.exchange(fluxvoice::NotificationBits::kNone,
                                           std::memory_order_acq_rel);
}

bool MiniaudioBackend::auto_routes_default() const {
    return ok_ && (ctx_.backend == ma_backend_wasapi || ctx_.backend == ma_backend_coreaudio);
}

void MiniaudioBackend::on_data(ma_device* dev, void* output, const void* input,
                               ma_uint32 frame_count) {
    auto* self = static_cast<MiniaudioBackend*>(dev->pUserData);
    if (self && self->process_fn_) {
        self->process_fn_(self->process_user_data_, static_cast<float*>(output),
                          static_cast<const float*>(input),
                          static_cast<std::size_t>(frame_count) * kStreamChannels);
    }
}

void MiniaudioBackend::on_notification(const ma_device_notification* notification) {
    if (!notification || !notification->pDevice) return;
    auto* self = static_cast<MiniaudioBackend*>(notification->pDevice->pUserData);
    if (!self) return;

    std::uint32_t bit = fluxvoice::NotificationBits::kNone;
    switch (notification->type) {
        case ma_device_notification_type_started:
            bit = fluxvoice::NotificationBits::kStarted;
            break;
        case ma_device_notification_type_stopped:
            bit = fluxvoice::NotificationBits::kStopped;
            break;
        case ma_device_notification_type_rerouted:
            bit = fluxvoice::NotificationBits::kRerouted;
            break;
        case ma_device_notification_type_interruption_began:
            bit = fluxvoice::NotificationBits::kInterruptionBegan;
            break;
        case ma_device_notification_type_interruption_ended:
            bit = fluxvoice::NotificationBits::kInterruptionEnded;
            break;
        case ma_device_notification_type_unlocked:
            bit = fluxvoice::NotificationBits::kUnlocked;
            break;
    }
    if (bit != fluxvoice::NotificationBits::kNone)
        self->pending_notifications_.fetch_or(bit, std::memory_order_relaxed);
}

std::optional<ma_device_id> MiniaudioBackend::resolve(
    const fluxvoice::EndpointKey& key) const {
    if (!ok_ || key.epoch != epoch_) return std::nullopt;
    const IdTable& table = key.direction == fluxvoice::EndpointDirection::Capture
                               ? capture_ids_
                               : playback_ids_;
    for (const IdEntry& e : table)
        if (e.token == key.token) return e.id;
    return std::nullopt;
}

std::uint64_t MiniaudioBackend::intern(IdTable& table, const ma_device_id& id) {
    for (const IdEntry& e : table)
        if (ma_device_id_equal(&e.id, &id)) return e.token;
    table.push_back(IdEntry{id, next_token_});
    return next_token_++;
}
