// AudioDeviceRegistry — implementation (control thread only).
#include "audio/AudioDeviceRegistry.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>

namespace fluxvoice {

Result<DeviceSnapshot> AudioDeviceRegistry::refresh() {
    const BackendStatus st = status();
    if (st.state == BackendState::Unavailable) {
        // Same rule as the other failure exits: a previously published
        // snapshot is retained but marked stale, never presented as fresh.
        if (published_.generation != 0) stale_ = true;
        return AudioError{AudioErrorCategory::BackendUnavailable, "refresh device list",
                          st.native_result, st.backend,
                          "No audio backend is available (initialisation failed)."};
    }
    if (st.state == BackendState::NullOnly) {
        // Successful context init is not proof of hardware: the null backend
        // "works" with no devices. Report offline mode, never an empty
        // working catalog (research §5).
        if (published_.generation != 0) stale_ = true;
        return AudioError{AudioErrorCategory::NullBackendOnly, "refresh device list",
                          st.native_result, st.backend,
                          "Offline/test mode: the null backend has no audio hardware; "
                          "device selection is unavailable."};
    }

    auto r = backend_.enumerate();
    if (!is_ok(r)) {
        AudioError err = std::get<AudioError>(std::move(r));
        if (published_.generation != 0) stale_ = true;  // retain as stale, not fresh
        return err;
    }
    DeviceSnapshot fresh = std::move(std::get<DeviceSnapshot>(r));

    if (published_.generation != 0 && published_.epoch == fresh.epoch &&
        same_catalog(published_.capture, fresh.capture) &&
        same_catalog(published_.playback, fresh.playback)) {
        stale_ = false;
        return published_;  // content unchanged: keep the published generation
    }

    fresh.generation = ++generation_;
    published_       = std::move(fresh);
    stale_           = false;
    return published_;
}

Result<EndpointInfo> AudioDeviceRegistry::capabilities(const EndpointKey& key) const {
    const BackendStatus st = status();
    if (st.state == BackendState::Unavailable)
        return AudioError{AudioErrorCategory::BackendUnavailable, "probe device",
                          st.native_result, st.backend, "No audio backend is available."};
    if (st.state == BackendState::NullOnly)
        return AudioError{AudioErrorCategory::NullBackendOnly, "probe device",
                          st.native_result, st.backend,
                          "Offline/test mode: the null backend has no audio hardware."};
    return backend_.probe(key);
}

Result<ResolvedEndpoint> AudioDeviceRegistry::resolve(const DeviceSelection& sel,
                                                      EndpointDirection dir,
                                                      bool strict_generation) const {
    const BackendStatus st = status();
    if (st.state == BackendState::Unavailable)
        return AudioError{AudioErrorCategory::BackendUnavailable, "resolve selection",
                          st.native_result, st.backend, "No audio backend is available."};
    if (st.state == BackendState::NullOnly)
        return AudioError{AudioErrorCategory::NullBackendOnly, "resolve selection",
                          st.native_result, st.backend,
                          "Offline/test mode: the null backend has no audio hardware."};

    if (std::holds_alternative<FollowSystemDefault>(sel)) {
        const EndpointDescriptor* def = default_descriptor(dir);
        return ResolvedEndpoint{true, def ? def->key : EndpointKey{}, published_.generation,
                                def ? def->name : std::string{}};
    }

    const ExplicitEndpoint& exp = std::get<ExplicitEndpoint>(sel);
    const EndpointKey& key      = exp.key;

    if (published_.generation == 0)
        return AudioError{AudioErrorCategory::DeviceChanged, "resolve selection", 0,
                          st.backend,
                          "No device catalog yet - list devices before selecting one."};
    if (key.epoch != published_.epoch)
        return AudioError{AudioErrorCategory::DeviceChanged, "resolve selection", 0,
                          st.backend,
                          "The audio context was recreated since this device was "
                          "selected - choose again."};
    if (strict_generation && exp.generation != 0 && exp.generation != published_.generation)
        return AudioError{AudioErrorCategory::DeviceChanged, "resolve selection", 0,
                          st.backend,
                          "The device catalog changed since this selection was made - "
                          "refresh and verify the selected device."};
    if (key.direction != dir)
        return AudioError{AudioErrorCategory::DeviceNotFound, "resolve selection", 0,
                          st.backend,
                          "The selected key belongs to the other endpoint direction."};

    const std::vector<EndpointDescriptor>& endpoints = published_.endpoints(dir);
    const auto it = std::find_if(endpoints.begin(), endpoints.end(),
                                 [&](const EndpointDescriptor& d) { return d.key == key; });
    if (it == endpoints.end())
        return AudioError{AudioErrorCategory::DeviceNotFound, "resolve selection", 0,
                          st.backend,
                          "The selected device is no longer present - pick another one."};

    return ResolvedEndpoint{false, it->key, published_.generation, it->name};
}

Result<ResolvedEndpoint> AudioDeviceRegistry::resolve_loopback(const DeviceSelection& sel,
                                                               bool strict_generation) const {
    const BackendStatus st = status();
    if (st.state == BackendState::Unavailable)
        return AudioError{AudioErrorCategory::BackendUnavailable,
                          "resolve system-playback loopback", st.native_result, st.backend,
                          "No audio backend is available."};
    if (st.state == BackendState::NullOnly)
        return AudioError{AudioErrorCategory::NullBackendOnly,
                          "resolve system-playback loopback", st.native_result, st.backend,
                          "Offline/test mode: the null backend has no audio hardware."};

    const AudioCapabilities caps = backend_.capabilities();
    if (caps.native_loopback != Capability::Supported) {
        std::string msg = "System-playback loopback capture is not supported on backend '" +
                          st.backend + "'.";
        if (st.backend == "coreaudio") {
            msg += " macOS Core Audio does not provide native loopback via miniaudio 0.11.25; "
                   "configure a virtual audio route (e.g. BlackHole) and select it with --input.";
        } else if (st.backend == "pulseaudio" || st.backend == "alsa") {
            msg += " On Linux PulseAudio/PipeWire, select the sink's monitor source as an "
                   "ordinary capture device with --input.";
        }
        return AudioError{AudioErrorCategory::FormatUnsupported,
                          "resolve system-playback loopback", 0, st.backend, std::move(msg)};
    }

    // Native WASAPI loopback captures a playback (render) endpoint.
    return resolve(sel, EndpointDirection::Playback, strict_generation);
}

Result<DeviceSelection> AudioDeviceRegistry::resolve_persistent_token(
    std::string_view token, EndpointDirection dir) const {
    const BackendStatus st = status();
    if (token == "default") return DeviceSelection{FollowSystemDefault{}};

    if (published_.generation == 0)
        return AudioError{AudioErrorCategory::DeviceChanged, "resolve persistent token", 0,
                          st.backend, "No device catalog published yet."};

    const std::string expected_prefix =
        "v1:" + published_.backend + ":" + to_string(dir) + ":";
    if (!token.starts_with(expected_prefix))
        return AudioError{AudioErrorCategory::DeviceChanged, "resolve persistent token", 0,
                          st.backend,
                          "Saved device token belongs to another backend, direction or version - "
                          "please select a device again."};

    const std::vector<EndpointDescriptor>& endpoints = published_.endpoints(dir);
    const EndpointDescriptor* match = nullptr;
    for (const EndpointDescriptor& d : endpoints) {
        const std::string candidate =
            !d.persistent_token.empty() ? d.persistent_token : (expected_prefix + d.name);
        if (candidate == token) {
            if (match != nullptr) {
                return AudioError{
                    AudioErrorCategory::DeviceChanged, "resolve persistent token", 0, st.backend,
                    "Saved device token matches multiple identical endpoints - please select "
                    "which one to use."};
            }
            match = &d;
        }
    }
    if (!match)
        return AudioError{AudioErrorCategory::DeviceNotFound, "resolve persistent token", 0,
                          st.backend, "Saved device is not currently connected."};

    return DeviceSelection{ExplicitEndpoint{match->key, published_.generation}};
}

const EndpointDescriptor* AudioDeviceRegistry::find_descriptor(const EndpointKey& key) const {
    if (published_.generation == 0 || key.epoch != published_.epoch) return nullptr;
    const auto& list = published_.endpoints(key.direction);
    const auto it = std::find_if(list.begin(), list.end(),
                                 [&](const EndpointDescriptor& d) { return d.key == key; });
    return it != list.end() ? &*it : nullptr;
}

const EndpointDescriptor* AudioDeviceRegistry::default_descriptor(EndpointDirection dir) const {
    if (published_.generation == 0) return nullptr;
    const auto& list = published_.endpoints(dir);
    const auto it = std::find_if(list.begin(), list.end(),
                                 [](const EndpointDescriptor& d) { return d.is_default; });
    return it != list.end() ? &*it : nullptr;
}

bool AudioDeviceRegistry::same_catalog(const std::vector<EndpointDescriptor>& a,
                                       const std::vector<EndpointDescriptor>& b) {
    if (a.size() != b.size()) return false;
    for (const EndpointDescriptor& da : a) {
        const auto it =
            std::find_if(b.begin(), b.end(), [&](const EndpointDescriptor& db) {
                return db.key == da.key;
            });
        if (it == b.end() || it->name != da.name || it->is_default != da.is_default ||
            it->is_monitor != da.is_monitor || it->monitor_of != da.monitor_of)
            return false;
    }
    return true;
}

} // namespace fluxvoice
