// DeviceSelectorModel — rows <-> selection policies, feedback preview.
#include "core/DeviceSelectorModel.hpp"

#include "audio/AudioTypes.hpp"
#include "audio/RouteGuard.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace fluxvoice::frontend {

namespace {

const std::vector<EndpointDescriptor>& list_for(const DeviceSnapshot& snap, EndpointDirection direction,
                                                CaptureMode mode) {
    if (direction == EndpointDirection::Capture && mode == CaptureMode::SystemPlaybackLoopback)
        return snap.playback;
    return snap.endpoints(direction);
}

const EndpointDescriptor* find_key(const std::vector<EndpointDescriptor>& list, const EndpointKey& key) {
    for (const EndpointDescriptor& d : list)
        if (d.key == key) return &d;
    return nullptr;
}

const EndpointDescriptor* find_default(const std::vector<EndpointDescriptor>& list) {
    for (const EndpointDescriptor& d : list)
        if (d.is_default) return &d;
    return nullptr;
}

const EndpointDescriptor* descriptor_for(const DeviceSelection& sel, const std::vector<EndpointDescriptor>& list) {
    if (const auto* e = std::get_if<ExplicitEndpoint>(&sel)) return find_key(list, e->key);
    return find_default(list);
}

} // namespace

DeviceRows build_rows(const DeviceSnapshot& snap, EndpointDirection direction, bool stale, CaptureMode capture_mode) {
    DeviceRows out;
    out.generation = snap.generation;
    out.backend    = snap.backend;
    out.stale      = stale;
    DeviceRow def;
    def.label          = "Follow system default";
    def.follow_default = true;
    out.rows.push_back(def);
    for (const EndpointDescriptor& d : list_for(snap, direction, capture_mode)) {
        DeviceRow r;
        r.label = d.name;
        if (d.is_default) r.label += "  (default)";
        if (d.is_monitor == Capability::Supported) r.label += "  [monitor]";
        r.key        = d.key;
        r.is_default = d.is_default;
        r.is_monitor = d.is_monitor;
        out.rows.push_back(r);
    }
    return out;
}

DeviceSelection selection_for_row(const DeviceRows& rows, std::size_t row) {
    if (row == 0 || row >= rows.rows.size() || rows.rows[row].follow_default) return FollowSystemDefault{};
    return ExplicitEndpoint{rows.rows[row].key, rows.generation};
}

std::optional<std::size_t> row_for_selection(const DeviceRows& rows, const DeviceSelection& sel) {
    if (std::holds_alternative<FollowSystemDefault>(sel)) return std::size_t{0};
    const auto& e = std::get<ExplicitEndpoint>(sel);
    for (std::size_t i = 1; i < rows.rows.size(); ++i)
        if (rows.rows[i].key == e.key) return i;
    return std::nullopt;
}

std::string describe_selection(const DeviceSelection& sel, const DeviceSnapshot& snap, EndpointDirection direction,
                               CaptureMode capture_mode) {
    if (std::holds_alternative<FollowSystemDefault>(sel)) return "Follow system default";
    const auto& e = std::get<ExplicitEndpoint>(sel);
    if (const EndpointDescriptor* d = find_key(list_for(snap, direction, capture_mode), e.key)) return d->name;
    return "(device no longer listed)";
}

std::string resolved_name(const DeviceSelection& sel, const DeviceSnapshot& snap, EndpointDirection direction,
                          CaptureMode capture_mode) {
    const EndpointDescriptor* d = descriptor_for(sel, list_for(snap, direction, capture_mode));
    return d ? d->name : std::string();
}

std::optional<std::string> feedback_preview(const DeviceSnapshot& snap, const DeviceSelection& capture,
                                            const DeviceSelection& playback, CaptureMode capture_mode) {
    const EndpointDescriptor* cap  = descriptor_for(capture, list_for(snap, EndpointDirection::Capture, capture_mode));
    const EndpointDescriptor* play = descriptor_for(playback, snap.playback);
    const std::string cap_name  = cap ? cap->name : std::string();
    const std::string play_name = play ? play->name : std::string();
    if (!fluxvoice::is_feedback_route(cap, play, cap_name, play_name, capture_mode)) return std::nullopt;
    if (capture_mode == CaptureMode::SystemPlaybackLoopback)
        return "This would capture FluxVoice's own playback endpoint ('" + play_name +
               "') and feed it back into the input. Choose a different playback device.";
    return "'" + cap_name + "' records what is played to '" + play_name +
           "': the output would feed back into the input. Pick the microphone (or the cable's "
           "other end) as capture.";
}

std::string explain_error(const AudioError& err) {
    std::string s = std::string(to_string(err.category));
    if (!err.operation.empty()) s += " while trying to " + err.operation;
    if (!err.message.empty()) s += ": " + err.message;
    switch (err.category) {
        case AudioErrorCategory::DeviceNotFound:
        case AudioErrorCategory::DeviceChanged:
            s += " Choose the device again from the current list; FluxVoice never substitutes another one.";
            break;
        case AudioErrorCategory::PermissionDenied:
            s += " Allow microphone access for FluxVoice in the OS privacy settings, then retry.";
            break;
        case AudioErrorCategory::DeviceBusy:
            s += " Another application holds the device exclusively; close it or pick another device.";
            break;
        case AudioErrorCategory::NullBackendOnly:
            s += " No audio hardware is reachable through this backend (offline/test mode).";
            break;
        default:
            break;
    }
    return s;
}

} // namespace fluxvoice::frontend
