// RouteGuard — refuses capture/playback pairs that would feed FluxVoice's own
// output back into its input (a virtual cable read from its far end, a sink
// monitor recording its own playback sink, or a system-playback loopback
// capturing its own playback endpoint).
//
// Milestone 1.5 / Stage 06:
//   1. Backend/adapter metadata is checked first when available
//      (`EndpointDescriptor::monitor_of == playback_key`, or
//      `CaptureMode::SystemPlaybackLoopback` targeting the same playback
//      endpoint).
//   2. Endpoint-name heuristics remain as a fallback when backend monitor
//      metadata is `Capability::Unknown` (miniaudio 0.11.25 does not expose
//      PulseAudio's `monitor_of_sink` field during basic enumeration):
//
//   capture                                  playback
//   "Monitor of X"                        <- "X"                                (PulseAudio / PipeWire sink monitor)
//   "CABLE Output (VB-Audio Virtual Cable)" <- "CABLE Input (VB-Audio Virtual Cable)"
//                                            (any "... Input ..." / "... Output ..." pair: CABLE-A/B, Hi-Fi Cable)
//   "BlackHole 2ch"                       <- "BlackHole 2ch"                    (macOS; one device, both directions)
#pragma once

#include "audio/AudioTypes.hpp"

#include <string>
#include <string_view>

/// True when recording `capture` would record what is played to `playback`.
inline bool is_feedback_route(std::string_view capture, std::string_view playback) {
    if (capture.empty() || playback.empty()) return false;

    constexpr std::string_view kMonitor = "Monitor of ";
    if (capture.starts_with(kMonitor) && capture.substr(kMonitor.size()) == playback)
        return true;

    // VB-Audio cables: the playback end is "<cable> Input", the capture end
    // "<cable> Output", with the rest of the name identical.
    constexpr std::string_view kIn = "Input", kOut = "Output";
    for (size_t p = playback.find(kIn); p != std::string_view::npos; p = playback.find(kIn, p + 1)) {
        std::string far_end(playback);
        far_end.replace(p, kIn.size(), kOut);
        if (capture == far_end) return true;
    }

    // Same device name in both directions is normal for USB headsets, so only
    // the known virtual device counts.
    return capture == playback && capture.starts_with("BlackHole");
}

namespace fluxvoice {

/// Metadata-first feedback guard (M1.5 stage 06).
/// Checks:
///   1. System-playback loopback targeting the same playback endpoint (by key
///      or by opened device name) as FluxVoice's output.
///   2. Native server monitor metadata (`capture_desc->monitor_of` matching
///      the playback endpoint key), independent of display names.
///   3. Name-based fallback (`is_feedback_route(capture_name, playback_name)`).
inline bool is_feedback_route(const EndpointDescriptor* capture_desc,
                              const EndpointDescriptor* playback_desc,
                              std::string_view opened_capture_name,
                              std::string_view opened_playback_name,
                              CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone) {
    if (capture_mode == CaptureMode::SystemPlaybackLoopback) {
        if (capture_desc && playback_desc && capture_desc->key == playback_desc->key)
            return true;
        if (!opened_capture_name.empty() && opened_capture_name == opened_playback_name)
            return true;
    }
    if (capture_desc && playback_desc && capture_desc->monitor_of.has_value() &&
        *capture_desc->monitor_of == playback_desc->key) {
        return true;
    }
    return ::is_feedback_route(opened_capture_name, opened_playback_name);
}

} // namespace fluxvoice
