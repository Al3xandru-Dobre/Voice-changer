// RouteGuard — refuses capture/playback pairs that would feed FluxVoice's own
// output back into its input (a virtual cable read from its far end).
//
// Recognised pairs, by the endpoint names miniaudio reports:
//
//   capture                                  playback
//   "Monitor of X"                        <- "X"                                (PulseAudio / PipeWire sink monitor)
//   "CABLE Output (VB-Audio Virtual Cable)" <- "CABLE Input (VB-Audio Virtual Cable)"
//                                            (any "... Input ..." / "... Output ..." pair: CABLE-A/B, Hi-Fi Cable)
//   "BlackHole 2ch"                       <- "BlackHole 2ch"                    (macOS; one device, both directions)
//
// A name heuristic, good enough for the Milestone 1 CLI. It cannot see routes
// it has no name for (e.g. a PipeWire virtual source layered on the sink).
// Milestone 1.5 / Stage 06 replaces it with backend metadata.
#pragma once

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
