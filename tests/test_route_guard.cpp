// RouteGuard — feedback-route detection tests (Milestone 1 / Stage 05).
//
// Names are the ones miniaudio reports on each backend.
#include "audio/RouteGuard.hpp"
#include "TestSupport.hpp"

namespace {

void test_detects_virtual_cable_far_end() {
    // PulseAudio / PipeWire null sink and its monitor.
    CHECK(is_feedback_route("Monitor of FluxVoice-Out", "FluxVoice-Out"));
    // VB-Cable and its A/B variants (WASAPI friendly names).
    CHECK(is_feedback_route("CABLE Output (VB-Audio Virtual Cable)",
                            "CABLE Input (VB-Audio Virtual Cable)"));
    CHECK(is_feedback_route("CABLE-A Output (VB-Audio Cable A)", "CABLE-A Input (VB-Audio Cable A)"));
    CHECK(is_feedback_route("Hi-Fi Cable Output (VB-Audio Hi-Fi Cable)",
                            "Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)"));
    // BlackHole is one device in both directions.
    CHECK(is_feedback_route("BlackHole 2ch", "BlackHole 2ch"));
}

void test_allows_real_routes() {
    // Microphone into a cable: the product route.
    CHECK(!is_feedback_route("Microphone Array (Intel Smart Sound Technology)",
                             "CABLE Input (VB-Audio Virtual Cable)"));
    CHECK(!is_feedback_route("sof-hda-dsp Digital Microphone", "FluxVoice-Out"));
    // A different cable's far end: two-cable measurement setups.
    CHECK(!is_feedback_route("CABLE-A Output (VB-Audio Cable A)",
                             "CABLE-B Input (VB-Audio Cable B)"));
    CHECK(!is_feedback_route("CABLE Output (VB-Audio Virtual Cable)",
                             "CABLE-A Input (VB-Audio Cable A)"));
    // Another sink's monitor (the Linux test source), or a name that only
    // starts like ours.
    CHECK(!is_feedback_route("Monitor of FluxVoice-Test-Src", "FluxVoice-Test-Out"));
    CHECK(!is_feedback_route("Monitor of FluxVoice-Out", "FluxVoice-Out2"));
    CHECK(!is_feedback_route("Monitor of FluxVoice-Out2", "FluxVoice-Out"));
    CHECK(!is_feedback_route("Monitor of Old FluxVoice-Out", "FluxVoice-Out"));
    // USB headsets often use one name for both directions.
    CHECK(!is_feedback_route("Jabra EVOLVE 30", "Jabra EVOLVE 30"));
    CHECK(!is_feedback_route("BlackHole 2ch", "BlackHole 16ch"));
    // Nothing known about an unnamed endpoint.
    CHECK(!is_feedback_route("", "CABLE Input (VB-Audio Virtual Cable)"));
    CHECK(!is_feedback_route("Monitor of ", ""));
}

void test_direction_matters() {
    // Recording the cable's playback end is not something a capture device
    // can be, but the pair must not match the wrong way round either.
    CHECK(!is_feedback_route("CABLE Input (VB-Audio Virtual Cable)",
                             "CABLE Output (VB-Audio Virtual Cable)"));
    CHECK(!is_feedback_route("FluxVoice-Out", "Monitor of FluxVoice-Out"));
}

void test_metadata_and_loopback_feedback_guard() {
    using namespace fluxvoice;

    const EndpointKey sink_a{EndpointDirection::Playback, 1, 10};
    const EndpointKey sink_b{EndpointDirection::Playback, 1, 11};
    const EndpointKey mon_a{EndpointDirection::Capture, 1, 20};

    EndpointDescriptor play_a{sink_a, "FluxVoice Virtual Sink", false, ""};
    EndpointDescriptor play_b{sink_b, "Headphones", true, ""};

    // Native server metadata links mon_a to sink_a even when the display name
    // has no "Monitor of " prefix.
    EndpointDescriptor cap_mon{mon_a, "Custom Remapped Source", false, "",
                               Capability::Supported, sink_a};
    CHECK(is_feedback_route(&cap_mon, &play_a, cap_mon.name, play_a.name,
                            CaptureMode::OrdinaryMicrophone));
    CHECK(!is_feedback_route(&cap_mon, &play_b, cap_mon.name, play_b.name,
                             CaptureMode::OrdinaryMicrophone));

    // System-playback loopback: capturing the same playback endpoint we play
    // into is a feedback loop; capturing another playback endpoint is valid.
    CHECK(is_feedback_route(&play_a, &play_a, play_a.name, play_a.name,
                            CaptureMode::SystemPlaybackLoopback));
    CHECK(!is_feedback_route(&play_b, &play_a, play_b.name, play_a.name,
                             CaptureMode::SystemPlaybackLoopback));
}

} // namespace

int main() {
    test_detects_virtual_cable_far_end();
    test_allows_real_routes();
    test_direction_matters();
    test_metadata_and_loopback_feedback_guard();
    return test::finish("route_guard");
}
