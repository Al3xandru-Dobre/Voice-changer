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

} // namespace

int main() {
    test_detects_virtual_cable_far_end();
    test_allows_real_routes();
    test_direction_matters();
    return test::finish("route_guard");
}
