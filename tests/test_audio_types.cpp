// AudioTypes — contract tests (Milestone 1.5 / Stage 02).
//
// This TU deliberately includes ONLY AudioTypes.hpp: it must compile without
// miniaudio, proving the application-facing vocabulary carries no native
// device identifiers (the target links no audio dependency at all).
#include "audio/AudioTypes.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

using namespace fluxvoice;

EndpointKey key(EndpointDirection d, ContextEpoch e, std::uint64_t t) {
    return EndpointKey{d, e, t};
}

void test_types_exist_without_miniaudio() {
    CHECK(to_string(EndpointDirection::Capture) == std::string("capture"));
    CHECK(to_string(EndpointDirection::Playback) == std::string("playback"));

    // Descriptors aggregate-initialize from plain data.
    const EndpointDescriptor d{{EndpointDirection::Capture, 1, 7}, "USB Mic", true, ""};
    CHECK(d.key.direction == EndpointDirection::Capture);
    CHECK(d.key.epoch == 1);
    CHECK(d.key.token == 7);
    CHECK(d.name == "USB Mic");
    CHECK(d.is_default);
    CHECK(d.persistent_token.empty());

    // A default snapshot is the "nothing published yet" state.
    const DeviceSnapshot snap;
    CHECK(snap.capture.empty());
    CHECK(snap.playback.empty());
    CHECK(snap.backend.empty());
    CHECK(snap.epoch == 0);
    CHECK(snap.generation == 0);
    CHECK(snap.endpoints(EndpointDirection::Capture).empty());
    CHECK(snap.endpoints(EndpointDirection::Playback).empty());

    // endpoints() picks the requested direction.
    DeviceSnapshot two;
    two.capture.push_back(d);
    CHECK(two.endpoints(EndpointDirection::Capture).size() == std::size_t{1});
    CHECK(two.endpoints(EndpointDirection::Playback).size() == std::size_t{0});
}

void test_key_equality_is_direction_and_epoch_scoped() {
    const EndpointKey a = key(EndpointDirection::Capture, 1, 7);
    CHECK(a == key(EndpointDirection::Capture, 1, 7));
    CHECK(!(a != key(EndpointDirection::Capture, 1, 7)));

    // Same token, other direction → a different endpoint (never infer
    // direction from a name).
    CHECK(a != key(EndpointDirection::Playback, 1, 7));
    // Same direction and token, other epoch → minted by another context;
    // a recreated backend must not honor it.
    CHECK(a != key(EndpointDirection::Capture, 2, 7));
    CHECK(a != key(EndpointDirection::Capture, 1, 8));

    // Epoch 0 is never minted by a backend — the default key matches nothing.
    const EndpointKey none{};
    CHECK(none.epoch == 0);
    CHECK(none != key(EndpointDirection::Capture, 1, 0));
}

void test_key_order_is_total_and_agrees_with_equality() {
    const std::vector<EndpointKey> keys = {
        key(EndpointDirection::Playback, 1, 3), key(EndpointDirection::Capture, 2, 1),
        key(EndpointDirection::Capture, 1, 2), key(EndpointDirection::Capture, 1, 1),
        key(EndpointDirection::Capture, 1, 2),  // duplicate on purpose
    };
    std::vector<EndpointKey> sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    CHECK(std::is_sorted(sorted.begin(), sorted.end()));
    // Equal keys sort adjacently.
    CHECK(sorted.size() == std::size_t{5});
    bool adjacent_equal = false;
    for (size_t i = 1; i < sorted.size(); ++i)
        if (sorted[i] == sorted[i - 1]) adjacent_equal = true;
    CHECK(adjacent_equal);
    // Direction orders before epoch and token.
    CHECK(key(EndpointDirection::Capture, 9, 9) < key(EndpointDirection::Playback, 1, 1));
    CHECK(!(key(EndpointDirection::Playback, 9, 9) < key(EndpointDirection::Capture, 1, 1)));
    // Within one direction, epoch orders before token.
    CHECK(key(EndpointDirection::Capture, 1, 99) < key(EndpointDirection::Capture, 2, 1));
}

void test_selection_is_a_policy_not_an_index() {
    const DeviceSelection follow{};
    CHECK(std::holds_alternative<FollowSystemDefault>(follow));

    const DeviceSelection mic = ExplicitEndpoint{key(EndpointDirection::Capture, 3, 11)};
    const ExplicitEndpoint& sel = std::get<ExplicitEndpoint>(mic);
    CHECK(sel.key == key(EndpointDirection::Capture, 3, 11));

    CHECK(follow != mic);  // a default policy is never equal to an endpoint
}

void test_capabilities_keep_unknown_apart_from_unsupported() {
    const AudioCapabilities unprobed;
    CHECK(unprobed.ordinary_io == Capability::Unknown);
    CHECK(unprobed.native_loopback == Capability::Unknown);
    CHECK(unprobed.monitor_capture == Capability::Unknown);
    CHECK(unprobed.virtual_routing == Capability::Unknown);

    // "Not probed" must never collapse into a verdict.
    CHECK(Capability::Unknown != Capability::Unsupported);
    CHECK(Capability::Unknown != Capability::Supported);
    CHECK(Capability::Unsupported != Capability::Supported);
}

void test_errors_carry_category_native_result_and_message() {
    const AudioError e{AudioErrorCategory::EnumerationFailed, "enumerate devices", -100,
                       "pulseaudio", "The audio backend failed to list devices."};
    CHECK(e.category == AudioErrorCategory::EnumerationFailed);
    CHECK(e.operation == "enumerate devices");
    CHECK(e.native_result == -100);
    CHECK(e.backend == "pulseaudio");
    CHECK(!e.message.empty());
    CHECK(e.message == "The audio backend failed to list devices.");

    // Every category keeps a distinct, non-empty printable name.
    const AudioErrorCategory all[] = {
        AudioErrorCategory::BackendUnavailable, AudioErrorCategory::NullBackendOnly,
        AudioErrorCategory::EnumerationFailed,  AudioErrorCategory::DeviceNotFound,
        AudioErrorCategory::DeviceChanged,      AudioErrorCategory::DeviceBusy,
        AudioErrorCategory::PermissionDenied,   AudioErrorCategory::FormatUnsupported,
        AudioErrorCategory::Unknown,
    };
    std::vector<std::string> names;
    for (const AudioErrorCategory c : all) names.emplace_back(to_string(c));
    for (const std::string& n : names) CHECK(!n.empty());
    std::sort(names.begin(), names.end());
    CHECK(std::adjacent_find(names.begin(), names.end()) == names.end());
    CHECK(to_string(AudioErrorCategory::Unknown) == std::string("unknown"));

    // Result<T>: value sits in alternative 0, error in alternative 1.
    const Result<int> ok_value{42};
    const Result<int> failed{AudioError{AudioErrorCategory::DeviceBusy, "open capture", 0, "", ""}};
    CHECK(is_ok(ok_value));
    CHECK(!is_ok(failed));
    CHECK(std::get<int>(ok_value) == 42);
    CHECK(std::get<AudioError>(failed).category == AudioErrorCategory::DeviceBusy);
}

void test_stream_status_defaults_to_idle() {
    const StreamStatus s;
    CHECK(s.state == StreamState::Idle);
    CHECK(std::holds_alternative<FollowSystemDefault>(s.requested_capture));
    CHECK(std::holds_alternative<FollowSystemDefault>(s.requested_playback));
    CHECK(!s.resolved_capture.has_value());
    CHECK(!s.resolved_playback.has_value());
    CHECK(s.backend.empty());
    CHECK(s.client_sample_rate == 0);
    CHECK(s.capture_backend_rate == 0);
    CHECK(s.playback_backend_rate == 0);

    // The tri-state lifecycle keeps its names (controller states, stage 05).
    CHECK(StreamState::Idle != StreamState::Running);
    CHECK(StreamState::Recovering != StreamState::Error);
}

void test_stage03_04_vocabulary_defaults_safely() {
    // BackendStatus: "nothing initialised yet" is distinguishable from a
    // real backend and from offline mode.
    const BackendStatus st;
    CHECK(st.state == BackendState::Unavailable);
    CHECK(st.backend.empty());
    CHECK(st.native_result == 0);
    CHECK(BackendState::Unavailable != BackendState::Ready);
    CHECK(BackendState::Ready != BackendState::NullOnly);

    // NativeFormat / EndpointInfo: empty probe results stay explicit.
    const NativeFormat nf;
    CHECK(nf.sample_rate == 0);
    CHECK(nf.channels == 0);
    CHECK(nf.format == 0);
    const EndpointInfo info;
    CHECK(info.key == EndpointKey{});
    CHECK(info.name.empty());
    CHECK(!info.is_default);
    CHECK(info.native_formats.empty());

    // ResolvedEndpoint: "nothing resolved yet".
    const ResolvedEndpoint re;
    CHECK(!re.use_default);
    CHECK(re.key == EndpointKey{});
    CHECK(re.generation == 0);
    CHECK(re.name.empty());
}

} // namespace

int main() {
    test_types_exist_without_miniaudio();
    test_key_equality_is_direction_and_epoch_scoped();
    test_key_order_is_total_and_agrees_with_equality();
    test_selection_is_a_policy_not_an_index();
    test_capabilities_keep_unknown_apart_from_unsupported();
    test_errors_carry_category_native_result_and_message();
    test_stream_status_defaults_to_idle();
    test_stage03_04_vocabulary_defaults_safely();
    return test::finish("audio_types");
}
