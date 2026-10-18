// AudioDeviceRegistry — deterministic fake-backend tests (Milestone 1.5 /
// Stages 03–04).
//
// Verifies without audio hardware:
//   * 0 / 1 / many devices per direction; duplicate display names keep
//     distinct keys;
//   * snapshot generation advances only when content changes (reorder alone
//     keeps the same generation);
//   * empty catalog vs. enumeration failure vs. backend unavailable vs. null
//     backend produce four distinct outcomes;
//   * failed refresh retains the previous snapshot marked stale;
//   * on-demand capabilities(key) probes a selected endpoint without opening
//     a stream;
//   * selection policy: FollowSystemDefault vs. ExplicitEndpoint across list
//     reorders, device disappearance, strict generation races, and context
//     epoch recreation;
//   * persistent tokens: unambiguous resolution vs. duplicate-name rejection.
#include "audio/AudioDeviceRegistry.hpp"
#include "TestSupport.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace fluxvoice;

class FakeBackend : public AudioBackend {
public:
    BackendStatus backend_status{BackendState::Ready, "fake", 0};
    ContextEpoch epoch = 1;
    std::vector<EndpointDescriptor> capture;
    std::vector<EndpointDescriptor> playback;
    std::optional<AudioError> enumerate_error;
    AudioCapabilities caps{Capability::Supported, Capability::Supported,
                           Capability::Unknown, Capability::Unknown};
    std::uint64_t enum_calls  = 0;
    std::uint64_t probe_calls = 0;

    EndpointDescriptor make_ep(EndpointDirection dir, std::uint64_t token,
                               const char* name, bool is_def = false) const {
        EndpointDescriptor d;
        d.key        = EndpointKey{dir, epoch, token};
        d.name       = name;
        d.is_default = is_def;
        d.persistent_token =
            std::string("v1:") + backend_status.backend + ":" + to_string(dir) + ":" + name;
        return d;
    }

    BackendStatus status() const override { return backend_status; }

    Result<DeviceSnapshot> enumerate() override {
        ++enum_calls;
        if (enumerate_error) return *enumerate_error;
        DeviceSnapshot s;
        s.backend    = backend_status.backend;
        s.epoch      = epoch;
        s.generation = enum_calls;
        s.capture    = capture;
        s.playback   = playback;
        return s;
    }

    Result<EndpointInfo> probe(const EndpointKey& key) const override {
        const_cast<FakeBackend*>(this)->probe_calls++;
        if (key.epoch != epoch) {
            return AudioError{AudioErrorCategory::DeviceNotFound, "probe device", 0,
                              backend_status.backend, "Stale epoch"};
        }
        const auto& list = key.direction == EndpointDirection::Capture ? capture : playback;
        for (const auto& d : list) {
            if (d.key == key) {
                EndpointInfo info;
                info.key        = d.key;
                info.name       = d.name;
                info.is_default = d.is_default;
                info.native_formats.push_back(NativeFormat{48000, 1, 5});
                return info;
            }
        }
        return AudioError{AudioErrorCategory::DeviceNotFound, "probe device", 0,
                          backend_status.backend, "Not found"};
    }

    AudioCapabilities capabilities() const override { return caps; }
};

void test_four_distinguishable_outcomes_and_stale_retention() {
    FakeBackend fb;
    AudioDeviceRegistry reg(fb);

    // 1. BackendUnavailable preserves exact native_result.
    fb.backend_status = BackendStatus{BackendState::Unavailable, "", -400};
    auto r_unavail = reg.refresh();
    CHECK(!is_ok(r_unavail));
    CHECK(std::get<AudioError>(r_unavail).category == AudioErrorCategory::BackendUnavailable);
    CHECK(std::get<AudioError>(r_unavail).native_result == -400);
    CHECK(reg.snapshot().generation == 0);

    // 2. NullBackendOnly is reported as offline mode, never as an empty catalog.
    fb.backend_status = BackendStatus{BackendState::NullOnly, "null", 0};
    auto r_null = reg.refresh();
    CHECK(!is_ok(r_null));
    CHECK(std::get<AudioError>(r_null).category == AudioErrorCategory::NullBackendOnly);

    // 3. Ready backend with 0 devices is a valid empty snapshot (generation 1).
    fb.backend_status = BackendStatus{BackendState::Ready, "wasapi", 0};
    auto r_empty = reg.refresh();
    CHECK(is_ok(r_empty));
    CHECK(reg.snapshot().generation == 1);
    CHECK(reg.snapshot().capture.empty());
    CHECK(reg.snapshot().playback.empty());
    CHECK(!reg.stale());

    // Populate devices -> generation 2.
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 2, "Speakers", true)};
    auto r_pop = reg.refresh();
    CHECK(is_ok(r_pop));
    CHECK(reg.snapshot().generation == 2);
    CHECK(!reg.stale());

    // 4. EnumerationFailed retains the previous snapshot and marks it stale.
    fb.enumerate_error = AudioError{AudioErrorCategory::EnumerationFailed, "enumerate devices",
                                    -1, "wasapi", "Backend device query failed"};
    auto r_fail = reg.refresh();
    CHECK(!is_ok(r_fail));
    CHECK(std::get<AudioError>(r_fail).category == AudioErrorCategory::EnumerationFailed);
    CHECK(reg.stale());
    CHECK(reg.snapshot().generation == 2);
    CHECK(reg.snapshot().capture.size() == 1);
    CHECK(reg.snapshot().capture[0].name == "USB Mic");

    // Recovery clears stale without bumping generation if content is unchanged.
    fb.enumerate_error = std::nullopt;
    auto r_rec = reg.refresh();
    CHECK(is_ok(r_rec));
    CHECK(!reg.stale());
    CHECK(reg.snapshot().generation == 2);
}

void test_duplicate_names_and_generation_stability_on_reorder() {
    FakeBackend fb;
    fb.backend_status = BackendStatus{BackendState::Ready, "pulseaudio", 0};
    // Two identical USB microphones with the exact same display name.
    const auto mic1 = fb.make_ep(EndpointDirection::Capture, 10, "Podcast USB Mic", true);
    const auto mic2 = fb.make_ep(EndpointDirection::Capture, 11, "Podcast USB Mic", false);
    const auto spk1 = fb.make_ep(EndpointDirection::Playback, 20, "Headphones", true);
    const auto spk2 = fb.make_ep(EndpointDirection::Playback, 21, "FluxVoice-Out", false);
    fb.capture  = {mic1, mic2};
    fb.playback = {spk1, spk2};

    AudioDeviceRegistry reg(fb);
    auto r1 = reg.refresh();
    CHECK(is_ok(r1));
    CHECK(reg.snapshot().generation == 1);
    CHECK(reg.snapshot().capture.size() == 2);
    CHECK(reg.snapshot().capture[0].key != reg.snapshot().capture[1].key);
    CHECK(reg.snapshot().capture[0].name == reg.snapshot().capture[1].name);

    // Pure reorder does NOT advance generation.
    fb.capture  = {mic2, mic1};
    fb.playback = {spk2, spk1};
    auto r2 = reg.refresh();
    CHECK(is_ok(r2));
    CHECK(reg.snapshot().generation == 1);

    // Default flag change DOES advance generation.
    fb.capture[0].is_default = true;
    fb.capture[1].is_default = false;
    auto r3 = reg.refresh();
    CHECK(is_ok(r3));
    CHECK(reg.snapshot().generation == 2);

    // On-demand probe queries only the requested key.
    CHECK(fb.probe_calls == 0);
    auto p = reg.capabilities(mic2.key);
    CHECK(is_ok(p));
    CHECK(fb.probe_calls == 1);
    CHECK(std::get<EndpointInfo>(p).native_formats.size() == 1);
    CHECK(std::get<EndpointInfo>(p).native_formats[0].sample_rate == 48000);
}

void test_selection_policy_survives_reorder_and_rejects_stale_or_missing() {
    FakeBackend fb;
    fb.backend_status = BackendStatus{BackendState::Ready, "wasapi", 0};
    const auto mic_a = fb.make_ep(EndpointDirection::Capture, 1, "Built-in Mic", true);
    const auto mic_b = fb.make_ep(EndpointDirection::Capture, 2, "Headset Mic", false);
    const auto mic_c = fb.make_ep(EndpointDirection::Capture, 3, "Webcam Mic", false);
    const auto spk   = fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true);
    fb.capture  = {mic_a, mic_b};
    fb.playback = {spk};

    AudioDeviceRegistry reg(fb);
    CHECK(is_ok(reg.refresh()));
    const uint64_t gen1 = reg.snapshot().generation;

    // User selects index 1 ("Headset Mic") against snapshot generation 1.
    const DeviceSelection sel_b = ExplicitEndpoint{reg.snapshot().capture[1].key, gen1};

    // Catalog reorders AND a new device is inserted at index 0 before open:
    // index 1 is now "Built-in Mic", while "Headset Mic" is at index 2.
    fb.capture = {mic_c, mic_a, mic_b};
    CHECK(is_ok(reg.refresh()));
    CHECK(reg.snapshot().generation == gen1 + 1);

    // Strict generation check catches the listing-to-open race.
    auto strict_res = reg.resolve(sel_b, EndpointDirection::Capture, /*strict_generation=*/true);
    CHECK(!is_ok(strict_res));
    CHECK(std::get<AudioError>(strict_res).category == AudioErrorCategory::DeviceChanged);

    // Re-resolving under policy finds "Headset Mic" by key — never "Built-in Mic" at index 1.
    auto policy_res = reg.resolve(sel_b, EndpointDirection::Capture, /*strict_generation=*/false);
    CHECK(is_ok(policy_res));
    CHECK(std::get<ResolvedEndpoint>(policy_res).key == mic_b.key);
    CHECK(std::get<ResolvedEndpoint>(policy_res).name == "Headset Mic");

    // Device disappears between listing and opening -> DeviceNotFound (never fallback).
    fb.capture = {mic_c, mic_a};
    CHECK(is_ok(reg.refresh()));
    auto gone_res = reg.resolve(sel_b, EndpointDirection::Capture);
    CHECK(!is_ok(gone_res));
    CHECK(std::get<AudioError>(gone_res).category == AudioErrorCategory::DeviceNotFound);

    // Wrong direction is rejected.
    auto wrong_dir = reg.resolve(sel_b, EndpointDirection::Playback);
    CHECK(!is_ok(wrong_dir));
    CHECK(std::get<AudioError>(wrong_dir).category == AudioErrorCategory::DeviceNotFound);

    // Context recreated (epoch advances): old keys fail with DeviceChanged even
    // if the new context reuses the same numeric token.
    fb.epoch   = 2;
    fb.capture = {fb.make_ep(EndpointDirection::Capture, 2, "Headset Mic", true)};
    CHECK(is_ok(reg.refresh()));
    auto old_epoch_res = reg.resolve(sel_b, EndpointDirection::Capture);
    CHECK(!is_ok(old_epoch_res));
    CHECK(std::get<AudioError>(old_epoch_res).category == AudioErrorCategory::DeviceChanged);

    // FollowSystemDefault always resolves to use_default = true.
    auto def_res = reg.resolve(FollowSystemDefault{}, EndpointDirection::Capture);
    CHECK(is_ok(def_res));
    CHECK(std::get<ResolvedEndpoint>(def_res).use_default);
}

void test_persistent_tokens_and_ambiguity() {
    FakeBackend fb;
    fb.backend_status = BackendStatus{BackendState::Ready, "wasapi", 0};
    const auto mic1 = fb.make_ep(EndpointDirection::Capture, 1, "Blue Yeti", true);
    const auto mic2 = fb.make_ep(EndpointDirection::Capture, 2, "Logitech Webcam", false);
    fb.capture = {mic1, mic2};

    AudioDeviceRegistry reg(fb);
    CHECK(is_ok(reg.refresh()));

    // "default" persists freely.
    auto def = reg.resolve_persistent_token("default", EndpointDirection::Capture);
    CHECK(is_ok(def));
    CHECK(std::holds_alternative<FollowSystemDefault>(std::get<DeviceSelection>(def)));

    // Unique versioned token resolves to the matching endpoint.
    auto uniq = reg.resolve_persistent_token("v1:wasapi:capture:Logitech Webcam",
                                             EndpointDirection::Capture);
    CHECK(is_ok(uniq));
    CHECK(std::get<ExplicitEndpoint>(std::get<DeviceSelection>(uniq)).key == mic2.key);

    // Token from another backend fails with DeviceChanged.
    auto wrong_backend = reg.resolve_persistent_token("v1:coreaudio:capture:Logitech Webcam",
                                                      EndpointDirection::Capture);
    CHECK(!is_ok(wrong_backend));
    CHECK(std::get<AudioError>(wrong_backend).category == AudioErrorCategory::DeviceChanged);

    // Two identical USB mics with the same name -> ambiguous, fails with DeviceChanged.
    const auto mic3 = fb.make_ep(EndpointDirection::Capture, 3, "Blue Yeti", false);
    fb.capture = {mic1, mic2, mic3};
    CHECK(is_ok(reg.refresh()));
    auto ambig = reg.resolve_persistent_token("v1:wasapi:capture:Blue Yeti",
                                              EndpointDirection::Capture);
    CHECK(!is_ok(ambig));
    CHECK(std::get<AudioError>(ambig).category == AudioErrorCategory::DeviceChanged);
}

} // namespace

int main() {
    test_four_distinguishable_outcomes_and_stale_retention();
    test_duplicate_names_and_generation_stability_on_reorder();
    test_selection_policy_survives_reorder_and_rejects_stale_or_missing();
    test_persistent_tokens_and_ambiguity();
    return test::finish("device_registry");
}
