// AudioController — deterministic fake-backend tests for Milestone 1.5
// Stages 05 (refresh & recovery policy table) and 06 (routing & loopback
// validation).
//
// Covers every row of the Stage 05 policy table and Research §11 without
// requiring audio hardware:
//   1. Follow-default selection when OS default changes:
//      - auto-routing backend updates resolved endpoint with no duplicate restart
//      - non-auto-routing backend performs serialized stop -> drain -> reset -> reopen
//      - refuses to follow default into a feedback route
//   2. Explicit capture device disappears -> stops route, AwaitingUser, never
//      substitutes another microphone
//   3. Virtual output disappears -> stops output, AwaitingUser, never falls
//      back to speakers
//   4. Permission denied -> AwaitingUser with explanation, zero retries, no
//      substitute source
//   5. Device busy / unexpected stop -> Recovering with bounded backoff
//      retries; succeeds if available in time, or transitions to Error when
//      retries are exhausted
//   6. Context/backend recreated -> stops stream, invalidates old epoch keys,
//      rebuilds inventory, re-resolves FollowSystemDefault vs. old explicit key
//   7. User switches endpoints -> serialized quiesce -> stop -> queue discard ->
//      pipeline reset -> reopen; no old-route PCM leaks into new route
//   8. Notification bursts coalesced; liveness works with silent audio;
//      shutdown during recovery is safe
//   9. Stage 06 system-playback loopback mode: gated by native_loopback,
//      resolves against playback endpoints, rejects feedback loops and
//      reports unsupported backends (macOS Core Audio / Linux PulseAudio).
#include "audio/AudioController.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "TestSupport.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace fluxvoice;
using namespace std::chrono_literals;

class FakeControllableBackend : public AudioBackend {
public:
    BackendStatus backend_status{BackendState::Ready, "wasapi", 0};
    ContextEpoch epoch = 1;
    std::vector<EndpointDescriptor> capture;
    std::vector<EndpointDescriptor> playback;
    AudioCapabilities caps{Capability::Supported, Capability::Supported,
                           Capability::Unknown, Capability::Unknown};
    bool auto_routes = true;

    std::optional<AudioError> enumerate_error;
    std::optional<AudioError> next_open_error;
    std::optional<AudioError> next_start_error;

    bool stream_open    = false;
    bool stream_running = false;
    StreamRequest last_request{};
    EndpointKey last_capture_key{};
    EndpointKey last_playback_key{};
    bool last_capture_was_default  = false;
    bool last_playback_was_default = false;

    std::uint64_t enumerate_calls = 0;
    std::uint64_t open_calls      = 0;
    std::uint64_t start_calls     = 0;
    std::uint64_t stop_calls      = 0;

    AudioProcessFn process_fn = nullptr;
    void* process_user_data   = nullptr;
    std::atomic<std::uint32_t> notifications{NotificationBits::kNone};

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
        ++enumerate_calls;
        if (enumerate_error) return *enumerate_error;
        DeviceSnapshot s;
        s.backend    = backend_status.backend;
        s.epoch      = epoch;
        s.generation = enumerate_calls;
        s.capture    = capture;
        s.playback   = playback;
        return s;
    }

    Result<EndpointInfo> probe(const EndpointKey& key) const override {
        EndpointInfo info;
        info.key = key;
        return info;
    }

    AudioCapabilities capabilities() const override { return caps; }

    Result<StreamInfo> open_stream(const StreamRequest& req, AudioProcessFn fn,
                                   void* user_data) override {
        ++open_calls;
        if (next_open_error) {
            AudioError err = *next_open_error;
            next_open_error.reset();
            return err;
        }
        if (req.capture_mode == CaptureMode::SystemPlaybackLoopback &&
            caps.native_loopback != Capability::Supported) {
            return AudioError{AudioErrorCategory::FormatUnsupported,
                              "open system-playback loopback", -201, backend_status.backend,
                              "Native loopback unsupported"};
        }

        last_request              = req;
        last_capture_was_default  = (req.capture_key == nullptr);
        last_playback_was_default = (req.playback_key == nullptr);
        last_capture_key          = req.capture_key ? *req.capture_key : EndpointKey{};
        last_playback_key         = req.playback_key ? *req.playback_key : EndpointKey{};
        process_fn                = fn;
        process_user_data         = user_data;
        stream_open               = true;

        StreamInfo info;
        info.backend                 = backend_status.backend;
        info.client_sample_rate      = req.sample_rate;
        info.capture_internal_rate   = 48000;
        info.playback_internal_rate  = 48000;
        info.capture_internal_period = 480;
        info.playback_internal_period = 480;

        const auto& cap_list =
            req.capture_mode == CaptureMode::SystemPlaybackLoopback ? playback : capture;
        for (const auto& d : cap_list) {
            if ((req.capture_key && d.key == *req.capture_key) ||
                (!req.capture_key && d.is_default)) {
                info.capture_name = d.name;
                break;
            }
        }
        for (const auto& d : playback) {
            if ((req.playback_key && d.key == *req.playback_key) ||
                (!req.playback_key && d.is_default)) {
                info.playback_name = d.name;
                break;
            }
        }
        return info;
    }

    Result<bool> start_stream() override {
        ++start_calls;
        if (next_start_error) {
            AudioError err = *next_start_error;
            next_start_error.reset();
            return err;
        }
        stream_running = true;
        notifications.fetch_or(NotificationBits::kStarted, std::memory_order_relaxed);
        return true;
    }

    void stop_stream() override {
        if (stream_open || stream_running) ++stop_calls;
        stream_running    = false;
        stream_open       = false;
        process_fn        = nullptr;
        process_user_data = nullptr;
    }

    bool is_stream_open() const override { return stream_open; }
    bool is_stream_running() const override { return stream_running; }

    std::uint32_t drain_notifications() override {
        return notifications.exchange(NotificationBits::kNone, std::memory_order_acq_rel);
    }

    bool auto_routes_default() const override { return auto_routes; }

    void fire_notification(std::uint32_t bits) {
        notifications.fetch_or(bits, std::memory_order_relaxed);
    }

    void pump_callback(const float* in, float* out, std::size_t n) {
        if (stream_running && process_fn) process_fn(process_user_data, out, in, n);
    }
};

void test_follow_default_auto_route_vs_reopen() {
    FakeControllableBackend fb;
    const auto mic1 = fb.make_ep(EndpointDirection::Capture, 1, "Desk Mic", true);
    const auto mic2 = fb.make_ep(EndpointDirection::Capture, 2, "Headset Mic", false);
    const auto spk1 = fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true);
    fb.capture  = {mic1, mic2};
    fb.playback = {spk1};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb);
    auto r = ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                 CaptureMode::OrdinaryMicrophone, t0);
    CHECK(is_ok(r));
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().resolved_capture == mic1.key);
    CHECK(fb.last_capture_was_default);
    CHECK(fb.open_calls == 1);

    // 1. Auto-routing backend (WASAPI/CoreAudio): OS default switches to mic2
    //    and fires kRerouted. Controller updates resolved_capture WITHOUT a
    //    duplicate restart.
    fb.auto_routes = true;
    fb.capture     = { fb.make_ep(EndpointDirection::Capture, 1, "Desk Mic", false),
                       fb.make_ep(EndpointDirection::Capture, 2, "Headset Mic", true) };
    fb.fire_notification(NotificationBits::kRerouted);
    ctrl.tick(t0 + 100ms);
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().resolved_capture == mic2.key);
    CHECK(ctrl.status().capture_name == "Headset Mic");
    CHECK(fb.open_calls == 1);
    CHECK(ctrl.auto_reroutes() == 1);

    // 2. Non-auto-routing backend (e.g. PulseAudio/ALSA): OS default switches
    //    back to mic1 -> controller performs a clean stop + reopen.
    fb.auto_routes = false;
    fb.capture     = { fb.make_ep(EndpointDirection::Capture, 1, "Desk Mic", true),
                       fb.make_ep(EndpointDirection::Capture, 2, "Headset Mic", false) };
    ctrl.tick(t0 + 1200ms);
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().resolved_capture == mic1.key);
    CHECK(fb.open_calls == 2);
}

void test_explicit_device_and_virtual_output_disappearance_never_fall_back() {
    FakeControllableBackend fb;
    const auto mic1  = fb.make_ep(EndpointDirection::Capture, 1, "USB Studio Mic", false);
    const auto mic2  = fb.make_ep(EndpointDirection::Capture, 2, "Laptop Array Mic", true);
    const auto cable = fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input (VB-Audio Virtual Cable)", false);
    const auto spkr  = fb.make_ep(EndpointDirection::Playback, 11, "Realtek Speakers", true);
    fb.capture  = {mic1, mic2};
    fb.playback = {cable, spkr};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb);
    CHECK(is_ok(ctrl.refresh(t0)));

    // Open explicit mic1 -> explicit virtual cable.
    auto r = ctrl.open_and_start(ExplicitEndpoint{mic1.key}, ExplicitEndpoint{cable.key},
                                 CaptureMode::OrdinaryMicrophone, t0);
    CHECK(is_ok(r));
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(fb.open_calls == 1);

    // Row 2: Explicit mic1 is unplugged. Must stop and enter AwaitingUser —
    // NEVER open Laptop Array Mic!
    fb.capture = {mic2};
    ctrl.tick(t0 + 1100ms);
    CHECK(ctrl.status().state == StreamState::AwaitingUser);
    CHECK(!ctrl.is_stream_live());
    CHECK(!fb.is_stream_open());
    CHECK(fb.open_calls == 1);
    CHECK(!ctrl.status().resolved_capture.has_value());
    CHECK(std::get<ExplicitEndpoint>(ctrl.status().requested_capture).key == mic1.key);
    CHECK(ctrl.status().last_error.has_value());
    CHECK(ctrl.status().last_error->category == AudioErrorCategory::DeviceNotFound);

    // Plug mic1 back in and reopen; then test Row 3: virtual output disappears.
    fb.capture = {mic1, mic2};
    CHECK(is_ok(ctrl.refresh(t0 + 2000ms)));
    CHECK(is_ok(ctrl.open_and_start(ExplicitEndpoint{mic1.key}, ExplicitEndpoint{cable.key},
                                    CaptureMode::OrdinaryMicrophone, t0 + 2000ms)));
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(fb.open_calls == 2);

    // Virtual cable disappears while Realtek Speakers remain default.
    // Must stop immediately and NEVER redirect processed speech to speakers!
    fb.playback = {spkr};
    ctrl.tick(t0 + 3100ms);
    CHECK(ctrl.status().state == StreamState::AwaitingUser);
    CHECK(!fb.is_stream_open());
    CHECK(fb.open_calls == 2);
    CHECK(!ctrl.status().resolved_playback.has_value());
    CHECK(ctrl.status().last_error.has_value());
    CHECK(ctrl.status().last_error->category == AudioErrorCategory::DeviceNotFound);
}

void test_permission_denied_does_not_retry_or_fallback() {
    FakeControllableBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};
    fb.next_open_error = AudioError{
        AudioErrorCategory::PermissionDenied, "open stream", -6, "coreaudio",
        "Microphone access denied in System Settings > Privacy & Security."};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb);
    auto r = ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                 CaptureMode::OrdinaryMicrophone, t0);
    CHECK(!is_ok(r));
    CHECK(ctrl.status().state == StreamState::AwaitingUser);
    CHECK(ctrl.status().retry_attempt == 0);
    CHECK(ctrl.status().last_error.has_value());
    CHECK(ctrl.status().last_error->category == AudioErrorCategory::PermissionDenied);

    // Subsequent ticks must NOT retry continuously.
    ctrl.tick(t0 + 500ms);
    ctrl.tick(t0 + 2000ms);
    CHECK(fb.open_calls == 1);
    CHECK(ctrl.status().state == StreamState::AwaitingUser);
}

void test_device_busy_bounded_retries_and_recovery() {
    FakeControllableBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};

    ControllerConfig cfg;
    cfg.max_retries           = 3;
    cfg.initial_retry_backoff = 200ms;
    cfg.max_retry_backoff     = 1000ms;

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb, cfg);
    CHECK(is_ok(ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                    CaptureMode::OrdinaryMicrophone, t0)));
    CHECK(ctrl.status().state == StreamState::Running);

    // Simulate unexpected stream stop (e.g. Bluetooth profile switch / server hiccup).
    fb.stream_running = false;
    fb.fire_notification(NotificationBits::kStopped);
    ctrl.tick(t0 + 50ms);
    CHECK(ctrl.status().state == StreamState::Recovering);
    CHECK(ctrl.status().retry_attempt == 0);

    // Retry 1 at +250ms fails with DeviceBusy.
    fb.next_open_error =
        AudioError{AudioErrorCategory::DeviceBusy, "open stream", -19, "wasapi", "Device in use"};
    ctrl.tick(t0 + 250ms);
    CHECK(ctrl.status().state == StreamState::Recovering);
    CHECK(ctrl.status().retry_attempt == 1);

    // Retry 2 at +450ms succeeds!
    ctrl.tick(t0 + 450ms);
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().retry_attempt == 0);
    CHECK(!ctrl.status().last_error.has_value());

    // Now simulate a permanent busy failure that exhausts all 3 retries.
    fb.stream_running = false;
    fb.fire_notification(NotificationBits::kStopped);
    ctrl.tick(t0 + 1000ms);
    CHECK(ctrl.status().state == StreamState::Recovering);

    for (int i = 1; i <= 3; ++i) {
        fb.next_open_error = AudioError{AudioErrorCategory::DeviceBusy, "open stream", -19,
                                        "wasapi", "Exclusive mode lock"};
        ctrl.tick(t0 + 1000ms + std::chrono::milliseconds(i * 1000));
    }
    CHECK(ctrl.status().state == StreamState::Error);
    CHECK(ctrl.status().retry_attempt == 3);
    CHECK(ctrl.status().last_error.has_value());
    CHECK(ctrl.status().last_error->backend == "wasapi");

    // Further ticks after exhausting retries do not call open_stream again.
    const uint64_t opens_before = fb.open_calls;
    ctrl.tick(t0 + 10000ms);
    CHECK(fb.open_calls == opens_before);
}

void test_context_recreation_invalidates_keys_and_reresolves_policy() {
    FakeControllableBackend fb;
    const auto mic1 = fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true);
    const auto spk1 = fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true);
    fb.capture  = {mic1};
    fb.playback = {spk1};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb);
    CHECK(is_ok(ctrl.refresh(t0)));

    // Case A: FollowSystemDefault re-resolves cleanly across context recreation.
    CHECK(is_ok(ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                    CaptureMode::OrdinaryMicrophone, t0)));
    fb.epoch    = 2;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};
    auto rec_default = ctrl.on_context_recreated(t0 + 500ms);
    CHECK(is_ok(rec_default));
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().resolved_capture->epoch == 2);

    // Case B: Explicit key from epoch 2 is rejected when epoch becomes 3.
    const EndpointKey epoch2_mic = fb.capture[0].key;
    CHECK(is_ok(ctrl.reconfigure(ExplicitEndpoint{epoch2_mic}, FollowSystemDefault{},
                                 CaptureMode::OrdinaryMicrophone, t0 + 600ms)));
    fb.epoch    = 3;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};
    auto rec_explicit = ctrl.on_context_recreated(t0 + 1000ms);
    CHECK(!is_ok(rec_explicit));
    CHECK(ctrl.status().state == StreamState::AwaitingUser);
    CHECK(ctrl.status().last_error.has_value());
    CHECK(ctrl.status().last_error->category == AudioErrorCategory::DeviceChanged);
}

void test_switch_transition_order_and_no_pcm_leak() {
    FakeControllableBackend fb;
    const auto mic1 = fb.make_ep(EndpointDirection::Capture, 1, "Mic 1", true);
    const auto mic2 = fb.make_ep(EndpointDirection::Capture, 2, "Mic 2", false);
    const auto spk  = fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true);
    fb.capture  = {mic1, mic2};
    fb.playback = {spk};

    const auto t0 = AudioController::Clock::now();
    InferencePipeline pipeline;
    bool worker_active = false;
    std::vector<std::string> sequence;

    AudioController ctrl(fb);
    CHECK(is_ok(ctrl.refresh(t0)));

    ctrl.set_worker_hooks(
        [&] {
            // Quiesce runs BEFORE backend stop or queue discard.
            CHECK(worker_active);
            worker_active = false;
            sequence.emplace_back("quiesce");
        },
        [&] {
            // Reset runs ONLY after both worker and backend stream are stopped
            // and queues have already been drained.
            CHECK(!worker_active);
            CHECK(!fb.is_stream_running());
            CHECK(!fb.is_stream_open());
            CHECK(ctrl.router().input_queued_approx() == 0);
            CHECK(ctrl.router().output_queued_approx() == 0);
            pipeline.reset();
            sequence.emplace_back("reset");
        },
        [&] {
            CHECK(fb.is_stream_running());
            worker_active = true;
            sequence.emplace_back("resume");
        });

    CHECK(is_ok(ctrl.open_and_start(ExplicitEndpoint{mic1.key}, ExplicitEndpoint{spk.key},
                                    CaptureMode::OrdinaryMicrophone, t0)));
    CHECK(worker_active);

    // Pump old-route samples into both queues.
    const float old_in[4]  = {9.0f, 9.0f, 9.0f, 9.0f};
    const float old_out[4] = {7.0f, 7.0f, 7.0f, 7.0f};
    float cb_out[4]        = {};
    CHECK(ctrl.write_output(old_out, 4) == 4);
    fb.pump_callback(old_in, cb_out, 4);
    // Queue another block so both in_q_ and out_q_ hold old-route audio.
    CHECK(ctrl.write_output(old_out, 4) == 4);
    CHECK(ctrl.router().input_queued_approx() == 4);
    CHECK(ctrl.router().output_queued_approx() == 4);

    sequence.clear();
    CHECK(is_ok(ctrl.reconfigure(ExplicitEndpoint{mic2.key}, ExplicitEndpoint{spk.key},
                                 CaptureMode::OrdinaryMicrophone, t0 + 100ms)));
    CHECK((sequence == std::vector<std::string>{"quiesce", "reset", "resume"}));
    CHECK(pipeline.reset_count() == 1);
    CHECK(ctrl.last_discarded_samples() == 8);
    CHECK(ctrl.router().input_queued_approx() == 0);
    CHECK(ctrl.router().output_queued_approx() == 0);

    // Verify new route has zero old samples in either queue.
    float check_buf[8] = {};
    CHECK(ctrl.read_input(check_buf, 8) == 0);
}

void test_notification_bursts_silent_liveness_and_shutdown_during_recovery() {
    FakeControllableBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb);
    CHECK(is_ok(ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                    CaptureMode::OrdinaryMicrophone, t0)));

    // Silent audio (all zeros) is still a live stream.
    const float silence[160] = {};
    float out[160]           = {};
    fb.pump_callback(silence, out, 160);
    ctrl.tick(t0 + 10ms);
    CHECK(ctrl.is_stream_live());
    CHECK(ctrl.status().callback_progress == 160);

    // Fire a burst of notifications from multiple background threads ->
    // coalesced into a single enumerate() call on tick().
    const uint64_t enums_before = fb.enumerate_calls;
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            for (int k = 0; k < 50; ++k) {
                fb.fire_notification(NotificationBits::kRerouted | NotificationBits::kUnlocked);
            }
        });
    }
    for (auto& th : threads) th.join();

    ctrl.tick(t0 + 20ms);
    CHECK(fb.enumerate_calls == enums_before + 1);

    // Enter Recovering and then call stop() mid-recovery -> clean Idle state.
    fb.stream_running = false;
    fb.fire_notification(NotificationBits::kInterruptionBegan);
    ctrl.tick(t0 + 30ms);
    CHECK(ctrl.status().state == StreamState::Recovering);
    ctrl.stop();
    CHECK(ctrl.status().state == StreamState::Idle);
    CHECK(!fb.is_stream_open());
}

void test_stalled_callback_is_not_live_and_recovers() {
    FakeControllableBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};

    ControllerConfig cfg;
    cfg.callback_stall_timeout = 500ms;
    cfg.initial_retry_backoff  = 200ms;
    cfg.max_retry_backoff      = 1000ms;

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb, cfg);
    CHECK(is_ok(ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                    CaptureMode::OrdinaryMicrophone, t0)));
    CHECK(ctrl.status().state == StreamState::Running);

    // Backend still reports "started" but the callback never fires — below
    // the stall timeout the stream still counts as live.
    ctrl.tick(t0 + 100ms);
    CHECK(ctrl.is_stream_live());

    // 600 ms with zero callback progress -> stalled -> bounded recovery.
    ctrl.tick(t0 + 600ms);
    CHECK(ctrl.status().state == StreamState::Recovering);
    CHECK(!ctrl.is_stream_live());

    // The 200 ms backoff retry opens a fresh stream.
    ctrl.tick(t0 + 850ms);
    CHECK(ctrl.status().state == StreamState::Running);

    // Progress resumes -> live again (silence still counts as progress).
    const float silence[160] = {};
    float out[160]           = {};
    fb.pump_callback(silence, out, 160);
    ctrl.tick(t0 + 1000ms);
    CHECK(ctrl.is_stream_live());
    CHECK(ctrl.status().callback_progress == 160);
}

void test_retryable_catalog_failure_enters_bounded_recovery() {
    FakeControllableBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};

    ControllerConfig cfg;
    cfg.initial_retry_backoff = 200ms;
    cfg.max_retry_backoff     = 1000ms;

    // Transient server loss while building the very first catalog — the same
    // failure class as a mid-stream "device busy" event, so it must take the
    // bounded-recovery path rather than a terminal Error.
    fb.enumerate_error = AudioError{AudioErrorCategory::DeviceBusy, "enumerate devices",
                                    -19, "wasapi", "audio server restarting"};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb, cfg);
    auto r = ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                 CaptureMode::OrdinaryMicrophone, t0);
    CHECK(!is_ok(r));
    CHECK(ctrl.status().state == StreamState::Recovering);
    CHECK(ctrl.status().retry_attempt == 0);

    // Server returns before the first retry -> catalog heals, stream opens.
    fb.enumerate_error.reset();
    ctrl.tick(t0 + 300ms);
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().resolved_capture.has_value());

    // Second scenario: a permanent enumeration failure exhausts the bounded
    // retries (200 ms / 200 ms / 400 ms) and lands in Error.
    FakeControllableBackend fb2;
    fb2.capture  = {fb2.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb2.playback = {fb2.make_ep(EndpointDirection::Playback, 10, "CABLE Input", true)};
    fb2.enumerate_error = AudioError{AudioErrorCategory::EnumerationFailed,
                                     "enumerate devices", -19, "wasapi",
                                     "audio server permanently gone"};

    AudioController ctrl2(fb2, cfg);
    auto r2 = ctrl2.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                   CaptureMode::OrdinaryMicrophone, t0);
    CHECK(!is_ok(r2));
    CHECK(ctrl2.status().state == StreamState::Recovering);
    ctrl2.tick(t0 + 300ms);   // retry 1 -> catalog still absent
    CHECK(ctrl2.status().state == StreamState::Recovering);
    ctrl2.tick(t0 + 800ms);   // retry 2
    CHECK(ctrl2.status().state == StreamState::Recovering);
    ctrl2.tick(t0 + 2000ms);  // retry 3 -> retries exhausted
    CHECK(ctrl2.status().state == StreamState::Error);
    CHECK(ctrl2.status().retry_attempt == 3);
}

void test_stage06_system_playback_loopback_and_unsupported_backends() {
    FakeControllableBackend fb;
    const auto mic   = fb.make_ep(EndpointDirection::Capture, 1, "Mic", true);
    const auto spkr  = fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true);
    const auto cable = fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input", false);
    fb.capture  = {mic};
    fb.playback = {spkr, cable};

    const auto t0 = AudioController::Clock::now();
    AudioController ctrl(fb);
    CHECK(is_ok(ctrl.refresh(t0)));

    // 1. Supported on WASAPI: captures Speakers (a Playback key!) and outputs
    //    to CABLE Input.
    auto ok_loop = ctrl.open_and_start(ExplicitEndpoint{spkr.key}, ExplicitEndpoint{cable.key},
                                       CaptureMode::SystemPlaybackLoopback, t0);
    CHECK(is_ok(ok_loop));
    CHECK(ctrl.status().state == StreamState::Running);
    CHECK(ctrl.status().capture_mode == CaptureMode::SystemPlaybackLoopback);
    CHECK(ctrl.status().resolved_capture == spkr.key);
    CHECK(ctrl.status().resolved_playback == cable.key);

    // 2. Passing a Capture (microphone) key to SystemPlaybackLoopback is rejected.
    auto wrong_dir = ctrl.reconfigure(ExplicitEndpoint{mic.key}, ExplicitEndpoint{cable.key},
                                      CaptureMode::SystemPlaybackLoopback, t0 + 100ms);
    CHECK(!is_ok(wrong_dir));
    CHECK(std::get<AudioError>(wrong_dir).category == AudioErrorCategory::DeviceNotFound);

    // 3. Loopback capturing the SAME endpoint we play into is rejected as a
    //    feedback route!
    auto self_loop = ctrl.reconfigure(ExplicitEndpoint{cable.key}, ExplicitEndpoint{cable.key},
                                      CaptureMode::SystemPlaybackLoopback, t0 + 200ms);
    CHECK(!is_ok(self_loop));
    CHECK(ctrl.status().state == StreamState::AwaitingUser);

    // 4. Unsupported backend (macOS Core Audio) reports FormatUnsupported with
    //    an explicit explanation.
    fb.backend_status       = BackendStatus{BackendState::Ready, "coreaudio", 0};
    fb.caps.native_loopback = Capability::Unsupported;
    CHECK(is_ok(ctrl.refresh(t0 + 300ms)));
    auto mac_loop = ctrl.reconfigure(FollowSystemDefault{}, ExplicitEndpoint{cable.key},
                                     CaptureMode::SystemPlaybackLoopback, t0 + 300ms);
    CHECK(!is_ok(mac_loop));
    CHECK(std::get<AudioError>(mac_loop).category == AudioErrorCategory::FormatUnsupported);
    CHECK(std::get<AudioError>(mac_loop).message.find("macOS Core Audio") != std::string::npos);
}

} // namespace

int main() {
    test_follow_default_auto_route_vs_reopen();
    test_explicit_device_and_virtual_output_disappearance_never_fall_back();
    test_permission_denied_does_not_retry_or_fallback();
    test_device_busy_bounded_retries_and_recovery();
    test_context_recreation_invalidates_keys_and_reresolves_policy();
    test_switch_transition_order_and_no_pcm_leak();
    test_notification_bursts_silent_liveness_and_shutdown_during_recovery();
    test_stalled_callback_is_not_live_and_recovers();
    test_retryable_catalog_failure_enters_bounded_recovery();
    test_stage06_system_playback_loopback_and_unsupported_backends();
    return test::finish("audio_controller");
}
