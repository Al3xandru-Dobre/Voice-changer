// FluxVoice-S2UT — entry point.
//
// Milestone 1 / 1.5 scope: DSP loopback, cross-platform device registry,
// selection policy, refresh/recovery controller, and optional system-playback
// loopback capture.
// Neural stages are wired in by Milestones 2–3.
//
// NOTE: miniaudio is a single-header library — MINIAUDIO_IMPLEMENTATION is
// defined in exactly one translation unit: src/audio/MiniaudioBackend.cpp.
// Application code speaks only the AudioTypes vocabulary (M1.5 / Stage 02);
// device listing, selection, refresh and recovery go through
// AudioDeviceRegistry and AudioController (stages 03–06).

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "audio/AudioController.hpp"
#include "audio/AudioDeviceRegistry.hpp"
#include "audio/AudioEngine.hpp"
#include "audio/AudioTypes.hpp"
#include "audio/ChunkAssembler.hpp"
#include "audio/MiniaudioBackend.hpp"
#include "audio/RouteGuard.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "util/ProcessMemory.hpp"

namespace {

enum class Mode { Passthrough, Queue, Chunked };

// Set by SIGINT/SIGTERM, Enter on stdin, or --duration expiry. A lock-free
// atomic is async-signal-safe to store to from a signal handler.
std::atomic<bool> g_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free);

extern "C" void on_stop_signal(int sig) {
    g_stop.store(true);
    std::signal(sig, SIG_DFL);  // a second Ctrl+C force-quits if shutdown hangs
}

void print_usage(const char* argv0) {
    std::cout <<
        "FluxVoice-S2UT — real-time textless voice changer (Milestone 1 / 1.5 build)\n"
        "\n"
        "Usage: " << argv0 << " [options]\n"
        "\n"
        "  --list-devices                Enumerate capture/playback devices and exit\n"
        "  --passthrough                 Copy input straight to output in the audio callback\n"
        "                                (lowest-latency loopback smoke test)\n"
        "  --queue-loopback              Route audio through the SPSC ring buffers + worker\n"
        "                                thread (default when no mode flag is given)\n"
        "  --chunked                     Queue loopback, but the worker regroups audio into\n"
        "                                80 ms model frames first (adds one hop of latency)\n"
        "  --input  <index|default>      Capture device: index from --list-devices, or\n"
        "                                \"default\" to follow the OS default endpoint\n"
        "  --output <index|default>      Playback device: index or \"default\" (same rules)\n"
        "  --loopback-input <index|default>\n"
        "                                System-playback capture: playback device index or\n"
        "                                \"default\" (WASAPI native loopback; mutually\n"
        "                                exclusive with --input)\n"
        "  --sample-rate <hz>            Sample rate (default 16000)\n"
        "  --duration <seconds>          Stop automatically after this many seconds\n"
        "                                (otherwise stop with Enter or Ctrl+C)\n"
        "  --stats <seconds>             Print a [stats] line (counters, queued audio, RSS)\n"
        "                                this often, for soak tests\n"
        "  -h, --help                    This help\n";
}

/// Parse a whole decimal string into an unsigned integer within [lo, hi].
/// Rejects empty input, signs, trailing junk and out-of-range values.
bool parse_uint(std::string_view s, uint32_t lo, uint32_t hi, uint32_t& out) {
    uint32_t v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || v < lo || v > hi)
        return false;
    out = v;
    return true;
}

void print_error(const fluxvoice::AudioError& err, const char* what) {
    std::cerr << what << ": " << fluxvoice::to_string(err.category)
              << " (backend '" << err.backend << "', result " << err.native_result
              << "): " << err.message << '\n';
}

/// --list-devices (M1.5 stages 03 & 06): backend name, snapshot generation,
/// native loopback capability, and the owned catalog. Null backend is
/// reported as offline mode, not as an empty working device list.
int list_devices(fluxvoice::AudioDeviceRegistry& registry) {
    const fluxvoice::BackendStatus st = registry.status();
    if (st.state == fluxvoice::BackendState::NullOnly) {
        std::cout << "Backend: " << st.backend
                  << " — offline/test mode: no audio hardware is available through this "
                     "backend.\n";
        return 0;
    }

    auto snap = registry.refresh();
    if (!fluxvoice::is_ok(snap)) {
        print_error(std::get<fluxvoice::AudioError>(snap), "Failed to list devices");
        return 1;
    }
    const fluxvoice::DeviceSnapshot& s = std::get<fluxvoice::DeviceSnapshot>(snap);
    const fluxvoice::AudioCapabilities caps = registry.backend_capabilities();
    const char* loopback_cap =
        caps.native_loopback == fluxvoice::Capability::Supported ? "supported"
        : caps.native_loopback == fluxvoice::Capability::Unsupported ? "unsupported"
                                                                     : "unknown";
    std::cout << "Backend: " << s.backend << " (snapshot generation " << s.generation
              << (registry.stale() ? ", stale" : "")
              << ", native loopback: " << loopback_cap << ")\n";
    const auto print = [](const char* title,
                          const std::vector<fluxvoice::EndpointDescriptor>& devs) {
        std::cout << title << '\n';
        for (size_t i = 0; i < devs.size(); ++i)
            std::cout << "  [" << i << "] " << devs[i].name
                      << (devs[i].is_default ? "  (default)" : "") << '\n';
    };
    print("Capture devices:", s.capture);
    print("Playback devices:", s.playback);
    return 0;
}

void print_stream_info(const StreamInfo& si, const fluxvoice::StreamStatus& status) {
    std::cout << "Backend : " << si.backend << '\n'
              << "Capture : " << si.capture_name
              << "  (backend " << si.capture_internal_rate << " Hz, period "
              << si.capture_internal_period << " frames)\n"
              << "Playback: " << si.playback_name
              << "  (backend " << si.playback_internal_rate << " Hz, period "
              << si.playback_internal_period << " frames)\n"
              << "Client  : " << si.client_sample_rate << " Hz mono f32, requested period "
              << AudioEngine::kDefaultPeriodMs << " ms\n";
    // Requested policy vs. the endpoint actually opened (M1.5 stages 04–06).
    const auto policy = [](const fluxvoice::DeviceSelection& sel) {
        return std::holds_alternative<fluxvoice::FollowSystemDefault>(sel) ? "default"
                                                                           : "explicit";
    };
    std::cout << "Select  : capture=" << policy(status.requested_capture) << " ("
              << fluxvoice::to_string(status.capture_mode) << ")"
              << ", playback=" << policy(status.requested_playback)
              << " (resolved against snapshot generation " << status.selection_generation << ")\n";
}

/// Glitch counters and derived values at one instant (control thread only).
struct Snapshot {
    uint64_t frames, overruns, underruns, drops;
    // Samples held between capture and playback (queues + worker). Every
    // callback enqueues and dequeues the same count, so this is exact by
    // conservation: underruns - overruns - drops.
    int64_t queued;
    size_t  rss;
};

Snapshot snapshot(const DuplexRouter& router, const std::atomic<uint64_t>& drops) {
    Snapshot s{};
    s.frames    = router.frames_processed();
    s.overruns  = router.input_overruns();
    s.underruns = router.output_underruns();
    s.drops     = drops.load(std::memory_order_relaxed);
    s.queued    = static_cast<int64_t>(s.underruns) - static_cast<int64_t>(s.overruns) -
                  static_cast<int64_t>(s.drops);
    s.rss       = fluxvoice::resident_bytes();
    return s;
}

/// One machine-readable line; tools/analyze_soak.py parses these.
std::string stats_line(double t, const Snapshot& s) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(1);
    o << "[stats] t=" << t << " frames=" << s.frames << " overruns=" << s.overruns
      << " underruns=" << s.underruns << " drops=" << s.drops << " queued=" << s.queued
      << " rss_kb=" << s.rss / 1024;
    return o.str();
}

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Passthrough: return "callback passthrough";
        case Mode::Queue:       return "SPSC queue loopback";
        case Mode::Chunked:     return "SPSC queue loopback, 80 ms chunked";
    }
    return "?";
}

} // namespace

int main(int argc, char** argv) {
    bool opt_list = false;
    Mode mode = Mode::Queue;
    int opt_input = -1, opt_output = -1, opt_loopback = -1;
    bool has_input_arg = false, has_loopback_arg = false;
    uint32_t sample_rate = AudioEngine::kDefaultSampleRate;
    uint32_t duration_s = 0;  // 0 = run until Enter / Ctrl+C
    uint32_t stats_s = 0;     // 0 = no periodic [stats] lines

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto value = [&](uint32_t lo, uint32_t hi, uint32_t& out) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << a << '\n';
                return false;
            }
            if (!parse_uint(argv[++i], lo, hi, out)) {
                std::cerr << "Invalid value for " << a << ": '" << argv[i]
                          << "' (expected an integer in [" << lo << ", " << hi << "])\n";
                return false;
            }
            return true;
        };
        // --input/--output/--loopback-input take a --list-devices index or the
        // literal "default" (explicit follow-the-OS-default, M1.5 stages 04–06).
        auto endpoint = [&](int& opt) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << a << '\n';
                return false;
            }
            const std::string_view val = argv[++i];
            if (val == "default") {
                opt = -1;
                return true;
            }
            uint32_t idx = 0;
            constexpr uint32_t kMaxIndex = std::numeric_limits<int>::max();
            if (!parse_uint(val, 0, kMaxIndex, idx)) {
                std::cerr << "Invalid value for " << a << ": '" << val
                          << "' (expected a device index or \"default\")\n";
                return false;
            }
            opt = static_cast<int>(idx);
            return true;
        };

        if (a == "--list-devices")            opt_list = true;
        else if (a == "--passthrough")        mode = Mode::Passthrough;
        else if (a == "--queue-loopback")     mode = Mode::Queue;
        else if (a == "--chunked")            mode = Mode::Chunked;
        else if (a == "--input")            { if (!endpoint(opt_input)) return 1; has_input_arg = true; }
        else if (a == "--output")           { if (!endpoint(opt_output)) return 1; }
        else if (a == "--loopback-input")   { if (!endpoint(opt_loopback)) return 1; has_loopback_arg = true; }
        else if (a == "--sample-rate")      { if (!value(8000, 192000, sample_rate)) return 1; }
        else if (a == "--duration")         { if (!value(1, 7 * 24 * 3600, duration_s)) return 1; }
        else if (a == "--stats")            { if (!value(1, 24 * 3600, stats_s)) return 1; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown option: " << a << '\n'; print_usage(argv[0]); return 1; }
    }

    if (has_input_arg && has_loopback_arg) {
        std::cerr << "--input and --loopback-input are mutually exclusive\n";
        return 1;
    }

    MiniaudioBackend backend;
    fluxvoice::ControllerConfig ctrl_cfg;
    ctrl_cfg.sample_rate = sample_rate;
    ctrl_cfg.period_ms   = AudioEngine::kDefaultPeriodMs;
    ctrl_cfg.passthrough = (mode == Mode::Passthrough && !has_loopback_arg);

    ChunkAssembler chunks(fluxvoice::PipelineTiming::kChunkSamples,
                          fluxvoice::PipelineTiming::kContextSamples);
    fluxvoice::InferencePipeline pipeline;

    std::atomic<bool> worker_run{false};
    std::atomic<uint64_t> output_drops{0};
    std::thread worker;

    fluxvoice::AudioController controller(backend, ctrl_cfg);
    fluxvoice::AudioDeviceRegistry& registry = controller.registry();

    const fluxvoice::BackendStatus backend_status = registry.status();
    if (backend_status.state == fluxvoice::BackendState::Unavailable) {
        std::cerr << "Failed to initialise an audio backend (result "
                  << backend_status.native_result << ")\n";
        return 1;
    }

    if (opt_list) return list_devices(registry);

    // Indices refer to the catalog enumerated by THIS run; keys are identity,
    // so a later reorder can never redirect an explicit selection (stage 04).
    auto snap = controller.refresh();
    if (!fluxvoice::is_ok(snap)) {
        print_error(std::get<fluxvoice::AudioError>(snap), "Cannot select audio devices");
        return 1;
    }
    const fluxvoice::DeviceSnapshot& catalog = controller.snapshot();

    const fluxvoice::CaptureMode capture_mode =
        has_loopback_arg ? fluxvoice::CaptureMode::SystemPlaybackLoopback
                         : fluxvoice::CaptureMode::OrdinaryMicrophone;

    fluxvoice::DeviceSelection in_sel{fluxvoice::FollowSystemDefault{}};
    fluxvoice::DeviceSelection out_sel{fluxvoice::FollowSystemDefault{}};
    if (has_loopback_arg && opt_loopback >= 0) {
        const auto& devs = catalog.playback;
        if (opt_loopback >= static_cast<int>(devs.size())) {
            std::cerr << "Bad --loopback-input index " << opt_loopback
                      << " (see Playback devices in --list-devices)\n";
            return 1;
        }
        in_sel = fluxvoice::ExplicitEndpoint{devs[opt_loopback].key, catalog.generation};
    } else if (opt_input >= 0) {
        const auto& devs = catalog.capture;
        if (opt_input >= static_cast<int>(devs.size())) {
            std::cerr << "Bad --input index " << opt_input
                      << " (see --list-devices)\n";
            return 1;
        }
        in_sel = fluxvoice::ExplicitEndpoint{devs[opt_input].key, catalog.generation};
    }
    if (opt_output >= 0) {
        const auto& devs = catalog.playback;
        if (opt_output >= static_cast<int>(devs.size())) {
            std::cerr << "Bad --output index " << opt_output
                      << " (see --list-devices)\n";
            return 1;
        }
        out_sel = fluxvoice::ExplicitEndpoint{devs[opt_output].key, catalog.generation};
    }

    const bool use_worker = (mode != Mode::Passthrough) || has_loopback_arg;

    auto stop_worker = [&] {
        worker_run.store(false, std::memory_order_relaxed);
        if (worker.joinable()) worker.join();
    };
    auto reset_dsp = [&] {
        chunks.reset();
        pipeline.reset();
    };
    auto start_worker = [&] {
        if (!use_worker || worker.joinable()) return;
        worker_run.store(true, std::memory_order_relaxed);
        worker = std::thread([&] {
            std::vector<float> buf(4096);
            auto emit = [&](const float* p, size_t n) {
                output_drops.fetch_add(n - controller.write_output(p, n),
                                       std::memory_order_relaxed);
            };
            while (worker_run.load(std::memory_order_relaxed)) {
                size_t n = controller.read_input(buf.data(), buf.size());
                if (n == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    continue;
                }
                if (mode != Mode::Chunked) {
                    emit(buf.data(), n);
                    continue;
                }

                const float* p = buf.data();
                while (n > 0) {
                    const size_t used = chunks.push(p, n);
                    p += used;
                    n -= used;
                    if (chunks.ready()) {
                        emit(chunks.hop_samples(), chunks.hop());
                        chunks.advance();
                    }
                }
            }
        });
    };

    controller.set_worker_hooks(stop_worker, reset_dsp, start_worker);

    auto start_res = controller.open_and_start(in_sel, out_sel, capture_mode);
    if (!fluxvoice::is_ok(start_res)) {
        print_error(std::get<fluxvoice::AudioError>(start_res), "Failed to start audio stream");
        return 1;
    }

    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);

    print_stream_info(controller.stream_info(), controller.status());
    std::cout << "FluxVoice running (" << mode_name(mode) << "). ";
    if (duration_s > 0) std::cout << "Stopping after " << duration_s << " s; Ctrl+C to stop early.\n";
    else                std::cout << "Press Enter or Ctrl+C to stop.\n";
    std::cout.flush();

    if (duration_s == 0) {
        std::thread([] { std::cin.get(); g_stop.store(true); }).detach();
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::seconds(duration_s);
    auto next_stats = t0 + std::chrono::seconds(stats_s);
    int exit_code = 0;
    while (!g_stop.load()) {
        const auto now = std::chrono::steady_clock::now();
        controller.tick(now);
        const fluxvoice::StreamState st = controller.status().state;
        if (st == fluxvoice::StreamState::AwaitingUser || st == fluxvoice::StreamState::Error) {
            if (controller.status().last_error) {
                print_error(*controller.status().last_error, "Stream stopped");
            }
            exit_code = 1;
            break;
        }
        if (duration_s > 0 && now >= deadline) break;
        if (stats_s > 0 && now >= next_stats) {
            const double t = std::chrono::duration<double>(now - t0).count();
            std::cout << stats_line(t, snapshot(controller.router(), output_drops)) << std::endl;
            next_stats += std::chrono::seconds(stats_s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Before teardown: callbacks that run after the worker stops would count
    // as underruns that are not the stream's fault (BEST_PRACTICES §3.5).
    const Snapshot end = snapshot(controller.router(), output_drops);
    const uint32_t active_rate =
        controller.stream_info().client_sample_rate > 0
            ? controller.stream_info().client_sample_rate
            : sample_rate;

    controller.stop();

    if (stats_s > 0) std::cout << stats_line(elapsed, end) << '\n';
    std::cout << "Stopped after " << elapsed << " s.\n"
              << "Stats: frames processed = " << end.frames
              << ", input overruns = " << end.overruns
              << ", output underruns = " << end.underruns
              << ", output drops = " << end.drops << '\n'
              << "Queued audio: " << end.queued << " samples ("
              << 1000.0 * static_cast<double>(end.queued) / static_cast<double>(active_rate)
              << " ms); RSS " << end.rss / 1024 << " KB\n";
    return exit_code;
}
