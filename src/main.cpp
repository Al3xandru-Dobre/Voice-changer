// FluxVoice-S2UT — entry point.
//
// Milestone 1 scope: DSP loopback & audio engine (device enumeration, duplex
// miniaudio stream, SPSC ring buffers, click-free pass-through).
// Neural stages are wired in by Milestones 2–3.
//
// NOTE: miniaudio is a single-header library — MINIAUDIO_IMPLEMENTATION must be
// defined in exactly one translation unit. This is it.

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

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

#include "audio/AudioEngine.hpp"
#include "audio/ChunkAssembler.hpp"
#include "audio/RouteGuard.hpp"
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
        "FluxVoice-S2UT — real-time textless voice changer (Milestone 1 build)\n"
        "\n"
        "Usage: " << argv0 << " [options]\n"
        "\n"
        "  --list-devices        Enumerate capture/playback devices and exit\n"
        "  --passthrough         Copy input straight to output in the audio callback\n"
        "                        (lowest-latency loopback smoke test)\n"
        "  --queue-loopback      Route audio through the SPSC ring buffers + worker\n"
        "                        thread (default when no mode flag is given)\n"
        "  --chunked             Queue loopback, but the worker regroups audio into\n"
        "                        80 ms model frames first (adds one hop of latency)\n"
        "  --input  <index>      Capture device index (see --list-devices)\n"
        "  --output <index>      Playback device index (see --list-devices)\n"
        "  --sample-rate <hz>    Sample rate (default 16000)\n"
        "  --duration <seconds>  Stop automatically after this many seconds\n"
        "                        (otherwise stop with Enter or Ctrl+C)\n"
        "  --stats <seconds>     Print a [stats] line (counters, queued audio, RSS)\n"
        "                        this often, for soak tests\n"
        "  -h, --help            This help\n";
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

void list_devices(AudioEngine& engine) {
    std::cout << "Backend: " << engine.backend_name() << '\n';
    auto print = [](const char* title, const std::vector<AudioDeviceInfo>& devs) {
        std::cout << title << '\n';
        for (size_t i = 0; i < devs.size(); ++i)
            std::cout << "  [" << i << "] " << devs[i].name
                      << (devs[i].is_default ? "  (default)" : "") << '\n';
    };
    print("Capture devices:", engine.capture_devices());
    print("Playback devices:", engine.playback_devices());
}

void print_stream_info(const StreamInfo& si) {
    std::cout << "Backend : " << si.backend << '\n'
              << "Capture : " << si.capture_name
              << "  (backend " << si.capture_internal_rate << " Hz, period "
              << si.capture_internal_period << " frames)\n"
              << "Playback: " << si.playback_name
              << "  (backend " << si.playback_internal_rate << " Hz, period "
              << si.playback_internal_period << " frames)\n"
              << "Client  : " << si.client_sample_rate << " Hz mono f32, requested period "
              << AudioEngine::kDefaultPeriodMs << " ms\n";
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

Snapshot snapshot(const AudioEngine& engine, const std::atomic<uint64_t>& drops) {
    Snapshot s{};
    s.frames    = engine.frames_processed();
    s.overruns  = engine.input_overruns();
    s.underruns = engine.output_underruns();
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
    int opt_input = -1, opt_output = -1;
    uint32_t sample_rate = AudioEngine::kDefaultSampleRate;
    uint32_t duration_s = 0;  // 0 = run until Enter / Ctrl+C
    uint32_t stats_s = 0;     // 0 = no periodic [stats] lines

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        // Consume the value of option `a` into `out`; false (with a message)
        // on a missing or malformed value.
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
        constexpr uint32_t kMaxIndex = std::numeric_limits<int>::max();
        uint32_t v = 0;

        if (a == "--list-devices")            opt_list = true;
        else if (a == "--passthrough")        mode = Mode::Passthrough;
        else if (a == "--queue-loopback")     mode = Mode::Queue;
        else if (a == "--chunked")            mode = Mode::Chunked;
        else if (a == "--input")            { if (!value(0, kMaxIndex, v)) return 1; opt_input = static_cast<int>(v); }
        else if (a == "--output")           { if (!value(0, kMaxIndex, v)) return 1; opt_output = static_cast<int>(v); }
        else if (a == "--sample-rate")      { if (!value(8000, 192000, sample_rate)) return 1; }
        else if (a == "--duration")         { if (!value(1, 7 * 24 * 3600, duration_s)) return 1; }
        else if (a == "--stats")            { if (!value(1, 24 * 3600, stats_s)) return 1; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown option: " << a << '\n'; print_usage(argv[0]); return 1; }
    }

    AudioEngine engine;
    if (!engine.context_ok()) {
        std::cerr << "Failed to initialise an audio backend\n";
        return 1;
    }

    if (opt_list) { list_devices(engine); return 0; }

    // Resolve device indices → ma_device_id. nullptr = OS default device.
    ma_device_id in_id{}, out_id{};
    const ma_device_id* in_ptr = nullptr;
    const ma_device_id* out_ptr = nullptr;
    if (opt_input >= 0) {
        auto devs = engine.capture_devices();
        if (opt_input >= static_cast<int>(devs.size())) { std::cerr << "Bad --input index\n"; return 1; }
        in_id = devs[opt_input].id; in_ptr = &in_id;
    }
    if (opt_output >= 0) {
        auto devs = engine.playback_devices();
        if (opt_output >= static_cast<int>(devs.size())) { std::cerr << "Bad --output index\n"; return 1; }
        out_id = devs[opt_output].id; out_ptr = &out_id;
    }

    if (!engine.open(in_ptr, out_ptr, sample_rate, mode == Mode::Passthrough)) {
        std::cerr << "Failed to open audio devices\n";
        return 1;
    }

    // Checked on the endpoints actually opened, so an OS-default input that
    // happens to be our own virtual cable is caught too.
    const StreamInfo& si = engine.stream_info();
    if (is_feedback_route(si.capture_name, si.playback_name)) {
        std::cerr << "Refusing to start: capture device \"" << si.capture_name
                  << "\" records playback device \"" << si.playback_name << "\",\n"
                  << "so FluxVoice would hear its own output (feedback loop).\n"
                  << "Pick the microphone with --input (see --list-devices).\n";
        return 1;
    }

    if (!engine.start()) {
        std::cerr << "Failed to start audio engine\n";
        return 1;
    }

    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);

    print_stream_info(si);
    std::cout << "FluxVoice running (" << mode_name(mode) << "). ";
    if (duration_s > 0) std::cout << "Stopping after " << duration_s << " s; Ctrl+C to stop early.\n";
    else                std::cout << "Press Enter or Ctrl+C to stop.\n";
    std::cout.flush();

    // Placeholder for the future inference worker thread. For Milestone 1 it
    // shovels samples from the input queue to the output queue so the queue
    // path is exercised end-to-end. In --chunked mode it first regroups them
    // into the 80 ms frames the M2 encoder will consume.
    std::atomic<bool> worker_run{true};
    std::atomic<uint64_t> output_drops{0};
    std::thread worker;
    if (mode != Mode::Passthrough) {
        worker = std::thread([&] {
            using fluxvoice::PipelineTiming;
            std::vector<float> buf(4096);
            ChunkAssembler chunks(PipelineTiming::kChunkSamples, PipelineTiming::kContextSamples);
            auto emit = [&](const float* p, size_t n) {
                output_drops.fetch_add(n - engine.write_output(p, n), std::memory_order_relaxed);
            };
            while (worker_run.load(std::memory_order_relaxed)) {
                size_t n = engine.read_input(buf.data(), buf.size());
                if (n == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    continue;
                }
                if (mode == Mode::Queue) { emit(buf.data(), n); continue; }

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
    }

    // Enter on stdin stops the stream — unless --duration was given, so that
    // scripted runs with a closed/redirected stdin keep running. Detached:
    // a thread blocked in a read can't be joined; process exit reclaims it.
    if (duration_s == 0) {
        std::thread([] { std::cin.get(); g_stop.store(true); }).detach();
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::seconds(duration_s);
    auto next_stats = t0 + std::chrono::seconds(stats_s);
    while (!g_stop.load()) {
        const auto now = std::chrono::steady_clock::now();
        if (duration_s > 0 && now >= deadline) break;
        if (stats_s > 0 && now >= next_stats) {
            const double t = std::chrono::duration<double>(now - t0).count();
            std::cout << stats_line(t, snapshot(engine, output_drops)) << std::endl;
            next_stats += std::chrono::seconds(stats_s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Before teardown: callbacks that run after the worker stops would count
    // as underruns that are not the stream's fault.
    const Snapshot end = snapshot(engine, output_drops);

    worker_run.store(false);
    if (worker.joinable()) worker.join();
    engine.stop();

    if (stats_s > 0) std::cout << stats_line(elapsed, end) << '\n';
    std::cout << "Stopped after " << elapsed << " s.\n"
              << "Stats: frames processed = " << end.frames
              << ", input overruns = " << end.overruns
              << ", output underruns = " << end.underruns
              << ", output drops = " << end.drops << '\n'
              << "Queued audio: " << end.queued << " samples ("
              << 1000.0 * static_cast<double>(end.queued) / static_cast<double>(engine.sample_rate())
              << " ms); RSS " << end.rss / 1024 << " KB\n";
    return 0;
}
