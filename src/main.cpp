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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

#include "audio/AudioEngine.hpp"

namespace {

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
        "                        thread (default when neither flag is given)\n"
        "  --input  <index>      Capture device index (see --list-devices)\n"
        "  --output <index>      Playback device index (see --list-devices)\n"
        "  --sample-rate <hz>    Sample rate (default 16000)\n"
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
    auto print = [](const char* title, const std::vector<AudioDeviceInfo>& devs) {
        std::cout << title << '\n';
        for (size_t i = 0; i < devs.size(); ++i)
            std::cout << "  [" << i << "] " << devs[i].name
                      << (devs[i].is_default ? "  (default)" : "") << '\n';
    };
    print("Capture devices:", engine.capture_devices());
    print("Playback devices:", engine.playback_devices());
}

} // namespace

int main(int argc, char** argv) {
    bool opt_list = false, opt_passthrough = false;
    int opt_input = -1, opt_output = -1;
    uint32_t sample_rate = AudioEngine::kDefaultSampleRate;

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
        else if (a == "--passthrough")        opt_passthrough = true;
        else if (a == "--queue-loopback")     opt_passthrough = false;
        else if (a == "--input")            { if (!value(0, kMaxIndex, v)) return 1; opt_input = static_cast<int>(v); }
        else if (a == "--output")           { if (!value(0, kMaxIndex, v)) return 1; opt_output = static_cast<int>(v); }
        else if (a == "--sample-rate")      { if (!value(8000, 192000, sample_rate)) return 1; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown option: " << a << '\n'; print_usage(argv[0]); return 1; }
    }

    AudioEngine engine;

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

    if (!engine.start(in_ptr, out_ptr, sample_rate, opt_passthrough)) {
        std::cerr << "Failed to start audio engine\n";
        return 1;
    }

    std::cout << "FluxVoice running @ " << sample_rate << " Hz ("
              << (opt_passthrough ? "callback passthrough" : "SPSC queue loopback")
              << "). Press Enter to stop.\n";

    // Placeholder for the future inference worker thread. For Milestone 1 it
    // simply shovels samples from the input queue to the output queue so the
    // queue path is exercised end-to-end.
    std::atomic<bool> worker_run{true};
    std::thread worker;
    if (!opt_passthrough) {
        worker = std::thread([&] {
            std::vector<float> buf(4096);
            while (worker_run.load(std::memory_order_relaxed)) {
                size_t n = engine.read_input(buf.data(), buf.size());
                if (n > 0) {
                    engine.write_output(buf.data(), n);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
        });
    }

    std::cin.get();

    worker_run.store(false);
    if (worker.joinable()) worker.join();
    engine.stop();

    std::cout << "Stats: input overruns = " << engine.input_overruns()
              << ", output underruns = " << engine.output_underruns() << '\n';
    return 0;
}
