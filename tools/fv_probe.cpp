// fv_probe — the "second application" for Milestone 1 / Stages 05–06.
//
// fluxvoice routes mic → virtual cable; fv_probe sits at the far end of such
// routes and measures them. It is a separate process with its own audio
// stream, like any app receiving FluxVoice's output.
//
//   fv_probe --list-devices
//   fv_probe record  --input I [--seconds S] [--rate R] out.wav
//   fv_probe tone    --output O --input I [--seconds S] [--freq F] [--rate R] out.wav
//   fv_probe latency --output O --input I [--count N] [--interval-ms M] [--rate R]
//                    [--csv out.csv]
//
// Devices are selected by index (--input/--output) or by name substring
// (--input-name/--output-name, case-insensitive, must match exactly one
// device). Scripts should prefer names: enumeration indices shift whenever
// audio drivers are installed or removed (BUG-001).
//
// record   Record capture device I to a 16-bit mono WAV, e.g. "CABLE Output":
//          does FluxVoice's output reach other apps?
// tone     Play a 440 Hz sine into O while recording I, e.g. O = the cable
//          fluxvoice listens on, I = the cable it plays into. Check the WAV
//          with tools/analyze_loopback.py (dropouts, clicks, pitch).
// latency  Play a 20 ms chirp every M ms into O and find each one in I. Both
//          ends are one duplex stream, so they share a sample clock. Reports
//          the O → … → I delay (median / p95 / range), missed markers and the
//          delay's drift over time. The figure includes fv_probe's own device
//          buffers: measure a baseline with O's cable read straight back, and
//          subtract it (Stage 06 doc).
//
// Device indices match `fluxvoice --list-devices` on the same backend.

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <readerwriterqueue.h>

#include "LatencyProbe.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

constexpr double kTwoPi = 6.283185307179586;

std::atomic<bool> g_stop{false};

extern "C" void on_stop_signal(int sig) {
    g_stop.store(true);
    std::signal(sig, SIG_DFL);
}

enum class Command { None, Record, Tone, Latency };

struct Options {
    Command     cmd      = Command::None;
    bool        list     = false;
    bool        loopback = false;  // record playback endpoint via native loopback (M1.5 stage 06)
    int         input = -1, output = -1;
    std::string input_name, output_name;  // substring match (see header note)
    uint32_t    rate = 48000, seconds = 10, freq = 440, count = 60, interval_ms = 500;
    std::string wav;  // record / tone output
    std::string csv;  // latency: one row per marker
};

void print_usage(const char* argv0) {
    std::cout <<
        "fv_probe — measure audio routes from the far end (FluxVoice M1 stages 05-06)\n"
        "\n"
        "Usage:\n"
        "  " << argv0 << " --list-devices\n"
        "  " << argv0 << " record  [--loopback] --input I [--seconds S] [--rate R] out.wav\n"
        "  " << argv0 << " tone    --output O --input I [--seconds S] [--freq F] [--rate R] out.wav\n"
        "  " << argv0 << " latency --output O --input I [--count N] [--interval-ms M] [--rate R]\n"
        "                   [--csv out.csv]\n"
        "\n"
        "  record    record capture device I to a 16-bit WAV\n"
        "  tone      play a sine into O while recording I to a WAV\n"
        "  latency   play a chirp every M ms into O, find it in I, report the delay\n"
        "\n"
        "  --input <index>       capture device (see --list-devices; or playback device\n"
        "                        when --loopback is used)\n"
        "  --output <index>      playback device (see --list-devices)\n"
        "  --input-name <text>   capture device by name substring (alternative to\n"
        "                        --input; error if the text matches no device or\n"
        "                        more than one)\n"
        "  --output-name <text>  playback device by name substring\n"
        "  --loopback            record a playback endpoint via native loopback\n"
        "                        (Windows WASAPI; valid with 'record' only)\n"
        "  --seconds <s>         record / tone length (default 10)\n"
        "  --freq <hz>           tone frequency (default 440)\n"
        "  --count <n>           latency markers (default 60)\n"
        "  --interval-ms <ms>    time between markers, and the largest delay that\n"
        "                        can be measured, plus 20 ms (default 500)\n"
        "  --rate <hz>           fv_probe's own sample rate (default 48000)\n"
        "  --csv <path>          latency: write marker,t_s,found,latency_ms,score rows\n"
        "  -h, --help            this help\n";
}

/// Parse a whole decimal string into an unsigned integer within [lo, hi].
bool parse_uint(std::string_view s, uint32_t lo, uint32_t hi, uint32_t& out) {
    uint32_t v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || v < lo || v > hi)
        return false;
    out = v;
    return true;
}

std::string to_lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

/// Resolve a device selection: an explicit index, or the unique device whose
/// name contains `name` (case-insensitive). Returns -1 with a message on
/// failure.
int resolve_device(const ma_device_info* devs, ma_uint32 n, int index,
                   const std::string& name, const char* what) {
    if (!name.empty()) {
        const std::string needle = to_lower(name);
        int hit = -1;
        for (ma_uint32 i = 0; i < n; ++i) {
            if (to_lower(devs[i].name).find(needle) == std::string::npos) continue;
            if (hit >= 0) {
                std::cerr << "Ambiguous " << what << " name \"" << name << "\": matches ["
                          << hit << "] " << devs[hit].name << " and [" << i << "] "
                          << devs[i].name << '\n';
                return -1;
            }
            hit = static_cast<int>(i);
        }
        if (hit < 0)
            std::cerr << "No " << what << " device name contains \"" << name
                      << "\" (see --list-devices)\n";
        else
            std::cout << "Matched " << what << " device [" << hit << "] "
                      << devs[hit].name << '\n';
        return hit;
    }
    if (index >= static_cast<int>(n)) {
        std::cerr << "Bad index for " << what << " (see --list-devices)\n";
        return -1;
    }
    return index;
}

/// Mono 16-bit PCM WAV, written as samples arrive; sizes patched by close().
class WavWriter {
public:
    bool open(const std::string& path, uint32_t rate) {
        f_.open(path, std::ios::binary);
        rate_ = rate;
        header(0);
        return static_cast<bool>(f_);
    }

    void write(const float* x, size_t n) {
        for (size_t i = 0; i < n; ++i)
            u16(static_cast<uint16_t>(static_cast<int16_t>(
                std::lround(std::clamp(x[i], -1.0f, 1.0f) * 32767.0f))));
        samples_ += n;
    }

    bool close() {
        f_.seekp(0);
        header(samples_ * 2);
        f_.close();
        return !f_.fail();
    }

private:
    void u16(uint32_t v) { f_.put(static_cast<char>(v)); f_.put(static_cast<char>(v >> 8)); }
    void u32(uint32_t v) { u16(v & 0xFFFF); u16(v >> 16); }
    void header(uint64_t data_bytes) {
        const auto d = static_cast<uint32_t>(data_bytes);
        f_.write("RIFF", 4); u32(36 + d); f_.write("WAVE", 4);
        f_.write("fmt ", 4); u32(16); u16(1 /* PCM */); u16(1 /* mono */);
        u32(rate_); u32(rate_ * 2); u16(2 /* block align */); u16(16 /* bits */);
        f_.write("data", 4); u32(d);
    }

    std::ofstream f_;
    uint32_t rate_ = 0;
    uint64_t samples_ = 0;
};

// The capture queue is cache-line aligned on purpose; MSVC's C4324 reports
// the padding that buys.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

/// Everything the audio callback touches; set up before the device starts.
struct Stream {
    Command cmd = Command::Record;
    double  phase = 0.0, phase_step = 0.0;  // tone
    std::vector<float>    marker;            // latency
    probe::MarkerSchedule sched;
    uint64_t pos = 0;                        // frames so far (callback thread only)

    // Capture → main thread. ~5 s at 48 kHz; the main thread drains it every
    // few ms. Index i of the queue is frame i of the stream (until overflow).
    moodycamel::ReaderWriterQueue<float> captured{1 << 18};
    std::atomic<uint64_t> overflows{0};
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

/// Real-time callback: render the probe signal, queue the capture. No
/// allocation, locks or I/O.
void data_callback(ma_device* dev, void* out, const void* in, ma_uint32 n) {
    auto* s = static_cast<Stream*>(dev->pUserData);
    if (out) {
        auto* o = static_cast<float*>(out);
        for (ma_uint32 i = 0; i < n; ++i) {
            if (s->cmd == Command::Tone) {
                o[i] = static_cast<float>(0.25 * std::sin(s->phase));
                s->phase = std::fmod(s->phase + s->phase_step, kTwoPi);
            } else {
                o[i] = probe::marker_sample(s->marker, s->sched, s->pos + i);
            }
        }
    }
    if (in) {
        const auto* x = static_cast<const float*>(in);
        for (ma_uint32 i = 0; i < n; ++i)
            if (!s->captured.try_enqueue(x[i])) {
                s->overflows.fetch_add(n - i, std::memory_order_relaxed);
                break;
            }
    }
    s->pos += n;
}

double ms(double samples, uint32_t rate) { return 1000.0 * samples / rate; }

/// Print the latency result; true when every marker was found.
bool report_latency(const probe::MarkerDetector& det, const Stream& s, const Options& opt) {
    const std::vector<probe::Detection> dets(det.detections().begin(),
                                             det.detections().begin() + static_cast<std::ptrdiff_t>(det.analysed()));
    const probe::Summary sum = probe::summarize(dets, s.sched, opt.rate);
    const double minutes = static_cast<double>(s.sched.interval) * static_cast<double>(dets.size()) /
                           opt.rate / 60.0;
    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "Markers : " << dets.size() << " of " << s.sched.count << " analysed, "
              << sum.found << " found, " << sum.missed << " missed\n";
    if (sum.found > 0) {
        std::cout << "Latency : median " << ms(sum.median, opt.rate) << " ms, p95 "
                  << ms(sum.p95, opt.rate) << " ms, min " << ms(sum.min, opt.rate) << " ms, max "
                  << ms(sum.max, opt.rate) << " ms\n";
        std::cout.precision(3);
        std::cout << "Drift   : " << ms(sum.slope_per_min, opt.rate) << " ms/min ("
                  << 1e6 * sum.slope_per_min / (60.0 * opt.rate) << " ppm) over "
                  << minutes << " min\n";
        const double window = static_cast<double>(s.sched.interval - s.marker.size());
        if (sum.max > 0.8 * window)
            std::cout << "Warning : delays approach the " << ms(window, opt.rate)
                      << " ms search window; raise --interval-ms\n";
    }
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);

    if (!opt.csv.empty()) {
        std::ofstream f(opt.csv);
        f << "marker,t_s,found,latency_ms,score\n";
        for (size_t k = 0; k < dets.size(); ++k)
            f << k << ',' << static_cast<double>(s.sched.start(k)) / opt.rate << ','
              << (dets[k].found ? 1 : 0) << ',' << ms(dets[k].lag, opt.rate) << ','
              << dets[k].score << '\n';
        if (!f) std::cerr << "Failed to write " << opt.csv << '\n';
    }
    return sum.found == s.sched.count;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
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
        constexpr uint32_t kMaxIndex = std::numeric_limits<int>::max();
        uint32_t v = 0;

        if (a == "-h" || a == "--help")      { print_usage(argv[0]); return 0; }
        else if (a == "--list-devices")      opt.list = true;
        else if (a == "--loopback")          opt.loopback = true;
        else if (a == "record"  && opt.cmd == Command::None) opt.cmd = Command::Record;
        else if (a == "tone"    && opt.cmd == Command::None) opt.cmd = Command::Tone;
        else if (a == "latency" && opt.cmd == Command::None) opt.cmd = Command::Latency;
        else if (a == "--input")           { if (!value(0, kMaxIndex, v)) return 1; opt.input = static_cast<int>(v); }
        else if (a == "--output")          { if (!value(0, kMaxIndex, v)) return 1; opt.output = static_cast<int>(v); }
        else if (a == "--input-name" || a == "--output-name") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << '\n'; return 1; }
            if (a == "--input-name") opt.input_name  = argv[++i];
            else                     opt.output_name = argv[++i];
        }
        else if (a == "--seconds")         { if (!value(1, 24 * 3600, opt.seconds)) return 1; }
        else if (a == "--freq")            { if (!value(20, 20000, opt.freq)) return 1; }
        else if (a == "--count")           { if (!value(1, 1000000, opt.count)) return 1; }
        else if (a == "--interval-ms")     { if (!value(100, 60000, opt.interval_ms)) return 1; }
        else if (a == "--rate")            { if (!value(8000, 192000, opt.rate)) return 1; }
        else if (a == "--csv") {
            if (i + 1 >= argc) { std::cerr << "Missing value for --csv\n"; return 1; }
            opt.csv = argv[++i];
        }
        else if (!a.empty() && a[0] != '-' && opt.cmd != Command::None && opt.wav.empty()) opt.wav = a;
        else { std::cerr << "Unexpected argument: " << a << '\n'; print_usage(argv[0]); return 1; }
    }

    if (!opt.list) {
        const bool wants_wav = opt.cmd == Command::Record || opt.cmd == Command::Tone;
        const bool has_input  = opt.input >= 0 || !opt.input_name.empty();
        const bool has_output = opt.output >= 0 || !opt.output_name.empty();
        const char* err = nullptr;
        if (opt.cmd == Command::None)                              err = "Missing command (record, tone or latency)";
        else if (opt.loopback && opt.cmd != Command::Record)       err = "--loopback is only valid with the record command";
        else if (opt.input >= 0 && !opt.input_name.empty())        err = "--input and --input-name are mutually exclusive";
        else if (opt.output >= 0 && !opt.output_name.empty())      err = "--output and --output-name are mutually exclusive";
        else if (!has_input)                                       err = "--input or --input-name is required";
        else if (opt.cmd != Command::Record && !has_output)        err = "--output or --output-name is required";
        else if (wants_wav && opt.wav.empty())                     err = "Missing output WAV path";
        else if (!wants_wav && !opt.wav.empty())                   err = "latency takes no WAV path";
        if (err) { std::cerr << err << '\n'; print_usage(argv[0]); return 1; }
    }

    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        std::cerr << "Failed to initialise an audio backend\n";
        return 1;
    }
    struct ContextGuard { ma_context* c; ~ContextGuard() { ma_context_uninit(c); } } ctx_guard{&ctx};

    ma_device_info* play = nullptr;
    ma_device_info* cap = nullptr;
    ma_uint32 n_play = 0, n_cap = 0;
    if (ma_context_get_devices(&ctx, &play, &n_play, &cap, &n_cap) != MA_SUCCESS) {
        std::cerr << "Failed to enumerate devices\n";
        return 1;
    }
    if (opt.list) {
        std::cout << "Backend: " << ma_get_backend_name(ctx.backend) << '\n';
        auto print = [](const char* title, const ma_device_info* d, ma_uint32 n) {
            std::cout << title << '\n';
            for (ma_uint32 i = 0; i < n; ++i)
                std::cout << "  [" << i << "] " << d[i].name << (d[i].isDefault ? "  (default)" : "") << '\n';
        };
        print("Capture devices:", cap, n_cap);
        print("Playback devices:", play, n_play);
        return 0;
    }

    if (opt.loopback && !ma_context_is_loopback_supported(&ctx)) {
        std::cerr << "System-playback loopback capture is not supported by backend '"
                  << ma_get_backend_name(ctx.backend) << "'\n";
        return 1;
    }

    // Resolve device selections, then copy the ids: the enumeration arrays are
    // only valid until the next call. In --loopback mode, capture.pDeviceID is
    // a PLAYBACK endpoint id (M1.5 stage 06).
    const ma_device_info* in_list = opt.loopback ? play : cap;
    const ma_uint32 n_in          = opt.loopback ? n_play : n_cap;
    const char* in_label          = opt.loopback ? "loopback playback" : "capture";
    const int in_idx = resolve_device(in_list, n_in, opt.input, opt.input_name, in_label);
    if (in_idx < 0) return 1;
    const ma_device_id in_id = in_list[in_idx].id;
    ma_device_id out_id{};
    if (opt.cmd != Command::Record) {
        const int out_idx = resolve_device(play, n_play, opt.output, opt.output_name, "playback");
        if (out_idx < 0) return 1;
        out_id = play[out_idx].id;
    }

    Stream s;
    s.cmd = opt.cmd;
    uint64_t total = static_cast<uint64_t>(opt.seconds) * opt.rate;
    if (opt.cmd == Command::Tone) s.phase_step = kTwoPi * opt.freq / opt.rate;
    if (opt.cmd == Command::Latency) {
        s.marker = probe::make_chirp(opt.rate, opt.rate / 50, 500.0, 3500.0, 0.5);  // 20 ms
        s.sched.first    = opt.rate;  // 1 s for both ends to settle
        s.sched.interval = static_cast<uint64_t>(opt.rate) * opt.interval_ms / 1000;
        s.sched.count    = opt.count;
        total = s.sched.total();
    }

    const ma_device_type dev_type =
        opt.cmd == Command::Record
            ? (opt.loopback ? ma_device_type_loopback : ma_device_type_capture)
            : ma_device_type_duplex;
    ma_device_config cfg = ma_device_config_init(dev_type);
    cfg.capture.pDeviceID  = &in_id;
    cfg.capture.format     = ma_format_f32;
    cfg.capture.channels   = 1;
    cfg.playback.pDeviceID = &out_id;
    cfg.playback.format    = ma_format_f32;
    cfg.playback.channels  = 1;
    cfg.sampleRate         = opt.rate;
    cfg.dataCallback       = &data_callback;
    cfg.pUserData          = &s;

    ma_device dev;
    if (ma_device_init(&ctx, &cfg, &dev) != MA_SUCCESS) {
        std::cerr << "Failed to open audio devices\n";
        return 1;
    }

    std::cout << "Backend : " << ma_get_backend_name(ctx.backend) << '\n';
    if (opt.cmd != Command::Record)
        std::cout << "Playback: " << dev.playback.name << "  (backend " << dev.playback.internalSampleRate
                  << " Hz, period " << dev.playback.internalPeriodSizeInFrames << " frames)\n";
    std::cout << "Capture : " << dev.capture.name << "  (backend " << dev.capture.internalSampleRate
              << " Hz, period " << dev.capture.internalPeriodSizeInFrames << " frames)\n"
              << "Client  : " << dev.sampleRate << " Hz mono f32\n";

    WavWriter wav;
    std::optional<probe::MarkerDetector> det;
    if (opt.cmd == Command::Latency) {
        det.emplace(s.marker, s.sched);
        std::cout << "Probing : " << opt.count << " chirps, one every " << opt.interval_ms
                  << " ms (" << static_cast<double>(total) / opt.rate << " s)\n";
    } else if (!wav.open(opt.wav, opt.rate)) {
        std::cerr << "Cannot write " << opt.wav << '\n';
        ma_device_uninit(&dev);
        return 1;
    } else {
        std::cout << "Recording " << opt.seconds << " s to " << opt.wav << '\n';
    }
    std::cout.flush();

    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);
    if (ma_device_start(&dev) != MA_SUCCESS) {
        std::cerr << "Failed to start audio devices\n";
        ma_device_uninit(&dev);
        return 1;
    }

    std::vector<float> buf(4096);
    uint64_t got = 0;
    size_t reported = 0;
    while (got < total && !g_stop.load()) {
        size_t n = 0;
        while (n < buf.size() && s.captured.try_dequeue(buf[n])) ++n;
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        n = static_cast<size_t>(std::min<uint64_t>(n, total - got));
        if (det) det->push(buf.data(), n);
        else     wav.write(buf.data(), n);
        got += n;

        // Progress for long (drift) runs: one line per 10 % of the markers.
        if (det && det->analysed() >= reported + std::max<size_t>(1, opt.count / 10)) {
            reported = det->analysed();
            const auto& d = det->detections()[reported - 1];
            std::cout << "  " << reported << '/' << opt.count << "  last: ";
            if (d.found) std::cout << static_cast<double>(std::lround(100.0 * ms(d.lag, opt.rate))) / 100.0 << " ms\n";
            else         std::cout << "missed\n";
            std::cout.flush();
        }
    }
    ma_device_uninit(&dev);

    const uint64_t overflows = s.overflows.load();
    if (overflows > 0)
        std::cerr << "Capture queue overflowed (" << overflows
                  << " samples lost): results are not trustworthy\n";

    bool ok = overflows == 0 && got == total;
    if (det) ok = report_latency(*det, s, opt) && ok;
    else if (!wav.close()) { std::cerr << "Failed to write " << opt.wav << '\n'; ok = false; }
    else std::cout << "Wrote " << got << " samples (" << static_cast<double>(got) / opt.rate
                   << " s) to " << opt.wav << '\n';
    return ok ? 0 : 1;
}
