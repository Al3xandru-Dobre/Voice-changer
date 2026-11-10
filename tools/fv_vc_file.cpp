// fv_vc_file — offline file -> file Mode A voice-conversion runner
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
//
// Runs the stateful ONNX encoder -> causal YIN pitch/energy tracker ->
// target-domain conditioning -> stateful ONNX unit vocoder chain step by step
// through InferencePipeline + OrtModel without any audio device.
//
// Interface:
//   fv_vc_file --model-dir <bundle> --voice <name>
//              [--frames-per-step <1|2|4>] [--dump-units [path]]
//              [--repeat <N>] [--pcm16] in.wav out.wav
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/ModelBundle.hpp"
#include "pipeline/ModelContract.hpp"
#include "pipeline/OrtModel.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "util/ProcessMemory.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
using namespace fluxvoice;

namespace {

struct Options {
    fs::path model_dir;
    std::string voice;
    int frames_per_step = 0; // 0 = auto-pick first available variant (s1 -> s2 -> s4)
    bool dump_units = false;
    fs::path dump_units_path;
    std::uint32_t repeat = 1;
    bool write_pcm16 = false;
    fs::path in_wav;
    fs::path out_wav;
};

void print_usage(const char* argv0) {
    std::cout <<
        "fv_vc_file — offline Mode A voice-conversion runner (FluxVoice M2 Stage 07)\n"
        "\n"
        "Usage:\n"
        "  " << argv0 << " --model-dir <bundle> --voice <name> [--frames-per-step <1|2|4>]\n"
        "             [--dump-units [path]] [--repeat <N>] [--pcm16] <in.wav> <out.wav>\n"
        "\n"
        "Options:\n"
        "  --model-dir <dir>         Model bundle directory (encoder_s*.onnx, vocoder_s*.onnx, voices.json)\n"
        "  --voice <name>            Target voice name from voices.json\n"
        "  --frames-per-step <1|2|4> Hop variant (default: first available of 1, 2, 4)\n"
        "  --dump-units [path]       Print discrete unit IDs (and optionally write .npy or .txt)\n"
        "  --repeat <N>              Loop input N times after warm-up and report RSS delta (default 1)\n"
        "  --pcm16                   Write 16-bit PCM WAV instead of 32-bit float WAV\n"
        "  -h, --help                Show this help and exit\n";
}

bool parse_uint(std::string_view s, std::uint32_t lo, std::uint32_t hi, std::uint32_t& out) {
    std::uint32_t v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || v < lo || v > hi)
        return false;
    out = v;
    return true;
}

std::uint16_t read_u16_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>(p[0] | (std::uint16_t{p[1]} << 8));
}

std::uint32_t read_u32_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

bool read_mono_wav(const fs::path& path, std::uint32_t& sample_rate,
                   std::vector<float>& samples, std::string& error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        error = "cannot open input WAV: " + path.string();
        return false;
    }
    const auto file_size = f.tellg();
    if (file_size < 44) {
        error = "input WAV is too short: " + path.string();
        return false;
    }
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(file_size));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(buf.data()), file_size)) {
        error = "failed to read input WAV: " + path.string();
        return false;
    }
    if (std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        error = "input file is not a RIFF/WAVE file: " + path.string();
        return false;
    }

    bool have_fmt = false;
    bool have_data = false;
    std::uint16_t format_tag = 0;
    std::uint16_t channels = 0;
    std::uint16_t bits_per_sample = 0;
    const std::uint8_t* data_ptr = nullptr;
    std::size_t data_bytes = 0;

    std::size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        const char* id = reinterpret_cast<const char*>(buf.data() + pos);
        const std::uint32_t chunk_size = read_u32_le(buf.data() + pos + 4);
        const std::size_t payload = pos + 8;
        if (payload + chunk_size > buf.size()) break;

        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                error = "malformed fmt chunk in " + path.string();
                return false;
            }
            format_tag = read_u16_le(buf.data() + payload);
            channels = read_u16_le(buf.data() + payload + 2);
            sample_rate = read_u32_le(buf.data() + payload + 4);
            bits_per_sample = read_u16_le(buf.data() + payload + 14);
            if (format_tag == 0xFFFEu && chunk_size >= 40) {
                format_tag = read_u16_le(buf.data() + payload + 24);
            }
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_ptr = buf.data() + payload;
            data_bytes = chunk_size;
            have_data = true;
            break;
        }
        pos = payload + chunk_size + (chunk_size & 1u);
    }

    if (!have_fmt || !have_data) {
        error = "WAV file missing fmt or data chunk: " + path.string();
        return false;
    }
    if (channels != 1) {
        error = "WAV must be mono (got " + std::to_string(channels) + " channels)";
        return false;
    }

    if (format_tag == 1 && bits_per_sample == 16) {
        const std::size_t n = data_bytes / 2;
        samples.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto u = read_u16_le(data_ptr + 2 * i);
            const auto s = static_cast<std::int16_t>(u);
            samples[i] = static_cast<float>(s) / 32768.0f;
        }
        return true;
    }
    if (format_tag == 3 && bits_per_sample == 32) {
        const std::size_t n = data_bytes / 4;
        samples.resize(n);
        std::memcpy(samples.data(), data_ptr, n * sizeof(float));
        return true;
    }

    error = "unsupported WAV format (format_tag=" + std::to_string(format_tag) +
            ", bits_per_sample=" + std::to_string(bits_per_sample) +
            "; expected 16-bit PCM or 32-bit float)";
    return false;
}

bool write_mono_wav(const fs::path& path, std::uint32_t sample_rate,
                    std::span<const float> samples, bool pcm16, std::string& error) {
    if (path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open output WAV for writing: " + path.string();
        return false;
    }
    auto u16 = [&](std::uint32_t v) {
        f.put(static_cast<char>(v & 0xFFu));
        f.put(static_cast<char>((v >> 8) & 0xFFu));
    };
    auto u32 = [&](std::uint32_t v) {
        u16(v & 0xFFFFu);
        u16((v >> 16) & 0xFFFFu);
    };

    const std::uint16_t format_tag = pcm16 ? 1u : 3u;
    const std::uint16_t bytes_per_sample = pcm16 ? 2u : 4u;
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * bytes_per_sample);

    f.write("RIFF", 4);
    u32(36u + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    u32(16u);
    u16(format_tag);
    u16(1u); // mono
    u32(sample_rate);
    u32(sample_rate * bytes_per_sample);
    u16(bytes_per_sample);
    u16(bytes_per_sample * 8u);
    f.write("data", 4);
    u32(data_bytes);

    if (pcm16) {
        for (float s : samples) {
            const float clamped = std::clamp(s, -1.0f, 1.0f);
            const auto q = static_cast<std::int16_t>(std::lround(clamped * 32767.0f));
            u16(static_cast<std::uint16_t>(q));
        }
    } else {
        f.write(reinterpret_cast<const char*>(samples.data()),
                static_cast<std::streamsize>(data_bytes));
    }

    if (!f) {
        error = "failed writing output WAV: " + path.string();
        return false;
    }
    return true;
}

bool write_units_file(const fs::path& path, std::span<const std::int64_t> units, std::string& error) {
    if (path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open units file for writing: " + path.string();
        return false;
    }
    if (path.extension() == ".npy") {
        // NumPy v1.0 header for a 1-D little-endian int64 array.
        std::string dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (" +
                           std::to_string(units.size()) + ",), }";
        const std::size_t preamble = 10;
        const std::size_t rem = (preamble + dict.size() + 1) % 64;
        const std::size_t pad = rem == 0 ? 0 : (64 - rem);
        dict.append(pad, ' ');
        dict.push_back('\n');
        const auto hdr_len = static_cast<std::uint16_t>(dict.size());
        const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
        f.write(magic, sizeof(magic));
        f.put(static_cast<char>(hdr_len & 0xFFu));
        f.put(static_cast<char>((hdr_len >> 8) & 0xFFu));
        f.write(dict.data(), static_cast<std::streamsize>(dict.size()));
        f.write(reinterpret_cast<const char*>(units.data()),
                static_cast<std::streamsize>(units.size() * sizeof(std::int64_t)));
    } else {
        for (std::int64_t u : units) f << u << '\n';
    }
    if (!f) {
        error = "failed writing units file: " + path.string();
        return false;
    }
    return true;
}

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(idx));
    const auto hi = static_cast<std::size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

bool parse_args(int argc, char** argv, Options& opts, bool& help_requested) {
    help_requested = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need_val = [&](std::string_view& out) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << a << '\n';
                return false;
            }
            out = argv[++i];
            return true;
        };

        if (a == "-h" || a == "--help") {
            help_requested = true;
            return true;
        } else if (a == "--model-dir") {
            std::string_view v;
            if (!need_val(v)) return false;
            opts.model_dir = v;
        } else if (a == "--voice") {
            std::string_view v;
            if (!need_val(v)) return false;
            opts.voice = std::string(v);
        } else if (a == "--frames-per-step") {
            std::string_view v;
            if (!need_val(v)) return false;
            std::uint32_t fps = 0;
            if (!parse_uint(v, 1, 4, fps) || (fps != 1 && fps != 2 && fps != 4)) {
                std::cerr << "Invalid --frames-per-step '" << v << "' (expected 1, 2 or 4)\n";
                return false;
            }
            opts.frames_per_step = static_cast<int>(fps);
        } else if (a == "--repeat") {
            std::string_view v;
            if (!need_val(v)) return false;
            if (!parse_uint(v, 1, 1000000, opts.repeat)) {
                std::cerr << "Invalid --repeat '" << v << "' (expected integer >= 1)\n";
                return false;
            }
        } else if (a == "--pcm16") {
            opts.write_pcm16 = true;
        } else if (a == "--dump-units") {
            opts.dump_units = true;
            // Optional path argument: consume next token if it does not start
            // with '-' and at least 2 positional arguments still follow it.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                int remaining_non_opts = 0;
                for (int j = i + 1; j < argc; ++j) {
                    if (argv[j][0] != '-') ++remaining_non_opts;
                }
                if (remaining_non_opts >= 3) {
                    opts.dump_units_path = argv[++i];
                }
            }
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << '\n';
            return false;
        } else {
            pos.emplace_back(a);
        }
    }

    if (opts.model_dir.empty()) {
        std::cerr << "Missing required --model-dir <bundle>\n";
        return false;
    }
    if (opts.voice.empty()) {
        std::cerr << "Missing required --voice <name>\n";
        return false;
    }
    if (pos.size() != 2) {
        std::cerr << "Expected <in.wav> and <out.wav> positional arguments\n";
        return false;
    }
    opts.in_wav = pos[0];
    opts.out_wav = pos[1];
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options opts;
    bool help_requested = false;
    if (!parse_args(argc, argv, opts, help_requested)) {
        print_usage(argv[0]);
        return 1;
    }
    if (help_requested) {
        print_usage(argv[0]);
        return 0;
    }

    if (!ort_available()) {
        std::cerr << "fv_vc_file: " << kBuiltWithoutOrtMessage << '\n';
        return 1;
    }

    ModelBundle bundle;
    std::string error;
    if (!bundle.load(opts.model_dir, opts.frames_per_step, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const VoiceEntry* selected = find_voice(bundle.voices, opts.voice, error);
    if (!selected) {
        std::cerr << error << '\n';
        return 1;
    }
    const VoiceEntry& voice = *selected;
    OrtModel& enc = bundle.encoder;
    OrtModel& voc = bundle.vocoder;
    const std::string variant = "s" + std::to_string(bundle.paths.frames_per_step);

    std::uint32_t wav_rate = 0;
    std::vector<float> in_samples;
    if (!read_mono_wav(opts.in_wav, wav_rate, in_samples, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (static_cast<int>(wav_rate) != enc.contract().sample_rate) {
        std::cerr << "Sample-rate mismatch: input WAV is " << wav_rate
                  << " Hz, model contract expects " << enc.contract().sample_rate
                  << " Hz (resample the input before running fv_vc_file)\n";
        return 1;
    }

    const std::size_t hop = static_cast<std::size_t>(enc.contract().hop());
    const std::size_t steps_per_pass = in_samples.size() / hop;
    if (steps_per_pass == 0) {
        std::cerr << "Input WAV (" << in_samples.size() << " samples) is shorter than one hop ("
                  << hop << " samples)\n";
        return 1;
    }
    const std::size_t usable_samples = steps_per_pass * hop;

    InferencePipeline pipeline(enc, voc, voice.stats, voice.embedding);
    if (!pipeline.warm_up(20)) {
        std::cerr << "Warm-up failed: " << pipeline.last_error() << '\n';
        return 1;
    }

    const std::size_t total_steps = steps_per_pass * static_cast<std::size_t>(opts.repeat);
    std::vector<float> out_samples(usable_samples, 0.0f);
    std::vector<double> step_us(total_steps, 0.0);
    std::vector<std::int64_t> dumped_units;
    if (opts.dump_units && enc.contract().content_kind == ContentKind::kDiscrete) {
        dumped_units.resize(steps_per_pass * static_cast<std::size_t>(enc.contract().frames_per_step), 0);
    }

    const std::size_t rss_before = resident_bytes();

    std::size_t step_idx = 0;
    for (std::uint32_t rep = 0; rep < opts.repeat; ++rep) {
        pipeline.reset();
        for (std::size_t s = 0; s < steps_per_pass; ++s) {
            const std::span<const float> in_hop(in_samples.data() + s * hop, hop);
            std::span<float> out_hop(out_samples.data() + s * hop, hop);

            const auto t0 = std::chrono::steady_clock::now();
            const PipelineStepResult res = pipeline.process(in_hop, out_hop);
            const auto t1 = std::chrono::steady_clock::now();
            step_us[step_idx++] = std::chrono::duration<double, std::micro>(t1 - t0).count();

            if (!res.ok) {
                std::cerr << "Inference step failed at repeat " << rep << ", step " << s << ": "
                          << res.error << '\n';
                return 1;
            }

            if (rep == 0 && !dumped_units.empty()) {
                const auto u = enc.units_out();
                std::copy(u.begin(), u.end(), dumped_units.begin() + static_cast<std::ptrdiff_t>(s * u.size()));
            }
        }
    }

    const std::size_t rss_after = resident_bytes();

    if (!write_mono_wav(opts.out_wav, wav_rate, out_samples, opts.write_pcm16, error)) {
        std::cerr << error << '\n';
        return 1;
    }

    if (opts.dump_units) {
        if (!opts.dump_units_path.empty()) {
            if (!write_units_file(opts.dump_units_path, dumped_units, error)) {
                std::cerr << error << '\n';
                return 1;
            }
        }
        std::cout << "Units (" << dumped_units.size() << " frames):";
        const std::size_t show = opts.dump_units_path.empty() ? dumped_units.size()
                                                              : std::min<std::size_t>(dumped_units.size(), 32);
        for (std::size_t i = 0; i < show; ++i) std::cout << ' ' << dumped_units[i];
        if (show < dumped_units.size()) std::cout << " ...";
        std::cout << '\n';
    }

    std::sort(step_us.begin(), step_us.end());
    const double p50_us = percentile(step_us, 50.0);
    const double p95_us = percentile(step_us, 95.0);
    const double max_us = step_us.back();
    const std::int64_t alg_samples = algorithmic_latency_samples(enc.contract(), voc.contract());
    const double alg_ms = 1000.0 * static_cast<double>(alg_samples) / enc.contract().sample_rate;
    const double hop_ms = 1000.0 * static_cast<double>(hop) / enc.contract().sample_rate;
    const std::int64_t rss_delta_bytes =
        static_cast<std::int64_t>(rss_after) - static_cast<std::int64_t>(rss_before);

    std::printf("ONNX Runtime       : %s\n", ort_runtime_version().c_str());
    std::printf("Bundle / variant   : %s (%s, hop %zu samples = %.1f ms @ %d Hz)\n",
                opts.model_dir.string().c_str(), variant.c_str(), hop, hop_ms,
                enc.contract().sample_rate);
    std::printf("Voice              : %s (%s)\n", voice.name.c_str(), voice.display.c_str());
    std::printf("Algorithmic latency: %lld samples (%.1f ms)\n",
                static_cast<long long>(alg_samples), alg_ms);
    std::printf("Steps              : %zu (%zu per pass x %u repeat)\n",
                total_steps, steps_per_pass, opts.repeat);
    std::printf("Step timing (us)   : p50=%.1f us  p95=%.1f us  max=%.1f us\n",
                p50_us, p95_us, max_us);
    std::printf("RSS (KB)           : before=%zu KB  after=%zu KB  delta=%+lld KB (%.3f MB)\n",
                rss_before / 1024, rss_after / 1024,
                static_cast<long long>(rss_delta_bytes / 1024),
                static_cast<double>(rss_delta_bytes) / (1024.0 * 1024.0));
    std::printf("Output WAV         : %s (%zu samples, %.2f s)\n",
                opts.out_wav.string().c_str(), out_samples.size(),
                static_cast<double>(out_samples.size()) / wav_rate);
    return 0;
}
