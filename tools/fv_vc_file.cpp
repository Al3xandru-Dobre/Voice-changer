// fv_vc_file — offline file -> file Mode A voice-conversion runner
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
//
// Runs the stateful ONNX encoder -> causal YIN pitch/energy tracker ->
// target-domain conditioning -> stateful ONNX unit vocoder chain step by step
// through InferencePipeline + OrtModel without any audio device.
//
// Interface:
//   fv_vc_file --model-dir <bundle> [--voice <name>]
//              [--frames-per-step <1|2|4>] [--dump-units [path]]
//              [--preset <name|path>] [--preset-dir <dir>] [--param name=value]
//              [--voice-b <name>] [--change <seconds>:<name>=<value>]
//              [--no-output-chain]
//              [--repeat <N>] [--pcm16] in.wav out.wav
//
// The output chain (gain -> gate -> brightness -> limiter) defaults to the
// full shipped chain; --no-output-chain leaves raw vocoder PCM for parity
// checks against runtimes that have no output stage.
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/ModelBundle.hpp"
#include "pipeline/ModelContract.hpp"
#include "pipeline/OrtModel.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "ToolCli.hpp"
#include "WavIo.hpp"
#include "util/ProcessMemory.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

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
using namespace fluxvoice::tools;

namespace {

struct Options {
    fs::path model_dir;
    std::string voice;
    std::string voice_b;
    std::string preset;
    fs::path preset_dir;
    std::vector<std::pair<const ParamDesc*, float>> assignments;
    std::vector<ScheduledChange> changes;
    bool no_output_chain = false;
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
        "  " << argv0 << " --model-dir <bundle> [--voice <name>] [--frames-per-step <1|2|4>]\n"
        "             [--preset <name|path>] [--param <name=value>] [--voice-b <name>]\n"
        "             [--change <seconds>:<name>=<value>] [--no-output-chain]\n"
        "             [--dump-units [path]] [--repeat <N>] [--pcm16] <in.wav> <out.wav>\n"
        "\n"
        "Options:\n"
        "  --model-dir <dir>         Model bundle directory (encoder_s*.onnx, vocoder_s*.onnx, voices.json)\n"
        "  --voice <name>          Target voice name from voices.json (default: first bundle voice)\n"
        "  --voice-b <name>        Blend endpoint voice B (default: follows --voice)\n"
        "  --preset <name|path>    Voice preset (searched in --preset-dir, exe/presets, source presets/,\n"
        "                          cwd/presets) or a .json path\n"
        "  --preset-dir <dir>      Extra preset directory searched first\n"
        "  --param <name=value>    Set a voice parameter (repeatable)\n"
        "  --change <s>:<n>=<v>    Apply a parameter change mid-run at the first hop starting\n"
        "                          at/after <s> seconds (repeatable; for click tests)\n"
        "  --no-output-chain       Raw vocoder PCM: no gain/gate/brightness/limiter (parity only)\n"
        "  --frames-per-step <1|2|4> Hop variant (default: first available of 1, 2, 4)\n"
        "  --dump-units [path]     Print discrete unit IDs (and optionally write .npy or .txt)\n"
        "  --repeat <N>            Loop input N times after warm-up and report RSS delta (default 1)\n"
        "  --pcm16                 Write 16-bit PCM WAV instead of 32-bit float WAV\n"
        "  -h, --help              Show this help and exit\n";
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
        } else if (a == "--voice-b") {
            std::string_view v;
            if (!need_val(v)) return false;
            opts.voice_b = std::string(v);
        } else if (a == "--preset") {
            std::string_view v;
            if (!need_val(v)) return false;
            opts.preset = std::string(v);
        } else if (a == "--preset-dir") {
            std::string_view v;
            if (!need_val(v)) return false;
            opts.preset_dir = fs::path(v);
        } else if (a == "--param") {
            std::string_view v;
            if (!need_val(v)) return false;
            if (!parse_param_option(v, opts.assignments)) return false;
        } else if (a == "--change") {
            std::string_view v;
            if (!need_val(v)) return false;
            if (!parse_change_option(v, opts.changes)) return false;
        } else if (a == "--no-output-chain") {
            opts.no_output_chain = true;
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

    // Same resolution order as the CLI: descriptor defaults -> preset params
    // and voice names -> --voice / --voice-b -> --param assignments.
    PresetLoadResult preset_res;
    if (!load_preset_option(opts.preset, opts.preset_dir, argv[0], preset_res)) return 1;

    ModelBundle bundle;
    std::string error;
    if (!bundle.load(opts.model_dir, opts.frames_per_step, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    VoiceParams params;
    if (!resolve_voice_params(bundle.voices, preset_res, opts.voice, opts.voice_b, opts.assignments,
                              params))
        return 1;
    const VoiceEntry& voice = bundle.voices[params.voice_a];
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

    std::vector<PipelineVoice> pipeline_voices;
    pipeline_voices.reserve(bundle.voices.size());
    for (const VoiceEntry& v : bundle.voices)
        pipeline_voices.push_back(
            {v.stats, std::span<const float>(v.embedding.data(), v.embedding.size())});
    PipelineOptions pipe_options;
    pipe_options.params = params;
    pipe_options.output_chain = !opts.no_output_chain;
    InferencePipeline pipeline(enc, voc, pipeline_voices, pipe_options);
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

    std::sort(opts.changes.begin(), opts.changes.end(),
              [](const ScheduledChange& a, const ScheduledChange& b) {
                  return a.seconds < b.seconds;
              });
    const VoiceParams initial_params = params;
    std::size_t step_idx = 0;
    for (std::uint32_t rep = 0; rep < opts.repeat; ++rep) {
        // --change schedules are per pass: restore the initial params snapped
        // (set_params then reset() pins the target without a ramp) and restart
        // the schedule at the pass-local position.
        params = initial_params;
        if (!pipeline.set_params(params)) {
            std::cerr << "Could not restore initial parameters\n";
            return 1;
        }
        pipeline.reset();
        std::size_t next_change = 0;
        for (std::size_t s = 0; s < steps_per_pass; ++s) {
            const std::span<const float> in_hop(in_samples.data() + s * hop, hop);
            std::span<float> out_hop(out_samples.data() + s * hop, hop);

            // --change: first hop whose start reaches the requested time.
            const std::size_t pos = s * hop;
            while (next_change < opts.changes.size() &&
                   static_cast<double>(pos) >=
                       opts.changes[next_change].seconds * wav_rate) {
                const ScheduledChange& c = opts.changes[next_change];
                params.set(c.desc->id, c.value);
                if (!pipeline.set_params(params)) {
                    std::cerr << "Rejected --change at " << c.seconds << " s\n";
                    return 1;
                }
                ++next_change;
            }

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
    if (initial_params.voice_b != initial_params.voice_a)
        std::printf("Voice B            : %s\n",
                    bundle.voices[initial_params.voice_b].name.c_str());
    // Effective non-default controls (the shipped output chain included).
    print_param_summary(initial_params, opts.changes);
    std::printf("Output chain       : %s\n", opts.no_output_chain ? "off (raw vocoder PCM)"
                                                                 : "gain->gate->brightness->limiter");
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
