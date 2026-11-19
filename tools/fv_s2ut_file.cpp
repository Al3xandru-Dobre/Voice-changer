// fv_s2ut_file — offline file -> file Mode B (simultaneous S2UT) runner
// (Milestone 3 / Stages 07-09, Documents/Milestone 3/07_cpp_translator_runtime_offline_runner.md).
//
// Runs encoder -> translator_enc -> WaitKPolicy -> translator_dec ->
// UnitProsodyFifo -> unit vocoder -> output chain step by step through
// InferencePipeline + OrtModel without any audio device. After the input ends
// the runner keeps feeding zero hops until the pipeline is idle (the kFlush
// tail has drained) or --max-tail-ms of padding was fed.
//
// Interface:
//   fv_s2ut_file --model-dir <bundle> --src-lang <code|id> --tgt-lang <code|id>
//                [--voice <name>] [--voice-b <name>] [--wait-k <1..10>] [--adaptive]
//                [--frames-per-step <1|2|4>] [--preset <name|path>] [--preset-dir <dir>]
//                [--param name=value]... [--change <seconds>:name=value]...
//                [--dump-trace <path.jsonl>] [--repeat <N>] [--raw] [--no-prosody-ctx]
//                [--max-tail-ms <ms>] [--max-rss-growth-kb <KB>] in.wav out.wav
//
// --raw leaves raw vocoder PCM (no gain/gate/brightness/limiter, exact-zero
// silence: parity runs only). --no-prosody-ctx feeds prosody_ctx = [0, 0] like
// the Python driver fvml/s2ut_stream.py. Exit codes: 0 ok, 1 error, 3 RSS
// growth above --max-rss-growth-kb.
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/ModelBundle.hpp"
#include "pipeline/ModelContract.hpp"
#include "pipeline/OrtModel.hpp"
#include "ToolCli.hpp"
#include "WavIo.hpp"
#include "util/ProcessMemory.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace fluxvoice;
using namespace fluxvoice::tools;

namespace {

struct Options {
    fs::path model_dir;
    std::string src_lang;
    std::string tgt_lang;
    std::string voice;
    std::string voice_b;
    std::string preset;
    fs::path preset_dir;
    ParamAssignments assignments;
    std::vector<ScheduledChange> changes;
    fs::path trace_path;
    int wait_k = 0; // 0 = bundle default for the pair
    bool adaptive = false;
    int frames_per_step = 4;
    std::uint32_t repeat = 1;
    bool raw = false;
    bool prosody_ctx = true;
    double max_tail_ms = -1.0; // < 0 = 400 + (32 + k + 2) * 80
    bool have_max_rss_growth = false;
    std::uint32_t max_rss_growth_kb = 0;
    fs::path in_wav;
    fs::path out_wav;
};

void print_usage(const char* argv0) {
    std::cout <<
        "fv_s2ut_file — offline Mode B simultaneous speech-to-unit-speech runner (FluxVoice M3 Stage 07)\n"
        "\n"
        "Usage:\n"
        "  " << argv0 << " --model-dir <bundle> --src-lang <code|id> --tgt-lang <code|id>\n"
        "             [--voice <name>] [--voice-b <name>] [--wait-k <1..10>] [--adaptive]\n"
        "             [--frames-per-step <1|2|4>] [--preset <name|path>] [--preset-dir <dir>]\n"
        "             [--param <name=value>]... [--change <seconds>:<name>=<value>]...\n"
        "             [--dump-trace <path.jsonl>] [--repeat <N>] [--raw] [--no-prosody-ctx]\n"
        "             [--max-tail-ms <ms>] [--max-rss-growth-kb <KB>] <in.wav> <out.wav>\n"
        "\n"
        "Options:\n"
        "  --model-dir <dir>         Mode B bundle (encoder/vocoder_s*.onnx, translator_enc/dec.onnx,\n"
        "                            voices.json, langs.json)\n"
        "  --src-lang <code|id>      Source language (langs.json code or id)\n"
        "  --tgt-lang <code|id>      Target language (must differ from the source)\n"
        "  --voice <name>            Target voice from voices.json (default: first bundle voice)\n"
        "  --voice-b <name>          Blend endpoint voice B (default: follows --voice)\n"
        "  --wait-k <1..10>          Wait-k (default: langs.json default_wait_k for the pair, else 3)\n"
        "  --adaptive                Adaptive wait-k policy (up to two writes per chunk step)\n"
        "  --frames-per-step <1|2|4> Hop variant (default 4)\n"
        "  --preset <name|path>      Voice preset (searched in --preset-dir, exe/presets, source presets/,\n"
        "                            cwd/presets) or a .json path\n"
        "  --preset-dir <dir>        Extra preset directory searched first\n"
        "  --param <name=value>      Set a Stage 09 control (repeatable)\n"
        "  --change <s>:<n>=<v>      Apply a control change at the first hop starting at/after <s> seconds\n"
        "                            (repeatable; acts on the translated output)\n"
        "  --dump-trace <path>       One JSON object per chunk step (first pass only); elapsed_ms is the\n"
        "                            compute time of the whole hop (process()) in which the step ran\n"
        "  --repeat <N>              Run N passes (reset between passes, no re-warm-up) and report RSS\n"
        "  --raw                     Raw vocoder PCM: no output chain (parity checks only)\n"
        "  --no-prosody-ctx          Feed prosody_ctx = [0, 0] (matches the Python driver)\n"
        "  --max-tail-ms <ms>        Cap on zero padding after the input (also bounds the extra zero\n"
        "                            hops that let the output chain ramp down to silence)\n"
        "                            (default 400 + (32 + k + 2) * 80)\n"
        "  --max-rss-growth-kb <KB>  Exit 3 when RSS grows by more than KB across the passes\n"
        "  -h, --help                Show this help and exit\n";
}

bool parse_args(int argc, char** argv, Options& opts, bool& help_requested) {
    help_requested = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto need_val = [&](std::string_view& out) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << a << '\n';
                return false;
            }
            out = argv[++i];
            return true;
        };
        std::string_view v;
        if (a == "-h" || a == "--help") {
            help_requested = true;
            return true;
        } else if (a == "--model-dir") {
            if (!need_val(v)) return false;
            opts.model_dir = v;
        } else if (a == "--src-lang") {
            if (!need_val(v)) return false;
            opts.src_lang = std::string(v);
        } else if (a == "--tgt-lang") {
            if (!need_val(v)) return false;
            opts.tgt_lang = std::string(v);
        } else if (a == "--voice") {
            if (!need_val(v)) return false;
            opts.voice = std::string(v);
        } else if (a == "--voice-b") {
            if (!need_val(v)) return false;
            opts.voice_b = std::string(v);
        } else if (a == "--preset") {
            if (!need_val(v)) return false;
            opts.preset = std::string(v);
        } else if (a == "--preset-dir") {
            if (!need_val(v)) return false;
            opts.preset_dir = fs::path(v);
        } else if (a == "--param") {
            if (!need_val(v)) return false;
            if (!parse_param_option(v, opts.assignments)) return false;
        } else if (a == "--change") {
            if (!need_val(v)) return false;
            if (!parse_change_option(v, opts.changes)) return false;
        } else if (a == "--dump-trace") {
            if (!need_val(v)) return false;
            opts.trace_path = v;
        } else if (a == "--wait-k") {
            if (!need_val(v)) return false;
            std::uint32_t k = 0;
            if (!parse_uint(v, 1, 10, k)) {
                std::cerr << "Invalid --wait-k '" << v << "' (expected 1..10)\n";
                return false;
            }
            opts.wait_k = static_cast<int>(k);
        } else if (a == "--adaptive") {
            opts.adaptive = true;
        } else if (a == "--frames-per-step") {
            if (!need_val(v)) return false;
            std::uint32_t fps = 0;
            if (!parse_uint(v, 1, 4, fps) || fps == 3) {
                std::cerr << "Invalid --frames-per-step '" << v << "' (expected 1, 2 or 4)\n";
                return false;
            }
            opts.frames_per_step = static_cast<int>(fps);
        } else if (a == "--repeat") {
            if (!need_val(v)) return false;
            if (!parse_uint(v, 1, 1000000, opts.repeat)) {
                std::cerr << "Invalid --repeat '" << v << "' (expected integer >= 1)\n";
                return false;
            }
        } else if (a == "--raw") {
            opts.raw = true;
        } else if (a == "--no-prosody-ctx") {
            opts.prosody_ctx = false;
        } else if (a == "--max-tail-ms") {
            if (!need_val(v)) return false;
            double ms = 0.0;
            if (!parse_finite_float(v, ms) || ms < 0.0) {
                std::cerr << "Invalid --max-tail-ms '" << v << "' (expected a number >= 0)\n";
                return false;
            }
            opts.max_tail_ms = ms;
        } else if (a == "--max-rss-growth-kb") {
            if (!need_val(v)) return false;
            if (!parse_uint(v, 0, 0xFFFFFFFFu, opts.max_rss_growth_kb)) {
                std::cerr << "Invalid --max-rss-growth-kb '" << v << "' (expected integer >= 0)\n";
                return false;
            }
            opts.have_max_rss_growth = true;
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
    if (opts.src_lang.empty()) {
        std::cerr << "Missing required --src-lang <code|id>\n";
        return false;
    }
    if (opts.tgt_lang.empty()) {
        std::cerr << "Missing required --tgt-lang <code|id>\n";
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

// One JSON object per chunk step (offline tool: allocation in the observer is
// fine; the pipeline itself stays allocation-free). Keys follow the Python
// golden checker: utt_id, step, t_src_ms, ...
class TraceWriter final : public S2utObserver {
public:
    TraceWriter(const fs::path& path, std::string utt_id, double sample_rate)
        : file_(path, std::ios::binary), utt_id_(std::move(utt_id)), sample_rate_(sample_rate) {}

    bool is_open() const { return file_.is_open(); }
    bool failed() const { return failed_ || !file_; }
    std::size_t lines() const { return lines_; }

    /// Buffers the event; flush_pending() writes it once the whole hop's
    /// process() duration is known (elapsed_ms = that hop's compute time).
    void on_chunk(const S2utChunkEvent& e) noexcept override {
        try {
            nlohmann::ordered_json j;
            j["utt_id"] = utt_id_;
            j["step"] = e.chunk;
            j["t_src_ms"] = static_cast<double>(e.input_samples) / sample_rate_ * 1000.0;
            j["t_tgt_frames_emitted"] = e.frames_pushed;
            j["elapsed_ms"] = static_cast<double>(e.trans_enc_us + e.trans_dec_us) / 1000.0;
            j["decision"] = to_string(e.decision);
            j["speech"] = e.speech;
            j["encoded"] = e.encoded;
            j["writes"] = e.writes;
            j["t_src"] = e.t_src;
            j["t_tgt"] = e.t_tgt;
            nlohmann::ordered_json units = nlohmann::ordered_json::array();
            for (const std::int64_t u : e.units) units.push_back(u);
            j["units"] = std::move(units);
            j["eos"] = e.eos;
            j["utterance_end"] = e.utterance_end;
            j["cap_end"] = e.cap_end;
            j["pad_frames"] = e.pad_frames;
            j["segment"] = e.segment;
            pending_.push_back(std::move(j));
        } catch (...) {
            failed_ = true;
        }
    }

    /// Write the buffered chunk events of the hop that just finished.
    void flush_pending(std::uint64_t hop_us) noexcept {
        try {
            for (nlohmann::ordered_json& j : pending_) {
                j["elapsed_ms"] = static_cast<double>(hop_us) / 1000.0; // key keeps its position
                file_ << j.dump() << '\n';
                ++lines_;
            }
        } catch (...) {
            failed_ = true;
        }
        pending_.clear();
    }

private:
    std::ofstream file_;
    std::string utt_id_;
    double sample_rate_;
    std::vector<nlohmann::ordered_json> pending_;
    std::size_t lines_ = 0;
    bool failed_ = false;
};

void print_stage(const char* name, const TimingSnapshot& t) {
    std::printf("  %-10s       : n=%llu  p50=%llu us  p95=%llu us  max=%llu us\n", name,
                static_cast<unsigned long long>(t.samples), static_cast<unsigned long long>(t.p50_us),
                static_cast<unsigned long long>(t.p95_us), static_cast<unsigned long long>(t.max_us));
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
        std::cerr << "fv_s2ut_file: " << kBuiltWithoutOrtMessage << '\n';
        return 1;
    }

    try {
        PresetLoadResult preset_res;
        if (!load_preset_option(opts.preset, opts.preset_dir, argv[0], preset_res)) return 1;

        auto bundle = std::make_unique<S2utBundle>(); // non-movable: translator_dec points at translator_enc
        std::string error;
        if (!bundle->load(opts.model_dir, opts.frames_per_step, error)) {
            std::cerr << error << '\n';
            return 1;
        }

        const auto codes = [&] {
            std::string list;
            for (const LangEntry& l : bundle->langs.langs) {
                if (!list.empty()) list += ", ";
                list += l.code + " (" + std::to_string(l.id) + ")";
            }
            return list;
        };
        const LangEntry* src = bundle->langs.find(opts.src_lang);
        if (!src) {
            std::cerr << "unknown source language '" << opts.src_lang << "' (available: " << codes() << ")\n";
            return 1;
        }
        const LangEntry* tgt = bundle->langs.find(opts.tgt_lang);
        if (!tgt) {
            std::cerr << "unknown target language '" << opts.tgt_lang << "' (available: " << codes() << ")\n";
            return 1;
        }
        if (src->id == tgt->id) {
            std::cerr << "source and target language must differ (both '" << src->code << "')\n";
            return 1;
        }

        VoiceParams params;
        if (!resolve_voice_params(bundle->voices, preset_res, opts.voice, opts.voice_b,
                                  opts.assignments, params))
            return 1;
        const VoiceEntry& voice = bundle->voices[params.voice_a];

        // wait-k: --wait-k, else a wait_k from the preset/--param, else the pair default.
        // Precedence: --wait-k > --param > preset > langs.json pair default.
        // "Explicit" is tracked, not inferred from the value: wait_k=3 equals
        // the descriptor default but is still a request.
        int wait_k = opts.wait_k;
        if (wait_k == 0) {
            bool k_explicit = std::any_of(opts.assignments.begin(), opts.assignments.end(),
                                          [](const auto& a) { return a.first->id == ParamId::WaitK; });
            if (!k_explicit && preset_res.ok())
                k_explicit = preset_res.preset->params[param_index(ParamId::WaitK)].has_value();
            wait_k = k_explicit ? static_cast<int>(std::lround(params.get(ParamId::WaitK)))
                                : bundle->langs.default_k(src->code, tgt->code);
        }

        OrtModel& enc = bundle->encoder;
        OrtModel& tenc = bundle->translator_enc;
        OrtModel& tdec = bundle->translator_dec;
        OrtModel& voc = bundle->vocoder;

        std::uint32_t wav_rate = 0;
        std::vector<float> in_samples;
        if (!read_mono_wav(opts.in_wav, wav_rate, in_samples, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        if (static_cast<int>(wav_rate) != enc.contract().sample_rate) {
            std::cerr << "Sample-rate mismatch: input WAV is " << wav_rate
                      << " Hz, model contract expects " << enc.contract().sample_rate
                      << " Hz (resample the input before running fv_s2ut_file)\n";
            return 1;
        }
        const std::size_t hop = static_cast<std::size_t>(enc.contract().hop());
        const std::size_t input_hops = (in_samples.size() + hop - 1) / hop;
        if (input_hops == 0) {
            std::cerr << "Input WAV is empty\n";
            return 1;
        }

        std::vector<PipelineVoice> pipeline_voices;
        pipeline_voices.reserve(bundle->voices.size());
        for (const VoiceEntry& v : bundle->voices)
            pipeline_voices.push_back(
                {v.stats, std::span<const float>(v.embedding.data(), v.embedding.size())});

        S2utOptions s2ut;
        s2ut.policy.k = wait_k;
        s2ut.policy.adaptive_mode = opts.adaptive || params.enabled(ParamId::AdaptivePolicy);
        s2ut.src_lang_id = src->id;
        s2ut.tgt_lang_id = tgt->id;
        s2ut.prosody_ctx = opts.prosody_ctx;
        PipelineOptions pipe_options;
        pipe_options.params = params;
        pipe_options.output_chain = !opts.raw;
        InferencePipeline pipeline(enc, tenc, tdec, voc, pipeline_voices, s2ut, pipe_options);
        if (!pipeline.warm_up(20)) {
            std::cerr << "Warm-up failed: " << pipeline.last_error() << '\n';
            return 1;
        }

        // Default tail: the pause that starts the flush, plus the longest flush
        // the policy allows (max_flush_steps), plus k + 2 chunks of slack —
        // read from the live policy so it tracks its config (k may also have
        // come from a preset).
        const WaitKPolicy& policy = *pipeline.waitk();
        const double max_tail_ms =
            opts.max_tail_ms >= 0.0
                ? opts.max_tail_ms
                : static_cast<double>(policy.config().tail_flush_ms) +
                      static_cast<double>(policy.config().max_flush_steps + policy.k() + 2) *
                          static_cast<double>(policy.chunk_ms());
        const double hop_ms = 1000.0 * static_cast<double>(hop) / wav_rate;
        const auto max_tail_hops = static_cast<std::size_t>(std::ceil(max_tail_ms / hop_ms));

        std::unique_ptr<TraceWriter> trace;
        if (!opts.trace_path.empty()) {
            if (opts.trace_path.has_parent_path()) {
                std::error_code ec;
                fs::create_directories(opts.trace_path.parent_path(), ec);
            }
            trace = std::make_unique<TraceWriter>(opts.trace_path, opts.in_wav.stem().string(),
                                                  static_cast<double>(wav_rate));
            if (!trace->is_open()) {
                std::cerr << "cannot open trace file for writing: " << opts.trace_path.string() << '\n';
                return 1;
            }
        }

        // Every buffer the passes touch is sized and faulted in before the RSS baseline.
        std::vector<float> out_samples((input_hops + max_tail_hops) * hop, 0.0f);
        std::vector<float> scratch_out(hop, 0.0f);
        std::vector<float> pad_hop(hop, 0.0f);
        std::size_t out_hops = 0;

        std::sort(opts.changes.begin(), opts.changes.end(),
                  [](const ScheduledChange& a, const ScheduledChange& b) { return a.seconds < b.seconds; });
        const VoiceParams initial_params = pipeline.params();
        const std::size_t rss_before = resident_bytes();

        for (std::uint32_t rep = 0; rep < opts.repeat; ++rep) {
            params = initial_params;
            if (!pipeline.set_params(params)) {
                std::cerr << "Could not restore initial parameters\n";
                return 1;
            }
            pipeline.reset();
            pipeline.set_s2ut_observer(rep == 0 ? trace.get() : nullptr);
            std::size_t next_change = 0;
            std::size_t h = 0;
            std::size_t tail_hops = 0;
            // After idle, keep rendering zero hops while the last sample is
            // audible so the output chain's ramp-down/gate release completes
            // (the file must not end on a non-zero sample); still bounded.
            float last_sample = 0.0f;
            while (h < input_hops ||
                   ((!pipeline.s2ut_idle() || std::fabs(last_sample) > 0.001f) && tail_hops < max_tail_hops)) {
                std::span<const float> in_hop;
                if (h < input_hops) {
                    const std::size_t begin = h * hop;
                    const std::size_t avail = std::min(hop, in_samples.size() - begin);
                    if (avail == hop) {
                        in_hop = std::span<const float>(in_samples.data() + begin, hop);
                    } else {
                        std::fill(pad_hop.begin(), pad_hop.end(), 0.0f);
                        std::copy_n(in_samples.data() + begin, avail, pad_hop.begin());
                        in_hop = pad_hop;
                    }
                } else {
                    std::fill(pad_hop.begin(), pad_hop.end(), 0.0f);
                    in_hop = pad_hop;
                    ++tail_hops;
                }
                std::span<float> out_hop = rep == 0 ? std::span<float>(out_samples.data() + h * hop, hop)
                                                    : std::span<float>(scratch_out);

                const std::size_t pos = h * hop;
                while (next_change < opts.changes.size() &&
                       static_cast<double>(pos) >= opts.changes[next_change].seconds * wav_rate) {
                    const ScheduledChange& c = opts.changes[next_change];
                    params.set(c.desc->id, c.value);
                    if (!pipeline.set_params(params)) {
                        std::cerr << "Rejected --change at " << c.seconds << " s\n";
                        return 1;
                    }
                    ++next_change;
                }

                const auto hop_begin = std::chrono::steady_clock::now();
                const PipelineStepResult res = pipeline.process(in_hop, out_hop);
                if (rep == 0 && trace)
                    trace->flush_pending(static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - hop_begin).count()));
                if (!res.ok) {
                    std::cerr << "Inference step failed at repeat " << rep << ", hop " << h << ": "
                              << res.error << '\n';
                    return 1;
                }
                last_sample = out_hop.empty() ? 0.0f : out_hop.back();
                ++h;
            }
            if (rep == 0) out_hops = h;
        }
        pipeline.set_s2ut_observer(nullptr);
        const std::size_t rss_after = resident_bytes();

        out_samples.resize(out_hops * hop);
        if (!write_mono_wav(opts.out_wav, wav_rate, out_samples, false, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        if (trace && trace->failed()) {
            std::cerr << "failed writing trace: " << opts.trace_path.string() << '\n';
            return 1;
        }

        const S2utTelemetry tel = pipeline.s2ut_telemetry();
        const std::int64_t onset = s2ut_onset_latency_samples(enc.contract(), tenc.contract(),
                                                              tdec.contract(), voc.contract(), wait_k);
        const std::int64_t rss_delta_bytes =
            static_cast<std::int64_t>(rss_after) - static_cast<std::int64_t>(rss_before);
        const TimingSnapshot hop_t = pipeline.timing();

        std::printf("ONNX Runtime       : %s\n", ort_runtime_version().c_str());
        std::printf("Bundle / variant   : %s (s%d, hop %zu samples = %.1f ms @ %d Hz)\n",
                    opts.model_dir.string().c_str(), bundle->paths.frames_per_step, hop, hop_ms,
                    enc.contract().sample_rate);
        std::printf("Languages          : %s (%d) -> %s (%d)\n", src->code.c_str(), src->id,
                    tgt->code.c_str(), tgt->id);
        std::printf("Wait-k             : %d%s\n", wait_k,
                    s2ut.policy.adaptive_mode ? " (adaptive)" : "");
        std::printf("Frames per step    : %d\n", bundle->paths.frames_per_step);
        std::printf("Voice              : %s (%s)\n", voice.name.c_str(), voice.display.c_str());
        if (initial_params.voice_b != initial_params.voice_a)
            std::printf("Voice B            : %s\n", bundle->voices[initial_params.voice_b].name.c_str());
        print_param_summary(initial_params, opts.changes);
        std::printf("Output chain       : %s\n",
                    opts.raw ? "off (raw vocoder PCM)" : "gain->gate->brightness->limiter");
        std::printf("Prosody context    : %s\n", opts.prosody_ctx ? "utterance mean" : "off ([0, 0])");
        std::printf("Onset latency      : %lld samples (%.1f ms)\n", static_cast<long long>(onset),
                    1000.0 * static_cast<double>(onset) / enc.contract().sample_rate);
        std::printf("Hops               : %zu (%zu input + %zu padding) per pass x %u pass(es)\n", out_hops,
                    input_hops, out_hops > input_hops ? out_hops - input_hops : 0, opts.repeat);
        std::printf("Chunk steps        : reads=%llu writes=%llu flushes=%llu\n",
                    static_cast<unsigned long long>(tel.reads), static_cast<unsigned long long>(tel.writes),
                    static_cast<unsigned long long>(tel.flushes));
        std::printf("Utterances         : %llu (eos=%llu cap=%llu)\n",
                    static_cast<unsigned long long>(tel.utterances),
                    static_cast<unsigned long long>(tel.eos_ends),
                    static_cast<unsigned long long>(tel.cap_ends));
        std::printf("Model steps        : decoder=%llu vocoder=%llu pad_frames=%llu fifo_stalls=%llu\n",
                    static_cast<unsigned long long>(tel.dec_steps),
                    static_cast<unsigned long long>(tel.voc_steps),
                    static_cast<unsigned long long>(tel.pad_frames),
                    static_cast<unsigned long long>(tel.fifo_stalls));
        std::printf("Stage timing       :\n");
        print_stage("enc", pipeline.stage_timing(PipelineStage::kEncoder));
        print_stage("trans_enc", pipeline.stage_timing(PipelineStage::kTranslatorEnc));
        print_stage("trans_dec", pipeline.stage_timing(PipelineStage::kTranslatorDec));
        print_stage("voc", pipeline.stage_timing(PipelineStage::kVocoder));
        std::printf("Hop timing         : n=%llu  p50=%llu us  p95=%llu us  max=%llu us  misses=%llu\n",
                    static_cast<unsigned long long>(hop_t.samples), static_cast<unsigned long long>(hop_t.p50_us),
                    static_cast<unsigned long long>(hop_t.p95_us), static_cast<unsigned long long>(hop_t.max_us),
                    static_cast<unsigned long long>(hop_t.misses));
        std::printf("RSS (KB)           : before=%zu KB  after=%zu KB  delta=%+lld KB (%.3f MB)\n",
                    rss_before / 1024, rss_after / 1024, static_cast<long long>(rss_delta_bytes / 1024),
                    static_cast<double>(rss_delta_bytes) / (1024.0 * 1024.0));
        if (trace)
            std::printf("Trace              : %s (%zu chunk steps)\n", opts.trace_path.string().c_str(),
                        trace->lines());
        std::printf("Output WAV         : %s (%zu samples, %.2f s)\n", opts.out_wav.string().c_str(),
                    out_samples.size(), static_cast<double>(out_samples.size()) / wav_rate);

        if (opts.have_max_rss_growth &&
            rss_delta_bytes > static_cast<std::int64_t>(opts.max_rss_growth_kb) * 1024) {
            std::cerr << "RSS grew by " << rss_delta_bytes / 1024 << " KB, above --max-rss-growth-kb "
                      << opts.max_rss_growth_kb << '\n';
            return 3;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fv_s2ut_file: " << e.what() << '\n';
        return 1;
    }
}
