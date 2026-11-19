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

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <variant>
#include <vector>

#include "audio/AudioController.hpp"
#include "audio/AudioDeviceRegistry.hpp"
#include "audio/AudioEngine.hpp"
#include "audio/AudioTypes.hpp"
#include "audio/ChunkAssembler.hpp"
#include "audio/MiniaudioBackend.hpp"
#include "audio/RouteGuard.hpp"
#include "pipeline/HostStats.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/LangCatalog.hpp"
#include "pipeline/LatencyGovernor.hpp"
#include "pipeline/ModeAHost.hpp"
#include "pipeline/ModelBundle.hpp"  // kNoTranslatorModelsMessage
#include "pipeline/ModelContract.hpp"
#include "pipeline/OrtModel.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "util/ProcessMemory.hpp"
#include "util/WorkerPriority.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <readerwriterqueue.h>

namespace {

enum class Mode { Passthrough, Queue, Chunked, VC, S2UT };

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
        "FluxVoice-S2UT — real-time textless voice changer / speech translator (Mode A / B)\n"
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
        "  --vc                          Voice-conversion mode (Mode A; requires ONNX Runtime)\n"
        "  --s2ut                        Simultaneous speech-to-unit translation (Mode B; requires\n"
        "                                ONNX Runtime and a bundle with translator models + langs.json)\n"
        "  --src-lang <code|id>          Source language from langs.json (requires --s2ut)\n"
        "  --tgt-lang <code|id>          Target language from langs.json (requires --s2ut; must differ\n"
        "                                from --src-lang: use --vc for same-language conversion)\n"
        "  --wait-k <1..10>              Wait-k in 80 ms chunks (requires --s2ut; default: the\n"
        "                                pair's default_wait_k in langs.json, else 3)\n"
        "  --list-langs                  With --model-dir <bundle>: print the bundle languages and\n"
        "                                default Wait-k per pair, then exit (no audio/ORT needed)\n"
        "  --model-dir <bundle>          Encoder/vocoder bundle and voices.json (requires --vc/--s2ut;\n"
        "                                with --list-params --json also reports langs.json)\n"
        "                                A --vc bundle that also holds translator models + langs.json\n"
        "                                starts as Mode A and can switch to Mode B from the console\n"
        "  --voice <name>                Target voice (requires --vc/--s2ut; default: first bundle voice)\n"
        "  --voice-b <name>              Blend endpoint voice B (requires --vc/--s2ut; follows --voice)\n"
        "  --preset <name|path>          Voice preset: name searched in --preset-dir, exe/presets,\n"
        "                                source presets/ and cwd/presets, or a .json path\n"
        "                                (requires --vc/--s2ut)\n"
        "  --param <name=value>          Set a voice parameter, repeatable (requires --vc/--s2ut;\n"
        "                                names in --list-params)\n"
        "  --preset-dir <dir>            User preset directory; also where console 'save' writes\n"
        "  --list-params [--json]        Print the voice-parameter table and exit\n"
        "  --list-presets                List the presets found in the search dirs and exit\n"
        "  --console                     Read console commands from stdin even under --duration;\n"
        "                                with --duration, stdin EOF stops reading commands but\n"
        "                                does not end the run (without it, EOF still stops).\n"
        "                                Commands: set <param> <value> | get [param] | voices |\n"
        "                                preset <name> | save <name> | mode vc|s2ut |\n"
        "                                lang <src> <tgt> | waitk <1..10> | langs | help\n"
        "  --frames-per-step <1|2|4>      Hop variant (default: smallest available for --vc, even for\n"
        "                                bundles with translator models (Mode A latency first); 4 for\n"
        "                                --s2ut. Pass --frames-per-step 4 with --vc when planning to\n"
        "                                switch to Mode B: at s1 every model runs in one 20 ms hop)\n"
        "  --latency-target-ms <ms>       Queued-audio target (default: model budget)\n"
        "  --list-voices                 Validate/list bundle voices and exit (no audio/ORT needed;\n"
        "                                requires --vc or --s2ut)\n"
        "  --input  <index|default>      Capture device: index from --list-devices, or\n"
        "                                \"default\" to follow the OS default endpoint\n"
        "  --output <index|default>      Playback device: index or \"default\" (same rules)\n"
        "  --loopback-input <index|default>\n"
        "                                System-playback capture: playback device index or\n"
        "                                \"default\" (WASAPI native loopback; mutually\n"
        "                                exclusive with --input)\n"
        "  --sample-rate <hz>            Sample rate (default 16000)\n"
        "  --period-ms <5..40>            Requested device period (default 20 ms)\n"
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

void print_stream_info(const StreamInfo& si, const fluxvoice::StreamStatus& status,
                       std::uint32_t requested_period_ms) {
    std::cout << "Backend : " << si.backend << '\n'
              << "Capture : " << si.capture_name
              << "  (backend " << si.capture_internal_rate << " Hz, period "
              << si.capture_internal_period << " frames)\n"
              << "Playback: " << si.playback_name
              << "  (backend " << si.playback_internal_rate << " Hz, period "
              << si.playback_internal_period << " frames)\n"
              << "Client  : " << si.client_sample_rate << " Hz mono f32, requested period "
              << requested_period_ms << " ms\n";
    const auto period_ms = [](std::uint32_t frames, std::uint32_t rate) {
        return rate ? 1000.0 * frames / rate : 0.0;
    };
    std::cout << "Periods : capture="
              << period_ms(si.capture_internal_period, si.capture_internal_rate)
              << " ms, playback="
              << period_ms(si.playback_internal_period, si.playback_internal_rate) << " ms\n";
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

/// The diagnostic modes' counters (passthrough / queue / chunked): the router's
/// and the application's, with no inference telemetry (`vc` stays false). The
/// neural modes use ModeAHost::stats() instead.
fluxvoice::HostStats legacy_stats(const DuplexRouter& router, const std::atomic<uint64_t>& drops) {
    fluxvoice::HostStats s;
    s.frames    = router.frames_processed();
    s.overruns  = router.input_overruns();
    s.underruns = router.output_underruns();
    s.drops     = drops.load(std::memory_order_relaxed);
    // These modes insert no startup silence and skip nothing (skipped = prefilled = 0).
    s.queued    = fluxvoice::queued_samples(s.underruns, s.overruns, s.drops,
                                           s.skipped, s.prefilled);
    s.rss       = fluxvoice::resident_bytes();
    return s;
}

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Passthrough: return "callback passthrough";
        case Mode::Queue:       return "SPSC queue loopback";
        case Mode::Chunked:     return "SPSC queue loopback, 80 ms chunked";
        case Mode::VC:          return "Mode A voice conversion";
        case Mode::S2UT:        return "Mode B simultaneous translation (s2ut)";
    }
    return "?";
}

// ---- Voice setup shared by --vc and --s2ut ---------------------------------

/// Index of `name` in `voices` (diagnostic on stderr when unknown).
bool voice_index_of(const std::vector<fluxvoice::VoiceEntry>& voices, std::string_view name,
                    std::uint16_t& out) {
    std::string err;
    const fluxvoice::VoiceEntry* v = fluxvoice::find_voice(voices, name, err);
    if (!v) {
        std::cerr << err << '\n';
        return false;
    }
    out = static_cast<std::uint16_t>(v - voices.data());
    return true;
}

/// Order: descriptor defaults (or the caller's pre-seeded `params`) -> preset
/// params + voice names -> --voice / --voice-b -> --param assignments in
/// command-line order.
bool build_initial_params(const std::vector<fluxvoice::VoiceEntry>& voices,
                          const fluxvoice::PresetLoadResult& preset_res,
                          const std::string& voice_name, const std::string& voice_b_name,
                          const std::vector<std::pair<const fluxvoice::ParamDesc*, float>>& assignments,
                          fluxvoice::VoiceParams& params, bool& voice_b_follows_a) {
    if (preset_res.ok()) {
        fluxvoice::apply_preset_params(*preset_res.preset, params);
        if (preset_res.preset->voice_a &&
            !voice_index_of(voices, *preset_res.preset->voice_a, params.voice_a))
            return false;
        if (preset_res.preset->voice_b &&
            !voice_index_of(voices, *preset_res.preset->voice_b, params.voice_b))
            return false;
        if (preset_res.preset->voice_b) voice_b_follows_a = false;
    }
    if (!voice_name.empty() && !voice_index_of(voices, voice_name, params.voice_a)) return false;
    if (!voice_b_name.empty()) {
        if (!voice_index_of(voices, voice_b_name, params.voice_b)) return false;
        voice_b_follows_a = false;
    }
    for (const auto& [d, v] : assignments) params.set(d->id, v);
    if (voice_b_follows_a) params.voice_b = params.voice_a;
    return true;
}

// ---- Translation controls (M3 Stage 08 CLI / Stage 09 console) -------------

/// "en es fr de"
std::string lang_codes(const fluxvoice::LangCatalog& c) {
    std::string out;
    for (const auto& l : c.langs) out += (out.empty() ? "" : " ") + l.code;
    return out;
}

int wait_k_of(const fluxvoice::VoiceParams& p) {
    return static_cast<int>(std::lround(p.get(fluxvoice::ParamId::WaitK)));
}

/// "es -> en (k=3, mode s2ut)"
std::string pair_status(const fluxvoice::LangCatalog& c, const fluxvoice::VoiceParams& p) {
    const auto* s = c.find(std::to_string(p.src_lang_id));
    const auto* t = c.find(std::to_string(p.tgt_lang_id));
    return (s ? s->code : std::string("?")) + " -> " + (t ? t->code : std::string("?")) +
           " (k=" + std::to_string(wait_k_of(p)) + ", mode " +
           (p.enabled(fluxvoice::ParamId::Translate) ? "s2ut" : "vc") + ")";
}

/// Parse a Wait-k console/CLI argument: an integer in [1, 10].
bool parse_wait_k(std::string_view text, int& out) {
    std::uint32_t v = 0;
    if (!parse_uint(text, 1, 10, v)) return false;
    out = static_cast<int>(v);
    return true;
}

enum class PairChange { Applied, UnknownLanguage, SameLanguage };

/// `lang <src> <tgt>`: validate against the catalog and, when valid, set the
/// ids and reset wait_k to the pair default (AppModel::set_language_pair
/// behaviour). Nothing is modified unless Applied; `message` is always set.
PairChange change_language_pair(const fluxvoice::LangCatalog& c, fluxvoice::VoiceParams& p,
                                std::string_view src, std::string_view tgt, std::string& message) {
    const fluxvoice::LangEntry* s = c.find(src);
    const fluxvoice::LangEntry* t = c.find(tgt);
    if (!s || !t) {
        message = "unknown language '" + std::string(!s ? src : tgt) +
                  "' (bundle languages: " + lang_codes(c) + ")";
        return PairChange::UnknownLanguage;
    }
    if (s->id == t->id) {
        message = "source and target are both '" + s->code +
                  "': use 'mode vc' for same-language voice conversion "
                  "(Mode A gives ~60-100 ms instead of ~330 ms); language pair unchanged";
        return PairChange::SameLanguage;
    }
    p.src_lang_id = static_cast<std::uint16_t>(s->id);
    p.tgt_lang_id = static_cast<std::uint16_t>(t->id);
    p.set(fluxvoice::ParamId::WaitK, static_cast<float>(c.default_k(s->code, t->code)));
    message = pair_status(c, p);
    return PairChange::Applied;
}

void print_langs(std::ostream& os, const fluxvoice::LangCatalog& c) {
    for (const auto& l : c.langs) os << "  " << l.code << "  " << l.id << "  " << l.name << '\n';
}

void print_default_wait_k(std::ostream& os, const fluxvoice::LangCatalog& c) {
    os << "Default wait-k (80 ms chunks) per pair (others: " << fluxvoice::kDefaultWaitK << "):\n";
    for (const auto& d : c.default_wait_k) os << "  " << d.src << "->" << d.tgt << "  " << d.wait_k << '\n';
}

void print_s2ut_summary(const fluxvoice::HostStats& s) {
    std::cout << "S2UT: decisions read=" << s.tel.reads << " write=" << s.tel.writes
              << " flush=" << s.tel.flushes << ", utterances=" << s.tel.utterances
              << " (eos=" << s.tel.eos_ends << ", cap=" << s.tel.cap_ends << ")"
              << ", decoder steps=" << s.tel.dec_steps << ", vocoder steps=" << s.tel.voc_steps
              << ", pad frames=" << s.tel.pad_frames << ", fifo stalls=" << s.tel.fifo_stalls
              << ", lag=" << s.tel.lag_ms() << " ms\n";
    for (std::size_t i = 0; i < s.stages.size(); ++i) {
        const auto& t = s.stages[i];
        std::cout << "  " << fluxvoice::to_string(static_cast<fluxvoice::PipelineStage>(i))
                  << ": n=" << t.samples << " p50/p95/max = " << t.p50_us << "/" << t.p95_us << "/"
                  << t.max_us << " us\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    bool opt_list = false;
    bool list_voices = false, vc_options = false, explicit_rate = false, selected_mode = false;
    bool list_params = false, list_presets = false, opt_json = false, opt_console = false;
    bool list_langs = false, s2ut_options = false;
    Mode mode = Mode::Queue;
    std::string model_dir, voice_name, voice_b_name, preset_arg, preset_dir;
    std::string src_lang_arg, tgt_lang_arg;
    std::uint32_t wait_k_arg = 0;  // 0 = pair default
    std::vector<std::string> param_args;
    std::uint32_t frames_per_step = 0, latency_target_ms = 0;
    int opt_input = -1, opt_output = -1, opt_loopback = -1;
    bool has_input_arg = false, has_loopback_arg = false;
    uint32_t sample_rate = AudioEngine::kDefaultSampleRate;
    std::uint32_t period_ms = AudioEngine::kDefaultPeriodMs;
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
        auto text = [&](std::string& out) {
            if (i + 1 >= argc || argv[i + 1][0] == '\0' || argv[i + 1][0] == '-') {
                std::cerr << "Missing value for " << a << '\n';
                return false;
            }
            out = argv[++i];
            vc_options = true;
            return true;
        };
        auto select_mode = [&](Mode choice) {
            if (selected_mode && choice != mode) {
                std::cerr << "Audio mode flags are mutually exclusive\n";
                return false;
            }
            mode = choice;
            selected_mode = true;
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
        else if (a == "--passthrough")      { if (!select_mode(Mode::Passthrough)) return 1; }
        else if (a == "--queue-loopback")   { if (!select_mode(Mode::Queue)) return 1; }
        else if (a == "--chunked")          { if (!select_mode(Mode::Chunked)) return 1; }
        else if (a == "--vc")               { if (!select_mode(Mode::VC)) return 1; }
        else if (a == "--s2ut")             { if (!select_mode(Mode::S2UT)) return 1; }
        else if (a == "--src-lang")         { s2ut_options = true; if (!text(src_lang_arg)) return 1; }
        else if (a == "--tgt-lang")         { s2ut_options = true; if (!text(tgt_lang_arg)) return 1; }
        else if (a == "--wait-k")           { s2ut_options = true; if (!value(1, 10, wait_k_arg)) return 1; }
        else if (a == "--list-langs")       { list_langs = true; }
        else if (a == "--model-dir")        { if (!text(model_dir)) return 1; }
        else if (a == "--voice")            { if (!text(voice_name)) return 1; }
        else if (a == "--voice-b")          { if (!text(voice_b_name)) return 1; }
        else if (a == "--preset")           { if (!text(preset_arg)) return 1; }
        else if (a == "--param") {
            std::string v;
            if (!text(v)) return 1;
            param_args.push_back(std::move(v));
        }
        else if (a == "--preset-dir") {
            if (i + 1 >= argc || argv[i + 1][0] == '\0' || argv[i + 1][0] == '-') {
                std::cerr << "Missing value for " << a << '\n';
                return 1;
            }
            preset_dir = argv[++i];
        }
        else if (a == "--list-params")      { list_params = true; }
        else if (a == "--list-presets")     { list_presets = true; }
        else if (a == "--json")             { opt_json = true; }
        else if (a == "--console")          { opt_console = true; }
        else if (a == "--frames-per-step") {
            vc_options = true;
            if (!value(1, 4, frames_per_step)) return 1;
            if (frames_per_step == 3) {
                std::cerr << "--frames-per-step must be 1, 2 or 4\n";
                return 1;
            }
        }
        else if (a == "--latency-target-ms") {
            vc_options = true;
            if (!value(1, 10000, latency_target_ms)) return 1;
        }
        else if (a == "--list-voices")      { list_voices = vc_options = true; }
        else if (a == "--input")            { if (!endpoint(opt_input)) return 1; has_input_arg = true; }
        else if (a == "--output")           { if (!endpoint(opt_output)) return 1; }
        else if (a == "--loopback-input")   { if (!endpoint(opt_loopback)) return 1; has_loopback_arg = true; }
        else if (a == "--sample-rate")      { if (!value(8000, 192000, sample_rate)) return 1; explicit_rate = true; }
        else if (a == "--period-ms")        { if (!value(5, 40, period_ms)) return 1; }
        else if (a == "--duration")         { if (!value(1, 7 * 24 * 3600, duration_s)) return 1; }
        else if (a == "--stats")            { if (!value(1, 24 * 3600, stats_s)) return 1; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown option: " << a << '\n'; print_usage(argv[0]); return 1; }
    }

    if (has_input_arg && has_loopback_arg) {
        std::cerr << "--input and --loopback-input are mutually exclusive\n";
        return 1;
    }
    if (opt_json && !list_params) {
        std::cerr << "--json requires --list-params\n";
        return 1;
    }

    // Standalone information commands: no --vc, no ORT, no audio device.
    std::error_code exe_ec;
    const std::filesystem::path exe_dir =
        std::filesystem::weakly_canonical(std::filesystem::path(argv[0]), exe_ec).parent_path();
    const auto preset_dirs = fluxvoice::preset_search_dirs(preset_dir, exe_dir);
    if (list_params) {
        if (opt_json && !model_dir.empty() &&
            std::filesystem::is_regular_file(std::filesystem::path(model_dir) / "langs.json", exe_ec)) {
            fluxvoice::LangCatalog catalog;
            std::string error;
            if (!fluxvoice::read_lang_catalog(std::filesystem::path(model_dir) / "langs.json",
                                              nullptr, catalog, error)) {
                std::cerr << error << '\n';
                return 1;
            }
            std::cout << fluxvoice::descriptor_table_json(&catalog);
            return 0;
        }
        std::cout << (opt_json ? fluxvoice::descriptor_table_json()
                               : fluxvoice::params_table_text());
        return 0;
    }
    if (list_langs) {
        if (model_dir.empty()) {
            std::cerr << "--list-langs requires --model-dir <bundle>\n";
            return 1;
        }
        fluxvoice::LangCatalog catalog;
        std::string error;
        if (!fluxvoice::read_lang_catalog(std::filesystem::path(model_dir) / "langs.json", nullptr,
                                          catalog, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        std::cout << "Languages (code  id  name):\n";
        print_langs(std::cout, catalog);
        print_default_wait_k(std::cout, catalog);
        return 0;
    }
    if (list_presets) {
        for (const auto& dir : preset_dirs) {
            for (const auto& file : fluxvoice::list_preset_files(dir)) {
                auto r = fluxvoice::load_preset_file(file);
                for (const std::string& w : r.warnings)
                    std::cerr << file.filename().string() << ": " << w << '\n';
                if (!r.ok()) {
                    std::cerr << file.filename().string() << ": " << r.error << '\n';
                    continue;
                }
                std::cout << r.preset->name << "  (" << file.string() << ")\n";
            }
        }
        return 0;
    }

    const bool neural = mode == Mode::VC || mode == Mode::S2UT;
    const char* const mode_flag = mode == Mode::S2UT ? "--s2ut" : "--vc";
    if (s2ut_options && mode != Mode::S2UT) {
        std::cerr << "--src-lang, --tgt-lang and --wait-k require --s2ut\n";
        return 1;
    }
    if (vc_options && !neural) {
        std::cerr << "Voice-conversion options require --vc or --s2ut\n";
        return 1;
    }
    if (neural && (has_loopback_arg || opt_list)) {
        std::cerr << mode_flag
                  << " requires ordinary duplex capture, not --loopback-input/--list-devices\n";
        return 1;
    }
    // Validate --preset and every --param BEFORE the ORT-availability check so
    // ORT-less builds exercise the same rejections.
    fluxvoice::PresetLoadResult preset_res;
    std::vector<std::pair<const fluxvoice::ParamDesc*, float>> cli_assignments;
    if (neural) {
        if (!preset_arg.empty()) {
            preset_res = fluxvoice::resolve_preset(preset_arg, preset_dirs);
            if (!preset_res.ok()) {
                std::cerr << preset_res.error << '\n';
                return 1;
            }
            for (const std::string& w : preset_res.warnings) std::cerr << "preset: " << w << '\n';
        }
        for (const std::string& assignment : param_args) {
            const fluxvoice::ParamDesc* d = nullptr;
            float v = 0.0f;
            std::string warning, error;
            if (!fluxvoice::parse_param_assignment(assignment, d, v, warning, error)) {
                std::cerr << "--param: " << error << '\n';
                return 1;
            }
            if (!warning.empty()) std::cerr << "--param: " << warning << '\n';
            cli_assignments.push_back({d, v});
        }
    }
    // --s2ut: every language/wait-k rejection happens here, before the ORT
    // guard, any model session or any audio device (M3 Stage 08).
    fluxvoice::LangCatalog langs;
    const fluxvoice::LangEntry* src_entry = nullptr;
    const fluxvoice::LangEntry* tgt_entry = nullptr;
    int wait_k = fluxvoice::kDefaultWaitK;
    if (mode == Mode::S2UT && !list_voices) {
        if (model_dir.empty()) {
            std::cerr << "--s2ut requires --model-dir <bundle>\n";
            return 1;
        }
        if (src_lang_arg.empty() || tgt_lang_arg.empty()) {
            std::cerr << "--s2ut requires --src-lang <code|id> and --tgt-lang <code|id>"
                      << " (see --list-langs --model-dir <bundle>)\n";
            return 1;
        }
        std::string error;
        if (!fluxvoice::read_lang_catalog(std::filesystem::path(model_dir) / "langs.json", nullptr,
                                          langs, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        src_entry = langs.find(src_lang_arg);
        if (!src_entry) {
            std::cerr << "unknown --src-lang '" << src_lang_arg << "' (bundle languages: "
                      << lang_codes(langs) << ")\n";
            return 1;
        }
        tgt_entry = langs.find(tgt_lang_arg);
        if (!tgt_entry) {
            std::cerr << "unknown --tgt-lang '" << tgt_lang_arg << "' (bundle languages: "
                      << lang_codes(langs) << ")\n";
            return 1;
        }
        if (src_entry->id == tgt_entry->id) {
            std::cerr << "--src-lang and --tgt-lang are both '" << src_entry->code
                      << "': translating a language into itself is voice conversion; use --vc\n";
            return 1;
        }
        wait_k = wait_k_arg != 0 ? static_cast<int>(wait_k_arg)
                                 : langs.default_k(src_entry->code, tgt_entry->code);
    }
    if (neural && !list_voices && !fluxvoice::ort_available()) {
        std::cerr << mode_flag << ": " << fluxvoice::kBuiltWithoutOrtMessage << '\n';
        return 1;
    }
    if (neural && model_dir.empty()) {
        std::cerr << mode_flag << " requires --model-dir <bundle>\n";
        return 1;
    }
    // The one live composition (models, pipeline, param channel, worker); the
    // GUI's EngineHost uses the same class. Declared before the controller and
    // detached (DetachGuard below) before the controller is destroyed.
    fluxvoice::ModeAHost host;
    // Control-side snapshot mirrored by the console (Stage 09). `voice_b`
    // follows `voice_a` until chosen explicitly by preset or --voice-b.
    fluxvoice::VoiceParams console_params;
    const std::vector<fluxvoice::VoiceEntry>* console_voices = nullptr;
    // Non-null exactly when translator models are loaded (Mode B capable).
    const fluxvoice::LangCatalog* console_langs = nullptr;
    bool voice_b_follows_a = true;
    // Console text for translation commands when no Mode B quartet is loaded.
    std::string translation_msg = fluxvoice::kNoTranslatorModelsMessage;
    if (neural) {
        if (list_voices) {
            try {
                std::string error;
                std::vector<fluxvoice::VoiceEntry> voices;
                if (!fluxvoice::read_voice_catalog(
                        std::filesystem::path(model_dir) / "voices.json", std::nullopt, voices, error)) {
                    std::cerr << error << '\n';
                    return 1;
                }
                for (const auto& voice : voices)
                    std::cout << voice.name << "  " << voice.display << "  [" << voice.license << "]\n";
                return 0;
            } catch (const std::exception& e) {
                std::cerr << "Cannot prepare " << (mode == Mode::S2UT ? "Mode B" : "Mode A") << ": "
                          << e.what() << '\n';
                return 1;
            }
        }

        // The CLI's preset / language / --param logic. ModeAHost calls this once
        // per composition attempt (twice when a --vc translator quartet is
        // unusable and the host retries as plain Mode A), after the models are
        // in and before the pipeline is built. Diagnostics go to stderr here, in
        // the order the pre-host main() printed them; returning false aborts the
        // load (the message was already printed, so the error text stays empty).
        const auto seed_params = [&](const fluxvoice::HostBundleInfo& info,
                                     fluxvoice::VoiceParams& params,
                                     fluxvoice::S2utOptions& s2ut_opts, std::string&) -> bool {
            if (!info.translator_fallback.empty()) {
                // Mode A must not depend on a valid Mode B quartet: the host
                // retries with two models; forget everything the first try set.
                std::cerr << "translator models present but unusable for live switching: "
                          << info.translator_fallback << "; running Mode A only\n";
                src_entry = tgt_entry = nullptr;
                wait_k = fluxvoice::kDefaultWaitK;
                voice_b_follows_a = true;
                translation_msg = "translator models are unusable for live switching (see the startup warning)";
            }
            if (!info.translator_unavailable.empty()) {
                // One language cannot be translated: plain Mode A bundle.
                translation_msg = std::string(info.translator_unavailable);
                std::cerr << "Note: " << translation_msg << "; running Mode A only\n";
            }
            const std::vector<fluxvoice::VoiceEntry>& voices = *info.voices;
            if (info.langs) {
                if (mode == Mode::VC) {
                    // Start as Mode A on en -> es (else the first two languages);
                    // a valid --preset src_lang/tgt_lang pair wins.
                    const auto& l = *info.langs;
                    const fluxvoice::LangEntry* en = l.find("en");
                    const fluxvoice::LangEntry* es = l.find("es");
                    src_entry = en && es ? en : &l.langs[0];
                    tgt_entry = en && es ? es : &l.langs[1];
                    if (preset_res.ok() && (preset_res.preset->src_lang || preset_res.preset->tgt_lang)) {
                        fluxvoice::VoiceParams probe;
                        std::string message;
                        const std::string s_code = preset_res.preset->src_lang ? *preset_res.preset->src_lang
                                                                               : src_entry->code;
                        const std::string t_code = preset_res.preset->tgt_lang ? *preset_res.preset->tgt_lang
                                                                               : tgt_entry->code;
                        if (change_language_pair(l, probe, s_code, t_code, message) == PairChange::Applied) {
                            src_entry = l.find(s_code);
                            tgt_entry = l.find(t_code);
                        } else {
                            std::cerr << "preset: languages skipped: " << message << '\n';
                        }
                    }
                    wait_k = l.default_k(src_entry->code, tgt_entry->code);
                } else if (preset_res.ok()) {
                    // CLI --src-lang/--tgt-lang win over the preset's pair.
                    for (const auto* pl : {&preset_res.preset->src_lang, &preset_res.preset->tgt_lang}) {
                        if (!*pl) continue;
                        const fluxvoice::LangEntry* e = info.langs->find(**pl);
                        if (!e || (e->id != src_entry->id && e->id != tgt_entry->id)) {
                            std::cerr << "preset: src_lang/tgt_lang ignored under --s2ut; using --src-lang "
                                      << src_entry->code << " --tgt-lang " << tgt_entry->code << '\n';
                            break;
                        }
                    }
                }
            }
            if (src_entry && tgt_entry) {
                // Seeded before presets/--param so those may override wait_k;
                // --wait-k below wins over everything.
                params.src_lang_id = static_cast<std::uint16_t>(src_entry->id);
                params.tgt_lang_id = static_cast<std::uint16_t>(tgt_entry->id);
                params.set(fluxvoice::ParamId::WaitK, static_cast<float>(wait_k));
            }
            if (!build_initial_params(voices, preset_res, voice_name, voice_b_name, cli_assignments,
                                      params, voice_b_follows_a))
                return false;
            if (wait_k_arg != 0) params.set(fluxvoice::ParamId::WaitK, static_cast<float>(wait_k_arg));
            if (mode == Mode::VC && params.enabled(fluxvoice::ParamId::Translate)) {
                std::cerr << "preset/--param: translate ignored under --vc (use --s2ut or the "
                             "console 'mode s2ut')\n";
                params.set(fluxvoice::ParamId::Translate, 0.0f);
            }
            if (mode == Mode::S2UT) params.set(fluxvoice::ParamId::Translate, 1.0f);
            console_params = params;
            if (explicit_rate && sample_rate != info.contract_rate) {
                std::cerr << "--sample-rate must equal the model contract rate (" << info.contract_rate << ")\n";
                return false;
            }
            sample_rate = info.contract_rate;
            if (info.langs) {
                s2ut_opts.policy.k = wait_k_of(params);
                s2ut_opts.policy.adaptive_mode = params.enabled(fluxvoice::ParamId::AdaptivePolicy);
                s2ut_opts.src_lang_id = params.src_lang_id;
                s2ut_opts.tgt_lang_id = params.tgt_lang_id;
                s2ut_opts.translate = (mode == Mode::S2UT);
            }
            return true;
        };

        // Mode B runs Stage 1 + 3 at s4 by default (the host applies that for
        // kRequired). A --vc run keeps the M2 default (smallest available
        // variant: Mode A latency comes first; pass --frames-per-step 4 when
        // planning to switch to Mode B) and also loads the translator models
        // when the bundle has them (kIfPresent), so the console can switch to
        // Mode B live at that same hop.
        fluxvoice::HostLoadRequest request;
        request.model_dir = model_dir;
        request.frames_per_step = static_cast<int>(frames_per_step);
        request.latency_target_ms = latency_target_ms;
        request.translator = mode == Mode::S2UT ? fluxvoice::TranslatorPolicy::kRequired
                                                : fluxvoice::TranslatorPolicy::kIfPresent;
        request.warm_up_steps = 20;
        fluxvoice::HostError load_error;
        if (!host.load(request, seed_params, load_error)) {
            if (!load_error.message.empty()) std::cerr << load_error.message << '\n';
            return 1;
        }
        console_voices = &host.voices();
        console_langs = host.langs();
        sample_rate = host.contract_rate();
    }

    MiniaudioBackend backend;
    fluxvoice::ControllerConfig ctrl_cfg;
    ctrl_cfg.sample_rate = sample_rate;
    ctrl_cfg.period_ms   = period_ms;
    ctrl_cfg.passthrough = (mode == Mode::Passthrough && !has_loopback_arg);
    ctrl_cfg.fade_underruns = neural;
    if (neural) host.configure(ctrl_cfg);  // contract rate, fade_underruns, governor owns skips

    ChunkAssembler chunks(fluxvoice::PipelineTiming::kChunkSamples,
                          fluxvoice::PipelineTiming::kContextSamples);
    fluxvoice::InferencePipeline pipeline;

    std::atomic<bool> worker_run{false};
    std::atomic<uint64_t> output_drops{0};
    std::array<float, 4096> buf{};
    std::thread worker;  // diagnostic modes only; the neural worker belongs to the host

    fluxvoice::AudioController controller(backend, ctrl_cfg);
    // Declared after the controller so it runs first on every exit path: the
    // host's hooks must be cleared while the controller is still alive.
    struct DetachGuard {
        fluxvoice::ModeAHost& host;
        ~DetachGuard() { host.detach(); }
    } detach_guard{host};
    if (neural) {
        // Creates the worker, the prepare hook (client rate == contract rate,
        // then prepare()) and the quiesce / reset / resume hooks.
        fluxvoice::HostError attach_error;
        if (!host.attach(controller, attach_error,
                         [](const char* method, bool elevated, int result) {
                             std::cerr << "[worker] priority=" << method
                                       << " elevated=" << (elevated ? "yes" : "no")
                                       << " result=" << result << '\n';
                         })) {
            std::cerr << attach_error.message << '\n';
            return 1;
        }
    }
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
            const fluxvoice::WorkerPriority priority;
            std::cerr << "[worker] priority=" << priority.method()
                      << " elevated=" << (priority.elevated() ? "yes" : "no")
                      << " result=" << priority.result() << '\n';
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

    // The neural worker's hooks were installed by host.attach().
    if (!neural) controller.set_worker_hooks(stop_worker, reset_dsp, start_worker);

    auto start_res = controller.open_and_start(in_sel, out_sel, capture_mode);
    if (!fluxvoice::is_ok(start_res)) {
        print_error(std::get<fluxvoice::AudioError>(start_res), "Failed to start audio stream");
        return 1;
    }

    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);

    print_stream_info(controller.stream_info(), controller.status(), period_ms);
    if (neural)
        std::cout << "Pipeline: s" << host.geometry().frames_per_step
                  << ", hop=" << host.geometry().hop
                  << " samples, algorithmic=" << host.geometry().algorithmic_samples()
                  << " samples, target=" << host.governor_target_samples()
                  << " samples, voice="
                  << console_voices->at(console_params.voice_a).name
                  << (console_params.voice_b != console_params.voice_a
                          ? ", voice_b=" + console_voices->at(console_params.voice_b).name
                          : "")
                  << '\n';
    if (console_langs) {
        const fluxvoice::VoiceParams& p0 = console_params;
        const std::int64_t onset = host.onset_latency_samples(wait_k_of(p0));
        std::cout << (mode == Mode::S2UT ? "Mode B: " : "Translator models loaded (starting as Mode A): ")
                  << pair_status(*console_langs, p0)
                  << ", wait-k=" << wait_k_of(p0)
                  << ", frames_per_step=" << host.geometry().frames_per_step
                  << ", algorithmic onset latency="
                  << 1000.0 * static_cast<double>(onset) / static_cast<double>(sample_rate)
                  << " ms\n";
    }
    std::cout << "FluxVoice running (" << mode_name(mode) << "). ";
    if (duration_s > 0) std::cout << "Stopping after " << duration_s << " s; Ctrl+C to stop early.\n";
    else                std::cout << "Press Enter or Ctrl+C to stop.\n";
    std::cout.flush();

    // Console input (M2 / Stage 09): a detached line-reader posts stdin lines
    // to an intentionally leaked SPSC queue — the reader may outlive main's
    // locals, and this is not a real-time thread so std::string allocation is
    // fine. Only this control loop drains it and prints.
    auto* console_lines = new moodycamel::ReaderWriterQueue<std::string>(64);
    if (duration_s == 0 || opt_console) {
        const bool eof_stops = duration_s == 0;
        std::thread([console_lines, eof_stops] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                console_lines->enqueue(std::move(line));
            }
            // stdin EOF: with --duration + --console just stop reading;
            // otherwise EOF ends the run like Enter did before.
            if (eof_stops) g_stop.store(true);
        }).detach();
    }

    const auto slugify = [](std::string_view name) {
        std::string slug;
        slug.reserve(name.size());
        for (char c : name)
            slug.push_back((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ? c
                           : (c >= 'A' && c <= 'Z')                        ? static_cast<char>(c - 'A' + 'a')
                                                                          : '-');
        while (!slug.empty() && slug.front() == '-') slug.erase(slug.begin());
        while (!slug.empty() && slug.back() == '-') slug.pop_back();
        return slug;
    };
    const auto voice_name_of = [&](std::uint16_t index) -> std::string_view {
        return console_voices && index < console_voices->size()
                   ? std::string_view(console_voices->at(index).name)
                   : std::string_view("?");
    };
    const auto print_param = [&](const fluxvoice::ParamDesc& d) {
        std::cout << d.name << " = " << fluxvoice::format_param_value(d, console_params.get(d.id))
                  << (d.unit.empty() ? "" : " ") << d.unit << '\n';
    };
    const auto handle_line = [&](const std::string& line) {
        if (line.empty()) {
            g_stop.store(true);
            return;
        }
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        const bool vc = console_voices != nullptr;
        if (cmd == "help") {
            std::cout << "Commands: set <param> <value> | set voice_a|voice_b <name>\n"
                         "          get [param] | preset <name> | save <name> | voices | help\n"
                         "          mode vc|s2ut         switch Mode A / Mode B (translator bundle)\n"
                         "          lang <src> <tgt>     language pair (codes or ids; resets wait_k)\n"
                         "          waitk <1..10>        shortcut for 'set wait_k'\n"
                         "          langs                list languages and the current pair\n"
                         "          (empty line stops)\n";
            return;
        }
        if (cmd != "set" && cmd != "get" && cmd != "preset" && cmd != "save" && cmd != "voices" &&
            cmd != "mode" && cmd != "lang" && cmd != "waitk" && cmd != "langs") {
            std::cout << "unknown command '" << cmd << "' (try 'help')\n";
            return;
        }
        if (!vc) {
            std::cout << cmd << ": voice controls need --vc or --s2ut\n";
            return;
        }
        if (cmd == "set") {
            std::string key, value;
            ss >> key >> value;
            if (key.empty() || value.empty()) {
                std::cout << "usage: set <param> <value>\n";
                return;
            }
            if (key == "voice_a" || key == "voice_b") {
                std::string err;
                const fluxvoice::VoiceEntry* v = fluxvoice::find_voice(*console_voices, value, err);
                if (!v) {
                    std::cout << err << '\n';
                    return;
                }
                const auto idx = static_cast<std::uint16_t>(v - console_voices->data());
                if (key == "voice_a") {
                    console_params.voice_a = idx;
                    if (voice_b_follows_a) console_params.voice_b = idx;
                } else {
                    console_params.voice_b = idx;
                    voice_b_follows_a = false;
                }
                host.push_params(console_params);
                std::cout << key << " = " << v->name << '\n';
                return;
            }
            const fluxvoice::ParamDesc* d = nullptr;
            float v = 0.0f;
            std::string warning, error;
            if (!fluxvoice::parse_param_assignment(key + "=" + value, d, v, warning, error)) {
                std::cout << "set: " << error << '\n';
                return;
            }
            if (!warning.empty()) std::cout << "set: " << warning << '\n';
            if (d->group == fluxvoice::ParamGroup::Translation && !console_langs) {
                std::cout << "set: " << translation_msg << '\n';
                return;
            }
            console_params.set(d->id, v);
            host.push_params(console_params);
            print_param(*d);
            return;
        }
        if (cmd == "mode") {
            std::string arg;
            ss >> arg;
            if (arg != "vc" && arg != "s2ut") {
                std::cout << "usage: mode vc|s2ut\n";
                return;
            }
            if (!console_langs) {
                std::cout << "mode: " << translation_msg << '\n';
                return;
            }
            console_params.set(fluxvoice::ParamId::Translate, arg == "s2ut" ? 1.0f : 0.0f);
            host.push_params(console_params);
            std::cout << "mode: " << pair_status(*console_langs, console_params) << '\n';
            return;
        }
        if (cmd == "lang") {
            std::string src, tgt, extra;
            ss >> src >> tgt >> extra;
            if (src.empty() || tgt.empty() || !extra.empty()) {
                std::cout << "usage: lang <src> <tgt>\n";
                return;
            }
            if (!console_langs) {
                std::cout << "lang: " << translation_msg << '\n';
                return;
            }
            fluxvoice::VoiceParams next = console_params;
            std::string message;
            const PairChange r = change_language_pair(*console_langs, next, src, tgt, message);
            if (r != PairChange::Applied) {
                std::cout << "lang: " << message << '\n';
                return;
            }
            console_params = next;
            host.push_params(console_params);
            std::cout << "lang: " << message << '\n';
            return;
        }
        if (cmd == "waitk") {
            std::string arg, extra;
            ss >> arg >> extra;
            int k = 0;
            if (arg.empty() || !extra.empty() || !parse_wait_k(arg, k)) {
                std::cout << "waitk: expected an integer in [1, 10]\n";
                return;
            }
            if (!console_langs) {
                std::cout << "waitk: " << translation_msg << '\n';
                return;
            }
            console_params.set(fluxvoice::ParamId::WaitK, static_cast<float>(k));
            host.push_params(console_params);
            print_param(fluxvoice::param_desc(fluxvoice::ParamId::WaitK));
            return;
        }
        if (cmd == "langs") {
            if (!console_langs) {
                std::cout << "langs: " << translation_msg << '\n';
                return;
            }
            print_langs(std::cout, *console_langs);
            std::cout << pair_status(*console_langs, console_params) << '\n';
            return;
        }
        if (cmd == "get") {
            std::string arg;
            ss >> arg;
            if (arg.empty()) {
                // Translation controls only exist with translator models loaded.
                for (const fluxvoice::ParamDesc& d : fluxvoice::kParams)
                    if (d.group != fluxvoice::ParamGroup::Translation || console_langs) print_param(d);
                if (console_langs)
                    std::cout << "translation = " << pair_status(*console_langs, console_params) << '\n';
                std::cout << "voice_a = " << voice_name_of(console_params.voice_a) << '\n'
                          << "voice_b = " << voice_name_of(console_params.voice_b) << '\n';
                return;
            }
            if (arg == "voice_a" || arg == "voice_b") {
                std::cout << arg << " = "
                          << voice_name_of(arg == "voice_a" ? console_params.voice_a
                                                            : console_params.voice_b)
                          << '\n';
                return;
            }
            const fluxvoice::ParamDesc* d = fluxvoice::find_param(arg);
            if (!d) {
                std::cout << "unknown parameter '" << arg << "'\n";
                return;
            }
            print_param(*d);
            return;
        }
        if (cmd == "preset") {
            std::string name;
            ss >> name;
            if (name.empty()) {
                std::cout << "usage: preset <name>\n";
                return;
            }
            auto r = fluxvoice::resolve_preset(name, preset_dirs);
            if (!r.ok()) {
                std::cout << r.error << '\n';
                return;
            }
            for (const std::string& w : r.warnings) std::cout << "preset: " << w << '\n';
            // Work on a copy and commit at the end: an unknown voice or an
            // invalid language pair must not leave the params half-applied,
            // and the pipeline only ever sees snapshots already validated here.
            fluxvoice::Preset preset = *r.preset;
            fluxvoice::VoiceParams next = console_params;
            std::uint16_t va = next.voice_a, vb = next.voice_b;
            bool vb_follows = voice_b_follows_a;
            const auto resolve = [&](const std::optional<std::string>& name_opt,
                                     std::uint16_t& out) -> bool {
                if (!name_opt) return true;
                std::string err;
                const fluxvoice::VoiceEntry* v = fluxvoice::find_voice(*console_voices, *name_opt, err);
                if (!v) {
                    std::cout << err << '\n';
                    return false;
                }
                out = static_cast<std::uint16_t>(v - console_voices->data());
                return true;
            };
            if (!resolve(preset.voice_a, va) || !resolve(preset.voice_b, vb)) return;
            if (preset.voice_b) vb_follows = false;
            // The translation mode switch goes through the same availability
            // check as 'mode'; the language pair through the same validation
            // as 'lang'. Omitted keys leave mode, languages and wait_k alone.
            const auto translate_req = preset.params[fluxvoice::param_index(fluxvoice::ParamId::Translate)];
            const auto wait_k_req = preset.params[fluxvoice::param_index(fluxvoice::ParamId::WaitK)];
            preset.params[fluxvoice::param_index(fluxvoice::ParamId::Translate)].reset();
            if (!console_langs) {
                // Nothing translation-related can be accepted without translator models.
                for (const fluxvoice::ParamDesc& d : fluxvoice::kParams)
                    if (d.group == fluxvoice::ParamGroup::Translation)
                        preset.params[fluxvoice::param_index(d.id)].reset();
                if (translate_req || preset.src_lang || preset.tgt_lang)
                    std::cout << "preset: translation settings ignored: "
                              << translation_msg << '\n';
            }
            fluxvoice::apply_preset_params(preset, next);
            next.voice_a = va;
            next.voice_b = vb_follows && preset.voice_a ? va : vb;
            if (console_langs && (preset.src_lang || preset.tgt_lang)) {
                const auto* cur_s = console_langs->find(std::to_string(next.src_lang_id));
                const auto* cur_t = console_langs->find(std::to_string(next.tgt_lang_id));
                const std::string s_code = preset.src_lang ? *preset.src_lang : (cur_s ? cur_s->code : "");
                const std::string t_code = preset.tgt_lang ? *preset.tgt_lang : (cur_t ? cur_t->code : "");
                fluxvoice::VoiceParams paired = next;
                std::string message;
                if (change_language_pair(*console_langs, paired, s_code, t_code, message) ==
                    PairChange::Applied) {
                    // An unchanged pair keeps wait_k; a preset that carries wait_k
                    // keeps it over the new pair's default.
                    if (paired.src_lang_id == next.src_lang_id && paired.tgt_lang_id == next.tgt_lang_id)
                        paired.set(fluxvoice::ParamId::WaitK, next.get(fluxvoice::ParamId::WaitK));
                    if (wait_k_req) paired.set(fluxvoice::ParamId::WaitK, *wait_k_req);
                    next = paired;
                } else {
                    std::cout << "preset: languages skipped: " << message << '\n';
                }
            }
            if (translate_req && console_langs)
                next.set(fluxvoice::ParamId::Translate, *translate_req);
            console_params = next;
            voice_b_follows_a = vb_follows;
            host.push_params(console_params);
            std::cout << "preset '" << r.preset->name << "' applied\n";
            return;
        }
        if (cmd == "save") {
            std::string name;
            ss >> name;
            if (preset_dir.empty()) {
                std::cout << "save: pass --preset-dir <dir> to enable saving\n";
                return;
            }
            const std::string slug = slugify(name);
            if (name.empty() || slug.empty()) {
                std::cout << "save: '" << name << "' has no usable filename characters\n";
                return;
            }
            // Mode A has no translation controls; don't serialize them.
            fluxvoice::Preset p = fluxvoice::preset_from_params(
                name, console_params, voice_name_of(console_params.voice_a),
                voice_name_of(console_params.voice_b));
            for (const fluxvoice::ParamDesc& d : fluxvoice::kParams)
                if (d.group == fluxvoice::ParamGroup::Translation)
                    p.params[fluxvoice::param_index(d.id)].reset();
            const std::filesystem::path file = std::filesystem::path(preset_dir) / (slug + ".json");
            std::string error;
            if (fluxvoice::save_preset_file(file, p, error))
                std::cout << "saved " << file.string() << '\n';
            else
                std::cout << error << '\n';
            return;
        }
        if (cmd == "voices") {
            for (std::size_t i = 0; i < console_voices->size(); ++i) {
                const auto& v = console_voices->at(i);
                std::cout << "  [" << i << "] " << v.name << "  " << v.display;
                if (i == console_params.voice_a) std::cout << "  [A]";
                if (i == console_params.voice_b && i != console_params.voice_a) std::cout << "  [B]";
                std::cout << '\n';
            }
            return;
        }
    };

    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::seconds(duration_s);
    auto next_stats = t0 + std::chrono::seconds(stats_s);
    int exit_code = 0;
    const auto take_snapshot = [&] {
        return neural ? host.stats() : legacy_stats(controller.router(), output_drops);
    };
    while (!g_stop.load()) {
        const auto now = std::chrono::steady_clock::now();
        if (host.failed()) {
            std::cerr << "Inference stopped after repeated failures: " << host.error() << '\n';
            exit_code = 1;
            break;
        }
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
        host.flush_params();
        std::string line;
        while (console_lines->try_dequeue(line)) handle_line(line);
        if (stats_s > 0 && now >= next_stats) {
            const double t = std::chrono::duration<double>(now - t0).count();
            std::cout << fluxvoice::format_stats_line(t, take_snapshot()) << std::endl;
            next_stats += std::chrono::seconds(stats_s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Before teardown: callbacks that run after the worker stops would count
    // as underruns that are not the stream's fault (BEST_PRACTICES §3.5).
    const fluxvoice::HostStats end = take_snapshot();
    const uint32_t active_rate =
        controller.stream_info().client_sample_rate > 0
            ? controller.stream_info().client_sample_rate
            : sample_rate;

    controller.stop();

    if (stats_s > 0) std::cout << fluxvoice::format_stats_line(elapsed, end) << '\n';
    std::cout << "Stopped after " << elapsed << " s.\n"
              << "Stats: frames processed = " << end.frames
              << ", input overruns = " << end.overruns
              << ", output underruns = " << end.underruns
              << ", output drops = " << end.drops << '\n'
              << "Governor: skipped = " << end.skipped << ", prefilled = " << end.prefilled << '\n'
              << "Queued audio: " << end.queued << " samples ("
              << 1000.0 * static_cast<double>(end.queued) / static_cast<double>(active_rate)
              << " ms); RSS " << end.rss / 1024 << " KB\n";
    if (end.s2ut) print_s2ut_summary(end);
    return exit_code;
}
