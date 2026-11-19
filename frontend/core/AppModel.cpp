// AppModel — state owner and command implementation.
#include "core/AppModel.hpp"

#include "audio/AudioTypes.hpp"
#include "core/DeviceSelectorModel.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "core/TrayModel.hpp"
#include "pipeline/ModelBundle.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace fluxvoice::frontend {

namespace fs = std::filesystem;

const char* to_string(Section s) {
    switch (s) {
        case Section::Live:      return "Live";
        case Section::Devices:   return "Devices";
        case Section::Voices:    return "Voices";
        case Section::Studio:    return "Voice Studio";
        case Section::Translate: return "Translate";
        case Section::Catalogue: return "Catalogue";
        case Section::Telemetry: return "Telemetry";
        case Section::Settings:  return "Settings";
        case Section::Count:     break;
    }
    return "?";
}

const char* to_string(ParamApply a) {
    switch (a) {
        case ParamApply::BypassInactive: return "Bypass - model controls inactive";
        case ParamApply::Stopped:        return "engine stopped: applied at start";
        case ParamApply::Pending:        return "waiting for the pipeline";
        case ParamApply::Applied:        return "applied by the pipeline";
        case ParamApply::Rejected:       return "rejected by the pipeline";
    }
    return "?";
}

namespace {

ControllerConfig default_controller_config() {
    ControllerConfig cfg;
    cfg.sample_rate = 16000;
    cfg.period_ms   = 20;
    return cfg;
}

std::string read_text(const fs::path& p, bool& ok) {
    std::ifstream f(p, std::ios::binary);
    ok = static_cast<bool>(f);
    if (!ok) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    const auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
    for (std::size_t i = 0; i < a.size(); ++i)
        if (lower(a[i]) != lower(b[i])) return false;
    return true;
}

} // namespace

AppModel::AppModel(AudioBackend& backend, AppPaths paths, Settings settings,
                   std::shared_ptr<const fluxvoice::HostModelLoader> loader)
    : engine_(backend, default_controller_config(), std::move(loader)),
      paths_(std::move(paths)),
      settings_(std::move(settings)) {}

AppModel::~AppModel() { loader_thread_.join(); }

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void AppModel::initialise(TimePoint now) {
    // load_bundle_state() rewrites settings_.active_bundle with whatever is
    // active; only a bundle the user had really chosen loads at startup.
    const std::string saved_active_bundle = settings_.active_bundle;
    // Bundles and catalogues.
    rescan_models();
    if (!paths_.assets_dir.empty()) {
        const fs::path idx = paths_.assets_dir / "catalogue" / "index.json";
        std::error_code ec;
        if (fs::is_regular_file(idx, ec)) catalog_.load_catalogue_index(idx);
    }
    if (!settings_.catalogue_index.empty()) {
        std::string err;
        if (!catalog_.load_catalogue_index(path_from_utf8(settings_.catalogue_index), &err))
            notify(Notice::Level::Warning, "catalogue index: " + err);
    }

    // Active bundle: the saved one when still present, else the first valid.
    if (!settings_.active_bundle.empty() && catalog_.bundle(settings_.active_bundle)) {
        active_bundle_ = settings_.active_bundle;
    } else {
        for (const BundleInfo& b : catalog_.bundles())
            if (b.valid() && !b.voices.empty()) { active_bundle_ = b.name; break; }
    }
    load_bundle_state();

    // Voices. Quiet assignment: the one notice below says what the saved name did.
    std::string voice_error;
    const std::string saved_voice_a = settings_.voice_a, saved_voice_b = settings_.voice_b;  // assign_voice rewrites settings_
    if (!saved_voice_a.empty() && !assign_voice(false, saved_voice_a, voice_error))
        notify(Notice::Level::Warning, "saved voice '" + saved_voice_a + "' is not in bundle '" + active_bundle_ + "'");
    if (voice_a_.empty()) {
        const auto names = voice_names();
        if (!names.empty()) static_cast<void>(assign_voice(false, names.front(), voice_error));
    }
    if (!saved_voice_b.empty()) static_cast<void>(assign_voice(true, saved_voice_b, voice_error));
    if (voice_b_.empty()) voice_b_ = voice_a_;
    // A saved B that differs from A was chosen explicitly; otherwise B follows A.
    voice_b_explicit_ = !voice_b_.empty() && voice_b_ != voice_a_;

    // Presets, then the saved control state on top.
    reload_presets();
    // The last preset is re-applied, so its label never sits on top of default
    // values (Stage 03). Its voice parameters only: voices, languages, Wait-k,
    // the mode and the gain are saved separately and are newer than the preset.
    if (!settings_.last_preset.empty()) {
        const Preset* last = nullptr;
        for (const Preset& p : presets_)
            if (p.name == settings_.last_preset) { last = &p; break; }
        if (last) {
            Preset voice_only = *last;
            for (const ParamDesc& d : kParams)
                if (d.group == ParamGroup::Translation) voice_only.params[param_index(d.id)].reset();
            apply_preset_params(voice_only, params_);
            current_preset_ = last->name;
            const auto& g = last->params[param_index(ParamId::Gain)];
            preset_modified_ = g.has_value() && *g != settings_.gain_db;
        } else {
            notify(Notice::Level::Info, "The last preset '" + settings_.last_preset + "' is no longer available.");
            settings_.last_preset.clear();
        }
    }
    params_.set(ParamId::Gain, settings_.gain_db);
    // The saved Wait-k survives a restart for every pair: restoring the pair
    // must not replace it by the pair default.
    if (!change_language_pair(settings_.src_lang, settings_.tgt_lang, false)) {
        notify(Notice::Level::Info, "Saved language pair " + settings_.src_lang + " -> " + settings_.tgt_lang +
                                        " is not available: " + translation_message_);
        src_lang_ = "en";
        tgt_lang_ = "es";
        static_cast<void>(change_language_pair(src_lang_, tgt_lang_, false));
    }
    params_.set(ParamId::WaitK, static_cast<float>(TranslationModel::clamp_wait_k(settings_.wait_k)));
    settings_.wait_k = wait_k();
    if (settings_.translate && !set_translate(true))
        notify(Notice::Level::Info, "translation stays off: " + translation_.unavailable_reason());

    // Studio draft: reopen the most recently modified project (BUG-021).
    std::error_code ec;
    if (!paths_.studio_dir.empty() && fs::is_directory(paths_.studio_dir, ec)) {
        std::string newest;
        fs::file_time_type newest_t{};
        for (const auto& e : fs::directory_iterator(paths_.studio_dir, ec)) {
            const std::string name = path_to_utf8(e.path().filename());
            if (!e.is_directory(ec) || !is_safe_project_name(name)) continue;
            const auto t = fs::last_write_time(e.path() / "draft.json", ec);
            if (ec) { ec.clear(); continue; }
            if (newest.empty() || t > newest_t || (t == newest_t && name < newest)) { newest = name; newest_t = t; }
        }
        if (!newest.empty() && !studio_open_project(newest))
            notify(Notice::Level::Warning, "Voice Studio draft '" + newest + "' could not be read.");
    }

    // Devices: enumerate, restore the policy, maybe start.
    engine_.tick(now);
    const auto snap = engine_.controller().refresh(now);
    if (!is_ok(snap)) {
        notify(Notice::Level::Error, explain_error(std::get<AudioError>(snap)));
    }
    capture_mode_ = settings_.capture_loopback && loopback_supported() ? CaptureMode::SystemPlaybackLoopback
                                                                       : CaptureMode::OrdinaryMicrophone;
    bool cap_ok = true, play_ok = true;
    restore_device_policy(cap_ok, play_ok);
    seen_generation_ = snapshot().generation;
    if (!cap_ok || !play_ok) {
        std::string names;
        if (unresolved_capture_) names += " capture '" + *unresolved_capture_ + "'";
        if (unresolved_playback_) names += std::string(names.empty() ? "" : " and") + " playback '" + *unresolved_playback_ + "'";
        notify(Notice::Level::Warning,
               "A remembered device is no longer listed (" + names.substr(1) + "). Choose it again in Devices; "
               "FluxVoice never substitutes another device.");
        section_ = Section::Devices;
    }

    window_visible_ = !settings_.start_minimized;
    push_params();
    last_state_ = engine_.status().state;

    // The model: `--bundle`, else the saved active bundle, loads on the loader
    // thread. "Start on launch" waits behind that load, so the first audio is
    // never an unannounced Bypass; if the load fails the engine is not started.
    const bool may_start = cap_ok && play_ok && settings_.start_engine_on_launch && backend_usable();
    fs::path startup = startup_bundle_;
    if (startup.empty() && !saved_active_bundle.empty())
        if (const BundleInfo* b = catalog_.bundle(saved_active_bundle); b && b->valid() && !b->voices.empty())
            startup = b->dir;
    if (!startup.empty()) {
        start_after_load_ = may_start;
        if (!load_bundle(startup)) start_after_load_ = false;
    } else if (may_start) {
        static_cast<void>(start_engine());
    }
    last_state_ = engine_.status().state;
}

void AppModel::tick(TimePoint now) {
    double dt = 0.0;
    if (has_last_tick_) dt = std::chrono::duration<double>(now - last_tick_).count();
    has_last_tick_ = true;
    last_tick_     = now;
    if (dt < 0.0 || dt > 5.0) dt = 0.0;
    uptime_s_ += dt;

    // A finished bundle load is installed here, on the control thread (the
    // same drain-in-tick pattern as poll_analysis and the native dialogs).
    if (loader_thread_.busy()) {
        std::optional<BundleLoadResult> done;
        try {
            done = loader_thread_.poll();
        } catch (const std::exception& e) {
            done.emplace();
            done->dir   = loading_dir_;
            done->error = std::string("Cannot load bundle: ") + e.what();
        } catch (...) {
            done.emplace();
            done->dir   = loading_dir_;
            done->error = "Cannot load bundle: unknown exception";
        }
        if (done) finish_bundle_load(std::move(*done));
    }

    engine_.tick(now);
    if (snapshot().generation != seen_generation_) {  // hot-plug: a missing remembered device may be back
        seen_generation_ = snapshot().generation;
        retry_unresolved_devices();
    }

    // State transitions worth a notice (the stream stopped without the user asking).
    const StreamState st = engine_.status().state;
    if (st != last_state_) {
        if (st == StreamState::AwaitingUser || st == StreamState::Error) {
            engine_requested_ = false;  // the stream is down until the user acts; the tray must not claim "on"
            if (engine_.status().last_error)
                notify(Notice::Level::Error, "Stream stopped: " + explain_error(*engine_.status().last_error));
        } else if (st == StreamState::Recovering)
            notify(Notice::Level::Warning, "Audio device interrupted; retrying.");
        last_state_ = st;
    }

    engine_.flush_params();  // deliver a snapshot parked by a full queue (BUG-023)

    // The pipeline stopped the worker for good: stop the stream and say why.
    // The host's failed() latch lasts until the next prepare(), so it is
    // handled once and never restarts anything by itself.
    if (engine_.host_failed()) {
        if (!failure_ack_) handle_pipeline_failure();
    } else {
        failure_ack_ = false;
    }

    // Stats log cadence: once per second of stream time. The elapsed dt counts
    // for the stream only when it was already running; a new epoch (below)
    // restarts the cadence, so the first line is due one second after the start.
    if (stream_running_) stats_acc_s_ += dt;
    update_stream_clock(st);

    history_acc_s_ += dt;
    if (history_acc_s_ >= 1.0) {
        history_acc_s_ = 0.0;
        const TelemetrySnapshot t = engine_.telemetry();
        if (t.output_underruns < last_underruns_) last_underruns_ = t.output_underruns;  // per-stream rebaseline (BUG-024)
        underruns_.push(static_cast<float>(t.output_underruns - last_underruns_));
        last_underruns_ = t.output_underruns;
        queued_ms_.push(t.queued_valid ? static_cast<float>(t.queued_ms()) : 0.0f);
        compute_p95_ms_.push(t.stats.vc ? static_cast<float>(static_cast<double>(t.stats.timing.p95_us) / 1000.0) : 0.0f);
    }
    if (stream_running_ && stats_acc_s_ >= 1.0) {
        stats_acc_s_ = 0.0;
        if (settings_.write_stats_log) append_stats_line(engine_.telemetry());
        else close_stats_log();
    }

    // WAV analysis runs on the studio's worker thread; its results are applied
    // here, on the UI/control thread, every frame and in any section, and the
    // draft is saved once per batch (Stage 03).
    if (studio_.poll_analysis() > 0) static_cast<void>(studio_save_draft());

    studio_poll_s_ += dt;
    if (section_ == Section::Studio && studio_poll_s_ >= 1.0) {
        studio_poll_s_ = 0.0;
        studio_status_ = studio_.poll_status(paths_.studio_dir);
    }
}

// ---------------------------------------------------------------------------
// Stream clock and stats log (M4 Stage 06)
// ---------------------------------------------------------------------------

fs::path AppModel::stats_log_file() const {
    return settings_.stats_log_path.empty() ? paths_.default_stats_log() : path_from_utf8(settings_.stats_log_path);
}

void AppModel::update_stream_clock(StreamState st) {
    if (st != StreamState::Running) {
        stream_running_       = false;
        stream_epoch_pending_ = false;
        close_stats_log();
        return;
    }
    if (!stream_running_ || stream_epoch_pending_) {
        stream_running_       = true;
        stream_epoch_pending_ = false;
        stream_t0_            = uptime_s_;
        stats_acc_s_          = 0.0;
        close_stats_log();  // the previous stream's file; this stream opens its own
    }
}

void AppModel::close_stats_log() {
    if (stats_log_.is_open()) stats_log_.close();
    stats_log_.clear();
    stats_open_path_.clear();
}

void AppModel::fail_stats_log(const std::string& what) {
    const fs::path file = stats_log_file();
    close_stats_log();
    settings_.write_stats_log = false;
    notify(Notice::Level::Error, "Stats log disabled: " + what + " (" + path_to_utf8(file) +
                                     "). Turn it on again in Settings once the path is writable.");
}

void AppModel::append_stats_line(const TelemetrySnapshot& t) {
    if (!settings_.write_stats_log) {
        close_stats_log();
        return;
    }
    const fs::path file = stats_log_file();
    if (stats_log_.is_open() && stats_open_path_ != file) close_stats_log();  // the path was changed in Settings
    if (!stats_log_.is_open()) {
        std::error_code ec;
        if (settings_.stats_log_path.empty()) fs::create_directories(file.parent_path(), ec);  // the config dir only
        stats_log_.open(file, std::ios::out | std::ios::app | std::ios::binary);
        if (!stats_log_.is_open()) {
            fail_stats_log("cannot open the file for appending");
            return;
        }
        stats_open_path_ = file;
        ++stats_opens_;
    }
    stats_log_ << t.stats_line(stream_seconds()) << '\n';
    stats_log_.flush();
    if (!stats_log_) {
        fail_stats_log("a write failed");
        return;
    }
    ++stats_lines_;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

bool AppModel::backend_usable() const {
    return backend_status().state == BackendState::Ready;
}

bool AppModel::start_engine() {
    if (bundle_loading()) {  // the stream never runs across a model change
        engine_message_ = "A model bundle is loading; the engine can start when it has finished.";
        notify(Notice::Level::Info, engine_message_);
        return false;
    }
    if (!backend_usable()) {
        engine_message_ = backend_status().state == BackendState::NullOnly
                              ? "No audio hardware is reachable (null backend, offline mode)."
                              : "No audio backend could be initialised.";
        notify(Notice::Level::Error, engine_message_);
        return false;
    }
    // BUG-020: a remembered explicit device that is not resolvable is never
    // replaced by the system default. Live, tray, Mini panel and hotkeys all
    // come through here.
    if (device_unresolved()) {
        engine_message_   = explain_error(unresolved_device_error());
        engine_requested_ = false;
        notify(Notice::Level::Error, engine_message_);
        return false;
    }
    if (const auto fb = feedback_warning()) {
        engine_message_ = *fb;
        notify(Notice::Level::Error, *fb);
        return false;
    }
    return start_engine_with(capture_sel_, playback_sel_, capture_mode_);
}

bool AppModel::start_engine_with(const DeviceSelection& cap, const DeviceSelection& play, CaptureMode mode) {
    const auto res = engine_.start(cap, play, mode);
    if (!is_ok(res)) {
        engine_message_   = explain_error(std::get<AudioError>(res));
        engine_requested_ = false;
        notify(Notice::Level::Error, engine_message_);
        return false;
    }
    engine_requested_ = true;
    engine_message_   = "Running: " + capture_name() + " -> " + playback_name();
    last_state_       = engine_.status().state;
    stream_epoch_pending_ = true;  // a start or reconfigure discarded the queues: new accounting epoch
    push_params();
    return true;
}

void AppModel::stop_engine() {
    engine_.stop();
    engine_requested_ = false;
    engine_message_   = "Stopped.";
    last_state_       = engine_.status().state;
}

void AppModel::on_unexpected_exception(std::string_view what) noexcept {
    try {
        engine_.stop();  // controller quiesces the worker through its hooks, then closes the stream
    } catch (...) {
        // The stream may be what failed; the stopped flag below still applies.
    }
    engine_requested_ = false;
    try {
        engine_message_ = "Audio stopped after an unexpected error: " + std::string(what);
        notify(Notice::Level::Error, engine_message_);
        last_state_ = engine_.status().state;
    } catch (...) {
        // Out of memory while formatting: the stream is already stopped.
    }
}

void AppModel::set_muted(bool m) {
    engine_.set_muted(m);
}

// ---------------------------------------------------------------------------
// Model bundle lifecycle
// ---------------------------------------------------------------------------

namespace {

bool same_directory(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    if (fs::equivalent(a, b, ec) && !ec) return true;
    return a.lexically_normal() == b.lexically_normal();
}

std::string bundle_name_of(const fs::path& dir) {
    const fs::path leaf = dir.has_filename() ? dir.filename() : dir.parent_path().filename();
    return path_to_utf8(leaf);
}

} // namespace

bool AppModel::load_bundle(const fs::path& dir) {
    if (loader_thread_.busy()) {
        notify(Notice::Level::Warning, "A model bundle is already loading; wait for it to finish.");
        return false;
    }
    // Remember the device policy and stop: the stream never runs across a model
    // change, and the old host is replaced only while the controller is Idle.
    resume_ = Resume{};
    if (engine_requested_) {
        resume_.valid    = true;
        resume_.capture  = capture_sel_;
        resume_.playback = playback_sel_;
        resume_.mode     = capture_mode_;
    }
    if (engine_requested_ || engine_.status().state != StreamState::Idle) stop_engine();

    // The seeder runs on the loader thread: it reads only these copies, never
    // the AppModel. Voice names that the bundle does not have fall back to its
    // first voice here; finish_bundle_load() reports them on the control thread.
    VoiceParams seed_params = params_;
    seed_params.set(ParamId::Translate, 0.0f);  // encoder + vocoder only until Stage 07
    const std::string seed_a = voice_a_, seed_b = voice_b_;
    fluxvoice::HostParamSeeder seeder = [seed_params, seed_a, seed_b](const fluxvoice::HostBundleInfo& info, VoiceParams& out,
                                                                      fluxvoice::S2utOptions&, std::string&) {
        const auto find = [&](const std::string& name) -> std::uint16_t {
            if (info.voices)
                for (std::size_t i = 0; i < info.voices->size(); ++i)
                    if ((*info.voices)[i].name == name) return static_cast<std::uint16_t>(i);
            return 0;
        };
        out         = seed_params;
        out.voice_a = find(seed_a);
        out.voice_b = seed_b.empty() ? out.voice_a : find(seed_b);
        return true;
    };

    fluxvoice::HostLoadRequest req;
    req.model_dir         = dir;
    req.frames_per_step   = settings_.frames_per_step;
    req.latency_target_ms = settings_.latency_target_ms;
    req.translator        = fluxvoice::TranslatorPolicy::kNever;

    loading_dir_  = dir;
    loading_name_ = bundle_name_of(dir);
    bundle_report_.reset();
    try {
        const std::shared_ptr<const fluxvoice::HostModelLoader> loader = engine_.loader();
        loader_thread_.start([loader, req, seeder]() -> BundleLoadResult {
            BundleLoadResult r;
            r.dir = req.model_dir;
            try {
                auto host = std::make_unique<fluxvoice::ModeAHost>(loader);
                fluxvoice::HostError err;
                if (host->load(req, seeder, err)) {
                    r.host = std::move(host);
                } else {
                    r.error = err.message.empty() ? std::string("the bundle could not be loaded") : err.message;
                }
            } catch (const std::exception& e) {
                r.error = std::string("Cannot load bundle: ") + e.what();
            } catch (...) {
                r.error = "Cannot load bundle: unknown exception";
            }
            return r;
        });
    } catch (const std::exception& e) {  // the thread could not be created
        BundleLoadResult failed;
        failed.dir   = dir;
        failed.error = std::string("Cannot start the loader thread: ") + e.what();
        finish_bundle_load(std::move(failed));
        return false;
    }
    engine_message_ = "Loading bundle '" + loading_name_ + "'...";
    return true;
}

bool AppModel::activate_bundle(std::string_view name, std::string_view voice, bool as_b) {
    const BundleInfo* b = catalog_.bundle(name);
    if (!b) {
        notify(Notice::Level::Warning, "No bundle named '" + std::string(name) + "'.");
        return false;
    }
    if (!b->valid() || b->voices.empty()) {
        notify(Notice::Level::Warning, "Bundle '" + b->name + "' cannot be used: " + (b->error.empty() ? "it has no voices" : b->error));
        return false;
    }
    if (!load_bundle(b->dir)) return false;
    pending_voice_   = std::string(voice);
    pending_voice_b_ = as_b;
    return true;
}

bool AppModel::use_voice(std::string_view bundle, std::string_view voice, bool as_b) {
    if (engine_.mode_a() && loaded_bundle_name_ == bundle && !bundle_loading())
        return as_b ? set_voice_b(voice) : set_voice_a(voice);
    return activate_bundle(bundle, voice, as_b);
}

bool AppModel::reload_bundle() {
    fs::path dir = engine_.mode_a() ? loaded_bundle_dir_ : fs::path{};
    if (dir.empty())
        if (const BundleInfo* b = active_bundle_info()) dir = b->dir;
    if (dir.empty()) {
        notify(Notice::Level::Warning, "There is no bundle to load.");
        return false;
    }
    return load_bundle(dir);
}

std::optional<BundleReport> AppModel::take_bundle_report() {
    std::optional<BundleReport> r = std::move(bundle_report_);
    bundle_report_.reset();
    return r;
}

void AppModel::sync_to_installed_host(std::string_view wanted_voice, bool wanted_as_b) {
    const fluxvoice::ModeAHost* host = engine_.host();
    if (!host) return;
    const std::string first = host->voices().empty() ? std::string() : host->voices().front().name;
    const std::string a = voice_a_, b = voice_b_;  // copies: assign_voice rewrites the members
    std::string err;
    // Voice names are resolved against the new bundle's table; an unknown one
    // falls back to the first voice and says so.
    if (a.empty() || !assign_voice(false, a, err)) {
        if (!first.empty()) static_cast<void>(assign_voice(false, first, err));
        if (!a.empty())
            notify(Notice::Level::Info, "Voice '" + a + "' is not in bundle '" + loaded_bundle_name_ + "'; using '" + first + "'.");
    }
    if (voice_b_explicit_ && !b.empty() && !assign_voice(true, b, err)) {
        voice_b_explicit_ = false;  // B follows A again
        voice_b_          = voice_a_;
        params_.voice_b   = params_.voice_a;
        notify(Notice::Level::Info, "Voice B '" + b + "' is not in bundle '" + loaded_bundle_name_ + "'; it follows voice A.");
    }
    if (!wanted_voice.empty() && !assign_voice(wanted_as_b, wanted_voice, err)) notify(Notice::Level::Warning, err);
}

void AppModel::finish_bundle_load(BundleLoadResult r) {
    const std::string name        = loading_name_.empty() ? bundle_name_of(r.dir) : loading_name_;
    const std::string wanted      = std::move(pending_voice_);
    const bool        wanted_as_b = pending_voice_b_;
    pending_voice_.clear();
    pending_voice_b_ = false;
    loading_name_.clear();
    Resume resume = std::move(resume_);
    resume_       = Resume{};
    const bool launch_start = start_after_load_;
    start_after_load_       = false;

    BundleReport rep;
    rep.dir  = r.dir;
    rep.name = name;
    std::string failure = std::move(r.error);
    bool installed = false;
    if (r.host) {
        rep.rate   = r.host->contract_rate();
        rep.hop    = r.host->geometry().hop;
        rep.voices = r.host->voices().size();
        std::string err;
        // The old host (if any) is detached and destroyed inside install_host(),
        // with the controller Idle; a refusal puts the previous host back.
        installed = engine_.install_host(std::move(r.host), name, err);
        if (!installed) failure = err.empty() ? std::string("the model host could not be installed") : err;
    }
    if (!engine_.mode_a()) loaded_bundle_name_.clear();

    if (installed) {
        loaded_bundle_name_ = name;
        loaded_bundle_dir_  = r.dir;
        // The catalogue follows what is loaded (a --bundle outside the models root is added).
        bool listed = false;
        for (const BundleInfo& b : catalog_.bundles())
            if (b.name == name && same_directory(b.dir, r.dir)) listed = true;
        if (!listed) static_cast<void>(catalog_.add_bundle_dir(r.dir));
        if (catalog_.bundle(name)) {
            active_bundle_ = name;
            load_bundle_state();  // languages, translator state of the catalogue entry; saves active_bundle
        }
        // The GUI loads encoder + vocoder only (until M4 Stage 07): the host reports
        // whether it owns translator models, and translate = 1 is rejected without them.
        translation_.set_translator_available(engine_.host_has_translator(), fluxvoice::kNoTranslatorModelsMessage);
        pipeline_failure_.clear();
        sync_to_installed_host(wanted, wanted_as_b);
        push_params();
        rep.ok = true;
        bundle_report_ = rep;
        engine_message_ = "Loaded bundle '" + name + "'.";
        notify(Notice::Level::Info, "Loaded bundle '" + name + "' (" + std::to_string(rep.rate) + " Hz, hop " +
                                        std::to_string(rep.hop) + " samples, " + std::to_string(rep.voices) + " voices).");
    } else {
        rep.ok      = false;
        rep.message = failure;
        std::string text = "Bundle '" + name + "' failed to load: " + failure + ". ";
        text += engine_.mode_a() ? "The previous bundle ('" + loaded_bundle_name_ + "') stays active."
                                 : "No model is loaded; the engine is in Bypass.";
        engine_message_ = text;
        notify(Notice::Level::Error, text);
        bundle_report_ = rep;
    }

    // Restart with the policy remembered when the load stopped a running stream.
    // After a failure that restores what was running (the previous bundle, or
    // Bypass if that is what ran); a failed "start on launch" load does not
    // start anything: Bypass is never entered silently.
    if (resume.valid) {
        static_cast<void>(start_engine_with(resume.capture, resume.playback, resume.mode));
    } else if (launch_start && installed) {
        static_cast<void>(start_engine());
    } else if (launch_start) {
        notify(Notice::Level::Warning, "The engine was not started because the model bundle did not load.");
    }
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

DeviceRows AppModel::capture_rows() const {
    return build_rows(snapshot(), EndpointDirection::Capture, snapshot_stale(), capture_mode_);
}

DeviceRows AppModel::playback_rows() const {
    return build_rows(snapshot(), EndpointDirection::Playback, snapshot_stale());
}

// Any explicit choice, including the "Follow system default" row, replaces
// the remembered-but-missing device. A row that is not in the list is an
// error: the selection and any unresolved state stay as they are.
bool AppModel::select_capture_row(std::size_t row) {
    const std::optional<DeviceSelection> sel = selection_for_row(capture_rows(), row);
    if (!sel) {
        notify(Notice::Level::Error, "Capture row " + std::to_string(row) + " is not in the device list; nothing changed.");
        return false;
    }
    capture_sel_ = *sel;
    unresolved_capture_.reset();
    return true;
}
bool AppModel::select_playback_row(std::size_t row) {
    const std::optional<DeviceSelection> sel = selection_for_row(playback_rows(), row);
    if (!sel) {
        notify(Notice::Level::Error, "Playback row " + std::to_string(row) + " is not in the device list; nothing changed.");
        return false;
    }
    playback_sel_ = *sel;
    unresolved_playback_.reset();
    return true;
}

void AppModel::set_capture_mode(CaptureMode m) {
    if (m == CaptureMode::SystemPlaybackLoopback && !loopback_supported()) {
        notify(Notice::Level::Warning, "System-playback capture is not supported by this backend.");
        return;
    }
    if (m != capture_mode_) capture_sel_ = FollowSystemDefault{};  // rows change list; never carry a key across
    capture_mode_ = m;
}

std::optional<std::string> AppModel::feedback_warning() const {
    return feedback_preview(snapshot(), capture_sel_, playback_sel_, capture_mode_);
}

bool AppModel::apply_devices() {
    // An unresolved side keeps its remembered policy: writing "default" here
    // would turn the substitution into a saved choice the user never made.
    // Likewise an explicit endpoint that left the list since it was chosen has
    // no name to remember; the stored policy stays and Start reports the loss.
    if (!unresolved_capture_) {
        if (auto p = policy_text(capture_sel_, EndpointDirection::Capture)) {
            settings_.capture_policy   = std::move(*p);
            settings_.capture_loopback = capture_mode_ == CaptureMode::SystemPlaybackLoopback;
        }
    }
    if (!unresolved_playback_)
        if (auto p = policy_text(playback_sel_, EndpointDirection::Playback)) settings_.playback_policy = std::move(*p);
    const bool ok = start_engine();
    static_cast<void>(save_settings());
    return ok;
}

void AppModel::refresh_devices() {
    const auto res = engine_.controller().refresh();
    if (!is_ok(res)) {
        notify(Notice::Level::Warning, "Device list refresh failed: " + explain_error(std::get<AudioError>(res)));
        return;
    }
    seen_generation_ = snapshot().generation;
    retry_unresolved_devices();
}

void AppModel::retry_unresolved_devices() {
    auto retry = [&](std::optional<std::string>& pending, DeviceSelection& sel, EndpointDirection dir, const char* what) {
        if (!pending) return;
        bool ok = false;
        const DeviceSelection s = selection_from_policy("name:" + *pending, dir, ok);
        if (!ok) return;
        sel = s;
        notify(Notice::Level::Info, std::string("Remembered ") + what + " device '" + *pending + "' is listed again and selected.");
        pending.reset();
    };
    retry(unresolved_capture_, capture_sel_, EndpointDirection::Capture, "capture");
    retry(unresolved_playback_, playback_sel_, EndpointDirection::Playback, "playback");
}

AudioError AppModel::unresolved_device_error() const {
    std::string msg;
    auto add = [&](const std::optional<std::string>& pending, EndpointDirection dir, const char* what) {
        if (!pending) return;
        if (!msg.empty()) msg += "; ";
        msg += std::string("the remembered ") + what + " device '" + *pending + "' " +
               (count_named(*pending, dir) > 1 ? "matches more than one listed device" : "is not in the current device list");
    };
    add(unresolved_capture_, EndpointDirection::Capture, "capture");
    add(unresolved_playback_, EndpointDirection::Playback, "playback");
    return AudioError{AudioErrorCategory::DeviceNotFound, "start the stream", 0, snapshot().backend, msg + "."};
}

Result<EndpointInfo> AppModel::probe(const DeviceSelection& sel, EndpointDirection dir) const {
    const AudioDeviceRegistry& reg = engine_.controller().registry();
    const CaptureMode mode = dir == EndpointDirection::Capture ? capture_mode_ : CaptureMode::OrdinaryMicrophone;
    const auto resolved = dir == EndpointDirection::Capture && mode == CaptureMode::SystemPlaybackLoopback
                              ? reg.resolve_loopback(sel) : reg.resolve(sel, dir);
    if (!is_ok(resolved)) return std::get<AudioError>(resolved);
    const ResolvedEndpoint& r = std::get<ResolvedEndpoint>(resolved);
    if (r.use_default) {
        const EndpointDescriptor* d = reg.default_descriptor(
            mode == CaptureMode::SystemPlaybackLoopback ? EndpointDirection::Playback : dir);
        if (!d) return AudioError{AudioErrorCategory::DeviceNotFound, "probe default endpoint", 0, snapshot().backend,
                                  "No default endpoint is listed for this direction."};
        return reg.capabilities(d->key);
    }
    return reg.capabilities(r.key);
}

std::string AppModel::capture_name() const {
    if (engine_.status().state == StreamState::Running && !engine_.stream_info().capture_name.empty())
        return engine_.stream_info().capture_name;
    if (unresolved_capture_) return *unresolved_capture_ + " (not listed)";
    return describe_selection(capture_sel_, snapshot(), EndpointDirection::Capture, capture_mode_);
}

std::string AppModel::playback_name() const {
    if (engine_.status().state == StreamState::Running && !engine_.stream_info().playback_name.empty())
        return engine_.stream_info().playback_name;
    if (unresolved_playback_) return *unresolved_playback_ + " (not listed)";
    return describe_selection(playback_sel_, snapshot(), EndpointDirection::Playback);
}

namespace {
constexpr std::string_view kPolicyName    = "name:";
constexpr std::string_view kPolicyDefault = "default";
}

std::optional<std::string> AppModel::policy_text(const DeviceSelection& sel, EndpointDirection dir) const {
    if (std::holds_alternative<FollowSystemDefault>(sel)) return std::string(kPolicyDefault);
    // Not via describe_selection(): its "(device no longer listed)" text must
    // never be saved as a device name.
    const std::string name =
        resolved_name(sel, snapshot(), dir, dir == EndpointDirection::Capture ? capture_mode_ : CaptureMode::OrdinaryMicrophone);
    if (name.empty()) return std::nullopt;
    return std::string(kPolicyName) + name;
}

int AppModel::count_named(std::string_view name, EndpointDirection dir) const {
    const auto& list = (dir == EndpointDirection::Capture && capture_mode_ == CaptureMode::SystemPlaybackLoopback)
                           ? snapshot().playback : snapshot().endpoints(dir);
    int matches = 0;
    for (const EndpointDescriptor& d : list)
        if (d.name == name) ++matches;
    return matches;
}

// `ok == false` means "explicit, but not resolvable now". The returned
// FollowSystemDefault is only a placeholder: the caller records the name in
// unresolved_* so start_engine() refuses (BUG-020). Only "" and "default"
// mean Follow system default; any other text that is not "name:<device>" is
// treated as an unresolvable explicit choice, never as the default (Stage 02).
DeviceSelection AppModel::selection_from_policy(std::string_view policy, EndpointDirection dir, bool& ok) const {
    ok = true;
    if (policy.empty() || policy == kPolicyDefault) return FollowSystemDefault{};
    if (!policy.starts_with(kPolicyName) || policy.size() == kPolicyName.size()) {
        ok = false;
        return FollowSystemDefault{};
    }
    const std::string_view name = policy.substr(kPolicyName.size());
    if (count_named(name, dir) != 1) {  // missing or ambiguous (two identical USB mics): ask again
        ok = false;
        return FollowSystemDefault{};
    }
    const auto& list = (dir == EndpointDirection::Capture && capture_mode_ == CaptureMode::SystemPlaybackLoopback)
                           ? snapshot().playback : snapshot().endpoints(dir);
    for (const EndpointDescriptor& d : list)
        if (d.name == name) return ExplicitEndpoint{d.key, snapshot().generation};
    ok = false;  // unreachable: count_named saw exactly one
    return FollowSystemDefault{};
}

void AppModel::restore_device_policy(bool& capture_ok, bool& playback_ok) {
    capture_sel_  = selection_from_policy(settings_.capture_policy, EndpointDirection::Capture, capture_ok);
    playback_sel_ = selection_from_policy(settings_.playback_policy, EndpointDirection::Playback, playback_ok);
    unresolved_capture_.reset();
    unresolved_playback_.reset();
    // The remembered name; for a policy without the "name:" prefix the whole
    // text, which retry_unresolved_devices() matches only as an exact name.
    const auto remembered = [](const std::string& policy) {
        return policy.starts_with(kPolicyName) ? policy.substr(kPolicyName.size()) : policy;
    };
    if (!capture_ok) unresolved_capture_ = remembered(settings_.capture_policy);
    if (!playback_ok) unresolved_playback_ = remembered(settings_.playback_policy);
}

// ---------------------------------------------------------------------------
// Parameters, voices, presets
// ---------------------------------------------------------------------------

void AppModel::push_params() {
    rejected_at_push_ = engine_.rejected_params();
    static_cast<void>(engine_.push_params(params_));
}

ParamApply AppModel::param_apply() const {
    const TelemetrySnapshot t = engine_.telemetry();
    if (!t.mode_a) return ParamApply::BypassInactive;
    if (!engine_requested_ || engine_.status().state == StreamState::Idle) return ParamApply::Stopped;
    // "Applied" is acceptance by InferencePipeline::set_params(), not delivery:
    // ModeAWorker drains the channel (applied_generation advances) and then
    // the pipeline may still refuse the snapshot.
    if (t.params_accepted == t.params_pushed) return ParamApply::Applied;
    if (t.params_rejected > rejected_at_push_) return ParamApply::Rejected;
    return ParamApply::Pending;
}

std::string AppModel::param_apply_text() const { return to_string(param_apply()); }

void AppModel::handle_pipeline_failure() {
    failure_ack_      = true;
    pipeline_failure_ = engine_.host_error();
    if (pipeline_failure_.empty()) pipeline_failure_ = "the pipeline stopped without an error message";
    engine_.stop();  // quiesces the worker through the controller hooks, then closes the stream
    engine_requested_ = false;
    engine_message_   = "Pipeline error: " + pipeline_failure_;
    notify(Notice::Level::Error, engine_message_);
    last_state_ = engine_.status().state;
}

bool AppModel::retry_after_failure() {
    pipeline_failure_.clear();  // failure_ack_ stays set until the host's latch clears (next prepare())
    return start_engine();
}

void AppModel::set_param(ParamId id, float v) {
    if (id == ParamId::Translate) {
        static_cast<void>(set_translate(v >= 0.5f));
        return;
    }
    const float before = params_.get(id);
    params_.set(id, v);
    if (params_.get(id) != before) {
        mark_custom();
        if (id == ParamId::Gain) settings_.gain_db = params_.get(id);
        if (id == ParamId::WaitK) settings_.wait_k = wait_k();
        push_params();
    }
}

void AppModel::reset_params() {
    const bool tr = translate();
    const float k = params_.get(ParamId::WaitK);
    const std::uint16_t s = params_.src_lang_id, t = params_.tgt_lang_id;
    const std::uint16_t a = params_.voice_a, b = params_.voice_b;
    params_ = VoiceParams{};
    params_.set(ParamId::Translate, tr ? 1.0f : 0.0f);  // voice reset keeps translation state (M3 / Stage 09)
    params_.set(ParamId::WaitK, k);
    params_.src_lang_id = s;
    params_.tgt_lang_id = t;
    params_.voice_a = a;
    params_.voice_b = b;
    current_preset_.clear();
    preset_modified_ = false;
    settings_.gain_db = 0.0f;
    push_params();
}

void AppModel::load_bundle_state() {
    // A different bundle: what the translator declares (chunk length,
    // adaptive readiness) is unknown until its models are loaded.
    translation_.set_translator_caps(TranslatorCaps{});
    const BundleInfo* b = active_bundle_info();
    if (!b) {
        translation_.set_translator_available(false, "no bundle loaded");
        translation_.set_table(builtin_lang_table());
        return;
    }
    translation_.set_translator_available(
        b->has_translator, "bundle '" + b->name + "' has no translator models (translator_enc.onnx / translator_dec.onnx)");
    if (b->has_langs) {
        bool ok = false;
        const LangTable t = parse_langs_json(read_text(b->dir / "langs.json", ok));
        if (ok && t.ok()) { LangTable copy = t; copy.from_bundle = true; translation_.set_table(copy); }
    } else {
        translation_.set_table(builtin_lang_table());
    }
    settings_.active_bundle = active_bundle_;
}

std::vector<std::string> AppModel::voice_names() const {
    if (const fluxvoice::ModeAHost* h = engine_.host(); h && active_bundle_ == loaded_bundle_name_) {
        std::vector<std::string> out;
        for (const fluxvoice::VoiceEntry& v : h->voices()) out.push_back(v.name);
        return out;  // the loaded bundle is the authority for names the worker will accept
    }
    return catalog_.voice_names(active_bundle_);
}

bool AppModel::set_active_bundle(std::string_view name) {
    const BundleInfo* b = catalog_.bundle(name);
    if (!b || !b->valid() || b->voices.empty()) return false;
    if (engine_.mode_a() && name != loaded_bundle_name_) {
        notify(Notice::Level::Warning, "Bundle '" + std::string(name) + "' is not the loaded one; activate it to switch.");
        return false;
    }
    active_bundle_ = std::string(name);
    load_bundle_state();
    voice_a_.clear();
    voice_b_.clear();
    voice_b_explicit_ = false;  // a new bundle starts with B following A
    std::string voice_error;
    static_cast<void>(assign_voice(false, b->voices.front().name, voice_error));
    mark_custom();
    if (translate() && !translation_.translator_available()) {
        params_.set(ParamId::Translate, 0.0f);
        notify(Notice::Level::Info, "Translation turned off: " + translation_.unavailable_reason());
    }
    push_params();
    return true;
}

std::optional<std::uint16_t> AppModel::resolve_voice(std::string_view name, std::string& error) const {
    if (engine_.mode_a()) return engine_.voice_index(name, error);  // the loaded bundle is the authority
    const BundleInfo* b = active_bundle_info();
    if (!b) {
        error = "no voice bundle is selected";
        return std::nullopt;
    }
    std::string known;
    for (std::size_t i = 0; i < b->voices.size(); ++i) {
        if (b->voices[i].name == name) return static_cast<std::uint16_t>(i);
        known += (known.empty() ? "" : ", ") + b->voices[i].name;
    }
    error = "voice '" + std::string(name) + "' is not in bundle '" + b->name + "' (available: " + known + ")";
    return std::nullopt;
}

bool AppModel::assign_voice(bool slot_b, std::string_view name, std::string& error) {
    const auto idx = resolve_voice(name, error);
    if (!idx) return false;  // rejected: the previous voice stays
    if (slot_b) {
        voice_b_          = std::string(name);
        params_.voice_b   = *idx;
        settings_.voice_b = voice_b_;
        voice_b_explicit_ = true;
        return true;
    }
    voice_a_          = std::string(name);
    params_.voice_a   = *idx;
    settings_.voice_a = voice_a_;
    if (!voice_b_explicit_ || voice_b_.empty()) {  // B follows A until the user chooses it
        voice_b_          = voice_a_;
        params_.voice_b   = *idx;
        settings_.voice_b = voice_b_;
    }
    return true;
}

bool AppModel::set_voice_a(std::string_view name) {
    std::string err;
    if (!assign_voice(false, name, err)) {
        notify(Notice::Level::Warning, err);
        return false;
    }
    mark_custom();
    push_params();
    return true;
}

bool AppModel::set_voice_b(std::string_view name) {
    std::string err;
    if (!assign_voice(true, name, err)) {
        notify(Notice::Level::Warning, err);
        return false;
    }
    mark_custom();
    push_params();
    return true;
}

void AppModel::reload_presets() {
    presets_.clear();
    preset_warnings_.clear();
    auto load_dir = [&](const fs::path& dir, bool builtin) {
        for (const fs::path& f : list_preset_files(dir)) {
            PresetLoadResult r = load_preset_file(f);
            for (const std::string& w : r.warnings) preset_warnings_.push_back(path_to_utf8(f.filename()) + ": " + w);
            if (!r.ok()) { preset_warnings_.push_back(path_to_utf8(f.filename()) + ": " + r.error); continue; }
            r.preset->builtin = builtin;
            presets_.push_back(std::move(*r.preset));
        }
    };
    if (!paths_.builtin_preset_dir.empty()) load_dir(paths_.builtin_preset_dir, true);
    if (!paths_.user_preset_dir.empty() && paths_.user_preset_dir != paths_.builtin_preset_dir)
        load_dir(paths_.user_preset_dir, false);
    // A user file named like a built-in would be unreachable by name (and the
    // CLI matches names case-insensitively): report it instead of listing two.
    std::vector<std::string> builtin_names;
    for (const Preset& b : presets_)
        if (b.builtin) builtin_names.push_back(b.name);
    std::erase_if(presets_, [&](const Preset& u) {
        if (u.builtin) return false;
        for (const std::string& b : builtin_names)
            if (iequals(b, u.name)) {
                preset_warnings_.push_back(path_to_utf8(u.path.filename()) + ": named like the built-in preset '" + b +
                                           "', so it is hidden; rename it.");
                return true;
            }
        return false;
    });
    // Built-ins first in the documented order, then user presets alphabetically.
    const auto rank = [](const Preset& p) {
        static const char* const order[] = {"Neutral", "Deep", "Bright", "Robot", "Whisper"};
        if (!p.builtin) return 100;
        for (int i = 0; i < 5; ++i)
            if (p.name == order[i]) return i;
        return 50;
    };
    std::stable_sort(presets_.begin(), presets_.end(), [&](const Preset& a, const Preset& b) {
        const int ra = rank(a), rb = rank(b);
        return ra != rb ? ra < rb : a.name < b.name;
    });
}

std::vector<std::string> AppModel::preset_names() const {
    std::vector<std::string> out;
    for (const Preset& p : presets_) out.push_back(p.name);
    return out;
}

bool AppModel::load_preset(std::string_view name) {
    const Preset* found = nullptr;
    for (const Preset& p : presets_)
        if (p.name == name) { found = &p; break; }
    if (!found) {
        notify(Notice::Level::Warning, "No preset named '" + std::string(name) + "'.");
        return false;
    }
    // Translation toggles inside a preset go through the availability check.
    Preset p = *found;
    std::optional<float> translate_req = p.params[param_index(ParamId::Translate)];
    p.params[param_index(ParamId::Translate)].reset();
    apply_preset_params(p, params_);
    std::string voice_error;
    if (p.voice_a && !assign_voice(false, *p.voice_a, voice_error))
        notify(Notice::Level::Warning, "Preset voice '" + *p.voice_a + "' is not in bundle '" + active_bundle_ + "'.");
    if (p.voice_b && !assign_voice(true, *p.voice_b, voice_error))
        notify(Notice::Level::Warning, "Preset voice '" + *p.voice_b + "' is not in bundle '" + active_bundle_ + "'.");
    if (p.src_lang || p.tgt_lang) {
        const std::string s = p.src_lang ? *p.src_lang : src_lang_;
        const std::string t = p.tgt_lang ? *p.tgt_lang : tgt_lang_;
        // A preset's own wait_k wins over the new pair's default.
        if (!change_language_pair(s, t, !p.has(ParamId::WaitK)))
            notify(Notice::Level::Warning, "Preset languages ignored: " + translation_message_);
    }
    if (translate_req && !set_translate(*translate_req >= 0.5f))
        notify(Notice::Level::Info, "Preset asked for translation: " + translation_message_);
    current_preset_  = found->name;
    preset_modified_ = false;
    settings_.last_preset = current_preset_;
    settings_.gain_db     = params_.get(ParamId::Gain);
    settings_.wait_k      = wait_k();
    push_params();
    return true;
}

fs::path AppModel::user_preset_path(std::string_view name) const {
    std::string file(name);
    for (char& c : file)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) c = '_';
    return paths_.user_preset_dir / (file + ".json");
}

std::optional<std::string> AppModel::preset_name_problem(std::string_view name) const {
    if (name.empty()) return std::string("Enter a preset name.");
    for (const Preset& p : presets_)
        if (p.builtin && iequals(p.name, name))
            return "'" + p.name + "' is a built-in preset; choose another name.";
    // Two names can share one file name ("My voice" and "My_voice"): never
    // overwrite a different preset's file. Re-saving the same name is fine.
    const fs::path file = user_preset_path(name);
    for (const Preset& p : presets_)
        if (!p.builtin && p.path == file && p.name != name)
            return "The file " + path_to_utf8(file.filename()) + " already holds the preset '" + p.name + "'.";
    return std::nullopt;
}

bool AppModel::save_preset_as(std::string_view name, bool include_translation) {
    if (const auto problem = preset_name_problem(name)) {
        notify(Notice::Level::Warning, "Preset not saved: " + *problem);
        return false;
    }
    if (paths_.user_preset_dir.empty()) {
        notify(Notice::Level::Error, "No user preset directory is configured (Settings).");
        return false;
    }
    Preset p = include_translation ? preset_from_params(std::string(name), params_, voice_a_, voice_b_, src_lang_, tgt_lang_)
                                   : preset_from_params(std::string(name), params_, voice_a_, voice_b_);
    if (!include_translation)  // a voice preset never changes the mode, Wait-k or languages when loaded
        for (const ParamDesc& d : kParams)
            if (d.group == ParamGroup::Translation) p.params[param_index(d.id)].reset();
    std::string err;
    if (!save_preset_file(user_preset_path(name), p, err)) {
        notify(Notice::Level::Error, err);
        return false;
    }
    reload_presets();
    current_preset_  = std::string(name);
    preset_modified_ = false;
    settings_.last_preset = current_preset_;
    notify(Notice::Level::Info, "Saved preset '" + std::string(name) + "'.");
    return true;
}

// ---------------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------------

bool AppModel::reject_translate(std::string message) {
    translation_message_ = std::move(message);
    if (params_.enabled(ParamId::Translate)) {
        params_.set(ParamId::Translate, 0.0f);
        push_params();
    }
    return false;
}

bool AppModel::set_translate(bool on) {
    // The worker's set_params() refuses translate = 1 without translator models
    // and ModeAWorker drops that result, so the control thread decides first
    // (BEST_PRACTICES 4.8). The GUI loads encoder + vocoder only until Stage 07.
    if (on && !engine_.host_has_translator())
        return reject_translate(engine_.mode_a() ? std::string(kNoTranslatorModelsMessage)
                                                 : std::string("no model loaded: translation needs a bundle with translator models"));
    if (on && !translation_.translator_available()) {
        translation_message_ = translation_.unavailable_reason();
        params_.set(ParamId::Translate, 0.0f);
        settings_.translate = false;
        push_params();
        return false;
    }
    if (on) {
        const PairValidation v = translation_.validate_pair(src_lang_, tgt_lang_);
        if (!v.ok) {
            translation_message_ = v.message;
            return false;
        }
    }
    params_.set(ParamId::Translate, on ? 1.0f : 0.0f);
    settings_.translate  = on;
    translation_message_ = on ? "Translating " + language_pair_label() : "Voice changer (Mode A)";
    push_params();
    return true;
}

bool AppModel::change_language_pair(std::string_view src, std::string_view tgt, bool apply_pair_default) {
    const PairValidation v = translation_.validate_pair(src, tgt);
    if (!v.ok) {
        translation_message_ = v.message;
        return false;
    }
    const bool pair_changed = src != src_lang_ || tgt != tgt_lang_;
    src_lang_ = std::string(src);
    tgt_lang_ = std::string(tgt);
    params_.src_lang_id = translation_.table().by_code(src)->id;
    params_.tgt_lang_id = translation_.table().by_code(tgt)->id;
    if (pair_changed && apply_pair_default) {
        params_.set(ParamId::WaitK, static_cast<float>(translation_.table().default_k(src, tgt)));
        settings_.wait_k = wait_k();
    }
    settings_.src_lang   = src_lang_;
    settings_.tgt_lang   = tgt_lang_;
    translation_message_ = v.message.empty() ? language_pair_label() + " (k=" + std::to_string(wait_k()) + ")" : v.message;
    push_params();
    return true;
}

void AppModel::set_translator_caps(const TranslatorCaps& caps) {
    translation_.set_translator_caps(caps);
    if (caps.known && !caps.adaptive_policy && params_.enabled(ParamId::AdaptivePolicy)) {
        params_.set(ParamId::AdaptivePolicy, 0.0f);
        notify(Notice::Level::Info, "Adaptive policy turned off: the loaded translator declares no readiness output.");
        push_params();
    }
}

std::vector<std::string> AppModel::language_pair_options() const {
    std::vector<std::string> out;
    const auto& langs = translation_.table().langs;
    for (const Lang& s : langs)
        for (const Lang& t : langs)
            if (s.code != t.code) out.push_back(s.code + " -> " + t.code);
    return out;
}

// ---------------------------------------------------------------------------
// Catalogue & studio
// ---------------------------------------------------------------------------

bool AppModel::add_bundle_dir(const fs::path& dir) {
    std::string err;
    if (!catalog_.add_bundle_dir(dir, &err)) {
        notify(Notice::Level::Error, err);
        return false;
    }
    if (active_bundle_.empty())
        for (const BundleInfo& b : catalog_.bundles())
            if (b.valid() && !b.voices.empty()) { static_cast<void>(set_active_bundle(b.name)); break; }
    return true;
}

void AppModel::rescan_models() {
    const std::vector<CatalogueIndex> keep = catalog_.indices();
    catalog_.clear();
    if (!paths_.models_dir.empty()) catalog_.scan_models_root(paths_.models_dir);
    for (const CatalogueIndex& i : keep) catalog_.load_catalogue_index(i.path);
    for (const BundleInfo& b : catalog_.bundles())
        if (!b.valid()) notify(Notice::Level::Warning, "Bundle '" + b.name + "' is invalid: " + b.error);
}

bool AppModel::studio_write_job(std::string& error) {
    if (paths_.studio_dir.empty()) { error = "no studio directory configured"; return false; }
    if (studio_.draft().base_bundle.empty()) studio_.draft().base_bundle = active_bundle_;
    // Job files go into the project the draft owns, never into another one.
    if (const auto conflict = studio_name_conflict()) { error = *conflict; return false; }
    if (!studio_save_draft()) {
        error = "the draft cannot be saved under voice name '" + studio_.draft().voice_name + "'";
        return false;
    }
    return studio_.write_job(paths_.repo_root, paths_.studio_dir, paths_.models_dir, error);
}

std::optional<std::string> AppModel::studio_name_conflict() const {
    const std::string& name = studio_.draft().voice_name;
    if (paths_.studio_dir.empty() || !is_safe_project_name(name) || name == studio_project_) return std::nullopt;
    std::error_code ec;
    if (!fs::exists(paths_.studio_dir / name, ec)) return std::nullopt;
    return "A Voice Studio project named '" + name + "' already exists. Open it, or choose another name.";
}

bool AppModel::studio_open_project(std::string_view name) {
    if (paths_.studio_dir.empty() || !is_safe_project_name(name)) return false;
    VoiceStudioModel loaded;
    loaded.requirements() = studio_.requirements();
    loaded.draft().voice_name = std::string(name);
    if (!loaded.load_draft(paths_.studio_dir)) return false;
    loaded.draft().voice_name = std::string(name);  // the directory is the identity, not the file's copy
    studio_         = std::move(loaded);
    studio_project_ = std::string(name);
    studio_status_.reset();
    return true;
}

bool AppModel::studio_save_draft() {
    const std::string name = studio_.draft().voice_name;
    if (paths_.studio_dir.empty() || studio_.name_problem(nullptr)) return false;  // half-typed or unsafe name
    if (studio_name_conflict()) return false;  // never overwrite another project; the UI offers to open it
    std::string err;
    if (!studio_.save_draft(paths_.studio_dir, err)) { notify(Notice::Level::Warning, err); return false; }
    if (!studio_project_.empty() && studio_project_ != name) {
        // Rename: the draft now lives under the new name. An old directory
        // holding only its draft.json (e.g. the prefix "ale" while typing
        // "alex") is removed; one with job files or status is left intact.
        const fs::path old_dir = paths_.studio_dir / studio_project_;
        std::error_code ec;
        bool only_draft = fs::is_regular_file(old_dir / "draft.json", ec);
        for (const auto& e : fs::directory_iterator(old_dir, ec))
            if (e.path().filename() != "draft.json") { only_draft = false; break; }
        if (only_draft && !ec && fs::remove(old_dir / "draft.json", ec)) fs::remove(old_dir, ec);
    }
    studio_project_ = name;
    return true;
}

// ---------------------------------------------------------------------------
// Tray
// ---------------------------------------------------------------------------

TraySpec AppModel::tray_spec() const {
    TrayInputs in;
    const StreamStatus& st = engine_.status();
    in.stream        = st.state;
    in.engine_on     = engine_requested_;
    in.muted         = muted();
    in.translate     = translate();
    in.translator_available = translation_.translator_available();
    in.window_visible = window_visible_;
    in.preset        = preset_modified_ ? current_preset_ + " (modified)" : current_preset_;
    in.presets       = preset_names();
    in.voice         = voice_a_;
    in.voices        = voice_names();
    in.language_pair = language_pair_label();
    in.language_pairs = language_pair_options();
    in.queued_ms     = engine_.telemetry().queued_ms();
    in.queued_valid  = engine_.telemetry().queued_valid;
    in.retry_attempt = st.retry_attempt;
    if (st.last_error) in.error_text = std::string(to_string(st.last_error->category)) + ": " + st.last_error->message;
    in.mode_label    = mode_label();
    in.device_unresolved = device_unresolved();
    if (in.device_unresolved) {
        const AudioError e = unresolved_device_error();
        in.error_text = std::string(to_string(e.category)) + ": " + e.message;
    }
    return build_tray_spec(in);
}

void AppModel::handle_tray_action(const TrayAction& a) {
    using T = TrayAction::Type;
    switch (a.type) {
        case T::ShowPanel:           show_panel_requested_ = true; break;
        case T::ToggleMiniPanel:     mini_toggle_requested_ = true; break;
        case T::ToggleEngine:        toggle_engine(); break;
        case T::ToggleMute:          toggle_mute(); break;
        case T::SetModeVoiceChanger: static_cast<void>(set_translate(false)); break;
        case T::SetModeTranslate:
            if (!set_translate(true)) notify(Notice::Level::Warning, translation_message_);
            break;
        case T::LoadPreset:          static_cast<void>(load_preset(a.arg)); break;
        case T::SetVoice:
            static_cast<void>(set_voice_a(a.arg));  // an unknown name is reported by set_voice_a
            break;
        case T::SetLanguagePair: {
            const std::size_t arrow = a.arg.find("->");
            if (arrow == std::string::npos) break;
            std::string s = a.arg.substr(0, arrow), t = a.arg.substr(arrow + 2);
            auto trim = [](std::string& x) {
                while (!x.empty() && x.front() == ' ') x.erase(x.begin());
                while (!x.empty() && x.back() == ' ') x.pop_back();
            };
            trim(s); trim(t);
            if (!set_language_pair(s, t)) notify(Notice::Level::Warning, translation_message_);
            break;
        }
        case T::Quit:                quit_requested_ = true; break;
    }
}

// ---------------------------------------------------------------------------
// Settings, status, notices
// ---------------------------------------------------------------------------

bool AppModel::save_settings(std::string* error) {
    std::string err;
    const bool ok = settings_.save(paths_.settings_file(), err);
    if (!ok) {
        if (error) *error = err;
        notify(Notice::Level::Warning, "Settings not saved: " + err);
    }
    return ok;
}

StatusSummary AppModel::status_summary() const {
    StatusSummary s;
    const StreamStatus& st = engine_.status();
    s.state = st.state;
    s.mode  = mode_label();
    switch (st.state) {
        case StreamState::Idle:         s.pill = "Idle"; break;
        case StreamState::Opening:      s.pill = "Opening"; break;
        case StreamState::Running:      s.pill = muted() ? "Running (muted)" : "Running"; break;
        case StreamState::Recovering:   s.pill = "Recovering"; break;
        case StreamState::AwaitingUser: s.pill = "Needs attention"; break;
        case StreamState::Error:        s.pill = "Error"; break;
        case StreamState::Stopping:     s.pill = "Stopping"; break;
    }
    const TelemetrySnapshot t = engine_.telemetry();
    std::ostringstream d;
    const std::string backend = snapshot().backend.empty() ? backend_status().backend : snapshot().backend;
    if (!backend.empty()) d << backend << "  |  ";
    if (st.state == StreamState::Running || st.state == StreamState::Recovering) {
        d << capture_name() << " -> " << playback_name() << "  |  " << t.sample_rate << " Hz  |  queued "
          << t.queued_ms_text() << "  |  " << t.output_underruns << " underruns, " << t.input_overruns << " overruns";



    } else if (st.last_error && (st.state == StreamState::AwaitingUser || st.state == StreamState::Error)) {
        d << explain_error(*st.last_error);
    } else if (device_unresolved()) {
        d << explain_error(unresolved_device_error());
    } else {
        d << "engine stopped";
        if (backend_status().state == BackendState::NullOnly) d << "  |  offline: null backend";
    }
    if (bundle_loading()) {
        s.pill = "Loading model";
        d.str(std::string());
        d << "loading bundle '" << loading_name_ << "'";
    }
    if (pipeline_failed()) {
        s.pill = "Pipeline error";
        d.str(std::string());
        d << "pipeline error: " << pipeline_failure_;
    }
    s.detail = d.str();
    return s;
}

void AppModel::notify(Notice::Level level, std::string text) {
    Notice n;
    n.level      = level;
    n.text       = std::move(text);
    n.at_seconds = uptime_s_;
    if (!notices_.empty() && notices_.back().text == n.text) return;  // collapse repeats
    notices_.push_back(std::move(n));
    while (notices_.size() > 50) notices_.pop_front();
}

} // namespace fluxvoice::frontend
