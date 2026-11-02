// AppModel — state owner and command implementation.
#include "core/AppModel.hpp"

#include "audio/AudioTypes.hpp"
#include "core/DeviceSelectorModel.hpp"
#include "core/Settings.hpp"
#include "core/TrayModel.hpp"
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

} // namespace

AppModel::AppModel(AudioBackend& backend, AppPaths paths, Settings settings)
    : engine_(backend, default_controller_config()), paths_(std::move(paths)), settings_(std::move(settings)) {}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void AppModel::initialise(TimePoint now) {
    // Bundles and catalogues.
    rescan_models();
    if (!paths_.assets_dir.empty()) {
        const fs::path idx = paths_.assets_dir / "catalogue" / "index.json";
        std::error_code ec;
        if (fs::is_regular_file(idx, ec)) catalog_.load_catalogue_index(idx);
    }
    if (!settings_.catalogue_index.empty()) {
        std::string err;
        if (!catalog_.load_catalogue_index(fs::path(settings_.catalogue_index), &err))
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

    // Voices.
    if (!settings_.voice_a.empty() && !set_voice_a(settings_.voice_a))
        notify(Notice::Level::Warning, "saved voice '" + settings_.voice_a + "' is not in bundle '" + active_bundle_ + "'");
    if (voice_a_.empty()) {
        const auto names = voice_names();
        if (!names.empty()) static_cast<void>(set_voice_a(names.front()));
    }
    if (!settings_.voice_b.empty()) static_cast<void>(set_voice_b(settings_.voice_b));
    if (voice_b_.empty()) voice_b_ = voice_a_;

    // Presets, then the saved control state on top.
    reload_presets();
    params_.set(ParamId::Gain, settings_.gain_db);
    params_.set(ParamId::WaitK, static_cast<float>(settings_.wait_k));
    if (!set_language_pair(settings_.src_lang, settings_.tgt_lang)) {
        src_lang_ = "en";
        tgt_lang_ = "es";
        static_cast<void>(set_language_pair(src_lang_, tgt_lang_));
    }
    if (settings_.translate && !set_translate(true))
        notify(Notice::Level::Info, "translation stays off: " + translation_.unavailable_reason());
    if (!settings_.last_preset.empty()) {
        for (const Preset& p : presets_)
            if (p.name == settings_.last_preset) current_preset_ = p.name;
    }

    // Studio draft (per voice name is unknown until typed; load the most
    // recent draft directory if there is exactly one).
    std::error_code ec;
    if (fs::is_directory(paths_.studio_dir, ec)) {
        std::vector<fs::path> drafts;
        for (const auto& e : fs::directory_iterator(paths_.studio_dir, ec))
            if (e.is_directory(ec) && fs::is_regular_file(e.path() / "draft.json", ec)) drafts.push_back(e.path());
        if (drafts.size() == 1) {
            studio_.draft().voice_name = drafts.front().filename().string();
            studio_.load_draft(paths_.studio_dir);
        }
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
    if (!cap_ok || !play_ok) {
        notify(Notice::Level::Warning,
               "A remembered device is no longer listed. Choose it again in Devices; FluxVoice never "
               "substitutes another device.");
        section_ = Section::Devices;
    } else if (settings_.start_engine_on_launch && backend_usable()) {
        static_cast<void>(start_engine());
    }

    window_visible_ = !settings_.start_minimized;
    push_params();
    last_state_ = engine_.status().state;
}

void AppModel::tick(TimePoint now) {
    double dt = 0.0;
    if (has_last_tick_) dt = std::chrono::duration<double>(now - last_tick_).count();
    has_last_tick_ = true;
    last_tick_     = now;
    if (dt < 0.0 || dt > 5.0) dt = 0.0;
    uptime_s_ += dt;

    engine_.tick(now);

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

    history_acc_s_ += dt;
    if (history_acc_s_ >= 1.0) {
        history_acc_s_ = 0.0;
        const TelemetrySnapshot t = engine_.telemetry();
        underruns_.push(static_cast<float>(t.output_underruns - last_underruns_));
        last_underruns_ = t.output_underruns;
        queued_ms_.push(static_cast<float>(t.queued_ms()));
    }

    studio_poll_s_ += dt;
    if (section_ == Section::Studio && studio_poll_s_ >= 1.0) {
        studio_poll_s_ = 0.0;
        studio_status_ = studio_.poll_status(paths_.studio_dir);
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

bool AppModel::backend_usable() const {
    return backend_status().state == BackendState::Ready;
}

bool AppModel::start_engine() {
    if (!backend_usable()) {
        engine_message_ = backend_status().state == BackendState::NullOnly
                              ? "No audio hardware is reachable (null backend, offline mode)."
                              : "No audio backend could be initialised.";
        notify(Notice::Level::Error, engine_message_);
        return false;
    }
    if (const auto fb = feedback_warning()) {
        engine_message_ = *fb;
        notify(Notice::Level::Error, *fb);
        return false;
    }
    const auto res = engine_.start(capture_sel_, playback_sel_, capture_mode_);
    if (!is_ok(res)) {
        engine_message_   = explain_error(std::get<AudioError>(res));
        engine_requested_ = false;
        notify(Notice::Level::Error, engine_message_);
        return false;
    }
    engine_requested_ = true;
    engine_message_   = "Running: " + capture_name() + " -> " + playback_name();
    last_state_       = engine_.status().state;
    push_params();
    return true;
}

void AppModel::stop_engine() {
    engine_.stop();
    engine_requested_ = false;
    engine_message_   = "Stopped.";
    last_state_       = engine_.status().state;
}

void AppModel::set_muted(bool m) {
    engine_.set_muted(m);
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

void AppModel::select_capture_row(std::size_t row) { capture_sel_ = selection_for_row(capture_rows(), row); }
void AppModel::select_playback_row(std::size_t row) { playback_sel_ = selection_for_row(playback_rows(), row); }

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
    settings_.capture_policy   = policy_text(capture_sel_, EndpointDirection::Capture);
    settings_.playback_policy  = policy_text(playback_sel_, EndpointDirection::Playback);
    settings_.capture_loopback = capture_mode_ == CaptureMode::SystemPlaybackLoopback;
    const bool ok = start_engine();
    static_cast<void>(save_settings());
    return ok;
}

void AppModel::refresh_devices() {
    const auto res = engine_.controller().refresh();
    if (!is_ok(res)) notify(Notice::Level::Warning, "Device list refresh failed: " + explain_error(std::get<AudioError>(res)));
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
    return describe_selection(capture_sel_, snapshot(), EndpointDirection::Capture, capture_mode_);
}

std::string AppModel::playback_name() const {
    if (engine_.status().state == StreamState::Running && !engine_.stream_info().playback_name.empty())
        return engine_.stream_info().playback_name;
    return describe_selection(playback_sel_, snapshot(), EndpointDirection::Playback);
}

std::string AppModel::policy_text(const DeviceSelection& sel, EndpointDirection dir) const {
    if (std::holds_alternative<FollowSystemDefault>(sel)) return "default";
    const std::string name = describe_selection(sel, snapshot(), dir, dir == EndpointDirection::Capture ? capture_mode_
                                                                                                        : CaptureMode::OrdinaryMicrophone);
    return "name:" + name;
}

DeviceSelection AppModel::selection_from_policy(std::string_view policy, EndpointDirection dir, bool& ok) const {
    ok = true;
    constexpr std::string_view kName = "name:";
    if (policy.empty() || policy == "default" || !policy.starts_with(kName)) return FollowSystemDefault{};
    const std::string_view name = policy.substr(kName.size());
    const auto& list = (dir == EndpointDirection::Capture && capture_mode_ == CaptureMode::SystemPlaybackLoopback)
                           ? snapshot().playback : snapshot().endpoints(dir);
    const EndpointDescriptor* found = nullptr;
    int matches = 0;
    for (const EndpointDescriptor& d : list)
        if (d.name == name) { found = &d; ++matches; }
    if (matches != 1) {  // missing or ambiguous (two identical USB mics): ask again
        ok = false;
        return FollowSystemDefault{};
    }
    return ExplicitEndpoint{found->key, snapshot().generation};
}

void AppModel::restore_device_policy(bool& capture_ok, bool& playback_ok) {
    capture_sel_  = selection_from_policy(settings_.capture_policy, EndpointDirection::Capture, capture_ok);
    playback_sel_ = selection_from_policy(settings_.playback_policy, EndpointDirection::Playback, playback_ok);
}

// ---------------------------------------------------------------------------
// Parameters, voices, presets
// ---------------------------------------------------------------------------

void AppModel::push_params() { engine_.push_params(params_); }

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

bool AppModel::set_active_bundle(std::string_view name) {
    const BundleInfo* b = catalog_.bundle(name);
    if (!b || !b->valid() || b->voices.empty()) return false;
    active_bundle_ = std::string(name);
    load_bundle_state();
    voice_a_.clear();
    voice_b_.clear();
    static_cast<void>(set_voice_a(b->voices.front().name));
    voice_b_ = voice_a_;
    params_.voice_b = params_.voice_a;
    if (translate() && !translation_.translator_available()) {
        params_.set(ParamId::Translate, 0.0f);
        notify(Notice::Level::Info, "Translation turned off: " + translation_.unavailable_reason());
    }
    push_params();
    return true;
}

std::optional<std::uint16_t> AppModel::voice_index(std::string_view name) const {
    const BundleInfo* b = active_bundle_info();
    if (!b) return std::nullopt;
    for (std::size_t i = 0; i < b->voices.size(); ++i)
        if (b->voices[i].name == name) return static_cast<std::uint16_t>(i);
    return std::nullopt;
}

bool AppModel::set_voice_a(std::string_view name) {
    const auto idx = voice_index(name);
    if (!idx) return false;
    voice_a_        = std::string(name);
    params_.voice_a = *idx;
    settings_.voice_a = voice_a_;
    if (voice_b_.empty()) { voice_b_ = voice_a_; params_.voice_b = *idx; }
    mark_custom();
    push_params();
    return true;
}

bool AppModel::set_voice_b(std::string_view name) {
    const auto idx = voice_index(name);
    if (!idx) return false;
    voice_b_        = std::string(name);
    params_.voice_b = *idx;
    settings_.voice_b = voice_b_;
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
            for (const std::string& w : r.warnings) preset_warnings_.push_back(f.filename().string() + ": " + w);
            if (!r.ok()) { preset_warnings_.push_back(f.filename().string() + ": " + r.error); continue; }
            r.preset->builtin = builtin;
            presets_.push_back(std::move(*r.preset));
        }
    };
    if (!paths_.builtin_preset_dir.empty()) load_dir(paths_.builtin_preset_dir, true);
    if (!paths_.user_preset_dir.empty() && paths_.user_preset_dir != paths_.builtin_preset_dir)
        load_dir(paths_.user_preset_dir, false);
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
    if (p.voice_a && !set_voice_a(*p.voice_a))
        notify(Notice::Level::Warning, "Preset voice '" + *p.voice_a + "' is not in bundle '" + active_bundle_ + "'.");
    if (p.voice_b && !set_voice_b(*p.voice_b))
        notify(Notice::Level::Warning, "Preset voice '" + *p.voice_b + "' is not in bundle '" + active_bundle_ + "'.");
    if (p.src_lang || p.tgt_lang) {
        const std::string s = p.src_lang ? *p.src_lang : src_lang_;
        const std::string t = p.tgt_lang ? *p.tgt_lang : tgt_lang_;
        if (!set_language_pair(s, t)) notify(Notice::Level::Warning, "Preset languages ignored: " + translation_message_);
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

bool AppModel::save_preset_as(std::string_view name) {
    if (name.empty()) return false;
    if (paths_.user_preset_dir.empty()) {
        notify(Notice::Level::Error, "No user preset directory is configured (Settings).");
        return false;
    }
    Preset p = preset_from_params(std::string(name), params_, voice_a_, voice_b_);
    std::string file = std::string(name);
    for (char& c : file)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) c = '_';
    std::string err;
    if (!save_preset_file(paths_.user_preset_dir / (file + ".json"), p, err)) {
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

bool AppModel::set_translate(bool on) {
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

bool AppModel::set_language_pair(std::string_view src, std::string_view tgt) {
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
    if (pair_changed) {
        params_.set(ParamId::WaitK, static_cast<float>(translation_.table().default_k(src, tgt)));
        settings_.wait_k = wait_k();
    }
    settings_.src_lang   = src_lang_;
    settings_.tgt_lang   = tgt_lang_;
    translation_message_ = v.message.empty() ? language_pair_label() + " (k=" + std::to_string(wait_k()) + ")" : v.message;
    push_params();
    return true;
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
    return studio_.write_job(paths_.repo_root, paths_.studio_dir, paths_.models_dir, error);
}

bool AppModel::studio_save_draft() {
    if (paths_.studio_dir.empty() || studio_.draft().voice_name.empty()) return false;
    std::string err;
    if (!studio_.save_draft(paths_.studio_dir, err)) { notify(Notice::Level::Warning, err); return false; }
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
    in.retry_attempt = st.retry_attempt;
    if (st.last_error) in.error_text = std::string(to_string(st.last_error->category)) + ": " + st.last_error->message;
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
            if (!set_voice_a(a.arg)) notify(Notice::Level::Warning, "Voice '" + a.arg + "' is not in the active bundle.");
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
        d << capture_name() << " -> " << playback_name() << "  |  " << t.sample_rate << " Hz  |  queued ";
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.0f ms", t.queued_ms());
        d << buf << "  |  " << t.output_underruns << " underruns, " << t.input_overruns << " overruns";
    } else if (st.last_error && (st.state == StreamState::AwaitingUser || st.state == StreamState::Error)) {
        d << explain_error(*st.last_error);
    } else {
        d << "engine stopped";
        if (backend_status().state == BackendState::NullOnly) d << "  |  offline: null backend";
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
