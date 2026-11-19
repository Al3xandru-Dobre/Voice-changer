// AppModel — the single owner of frontend state (frontend/README.md §8).
//
// Views read it and call its commands; the tray and the Mini panel go through
// the same commands, so a preset switched from the tray behaves exactly like
// one switched in the Voices panel. Every method runs on the UI/control
// thread. The model never touches the audio callback; it talks to the worker
// through EngineHost::push_params (SPSC) and reads telemetry from atomics.
#pragma once

#include "audio/AudioBackend.hpp"
#include "audio/AudioTypes.hpp"
#include "core/DeviceSelectorModel.hpp"
#include "core/EngineHost.hpp"
#include "core/LoaderThread.hpp"
#include "core/Settings.hpp"
#include "core/Telemetry.hpp"
#include "core/TranslationModel.hpp"
#include "core/TrayModel.hpp"
#include "core/VoiceCatalog.hpp"
#include "core/VoiceStudio.hpp"
#include "pipeline/ModeAHost.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

enum class Section : std::uint8_t { Live, Devices, Voices, Studio, Translate, Catalogue, Telemetry, Settings, Count };
const char* to_string(Section s);

struct AppPaths {
    std::filesystem::path config_dir;          // settings.json
    std::filesystem::path repo_root;           // python_training/, presets/ (may be empty when installed)
    std::filesystem::path models_dir;          // bundle root
    std::filesystem::path builtin_preset_dir;  // presets/
    std::filesystem::path user_preset_dir;
    std::filesystem::path studio_dir;
    std::filesystem::path assets_dir;          // frontend/assets

    std::filesystem::path settings_file() const { return config_dir / "settings.json"; }
    /// Where "Write stats log" appends when Settings::stats_log_path is empty.
    std::filesystem::path default_stats_log() const { return config_dir / "stats.log"; }
};

struct Notice {
    enum class Level : std::uint8_t { Info, Warning, Error };
    Level       level = Level::Info;
    std::string text;
    double      at_seconds = 0.0;  // model uptime when raised
};

struct StatusSummary {
    StreamState state = StreamState::Idle;
    std::string pill;    // "Running", "Awaiting user", ...
    std::string mode;    // "Bypass - no model loaded", "Mode A - <bundle>" (EngineHost::mode_label())
    std::string detail;  // "wasapi - Mic -> CABLE Input - 16 kHz - queued 40 ms - 0 underruns"
};

/// What "applied" means for the parameter indicator (M4 Stage 05): accepted by
/// the pipeline, not merely delivered to the worker.
enum class ParamApply : std::uint8_t {
    BypassInactive,  // no model loaded: only gain, gate, brightness and limiter act
    Stopped,         // engine stopped; the newest snapshot is applied at start
    Pending,         // pushed, not yet accepted
    Applied,         // accepted_generation == pushed_generation
    Rejected,        // the pipeline refused a snapshot since the newest push
};
const char* to_string(ParamApply a);

/// Outcome of one bundle load, drained once with AppModel::take_bundle_report()
/// (the `--bundle` startup line prints it; the Live notice already told the user).
struct BundleReport {
    bool                  ok = false;
    std::filesystem::path dir;
    std::string           name;           // bundle name (directory name)
    std::uint32_t         rate = 0;       // contract sample rate (ok only)
    std::size_t           hop = 0;        // samples per hop at that rate (ok only)
    std::size_t           voices = 0;     // voices in voices.json (ok only)
    std::string           message;        // the failure text (!ok)
};

class AppModel {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// `loader` builds the models of every bundle (the ONNX Runtime bundle
    /// loader by default; tests inject fakes).
    AppModel(AudioBackend& backend, AppPaths paths, Settings settings,
             std::shared_ptr<const fluxvoice::HostModelLoader> loader = fluxvoice::ort_bundle_loader());
    /// Joins the loader thread (an undrained result is destroyed afterwards).
    ~AppModel();

    AppModel(const AppModel&)            = delete;
    AppModel& operator=(const AppModel&) = delete;

    /// `--bundle`: load this bundle at startup instead of Settings::active_bundle.
    /// Call before initialise().
    void set_startup_bundle(std::filesystem::path dir) { startup_bundle_ = std::move(dir); }

    /// Scan bundles, presets and catalogues, restore the saved state and
    /// device policy, load the saved/`--bundle` bundle on the loader thread,
    /// then optionally start the engine (Settings; after the load result).
    void initialise(TimePoint now = Clock::now());

    /// Per frame: controller tick, 1 Hz telemetry history, studio status poll.
    void tick(TimePoint now = Clock::now());

    // --- navigation -------------------------------------------------------
    Section section() const { return section_; }
    void    navigate(Section s) { section_ = s; }

    // --- engine ----------------------------------------------------------
    EngineHost&       engine()       { return engine_; }
    const EngineHost& engine() const { return engine_; }
    bool engine_on() const { return engine_requested_; }
    bool start_engine();
    void stop_engine();
    void toggle_engine() { engine_on() ? stop_engine() : static_cast<void>(start_engine()); }
    /// Recovery for an exception that escaped a UI frame (BUG-022): stop the
    /// stream in order (worker quiesced, then the device closed), say so in the
    /// engine message and as an Error notice carrying `what`, and leave every
    /// other piece of state (parameters, selections, bundle, settings) as it
    /// was so the frame loop can carry on. Never throws; repeats collapse
    /// into one notice.
    void on_unexpected_exception(std::string_view what) noexcept;
    bool muted() const { return engine_.muted(); }
    void set_muted(bool m);

    // --- model engine (M4 Stage 05) -----------------------------------------
    /// True when a model bundle drives the stream; false is Bypass.
    bool mode_a() const { return engine_.mode_a(); }
    /// "Bypass - no model loaded" / "Mode A - <bundle>".
    std::string mode_label() const { return engine_.mode_label(); }
    /// The pipeline stopped the worker (fatal error): the message, or empty.
    /// The stream is stopped and nothing restarts by itself.
    const std::string& pipeline_failure() const { return pipeline_failure_; }
    bool pipeline_failed() const { return !pipeline_failure_.empty(); }
    /// Live "Retry": clear the failure and start again with the same bundle and devices.
    bool retry_after_failure();
    /// Live "Choose another bundle": open the Catalogue.
    void choose_another_bundle() { navigate(Section::Catalogue); }

    // --- model bundle lifecycle (control thread; the load runs on a loader thread) ---
    /// Load the bundle in `dir`: stop the stream (remembering the device policy
    /// when it was running), build a ModeAHost off-thread (models, warm-up),
    /// and in tick() install it and restart. A failed load keeps the previous
    /// bundle active and reports an Error notice: it never falls back to
    /// Bypass. One load at a time; false when one is running.
    bool load_bundle(const std::filesystem::path& dir);
    /// Select catalogue bundle `name` and load it. `voice` (optional) becomes
    /// voice A (or B with `as_b`) once it is loaded.
    bool activate_bundle(std::string_view name, std::string_view voice = {}, bool as_b = false);
    /// Catalogue / Studio "use this voice": names, never rows. Switches the
    /// bundle first when `bundle` is not the loaded one.
    bool use_voice(std::string_view bundle, std::string_view voice, bool as_b = false);
    /// Load the current bundle again (Settings: frames per step, latency target).
    bool reload_bundle();
    bool bundle_loading() const { return loader_thread_.busy(); }
    /// "identity" while loading `identity`; empty otherwise.
    const std::string& loading_bundle() const { return loading_name_; }
    /// Name of the loaded (Mode A) bundle; empty in Bypass.
    const std::string& loaded_bundle() const { return loaded_bundle_name_; }
    /// The newest unreported load outcome, once.
    std::optional<BundleReport> take_bundle_report();
    void toggle_mute() { set_muted(!muted()); }
    const std::string& engine_message() const { return engine_message_; }
    bool backend_usable() const;

    // --- devices ---------------------------------------------------------
    const DeviceSnapshot& snapshot() const { return engine_.controller().snapshot(); }
    bool snapshot_stale() const { return engine_.controller().snapshot_stale(); }
    BackendStatus     backend_status() const { return engine_.controller().registry().status(); }
    AudioCapabilities capabilities() const { return engine_.controller().registry().backend_capabilities(); }
    bool loopback_supported() const { return capabilities().native_loopback == Capability::Supported; }
    DeviceRows capture_rows() const;
    DeviceRows playback_rows() const;
    const DeviceSelection& capture_selection() const { return capture_sel_; }
    const DeviceSelection& playback_selection() const { return playback_sel_; }
    CaptureMode capture_mode() const { return capture_mode_; }
    /// False (with an error notice, selection unchanged) for a row that is not
    /// in the current list; only the "Follow system default" row selects it.
    bool select_capture_row(std::size_t row);
    bool select_playback_row(std::size_t row);
    void set_capture_mode(CaptureMode m);
    std::optional<std::string> feedback_warning() const;
    /// Start or reconfigure with the pending selections; persists the policy.
    bool apply_devices();
    void refresh_devices();
    Result<EndpointInfo> probe(const DeviceSelection& sel, EndpointDirection dir) const;
    std::string capture_name() const;
    std::string playback_name() const;
    /// A remembered `name:` device that is missing or ambiguous in the current
    /// list (BUG-020). While either is set, start_engine() refuses instead of
    /// following the system default; choosing a row clears it, and a device
    /// list refresh restores the explicit endpoint once the name resolves.
    const std::optional<std::string>& unresolved_capture() const { return unresolved_capture_; }
    const std::optional<std::string>& unresolved_playback() const { return unresolved_playback_; }
    bool device_unresolved() const { return unresolved_capture_.has_value() || unresolved_playback_.has_value(); }

    // --- parameters -------------------------------------------------------
    const VoiceParams& params() const { return params_; }
    /// Drives the indicator next to the sliders: accepted vs. pushed generation.
    ParamApply param_apply() const;
    std::string param_apply_text() const;
    void set_param(ParamId id, float v);
    void reset_param(ParamId id) { set_param(id, param_desc(id).def); }
    void reset_params();

    // --- voices -----------------------------------------------------------
    const std::string& active_bundle() const { return active_bundle_; }
    /// Catalogue selection only; never loads. In Mode A the loaded bundle is the
    /// active one, so another name is refused (use activate_bundle()).
    bool set_active_bundle(std::string_view name);
    const BundleInfo* active_bundle_info() const { return catalog_.bundle(active_bundle_); }
    /// The loaded bundle's voices in Mode A (the host is the authority), else the catalogue's.
    std::vector<std::string> voice_names() const;
    const std::string& voice_a() const { return voice_a_; }
    const std::string& voice_b() const { return voice_b_; }
    bool set_voice_a(std::string_view name);
    bool set_voice_b(std::string_view name);
    const VoiceEntry* voice_entry(std::string_view name) const { return catalog_.find_voice(active_bundle_, name); }

    // --- presets ----------------------------------------------------------
    const std::vector<Preset>& presets() const { return presets_; }
    std::vector<std::string> preset_names() const;
    const std::string& current_preset() const { return current_preset_; }
    bool preset_modified() const { return preset_modified_; }
    bool load_preset(std::string_view name);
    /// Writes a user preset. By default only voice parameters are saved
    /// (pitch, timbre, output, voices); the translation group (translate,
    /// wait_k, adaptive_policy) and the language pair only when
    /// `include_translation` is set (M3 / Stage 09 rule). Refused, with a
    /// notice, for any name preset_name_problem() reports.
    bool save_preset_as(std::string_view name, bool include_translation = false);
    /// Why `name` cannot be saved: empty, the name of a built-in preset
    /// (case-insensitive, so it cannot shadow one), or the file name of a
    /// different user preset. nullopt when it is acceptable.
    std::optional<std::string> preset_name_problem(std::string_view name) const;
    const std::vector<std::string>& preset_warnings() const { return preset_warnings_; }
    void reload_presets();

    // --- translation ------------------------------------------------------
    TranslationModel&       translation()       { return translation_; }
    const TranslationModel& translation() const { return translation_; }
    bool translate() const { return params_.enabled(ParamId::Translate); }
    bool set_translate(bool on);
    const std::string& src_lang() const { return src_lang_; }
    const std::string& tgt_lang() const { return tgt_lang_; }
    /// A user-chosen pair change applies the pair's default Wait-k.
    bool set_language_pair(std::string_view src, std::string_view tgt) { return change_language_pair(src, tgt, true); }
    void swap_languages() {
        const std::string s = tgt_lang_, t = src_lang_;  // copies: set_language_pair rewrites both members
        static_cast<void>(set_language_pair(s, t));
    }
    int  wait_k() const { return static_cast<int>(params_.get(ParamId::WaitK)); }
    void set_wait_k(int k) { set_param(ParamId::WaitK, static_cast<float>(TranslationModel::clamp_wait_k(k))); }
    std::string language_pair_label() const { return src_lang_ + " -> " + tgt_lang_; }
    std::vector<std::string> language_pair_options() const;
    const std::string& translation_message() const { return translation_message_; }
    /// Called by the engine host once the bundle's translator contract is
    /// read (M4 Stage 05 / 07). A bundle that declares no adaptive readiness
    /// turns adaptive_policy off; a bundle change resets the caps to unknown.
    void set_translator_caps(const TranslatorCaps& caps);

    // --- catalogue & studio -----------------------------------------------
    VoiceCatalog&       catalog()       { return catalog_; }
    const VoiceCatalog& catalog() const { return catalog_; }
    bool add_bundle_dir(const std::filesystem::path& dir);
    void rescan_models();
    VoiceStudioModel&       studio()       { return studio_; }
    const VoiceStudioModel& studio() const { return studio_; }
    bool studio_write_job(std::string& error);
    /// Persist the draft under studio_dir / voice_name (BUG-021). Refuses a
    /// name that fails is_safe_project_name() and never writes over another
    /// project's draft (see studio_name_conflict). Renaming the current
    /// project moves its draft, so typing a name produces one directory.
    bool studio_save_draft();
    /// The typed voice name belongs to another project already on disk;
    /// the UI offers studio_open_project() instead of overwriting it.
    std::optional<std::string> studio_name_conflict() const;
    bool studio_open_project(std::string_view name);
    /// Project whose directory holds the current draft ("" = not saved yet).
    const std::string& studio_project() const { return studio_project_; }
    std::optional<StudioStatus> studio_status() const { return studio_status_; }

    // --- tray & window intents --------------------------------------------
    TraySpec tray_spec() const;
    void handle_tray_action(const TrayAction& a);
    bool window_visible() const { return window_visible_; }
    void set_window_visible(bool v) { window_visible_ = v; }
    bool mini_panel() const { return mini_panel_; }
    void set_mini_panel(bool on) { mini_panel_ = on; }
    bool take_quit_request()       { return take(quit_requested_); }
    bool take_show_panel_request() { return take(show_panel_requested_); }
    bool take_mini_toggle_request(){ return take(mini_toggle_requested_); }

    // --- settings ---------------------------------------------------------
    Settings&       settings()       { return settings_; }
    const Settings& settings() const { return settings_; }
    const AppPaths& paths() const { return paths_; }
    bool save_settings(std::string* error = nullptr);

    // --- telemetry & status -----------------------------------------------
    TelemetrySnapshot telemetry() const { return engine_.telemetry(); }
    const History<60>& underrun_history() const { return underruns_; }
    const History<60>& queued_history() const { return queued_ms_; }
    /// Whole-hop compute p95 in ms, one point per second (0 in Bypass).
    const History<60>& compute_history() const { return compute_p95_ms_; }
    double uptime_seconds() const { return uptime_s_; }
    StatusSummary status_summary() const;

    // --- stream clock & stats log (M4 Stage 06) -----------------------------
    /// Seconds since the current stream started: rebased by every start,
    /// reconfigure and recovery (the counters rebaseline then too, BUG-024);
    /// 0 while no stream is running. This is `t` of the `[stats]` line.
    double stream_seconds() const { return stream_running_ ? uptime_s_ - stream_t0_ : 0.0; }
    /// `telemetry().stats_line(stream_seconds())`: the Copy button's line.
    std::string stats_line() const { return engine_.telemetry().stats_line(stream_seconds()); }
    /// The file "Write stats log" appends to: Settings::stats_log_path, or
    /// `<config dir>/stats.log` when it is empty.
    std::filesystem::path stats_log_file() const;
    /// The log file is open (a stream runs and logging is on and healthy).
    bool stats_log_active() const { return stats_log_.is_open(); }
    /// Lines written to the log, and times it was opened (once per stream).
    std::uint64_t stats_log_lines() const { return stats_lines_; }
    std::uint64_t stats_log_opens() const { return stats_opens_; }

    // --- notices ----------------------------------------------------------
    const std::deque<Notice>& notices() const { return notices_; }
    void notify(Notice::Level level, std::string text);
    void clear_notices() { notices_.clear(); }

private:
    /// What the loader thread hands back: a loaded, unattached host, or the error.
    struct BundleLoadResult {
        std::unique_ptr<fluxvoice::ModeAHost> host;
        std::filesystem::path                 dir;
        std::string                           error;
    };

    static bool take(bool& flag) { const bool v = flag; flag = false; return v; }

    void push_params();
    /// `apply_pair_default` false keeps the current Wait-k: used when a saved
    /// value or a preset's wait_k must win over the pair default.
    bool change_language_pair(std::string_view src, std::string_view tgt, bool apply_pair_default);
    std::filesystem::path user_preset_path(std::string_view name) const;
    void load_bundle_state();
    void restore_device_policy(bool& capture_ok, bool& playback_ok);
    void retry_unresolved_devices();
    AudioError unresolved_device_error() const;
    int count_named(std::string_view name, EndpointDirection dir) const;
    /// Persistable policy text, or nullopt when an explicit endpoint is no
    /// longer listed (its name is unknown and must not be invented).
    std::optional<std::string> policy_text(const DeviceSelection& sel, EndpointDirection dir) const;
    DeviceSelection selection_from_policy(std::string_view policy, EndpointDirection dir, bool& ok) const;
    /// Index of voice `name` on the control thread: the loaded host's table in
    /// Mode A (`ModeAHost::voice_index`), else the catalogue's. nullopt with an
    /// error naming the choices.
    std::optional<std::uint16_t> resolve_voice(std::string_view name, std::string& error) const;
    /// Set voice A/B by name. An unknown name changes nothing and fills `error`.
    bool assign_voice(bool slot_b, std::string_view name, std::string& error);
    /// Set `params_.translate` off with the reason and report failure.
    bool reject_translate(std::string message);
    void handle_pipeline_failure();
    /// What start_engine() does with explicit selections (the load/resume path
    /// replays the ones remembered when the stream was stopped).
    bool start_engine_with(const DeviceSelection& cap, const DeviceSelection& play, CaptureMode mode);
    void finish_bundle_load(BundleLoadResult r);
    /// After a host was installed: re-resolve voice A/B by name against its
    /// table (unknown -> first voice, with a notice), refresh the translator state.
    void sync_to_installed_host(std::string_view wanted_voice, bool wanted_as_b);
    void update_stream_clock(StreamState st);
    /// Append one line (opening the file first when needed); a failure
    /// disables logging with an Error notice.
    void append_stats_line(const TelemetrySnapshot& t);
    void close_stats_log();
    void fail_stats_log(const std::string& what);
    void mark_custom() { preset_modified_ = !current_preset_.empty(); }

    EngineHost        engine_;
    // Declared after engine_: destroyed first, so the loader thread is joined
    // while the engine (and the loader it shares) still exist.
    LoaderThread<BundleLoadResult> loader_thread_;
    AppPaths          paths_;
    Settings          settings_;
    VoiceCatalog      catalog_;
    TranslationModel  translation_;
    VoiceStudioModel  studio_;

    Section           section_ = Section::Live;
    bool              engine_requested_ = false;
    std::string       engine_message_;
    DeviceSelection   capture_sel_{FollowSystemDefault{}};
    DeviceSelection   playback_sel_{FollowSystemDefault{}};
    CaptureMode       capture_mode_ = CaptureMode::OrdinaryMicrophone;
    std::optional<std::string> unresolved_capture_;   // remembered name, not resolvable now
    std::optional<std::string> unresolved_playback_;
    std::uint64_t     seen_generation_ = 0;           // snapshot generation last checked for them

    VoiceParams       params_;
    std::uint64_t     rejected_at_push_ = 0;   // engine_.rejected_params() when the newest snapshot was pushed
    std::string       active_bundle_;
    std::string       voice_a_;
    std::string       voice_b_;
    bool              voice_b_explicit_ = false;  // voice B follows voice A until the user chooses it
    std::vector<Preset> presets_;
    std::string       current_preset_;
    bool              preset_modified_ = false;
    // Bundle lifecycle. `resume_*` is the device policy remembered when a load
    // stopped a running stream.
    struct Resume {
        bool            valid = false;
        DeviceSelection capture{FollowSystemDefault{}};
        DeviceSelection playback{FollowSystemDefault{}};
        CaptureMode     mode = CaptureMode::OrdinaryMicrophone;
    };
    Resume                       resume_;
    std::filesystem::path        startup_bundle_;     // --bundle
    bool                         start_after_load_ = false;  // Settings "start on launch", deferred behind the load
    std::filesystem::path        loading_dir_;
    std::string                  loading_name_;
    std::string                  loaded_bundle_name_;
    std::filesystem::path        loaded_bundle_dir_;
    std::string                  pending_voice_;      // voice to select once the load installs
    bool                         pending_voice_b_ = false;
    std::optional<BundleReport>  bundle_report_;
    std::string       pipeline_failure_;       // fatal pipeline error text; empty = none
    bool              failure_ack_ = false;    // the host's failed() latch was handled (until it clears)
    std::vector<std::string> preset_warnings_;
    std::string       src_lang_ = "en";
    std::string       tgt_lang_ = "es";
    std::string       translation_message_;

    std::optional<StudioStatus> studio_status_;
    std::string       studio_project_;

    bool window_visible_        = true;
    bool mini_panel_            = false;
    bool quit_requested_        = false;
    bool show_panel_requested_  = false;
    bool mini_toggle_requested_ = false;

    History<60>   underruns_;
    History<60>   queued_ms_;
    History<60>   compute_p95_ms_;
    std::uint64_t last_underruns_ = 0;
    double        uptime_s_       = 0.0;
    double        history_acc_s_  = 0.0;
    double        studio_poll_s_  = 0.0;
    bool          has_last_tick_  = false;
    TimePoint     last_tick_{};
    StreamState   last_state_ = StreamState::Idle;

    // Stream clock and stats log. The file is opened (append) lazily when the
    // first line of a stream is due and closed when the stream ends or is
    // rebased; never reopened per line.
    bool          stream_running_       = false;
    bool          stream_epoch_pending_ = false;  // a start/reconfigure happened: rebase at the next tick
    double        stream_t0_            = 0.0;
    double        stats_acc_s_          = 0.0;
    std::ofstream stats_log_;
    std::filesystem::path stats_open_path_;
    std::uint64_t stats_lines_ = 0;
    std::uint64_t stats_opens_ = 0;

    std::deque<Notice> notices_;
};

} // namespace fluxvoice::frontend
