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
#include "core/Settings.hpp"
#include "core/Telemetry.hpp"
#include "core/TranslationModel.hpp"
#include "core/TrayModel.hpp"
#include "core/VoiceCatalog.hpp"
#include "core/VoiceStudio.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
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
    std::string detail;  // "wasapi - Mic -> CABLE Input - 16 kHz - queued 40 ms - 0 underruns"
};

class AppModel {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    AppModel(AudioBackend& backend, AppPaths paths, Settings settings);

    /// Scan bundles, presets and catalogues, restore the saved state and
    /// device policy, optionally start the engine (Settings).
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
    bool muted() const { return engine_.muted(); }
    void set_muted(bool m);
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
    void select_capture_row(std::size_t row);
    void select_playback_row(std::size_t row);
    void set_capture_mode(CaptureMode m);
    std::optional<std::string> feedback_warning() const;
    /// Start or reconfigure with the pending selections; persists the policy.
    bool apply_devices();
    void refresh_devices();
    Result<EndpointInfo> probe(const DeviceSelection& sel, EndpointDirection dir) const;
    std::string capture_name() const;
    std::string playback_name() const;

    // --- parameters -------------------------------------------------------
    const VoiceParams& params() const { return params_; }
    void set_param(ParamId id, float v);
    void reset_param(ParamId id) { set_param(id, param_desc(id).def); }
    void reset_params();

    // --- voices -----------------------------------------------------------
    const std::string& active_bundle() const { return active_bundle_; }
    bool set_active_bundle(std::string_view name);
    const BundleInfo* active_bundle_info() const { return catalog_.bundle(active_bundle_); }
    std::vector<std::string> voice_names() const { return catalog_.voice_names(active_bundle_); }
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
    bool save_preset_as(std::string_view name);
    const std::vector<std::string>& preset_warnings() const { return preset_warnings_; }
    void reload_presets();

    // --- translation ------------------------------------------------------
    TranslationModel&       translation()       { return translation_; }
    const TranslationModel& translation() const { return translation_; }
    bool translate() const { return params_.enabled(ParamId::Translate); }
    bool set_translate(bool on);
    const std::string& src_lang() const { return src_lang_; }
    const std::string& tgt_lang() const { return tgt_lang_; }
    bool set_language_pair(std::string_view src, std::string_view tgt);
    void swap_languages() {
        const std::string s = tgt_lang_, t = src_lang_;  // copies: set_language_pair rewrites both members
        static_cast<void>(set_language_pair(s, t));
    }
    int  wait_k() const { return static_cast<int>(params_.get(ParamId::WaitK)); }
    void set_wait_k(int k) { set_param(ParamId::WaitK, static_cast<float>(TranslationModel::clamp_wait_k(k))); }
    std::string language_pair_label() const { return src_lang_ + " -> " + tgt_lang_; }
    std::vector<std::string> language_pair_options() const;
    const std::string& translation_message() const { return translation_message_; }

    // --- catalogue & studio -----------------------------------------------
    VoiceCatalog&       catalog()       { return catalog_; }
    const VoiceCatalog& catalog() const { return catalog_; }
    bool add_bundle_dir(const std::filesystem::path& dir);
    void rescan_models();
    VoiceStudioModel&       studio()       { return studio_; }
    const VoiceStudioModel& studio() const { return studio_; }
    bool studio_write_job(std::string& error);
    bool studio_save_draft();
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
    double uptime_seconds() const { return uptime_s_; }
    StatusSummary status_summary() const;

    // --- notices ----------------------------------------------------------
    const std::deque<Notice>& notices() const { return notices_; }
    void notify(Notice::Level level, std::string text);
    void clear_notices() { notices_.clear(); }

private:
    static bool take(bool& flag) { const bool v = flag; flag = false; return v; }

    void push_params();
    void load_bundle_state();
    void restore_device_policy(bool& capture_ok, bool& playback_ok);
    std::string policy_text(const DeviceSelection& sel, EndpointDirection dir) const;
    DeviceSelection selection_from_policy(std::string_view policy, EndpointDirection dir, bool& ok) const;
    std::optional<std::uint16_t> voice_index(std::string_view name) const;
    void mark_custom() { preset_modified_ = !current_preset_.empty(); }

    EngineHost        engine_;
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

    VoiceParams       params_;
    std::string       active_bundle_;
    std::string       voice_a_;
    std::string       voice_b_;
    std::vector<Preset> presets_;
    std::string       current_preset_;
    bool              preset_modified_ = false;
    std::vector<std::string> preset_warnings_;
    std::string       src_lang_ = "en";
    std::string       tgt_lang_ = "es";
    std::string       translation_message_;

    std::optional<StudioStatus> studio_status_;

    bool window_visible_        = true;
    bool mini_panel_            = false;
    bool quit_requested_        = false;
    bool show_panel_requested_  = false;
    bool mini_toggle_requested_ = false;

    History<60>   underruns_;
    History<60>   queued_ms_;
    std::uint64_t last_underruns_ = 0;
    double        uptime_s_       = 0.0;
    double        history_acc_s_  = 0.0;
    double        studio_poll_s_  = 0.0;
    bool          has_last_tick_  = false;
    TimePoint     last_tick_{};
    StreamState   last_state_ = StreamState::Idle;

    std::deque<Notice> notices_;
};

} // namespace fluxvoice::frontend
