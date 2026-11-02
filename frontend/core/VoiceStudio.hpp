// VoiceStudio — the custom-voice creation workflow (frontend/README.md §4.4),
// a guided form of Milestone 2 / Stage 10's recording protocol:
//   1. name & consent   2. setup checklist   3. sessions & takes
//   4. room tone & live-mic set   5. prepare & fine-tune commands   6. install
//
// The model analyses WAV takes (duration, format, peak, RMS, clipping, SNR
// estimate against the quietest blocks), persists the draft, writes the job
// manifest + take list the training scripts consume, and polls status.json.
// No audio is recorded here: the protocol wants the user's own DAW/recorder
// at 48 kHz / 24-bit with a fixed mic position; the Studio checks the result.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

class VoiceCatalog;

struct WavStats {
    bool        ok = false;
    std::string error;
    std::uint32_t sample_rate = 0;
    std::uint16_t channels    = 0;
    std::uint16_t bits        = 0;        // 16/24/32 PCM, 32 float
    bool          is_float    = false;
    double        seconds     = 0.0;
    float         peak_dbfs   = -120.0f;
    float         rms_dbfs    = -120.0f;
    float         noise_floor_dbfs = -120.0f;  // 10th percentile of 10 ms block RMS
    float         speech_dbfs      = -120.0f;  // 90th percentile of 10 ms block RMS
    float         snr_db           = 0.0f;     // speech - noise floor
    std::size_t   clipped_samples  = 0;        // |x| >= 0.999 FS
    bool clipped() const { return clipped_samples > 0; }
};

/// Statistics over interleaved samples (channels averaged to mono for levels).
WavStats analyze_samples(const float* interleaved, std::size_t frames, std::uint16_t channels,
                         std::uint32_t sample_rate);

/// Parse a RIFF/WAVE file (PCM 16/24/32-bit or IEEE float 32, incl. WAVE_FORMAT_EXTENSIBLE)
/// and compute its statistics without loading it whole.
WavStats analyze_wav_file(const std::filesystem::path& path);

/// Why a take is excluded from the clean total, per the Stage 10 rules.
enum class TakeVerdict : std::uint8_t { Clean, Clipped, LowSnr, NotMono, Unreadable, TooShort };
const char* to_string(TakeVerdict v);

struct Take {
    std::filesystem::path path;
    std::string session;  // free text, e.g. "s01"
    WavStats    stats;
    TakeVerdict verdict = TakeVerdict::Unreadable;
};

TakeVerdict judge_take(const WavStats& s, double min_seconds = 2.0, float min_snr_db = 30.0f);

enum class StudioStep : std::uint8_t { NameConsent, Setup, Takes, RoomTone, Prepare, Install, Count };
const char* to_string(StudioStep s);

constexpr std::size_t kSetupItemCount = 7;
extern const std::array<const char*, kSetupItemCount> kSetupItems;
constexpr std::size_t kMaterialItemCount = 6;
extern const std::array<const char*, kMaterialItemCount> kMaterialItems;

struct StudioDraft {
    std::string voice_name;     // lowercase [a-z0-9_-]+
    std::string display_name;
    bool        is_own_voice = true;
    bool        consent      = false;
    std::string consent_note;   // required when !is_own_voice
    std::array<bool, kSetupItemCount>    setup{};
    std::array<bool, kMaterialItemCount> material{};
    std::vector<Take> takes;
    std::vector<Take> room_tone;
    std::vector<Take> live_mic;
    std::string base_bundle;    // bundle whose vocoder is fine-tuned
    std::string target_bundle;  // bundle directory name to export into
};

struct StudioStatus {
    std::string stage;      // prepare|features|finetune|export|done|error
    float       progress = 0.0f;
    std::string message;
    double      eta_s = -1.0;
};

enum class Shell : std::uint8_t { Posix, PowerShell };

struct StudioCommands {
    std::string prepare;
    std::string finetune;
    std::string export_;
};

class VoiceStudioModel {
public:
    static constexpr double kMinimumMinutes = 60.0;
    static constexpr double kIdealMinutes   = 180.0;
    static constexpr double kRoomToneMinimumSeconds = 25.0;
    static constexpr double kLiveMicTargetMinutes   = 10.0;

    /// Thresholds from the Stage 10 protocol; adjustable for dry runs and tests.
    struct Requirements {
        double minimum_minutes   = kMinimumMinutes;
        double room_tone_seconds = kRoomToneMinimumSeconds;
    };
    Requirements&       requirements()       { return req_; }
    const Requirements& requirements() const { return req_; }

    StudioDraft& draft() { return draft_; }
    const StudioDraft& draft() const { return draft_; }

    StudioStep step() const { return step_; }
    void go_to(StudioStep s) { step_ = s; }
    /// True when the current step's requirements are met; `why` explains otherwise.
    bool can_advance(std::string* why = nullptr) const;
    bool next();  // false when !can_advance()
    void prev();

    /// Validation of the voice name against the voices.json rule plus a
    /// collision check against installed voices (nullopt = acceptable).
    std::optional<std::string> name_problem(const VoiceCatalog* installed) const;

    // Takes
    const Take& add_take(const std::filesystem::path& wav, std::string session = {});
    const Take& add_room_tone(const std::filesystem::path& wav);
    const Take& add_live_mic(const std::filesystem::path& wav);
    void remove_take(std::size_t index);
    double clean_minutes() const;
    double excluded_minutes() const;
    double room_tone_seconds() const;
    double live_mic_minutes() const;
    float  progress_to_minimum() const;  // 0..1 against kMinimumMinutes

    // Outputs
    std::filesystem::path project_dir(const std::filesystem::path& studio_root) const;
    StudioCommands commands(const std::filesystem::path& repo_root, const std::filesystem::path& studio_root,
                            const std::filesystem::path& models_root, Shell shell) const;
    std::string job_json(const std::filesystem::path& repo_root, const std::filesystem::path& studio_root,
                         const std::filesystem::path& models_root) const;
    std::string takes_tsv() const;
    /// Writes job.json, takes.tsv and run_finetune.(sh|ps1) into the project dir.
    bool write_job(const std::filesystem::path& repo_root, const std::filesystem::path& studio_root,
                   const std::filesystem::path& models_root, std::string& error) const;
    bool job_written(const std::filesystem::path& studio_root) const;

    // Persistence
    std::string draft_json() const;
    bool load_draft_json(std::string_view text, std::string* error = nullptr);
    bool save_draft(const std::filesystem::path& studio_root, std::string& error) const;
    bool load_draft(const std::filesystem::path& studio_root);

    /// status.json written by the training side (frontend/README.md §9.6).
    std::optional<StudioStatus> poll_status(const std::filesystem::path& studio_root) const;

    /// The voice exists in an installed bundle (step 6 condition).
    bool installed(const VoiceCatalog& catalog) const;

private:
    StudioDraft  draft_;
    StudioStep   step_ = StudioStep::NameConsent;
    Requirements req_{};
};

/// Quote a path for the given shell.
std::string shell_quote(const std::filesystem::path& p, Shell shell);

} // namespace fluxvoice::frontend
