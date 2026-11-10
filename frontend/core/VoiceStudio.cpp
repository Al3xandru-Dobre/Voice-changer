// VoiceStudio — WAV analysis, wizard rules, job manifest, draft persistence.
#include "core/VoiceStudio.hpp"

#include "core/VoiceCatalog.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fluxvoice::frontend {

namespace fs = std::filesystem;
using nlohmann::json;

const std::array<const char*, kSetupItemCount> kSetupItems = {
    "One microphone (USB, or XLR + interface) with a pop filter",
    "Fixed distance of 15-30 cm, same position and gain in every session",
    "Headphones for monitoring, never speakers",
    "Quiet, dry room (soft furnishings, closet or reflection filter)",
    "Recorder set to 48 kHz / 24-bit WAV, mono",
    "Peaks around -6 dBFS, never clipping",
    "Sessions of at most 45 minutes, with breaks and water",
};

const std::array<const char*, kMaterialItemCount> kMaterialItems = {
    "CMU ARCTIC prompts",
    "Harvard sentences",
    "Read passages",
    "Questions and exclamations",
    "Emotional reads (happy, sad, emphatic)",
    "10-15 min of spontaneous speech",
};

const char* to_string(TakeVerdict v) {
    switch (v) {
        case TakeVerdict::Clean:      return "clean";
        case TakeVerdict::Clipped:    return "clipped (sample >= 0.999 FS)";
        case TakeVerdict::LowSnr:     return "SNR below 30 dB";
        case TakeVerdict::NotMono:    return "not mono";
        case TakeVerdict::Unreadable: return "unreadable";
        case TakeVerdict::TooShort:   return "shorter than 2 s";
    }
    return "?";
}

const char* to_string(StudioStep s) {
    switch (s) {
        case StudioStep::NameConsent: return "Name & consent";
        case StudioStep::Setup:       return "Setup checklist";
        case StudioStep::Takes:       return "Sessions & takes";
        case StudioStep::RoomTone:    return "Room tone & live-mic set";
        case StudioStep::Prepare:     return "Prepare & fine-tune";
        case StudioStep::Install:     return "Install";
        case StudioStep::Count:       break;
    }
    return "?";
}

// ---------------------------------------------------------------------------
// WAV analysis
// ---------------------------------------------------------------------------

namespace {

float to_db(double lin) { return lin <= 1e-9 ? -120.0f : static_cast<float>(20.0 * std::log10(lin)); }

/// Streaming accumulator over mono samples: peak, energy, clipping and 10 ms
/// block RMS values for the percentile-based SNR estimate.
class StatsAccumulator {
public:
    explicit StatsAccumulator(std::uint32_t sample_rate)
        : block_len_(std::max<std::size_t>(1, sample_rate / 100)) {}

    void push(float x) {
        const float a = std::fabs(x);
        if (a > peak_) peak_ = a;
        if (a >= 0.999f) ++clipped_;
        const double sq = static_cast<double>(x) * static_cast<double>(x);
        sum_sq_ += sq;
        block_sq_ += sq;
        ++n_;
        if (++block_fill_ == block_len_) flush_block();
    }

    WavStats finish(std::uint32_t sample_rate, std::uint16_t channels) {
        if (block_fill_ > 0) flush_block();
        WavStats s;
        s.ok          = true;
        s.sample_rate = sample_rate;
        s.channels    = channels;
        s.seconds     = sample_rate == 0 ? 0.0 : static_cast<double>(n_) / sample_rate;
        s.peak_dbfs   = to_db(peak_);
        s.rms_dbfs    = n_ == 0 ? -120.0f : to_db(std::sqrt(sum_sq_ / static_cast<double>(n_)));
        s.clipped_samples = clipped_;
        if (!blocks_.empty()) {
            std::sort(blocks_.begin(), blocks_.end());
            const auto pct = [&](double p) {
                const std::size_t i = static_cast<std::size_t>(p * static_cast<double>(blocks_.size() - 1) + 0.5);
                return blocks_[std::min(i, blocks_.size() - 1)];
            };
            s.noise_floor_dbfs = to_db(pct(0.10));
            s.speech_dbfs      = to_db(pct(0.90));
            s.snr_db           = s.speech_dbfs - s.noise_floor_dbfs;
        }
        return s;
    }

private:
    void flush_block() {
        blocks_.push_back(std::sqrt(block_sq_ / static_cast<double>(block_fill_)));
        block_sq_   = 0.0;
        block_fill_ = 0;
    }

    std::size_t block_len_;
    std::size_t block_fill_ = 0;
    double      block_sq_   = 0.0;
    std::vector<double> blocks_;
    float       peak_    = 0.0f;
    double      sum_sq_  = 0.0;
    std::size_t clipped_ = 0;
    std::size_t n_       = 0;
};

std::uint32_t le32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint16_t le16(const unsigned char* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

float decode_sample(const unsigned char* p, std::uint16_t bits, bool is_float) {
    if (is_float) {
        float f = 0.0f;
        std::memcpy(&f, p, 4);
        return f;
    }
    switch (bits) {
        case 16: {
            const std::int16_t v = static_cast<std::int16_t>(le16(p));
            return static_cast<float>(v) / 32768.0f;
        }
        case 24: {
            std::int32_t v = static_cast<std::int32_t>(p[0]) | (static_cast<std::int32_t>(p[1]) << 8) |
                             (static_cast<std::int32_t>(p[2]) << 16);
            if (v & 0x800000) v |= static_cast<std::int32_t>(0xFF000000u);
            return static_cast<float>(v) / 8388608.0f;
        }
        case 32: {
            const std::int32_t v = static_cast<std::int32_t>(le32(p));
            return static_cast<float>(static_cast<double>(v) / 2147483648.0);
        }
        default:
            return 0.0f;
    }
}

} // namespace

WavStats analyze_samples(const float* interleaved, std::size_t frames, std::uint16_t channels,
                         std::uint32_t sample_rate) {
    if (channels == 0 || sample_rate == 0) {
        WavStats s;
        s.error = "invalid format";
        return s;
    }
    StatsAccumulator acc(sample_rate);
    for (std::size_t i = 0; i < frames; ++i) {
        float m = 0.0f;
        for (std::uint16_t c = 0; c < channels; ++c) m += interleaved[i * channels + c];
        acc.push(m / static_cast<float>(channels));
    }
    return acc.finish(sample_rate, channels);
}

WavStats analyze_wav_file(const fs::path& path) {
    WavStats s;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        s.error = "cannot open file";
        return s;
    }
    unsigned char hdr[12];
    if (!f.read(reinterpret_cast<char*>(hdr), 12) || std::memcmp(hdr, "RIFF", 4) != 0 ||
        std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        s.error = "not a RIFF/WAVE file";
        return s;
    }
    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t sample_rate = 0;
    bool have_fmt = false;
    while (f) {
        unsigned char ch[8];
        if (!f.read(reinterpret_cast<char*>(ch), 8)) break;
        const std::uint32_t size = le32(ch + 4);
        if (std::memcmp(ch, "fmt ", 4) == 0) {
            std::vector<unsigned char> fmt(size);
            if (size < 16 || !f.read(reinterpret_cast<char*>(fmt.data()), size)) {
                s.error = "truncated fmt chunk";
                return s;
            }
            format      = le16(fmt.data());
            channels    = le16(fmt.data() + 2);
            sample_rate = le32(fmt.data() + 4);
            bits        = le16(fmt.data() + 14);
            if (format == 0xFFFE && size >= 26) format = le16(fmt.data() + 24);  // extensible: sub-format
            have_fmt = true;
            if (size & 1u) f.seekg(1, std::ios::cur);
        } else if (std::memcmp(ch, "data", 4) == 0) {
            if (!have_fmt) {
                s.error = "data chunk before fmt chunk";
                return s;
            }
            const bool is_float = (format == 3);
            if (!((format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (is_float && bits == 32))) {
                s.error = "unsupported sample format (need PCM 16/24/32 or float 32)";
                return s;
            }
            if (channels == 0 || sample_rate == 0) {
                s.error = "invalid fmt chunk";
                return s;
            }
            const std::size_t frame_bytes = static_cast<std::size_t>(channels) * (bits / 8u);
            StatsAccumulator acc(sample_rate);
            std::vector<unsigned char> buf(frame_bytes * 4096);
            std::size_t remaining = size;
            while (remaining > 0 && f) {
                const std::size_t want = std::min(buf.size(), remaining - (remaining % frame_bytes));
                if (want == 0) break;
                f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(want));
                const std::size_t got = static_cast<std::size_t>(f.gcount());
                if (got == 0) break;
                remaining -= got;
                for (std::size_t off = 0; off + frame_bytes <= got; off += frame_bytes) {
                    float m = 0.0f;
                    for (std::uint16_t c = 0; c < channels; ++c)
                        m += decode_sample(buf.data() + off + c * (bits / 8u), bits, is_float);
                    acc.push(m / static_cast<float>(channels));
                }
            }
            s          = acc.finish(sample_rate, channels);
            s.bits     = bits;
            s.is_float = is_float;
            return s;
        } else {
            f.seekg(static_cast<std::streamoff>(size + (size & 1u)), std::ios::cur);
        }
    }
    s.error = have_fmt ? "no data chunk" : "no fmt chunk";
    return s;
}

TakeVerdict judge_take(const WavStats& s, double min_seconds, float min_snr_db) {
    if (!s.ok) return TakeVerdict::Unreadable;
    if (s.channels != 1) return TakeVerdict::NotMono;
    if (s.seconds < min_seconds) return TakeVerdict::TooShort;
    if (s.clipped()) return TakeVerdict::Clipped;
    if (s.snr_db < min_snr_db) return TakeVerdict::LowSnr;
    return TakeVerdict::Clean;
}

// ---------------------------------------------------------------------------
// Wizard
// ---------------------------------------------------------------------------

bool VoiceStudioModel::can_advance(std::string* why) const {
    auto fail = [&](const char* msg) {
        if (why) *why = msg;
        return false;
    };
    switch (step_) {
        case StudioStep::NameConsent:
            if (!is_valid_voice_name(draft_.voice_name)) return fail("Voice name must be lowercase letters, digits, '-' or '_'.");
            if (draft_.display_name.empty()) return fail("Give the voice a display name.");
            if (!draft_.consent) return fail("Confirm you have the right to use this voice.");
            if (!draft_.is_own_voice && draft_.consent_note.empty())
                return fail("For another person's voice, record where the written consent is kept.");
            return true;
        case StudioStep::Setup:
            for (bool b : draft_.setup)
                if (!b) return fail("Tick every setup item; bad audio cannot be fixed in training.");
            return true;
        case StudioStep::Takes:
            if (clean_minutes() < req_.minimum_minutes) {
                if (why) *why = "At least " + std::to_string(static_cast<int>(req_.minimum_minutes)) +
                                " clean minutes are needed (180 is ideal).";
                return false;
            }
            return true;
        case StudioStep::RoomTone:
            if (room_tone_seconds() < req_.room_tone_seconds)
                return fail("Add at least 30 s of room tone (the SNR check depends on it).");
            return true;
        case StudioStep::Prepare:
            return true;  // the job can be re-written any time; Install checks the result
        case StudioStep::Install:
        case StudioStep::Count:
            return false;
    }
    return false;
}

bool VoiceStudioModel::next() {
    if (!can_advance() || step_ == StudioStep::Install) return false;
    step_ = static_cast<StudioStep>(static_cast<std::uint8_t>(step_) + 1);
    return true;
}

void VoiceStudioModel::prev() {
    if (step_ == StudioStep::NameConsent) return;
    step_ = static_cast<StudioStep>(static_cast<std::uint8_t>(step_) - 1);
}

std::optional<std::string> VoiceStudioModel::name_problem(const VoiceCatalog* installed) const {
    if (!is_valid_voice_name(draft_.voice_name))
        return std::string("use lowercase letters, digits, '-' or '_' (voices.json rule)");
    if (installed) {
        for (const BundleInfo& b : installed->bundles())
            for (const VoiceEntry& v : b.voices)
                if (v.name == draft_.voice_name) return "a voice named '" + v.name + "' already exists in bundle '" + b.name + "'";
    }
    return std::nullopt;
}

namespace {

Take make_take(const fs::path& wav, std::string session) {
    Take t;
    t.path    = wav;
    t.session = std::move(session);
    t.stats   = analyze_wav_file(wav);
    t.verdict = judge_take(t.stats);
    return t;
}

double minutes_of(const std::vector<Take>& takes, bool clean_only) {
    double s = 0.0;
    for (const Take& t : takes)
        if (!clean_only || t.verdict == TakeVerdict::Clean) s += t.stats.seconds;
    return s / 60.0;
}

} // namespace

const Take& VoiceStudioModel::add_take(const fs::path& wav, std::string session) {
    draft_.takes.push_back(make_take(wav, std::move(session)));
    return draft_.takes.back();
}

const Take& VoiceStudioModel::add_room_tone(const fs::path& wav) {
    Take t = make_take(wav, "room-tone");
    // Room tone is silence: the SNR rule does not apply, only readability/mono.
    if (t.verdict == TakeVerdict::LowSnr || t.verdict == TakeVerdict::TooShort) t.verdict = TakeVerdict::Clean;
    draft_.room_tone.push_back(std::move(t));
    return draft_.room_tone.back();
}

const Take& VoiceStudioModel::add_live_mic(const fs::path& wav) {
    draft_.live_mic.push_back(make_take(wav, "live-mic"));
    return draft_.live_mic.back();
}

void VoiceStudioModel::remove_take(std::size_t index) {
    if (index < draft_.takes.size()) draft_.takes.erase(draft_.takes.begin() + static_cast<std::ptrdiff_t>(index));
}

double VoiceStudioModel::clean_minutes() const { return minutes_of(draft_.takes, true); }

double VoiceStudioModel::excluded_minutes() const {
    return minutes_of(draft_.takes, false) - minutes_of(draft_.takes, true);
}

double VoiceStudioModel::room_tone_seconds() const {
    double s = 0.0;
    for (const Take& t : draft_.room_tone)
        if (t.verdict == TakeVerdict::Clean) s += t.stats.seconds;
    return s;
}

double VoiceStudioModel::live_mic_minutes() const { return minutes_of(draft_.live_mic, false); }

float VoiceStudioModel::progress_to_minimum() const {
    const double p = req_.minimum_minutes <= 0.0 ? 1.0 : clean_minutes() / req_.minimum_minutes;
    return static_cast<float>(p > 1.0 ? 1.0 : p);
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

std::string shell_quote(const fs::path& p, Shell shell) {
    const std::string s = p.string();
    if (shell == Shell::PowerShell) {
        std::string out = "'";
        for (char c : s) { if (c == '\'') out += "''"; else out += c; }
        return out + "'";
    }
    std::string out = "'";
    for (char c : s) { if (c == '\'') out += "'\\''"; else out += c; }
    return out + "'";
}

fs::path VoiceStudioModel::project_dir(const fs::path& studio_root) const {
    return studio_root / (draft_.voice_name.empty() ? std::string("untitled") : draft_.voice_name);
}

StudioCommands VoiceStudioModel::commands(const fs::path& repo_root, const fs::path& studio_root,
                                          const fs::path& models_root, Shell shell) const {
    const fs::path training = repo_root / "python_training";
    const fs::path python   = shell == Shell::PowerShell ? training / ".venv" / "Scripts" / "python.exe"
                                                         : training / ".venv" / "bin" / "python";
    const fs::path project  = project_dir(studio_root);
    const std::string target = draft_.target_bundle.empty() ? draft_.voice_name + "-bundle" : draft_.target_bundle;
    const fs::path bundle_dir = models_root / target;
    const fs::path ckpt_dir   = repo_root / "checkpoints" / ("vocoder_" + draft_.voice_name);
    auto q = [&](const fs::path& p) { return shell_quote(p, shell); };

    StudioCommands c;
    // Stage 10 preprocessing: HPF, -23 LUFS, VAD 2-12 s, clip/SNR rejection, resample, manifest, 5 min hold-out.
    c.prepare = q(python) + " " + q(training / "prepare_target_voice.py") + " --voice " + draft_.voice_name +
                " --takes " + q(project / "takes.tsv") + " --room-tone " + q(project / "room_tone.tsv") +
                " --out " + q(project / "prepared") + " --holdout-minutes 5";
    // Stage 10 fine-tune: new speaker row initialised from the nearest row, 50/50 own/public batches, perturbation.
    c.finetune = q(python) + " " + q(training / "train_unit_vocoder.py") + " --fine-tune --speaker " +
                 draft_.voice_name + " --data " + q(project / "prepared") +
                 (draft_.base_bundle.empty() ? std::string() : " --base-bundle " + q(models_root / draft_.base_bundle)) +
                 " --out " + q(ckpt_dir);
    // Stage 06 export into a new bundle version, private licence.
    c.export_ = q(python) + " " + q(training / "export_stateful_onnx.py") + " --role vocoder --checkpoint " +
                q(ckpt_dir / "best.pt") + " --bundle " + q(bundle_dir) +
                " --license \"private - do not distribute\"";
    return c;
}

namespace {

json take_to_json(const Take& t) {
    json j;
    j["path"]        = t.path.string();
    j["session"]     = t.session;
    j["verdict"]     = to_string(t.verdict);
    j["ok"]          = t.stats.ok;
    j["error"]       = t.stats.error;
    j["sample_rate"] = t.stats.sample_rate;
    j["channels"]    = t.stats.channels;
    j["bits"]        = t.stats.bits;
    j["is_float"]    = t.stats.is_float;
    j["seconds"]     = t.stats.seconds;
    j["peak_dbfs"]   = t.stats.peak_dbfs;
    j["rms_dbfs"]    = t.stats.rms_dbfs;
    j["noise_floor_dbfs"] = t.stats.noise_floor_dbfs;
    j["speech_dbfs"]      = t.stats.speech_dbfs;
    j["snr_db"]           = t.stats.snr_db;
    j["clipped_samples"]  = t.stats.clipped_samples;
    return j;
}

TakeVerdict verdict_from_string(std::string_view s) {
    for (int i = 0; i <= static_cast<int>(TakeVerdict::TooShort); ++i) {
        const auto v = static_cast<TakeVerdict>(i);
        if (s == to_string(v)) return v;
    }
    return TakeVerdict::Unreadable;
}

Take take_from_json(const json& j) {
    Take t;
    auto str = [&](const char* k) { auto it = j.find(k); return it != j.end() && it->is_string() ? it->get<std::string>() : std::string(); };
    auto num = [&](const char* k, double def) { auto it = j.find(k); return it != j.end() && it->is_number() ? it->get<double>() : def; };
    t.path    = fs::path(str("path"));
    t.session = str("session");
    t.stats.ok          = j.value("ok", false);
    t.stats.error       = str("error");
    t.stats.sample_rate = static_cast<std::uint32_t>(num("sample_rate", 0));
    t.stats.channels    = static_cast<std::uint16_t>(num("channels", 0));
    t.stats.bits        = static_cast<std::uint16_t>(num("bits", 0));
    t.stats.is_float    = j.value("is_float", false);
    t.stats.seconds     = num("seconds", 0.0);
    t.stats.peak_dbfs   = static_cast<float>(num("peak_dbfs", -120.0));
    t.stats.rms_dbfs    = static_cast<float>(num("rms_dbfs", -120.0));
    t.stats.noise_floor_dbfs = static_cast<float>(num("noise_floor_dbfs", -120.0));
    t.stats.speech_dbfs      = static_cast<float>(num("speech_dbfs", -120.0));
    t.stats.snr_db           = static_cast<float>(num("snr_db", 0.0));
    t.stats.clipped_samples  = static_cast<std::size_t>(num("clipped_samples", 0));
    t.verdict = verdict_from_string(str("verdict"));
    return t;
}

bool write_text(const fs::path& p, const std::string& text, std::string& error) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) { error = "cannot write " + p.string(); return false; }
    f << text;
    if (!f) { error = "write failed for " + p.string(); return false; }
    return true;
}

} // namespace

std::string VoiceStudioModel::takes_tsv() const {
    std::ostringstream o;
    o << "path\tsession\tseconds\tpeak_dbfs\tsnr_db\tverdict\n";
    for (const Take& t : draft_.takes)
        o << t.path.string() << '\t' << t.session << '\t' << t.stats.seconds << '\t' << t.stats.peak_dbfs << '\t'
          << t.stats.snr_db << '\t' << to_string(t.verdict) << '\n';
    return o.str();
}

std::string VoiceStudioModel::job_json(const fs::path& repo_root, const fs::path& studio_root,
                                       const fs::path& models_root) const {
    json j;
    j["schema"]        = 1;
    j["voice"]         = draft_.voice_name;
    j["display"]       = draft_.display_name;
    j["is_own_voice"]  = draft_.is_own_voice;
    j["consent"]       = draft_.consent;
    j["consent_note"]  = draft_.consent_note;
    j["base_bundle"]   = draft_.base_bundle;
    j["target_bundle"] = draft_.target_bundle;
    j["license"]       = "private - do not distribute";
    j["clean_minutes"] = clean_minutes();
    j["takes"]      = json::array();
    j["room_tone"]  = json::array();
    j["live_mic"]   = json::array();
    for (const Take& t : draft_.takes)     j["takes"].push_back(take_to_json(t));
    for (const Take& t : draft_.room_tone) j["room_tone"].push_back(take_to_json(t));
    for (const Take& t : draft_.live_mic)  j["live_mic"].push_back(take_to_json(t));
    const StudioCommands posix = commands(repo_root, studio_root, models_root, Shell::Posix);
    const StudioCommands ps    = commands(repo_root, studio_root, models_root, Shell::PowerShell);
    j["commands"]["posix"]      = {{"prepare", posix.prepare}, {"finetune", posix.finetune}, {"export", posix.export_}};
    j["commands"]["powershell"] = {{"prepare", ps.prepare}, {"finetune", ps.finetune}, {"export", ps.export_}};
    return j.dump(2) + "\n";
}

bool VoiceStudioModel::write_job(const fs::path& repo_root, const fs::path& studio_root, const fs::path& models_root,
                                 std::string& error) const {
    const fs::path dir = project_dir(studio_root);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { error = "cannot create " + dir.string(); return false; }
    if (!write_text(dir / "job.json", job_json(repo_root, studio_root, models_root), error)) return false;
    if (!write_text(dir / "takes.tsv", takes_tsv(), error)) return false;
    {
        std::ostringstream o;
        o << "path\tseconds\n";
        for (const Take& t : draft_.room_tone) o << t.path.string() << '\t' << t.stats.seconds << '\n';
        if (!write_text(dir / "room_tone.tsv", o.str(), error)) return false;
    }
    const StudioCommands posix = commands(repo_root, studio_root, models_root, Shell::Posix);
    const StudioCommands ps    = commands(repo_root, studio_root, models_root, Shell::PowerShell);
    if (!write_text(dir / "run_finetune.sh",
                    "#!/bin/sh\nset -e\n" + posix.prepare + "\n" + posix.finetune + "\n" + posix.export_ + "\n", error))
        return false;
    if (!write_text(dir / "run_finetune.ps1",
                    "$ErrorActionPreference = 'Stop'\n" + ps.prepare + "\n" + ps.finetune + "\n" + ps.export_ + "\n", error))
        return false;
    return true;
}

bool VoiceStudioModel::job_written(const fs::path& studio_root) const {
    std::error_code ec;
    return fs::is_regular_file(project_dir(studio_root) / "job.json", ec);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

std::string VoiceStudioModel::draft_json() const {
    json j;
    j["schema"]        = 1;
    j["step"]          = static_cast<int>(step_);
    j["voice_name"]    = draft_.voice_name;
    j["display_name"]  = draft_.display_name;
    j["is_own_voice"]  = draft_.is_own_voice;
    j["consent"]       = draft_.consent;
    j["consent_note"]  = draft_.consent_note;
    j["base_bundle"]   = draft_.base_bundle;
    j["target_bundle"] = draft_.target_bundle;
    j["setup"]    = json::array();
    j["material"] = json::array();
    for (bool b : draft_.setup) j["setup"].push_back(b);
    for (bool b : draft_.material) j["material"].push_back(b);
    j["takes"] = json::array();
    j["room_tone"] = json::array();
    j["live_mic"] = json::array();
    for (const Take& t : draft_.takes)     j["takes"].push_back(take_to_json(t));
    for (const Take& t : draft_.room_tone) j["room_tone"].push_back(take_to_json(t));
    for (const Take& t : draft_.live_mic)  j["live_mic"].push_back(take_to_json(t));
    return j.dump(2) + "\n";
}

bool VoiceStudioModel::load_draft_json(std::string_view text, std::string* error) {
    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object() || j.value("schema", 0) != 1) {
        if (error) *error = "draft.json is not a schema-1 object";
        return false;
    }
    StudioDraft d;
    d.voice_name    = j.value("voice_name", std::string());
    d.display_name  = j.value("display_name", std::string());
    d.is_own_voice  = j.value("is_own_voice", true);
    d.consent       = j.value("consent", false);
    d.consent_note  = j.value("consent_note", std::string());
    d.base_bundle   = j.value("base_bundle", std::string());
    d.target_bundle = j.value("target_bundle", std::string());
    if (auto it = j.find("setup"); it != j.end() && it->is_array())
        for (std::size_t i = 0; i < d.setup.size() && i < it->size(); ++i) d.setup[i] = (*it)[i].is_boolean() && (*it)[i].get<bool>();
    if (auto it = j.find("material"); it != j.end() && it->is_array())
        for (std::size_t i = 0; i < d.material.size() && i < it->size(); ++i) d.material[i] = (*it)[i].is_boolean() && (*it)[i].get<bool>();
    auto load_takes = [&](const char* key, std::vector<Take>& out) {
        if (auto it = j.find(key); it != j.end() && it->is_array())
            for (const json& t : *it)
                if (t.is_object()) out.push_back(take_from_json(t));
    };
    load_takes("takes", d.takes);
    load_takes("room_tone", d.room_tone);
    load_takes("live_mic", d.live_mic);
    const int step = j.value("step", 0);
    step_  = static_cast<StudioStep>(step < 0 || step >= static_cast<int>(StudioStep::Count) ? 0 : step);
    draft_ = std::move(d);
    return true;
}

bool VoiceStudioModel::save_draft(const fs::path& studio_root, std::string& error) const {
    const fs::path dir = project_dir(studio_root);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { error = "cannot create " + dir.string(); return false; }
    return write_text(dir / "draft.json", draft_json(), error);
}

bool VoiceStudioModel::load_draft(const fs::path& studio_root) {
    std::ifstream f(project_dir(studio_root) / "draft.json", std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    return load_draft_json(ss.str());
}

std::optional<StudioStatus> VoiceStudioModel::poll_status(const fs::path& studio_root) const {
    std::ifstream f(project_dir(studio_root) / "status.json", std::ios::binary);
    if (!f) return std::nullopt;
    std::stringstream ss;
    ss << f.rdbuf();
    const json j = json::parse(ss.str(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    StudioStatus s;
    s.stage    = j.value("stage", std::string("unknown"));
    s.progress = j.value("progress", 0.0f);
    s.message  = j.value("message", std::string());
    s.eta_s    = j.value("eta_s", -1.0);
    if (s.progress < 0.0f) s.progress = 0.0f;
    if (s.progress > 1.0f) s.progress = 1.0f;
    return s;
}

bool VoiceStudioModel::installed(const VoiceCatalog& catalog) const {
    for (const BundleInfo& b : catalog.bundles())
        for (const VoiceEntry& v : b.voices)
            if (v.name == draft_.voice_name) return true;
    return false;
}

} // namespace fluxvoice::frontend
