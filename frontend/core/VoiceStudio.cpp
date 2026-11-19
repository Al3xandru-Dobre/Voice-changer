// VoiceStudio — WAV analysis, wizard rules, job manifest, draft persistence.
#include "core/VoiceStudio.hpp"

#include "core/JsonRead.hpp"
#include "core/Paths.hpp"
#include "core/VoiceCatalog.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
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
        case TakeVerdict::NonFinite:  return "contains non-finite samples (NaN/Inf)";
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

    /// Non-finite samples are counted and left out of every level.
    void push(float x) {
        if (!std::isfinite(x)) { ++non_finite_; return; }
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
        s.seconds     = sample_rate == 0 ? 0.0 : static_cast<double>(n_ + non_finite_) / sample_rate;
        s.peak_dbfs   = to_db(peak_);
        s.rms_dbfs    = n_ == 0 ? -120.0f : to_db(std::sqrt(sum_sq_ / static_cast<double>(n_)));
        s.clipped_samples = clipped_;
        s.non_finite_samples = non_finite_;
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
    std::size_t non_finite_ = 0;
};

std::uint32_t le32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint16_t le16(const unsigned char* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

/// Channel average in double; NaN when any channel is non-finite (so push() counts it).
float mono_of(const float* frame, std::uint16_t channels) {
    double m = 0.0;
    for (std::uint16_t c = 0; c < channels; ++c) {
        if (!std::isfinite(frame[c])) return std::numeric_limits<float>::quiet_NaN();
        m += static_cast<double>(frame[c]);
    }
    return static_cast<float>(m / channels);
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
        acc.push(mono_of(interleaved + i * channels, channels));
    }
    return acc.finish(sample_rate, channels);
}

namespace {

constexpr std::uint32_t kMinSampleRate = 8000;
constexpr std::uint32_t kMaxSampleRate = 384000;
constexpr std::uint16_t kMaxChannels   = 8;
constexpr std::size_t   kFmtReadMax    = 40;  // WAVE_FORMAT_EXTENSIBLE needs 26; the rest is skipped

WavStats analyze_wav_impl(const fs::path& path, const std::atomic<bool>* cancel) {
    WavStats s;
    std::error_code ec;
    const std::uint64_t file_size = fs::is_regular_file(path, ec) ? static_cast<std::uint64_t>(fs::file_size(path, ec)) : 0;
    std::ifstream f(path, std::ios::binary);
    if (ec || !f) {
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
    std::uint64_t offset = 12;  // start of the next chunk header; strictly increases by >= 8 per chunk
    while (offset + 8 <= file_size) {
        f.clear();
        f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        unsigned char ch[8];
        if (!f.read(reinterpret_cast<char*>(ch), 8)) break;
        const std::uint64_t size  = le32(ch + 4);
        const std::uint64_t body  = offset + 8;
        const std::uint64_t avail = file_size - body;  // body <= file_size here
        if (std::memcmp(ch, "fmt ", 4) == 0) {
            if (size < 16 || size > avail) {
                s.error = size < 16 ? "truncated fmt chunk" : "fmt chunk larger than the file";
                return s;
            }
            unsigned char fmt[kFmtReadMax] = {};
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(size, kFmtReadMax));
            if (!f.read(reinterpret_cast<char*>(fmt), static_cast<std::streamsize>(n))) {
                s.error = "truncated fmt chunk";
                return s;
            }
            format      = le16(fmt);
            channels    = le16(fmt + 2);
            sample_rate = le32(fmt + 4);
            const std::uint16_t block_align = le16(fmt + 12);
            bits        = le16(fmt + 14);
            if (format == 0xFFFE && size >= 26) format = le16(fmt + 24);  // extensible: sub-format
            if (!((format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32))) {
                s.error = "unsupported sample format (need PCM 16/24/32 or float 32)";
                return s;
            }
            if (channels == 0 || channels > kMaxChannels || sample_rate < kMinSampleRate || sample_rate > kMaxSampleRate) {
                s.error = "invalid fmt chunk (channels or sample rate out of range)";
                return s;
            }
            if (block_align != channels * (bits / 8u)) {
                s.error = "invalid fmt chunk (block align does not match channels and bit depth)";
                return s;
            }
            have_fmt = true;
        } else if (std::memcmp(ch, "data", 4) == 0) {
            if (!have_fmt) {
                s.error = "data chunk before fmt chunk";
                return s;
            }
            const bool is_float = (format == 3);
            const std::size_t sample_bytes = bits / 8u;
            const std::size_t frame_bytes  = static_cast<std::size_t>(channels) * sample_bytes;
            // A data size larger than the file (truncated or streamed file) is clamped to what is there.
            std::uint64_t remaining = std::min(size, avail);
            remaining -= remaining % frame_bytes;
            StatsAccumulator acc(sample_rate);
            std::vector<unsigned char> buf(frame_bytes * 4096);
            while (remaining > 0 && f) {
                if (cancel && cancel->load(std::memory_order_relaxed)) {
                    s.error = "cancelled";
                    return s;
                }
                const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(buf.size(), remaining));
                f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(want));
                const std::size_t got = static_cast<std::size_t>(f.gcount());
                if (got == 0) break;
                remaining -= std::min<std::uint64_t>(got, remaining);
                for (std::size_t off = 0; off + frame_bytes <= got; off += frame_bytes) {
                    float frame[kMaxChannels];
                    for (std::uint16_t c = 0; c < channels; ++c)
                        frame[c] = decode_sample(buf.data() + off + c * sample_bytes, bits, is_float);
                    acc.push(mono_of(frame, channels));
                }
            }
            s          = acc.finish(sample_rate, channels);
            s.bits     = bits;
            s.is_float = is_float;
            return s;
        }
        offset = body + size + (size & 1u);  // RIFF chunks are word aligned: odd sizes carry a pad byte
    }
    s.error = have_fmt ? "no data chunk" : "no fmt chunk";
    return s;
}

} // namespace

WavStats analyze_wav_file(const fs::path& path) { return analyze_wav_impl(path, nullptr); }

TakeVerdict judge_take(const WavStats& s, double min_seconds, float min_snr_db) {
    if (!s.ok) return TakeVerdict::Unreadable;
    if (s.non_finite_samples > 0) return TakeVerdict::NonFinite;
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
            if (!is_safe_project_name(draft_.voice_name)) return fail("That voice name is reserved by Windows; choose another.");
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
            if (room_tone_seconds() < req_.room_tone_seconds) {
                if (why) *why = "Add at least " + std::to_string(static_cast<int>(std::ceil(req_.room_tone_seconds))) +
                                " s of room tone (the SNR check depends on it).";
                return false;
            }
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
    if (!is_safe_project_name(draft_.voice_name))
        return std::string("'" + draft_.voice_name + "' is a reserved device name on Windows; choose another");
    if (installed) {
        for (const BundleInfo& b : installed->bundles())
            for (const VoiceEntry& v : b.voices)
                if (v.name == draft_.voice_name) return "a voice named '" + v.name + "' already exists in bundle '" + b.name + "'";
    }
    return std::nullopt;
}

namespace {

Take take_from_stats(const fs::path& wav, std::string session, WavStats stats) {
    Take t;
    t.path    = wav;
    t.session = std::move(session);
    t.stats   = std::move(stats);
    t.verdict = judge_take(t.stats);
    return t;
}

Take make_take(const fs::path& wav, std::string session) {
    return take_from_stats(wav, std::move(session), analyze_wav_file(wav));
}

/// Room tone is silence: the SNR rule does not apply, only readability/mono/finiteness.
void relax_room_tone(Take& t) {
    if (t.verdict == TakeVerdict::LowSnr || t.verdict == TakeVerdict::TooShort) t.verdict = TakeVerdict::Clean;
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
    relax_room_tone(t);
    draft_.room_tone.push_back(std::move(t));
    return draft_.room_tone.back();
}

const Take& VoiceStudioModel::add_live_mic(const fs::path& wav) {
    draft_.live_mic.push_back(make_take(wav, "live-mic"));
    return draft_.live_mic.back();
}

// ---------------------------------------------------------------------------
// Asynchronous analysis: one worker thread per model, owned through worker_.
// The thread only reads files and computes WavStats; it never touches the
// model. poll_analysis() (calling thread) moves finished results into the draft.
// ---------------------------------------------------------------------------

struct VoiceStudioModel::Worker {
    struct Request { TakeKind kind; fs::path path; std::string session; };
    struct Result  { TakeKind kind; fs::path path; std::string session; WavStats stats; };

    std::mutex              mu;
    std::condition_variable cv;
    std::deque<Request>     requests;
    std::deque<Result>      results;
    std::size_t             in_flight = 0;  // requested, not yet applied
    std::atomic<bool>       stop{false};
    std::thread             thread;

    ~Worker() {
        stop.store(true);
        {
            const std::lock_guard<std::mutex> lock(mu);
            cv.notify_all();
        }
        if (thread.joinable()) thread.join();
    }

    void run() {
        for (;;) {
            Request req;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return stop.load() || !requests.empty(); });
                if (stop.load()) return;
                req = std::move(requests.front());
                requests.pop_front();
            }
            WavStats stats = analyze_wav_impl(req.path, &stop);
            if (stop.load()) return;
            const std::lock_guard<std::mutex> lock(mu);
            results.push_back(Result{req.kind, std::move(req.path), std::move(req.session), std::move(stats)});
        }
    }
};

VoiceStudioModel::VoiceStudioModel() = default;
VoiceStudioModel::~VoiceStudioModel() = default;
VoiceStudioModel::VoiceStudioModel(VoiceStudioModel&&) noexcept = default;
VoiceStudioModel& VoiceStudioModel::operator=(VoiceStudioModel&&) noexcept = default;

void VoiceStudioModel::request_take_analysis(TakeKind kind, const fs::path& wav, std::string session) {
    if (!worker_) {
        worker_ = std::make_unique<Worker>();
        Worker* w = worker_.get();
        w->thread = std::thread([w] { w->run(); });
    }
    {
        const std::lock_guard<std::mutex> lock(worker_->mu);
        worker_->requests.push_back(Worker::Request{kind, wav, std::move(session)});
        ++worker_->in_flight;
    }
    worker_->cv.notify_one();
}

std::size_t VoiceStudioModel::poll_analysis() {
    if (!worker_) return 0;
    std::deque<Worker::Result> done;
    {
        const std::lock_guard<std::mutex> lock(worker_->mu);
        done.swap(worker_->results);
        worker_->in_flight -= done.size();
    }
    for (Worker::Result& r : done) {
        const bool room = r.kind == TakeKind::RoomTone;
        Take t = take_from_stats(r.path, room ? std::string("room-tone") : (r.kind == TakeKind::LiveMic ? std::string("live-mic") : std::move(r.session)),
                                 std::move(r.stats));
        if (room) relax_room_tone(t);
        std::vector<Take>& list = r.kind == TakeKind::Take ? draft_.takes : (room ? draft_.room_tone : draft_.live_mic);
        list.push_back(std::move(t));
    }
    return done.size();
}

std::size_t VoiceStudioModel::analysis_pending_count() const {
    if (!worker_) return 0;
    const std::lock_guard<std::mutex> lock(worker_->mu);
    return worker_->in_flight;
}

bool VoiceStudioModel::analysis_pending() const { return analysis_pending_count() > 0; }

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

namespace {

/// Single-quote `s` for the shell; control characters (line breaks) become spaces.
std::string quote_text(const std::string& s, Shell shell) {
    std::string out = "'";
    for (const char ch : s) {
        if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f) out += ' ';
        else if (ch == '\'') out += shell == Shell::PowerShell ? "''" : "'\\''";
        else out += ch;
    }
    return out + "'";
}

} // namespace

std::string shell_quote(const fs::path& p, Shell shell) {
    return quote_text(shell == Shell::Posix ? path_to_utf8_generic(p) : path_to_utf8(p), shell);
}

bool is_safe_project_name(std::string_view name) {
    if (!is_valid_voice_name(name)) return false;
    static constexpr std::string_view kReserved[] = {"con", "prn", "aux", "nul"};
    for (std::string_view r : kReserved)
        if (name == r) return false;
    if (name.size() == 4 && (name.starts_with("com") || name.starts_with("lpt")) && name[3] >= '0' && name[3] <= '9')
        return false;
    return true;
}

fs::path VoiceStudioModel::project_dir(const fs::path& studio_root) const {
    if (!is_safe_project_name(draft_.voice_name)) return {};
    return studio_root / draft_.voice_name;
}

std::string checked_line(const std::string& cmd, Shell shell) {
    if (shell == Shell::PowerShell) return cmd + "\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }";
    return cmd;
}

namespace {

/// `<voice>_sNN_TTTT.wav` (take) or `<voice>_sNN_roomtone.wav`: the names prepare_target_voice.py scans for.
bool matches_prepare_name(const fs::path& p, const std::string& voice, bool room_tone) {
    if (p.extension() != ".wav") return false;
    const std::string stem = path_to_utf8(p.stem());
    const std::string head = voice + "_s";
    if (stem.size() <= head.size() + 2 || stem.compare(0, head.size(), head) != 0) return false;
    if (!std::isdigit(static_cast<unsigned char>(stem[head.size()])) || !std::isdigit(static_cast<unsigned char>(stem[head.size() + 1])))
        return false;
    if (stem[head.size() + 2] != '_') return false;
    const std::string tail = stem.substr(head.size() + 3);
    if (room_tone) return tail == "roomtone";
    return tail.size() == 4 && std::all_of(tail.begin(), tail.end(), [](char ch) { return ch >= '0' && ch <= '9'; });
}

} // namespace

StudioCommands VoiceStudioModel::commands(const fs::path& repo_root, const fs::path& studio_root,
                                          const fs::path& models_root, Shell shell) const {
    static_cast<void>(models_root);
    StudioCommands c;
    const fs::path project = project_dir(studio_root);
    if (project.empty()) {
        c.prepare_unavailable = c.finetune_unavailable = c.export_unavailable = "the voice name is not a valid project name";
        return c;
    }
    const fs::path training = repo_root / "python_training";
    const fs::path python   = shell == Shell::PowerShell ? training / ".venv" / "Scripts" / "python.exe"
                                                         : training / ".venv" / "bin" / "python";
    auto q = [&](const fs::path& p) { return shell_quote(p, shell); };
    const std::string call = (shell == Shell::PowerShell ? "& " : std::string()) + q(python) + " ";

    // Step 1 — prepare_target_voice.py scans one directory (recursively) for
    // <voice>_sNN_TTTT.wav takes and <voice>_sNN_roomtone.wav room tones.
    fs::path input_dir;
    bool same_dir = !draft_.takes.empty(), names_ok = true;
    for (const std::vector<Take>* list : {&draft_.takes, &draft_.room_tone}) {
        for (const Take& t : *list) {
            const fs::path parent = t.path.parent_path().lexically_normal();
            if (input_dir.empty()) input_dir = parent;
            else if (parent != input_dir) same_dir = false;
            if (!matches_prepare_name(t.path, draft_.voice_name, list == &draft_.room_tone)) names_ok = false;
        }
    }
    if (draft_.takes.empty()) {
        c.prepare_unavailable = "add takes first";
    } else if (!same_dir) {
        c.prepare_unavailable = "the preparation step reads one directory: keep every take and room-tone file in the same folder";
    } else if (!names_ok) {
        c.prepare_unavailable = "the preparation step needs takes named " + draft_.voice_name + "_sNN_TTTT.wav and room tone named " +
                                draft_.voice_name + "_sNN_roomtone.wav (NN = session, TTTT = take number)";
    } else {
        c.prepare = call + q(training / "prepare_target_voice.py") + " --input " + q(input_dir) + " --speaker=" + draft_.voice_name +
                    " --out " + q(project / "prepared") +
                    (draft_.display_name.empty() ? std::string() : " --display " + quote_text(draft_.display_name, shell));
    }

    // Steps 2-3 — unit extraction + fine-tune need inputs the Studio does not hold.
    c.finetune_unavailable =
        "unavailable until M2 Stage 10 lands: the fine-tune needs the Stage 05 winner encoder checkpoint (unit extraction), the "
        "base vocoder checkpoint and the public training manifest and unit directories (50/50 batches); the Studio does not know "
        "them and no existing option names a base bundle. Run the recipe in Documents/Milestone 2/10_target_voice_finetune.md by hand.";
    // Step 4 — the export reads the checkpoint that the fine-tune writes.
    c.export_unavailable = "unavailable until M2 Stage 10 lands: it exports the checkpoint written by the fine-tune step, which cannot be generated yet.";
    return c;
}

namespace {

json take_to_json(const Take& t) {
    json j;
    j["path"]        = path_to_utf8(t.path);
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
    j["non_finite_samples"] = t.stats.non_finite_samples;
    return j;
}

TakeVerdict verdict_from_string(std::string_view s) {
    for (int i = 0; i <= static_cast<int>(TakeVerdict::NonFinite); ++i) {
        const auto v = static_cast<TakeVerdict>(i);
        if (s == to_string(v)) return v;
    }
    return TakeVerdict::Unreadable;
}

Take take_from_json(const json& j) {
    Take t;
    auto str = [&](const char* k) { return read_string(j, k, std::string()); };
    auto num = [&](const char* k, double def) { return read_number(j, k, def); };
    auto flt = [&](const char* k, float def) { return read_float(j, k, def); };
    t.path    = path_from_utf8(str("path"));
    t.session = str("session");
    t.stats.ok          = read_bool(j, "ok", false);
    t.stats.error       = str("error");
    t.stats.sample_rate = read_integral<std::uint32_t>(j, "sample_rate", 0);
    t.stats.channels    = read_integral<std::uint16_t>(j, "channels", 0);
    t.stats.bits        = read_integral<std::uint16_t>(j, "bits", 0);
    t.stats.is_float    = read_bool(j, "is_float", false);
    t.stats.seconds     = num("seconds", 0.0);
    t.stats.peak_dbfs   = flt("peak_dbfs", -120.0f);
    t.stats.rms_dbfs    = flt("rms_dbfs", -120.0f);
    t.stats.noise_floor_dbfs = flt("noise_floor_dbfs", -120.0f);
    t.stats.speech_dbfs      = flt("speech_dbfs", -120.0f);
    t.stats.snr_db           = flt("snr_db", 0.0f);
    t.stats.clipped_samples  = read_integral<std::size_t>(j, "clipped_samples", 0);
    t.stats.non_finite_samples = read_integral<std::size_t>(j, "non_finite_samples", 0);
    t.verdict = verdict_from_string(str("verdict"));
    return t;
}

bool write_text(const fs::path& p, const std::string& text, std::string& error) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) { error = "cannot write " + path_to_utf8(p); return false; }
    f << text;
    if (!f) { error = "write failed for " + path_to_utf8(p); return false; }
    return true;
}

} // namespace

std::string VoiceStudioModel::takes_tsv() const {
    std::ostringstream o;
    o << "path\tsession\tseconds\tpeak_dbfs\tsnr_db\tverdict\n";
    for (const Take& t : draft_.takes)
        o << path_to_utf8(t.path) << '\t' << t.session << '\t' << t.stats.seconds << '\t' << t.stats.peak_dbfs << '\t'
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
    j["commands"]["posix"]      = json::object();
    j["commands"]["powershell"] = json::object();
    j["unavailable"]            = json::object();
    const auto put = [&](const char* name, const std::string& px, const std::string& pw, const std::string& why) {
        if (why.empty()) {
            j["commands"]["posix"][name]      = px;
            j["commands"]["powershell"][name] = pw;
        } else {
            j["unavailable"][name] = why;
        }
    };
    put("prepare", posix.prepare, ps.prepare, posix.prepare_unavailable);
    put("finetune", posix.finetune, ps.finetune, posix.finetune_unavailable);
    put("export", posix.export_, ps.export_, posix.export_unavailable);
    return j.dump(2) + "\n";
}

bool VoiceStudioModel::write_job(const fs::path& repo_root, const fs::path& studio_root, const fs::path& models_root,
                                 std::string& error) const {
    if (!is_safe_project_name(draft_.voice_name)) {
        error = "voice name '" + draft_.voice_name + "' is not a valid project name";
        return false;
    }
    const fs::path dir = project_dir(studio_root);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { error = "cannot create " + path_to_utf8(dir); return false; }
    if (!write_text(dir / "job.json", job_json(repo_root, studio_root, models_root), error)) return false;
    if (!write_text(dir / "takes.tsv", takes_tsv(), error)) return false;
    {
        std::ostringstream o;
        o << "path\tseconds\n";
        for (const Take& t : draft_.room_tone) o << path_to_utf8(t.path) << '\t' << t.stats.seconds << '\n';
        if (!write_text(dir / "room_tone.tsv", o.str(), error)) return false;
    }
    const StudioCommands posix = commands(repo_root, studio_root, models_root, Shell::Posix);
    const StudioCommands ps    = commands(repo_root, studio_root, models_root, Shell::PowerShell);
    std::string sh = "#!/bin/sh\nset -eu\n";
    std::string pw = "$ErrorActionPreference = 'Stop'\n";
    const auto add = [&](const char* name, const std::string& px, const std::string& pws, const std::string& why) {
        if (why.empty()) {
            sh += checked_line(px, Shell::Posix) + "\n";
            pw += checked_line(pws, Shell::PowerShell) + "\n";
        } else {
            sh += std::string("# step '") + name + "' not generated: " + why + "\n";
            pw += std::string("# step '") + name + "' not generated: " + why + "\n";
        }
    };
    add("prepare", posix.prepare, ps.prepare, posix.prepare_unavailable);
    add("finetune", posix.finetune, ps.finetune, posix.finetune_unavailable);
    add("export", posix.export_, ps.export_, posix.export_unavailable);
    if (!write_text(dir / "run_finetune.sh", sh, error)) return false;
    if (!write_text(dir / "run_finetune.ps1", pw, error)) return false;
    return true;
}

bool VoiceStudioModel::job_written(const fs::path& studio_root) const {
    const fs::path dir = project_dir(studio_root);
    std::error_code ec;
    return !dir.empty() && fs::is_regular_file(dir / "job.json", ec);
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
    if (j.is_discarded() || !j.is_object() || read_int(j, "schema", 0) != 1) {
        if (error) *error = "draft.json is not a schema-1 object";
        return false;
    }
    StudioDraft d;
    d.voice_name    = read_string(j, "voice_name", std::string());
    d.display_name  = read_string(j, "display_name", std::string());
    d.is_own_voice  = read_bool(j, "is_own_voice", true);
    d.consent       = read_bool(j, "consent", false);
    d.consent_note  = read_string(j, "consent_note", std::string());
    d.base_bundle   = read_string(j, "base_bundle", std::string());
    d.target_bundle = read_string(j, "target_bundle", std::string());
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
    const int step = read_integral<int>(j, "step", 0, 0, static_cast<int>(StudioStep::Count) - 1);
    step_  = static_cast<StudioStep>(step);
    draft_ = std::move(d);
    return true;
}

bool VoiceStudioModel::save_draft(const fs::path& studio_root, std::string& error) const {
    if (!is_safe_project_name(draft_.voice_name)) {
        error = "voice name '" + draft_.voice_name + "' is not a valid project name";
        return false;
    }
    const fs::path dir = project_dir(studio_root);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { error = "cannot create " + path_to_utf8(dir); return false; }
    return write_text(dir / "draft.json", draft_json(), error);
}

bool VoiceStudioModel::load_draft(const fs::path& studio_root) {
    if (!is_safe_project_name(draft_.voice_name)) return false;
    std::ifstream f(project_dir(studio_root) / "draft.json", std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    return load_draft_json(ss.str());
}

std::optional<StudioStatus> VoiceStudioModel::poll_status(const fs::path& studio_root) const {
    const fs::path dir = project_dir(studio_root);
    if (dir.empty()) return std::nullopt;
    std::ifstream f(dir / "status.json", std::ios::binary);
    if (!f) return std::nullopt;
    std::stringstream ss;
    ss << f.rdbuf();
    const json j = json::parse(ss.str(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    StudioStatus s;
    s.stage    = read_string(j, "stage", std::string("unknown"));
    s.progress = read_float(j, "progress", 0.0f);
    s.message  = read_string(j, "message", std::string());
    s.eta_s    = read_number(j, "eta_s", -1.0);
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
