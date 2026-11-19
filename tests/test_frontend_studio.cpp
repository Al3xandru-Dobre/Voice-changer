// Voice Studio — headless tests (Milestone 4 Stage 03 residuals, no display,
// no audio hardware): hostile WAV headers, non-finite samples, analysis off
// the UI thread, room-tone threshold, and job commands that only use scripts
// and options that exist in python_training/ (BUG-025).
//
// Usage: test_frontend_studio [repo-root]
#include "TestSupport.hpp"

#include "core/VoiceStudio.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace fluxvoice::frontend;

fs::path g_repo = ".";
fs::path g_tmp;

// ---------------------------------------------------------------------------
// WAV fixtures
// ---------------------------------------------------------------------------
void put16(std::string& s, std::uint16_t v) {
    s.push_back(static_cast<char>(v & 0xFF));
    s.push_back(static_cast<char>(v >> 8));
}
void put32(std::string& s, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

struct Fmt {
    std::uint16_t format = 1, channels = 1, bits = 16;
    std::uint32_t rate = 16000;
    std::optional<std::uint16_t> block_align;  // default: channels * bits / 8
    std::uint32_t chunk_size = 16;             // declared size of the fmt chunk
};

std::string fmt_chunk(const Fmt& f) {
    std::string s = "fmt ";
    put32(s, f.chunk_size);
    put16(s, f.format);
    put16(s, f.channels);
    put32(s, f.rate);
    put32(s, f.rate * f.channels * (f.bits / 8u));
    put16(s, f.block_align.value_or(static_cast<std::uint16_t>(f.channels * (f.bits / 8u))));
    put16(s, f.bits);
    return s;
}

std::string pcm_bytes(const std::vector<float>& mono, std::uint16_t bits) {
    std::string d;
    for (float x : mono) {
        const double v = x < -1.0f ? -1.0 : (x > 1.0f ? 1.0 : static_cast<double>(x));
        if (bits == 16) {
            put16(d, static_cast<std::uint16_t>(static_cast<std::int16_t>(std::lround(v * 32767.0))));
        } else {
            const auto i = static_cast<std::int32_t>(std::lround(v * 8388607.0));
            d.push_back(static_cast<char>(i & 0xFF));
            d.push_back(static_cast<char>((i >> 8) & 0xFF));
            d.push_back(static_cast<char>((i >> 16) & 0xFF));
        }
    }
    return d;
}

std::string float_bytes(const std::vector<float>& mono) {
    std::string d(mono.size() * 4, '\0');
    if (!mono.empty()) std::memcpy(d.data(), mono.data(), d.size());
    return d;
}

std::string riff(const std::string& body) {
    std::string s = "RIFF";
    put32(s, static_cast<std::uint32_t>(4 + body.size()));
    s += "WAVE";
    return s + body;
}

std::string data_chunk(const std::string& bytes, std::optional<std::uint32_t> declared = std::nullopt) {
    std::string s = "data";
    put32(s, declared.value_or(static_cast<std::uint32_t>(bytes.size())));
    s += bytes;
    if (!declared && (bytes.size() & 1u)) s.push_back('\0');
    return s;
}

void write_file(const fs::path& p, const std::string& bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

/// Speech-like test signal: alternating loud and quiet 10 ms blocks (SNR well over 30 dB).
std::vector<float> take_signal(double seconds, std::uint32_t rate) {
    const auto n = static_cast<std::size_t>(seconds * rate);
    std::vector<float> v(n);
    const std::size_t block = rate / 100;
    for (std::size_t i = 0; i < n; ++i) {
        const float amp = ((i / block) % 2 == 0) ? 0.3f : 0.001f;
        v[i] = amp * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 220.0 * static_cast<double>(i) / rate));
    }
    return v;
}

std::string pcm_file(const std::vector<float>& mono, std::uint16_t bits, std::uint32_t rate = 16000) {
    Fmt f;
    f.bits = bits;
    f.rate = rate;
    return riff(fmt_chunk(f) + data_chunk(pcm_bytes(mono, bits)));
}

// ---------------------------------------------------------------------------
// WAV header hardening
// ---------------------------------------------------------------------------
void test_hostile_wavs() {
    const std::vector<float> sig = take_signal(2.5, 16000);
    const fs::path d = g_tmp / "wav";

    // Valid 16-bit and 24-bit files give correct stats.
    write_file(d / "ok16.wav", pcm_file(sig, 16));
    write_file(d / "ok24.wav", pcm_file(sig, 24));
    const WavStats s16 = analyze_wav_file(d / "ok16.wav"), s24 = analyze_wav_file(d / "ok24.wav");
    CHECK(s16.ok && s16.bits == 16 && s16.channels == 1 && std::fabs(s16.seconds - 2.5) < 1e-6);
    CHECK(s24.ok && s24.bits == 24 && std::fabs(s24.seconds - 2.5) < 1e-6);
    CHECK(judge_take(s16) == TakeVerdict::Clean && judge_take(s24) == TakeVerdict::Clean);

    // fmt chunk claiming ~4 GiB: a clean error, not a 4 GiB allocation.
    {
        Fmt f;
        f.chunk_size = 0xFFFFFFF0u;
        write_file(d / "fmt_huge.wav", riff(fmt_chunk(f) + data_chunk(pcm_bytes(sig, 16))));
        const auto t0 = std::chrono::steady_clock::now();
        const WavStats s = analyze_wav_file(d / "fmt_huge.wav");
        CHECK(!s.ok && !s.error.empty());
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2));
    }
    // data size larger than the file: clamped to the bytes that are there.
    {
        Fmt f;
        write_file(d / "data_big.wav", riff(fmt_chunk(f) + data_chunk(pcm_bytes(sig, 16), 0xFFFFFFF0u)));
        const WavStats s = analyze_wav_file(d / "data_big.wav");
        CHECK(s.ok && std::fabs(s.seconds - 2.5) < 1e-6);
    }
    // channels 0 and 200, bits 7, sample-rate and block-align lies.
    for (const std::uint16_t ch : {std::uint16_t{0}, std::uint16_t{200}}) {
        Fmt f;
        f.channels = ch;
        f.block_align = 2;
        write_file(d / ("ch" + std::to_string(ch) + ".wav"), riff(fmt_chunk(f) + data_chunk(pcm_bytes(sig, 16))));
        CHECK(!analyze_wav_file(d / ("ch" + std::to_string(ch) + ".wav")).ok);
    }
    {
        Fmt f;
        f.bits = 7;
        f.block_align = 1;
        write_file(d / "bits7.wav", riff(fmt_chunk(f) + data_chunk(pcm_bytes(sig, 16))));
        CHECK(!analyze_wav_file(d / "bits7.wav").ok);
        Fmt r;
        r.rate = 3;
        write_file(d / "rate3.wav", riff(fmt_chunk(r) + data_chunk(pcm_bytes(sig, 16))));
        CHECK(!analyze_wav_file(d / "rate3.wav").ok);
        Fmt b;
        b.block_align = 7;
        write_file(d / "align.wav", riff(fmt_chunk(b) + data_chunk(pcm_bytes(sig, 16))));
        CHECK(!analyze_wav_file(d / "align.wav").ok);
        Fmt e;
        e.format = 2;  // ADPCM
        write_file(d / "adpcm.wav", riff(fmt_chunk(e) + data_chunk(pcm_bytes(sig, 16))));
        CHECK(!analyze_wav_file(d / "adpcm.wav").ok);
    }
    // zero-size unknown chunk(s) before the data: skipped, no endless loop.
    {
        Fmt f;
        std::string body = fmt_chunk(f);
        for (int i = 0; i < 3; ++i) body += std::string("junk") + std::string(4, '\0');
        body += data_chunk(pcm_bytes(sig, 16));
        write_file(d / "zero_chunk.wav", riff(body));
        const WavStats s = analyze_wav_file(d / "zero_chunk.wav");
        CHECK(s.ok && std::fabs(s.seconds - 2.5) < 1e-6);
    }
    // odd-sized chunk with its pad byte (valid), then data.
    {
        Fmt f;
        std::string body = fmt_chunk(f);
        body += "LIST";
        put32(body, 3);
        body += "abc";
        body.push_back('\0');  // pad
        body += data_chunk(pcm_bytes(sig, 16));
        write_file(d / "odd_pad.wav", riff(body));
        const WavStats s = analyze_wav_file(d / "odd_pad.wav");
        CHECK(s.ok && std::fabs(s.seconds - 2.5) < 1e-6);
    }
    // truncated headers.
    {
        const std::string full = pcm_file(sig, 16);
        write_file(d / "trunc_riff.wav", full.substr(0, 8));
        write_file(d / "trunc_fmt.wav", full.substr(0, 30));
        write_file(d / "trunc_chunk.wav", full.substr(0, 40));
        write_file(d / "empty.wav", "");
        CHECK(!analyze_wav_file(d / "trunc_riff.wav").ok);
        CHECK(!analyze_wav_file(d / "trunc_fmt.wav").ok);
        CHECK(!analyze_wav_file(d / "empty.wav").ok);
        const WavStats s = analyze_wav_file(d / "trunc_chunk.wav");  // fmt ok, data cut to 0 bytes
        CHECK(!s.ok || s.seconds == 0.0);
    }
    // data before fmt.
    write_file(d / "data_first.wav", riff(data_chunk(pcm_bytes(sig, 16)) + fmt_chunk(Fmt{})));
    CHECK(!analyze_wav_file(d / "data_first.wav").ok);
}

// ---------------------------------------------------------------------------
// Non-finite samples
// ---------------------------------------------------------------------------
std::string float_file(const std::vector<float>& mono) {
    Fmt f;
    f.format = 3;
    f.bits = 32;
    return riff(fmt_chunk(f) + data_chunk(float_bytes(mono)));
}

void test_non_finite() {
    const fs::path d = g_tmp / "nf";
    std::vector<float> sig = take_signal(3.0, 16000);
    write_file(d / "finite.wav", float_file(sig));
    const WavStats fin = analyze_wav_file(d / "finite.wav");
    CHECK(fin.ok && fin.is_float && fin.non_finite_samples == 0 && judge_take(fin) == TakeVerdict::Clean);

    std::vector<float> nan = sig;
    nan[500] = std::nanf("");
    write_file(d / "nan.wav", float_file(nan));
    const WavStats sn = analyze_wav_file(d / "nan.wav");
    CHECK(sn.non_finite_samples == 1 && judge_take(sn) == TakeVerdict::NonFinite);
    CHECK(contains(to_string(TakeVerdict::NonFinite), "non-finite"));
    // The level statistics are not polluted by the bad sample.
    CHECK(std::isfinite(sn.peak_dbfs) && std::isfinite(sn.rms_dbfs) && std::isfinite(sn.snr_db));

    std::vector<float> inf = sig;
    inf[7] = std::numeric_limits<float>::infinity();
    write_file(d / "inf.wav", float_file(inf));
    const WavStats si = analyze_wav_file(d / "inf.wav");
    CHECK(si.non_finite_samples == 1 && judge_take(si) == TakeVerdict::NonFinite);

    // In-memory analysis and the model path reject it too.
    CHECK(analyze_samples(nan.data(), nan.size(), 1, 16000).non_finite_samples == 1);
    VoiceStudioModel m;
    CHECK(m.add_take(d / "nan.wav", "s01").verdict == TakeVerdict::NonFinite);
    CHECK(m.clean_minutes() == 0.0 && m.excluded_minutes() > 0.0);
    CHECK(m.add_take(d / "finite.wav", "s01").verdict == TakeVerdict::Clean);
    // A rejected verdict survives the draft round trip.
    VoiceStudioModel back;
    CHECK(back.load_draft_json(m.draft_json()));
    CHECK(back.draft().takes.size() == 2 && back.draft().takes[0].verdict == TakeVerdict::NonFinite &&
          back.draft().takes[0].stats.non_finite_samples == 1);
}

// ---------------------------------------------------------------------------
// Asynchronous analysis
// ---------------------------------------------------------------------------
bool wait_for_analysis(VoiceStudioModel& m, std::size_t expected, std::size_t* applied) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        *applied += m.poll_analysis();
        if (*applied >= expected && !m.analysis_pending()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

void test_async() {
    const fs::path d = g_tmp / "async";
    const std::vector<float> sig = take_signal(3.0, 16000);
    write_file(d / "a.wav", pcm_file(sig, 16));
    write_file(d / "b.wav", pcm_file(sig, 24));
    write_file(d / "bad.wav", "not a wav");

    VoiceStudioModel m;
    CHECK(!m.analysis_pending() && m.poll_analysis() == 0);
    m.request_take_analysis(TakeKind::Take, d / "a.wav", "s01");
    m.request_take_analysis(TakeKind::RoomTone, d / "b.wav");
    m.request_take_analysis(TakeKind::LiveMic, d / "a.wav");
    m.request_take_analysis(TakeKind::Take, d / "bad.wav", "s02");
    CHECK(m.analysis_pending());
    CHECK(m.draft().takes.empty());  // nothing is applied before poll_analysis()
    std::size_t applied = 0;
    CHECK(wait_for_analysis(m, 4, &applied));
    CHECK(applied == 4 && !m.analysis_pending() && m.analysis_pending_count() == 0);
    CHECK(m.draft().takes.size() == 2 && m.draft().room_tone.size() == 1 && m.draft().live_mic.size() == 1);

    // Same result as the synchronous path (order of requests is kept).
    VoiceStudioModel sync;
    const Take& st = sync.add_take(d / "a.wav", "s01");
    const Take& at = m.draft().takes[0];
    CHECK(at.session == "s01" && at.verdict == st.verdict && at.stats.ok == st.stats.ok &&
          at.stats.seconds == st.stats.seconds && at.stats.peak_dbfs == st.stats.peak_dbfs &&
          at.stats.snr_db == st.stats.snr_db && at.stats.bits == st.stats.bits);
    CHECK(m.draft().takes[1].path == d / "bad.wav" && m.draft().takes[1].verdict == TakeVerdict::Unreadable);
    CHECK(m.draft().room_tone[0].session == "room-tone" && m.draft().live_mic[0].session == "live-mic");

    // Destroying a model with pending work must not crash or leak a thread.
    {
        VoiceStudioModel doomed;
        for (int i = 0; i < 20; ++i) doomed.request_take_analysis(TakeKind::Take, d / "a.wav", "s01");
    }
    // Moving a model with pending work is safe and the work follows the object.
    {
        VoiceStudioModel src;
        for (int i = 0; i < 5; ++i) src.request_take_analysis(TakeKind::Take, d / "b.wav", "s03");
        VoiceStudioModel moved(std::move(src));
        CHECK(!src.analysis_pending() && src.poll_analysis() == 0);  // NOLINT(bugprone-use-after-move): defined state
        CHECK(moved.analysis_pending());
        std::size_t got = 0;
        CHECK(wait_for_analysis(moved, 5, &got));
        CHECK(got == 5 && moved.draft().takes.size() == 5);

        VoiceStudioModel target;
        for (int i = 0; i < 10; ++i) target.request_take_analysis(TakeKind::Take, d / "a.wav", "old");
        VoiceStudioModel other;
        other.request_take_analysis(TakeKind::Take, d / "b.wav", "new");
        target = std::move(other);  // joins the overwritten model's worker
        std::size_t g2 = 0;
        CHECK(wait_for_analysis(target, 1, &g2));
        CHECK(g2 == 1 && target.draft().takes.size() == 1 && target.draft().takes[0].session == "new");
    }
}

// ---------------------------------------------------------------------------
// Room tone threshold
// ---------------------------------------------------------------------------
void test_room_tone() {
    const fs::path d = g_tmp / "room";
    CHECK(VoiceStudioModel::kRoomToneMinimumSeconds == 30.0);
    VoiceStudioModel m;
    CHECK(m.requirements().room_tone_seconds == 30.0);
    m.draft().voice_name = "me";
    m.draft().display_name = "Me";
    m.draft().consent = true;
    m.go_to(StudioStep::RoomTone);
    const auto quiet = [](double seconds) { return std::vector<float>(static_cast<std::size_t>(seconds * 16000.0 + 0.5), 0.0005f); };
    write_file(d / "short.wav", pcm_file(quiet(29.9), 16));
    write_file(d / "exact.wav", pcm_file(quiet(30.0), 16));
    CHECK(m.add_room_tone(d / "short.wav").verdict == TakeVerdict::Clean);
    std::string why;
    CHECK(!m.can_advance(&why) && contains(why, "30"));
    VoiceStudioModel m2;
    m2.go_to(StudioStep::RoomTone);
    CHECK(m2.add_room_tone(d / "exact.wav").verdict == TakeVerdict::Clean);
    CHECK(m2.room_tone_seconds() >= 30.0 && m2.can_advance());
    // The message follows the configured threshold (one constant, no hard-coded text).
    m2.requirements().room_tone_seconds = 45.0;
    CHECK(!m2.can_advance(&why) && contains(why, "45") && !contains(why, "30"));
}

// ---------------------------------------------------------------------------
// Project directory refusal (BUG-021 residual)
// ---------------------------------------------------------------------------
void test_project_dir() {
    VoiceStudioModel m;
    for (const char* bad : {"", "../x", "Me", "con", "a/b", "nul"}) {
        m.draft().voice_name = bad;
        CHECK(m.project_dir(g_tmp / "studio").empty());
        CHECK(!m.job_written(g_tmp / "studio"));
        CHECK(!m.poll_status(g_tmp / "studio").has_value());
        const StudioCommands c = m.commands(g_repo, g_tmp / "studio", g_tmp / "models", Shell::Posix);
        CHECK(c.prepare.empty() && c.finetune.empty() && c.export_.empty());
        CHECK(!c.prepare_unavailable.empty());
    }
    m.draft().voice_name = "me";
    CHECK(m.project_dir(g_tmp / "studio") == g_tmp / "studio" / "me");
}

// ---------------------------------------------------------------------------
// BUG-025: generated commands against the real scripts
// ---------------------------------------------------------------------------
std::set<std::string> script_options(const std::string& text) {
    // Every "--option" string literal that appears inside an add_argument(...) call.
    std::set<std::string> out;
    std::size_t pos = 0;
    while ((pos = text.find("add_argument(", pos)) != std::string::npos) {
        pos += 13;
        const std::size_t end = text.find(')', pos);
        const std::string call = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        std::size_t q = 0;
        while ((q = call.find("\"--", q)) != std::string::npos) {
            const std::size_t e = call.find('"', q + 1);
            if (e == std::string::npos) break;
            out.insert(call.substr(q + 1, e - q - 1));
            q = e + 1;
        }
    }
    return out;
}

/// Tokenise a command line (single-quoted words, '' / '\'' escapes) into words.
std::vector<std::string> words(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_q = false, have = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_q) {
            if (c == '\'') {
                if (i + 1 < line.size() && line[i + 1] == '\'') { cur += '\''; ++i; }
                else in_q = false;
            } else cur += c;
        } else if (c == '\'') {
            in_q = true;
            have = true;
        } else if (c == ' ') {
            if (have || !cur.empty()) out.push_back(cur);
            cur.clear();
            have = false;
        } else {
            cur += c;
        }
    }
    if (have || !cur.empty()) out.push_back(cur);
    return out;
}

void check_command_options(const std::string& cmd, bool powershell) {
    const std::vector<std::string> w = words(cmd);
    CHECK(w.size() >= 2);
    if (w.size() < 2) return;
    std::size_t i = 0;
    if (powershell) {
        CHECK(cmd.rfind("& ", 0) == 0);
        CHECK(w[0] == "&");
        i = 1;
    }
    ++i;  // the interpreter
    const fs::path script = w[i];
    CHECK(script.parent_path().filename() == "python_training");
    CHECK(fs::is_regular_file(script));
    const fs::path src = g_repo / "python_training" / script.filename();
    CHECK(fs::is_regular_file(src));
    const std::set<std::string> known = script_options(slurp(src));
    CHECK(!known.empty());
    for (std::size_t k = i + 1; k < w.size(); ++k) {
        if (w[k].rfind("--", 0) != 0) continue;
        const std::string opt = w[k].substr(0, w[k].find('='));
        if (known.count(opt) == 0) std::printf("unknown option %s for %s\n", opt.c_str(), script.string().c_str());
        CHECK(known.count(opt) == 1);
    }
}

void test_job_commands() {
    const fs::path in = g_tmp / "raw";
    const std::vector<float> sig = take_signal(2.5, 16000);
    VoiceStudioModel m;
    m.draft().voice_name = "me";
    m.draft().display_name = "Me (it's private)";
    m.draft().consent = true;
    m.draft().base_bundle = "identity";
    write_file(in / "me_s01_0001.wav", pcm_file(sig, 16));
    write_file(in / "me_s01_roomtone.wav", pcm_file(sig, 16));
    m.add_take(in / "me_s01_0001.wav", "s01");
    m.add_room_tone(in / "me_s01_roomtone.wav");

    const fs::path studio = g_tmp / "studio", models = g_tmp / "models";
    for (const Shell shell : {Shell::Posix, Shell::PowerShell}) {
        const StudioCommands c = m.commands(g_repo, studio, models, shell);
        CHECK(!c.prepare.empty() && c.prepare_unavailable.empty());
        check_command_options(c.prepare, shell == Shell::PowerShell);
        CHECK(contains(c.prepare, "prepare_target_voice.py") && contains(c.prepare, "--speaker=me"));
        CHECK(contains(c.prepare, "prepared"));
        // Steps the existing CLIs cannot express are not generated and say why.
        CHECK(c.finetune.empty() && c.export_.empty());
        CHECK(contains(c.finetune_unavailable, "unavailable until M2 Stage 10 lands"));
        CHECK(contains(c.export_unavailable, "unavailable until M2 Stage 10 lands"));
    }
    // Quoting: the display name keeps its apostrophe in both shells.
    CHECK(contains(m.commands(g_repo, studio, models, Shell::PowerShell).prepare, "'Me (it''s private)'"));
    CHECK(contains(m.commands(g_repo, studio, models, Shell::Posix).prepare, "'Me (it'\\''s private)'"));

    // Written job files: only available steps, PowerShell lines are `& ...` + exit-code check.
    std::string err;
    CHECK(m.write_job(g_repo, studio, models, err) && err.empty());
    const std::string sh = slurp(studio / "me" / "run_finetune.sh");
    const std::string ps = slurp(studio / "me" / "run_finetune.ps1");
    CHECK(sh.rfind("#!/bin/sh\nset -eu\n", 0) == 0);
    CHECK(contains(sh, "prepare_target_voice.py") && contains(ps, "prepare_target_voice.py"));
    for (const std::string* f : {&sh, &ps}) {
        CHECK(!contains(*f, "train_unit_vocoder.py") && !contains(*f, "export_stateful_onnx.py"));
        CHECK(!contains(*f, "--fine-tune") && !contains(*f, "--base-bundle"));
        CHECK(contains(*f, "step 'finetune' not generated: unavailable"));
        CHECK(contains(*f, "step 'export' not generated: unavailable"));
    }
    {
        std::istringstream in_ps(ps);
        std::string line, prev;
        int commands = 0;
        while (std::getline(in_ps, line)) {
            if (line.empty() || line[0] == '#' || line[0] == '$') { prev = line; continue; }
            if (line.rfind("& ", 0) == 0) {
                ++commands;
                std::string next;
                CHECK(std::getline(in_ps, next) && next == "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }");
            } else {
                CHECK(line.rfind("if ($LASTEXITCODE", 0) == 0);  // only the check lines may follow
            }
        }
        CHECK(commands == 1);
    }
    const std::string job = slurp(studio / "me" / "job.json");
    CHECK(contains(job, "\"unavailable\"") && contains(job, "prepare_target_voice.py"));
    CHECK(!contains(job, "train_unit_vocoder.py") && !contains(job, "export_stateful_onnx.py"));

    // Files that do not follow the scanned naming, or live in several folders, disable the prepare step.
    VoiceStudioModel odd;
    odd.draft().voice_name = "me";
    write_file(in / "take.wav", pcm_file(sig, 16));
    odd.add_take(in / "take.wav", "s01");
    StudioCommands c = odd.commands(g_repo, studio, models, Shell::Posix);
    CHECK(c.prepare.empty() && contains(c.prepare_unavailable, "me_sNN_TTTT.wav"));
    write_file(g_tmp / "raw2" / "me_s01_0002.wav", pcm_file(sig, 16));
    VoiceStudioModel split;
    split.draft().voice_name = "me";
    split.add_take(in / "me_s01_0001.wav", "s01");
    split.add_take(g_tmp / "raw2" / "me_s01_0002.wav", "s01");
    c = split.commands(g_repo, studio, models, Shell::Posix);
    CHECK(c.prepare.empty() && contains(c.prepare_unavailable, "same folder"));
    CHECK(VoiceStudioModel{}.commands(g_repo, studio, models, Shell::Posix).prepare.empty());
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_repo = argv[1];
    g_tmp = fs::temp_directory_path() / ("fv_studio_test_" + std::to_string(static_cast<unsigned long long>(
                std::chrono::steady_clock::now().time_since_epoch().count())));
    std::error_code ec;
    fs::remove_all(g_tmp, ec);
    fs::create_directories(g_tmp);

    test_hostile_wavs();
    test_non_finite();
    test_async();
    test_room_tone();
    test_project_dir();
    test_job_commands();

    fs::remove_all(g_tmp, ec);
    return test::finish("frontend_studio");
}
