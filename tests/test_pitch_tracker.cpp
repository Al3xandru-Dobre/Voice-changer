// PitchTracker — C++ port vs. the Python reference (Milestone 2 / Stage 03).
//
// Usage: test_pitch_tracker <tests/data/pitch> [--bench FRAMES]
//
// Covers:
//   * the C++ defaults equal the constants in header.json (Python's values)
//   * golden vectors from python_training/make_pitch_golden.py, lookahead
//     0/1/2: |df0| <= 0.1 Hz on frames voiced in both, voicing and the voiced
//     decision identical on >= 99.5% of frames, |dlog_energy| <= 1e-4
//   * push() performs no heap allocation after construction
//   * reset() replays a fresh tracker exactly
//   * whitening converges to the stats of a synthetic AR(1) log-f0 track
//   * the target mapping reproduces closed-form values
//   * prints the per-frame cost of push() (Stage 03 DoD: <= 0.3 ms, Release)
#include "dsp/PitchTracker.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <random>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "golden vectors are little-endian; add a byte swap for this host");

// ThreadSanitizer ships its own operator new/delete, which a replacement would
// collide with at link time; the allocation check is skipped in TSan builds.
#if defined(__SANITIZE_THREAD__)
#define FLUXVOICE_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FLUXVOICE_TSAN 1
#endif
#endif

#ifndef FLUXVOICE_TSAN
static std::atomic<long> g_allocs{0};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif // FLUXVOICE_TSAN

namespace {

constexpr int kColumns = 7;   // header["features"]
enum Col { kF0, kAper, kVoicing, kLogEnergy, kHeld, kZLogf0, kZEnergy };

std::string g_dir = "tests/data/pitch";

// ---------------------------------------------------------------------------
// Minimal readers for the flat header.json (numbers, number arrays, string
// arrays at known keys) and the raw little-endian files.
// ---------------------------------------------------------------------------
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

size_t value_pos(const std::string& json, const char* key) {
    const std::string quoted = std::string("\"") + key + "\"";
    size_t p = json.find(quoted);
    if (p == std::string::npos) {
        std::printf("header.json has no key %s\n", key);
        std::exit(2);
    }
    p = json.find(':', p + quoted.size());
    return json.find_first_not_of(" \t\r\n", p + 1);
}

double json_number(const std::string& json, const char* key) {
    return std::strtod(json.c_str() + value_pos(json, key), nullptr);
}

std::vector<double> json_numbers(const std::string& json, const char* key) {
    std::vector<double> out;
    const char* p = json.c_str() + value_pos(json, key) + 1;   // after '['
    while (true) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') ++p;
        if (*p == ']') return out;
        char* end = nullptr;
        out.push_back(std::strtod(p, &end));
        p = end;
    }
}

std::vector<std::string> json_strings(const std::string& json, const char* key) {
    std::vector<std::string> out;
    size_t p = value_pos(json, key) + 1;
    const size_t close = json.find(']', p);
    while (true) {
        const size_t a = json.find('"', p);
        if (a == std::string::npos || a > close) return out;
        const size_t b = json.find('"', a + 1);
        out.push_back(json.substr(a + 1, b - a - 1));
        p = b + 1;
    }
}

std::vector<float> read_pcm16(const std::string& path) {
    const std::string raw = read_file(path);
    std::vector<float> x(raw.size() / 2);
    for (size_t i = 0; i < x.size(); ++i) {
        std::int16_t v;
        std::memcpy(&v, raw.data() + 2 * i, 2);
        x[i] = static_cast<float>(v) / 32768.0f;
    }
    return x;
}

std::vector<float> read_f32(const std::string& path) {
    const std::string raw = read_file(path);
    std::vector<float> x(raw.size() / 4);
    std::memcpy(x.data(), raw.data(), x.size() * 4);
    return x;
}

bool close_rel(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b));
}

PitchTrackerConfig config_from_header(const std::string& h, int lookahead) {
    PitchTrackerConfig c;
    c.sample_rate = static_cast<int>(json_number(h, "sample_rate"));
    c.frame_samples = static_cast<int>(json_number(h, "frame_samples"));
    c.window_samples = static_cast<int>(json_number(h, "window_samples"));
    c.f0_min_hz = json_number(h, "f0_min_hz");
    c.f0_max_hz = json_number(h, "f0_max_hz");
    const std::vector<double> t = json_numbers(h, "thresholds");
    CHECK(t.size() == c.thresholds.size());
    for (size_t i = 0; i < c.thresholds.size() && i < t.size(); ++i) c.thresholds[i] = t[i];
    c.f0_threshold = json_number(h, "f0_threshold");
    c.voiced_max_aperiodicity = json_number(h, "voiced_max_aperiodicity");
    c.voiced_min_z_energy = json_number(h, "voiced_min_z_energy");
    c.hpf_hz = json_number(h, "hpf_hz");
    c.hpf_q = json_number(h, "hpf_q");
    c.energy_eps = json_number(h, "energy_eps");
    c.silence_mean_square = json_number(h, "silence_mean_square");
    c.norm_tau_s = json_number(h, "norm_tau_s");
    c.logf0_std_floor = json_number(h, "logf0_std_floor");
    c.energy_std_floor = json_number(h, "energy_std_floor");
    c.hold_relax_s = json_number(h, "hold_relax_s");
    c.logf0_prior_mean = json_number(h, "logf0_prior_mean");
    c.logf0_prior_std = json_number(h, "logf0_prior_std");
    c.energy_prior_mean = json_number(h, "energy_prior_mean");
    c.energy_prior_std = json_number(h, "energy_prior_std");
    c.lookahead_frames = lookahead;
    return c;
}

/// Frames of `x` (zero-padded to whole frames + L lookahead frames), aligned
/// like python's features.extract(): the L leading outputs are dropped.
std::vector<PitchFrame> run(PitchTracker& tr, const std::vector<float>& x) {
    const size_t hop = tr.frame_samples();
    const size_t la = static_cast<size_t>(tr.config().lookahead_frames);
    const size_t frames = (x.size() + hop - 1) / hop;
    std::vector<float> padded((frames + la) * hop, 0.0f);
    std::copy(x.begin(), x.end(), padded.begin());
    std::vector<PitchFrame> out;
    out.reserve(frames + la);
    for (size_t i = 0; i < frames + la; ++i) out.push_back(tr.push(padded.data() + i * hop));
    out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(la));
    return out;
}

// ---------------------------------------------------------------------------

void test_defaults_match_header() {
    const std::string h = read_file(g_dir + "/header.json");
    const PitchTrackerConfig d;
    const PitchTrackerConfig j = config_from_header(h, 0);
    CHECK(d.sample_rate == j.sample_rate);
    CHECK(d.frame_samples == j.frame_samples);
    CHECK(d.window_samples == j.window_samples);
    CHECK(close_rel(d.f0_min_hz, j.f0_min_hz));
    CHECK(close_rel(d.f0_max_hz, j.f0_max_hz));
    for (size_t i = 0; i < d.thresholds.size(); ++i) CHECK(close_rel(d.thresholds[i], j.thresholds[i]));
    CHECK(close_rel(d.f0_threshold, j.f0_threshold));
    CHECK(close_rel(d.voiced_max_aperiodicity, j.voiced_max_aperiodicity));
    CHECK(close_rel(d.voiced_min_z_energy, j.voiced_min_z_energy));
    CHECK(close_rel(d.hpf_hz, j.hpf_hz));
    CHECK(close_rel(d.hpf_q, j.hpf_q));
    CHECK(close_rel(d.energy_eps, j.energy_eps));
    CHECK(close_rel(d.silence_mean_square, j.silence_mean_square));
    CHECK(close_rel(d.norm_tau_s, j.norm_tau_s));
    CHECK(close_rel(d.logf0_std_floor, j.logf0_std_floor));
    CHECK(close_rel(d.energy_std_floor, j.energy_std_floor));
    CHECK(close_rel(d.hold_relax_s, j.hold_relax_s));
    CHECK(close_rel(d.logf0_prior_mean, j.logf0_prior_mean));
    CHECK(close_rel(d.logf0_prior_std, j.logf0_prior_std));
    CHECK(close_rel(d.energy_prior_mean, j.energy_prior_mean));
    CHECK(close_rel(d.energy_prior_std, j.energy_prior_std));

    // Derived values and the filter coefficients.
    CHECK(d.tau_min() == static_cast<int>(json_number(h, "tau_min")));
    CHECK(d.tau_max() == static_cast<int>(json_number(h, "tau_max")));
    CHECK(d.integration_samples() == static_cast<int>(json_number(h, "integration_samples")));
    CHECK(d.valid());
    const BiquadCoefficients c = BiquadCoefficients::high_pass(d.hpf_hz, d.sample_rate, d.hpf_q);
    const std::vector<double> b = json_numbers(h, "hpf_b");
    const std::vector<double> a = json_numbers(h, "hpf_a");
    CHECK(b.size() == 3 && a.size() == 3);
    if (b.size() == 3 && a.size() == 3) {
        CHECK(close_rel(c.b0, b[0]) && close_rel(c.b1, b[1]) && close_rel(c.b2, b[2]));
        CHECK(a[0] == 1.0 && close_rel(c.a1, a[1]) && close_rel(c.a2, a[2]));
    }
    CHECK(static_cast<int>(json_strings(h, "features").size()) == kColumns);
}

void test_golden_vectors() {
    const std::string h = read_file(g_dir + "/header.json");
    const std::vector<std::string> cases = json_strings(h, "cases");
    CHECK(cases.size() >= 6);
    for (const std::string& name : cases) {
        const size_t us = name.rfind("_L");
        const int la = std::atoi(name.c_str() + us + 2);
        const std::vector<float> x = read_pcm16(g_dir + "/" + name.substr(0, us) + ".pcm16");
        const std::vector<float> want = read_f32(g_dir + "/" + name + ".f32");

        PitchTracker tr(config_from_header(h, la));
        const std::vector<PitchFrame> got = run(tr, x);
        CHECK(want.size() == got.size() * kColumns);
        if (want.size() != got.size() * kColumns) continue;

        size_t voicing_same = 0, decision_same = 0, f0_bad = 0, energy_bad = 0, other_bad = 0;
        double max_df0 = 0.0, max_de = 0.0;
        for (size_t t = 0; t < got.size(); ++t) {
            const float* w = want.data() + t * kColumns;
            const PitchFrame& g = got[t];
            voicing_same += g.voicing == w[kVoicing] ? 1 : 0;
            decision_same += (g.f0_hz > 0.0f) == (w[kF0] > 0.0f) ? 1 : 0;
            if (g.f0_hz > 0.0f && w[kF0] > 0.0f) {
                const double df = std::fabs(g.f0_hz - w[kF0]);
                max_df0 = std::max(max_df0, df);
                f0_bad += df > 0.1 ? 1 : 0;
            }
            const double de = std::fabs(g.log_energy - w[kLogEnergy]);
            max_de = std::max(max_de, de);
            energy_bad += de > 1e-4 ? 1 : 0;
            other_bad += std::fabs(g.aperiodicity - w[kAper]) > 1e-4 ||
                         std::fabs(g.z_energy - w[kZEnergy]) > 1e-4 ||
                         std::fabs(g.logf0_held - w[kHeld]) > 1e-4 ||
                         std::fabs(g.z_logf0 - w[kZLogf0]) > 1e-3 ? 1 : 0;
        }
        const double n = static_cast<double>(got.size());
        std::printf("  %-13s frames %3zu  voicing same %.4f  decision same %.4f  max|df0| %.2e Hz"
                    "  max|dlogE| %.2e\n",
                    name.c_str(), got.size(), voicing_same / n, decision_same / n, max_df0, max_de);
        CHECK(voicing_same >= 0.995 * n);
        CHECK(decision_same >= 0.995 * n);
        CHECK(f0_bad == 0);
        CHECK(energy_bad == 0);
        CHECK(other_bad == 0);
    }
}

void test_lookahead_delays_output() {
    // A tone that starts at frame 10: with L = 2, the first voiced output is
    // two pushes later than with L = 0, and describes the same frame.
    std::vector<float> x(320 * 40, 0.0f);
    for (size_t i = 320 * 10; i < x.size(); ++i)
        x[i] = 0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 150.0 * i / 16000.0));
    auto first_voiced_push = [&](int la) {
        PitchTrackerConfig c;
        c.lookahead_frames = la;
        PitchTracker tr(c);
        for (size_t p = 0; p < 40; ++p)
            if (tr.push(x.data() + p * 320).f0_hz > 0.0f) return static_cast<int>(p);
        return -1;
    };
    const int p0 = first_voiced_push(0);
    const int p2 = first_voiced_push(2);
    CHECK(p0 >= 10);
    CHECK(p2 >= 0 && p2 - 2 <= p0 && p2 - 2 >= p0 - 1);   // centred window may see it a frame early
}

void test_no_allocation_and_reset() {
    const std::vector<float> x = read_pcm16(g_dir + "/libri.pcm16");
    PitchTrackerConfig c;
    c.lookahead_frames = 1;
    PitchTracker tr(c);
    const size_t frames = x.size() / 320;
    std::vector<PitchFrame> first(frames), second(frames);

#ifndef FLUXVOICE_TSAN
    const long before = g_allocs.load();
#endif
    for (size_t i = 0; i < frames; ++i) first[i] = tr.push(x.data() + i * 320);
    tr.reset();
    for (size_t i = 0; i < frames; ++i) second[i] = tr.push(x.data() + i * 320);
#ifndef FLUXVOICE_TSAN
    CHECK(g_allocs.load() == before);
#endif
    bool same = true;
    for (size_t i = 0; i < frames; ++i)
        same = same && std::memcmp(&first[i], &second[i], sizeof(PitchFrame)) == 0;
    CHECK(same);
}

void test_whitening_converges() {
    // AR(1) log-f0 around log(120 Hz) with stationary std 0.2, from the
    // population prior (5.087 +- 0.427): after 20 s the prior is forgotten,
    // and over the next 200 s the EMA tracks the true stats.
    const double mu = std::log(120.0), sd = 0.2, phi = 0.9;
    const double sigma_e = sd * std::sqrt(1.0 - phi * phi);
    const PitchTrackerConfig c;
    RunningNorm norm(c.logf0_prior_mean, c.logf0_prior_std, c.norm_alpha(), c.logf0_std_floor);
    std::mt19937 rng(7);
    auto gauss = [&rng]() {   // Box-Muller: same numbers on every standard library
        const double u1 = (static_cast<double>(rng()) + 1.0) / 4294967297.0;
        const double u2 = static_cast<double>(rng()) / 4294967296.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
    };
    double x = mu;
    for (int t = 0; t < 1000; ++t) {   // 20 s
        x = mu + phi * (x - mu) + sigma_e * gauss();
        norm.update(x);
    }
    CHECK(std::fabs(norm.mean() - mu) < 0.1);
    double mean_sum = 0.0, std_sum = 0.0, z_sq = 0.0;
    const int n = 10000;               // 200 s
    for (int t = 0; t < n; ++t) {
        x = mu + phi * (x - mu) + sigma_e * gauss();
        norm.update(x);
        mean_sum += norm.mean();
        std_sum += norm.std();
        z_sq += norm.z(x) * norm.z(x);
    }
    std::printf("  AR(1): mean %.4f (true %.4f)  std %.4f (true %.4f)  rms z %.3f\n",
                mean_sum / n, mu, std_sum / n, sd, std::sqrt(z_sq / n));
    CHECK(std::fabs(mean_sum / n - mu) < 0.02);
    CHECK(std::fabs(std_sum / n - sd) < 0.15 * sd);
    CHECK(std::fabs(std::sqrt(z_sq / n) - 1.0) < 0.15);

    // Constant input: the std floor keeps z finite.
    RunningNorm flat(1.0, 0.0, c.norm_alpha(), c.logf0_std_floor);
    for (int t = 0; t < 100; ++t) flat.update(1.0);
    CHECK(flat.std() == c.logf0_std_floor);
    CHECK(std::isfinite(flat.z(5.0)));
}

void test_target_mapping() {
    const TargetVoiceStats v{std::log(200.0), 0.15, -6.0, 2.0};
    // z = 0 lands on the target's mean; +12 semitones doubles f0.
    CHECK(std::fabs(map_logf0_to_target(0.0, v) - std::log(200.0)) < 1e-12);
    CHECK(std::fabs(std::exp(map_logf0_to_target(0.0, v, 1.0, 12.0)) - 400.0) < 1e-9);
    CHECK(std::fabs(std::exp(map_logf0_to_target(0.0, v, 1.0, -12.0)) - 100.0) < 1e-9);
    // mu + scale * sigma * z + shift * ln2 / 12, by hand.
    CHECK(std::fabs(map_logf0_to_target(2.0, v, 0.5, 3.0) -
                    (std::log(200.0) + 0.5 * 0.15 * 2.0 + 3.0 * std::log(2.0) / 12.0)) < 1e-12);
    // scale = 0 flattens intonation.
    CHECK(map_logf0_to_target(-1.7, v, 0.0) == map_logf0_to_target(2.3, v, 0.0));
    CHECK(std::fabs(map_energy_to_target(1.5, v) - (-6.0 + 2.0 * 1.5)) < 1e-12);
}

void bench(size_t frames) {
    const std::vector<float> x = read_pcm16(g_dir + "/libri.pcm16");
    const size_t n = x.size() / 320;
    for (int la : {0, 1}) {
        PitchTrackerConfig c;
        c.lookahead_frames = la;
        PitchTracker tr(c);
        std::vector<double> us(frames);
        float sink = 0.0f;
        for (size_t i = 0; i < frames; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            sink += tr.push(x.data() + (i % n) * 320).f0_hz;
            us[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
        std::sort(us.begin(), us.end());
        double mean = 0.0;
        for (double v : us) mean += v;
        std::printf("  push() L=%d: mean %.1f us  p50 %.1f  p99 %.1f  max %.1f per 20 ms frame"
                    " (%zu frames, checksum %.1f)\n",
                    la, mean / frames, us[frames / 2], us[frames * 99 / 100], us.back(), frames,
                    static_cast<double>(sink));
    }
}

} // namespace

int main(int argc, char** argv) {
    size_t bench_frames = 2000;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) bench_frames = std::strtoul(argv[++i], nullptr, 10);
        else g_dir = argv[i];
    }
    test_defaults_match_header();
    test_golden_vectors();
    test_lookahead_delays_output();
    test_no_allocation_and_reset();
    test_whitening_converges();
    test_target_mapping();
    bench(std::max<size_t>(bench_frames, 100));
    return test::finish("pitch_tracker");
}
