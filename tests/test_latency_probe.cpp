// LatencyProbe — marker detector tests (Milestone 1 / Stage 06).
//
// Each test builds the probe's marker stream, passes it through a simulated
// route (delay, gain, noise, low-pass filter, dropouts) and checks that the
// detector recovers the route's delay.
#include "LatencyProbe.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr double kRate = 48000.0;
constexpr size_t kMarkerLen = 960;  // 20 ms
// Noise moves the interpolated peak by up to ~0.07 samples (1.5 us at 48 kHz).
constexpr double kNoisyTol = 0.25;

std::vector<float> marker() { return probe::make_chirp(kRate, kMarkerLen, 500.0, 3500.0, 0.5); }

probe::MarkerSchedule schedule(size_t count = 12) {
    probe::MarkerSchedule s;
    s.first = 4800;        // 100 ms
    s.interval = 12000;    // 250 ms
    s.count = count;
    return s;
}

/// Deterministic white noise in [-amp, amp].
struct Noise {
    uint32_t state = 12345;
    float next(float amp) {
        state = state * 1664525u + 1013904223u;
        return amp * (static_cast<float>(state >> 8) / 8388608.0f - 1.0f);
    }
};

/// Route that delays marker k by delays[k] samples, scales and adds noise.
std::vector<float> route(const std::vector<float>& m, const probe::MarkerSchedule& s,
                         const std::vector<size_t>& delays, float gain, float noise) {
    std::vector<float> y(static_cast<size_t>(s.total()), 0.0f);
    for (size_t k = 0; k < s.count; ++k)
        for (size_t j = 0; j < m.size(); ++j) {
            const size_t n = static_cast<size_t>(s.start(k)) + delays[k] + j;
            if (n < y.size()) y[n] += gain * m[j];
        }
    Noise rng;
    for (float& v : y) v += rng.next(noise);
    return y;
}

probe::MarkerDetector detect(const std::vector<float>& rec, const probe::MarkerSchedule& s) {
    probe::MarkerDetector det(marker(), s);
    det.push(rec.data(), rec.size());
    return det;
}

void test_marker_shape() {
    const auto m = marker();
    CHECK(m.size() == kMarkerLen);
    CHECK(m.front() == 0.0f);
    CHECK(std::abs(m.back()) < 1e-6f);
    float peak = 0.0f;
    for (float v : m) peak = std::max(peak, std::abs(v));
    CHECK(peak > 0.3f && peak <= 0.5f);

    const auto s = schedule(3);
    CHECK(probe::marker_sample(m, s, 0) == 0.0f);
    CHECK(probe::marker_sample(m, s, s.first + 100) == m[100]);
    CHECK(probe::marker_sample(m, s, s.start(2) + 5) == m[5]);
    CHECK(probe::marker_sample(m, s, s.start(1) + kMarkerLen) == 0.0f);  // gap
    CHECK(probe::marker_sample(m, s, s.total() + 5) == 0.0f);            // after the last
}

void test_constant_delay_with_noise() {
    const auto s = schedule();
    const std::vector<size_t> delays(s.count, 3217);
    const auto det = detect(route(marker(), s, delays, 0.2f, 0.02f), s);
    CHECK(det.done());
    for (const auto& d : det.detections()) {
        CHECK(d.found);
        CHECK(std::abs(d.lag - 3217.0) < kNoisyTol);
        CHECK(d.score > 0.9);
    }
    const auto sum = probe::summarize(det.detections(), s, kRate);
    CHECK(sum.found == s.count && sum.missed == 0);
    CHECK(std::abs(sum.median - 3217.0) < kNoisyTol);
    CHECK(std::abs(sum.slope_per_min) < 3.0);  // samples/min; noise alone, over 3 s of markers
}

void test_zero_and_maximum_delay() {
    const auto s = schedule(4);
    const size_t max_delay = static_cast<size_t>(s.interval) - kMarkerLen;
    const auto det0 = detect(route(marker(), s, std::vector<size_t>(s.count, 0), 1.0f, 0.0f), s);
    const auto detm = detect(route(marker(), s, std::vector<size_t>(s.count, max_delay), 1.0f, 0.0f), s);
    for (size_t k = 0; k < s.count; ++k) {
        CHECK(det0.detections()[k].found && det0.detections()[k].lag == 0.0);
        CHECK(detm.detections()[k].found &&
              detm.detections()[k].lag == static_cast<double>(max_delay));
    }
}

void test_piecewise_push_matches_whole() {
    const auto s = schedule();
    const auto rec = route(marker(), s, std::vector<size_t>(s.count, 1500), 0.5f, 0.01f);
    const auto whole = detect(rec, s);

    probe::MarkerDetector det(marker(), s);
    const size_t sizes[] = {1, 480, 441, 7, 4096, 160};
    size_t at = 0, i = 0;
    while (at < rec.size()) {
        const size_t n = std::min(sizes[i++ % 6], rec.size() - at);
        det.push(rec.data() + at, n);
        at += n;
    }
    CHECK(det.done());
    for (size_t k = 0; k < s.count; ++k) {
        CHECK(det.detections()[k].found == whole.detections()[k].found);
        CHECK(det.detections()[k].lag == whole.detections()[k].lag);
    }
}

void test_exact_sample_count_completes() {
    // fv_probe records exactly total() samples; that must close every window.
    const auto s = schedule(3);
    const auto rec = route(marker(), s, std::vector<size_t>(s.count, 700), 0.5f, 0.0f);
    probe::MarkerDetector det(marker(), s);
    det.push(rec.data(), rec.size() - 1);
    CHECK(!det.done() && det.analysed() == s.count - 1);
    det.push(rec.data() + rec.size() - 1, 1);
    CHECK(det.done() && det.analysed() == s.count);
}

/// Linear-phase low-pass (windowed sinc, 4 kHz cutoff), like a resampler's
/// filter: delays every frequency by exactly (taps - 1) / 2 samples.
std::vector<float> lowpass(const std::vector<float>& x, size_t taps) {
    std::vector<double> h(taps);
    const double centre = static_cast<double>(taps - 1) / 2.0;
    double sum = 0;
    for (size_t i = 0; i < taps; ++i) {
        const double t = static_cast<double>(i) - centre;
        const double fc = 4000.0 / kRate;
        const double sinc = t == 0.0 ? 2 * fc : std::sin(2 * probe::kPi * fc * t) / (probe::kPi * t);
        const double w = 0.54 - 0.46 * std::cos(2 * probe::kPi * static_cast<double>(i) /
                                                static_cast<double>(taps - 1));
        h[i] = sinc * w;
        sum += h[i];
    }
    std::vector<float> y(x.size(), 0.0f);
    for (size_t n = 0; n < x.size(); ++n) {
        double acc = 0;
        for (size_t i = 0; i < taps && i <= n; ++i) acc += h[i] / sum * x[n - i];
        y[n] = static_cast<float>(acc);
    }
    return y;
}

void test_lowpass_route() {
    const auto s = schedule();
    const auto rec = route(marker(), s, std::vector<size_t>(s.count, 2000), 0.5f, 0.0f);
    // 63 taps: whole-sample delay. 64 taps: a half-sample delay, which only
    // the sub-sample interpolation can resolve.
    for (const auto& [taps, expected] : {std::pair<size_t, double>{63, 2031.0}, {64, 2031.5}}) {
        const auto det = detect(lowpass(rec, taps), s);
        for (const auto& d : det.detections()) {
            CHECK(d.found);
            CHECK(std::abs(d.lag - expected) < 0.05);
        }
    }
}

void test_drift_and_missing_markers() {
    // Delay grows 3 samples per marker; marker 5 never arrives (dropout).
    const auto s = schedule(20);
    std::vector<size_t> delays(s.count);
    for (size_t k = 0; k < s.count; ++k) delays[k] = 1000 + 3 * k;
    auto rec = route(marker(), s, delays, 0.5f, 0.01f);
    for (size_t n = 0; n < s.interval; ++n) rec[static_cast<size_t>(s.start(5) + n)] = 0.0f;

    const auto det = detect(rec, s);
    CHECK(!det.detections()[5].found);
    for (size_t k = 0; k < s.count; ++k)
        if (k != 5) CHECK(det.detections()[k].found && std::abs(det.detections()[k].lag - delays[k]) < kNoisyTol);

    const auto sum = probe::summarize(det.detections(), s, kRate);
    CHECK(sum.found == s.count - 1 && sum.missed == 1);
    CHECK(sum.min == det.detections()[0].lag);
    CHECK(std::abs(sum.max - delays.back()) < kNoisyTol);
    // 3 samples per 250 ms interval = 720 samples per minute.
    CHECK(std::abs(sum.slope_per_min - 720.0) < 1.0);
}

void test_noise_only_is_not_a_marker() {
    const auto s = schedule(6);
    std::vector<float> rec(static_cast<size_t>(s.total()));
    Noise rng;
    for (float& v : rec) v = rng.next(0.3f);
    const auto det = detect(rec, s);
    for (const auto& d : det.detections()) CHECK(!d.found && d.score < 0.3);
    CHECK(probe::summarize(det.detections(), s, kRate).found == 0);
}

void test_percentiles() {
    probe::MarkerSchedule s = schedule(20);
    std::vector<probe::Detection> dets(20);
    for (size_t k = 0; k < 20; ++k) { dets[k].found = true; dets[k].lag = static_cast<double>(k + 1); }
    const auto sum = probe::summarize(dets, s, kRate);
    CHECK(sum.median == 10.5);
    CHECK(sum.p95 == 19.0);  // nearest rank: ceil(0.95 * 20) = 19th
    CHECK(sum.min == 1.0 && sum.max == 20.0);
}

void test_speech_burst_shape() {
    for (double rate : {48000.0, 16000.0}) {
        const auto m = probe::make_speech_burst(rate, 0.5);
        const std::size_t expected_len =
            static_cast<std::size_t>(std::llround(rate * probe::kSpeechBurstSeconds));
        CHECK(m.size() == expected_len);
        CHECK(m.front() == 0.0f);
        CHECK(m.back() == 0.0f);
        float peak = 0.0f;
        for (float v : m) {
            CHECK(std::isfinite(v));
            peak = std::max(peak, std::abs(v));
        }
        CHECK(peak > 0.45f && peak <= 0.5001f);

        auto rms_slice = [&](double t0, double t1) {
            const auto a = static_cast<std::size_t>(t0 * rate);
            const auto b = std::min(m.size(), static_cast<std::size_t>(t1 * rate));
            double s = 0.0;
            for (std::size_t i = a; i < b; ++i) s += static_cast<double>(m[i]) * m[i];
            return std::sqrt(s / static_cast<double>(b - a));
        };
        const double pre  = rms_slice(0.000, 0.018);
        const double body = rms_slice(0.030, 0.115);
        const double post = rms_slice(0.130, 0.150);
        CHECK(body > 0.08);
        CHECK(pre == 0.0 && post == 0.0);
    }
}

/// Signal-driven causal VC simulator: replaces the input's pitch and timbre by
/// tracking short-time low/high-band envelopes, re-exciting shifted vocal-tract
/// formants with a target-pitch glottal carrier + unvoiced aspiration, applying
/// soft limiting, delaying by `delays[k]` samples, and adding white noise.
std::vector<float> simulate_vc_stream(const std::vector<float>& m,
                                      const probe::MarkerSchedule& s,
                                      const std::vector<std::size_t>& delays,
                                      double rate, double target_f0_hz,
                                      double formant_scale, float gain, float noise_amp) {
    const auto clean = route(m, s, delays, 1.0f, 0.0f);
    std::vector<float> out(clean.size(), 0.0f);

    const double env_alpha = 1.0 - std::exp(-1.0 / (0.0012 * rate));
    const double lp_alpha  = 1.0 - std::exp(-2.0 * probe::kPi * 900.0 / rate);
    auto f1 = probe::detail::Resonator::make(rate, 650.0 * formant_scale, 100.0);
    auto f2 = probe::detail::Resonator::make(rate, 1450.0 * formant_scale, 130.0);
    auto f3 = probe::detail::Resonator::make(rate, 2550.0 * formant_scale, 180.0);
    auto asp = probe::detail::Resonator::make(rate, 2300.0 * formant_scale, 900.0);

    const double max_harm_hz = std::min(3800.0, 0.44 * rate);
    double lp = 0.0, env_voicing = 0.0, env_fricative = 0.0;
    double phase = 1.3;  // arbitrary carrier phase unrelated to source
    Noise carrier_rng{987654321u};
    Noise ch_noise{24681357u};

    for (std::size_t n = 0; n < clean.size(); ++n) {
        const double x = clean[n];
        lp += lp_alpha * (x - lp);
        const double hp = x - lp;
        env_voicing   += env_alpha * (std::abs(lp) - env_voicing);
        env_fricative += env_alpha * (std::abs(hp) - env_fricative);

        const double f0 = target_f0_hz * (1.0 + 0.04 * std::sin(2.0 * probe::kPi * 5.5 * n / rate));
        phase = std::fmod(phase + 2.0 * probe::kPi * f0 / rate, 2.0 * probe::kPi);
        const int n_harm = std::min(22, std::max(1, static_cast<int>(max_harm_hz / f0)));
        double glottal = 0.0;
        for (int h = 1; h <= n_harm; ++h)
            glottal += std::sin(static_cast<double>(h) * phase) / (1.0 + 0.09 * h * h);

        const double voiced = 1.00 * f1.step(glottal) +
                              0.75 * f2.step(glottal) +
                              0.40 * f3.step(glottal);
        const double unvoiced = asp.step(carrier_rng.next(1.0f));
        const double y = 2.6 * env_voicing * voiced + 2.2 * env_fricative * unvoiced;
        out[n] = gain * static_cast<float>(std::tanh(1.4 * y)) + ch_noise.next(noise_amp);
    }
    return out;
}

void test_speech_vc_simulated_route() {
    // 1. Verify why Stage 11 needs the speech envelope marker: the waveform
    //    chirp correlator fails completely once a VC model replaces pitch/timbre.
    {
        probe::MarkerSchedule s;
        s.first = 4800;
        s.interval = 24000;  // 500 ms
        s.count = 6;
        const auto c = marker();
        const std::vector<std::size_t> delays(s.count, 3217);
        const auto vc_chirp = simulate_vc_stream(c, s, delays, kRate, 215.0, 1.22, 0.4f, 0.01f);
        probe::MarkerDetector chirp_det(c, s, 0.5, probe::MarkerKind::Chirp, kRate);
        chirp_det.push(vc_chirp.data(), vc_chirp.size());
        CHECK(probe::summarize(chirp_det.detections(), s, kRate).found == 0);
    }

    // 2. Stage 11 DoD: simulated VC routes (upward female-like +7 st & shifted
    //    formants at 48 kHz, downward baritone -5 st & shifted formants at
    //    48 kHz, and 16 kHz pipeline rate) recover known delays within +/- 2 ms.
    struct Case {
        double rate;
        double target_f0;
        double formant_scale;
        std::size_t delay_samples;
    };
    const Case cases[] = {
        {48000.0, 215.0, 1.22, 3217},  // +7.4 st, brighter formants, 67.02 ms
        {48000.0,  98.0, 0.84, 4583},  // -5.9 st, darker formants,   95.48 ms
        {16000.0, 195.0, 1.18, 1237},  // 16 kHz native VC rate,      77.31 ms
    };

    for (const Case& tc : cases) {
        probe::MarkerSchedule s;
        s.first    = static_cast<std::uint64_t>(tc.rate / 10);    // 100 ms
        s.interval = static_cast<std::uint64_t>(tc.rate / 2);     // 500 ms
        s.count    = 8;

        const auto speech = probe::make_speech_burst(tc.rate, 0.5);
        const std::vector<std::size_t> delays(s.count, tc.delay_samples);
        const auto vc_rec = simulate_vc_stream(
            speech, s, delays, tc.rate, tc.target_f0, tc.formant_scale, 0.35f, 0.015f);

        probe::MarkerDetector det(speech, s, 0.65, probe::MarkerKind::Speech, tc.rate);
        det.push(vc_rec.data(), vc_rec.size());
        CHECK(det.done());

        const double tol_samples = 0.002 * tc.rate;  // +/- 2.0 ms
        for (const auto& d : det.detections()) {
            CHECK(d.found);
            CHECK(d.score >= 0.75);
            CHECK(std::abs(d.lag - static_cast<double>(tc.delay_samples)) <= tol_samples);
        }
        const auto sum = probe::summarize(det.detections(), s, tc.rate);
        CHECK(sum.found == s.count && sum.missed == 0);
        CHECK(std::abs(sum.median - static_cast<double>(tc.delay_samples)) <= tol_samples);
    }
}

void test_speech_rejects_noise_tone_and_click() {
    probe::MarkerSchedule s;
    s.first = 4800;
    s.interval = 24000;
    s.count = 4;
    const auto speech = probe::make_speech_burst(kRate, 0.5);

    // Stationary white noise must stay far below threshold.
    std::vector<float> noisy(static_cast<std::size_t>(s.total()));
    Noise rng;
    for (float& v : noisy) v = rng.next(0.35f);
    probe::MarkerDetector det_noise(speech, s, 0.65, probe::MarkerKind::Speech, kRate);
    det_noise.push(noisy.data(), noisy.size());
    for (const auto& d : det_noise.detections()) CHECK(!d.found && d.score < 0.25);

    // Steady sine tone has flat envelope -> zero modulation correlation.
    std::vector<float> tone(static_cast<std::size_t>(s.total()));
    for (std::size_t i = 0; i < tone.size(); ++i)
        tone[i] = static_cast<float>(0.4 * std::sin(2.0 * probe::kPi * 440.0 * i / kRate));
    probe::MarkerDetector det_tone(speech, s, 0.65, probe::MarkerKind::Speech, kRate);
    det_tone.push(tone.data(), tone.size());
    for (const auto& d : det_tone.detections()) CHECK(!d.found && d.score < 0.1);

    // Single 1 ms transient click lacks the two-syllable speech envelope.
    std::vector<float> clicks(static_cast<std::size_t>(s.total()), 0.0f);
    for (std::size_t k = 0; k < s.count; ++k)
        for (std::size_t j = 0; j < 48; ++j)
            clicks[static_cast<std::size_t>(s.start(k)) + 3000 + j] = 0.8f;
    probe::MarkerDetector det_click(speech, s, 0.65, probe::MarkerKind::Speech, kRate);
    det_click.push(clicks.data(), clicks.size());
    for (const auto& d : det_click.detections()) CHECK(!d.found);
}

void test_speech_piecewise_and_zero_delay() {
    probe::MarkerSchedule s;
    s.first = 4800;
    s.interval = 24000;
    s.count = 4;
    const auto speech = probe::make_speech_burst(kRate, 0.5);

    const auto rec0 = route(speech, s, std::vector<std::size_t>(s.count, 0), 0.7f, 0.0f);
    probe::MarkerDetector det0(speech, s, 0.65, probe::MarkerKind::Speech, kRate);
    det0.push(rec0.data(), rec0.size());
    for (const auto& d : det0.detections()) {
        CHECK(d.found);
        CHECK(std::abs(d.lag) < 1.0);
    }

    const auto rec = route(speech, s, std::vector<std::size_t>(s.count, 2765), 0.6f, 0.01f);
    probe::MarkerDetector whole(speech, s, 0.65, probe::MarkerKind::Speech, kRate);
    whole.push(rec.data(), rec.size());

    probe::MarkerDetector piece(speech, s, 0.65, probe::MarkerKind::Speech, kRate);
    const std::size_t sizes[] = {3, 257, 1024, 480, 19, 4096};
    std::size_t at = 0, idx = 0;
    while (at < rec.size()) {
        const std::size_t n = std::min(sizes[idx++ % 6], rec.size() - at);
        piece.push(rec.data() + at, n);
        at += n;
    }
    CHECK(piece.done());
    for (std::size_t k = 0; k < s.count; ++k) {
        CHECK(piece.detections()[k].found == whole.detections()[k].found);
        CHECK(piece.detections()[k].lag == whole.detections()[k].lag);
        CHECK(std::abs(piece.detections()[k].lag - 2765.0) < 24.0);  // < 0.5 ms
    }
}

} // namespace

int main() {
    test_marker_shape();
    test_constant_delay_with_noise();
    test_zero_and_maximum_delay();
    test_piecewise_push_matches_whole();
    test_exact_sample_count_completes();
    test_lowpass_route();
    test_drift_and_missing_markers();
    test_noise_only_is_not_a_marker();
    test_percentiles();
    test_speech_burst_shape();
    test_speech_vc_simulated_route();
    test_speech_rejects_noise_tone_and_click();
    test_speech_piecewise_and_zero_delay();
    return test::finish("latency_probe");
}
