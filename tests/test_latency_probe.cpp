// LatencyProbe — marker detector tests (Milestone 1 / Stage 06; speech marker
// Milestone 2 / Stage 11; phrase marker Milestone 3 / Stage 11).
//
// Each test builds the probe's marker stream, passes it through a simulated
// route (delay, gain, noise, low-pass filter, dropouts, a voice converter, a
// Wait-k translator) and checks that the detector recovers the route's delay.
#include "LatencyProbe.hpp"
#include "TestSupport.hpp"
#include "pipeline/WaitKPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
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

// ---------------------------------------------------------------------------
// Phrase marker (Milestone 3 / Stage 11): onset delay + flush tail lag.
// ---------------------------------------------------------------------------

probe::MarkerSchedule phrase_schedule(double rate, std::size_t count, double interval_ms = 3500.0,
                                      double first_ms = 240.0) {
    probe::MarkerSchedule s;
    s.first    = static_cast<std::uint64_t>(std::llround(rate * first_ms / 1000.0));
    s.interval = static_cast<std::uint64_t>(std::llround(rate * interval_ms / 1000.0));
    s.count    = count;
    return s;
}

double rms(const std::vector<float>& x, std::size_t a, std::size_t b) {
    double acc = 0.0;
    for (std::size_t i = a; i < b; ++i) acc += static_cast<double>(x[i]) * x[i];
    return std::sqrt(acc / static_cast<double>(b - a));
}

void test_phrase_shape() {
    for (double rate : {48000.0, 16000.0}) {
        const auto m = probe::make_phrase(rate);
        CHECK(m.size() == static_cast<std::size_t>(std::llround(rate * probe::kPhraseSeconds)));
        CHECK(m == probe::make_phrase(rate));  // deterministic
        CHECK(probe::make_phrase(rate, 0.6).size() == static_cast<std::size_t>(std::llround(rate * 0.6)));
        CHECK(m.front() == 0.0f);
        float peak = 0.0f;
        for (float v : m) {
            CHECK(std::isfinite(v));
            peak = std::max(peak, std::abs(v));
        }
        CHECK(peak > 0.4999f && peak <= 0.5001f);

        // Sharp edges: the first and last 5 ms already carry speech-level energy.
        const double body = rms(m, 0, m.size());
        const auto edge = static_cast<std::size_t>(rate * 0.005);
        CHECK(rms(m, 0, edge) > 0.2 * body);
        CHECK(rms(m, m.size() - edge, m.size()) > 0.2 * body);

        // No gap: every 20 ms frame stays far above the pipeline's -55 dBFS
        // VAD, so a Wait-k policy sees one utterance of 15 chunks.
        const auto frame = static_cast<std::size_t>(rate * 0.020);
        for (std::size_t a = 0; a + frame <= m.size(); a += frame) {
            const double r = rms(m, a, a + frame);
            CHECK(r > 0.2 * body);
            CHECK(20.0 * std::log10(r) > -35.0);
        }
    }
}

probe::PhraseDetector phrase_detect(const std::vector<float>& phrase, const probe::MarkerSchedule& s,
                                    double rate, const std::vector<float>& rec) {
    probe::PhraseDetector det(phrase, s, rate);
    det.push(rec.data(), rec.size());
    return det;
}

void test_phrase_pure_delay() {
    // A route that only delays reads the same delay for the onset and for the
    // tail, and the output is as long as the phrase (less the few ms of the
    // 2 ms edge ramps that lie under the -30 dB threshold).
    const auto phrase = probe::make_phrase(kRate);
    const auto s = phrase_schedule(kRate, 3);
    const double len = static_cast<double>(phrase.size());

    // Whole steps of the 1 ms search grid: exact, including a zero delay.
    for (std::size_t delay : {std::size_t{0}, std::size_t{4800}, std::size_t{96000}}) {
        const auto det = phrase_detect(phrase, s, kRate,
                                       route(phrase, s, std::vector<std::size_t>(s.count, delay), 1.0f, 0.0f));
        CHECK(det.done());
        for (const auto& d : det.detections()) {
            CHECK(d.found && d.miss == probe::PhraseMiss::None);
            CHECK(d.onset_delay == static_cast<double>(delay));
            CHECK(d.flush_tail == static_cast<double>(delay));
            CHECK(d.duration <= len && d.duration > len - 240.0);
            CHECK(d.duration == det.detections()[0].duration);
        }
    }

    // Arbitrary delays, attenuated and noisy (46 dB SNR): within 0.1 ms.
    for (std::size_t delay : {std::size_t{1}, std::size_t{4583}, std::size_t{37001}}) {
        const auto det = phrase_detect(phrase, s, kRate,
                                       route(phrase, s, std::vector<std::size_t>(s.count, delay), 0.2f, 0.0005f));
        for (const auto& d : det.detections()) {
            CHECK(d.found);
            CHECK(std::abs(d.onset_delay - static_cast<double>(delay)) < 4.8);
            CHECK(std::abs(d.flush_tail - static_cast<double>(delay)) < 4.8);
            CHECK(d.duration <= len + 9.6 && d.duration > len - 240.0);
            CHECK(d.peak_dbfs > -30.0 && d.peak_dbfs < -10.0);
            CHECK(d.floor_dbfs < -65.0);
        }
    }
}

/// What a simulated translator did with one phrase, in stream samples.
struct WaitKTruth {
    bool emitted = false;
    std::size_t onset = 0;   // first sample of the first emitted hop
    std::size_t offset = 0;  // one past the last sample of the last emitted hop
};

/// Simulated Mode B route, driven by the real WaitKPolicy the way
/// InferencePipeline drives it: 80 ms chunks on a grid `grid_offset` samples
/// into the stream, a -55 dBFS frame VAD, one translated chunk per kWrite /
/// kFlush step, and a decoder that predicts <EOS> once it has caught up with
/// the source. Each translated chunk leaves when its chunk step runs, `io_delay`
/// samples later, as different speech (own pitch, formants and rhythm), with
/// the pipeline's 2 ms ramps at the silence edges, plus channel noise.
std::vector<float> simulate_waitk_route(const std::vector<float>& in, const probe::MarkerSchedule& s,
                                        double rate, int k, std::size_t grid_offset,
                                        std::size_t io_delay, float gain, float noise_amp,
                                        std::vector<WaitKTruth>& truth) {
    const auto chunk = static_cast<std::size_t>(rate * 0.080);
    const auto frame = chunk / 4;
    const auto ramp = static_cast<std::size_t>(rate * 0.002);
    std::vector<float> out(in.size(), 0.0f);
    truth.assign(s.count, WaitKTruth{});

    WaitKPolicy policy(k);
    auto f1 = probe::detail::Resonator::make(rate, 480.0, 110.0);
    auto f2 = probe::detail::Resonator::make(rate, 1750.0, 150.0);
    double phase = 0.7;
    std::size_t voice_n = 0;  // the translated voice's own clock
    bool was_emitting = false;
    float last = 0.0f;

    for (std::size_t c = 0; grid_offset + (c + 1) * chunk <= in.size(); ++c) {
        const std::size_t a = grid_offset + c * chunk;
        bool speech = false;
        for (std::size_t f = 0; f < 4; ++f) {
            const std::vector<float> fr(in.begin() + static_cast<std::ptrdiff_t>(a + f * frame),
                                        in.begin() + static_cast<std::ptrdiff_t>(a + (f + 1) * frame));
            if (20.0 * std::log10(rms(fr, 0, fr.size()) + 1e-8) > -55.0) speech = true;
        }
        policy.on_chunk_energy(speech, policy.chunk_ms());
        if (policy.source_wanted()) policy.notify_source_unit();
        const int writes = policy.max_writes_this_step();
        const bool emit = writes > 0;
        if (emit) policy.notify_target_unit(policy.flushing() && policy.t_tgt() + 1 >= policy.t_src());

        // The hop computed in this chunk step plays right after the chunk.
        const std::size_t o = a + chunk + io_delay;
        if (o + chunk > out.size()) break;
        if (emit) {
            for (std::size_t i = 0; i < chunk; ++i, ++voice_n) {
                const double t = static_cast<double>(voice_n) / rate;
                phase = std::fmod(phase + 2.0 * probe::kPi * 190.0 / rate, 2.0 * probe::kPi);
                double glottal = 0.0;
                for (int h = 1; h <= 12; ++h)
                    glottal += std::sin(static_cast<double>(h) * phase) / (1.0 + 0.12 * h * h);
                const double env = 0.55 + 0.45 * std::pow(std::sin(probe::kPi * 5.5 * t), 2.0);
                out[o + i] = gain * static_cast<float>(std::tanh(env * (f1.step(glottal) + 0.6 * f2.step(glottal))));
            }
            if (!was_emitting)
                for (std::size_t i = 0; i < ramp; ++i)
                    out[o + i] *= static_cast<float>(i + 1) / static_cast<float>(ramp);
            last = out[o + chunk - 1];
            for (std::size_t t = 0; t < s.count; ++t) {
                if (o < s.start(t) || o >= s.start(t) + s.interval) continue;
                if (!truth[t].emitted) truth[t].onset = o;
                truth[t].emitted = true;
                truth[t].offset = o + chunk;
            }
        } else if (was_emitting) {
            for (std::size_t i = 0; i < ramp; ++i)
                out[o + i] = last * static_cast<float>(ramp - i - 1) / static_cast<float>(ramp);
        }
        was_emitting = emit;
    }

    Noise rng{97531u};
    for (float& v : out) v += rng.next(noise_amp);
    return out;
}

void test_phrase_waitk_route() {
    // Stage 11 DoD: a simulated Wait-k route (k chunks of 80 ms buffered
    // before the first translated chunk, the tail drained after the 400 ms
    // pause rule) gives back its onset delay and its flush tail within 5 ms.
    struct Case { double rate; int k; double grid_ms; double io_ms; };
    const Case cases[] = {
        {48000.0, 1, 0.0, 27.5},  {48000.0, 2, 0.0, 27.5}, {48000.0, 3, 0.0, 27.5},
        {48000.0, 4, 0.0, 41.0},  {48000.0, 5, 0.0, 27.5}, {48000.0, 8, 0.0, 27.5},
        {48000.0, 3, 33.0, 12.0}, {48000.0, 5, 71.0, 12.0},  // chunk grid out of phase
        {16000.0, 3, 0.0, 27.5},  {16000.0, 4, 50.0, 20.0},  // the pipeline's own rate
    };
    for (const Case& tc : cases) {
        const auto ms = [&](double v) { return v * tc.rate / 1000.0; };
        const auto phrase = probe::make_phrase(tc.rate);
        // 3500 ms is not a multiple of the 80 ms chunk, so the four trials
        // meet the chunk grid at four phases (0 / 60 / 40 / 20 ms later).
        const auto s = phrase_schedule(tc.rate, 4);
        const auto in = route(phrase, s, std::vector<std::size_t>(s.count, 0), 1.0f, 0.0f);
        const auto grid = static_cast<std::size_t>(ms(tc.grid_ms));
        const auto io = static_cast<std::size_t>(ms(tc.io_ms));
        std::vector<WaitKTruth> truth;
        const auto rec = simulate_waitk_route(in, s, tc.rate, tc.k, grid, io, 0.35f, 0.0004f, truth);

        const auto det = phrase_detect(phrase, s, tc.rate, rec);
        CHECK(det.done());
        const double tol = ms(5.0);
        const double chunk = ms(80.0);
        const int flush_steps = std::max(tc.k - 1, 1);
        for (std::size_t t = 0; t < s.count; ++t) {
            const auto& d = det.detections()[t];
            CHECK(truth[t].emitted);
            CHECK(d.found && d.miss == probe::PhraseMiss::None);
            const double true_onset = static_cast<double>(truth[t].onset) - static_cast<double>(s.start(t));
            const double true_tail  = static_cast<double>(truth[t].offset) -
                                      static_cast<double>(s.start(t) + phrase.size());
            CHECK(std::abs(d.onset_delay - true_onset) <= tol);
            CHECK(std::abs(d.flush_tail - true_tail) <= tol);
            CHECK(std::abs(d.duration - static_cast<double>(truth[t].offset - truth[t].onset)) <= 2.0 * tol);

            // The route's own law, so the simulation cannot drift from the
            // policy: the first chunk leaves when the k-th speech chunk closes
            // (k x 80 ms after an onset on the chunk grid, up to one chunk
            // sooner off the grid), and the tail outlasts the input by the
            // five written silent chunks, the flush trigger and the drain.
            const double to_grid = std::fmod(chunk - std::fmod(static_cast<double>(s.start(t)) - static_cast<double>(grid) + 1000.0 * chunk, chunk), chunk);
            const double wait = to_grid == 0.0 ? chunk : to_grid;  // rest of the chunk the onset falls in
            CHECK(true_onset == wait + (tc.k - 1) * chunk + static_cast<double>(io));
            CHECK(true_onset > (tc.k - 1) * chunk + static_cast<double>(io) &&
                  true_onset <= tc.k * chunk + static_cast<double>(io));
            CHECK(true_tail >= (6 + flush_steps) * chunk + static_cast<double>(io) &&
                  true_tail < (7 + flush_steps) * chunk + static_cast<double>(io));
            CHECK(true_tail > true_onset + ms(300.0));  // the two figures cannot be swapped
        }

        const auto sum = probe::summarize_phrase(det.detections(), s, tc.rate);
        CHECK(sum.found == s.count && sum.missed == 0);
        CHECK(sum.onset.min > (tc.k - 1) * chunk + static_cast<double>(io) - tol);
        CHECK(sum.onset.max <= tc.k * chunk + static_cast<double>(io) + tol);
        if (tc.grid_ms == 0.0) {
            // Trial 0 starts on the grid (240 ms = 3 chunks): exactly k x 80 ms
            // of Wait-k buffering and (6 + flush) x 80 ms of tail on top of I/O.
            const auto& d0 = det.detections()[0];
            CHECK(std::abs(d0.onset_delay - (tc.k * chunk + static_cast<double>(io))) <= tol);
            CHECK(std::abs(d0.flush_tail - ((6 + flush_steps) * chunk + static_cast<double>(io))) <= tol);
        }
    }
}

void test_phrase_ignores_clicks_and_pauses() {
    const auto phrase = probe::make_phrase(kRate);
    const auto s = phrase_schedule(kRate, 3);
    const std::size_t delay = 14400;  // 300 ms
    const auto clean = route(phrase, s, std::vector<std::size_t>(s.count, delay), 0.3f, 0.0f);
    const auto base = phrase_detect(phrase, s, kRate, clean);

    // 1 ms clicks 150 ms before the output and 300 ms after it move neither edge.
    auto clicked = clean;
    for (std::size_t k = 0; k < s.count; ++k) {
        const auto at = static_cast<std::size_t>(s.start(k)) + delay;
        for (std::size_t j = 0; j < 48; ++j) {
            clicked[at - 7200 + j] = 0.8f;
            clicked[at + phrase.size() + 14400 + j] = -0.8f;
        }
    }
    // A 150 ms pause inside the output is still one answer.
    auto paused = clean;
    for (std::size_t k = 0; k < s.count; ++k) {
        const auto at = static_cast<std::size_t>(s.start(k)) + delay + 24000;
        std::fill(paused.begin() + static_cast<std::ptrdiff_t>(at),
                  paused.begin() + static_cast<std::ptrdiff_t>(at + 7200), 0.0f);
    }
    // A 7 ms dropout 10 ms into the output (a glitch, or a stop consonant)
    // does not turn the first 10 ms into a click and move the onset.
    auto dipped = clean;
    for (std::size_t k = 0; k < s.count; ++k) {
        const auto at = static_cast<std::size_t>(s.start(k)) + delay + 480;
        std::fill(dipped.begin() + static_cast<std::ptrdiff_t>(at),
                  dipped.begin() + static_cast<std::ptrdiff_t>(at + 336), 0.0f);
    }
    const auto det_click = phrase_detect(phrase, s, kRate, clicked);
    const auto det_pause = phrase_detect(phrase, s, kRate, paused);
    const auto det_dip = phrase_detect(phrase, s, kRate, dipped);
    for (std::size_t k = 0; k < s.count; ++k) {
        CHECK(base.detections()[k].found && base.detections()[k].onset_delay == static_cast<double>(delay));
        // The clicks are 18 dB louder than the answer, yet the threshold
        // follows the answer's own peak: both edges stay put.
        CHECK(det_click.detections()[k].found);
        CHECK(det_click.detections()[k].peak_dbfs > base.detections()[k].peak_dbfs + 6.0);
        CHECK(det_click.detections()[k].onset_delay == base.detections()[k].onset_delay);
        CHECK(det_click.detections()[k].flush_tail == base.detections()[k].flush_tail);
        CHECK(det_pause.detections()[k].found);
        CHECK(det_pause.detections()[k].onset_delay == base.detections()[k].onset_delay);
        CHECK(det_pause.detections()[k].flush_tail == base.detections()[k].flush_tail);
        CHECK(det_dip.detections()[k].found);
        CHECK(det_dip.detections()[k].onset_delay == base.detections()[k].onset_delay);
    }
}

void test_phrase_misses() {
    const auto phrase = probe::make_phrase(kRate);
    const auto s = phrase_schedule(kRate, 3);
    const auto n = static_cast<std::size_t>(s.total());
    using probe::PhraseMiss;

    auto all = [&](const std::vector<float>& rec, PhraseMiss expected) {
        const auto det = phrase_detect(phrase, s, kRate, rec);
        for (const auto& d : det.detections()) CHECK(!d.found && d.miss == expected);
        const auto sum = probe::summarize_phrase(det.detections(), s, kRate);
        CHECK(sum.found == 0 && sum.missed == s.count && sum.onset.missed == s.count);
    };

    all(std::vector<float>(n, 0.0f), PhraseMiss::Silent);

    std::vector<float> noise(n);
    Noise rng;
    for (float& v : noise) v = rng.next(0.3f);
    all(noise, PhraseMiss::LowSnr);

    std::vector<float> tone(n);
    for (std::size_t i = 0; i < n; ++i)
        tone[i] = static_cast<float>(0.4 * std::sin(2.0 * probe::kPi * 440.0 * static_cast<double>(i) / kRate));
    all(tone, PhraseMiss::LowSnr);

    std::vector<float> clicks(n, 0.0f);
    for (std::size_t k = 0; k < s.count; ++k)
        for (std::size_t j = 0; j < 48; ++j) clicks[static_cast<std::size_t>(s.start(k)) + 30000 + j] = 0.8f;
    all(clicks, PhraseMiss::NoSpeech);

    // 60 ms of the phrase is long enough to be a run but not an answer.
    std::vector<float> blip(n, 0.0f);
    for (std::size_t k = 0; k < s.count; ++k)
        std::copy(phrase.begin(), phrase.begin() + 2880,
                  blip.begin() + static_cast<std::ptrdiff_t>(s.start(k) + 30000));
    all(blip, PhraseMiss::NoSpeech);

    // An answer that runs past its window is a miss, never a short tail; its
    // remainder then spoils the start of the next window.
    const std::size_t late = static_cast<std::size_t>(s.interval) - phrase.size();  // ends at the next start
    const auto det = phrase_detect(phrase, s, kRate,
                                   route(phrase, s, std::vector<std::size_t>(s.count, late), 0.5f, 0.0f));
    CHECK(det.detections()[0].miss == PhraseMiss::Truncated && !det.detections()[0].found);
    CHECK(det.detections()[1].miss == PhraseMiss::Overlap && !det.detections()[1].found);
    CHECK(det.detections()[2].miss == PhraseMiss::Overlap);

    // The longest tail that still fits is measured.
    const auto fits = static_cast<std::size_t>(det.max_flush_tail());
    CHECK(fits < late && fits + 9600 > late);
    const auto det_fit = phrase_detect(phrase, s, kRate,
                                       route(phrase, s, std::vector<std::size_t>(s.count, fits), 0.5f, 0.0f));
    for (const auto& d : det_fit.detections())
        CHECK(d.found && std::abs(d.flush_tail - static_cast<double>(fits)) < 4.8);

    CHECK(std::string_view(probe::to_string(PhraseMiss::Truncated)) == "truncated");
    CHECK(std::string_view(probe::to_string(PhraseMiss::None)) == "none");
}

void test_phrase_piecewise_and_construction() {
    const auto phrase = probe::make_phrase(kRate);
    const auto s = phrase_schedule(kRate, 3);
    const auto rec = route(phrase, s, std::vector<std::size_t>(s.count, 12345), 0.4f, 0.0005f);
    const auto whole = phrase_detect(phrase, s, kRate, rec);

    probe::PhraseDetector piece(phrase, s, kRate);
    CHECK(piece.required_samples() == s.total() - 2400);  // 50 ms pre-roll
    const std::size_t sizes[] = {3, 257, 1024, 480, 19, 4096};
    std::size_t at = 0, idx = 0;
    const auto need = static_cast<std::size_t>(piece.required_samples());
    while (at + 1 < need) {
        const std::size_t n = std::min(sizes[idx++ % 6], need - 1 - at);
        piece.push(rec.data() + at, n);
        at += n;
    }
    CHECK(!piece.done() && piece.analysed() == s.count - 1);
    piece.push(rec.data() + at, 1);
    CHECK(piece.done());
    for (std::size_t k = 0; k < s.count; ++k) {
        CHECK(piece.detections()[k].found == whole.detections()[k].found);
        CHECK(piece.detections()[k].onset_delay == whole.detections()[k].onset_delay);
        CHECK(piece.detections()[k].flush_tail == whole.detections()[k].flush_tail);
    }

    auto throws = [](auto&& make) {
        try { make(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    // No room for the phrase plus its margins; a first phrase inside the
    // pre-roll; a silent "phrase"; and the wrong detector for the marker.
    CHECK(throws([&] { probe::PhraseDetector d(phrase, phrase_schedule(kRate, 2, 1250.0), kRate); }));
    CHECK(throws([&] { probe::PhraseDetector d(phrase, phrase_schedule(kRate, 2, 3500.0, 20.0), kRate); }));
    CHECK(throws([&] { probe::PhraseDetector d(std::vector<float>(57600, 0.0f), s, kRate); }));
    CHECK(throws([&] { probe::MarkerDetector d(phrase, s, 0.5, probe::MarkerKind::Phrase, kRate); }));
    CHECK(!throws([&] { probe::PhraseDetector d(phrase, phrase_schedule(kRate, 2, 1400.0, 50.0), kRate); }));
}

void test_phrase_summary() {
    const auto s = phrase_schedule(kRate, 20);
    std::vector<probe::PhraseDetection> dets(20);
    for (std::size_t k = 0; k < 20; ++k) {
        dets[k].found = true;
        dets[k].miss = probe::PhraseMiss::None;
        dets[k].onset_delay = static_cast<double>(k + 1);
        dets[k].flush_tail  = 1000.0 + 2.0 * static_cast<double>(k);
        dets[k].duration    = 500.0;
    }
    auto sum = probe::summarize_phrase(dets, s, kRate);
    CHECK(sum.found == 20 && sum.missed == 0);
    CHECK(sum.onset.median == 10.5 && sum.onset.p95 == 19.0);
    CHECK(sum.onset.min == 1.0 && sum.onset.max == 20.0);
    CHECK(sum.tail.median == 1019.0 && sum.tail.p95 == 1036.0);
    CHECK(sum.tail.min == 1000.0 && sum.tail.max == 1038.0);
    CHECK(sum.duration.median == 500.0 && sum.duration.found == 20);

    dets[19].found = false;  // the largest values drop out of every statistic
    dets[19].miss = probe::PhraseMiss::Truncated;
    sum = probe::summarize_phrase(dets, s, kRate);
    CHECK(sum.found == 19 && sum.missed == 1 && sum.tail.missed == 1);
    CHECK(sum.onset.max == 19.0 && sum.tail.max == 1036.0 && sum.onset.median == 10.0);
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
    test_phrase_shape();
    test_phrase_pure_delay();
    test_phrase_waitk_route();
    test_phrase_ignores_clicks_and_pauses();
    test_phrase_misses();
    test_phrase_piecewise_and_construction();
    test_phrase_summary();
    return test::finish("latency_probe");
}
