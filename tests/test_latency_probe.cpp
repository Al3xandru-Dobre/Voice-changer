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
    return test::finish("latency_probe");
}
