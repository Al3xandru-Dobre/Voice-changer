// OverlapAddProcessor — crossfade tests (Milestone 1 / Stage 04).
//
// Step k's input covers stream samples [k*chunk, k*chunk + chunk + overlap);
// its tail and step k+1's head are the same instants and get crossfaded.
// Signals are generated, so every check is exact up to float rounding.
#include "audio/OverlapAdd.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using fluxvoice::PipelineTiming;
using Frame = std::vector<float>;

constexpr double kTwoPi = 6.283185307179586;
constexpr double kRate  = static_cast<double>(PipelineTiming::kSampleRate);
constexpr float  kEps   = 1e-6f;

constexpr size_t kChunk   = PipelineTiming::kChunkSamples;
constexpr size_t kOverlap = PipelineTiming::kOverlapSamples;

struct Geometry { size_t chunk, overlap; };
constexpr Geometry kGeometries[] = {{kChunk, kOverlap}, {8, 4}, {5, 1}, {16, 16}};

/// Feed frames through `ola`; returns the concatenated output chunks.
std::vector<float> run(OverlapAddProcessor& ola, const std::vector<Frame>& frames) {
    std::vector<float> out(frames.size() * ola.chunk_size());
    for (size_t k = 0; k < frames.size(); ++k)
        ola.process(frames[k].data(), out.data() + k * ola.chunk_size());
    return out;
}

/// Frames cut from one continuous signal, as an ideal vocoder would emit them.
std::vector<Frame> slice(const std::vector<float>& x, size_t chunk, size_t overlap) {
    std::vector<Frame> frames;
    for (size_t start = 0; start + chunk + overlap <= x.size(); start += chunk)
        frames.emplace_back(x.begin() + start, x.begin() + start + chunk + overlap);
    return frames;
}

/// Frames the way a chunked vocoder emits them: each one a sine with its own
/// gain (0.6 … 1.0) and phase (±0.6 rad), so neighbours disagree where they
/// overlap.
std::vector<Frame> mismatched_frames(size_t n_frames, double freq, double amp) {
    std::vector<Frame> frames;
    for (size_t k = 0; k < n_frames; ++k) {
        const double gain  = 0.6 + 0.1 * static_cast<double>((k * 7) % 5);
        const double phase = 0.3 * static_cast<double>((k * 3) % 5) - 0.6;
        Frame f(kChunk + kOverlap);
        for (size_t i = 0; i < f.size(); ++i) {
            const double t = static_cast<double>(k * kChunk + i) / kRate;
            f[i] = static_cast<float>(gain * amp * std::sin(kTwoPi * freq * t + phase));
        }
        frames.push_back(std::move(f));
    }
    return frames;
}

std::vector<float> sine(size_t n, double freq, double amp) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = static_cast<float>(amp * std::sin(kTwoPi * freq * static_cast<double>(i) / kRate));
    return x;
}

/// Largest |y[i] - y[i-1]| for i > from.
float max_step(const std::vector<float>& y, size_t from) {
    float m = 0.0f;
    for (size_t i = from + 1; i < y.size(); ++i) m = std::max(m, std::abs(y[i] - y[i - 1]));
    return m;
}

/// Largest |y[i] - ref[i]| for i >= from (ref may be longer than y).
float max_abs_diff(const std::vector<float>& y, const std::vector<float>& ref, size_t from) {
    float m = 0.0f;
    for (size_t i = from; i < y.size(); ++i) m = std::max(m, std::abs(y[i] - ref[i]));
    return m;
}

void test_fades_sum_to_one() {
    for (const Geometry g : kGeometries) {
        OverlapAddProcessor ola(g.chunk, g.overlap);
        const float* in  = ola.fade_in();
        const float* out = ola.fade_out();
        CHECK(in[0] == 0.0f);             // the new chunk starts silent ...
        CHECK(in[g.overlap - 1] < 1.0f);  // ... and the last tap stays below 1 (phase = i / overlap)
        float worst = 0.0f;
        bool rising = true;
        for (size_t i = 0; i < g.overlap; ++i) {
            worst = std::max(worst, std::abs(in[i] + out[i] - 1.0f));
            if (i > 0) rising &= in[i] > in[i - 1];
        }
        CHECK(worst <= kEps);
        CHECK(rising);
    }
}

void test_constant_signal_is_transparent() {
    for (const Geometry g : kGeometries) {
        for (float c : {0.37f, -0.8f}) {
            OverlapAddProcessor ola(g.chunk, g.overlap);
            const auto y = run(ola, std::vector<Frame>(6, Frame(g.chunk + g.overlap, c)));
            // No history yet: the first chunk fades in from silence ...
            float head = 0.0f;
            for (size_t i = 0; i < g.overlap; ++i)
                head = std::max(head, std::abs(y[i] - c * ola.fade_in()[i]));
            CHECK(head <= kEps);
            // ... after that the splices are invisible.
            CHECK(max_abs_diff(y, std::vector<float>(y.size(), c), g.overlap) <= kEps);
        }
    }
}

void test_continuous_sine_has_no_boundary_discontinuity() {
    const double amp = 0.5;
    for (double freq : {50.0, 440.0, 1234.5, 7000.0}) {
        const auto x = sine(kChunk * 40 + kOverlap, freq, amp);
        OverlapAddProcessor ola(kChunk, kOverlap);
        const auto y = run(ola, slice(x, kChunk, kOverlap));
        CHECK(y.size() == kChunk * 40);
        // Past the initial fade-in the stream comes back unchanged ...
        CHECK(max_abs_diff(y, x, kOverlap) <= kEps);
        // ... so no step, at a chunk boundary or anywhere else, exceeds the
        // sine's own maximum slope.
        const double slope = kTwoPi * freq * amp / kRate;
        CHECK(max_step(y, kOverlap) <= slope + kEps);
    }
}

void test_mismatched_frames_are_smoothed() {
    // Blending a (previous tail) into b (new head) with weight w gives
    //   Δy = w'·Δb + (1 - w')·Δa + Δw·(b - a)
    // so a step can't exceed the steeper signal's slope plus max|b - a| times
    // the largest Hann weight step, sin(π / 2N).
    const double freq = 440.0, amp = 0.5;
    const auto frames = mismatched_frames(30, freq, amp);
    OverlapAddProcessor ola(kChunk, kOverlap);
    const auto y = run(ola, frames);

    const double slope = kTwoPi * freq * amp / kRate;  // every gain is ≤ 1
    const double bound = slope + 2.0 * amp * std::sin(kTwoPi / (4.0 * static_cast<double>(kOverlap)));
    CHECK(max_step(y, kOverlap) <= bound + kEps);

    // Butt-joining the same frames (what OLA replaces) jumps well past the
    // bound at the boundaries, so the check above has teeth.
    std::vector<float> naive;
    for (const auto& f : frames) naive.insert(naive.end(), f.begin(), f.begin() + kChunk);
    CHECK(max_step(naive, 0) > 2.0 * bound);
}

void test_reset_clears_tail() {
    const auto frames = mismatched_frames(4, 440.0, 0.5);
    OverlapAddProcessor ola(kChunk, kOverlap);

    // With silent input the output head is prev_tail * fade_out, and fade_out
    // is non-zero across the overlap, so any leftover tail would show up.
    run(ola, frames);
    ola.reset();
    const std::vector<Frame> silence(1, Frame(kChunk + kOverlap, 0.0f));
    CHECK(max_abs_diff(run(ola, silence), std::vector<float>(kChunk, 0.0f), 0) == 0.0f);

    // After reset() the processor behaves exactly like a new one.
    run(ola, frames);
    ola.reset();
    OverlapAddProcessor fresh(kChunk, kOverlap);
    CHECK(run(ola, frames) == run(fresh, frames));
}

void test_zero_overlap_is_plain_copy() {
    OverlapAddProcessor ola(4, 0);
    const std::vector<Frame> frames = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    CHECK(run(ola, frames) == std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8}));
}

void test_writes_exactly_one_chunk() {
    for (const Geometry g : kGeometries) {
        OverlapAddProcessor ola(g.chunk, g.overlap);
        const Frame in(g.chunk + g.overlap, 0.25f);
        std::vector<float> out(g.chunk + 8, 99.0f);
        ola.process(in.data(), out.data());
        ola.process(in.data(), out.data());
        CHECK(std::all_of(out.begin() + static_cast<std::ptrdiff_t>(g.chunk), out.end(),
                          [](float s) { return s == 99.0f; }));
    }
}

} // namespace

int main() {
    test_fades_sum_to_one();
    test_constant_signal_is_transparent();
    test_continuous_sine_has_no_boundary_discontinuity();
    test_mismatched_frames_are_smoothed();
    test_reset_clears_tail();
    test_zero_overlap_is_plain_copy();
    test_writes_exactly_one_chunk();
    return test::finish("overlap_add");
}
