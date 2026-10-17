// ola_sine_test — WAVs for hearing what OverlapAddProcessor does
// (Milestone 1 / Stage 04).
//
// A 440 Hz tone is cut into vocoder-style frames (80 ms chunk + 20 ms overlap)
// whose gain and phase change from frame to frame, the way independently
// generated vocoder chunks disagree. Writes three 16 kHz mono 16-bit WAVs:
//
//   ola_reference.wav   the continuous tone, no chunking
//   ola_naive.wav       the frames butt-joined: a click every 80 ms
//   ola_crossfaded.wav  the same frames through OverlapAddProcessor
//
// and prints the largest sample-to-sample step of each. Listen to them, or
// run tools/analyze_loopback.py on them: the crossfaded file should come out
// clean and the naive one should not. (The crossfaded tone still wobbles in
// level and pitch where frames disagree; OLA removes clicks, not mismatch.)
//
// Usage: ola_sine_test [output_dir]
#include "audio/OverlapAdd.hpp"
#include "pipeline/PipelineTiming.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

using fluxvoice::PipelineTiming;

constexpr double   kTwoPi   = 6.283185307179586;
constexpr uint32_t kRate    = PipelineTiming::kSampleRate;
constexpr size_t   kChunk   = PipelineTiming::kChunkSamples;
constexpr size_t   kOverlap = PipelineTiming::kOverlapSamples;
constexpr size_t   kFrames  = 5 * kRate / kChunk;  // 5 s
constexpr double   kFreq    = 440.0;
constexpr double   kAmp     = 0.25;

/// Write mono float samples as 16-bit PCM.
bool write_wav(const std::filesystem::path& path, const std::vector<float>& x) {
    std::ofstream f(path, std::ios::binary);
    auto u16 = [&](uint32_t v) { f.put(static_cast<char>(v)); f.put(static_cast<char>(v >> 8)); };
    auto u32 = [&](uint32_t v) { u16(v & 0xFFFF); u16(v >> 16); };
    const uint32_t data_bytes = static_cast<uint32_t>(x.size() * 2);

    f.write("RIFF", 4); u32(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(1 /* PCM */); u16(1 /* mono */);
    u32(kRate); u32(kRate * 2); u16(2 /* block align */); u16(16 /* bits */);
    f.write("data", 4); u32(data_bytes);
    for (float s : x) {
        const auto q = static_cast<int16_t>(std::lround(std::clamp(s, -1.0f, 1.0f) * 32767.0f));
        u16(static_cast<uint16_t>(q));
    }
    return static_cast<bool>(f);
}

float max_step(const std::vector<float>& y) {
    float m = 0.0f;
    for (size_t i = 1; i < y.size(); ++i) m = std::max(m, std::abs(y[i] - y[i - 1]));
    return m;
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : ".";

    std::vector<float> reference, naive, crossfaded(kFrames * kChunk);
    std::vector<float> frame(kChunk + kOverlap);
    OverlapAddProcessor ola(kChunk, kOverlap);

    for (size_t k = 0; k < kFrames; ++k) {
        // Per-frame gain 0.8 … 1.0 and phase ±0.4 rad.
        const double gain  = 0.8 + 0.05 * static_cast<double>((k * 7) % 5);
        const double phase = 0.2 * static_cast<double>((k * 3) % 5) - 0.4;
        for (size_t i = 0; i < frame.size(); ++i) {
            const double t = static_cast<double>(k * kChunk + i) / kRate;
            frame[i] = static_cast<float>(gain * kAmp * std::sin(kTwoPi * kFreq * t + phase));
            if (i < kChunk) reference.push_back(static_cast<float>(kAmp * std::sin(kTwoPi * kFreq * t)));
        }
        naive.insert(naive.end(), frame.begin(), frame.begin() + kChunk);
        ola.process(frame.data(), crossfaded.data() + k * kChunk);
    }

    struct Output { const char* name; const std::vector<float>& x; };
    const Output outputs[] = {{"ola_reference.wav", reference},
                              {"ola_naive.wav", naive},
                              {"ola_crossfaded.wav", crossfaded}};
    for (const Output& o : outputs) {
        const auto path = dir / o.name;
        if (!write_wav(path, o.x)) {
            std::fprintf(stderr, "Failed to write %s\n", path.string().c_str());
            return 1;
        }
        std::printf("%-20s max step %.4f\n", o.name, max_step(o.x));
    }
    return 0;
}
