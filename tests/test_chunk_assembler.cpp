// ChunkAssembler — frame geometry tests (Milestone 1 / Stage 03).
//
// Frames must be identical however the stream is sliced into pushes:
// frame k = [ x[k*hop - context, k*hop) | x[k*hop, (k+1)*hop) ], with samples
// before the stream start read as 0.
#include "audio/ChunkAssembler.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

using fluxvoice::PipelineTiming;

/// Reference: slice the whole signal directly.
std::vector<std::vector<float>> reference_frames(const std::vector<float>& x, size_t hop, size_t ctx) {
    std::vector<std::vector<float>> frames;
    for (size_t start = 0; start + hop <= x.size(); start += hop) {
        std::vector<float> f(ctx + hop);
        for (size_t i = 0; i < ctx + hop; ++i) {
            const long src = static_cast<long>(start + i) - static_cast<long>(ctx);
            f[i] = src < 0 ? 0.0f : x[static_cast<size_t>(src)];
        }
        frames.push_back(std::move(f));
    }
    return frames;
}

/// Feed `x` to an assembler in pushes of the given (cycled) sizes.
std::vector<std::vector<float>> assemble(const std::vector<float>& x, size_t hop, size_t ctx,
                                         const std::vector<size_t>& push_sizes) {
    ChunkAssembler a(hop, ctx);
    std::vector<std::vector<float>> frames;
    size_t pos = 0, k = 0;
    while (pos < x.size()) {
        size_t n = std::min(push_sizes[k++ % push_sizes.size()], x.size() - pos);
        const float* p = x.data() + pos;
        pos += n;
        while (n > 0) {
            const size_t used = a.push(p, n);
            p += used;
            n -= used;
            if (a.ready()) {
                frames.emplace_back(a.frame(), a.frame() + a.frame_size());
                a.advance();
            }
        }
    }
    return frames;
}

std::vector<float> ramp(size_t n) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = static_cast<float>(i + 1);
    return x;
}

void test_m2_geometry_any_push_pattern() {
    const size_t hop = PipelineTiming::kChunkSamples, ctx = PipelineTiming::kContextSamples;
    const auto x = ramp(hop * 23 + 777);  // includes a trailing partial hop
    const auto ref = reference_frames(x, hop, ctx);
    CHECK(ref.size() == 23);

    const std::vector<std::vector<size_t>> patterns = {
        {320}, {1}, {441}, {1280}, {1600}, {5000}, {7, 320, 13, 2048, 1}, {hop * 23 + 777}};
    for (const auto& pat : patterns) {
        const auto got = assemble(x, hop, ctx, pat);
        CHECK(got.size() == ref.size());
        for (size_t f = 0; f < got.size() && f < ref.size(); ++f) CHECK(got[f] == ref[f]);
    }
}

void test_first_frame_context_is_silence_and_hop_pointer() {
    ChunkAssembler a(4, 2);
    CHECK(a.frame_size() == 6);
    const float x[4] = {1, 2, 3, 4};
    CHECK(a.push(x, 4) == 4);
    CHECK(a.ready());
    CHECK(a.frame()[0] == 0.0f && a.frame()[1] == 0.0f);
    CHECK(a.hop_samples()[0] == 1.0f && a.hop_samples()[3] == 4.0f);

    a.advance();
    CHECK(!a.ready());
    CHECK(a.pending() == 0);
    CHECK(a.frame()[0] == 3.0f && a.frame()[1] == 4.0f);  // carried context
}

void test_push_stops_at_frame_boundary() {
    ChunkAssembler a(4, 1);
    const float x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK(a.push(x, 10) == 4);  // only fills the current frame
    CHECK(a.ready());
    CHECK(a.push(x + 4, 6) == 0);  // full: consumes nothing until advance()
    a.advance();
    CHECK(a.push(x + 4, 6) == 4);
    CHECK(a.frame()[0] == 4.0f && a.hop_samples()[0] == 5.0f);
}

void test_reset_drops_history_and_partial() {
    ChunkAssembler a(4, 2);
    const float x[6] = {1, 2, 3, 4, 5, 6};
    a.push(x, 4);
    a.advance();
    a.push(x + 4, 2);
    CHECK(a.pending() == 2);
    a.reset();
    CHECK(a.pending() == 0);
    a.push(x, 4);
    CHECK(a.frame()[0] == 0.0f && a.frame()[1] == 0.0f);
}

void test_zero_context() {
    const auto x = ramp(100);
    const auto got = assemble(x, 10, 0, {3, 7, 11});
    const auto ref = reference_frames(x, 10, 0);
    CHECK(got == ref);
}

} // namespace

int main() {
    test_m2_geometry_any_push_pattern();
    test_first_frame_context_is_silence_and_hop_pointer();
    test_push_stops_at_frame_boundary();
    test_reset_drops_history_and_partial();
    test_zero_context();
    return test::finish("chunk_assembler");
}
