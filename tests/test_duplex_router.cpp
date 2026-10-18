// DuplexRouter — real-time callback path tests (no audio device needed).
//
// Covers Milestone 1 / Stages 02–03:
//   * passthrough copies input, zero-fills when capture is absent
//   * queue mode round-trips samples in order through worker read/write
//   * underruns zero-fill and are counted; overruns drop and are counted
//   * arbitrary callback frame counts (periods are hints)
//   * process() performs no heap allocation in either mode
#include "audio/DuplexRouter.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

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
// ---------------------------------------------------------------------------
// Global allocation counter: every operator new in this process bumps it.
// ---------------------------------------------------------------------------
static std::atomic<long> g_allocs{0};

// GCC can't see that these replacements pair malloc with free on purpose.
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

void test_passthrough() {
    DuplexRouter r(1024);
    r.set_passthrough(true);
    float in[5] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    float out[5] = {9, 9, 9, 9, 9};
    r.process(out, in, 5);
    for (int i = 0; i < 5; ++i) CHECK(out[i] == in[i]);

    r.process(out, nullptr, 5);  // capture absent → silence
    for (int i = 0; i < 5; ++i) CHECK(out[i] == 0.0f);

    CHECK(r.input_overruns() == 0);
    CHECK(r.output_underruns() == 0);
    CHECK(r.frames_processed() == 10);
}

void test_null_output_buffer_capture_only_contract() {
    // Capture-only / native loopback callbacks pass output == nullptr (M1.5
    // stages 05–06). process() must enqueue captured samples without
    // dereferencing output or counting false output underruns.
    DuplexRouter r(1024);
    float in[4] = {0.25f, 0.5f, -0.25f, -0.5f};
    r.process(nullptr, in, 4);
    CHECK(r.frames_processed() == 4);
    CHECK(r.output_underruns() == 0);
    CHECK(r.input_overruns() == 0);

    float drained[4] = {};
    CHECK(r.read_input(drained, 4) == 4);
    for (int i = 0; i < 4; ++i) CHECK(drained[i] == in[i]);

    // Both null is a safe no-op.
    r.process(nullptr, nullptr, 16);
    CHECK(r.frames_processed() == 4);
}

void test_bounded_output_and_discard_queues() {
    DuplexRouter r(1024);
    std::vector<float> chunk(100, 1.0f);
    // Bound output queue to 150 samples: first 100 fit, second 100 writes at most 50.
    const size_t w1 = r.write_output_bounded(chunk.data(), chunk.size(), 150);
    const size_t w2 = r.write_output_bounded(chunk.data(), chunk.size(), 150);
    const size_t w3 = r.write_output_bounded(chunk.data(), chunk.size(), 150);
    CHECK(w1 == 100);
    CHECK(w2 == 50);
    CHECK(w3 == 0);
    CHECK(r.output_queued_approx() == 150);

    // Enqueue some input samples too, then discard both queues after quiesce.
    r.process(nullptr, chunk.data(), 40);
    CHECK(r.input_queued_approx() == 40);
    const size_t discarded = r.discard_queues();
    CHECK(discarded == 190);
    CHECK(r.input_queued_approx() == 0);
    CHECK(r.output_queued_approx() == 0);
}

void test_queue_roundtrip_with_odd_frame_counts() {
    DuplexRouter r(4096);
    std::vector<float> worker_buf(4096);
    std::vector<float> received;
    float next = 1.0f;

    // Period sizes are hints: feed deliberately irregular callback sizes.
    const size_t sizes[] = {320, 1, 7, 441, 320, 2048, 13, 320};
    for (size_t n : sizes) {
        std::vector<float> in(n), out(n);
        for (auto& s : in) s = next++;
        r.process(out.data(), in.data(), n);
        for (float s : out) if (s != 0.0f) received.push_back(s);

        // Worker: drain input, echo to output.
        size_t got = r.read_input(worker_buf.data(), worker_buf.size());
        CHECK(got == n);
        CHECK(r.write_output(worker_buf.data(), got) == got);
    }
    // One more callback with no capture to flush the last echoed block.
    std::vector<float> out(4096);
    r.process(out.data(), nullptr, out.size());
    for (float s : out) if (s != 0.0f) received.push_back(s);

    // Everything we fed came back, in order, nothing duplicated.
    CHECK(received.size() == static_cast<size_t>(next - 1.0f));
    for (size_t i = 0; i < received.size(); ++i) CHECK(received[i] == static_cast<float>(i + 1));
    CHECK(r.input_overruns() == 0);
}

void test_underrun_zero_fills_and_counts() {
    DuplexRouter r(1024);
    float primed[3] = {0.5f, 0.6f, 0.7f};
    CHECK(r.write_output(primed, 3) == 3);

    float out[8];
    for (auto& s : out) s = 9.0f;
    r.process(out, nullptr, 8);
    CHECK(out[0] == 0.5f && out[1] == 0.6f && out[2] == 0.7f);
    for (int i = 3; i < 8; ++i) CHECK(out[i] == 0.0f);
    CHECK(r.output_underruns() == 5);
}

void test_overrun_drops_and_counts() {
    // ReaderWriterQueue rounds capacity up internally, so measure the real
    // capacity by filling it, then check the overflow is counted exactly.
    DuplexRouter r(64);
    std::vector<float> in(10000, 0.25f), out(10000);
    r.process(out.data(), in.data(), in.size());

    std::vector<float> drained(10000);
    const size_t stored = r.read_input(drained.data(), drained.size());
    CHECK(stored > 0 && stored < in.size());
    CHECK(r.input_overruns() == in.size() - stored);
}

#ifndef FLUXVOICE_TSAN
void test_process_never_allocates() {
    DuplexRouter r(1 << 16);
    std::vector<float> in(4096, 0.1f), out(4096), worker(4096);

    for (bool passthrough : {true, false}) {
        r.set_passthrough(passthrough);
        const long before = g_allocs.load();
        for (int k = 0; k < 500; ++k) {
            const size_t n = 1 + static_cast<size_t>(k * 37) % 1500;  // irregular sizes
            r.process(out.data(), in.data(), n);
            r.process(out.data(), nullptr, n);
            if (!passthrough) {  // keep queues cycling, incl. overflow/underflow edges
                const size_t got = r.read_input(worker.data(), worker.size());
                r.write_output(worker.data(), got / 2);
            }
        }
        // Worker-side calls are included: try_enqueue/try_dequeue never allocate.
        CHECK(g_allocs.load() == before);
    }
}
#endif // FLUXVOICE_TSAN

void test_concurrent_callback_and_worker() {
    // A "callback" thread and a worker thread run concurrently, as in the real
    // engine. Samples carry increasing values; whatever reaches the output must
    // still be strictly increasing (no reordering, duplication or tearing), and
    // every sample is accounted for as delivered, dropped or still in flight.
    // Run under -fsanitize=thread to check the SPSC hand-off for data races.
    DuplexRouter r(4096);
    constexpr size_t kTotal = 400000;
    std::atomic<bool> done{false};
    std::vector<float> delivered;
    delivered.reserve(kTotal);

    std::thread callback([&] {
        std::vector<float> in(1024), out(1024);
        size_t fed = 0, k = 0;
        while (fed < kTotal) {
            const size_t n = std::min<size_t>(1 + (k++ * 131) % 1000, kTotal - fed);
            for (size_t i = 0; i < n; ++i) in[i] = static_cast<float>(fed + i + 1);
            fed += n;
            r.process(out.data(), in.data(), n);
            for (size_t i = 0; i < n; ++i) if (out[i] != 0.0f) delivered.push_back(out[i]);
            if (k % 16 == 0) std::this_thread::yield();
        }
        done.store(true);
    });
    std::thread worker([&] {
        std::vector<float> buf(512);
        while (!done.load()) {
            const size_t got = r.read_input(buf.data(), buf.size());
            if (got == 0) { std::this_thread::yield(); continue; }
            r.write_output(buf.data(), got);  // excess is dropped, like main.cpp
        }
    });
    callback.join();
    worker.join();

    bool increasing = true;
    for (size_t i = 1; i < delivered.size(); ++i) increasing &= delivered[i] > delivered[i - 1];
    CHECK(increasing);
    CHECK(!delivered.empty());
    CHECK(delivered.size() + r.input_overruns() <= kTotal);
    CHECK(r.frames_processed() == kTotal);
}

} // namespace

int main() {
    test_passthrough();
    test_null_output_buffer_capture_only_contract();
    test_bounded_output_and_discard_queues();
    test_queue_roundtrip_with_odd_frame_counts();
    test_underrun_zero_fills_and_counts();
    test_overrun_drops_and_counts();
#ifndef FLUXVOICE_TSAN
    test_process_never_allocates();
#endif
    test_concurrent_callback_and_worker();
    return test::finish("duplex_router");
}
