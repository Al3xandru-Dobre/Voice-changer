#include "pipeline/TimingHistogram.hpp"
#include "TestSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// A scripted microsecond clock, the shape of PipelineOptions::clock_us: each
// read returns the current time, then advances it by the next scripted step.
struct ScriptedClock {
    static inline std::uint64_t now = 0;
    static inline std::vector<std::uint64_t> steps;
    static inline std::size_t next = 0;
    static std::uint64_t read() noexcept {
        const std::uint64_t t = now;
        now += steps[next++ % steps.size()];
        return t;
    }
};

// One worker-style bracket: t0, the work (the clock advances), t1, record.
void bracket(fluxvoice::TimingHistogram& h, std::uint64_t work_us) {
    ScriptedClock::steps = {work_us, 0};   // read #1 advances by the work, read #2 by nothing
    ScriptedClock::next = 0;
    const std::uint64_t t0 = ScriptedClock::read();
    const std::uint64_t t1 = ScriptedClock::read();
    h.record(t1 - t0);
}

// Durations come from the clock, so p50 / p95 / max / misses are known by
// construction: 90 x 120 us, 5 x exactly one hop (not a miss), 5 x one bin
// over it (a miss). Percentiles are 50 us bin upper bounds capped at the max.
void test_fake_clock_driven() {
    using namespace fluxvoice;
    TimingHistogram h(20000);
    for (int i = 0; i < 90; ++i) bracket(h, 120);
    for (int i = 0; i < 5; ++i) bracket(h, 20000);
    for (int i = 0; i < 5; ++i) bracket(h, 20050);
    TimingSnapshot s = h.snapshot();
    CHECK(s.samples == 100);
    CHECK(s.p50_us == 150);     // bin 2 (100..149 us) upper bound
    CHECK(s.p95_us == 20050);   // rank 95 is the last 20000 us sample: bin 400, capped at max
    CHECK(s.max_us == 20050);
    CHECK(s.misses == 5);       // only the 20050 us hops; == hop is on time

    // A stage that is always exactly one hop long never misses; one us more always does.
    h.reset();
    for (int i = 0; i < 8; ++i) bracket(h, 20000);
    s = h.snapshot();
    CHECK(s.misses == 0 && s.p50_us == 20000 && s.p95_us == 20000 && s.max_us == 20000);
    h.reset();
    for (int i = 0; i < 8; ++i) bracket(h, 20001);
    s = h.snapshot();
    CHECK(s.misses == 8 && s.p50_us == 20001 && s.max_us == 20001);
}

} // namespace

int main() {
    using namespace fluxvoice;
    TimingHistogram h(20000);
    CHECK(h.snapshot().samples == 0);
    for (int i = 0; i < 50; ++i) h.record(49);
    for (int i = 0; i < 45; ++i) h.record(150);
    for (int i = 0; i < 4; ++i) h.record(20000);
    h.record(20001);
    auto s = h.snapshot();
    CHECK(s.samples == 100 && s.p50_us == 50 && s.p95_us == 200);
    CHECK(s.max_us == 20001 && s.misses == 1); // equal to hop is not a miss
    h.reset();
    for (int i = 0; i < 20; ++i) h.record(70000);
    s = h.snapshot();
    CHECK(s.p50_us == 70000 && s.p95_us == 70000 && s.misses == 20);
    h.reset();
    h.record(0);
    CHECK(h.snapshot().p50_us == 0);
    h.reset();

    std::thread worker([&] {
        for (std::uint64_t i = 0; i < 10000; ++i) h.record(i);
    });
    for (int i = 0; i < 10; ++i) {
        s = h.snapshot();
        CHECK(s.p50_us <= s.max_us && s.p95_us <= s.max_us);
    }
    worker.join();
    s = h.snapshot();
    CHECK(s.samples == 10000 && s.max_us == 9999 && s.misses == 0);
    CHECK(s.p50_us == 5000 && s.p95_us == 9500);
    bool rejected = false;
    try { TimingHistogram invalid(0); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    test_fake_clock_driven();
    return test::finish("timing_histogram");
}
