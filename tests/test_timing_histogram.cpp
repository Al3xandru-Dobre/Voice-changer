#include "pipeline/TimingHistogram.hpp"
#include "TestSupport.hpp"

#include <cstdint>
#include <stdexcept>
#include <thread>

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
    return test::finish("timing_histogram");
}
