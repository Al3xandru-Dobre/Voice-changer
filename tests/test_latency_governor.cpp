#include "pipeline/LatencyGovernor.hpp"
#include "audio/DuplexRouter.hpp"
#include "TestSupport.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace {
using namespace fluxvoice;

void test_policy() {
    LatencyGovernor g({16000, 320, 560, 3, 0.001f});
    std::array<float, 320> quiet{}, loud{};
    loud.fill(0.1f);
    CHECK(g.hard_cap_samples() == 2960);
    CHECK(g.skip_output(880, quiet) == 0); // threshold is strictly above target + hop
    CHECK(g.skip_output(881, quiet) == 0);
    CHECK(g.skip_output(881, quiet) == 0);
    CHECK(g.skip_output(881, quiet) == 320);
    g.reset();
    CHECK(g.skip_output(881, quiet) == 0); // reset clears sustained-high history
    CHECK(g.skip_output(560, quiet) == 0);
    CHECK(g.skip_output(881, quiet) == 0); // low occupancy clears it, too
    for (int i = 0; i < 10; ++i) CHECK(g.skip_output(1200, loud) == 0);
    CHECK(g.skip_output(1200, quiet) == 320);
    CHECK(g.skip_output(1200, loud, true) == 320); // closed output gate is quiet
    quiet[1] = 0.001f;
    CHECK(g.skip_output(1200, quiet) == 320);
    quiet[1] = 0.0011f;
    CHECK(g.skip_output(1200, quiet) == 0);
    quiet[1] = std::numeric_limits<float>::quiet_NaN();
    CHECK(g.skip_output(1200, quiet) == 0);
    CHECK(g.skip_output(2960, loud) == 0); // hard cap is strict, not >=
    g.reset();
    CHECK(g.skip_output(2961, loud) == 320); // no quiet/sustained requirement
    CHECK(g.skip_output(5000, std::span(loud).first(17)) == 17);
    CHECK(g.skip_output(-100, loud) == 0);

    // For long hops the hard cap can be below the ordinary one-hop threshold.
    LatencyGovernor long_hop({16000, 3200, 560, 3, 0.001f});
    CHECK(long_hop.skip_output(2961, loud) == 320);
    CHECK(LatencyGovernor::default_target(320, 320, 16000) == 560);
    CHECK(LatencyGovernor::default_target(640, 320, 16000) == 880);
    CHECK(LatencyGovernor::default_target(640, 640, 16000) == 1040);
    CHECK(LatencyGovernor::default_target(1280, 1280, 16000) == 2080);
    bool rejected = false;
    try { LatencyGovernor invalid({16000, 0, 560, 3, 0.001f}); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

void test_conservation_with_prefill_and_recovery() {
    DuplexRouter router(4096);
    LatencyGovernor g({16000, 320, 320, 2, 0.001f});
    std::array<float, 320> input{}, output{}, work{};
    std::uint64_t prefilled = router.write_output(input.data(), g.target_samples());
    std::uint64_t drops = 0, skipped = 0;
    CHECK(prefilled == 320);
    // Simulate a 3-hop inference stall, then process every captured hop. No
    // model input is skipped, even as quiet output is discarded for recovery.
    for (int i = 0; i < 3; ++i) router.process(output.data(), input.data(), input.size());
    std::size_t consumed = 0;
    while (router.read_input(work.data(), work.size()) == work.size()) {
        consumed += work.size();
        const auto queued = queued_samples(router.output_underruns(), router.input_overruns(),
                                            drops, skipped, prefilled);
        const auto skip = g.skip_output(queued, work);
        skipped += skip;
        const auto emit = std::span(work).subspan(skip);
        drops += emit.size() - router.write_output(emit.data(), emit.size());
    }
    CHECK(consumed == 960 && skipped == 320);
    CHECK(queued_samples(router.output_underruns(), router.input_overruns(), drops,
                         skipped, prefilled) == static_cast<std::int64_t>(
                             router.input_queued_approx() + router.output_queued_approx()));
    // Round-trip more callbacks with irregular sizes, including a short write.
    for (std::size_t n : {std::size_t{17}, std::size_t{320}, std::size_t{91}}) {
        router.process(output.data(), input.data(), n);
        const auto read = router.read_input(work.data(), n);
        CHECK(read == n);
        const auto written = router.write_output_bounded(work.data(), read, 500);
        drops += read - written;
        CHECK(queued_samples(router.output_underruns(), router.input_overruns(), drops,
                             skipped, prefilled) == static_cast<std::int64_t>(
                                 router.input_queued_approx() + router.output_queued_approx()));
    }
}
} // namespace

int main() {
    test_policy();
    test_conservation_with_prefill_and_recovery();
    return test::finish("latency_governor");
}
