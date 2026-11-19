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
    CHECK(LatencyGovernor::default_target(640, 640, 16000) == 1040);
    CHECK(LatencyGovernor::default_target(1280, 1280, 16000) == 2080);
    bool rejected = false;
    try { LatencyGovernor invalid({16000, 0, 560, 3, 0.001f}); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

// Mode B burst cushion (Milestone 3 / Stage 08): target {16000, 320, 560, 3,
// 0.001, cushion 1000} -> effective target 1560 and hard cap 3960 while the
// cushion is active, 560 / 2960 otherwise.
void test_burst_cushion_and_hold() {
    GovernorConfig cfg{16000, 320, 560, 3, 0.001f, 1000};
    LatencyGovernor g(cfg);
    std::array<float, 320> quiet{}, loud{};
    loud.fill(0.1f);

    // Idle: the cushion is configured but inactive.
    CHECK(g.burst_cushion_samples() == 1000);
    CHECK(!g.cushion_active() && !g.hold_active());
    CHECK(g.target_samples() == 560 && g.effective_target_samples() == 560);
    CHECK(g.hard_cap_samples() == 2960);

    g.set_translation_state(true, false);
    CHECK(g.cushion_active() && !g.hold_active());
    CHECK(g.target_samples() == 560);                    // the startup target is untouched
    CHECK(g.effective_target_samples() == 1560);
    CHECK(g.hard_cap_samples() == 3960);
    // 1560 + 320 = 1880: quiet queue at or below stays, never trimmed.
    for (int i = 0; i < 10; ++i) CHECK(g.skip_output(1880, quiet) == 0);
    // One sample above effective target + hop: trimmed after sustained_hops,
    // by min(hop, excess) with the excess measured from the raised target.
    CHECK(g.skip_output(1881, quiet) == 0);
    CHECK(g.skip_output(1881, quiet) == 0);
    CHECK(g.skip_output(1881, quiet) == 320);
    // Without the cushion the same occupancy would already have been trimmed
    // earlier (the cushion is what held it back).
    g.set_translation_state(false, false);
    CHECK(g.effective_target_samples() == 560 && g.hard_cap_samples() == 2960);
    g.reset();
    CHECK(g.skip_output(1000, quiet) == 0);
    CHECK(g.skip_output(1000, quiet) == 0);
    CHECK(g.skip_output(1000, quiet) == 320);

    // The cushion raises the hard cap: 3960 is not a hard drop, 3961 is.
    g.reset();
    g.set_translation_state(true, true);
    CHECK(g.skip_output(3960, loud) == 0);
    CHECK(g.skip_output(3961, loud) == 320);
    g.set_translation_state(false, true);
    CHECK(g.skip_output(2961, loud) == 320);             // 2961 > 2960 once the cushion is gone

    // Hold suppresses soft (quiet / sustained) drops, however long it lasts...
    g.reset();
    g.set_translation_state(false, true);
    for (int i = 0; i < 20; ++i) CHECK(g.skip_output(1200, quiet) == 0);
    CHECK(g.skip_output(1200, quiet, true) == 0);        // gate-closed is also a soft drop
    // ...but never a hard-cap drop, and it does not alter the thresholds.
    CHECK(g.hard_cap_samples() == 2960 && g.effective_target_samples() == 560);
    CHECK(g.skip_output(2961, loud) == 320);
    CHECK(g.skip_output(2961, std::span(loud).first(10)) == 10);
    // The sustained counter kept running while held: releasing the hold trims
    // at once for quiet output.
    g.set_translation_state(false, false);
    CHECK(g.skip_output(1200, quiet) == 320);

    // reset() clears both flags and the sustained history.
    g.set_translation_state(true, true);
    g.reset();
    CHECK(!g.cushion_active() && !g.hold_active());
    CHECK(g.effective_target_samples() == 560 && g.hard_cap_samples() == 2960);
}

void test_cushion_validation_and_zero_cushion() {
    GovernorConfig bad{16000, 320, 560, 3, 0.001f, 160001};
    bool rejected = false;
    try { LatencyGovernor g(bad); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    GovernorConfig edge = bad;
    edge.burst_cushion_samples = 160000;                 // exactly 10 * sample_rate is allowed
    LatencyGovernor ok(edge);
    CHECK(ok.burst_cushion_samples() == 160000);

    // Cushion 0 (Mode A): the hint changes nothing about the targets.
    LatencyGovernor a({16000, 320, 560, 3, 0.001f});
    CHECK(a.burst_cushion_samples() == 0);
    a.set_translation_state(true, false);
    CHECK(a.cushion_active());
    CHECK(a.effective_target_samples() == 560 && a.hard_cap_samples() == 2960);
    std::array<float, 320> quiet{};
    CHECK(a.skip_output(880, quiet) == 0);
    CHECK(a.skip_output(881, quiet) == 0);
    CHECK(a.skip_output(881, quiet) == 0);
    CHECK(a.skip_output(881, quiet) == 320);             // identical to the Mode A policy test
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
    test_burst_cushion_and_hold();
    test_cushion_validation_and_zero_cushion();
    test_conservation_with_prefill_and_recovery();
    return test::finish("latency_governor");
}
