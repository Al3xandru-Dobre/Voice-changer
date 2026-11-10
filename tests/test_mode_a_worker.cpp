#include "pipeline/ModeAWorker.hpp"
#include "AllocationCounter.hpp"
#include "FakeModelStep.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {
using namespace fluxvoice;

void test_stream_and_accounting(int fps) {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder, fps));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder, fps));
    const std::array<float, 4> embedding{};
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding);
    ModeAWorker worker(router, pipeline);
    CHECK(pipeline.warm_up(3));
    CHECK(worker.prepare());
    const auto target = pipeline.governor().target_samples();
    CHECK(router.output_queued_approx() == target);
    CHECK(worker.snapshot().queued == static_cast<std::int64_t>(target));
    std::vector<float> sent, got;
    sent.reserve(32000);
    got.reserve(32000);
    std::array<float, 160> in{}, out{};
    const long before = test::allocation_count();
    for (std::size_t k = 0; k < 100; ++k) {
        const std::size_t n = 1 + k * 31 % in.size();
        for (std::size_t i = 0; i < n; ++i)
            in[i] = static_cast<float>((sent.size() + i) % 97) / 128.0f;
        sent.insert(sent.end(), in.begin(), in.begin() + static_cast<std::ptrdiff_t>(n));
        router.process(out.data(), in.data(), n);
        got.insert(got.end(), out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n));
        CHECK(worker.pump());
        const auto s = worker.snapshot();
        CHECK(s.underruns == 0 && s.overruns == 0 && s.skipped == 0 && s.drops == 0);
        CHECK(s.queued == static_cast<std::int64_t>(target));
    }
    if constexpr (test::kAllocationCounting) CHECK(test::allocation_count() == before);
    for (std::size_t i = 0; i < got.size(); ++i)
        CHECK(got[i] == (i < target ? 0.0f : sent[i - target]));
    router.discard_queues();
    worker.reset();
    CHECK(worker.snapshot().queued == 0);
    CHECK(worker.prepare());
    CHECK(worker.snapshot().queued == static_cast<std::int64_t>(target));
}

void test_stall_recovery_and_failures() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    PipelineOptions options;
    options.max_consecutive_errors = 2;
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, options);
    ModeAWorker worker(router, pipeline);
    CHECK(worker.prepare());
    std::array<float, 320> in{}, out{};
    for (int i = 0; i < 6; ++i) router.process(out.data(), in.data(), in.size());
    const auto before = worker.snapshot();
    CHECK(before.queued > static_cast<std::int64_t>(pipeline.governor().target_samples() + 320));
    const auto steps_before = enc.steps;
    CHECK(worker.pump());
    const auto after = worker.snapshot();
    CHECK(enc.steps == steps_before + 6); // input/model timeline is never skipped
    CHECK(after.skipped > 0 && after.drops == 0);
    CHECK(after.queued == static_cast<std::int64_t>(after.prefilled + after.underruns -
                                                 after.overruns - after.drops - after.skipped));
    CHECK(after.queued == static_cast<std::int64_t>(router.output_queued_approx()));
    CHECK(after.queued <= static_cast<std::int64_t>(pipeline.governor().target_samples() + 320));

    // Failed startup/reconfigure must not layer a second prefill on old PCM.
    CHECK(!worker.prepare());
    router.discard_queues();
    worker.reset();
    CHECK(worker.snapshot().underruns == 0 && worker.snapshot().queued == 0);
    CHECK(worker.prepare());
    voc.always_fail = true;
    for (int i = 0; i < 2; ++i) router.process(out.data(), in.data(), in.size());
    CHECK(!worker.pump() && worker.failed());
    CHECK(worker.error() == "fake Run failure");
    const auto failed_steps = enc.steps;
    CHECK(!worker.pump() && enc.steps == failed_steps); // fatal error is latched
    voc.always_fail = false;
    router.discard_queues();
    CHECK(worker.prepare());
    CHECK(!worker.failed());
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());
}
} // namespace

int main() {
    for (int fps : {1, 2, 4}) test_stream_and_accounting(fps);
    test_stall_recovery_and_failures();
    return test::finish("mode_a_worker");
}
