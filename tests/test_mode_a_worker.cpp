#include "pipeline/ModeAWorker.hpp"
#include "AllocationCounter.hpp"
#include "FakeModelStep.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {
using namespace fluxvoice;

// output_chain = false keeps these plumbing/accounting tests on raw vocoder
// PCM (the shipped chain has its own coverage in test_voice_params).
PipelineOptions raw_options() {
    PipelineOptions o;
    o.output_chain = false;
    return o;
}

void test_stream_and_accounting(int fps) {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder, fps));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder, fps));
    const std::array<float, 4> embedding{};
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, raw_options());
    ParamChannel params;
    ModeAWorker worker(router, pipeline, &params);
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
        // `held` is the partial hop the worker has read and not yet written; with
        // it the queues' contents reproduce `queued` after every pump (M4 Stage 06).
        const std::size_t hop = pipeline.geometry().hop;
        CHECK(s.held == sent.size() % hop);
        CHECK(static_cast<std::int64_t>(router.input_queued_approx() + router.output_queued_approx() +
                                        s.held) == s.queued);
    }
    if constexpr (test::kAllocationCounting) CHECK(test::allocation_count() == before);
    for (std::size_t i = 0; i < got.size(); ++i)
        CHECK(got[i] == (i < target ? 0.0f : sent[i - target]));
    router.discard_queues();
    worker.reset();
    CHECK(worker.snapshot().queued == 0 && worker.snapshot().held == 0);
    CHECK(worker.prepare());
    CHECK(worker.snapshot().queued == static_cast<std::int64_t>(target));
}

// `held`: read from the input queue, not yet written, dropped or skipped.
void test_held_samples() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, raw_options());
    ModeAWorker worker(router, pipeline);
    CHECK(worker.prepare() && worker.snapshot().held == 0);
    std::array<float, 160> in{}, out{};   // half a 320-sample hop
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.snapshot().held == 0);   // still in the input queue, which is counted separately
    CHECK(worker.pump());
    CHECK(worker.snapshot().held == 160 && router.input_queued_approx() == 0);
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());                 // the hop completes and is written
    CHECK(worker.snapshot().held == 0);
    CHECK(static_cast<std::int64_t>(router.output_queued_approx()) == worker.snapshot().queued);
    // A skipped hop is released too (output only; the input was consumed).
    for (int i = 0; i < 6; ++i) router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump() && worker.snapshot().skipped > 0 && worker.snapshot().held == 0);
    CHECK(static_cast<std::int64_t>(router.output_queued_approx()) == worker.snapshot().queued);
}

void test_stall_recovery_and_failures() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    PipelineOptions options = raw_options();
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

void test_param_channel_delivery() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, raw_options());
    ParamChannel channel;
    ModeAWorker worker(router, pipeline, &channel);
    CHECK(worker.prepare());
    CHECK(pipeline.params().get(ParamId::PitchShift) == 0.0f);

    std::array<float, 320> in{}, out{};
    VoiceParams p;
    p.set(ParamId::PitchShift, 4.0f);
    p.set(ParamId::TimbreBlend, 0.5f);
    const std::uint64_t gen = channel.push(p);
    CHECK(channel.pushed_generation() == gen);
    // The worker drains the mailbox at the next hop before process().
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());
    CHECK(pipeline.params().get(ParamId::PitchShift) == 4.0f);
    CHECK(pipeline.params().get(ParamId::TimbreBlend) == 0.5f);
    CHECK(channel.applied_generation() == gen);
}

// Delivery (drain) and acceptance (set_params) are separate facts: a refused
// snapshot is counted and must not advance the accepted generation.
void test_accepted_and_rejected_params() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, raw_options());
    ParamChannel channel;
    ModeAWorker worker(router, pipeline, &channel);
    CHECK(worker.prepare());
    CHECK(worker.accepted_generation() == 0 && worker.rejected_params() == 0);

    std::array<float, 320> in{}, out{};
    VoiceParams good;
    good.set(ParamId::PitchShift, 2.0f);
    const std::uint64_t g1 = channel.push(good);
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());
    CHECK(channel.applied_generation() == g1 && worker.accepted_generation() == g1);
    CHECK(worker.rejected_params() == 0);

    VoiceParams bad = good;
    bad.set(ParamId::PitchShift, 5.0f);
    bad.voice_a = 9;  // only one voice exists: set_params() refuses
    const std::uint64_t g2 = channel.push(bad);
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());
    CHECK(channel.applied_generation() == g2);   // delivered...
    CHECK(worker.accepted_generation() == g1);   // ...but not accepted
    CHECK(worker.rejected_params() == 1);
    CHECK(pipeline.params().get(ParamId::PitchShift) == 2.0f);  // previous kept

    // Several snapshots between hops: the newest decides, and is reported by generation.
    channel.push(bad);
    VoiceParams next = good;
    next.set(ParamId::PitchShift, -1.0f);
    const std::uint64_t g4 = channel.push(next);
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());
    CHECK(worker.accepted_generation() == g4 && worker.rejected_params() == 1);
    CHECK(pipeline.params().get(ParamId::PitchShift) == -1.0f);

    // reset() (stream restart) keeps the lifetime counters.
    router.discard_queues();
    worker.reset();
    CHECK(worker.accepted_generation() == g4 && worker.rejected_params() == 1);
}

void test_meters() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, raw_options());
    ModeAWorker worker(router, pipeline);
    CHECK(worker.prepare());
    CHECK(worker.meters().in_peak == 0.0f && worker.meters().out_rms == 0.0f);
    std::array<float, 320> in{}, out{};
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = (i % 2 == 0) ? 0.5f : -0.25f;
    router.process(out.data(), in.data(), in.size());
    CHECK(worker.pump());
    const WorkerMeters m = worker.meters();
    CHECK(m.in_peak == 0.5f);
    CHECK(std::fabs(m.in_rms - std::sqrt((0.25f + 0.0625f) / 2.0f)) < 1e-6f);
    // Raw model output is the input hop (no chain): same levels at the output.
    CHECK(m.out_peak == 0.5f && std::fabs(m.out_rms - m.in_rms) < 1e-6f);
    // An empty read leaves the last values (display only).
    CHECK(worker.pump());
    CHECK(worker.meters().in_peak == 0.5f);
}

// The mute flag is applied by the worker at hop start, through the shipped
// output chain (output_chain = true).
void test_mute_applied_at_hop_start() {
    DuplexRouter router;
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    const std::array<float, 4> embedding{};
    PipelineOptions options;
    options.params.set(ParamId::GateThreshold, -80.0f);
    InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, options);
    ModeAWorker worker(router, pipeline);
    CHECK(worker.prepare());
    CHECK(!worker.muted());
    std::array<float, 320> in{}, out{};
    std::vector<float> got;
    std::size_t n = 0;
    const auto hop = [&] {
        for (float& x : in) x = 0.3f * std::sin(0.1f * static_cast<float>(n++));  // never DC
        router.process(out.data(), in.data(), in.size());
        got.insert(got.end(), out.begin(), out.end());
        CHECK(worker.pump());
    };
    for (int i = 0; i < 6; ++i) hop();
    worker.set_muted(true);
    CHECK(worker.muted());
    for (int i = 0; i < 8; ++i) hop();
    worker.set_muted(false);
    for (int i = 0; i < 8; ++i) hop();
    // 560 samples of prefill, then the stream: muted hops 6..13 land at 560 + 6*320 ...
    const std::size_t muted_from = 560 + 7 * 320, muted_to = 560 + 14 * 320;
    for (std::size_t i = muted_from; i < muted_to; ++i) CHECK(std::fabs(got[i]) < 1e-3f);
    const auto peak = [&](std::size_t hop_index) {
        float p = 0.0f;
        for (std::size_t i = 0; i < 320; ++i) p = std::max(p, std::fabs(got[560 + hop_index * 320 + i]));
        return p;
    };
    CHECK(peak(5) > 0.25f);    // before the mute
    CHECK(peak(17) > 0.25f);   // after the unmute
}

void test_lookahead_does_not_inflate_queue_target() {
    std::size_t base_target = 0;
    for (int lookahead : {0, 1, 2}) {
        DuplexRouter router;
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder, 1, lookahead));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder, 1, 0));
        const std::array<float, 4> embedding{};
        InferencePipeline pipeline(enc, voc, {4.8, 0.25, -4.0, 1.5}, embedding, raw_options());
        const std::size_t target = pipeline.governor().target_samples();
        if (lookahead == 0) base_target = target;
        CHECK(target == 560);
        CHECK(target == base_target);
        CHECK(pipeline.geometry().model_delay_samples() == static_cast<std::size_t>(lookahead * 320));
        CHECK(pipeline.geometry().algorithmic_samples() ==
              pipeline.geometry().queue_samples() + pipeline.geometry().model_delay_samples());

        ModeAWorker worker(router, pipeline);
        CHECK(worker.prepare());
        std::vector<float> got;
        got.reserve(320 * 12);
        std::array<float, 320> in{}, out{};
        for (int step = 0; step < 12; ++step) {
            in.fill(0.0f);
            if (step == 0) in[7] = 1.0f;
            router.process(out.data(), in.data(), in.size());
            got.insert(got.end(), out.begin(), out.end());
            CHECK(worker.pump());
        }
        const std::size_t expected_pos = target + pipeline.geometry().model_delay_samples() + 7;
        CHECK(got[expected_pos] == 1.0f);
        CHECK(std::count(got.begin(), got.end(), 1.0f) == 1);
    }
}
} // namespace

int main() {
    for (int fps : {1, 2, 4}) test_stream_and_accounting(fps);
    test_held_samples();
    test_stall_recovery_and_failures();
    test_param_channel_delivery();
    test_accepted_and_rejected_params();
    test_meters();
    test_mute_applied_at_hop_start();
    test_lookahead_does_not_inflate_queue_target();
    return test::finish("mode_a_worker");
}
