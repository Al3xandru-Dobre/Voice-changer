// ModeAHost tests (Milestone 4 / Stage 05). ORT-free and hardware-free: fake
// models (FakeModelStep / FakeTranslatorStep behind a fake HostModelLoader),
// the fake audio backend and a real AudioController, so the worker thread, the
// controller hooks and the prepare hook run exactly as they do in
// `fluxvoice --vc` and the GUI.
//
// Real worker thread: every wait polls a counter with a generous deadline, and
// nothing asserts on timing, only on state reached.
#include "pipeline/ModeAHost.hpp"

#include "pipeline/OrtModel.hpp"  // kBuiltWithoutOrtMessage

#include "AllocationCounter.hpp"
#include "FakeAudioBackend.hpp"
#include "FakeModelStep.hpp"
#include "FakeTranslatorStep.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace fluxvoice;
using namespace std::chrono_literals;
using fluxvoice::testing::FakeAudioBackend;

constexpr std::size_t kHop = 320;  // fps 1 fake contract
constexpr std::size_t kTarget = 560;

// ---- Probe: counters that outlive the fakes --------------------------------

struct Probe {
    std::atomic<long> enc_steps{0}, voc_steps{0};
    std::atomic<long> late_calls{0};      // step()/reset() after the models began to be released
    std::atomic<bool> released{false};
    std::atomic<bool> fail_vocoder{false};
    std::atomic<int> slow_vocoder_ms{0};  // makes a burst long enough to catch a missing quiesce
    std::atomic<int> release_window_ms{0};
};

/// FakeModelStep + counters. The wrapper only observes; the maths is the fake's.
class TrackedStep final : public ModelStep {
public:
    TrackedStep(ModelContract c, std::shared_ptr<Probe> probe, bool vocoder)
        : inner_(std::move(c)), probe_(std::move(probe)), vocoder_(vocoder) {}

    const ModelContract& contract() const noexcept override { return inner_.contract(); }
    std::span<float> input_f32(std::string_view n) noexcept override { return inner_.input_f32(n); }
    std::span<std::int64_t> input_i64(std::string_view n) noexcept override { return inner_.input_i64(n); }
    std::span<const float> output_f32(std::string_view n) const noexcept override { return inner_.output_f32(n); }
    std::span<const std::int64_t> output_i64(std::string_view n) const noexcept override {
        return inner_.output_i64(n);
    }
    bool step() noexcept override {
        if (probe_->released.load(std::memory_order_relaxed)) probe_->late_calls.fetch_add(1);
        if (!vocoder_) {
            probe_->enc_steps.fetch_add(1, std::memory_order_relaxed);
        } else {
            probe_->voc_steps.fetch_add(1, std::memory_order_relaxed);
            const int slow = probe_->slow_vocoder_ms.load(std::memory_order_relaxed);
            if (slow > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slow));
            if (probe_->fail_vocoder.load(std::memory_order_relaxed)) return false;
        }
        return inner_.step();
    }
    void reset() noexcept override {
        if (probe_->released.load(std::memory_order_relaxed)) probe_->late_calls.fetch_add(1);
        inner_.reset();
    }
    std::string_view last_error() const noexcept override { return inner_.last_error(); }

private:
    test::FakeModelStep inner_;
    std::shared_ptr<Probe> probe_;
    bool vocoder_;
};

std::vector<VoiceEntry> fake_voices() {
    std::vector<VoiceEntry> v(2);
    const char* names[] = {"alpha", "beta"};
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i].name = names[i];
        v[i].display = names[i];
        v[i].license = "CC0-1.0";
        v[i].source = "test";
        v[i].stats = {4.8, 0.25, -4.0, 1.5};
        v[i].embedding.assign(4, 0.0f);
    }
    return v;
}

/// Two-model (Mode A) set. Releasing it opens a window in which a worker that
/// is still running would be counted by the probe.
class FakeModels final : public HostModels {
public:
    FakeModels(std::shared_ptr<Probe> probe, int fps)
        : probe_(probe),
          enc_(test::fake_contract(ModelRole::kEncoder, fps), probe, false),
          voc_(test::fake_contract(ModelRole::kVocoder, fps), probe, true),
          voices_(fake_voices()) {}
    ~FakeModels() override {
        probe_->released.store(true);
        const int ms = probe_->release_window_ms.load();
        if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    ModelStep& encoder() noexcept override { return enc_; }
    ModelStep& vocoder() noexcept override { return voc_; }
    const std::vector<VoiceEntry>& voices() const noexcept override { return voices_; }

private:
    std::shared_ptr<Probe> probe_;
    TrackedStep enc_, voc_;
    std::vector<VoiceEntry> voices_;
};

struct TranslatorKnobs {
    bool no_readiness = false;             // translator_enc without a readiness head
    bool stateful_vocoder_broken = false;  // overlap > 0: the Mode B constructor refuses
    bool warm_up_fails = false;
    int lang_count = 4;
};

/// Mode B quartet (the offline twins of the ONNX fixtures).
class FakeTranslatorModels final : public HostModels {
public:
    explicit FakeTranslatorModels(const TranslatorKnobs& k)
        : enc_(4), tenc_(make_tenc_contract(k)), tdec_(tenc_),
          voc_(4, k.stateful_vocoder_broken ? 80 : 0), voices_(fake_voices()) {
        const char* codes[] = {"en", "es", "fr", "de"};
        for (int i = 0; i < k.lang_count && i < 4; ++i) langs_.langs.push_back({codes[i], i, codes[i]});
        tdec_.fail_next = k.warm_up_fails;
    }
    ModelStep& encoder() noexcept override { return enc_; }
    ModelStep& vocoder() noexcept override { return voc_; }
    ModelStep* translator_enc() noexcept override { return &tenc_; }
    ModelStep* translator_dec() noexcept override { return &tdec_; }
    const std::vector<VoiceEntry>& voices() const noexcept override { return voices_; }
    const LangCatalog* langs() const noexcept override { return &langs_; }

private:
    static ModelContract make_tenc_contract(const TranslatorKnobs& k) {
        ModelContract c = test::translator_enc_contract();
        if (k.no_readiness) c.has_readiness_output = false;
        return c;
    }
    test::FakeDualEncoder enc_;
    test::FakeTranslatorEnc tenc_;
    test::FakeTranslatorDec tdec_;
    test::FakeUnitVocoder voc_;
    std::vector<VoiceEntry> voices_;
    LangCatalog langs_;
};

class FakeLoader final : public HostModelLoader {
public:
    std::shared_ptr<Probe> probe = std::make_shared<Probe>();
    int fps = 1;
    bool translator_files = false;      // has_translator()
    std::string load_error;             // plain load fails with this
    std::string translator_load_error;  // translator load fails with this
    TranslatorKnobs translator;
    mutable std::vector<std::pair<int, bool>> loads;  // (frames_per_step, with_translator)
    mutable std::vector<int> has_translator_queries;

    bool has_translator(const std::filesystem::path&, int frames_per_step) const override {
        has_translator_queries.push_back(frames_per_step);
        return translator_files;
    }
    std::unique_ptr<HostModels> load(const std::filesystem::path&, int frames_per_step,
                                     bool with_translator, std::string& error) const override {
        loads.emplace_back(frames_per_step, with_translator);
        if (with_translator) {
            if (!translator_load_error.empty()) {
                error = translator_load_error;
                return nullptr;
            }
            return std::make_unique<FakeTranslatorModels>(translator);
        }
        if (!load_error.empty()) {
            error = load_error;
            return nullptr;
        }
        return std::make_unique<FakeModels>(probe, fps);
    }
};

// ---- Seeder ------------------------------------------------------------------

struct SeederLog {
    int calls = 0;
    bool refuse = false;
    std::string refuse_message;
    std::vector<std::string> fallbacks, unavailable;
    std::vector<bool> had_langs;
    bool translate_on = false;  // start on the translator route
    bool adaptive = false;      // request adaptive_policy
};

HostParamSeeder make_seeder(SeederLog& log) {
    return [&log](const HostBundleInfo& info, VoiceParams& params, S2utOptions& s2ut,
                  std::string& error) {
        ++log.calls;
        log.fallbacks.emplace_back(info.translator_fallback);
        log.unavailable.emplace_back(info.translator_unavailable);
        log.had_langs.push_back(info.langs != nullptr);
        if (log.refuse) {
            error = log.refuse_message;
            return false;
        }
        params.set(ParamId::GateThreshold, -80.0f);  // the 0.3 test sine is always above it
        if (info.langs) {
            params.src_lang_id = 1;
            params.tgt_lang_id = 0;
            s2ut.src_lang_id = 1;
            s2ut.tgt_lang_id = 0;
            s2ut.policy.k = 3;
            s2ut.policy.adaptive_mode = log.adaptive;
            s2ut.translate = log.translate_on;
            if (log.adaptive) params.set(ParamId::AdaptivePolicy, 1.0f);
        }
        return true;
    };
}

// ---- Stream rig ---------------------------------------------------------------

constexpr std::size_t kMaxBlock = kHop * 64;
std::array<float, kMaxBlock> g_in{}, g_out{};

template <class Pred>
bool wait_until(Pred done, std::chrono::milliseconds limit = 5000ms) {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (!done()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(300us);
    }
    return true;
}

struct Stream {
    FakeAudioBackend fb;
    std::unique_ptr<AudioController> ctl;
    AudioController::TimePoint t0 = AudioController::Clock::now();
    std::uint64_t phase = 0;

    Stream() {
        fb.capture = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    }
    void make(const ModeAHost& host, std::optional<std::uint32_t> rate_override = std::nullopt) {
        ControllerConfig cfg;
        cfg.callback_stall_timeout = 500ms;
        cfg.initial_retry_backoff = 200ms;
        host.configure(cfg);
        if (rate_override) cfg.sample_rate = *rate_override;
        ctl = std::make_unique<AudioController>(fb, cfg);
    }
    bool start() {
        return is_ok(ctl->open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                                         CaptureMode::OrdinaryMicrophone, t0));
    }
    /// One audio callback of `n` samples of `amp` * sine (continuous across calls).
    void callback(std::size_t n, double amp = 0.3) {
        for (std::size_t i = 0; i < n; ++i)
            g_in[i] = static_cast<float>(amp * std::sin(6.283185307179586 * 440.0 *
                                                         static_cast<double>(phase++) / 16000.0));
        fb.pump_callback(g_in.data(), g_out.data(), n);
    }
};

struct Fx {
    std::shared_ptr<FakeLoader> loader = std::make_shared<FakeLoader>();
    Stream stream;  // outlives the host: the host detaches from a live controller
    ModeAHost host{loader};
    SeederLog log;
    HostLoadRequest req;
    HostError err;

    Fx() { req.model_dir = "fake"; }
    bool load() { return host.load(req, make_seeder(log), err); }
    /// load + controller + attach + start.
    bool run(HostWorkerStartCallback on_start = {}) {
        if (!load()) return false;
        stream.make(host);
        if (!host.attach(*stream.ctl, err, std::move(on_start))) return false;
        return stream.start();
    }
    /// The worker finished hop number `want`: its encoder ran AND its output is
    /// back in the queue (the prefill is only 1.75 hops deep, so the next
    /// callback would underrun if it popped before the write).
    bool wait_hop(long want) {
        return wait_until([&] {
            return loader->probe->enc_steps.load() >= want &&
                   stream.ctl->router().output_queued_approx() >= kTarget;
        });
    }
    /// Feed `hops` hops one at a time, waiting for the worker after each.
    bool feed_hops(std::size_t hops, double amp = 0.3) {
        for (std::size_t h = 0; h < hops; ++h) {
            const long want = loader->probe->enc_steps.load() + 1;
            stream.callback(kHop, amp);
            if (!wait_hop(want)) return false;
        }
        return true;
    }
    /// Worker idle: nothing left to read and the counters stopped moving.
    bool settle() {
        WorkerSnapshot last{};
        int stable = 0;
        return wait_until([&] {
            const WorkerSnapshot s = host.snapshot();
            const bool same = s.queued == last.queued && s.skipped == last.skipped &&
                              s.drops == last.drops && s.underruns == last.underruns;
            last = s;
            stable = (same && stream.ctl->router().input_queued_approx() == 0) ? stable + 1 : 0;
            std::this_thread::sleep_for(2ms);
            return stable >= 5;
        });
    }
};

std::int64_t identity_of(const WorkerSnapshot& s) {
    return static_cast<std::int64_t>(s.prefilled) + static_cast<std::int64_t>(s.underruns) -
           static_cast<std::int64_t>(s.overruns) - static_cast<std::int64_t>(s.drops) -
           static_cast<std::int64_t>(s.skipped);
}

double rms(const std::vector<float>& v, std::size_t from, std::size_t to) {
    double s = 0.0;
    for (std::size_t i = from; i < to; ++i) s += static_cast<double>(v[i]) * v[i];
    return std::sqrt(s / static_cast<double>(to - from));
}

// ---- Tests --------------------------------------------------------------------

void test_start_and_conservation() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.err.ok());
    CHECK(fx.host.loaded() && fx.host.attached());
    CHECK(fx.stream.ctl->status().state == StreamState::Running);
    CHECK(fx.host.contract_rate() == 16000 && fx.host.geometry().hop == kHop);
    CHECK(fx.host.governor_target_samples() == kTarget);

    // The prepare hook ran: prefill is in the queue and in the accounting.
    WorkerSnapshot s = fx.host.snapshot();
    CHECK(s.prefilled == kTarget && s.queued == static_cast<std::int64_t>(kTarget));
    CHECK(fx.stream.ctl->router().output_queued_approx() == kTarget);

    const long enc0 = fx.loader->probe->enc_steps.load(), voc0 = fx.loader->probe->voc_steps.load();
    CHECK(fx.feed_hops(20));
    CHECK(fx.settle());
    CHECK(fx.loader->probe->enc_steps.load() - enc0 == 20);  // hops reach both fake models
    CHECK(fx.loader->probe->voc_steps.load() - voc0 == 20);
    s = fx.host.snapshot();
    CHECK(s.underruns == 0 && s.overruns == 0 && s.drops == 0 && s.skipped == 0);
    CHECK(s.queued == identity_of(s) && s.queued == static_cast<std::int64_t>(kTarget));
    CHECK(static_cast<std::int64_t>(fx.stream.ctl->router().output_queued_approx()) == s.queued);
    CHECK(fx.host.timing().samples > 0);
    CHECK(fx.host.pipeline().errors() == 0);

    // A burst the governor must trim: 8 silent hops in one callback drain the
    // prefill (underruns) and refill the queue past the target (skipped; the
    // governor only ever trims quiet audio).
    fx.stream.callback(8 * kHop, 0.0);
    CHECK(fx.settle());
    s = fx.host.snapshot();
    CHECK(s.skipped > 0 && s.underruns > 0 && s.prefilled == kTarget);
    CHECK(s.queued == identity_of(s));  // prefilled and skipped both in the identity
    CHECK(static_cast<std::int64_t>(fx.stream.ctl->router().output_queued_approx()) == s.queued);
    CHECK(s.queued <= static_cast<std::int64_t>(fx.host.governor_target_samples() + kHop));

    // The worker meters follow the audio.
    CHECK(fx.feed_hops(4, 0.25));
    const WorkerMeters m = fx.host.meters();
    CHECK(m.in_peak > 0.2f && m.in_peak <= 0.25f + 1e-6f);
    CHECK(m.in_rms > 0.1f && m.in_rms < m.in_peak);
    CHECK(m.out_peak > 0.0f);
}

void test_worker_start_callback_and_priority() {
    Fx fx;
    std::atomic<int> started{0};
    std::string method;
    std::atomic<int> elevated{-1};
    const bool ok = fx.run([&](const char* m, bool e, int) {
        method = m;
        elevated.store(e ? 1 : 0);
        started.fetch_add(1);
    });
    CHECK(ok);
    CHECK(wait_until([&] { return started.load() == 1; }));
    CHECK(!method.empty() && method == fx.host.priority_method());
    CHECK((elevated.load() == 1) == fx.host.priority_elevated());
    // A stream restart relaunches the worker: the callback fires once per start.
    const auto t = fx.stream.t0;
    CHECK(is_ok(fx.stream.ctl->reconfigure(FollowSystemDefault{}, FollowSystemDefault{},
                                           CaptureMode::OrdinaryMicrophone, t + 200ms)));
    CHECK(wait_until([&] { return started.load() == 2; }));
}

void test_configure_and_rate_check() {
    Fx fx;
    CHECK(fx.load());
    ControllerConfig cfg;
    cfg.sample_rate = 48000;
    cfg.fade_underruns = false;
    cfg.max_queued_samples = 5120;
    fx.host.configure(cfg);
    CHECK(cfg.sample_rate == 16000 && cfg.fade_underruns && cfg.max_queued_samples == 0);

    // The prepare hook refuses a client rate that differs from the contract rate.
    fx.stream.make(fx.host, 48000u);
    CHECK(fx.host.attach(*fx.stream.ctl, fx.err));
    CHECK(!fx.stream.start());
    CHECK(fx.stream.ctl->status().state != StreamState::Running);

    // An unloaded host leaves the config alone.
    ModeAHost empty(fx.loader);
    ControllerConfig untouched;
    untouched.sample_rate = 22050;
    empty.configure(untouched);
    CHECK(untouched.sample_rate == 22050 && !untouched.fade_underruns);
}

void test_params_accepted_vs_rejected() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.feed_hops(2));
    CHECK(fx.host.pushed_generation() == 0 && fx.host.accepted_generation() == 0 &&
          fx.host.rejected_params() == 0);

    VoiceParams p;
    p.set(ParamId::PitchShift, 3.0f);
    p.set(ParamId::GateThreshold, -80.0f);
    p.voice_a = 1;
    p.voice_b = 1;
    const std::uint64_t g1 = fx.host.push_params(p);
    CHECK(g1 >= 1 && fx.host.flush_params());
    CHECK(fx.feed_hops(1));  // applied at the next hop start
    CHECK(wait_until([&] { return fx.host.accepted_generation() == g1; }));
    CHECK(fx.host.applied_generation() == g1 && fx.host.pushed_generation() == g1);
    CHECK(fx.host.rejected_params() == 0);

    // Out-of-range voice index: delivered, refused, counted, previous kept.
    VoiceParams bad = p;
    bad.voice_a = 99;
    const std::uint64_t g2 = fx.host.push_params(bad);
    CHECK(g2 > g1);
    CHECK(fx.feed_hops(1));
    CHECK(wait_until([&] { return fx.host.applied_generation() == g2; }));
    CHECK(wait_until([&] { return fx.host.rejected_params() == 1; }));
    CHECK(fx.host.accepted_generation() == g1);  // not advanced by the refused snapshot
    CHECK(fx.host.pushed_generation() == g2);

    // The next valid snapshot is accepted; the refusal stays counted.
    VoiceParams good = p;
    good.set(ParamId::PitchShift, -2.0f);
    const std::uint64_t g3 = fx.host.push_params(good);
    CHECK(fx.feed_hops(1));
    CHECK(wait_until([&] { return fx.host.accepted_generation() == g3; }));
    CHECK(fx.host.rejected_params() == 1);

    // Params are worker-owned: read them once the worker is quiesced.
    fx.stream.ctl->stop();
    CHECK(fx.host.pipeline().params().get(ParamId::PitchShift) == -2.0f);
    CHECK(fx.host.pipeline().params().voice_a == 1);
    CHECK(fx.host.accepted_generation() == g3 && fx.host.applied_generation() == g3);
}

void test_voice_lookup() {
    Fx fx;
    std::string error;
    CHECK(!fx.host.voice_index("alpha", error).has_value() && !error.empty());  // not loaded yet
    CHECK(fx.host.voices().empty());
    CHECK(fx.load());
    CHECK(fx.host.voices().size() == 2);
    error.clear();
    CHECK(fx.host.voice_index("alpha", error) == std::optional<std::uint16_t>(0));
    CHECK(fx.host.voice_index("beta", error) == std::optional<std::uint16_t>(1));
    error.clear();
    const auto none = fx.host.voice_index("gamma", error);
    CHECK(!none.has_value());
    CHECK(error.find("unknown voice 'gamma'") != std::string::npos);
    CHECK(error.find("alpha") != std::string::npos && error.find("beta") != std::string::npos);
    CHECK(!fx.host.voice_index("", error).has_value());
    CHECK(!fx.host.voice_index("Alpha", error).has_value());  // exact names only
}

void test_bundle_swap_quiesces_before_models_die() {
    auto old_loader = std::make_shared<FakeLoader>();
    old_loader->probe->slow_vocoder_ms = 2;  // a burst keeps the worker busy for tens of ms
    old_loader->probe->release_window_ms = 30;
    const std::shared_ptr<Probe> old_probe = old_loader->probe;
    Stream stream;
    auto old_host = std::make_unique<ModeAHost>(old_loader);
    SeederLog log;
    HostError err;
    HostLoadRequest req;
    CHECK(old_host->load(req, make_seeder(log), err));
    stream.make(*old_host);
    CHECK(old_host->attach(*stream.ctl, err));
    CHECK(stream.start());
    stream.callback(kHop);
    CHECK(wait_until([&] { return old_probe->enc_steps.load() >= 1; }));

    // Stop while the worker still has a long burst to chew through: the quiesce
    // hook must join it, so nothing steps after stop() returns.
    stream.callback(40 * kHop);
    std::this_thread::sleep_for(6ms);
    stream.ctl->stop();
    const long at_stop = old_probe->enc_steps.load();
    std::this_thread::sleep_for(60ms);
    CHECK(old_probe->enc_steps.load() == at_stop);

    old_host->detach();
    CHECK(!old_host->attached());
    // Detached: the controller no longer calls into the host (hooks cleared).
    CHECK(stream.start());
    stream.callback(2 * kHop);
    std::this_thread::sleep_for(20ms);
    CHECK(old_probe->enc_steps.load() == at_stop);
    stream.ctl->stop();

    // Destroy the old host and with it the old models. The old fakes stay alive
    // for a 30 ms window after their release flag is set: a worker still running
    // would be counted.
    old_host.reset();
    CHECK(old_probe->released.load());
    CHECK(old_probe->late_calls.load() == 0);

    // A new host on the same controller.
    auto next_loader = std::make_shared<FakeLoader>();
    ModeAHost next(next_loader);
    SeederLog next_log;
    CHECK(next.load(req, make_seeder(next_log), err));
    CHECK(next.attach(*stream.ctl, err));
    CHECK(stream.start());
    const long next0 = next_loader->probe->enc_steps.load();
    stream.callback(kHop);
    CHECK(wait_until([&] { return next_loader->probe->enc_steps.load() > next0; }));
    CHECK(old_probe->late_calls.load() == 0 && old_probe->enc_steps.load() == at_stop);
    stream.ctl->stop();
    next.detach();
}

void test_destroy_while_running_joins_worker() {
    // The defensive path: the host dies while its stream is live and its
    // worker is busy. The destructor must join before the models are released
    // and must clear the controller hooks.
    auto loader = std::make_shared<FakeLoader>();
    loader->probe->slow_vocoder_ms = 2;
    loader->probe->release_window_ms = 30;
    const std::shared_ptr<Probe> probe = loader->probe;
    Stream stream;
    {
        ModeAHost host(loader);
        SeederLog log;
        HostError err;
        HostLoadRequest req;
        CHECK(host.load(req, make_seeder(log), err));
        stream.make(host);
        CHECK(host.attach(*stream.ctl, err));
        CHECK(stream.start());
        stream.callback(40 * kHop);
        std::this_thread::sleep_for(6ms);
    }  // ~ModeAHost: detach + join + destroy pipeline + models
    CHECK(probe->released.load());
    CHECK(probe->late_calls.load() == 0);
    const long after = probe->enc_steps.load();
    std::this_thread::sleep_for(40ms);
    CHECK(probe->enc_steps.load() == after);
    // The controller was left with no hooks pointing at the dead host: a stop
    // and a restart run without calling into it.
    stream.ctl->stop();
    CHECK(stream.ctl->status().state == StreamState::Idle);
    CHECK(stream.start());
    stream.callback(kHop);
    stream.ctl->stop();
}

void test_device_loss_and_recovery() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.feed_hops(3));
    // An overload leaves the epoch's accounting dirty (underruns).
    fx.stream.callback(8 * kHop);
    CHECK(fx.settle());
    CHECK(fx.host.snapshot().underruns > 0);
    const auto t0 = fx.stream.t0;
    fx.stream.ctl->tick(t0 + 100ms);  // progress seen
    CHECK(fx.stream.ctl->status().state == StreamState::Running);

    // No callback progress for 500 ms: stalled, bounded recovery. The worker is
    // quiesced and reset; prepare() has not run yet.
    fx.stream.ctl->tick(t0 + 700ms);
    CHECK(fx.stream.ctl->status().state == StreamState::Recovering);
    CHECK(fx.host.loaded() && fx.host.attached());
    CHECK(fx.host.snapshot().prefilled == 0 && fx.host.snapshot().underruns == 0);

    // The retry reopens the stream: the prepare hook runs again (rate check +
    // prefill) and the worker is relaunched on the same bundle.
    const std::size_t loads = fx.loader->loads.size();
    fx.stream.ctl->tick(t0 + 1000ms);
    CHECK(fx.stream.ctl->status().state == StreamState::Running);
    CHECK(fx.loader->loads.size() == loads && fx.log.calls == 1);  // no reload, no re-seed
    const WorkerSnapshot s = fx.host.snapshot();
    CHECK(s.prefilled == kTarget && s.queued == static_cast<std::int64_t>(kTarget));
    CHECK(s.underruns == 0 && s.skipped == 0);
    CHECK(fx.stream.ctl->router().output_queued_approx() == kTarget);
    const long enc0 = fx.loader->probe->enc_steps.load();
    CHECK(fx.feed_hops(3));
    CHECK(fx.loader->probe->enc_steps.load() - enc0 == 3);
    CHECK(!fx.host.failed());
}

void test_fatal_pipeline_error() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.feed_hops(2));
    CHECK(!fx.host.failed() && fx.host.error().empty());

    // Every vocoder step fails; after max_consecutive_errors (8) the pipeline
    // asks to stop and the worker latches the failure and exits.
    fx.loader->probe->fail_vocoder = true;
    fx.stream.callback(12 * kHop);
    CHECK(wait_until([&] { return fx.host.failed(); }));
    CHECK(!fx.host.error().empty());
    CHECK(fx.host.error().find("fake Run failure") != std::string::npos);
    CHECK(fx.host.pipeline().errors() >= 8);
    const long frozen = fx.loader->probe->enc_steps.load();
    fx.stream.callback(2 * kHop);
    std::this_thread::sleep_for(40ms);
    CHECK(fx.loader->probe->enc_steps.load() == frozen);  // the worker stopped

    // The failure stays visible until the stream is restarted ("Retry").
    fx.stream.ctl->stop();
    CHECK(fx.host.failed() && !fx.host.error().empty());
    fx.loader->probe->fail_vocoder = false;
    CHECK(fx.stream.start());
    CHECK(!fx.host.failed() && fx.host.error().empty());
    CHECK(fx.feed_hops(2));
    CHECK(!fx.host.failed());
}

void test_steady_state_allocates_nothing() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.feed_hops(6));
    VoiceParams p;
    p.set(ParamId::GateThreshold, -80.0f);
    fx.host.push_params(p);  // control-thread work, before the measured window
    CHECK(fx.feed_hops(2));
    if constexpr (test::kAllocationCounting) {
        const long before = test::allocation_count();
        const long enc0 = fx.loader->probe->enc_steps.load();
        for (int i = 0; i < 60; ++i) {
            fx.stream.callback(kHop);
            const WorkerSnapshot s = fx.host.snapshot();  // any-thread accessors allocate nothing
            const WorkerMeters m = fx.host.meters();
            (void)s;
            (void)m;
            if (!fx.wait_hop(enc0 + i + 1)) break;
        }
        CHECK(fx.loader->probe->enc_steps.load() - enc0 == 60);
        CHECK(test::allocation_count() == before);
    }
}

void test_mute_through_host() {
    Fx fx;
    CHECK(fx.run());
    CHECK(!fx.host.muted());
    std::vector<float> out;  // every callback's output, in order
    const auto hop = [&](bool change, bool mute) {
        if (change) fx.host.set_muted(mute);
        const long want = fx.loader->probe->enc_steps.load() + 1;
        fx.stream.callback(kHop);
        out.insert(out.end(), g_out.begin(), g_out.begin() + kHop);
        CHECK(fx.wait_hop(want));
    };
    for (int h = 0; h < 10; ++h) hop(false, false);
    for (int h = 10; h < 40; ++h) hop(h == 10, true);
    CHECK(fx.host.muted());
    for (int h = 40; h < 80; ++h) hop(h == 40, false);
    CHECK(!fx.host.muted());

    // The output lags the input by the 560-sample prefill (two hops).
    const double before = rms(out, 4 * kHop, 10 * kHop);
    CHECK(before > 0.15);
    // Muted (after the 5 ms ramp): below -60 dBFS.
    for (std::size_t i = 14 * kHop; i < 41 * kHop; ++i) CHECK(std::fabs(out[i]) < 0.001f);
    // Unmuted again: the level is back.
    const double after = rms(out, 60 * kHop, 80 * kHop);
    CHECK(std::fabs(after / before - 1.0) < 0.03);
}

void test_load_errors_leave_host_unloaded() {
    {   // kSeed: the seeder refuses; its message is carried, nothing else is built.
        Fx fx;
        fx.log.refuse = true;
        fx.log.refuse_message = "unknown voice 'x'";
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kSeed && fx.err.message == "unknown voice 'x'");
        CHECK(!fx.host.loaded() && fx.log.calls == 1);
        // The host is reusable: the next load succeeds.
        fx.log.refuse = false;
        CHECK(fx.load() && fx.err.ok() && fx.host.loaded());
        // ...and a later failure unloads it again.
        fx.log.refuse = true;
        CHECK(!fx.load() && !fx.host.loaded());
    }
    {   // kSeed through an exception from the seeder
        Fx fx;
        HostError e;
        const HostParamSeeder throwing = [](const HostBundleInfo&, VoiceParams&, S2utOptions&,
                                            std::string&) -> bool { throw std::runtime_error("boom"); };
        CHECK(!fx.host.load(fx.req, throwing, e));
        CHECK(e.kind == HostErrorKind::kSeed && e.message.find("boom") != std::string::npos);
        CHECK(!fx.host.loaded());
    }
    {   // kWarmUp: a model fails during warm-up; the stream must not start.
        Fx fx;
        fx.loader->probe->fail_vocoder = true;
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kWarmUp);
        CHECK(fx.err.message.find("Warm-up failed: ") == 0 &&
              fx.err.message.find("fake Run failure") != std::string::npos);
        CHECK(!fx.host.loaded());
    }
    {   // kQueueCapacity: latency target + one hop does not fit the audio queue
        Fx fx;
        fx.req.latency_target_ms = 5000;  // 80000 samples > 65536
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kQueueCapacity);
        CHECK(fx.err.message == "Latency target plus one hop exceeds the audio queue capacity");
        CHECK(!fx.host.loaded());
        fx.req.latency_target_ms = 100;  // 1600 samples: fits, and is honoured
        CHECK(fx.load() && fx.host.governor_target_samples() == 1600);
    }
    {   // kBundle
        Fx fx;
        fx.loader->load_error = "Failed to load encoder: nope";
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kBundle && fx.err.message == "Failed to load encoder: nope");
        CHECK(!fx.host.loaded());
        ModeAHost no_loader(nullptr);
        HostError e;
        CHECK(!no_loader.load(fx.req, make_seeder(fx.log), e) && e.kind == HostErrorKind::kBundle);
    }
    {   // kBuiltWithoutOrt: the shipped loader's message maps to its own kind
        Fx fx;
        fx.loader->load_error = kBuiltWithoutOrtMessage;
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kBuiltWithoutOrt && !fx.host.loaded());
        CHECK(fx.log.calls == 0);
    }
    {   // kPipeline: the constructor refuses the models (voice index out of range)
        Fx fx;
        HostError e;
        const HostParamSeeder bad_voice = [](const HostBundleInfo&, VoiceParams& p, S2utOptions&,
                                             std::string&) {
            p.voice_a = 7;
            return true;
        };
        CHECK(!fx.host.load(fx.req, bad_voice, e));
        CHECK(e.kind == HostErrorKind::kPipeline && e.message.find("Cannot prepare Mode A: ") == 0);
        CHECK(!fx.host.loaded());
    }
    {   // kState: wrong call order
        Fx fx;
        HostError e;
        AudioController other(fx.stream.fb);
        CHECK(!fx.host.attach(other, e) && e.kind == HostErrorKind::kState);  // attach before load
        CHECK(fx.run());
        const std::uint64_t before = fx.host.pushed_generation();
        CHECK(!fx.load());  // load while attached
        CHECK(fx.err.kind == HostErrorKind::kState);
        CHECK(fx.host.loaded() && fx.host.attached());  // untouched
        CHECK(fx.host.pushed_generation() == before);
        CHECK(!fx.host.attach(*fx.stream.ctl, e) && e.kind == HostErrorKind::kState);  // twice
        CHECK(fx.feed_hops(2));  // still running
    }
}

void test_translator_policy_and_fallback() {
    using Loads = std::vector<std::pair<int, bool>>;
    {   // kIfPresent, quartet usable: Mode B capable, seeder once, caller's fps (0)
        Fx fx;
        fx.loader->translator_files = true;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(fx.load());
        CHECK(fx.host.load_report().translator && !fx.host.load_report().translator_fallback);
        CHECK(fx.host.load_report().translator_unavailable.empty());
        CHECK(fx.host.langs() != nullptr && fx.host.langs()->langs.size() == 4);
        CHECK(fx.host.pipeline().has_translator() && fx.host.pipeline().mode() == PipelineMode::kVC);
        CHECK(fx.log.calls == 1 && fx.log.had_langs[0] && fx.log.fallbacks[0].empty());
        CHECK(fx.loader->has_translator_queries == std::vector<int>{0});
        CHECK((fx.loader->loads == Loads{{0, true}}));
        CHECK(fx.host.onset_latency_samples(3) > 0);
    }
    {   // kIfPresent, no quartet on disk: plain Mode A, never asks for translators
        Fx fx;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        fx.req.frames_per_step = 2;
        CHECK(fx.load());
        CHECK(!fx.host.load_report().translator && fx.host.langs() == nullptr);
        CHECK((fx.loader->loads == Loads{{2, false}}));
        CHECK(fx.host.onset_latency_samples(3) == 0);
    }
    {   // kNever never looks for translators even when the files are there
        Fx fx;
        fx.loader->translator_files = true;
        CHECK(fx.load() && fx.loader->has_translator_queries.empty());
        CHECK((fx.loader->loads == Loads{{0, false}}));
    }
    {   // kIfPresent, translator load fails: Mode A, reason reported (the seeder runs once, on Mode A)
        Fx fx;
        fx.loader->translator_files = true;
        fx.loader->translator_load_error = "Failed to load translator_enc: broken";
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(fx.load());
        const HostLoadReport& r = fx.host.load_report();
        CHECK(!r.translator && r.translator_fallback);
        CHECK(r.translator_unavailable == "Failed to load translator_enc: broken");
        CHECK(fx.log.calls == 1 && fx.log.fallbacks[0] == r.translator_unavailable && !fx.log.had_langs[0]);
        CHECK(fx.host.langs() == nullptr && !fx.host.pipeline().has_translator());
    }
    {   // kIfPresent, Mode B constructor refuses (stateful vocoder required): Mode A, seeder called twice
        Fx fx;
        fx.loader->translator_files = true;
        fx.loader->translator.stateful_vocoder_broken = true;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(fx.load());
        const HostLoadReport& r = fx.host.load_report();
        CHECK(!r.translator && r.translator_fallback && !r.translator_unavailable.empty());
        CHECK(fx.log.calls == 2);
        CHECK(fx.log.had_langs[0] && !fx.log.had_langs[1]);
        CHECK(fx.log.fallbacks[0].empty() && fx.log.fallbacks[1] == r.translator_unavailable);
        CHECK(fx.host.langs() == nullptr && !fx.host.pipeline().has_translator());
        CHECK((fx.loader->loads == Loads{{0, true}, {0, false}}));
        // The fallback host is a working Mode A host.
        CHECK(fx.host.loaded());
        fx.stream.make(fx.host);
        CHECK(fx.host.attach(*fx.stream.ctl, fx.err) && fx.stream.start());
        CHECK(fx.feed_hops(2));
    }
    {   // kIfPresent, warm-up fails with the translator: same fallback, seeder twice
        Fx fx;
        fx.loader->translator_files = true;
        fx.loader->translator.warm_up_fails = true;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(fx.load());
        const HostLoadReport& r = fx.host.load_report();
        CHECK(!r.translator && r.translator_fallback);
        CHECK(r.translator_unavailable.find("warm-up failed: ") == 0);
        CHECK(fx.log.calls == 2 && !fx.host.pipeline().has_translator());
    }
    {   // a "readiness" refusal stays a hard error under kIfPresent (no fallback)
        Fx fx;
        fx.loader->translator_files = true;
        fx.loader->translator.no_readiness = true;
        fx.log.adaptive = true;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kPipeline);
        CHECK(fx.err.message.find("adaptive_policy needs a translator with a readiness head (") == 0);
        CHECK(fx.log.calls == 1 && !fx.host.loaded());
    }
    {   // fewer than 2 languages: plain Mode A, not a failure, announced to the seeder
        Fx fx;
        fx.loader->translator_files = true;
        fx.loader->translator.lang_count = 1;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(fx.load());
        const HostLoadReport& r = fx.host.load_report();
        CHECK(!r.translator && !r.translator_fallback);
        CHECK(r.translator_unavailable ==
              "langs.json lists fewer than 2 languages; translation is unavailable");
        CHECK(fx.log.calls == 1 && fx.log.unavailable[0] == r.translator_unavailable);
        CHECK(fx.log.fallbacks[0].empty() && !fx.log.had_langs[0]);
        CHECK(fx.host.langs() == nullptr);
        CHECK((fx.loader->loads == Loads{{0, true}, {0, false}}));
    }
    {   // kRequired: s4 by default, and every translator problem is a hard error
        Fx fx;
        fx.req.translator = TranslatorPolicy::kRequired;
        fx.log.translate_on = true;
        CHECK(fx.load());
        CHECK((fx.loader->loads == Loads{{4, true}}));
        CHECK(fx.loader->has_translator_queries.empty());
        CHECK(fx.host.load_report().translator && fx.host.pipeline().mode() == PipelineMode::kS2UT);
        CHECK(fx.host.geometry().frames_per_step == 4 && fx.host.geometry().chunk_frames == 4);

        Fx fixed;  // an explicit frames_per_step wins over the s4 default
        fixed.req.translator = TranslatorPolicy::kRequired;
        fixed.req.frames_per_step = 2;
        fixed.load();  // the fake quartet is s4-only and ignores the request
        CHECK(fixed.loader->loads.size() == 1 && fixed.loader->loads[0].first == 2);

        Fx broken;
        broken.req.translator = TranslatorPolicy::kRequired;
        broken.loader->translator_load_error = "Failed to load translator_dec: broken";
        CHECK(!broken.load());
        CHECK(broken.err.kind == HostErrorKind::kBundle &&
              broken.err.message == "Failed to load translator_dec: broken");
        CHECK(broken.log.calls == 0 && broken.loader->loads.size() == 1);  // no fallback

        Fx ctor;
        ctor.req.translator = TranslatorPolicy::kRequired;
        ctor.loader->translator.stateful_vocoder_broken = true;
        CHECK(!ctor.load());
        CHECK(ctor.err.kind == HostErrorKind::kPipeline &&
              ctor.err.message.find("Cannot prepare Mode B: ") == 0);
        CHECK(ctor.log.calls == 1 && ctor.loader->loads.size() == 1);

        Fx warm;
        warm.req.translator = TranslatorPolicy::kRequired;
        warm.loader->translator.warm_up_fails = true;
        CHECK(!warm.load());
        CHECK(warm.err.kind == HostErrorKind::kWarmUp && warm.err.message.find("Warm-up failed: ") == 0);
        CHECK(warm.log.calls == 1);
    }
    {   // the ORT-less message is not a "translator problem": no fallback, own kind
        Fx fx;
        fx.loader->translator_files = true;
        fx.loader->translator_load_error = kBuiltWithoutOrtMessage;
        fx.req.translator = TranslatorPolicy::kIfPresent;
        CHECK(!fx.load());
        CHECK(fx.err.kind == HostErrorKind::kBuiltWithoutOrt);
        CHECK(fx.loader->loads.size() == 1 && fx.log.calls == 0);
    }
}

} // namespace

// ---- M4 Stage 06: stats() and the [stats] line ---------------------------------

// The strings were produced by the pre-Stage-06 `stats_line` of src/main.cpp
// (copied verbatim into a scratch program and run), not by format_stats_line,
// so a changed field order, separator or precision breaks here and in
// tools/analyze_soak.py. Binary-exact fractions keep the rounding unambiguous.
void test_stats_line_golden() {
    {   // diagnostic modes: the legacy counters only, negative queued, rss in KB
        HostStats s;
        s.frames = 480000; s.overruns = 1; s.underruns = 2; s.drops = 3;
        s.queued = -5; s.rss = 123456789;
        CHECK(format_stats_line(61.96, s) ==
              "[stats] t=62.0 frames=480000 overruns=1 underruns=2 drops=3 queued=-5 "
              "skipped=0 prefilled=0 rss_kb=120563");
        s.s2ut = false;   // translator fields never appear without s2ut
        s.tel.reads = 9;
        CHECK(format_stats_line(61.96, s).find("s2ut_") == std::string::npos);
    }
    HostStats vc;
    vc.vc = true;
    vc.frames = 160000; vc.underruns = 3; vc.queued = 880; vc.skipped = 160; vc.prefilled = 1040;
    vc.rss = 64u * 1024u * 1024u;
    vc.timing.p50_us = 1250; vc.timing.p95_us = 2400; vc.timing.max_us = 5100; vc.timing.misses = 2;
    vc.errors = 1;
    vc.lat_ms = 57.8125;   // printed with one decimal
    CHECK(format_stats_line(10.0, vc) ==
          "[stats] t=10.0 frames=160000 overruns=0 underruns=3 drops=0 queued=880 skipped=160 "
          "prefilled=1040 rss_kb=65536 inf_p50_us=1250 inf_p95_us=2400 inf_max_us=5100 misses=2 "
          "errors=1 lat_ms=57.8");
    HostStats b = vc;   // VC + S2UT: lag_ms is negative during a flush
    b.lat_ms = 133.4375;
    b.s2ut = true;
    b.tel.reads = 12; b.tel.writes = 30; b.tel.flushes = 4; b.tel.lag_chunks = -2;
    b.tel.chunk_ms = 80; b.tel.utterances = 2; b.tel.cap_ends = 1;
    CHECK(format_stats_line(3725.5625, b) ==
          "[stats] t=3725.6 frames=160000 overruns=0 underruns=3 drops=0 queued=880 skipped=160 "
          "prefilled=1040 rss_kb=65536 inf_p50_us=1250 inf_p95_us=2400 inf_max_us=5100 misses=2 "
          "errors=1 lat_ms=133.4 s2ut_r=12 s2ut_w=30 s2ut_f=4 lag_ms=-160.0 s2ut_utt=2 s2ut_cap=1");
}

void test_stats_unattached() {
    {   // nothing loaded: defaults, no throw, no device terms
        ModeAHost host(std::make_shared<FakeLoader>());
        const HostStats s = host.stats();
        CHECK(s.vc && s.frames == 0 && s.queued == 0 && s.occupancy == -1);
        CHECK(s.sample_rate == 0 && s.hop == 0 && s.governor_target == 0);
        CHECK(s.latency.total_ms == 0.0 && s.lat_ms == 0.0 && !s.s2ut);
    }
    Fx fx;   // loaded, not attached: the pipeline's own values, zero device terms
    CHECK(fx.load());
    const HostStats s = fx.host.stats();
    CHECK(s.vc && !s.s2ut && s.frames == 0 && s.queued == 0 && s.prefilled == 0);
    CHECK(s.occupancy == -1 && s.occupancy_tolerance == 0);
    CHECK(s.sample_rate == 16000 && s.hop == kHop);
    CHECK(s.algorithmic_samples == fx.host.geometry().algorithmic_samples());
    CHECK(s.governor_target == kTarget && s.governor_hard_cap == kTarget + 2400);
    CHECK(s.latency.capture_device_ms == 0.0 && s.latency.playback_device_ms == 0.0);
    CHECK(s.latency.queued_ms == 0.0);
    CHECK(s.latency.model_delay_ms ==
          1000.0 * static_cast<double>(fx.host.geometry().model_delay_samples()) / 16000.0);
    CHECK(s.lat_ms == s.latency.total_ms && s.latency.total_ms == s.latency.model_delay_ms);
    for (const TimingSnapshot& t : s.stages) CHECK(t.samples == 0);   // warm-up telemetry was cleared
}

void test_stats_live_mode_a() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.feed_hops(20));
    CHECK(fx.settle());
    const WorkerSnapshot w = fx.host.snapshot();
    const HostStats s = fx.host.stats();
    CHECK(s.vc && !s.s2ut);
    // Counters come from the worker snapshot; frames from the attached router.
    CHECK(s.overruns == w.overruns && s.underruns == w.underruns && s.drops == w.drops);
    CHECK(s.skipped == w.skipped && s.prefilled == w.prefilled && s.queued == w.queued);
    CHECK(s.frames == fx.stream.ctl->router().frames_processed() && s.frames >= 20 * kHop);
    CHECK(s.queued == identity_of(w) && s.queued == static_cast<std::int64_t>(kTarget));
    CHECK(s.rss > 0);
    // Whole-hop and per-stage timing (stages always, with or without a translator).
    CHECK(s.timing.samples >= 20 && s.errors == 0);
    CHECK(s.stages[static_cast<std::size_t>(PipelineStage::kEncoder)].samples == s.timing.samples);
    CHECK(s.stages[static_cast<std::size_t>(PipelineStage::kVocoder)].samples == s.timing.samples);
    CHECK(s.stages[static_cast<std::size_t>(PipelineStage::kFeatures)].samples == s.timing.samples);
    CHECK(s.stages[static_cast<std::size_t>(PipelineStage::kOutputChain)].samples == s.timing.samples);
    CHECK(s.stages[static_cast<std::size_t>(PipelineStage::kTranslatorDec)].samples == 0);
    // Geometry and governor.
    const PipelineGeometry& g = fx.host.geometry();
    CHECK(s.sample_rate == 16000 && s.hop == g.hop && s.algorithmic_samples == g.algorithmic_samples());
    CHECK(s.governor_target == kTarget && s.governor_hard_cap == kTarget + 2400);
    // Latency: the fake backend opens 480-frame periods at 48 kHz (10 ms each).
    CHECK(s.latency.capture_device_ms == 10.0 && s.latency.playback_device_ms == 10.0);
    CHECK(s.latency.queued_ms == 1000.0 * static_cast<double>(kTarget) / 16000.0);
    CHECK(s.latency.model_delay_ms == 1000.0 * static_cast<double>(g.model_delay_samples()) / 16000.0);
    const double expect = 1000.0 * (static_cast<double>(g.model_delay_samples()) +
                                    static_cast<double>(kTarget)) / 16000.0 + 10.0 + 10.0;
    CHECK(s.lat_ms == s.latency.total_ms && std::abs(s.lat_ms - expect) < 1e-9);
    // Identity check: at quiescence the measured occupancy is exactly `queued`.
    // Tolerance = one device period at the client rate (480 @ 48 kHz -> 160) + one hop.
    CHECK(s.occupancy == s.queued);
    CHECK(s.occupancy_tolerance == 160 + static_cast<std::int64_t>(kHop));
    // A governor trim keeps the identity, with skipped in it.
    fx.stream.callback(8 * kHop, 0.0);
    CHECK(fx.settle());
    const HostStats t = fx.host.stats();
    CHECK(t.skipped > 0 && t.occupancy == t.queued);
}

// What the worker holds between its input read and its output write is part of
// the occupancy: a partial hop sits in the ChunkAssembler, in neither queue.
void test_stats_occupancy_counts_held_samples() {
    Fx fx;
    CHECK(fx.run());
    CHECK(fx.feed_hops(4));
    CHECK(fx.settle());
    fx.stream.callback(100, 0.3);   // 100 of the next 320 samples: popped from out_q, read, not processed
    CHECK(wait_until([&] { return fx.host.snapshot().held == 100; }));
    const HostStats s = fx.host.stats();
    const std::int64_t queues =
        static_cast<std::int64_t>(fx.stream.ctl->router().input_queued_approx() +
                                  fx.stream.ctl->router().output_queued_approx());
    CHECK(s.queued == static_cast<std::int64_t>(kTarget));
    CHECK(queues == s.queued - 100);        // the queues alone fall short by exactly the held samples
    CHECK(s.occupancy == s.queued);         // with `held` the identity closes
    // Completing the hop releases them.
    fx.stream.callback(kHop - 100, 0.3);
    CHECK(fx.settle());
    CHECK(fx.host.snapshot().held == 0);
    CHECK(fx.host.stats().occupancy == fx.host.stats().queued);
}

void test_stats_live_mode_b() {
    Fx fx;
    fx.loader->translator_files = true;
    fx.req.translator = TranslatorPolicy::kIfPresent;
    fx.log.translate_on = true;   // start on the translator route
    CHECK(fx.load());
    fx.stream.make(fx.host);
    CHECK(fx.host.attach(*fx.stream.ctl, fx.err) && fx.stream.start());
    const std::size_t hop = fx.host.geometry().hop;
    CHECK(hop == 1280);   // fps 4
    for (std::uint64_t i = 1; i <= 6; ++i) {
        fx.stream.callback(hop, 0.3);
        CHECK(wait_until([&] { return fx.host.timing().samples >= i; }));
    }
    CHECK(fx.settle());
    const HostStats s = fx.host.stats();
    CHECK(s.vc && s.s2ut && s.tel.chunk_ms == 80);
    CHECK(s.tel.reads + s.tel.writes + s.tel.flushes == 6);    // one chunk step per hop at fps 4
    CHECK(s.tel.reads == fx.host.pipeline().s2ut_telemetry().reads);
    CHECK(s.hop == hop && s.timing.samples == 6);
    for (PipelineStage st : {PipelineStage::kEncoder, PipelineStage::kFeatures,
                             PipelineStage::kOutputChain})
        CHECK(s.stages[static_cast<std::size_t>(st)].samples == 6);
    CHECK(s.stages[static_cast<std::size_t>(PipelineStage::kTranslatorEnc)].samples > 0);
    // The governor entries are the effective ones: 2080, or 2080 + 3840 while translating.
    CHECK(s.governor_target == fx.host.pipeline().governor_target_samples());
    CHECK(s.governor_hard_cap == s.governor_target + 2400);
    CHECK(s.governor_target == 2080 || s.governor_target == 2080 + 3840);
    CHECK(s.occupancy == s.queued);
}

int main() {
    test_start_and_conservation();
    test_worker_start_callback_and_priority();
    test_configure_and_rate_check();
    test_params_accepted_vs_rejected();
    test_voice_lookup();
    test_bundle_swap_quiesces_before_models_die();
    test_destroy_while_running_joins_worker();
    test_device_loss_and_recovery();
    test_fatal_pipeline_error();
    test_steady_state_allocates_nothing();
    test_mute_through_host();
    test_load_errors_leave_host_unloaded();
    test_translator_policy_and_fallback();
    test_stats_line_golden();
    test_stats_unattached();
    test_stats_live_mode_a();
    test_stats_occupancy_counts_held_samples();
    test_stats_live_mode_b();
    return test::finish("mode_a_host");
}
