// ModeAHost — the shared live composition of the neural pipeline
// (Milestone 4 / Stage 05; see ModeAHost.hpp for the contract).
//
// This file has no ONNX Runtime dependency: the models arrive through
// HostModelLoader (OrtBundleLoader.cpp ships the ORT one, tests ship fakes).
// The only ORT-side name used is the inline kBuiltWithoutOrtMessage constant.
#include "pipeline/ModeAHost.hpp"

#include "audio/DuplexRouter.hpp"
#include "pipeline/ModelContract.hpp"
#include "pipeline/OrtModel.hpp"  // kBuiltWithoutOrtMessage (header-only constant)
#include "util/ProcessMemory.hpp"
#include "util/WorkerPriority.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <span>
#include <stdexcept>
#include <utility>

namespace fluxvoice {

namespace {

constexpr const char* kQueueCapacityMessage =
    "Latency target plus one hop exceeds the audio queue capacity";
constexpr const char* kFewLanguagesMessage =
    "langs.json lists fewer than 2 languages; translation is unavailable";
constexpr const char* kLoaderMissingMessage = "no model loader configured";
constexpr const char* kNoTranslatorPairMessage = "loader returned no translator models";

bool is_built_without_ort(std::string_view text) noexcept {
    return text.find(kBuiltWithoutOrtMessage) != std::string_view::npos;
}

std::vector<PipelineVoice> make_pipeline_voices(const std::vector<VoiceEntry>& voices) {
    std::vector<PipelineVoice> out;
    out.reserve(voices.size());
    for (const VoiceEntry& v : voices)
        out.push_back({v.stats, std::span<const float>(v.embedding.data(), v.embedding.size())});
    return out;
}

// One composition attempt. Member order is destruction order in reverse: the
// pipeline (which holds references into the models) is always destroyed first.
struct Attempt {
    std::unique_ptr<HostModels> models;
    std::unique_ptr<InferencePipeline> pipeline;
    HostLoadReport report{};
    HostError error{};
    std::string fallback_reason;  // set with Outcome::kFallback
};

enum class Outcome { kOk, kFailed, kFallback };

Outcome fail_with(Attempt& a, HostErrorKind kind, std::string message) {
    a.error.kind = kind;
    a.error.message = std::move(message);
    return Outcome::kFailed;
}

/// A translator path that cannot be used: a hard error when the translator is
/// required (CLI --s2ut) or when the build has no ORT at all, otherwise a
/// request to retry as plain Mode A (CLI --vc; "rc 2" of the old main.cpp).
Outcome translator_unusable(Attempt& a, bool required, std::string reason) {
    if (is_built_without_ort(reason))
        return fail_with(a, HostErrorKind::kBuiltWithoutOrt, std::move(reason));
    if (required) return fail_with(a, HostErrorKind::kBundle, std::move(reason));
    a.fallback_reason = std::move(reason);
    return Outcome::kFallback;
}

/// Models -> seeder -> InferencePipeline -> warm_up -> queue-capacity check.
/// `with_translator` selects the Mode B constructor; `fallback_in` is the
/// reason of the previous (translator) attempt, "" on the first one.
Outcome run_attempt(const HostModelLoader& loader, const HostLoadRequest& req,
                    const HostParamSeeder& seed, int bundle_fps, bool with_translator,
                    bool required, std::string_view fallback_in, Attempt& a) {
    enum class Phase { kBundle, kSeed, kPipeline, kWarmUp } phase = Phase::kBundle;
    try {
        std::string error;
        std::string_view unavailable;
        if (with_translator) {
            a.models = loader.load(req.model_dir, bundle_fps, true, error);
            if (!a.models) return translator_unusable(a, required, std::move(error));
            if (!a.models->translator_enc() || !a.models->translator_dec() || !a.models->langs())
                return translator_unusable(a, required, kNoTranslatorPairMessage);
            if (!required && a.models->langs()->langs.size() < 2) {
                // One language cannot be translated: plain Mode A bundle. Not a
                // failure; the caller decides how to tell the user.
                a.models.reset();
                a.report.translator_unavailable = kFewLanguagesMessage;
                unavailable = a.report.translator_unavailable;
                with_translator = false;
            }
        }
        if (!with_translator) {
            a.models = loader.load(req.model_dir, req.frames_per_step, false, error);
            if (!a.models) {
                const HostErrorKind kind = is_built_without_ort(error) ? HostErrorKind::kBuiltWithoutOrt
                                                                       : HostErrorKind::kBundle;
                return fail_with(a, kind, std::move(error));
            }
        }

        const std::vector<VoiceEntry>& voices = a.models->voices();
        ModelStep& encoder = a.models->encoder();
        ModelStep& vocoder = a.models->vocoder();
        const auto contract_rate = static_cast<std::uint32_t>(encoder.contract().sample_rate);

        phase = Phase::kSeed;
        HostBundleInfo info;
        info.voices = &voices;
        info.langs = with_translator ? a.models->langs() : nullptr;
        info.contract_rate = contract_rate;
        info.translator_fallback = fallback_in;
        info.translator_unavailable = unavailable;
        VoiceParams params;
        S2utOptions s2ut;
        std::string seed_error;
        if (seed && !seed(info, params, s2ut, seed_error))
            return fail_with(a, HostErrorKind::kSeed, std::move(seed_error));

        phase = Phase::kPipeline;
        const std::vector<PipelineVoice> pipeline_voices = make_pipeline_voices(voices);
        PipelineOptions options;
        options.latency_target_samples = std::size_t{contract_rate} * req.latency_target_ms / 1000;
        options.params = params;
        if (with_translator) {
            try {
                a.pipeline = std::make_unique<InferencePipeline>(
                    encoder, *a.models->translator_enc(), *a.models->translator_dec(), vocoder,
                    pipeline_voices, s2ut, options);
            } catch (const std::invalid_argument& e) {
                if (std::string_view(e.what()).find("readiness") != std::string_view::npos)
                    return fail_with(
                        a, HostErrorKind::kPipeline,
                        std::string("adaptive_policy needs a translator with a readiness head (") +
                            e.what() + ")");
                if (required) throw;  // reported below as "Cannot prepare Mode B"
                a.fallback_reason = e.what();
                return Outcome::kFallback;
            }
        } else {
            a.pipeline = std::make_unique<InferencePipeline>(encoder, vocoder, pipeline_voices, options);
        }

        phase = Phase::kWarmUp;
        if (!a.pipeline->warm_up(req.warm_up_steps)) {
            if (with_translator && !required) {
                a.fallback_reason = "warm-up failed: " + std::string(a.pipeline->last_error());
                return Outcome::kFallback;
            }
            return fail_with(a, HostErrorKind::kWarmUp,
                             "Warm-up failed: " + std::string(a.pipeline->last_error()));
        }
        if (a.pipeline->governor().target_samples() + a.pipeline->geometry().hop >
            DuplexRouter::kQueueCapacity)
            return fail_with(a, HostErrorKind::kQueueCapacity, kQueueCapacityMessage);
        a.report.translator = a.pipeline->has_translator();
        return Outcome::kOk;
    } catch (const std::exception& e) {
        const HostErrorKind kind = phase == Phase::kBundle     ? HostErrorKind::kBundle
                                   : phase == Phase::kSeed     ? HostErrorKind::kSeed
                                   : phase == Phase::kPipeline ? HostErrorKind::kPipeline
                                                               : HostErrorKind::kWarmUp;
        return fail_with(a, kind,
                         std::string("Cannot prepare ") + (required ? "Mode B" : "Mode A") + ": " +
                             e.what());
    } catch (...) {
        return fail_with(a, HostErrorKind::kPipeline,
                         std::string("Cannot prepare ") + (required ? "Mode B" : "Mode A") +
                             ": unknown exception");
    }
}

const std::vector<VoiceEntry> kNoVoices;
const PipelineGeometry kNoGeometry{};

} // namespace

// ---- Construction / load ----------------------------------------------------

ModeAHost::ModeAHost(std::shared_ptr<const HostModelLoader> loader) : loader_(std::move(loader)) {}

ModeAHost::~ModeAHost() {
    // The controller must still be alive here (or already detached): the hooks
    // capture `this` and are cleared by detach().
    if (controller_) detach();
    stop_worker();
}

bool ModeAHost::load(const HostLoadRequest& request, const HostParamSeeder& seed, HostError& error) {
    error = HostError{};
    if (controller_) {
        error = {HostErrorKind::kState, "load() called while attached to an audio controller"};
        return false;
    }
    try {
        // Unload first: the pipeline references the models, so it goes first.
        // A failed load leaves the host unloaded.
        stop_worker();
        pipeline_.reset();
        models_.reset();
        report_ = HostLoadReport{};
        // Snapshots queued for the previous bundle must not reach this one
        // (voice indices differ). SPSC roles may change: no worker is running.
        {
            VoiceParams stale;
            params_.flush();
            while (params_.drain(stale)) {}
        }
        if (!loader_) {
            error = {HostErrorKind::kBundle, kLoaderMissingMessage};
            return false;
        }

        // Mode B runs at s4 by default. Mode A keeps the smallest available
        // variant (the caller's frames_per_step, 0 = bundle default) and still
        // loads the translator models when the bundle has them (kIfPresent),
        // at that same hop, so the console can switch to Mode B live.
        const bool required = request.translator == TranslatorPolicy::kRequired;
        const int bundle_fps = request.frames_per_step != 0 ? request.frames_per_step
                                                            : (required ? 4 : 0);
        const bool with_translator =
            required || (request.translator == TranslatorPolicy::kIfPresent &&
                         loader_->has_translator(request.model_dir, bundle_fps));

        Attempt first;
        Outcome outcome =
            run_attempt(*loader_, request, seed, bundle_fps, with_translator, required, {}, first);
        Attempt second;
        Attempt* last = &first;
        if (outcome == Outcome::kFallback) {
            // Mode A must not depend on a valid Mode B quartet: retry with two models.
            const std::string reason = first.fallback_reason;
            first.pipeline.reset();
            first.models.reset();
            outcome = run_attempt(*loader_, request, seed, bundle_fps, false, required, reason, second);
            second.report.translator_unavailable = reason;
            second.report.translator_fallback = true;
            last = &second;
        }
        if (outcome != Outcome::kOk) {
            error = std::move(last->error);
            return false;
        }
        models_ = std::move(last->models);
        pipeline_ = std::move(last->pipeline);
        report_ = std::move(last->report);
        return true;
    } catch (const std::exception& e) {
        pipeline_.reset();
        models_.reset();
        report_ = HostLoadReport{};
        error = {HostErrorKind::kPipeline, std::string("Cannot prepare Mode A: ") + e.what()};
        return false;
    } catch (...) {
        pipeline_.reset();
        models_.reset();
        report_ = HostLoadReport{};
        error = {HostErrorKind::kPipeline, "Cannot prepare Mode A: unknown exception"};
        return false;
    }
}

bool ModeAHost::loaded() const noexcept { return pipeline_ != nullptr; }
const HostLoadReport& ModeAHost::load_report() const noexcept { return report_; }

const std::vector<VoiceEntry>& ModeAHost::voices() const noexcept {
    return models_ ? models_->voices() : kNoVoices;
}

std::optional<std::uint16_t> ModeAHost::voice_index(std::string_view name, std::string& error) const {
    if (!models_) {
        error = "no model bundle loaded";
        return std::nullopt;
    }
    const std::vector<VoiceEntry>& list = models_->voices();
    const VoiceEntry* v = find_voice(list, name, error);
    if (!v) return std::nullopt;
    return static_cast<std::uint16_t>(v - list.data());
}

const LangCatalog* ModeAHost::langs() const noexcept {
    return models_ && pipeline_ && pipeline_->has_translator() ? models_->langs() : nullptr;
}

std::uint32_t ModeAHost::contract_rate() const noexcept {
    return pipeline_ ? pipeline_->geometry().sample_rate : 0;
}

const PipelineGeometry& ModeAHost::geometry() const {
    return pipeline_ ? pipeline_->geometry() : kNoGeometry;
}

std::size_t ModeAHost::governor_target_samples() const noexcept {
    return pipeline_ ? pipeline_->governor().target_samples() : 0;
}

const InferencePipeline& ModeAHost::pipeline() const {
    // An unloaded host hands out the legacy shell: every any-thread accessor
    // answers with zeros instead of dereferencing nothing.
    static const InferencePipeline kUnloaded;
    return pipeline_ ? *pipeline_ : kUnloaded;
}

std::int64_t ModeAHost::onset_latency_samples(int wait_k) const {
    if (!models_ || !models_->translator_enc() || !models_->translator_dec()) return 0;
    return s2ut_onset_latency_samples(models_->encoder().contract(),
                                      models_->translator_enc()->contract(),
                                      models_->translator_dec()->contract(),
                                      models_->vocoder().contract(), wait_k);
}

// ---- Stream wiring ----------------------------------------------------------

void ModeAHost::configure(ControllerConfig& cfg) const {
    if (!loaded()) return;
    cfg.sample_rate = contract_rate();
    cfg.fade_underruns = true;
    cfg.max_queued_samples = 0;  // the governor owns intentional skips
}

bool ModeAHost::attach(AudioController& controller, HostError& error,
                       HostWorkerStartCallback on_worker_start) {
    error = HostError{};
    if (!loaded()) {
        error = {HostErrorKind::kState, "attach() before a successful load()"};
        return false;
    }
    if (controller_) {
        error = {HostErrorKind::kState, "attach() while already attached"};
        return false;
    }
    try {
        worker_ = std::make_unique<ModeAWorker>(controller.router(), *pipeline_, &params_);
    } catch (const std::invalid_argument& e) {
        error = {HostErrorKind::kQueueCapacity, e.what()};
        return false;
    } catch (const std::exception& e) {
        error = {HostErrorKind::kPipeline, e.what()};
        return false;
    }
    worker_->set_muted(muted_.load(std::memory_order_relaxed));
    controller_ = &controller;
    on_worker_start_ = std::move(on_worker_start);
    const std::uint32_t rate = contract_rate();
    controller.set_prepare_hook([this, &controller, rate] {
        return controller.stream_info().client_sample_rate == rate && worker_ && worker_->prepare();
    });
    controller.set_worker_hooks([this] { stop_worker(); }, [this] { reset_worker(); },
                                [this] { start_worker(); });
    return true;
}

void ModeAHost::detach() {
    stop_worker();
    if (controller_) {
        controller_->set_prepare_hook({});
        controller_->set_worker_hooks({}, {}, {});
        controller_ = nullptr;
    }
    if (worker_) {
        const auto accepted = worker_->accepted_generation();
        if (accepted > retired_accepted_.load(std::memory_order_relaxed))
            retired_accepted_.store(accepted, std::memory_order_relaxed);
        retired_rejected_.fetch_add(worker_->rejected_params(), std::memory_order_relaxed);
        worker_.reset();
    }
    on_worker_start_ = {};
}

bool ModeAHost::attached() const noexcept { return controller_ != nullptr; }

void ModeAHost::stop_worker() {
    run_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
}

void ModeAHost::reset_worker() {
    if (worker_) worker_->reset();
}

void ModeAHost::start_worker() {
    if (!worker_ || !controller_ || thread_.joinable()) return;
    run_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] {
        const WorkerPriority priority;
        priority_method_.store(priority.method(), std::memory_order_relaxed);
        priority_elevated_.store(priority.elevated(), std::memory_order_relaxed);
        if (on_worker_start_) on_worker_start_(priority.method(), priority.elevated(), priority.result());
        while (run_.load(std::memory_order_relaxed)) {
            if (!worker_->pump()) break;
            if (controller_->router().input_queued_approx() == 0)
                std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });
}

// ---- Controls ---------------------------------------------------------------

std::uint64_t ModeAHost::push_params(const VoiceParams& params) { return params_.push(params); }
bool ModeAHost::flush_params() { return params_.flush(); }
std::uint64_t ModeAHost::pushed_generation() const noexcept { return params_.pushed_generation(); }
std::uint64_t ModeAHost::applied_generation() const noexcept { return params_.applied_generation(); }

std::uint64_t ModeAHost::accepted_generation() const noexcept {
    const std::uint64_t retired = retired_accepted_.load(std::memory_order_relaxed);
    const std::uint64_t live = worker_ ? worker_->accepted_generation() : 0;
    return live > retired ? live : retired;
}

std::uint64_t ModeAHost::rejected_params() const noexcept {
    return retired_rejected_.load(std::memory_order_relaxed) +
           (worker_ ? worker_->rejected_params() : 0);
}

void ModeAHost::set_muted(bool muted) noexcept {
    muted_.store(muted, std::memory_order_relaxed);
    if (worker_) worker_->set_muted(muted);
}

bool ModeAHost::muted() const noexcept { return muted_.load(std::memory_order_relaxed); }

// ---- Telemetry --------------------------------------------------------------

WorkerSnapshot ModeAHost::snapshot() const noexcept {
    return worker_ ? worker_->snapshot() : WorkerSnapshot{};
}

WorkerMeters ModeAHost::meters() const noexcept {
    return worker_ ? worker_->meters() : WorkerMeters{};
}

TimingSnapshot ModeAHost::timing() const noexcept {
    return pipeline_ ? pipeline_->timing() : TimingSnapshot{};
}

HostStats ModeAHost::stats() const {
    HostStats s;
    s.vc = true;
    const WorkerSnapshot w = snapshot();
    s.overruns = w.overruns;
    s.underruns = w.underruns;
    s.drops = w.drops;
    s.skipped = w.skipped;
    s.prefilled = w.prefilled;
    s.queued = w.queued;
    s.rss = resident_bytes();
    if (pipeline_) {
        const PipelineGeometry& g = pipeline_->geometry();
        s.timing = pipeline_->timing();
        s.errors = pipeline_->errors();
        s.stages = pipeline_->stage_timings();
        if (pipeline_->has_translator()) {
            s.s2ut = true;
            s.tel = pipeline_->s2ut_telemetry();
        }
        s.sample_rate = g.sample_rate;
        s.hop = g.hop;
        s.algorithmic_samples = g.algorithmic_samples();
        s.governor_target = pipeline_->governor_target_samples();
        s.governor_hard_cap = pipeline_->governor_hard_cap_samples();
    }
    StreamInfo stream;  // zero device terms until a controller is attached
    if (controller_) {
        const DuplexRouter& router = controller_->router();
        stream = controller_->stream_info();
        s.frames = router.frames_processed();
        // Identity check: what the queues and the worker hold right now. The
        // counters behind `queued` and these sizes are read at slightly
        // different instants, hence the tolerance of one device period plus one hop.
        s.occupancy = static_cast<std::int64_t>(router.input_queued_approx()) +
                      static_cast<std::int64_t>(router.output_queued_approx()) +
                      static_cast<std::int64_t>(w.held);
        const std::uint32_t client_rate =
            stream.client_sample_rate ? stream.client_sample_rate : s.sample_rate;
        const auto period_samples = [client_rate](std::uint32_t frames, std::uint32_t rate) {
            return rate ? (std::uint64_t{frames} * client_rate + rate - 1) / rate : std::uint64_t{0};
        };
        const std::uint64_t device_period =
            std::max(period_samples(stream.capture_internal_period, stream.capture_internal_rate),
                     period_samples(stream.playback_internal_period, stream.playback_internal_rate));
        s.occupancy_tolerance = static_cast<std::int64_t>(device_period + s.hop);
    }
    s.latency = estimate_latency(geometry(), s.queued, stream);
    s.lat_ms = s.latency.total_ms;
    return s;
}

bool ModeAHost::failed() const noexcept { return worker_ && worker_->failed(); }

std::string ModeAHost::error() const {
    return worker_ ? std::string(worker_->error()) : std::string{};
}

const char* ModeAHost::priority_method() const noexcept {
    return priority_method_.load(std::memory_order_relaxed);
}

bool ModeAHost::priority_elevated() const noexcept {
    return priority_elevated_.load(std::memory_order_relaxed);
}

} // namespace fluxvoice
