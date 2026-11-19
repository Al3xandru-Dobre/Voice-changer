// ModeAHost — the one live composition of the neural pipeline, shared by the
// CLI (`fluxvoice --vc` / `--s2ut`) and the GUI (frontend/core/EngineHost)
// (Milestone 4 / Stage 05). It owns, in this order, the models, the
// InferencePipeline, the ParamChannel and the ModeAWorker, so destruction runs
// worker -> channel -> pipeline -> models.
//
// Ordering rules it enforces (easy to get wrong in a second copy):
//   * warm_up() runs inside load(), before any stream exists;
//   * the prepare hook runs after the controller drained both queues and
//     before callbacks start (contract rate check, then ModeAWorker::prepare());
//   * SPSC roles change only while the worker is quiesced (controller hooks);
//   * the caller reads its final snapshot() before stopping the controller.
//
// Despite the name, a host may own a translator quartet (S2utBundle): the
// worker loop is the same ModeAWorker for both routes, and the CLI's `--vc`
// already loads translator models when the bundle has them. The GUI requests
// TranslatorPolicy::kNever until M4 Stage 07.
//
// Threads:
//   * load(): control thread or a short-lived loader thread, never while the
//     host is attached to a controller;
//   * attach() / detach() / push_params() / flush_params() / set_muted() /
//     error(): control thread;
//   * generations, snapshot(), meters(), timing(), failed(), priority_*():
//     any thread (relaxed atomics; values may lag the worker by a hop).
#pragma once

#include "audio/AudioController.hpp"
#include "pipeline/HostStats.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/LangCatalog.hpp"
#include "pipeline/ModeAWorker.hpp"
#include "pipeline/ModelStep.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "pipeline/TimingHistogram.hpp"
#include "pipeline/VoiceCatalog.hpp"
#include "voice/ParamChannel.hpp"
#include "voice/VoiceParams.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fluxvoice {

// ---- Model source seam ------------------------------------------------------

/// The models one host runs. The ORT loader wraps ModelBundle / S2utBundle;
/// tests return FakeModelStep sets. Translator models come as a pair or not
/// at all. Every reference stays valid for the object's lifetime.
class HostModels {
public:
    virtual ~HostModels() = default;
    virtual ModelStep& encoder() noexcept = 0;
    virtual ModelStep& vocoder() noexcept = 0;
    virtual ModelStep* translator_enc() noexcept { return nullptr; }
    virtual ModelStep* translator_dec() noexcept { return nullptr; }
    virtual const std::vector<VoiceEntry>& voices() const noexcept = 0;
    /// Non-null exactly when the translator pair is present.
    virtual const LangCatalog* langs() const noexcept { return nullptr; }
};

class HostModelLoader {
public:
    virtual ~HostModelLoader() = default;
    /// True when `dir` holds a translator quartet for `frames_per_step`
    /// (file layout only; nothing is loaded).
    virtual bool has_translator(const std::filesystem::path& dir, int frames_per_step) const = 0;
    /// Encoder + vocoder, plus the translator pair when `with_translator`.
    /// nullptr and `error` on failure; never throws.
    virtual std::unique_ptr<HostModels> load(const std::filesystem::path& dir, int frames_per_step,
                                             bool with_translator, std::string& error) const = 0;
};

/// The shipped loader (ModelBundle / S2utBundle over OrtModel). In a build
/// without ONNX Runtime every load fails with kBuiltWithoutOrtMessage.
std::shared_ptr<const HostModelLoader> ort_bundle_loader();

// ---- Load request and results ----------------------------------------------

enum class TranslatorPolicy {
    kNever,      // encoder + vocoder only (GUI until M4 Stage 07)
    kIfPresent,  // load the quartet when the bundle has one; fall back to Mode A (CLI --vc)
    kRequired,   // the quartet or an error (CLI --s2ut)
};

struct HostLoadRequest {
    std::filesystem::path model_dir;
    int frames_per_step = 0;              // 0 = the bundle's default variant
    std::uint32_t latency_target_ms = 0;  // 0 = master-budget default (LatencyGovernor)
    TranslatorPolicy translator = TranslatorPolicy::kNever;
    std::size_t warm_up_steps = 20;
};

enum class HostErrorKind {
    kNone = 0,
    kBuiltWithoutOrt,  // ORT-less binary
    kBundle,           // locate / load / validate failed
    kSeed,             // the seeder refused (unknown voice, bad rate, ...)
    kPipeline,         // InferencePipeline construction refused the models
    kWarmUp,           // warm-up failed; the stream must not start
    kQueueCapacity,    // latency target + one hop exceeds the audio queue
    kState,            // wrong call order (not loaded, still attached, ...)
};

struct HostError {
    HostErrorKind kind = HostErrorKind::kNone;
    /// Final, user-facing text (the CLI prints it as is). Empty only for kSeed
    /// when the seeder already reported the problem itself.
    std::string message;
    bool ok() const noexcept { return kind == HostErrorKind::kNone; }
};

/// What load() knows once the models are in, before the pipeline exists.
struct HostBundleInfo {
    const std::vector<VoiceEntry>* voices = nullptr;
    const LangCatalog* langs = nullptr;   // non-null: translator models loaded
    std::uint32_t contract_rate = 0;
    /// Non-empty on the Mode A retry after an unusable translator path
    /// (kIfPresent only); the caller decides how to report it.
    std::string_view translator_fallback;
    /// Non-empty when a kIfPresent load found a translator quartet but chose
    /// plain Mode A on purpose (fewer than 2 languages); not a failure, so
    /// translator_fallback stays empty. The seeder may print a note.
    std::string_view translator_unavailable;
};

/// Fills the initial VoiceParams (voice indices, controls) and, for a
/// translator load, the S2utOptions. Returning false aborts the load with
/// kSeed and `error` (may be empty when the seeder already reported).
using HostParamSeeder =
    std::function<bool(const HostBundleInfo& info, VoiceParams& params, S2utOptions& s2ut, std::string& error)>;

struct HostLoadReport {
    bool translator = false;             // translator models loaded and usable (Mode B capable)
    std::string translator_unavailable;  // why a kIfPresent request ended as Mode A ("" otherwise)
    bool translator_fallback = false;    // true: the translator path failed; false: e.g. < 2 languages
};

/// Called on the worker thread once per worker start, right after the
/// scheduling request (src/util/WorkerPriority.hpp). Must not block.
using HostWorkerStartCallback = std::function<void(const char* method, bool elevated, int result)>;

// ---- The host ---------------------------------------------------------------

class ModeAHost {
public:
    explicit ModeAHost(std::shared_ptr<const HostModelLoader> loader = ort_bundle_loader());
    ~ModeAHost();  // detach() if attached, then worker -> channel -> pipeline -> models

    ModeAHost(const ModeAHost&) = delete;
    ModeAHost& operator=(const ModeAHost&) = delete;

    // -- Load (never while attached) ----------------------------------------
    /// Models -> seeder -> InferencePipeline -> warm_up -> queue-capacity check.
    /// On failure the host is left unloaded and `error` says why; nothing throws.
    bool load(const HostLoadRequest& request, const HostParamSeeder& seed, HostError& error);
    bool loaded() const noexcept;
    const HostLoadReport& load_report() const noexcept;

    // Valid after a successful load().
    const std::vector<VoiceEntry>& voices() const noexcept;
    /// Index of the voice called `name` (panels submit names, never rows);
    /// nullopt and an error naming the available voices otherwise.
    std::optional<std::uint16_t> voice_index(std::string_view name, std::string& error) const;
    const LangCatalog* langs() const noexcept;  // non-null when translator models are loaded
    std::uint32_t contract_rate() const noexcept;
    const PipelineGeometry& geometry() const;
    std::size_t governor_target_samples() const noexcept;
    /// Any-thread accessors only (timing(), stage_timing(), mode(),
    /// has_translator(), s2ut_telemetry(), errors(), geometry()).
    const InferencePipeline& pipeline() const;
    /// Algorithmic onset latency of the Mode B route at `wait_k` (CLI start
    /// banner); 0 when no translator models are loaded.
    std::int64_t onset_latency_samples(int wait_k) const;

    // -- Stream wiring (control thread, controller Idle) ---------------------
    /// Contract rate, fade_underruns = true, max_queued_samples = 0 (the
    /// governor owns intentional skips).
    void configure(ControllerConfig& cfg) const;
    /// Creates the ModeAWorker, installs the prepare hook (client rate ==
    /// contract rate, then prepare()) and the quiesce / reset / resume hooks;
    /// the worker thread takes WorkerPriority. kQueueCapacity / kState on failure.
    bool attach(AudioController& controller, HostError& error,
                HostWorkerStartCallback on_worker_start = {});
    /// Controller must be stopped. Joins the worker and clears the hooks.
    void detach();
    bool attached() const noexcept;

    // -- Controls (control thread) -------------------------------------------
    /// Full snapshot; returns its generation. Newest wins (ParamChannel).
    std::uint64_t push_params(const VoiceParams& params);
    /// Retry a parked snapshot (call each control tick). True when none is left.
    bool flush_params();
    std::uint64_t pushed_generation() const noexcept;
    /// Drained by the worker (delivery).
    std::uint64_t applied_generation() const noexcept;
    /// Accepted by InferencePipeline::set_params() (what "applied" means in a UI).
    std::uint64_t accepted_generation() const noexcept;
    /// Snapshots the pipeline refused (previous parameters kept).
    std::uint64_t rejected_params() const noexcept;
    /// Ramped output mute inside the pipeline's output chain.
    void set_muted(bool muted) noexcept;
    bool muted() const noexcept;

    // -- Telemetry (any thread) ----------------------------------------------
    WorkerSnapshot snapshot() const noexcept;
    WorkerMeters meters() const noexcept;
    TimingSnapshot timing() const noexcept;
    /// One `[stats]` snapshot of the attached stream (control thread; M4
    /// Stage 06): worker counters, whole-hop and per-stage timing, the
    /// latency estimate, governor target and hard cap. `vc` is true; an
    /// unattached host returns the counters it has and zero device terms.
    HostStats stats() const;
    bool failed() const noexcept;
    /// Control thread, after failed(): the pipeline's stable last error.
    std::string error() const;
    const char* priority_method() const noexcept;
    bool priority_elevated() const noexcept;

private:
    void stop_worker();
    void reset_worker();
    void start_worker();

    std::shared_ptr<const HostModelLoader> loader_;
    HostLoadReport report_{};
    // Declaration order is destruction order in reverse: the worker goes
    // first, the models last (InferencePipeline holds references to them).
    std::unique_ptr<HostModels> models_;
    std::unique_ptr<InferencePipeline> pipeline_;
    ParamChannel params_;
    std::unique_ptr<ModeAWorker> worker_;
    AudioController* controller_ = nullptr;
    HostWorkerStartCallback on_worker_start_;
    std::thread thread_;
    std::atomic<bool> run_{false};
    std::atomic<bool> muted_{false};
    std::atomic<const char*> priority_method_{"not requested"};
    std::atomic<bool> priority_elevated_{false};
    // Worker counters carried over when detach() destroys the worker, so a
    // re-attached host keeps reporting monotonic accepted / rejected values.
    std::atomic<std::uint64_t> retired_accepted_{0};
    std::atomic<std::uint64_t> retired_rejected_{0};
};

} // namespace fluxvoice
