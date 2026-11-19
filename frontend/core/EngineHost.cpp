// EngineHost — controller composition and the Bypass worker loop.
#include "core/EngineHost.hpp"

#include "audio/AudioTypes.hpp"
#include "core/Meters.hpp"
#include "core/Telemetry.hpp"
#include "pipeline/HostStats.hpp"
#include "pipeline/PipelineGeometry.hpp"
#include "util/ProcessMemory.hpp"
#include "util/WorkerPriority.hpp"
#include "voice/VoiceParams.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace fluxvoice::frontend {

namespace {

void apply_targets(VoiceOutputChain& chain, const VoiceParams& p) {
    chain.set_targets(p.get(ParamId::Gain), p.get(ParamId::GateThreshold), p.enabled(ParamId::Limiter),
                      p.get(ParamId::Brightness));
}

} // namespace

EngineHost::EngineHost(AudioBackend& backend, ControllerConfig cfg,
                       std::shared_ptr<const fluxvoice::HostModelLoader> loader)
    : backend_(backend), controller_(backend, cfg), bypass_cfg_(cfg), loader_(std::move(loader)) {
    (void)backend_;
    reset_dsp();  // sizes the 20 ms block assembler and configures the chain at the configured rate
    install_bypass_hooks();
}

EngineHost::~EngineHost() {
    controller_.stop();  // quiesces the worker through the hooks before closing the stream
    stop_worker();
    if (host_) host_->detach();  // hooks cleared before the controller goes
    host_.reset();
}

void EngineHost::install_bypass_hooks() {
    controller_.set_prepare_hook({});
    controller_.set_worker_hooks([this] { stop_worker(); }, [this] { reset_dsp(); }, [this] { start_worker(); });
}

std::string EngineHost::mode_label() const {
    return host_ ? "Mode A - " + bundle_name_ : "Bypass - no model loaded";
}

std::optional<std::uint16_t> EngineHost::voice_index(std::string_view name, std::string& error) const {
    if (!host_) {
        error = "no model bundle is loaded";
        return std::nullopt;
    }
    return host_->voice_index(name, error);
}
bool EngineHost::host_has_translator() const { return host_ && host_->langs() != nullptr; }
bool EngineHost::host_failed() const { return host_ && host_->failed(); }
std::string EngineHost::host_error() const { return host_ ? host_->error() : std::string{}; }
std::uint64_t EngineHost::rejected_params() const { return host_ ? host_->rejected_params() : 0; }

std::uint64_t EngineHost::worker_hops() const {
    return host_ ? host_->timing().samples : hops_.load(std::memory_order_relaxed);
}
const char* EngineHost::priority_method() const {
    return host_ ? host_->priority_method() : priority_method_.load(std::memory_order_acquire);
}
bool EngineHost::priority_elevated() const {
    return host_ ? host_->priority_elevated() : priority_elevated_.load(std::memory_order_acquire);
}

void EngineHost::restore_bypass_config() {
    ControllerConfig c = controller_.config();
    c.sample_rate        = bypass_cfg_.sample_rate;
    c.fade_underruns     = bypass_cfg_.fade_underruns;
    c.max_queued_samples = bypass_cfg_.max_queued_samples;
    controller_.set_config(c);
}

bool EngineHost::install_host(std::unique_ptr<fluxvoice::ModeAHost> host, std::string bundle_name, std::string& error) {
    using fluxvoice::HostError;
    if (!host || !host->loaded()) {
        error = "no loaded model host to install";
        return false;
    }
    if (controller_.status().state != StreamState::Idle) {
        error = "the stream must be stopped before the model changes";
        return false;
    }
    // The previous host stays alive (detached) until the new one is attached:
    // on failure it is put back, so a refused bundle never turns into Bypass.
    std::unique_ptr<fluxvoice::ModeAHost> previous = std::move(host_);
    const std::string previous_name = std::move(bundle_name_);
    if (previous) previous->detach();  // joins its worker, clears the controller hooks

    ControllerConfig cfg = controller_.config();
    host->configure(cfg);
    controller_.set_config(cfg);  // the contract rate is applied while Idle only
    HostError herr;
    if (!host->attach(controller_, herr)) {
        error = herr.message.empty() ? std::string("the model host refused to attach") : herr.message;
        if (previous) {
            ControllerConfig back = controller_.config();
            previous->configure(back);
            controller_.set_config(back);
            HostError rerr;
            if (previous->attach(controller_, rerr)) {
                host_        = std::move(previous);
                bundle_name_ = previous_name;
                host_->set_muted(muted());
                return false;
            }
            error += "; the previous bundle could not be re-attached (" + rerr.message + ")";
        }
        restore_bypass_config();
        install_bypass_hooks();
        reset_dsp();
        return false;
    }
    host->set_muted(muted());
    host_        = std::move(host);
    bundle_name_ = std::move(bundle_name);
    host_drops_closed_ = 0;
    return true;  // `previous` is destroyed here: worker joined, hooks cleared, stream idle
}

bool EngineHost::remove_host(std::string& error) {
    if (controller_.status().state != StreamState::Idle) {
        error = "the stream must be stopped before the model changes";
        return false;
    }
    if (!host_) return true;
    host_->detach();
    host_.reset();
    bundle_name_.clear();
    restore_bypass_config();
    install_bypass_hooks();
    reset_dsp();
    return true;
}

Result<StreamStatus> EngineHost::start(const DeviceSelection& capture, const DeviceSelection& playback,
                                       CaptureMode capture_mode, TimePoint now) {
    const StreamState st = controller_.status().state;
    if (st == StreamState::Running || st == StreamState::Recovering || st == StreamState::Opening)
        return controller_.reconfigure(capture, playback, capture_mode, now);
    return controller_.open_and_start(capture, playback, capture_mode, now);
}

void EngineHost::stop() {
    // The final worker counters are read before the controller stops
    // (ModeAHost.hpp), once per stream: a stop() while already Idle would
    // otherwise add the last stream's drops to the lifetime total again.
    if (host_ && controller_.status().state != StreamState::Idle) host_drops_closed_ += host_->snapshot().drops;
    controller_.stop();
}

std::uint64_t EngineHost::push_params(const VoiceParams& p) {
    // ParamChannel::push parks the newest snapshot when the queue is full and
    // retries it on flush(); the newest snapshot always wins (BUG-023).
    if (host_) return host_->push_params(p);
    return params_.push(p);
}

void EngineHost::flush_params() {
    if (host_) {
        static_cast<void>(host_->flush_params());
    } else {
        static_cast<void>(params_.flush());
    }
}

void EngineHost::set_muted(bool m) {
    muted_.store(m, std::memory_order_relaxed);  // Bypass worker reads it each loop
    if (host_) host_->set_muted(m);              // ramped inside the pipeline's output chain
}

std::uint32_t EngineHost::stream_rate() const {
    return controller_.stream_info().client_sample_rate != 0 ? controller_.stream_info().client_sample_rate
                                                             : controller_.config().sample_rate;
}

TelemetrySnapshot EngineHost::telemetry() const {
    TelemetrySnapshot t;
    const DuplexRouter& r = controller_.router();
    t.frames                    = r.frames_processed();
    t.lifetime_input_overruns   = r.input_overruns();
    t.lifetime_output_underruns = r.output_underruns();
    t.sample_rate               = stream_rate();
    t.rss_bytes                 = fluxvoice::resident_bytes();
    // Independently clocked loopback callbacks do not obey the identity.
    t.queued_valid = controller_.status().capture_mode != CaptureMode::SystemPlaybackLoopback;
    t.stream_running = controller_.status().state == StreamState::Running;

    if (host_) {
        // Mode A: the worker owns the accounting. One HostStats snapshot is the
        // source of every figure below (and of the `[stats]` line): the full
        // identity (MEMORY.md §2) queued = prefilled + underruns - overruns -
        // drops - skipped, rebaselined by the worker's reset() after every
        // queue discard (BUG-024).
        const fluxvoice::HostStats hs = host_->stats();
        t.stats                   = hs;
        t.input_overruns          = hs.overruns;
        t.output_underruns        = hs.underruns;
        t.output_drops            = hs.drops;
        t.queued_samples          = hs.queued;
        t.lifetime_output_drops   = host_drops_closed_ + hs.drops;
        const bool running        = t.stream_running;
        const fluxvoice::WorkerMeters m = host_->meters();
        t.input_peak_dbfs  = running ? meters::to_dbfs(m.in_peak) : -120.0f;
        t.input_rms_dbfs   = running ? meters::to_dbfs(m.in_rms) : -120.0f;
        t.output_peak_dbfs = running ? meters::to_dbfs(m.out_peak) : -120.0f;
        t.output_rms_dbfs  = running ? meters::to_dbfs(m.out_rms) : -120.0f;
        t.mode_a           = true;
        t.params_pushed    = host_->pushed_generation();
        t.params_applied   = host_->applied_generation();
        t.params_accepted  = host_->accepted_generation();
        t.params_rejected  = host_->rejected_params();
        return t;
    }

    // Bypass: per-stream counters rebaselined by reset_dsp() after every queue
    // discard, so residue of earlier streams stays out of the figure (BUG-024).
    t.input_overruns   = r.input_overruns() - overrun_base_;
    t.output_underruns = r.output_underruns() - underrun_base_;
    t.output_drops     = output_drops_.load(std::memory_order_relaxed);
    t.queued_samples   = static_cast<std::int64_t>(t.output_underruns) - static_cast<std::int64_t>(t.input_overruns) -
                         static_cast<std::int64_t>(t.output_drops);
    t.lifetime_output_drops = lifetime_drops_.load(std::memory_order_relaxed);
    t.input_peak_dbfs  = meters::to_dbfs(in_peak_.load(std::memory_order_relaxed));
    t.input_rms_dbfs   = meters::to_dbfs(in_rms_.load(std::memory_order_relaxed));
    t.output_peak_dbfs = meters::to_dbfs(out_peak_.load(std::memory_order_relaxed));
    t.output_rms_dbfs  = meters::to_dbfs(out_rms_.load(std::memory_order_relaxed));
    t.mode_a           = false;
    t.params_pushed    = params_.pushed_generation();
    t.params_applied   = params_.applied_generation();
    t.params_accepted  = 0;  // Bypass has no pipeline to accept anything
    t.params_rejected  = 0;

    // The legacy `[stats]` snapshot, as the CLI's diagnostic modes print it:
    // no inference fields (`vc` false), no governor (skipped = prefilled = 0).
    fluxvoice::HostStats& s = t.stats;
    s.frames      = t.frames;
    s.overruns    = t.input_overruns;
    s.underruns   = t.output_underruns;
    s.drops       = t.output_drops;
    s.queued      = t.queued_samples;
    s.rss         = t.rss_bytes;
    s.sample_rate = t.sample_rate;
    s.hop         = blocks_ ? blocks_->hop() : 0;
    // Queued + device terms only: a geometry with a rate and no model delay.
    fluxvoice::PipelineGeometry g;
    g.sample_rate = t.sample_rate;
    s.latency     = fluxvoice::estimate_latency(g, t.queued_samples, controller_.stream_info());
    s.lat_ms      = s.latency.total_ms;
    // Occupancy = input queue + output queue + the worker's held samples; it
    // equals `queued` at quiescence. The counters and the queues are read at
    // different instants, so a running stream may differ by one device period
    // plus one hop (the worker publishes held samples after each read).
    s.occupancy = static_cast<std::int64_t>(r.input_queued_approx() + r.output_queued_approx() +
                                            held_.load(std::memory_order_relaxed));
    const StreamInfo& si = controller_.stream_info();
    const auto period_samples = [&](std::uint32_t frames, std::uint32_t rate) -> std::int64_t {
        return rate == 0 ? 0 : static_cast<std::int64_t>(std::uint64_t{frames} * t.sample_rate / rate);
    };
    s.occupancy_tolerance = std::max(period_samples(si.capture_internal_period, si.capture_internal_rate),
                                     period_samples(si.playback_internal_period, si.playback_internal_rate)) +
                            static_cast<std::int64_t>(s.hop);
    return t;
}

void EngineHost::stop_worker() {
    worker_run_.store(false, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    in_peak_.store(0.0f, std::memory_order_relaxed);
    in_rms_.store(0.0f, std::memory_order_relaxed);
    out_peak_.store(0.0f, std::memory_order_relaxed);
    out_rms_.store(0.0f, std::memory_order_relaxed);
    held_.store(0, std::memory_order_relaxed);
}

void EngineHost::reset_dsp() {
    const std::uint32_t rate = stream_rate();
    const std::size_t block  = std::max<std::size_t>(1, static_cast<std::size_t>(rate) * kBypassBlockMs / 1000);
    blocks_ = std::make_unique<ChunkAssembler>(block, 0);
    chain_.configure(rate);
    apply_targets(chain_, bypass_params_);
    chain_.set_muted(muted_.load(std::memory_order_relaxed));  // a muted stream restarts silent
    chain_.reset();
    // The controller discarded both queues just before this hook: begin a new
    // per-stream accounting epoch so the queued readout starts at zero
    // (BUG-024, MEMORY.md §2 "rebaseline after queue discards").
    underrun_base_ = controller_.router().output_underruns();
    overrun_base_  = controller_.router().input_overruns();
    output_drops_.store(0, std::memory_order_relaxed);
    held_.store(0, std::memory_order_relaxed);
}

void EngineHost::start_worker() {
    if (worker_.joinable()) return;
    worker_run_.store(true, std::memory_order_relaxed);
    worker_ = std::thread([this] { bypass_loop(); });
}

void EngineHost::bypass_loop() {
    // Same request as the CLI worker (src/main.cpp); reverted when the thread ends.
    const WorkerPriority priority;
    priority_method_.store(priority.method(), std::memory_order_release);
    priority_elevated_.store(priority.elevated(), std::memory_order_release);
    std::vector<float> buf(4096);
    std::vector<float> block(blocks_->hop());
    VoiceParams msg{};
    auto emit = [&](const float* p, std::size_t n) {
        const std::size_t written = controller_.write_output(p, n);
        const std::size_t dropped = n - written;
        output_drops_.fetch_add(dropped, std::memory_order_relaxed);
        lifetime_drops_.fetch_add(dropped, std::memory_order_relaxed);
    };
    while (worker_run_.load(std::memory_order_relaxed)) {
        // Newest parameter snapshot wins (M2 / Stage 09: latest wins, no locks).
        // Only the output stage exists in Bypass; every other control is ignored.
        if (params_.drain(msg)) {
            bypass_params_ = msg;
            apply_targets(chain_, bypass_params_);
        }
        chain_.set_muted(muted_.load(std::memory_order_relaxed));

        const std::size_t n = controller_.read_input(buf.data(), buf.size());
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        float ip = 0.0f, ir = 0.0f;
        meters::measure(buf.data(), n, ip, ir);
        in_peak_.store(ip, std::memory_order_relaxed);
        in_rms_.store(ir, std::memory_order_relaxed);

        const float* p = buf.data();
        std::size_t left = n;
        held_.store(blocks_->pending() + n, std::memory_order_relaxed);  // read, not yet emitted
        while (left > 0) {
            const std::size_t used = blocks_->push(p, left);
            p += used;
            left -= used;
            if (blocks_->ready()) {
                std::copy_n(blocks_->hop_samples(), block.size(), block.begin());
                // The gate decides on the block's input level, like the pipeline's
                // per-frame energy (one 20 ms frame per block).
                float bp = 0.0f, br = 0.0f;
                meters::measure(block.data(), block.size(), bp, br);
                const float level = meters::to_dbfs(br);
                chain_.process(std::span<float>(block), std::span<const float>(&level, 1), block.size());
                float op = 0.0f, orms = 0.0f;
                meters::measure(block.data(), block.size(), op, orms);
                out_peak_.store(op, std::memory_order_relaxed);
                out_rms_.store(orms, std::memory_order_relaxed);
                emit(block.data(), block.size());
                blocks_->advance();
                hops_.fetch_add(1, std::memory_order_relaxed);
            }
            held_.store(blocks_->pending() + left, std::memory_order_relaxed);
        }
    }
}

} // namespace fluxvoice::frontend
