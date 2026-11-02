// EngineHost — controller composition and the worker loop.
#include "core/EngineHost.hpp"

#include "audio/AudioTypes.hpp"
#include "core/OutputChain.hpp"
#include "core/Telemetry.hpp"
#include "pipeline/PipelineTiming.hpp"
#include "util/ProcessMemory.hpp"
#include "voice/VoiceParams.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace fluxvoice::frontend {

EngineHost::EngineHost(AudioBackend& backend, ControllerConfig cfg)
    : backend_(backend),
      controller_(backend, cfg),
      chunks_(PipelineTiming::kChunkSamples, PipelineTiming::kContextSamples),
      params_q_(64) {
    (void)backend_;
    chain_.configure(cfg.sample_rate);
    controller_.set_worker_hooks([this] { stop_worker(); }, [this] { reset_dsp(); }, [this] { start_worker(); });
}

EngineHost::~EngineHost() {
    controller_.stop();  // quiesces the worker through the hooks before closing the stream
    stop_worker();
}

Result<StreamStatus> EngineHost::start(const DeviceSelection& capture, const DeviceSelection& playback,
                                       CaptureMode capture_mode, TimePoint now) {
    const StreamState st = controller_.status().state;
    if (st == StreamState::Running || st == StreamState::Recovering || st == StreamState::Opening)
        return controller_.reconfigure(capture, playback, capture_mode, now);
    return controller_.open_and_start(capture, playback, capture_mode, now);
}

void EngineHost::stop() { controller_.stop(); }

std::uint64_t EngineHost::push_params(const VoiceParams& p) {
    ParamMessage m;
    m.params     = p;
    m.generation = ++next_generation_;
    // Full queue: the worker is far behind (or not running); the newest
    // snapshot wins at the next push, so dropping is harmless.
    if (params_q_.try_enqueue(m)) params_pushed_.store(m.generation, std::memory_order_relaxed);
    return m.generation;
}

TelemetrySnapshot EngineHost::telemetry() const {
    TelemetrySnapshot t;
    const DuplexRouter& r = controller_.router();
    t.frames           = r.frames_processed();
    t.input_overruns   = r.input_overruns();
    t.output_underruns = r.output_underruns();
    t.output_drops     = output_drops_.load(std::memory_order_relaxed);
    t.queued_samples   = static_cast<std::int64_t>(t.output_underruns) - static_cast<std::int64_t>(t.input_overruns) -
                         static_cast<std::int64_t>(t.output_drops);
    t.sample_rate      = controller_.stream_info().client_sample_rate != 0 ? controller_.stream_info().client_sample_rate
                                                                           : controller_.config().sample_rate;
    t.rss_bytes        = fluxvoice::resident_bytes();
    t.input_peak_dbfs  = OutputChain::to_dbfs(in_peak_.load(std::memory_order_relaxed));
    t.input_rms_dbfs   = OutputChain::to_dbfs(in_rms_.load(std::memory_order_relaxed));
    t.output_peak_dbfs = OutputChain::to_dbfs(out_peak_.load(std::memory_order_relaxed));
    t.output_rms_dbfs  = OutputChain::to_dbfs(out_rms_.load(std::memory_order_relaxed));
    t.params_pushed    = params_pushed_.load(std::memory_order_relaxed);
    t.params_applied   = params_applied_.load(std::memory_order_relaxed);
    return t;
}

void EngineHost::stop_worker() {
    worker_run_.store(false, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    in_peak_.store(0.0f, std::memory_order_relaxed);
    in_rms_.store(0.0f, std::memory_order_relaxed);
    out_peak_.store(0.0f, std::memory_order_relaxed);
    out_rms_.store(0.0f, std::memory_order_relaxed);
}

void EngineHost::reset_dsp() {
    chunks_.reset();
    pipeline_.reset();
    chain_.configure(controller_.stream_info().client_sample_rate != 0 ? controller_.stream_info().client_sample_rate
                                                                       : controller_.config().sample_rate);
    chain_.reset();
}

void EngineHost::start_worker() {
    if (worker_.joinable()) return;
    worker_run_.store(true, std::memory_order_relaxed);
    worker_ = std::thread([this] { worker_loop(); });
}

void EngineHost::worker_loop() {
    std::vector<float> buf(4096);
    std::vector<float> hop(chunks_.hop());
    ParamMessage msg;
    auto emit = [&](const float* p, std::size_t n) {
        const std::size_t written = controller_.write_output(p, n);
        output_drops_.fetch_add(n - written, std::memory_order_relaxed);
    };
    while (worker_run_.load(std::memory_order_relaxed)) {
        // Newest parameter snapshot wins (M2 / Stage 09: latest wins, no locks).
        bool got_params = false;
        while (params_q_.try_dequeue(msg)) got_params = true;
        if (got_params) {
            chain_.set_gain_db(msg.params.get(ParamId::Gain));
            chain_.set_limiter(msg.params.enabled(ParamId::Limiter));
            params_applied_.store(msg.generation, std::memory_order_relaxed);
        }
        chain_.set_muted(muted_.load(std::memory_order_relaxed));

        const std::size_t n = controller_.read_input(buf.data(), buf.size());
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        float ip = 0.0f, ir = 0.0f;
        OutputChain::measure(buf.data(), n, ip, ir);
        in_peak_.store(ip, std::memory_order_relaxed);
        in_rms_.store(ir, std::memory_order_relaxed);

        const float* p = buf.data();
        std::size_t left = n;
        while (left > 0) {
            const std::size_t used = chunks_.push(p, left);
            p += used;
            left -= used;
            if (chunks_.ready()) {
                // Inference lands here (M2 / Stage 08). Today the hop passes
                // through the output chain unchanged.
                for (std::size_t i = 0; i < hop.size(); ++i) hop[i] = chunks_.hop_samples()[i];
                chain_.process(hop.data(), hop.size());
                out_peak_.store(chain_.last_peak(), std::memory_order_relaxed);
                out_rms_.store(chain_.last_rms(), std::memory_order_relaxed);
                emit(hop.data(), hop.size());
                chunks_.advance();
                hops_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

} // namespace fluxvoice::frontend
