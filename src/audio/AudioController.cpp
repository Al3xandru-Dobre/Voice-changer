// AudioController — implementation (control thread only).
#include "audio/AudioController.hpp"
#include "audio/RouteGuard.hpp"

#include <algorithm>
#include <utility>

namespace fluxvoice {

namespace {

bool is_non_retryable(AudioErrorCategory c) {
    switch (c) {
        case AudioErrorCategory::PermissionDenied:
        case AudioErrorCategory::DeviceNotFound:
        case AudioErrorCategory::DeviceChanged:
        case AudioErrorCategory::FormatUnsupported:
        case AudioErrorCategory::NullBackendOnly:
            return true;
        case AudioErrorCategory::BackendUnavailable:
        case AudioErrorCategory::EnumerationFailed:
        case AudioErrorCategory::DeviceBusy:
        case AudioErrorCategory::Unknown:
            return false;
    }
    return false;
}

} // namespace

AudioController::AudioController(AudioBackend& backend, ControllerConfig cfg)
    : backend_(backend), cfg_(cfg), registry_(backend) {
    status_.backend = backend_.status().backend;
}

AudioController::~AudioController() {
    stop();
}

void AudioController::set_worker_hooks(std::function<void()> on_quiesce,
                                       std::function<void()> on_reset,
                                       std::function<void()> on_resume) {
    on_quiesce_ = std::move(on_quiesce);
    on_reset_   = std::move(on_reset);
    on_resume_  = std::move(on_resume);
}

Result<DeviceSnapshot> AudioController::refresh(TimePoint now) {
    return refresh_catalog_internal(now);
}

Result<DeviceSnapshot> AudioController::refresh_catalog_internal(TimePoint now) {
    auto r = registry_.refresh();
    if (is_ok(r)) {
        refresh_failures_  = 0;
        next_refresh_time_ = now + cfg_.refresh_interval;
        if (active_epoch_ == 0) active_epoch_ = registry_.snapshot().epoch;
    } else {
        ++refresh_failures_;
        const std::uint32_t shift = std::min<std::uint32_t>(refresh_failures_ - 1, 5u);
        const auto backoff = std::min(cfg_.max_retry_backoff,
                                      cfg_.initial_retry_backoff * (1 << shift));
        next_refresh_time_ = now + backoff;
    }
    return r;
}

bool AudioController::is_stream_live() const {
    if (status_.state != StreamState::Running) return false;
    if (!backend_.is_stream_running()) return false;
    if (callback_stalled_) return false;
    if ((last_notifications_ &
         (NotificationBits::kStopped | NotificationBits::kInterruptionBegan)) != 0)
        return false;
    return true;
}

void AudioController::quiesce_and_stop_stream(bool reset_dsp) {
    const bool was_active = backend_.is_stream_open() || status_.state == StreamState::Running ||
                            status_.state == StreamState::Opening ||
                            status_.state == StreamState::Recovering;
    if (!was_active && status_.state == StreamState::Idle) return;

    status_.state = StreamState::Stopping;

    // 1. Quiesce the worker thread first so it is neither consuming in_q_ nor
    //    producing out_q_ nor running InferencePipeline steps.
    if (was_active && on_quiesce_) on_quiesce_();

    // 2. Stop and uninit the backend audio stream so no RT callback can run.
    backend_.stop_stream();
    status_.backend_stream_started = false;

    // 3. Only after both producer and consumer threads are stopped, drain and
    //    discard both SPSC queues so no old-route PCM can leak into the next
    //    stream.
    last_discarded_samples_ = router_.discard_queues();

    // 4. Reset OLA / chunk assembler / stateful model caches.
    if (was_active && reset_dsp && on_reset_) on_reset_();
}

void AudioController::stop() {
    quiesce_and_stop_stream(/*reset_dsp=*/true);
    status_.state             = StreamState::Idle;
    status_.resolved_capture  = std::nullopt;
    status_.resolved_playback = std::nullopt;
    status_.retry_attempt     = 0;
    ever_opened_              = false;
    callback_stalled_         = false;
}

void AudioController::schedule_next_retry(TimePoint now) {
    const std::uint32_t shift =
        status_.retry_attempt > 0 ? std::min<std::uint32_t>(status_.retry_attempt - 1, 5u) : 0u;
    const auto delay =
        std::min(cfg_.max_retry_backoff, cfg_.initial_retry_backoff * (1 << shift));
    next_retry_time_ = now + delay;
}

void AudioController::enter_failure_state(const AudioError& err, TimePoint now) {
    status_.last_error        = err;
    status_.resolved_capture  = std::nullopt;
    status_.resolved_playback = std::nullopt;
    if (is_non_retryable(err.category)) {
        status_.state = StreamState::AwaitingUser;
    } else if (cfg_.max_retries == 0) {
        status_.state = StreamState::Error;
    } else {
        // Same policy as a mid-stream "device busy / server loss" failure:
        // bounded retries with backoff, not an instant terminal error.
        status_.state         = StreamState::Recovering;
        status_.retry_attempt = 0;
        schedule_next_retry(now);
    }
}

Result<StreamStatus> AudioController::open_and_start(const DeviceSelection& capture,
                                                     const DeviceSelection& playback,
                                                     CaptureMode capture_mode,
                                                     TimePoint now) {
    if (backend_.is_stream_open() || status_.state == StreamState::Running ||
        status_.state == StreamState::Recovering) {
        quiesce_and_stop_stream(/*reset_dsp=*/true);
    }

    status_.requested_capture  = capture;
    status_.requested_playback = playback;
    status_.capture_mode       = capture_mode;
    status_.retry_attempt      = 0;
    status_.last_error         = std::nullopt;
    ever_opened_               = true;

    if (registry_.snapshot().generation == 0) {
        auto snap_res = refresh_catalog_internal(now);
        if (!is_ok(snap_res)) {
            AudioError err = std::get<AudioError>(std::move(snap_res));
            enter_failure_state(err, now);
            return err;
        }
    }

    return open_resolved_internal(now, /*strict_generation=*/true);
}

Result<StreamStatus> AudioController::reconfigure(const DeviceSelection& capture,
                                                  const DeviceSelection& playback,
                                                  CaptureMode capture_mode,
                                                  TimePoint now) {
    // Always run the full switch sequence (quiesce -> stop -> discard -> reset)
    // before opening the new route.
    quiesce_and_stop_stream(/*reset_dsp=*/true);
    status_.requested_capture  = capture;
    status_.requested_playback = playback;
    status_.capture_mode       = capture_mode;
    status_.retry_attempt      = 0;
    status_.last_error         = std::nullopt;
    ever_opened_               = true;

    if (registry_.snapshot().generation == 0) {
        auto snap_res = refresh_catalog_internal(now);
        if (!is_ok(snap_res)) {
            AudioError err = std::get<AudioError>(std::move(snap_res));
            enter_failure_state(err, now);
            return err;
        }
    }

    return open_resolved_internal(now, /*strict_generation=*/true);
}

Result<StreamStatus> AudioController::on_context_recreated(TimePoint now) {
    const bool should_reopen = ever_opened_;
    quiesce_and_stop_stream(/*reset_dsp=*/true);
    status_.resolved_capture  = std::nullopt;
    status_.resolved_playback = std::nullopt;
    status_.retry_attempt     = 0;

    auto snap_res = refresh_catalog_internal(now);
    if (!is_ok(snap_res)) {
        AudioError err = std::get<AudioError>(std::move(snap_res));
        enter_failure_state(err, now);
        return err;
    }
    active_epoch_ = registry_.snapshot().epoch;

    if (!should_reopen) {
        status_.state = StreamState::Idle;
        return status_;
    }

    // Re-resolve saved policy in the new epoch: FollowSystemDefault reopens on
    // the new default; explicit keys from the dead epoch fail with
    // DeviceChanged and transition to AwaitingUser.
    return open_resolved_internal(now, /*strict_generation=*/false);
}

Result<StreamStatus> AudioController::open_resolved_internal(TimePoint now,
                                                             bool strict_generation) {
    status_.state = StreamState::Opening;
    router_.set_passthrough(cfg_.passthrough &&
                            status_.capture_mode == CaptureMode::OrdinaryMicrophone);

    const auto fail = [&](AudioError err) -> Result<StreamStatus> {
        status_.last_error        = err;
        status_.resolved_capture  = std::nullopt;
        status_.resolved_playback = std::nullopt;
        if (is_non_retryable(err.category)) {
            status_.state = StreamState::AwaitingUser;
        } else if (cfg_.max_retries > 0 && status_.retry_attempt == 0) {
            status_.state         = StreamState::Recovering;
            status_.retry_attempt = 0;
            schedule_next_retry(now);
        } else if (status_.retry_attempt < cfg_.max_retries) {
            status_.state = StreamState::Recovering;
            schedule_next_retry(now);
        } else {
            status_.state = StreamState::Error;
        }
        return err;
    };

    const auto in_res =
        status_.capture_mode == CaptureMode::SystemPlaybackLoopback
            ? registry_.resolve_loopback(status_.requested_capture, strict_generation)
            : registry_.resolve(status_.requested_capture, EndpointDirection::Capture,
                                strict_generation);
    if (!is_ok(in_res)) return fail(std::get<AudioError>(in_res));

    const auto out_res =
        registry_.resolve(status_.requested_playback, EndpointDirection::Playback,
                          strict_generation);
    if (!is_ok(out_res)) return fail(std::get<AudioError>(out_res));

    const ResolvedEndpoint& in_ep  = std::get<ResolvedEndpoint>(in_res);
    const ResolvedEndpoint& out_ep = std::get<ResolvedEndpoint>(out_res);

    const EndpointDirection cap_dir =
        status_.capture_mode == CaptureMode::SystemPlaybackLoopback
            ? EndpointDirection::Playback
            : EndpointDirection::Capture;
    const EndpointDescriptor* cap_desc =
        in_ep.use_default ? registry_.default_descriptor(cap_dir)
                          : registry_.find_descriptor(in_ep.key);
    const EndpointDescriptor* play_desc =
        out_ep.use_default ? registry_.default_descriptor(EndpointDirection::Playback)
                           : registry_.find_descriptor(out_ep.key);

    // Pre-open feedback check (catches monitor_of metadata and loopback onto
    // our own playback endpoint even before opening the device).
    if (is_feedback_route(cap_desc, play_desc,
                          cap_desc ? cap_desc->name : in_ep.name,
                          play_desc ? play_desc->name : out_ep.name,
                          status_.capture_mode)) {
        return fail(AudioError{
            AudioErrorCategory::FormatUnsupported, "check route feedback", 0,
            registry_.snapshot().backend,
            "Refusing to start: capture endpoint records FluxVoice's own playback endpoint "
            "(feedback loop). Pick a microphone with --input."});
    }

    StreamRequest req;
    req.capture_mode = status_.capture_mode;
    req.capture_key  = in_ep.use_default ? nullptr : &in_ep.key;
    req.playback_key = out_ep.use_default ? nullptr : &out_ep.key;
    req.sample_rate  = cfg_.sample_rate;
    req.period_ms    = cfg_.period_ms;
    req.passthrough  = cfg_.passthrough &&
                       status_.capture_mode == CaptureMode::OrdinaryMicrophone;

    auto open_res = backend_.open_stream(req, &AudioController::data_callback, this);
    if (!is_ok(open_res)) return fail(std::get<AudioError>(std::move(open_res)));

    info_ = std::move(std::get<StreamInfo>(open_res));

    // Post-open feedback check on the actual opened endpoint names (catches
    // OS-default endpoints when the catalog had no default flag).
    if (is_feedback_route(cap_desc, play_desc, info_.capture_name, info_.playback_name,
                          status_.capture_mode)) {
        backend_.stop_stream();
        return fail(AudioError{
            AudioErrorCategory::FormatUnsupported, "check route feedback", 0, info_.backend,
            "Refusing to start: capture device \"" + info_.capture_name +
                "\" records playback device \"" + info_.playback_name + "\" (feedback loop)."});
    }

    auto start_res = backend_.start_stream();
    if (!is_ok(start_res)) {
        backend_.stop_stream();
        return fail(std::get<AudioError>(std::move(start_res)));
    }

    // Drain any initial kStarted notification so liveness starts clean.
    last_notifications_ = backend_.drain_notifications() & ~NotificationBits::kStarted;
    active_epoch_       = registry_.snapshot().epoch;

    status_.state = StreamState::Running;
    if (!in_ep.use_default) {
        status_.resolved_capture = in_ep.key;
    } else if (cap_desc) {
        status_.resolved_capture = cap_desc->key;
    } else {
        status_.resolved_capture = std::nullopt;
    }
    if (!out_ep.use_default) {
        status_.resolved_playback = out_ep.key;
    } else if (play_desc) {
        status_.resolved_playback = play_desc->key;
    } else {
        status_.resolved_playback = std::nullopt;
    }
    status_.backend                 = info_.backend;
    status_.capture_name            = info_.capture_name;
    status_.playback_name           = info_.playback_name;
    status_.client_sample_rate      = info_.client_sample_rate;
    status_.capture_backend_rate    = info_.capture_internal_rate;
    status_.playback_backend_rate   = info_.playback_internal_rate;
    status_.capture_backend_period  = info_.capture_internal_period;
    status_.playback_backend_period = info_.playback_internal_period;
    status_.selection_generation    = in_ep.generation;
    status_.backend_stream_started  = backend_.is_stream_running();
    status_.callback_progress       = router_.frames_processed();
    status_.retry_attempt           = 0;
    status_.last_error              = std::nullopt;

    // Liveness baseline: the progress counter is cumulative, so the watermark
    // is the current value; the clock starts now.
    last_callback_progress_ = router_.frames_processed();
    progress_since_         = now;
    callback_stalled_       = false;

    if (on_resume_) on_resume_();
    return status_;
}

void AudioController::tick(TimePoint now) {
    // 1. Drain notification bitset (coalesces any burst into one bitmask).
    const std::uint32_t notes = backend_.drain_notifications();
    last_notifications_       = notes;

    // Callback-progress liveness (stage 05): while Running, the counter must
    // keep advancing — a frozen counter means the RT thread is dead even if
    // the backend still reports "started". Silence still counts as progress.
    status_.callback_progress      = router_.frames_processed();
    status_.backend_stream_started = backend_.is_stream_running();
    if (status_.state == StreamState::Running) {
        if (progress_since_ == TimePoint{}) progress_since_ = now;
        if (status_.callback_progress != last_callback_progress_) {
            last_callback_progress_ = status_.callback_progress;
            progress_since_         = now;
            callback_stalled_       = false;
        } else if (!callback_stalled_ && cfg_.callback_stall_timeout.count() > 0 &&
                   now - progress_since_ >= cfg_.callback_stall_timeout) {
            callback_stalled_ = true;
        }
    } else {
        callback_stalled_ = false;
    }

    // 2. Refresh catalog when periodic interval elapses or notification hints arrive.
    const bool hint_refresh =
        (notes & (NotificationBits::kRerouted | NotificationBits::kStopped |
                  NotificationBits::kInterruptionBegan | NotificationBits::kInterruptionEnded)) !=
        0;
    const bool periodic_due =
        cfg_.refresh_interval.count() > 0 && now >= next_refresh_time_;

    if (hint_refresh || periodic_due) {
        auto r = refresh_catalog_internal(now);
        if (is_ok(r) && active_epoch_ != 0 &&
            registry_.snapshot().epoch != active_epoch_) {
            // Backend context epoch changed underneath us.
            active_epoch_ = registry_.snapshot().epoch;
            if (ever_opened_) {
                quiesce_and_stop_stream(/*reset_dsp=*/true);
                (void)open_resolved_internal(now, /*strict_generation=*/false);
                return;
            }
        }
    }

    const EndpointDirection cap_dir =
        status_.capture_mode == CaptureMode::SystemPlaybackLoopback
            ? EndpointDirection::Playback
            : EndpointDirection::Capture;

    // 3. Running state: evaluate policy table.
    if (status_.state == StreamState::Running) {
        if (!registry_.stale() && registry_.snapshot().generation > 0) {
            // Policy: Explicit capture device disappears -> stop route; show
            // missing selection; NEVER record another microphone.
            if (std::holds_alternative<ExplicitEndpoint>(status_.requested_capture)) {
                auto cap_check =
                    registry_.resolve(status_.requested_capture, cap_dir, /*strict=*/false);
                if (!is_ok(cap_check)) {
                    quiesce_and_stop_stream(/*reset_dsp=*/true);
                    status_.state            = StreamState::AwaitingUser;
                    status_.resolved_capture = std::nullopt;
                    status_.last_error       = std::get<AudioError>(std::move(cap_check));
                    return;
                }
            }

            // Policy: Explicit playback / virtual output disappears -> stop
            // output; request repair; NEVER redirect processed speech to
            // speakers.
            if (std::holds_alternative<ExplicitEndpoint>(status_.requested_playback)) {
                auto play_check = registry_.resolve(status_.requested_playback,
                                                    EndpointDirection::Playback, /*strict=*/false);
                if (!is_ok(play_check)) {
                    quiesce_and_stop_stream(/*reset_dsp=*/true);
                    status_.state             = StreamState::AwaitingUser;
                    status_.resolved_playback = std::nullopt;
                    status_.last_error        = std::get<AudioError>(std::move(play_check));
                    return;
                }
            }

            // Policy: Follow-default selection when OS default changes.
            const bool follow_cap =
                std::holds_alternative<FollowSystemDefault>(status_.requested_capture);
            const bool follow_play =
                std::holds_alternative<FollowSystemDefault>(status_.requested_playback);

            if (follow_cap || follow_play) {
                const EndpointDescriptor* def_cap =
                    follow_cap ? registry_.default_descriptor(cap_dir)
                               : (status_.resolved_capture
                                      ? registry_.find_descriptor(*status_.resolved_capture)
                                      : nullptr);
                const EndpointDescriptor* def_play =
                    follow_play ? registry_.default_descriptor(EndpointDirection::Playback)
                                : (status_.resolved_playback
                                       ? registry_.find_descriptor(*status_.resolved_playback)
                                       : nullptr);

                const std::optional<EndpointKey> new_cap_key =
                    def_cap ? std::optional<EndpointKey>(def_cap->key) : std::nullopt;
                const std::optional<EndpointKey> new_play_key =
                    def_play ? std::optional<EndpointKey>(def_play->key) : std::nullopt;

                const bool cap_changed  = follow_cap && (new_cap_key != status_.resolved_capture);
                const bool play_changed = follow_play && (new_play_key != status_.resolved_playback);

                if (cap_changed || play_changed || (notes & NotificationBits::kRerouted) != 0) {
                    // Never follow a default change into a feedback route!
                    const std::string new_cap_name =
                        def_cap ? def_cap->name : status_.capture_name;
                    const std::string new_play_name =
                        def_play ? def_play->name : status_.playback_name;
                    if (is_feedback_route(def_cap, def_play, new_cap_name, new_play_name,
                                          status_.capture_mode)) {
                        quiesce_and_stop_stream(/*reset_dsp=*/true);
                        status_.state      = StreamState::AwaitingUser;
                        status_.last_error = AudioError{
                            AudioErrorCategory::FormatUnsupported, "follow default route", 0,
                            status_.backend,
                            "OS default changed to a feedback loop with FluxVoice's output; "
                            "stream stopped."};
                        return;
                    }

                    if (backend_.auto_routes_default() && backend_.is_stream_running()) {
                        // Backend auto-routed the stream; update resolved
                        // endpoints in place without a duplicate restart.
                        if (follow_cap) {
                            status_.resolved_capture = new_cap_key;
                            if (def_cap) {
                                status_.capture_name = def_cap->name;
                                info_.capture_name   = def_cap->name;
                            }
                        }
                        if (follow_play) {
                            status_.resolved_playback = new_play_key;
                            if (def_play) {
                                status_.playback_name = def_play->name;
                                info_.playback_name   = def_play->name;
                            }
                        }
                        status_.selection_generation = registry_.snapshot().generation;
                        ++auto_reroutes_;
                    } else {
                        // Reopen on the new OS default.
                        quiesce_and_stop_stream(/*reset_dsp=*/true);
                        (void)open_resolved_internal(now, /*strict_generation=*/false);
                        return;
                    }
                }
            }
        }

        // Policy: Unexpected stream stop / interruption / dead callback
        // thread -> bounded recovery.
        if ((notes & (NotificationBits::kStopped | NotificationBits::kInterruptionBegan)) != 0 ||
            !backend_.is_stream_running() || callback_stalled_) {
            quiesce_and_stop_stream(/*reset_dsp=*/true);
            status_.last_error = AudioError{
                AudioErrorCategory::DeviceBusy, "stream interrupted", 0, status_.backend,
                "Audio stream stopped unexpectedly; attempting bounded recovery."};
            if (cfg_.max_retries == 0) {
                status_.state = StreamState::Error;
            } else {
                status_.state         = StreamState::Recovering;
                status_.retry_attempt = 0;
                schedule_next_retry(now);
            }
            return;
        }
    }

    // 4. Recovering state: bounded retries with backoff.
    if (status_.state == StreamState::Recovering) {
        if (!registry_.stale() && registry_.snapshot().generation > 0) {
            if (std::holds_alternative<ExplicitEndpoint>(status_.requested_capture)) {
                auto cap_check =
                    registry_.resolve(status_.requested_capture, cap_dir, /*strict=*/false);
                if (!is_ok(cap_check)) {
                    status_.state            = StreamState::AwaitingUser;
                    status_.resolved_capture = std::nullopt;
                    status_.last_error       = std::get<AudioError>(std::move(cap_check));
                    return;
                }
            }
            if (std::holds_alternative<ExplicitEndpoint>(status_.requested_playback)) {
                auto play_check = registry_.resolve(status_.requested_playback,
                                                    EndpointDirection::Playback, /*strict=*/false);
                if (!is_ok(play_check)) {
                    status_.state             = StreamState::AwaitingUser;
                    status_.resolved_playback = std::nullopt;
                    status_.last_error        = std::get<AudioError>(std::move(play_check));
                    return;
                }
            }
        }

        const bool resume_hint = (notes & NotificationBits::kInterruptionEnded) != 0;
        if (resume_hint || now >= next_retry_time_) {
            if (status_.retry_attempt >= cfg_.max_retries) {
                status_.state = StreamState::Error;
                return;
            }
            const std::uint32_t attempt = status_.retry_attempt + 1;
            status_.retry_attempt       = attempt;

            // A retry can only succeed once a catalog exists: an earlier
            // enumeration failure is what put us here. Heal it first.
            if (registry_.snapshot().generation == 0) {
                auto snap_res = refresh_catalog_internal(now);
                if (!is_ok(snap_res)) {
                    const AudioError& err = std::get<AudioError>(snap_res);
                    status_.last_error    = err;
                    if (is_non_retryable(err.category)) {
                        status_.state = StreamState::AwaitingUser;
                    } else if (attempt >= cfg_.max_retries) {
                        status_.state = StreamState::Error;
                    } else {
                        status_.state = StreamState::Recovering;
                        schedule_next_retry(now);
                    }
                    return;
                }
            }

            auto res = open_resolved_internal(now, /*strict_generation=*/false);
            if (!is_ok(res)) {
                status_.retry_attempt = attempt;
                const AudioError& err = std::get<AudioError>(res);
                if (is_non_retryable(err.category)) {
                    status_.state = StreamState::AwaitingUser;
                } else if (attempt >= cfg_.max_retries) {
                    status_.state = StreamState::Error;
                } else {
                    status_.state = StreamState::Recovering;
                    schedule_next_retry(now);
                }
            }
        }
    }
}

void AudioController::data_callback(void* user_data, float* output, const float* input,
                                    std::size_t sample_count) {
    auto* self = static_cast<AudioController*>(user_data);
    self->router_.process(output, input, sample_count);
}

} // namespace fluxvoice
