// AudioBackend — the seam between application code and an audio backend.
//
// Deliberately small (Milestone 1.5 / Stage 02): just enough surface for the
// device registry (stage 03) and its fake-backend tests to be written against
// something that is not miniaudio. It speaks only the AudioTypes vocabulary;
// native identifiers stay inside MiniaudioBackend. This is not a plugin
// system — there is exactly one real implementation.
//
// Control-thread only: implementations may enumerate and probe, which is
// never real-time safe.
#pragma once

#include "audio/AudioTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

class AudioBackend {
public:
    /// Real-time callback signature: plain function pointer + opaque context
    /// so the audio callback path has zero std::function / heap overhead.
    using AudioProcessFn = void (*)(void* user_data, float* output, const float* input,
                                    std::size_t sample_count);

    virtual ~AudioBackend() = default;

    /// Context-level status: initialisation outcome, the selected backend
    /// name, and the exact native result code on failure.
    virtual fluxvoice::BackendStatus status() const = 0;

    /// Enumerate both directions in one serialized call and return an owned
    /// snapshot. Implementations must copy everything out of the backend's
    /// arrays before returning (they are invalidated by the next call).
    virtual fluxvoice::Result<fluxvoice::DeviceSnapshot> enumerate() = 0;

    /// On-demand probe of one endpoint's native formats (stage 03). Opens no
    /// stream; callers use it for a *selected* device, never per list entry.
    virtual fluxvoice::Result<fluxvoice::EndpointInfo> probe(
        const fluxvoice::EndpointKey& key) const = 0;

    /// What this backend can do; capabilities not yet probed stay Unknown.
    virtual fluxvoice::AudioCapabilities capabilities() const = 0;

    // ---- Stream lifecycle & notifications (stages 05–06) ----

    /// Open a stream without starting it, returning the backend-negotiated
    /// StreamInfo or a typed AudioError.
    virtual fluxvoice::Result<fluxvoice::StreamInfo> open_stream(
        const fluxvoice::StreamRequest& req, AudioProcessFn process_fn, void* user_data) {
        (void)req;
        (void)process_fn;
        (void)user_data;
        return fluxvoice::AudioError{fluxvoice::AudioErrorCategory::BackendUnavailable,
                                     "open stream", 0, "", "Stream open not implemented."};
    }

    /// Start the opened stream; callbacks begin to run.
    virtual fluxvoice::Result<bool> start_stream() {
        return fluxvoice::AudioError{fluxvoice::AudioErrorCategory::BackendUnavailable,
                                     "start stream", 0, "", "Stream start not implemented."};
    }

    /// Stop and uninitialise the opened stream (no-op when none is open).
    virtual void stop_stream() {}

    /// True when a stream has been initialized and not yet uninitialized.
    virtual bool is_stream_open() const { return false; }

    /// True when the backend reports the active stream is currently started.
    virtual bool is_stream_running() const { return false; }

    /// Atomically drain and clear pending notification bits set by the
    /// backend's notification callback.
    virtual std::uint32_t drain_notifications() { return fluxvoice::NotificationBits::kNone; }

    /// True when the backend automatically re-routes streams opened with a
    /// null (FollowSystemDefault) device ID on OS default changes, so the
    /// controller should not trigger a duplicate restart.
    virtual bool auto_routes_default() const { return false; }
};
