// MiniaudioBackend — the AudioBackend implementation over miniaudio 0.11.25.
//
// Owns the one ma_context (its address is stable for the object's lifetime)
// and the private EndpointKey-to-ma_device_id mapping (research §6 rules 2–4):
//   * enumeration copies everything out of ma_context_get_devices' arrays
//     before they are invalidated by the next call;
//   * native ids are compared with ma_device_id_equal(), never hashed or
//     serialized;
//   * each instance draws a fresh epoch, so keys minted by a previous context
//     never resolve here.
//
// The id table persists across enumerate() calls within one context, so the
// same physical device keeps the same key from snapshot to snapshot —
// snapshots may reorder freely without changing identity.
//
// Control-thread only: no method here is real-time safe.
#pragma once

#include <miniaudio.h>

#include "audio/AudioBackend.hpp"
#include "audio/AudioTypes.hpp"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class MiniaudioBackend : public AudioBackend {
public:
    MiniaudioBackend();
    ~MiniaudioBackend() override;

    MiniaudioBackend(const MiniaudioBackend&)            = delete;
    MiniaudioBackend& operator=(const MiniaudioBackend&) = delete;

    bool ok() const { return ok_; }

    /// True when context init succeeded on the null (no-hardware) backend —
    /// offline/test mode, not a working microphone (stage 03 reports it).
    bool is_null_backend() const { return ok_ && ctx_.backend == ma_backend_null; }

    fluxvoice::ContextEpoch epoch() const { return epoch_; }

    // ---- AudioBackend ----
    fluxvoice::BackendStatus status() const override;
    fluxvoice::Result<fluxvoice::DeviceSnapshot> enumerate() override;
    fluxvoice::Result<fluxvoice::EndpointInfo> probe(
        const fluxvoice::EndpointKey& key) const override;
    fluxvoice::AudioCapabilities capabilities() const override;
    fluxvoice::Result<fluxvoice::StreamInfo> open_stream(
        const fluxvoice::StreamRequest& req, AudioProcessFn process_fn,
        void* user_data) override;
    fluxvoice::Result<bool> start_stream() override;
    void stop_stream() override;
    bool is_stream_open() const override { return dev_ok_; }
    bool is_stream_running() const override;
    std::uint32_t drain_notifications() override;
    bool auto_routes_default() const override;
    std::string backend_name() const;  // concrete convenience (off the seam)

    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "RT notification bitset must be lock-free on this target");

    // ---- Key mapping (private to the audio implementation) ----

    /// Resolve a key minted by this backend to its native id. Engaged result
    /// only when the epoch and direction match and the token is known;
    /// stale/foreign keys resolve to nothing (stage 04 turns that into
    /// DeviceChanged / DeviceNotFound).
    std::optional<ma_device_id> resolve(const fluxvoice::EndpointKey& key) const;

    /// Map a raw ma_result code to a structured AudioError (stages 03–06).
    fluxvoice::AudioError make_error(ma_result r, const char* operation) const;

    /// The shared context — AudioEngine opens its stream on it.
    ma_context& context() { return ctx_; }

private:
    static void on_data(ma_device* dev, void* output, const void* input, ma_uint32 frame_count);
    static void on_notification(const ma_device_notification* notification);

    struct IdEntry {
        ma_device_id   id{};
        std::uint64_t  token = 0;
    };
    using IdTable = std::vector<IdEntry>;

    /// Return the stable token for `id`, assigning a fresh one on first sight.
    std::uint64_t intern(IdTable& table, const ma_device_id& id);

    /// Process-wide epoch source; starts at 1 so epoch 0 means "never minted".
    static fluxvoice::ContextEpoch mint_epoch() {
        static std::atomic<fluxvoice::ContextEpoch> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    ma_context ctx_{};
    bool       ok_ = false;
    ma_result  init_result_ = MA_SUCCESS;  // exact ma_context_init outcome

    ma_device  dev_{};
    bool       dev_ok_ = false;
    ma_device  play_dev_{};
    bool       play_dev_ok_ = false;
    AudioProcessFn process_fn_ = nullptr;
    void*          process_user_data_ = nullptr;
    std::atomic<std::uint32_t> pending_notifications_{0};

    fluxvoice::ContextEpoch epoch_ = 0;
    IdTable    capture_ids_;
    IdTable    playback_ids_;
    std::uint64_t next_token_ = 1;
    std::uint64_t generation_ = 0;
};
