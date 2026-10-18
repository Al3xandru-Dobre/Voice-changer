// AudioDeviceRegistry — owned, immutable device-catalog snapshots plus
// selection resolution over one AudioBackend (Milestone 1.5 / Stages 03–04).
//
// Invariants (research §5–§7, BEST_PRACTICES §2):
//   * one backend/context per registry, supplied by the caller and outliving
//     it (AudioEngine owns the MiniaudioBackend today; stage 05's
//     AudioController takes over the whole lifecycle) — the context address
//     is stable and every call runs on the control thread;
//   * a new snapshot `generation` is published only when endpoint content
//     (keys, names, default flags) actually changes — a reordered catalog is
//     not a change;
//   * a failed refresh retains the previous snapshot, marked stale — never
//     replaced by an empty or "fresh" one;
//   * the null backend is reported as offline mode (NullBackendOnly), never
//     as a working empty catalog;
//   * per-device probing happens only on demand for a selected endpoint —
//     never to populate a list, and never by opening a stream;
//   * resolving a selection NEVER falls back: a follow-default selection
//     opens the OS default, an explicit key must still name its own device,
//     and stale/vanished keys fail with DeviceChanged / DeviceNotFound.
//
// Control-thread only.
#pragma once

#include "audio/AudioBackend.hpp"
#include "audio/AudioTypes.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace fluxvoice {

class AudioDeviceRegistry {
public:
    /// `backend` must outlive the registry (see class comment).
    explicit AudioDeviceRegistry(AudioBackend& backend) : backend_(backend) {}

    AudioDeviceRegistry(const AudioDeviceRegistry&)            = delete;
    AudioDeviceRegistry& operator=(const AudioDeviceRegistry&) = delete;

    /// Backend/context status; the exact init failure code is preserved in
    /// BackendStatus::native_result ("init()" per the stage plan — the
    /// context itself is created by the backend, once, address-stable).
    BackendStatus status() const { return backend_.status(); }

    /// Backend-level capabilities (ordinary I/O, native loopback, etc.).
    AudioCapabilities backend_capabilities() const { return backend_.capabilities(); }

    /// Re-enumerate both directions in one serialized call and publish a new
    /// snapshot if content changed. Returns the current snapshot, or the
    /// error; on error the previous snapshot stays available via snapshot().
    Result<DeviceSnapshot> refresh();

    /// Last published snapshot (generation 0 = nothing published yet).
    const DeviceSnapshot& snapshot() const { return published_; }

    /// True when the last refresh failed but an older snapshot is retained.
    bool stale() const { return stale_; }

    /// On-demand native-format probe for one endpoint (stage 03). Opens no
    /// stream; the key must belong to this backend's epoch.
    Result<EndpointInfo> capabilities(const EndpointKey& key) const;

    /// Resolve a selection policy against the published snapshot (stage 04).
    /// FollowSystemDefault -> "open the OS default"; ExplicitEndpoint -> the
    /// same concrete endpoint wherever it now sits in the catalog. A key from
    /// another context epoch fails with DeviceChanged; a vanished device
    /// fails with DeviceNotFound — a selection is never redirected to
    /// whatever now occupies its old position.
    /// When `strict_generation` is true and an ExplicitEndpoint carries a
    /// non-zero `generation` that differs from `snapshot().generation`,
    /// returns `DeviceChanged` (detecting listing-to-open races).
    Result<ResolvedEndpoint> resolve(const DeviceSelection& sel, EndpointDirection dir,
                                     bool strict_generation = false) const;

    /// Resolve a system-playback loopback capture selection (stage 06).
    /// Gated by `backend.capabilities().native_loopback == Capability::Supported`
    /// and resolves against Playback endpoints (`ma_device_type_loopback`
    /// captures a render endpoint).
    Result<ResolvedEndpoint> resolve_loopback(const DeviceSelection& sel,
                                              bool strict_generation = false) const;

    /// Resolve a persisted selection token ("default" or a versioned
    /// "v1:<backend>:<direction>:<name>" token, stage 04). Fails with
    /// DeviceChanged if the token is ambiguous (e.g. two identical USB mics)
    /// or from another backend, and DeviceNotFound if absent.
    Result<DeviceSelection> resolve_persistent_token(std::string_view token,
                                                     EndpointDirection dir) const;

    /// Find the descriptor for `key` in the published snapshot, or nullptr.
    const EndpointDescriptor* find_descriptor(const EndpointKey& key) const;

    /// Find the current default descriptor for `dir` in the published snapshot, or nullptr.
    const EndpointDescriptor* default_descriptor(EndpointDirection dir) const;

private:
    /// Order-insensitive catalog comparison by key (keys are unique within a
    /// snapshot, so equal size + every a-side entry found means a bijection).
    static bool same_catalog(const std::vector<EndpointDescriptor>& a,
                             const std::vector<EndpointDescriptor>& b);

    AudioBackend& backend_;
    DeviceSnapshot published_{};
    std::uint64_t generation_ = 0;
    bool stale_ = false;
};

} // namespace fluxvoice

using fluxvoice::AudioDeviceRegistry;
