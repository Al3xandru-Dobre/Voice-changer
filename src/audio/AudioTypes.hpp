// AudioTypes — the application-facing audio vocabulary.
//
// Everything the CLI, the future UI (Milestone 4) and the tests need in order
// to describe endpoints, selections, capabilities, failures and stream state
// — without including <miniaudio.h> or any native device identifier
// (Milestone 1.5 / Stage 02).
//
// Identity rules (research §6 "Identity and persistence rules"):
//   * EndpointKey is an opaque session key, meaningful only to the backend
//     that minted it and scoped to one direction and one context epoch. It is
//     not a serialized native id and never persists across processes.
//   * Names are labels, not keys — two identical USB microphones must remain
//     two distinct endpoints.
//   * "Not probed" is Capability::Unknown, never Capability::Unsupported.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace fluxvoice {

// ---------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------

/// Capture (input) or Playback (output). Never inferred from a device name.
enum class EndpointDirection : std::uint8_t { Capture, Playback };

inline const char* to_string(EndpointDirection d) {
    switch (d) {
        case EndpointDirection::Capture:  return "capture";
        case EndpointDirection::Playback: return "playback";
    }
    return "?";
}

/// Identifies the backend/context generation a key was minted in. Backends
/// draw epochs from a process-wide counter starting at 1, so keys from a dead
/// context can never alias keys from a recreated one (epoch 0 = never minted).
using ContextEpoch = std::uint32_t;

/// Opaque session key for one endpoint. Only the backend that minted it can
/// resolve it — MiniaudioBackend keeps the private token-to-native-id mapping
/// and compares ids with ma_device_id_equal().
struct EndpointKey {
    EndpointDirection direction{};
    ContextEpoch      epoch{};
    std::uint64_t     token{};

    friend bool operator==(const EndpointKey&, const EndpointKey&) = default;
};

/// Total order over (direction, epoch, token), consistent with operator==.
inline bool operator<(const EndpointKey& a, const EndpointKey& b) {
    if (a.direction != b.direction) return a.direction < b.direction;
    if (a.epoch != b.epoch) return a.epoch < b.epoch;
    return a.token < b.token;
}

///Tri-state: Unknown means "not probed", never "probed and unsupported".
enum class Capability : std::uint8_t { Unknown, Supported, Unsupported };

/// One enumerated endpoint, as shown in a device list. The direction is
/// carried by `key`; `name` is a display label only, never an identity.
struct EndpointDescriptor {
    EndpointKey key{};
    std::string name;
    bool is_default = false;
    /// Optional versioned, backend-namespaced token for persistence
    /// (stage 04); empty = none.
    std::string persistent_token;
    /// Native server metadata classification for sink monitor sources
    /// (stage 06): Unknown unless populated by native server metadata —
    /// never inferred from a ".monitor" or "Monitor of " name suffix.
    Capability is_monitor = Capability::Unknown;
    /// When `is_monitor == Capability::Supported`, the playback sink key
    /// this monitor records, if provided by the backend/adapter.
    std::optional<EndpointKey> monitor_of;
};

/// An owned, immutable copy of the whole device catalog at one instant.
/// `generation` grows monotonically with every published snapshot; keys
/// minted in `epoch` (also embedded in every key) are valid only within it.
struct DeviceSnapshot {
    std::vector<EndpointDescriptor> capture;
    std::vector<EndpointDescriptor> playback;
    std::string backend;          // "wasapi", "coreaudio", "pulseaudio", "null", ...
    ContextEpoch  epoch{};
    std::uint64_t generation{};   // 0 = no snapshot published yet

    const std::vector<EndpointDescriptor>& endpoints(EndpointDirection d) const {
        return d == EndpointDirection::Capture ? capture : playback;
    }
};

// ---------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------

/// Selecting a device means selecting a *policy*, not a row number.
struct FollowSystemDefault {               // open with a null device id
    friend bool operator==(const FollowSystemDefault&, const FollowSystemDefault&) = default;
};

struct ExplicitEndpoint {
    EndpointKey key{};
    /// Optional snapshot generation this selection was made against (0 =
    /// resolve by key against current snapshot within the same epoch).
    std::uint64_t generation = 0;
    friend bool operator==(const ExplicitEndpoint&, const ExplicitEndpoint&) = default;
};

using DeviceSelection = std::variant<FollowSystemDefault, ExplicitEndpoint>;

/// How the capture side of a stream is sourced (M1.5 stage 06).
/// OrdinaryMicrophone opens a Capture endpoint; SystemPlaybackLoopback
/// captures a Playback endpoint via native loopback (WASAPI
/// ma_device_type_loopback), gated by AudioCapabilities::native_loopback.
enum class CaptureMode : std::uint8_t {
    OrdinaryMicrophone,
    SystemPlaybackLoopback,
};

inline const char* to_string(CaptureMode m) {
    switch (m) {
        case CaptureMode::OrdinaryMicrophone:     return "microphone";
        case CaptureMode::SystemPlaybackLoopback: return "system-playback loopback";
    }
    return "?";
}

// ---------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------

/// What the current backend can do. Ordinary I/O is known once a context
/// exists; the rest are probed lazily where the backend can see them at all.
struct AudioCapabilities {
    Capability ordinary_io     = Capability::Unknown;  // capture + playback
    Capability native_loopback = Capability::Unknown;  // system-playback capture (WASAPI in miniaudio 0.11.25)
    Capability monitor_capture = Capability::Unknown;  // server monitor sources as capture (PulseAudio/PipeWire)
    Capability virtual_routing = Capability::Unknown;  // app-configured virtual cable/sink route
};

// ---------------------------------------------------------------------
// Errors & results
// ---------------------------------------------------------------------

enum class AudioErrorCategory : std::uint8_t {
    BackendUnavailable,  // no audio backend could be initialised
    NullBackendOnly,     // context came up on the null (no-hardware) backend
    EnumerationFailed,   // device listing failed
    DeviceNotFound,      // a key no longer resolves to an endpoint
    DeviceChanged,       // selection was made against an older snapshot
    DeviceBusy,          // exclusive use by another application / server loss
    PermissionDenied,    // OS consent missing
    FormatUnsupported,   // requested format rejected by the endpoint
    Unknown,
};

inline const char* to_string(AudioErrorCategory c) {
    switch (c) {
        case AudioErrorCategory::BackendUnavailable: return "backend unavailable";
        case AudioErrorCategory::NullBackendOnly:    return "null backend only";
        case AudioErrorCategory::EnumerationFailed:  return "enumeration failed";
        case AudioErrorCategory::DeviceNotFound:     return "device not found";
        case AudioErrorCategory::DeviceChanged:      return "device changed";
        case AudioErrorCategory::DeviceBusy:         return "device busy";
        case AudioErrorCategory::PermissionDenied:   return "permission denied";
        case AudioErrorCategory::FormatUnsupported:  return "format unsupported";
        case AudioErrorCategory::Unknown:            return "unknown";
    }
    return "unknown";
}

/// Everything known about one failed audio operation.
struct AudioError {
    AudioErrorCategory category = AudioErrorCategory::Unknown;
    std::string operation;    // "enumerate devices", "open capture", ...
    int native_result = 0;    // raw backend result (ma_result) as a plain int; 0 = none
    std::string backend;      // backend name; empty when no context exists
    std::string message;      // user-facing explanation / suggested next step
};

/// C++20 result type (std::expected arrives with C++23).
template <typename T>
using Result = std::variant<T, AudioError>;

/// True when `r` holds a value (alternative 0) rather than an AudioError.
template <typename T>
inline bool is_ok(const Result<T>& r) { return r.index() == 0; }

// ---------------------------------------------------------------------
// Backend status, per-endpoint probes & resolved selections (stages 03–04)
// ---------------------------------------------------------------------

/// Context-level outcome: what happened at initialisation and which backend
/// was selected. `native_result` carries the exact backend init code when
/// Unavailable (0 otherwise); `backend` names the selected backend on
/// success and is empty when none could be created.
enum class BackendState : std::uint8_t {
    Unavailable,  // context init failed — exact native result preserved
    Ready,        // a real backend is selected
    NullOnly,     // the null (no-hardware) backend — offline/test mode
};

struct BackendStatus {
    BackendState state = BackendState::Unavailable;
    std::string backend;
    int native_result = 0;
};

/// One native data-format entry of an endpoint. 0 channels / 0 sample rate
/// mean "any", following the backend's convention; `format` is the backend
/// sample-format enum value as a plain int (0 = unknown).
struct NativeFormat {
    std::uint32_t sample_rate = 0;
    std::uint32_t channels = 0;
    int format = 0;
};

/// On-demand per-endpoint probe (stage 03): the backend's device-info query,
/// never an opened stream. Used to inspect a *selected* device only — never
/// to populate a device list.
struct EndpointInfo {
    EndpointKey key{};
    std::string name;
    bool is_default = false;
    std::vector<NativeFormat> native_formats;  // may be empty when the backend reports none
};

/// A selection resolved against one published snapshot (stage 04): either
/// "open the OS default" or one concrete endpoint, plus the snapshot
/// generation it was resolved against (so the controller can detect races).
struct ResolvedEndpoint {
    bool use_default = false;
    EndpointKey key{};             // the concrete endpoint when !use_default
    std::uint64_t generation = 0;  // snapshot generation resolved against
    std::string name;              // display name of the concrete endpoint ("" for default)
};

// ---------------------------------------------------------------------
// Stream configuration, notifications & status
// ---------------------------------------------------------------------

/// Lock-free notification hint bitset (M1.5 stage 05). Set by the backend's
/// notificationCallback via atomic fetch_or; drained on the control thread.
namespace NotificationBits {
constexpr std::uint32_t kNone              = 0;
constexpr std::uint32_t kStarted           = 1u << 0;
constexpr std::uint32_t kStopped           = 1u << 1;
constexpr std::uint32_t kRerouted          = 1u << 2;
constexpr std::uint32_t kInterruptionBegan = 1u << 3;
constexpr std::uint32_t kInterruptionEnded = 1u << 4;
constexpr std::uint32_t kUnlocked          = 1u << 5;
} // namespace NotificationBits

/// Parameters for opening a stream on an AudioBackend (stages 05–06).
struct StreamRequest {
    CaptureMode capture_mode        = CaptureMode::OrdinaryMicrophone;
    const EndpointKey* capture_key  = nullptr;  // nullptr = OS default
    const EndpointKey* playback_key = nullptr;  // nullptr = OS default
    std::uint32_t sample_rate       = 16000;
    std::uint32_t period_ms         = 20;
    bool passthrough                = false;
};

/// What was actually opened — filled when a stream is initialised.
/// `client_sample_rate` is the rate our callback sees; the `*_internal_*`
/// fields are the backend-side values (the OS audio server may still convert
/// further). Mixing the two up is the classic "chipmunk voice" bug.
struct StreamInfo {
    std::string backend;
    std::string capture_name;
    std::string playback_name;
    std::uint32_t client_sample_rate       = 0;
    std::uint32_t capture_internal_rate    = 0;
    std::uint32_t playback_internal_rate   = 0;
    std::uint32_t capture_internal_period  = 0;  // frames
    std::uint32_t playback_internal_period = 0;  // frames
};

/// Controller-side stream lifecycle (M1.5 stage 05). Never infer liveness
/// from a device flag or from "samples are non-zero" — the user may be silent.
enum class StreamState : std::uint8_t {
    Idle, Opening, Running, Recovering, AwaitingUser, Error, Stopping,
};

inline const char* to_string(StreamState s) {
    switch (s) {
        case StreamState::Idle:         return "idle";
        case StreamState::Opening:      return "opening";
        case StreamState::Running:      return "running";
        case StreamState::Recovering:   return "recovering";
        case StreamState::AwaitingUser: return "awaiting user";
        case StreamState::Error:        return "error";
        case StreamState::Stopping:     return "stopping";
    }
    return "?";
}

/// Published view of the stream: what was requested, what actually opened,
/// and the client-vs-backend formats (mixing the two up is the classic
/// "chipmunk voice" bug — see README §4).
struct StreamStatus {
    StreamState state = StreamState::Idle;

    DeviceSelection requested_capture{FollowSystemDefault{}};
    DeviceSelection requested_playback{FollowSystemDefault{}};
    std::optional<EndpointKey> resolved_capture;   // endpoint actually open, when queryable
    std::optional<EndpointKey> resolved_playback;

    std::string backend;
    std::uint32_t client_sample_rate    = 0;  // the rate our code sees
    std::uint32_t capture_backend_rate  = 0;  // device/backend-side rates
    std::uint32_t playback_backend_rate = 0;

    CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone;
    std::string capture_name;
    std::string playback_name;
    std::uint32_t capture_backend_period  = 0;
    std::uint32_t playback_backend_period = 0;
    std::uint64_t selection_generation    = 0;
    bool backend_stream_started           = false;
    std::uint64_t callback_progress       = 0;
    std::uint32_t retry_attempt           = 0;
    std::optional<AudioError> last_error;
};

} // namespace fluxvoice
