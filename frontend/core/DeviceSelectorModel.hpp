// DeviceSelectorModel — turns an immutable DeviceSnapshot into dropdown rows
// and a chosen row back into a DeviceSelection *policy* that carries the
// snapshot generation (Milestone 4 scope, M1.5 stages 03–06 rules):
//   * row 0 is always "Follow system default";
//   * an explicit row becomes ExplicitEndpoint{key, generation}, never an index;
//   * in system-playback loopback mode the capture rows come from the playback
//     list, because native loopback records a render endpoint;
//   * the feedback guard is previewed on the names the user is about to open.
#pragma once

#include "audio/AudioTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

struct DeviceRow {
    std::string label;             // what the dropdown shows
    bool        follow_default = false;
    EndpointKey key{};
    bool        is_default = false;
    Capability  is_monitor = Capability::Unknown;
};

struct DeviceRows {
    std::vector<DeviceRow> rows;
    std::uint64_t generation = 0;
    std::string backend;
    bool stale = false;
};

/// Rows for one direction. `direction` is the *stream* direction; with
/// `CaptureMode::SystemPlaybackLoopback` the Capture rows are built from the
/// snapshot's playback endpoints.
DeviceRows build_rows(const DeviceSnapshot& snap, EndpointDirection direction, bool stale,
                      CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone);

/// Row -> policy. Out-of-range rows fall back to FollowSystemDefault.
DeviceSelection selection_for_row(const DeviceRows& rows, std::size_t row);

/// Policy -> row index in `rows`, or nullopt when an explicit key is not listed
/// (the UI then shows "device no longer listed" instead of picking another).
std::optional<std::size_t> row_for_selection(const DeviceRows& rows, const DeviceSelection& sel);

/// Human text for a policy against a snapshot ("Follow system default",
/// the device name, or "(device no longer listed)").
std::string describe_selection(const DeviceSelection& sel, const DeviceSnapshot& snap,
                               EndpointDirection direction,
                               CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone);

/// Name the policy would open right now (default endpoint's name for
/// FollowSystemDefault), or empty when unknown.
std::string resolved_name(const DeviceSelection& sel, const DeviceSnapshot& snap, EndpointDirection direction,
                          CaptureMode capture_mode = CaptureMode::OrdinaryMicrophone);

/// Preview of the feedback guard (RouteGuard.hpp) for the pair the user is
/// about to apply. Engaged with an explanation when the route would feed
/// FluxVoice's output back into its input.
std::optional<std::string> feedback_preview(const DeviceSnapshot& snap, const DeviceSelection& capture,
                                            const DeviceSelection& playback, CaptureMode capture_mode);

/// Short user-facing text for a controller/registry failure plus the next step.
std::string explain_error(const AudioError& err);

} // namespace fluxvoice::frontend
