# Cross-platform audio input/output discovery in C++

- **Project:** FluxVoice-S2UT
- **Research date:** 2026-10-17
- **Status:** Research and proposed implementation plan; no implementation changes

> **Implementation tracking.** This report is preserved as a dated research
> snapshot. Its §10 plan is scheduled in
> [Milestone 1.5 — Cross-platform audio devices](../Milestone%201.5/00_master.md)
> (research stages 1–4 and 6 → M1.5 stages 02–06; research stage 5 → Milestone 4
> UI). Corrections it raised against existing plans were folded into Milestone 1
> stages 02, 03, 05 and 06. Where this snapshot and a stage file disagree, the
> stage file is authoritative.
>
> Superseded since writing: §10's note that "the current CMake file does not
> define test targets" — CTest was added in Milestone 1 / Stage 01.

## 1. Recommendation

Keep **miniaudio**, already pinned to **0.11.25**, for device enumeration and
ordinary capture/playback. Add a small C++ device registry and stream controller
around it so the CLI, future UI, and inference pipeline do not depend on native
OS audio APIs or miniaudio device IDs.

Use one application-facing API on Windows, macOS, and Linux, while acknowledging
that the underlying backends, permission flows, and routing facilities differ.
“OS-agnostic” should mean **portable application code and consistent behavior**,
not identical audio capabilities or an absence of platform-specific packaging.

The recommended first delivery is:

1. Discover capture and playback endpoints, including OS defaults.
2. Select an endpoint reliably without depending on its position or name.
3. Refresh the catalog after hardware/default-device changes.
4. Recover safely from device loss and report actionable errors.
5. Keep system-playback capture and virtual-microphone setup as explicit,
   capability-gated features rather than pretending they work uniformly.

Replacing miniaudio is not justified by the current requirements.

## 2. Scope and evidence

### What “audio detection” means here

| Requirement | Meaning | Scope |
|---|---|---|
| Input discovery | List microphone, interface, and virtual capture endpoints | Core |
| Output discovery | List speaker, headphone, interface, and virtual playback endpoints | Core |
| Default detection | Identify the backend's default input and output | Core |
| Change detection | Notice endpoint additions/removals and default changes | Core |
| System-playback capture | Capture the audio being played by other applications | Optional, backend-dependent |
| Virtual microphone | Make processed audio selectable as an input in another application | Product routing requirement; separate setup |
| Signal/activity detection | Determine whether a selected stream contains sound | Separate DSP/metering work, not device discovery |

A capture endpoint is not necessarily a physical microphone, and a playback
endpoint is not necessarily a physical speaker. A single hardware interface may
appear as several endpoints. Discovery should expose what the audio backend
provides, not attempt to reconstruct a universal physical-hardware inventory.

### Research method and limitations

This report is based on a **local source/documentation audit**:

- The project's C++ source, CMake configuration, and milestone plans.
- The fetched miniaudio **0.11.25** header, examples, README, and license.
- The installed PortAudio public header.
- Installed PipeWire **1.0.5** manuals describing PulseAudio compatibility and
  virtual-device modules.

No network requests, device recordings, driver installations, or audio-server
configuration changes were performed. Online documentation access was requested
but has not been approved at the time of writing. External links below are
reading/verification targets, **not claims of live web verification**. This is
not a survey of the latest library releases.

Library behavior verified in the local sources is distinguished below from
proposed application policy. Exact current Apple permission requirements,
third-party driver distribution terms, and unaudited library APIs still need
upstream verification before implementation or release.

## 3. Current project baseline

| Location | Existing behavior | Gap relevant to this work |
|---|---|---|
| [`CMakeLists.txt`](../../CMakeLists.txt) | C++20; miniaudio 0.11.25 and ReaderWriterQueue v1.0.7; OS-specific link settings | Keep dependency pinning; add tests when implementing |
| [`AudioEngine.hpp`](../../src/audio/AudioEngine.hpp) | Owns a context; enumerates both directions; copies IDs/names/default flags; starts mono float32 duplex audio | No catalog snapshots, refresh policy, typed failures, or device-change handling |
| [`main.cpp`](../../src/main.cpp) | `--list-devices`, `--input`, `--output`; omitted IDs use defaults | Selection indices are temporary; invalid enumeration/start states have little diagnostic detail |
| [`AppWindow.cpp`](../../src/ui/AppWindow.cpp) | Placeholder for the future UI | Device dropdowns should consume the same registry as the CLI |
| [M1 duplex plan](../Milestone%201/02_duplex_audio_io.md) | Basic device enumeration and passthrough | Extend with variable-frame, identity, and lifecycle tests |
| [M1 virtual-cable plan](../Milestone%201/05_virtual_cable_loopback.md) | Windows/macOS cable and Linux sink/monitor routing | Setup remains platform-specific; client/native format terminology needs clarification |

The existing wrapper correctly copies `ma_device_id` values instead of retaining
miniaudio's enumeration pointers. Keep this behavior. Its empty-list and `false`
return values, however, currently obscure whether an operation failed or
successfully found no devices.

## 4. Options considered

| Option | Fit for this project | Costs / limitations | Decision |
|---|---|---|---|
| **miniaudio** | Already integrated; portable C API callable from C++; capture/playback, enumeration, conversion, notifications | No native PipeWire backend in this pinned version; native loopback only on WASAPI; application must define identity and recovery policy | **Use now** |
| **PortAudio** | Installed header confirms device counts, defaults, device information, streams, and format-support queries | Replaces working integration; integer device indices do not provide application-level persistence or solve routing/permissions | Viable alternative if a measured backend problem justifies migration |
| **RtAudio** | A C++ audio-I/O library worth evaluating if a replacement is needed | Not integrated or audited here; current backend coverage, identifier lifetime, notifications, and license must be checked | Shortlist only, not a verified recommendation |
| **JUCE audio devices** | Potential broader application/audio-framework option | Not integrated or audited here; larger architectural change for a project already planning Dear ImGui; current license/distribution terms need review | Revisit only if broader framework requirements emerge |
| **Native APIs everywhere** | Maximum access to each OS's routing and notification facilities | Multiple implementations, more lifecycle/threading code, and a larger test surface | Use only for gaps behind narrow optional adapters |

miniaudio's local license offers a choice of public-domain dedication or MIT
No Attribution (MIT-0). The installed PortAudio header contains an MIT-style
license requiring preservation of its notice. Review exact versions and all
distribution components before shipping; an audio-I/O library's license does
not cover a separately installed virtual driver.

No new dependency is needed for the proposed discovery layer. Use C++20 standard
types and the queue library already present.

## 5. Verified miniaudio capabilities and boundaries

### Backend matrix for the pinned version

| Platform/path | Ordinary discovery and capture/playback | Capturing system playback | Sending processed audio to another app |
|---|---|---|---|
| Windows / WASAPI | Supported by miniaudio | Native `ma_device_type_loopback`; selects a playback endpoint as the capture target | Write to an installed virtual cable's playback endpoint; the other app selects its corresponding capture endpoint |
| macOS / Core Audio | Supported by miniaudio | **Not** provided by miniaudio's native loopback mode in 0.11.25; use an explicitly configured virtual route or investigate a native adapter | Use an installed/configured virtual audio device |
| Linux / PulseAudio | Supported by miniaudio | Capture a sink's monitor source as an ordinary capture endpoint when exposed by the server | Write to a virtual sink; the other app may select its monitor or a separately configured virtual source |
| Linux / PipeWire via `pipewire-pulse` | Uses miniaudio's PulseAudio backend when the compatibility server/client library is available | Same monitor-source approach where exposed and permitted | PipeWire/Pulse-compatible routing; validate with the receiving application |
| Linux / ALSA fallback | Ordinary ALSA enumeration/I/O | No general desktop-mix capture promise | No promise of access to the desktop server's virtual devices |

These are API/routing findings, **not completed hardware certifications**.

Important details from the audited sources:

- `ma_context_init(nullptr, 0, ...)` tries available backends in priority order
  and selects one context backend. It does not merge inventories from every
  backend. Linux prioritizes PulseAudio ahead of ALSA in this version. [MA1]
- There is no `ma_backend_pipewire` in 0.11.25. The installed `pipewire-pulse`
  manual describes a PulseAudio-compatible replacement server, which explains
  the compatibility route; it is not native PipeWire integration. [PW1]
- The default backend search includes the **null backend**. Successful context
  initialization alone is not proof that real audio hardware is available.
  Report null-backend operation as test/offline mode, not a working microphone.
  [MA1]
- `ma_context_is_loopback_supported()` reports native loopback support for the
  selected context. In this version, the built-in backend implementation returns
  true only for WASAPI. A false result does **not** mean a Linux monitor source
  cannot be opened as ordinary capture. [MA1, MA2]
- The WASAPI loopback example configures `capture.pDeviceID` with a
  **playback-device ID**. It is not a microphone ID. [MA2]
- miniaudio's WASAPI default-endpoint code uses the Windows **`eConsole` role**.
  Do not label this “default communications microphone” or assume that role is
  equivalent. Add a native role adapter only if that becomes a requirement.
  [MA1]

### Enumeration is a snapshot, not an availability guarantee

`ma_context_get_devices()` returns context-owned arrays that become invalid on
the next call or context destruction. Its API documentation explicitly warns
against concurrent use of these returned arrays. Make an application-owned copy
before another refresh, and serialize enumeration on the control thread. Retrieve
both input and output lists in one call to reduce inconsistent snapshots. [MA1]

Basic enumeration does not establish all supported formats, permission status,
or whether another application holds a device exclusively. Use
`ma_context_get_device_info()` for selected-device capability queries outside
the enumeration callback; the actual open/start result remains authoritative.
Do not probe every microphone by starting capture just to populate a dropdown.

### Notifications do not replace catalog refresh

`notificationCallback` describes changes to an **opened stream**: started,
stopped, rerouted, and interruptions. It is not a portable subscription to every
device being connected to the system. Notification coverage varies by backend;
some reroutes are not reported, and a stopped event does not accompany every
reroute. Never restart or uninitialize a device inside this callback. [MA1]

Use notifications as refresh/recovery hints, with periodic re-enumeration for a
portable baseline.

## 6. Proposed C++ architecture

```text
CLI / future ImGui UI
        |
        v
AudioDeviceRegistry + AudioController          control thread
        |
        +---- MiniaudioBackend                 default implementation
        |       context, IDs, enumeration, stream lifecycle
        |
        +---- optional platform adapters       only for verified gaps
                permissions, native system capture, routing setup

capture callback -> existing input SPSC -> inference worker
inference worker -> existing output SPSC -> playback callback
```

Start with concrete classes and a small backend boundary that tests can fake.
Do not build a plugin system or require three native implementations. Keep
`<miniaudio.h>` and native identifiers inside the audio implementation rather
than exposing them to the UI or inference code.

### Application-facing data contract

| Type / field | Purpose |
|---|---|
| `EndpointDirection` | Capture or playback; do not infer direction from a name |
| `EndpointKey` | Opaque session key, scoped to backend/context epoch and direction |
| `EndpointDescriptor` | Key, display name, direction, default flag, optional persistent token |
| `DeviceSnapshot` | Owned capture/playback descriptors, backend identity, monotonically increasing generation |
| `DeviceSelection` | Explicit `FollowSystemDefault` or `ExplicitEndpoint(key)` choice |
| `AudioCapabilities` | Separate status for ordinary I/O, native loopback, monitor capture, and configured virtual routing |
| `AudioError` | Operation, category, original `ma_result`, backend, and a useful user-facing explanation |
| `StreamStatus` | Controller state, requested endpoint, resolved endpoint where queryable, and client/backend formats |

Represent unknown capability metadata explicitly; do not turn “not probed” into
“unsupported.” Permissions and device availability are runtime outcomes, not
permanent hardware capabilities. In C++20, use a small result type or
`std::variant<T, AudioError>` rather than requiring C++23 `std::expected`.

### Identity and persistence rules

1. **Indices are presentation only.** Keep existing CLI index flags for manual
   smoke tests, but resolve them through a current snapshot before opening.
   A future UI submits a key plus snapshot generation, never a row number alone.
2. **Names are labels, not keys.** Two USB microphones may have identical names;
   descriptions can change with profiles, localization, or user settings.
3. **Keep native IDs private.** Compare miniaudio-provided IDs within the same
   backend/context epoch using `ma_device_id_equal()`. Do not serialize or hash
   the raw union as a portable, permanent identifier.
4. **Invalidate stale sessions.** Recreating the context or changing backends
   invalidates old session keys. A refresh/open race must return “device changed”
   or retry resolution safely, never select whatever now occupies the old index.
5. **Persist policy first.** Persist “follow default” directly. For explicit
   endpoints, use a versioned backend-native token only where supported and
   tested, with backend/direction namespace. A token is best-effort, not proof
   of physical identity across reinstalls, profile changes, or server restarts.
   If it cannot be resolved unambiguously, ask the user to select again.
6. **Default is not a concrete endpoint.** Opening a default selection passes
   a null device ID. Opening a fixed selection passes the copied current ID.
   Track requested policy separately from the endpoint actually in use.

## 7. Refresh, selection, and recovery policy

### Proposed refresh algorithm

1. Initialize the backend/context and preserve the exact failure result.
2. Enumerate both directions, copy results, and publish an owned snapshot.
3. Refresh on user request, notification hints, and periodically while the app
   is active. Start with a configurable **one-second interval**; this is an
   application policy target, not a library timing guarantee.
4. Compare IDs, direction, names, and default flags with the previous snapshot.
   Publish a new generation only when relevant data changes.
5. Coalesce events and retry temporary enumeration errors with bounded backoff.
   On a refresh error, retain the last snapshot as **stale**, not fresh/empty.
6. Reject stale selections or re-resolve them under the controller's policy
   before attempting a stream open.

All enumeration, capability queries, open/close, and retries run on one control
thread. UI access uses copied/immutable snapshots with ordinary synchronization
outside the audio callback.

### Stream lifecycle

```text
Idle -> Opening -> Running
                    |
                    v
                Recovering -> Opening
                    |
                    v
              AwaitingUser / Error

Running / Recovering / Error -> Stopping -> Idle
```

Do not use the current `dev_ok_` initialization flag as the sole indication that
audio is still flowing. Track controller state, notifications, backend state,
and callback progress for diagnostics. Lack of nonzero samples is not a failure:
the user may simply be silent.

| Event | Proposed behavior |
|---|---|
| Default selection; OS default changes | Honor follow-default policy; account for backend auto-routing before reopening, avoiding duplicate restarts |
| Explicit device disappears | Stop that route and show the missing selection; do not silently record a different microphone |
| Virtual output disappears | Stop output and request repair; never silently redirect processed speech to speakers |
| Permission denied | Explain the required user action; do not retry continuously or substitute a different source |
| Device busy / temporary server loss | Bounded retries; show backend/reason and offer a deliberate alternative |
| Context/backend must be recreated | Stop streams, invalidate session keys, rebuild inventory, re-resolve saved policy |
| User requests another endpoint | Perform a serialized stop/reconfigure/start transition |

Preserve the backend's distinction between default and fixed routing. The
audited WASAPI code enables automatic routing only for null/default selections;
the PulseAudio code sets `DONT_MOVE` for explicitly named endpoints. Validate
these behaviors with the target audio server rather than relying on polling
alone to prevent unwanted fallback. Strong routing guarantees require checking
backend/server behavior, not just avoiding fallback in application code. [MA1]

For stop/reconfigure/start, first quiesce the worker and callbacks, then discard
old queued PCM and reinitialize stream-specific DSP/OLA state. Define whether a
capture discontinuity also resets model state before reconnecting the inference
pipeline. Do not let a previous stream's samples leak into the new route.
Keep `ma_context`/`ma_device` addresses stable throughout their lifetimes.

Use a lock-free atomic pending-event flag/bitset in callbacks, with diagnostics
assembled later on the control thread. Do not assume notifications have a single
producer or feed them into an SPSC queue from multiple backend threads. Verify
callback atomics are lock-free on supported build targets.

## 8. Formats, clocks, and real-time constraints

Keep the model-facing format independent of hardware format: mono float32 at the
selected model's rate, initially the existing 16 kHz path. Let miniaudio perform
device-side conversion for the basic duplex path. Log:

- Requested and resulting **client** format/rate/channel count.
- Backend-side capture and playback rates, channels, formats, and period sizes.
- Selected backend, endpoint-selection policy, stream state, and error codes.

In particular, **`ma_device.sampleRate` is the client-side rate**, not necessarily
the hardware/backend rate. Inspect the pinned version's
`capture.internalSampleRate` and `playback.internalSampleRate` for backend-side
values; those still need not describe every conversion performed by an OS audio
server. The existing virtual-cable plan's wording about the “negotiated rate”
should be clarified when implementation begins.

Do not build model chunk boundaries from callback counts. The header describes
fixed-sized callback configuration, but also warns that requested period sizes
are hints. Always process the supplied `frameCount` and assemble exact model
chunks from accumulated samples in the worker. At 16 kHz, an 80 ms chunk is
**1,280 frames**, regardless of callback scheduling.

Preserve the existing real-time rules:

- No allocation, filesystem/network access, logging, mutexes, or lifecycle calls
  in audio callbacks.
- Keep the input queue capture-producer/worker-consumer and the output queue
  worker-producer/playback-consumer; UI and refresh threads never consume PCM.
- Zero-fill playback underruns and count overflow/underrun samples.
- Preallocated queue capacity is a safety limit, not a target buffering delay.
  Bound queued-audio latency and define an overload/drop policy.

Native loopback capture and monitoring playback may require **separate streams**
rather than the current single duplex device. The present callback assumes a
non-null output buffer, so it cannot simply be reused for capture-only/loopback
mode without restructuring. Route these modes explicitly and test null-buffer
contracts.

Independent devices can also have independent clocks. Format conversion at
nominal sample rates alone does not prove long-term drift correction. Measure
queue occupancy and latency during a soak test with mismatched physical/virtual
devices. If needed, add a controlled resampling/clock-recovery stage; do not hide
drift by indefinitely increasing buffers.

## 9. System playback, virtual routing, and permissions

### System-playback capture

- **Windows:** provide a separate “system output” capture mode backed by WASAPI
  loopback, gated by `ma_context_is_loopback_supported()`. Treat render-endpoint
  selection and normal microphone selection as different operations.
- **Linux:** expose server-provided monitor sources as capture choices. miniaudio's
  PulseAudio enumeration copies source names/descriptions but does not expose a
  universal “this is a monitor” field in the application descriptor. Name
  suffixes alone are not a reliable classification contract; use native server
  metadata in an optional adapter if automatic classification is required.
- **macOS:** do not promise native system capture through miniaudio. A configured
  virtual route can be selected as ordinary input; alternatives such as Core
  Audio process taps or ScreenCaptureKit require a separate Apple-documentation,
  minimum-OS-version, permission, and implementation investigation.

Capturing one render endpoint is not a promise to capture every application's
audio, every output device, protected content, or isolated per-application audio.
Prevent feedback when the application's own output returns through the selected
monitor/cable. Do not automatically select the app's virtual output as its input.

### Virtual microphone / processed output

```text
physical mic -> FluxVoice processing -> virtual playback endpoint
                                        |
                                        v
                              virtual capture endpoint -> receiving app
```

miniaudio opens existing endpoints; it does **not** install an OS-wide microphone
driver. The project already proposes VB-Cable on Windows, BlackHole on macOS,
and a virtual sink/monitor on Linux. Validate each route explicitly; third-party
installation, signing, licensing, and redistribution remain separate concerns.

On Linux, a monitor source may work directly in the receiving app, but some apps
filter monitor inputs or need an explicitly configured virtual source. The local
PipeWire manuals confirm null-sink and virtual-source modules exist, not that
every receiving app presents them identically.

Any routing setup helper should be a separate, opt-in operation. Do not install
drivers, change system defaults, create persistent server configuration, or run
privileged commands during enumeration.

### Permissions and privacy

Enumerate first; start capture only on explicit user action. Treat permission
failure separately from “no devices found.” Packaging must account for macOS
microphone usage descriptions and relevant entitlements/consent flows, Windows
privacy settings and packaging rules, and Linux sandbox/session restrictions.
Exact platform requirements must be checked against official documentation for
the supported OS versions; this audit did not verify them online.

Do not circumvent platform consent or request broad access just to populate a
list. Normal diagnostics should not include audio, and persisted device tokens
should remain local unless the user intentionally exports diagnostics.

## 10. Implementation plan

The following stages are proposed work, not completed tasks or time estimates.

| Stage | Work | Intended locations | Completion criteria |
|---|---|---|---|
| 1 — Define the contract | Add descriptors, keys, selection policy, snapshots, and typed errors; isolate native IDs | New `src/audio/AudioTypes.hpp`; refactor `AudioEngine.hpp` as needed | CLI and fake tests can describe devices without importing native IDs |
| 2 — Build the registry | One owned context, serialized input/output enumeration, copied snapshots, backend diagnostics | New `AudioDeviceRegistry` and miniaudio implementation files | Duplicate names, empty catalogs, null backend, and enumeration failures are distinguishable |
| 3 — Connect selection | Use shared registry in CLI; retain index flags; add stable session selection and default policy | `src/main.cpp`, audio controller | Reordered catalogs cannot redirect an explicit selection; stale keys fail safely |
| 4 — Add refresh/recovery | Notification hints, polling, bounded retries, safe stop/reopen, stale-snapshot state | New `AudioController`; audio lifecycle code | Plug/unplug and default changes obey the policy table without RT-thread lifecycle calls |
| 5 — Integrate the UI | Two direction-specific selectors, explicit “follow default,” refresh and error status | `src/ui/AppWindow.cpp` when M4 UI work lands | UI and CLI use the same discovery and selection rules |
| 6 — Validate routing | Test installed cables/monitors; add optional Windows loopback capture mode and only necessary adapters | Audio implementation; M1 routing/verification plans | Each advertised routing mode works in a second application on its target OS |

Stages 1–4 belong with Milestone 1's audio substrate, not the ONNX pipeline.
UI work aligns with Milestone 4. Native macOS system capture and automatic
virtual-device provisioning are separate proposals, not prerequisites for
portable ordinary device discovery.

When implementation starts, introduce CTest with dependency-free C++ tests or
another project-approved test framework. The current CMake file does not define
test targets. Keep `MINIAUDIO_IMPLEMENTATION` in exactly one translation unit if
audio code moves out of `main.cpp`; do not duplicate it across new files.

## 11. Verification and acceptance

### Deterministic tests without hardware

Use a fake backend to test:

- Zero, one, and multiple devices in each direction; duplicate display names.
- Device-list reorder and disappearance between listing and opening.
- Default changes with no device additions/removals.
- Successful empty enumeration versus enumeration/backend failure.
- Old keys after context recreation and backend changes.
- Fixed selection versus follow-default policy and denied fallback.
- Refresh failure retaining a visibly stale snapshot.
- Notification bursts, retry limits, shutdown during recovery, and no callbacks
  referring to destroyed registry/controller state.
- Switching streams discarding old samples only after producers/consumers stop.
- Arbitrary callback frame counts forming exact model chunks.

### Hardware/OS matrix

Run the following on each supported Windows/WASAPI, macOS/Core Audio, and
Linux/PipeWire-Pulse environment; also test PulseAudio and ALSA fallback where
they are advertised as supported.

| Scenario | Acceptance |
|---|---|
| List devices without capture | Separate input/output lists; correct observed default flags; no recording begins |
| USB plug/unplug | Catalog updates within two configured polling intervals under normal load; removed explicit selection is not silently replaced |
| Change OS default | Follow-default selection follows it; explicit selection remains fixed or fails clearly |
| Open 44.1/48 kHz device with 16 kHz client | Correct speed/pitch; distinct client/backend diagnostics |
| Different capture/playback devices | At least 30 minutes of measured streaming; no sustained latency/queue growth or unexplained disconnects |
| Bluetooth profile change, sleep/resume | Refresh/recovery or an actionable error, never a stale “running” status |
| Denied microphone access / busy device / server restart | Correct failure category; bounded retries; no unauthorized fallback |
| Virtual route | A second application records processed output; cable endpoints are distinguished from physical monitoring |
| Missing virtual output | No fallback to speakers; route stops safely |
| Optional system-playback mode | Supported path records the selected source; unsupported path says so explicitly |

Build on Linux/GCC or Clang, Windows/MSVC, and macOS/Apple Clang using the pinned
dependencies. CI compilation and fake tests establish portability of the code,
not hardware or permission correctness. Keep manual OS evidence separate from
automated test results.

Existing documented smoke checks are `--list-devices`, `--passthrough`, and
queue-loopback mode. Run audible tests deliberately with feedback-safe routing;
they were not run as part of this documentation task.

## 12. Open decisions and follow-up research

1. Confirm supported OS versions and distribution targets before implementing
   permissions, sandbox integration, or native macOS system capture.
2. Confirm whether system-playback capture is a release requirement or whether
   portable mic-to-virtual-output routing is sufficient initially.
3. Decide whether automatic fallback is ever allowed. Recommended default:
   **never replace an explicitly chosen microphone or virtual output silently**.
4. Define which backend-native tokens merit persistent storage and how unresolved
   selections are presented. Avoid promising universal hardware identity.
5. Validate the one-second refresh interval, routing behavior, conversion quality,
   and independent-device clock behavior on actual target machines.
6. With online access approved, check official miniaudio/OS documentation against
   the pinned version, review current alternative-library and virtual-driver
   terms, and record platform-specific permission/setup requirements.

**Bottom line:** a portable discovery API is practical now using the dependency
already present. The work is mostly safe identity, snapshots, selection policy,
and lifecycle management—not writing an audio backend per OS. System-playback
capture and virtual microphones must remain explicit capabilities with
platform-specific setup hidden behind a common application interface.

## Sources and verification targets

### Sources actually inspected locally

- **[P1] Project:** the files linked in section 3, plus
  [`third_party/README.md`](../../third_party/README.md) and
  [`Documents/README.md`](../README.md).
- **[MA1] miniaudio 0.11.25:** local
  `build/_deps/miniaudio-src/miniaudio.h`; header version confirmed against the
  checkout tag. See sections 1.1, 15, and 17; API documentation for
  `ma_context_get_devices`, `ma_context_get_device_info`, device configuration,
  notifications, and data callbacks; implementations of
  `ma_is_loopback_supported`, `ma_context_init`,
  `ma_context_enumerate_devices_source_callback__pulse`, and WASAPI default
  endpoint selection. Versioned upstream mirror:
  <https://github.com/mackron/miniaudio/blob/0.11.25/miniaudio.h>.
- **[MA2] Loopback example:** local
  `build/_deps/miniaudio-src/examples/simple_loopback.c`. Versioned upstream
  mirror:
  <https://github.com/mackron/miniaudio/blob/0.11.25/examples/simple_loopback.c>.
- **[MA3] miniaudio README/license:** local
  `build/_deps/miniaudio-src/README.md` and `LICENSE`. Upstream:
  <https://github.com/mackron/miniaudio/tree/0.11.25>.
- **[PA1] PortAudio:** `/usr/include/portaudio.h`, particularly
  `Pa_GetDeviceCount`, `Pa_GetDefaultInputDevice`,
  `Pa_GetDefaultOutputDevice`, `Pa_GetDeviceInfo`, and `Pa_IsFormatSupported`.
  Upstream documentation: <https://www.portaudio.com/docs.html>.
- **[PW1] PipeWire 1.0.5 manuals:** local `pipewire-pulse(1)`,
  `pipewire-pulse-module-null-sink(7)`, and
  `pipewire-pulse-module-virtual-source(7)`, stored under
  `/usr/share/man/`. Upstream documentation: <https://docs.pipewire.org/>.

### Online follow-up targets, not consulted for this report

- miniaudio manual: <https://miniaud.io/docs/manual/>.
- Microsoft WASAPI loopback documentation:
  <https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording>.
- Apple Core Audio documentation:
  <https://developer.apple.com/documentation/coreaudio>.
- Apple ScreenCaptureKit documentation:
  <https://developer.apple.com/documentation/screencapturekit>.
- RtAudio upstream: <https://github.com/thestk/rtaudio>.
- JUCE upstream: <https://github.com/juce-framework/JUCE>.
