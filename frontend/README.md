# FluxVoice frontend — UI/UX design & implementation guide

> Milestone 4 deliverable: [`Documents/Milestone 4/00_master.md`](../Documents/Milestone%204/00_master.md) ·
> Engine rules this UI obeys: [`MEMORY.md`](../MEMORY.md), [`Documents/BEST_PRACTICES.md`](../Documents/BEST_PRACTICES.md) ·
> Controls it renders: [M2 / Stage 09](../Documents/Milestone%202/09_voice_controls_presets.md), [M3 / Stage 09](../Documents/Milestone%203/09_s2ut_controls_language_switching.md) ·
> Devices it selects: [Milestone 1.5](../Documents/Milestone%201.5/00_master.md)

The frontend is a C++20 desktop application (`fluxvoice_gui`) for Windows and
Linux (macOS builds through the same code paths and CI leg). It is built from
C++ libraries only: **SDL3** (window, OpenGL context, events, native file
dialogs, system tray) and **Dear ImGui** (widgets). There is no web view, no
scripting runtime, and no network access from the application.

This document is the authoritative UX specification for the frontend and a
map of the code that implements it. Sections 1–7 describe the product; sections
8–11 describe the implementation, data formats and verification status.

---

## 1. Scope and relation to the engine

What the frontend does today:

- composes the Milestone 1.5 engine (`MiniaudioBackend` → `AudioController` →
  worker thread) exactly as the CLI does, and drives it from a panel and from
  the system tray;
- selects capture/playback devices by **endpoint key + snapshot generation**,
  with the same selection rules as the CLI (never a row index, never a silent
  fallback);
- exposes every voice and translation control from the shared
  `VoiceParams` descriptor table (`src/voice/VoiceParams.hpp`) and the preset
  format (`presets/*.json`);
- loads installed voice bundles (`models/<bundle>/voices.json`), language
  tables (`langs.json`) and a browsable voice **catalogue index**;
- hosts the **Voice Studio**, a guided workflow for recording and preparing a
  custom target voice following the M2 Stage 10 protocol;
- shows lock-free telemetry (frames, overruns, underruns, drops, queued audio,
  RSS) read from the engine's atomics.

What it deliberately does not do yet, because the engine stages it depends on
are not implemented (see §11): run ONNX inference, apply pitch/timbre
conditioning (the sliders are delivered to the worker and shown as "applied",
the worker records them), launch training jobs, or download voices. The UI is
built so these light up without UI changes when the engine stages land: the
sliders read `kParams`, the translation panel reads the bundle's `langs.json`,
and the Studio writes the job manifest the training scripts will consume.

## 2. Design principles

These follow directly from the project's invariants.

1. **Policy, not position.** A device dropdown row is presentation only. The
   UI submits `(EndpointKey, snapshot generation)` or "Follow system default".
   If the catalogue changed under the user, the engine answers
   `DeviceChanged`/`DeviceNotFound` and the UI asks the user to choose again.
   It never re-selects on the user's behalf.
2. **No silent fallback.** A missing virtual output stops the route and the UI
   says so (status pill + message + "Choose device"). It never routes processed
   speech to speakers by itself.
3. **Controls act on conditioning.** Pitch, intonation, timbre and voicing are
   vocoder inputs. There is no dry/wet mix slider (comb-filter hazard, M2
   research §11). The only audio-domain controls are output gain, gate,
   limiter and mute.
4. **Honest status.** Liveness is derived from the controller state plus the
   callback progress counter, never from "samples are non-zero". Stale
   snapshots, offline (null) backends and unknown capabilities are shown as
   such, not hidden.
5. **Real-time sanctity.** The UI thread is the control thread: it ticks
   `AudioController`, opens/closes streams and refreshes catalogues. It never
   touches the audio callback. Parameter changes travel to the worker over a
   lock-free SPSC queue; telemetry comes back through atomics.
6. **Minimised is a first-class state.** Everything a user needs mid-session
   (engine on/off, mute, preset, voice, mode, language pair) is reachable from
   the tray menu and from the compact Mini panel, without raising the main
   window.
7. **Offline and private by default.** The catalogue can describe remote voice
   packs, but installation is a link opened in the system browser plus a
   folder the user picks. Private voices ("Me") are badged and never leave the
   machine.

## 3. Users and jobs

| User | Primary job | What they touch most |
|---|---|---|
| Streamer / gamer with a virtual microphone | Start the engine, pick a voice, keep playing | Tray menu, Mini panel, Live panel |
| Multilingual caller | Speak one language, be heard in another, in their own voice | Translate panel, language pair in tray |
| Voice creator | Record their own voice and ship it as a selectable target | Voice Studio, Catalogue ("Me" voices) |
| Developer / tester | Verify routing, latency, device recovery | Devices panel, Telemetry panel |

## 4. Information architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ FluxVoice                                     [Mini panel] [Hide to tray]│
├──────────┬───────────────────────────────────────────────────────────────┤
│ Live     │                                                               │
│ Devices  │   <section content>                                           │
│ Voices   │                                                               │
│ Studio   │                                                               │
│ Translate│                                                               │
│ Catalogue│                                                               │
│ Telemetry│                                                               │
│ Settings │                                                               │
├──────────┴───────────────────────────────────────────────────────────────┤
│ ● Running · wasapi · Mic → CABLE Input · 16 kHz · queued 40 ms · 0 underruns│
└──────────────────────────────────────────────────────────────────────────┘
```

A left navigation rail, one content area, and a persistent status bar. The
status bar always shows the stream state pill, backend, the resolved
capture → playback names, client sample rate, queued audio and the glitch
counters, so the user never has to leave a section to know whether audio is
flowing.

### 4.1 Live (home)

The one screen a user needs during a session.

- **Engine** toggle (large): Start / Stop. Disabled with an explanation when
  no backend is available. Uses the selections stored in Settings.
- **Mute** toggle: mutes the output with a 20 ms ramp (no click). Shown in
  amber when active.
- **Mode** segmented control: *Voice changer* (Mode A) / *Translate* (Mode B).
  Mode B is disabled, with the reason, when the loaded bundle has no translator
  models.
- **Voice** and **Preset** quick pickers (the same lists as the Voices panel).
- **Device summary**: capture and playback names, "Follow system default"
  badge where applicable, and a "Change…" button that jumps to Devices.
- **Meters**: input and output peak/RMS bars (computed on the worker thread,
  published through atomics).
- **Latency card**: queued audio in ms (from the conservation identity), the
  requested vs. backend period, and a 30 s sparkline of underruns.
- **Status / error card**: appears only when the stream is not Running.
  Shows the typed `AudioError` category, the backend's message and the next
  step ("Choose another device", "Retry", "Allow microphone access").

### 4.2 Devices

Implements the M4 scope "device selectors" on top of M1.5.

- Two direction-specific dropdowns. Row 0 is always **Follow system default**.
  Each row shows the name, a `(default)` marker and, when known, a monitor
  badge. Rows are rendered from the immutable snapshot; the selected row maps
  to an `ExplicitEndpoint{key, generation}`.
- **Capture mode**: Microphone, or System-playback loopback (only when the
  backend reports `native_loopback = Supported`; otherwise shown disabled with
  the capability state spelled out).
- **Refresh** button and the snapshot generation; a **stale** badge when the
  last refresh failed and an older snapshot is retained.
- **Feedback guard preview**: before applying, the UI runs `is_feedback_route`
  on the chosen names and refuses the pair with an explanation.
- **Apply**: `reconfigure()` when running, `open_and_start()` when idle. The
  result replaces the status card. `DeviceChanged` and `DeviceNotFound` are
  rendered as "choose again" states.
- **Inspect** (on demand only): probes the selected endpoint's native formats
  via the registry. Never used to populate the list.
- **Stream status table**: backend, requested policy vs. resolved endpoint,
  client rate vs. backend rates, periods, selection generation, callback
  progress, retry attempt.

### 4.3 Voices

- **Voice A / Voice B** dropdowns from the active bundle's `voices.json`,
  grouped by bundle; private voices show a lock badge and their licence text.
- **Timbre blend** slider between A and B (marked *experimental*, as the
  stage file requires).
- **Sliders** generated from `kParams`: MVP tier always visible, Stretch tier
  under "Advanced". Each slider shows name, unit, range and default; a
  double-click resets it. Integer parameters render as integer sliders,
  toggles as checkboxes. The UI never hard-codes a range.
- **Preset bar**: built-in and user presets (`presets/`, plus the user preset
  directory), *Save as…*, *Reset to Neutral*. Loading a voice-only preset
  leaves translation state untouched (M3 Stage 09 rule). Warnings from the
  loader (clamped values, unknown keys) are shown inline.
- **Applied state**: a small indicator confirms the worker consumed the latest
  parameter snapshot.

### 4.4 Voice Studio (create a custom voice)

A six-step wizard that turns the M2 Stage 10 recording protocol into a guided
flow. Steps can be revisited; the model persists the draft to disk so a
multi-session recording project survives restarts.

1. **Name & consent.** Voice name (lowercase `[a-z0-9_-]+`, validated live
   against the `voices.json` rule), display name, and an explicit consent
   checkbox: "I have the right to use this voice". For a voice other than the
   user's own, a consent note field is required.
2. **Setup checklist.** Equipment, room and format requirements from the
   protocol (one mic, pop filter, fixed distance, headphones, quiet dry room,
   48 kHz / 24-bit mono, peaks around −6 dBFS). Each item is a checkbox; the
   step is complete when all are ticked.
3. **Sessions & takes.** Add WAV files (native file dialog). For each file the
   Studio reads the header and samples and reports duration, sample rate, bit
   depth, channels, peak dBFS, RMS dBFS, clipping and an SNR estimate against
   the quietest 10 % of frames. Files that clip, are stereo, or fall below the
   30 dB SNR floor are flagged and excluded from the total. A progress bar
   tracks clean minutes against the 60-minute minimum and 180-minute ideal,
   and a material checklist (ARCTIC, Harvard, passages, questions, emotional
   reads, spontaneous) reminds the user what to cover.
4. **Room tone & live-mic set.** Two dedicated slots: 30 s of room tone per
   session, and the 10-minute live-mic source set recorded through the
   everyday microphone (used as Stage 11 evaluation input).
5. **Prepare & fine-tune.** Shows the exact commands for
   `prepare_target_voice.py`, `train_unit_vocoder.py` (fine-tune mode) and
   `export_stateful_onnx.py`, with the chosen paths filled in, for the
   current OS shell. *Copy* puts them on the clipboard; *Write job* writes
   `studio/<voice>/job.json` and `run_finetune.sh|.ps1`. The Studio polls
   `studio/<voice>/status.json` and renders stage/progress/ETA when the job
   writes it.
6. **Install.** When `models/<bundle>/voices.json` contains the new name, the
   voice appears here with licence `private — do not distribute`, and can be
   set as Voice A directly.

### 4.5 Translate

- **Source** and **Target** language dropdowns from the bundle's `langs.json`
  (falls back to the shipped default table when the bundle has none), with a
  **Swap** button.
- **Same-language guard**: selecting the same code in both shows an inline
  message recommending Mode A (60–100 ms vs. ~330 ms) and disables Apply.
- **Wait-k** slider `1..10` with a live estimate "`k × 80 ms` extra onset
  latency", and the pair's default k from `langs.json`.
- **Adaptive policy** toggle (Stretch tier; hidden unless the bundle declares
  the capability).
- **Translation ON/OFF** mirrors the Live mode control.
- **Lag telemetry**: reads / writes / flushes and `lag_ms` when Mode B is
  running.
- **Unavailable state**: when the bundle is Mode A only, the whole panel is
  readable but disabled, with "bundle has no translator models
  (translator_enc.onnx / translator_dec.onnx)".

### 4.6 Catalogue

Browse voices from two sources: **Installed** (every `voices.json` found under
the models directory) and **Catalogue** (one or more `index.json` files
describing voice packs that can be obtained). One schema (§9.4) serves both.

- Search box (name, display name, tags, language, licence) and filter chips:
  language, licence class (*commercial-clean* / *research-only* /
  *private*), gender/style tags, installed only.
- Sort by name, language, or licence.
- Result list with a detail pane: display name, bundle, languages, licence,
  attribution, source dataset, embedding dimension, f0/energy statistics,
  tags, size, and actions:
  - installed voice: **Set as Voice A**, **Set as Voice B**, **Open folder**;
  - catalogue entry: **Open page** (system browser), **Add bundle folder…**
    (after the user obtained the files).
- Licence class is computed from the licence text (`CC-BY-NC`, `NC` →
  research-only; `private` → private; otherwise commercial-clean) and shown as
  a coloured badge, mirroring the licence hygiene rule for shipped bundles.

### 4.7 Telemetry

For developers and soak tests: live counters (frames, input overruns, output
underruns, output drops, queued samples and ms, RSS), queue identity check
(`queued = underruns − overruns − drops`), a 60 s history of each counter's
delta, the applied parameter generation, and a **Copy `[stats]` line** button
producing the same format `tools/analyze_soak.py` parses.

### 4.8 Settings

- Start minimised to tray; close button hides to tray (vs. quits); start the
  engine on launch.
- Theme: dark / light / follow system.
- Models directory, user preset directory, Studio directory.
- Remembered device policy: *Follow system default* persists; an explicit
  device is remembered by name as a hint only and is re-resolved from the
  current snapshot at start, asking the user if it cannot be found (M1.5
  decision 4).
- Tray availability: "available" or the platform-specific reason it is not
  (for example, no AppIndicator library on this Linux desktop).
- About: versions of SDL3, Dear ImGui, backend name, licences.

### 4.9 Mini panel

A compact (about 340 × 120 px), always-on-top window toggled from the title
bar, the tray or `Ctrl+M`: engine toggle, mute, voice picker, preset picker,
latency figure and the status dot. Dragging it moves it; it remembers its
position. Closing the mini panel restores the main window.

### 4.10 Keyboard

`Ctrl+1`…`Ctrl+8` switch sections, `Space` toggles the engine when the Live
panel has focus, `M` toggles mute, `Ctrl+M` toggles the Mini panel, `Esc`
closes dialogs. Every control is reachable with Tab (ImGui keyboard
navigation is enabled). Text contrast meets WCAG AA in both themes (§7).

## 5. Key flows

**First run.** Live shows the engine toggle disabled with "No devices
selected yet" → user opens Devices → the catalogue is already enumerated →
picks Follow default for capture and the virtual cable for playback → the
feedback-guard preview passes → Apply → Running. Settings remembers the
policy.

**Switch preset while minimised.** Tray → Preset ▸ → "Deep". The tray model
posts a `LoadPreset("Deep")` action; the control thread applies the preset to
`VoiceParams`, pushes the snapshot to the worker, updates the tooltip
("FluxVoice · Running · Deep · 42 ms").

**Device unplugged.** The controller enters `AwaitingUser` with
`DeviceNotFound`. The tray icon turns red, the tooltip carries the message,
and the Live status card offers "Choose device". Nothing is substituted.

**Set a translation pair.** Translate → Source `es`, Target `en` → Wait-k
shows the pair default (3) → Apply → `translate=1`, `src/tgt` ids set, worker
resets translator state only (engine stage work), UI shows lag telemetry.

**Create a custom voice.** Studio steps 1–6 as in §4.4. The draft is saved
to `studio/<voice>/draft.json` after every change.

## 6. Tray and minimised use

Icon states (rendered procedurally, no binary assets):

| State | Icon | Tooltip |
|---|---|---|
| Idle | grey waveform | `FluxVoice · Idle` |
| Running | green waveform | `FluxVoice · Running · <preset> · <queued> ms` |
| Muted | green with a slash | `FluxVoice · Muted` |
| Recovering / Opening | amber | `FluxVoice · Recovering (attempt n)` |
| Awaiting user / Error | red | `FluxVoice · <error category>: <message>` |

Menu:

```text
Show panel
Mini panel
──────────
[x] Engine on
[ ] Mute
──────────
Mode ▸        ( ) Voice changer   (•) Translate
Preset ▸      Neutral · Deep · Bright · Robot · Whisper · <user presets>
Voice ▸       <voices of the active bundle>
Languages ▸   es → en (k=3) · en → de (k=4) · <recent pairs>
──────────
Quit
```

Behaviour: closing the window hides it when *close to tray* is on (default
on when a tray is available); *start minimised* launches hidden. While hidden
the main loop waits on events with a 20 ms timeout, keeps ticking the
controller and the tray, and renders nothing. If the platform has no tray
(for example a Linux session without `libayatana-appindicator3`), close means
quit, the Settings page explains why, and *start minimised* falls back to a
minimised window.

Platform notes:

- **Windows**: SDL3 uses `Shell_NotifyIcon`; the menu is a native context
  menu. Taskbar thumbnail buttons and global hotkeys are listed in §11.
- **Linux**: SDL3 loads `libayatana-appindicator3` or `libappindicator3` at
  runtime (GTK 3). GNOME needs the AppIndicator extension; KDE, XFCE, Cinnamon
  and MATE work out of the box. X11 and Wayland sessions are both supported
  through SDL's video drivers.
- **macOS**: menu-bar status item through Cocoa (not a release target yet,
  but the CI leg builds it).

## 7. Visual design

- **Type**: the system UI font when found (Segoe UI on Windows; DejaVu Sans,
  Noto Sans or Liberation Sans on Linux; Helvetica on macOS), base size 16 px,
  scaled by the display scale SDL reports. Falls back to ImGui's embedded
  font.
- **Colour**: a sober dark theme by default and a light theme; one accent
  (teal, `#2BB0A6`) for primary actions and the selected rail item; semantic
  colours for status only: green running, amber recovering/muted, red error,
  grey idle. Contrast of body text on background is at least 7:1 (dark) and
  8:1 (light).
- **Shape and spacing**: 6 px frame rounding, 8 px item spacing, 12 px
  section padding, 1 px borders on frames in the light theme only.
- **Density**: one primary action per card; secondary actions as small
  buttons. Long text wraps; nothing truncates silently (device names are
  shown in full with a tooltip for the overflow).
- **Motion**: none beyond ImGui's defaults; meters update at frame rate.

## 8. Architecture

```text
frontend/
  core/        toolkit-free model layer (C++20, nlohmann/json, engine headers)
  platform/    Tray abstraction (NullTray) + SDL3 implementation; config paths
  imgui/       GuiApp (SDL3 + OpenGL3 + ImGui lifecycle), Theme, Views
  app/         main.cpp (argument parsing, backend construction, run loop)
  assets/      langs.default.json, catalogue/index.json (sample)
```

Threads:

| Thread | Owns | Talks to |
|---|---|---|
| UI = control | SDL event loop, ImGui frame, `AudioController::tick`, catalogue refresh, device open/close, preset loading, tray updates | worker via `ReaderWriterQueue<VoiceParams>` (SPSC, producer); engine via `AudioController` |
| Worker | `ChunkAssembler`, `InferencePipeline` (shell), `OutputChain` (mute/gain/limiter), meters | SPSC queues of `DuplexRouter`; publishes meters/applied-generation via atomics |
| Audio callback | `DuplexRouter::process` only | nothing else |

Core types (all in `namespace fluxvoice::frontend`):

- `EngineHost` — composes backend, controller, worker; applies worker hooks
  (quiesce / reset / resume) exactly like `src/main.cpp`; exposes
  `TelemetrySnapshot`, `push_params()`, `set_muted()`.
- `DeviceSelectorModel` — builds dropdown rows from a `DeviceSnapshot`
  (Follow default first) and converts a chosen row into a `DeviceSelection`
  carrying the snapshot generation; evaluates the feedback guard.
- `VoiceCatalog` — loads `voices.json` (schema 1, strict as the Python
  validator) from a models root, catalogue indices, search/filter/sort,
  licence classification.
- `TranslationModel` — `langs.json` loading with default table fallback,
  pair validation, wait-k defaults and clamping, translator availability.
- `VoiceStudioModel` — wizard state machine, draft persistence, WAV
  analysis (`WavStats`), job manifest and command generation.
- `TrayModel` — declarative menu spec + icon state + tooltip from the app
  state; `TrayIcon` renders the icon pixels; `TrayAction` queue.
- `AppModel` — owns the above plus `VoiceParams`, presets and `Settings`;
  the single place that mutates state. Views only read it and call its
  commands.
- `OutputChain` — smoothed gain, mute and soft limiter applied on the worker.

The ImGui views are thin: each `draw_*` function reads the model and calls a
command. Nothing in `imgui/` or `platform/` is required by the tests; the
model layer is tested headless with a fake `AudioBackend`
(`tests/test_frontend_model.cpp`).

Build: `-DFLUXVOICE_BUILD_GUI=ON` (default) fetches SDL3 3.4.18 (release
tarball, SHA-256 pinned, static, audio/joystick/haptic/sensor/camera/GPU
subsystems off) and Dear ImGui v1.92.9 (git tag) via FetchContent, mirroring
the existing miniaudio / readerwriterqueue / nlohmann pins. With
`-DFLUXVOICE_FETCH_DEPS=OFF` the sources are expected under
`third_party/SDL3` and `third_party/imgui`. Third-party sources compile in
their own targets as SYSTEM code; the project's warning gate applies only to
first-party files. Headers under `frontend/` are included in the
`fluxvoice_header_checks` isolation build.

Run:

```bash
./build/frontend/fluxvoice_gui                        # main panel
./build/frontend/fluxvoice_gui --minimized            # start in the tray (minimised window when no tray)
./build/frontend/fluxvoice_gui --mini                 # open as the compact Mini panel
./build/frontend/fluxvoice_gui --section catalogue    # open on a section (live, devices, voices, studio,
                                                      #   translate, catalogue, telemetry, settings)
./build/frontend/fluxvoice_gui --models-dir models    # bundle root (default: Settings, ./models, then tests/data/fixtures)
./build/frontend/fluxvoice_gui --preset-dir ~/.config/fluxvoice/presets --config-dir ~/.config/fluxvoice
./build/frontend/fluxvoice_gui --smoke-frames 40 --screenshot out.ppm   # render N frames, dump the frame, exit
# headless check on Linux:
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1280x800x24" ./build/frontend/fluxvoice_gui --smoke-frames 40 --screenshot live.ppm
```

## 9. Data formats

### 9.1 `voices.json` (schema 1, produced by `export_stateful_onnx.py`)

Read with the same rules as `fvml.contract.validate_voices`: integer schema
`1` only, keys limited to `schema`/`voices`, non-empty voice list, names
lowercase `[a-z0-9_-]+` and unique, `display`/`license`/`source` non-empty
strings, numeric `logf0_mean`/`logf0_std`/`energy_mean`/`energy_std`
(standard deviations positive), `embedding` a non-empty list of finite
numbers, optional `attribution` string. A bundle that fails validation is
listed as *invalid* with the diagnostic, not silently skipped.

### 9.2 `langs.json` (M3 Stage 02)

`{"schema": 1, "langs": [{"code": "en", "id": 0, "name": "English"}, …],
"default_wait_k": {"es->en": 3, …}}`. The shipped fallback is
`frontend/assets/langs.default.json` with `en`, `es`, `fr`, `de` and the
pair defaults from the stage files.

### 9.3 Presets (`presets/*.json`, M2/M3 Stage 09)

```json
{ "schema": 1, "name": "Deep", "description": "…",
  "voice_a": "<voice name>", "voice_b": "<voice name>",
  "params": { "pitch_shift": -4, "intonation": 0.9, "translate": 1, "wait_k": 3 },
  "src_lang": "es", "tgt_lang": "en" }
```

Every key except `schema`, `name` and `params` is optional; omitted keys leave
the current value unchanged. The loader clamps out-of-range values with a
warning, warns on unknown keys and rejects any schema other than `1`.

### 9.4 Catalogue index (`index.json`, introduced here)

```json
{ "schema": 1, "title": "FluxVoice public voices",
  "entries": [ { "id": "hifitts-92", "display": "Hi-Fi TTS speaker 92",
                 "bundle": "fluxvoice-en-v1", "languages": ["en"],
                 "license": "CC BY 4.0", "attribution": "Hi-Fi TTS (Bakhturina et al.)",
                 "tags": ["female", "neutral"], "size_mb": 42,
                 "url": "https://…/fluxvoice-en-v1.zip", "page": "https://…",
                 "sample": "https://…/preview.wav" } ] }
```

Only `id` and `display` are required. The application never downloads;
`url`/`page` are opened in the system browser.

### 9.5 Settings (`<config dir>/fluxvoice/settings.json`)

Written by `Settings::save`; unknown keys are preserved, missing keys take
defaults. Config dir: `%APPDATA%` on Windows, `$XDG_CONFIG_HOME` or
`~/.config` on Linux, `~/Library/Application Support` on macOS.

### 9.6 Studio job contract (`studio/<voice>/job.json`, `status.json`)

`job.json` is written by the Studio: voice name, display, consent note,
input WAV list with per-file stats, room-tone and live-mic lists, target
bundle path, and the three command lines. `status.json` is expected from the
training side: `{"stage": "prepare|features|finetune|export|done|error",
"progress": 0.0–1.0, "message": "…", "eta_s": 123}`. The Studio polls it
once per second while the Studio section is visible.

## 10. Verification

Hardware-free, in CTest: `voice_params` (descriptor table and presets),
`frontend_model` (device rows and selection policy, catalogue loading and
validation, translation guard, studio wizard and WAV statistics, tray menu
spec and icon, settings round-trip, output-chain click bound, and the
`AppModel` driving an `AudioController` over `tests/FakeAudioBackend.hpp`:
start, device loss without substitution, presets, tray actions, policy
restore), plus `gui_help` / `gui_rejects_*` (argument handling before SDL
initialises). All first-party headers, including `frontend/**`, compile in
isolation in `fluxvoice_header_checks`.

Evidence recorded 2026-11-02 (BEST_PRACTICES §5.6; container: Ubuntu 24.04
x86-64, CMake 3.28.3, Ninja):

| Leg | Compiler | Configure / build / CTest | Result |
|---|---|---|---|
| Linux GCC | GCC 13.3.0 | `cmake -S . -B build-gcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON -DFLUXVOICE_BUILD_TESTS=ON`, all targets, `ctest --test-dir build-gcc` | PASS, 51/51 tests |
| Linux Clang | Clang 18.1.3 | same with `build-clang` | PASS, 51/51 tests |
| Extra strictness | GCC + Clang | first-party `frontend/**`, `src/voice/**` and the two tests compiled with `-Wshadow -Wconversion -Wsign-conversion -Werror` to approximate MSVC `/W4` (C4244/C4267/C4456-9) | PASS |
| Headless run | GCC build, Xvfb + Mesa llvmpipe | every section and the Mini panel rendered for 40 frames with `--smoke-frames 40 --screenshot`, 0 `[imgui-error]` lines, PPM frames inspected | PASS |
| Windows MSVC | — | CI leg `windows-msvc` | NOT RUN here (CI) |
| macOS AppleClang | — | CI leg `macos-clang` | NOT RUN here (CI) |

The Windows and macOS legs run in `.github/workflows/build.yml`; the Linux
legs there install the SDL3 X11 build dependencies first. Live audio routing,
the real tray on each desktop, and native dialogs are manual checks on real
hardware and are not claimed by the table above.

## 11. Not in this iteration

- Inference wiring, conditioning application, latency governor, translator
  state resets (M2 Stages 07–09, M3 Stages 07–09). The UI delivers
  `VoiceParams` to the worker; applying them is engine work.
- Global hotkeys (Windows `RegisterHotKey`; Wayland needs the
  GlobalShortcuts portal) and Windows taskbar thumbnail buttons
  (`ITaskbarList3`). Both are additive to `platform/`.
- Desktop notifications, in-app downloads, preview playback of catalogue
  samples.
- GPU execution-provider toggles and quantisation switches (M4 optimisation
  half; they will be Settings entries bound to ORT session options).
