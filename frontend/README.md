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
- **runs the real Mode A pipeline**: it loads a model bundle (encoder, vocoder,
  `voices.json`) on a loader thread, warms it up, and converts live speech
  through `fluxvoice::ModeAHost` (`src/pipeline/ModeAHost.hpp`), the same host
  `fluxvoice --vc` uses. Pitch, intonation, timbre blend, voicing and the
  output stage all act on GUI audio. Without a loaded bundle the engine runs
  in **Bypass** (microphone through gain, gate, brightness, limiter and mute
  only) and says so in Live, the tray tooltip and the status bar;
- selects capture/playback devices by **endpoint key + snapshot generation**,
  with the same selection rules as the CLI (never a row index, never a silent
  fallback);
- exposes every voice and translation control from the shared
  `VoiceParams` descriptor table (`src/voice/VoiceParams.hpp`) and the preset
  format (`presets/*.json`). The "applied" indicator means *accepted by the
  pipeline*, not delivered to the worker;
- loads installed voice bundles (`models/<bundle>/voices.json`), language
  tables (`langs.json`) and a browsable voice **catalogue index**;
- hosts the **Voice Studio**, a guided workflow for recording and preparing a
  custom target voice following the M2 Stage 10 protocol;
- shows lock-free telemetry (frames, overruns, underruns, drops, queued audio,
  RSS) read from the engine's atomics.

What it deliberately does not do yet (see §11): translate (Mode B is rejected
on the control thread with "bundle has no translator models" until M4 Stage 07),
show the latency and per-stage timing panels (Stage 06), launch training jobs,
or download voices. A model that fails to load or a pipeline that stops at run
time is reported, never replaced by Bypass silently.

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
- **Meters**: input and output level meters (computed on the worker thread,
  published through atomics): an RMS bar with a peak tick that turns red above
  −3 dBFS. The Mini panel uses the same widget.
- **Latency card** ("Estimated end-to-end latency", M4 Stage 06): the same
  number as the CLI's `lat_ms` (`fluxvoice::estimate_latency`, the BUG-026
  formula), with its terms broken out: capture device period, queued audio,
  model delay (lookahead + OLA tail) and playback device period. Compute p95
  per hop and the algorithmic latency (contract) are shown beside it for
  information and are **not** added to the sum: compute runs while the queued
  audio plays. In Bypass the card shows the queued and device terms only and
  says so. Always labelled "estimated"; the measured value comes from
  `fv_probe latency` across two cables (Stage 12). In system-playback loopback
  the queued term is undefined and the card shows "n/a (independent clocks)".
  The Levels card keeps the meters, the glitch counters and a 60 s sparkline of
  underruns.
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
  - *Save as…* writes the voice parameters only (pitch, timbre, output,
    voices). The translation group (`translate`, `wait_k`, `adaptive_policy`)
    and the language pair are written only when the *Include translation state
    (mode, Wait-k, languages)* checkbox is ticked.
  - A name equal to a built-in preset (case-insensitive) is refused at save
    time: the Save button is disabled and shows the reason. Two names that map
    to the same file name (for example "My voice" and "My_voice") are refused
    too. A user preset file that carries a built-in's name is hidden, with a
    warning in the preset bar.
  - The last preset is re-applied at start: its voice parameters only (voices,
    languages, Wait-*k*, mode and gain come from Settings, which is newer). If
    the preset no longer exists, the label is cleared and an Info notice is
    shown.
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
   30 dB SNR floor, or contain NaN/Inf samples, are flagged and excluded from
   the total. Damaged or hostile WAV headers are rejected before any
   allocation. Analysis runs on a worker thread (the view shows "analysing N
   file(s)…"); results are applied on the UI thread. A progress bar
   tracks clean minutes against the 60-minute minimum and 180-minute ideal,
   and a material checklist (ARCTIC, Harvard, passages, questions, emotional
   reads, spontaneous) reminds the user what to cover.
4. **Room tone & live-mic set.** Two dedicated slots: at least 30 s of room
   tone per session, and the 10-minute live-mic source set recorded through the
   everyday microphone (used as Stage 11 evaluation input).
5. **Prepare & fine-tune.** Shows the commands for the three steps for the
   current OS shell, with the chosen paths filled in, but only for steps that
   can run today; an unavailable step shows its reason instead of a command.
   - *Prepare* (`prepare_target_voice.py --input <dir> --speaker=<voice> --out
     <project>/prepared [--display <name>]`) is available only when every take
     and room-tone file is in one folder and named `<voice>_sNN_TTTT.wav` /
     `<voice>_sNN_roomtone.wav`; otherwise the failing rule is shown.
   - *Fine-tune* is always "unavailable until M2 Stage 10 lands" (it needs the
     Stage 05 encoder checkpoint, the base vocoder checkpoint, a training
     manifest and unit directories that the Studio does not collect).
   - *Export* is unavailable (it would export the fine-tune's checkpoint).
   Unavailable steps are left out of `job.json` `"commands"` (they are listed
   in a separate `"unavailable"` object) and appear only as comments in the run
   scripts. *Copy* is disabled when nothing is available. *Write job* writes
   `studio/<voice>/job.json` and `run_finetune.sh|.ps1` (POSIX `#!/bin/sh` with
   `set -eu`; PowerShell with `&` calls, a `$LASTEXITCODE` check after each
   line and single-quote quoting). `takes.tsv` and `room_tone.tsv` are still
   written but nothing consumes them yet. The Studio polls
   `studio/<voice>/status.json` and renders stage/progress/ETA when the job
   writes it.
6. **Install.** When `models/<bundle>/voices.json` contains the new name, the
   voice appears here with licence `private — do not distribute`, and can be
   set as Voice A directly.

### 4.5 Translate

- **Source** and **Target** language dropdowns from the bundle's `langs.json`
  (falls back to the shipped default table when the bundle has none), with a
  **Swap** button.
- **Same-language guard**: there is no Apply button. A change applies at once;
  an invalid or same-language pair is rejected with an inline message
  recommending Mode A (60–100 ms vs. ~330 ms).
- **Wait-k** slider with its range, clamp and default read from
  `param_desc(ParamId::WaitK)` (1..10, default 3 today), the same source used
  to validate `langs.json` `default_wait_k` and Settings. The onset estimate is
  "`k × chunk` extra onset latency", where the chunk length comes from the
  translator contract (`chunk_samples / sample_rate`). The contract is embedded
  in the `.onnx` files and the GUI links no ONNX Runtime yet, so the chunk
  length stays unknown, and the text shows no ms figure, until a later stage
  loads the bundle and calls `AppModel::set_translator_caps()`. The saved
  Wait-*k* survives a restart for every pair; a pair change made by the user
  applies the pair's default; a preset's own `wait_k` wins over the pair
  default, and a pair-only preset gets the pair default.
- **Adaptive policy** toggle (Stretch tier; hidden until the bundle declares
  the capability, that is until the translator reports a readiness output;
  switched off if a bundle change drops it).
- **Translation ON/OFF** mirrors the Live mode control.
- **Lag telemetry**: reads / writes / flushes and `lag_ms` when Mode B is
  running.
- **Unavailable state**: when the bundle is Mode A only, every card
  (Languages, Wait-*k*, Mode) is readable but disabled, with "bundle has no translator models
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
  - catalogue entry: **Open page** / **Open download link** (system browser;
    `http://` or `https://` with a host only, otherwise the button is disabled
    with a tooltip), **Add bundle folder…** (after the user obtained the
    files). `GuiApp::open_url` refuses any other scheme with a Warning notice;
    local folders go through `open_path`.
- Licence class is computed from the licence text (`CC-BY-NC`, `NC` →
  research-only; `private` → private; otherwise commercial-clean) and shown as
  a coloured badge, mirroring the licence hygiene rule for shipped bundles.

### 4.7 Telemetry

For developers and soak tests; every figure is read lock-free and the panel
redraws every frame (at least once per second).

- **Latency card**: the same "Estimated end-to-end latency" card as Live.
- **Stage table**: one row per `PipelineStage` (`fluxvoice::to_string`: feat,
  enc, trans_enc, trans_dec, voc, chain) plus a `hop` row for the whole hop;
  columns hops, p50, p95, max in ms and *share of hop* (p95 / hop duration). A
  stage that never ran (no translator) shows "-". Under it, a 60 s sparkline of
  the whole-hop p95 in ms (one point per second, pushed in `AppModel::tick`).
  Bypass has no model stages and says so.
- **Counters**: frames, overruns, underruns, drops (per stream and lifetime),
  deadline misses, skipped samples (latency governor), prefilled samples
  (startup silence), the governor target and hard cap in ms, queued audio,
  RSS, parameter generations, worker hops.
- **Identity check**: `queued` (the identity: Mode A `prefilled + underruns -
  overruns - drops - skipped`, Bypass `underruns - overruns - drops`) against
  the measured occupancy (input queue + output queue + samples the worker
  holds), the difference and the tolerance (one device period plus one hop).
  Verdict "consistent" when |difference| <= tolerance, otherwise "MISMATCH" in
  the error colour (an accounting error: a missing term or a missed
  rebaseline); "n/a" when the stream is not running, in system-playback
  loopback (the identity does not hold) or when the occupancy is unknown. The
  note stays: queue occupancy cannot show hardware clock drift.
- **`[stats]` line** and a **Copy** button: exactly the line `src/main.cpp`
  prints (`fluxvoice::format_stats_line`; Bypass prints the legacy line, as the
  CLI's diagnostic modes do), with `t` = seconds since the stream started.
  `tools/analyze_soak.py` parses it, and CTest `gui_stats_parse` checks lines
  the GUI model produced.
- **Stats log** (Settings > "Write stats log", off by default): the UI thread
  appends that line once per second of stream time to `<config dir>/stats.log`
  (or the chosen file). The file is opened (append) once per stream, `t`
  restarts at 0 for every stream (start, reconfigure, device recovery), and a
  write error turns the log off with an Error notice.

### 4.8 Settings

- Start minimised to tray; close button hides to tray (vs. quits); start the
  engine on launch.
- Theme: dark / light / follow system.
- Models directory, user preset directory, Studio directory.
- Remembered device policy: *Follow system default* persists; an explicit
  device is remembered by name as a hint only and is re-resolved from the
  current snapshot at start, asking the user if it cannot be found (M1.5
  decision 4). A stored policy that is neither empty, `"default"` nor
  `"name:<device>"` is treated as an unresolved explicit choice (Start is
  refused), never as Follow default.
- A malformed `settings.json` is renamed to `settings.json.bad` on load (an
  older `.bad` is replaced), defaults are used, and a warning naming the `.bad`
  path is shown in the UI notices and on stderr. See §9.5.
- Tray availability: "available" or the platform-specific reason it is not
  (for example, no AppIndicator library on this Linux desktop).
- Model (M4 Stage 05): frames per step (0 = the bundle's default variant,
  else 1 / 2 / 4) and latency target in ms (0 = the master-budget default);
  "Apply now" reloads the loaded bundle with them.
- Stats log (M4 Stage 06): "Write stats log" plus a path (default
  `<config dir>/stats.log`). While a stream runs, the UI thread appends the
  same `[stats]` line the CLI prints, once per second of stream time (`t`
  restarts with each stream), so `tools/analyze_soak.py` reads GUI soak logs.
  A write error switches logging off for the session with an error notice; a
  user-chosen path never creates directories.
- About: versions of SDL3, Dear ImGui, backend name, licences.

### 4.9 Mini panel

A compact (about 340 × 120 px), always-on-top window toggled from the title
bar, the tray or `Ctrl+M`: engine toggle, mute, voice picker, preset picker,
latency figure and the status dot. Dragging it moves it; it remembers its
position (negative coordinates are allowed) and restores it only when the
point lies inside a connected display, otherwise the window manager places it.
Closing the mini panel restores the main window.

### 4.10 Keyboard

`Ctrl+1`…`Ctrl+8` switch sections, `M` toggles mute, `Ctrl+M` toggles the
Mini panel, `Ctrl+H` hides to the tray and `Ctrl+Q` quits (both global, also in
the Help menu). `Space` toggles the engine only when the Live section is shown
in the main window and its content panel has keyboard focus; never in Mini
mode, never while text input is active or a widget is held. Known limitation:
after clicking the navigation rail, click the Live panel once before `Space`
works. `Esc` closes the Save preset dialog (the only modal; file dialogs are
native). Every control is reachable with Tab (ImGui keyboard
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
shows the pair default (3) → the pair applies at once (no Apply button) →
`translate=1`, `src/tgt` ids set, worker
resets translator state only (engine stage work), UI shows lag telemetry.

**Create a custom voice.** Studio steps 1–6 as in §4.4. The draft is saved
to `studio/<voice>/draft.json` when the name field loses focus and after
each batch of analysed takes, never per keystroke.

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
Languages ▸   every ordered pair of the language table (no k, no recent pairs)
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
| UI = control | SDL event loop, ImGui frame, `AudioController::tick`, catalogue refresh, device open/close, preset loading, tray updates, **installing a loaded bundle** (`AppModel::tick` drains the loader result), voice-name resolution, `translate` pre-validation | worker via `ParamChannel` (SPSC, producer; `ModeAHost::push_params` in Mode A); engine via `AudioController` |
| Loader (short-lived, at most one) | `ModeAHost::load`: models, voice table, `InferencePipeline`, warm-up, queue-capacity check. Reads only copies of the parameters and voice names captured when the request was made | one-slot result queue under a mutex, drained by `AppModel::tick` |
| Worker | **Mode A:** `ModeAWorker` inside `ModeAHost` (the same worker `fluxvoice --vc` runs), at `WorkerPriority`. **Bypass:** 20 ms blocks through `VoiceOutputChain` | SPSC queues of `DuplexRouter`; publishes meters and the pushed / delivered / accepted / rejected generations via atomics |
| Audio callback | `DuplexRouter::process` only | nothing else |

Core types (all in `namespace fluxvoice::frontend`):

- `EngineHost` — composes backend and controller and runs in one of **two
  modes** (`EngineHost::Mode`):
  - *Bypass* (no bundle loaded): the capture stream is regrouped into 20 ms
    blocks and run through `fluxvoice::VoiceOutputChain` (gain, gate,
    brightness, limiter, ramped mute) with its own `ParamChannel`. Pitch,
    timbre and voice controls have nothing to act on: the UI shows "Bypass -
    model controls inactive" and never "applied". Labelled "Bypass - no model
    loaded" in Live, the tray tooltip and the status bar (ASCII on purpose).
  - *Mode A* (`install_host()` with a loaded `std::unique_ptr<ModeAHost>`):
    `host->configure(cfg)` then `controller.set_config(cfg)` while Idle, then
    `host->attach(controller)`. The host owns the worker, the parameter
    channel, the hooks and the output chain; `EngineHost` only forwards
    `push_params` / `flush_params` / `set_muted` and reads `host->meters()`
    and `host->snapshot()`. There is no second worker loop, parameter queue or
    output chain for Mode A in `frontend/`. `telemetry().queued_samples` is
    the worker's full identity (`prefilled + underruns - overruns - drops -
    skipped`); Bypass keeps its rebaselined `underruns - overruns - drops`.
    `telemetry()` carries one `fluxvoice::HostStats` (M4 Stage 06): Mode A's
    comes from `host->stats()` (the same snapshot the CLI formats), Bypass
    builds a legacy one (`vc` false, `skipped = prefilled = 0`, queued +
    device terms for the latency estimate) and publishes an occupancy of its
    own: input queue + output queue + the samples its worker holds
    (`held_`, a relaxed atomic the worker stores after each read and block).
    `TelemetrySnapshot` derives the stage rows, the identity verdict and the
    latency card from it, so the views hold no formulas.
  Installing replaces a previous host only while the controller is Idle (the
  old host is detached, the new one attached, the old one destroyed last; a
  refused attach puts the previous host back). `remove_host()` re-installs the
  Bypass hooks and config. The model source is an injectable
  `std::shared_ptr<const fluxvoice::HostModelLoader>` (default
  `fluxvoice::ort_bundle_loader()`), shared with `AppModel`, so tests inject
  fakes.
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
  commands. It also keeps the stream clock (`stream_seconds()`, rebased per
  stream by every start, reconfigure and recovery) and the optional stats log:
  `tick()` appends `telemetry().stats_line(stream_seconds())` once per second
  of stream time, opening the file once per stream and turning the log off
  with a notice on any write error; the path goes through `path_from_utf8` and
  is committed in Settings on Enter or focus loss, never per keystroke.
- `LoaderThread<T>` — one short-lived thread plus a one-slot result queue;
  `AppModel` uses it for bundle loads (same drain-in-`tick` pattern as the
  Studio analysis and the native dialogs). `Meters.hpp` — `to_dbfs` /
  `measure` for the Bypass meters.

Bundle lifecycle (`AppModel`, control thread):

- `load_bundle(dir)` stops a running stream (remembering the device policy),
  captures copies of the parameters and voice names for the seeder, and starts
  the loader thread with a `HostLoadRequest` built from Settings
  (`frames_per_step`, `latency_target_ms`) and `TranslatorPolicy::kNever`
  (Stage 07 enables translation). `tick()` drains the result: on success the
  host is installed, voices are re-resolved by name against its table (unknown
  name: first voice, with a notice), parameters are pushed, and the stream
  restarts with the remembered policy; on failure the **previous bundle stays
  active** (restarted if it was running) and the error is an Error notice -
  never a silent fall back to Bypass. One load at a time; Start is disabled
  while loading; the destructor joins the loader. `activate_bundle()` /
  `use_voice()` (Voices, Catalogue, Studio) select by name and load.
- The saved `active_bundle` (or `--bundle`) loads at startup; "start engine on
  launch" waits behind that load, and a failed startup load starts nothing.
- A fatal pipeline error (`host->failed()`) is handled once in `tick()`: the
  stream stops, Live shows `host->error()` with "Retry" (same bundle and
  devices) and "Choose another bundle" (Catalogue). Nothing restarts by itself.
- "Applied" is `accepted_generation() == pushed_generation()`; a rising
  `rejected_params()` shows "rejected by the pipeline".
- Voice A/B names map to indices with `host->voice_index()` on the control
  thread; B follows A until chosen explicitly; an unknown name is rejected
  with a message naming the choices and the previous voice stays.
  `translate = 1` is rejected on the control thread with
  `kNoTranslatorModelsMessage` ("bundle has no translator models").

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
./build/frontend/fluxvoice_gui --bundle models/mybundle   # load this bundle at startup (overrides the saved active bundle)
./build/frontend/fluxvoice_gui --preset-dir ~/.config/fluxvoice/presets --config-dir ~/.config/fluxvoice
./build/frontend/fluxvoice_gui --smoke-frames 40 --screenshot out.ppm   # render N frames, dump the frame, exit
# With --bundle, stderr gets one line when the load result is drained, and a smoke run
# does not end before it (bounded, ~10 s):
#   [bundle] loaded '<dir>' rate=16000 hop=320 voices=2
#   [bundle] error: <message>      (an ORT-less build: "...built without ONNX Runtime...")
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
defaults. Config dir: `%APPDATA%` on Windows (read with `_wdupenv_s` and kept
as a wide string, so a non-ASCII profile path survives), `$XDG_CONFIG_HOME` or
`~/.config` on Linux, `~/Library/Application Support` on macOS. Saves are
atomic (`settings.json.tmp` plus rename). A malformed file is moved to
`settings.json.bad` on load (and before a save that finds one); if that move
fails the save is refused and the file is left untouched.

The Mini panel position is stored as `"mini_position": [x, y]`, written only
when set. Legacy `mini_x` / `mini_y` keys are read once (legacy `-1,-1` or a
half pair means no position), migrated, and never written again.

Model and telemetry keys (M4 Stages 05–06): `"active_bundle"` (bundle name,
loaded asynchronously at start), `"frames_per_step"` (0, 1, 2 or 4),
`"latency_target_ms"` (0–5000; 0 = default), `"write_stats_log"` (bool) and
`"stats_log_path"` (UTF-8 path). All numbers go through the checked readers
in `core/JsonRead.hpp`: a wrong type, a non-finite or out-of-range value is
replaced by the default with a warning.

### 9.6 Studio job contract (`studio/<voice>/job.json`, `status.json`)

`job.json` is written by the Studio: voice name, display, consent note,
input WAV list with per-file stats, room-tone and live-mic lists, target
bundle path, and the three command lines. `status.json` is expected from the
training side: `{"stage": "prepare|features|finetune|export|done|error",
"progress": 0.0–1.0, "message": "…", "eta_s": 123}`. The Studio polls it
once per second while the Studio section is visible.

## 10. Verification

Hardware-free, in CTest: `voice_params` (descriptor table and presets),
`frontend_model` (device rows and selection policy, device-policy rules,
preset and translation rules, catalogue loading and validation, translation
guard, studio wizard and WAV statistics, Studio analysis drained by
`AppModel::tick`, tray menu spec and icon, settings round-trip, output-chain
click bound, and the `AppModel` driving an `AudioController` over
`tests/FakeAudioBackend.hpp`: start, device loss without substitution,
presets, tray actions, policy restore), `frontend_shell` (shell, keyboard and
settings behaviour, `tests/test_frontend_shell.cpp`), `frontend_studio`
(Studio drafts, WAV parser, job scripts, `tests/test_frontend_studio.cpp`),
plus `gui_help` / `gui_rejects_*` (argument handling before SDL
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

Evidence recorded 2026-11-18 (local machine, Ubuntu 24.04 x86-64, CMake
3.28.3, Ninja 1.11.1; base commit `fe2fb68` plus the uncommitted working tree;
GUI on, ORT off):

| Leg | Compiler | Configure / build / CTest | Result |
|---|---|---|---|
| Linux GCC | GCC 13.3.0 (`build-m4-gcc`) | `cmake -S . -B build-m4-gcc -G Ninja -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_BUILD_TESTS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON`, `cmake --build` (all targets incl. `fluxvoice_header_checks`, `fluxvoice_gui`), `ctest -E python_training` | PASS, 105/105 |
| Linux Clang | Clang 18.1.3 (`build-clang`) | same | PASS, 105/105 |
| Mutation spot-check | GCC | 7 rules broken one at a time (non-finite verdict, built-in preset name refusal, Wait-*k* pair-default override, malformed device policy, URL space/DEL rejection, PowerShell `$LASTEXITCODE` check, `AppModel::tick` draining Studio analysis) | each caught by a failing suite |
| Windows MSVC | — | — | NOT RUN (no toolchain on this machine; no CI run of this tree) |
| macOS AppleClang | — | — | NOT RUN (same) |
| ORT-enabled build | — | — | NOT RUN |
| Headless GUI smoke (`--smoke-frames`) | — | — | NOT RUN (no Xvfb on this machine) |
| Manual desktop checks (tray on Windows/KDE/GNOME/Wayland, native dialogs, close-to-tray, start minimised) | — | — | NOT RUN |
| Hardware device matrix | — | — | NOT RUN |

The `_wdupenv_s` branch in `frontend/core/Paths.cpp` has not been compiled by
any compiler yet.

Evidence recorded 2026-11-19 (M4 Stage 04; same machine and base; preset
trees `build-linux-*`, `cmake --workflow --preset <p>`, all targets, strict
warnings as errors). Details and mutation table:
[Stage 04 evidence](../Documents/Milestone%204/04_build_gate_frontend_hardening.md#evidence-2026-11-19).

| Leg | Result |
|---|---|
| Linux GCC 13.3.0 / Clang 18.1.3, ORT-less | PASS, 121/121 each |
| Linux GCC 13.3.0 / Clang 18.1.3, ORT 1.31.0 | PASS, 122/122 each |
| Headless GUI smoke in CTest (`gui_smoke_<section>` ×8 + `gui_smoke_mini`, Xvfb + llvmpipe, isolated config dirs) | PASS in all four trees |
| Windows MSVC, macOS AppleClang (incl. the new ORT legs) | NOT RUN (no toolchain; tree not pushed) |
| Manual desktop checks, hardware device matrix | NOT RUN |

`GuiApp::run` returns 2 when it shuts down after 30 consecutive failing
frames (BUG-022 guard); a single failing frame stops the stream, shows the
error and keeps the GUI running.

After M4 Stages 05–06 (same day, same trees): ORT-less 123/123 and ORT
125/125 on GCC and Clang, including `mode_a_host`, `gui_smoke_bundle` (the GUI
loads and warms up `tests/data/fixtures/identity`) and `gui_stats_parse` (GUI
`[stats]` lines through `tools/analyze_soak.py --check-format`). Evidence and
mutation tables: [Stage 05](../Documents/Milestone%204/05_gui_engine_host_mode_a.md#evidence-2026-11-19),
[Stage 06](../Documents/Milestone%204/06_live_latency_telemetry.md#evidence-2026-11-19).
Live audio (conversion, telemetry with a running engine) is NOT RUN.

The Windows and macOS legs run in `.github/workflows/build.yml`; the Linux
legs there install the SDL3 X11 build dependencies first. Live audio routing,
the real tray on each desktop, and native dialogs are manual checks on real
hardware and are not claimed by the table above.

## 11. Not in this iteration

- Mode B in the GUI (translator models, language-pair runtime, translator
  state resets: M4 Stage 07). The GUI loads encoder and vocoder only
  (`TranslatorPolicy::kNever`); `translate = 1` is rejected on the control
  thread until then. Mode A inference, conditioning, the latency governor and
  the fatal-error path are delivered (Stage 05).
- The Stage 06 latency and per-stage timing panels (the Telemetry panel shows
  the Stage 05 counters only: pushed / delivered / accepted / rejected
  snapshots, queued audio identity).
- Global hotkeys (Windows `RegisterHotKey`; Wayland needs the
  GlobalShortcuts portal) and Windows taskbar thumbnail buttons
  (`ITaskbarList3`). Both are additive to `platform/`.
- Desktop notifications, in-app downloads, preview playback of catalogue
  samples.
- GPU execution-provider toggles and quantisation switches (M4 optimisation
  half; they will be Settings entries bound to ORT session options).
