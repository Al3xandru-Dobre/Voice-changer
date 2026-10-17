# Stage 09 — Voice controls & presets

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 08](08_realtime_mode_a_pipeline.md) · [Stage 10 →](10_target_voice_finetune.md)

**Status:** ☐ not started
**Depends on:** Stage 08 (pipeline + hooks), Stage 05 (control-ready conditioning; readiness checks)
**Research:** §11 voice-control design · §4.2 target-domain mapping · §5 speaker-embedding FiLM

## Objective

Let the user shape the output voice live, and save and recall those settings as
presets. The engine-side work lands here:
- a single **parameter descriptor table**;
- a lock-free path from the control thread to the worker;
- smoothing, so changes don't click;
- a JSON preset format with built-in presets;
- CLI flags and live console commands.

The descriptor table is the spec that
[Milestone 4](../Milestone%204/00_master.md)'s ImGui sliders consume, so the
sliders need no new engine work.

## Context & references

- **Controls act on the vocoder's conditioning, not on audio** (research §11):
  - pitch shift and intonation scale modify `logf0_tgt` in the target-domain
    mapping (`logf0_tgt = μ_tgt + scale·σ_tgt·z + shift·ln2/12`, Stage 03);
  - timbre blend interpolates two `voices.json` embeddings into `spk_emb`
    (Stage 05 trained FiLM on a vector for exactly this).

  None of these adds latency.
- Stage 05's readiness checks (±12 st shift, scale 0/1/2, 50/50 blend) give the
  first evidence of usable ranges.
- **Console input.** The stdin stop thread (`src/main.cpp:289-291`) treats Enter
  as stop. With `--duration`, stdin isn't read at all (`:286-288`), so scripted
  runs keep working.
- **Locking.** `readerwriterqueue` is already a dependency for lock-free SPSC
  hand-off (M1 / Stage 03). nlohmann/json arrives in Stage 02.

## Parameters

| Name | Unit | Range | Default | Acts on | Smoothing | Tier |
|---|---|---|---|---|---|---|
| `pitch_shift` | semitones | −12 … +12 | 0 | `logf0_tgt` offset | per-frame ramp | MVP |
| `intonation` | × | 0 … 2 | 1 | Scales whitened z (0 = robot monotone) | per-frame ramp | MVP |
| `voice_a` / `voice_b` | voice name | `voices.json` | bundle default / = `voice_a` | Embedding endpoints | — (switch crossfades via blend) | MVP |
| `timbre_blend` | 0…1 | 0 … 1 | 0 | `spk_emb = (1−α)·e_A + α·e_B` | per-frame ramp | MVP (experimental quality) |
| `gain` | dB | −24 … +12 | 0 | Output samples | per-sample one-pole (~20 ms) | MVP |
| `gate_threshold` | dBFS | −80 … −20 | −55 | Output gate on input energy (attack/hold/release) | envelope | MVP |
| `limiter` | on/off | — | on | Soft clip, ceiling −1 dBFS, no lookahead | — | MVP |
| `vibrato_rate` / `vibrato_depth` | Hz / cents | 0…8 / 0…100 | 0 / 0 | LFO on `logf0_tgt` | per-frame | Stretch |
| `whisper` | 0…1 | 0 … 1 | 0 | Voicing input scaled towards 0 | per-frame ramp | Stretch |
| `brightness` | dB tilt | −6 … +6 | 0 | First-order tilt EQ on output | per-sample | Stretch |

**Deliberately excluded:**
- **Dry/wet mix.** The wet path lags the mic by the pipeline latency, so mixing
  comb-filters and lets the real voice through.
- **Formant DSP.** The timbre blend covers it; revisit in M4 or later.

## Tasks

- [ ] **`src/voice/VoiceParams.hpp`:**
  - [ ] `constexpr ParamDesc kParams[]` (name, unit, min, max, default,
        smoothing kind, tier).
  - [ ] A `VoiceParams` value type (float array indexed by an enum + two voice
        indices); `clamp()`.
  - [ ] Lookup by name.
- [ ] **Control → worker.**
  - [ ] Control thread pushes a full `VoiceParams` snapshot into a preallocated
        `moodycamel::ReaderWriterQueue<VoiceParams>`.
  - [ ] Worker drains to the newest at each hop start. Latest wins; no locks,
        no allocation.
- [ ] **Apply in `InferencePipeline`:**
  - [ ] the f0 mapping uses `pitch_shift`, `intonation` (and vibrato/whisper
        when enabled);
  - [ ] `spk_emb` is recomputed per frame from the smoothed `timbre_blend`;
  - [ ] the output chain is gain → gate → limiter (→ brightness).
  - [ ] Switching `voice_a` while `timbre_blend` = 0 crossfades through a short
        internal blend ramp, so it doesn't jump.
- [ ] **Presets.**
  - [ ] Format: `presets/*.json` with `{"schema": 1, "name", "description",
        "voice_a", "voice_b", "params": {...}}`.
  - [ ] Loader: clamps out-of-range values with a warning, warns on unknown keys,
        rejects an unknown schema.
  - [ ] Built-ins in `presets/`: **Neutral**, **Deep** (−4 st, intonation 0.9),
        **Bright** (+4 st, intonation 1.1), **Robot** (intonation 0), and
        **Whisper** (stretch tier).
  - [ ] `--preset-dir` for user presets.
- [ ] **CLI:**
  - [ ] `--preset <name|path>`
  - [ ] repeatable `--param name=value`
  - [ ] `--list-params` (human table; `--list-params --json` prints the descriptor
        table as JSON, the M4 slider spec)
  - [ ] `--list-presets`
- [ ] **Console.** Replace the stop thread with a line-reader thread that posts
      commands to the control loop through an SPSC queue. Commands:
  - empty line = stop (unchanged)
  - `set <param> <value>`, `get [param]`, `preset <name>`, `save <name>`,
    `voices`, `help`

  Not read under `--duration`, unless `--console` is given.
- [ ] **`fv_vc_file`** gains `--preset` / `--param`, so every control is measurable
      offline.
- [ ] **Tests** (`tests/test_voice_params.cpp`, ORT-free):
  - [ ] every descriptor has min < default < max (or equal bounds for toggles)
  - [ ] preset round-trip; clamping; unknown-key warning; bad schema rejected
  - [ ] f0-mapping arithmetic vs. closed form
  - [ ] blend arithmetic
  - [ ] a step change in gain/blend through the fake pipeline stays under the
        click bound

## Implementation notes

- **One table, four consumers.** CLI flags, console commands, preset keys and the
  M4 sliders all come from `kParams`. Adding a parameter is one line plus its
  `apply` code, and the UI picks it up automatically.
- **Range honesty.** Stage 05's readiness checks show where quality falls apart
  (e.g. large shifts outside the voice's training range). Set descriptor ranges
  to what passes this stage's DoD, not to what is mathematically possible.
- **Timbre blend is experimental.** With 4–8 training voices the embedding space
  is sparse, so midpoints may sound less natural. Presets should use trained
  voices (blend 0 or 1) unless the blend DoD passes comfortably.
- **The gate does double duty.** It is a user control, and it stops the vocoder
  from producing babble from room noise. It also tells the latency governor when
  output may be dropped (Stage 08).
- **Formant-preserving pitch shift comes free.** The vocoder renders the voice's
  formants from `spk_emb`, so shifting f0 doesn't produce the chipmunk effect of
  waveform pitch-shifting.

## Files touched

- `src/voice/VoiceParams.hpp` (new), `src/voice/Presets.{hpp,cpp}` (new),
  `presets/*.json` (new), `src/pipeline/InferencePipeline.{hpp,cpp}`,
  `src/main.cpp` (CLI, console thread), `tools/fv_vc_file.cpp`,
  `tests/test_voice_params.cpp` (new), `CMakeLists.txt`

## Definition of done

Measured offline with `fv_vc_file` on `test_clean_200`, unless noted:
- `pitch_shift` N ∈ {−7, −3, +3, +7} st: median output-f0 ratio within ±0.3 st of
  N.
- `intonation` s ∈ {0.5, 1.5}: voiced log-f0 standard-deviation ratio (output vs.
  `intonation`=1) within s ± 10%; s = 0 gives a std below 0.1 semitone (robot).
- `timbre_blend` α ∈ {0, 0.25, 0.5, 0.75, 1}: ECAPA cosine to voice A decreases
  monotonically, and UTMOS at α = 0.5 is ≥ 3.0.
- A step change of any MVP parameter adds no sample-to-sample delta above the
  99.9th percentile of the unchanged output (click test).
- WER for each built-in preset ≤ Neutral's WER + 3 pts.
- Live: `set`, `preset` and `--preset` change the sound within one hop + the
  smoothing time, with no `misses`.
