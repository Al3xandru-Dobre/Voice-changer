# Stage 09 — Synthetic-audio marking & disclosure (decision gate)

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 08](08_runtime_profile_integration.md) · [Stage 10 →](10_gui_quick_voice_privacy_centre.md)

**Status:** ☐ not started
**Depends on:** [Stage 01](01_privacy_consent_misuse_architecture.md) (regulatory checkpoints, open decisions 1 and 8),
[Stage 08](08_runtime_profile_integration.md) (`marking_required` hook),
[M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md) (stateful export),
[M2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) (budget, warm-up, timing histogram)
**Owner persona:** `@onnx-ort-runtime` (runtime), `@ml-speech-trainer` (export, robustness evaluation)

## Objective

Decide, by measurement, whether converted audio can carry a machine-readable
mark in real time inside the Mode A budget. If it can, ship it with a public
detector. If it cannot, apply the written fallback. In both cases give the
user a clear way to disclose that a voice is synthetic.

## Context & references

- The EU AI Act, Art. 50(2), requires providers of systems that generate
  synthetic audio to mark the output in a machine-readable, detectable way,
  as far as technically feasible; Art. 50(4) requires deployers to disclose
  deep fakes. Both apply since 2 August 2026 (references in the
  [master plan](00_master.md)). A live voice stream cannot carry metadata
  into a call: an in-band mark is the only form that travels with the audio.
- The repository has no marking code and no detector.
- The output chain order is gain → gate → brightness → limiter
  (`src/voice/VoiceOutputChain.hpp`); the limiter is a soft clip with no
  lookahead.
- Every model in the pipeline is a stateful, fixed-shape graph with
  preallocated bindings, warmed up before `engine.start()`. A marker has to
  follow the same rules or it does not run.

## What is marked

- **Policy (open decision 8):** when a marker is active, all converted
  output is marked.
- **Requirement:** a profile with `marking_required` (unverified or
  third-party) cannot be used without an active marker. The pipeline
  **fails closed**: if the marker fails at run time, conversion into such a
  profile stops with a visible error. It never continues unmarked.
- **The payload identifies the product, never the person.** A fixed code for
  "FluxVoice, payload version 1". No user, profile, device or session
  identifier is ever embedded: that would turn the mark into a tracking tag.

## Candidates

| # | Candidate | For | To verify |
|---|---|---|---|
| 1 | **AudioSeal** (code and weights MIT; 16 kHz; streaming generator since 0.2; sample-level detector) | Designed for speech; localised detection; licence fits the shipped bundle | Exportable to a stateful ONNX graph; per-hop cost on one core; any lookahead; robustness on *our* output through a call codec |
| 2 | In-house spread-spectrum mark (a keyed pseudo-noise sequence shaped below the speech spectrum; correlation detector) | Microseconds per hop; no model; no lookahead | Robustness to low-bitrate codecs is expected to be weak |
| 3 | Metadata only | Free | Useless for a live stream. Done in addition for files: `fv_vc_file` writes a `LIST/INFO` note that the audio is voice-converted |

## Decision gate

Written before measuring. A candidate is **accepted** when all hold, on the
shipped runtime, for hop variants s1, s2 and s4:

- **Cost:** added per-hop p99 ≤ 15 % of the hop on one core.
- **Latency:** no lookahead and no added algorithmic delay; the M2 **Must**
  line (T−B ≤ 100 ms) still holds end to end.
- **Detection:** true-positive rate ≥ 95 % at a false-positive rate ≤ 1 % on
  5-second segments, after each of: Opus at 24 kbit/s, resampling 16 → 48 →
  16 kHz, additive noise at 20 dB SNR, and the virtual-cable loopback. The
  false-positive rate is measured on unmarked converted audio and on natural
  speech.
- **Quality:** UTMOS change ≥ −0.05, WER change ≤ +0.5 points and target
  cosine change ≥ −0.01 on `test_clean_200`, plus written listening notes.

If more than one passes, choose the higher detection rate after Opus.

**If none passes**, record "evaluated and rejected" with the numbers and
apply the fallback: personal profiles are limited to the user's own
*verified* voice, third-party profiles stay disabled (open decision 1), and
disclosure rests on the interface below and the file metadata. Re-evaluate
when a new candidate appears.

## Tasks

- [ ] `python_training/eval_marking.py`: applies a marker to converted
      audio, runs the degradations (Opus round trip through an optional
      local encoder, skipped with a recorded reason when absent), reports
      detection curves and the quality deltas. Unit-tested with a fake
      marker.
- [ ] **Candidate 1:** export the streaming generator as
      `marker_s{1,2,4}.onnx` with every convolution cache exposed as
      `state_in_*` / `state_out_*`, and the detector as `marker_det.onnx`.
      Chunked ONNX Runtime output against the reference implementation for
      more than 20 seconds. Record the upstream version and the weights'
      licence text in `MODEL_CARD.md`. If the generator cannot be made
      strictly causal, record the delay; it counts against the gate.
- [ ] **Candidate 2:** `src/dsp/SpreadSpectrumMark.hpp` (generator and
      detector, allocation-free) with a Python twin and golden vectors.
- [ ] Contract: role `marker` in v3 (fixed-shape audio in, audio out,
      states); validators and fixtures; a pass-through fixture marker for
      the C++ tests.
- [ ] Runtime: marker stage on the worker thread, preallocated, ping-pong
      bindings, warmed up before `engine.start()`, `Ort::Exception` caught
      inside the wrapper. Measure placement before and after the limiter and
      keep the one with the higher detection rate. Its time is a separate
      row in the stage histogram.
- [ ] Fail-closed behaviour and the `marking_required` check from Stage 08,
      replacing the stub and removing `--allow-unmarked-profiles`.
- [ ] `tools/fv_wm_detect.cpp`: `fv_wm_detect <file>` prints present /
      absent / inconclusive with a score per 5-second segment, and resamples
      through `AudioFileReader`. Registered in CTest with marked and
      unmarked fixtures.
- [ ] `fv_vc_file`: the `LIST/INFO` note on every output file.
- [ ] **Disclosure, specified here and built in Stage 10:**
  - a persistent "AI voice active" state in the window, the Mini panel and
    the tray icon while conversion runs;
  - a first-use notice: tell listeners when they could otherwise be misled;
  - a copyable disclosure sentence for stream descriptions and profiles;
  - "Check a recording for the FluxVoice mark" using the detector.
- [ ] Tests: marker stage allocates 0 bytes per hop; a failing fake marker
      stops conversion into a `marking_required` profile and not into a
      bundle voice when policy marking is off; detector fixtures.
- [ ] Fill the results table and record the decision in the
      [master plan](00_master.md).

## Results *(fill in)*

| Candidate | Hop | Added p99 (ms / % hop) | Added delay (ms) | TPR @ 1 % FPR: clean / Opus 24k / resample / noise / cable | ΔUTMOS | ΔWER | Δ`tgt_cos` | Accepted |
|---|---|---|---|---|---|---|---|---|
| AudioSeal | s1 / s2 / s4 | | | | | | | |
| Spread spectrum | s1 / s2 / s4 | | | | | | | |

Decision: …, date. Placement: before / after the limiter.

## Implementation notes

- **What a mark is for.** It lets a platform, a call service or the person
  whose voice it is check a recording. It helps the honest majority and
  leaves evidence. It can be removed by someone who tries, and it does not
  survive every channel. It is never described as a security boundary.
- **A public detector helps removal as well as detection.** That trade is
  accepted: a mark nobody else can check protects nobody.
- **Why the payload is constant.** A per-user payload would let recordings
  be linked to a person. That conflicts with the purpose-limitation rule in
  Stage 01 and with why users would trust the application at all.
- **Decisions are made on the shipped runtime**
  ([BUG-015](../Bug%20Report/BUG-015_gate_decides_on_pytorch_clock.md)):
  ONNX Runtime per-hop timings, not PyTorch eager.

## Files touched

- `python_training/eval_marking.py` (new), `fvml/export.py`,
  `export_stateful_onnx.py`, `fvml/contract.py`,
  `src/pipeline/ModelContract.hpp`, `ModelBundle.{hpp,cpp}`,
  `InferencePipeline.{hpp,cpp}`, `src/dsp/SpreadSpectrumMark.hpp` (new),
  `tools/fv_wm_detect.cpp` (new), `tools/fv_vc_file.cpp`, `src/main.cpp`,
  `tools/make_fixture_models.py`, `tests/test_marking.cpp` (new),
  `tests/test_inference_pipeline.cpp`, `CMakeLists.txt`, this file (results)

## Definition of done

- The results table has measured rows for both candidates and a dated
  decision made by the written rule.
- If a candidate is accepted: marked output passes `fv_wm_detect`, the
  fail-closed test passes, the per-hop cost is in the histogram and inside
  the budget.
- If none is accepted: the fallback is in force in the code (third-party
  profiles disabled, unverified profiles not selectable), and the master
  plan records the deviation.
- The disclosure specification is complete enough for Stage 10 to build.
