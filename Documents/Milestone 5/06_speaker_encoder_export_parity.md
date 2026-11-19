# Stage 06 — Speaker-encoder export & enrolment parity

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 05](05_on_device_embedding_refinement.md) · [Stage 07 →](07_cpp_enrolment_engine_cli.md)

**Status:** ☐ not started
**Depends on:** [Stage 02](02_voice_profile_contract_store.md) (contract v3, `speaker_space`),
[Stage 04](04_speaker_encoder_vocoder_gate.md) (chosen checkpoints, `fvml/enroll.py`),
[M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md) (exporter, parity tooling).
The fixture path needs no trained weights and can start after Stage 02.
**Owner persona:** `@onnx-ort-runtime`

## Objective

Export the speaker encoder as `speaker_enc.onnx`, define the enrolment
algorithm precisely enough that Python and C++ produce the same profile from
the same samples, and pin it with golden vectors built from synthetic audio.

## Context & verified starting point (2026-11-18)

- `export_stateful_onnx.py` exports one fixed-shape graph per role and hop
  variant, embeds the contract in `metadata_props["fluxvoice.contract"]`,
  writes `voices.json`, `export_report.json`, `MODEL_CARD.md` and golden
  files, and reports any exporter fallback
  ([BUG-016](../Bug%20Report/BUG-016_gate_export_silent_exporter_fallback.md)).
  Re-export clears stale goldens
  ([BUG-013](../Bug%20Report/BUG-013_stale_golden_survives_reexport.md)).
- `voices.json` embeddings are read from the vocoder's speaker table
  (`UnitVocoder.voices_json`). After Stage 04 they come from the enrolment
  function instead.
- The project's rule for anything implemented twice is one reference, one
  port, and golden vectors between them: causal YIN
  (`tests/test_pitch_tracker.cpp`) and the Wait-*k* policy
  (`tests/data/waitk/golden_trajectories.txt`).

## The enrolment algorithm (`enroll_algo = 1`)

Input: mono float samples at the contract's `sample_rate`, and the bundle's
`speaker_enc` contract. Output: an embedding and four statistics, or a typed
refusal.

1. **Speech selection.** Energy-based speech detection on 20 ms frames (the
   rules of `fvml/target_voice.py::vad_segments`, with the noise floor taken
   as the 10th percentile of frame levels when no room tone is given).
   Speech regions from all files are concatenated in input order.
2. **Cap.** If more than 10 minutes of speech remain, take evenly spaced
   regions to reach 10 minutes. The choice depends only on the region list.
3. **Windows.** Slide `window_samples` (3 s) with `window_hop_samples`
   (1.5 s). The last window is kept if it holds at least
   `min_speech_samples`; it is zero-padded.
4. **Encode.** Run `speaker_enc` on each window → one vector per window.
5. **Pool.** Weighted mean with the window's speech sample count as weight;
   L2-normalise; multiply by `embedding_scale`.
6. **Consistency.** Cosine of each window (and of each file's own mean)
   against the leave-one-out centroid. Reported, used for the "more than one
   voice?" warning in Stage 07; it does not change the embedding.
7. **Statistics.** Run the causal tracker over the selected speech with the
   live configuration: `logf0_mean` / `logf0_std` over voiced frames,
   `energy_mean` / `energy_std` over speech frames, with the same floors as
   `compute_speaker_stats()`.
8. **Refusals.** Fewer than 30 s of speech (the Stage 03 threshold, once
   measured), fewer than 500 voiced frames, or non-finite output.

Any change to these steps is a new `enroll_algo` number and new goldens.

## Tasks

- [ ] `fvml/enroll.py` finalised as the reference for `enroll_algo = 1`,
      returning the embedding, statistics, window count, weights and
      consistency scores. No torch dependency outside the encoder call, so
      the same function runs over PyTorch and over ONNX Runtime.
- [ ] **Frame selection for statistics.** Make `compute_speaker_stats()` and
      `enroll()` select frames by the same rule (speech frames for energy),
      and recompute the curated voices' statistics with it. Change both or
      neither.
- [ ] `fvml/export.py`: `speaker_enc_graph()`: fixed-shape
      `audio [1, window_samples]` → `spk_emb [1, spk_emb_dim]`, mel front-end
      inside the graph, no states.
- [ ] `export_stateful_onnx.py --role speaker_enc`: writes `speaker_enc.onnx`
      with a v3 contract; computes `speaker_space` from the encoder and the
      vocoder FiLM weights and writes it into both contracts and into
      `export_report.json`.
- [ ] `voices.json` for the curated voices produced by `enroll()` over a
      fixed, listed set of each voice's utterances. The embedding table is no
      longer the source.
- [ ] `MODEL_CARD.md`: speaker-encoder training corpora and attribution; the
      "private" banner logic is unchanged (a default bundle holds no private
      voice).
- [ ] **Parity, PyTorch against ONNX Runtime** (`parity.py`): per-window
      embedding cosine ≥ 0.9999 and a relative difference ≤ 1e-4 on
      non-silent input, over at least 200 windows from the Stage 03 lists;
      full `enroll()` over ORT against PyTorch: cosine ≥ 0.9999, statistics
      identical.
- [ ] **Synthetic fixtures** (`tools/make_enroll_fixtures.py`, standard
      library only, `--self-test`, registered in CTest): deterministic
      speech-like signals (a harmonic source with a slow F0 contour, two
      formant resonances, pauses, a noise floor), written as 16 kHz and
      44.1 kHz WAV. Three "voices", one two-voice file, one too-short file,
      one clipped file.
- [ ] **Golden vectors** `tests/data/enroll/golden/*.json`, computed by
      `make_enroll_golden.py` with the fixture `speaker_enc`: selected
      regions, window count and weights, embedding, statistics, consistency
      scores and the expected refusal for the bad files.
- [ ] `tools/make_fixture_models.py`: the `speaker_enc` fixture from
      Stage 02 gains a non-trivial but tiny body (one convolution, mean and
      standard-deviation pooling) so that parity is meaningful.
- [ ] `models/README.md`: bundle layout with `speaker_enc.onnx`.
- [ ] Python tests (`test_export.py`, `test_enroll.py`) and CTest
      registration.

## Implementation notes

- **Fixed shapes, host-side pooling.** A fixed 3-second graph follows the
  project's export discipline and keeps the variable-length logic in code
  that both languages can test line by line.
- **Synthetic fixtures only.** The repository must not contain a real
  person's voice, even as a test file. The fixture signals need to exercise
  speech detection, pitch tracking and windowing, not sound like speech.
- **Resampling is outside the algorithm.** `enroll()` starts from samples at
  the model rate. How C++ gets there from a file is Stage 07's concern, with
  its own tolerance.
- **Operator set.** As for the other roles: IR ≤ 13, opset ≤ 26 for the
  pinned ONNX Runtime 1.31.

## Files touched

- `python_training/fvml/enroll.py`, `fvml/export.py`, `fvml/vocoder.py`,
  `export_stateful_onnx.py`, `train_unit_vocoder.py`, `parity.py`,
  `make_enroll_golden.py` (new), `tools/make_enroll_fixtures.py` (new),
  `tools/make_fixture_models.py`, `tests/data/enroll/**` (new),
  `python_training/tests/test_enroll.py`, `test_export.py`,
  `models/README.md`, `CMakeLists.txt`

## Definition of done

- A bundle exported from the Stage 04 checkpoints contains
  `speaker_enc.onnx`, matching `speaker_space` values, and a `voices.json`
  produced by `enroll()`; the contract validators accept it.
- PyTorch ↔ ONNX Runtime parity meets the thresholds above, recorded in
  `export_report.json`.
- The golden vectors exist, are reproducible from the scripts, and the
  Python test that checks them fails when any step of the algorithm is
  changed.
- If no trained checkpoints exist yet, the fixture path (fixture model,
  fixtures, goldens, tests) is complete and the trained-bundle rows are
  marked pending with the date.
