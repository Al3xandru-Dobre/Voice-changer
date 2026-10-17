# Textless voice conversion (Mode A): encoder, vocoder, export and runtime

- **Project:** FluxVoice-S2UT
- **Research date:** 2026-10-18
- **Status:** Research and proposed plan; no implementation changes

> **Implementation tracking.** This report is preserved as a dated research
> snapshot. Its recommendations are scheduled in
> [Milestone 2 — Textless VC mode](../Milestone%202/00_master.md), stages 01–11
> (mapping in §13). It also adds notes to the Milestone 3 and 4 masters and to
> [Milestone 1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md).
> Where this snapshot and a stage file disagree, the stage file is
> authoritative.

## 1. Summary, scope and method

**Recommendation.**
- **Build the Mode A pipeline so it works with any hop length.** Pick the
  content encoder in a measured bake-off where latency is the main criterion.
- **Expect a causal 50 Hz content encoder to win.** It follows the StreamVC
  recipe: a small causal conv net trained to predict HuBERT k-means labels.
  It runs at 16 kHz with 20 ms frames, and its lookahead can be set to 0, 1 or 2
  frames.
- **Give Mimi's semantic codebook a short kill test, not a full arm.** Its fixed
  80 ms frame alone costs more than three quarters of the README's 100 ms budget
  (§3). Published voice-conversion tests also show the codebook on its own loses
  most of the linguistic content (§2.3).
- **Vocoder.** Use a causal, stateful, multi-speaker HiFi-GAN conditioned on:
  - discrete unit ids
  - target-domain log-F0, voicing and energy from a causal YIN tracker (§4)
  - a speaker-embedding vector

  These conditioning inputs are also what make the voice controls cheap (§11).
- **Deployment.** Export to fixed-shape stateful ONNX and run it on the CPU
  execution provider, ONNX Runtime 1.31.x, with one intra-op thread on the
  worker thread (§8, §9).

**Scope.**
- In:
  - Mode A only (mic → units → vocoder → target voice).
  - The encoder, pitch features, vocoder, training data, evaluation, export,
    the C++ runtime, real-time scheduling and voice controls.
- Out:
  - Translation (Milestone 3).
  - GPU execution providers, quantization and the GUI (Milestone 4).

**Method.**
- Web search and primary-source reading on 2026-10-17/18: arXiv papers, Hugging
  Face model cards, the ONNX Runtime and PyTorch documentation, GitHub issues
  and PRs.
- Each number below carries its source. Secondary summaries (reviews, news
  write-ups) were used only to locate primary sources. Values that could not be
  confirmed from a primary source are marked *unverified*.
- **Limitations:**
  - No experiment was run.
  - Latency figures for FluxVoice are estimates until Stage 08/11 measure them.
  - The Mimi-related VC results below come from codec token-swap or kNN
    protocols, not from a unit vocoder trained on q0. That is why §2.7
    recommends a kill test rather than outright rejection.

## 2. Encoder landscape

### 2.1 What the M2 encoder must do

| Requirement | Why |
|---|---|
| Strictly causal (bounded, configurable lookahead) | Every lookahead frame is latency (§3) |
| Speaker-independent output | Mode A depends on the encoder removing the source timbre |
| Discrete units | Milestone 3 translates unit sequences; the vocoder must accept the translator's output |
| Cheap on one CPU core | The per-hop compute counts as latency (§3) |
| Frame rate ≥ hop rate | The model frame length is the smallest possible hop |

### 2.2 Kyutai Mimi

Facts from the transformers `MimiConfig` defaults and the kyutai/mimi model
card:
- 24 kHz mono input, 12.5 Hz frames (one per 80 ms), 1.1 kbps.
- RVQ with `num_quantizers=32` of `codebook_size=2048`. `num_semantic_quantizers=1`:
  q0 is distilled from WavLM.
- A causal SEANet conv encoder (`use_causal_conv=True`, ratios `[8, 6, 5, 4]`)
  followed by an 8-layer transformer with `sliding_window=250`. The transformer
  runs at 25 Hz before the final downsample, so the window is about 10 s.
- `MimiModel.encode(input_values, …, encoder_past_key_values, padding_cache,
  use_streaming)` supports streaming in Python.
- 96.2M parameters (encoder and decoder), CC-BY 4.0.

Community ONNX precedent:
- `maai-kyoto/continuous-mimi-onnx` exports the streaming encoder with an
  80 ms emit. Its KV cache is a ring buffer of `sliding_window + steps − 1` slots
  with absolute-position masks.
- Before a fix in 2026-10, its output drifted from Torch once the ring wrapped
  at 10 s: cosine similarity fell to about 0.97 and the error to about 5e-2.
  After the fix, parity over 400 steps is ≤ 1.3e-6.
- It ships 1-frame and 2-frame variants.

### 2.3 Evidence on Mimi q0 for voice conversion

- **Kanade** (arXiv 2602.00594, Table 2). The VC test swaps first-layer content
  tokens from the source with the rest from a reference. Mimi scores WER 120.3,
  CER 86.8, UTMOS 3.086, similarity 81.7, F0 correlation 0.213. Kanade 12.5 Hz
  scores WER 1.6. The authors say Mimi's "original content is nearly lost and
  the reference speech is reconstructed".
- **FocalCodec-Stream** (arXiv 2509.16195, VCTK VC). Mimi with 6 codebooks:
  UTMOS 2.62, dWER 110.00, similarity 91.3. FocalCodec-S@50-65k: 3.10 / 22.71 /
  92.5.
- **Codec interpretability** (arXiv 2506.04492). Mimi's higher RVQ levels "still
  contain a considerable amount of linguistic content", which the authors
  attribute partly to the 12.5 Hz rate.

**Reading.** At 12.5 Hz the first codebook does not hold all the content. Some
of it leaks into the acoustic codebooks, which a speaker-independent pipeline
must discard. These tests use codec decoders, not a unit vocoder trained on q0,
so the evidence justifies a time-boxed kill test (PNMI, a speaker probe and a
quick vocoder) rather than a full bake-off arm.

### 2.4 StreamVC recipe

StreamVC (Google, ICASSP 2024; arXiv 2401.03078):
- **Content encoder.** A causal SoundStream-style conv net (C=64, D=64, no FiLM)
  trained with cross-entropy to predict HuBERT-Base layer-7 labels, clustered
  with mini-batch k-means into 100 centroids, at 50 Hz (320 samples).
- **Units.** Its pre-softmax output is used as soft units.
- **Disentanglement.** Blocking decoder gradients into the content encoder is
  "essential" to stop extra speaker information leaking.
- **Pitch.** YIN f0 at thresholds 0.05/0.10/0.15, plus cumulative-mean-normalized
  difference values and an unvoiced flag: 9 values per 20 ms frame. Energy is
  the sample variance.
  - f0 is whitened with utterance mean and standard deviation in training, and
    with running averages when streaming.
  - Ablation: whitening vs. random offset/scale gives speaker similarity 77.81 vs
    75.59 and f0 PCC 0.842 vs 0.704. With no f0 at all, PCC falls to 0.461.
- **Latency.** A 2-frame lookahead (output paired with input s_{t−2}) gives 60 ms
  architectural latency. Compute is 10.8 ms per 20 ms chunk on one Pixel-7 core
  (XNNPACK), and end-to-end latency is 70.8 ms.

### 2.5 Other candidates (watch list)

| Candidate | Facts | Why not the M2 default |
|---|---|---|
| FocalCodec-Stream (2509.16195) | 16 kHz in, 24 kHz out, 50 Hz, single binary codebook (2k/4k/65k), 0.55–0.80 kbps, 80 ms theoretical latency including up to 3 frames of chunked-attention lookahead, ~249M params | Too heavy for one CPU core at 50 Hz; code release announced but not verified |
| Kanade (2602.00594) | 12.5/25 Hz FSQ tokens (vocabulary 12,800) from WavLM Base+ layers 6+9, global speaker embedding, ~120M trainable parameters; VC WER 1.6 / 0.7 | Not described as causal; only a chunked streaming prototype |
| LLVC (2311.00873) | Waveform-to-waveform any-to-one VC, < 20 ms latency at 16 kHz, 2.8× real time on a consumer CPU (author claims), MIT code | Not unit-based, so it cannot feed M3. Useful as a CPU-budget reference |
| speech-resynthesis (2104.00355) | HuBERT units + F0 + speaker → HiFi-GAN; HuBERT-100 VC WER 12.23, EER 0.31 on LJ | Non-causal reference design; its conditioning layout is reused |

### 2.6 Teacher model for the 50 Hz student

| Teacher | Facts | Use |
|---|---|---|
| HuBERT-Base (`facebook/hubert-base-ls960`) | English (LibriSpeech 960 h), ~95M params, fairseq lineage (MIT); loads in `transformers` | **M2 default**; StreamVC's teacher |
| mHuBERT-147 (`utter-project/mHuBERT-147`) | 147 languages, ~95M params, K=1000 Faiss units, ~90k h; **CC-BY-NC-SA-4.0** | Milestone 3 option for multilingual units; the licence forbids commercial use |

`fairseq` and `textlesslib` do not install cleanly on current Python versions,
so plan on `transformers` for feature extraction and `scikit-learn` for
MiniBatchKMeans.

### 2.7 Conclusion

- **Candidate A:** the StreamVC-style causal student, 50 Hz, 16 kHz, discrete
  km100 ids, lookahead L ∈ {0, 1, 2}.
- **Candidate M:** Mimi q0, 12.5 Hz, 24 kHz, behind a one-week kill test.
- Decide in a gate that ranks eligible candidates by projected latency (§3,
  Stage 05).

## 3. Latency analysis

The added delay T−B (fluxvoice in → out, excluding the virtual cable) breaks
down into:

| Component | Meaning |
|---|---|
| I/O floor | Capture period + playback buffering, measured as passthrough T−B. M1/06 has no numbers yet |
| Framing | One hop must accumulate before it can be processed |
| Lookahead | L future frames per model (encoder + vocoder) |
| Compute | Per-hop processing; output must be queued ahead of it, so its p99 counts |
| Governor margin | Jitter headroom kept in the output queue |
| OLA overlap | Only if the vocoder emits `chunk + overlap` (it needs lookahead to do so) |

**Underruns are permanent.** Milestone 1 / Stage 06 found that every underrun in
the queue design adds its length to the delay permanently (`queued =
underruns − overruns − drops`). A slow first ONNX `Run` or one compute spike
would therefore raise latency for the rest of the session. Two consequences:
- warm the sessions up before the stream starts;
- add a governor that gives the excess back during silence (§10).

**Budget** (ms, 16 kHz, 10 ms device periods, estimates):

| Component | A0: L=0, hop 20 | A1: L=1, hop 20 | A2: L=2, hop 20 | A1-40: L=1, hop 40 | M: Mimi, hop 80 @24 kHz |
|---|---|---|---|---|---|
| I/O floor | 25 | 25 | 25 | 25 | 25 |
| Framing = hop | 20 | 20 | 20 | 40 | 80 |
| Lookahead (L × 20) | 0 | 20 | 40 | 20 | 0 |
| Compute p99 (≤ 50% of hop) | 10 | 10 | 10 | 20 | 40 |
| Governor margin | 5 | 5 | 5 | 5 | 10 |
| **fluxvoice-added T−B** | **60** | **80** | **100** | **110** | **155** |
| Algorithmic only (framing + lookahead) | 20 | 40 | 60 | 60 | 80 |

Observations:
- **20 ms periods add about 20 ms;** an enabled OLA overlap adds the overlap.
- **Hop length is a free choice only with a 50 Hz encoder.** It must be a whole
  number of model frames.
- **Smaller hops cost more per second.** Each ONNX `Run` has a fixed overhead,
  so measure compute at 1, 2 and 4 frames per step.
- **Reaching the README's 60–100 ms range** needs L ≤ 1 with 10 ms periods
  (A0/A1). StreamVC's 70.8 ms on a phone (A2-like) shows the budget is
  realistic.

## 4. Pitch and energy features

### 4.1 Causality of candidate trackers

| Tracker | Facts | Causal? |
|---|---|---|
| SwiftF0 (2508.18440, MIT) | 16 kHz, 16 ms hop, tiny CNN on an STFT, ships ONNX; "streaming needs 176 ms of audio after a frame before that frame is final" (README) | **No** — 176 ms lookahead |
| FCPE (2509.15140) | Lynx-Net on log-mel, 360 pitch bins, RTF 0.0062 on an RTX 4090 | No causal mode documented |
| RMVPE | Popular in RVC; bidirectional recurrent layers (architecture-based reading, *unverified* for streaming variants) | No |
| YIN (classic) | Difference function on a past-sample window; StreamVC reports negligible compute | **Yes** |

The parameter count of SwiftF0 differs between the paper (95,842) and the
README (14,386). That doesn't matter here, because the lookahead rules it out.

### 4.2 Recommended features (adapted from StreamVC)

- **Pre-filter.** A high-pass filter at about 70 Hz removes rumble and DC that
  bias YIN and energy.
- **YIN.** Run it every 20 ms on a 40 ms window of *past* samples (two periods at
  50 Hz F0), plus the contract lookahead if L > 0. Search 55–1000 Hz.
- **Outputs per frame:**
  - f0 at threshold 0.10 (the primary track)
  - aperiodicity: the CMND minimum
  - soft voicing: the mean of the voiced flags at 0.05/0.10/0.15
  - log-energy
- **Whitening.** Use running (EMA) mean and standard deviation of log-F0 over
  voiced frames, and of log-energy over all frames. Start from population
  priors so early frames are sane, and floor the standard deviation.
- **Target-domain conditioning (our change).** StreamVC feeds whitened f0 and
  lets the decoder learn the target range. We need explicit pitch controls, so:
  - the vocoder is trained on the *target speaker's own* log-F0;
  - at run time the source contour is whitened, then mapped into the target
    domain with that voice's stats: `logf0_tgt = μ_tgt + scale · σ_tgt · z +
    shift_st · ln2 / 12`.

  The source's absolute pitch range never reaches the vocoder, which is the
  leakage StreamVC's ablation warns about.
- **Energy.** Map it the same way. Microphone gain must not push the
  conditioning outside the training range.
- **Training/run-time parity.** The same YIN implementation is used for training
  features (Python) and at run time (C++), and golden vectors keep the two
  identical.

## 5. Causal vocoders

- **Baseline.** A unit HiFi-GAN as in speech-resynthesis, converted to causal
  form:
  - every convolution is left-padded by `(k−1)·d`;
  - transposed convolutions trim all excess on the right (the same convention as
    Mimi's `trim_right_ratio=1.0`).

  With no lookahead, stateful chunked inference is **exactly** equal to offline
  inference. The streaming state is the per-layer left context.
- **Size.** From the HiFi-GAN paper (Kong et al. 2020): generator V1 ≈ 13.9M
  parameters, V2 ≈ 0.9M, V3 ≈ 1.5M. A V2-class generator fits the one-core
  budget and an 8 GB training GPU.
- **iSTFT / Vocos heads.** They are cheaper, but each output frame depends on its
  neighbours through overlap-add (arXiv 2609.25379). Streaming them needs
  stateful iSTFT OLA. A block-streaming comparison of HiFi-GAN, iSTFTNet, Vocos
  and Wavehax (arXiv 2506.03554) swaps LayerNorm for BatchNorm in Vocos for
  streaming. Treat this as a later optimisation.
- **The causal quality gap.** A causal AudioDec-based VC (arXiv 2510.09245)
  reports lower intelligibility, quality and similarity without future context.
  Mitigations, all measured in the gate:
  - one or two frames of lookahead;
  - distilling from a non-causal teacher (FocalCodec-Stream, DLL-APNet arXiv
    2509.13667).
- **Conditioning for controls.**
  - Speaker identity enters through FiLM layers driven by a speaker-embedding
    *vector*, as StreamVC does.
  - A learned table holds one row per training speaker. Blending two rows gives
    a timbre slider; interpolation quality is experimental and gets its own DoD.
  - Unit ids enter through an embedding table.
  - Log-F0, voicing and energy enter as frame-rate features.
- **OLA defaults off.** The M1 `OverlapAddProcessor` expects `chunk + overlap`
  samples. A stateful causal vocoder already produces continuous output and
  could only emit `overlap` extra samples by looking ahead, which costs latency.
  OLA stays available through a contract field for stateless fallback models.
- **Discrete vs. soft content.**
  - Discrete ids keep compatibility with Milestone 3 and disentangle speakers
    better.
  - Training the vocoder on the **teacher's** ids lets one vocoder serve every
    student variant; it is then fine-tuned on the winner's ids.
  - Soft units are a fallback if discrete ids fail the WER floor.

## 6. Training data and recording

| Dataset | Facts | Use |
|---|---|---|
| LJSpeech | 1 speaker, ~24 h, 22.05 kHz, public domain | Bootstrap voice |
| Hi-Fi TTS (OpenSLR 109) | 10 speakers, ≥ 17 h each (~291.6 h), 44.1 kHz, CC BY 4.0 | Multi-speaker vocoder bootstrap; voices for presets and blending (attribution required) |
| LibriSpeech (OpenSLR 12) | 960 h, 16 kHz, CC BY 4.0 | Teacher labels, student training (train-clean-100/360), evaluation (test-clean) |
| VCTK | 110 speakers, 48 kHz, CC BY 4.0 | Optional accent evaluation |
| MUSAN (OpenSLR 17) / RIRs (OpenSLR 26/28) | Noise, music and babble; simulated and real room impulse responses | Student input augmentation; licences vary by subset, so record them in the Stage 01 licence table |

**Own-voice recording protocol:**
- 48 kHz/24-bit, dry treated room, one microphone, fixed distance (15–30 cm),
  pop filter, peaks around −6 dBFS.
- 1–3 h of phonetically varied material: CMU ARCTIC prompts, Harvard sentences,
  read passages and spontaneous speech, *including questions and emotional
  reads*, because the vocoder must cover the pitch range the controls will ask
  for.
- 30 s of room tone per session.

**Preprocessing.** High-pass filter, −23 LUFS loudness, VAD segmentation into
2–12 s, reject clipped segments, resample to the pipeline rate.

**Robustness.** NANSY-style information perturbation: formant/pitch shift and
random EQ applied to training audio *before* unit extraction. It makes the
vocoder robust to the residual speaker information in units from other people.

**Realistic evaluation input.** About 10 minutes of the user speaking through
the real microphone and room. It is the only test input that matches
deployment.

## 7. Evaluation methodology

| Metric | Tool | Notes |
|---|---|---|
| WER / CER | Whisper large-v3 + its text normaliser | VERSA's default WER model; aggregate at corpus level |
| Target / source similarity | ECAPA-TDNN cosine (SpeechBrain `spkrec-ecapa-voxceleb`); EER | Similarity to the *source* is the leakage indicator |
| Naturalness | UTMOS (UTMOS22-strong, or UTMOSv2) | Reference-free; record the version |
| Prosody | F0 Pearson correlation against an offline (non-causal) reference tracker | |
| Unit quality | Phone purity, cluster purity, PNMI (HuBERT paper) against Montreal Forced Aligner LibriSpeech alignments | Compute at each model's own frame rate |
| Speaker leakage in units | Speaker-ID probe on unit sequences (chance = 1/N) | Lower is better |
| Cost | Single-core per-step p95, RTF | PyTorch with `torch.set_num_threads(1)`, then ORT with `intra_op=1` |

- **VERSA** (arXiv 2412.17667) bundles Whisper WER, UTMOS/DNSMOS and speaker
  similarity (via ESPnet-SPK). A small custom `eval_vc.py` is preferred so the
  ECAPA model and versions are pinned and the dependency footprint stays small.
- **The same metric set** (Whisper WER, UTMOSv2, ECAPA cosine, EER) is common
  practice in recent VC papers, e.g. arXiv 2507.04817.

## 8. Stateful ONNX export

- **Exporter.**
  - Since PyTorch 2.9, `torch.onnx.export` defaults to `dynamo=True` (the
    `torch.export` pipeline). `dynamic_shapes` replaces `dynamic_axes`.
  - `dynamo=False` keeps the legacy TorchScript exporter, which is nearing end of
    life.
  - The new path may surface capture errors the old one hid. Try it first, and
    record which path produced each model.
- **Explicit state.** Wrap each model so every recurrent tensor (conv left
  context, KV cache, positions) is a named graph input/output pair. The graph
  then has no side effects.
- **Pitfalls seen in the wild:**
  - A Python-int cache offset is frozen at its trace value, which produced
    garbled audio and NaNs (go-pocket-tts PR #43). Pass positions as tensors, or
    use concat + fixed-window slicing.
  - An exported encoder kept the trace's sequence length.
  - KV ring-buffer wrap drift (§2.2). Parity runs must last more than **2× the
    longest attention window**.
- **Shapes.** Use fixed shapes (batch 1, one hop) per frames-per-step variant
  (1, 2, 4). This enables memory-pattern planning and avoids allocations that
  depend on shape. The 1-frame/2-frame precedent comes from maai-kyoto.
- **Contract in the model.** Store it in `ModelProto.metadata_props` and read it
  in C++ with `Ort::Session::GetModelMetadata()` →
  `LookupCustomMetadataMapAllocated`, as sherpa-onnx does for model type and
  sample rate.
- **Checks.**
  - `onnx.checker.check_model(full_check=True)`.
  - ORT chunked vs. PyTorch offline on the same audio.
  - ORT 1.31 stays on ONNX 1.22: its ONNX 1.23 integration was reverted, so it
    has no new opsets. Use the exporter's default opset and verify it loads in
    the pinned ORT.

## 9. ONNX Runtime C++ deployment

- **Version.** v1.31.0 is the newest release (October 2026). It stabilised the
  model-package and EPContext APIs and parallelised weight prepacking. Building
  from source has required C++20 since 1.25, which matches the project.
- **Packaging.**
  - The Windows zip holds `include/` + `lib/` but no CMake config files
    (reported for 1.22, issue #28468). Define an IMPORTED SHARED target by hand.
  - Windows 11 ships its own `onnxruntime.dll` in System32, and apps have loaded
    it instead of their own copy (reports from .NET and vvvv users). So:
    - copy the DLL next to every executable;
    - log `Ort::GetVersionString()`;
    - refuse a version mismatch.
- **Threading.**
  - Defaults: intra-op threads = physical cores, spinning on. That pegs cores
    between runs (ORT threading docs; reports from other projects).
  - For small real-time models:
    - `intra_op_num_threads = 1`, so the run executes on the calling worker
      thread and inherits its priority;
    - `inter_op = 1` and sequential execution;
    - global thread pools shared by all sessions (`DisablePerSessionThreads`);
    - spinning off;
    - denormals as zero (`SetGlobalDenormalAsZero` with global pools).
- **Zero copy.** Create `Ort::Value::CreateTensor` views over preallocated
  buffers, and bind inputs/outputs with `Ort::IoBinding`. Use two prebuilt
  bindings (state A→B and B→A) and alternate between them, so no rebinding
  happens per hop.
- **Warm-up.** The first `Run` triggers allocations and kernel selection, so run
  warm-up passes before the audio stream starts.
- **Execution providers.** M2 uses CPU only.
  - DirectML is in sustained engineering (maintenance mode).
  - Microsoft now positions Windows ML (WinML) as the preferred Windows path,
    with automatic EP selection on Windows 11 24H2+.
  - Both are Milestone 4 concerns.
- **Errors.** The C++ API throws `Ort::Exception`. Catch it inside the model
  wrapper, because the worker thread must never throw.

## 10. Real-time scheduling and a fluent stream

- **Worker priority.**
  - Windows: MMCSS "Pro Audio" (`AvSetMmThreadCharacteristicsW`, avrt.lib).
  - Linux: best-effort SCHED_FIFO or nice.
  - macOS: `QOS_CLASS_USER_INTERACTIVE`.

  With `intra_op = 1` the whole inference runs on the worker thread. The README
  §4 table still lists the worker as "Normal" and must be updated.
- **Latency governor.** Keep output-queue occupancy near a target computed from
  the contract.
  - When it stays above target plus hysteresis, drop output samples while the
    output is quiet (silence is inaudible to drop).
  - A hard cap drops anything above it regardless.
  - Count the drops (`skipped`), so the `queued` identity stays exact.
- **Underrun fade.** On underrun, the callback ramps from the last sample to zero
  over about 2 ms instead of stepping, and ramps back in on resume. This is
  RT-safe: only a few scalars of state.
- **Noise gate.** Gating the output on input energy keeps the vocoder from
  producing babble from room noise.
- **Device periods.** miniaudio's `periodSizeInMilliseconds` is a hint.
  `ma_performance_profile_low_latency` with 10 ms periods is worth measuring
  first, because the I/O floor decides whether a 20 ms hop pays off.

## 11. Voice-control design

**Principle.** Change the *conditioning*, not the audio. Pitch, intonation,
timbre and voicing are vocoder inputs, so changing them costs nothing in latency
or DSP.

| Control | Acts on | MVP? |
|---|---|---|
| Pitch shift (±12 st) | `logf0_tgt += shift · ln2/12` | Yes |
| Intonation scale (0–2; 0 = robot monotone) | Multiplies whitened z | Yes |
| Timbre blend (voice A ↔ B) | `spk_emb = (1−α)·e_A + α·e_B` | Yes (experimental quality) |
| Output gain, noise gate, safety limiter | Output samples | Yes |
| Vibrato (rate, depth) | LFO on `logf0_tgt` | Stretch |
| Whisper | Voicing input forced towards 0 | Stretch |
| Brightness | Tilt EQ on output | Stretch |

**Excluded:**
- **Dry/wet mix.** The wet path is 60–100 ms late, so mixing produces comb
  filtering and lets the real voice through.
- **Formant DSP.** Timbre blending covers the need; revisit later.

**Engineering:**
- One descriptor table (name, unit, range, default, smoothing) drives the CLI
  flags, console commands, preset JSON keys and the future ImGui sliders.
- The control thread posts snapshots to the worker without locks.
- Smoothing is per-sample for gain and per-frame for conditioning, so changes
  don't produce zipper noise.

## 12. Licences

| Asset | Licence | Consequence |
|---|---|---|
| Mimi weights | CC-BY 4.0 | Attribution |
| HuBERT-Base (fairseq lineage) | MIT | — |
| mHuBERT-147 | CC-BY-NC-SA-4.0 | Non-commercial; models derived from it inherit the risk (M3 decision) |
| LJSpeech | Public domain | — |
| Hi-Fi TTS, LibriSpeech, VCTK | CC BY 4.0 | Attribution in the model card and README credits |
| MUSAN, RIR corpora | Mixed by subset | Check each subset at download |
| SwiftF0 | MIT | Not used (non-causal) |
| ONNX Runtime | MIT | Ship the licence with the DLL |
| nlohmann/json 3.12.0 | MIT | — |

## 13. Recommendations → stages

| Recommendation | Stage |
|---|---|
| Pinned Python 3.12 toolchain, datasets, frozen evaluation lists, `eval_vc.py` | 01 |
| Model contract in ONNX metadata; fixture models; nlohmann/json | 02 |
| Causal YIN + energy, whitening, Python/C++ parity | 03 |
| HuBERT km100 teacher; Mimi kill test; StreamVC-style student, L ∈ {0, 1, 2} | 04 |
| Causal multi-speaker vocoder; latency-first decision gate | 05 |
| Fixed-shape stateful export, parity > 2× window, `voices.json` | 06 |
| ORT 1.31.x, imported target + DLL copy + version check, global thread pool, IoBinding ping-pong | 07 |
| 10 ms periods, warm-up, governor, underrun fade, MMCSS, telemetry | 08 |
| Conditioning-domain controls, descriptor table, presets | 09 |
| Own-voice recording protocol, perturbation augmentation, fine-tune | 10 |
| Envelope latency marker, live-vs-offline quality, VC soak | 11 |

## 14. Open questions

1. **Does a unit vocoder trained on Mimi q0 recover intelligibility the codec
   tests didn't show?** Stage 04/05 kill test.
2. **Discrete km100 or soft units at 50 Hz?** Discrete is the default; soft units
   only if discrete misses the WER floor.
3. **How far does timbre interpolation between only 4–8 training speakers stay
   natural?** Stage 09 DoD.
4. **What is the real I/O floor at 10 ms periods on WASAPI shared mode?** This
   decides 20 vs. 40 ms hops. Stage 08.
5. **Milestone 3 language coverage.** An English HuBERT teacher limits source
   languages; mHuBERT-147 brings a non-commercial licence.

## Sources

- StreamVC — arXiv 2401.03078: https://arxiv.org/html/2401.03078v1
- Kanade tokenizer — arXiv 2602.00594: https://arxiv.org/html/2602.00594
- FocalCodec-Stream — arXiv 2509.16195: https://arxiv.org/html/2509.16195v1
- Bringing Interpretability to Neural Audio Codecs — arXiv 2506.04492: https://arxiv.org/html/2506.04492
- Speech Resynthesis from Discrete Disentangled SSL Representations — arXiv 2104.00355: https://arxiv.org/pdf/2104.00355
- LLVC, Low-latency Real-time Voice Conversion on CPU — arXiv 2311.00873: https://arxiv.org/pdf/2311.00873 · https://github.com/KoeAI/LLVC
- SwiftF0 — arXiv 2508.18440: https://arxiv.org/abs/2508.18440v1 · https://github.com/lars76/swift-f0
- FCPE — arXiv 2509.15140: https://arxiv.org/html/2509.15140v1
- mHuBERT-147 — arXiv 2406.06371: https://huggingface.co/utter-project/mHuBERT-147
- Causal AudioDec-based VC — arXiv 2510.09245: https://arxiv.org/html/2510.09245v1
- Narrowband streaming neural compression — arXiv 2609.25379: https://arxiv.org/pdf/2609.25379
- Comparative analysis of streaming vocoders — arXiv 2506.03554: https://arxiv.org/html/2506.03554v1
- DLL-APNet — arXiv 2509.13667: https://web3.arxiv.org/abs/2509.13667
- VERSA — arXiv 2412.17667: https://arxiv.org/html/2412.17667v2
- Fast-VGAN (evaluation practice) — arXiv 2507.04817: https://arxiv.org/pdf/2507.04817
- Kyutai Mimi model card: https://huggingface.co/kyutai/mimi
- Transformers Mimi docs: https://huggingface.co/docs/transformers/main/en/model_doc/mimi
- Continuous Mimi ONNX export: https://huggingface.co/maai-kyoto/continuous-mimi-onnx
- go-pocket-tts PR #43 (frozen cache offset): https://github.com/CWBudde/go-pocket-tts/pull/43
- PyTorch 2.9 release notes (dynamo exporter default): https://newreleases.io/project/pypi/torch/release/2.9.0 · https://docs.pytorch.org/docs/main/onnx.html
- ONNX Runtime threading: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- ONNX Runtime v1.31.0 release notes: https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.31.0
- ONNX Runtime v1.30.0 release notes: https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.30.0
- ONNX Runtime v1.25.0 (C++20 for source builds): https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0
- Windows zip lacks CMake files (VTK forum, issue #28468): https://discourse.vtk.org/t/vtk-9-6-onnx-runtime-installation/16410
- System32 `onnxruntime.dll` shadowing reports: https://discourse.vvvv.org/t/bug-with-native-net-lib-dependencies-in-editor/21689 · https://lightrun.com/answers/dotnet-interactive-external-resolving-onnx-runtime-fails
- DirectML sustained engineering / WinML: https://app.semanticdiff.com/gh/microsoft/onnxruntime/pull/25901/overview
- Hi-Fi TTS (OpenSLR 109): https://openslr.org/109 · https://arxiv.org/pdf/2104.01497
- nlohmann/json 3.12.0: https://github.com/nlohmann/json/releases/tag/v3.12.0
