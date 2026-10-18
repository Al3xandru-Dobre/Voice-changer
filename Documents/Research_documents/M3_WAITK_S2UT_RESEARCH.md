# Simultaneous speech-to-unit translation (Mode B): policy, multilingual units, prosody and runtime

- **Project:** FluxVoice-S2UT
- **Research date:** 2026-10-18
- **Status:** Research and proposed plan; no implementation changes

> **Implementation tracking.** This report is preserved as a dated research
> snapshot. Its recommendations are scheduled in
> [Milestone 3 — Wait-k S2UT mode](../Milestone%203/00_master.md), stages 01–11
> (mapping in §11). It also updates cross-milestone notes in
> [Milestone 2 / Stage 02 & 04](../Milestone%202/00_master.md) and
> [Milestone 4](../Milestone%204/00_master.md). Where this snapshot and a stage
> file disagree, the stage file is authoritative.

## 1. Summary, scope and method

**Recommendation.**
- **Target bidirectional `En↔{Es,Fr,De}` (6 directions) in a single multi-pair
  model** conditioned on `<src_lang>` and `<tgt_lang>` tokens.
- **Feed Stage 2 (the S2UT translator) with Stage 1's continuous pre-quantization
  latents (`content` `[1, frames_per_step, D]` f32), not hard source unit IDs.**
  Sharing M2's 50 Hz causal conv encoder trunk avoids running a second acoustic
  frontend in Mode B, enables instant Mode A↔B switching, and preserves fine
  cross-lingual phonetic detail that hard k-means quantization drops on the
  source side.
- **Train with StreamSpeech-style multi-task auxiliary CTC supervision (source
  ASR CTC + target text S2TT CTC), then export a 100% textless graph.** Pure
  single-pass S2UT without intermediate text supervision collapses on high
  reordering pairs (`De→En` drops to 2.46 ASR-BLEU in published CVSS-C tests),
  whereas multi-task CTC alignment reaches 19.72–27.58 ASR-BLEU with a ~70M
  parameter model (§2.3). Dropping the CTC heads at ONNX export keeps run-time
  inference strictly latent-to-unit.
- **Operate `WaitKPolicy` on 80 ms translation chunks ($C = 4$ frames at 50 Hz),
  and split the exported translator into `translator_enc.onnx` and
  `translator_dec.onnx` with static ring-buffer KV caches.** Counting $k$ in
  80 ms chunks preserves the README's `k ∈ [1..10]` (80–800 ms initial wait),
  keeps transformer sequence lengths $4\times$ shorter than 20 ms frame-by-frame
  attention, and allows `kRead` steps to run only the encoder subgraph while
  `kFlush` steps run only the decoder subgraph (§4, §8).
- **Predict target-frame prosody (`f0_z`, `voicing`, `aperiodicity`, `energy_z`)
  from the translator decoder, and decouple the output noise gate during
  `kFlush`.** In Mode B, target syllables do not align 1:1 in time with live
  microphone frames, and during tail flush (> 400 ms after speech ends) the live
  microphone is silent while the translator is still speaking (§5).

**Scope.**
- In:
  - Mode B (`mic → Stage 1 latents → Stage 2 S2UT translator → Stage 3 unit
    vocoder → target voice`).
  - Bidirectional `En↔{Es,Fr,De}` datasets, SimulEval latency/quality harness,
    multilingual unit teacher + vocoder upgrade, `WaitKPolicy` at 50 Hz,
    causal S2UT architecture and policy gate, translated prosody prediction,
    two-graph ring-KV ONNX export, C++ runtime, live Mode B pipeline, and
    cross-lingual voice preservation ("Me" voice speaking Es/Fr/De and En).
- Out:
  - GPU execution providers, FP16/INT8 quantization and Dear ImGui widgets
    (Milestone 4).
  - Unwritten languages without parallel text for training-time CTC supervision.

**Method.**
- Primary-source review on 2026-10-18: ACL / ICML / ICASSP / IEEE TASLP papers,
  official repositories (`ictnlp/StreamSpeech`, `kyutai-labs/hibiki`,
  `facebookresearch/fairseq`, `choijeongsoo/utut`, `facebookresearch/SimulEval`),
  and Hugging Face dataset/model cards (`google/cvss`, `facebook/covost2`,
  `facebook/voxpopuli`, `ylacombe/cml-tts`).

## 2. Simultaneous S2UT landscape & policy bake-off

### 2.1 Requirements for the FluxVoice Stage 2 translator

| Requirement | Why |
|---|---|
| Causal encoder and decoder with bounded state | Continuous streaming without sentence-boundary VAD waiting |
| Fits one CPU core alongside Stage 1 + Stage 3 | Milestone 3 runs on CPU EP (`intra_op = 1`); GPU is Milestone 4 |
| Trainable on an 8 GB laptop GPU (or short cloud run) | Hardware constraint inherited from Milestone 2 (`RTX 4060 Laptop 8 GB`) |
| Bidirectional `En↔{Es,Fr,De}` in one checkpoint | Avoids shipping 6 separate translator binaries |
| Emits discrete target units + prosody at the vocoder's rate | Plugs directly into the Milestone 2 causal Unit-HiFi-GAN |

### 2.2 Kyutai Hibiki & SeamlessStreaming (kill analysis)

`Documents/Milestone 3/00_master.md` (draft) asked for an opening spike
comparing **Kyutai Hibiki** (`kyutai-labs/hibiki`, Labiausse et al. 2025,
arXiv:2502.03382) and **Meta SeamlessStreaming** (`seamless_communication`)
against a Wait-*k* / CTC Conformer-Transformer. Primary-source inspection settles
both without needing a training run:

| System | Architecture & size | Codec / rate | Languages & licence | Verdict for FluxVoice M3 |
|---|---|---|---|---|
| **Hibiki** (arXiv:2502.03382) | Decoder-only multistream RQ-Transformer: 24-layer Temporal Transformer ($d=2560$, 20 heads) + Depth Transformer; **1B** (`hibiki-1b`, 8 RVQ streams) and **2B** (`hibiki-2b`, 16 RVQ streams) | Hard-wired to **Mimi** @ 24 kHz, 12.5 Hz (80 ms), 8–16 acoustic codebooks | **Fr→En only** released; trained on synthetic contextual-aligned data | **Drop as backbone.** 1B–2B params in FP32 (4–8 GB weights alone) cannot run on 1 CPU core or train on 8 GB VRAM, is Fr→En only, and requires 8–16 Mimi RVQ streams rather than speaker-disentangled semantic units |
| **SeamlessStreaming** (EMMA) | XLS-R frontend + 2-pass UnitY + EMMA monotonic attention; **~2.3B** params | 16 kHz in, mHuBERT km1000 units | 100 in / 36 out; **CC-BY-NC 4.0** | **Drop as backbone.** 2.3B params exceeds CPU and 8 GB VRAM budgets; non-commercial licence |
| **Fairseq Direct S2UT** (Lee et al., ACL 2022) | Speech encoder + discrete-unit Transformer decoder; **~73M** params | HuBERT / mHuBERT 50 Hz units (stacked or duration-reduced) | Pair-specific; MIT code | **Baseline architecture**, but fails on `De→En` (2.46 BLEU) without multi-task CTC supervision (§2.3) |
| **StreamSpeech** (Zhang et al., ACL 2024, arXiv:2406.03049) | Causal chunked Conformer encoder + auxiliary ASR/S2TT CTC alignment + decoder; **~70M** params | 16 kHz speech → mHuBERT km1000 units | Tested on `Fr/Es/De→En`; MIT code | **Adopt training recipe & CTC policy option**, scaled to a 25–45M param causal Conformer-Transformer on Stage 1 latents |
| **UTUT** (Kim et al., IEEE TASLP 2024, arXiv:2308.01831) | Many-to-many encoder-decoder with `<src_lang>` / `<tgt_lang>` tokens on mHuBERT units | mHuBERT L11 km1000 (50 Hz) | Bidirectional `En,Es,Fr,De,It,Nl` | **Adopt language-tag conditioning** for bidirectional `En↔{Es,Fr,De}` |

### 2.3 Why multi-task CTC supervision is essential during training

StreamSpeech (ACL 2024, Table 1) benchmarked direct S2UT models of comparable
size (~67–93M parameters) on CVSS-C test sets:

| Model | Params | `Fr→En` ASR-BLEU | `Es→En` ASR-BLEU | `De→En` ASR-BLEU | Average |
|---|---:|---:|---:|---:|---:|
| Ground truth target audio | — | 84.52 | 88.54 | 75.53 | 82.86 |
| Direct S2UT (Lee et al. 2022, unit loss only) | 73M | 20.91 | 16.94 | **2.46** | 13.44 |
| Translatotron 2 (two-pass auxiliary text) | 87M | 25.49 | 22.35 | 16.24 | 21.36 |
| UnitY (two-pass text + unit) | 67M | 26.90 | 23.93 | 18.19 | 23.01 |
| **StreamSpeech** (chunked Conformer + ASR/S2TT CTC) | 70M | **27.58** | **26.16** | **19.72** | **24.49** |

**Key takeaway:** Direct S2UT trained only with cross-entropy on 50 Hz target
units struggles to learn long-range syntactic reordering (especially German
verb-second / verb-final clauses, where `De→En` collapses to 2.46 BLEU). Adding
two lightweight linear CTC heads during training —
1. **Source ASR CTC** on top of the causal translator encoder ($L_{\text{asr}}$), and
2. **Target S2TT CTC** on top of the encoder/decoder bridge ($L_{\text{s2tt}}$) —
gives a $+11$ BLEU average jump (and $2.46 \to 19.72$ on `De→En`). Because CTC
heads are simple linear projections over SentencePiece subwords, they can either
be **stripped at ONNX export** (for pure Wait-*k* execution) or **retained as a
single blank/non-blank scalar** to drive StreamSpeech's adaptive READ/WRITE
policy.

### 2.4 Simultaneous policy: Strict Wait-*k* vs. CTC-Adaptive

1. **Strict Wait-*k* (`WaitKPolicy`):**
   - Waits until $t_{\text{src}} - t_{\text{tgt}} \ge k$ chunks (each chunk =
     80 ms = 4 frames at 50 Hz), then emits one target chunk per source chunk.
   - Deterministic latency: initial wait is exactly $k \times 80\text{ ms}$
     (e.g. $k=3 \implies 240\text{ ms}$, $k=5 \implies 400\text{ ms}$).
   - Weakness: fixed 1:1 chunk pacing during `kWrite` assumes target speech has
     the same speaking rate as source speech; Spanish and French often have
     15–25% more syllables per clause than English, pushing remaining content
     into the tail flush or requiring duration compression.
2. **CTC-Adaptive Wait-*k* (StreamSpeech hybrid):**
   - Enforces a minimum floor of $k_{\min}$ chunks, then uses the expected
     target token count from the auxiliary CTC head (or a learned write-Read
     gate probability $p_{\text{write}} \in [0, 1]$) to emit 0, 1, or 2 target
     unit chunks per source chunk when target phrasing is longer or shorter than
     the source.
   - Stage 05 runs a controlled gate comparing **Strict Wait-*k*** ($k \in \{2,
     3, 5\}$) against **Adaptive Wait-*k*** on SimulEval Average Lagging (AL) vs.
     ASR-BLEU.

## 3. Multilingual unit & encoder representation (`En↔{Es,Fr,De}`)

### 3.1 Multilingual teacher options

Milestone 2's default teacher (`facebook/hubert-base-ls960`) was trained on 960 h
of English LibriSpeech only. For bidirectional `En↔{Es,Fr,De}`, both source
latents and target discrete units must cover English, Spanish, French and German
phonetics:

| Teacher | Languages & data | Rate & vocab | Licence | Assessment |
|---|---|---|---|---|
| **mHuBERT-vp** (`mhubert_base_vp_en_es_fr_it3`, Lee et al. 2022 / Popuri et al. 2022) | Trained on VoxPopuli `En, Es, Fr` (100k subset); layer 11 + k-means $K=1000$; used by Fairseq S2UT, StreamSpeech, and UTUT (which also validated it on `De` and `Nl`) | 50 Hz (20 ms), $K \in \{500, 1000\}$ | **MIT** (fairseq release) | **Recommended default.** Permissive licence, proven in StreamSpeech (`Fr/Es/De→En`) and UTUT (`En↔Es/Fr/De/It/Nl`) |
| **mHuBERT-147** (`utter-project/mHuBERT-147`, Boito et al. 2024) | 147 languages (~90k h), 95M params, Faiss $K=1000$ | 50 Hz (20 ms), $K=1000$ | **CC-BY-NC-SA-4.0** | Strong German coverage, but non-commercial share-alike licence; evaluate in Stage 03 probe as an opt-in research teacher |
| **WavLM-Base+ / MMS-300M** | Multilingual SSL | 50 Hz | MIT / CC-BY-NC | 300M params (MMS) is heavier; fallback only if `mhubert-vp` German phone purity lags by > 5% |

Refitting (or augmenting) the $K=1000$ k-means codebook on balanced `{en, es, fr,
de}` layer-11 features from `mhubert_base_vp_en_es_fr_it3` ensures German
front-rounded vowels (`/y:/`, `/ø:/`) and uvular/velar fricatives (`/x/`, `/ç/`)
get dedicated centroids even though the base SSL pretraining was En/Es/Fr. Stage
03 measures phone purity and PNMI across all 4 languages before freezing the
multilingual codebook.

### 3.2 Shared M2 continuous latents (`content` f32) + discrete `units` i64

In Milestone 2, the 50 Hz causal conv student (`fvml/content_encoder.py`) has a
causal SoundStream-style trunk (output dimension $D = 64$ or $128$) followed by
linear classification heads predicting teacher cluster IDs.

For Milestone 3:
- The student trunk is trained on `{en, es, fr, de}` speech (with MUSAN/RIR
  augmentation) to predict the multilingual teacher's unit IDs.
- At export time, the encoder emits **two outputs from the same forward pass**:
  1. `units`: `[1, frames_per_step]` `int64` (argmax cluster IDs for Mode A direct
     VC);
  2. `latents`: `[1, frames_per_step, D]` `float32` (layer-normalised pre-head
     continuous bottleneck features for Mode B's Stage 2 translator).
- Because the student trunk is trained solely against speaker-independent teacher
  units (with no vocoder gradient flowing back), `latents` is already stripped of
  source speaker identity while retaining soft phonetic probabilities across the
  codebook.

### 3.3 Multilingual causal Unit-HiFi-GAN vocoder

To synthesize both `X→En` and `En→{Es,Fr,De}` in any target voice (including the
user's "Me" voice from M2 Stage 10):
- The Stage 3 causal Unit-HiFi-GAN vocoder is trained on the unified
  multilingual unit vocabulary ($K=1000$) using clean target audio across all 4
  languages:
  - **English:** LJSpeech + chosen Hi-Fi TTS speakers (from M2);
  - **Spanish, French, German:** CSS10 single-speaker studio sets (~24 h `es`,
    ~19 h `fr`, ~16 h `de`) + selected high-SNR speakers from **CML-TTS** (24 kHz
    LibriVox TTS subset, CC-BY 4.0, resampled to 16 kHz).
- Every speaker gets a row in the shared `nn.Embedding(n_speakers, 64)` FiLM
  table. Because units are language-shared and speaker identity enters only
  through FiLM + target-domain F0/energy stats (`voices.json`), **any English
  speaker embedding (including `--voice me`) can synthesize Spanish, French, or
  German units in that person's voice.**

## 4. Latency analysis & Wait-*k* at 50 Hz

### 4.1 Understanding the two latency definitions in Mode B

1. **First-audio / chunk-pipeline latency ($T - B$, README target ~300–500 ms):**
   The time from speech onset at the microphone until the first translated audio
   emerges from the output device. It equals:
   $$\text{Latency}_{\text{onset}} = \text{I/O floor} + k \times \text{chunk\_ms} + \text{Encoder lookahead} + \text{Compute}_{p99} + \text{Governor margin}$$
2. **Simultaneous translation Average Lagging ($\text{AL}$, SimulEval):**
   The average delay across an entire utterance between when a source word is
   spoken and when its corresponding target word is heard, accounting for
   cross-lingual word-order reordering and speaking-rate ratio $\gamma =
   |Y| / |X|$.

### 4.2 Why 80 ms translation chunks ($C = 4$ frames at 50 Hz) are optimal

If `WaitKPolicy` stepped every single 20 ms frame ($C = 1$):
- $k \in [1..10]$ would only cover 20–200 ms (less than one syllable!), whereas
  resolving cross-lingual word order needs at least 240–480 ms (1–3 words);
- the S2UT transformer would run 50 encoder-decoder steps per second, $4\times$
  more attention compute and $4\times$ longer KV caches for the same temporal
  window.

Instead, we group $C = 4$ frames (80 ms at 50 Hz) into one **translation step**:
- Stage 1 runs at the hop rate (e.g. 20 ms or 40 ms for low-latency Mode A, or
  80 ms `s4` in Mode B) and accumulates $C = 4$ latent frames `[1, 4, D]`.
- `WaitKPolicy` counts $t_{\text{src}}$ and $t_{\text{tgt}}$ in **80 ms chunks**,
  matching `src/pipeline/WaitKPolicy.hpp` and the README (`k ∈ [1..10]` $\implies$
  80–800 ms initial wait).
- On each `kWrite` or `kFlush` step, `translator_dec` emits **4 target unit
  frames** `[1, 4]` (Lee et al. 2022's **stacked unit decoding**, predicting 4
  consecutive 20 ms unit distributions per decoder step) plus **4 target prosody
  frames** `[1, 4, 4]`, which feed the 50 Hz vocoder directly!

### 4.3 Mode B onset latency budget (16 kHz, 80 ms translation chunks)

| Component | $k=2$ (aggressive) | $k=3$ (default) | $k=4$ (balanced) | $k=5$ (high quality, De↔En) |
|---|---:|---:|---:|---:|
| I/O floor (passthrough $T - B$ @ 10 ms period) | 25 ms | 25 ms | 25 ms | 25 ms |
| Encoder lookahead ($L = 1 \times 20\text{ ms}$) | 20 ms | 20 ms | 20 ms | 20 ms |
| Wait-*k* buffering ($k \times 80\text{ ms}$ chunks, incl. framing) | 160 ms | 240 ms | 320 ms | 400 ms |
| Compute $p_{99}$ (Enc + Trans Enc/Dec + Vocoder, $\le 50\%$ of 80 ms) | 35 ms | 35 ms | 35 ms | 35 ms |
| Governor jitter margin | 10 ms | 10 ms | 10 ms | 10 ms |
| **FluxVoice Mode B onset $T - B$** | **250 ms** | **330 ms** | **410 ms** | **490 ms** |
| Algorithmic latency only ($k \times 80 + L \times 20$) | 180 ms | 260 ms | 340 ms | 420 ms |

- All four primary configurations ($k \in \{2, 3, 4, 5\}$) land squarely inside
  or below the **README ~300–500 ms target**!
- The slider `k ∈ [1..10]` exposes the full range from ~170 ms ($k=1$, monotone
  phrase glossing) to ~890 ms ($k=10$, clause-level reordering).

## 5. Prosody & voice transfer across translation

### 5.1 The Mode B prosody mismatch problem

In Mode A (voice conversion), source frame $t$ and output frame $t$ are the same
syllable in the same language, so `InferencePipeline` feeds the live microphone's
whitened `logf0`, `voicing`, `aperiodicity`, and `energy` (delayed by $L$ frames)
directly into the vocoder.

In Mode B, feeding live microphone prosody directly into the vocoder fails in
three ways:
1. **READ steps (`t_src - t_tgt < k`):** The user is actively speaking into the
   mic (`voicing = 1`, high energy), but the translator is buffering and has no
   target words to emit yet.
2. **Word-order & syllable mismatch (`kWrite`):** When translating *"the red
   car"* $\to$ *"el coche rojo"*, target vowels and unvoiced fricatives do not
   occur on the same 20 ms frames as the source audio $k$ chunks later. Forcing
   source `voicing = 0` onto a translated vowel (or `voicing = 1` onto a
   translated `/s/`) creates buzzes and whispered dropouts.
3. **Tail flush (`kFlush`, after > 400 ms of mic silence):** The user has
   **stopped speaking** (`voicing = 0`, mic energy below the noise gate!). If the
   vocoder took `voicing` and `energy` from the live mic, or if Stage 09's noise
   gate gated on live mic energy, **the last $k$ chunks of every translated
   sentence would be muted or whispered!**

### 5.2 Solution: Translator prosody head + source-style summary + gate override

1. **Utterance-level source prosody transfer (`prosody_ctx`):**
   - The C++ `PitchTracker` + `RunningNorm` (from M2 Stage 03) still tracks the
     speaker's live pitch deviation and energy dynamics on voiced source frames,
     computing a smoothed 2-scalar summary per chunk: `[mean_z_f0, mean_z_energy]`
     (how high/excited the speaker's current phrase is relative to their baseline).
2. **Translator prosody predictor head:**
   - Alongside the stacked target unit logits `[1, C, V]`, `translator_dec` has a
     lightweight 2-layer causal conv/linear prosody head conditioned on the
     decoder hidden state, the chosen target unit embeddings, and `prosody_ctx`:
     $$\hat{p}_t = (\hat{z}_{\text{f0}, t},\; \hat{v}_t,\; \hat{a}_t,\; \hat{z}_{e, t}) \in \mathbb{R}^{C \times 4}$$
   - During training, this head is supervised with L1 + BCE loss against the
     ground-truth target audio's Stage 03 YIN features (`z_f0`, `voicing`,
     `aperiodicity`, `z_energy`).
3. **Target-domain mapping & Stage 09 controls:**
   - In C++, the predicted $\hat{z}_{\text{f0}, t}$ and $\hat{z}_{e, t}$ pass
     through the **exact same target-domain mapping** as Mode A:
     $$\text{logf0}_{\text{tgt}} = \mu_{\text{tgt}} + \text{scale} \cdot \sigma_{\text{tgt}} \cdot \hat{z}_{\text{f0}, t} + \text{shift\_st} \cdot \frac{\ln 2}{12}$$
   - Thus all Stage 09 voice controls (`pitch_shift`, `intonation`,
     `timbre_blend`, `voice_a`/`voice_b`, `gain`, `limiter`) work identically in
     Mode B at zero extra cost!
4. **READ bypass and FLUSH gate override:**
   - On `Decision::kRead`: skip `translator_dec` and skip `vocoder.step()`; emit
     80 ms of zero samples (ramped cleanly by the output fade if transitioning
     from speech).
   - On `Decision::kWrite` and `Decision::kFlush`: drive the output noise gate
     from the **predicted target energy/voicing** (or hold the gate open whenever
     non-`<SILENCE>` target units are emitted), never from instantaneous mic
     energy.

## 6. Bidirectional datasets (`En↔{Es,Fr,De}`)

| Dataset | Directions used | Hours (`Es / Fr / De`) | Target audio type | Licence | Role in Milestone 3 |
|---|---|---|---|---|---|
| **CVSS-C** (`google/cvss`) | `Es→En`, `Fr→En`, `De→En` | Src: `113 / 264 / 184` h; Tgt: `70 / 174 / 112` h | Single high-quality canonical English TTS voice (22.05 kHz → 16 kHz) | **CC BY 4.0** | Primary training + SimulEval benchmark for `{Es,Fr,De}→En` |
| **CoVoST 2** (`facebook/covost2`) + multi-speaker TTS synthesis | `En→De` (official 289k utts, ~364 h) + `En→Es`, `En→Fr` (reverse pairs from CoVoST 2 / Europarl-ST synthesized with Piper/VITS CSS10/CML voices) | ~150–360 h per `En→X` pair | Clean synthesized `De`, `Es`, `Fr` target waveforms → mHuBERT units | **CC0** (CoVoST 2) + **CC BY 4.0** (Europarl-ST) | Primary training + benchmark for `En→{Es,Fr,De}` |
| **CSS10** (`kyubyong/css10`) | Monolingual `Es`, `Fr`, `De` | `24 h Es`, `19 h Fr`, `16 h De` | Single-speaker studio LibriVox reads | **Apache 2.0 / CC0** | Multilingual Unit-HiFi-GAN vocoder target voices (`Es`, `Fr`, `De`) |
| **CML-TTS** (`ylacombe/cml-tts`, OpenSLR 146) | Monolingual `Es`, `Fr`, `De` | Curated subset: 2–3 studio-grade speakers per language (~15 h each) | 24 kHz clean audiobooks (resampled to 16 kHz) | **CC BY 4.0** | Multi-speaker diversity for the multilingual vocoder |
| **VoxPopuli** (`facebook/voxpopuli`) | `En↔{Es,Fr,De}` transcribed + S2S | `543 h En`, `282 h De`, `211 h Fr`, `166 h Es` | Real EU Parliament speech | **CC BY-NC 4.0** | Optional evaluation / research set only (flagged in `DATASETS.md` due to NC clause) |

**Why synthesize target speech for `En→{Es,Fr,De}` (the CVSS-C recipe in reverse)?**
CVSS-C proved that training S2UT on human source speech paired with clean,
canonical TTS target units converges far better than noisy human-interpreted
target audio (where overlapping interpreters, hesitations, and room acoustics
corrupt target unit labels). Because Stage 3 (the Unit-HiFi-GAN vocoder) renders
the final voice from discrete units + speaker embedding, the translator only
needs clean target *unit sequences* and normalized prosody contours during
training.

## 7. Evaluation methodology

Every Mode B candidate and stage gate is scored by a unified `eval_s2ut.py`
script on frozen 200-utterance test lists across all 6 directions
(`es_en`, `fr_en`, `de_en`, `en_es`, `en_fr`, `en_de`):

| Metric | Tool & configuration | Purpose |
|---|---|---|
| **ASR-BLEU** | Whisper `large-v3` (forced target `language={en,es,fr,de}`, `task="transcribe"`) + SacreBLEU (`13a` tokenizer, lowercase/normalized) | Primary translation quality metric (standard in CVSS, StreamSpeech, Fairseq S2UT) |
| **ASR-chrF++** | SacreBLEU `chrf` (`word_order=2`) on Whisper transcripts | Morphologically richer metric for German (`de`) and Romance inflections (`es`, `fr`) |
| **Average Lagging (`AL`, `AL_CA`)** | SimulEval word/unit timestamp alignment (Ma et al. 2020): algorithmic `AL` (ms) and computation-aware `AL_CA` (ms) | Measures real simultaneous translation delay across the whole utterance |
| **First-audio onset ($T - B$)** | `fv_probe latency --marker speech` (or offline sample index of first voiced translated frame minus source onset) | Verifies the README's ~300–500 ms onset target |
| **Speaker similarity** | SpeechBrain `spkrec-ecapa-voxceleb` cosine to target voice reference clips | Verifies cross-lingual voice preservation (including `--voice me`) |
| **Naturalness** | UTMOS | Reference-free acoustic quality check across all 4 output languages |
| **Hallucination / flush check** | Silence-only and noise-only input clips + post-utterance tail duration | Ensures `kFlush` terminates cleanly on `<EOS>` without babble loops |

## 8. Stateful ONNX export for encoder-decoder cross-attention

### 8.1 Why split Stage 2 into `translator_enc.onnx` and `translator_dec.onnx`

A single monolithic encoder-decoder ONNX graph is awkward for simultaneous
translation because:
- During **`kRead`** ($t_{\text{src}} - t_{\text{tgt}} < k$), only the translator
  encoder needs to advance; running the decoder and throwing away its output
  wastes ~50% of Stage 2 CPU time and corrupts the autoregressive decoder
  self-attention KV cache unless masked out.
- During **`kFlush`** (after mic silence > 400 ms), no new source speech arrives
  (or only silence arrives), while the decoder steps multiple times until the
  cached source tokens are drained or `<EOS>` is emitted.
- Under an **adaptive policy**, one encoder step may trigger 0, 1, or 2 decoder
  steps.

Splitting Stage 2 into two fixed-shape stateful graphs makes `kRead`, `kWrite`,
and `kFlush` trivial in C++:
1. **`translator_enc.onnx`**:
   - Inputs: `latents` `[1, C, D]` f32, `src_lang` `[1]` i64, `src_pos` `[1]` i64,
     self-attention KV ring states `state_in.enc_kv_*`, conv left-context states,
     and cross-attention memory ring `state_in.mem_k_*`, `state_in.mem_v_*`.
   - Outputs: updated `state_out.*` (including the new slot written into
     `mem_k_*` / `mem_v_*` at index `src_pos mod W_src`) + optional
     `readiness_logit` `[1, 1]` f32 (for CTC-adaptive policy).
2. **`translator_dec.onnx`**:
   - Inputs: `prev_units` `[1, C]` i64, `tgt_lang` `[1]` i64, `src_pos` `[1]` i64,
     `tgt_pos` `[1]` i64, `prosody_ctx` `[1, 2]` f32, cross-attention memory
     `mem_k_*` / `mem_v_*` (read-only view from `translator_enc`!), and decoder
     self-attention KV ring `state_in.dec_kv_*`.
   - Outputs: `units` `[1, C]` i64, `prosody` `[1, C, 4]` f32, `eos_prob` `[1, 1]`
     f32, and updated `state_out.dec_kv_*`, `state_out.tgt_pos`.

### 8.2 Ring-buffer KV caches with static shapes

Dynamic sequence dimensions (`[1, H, T_dynamic, D_h]`) cause ONNX Runtime's arena
to reallocate buffers as $T$ grows and slow down attention linearly over long
utterances. Instead:
- Fix the encoder sliding window to $W_{\text{src}} = 128$ chunks ($128 \times
  80\text{ ms} = 10.24\text{ s}$) and the decoder sliding window to
  $W_{\text{tgt}} = 128$ chunks ($10.24\text{ s}$).
- Update the current slot in-place via `torch.scatter` / one-hot index update
  driven by tensor state `src_pos` / `tgt_pos`, and compute the causal +
  valid-range mask `(pos_idx <= current_pos) & (current_pos - pos_idx < W)`
  purely from tensor operations (avoiding the frozen Python-int offset bug
  documented in M2 research §8).
- When `WaitKPolicy::reset()` triggers at the end of a tail flush, C++ zeroes
  `src_pos`, `tgt_pos`, and the KV state banks in microseconds.

## 9. C++ runtime & `WaitKPolicy` integration

### 9.1 `WaitKPolicy` enhancements (`src/pipeline/WaitKPolicy.hpp`)

The existing `WaitKPolicy` header is a clean 67-line state machine (`kRead`,
`kWrite`, `kFlush`), but needs four targeted upgrades for real-time 50 Hz S2UT:
1. **Frame-to-chunk accumulator:** Track 20 ms encoder frames and trigger a
   policy decision once per `chunk_frames = 4` (80 ms), or support `s4` (4 frames
   per step) directly.
2. **Bounded tail flush & `<EOS>` termination:** Currently `flush_pending()`
   stays `true` forever until external code calls `reset()`! Add:
   - automatic completion when `t_tgt >= max_tgt_for_flush` (bounded by
     $\lceil \gamma_{\max} \cdot t_{\text{src}} \rceil$, e.g. $1.5 \times
     t_{\text{src}}$) **or** when `translator_dec` emits `eos_prob > threshold`
     (or `<EOS>` unit ID);
   - automatic `reset()` transition once the flush finishes draining.
3. **Mid-sentence pause immunity:** A brief 150–300 ms pause between words must
   not prematurely reset the encoder context; `kTailFlushMs = 400` handles short
   pauses, and during `kFlush` if new speech arrives before flush completes, the
   policy transitions smoothly back to `kWrite` without dropping cached context.
4. **Adaptive policy hook:** Optional `notify_readiness(float score)` so the same
   state machine can run either strict Wait-*k* (`t_src - t_tgt >= k`) or
   CTC-adaptive Wait-*k* (`t_src - t_tgt >= k_min && score >= threshold`).

### 9.2 `LatencyGovernor` in Mode B

In Mode A, `LatencyGovernor` (M2 Stage 08) drops output silence whenever `queued`
exceeds `target_ms`. In Mode B:
- During `kRead`, the pipeline intentionally outputs 80 ms of silence per chunk
  so the SPSC output queue stays fed at the device sample rate. That does **not**
  increase `queued` sample count (1 chunk in = 1 chunk of silence out), so the
  conservation identity `queued = underruns - overruns - drops - skipped`
  still holds!
- However, if an adaptive policy or a multi-step `kFlush` emits two target chunks
  during a single hop, `queued` intentionally rises by one chunk (80 ms) while
  the translated phrase plays out. The governor's target ceiling in Mode B must
  allow a **burst cushion** of `max_burst_chunks × 80 ms` (e.g. +240 ms during
  active speech/flush) and only trim excess silence **after** `WaitKPolicy` has
  completed `kFlush` and returned to idle!

## 10. Licences

| Asset | Licence | Consequence for FluxVoice |
|---|---|---|
| **CVSS-C / CVSS-T** (`google/cvss`) | CC BY 4.0 | Commercial-friendly; requires attribution in `MODEL_CARD.md` and `README.md` |
| **CoVoST 2** (`facebook/covost2`) | CC0 1.0 | Public domain |
| **Europarl-ST** | CC BY 4.0 | Commercial-friendly with attribution |
| **CSS10** (`kyubyong/css10`) | Apache 2.0 / CC0 (LibriVox) | Commercial-friendly |
| **CML-TTS** (`ylacombe/cml-tts`, OpenSLR 146) | CC BY 4.0 | Commercial-friendly with attribution |
| **mHuBERT-vp** (`mhubert_base_vp_en_es_fr_it3`, fairseq) | MIT | Commercial-friendly; recommended default teacher |
| **mHuBERT-147** (`utter-project/mHuBERT-147`) | CC BY-NC-SA 4.0 | **Non-commercial**; models distilled from it inherit NC risk — keep optional behind `--teacher mhubert147` |
| **VoxPopuli** (`facebook/voxpopuli`) | CC BY-NC 4.0 | **Non-commercial**; do not use in default commercial bundle weights (use CoVoST 2 + CVSS-C + Europarl-ST instead) |

## 11. Recommendations → stages

| Recommendation | Stage |
|---|---|
| Bidirectional `En↔{Es,Fr,De}` data prep (CVSS-C, CoVoST 2 + TTS, Europarl-ST, CSS10, CML-TTS) & `eval_s2ut.py` (ASR-BLEU, chrF++, AL, UTMOS, ECAPA) | 01 |
| Contract v2 (`translator_enc`, `translator_dec`, `langs`, `chunk_frames`, ring-KV states, prosody head) + deterministic ONNX translator fixtures | 02 |
| Multilingual unit teacher probe (`mhubert-vp` km1000 vs. `mHuBERT-147`), dual-output 50 Hz student (`latents` f32 + `units` i64), multilingual Unit-HiFi-GAN vocoder | 03 |
| `WaitKPolicy` C++/Python upgrade (chunk ratio, bounded flush, `<EOS>` stop, mid-flush speech resume, adaptive hook, unit/prosody FIFO) + unit tests | 04 |
| Causal Conformer-Transformer S2UT training with multi-task ASR/S2TT CTC supervision & policy decision gate (Strict Wait-*k* vs. CTC-Adaptive; Hibiki kill record) | 05 |
| Translator prosody predictor head + fixed-shape ring-KV ONNX export (`translator_enc.onnx` + `translator_dec.onnx`) + 60 s multi-utterance parity | 06 |
| Two-session `OrtModel` execution with shared `mem_kv` IoBinding + `fv_s2ut_file` offline WAV→WAV runner | 07 |
| Real-time Mode B (`fluxvoice --s2ut`), `kRead` vocoder bypass, `kFlush` noise-gate override, Mode B `LatencyGovernor` burst cushion, `[stats]` telemetry | 08 |
| Live S2UT controls (`translate`, `wait_k`, `src_lang`, `tgt_lang`), glitch-free Mode A↔B & language hot-switching | 09 |
| Cross-lingual "Me" voice validation & fine-tune across all 6 `En↔{Es,Fr,De}` directions | 10 |
| End-to-end Mode B verification (multi-utterance `fv_probe` latency + AL, live-cable vs. offline ASR-BLEU across 6 pairs, 30-min soak, doc updates) | 11 |

## Sources

- StreamSpeech (ACL 2024) — arXiv:2406.03049: https://aclanthology.org/2024.acl-long.485/ · https://github.com/ictnlp/StreamSpeech
- Hibiki (ICML 2025) — arXiv:2502.03382: https://arxiv.org/abs/2502.03382 · https://github.com/kyutai-labs/hibiki
- Direct S2UT with Discrete Units (Lee et al., ACL 2022) — arXiv:2107.05604: https://aclanthology.org/2022.acl-long.235/ · https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_speech
- Enhanced Direct S2ST / Textless S2ST on Real Data (Popuri et al. 2022) — arXiv:2204.02967: https://arxiv.org/abs/2204.02967
- UTUT: Textless Unit-to-Unit Training for Many-to-Many Multilingual S2ST (Kim et al., IEEE TASLP 2024) — arXiv:2308.01831: https://arxiv.org/abs/2308.01831 · https://github.com/choijeongsoo/utut
- SimulEval (Ma et al., EMNLP 2020) — arXiv:2007.16193: https://github.com/facebookresearch/SimulEval
- CVSS Corpus (Jia et al., LREC 2022) — arXiv:2201.03713: https://github.com/google-research-datasets/cvss · https://huggingface.co/datasets/google/cvss
- CoVoST 2 (Wang et al. 2020) — arXiv:2007.10310: https://huggingface.co/datasets/facebook/covost2
- VoxPopuli (Wang et al., ACL 2021) — https://github.com/facebookresearch/voxpopuli
- CSS10 (Park & Mulc 2019) — https://github.com/kyubyong/css10
- CML-TTS (OpenSLR 146) — arXiv:2306.10097: https://www.openslr.org/146/ · https://huggingface.co/datasets/ylacombe/cml-tts
- mHuBERT-147 (Boito et al. 2024) — arXiv:2406.06371: https://huggingface.co/utter-project/mHuBERT-147
