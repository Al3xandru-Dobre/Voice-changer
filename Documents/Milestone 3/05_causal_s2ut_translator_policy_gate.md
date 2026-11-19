# Stage 05 — Causal S2UT translator & policy decision gate

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 04](04_waitk_policy_streaming_state_machine.md) · [Stage 06 →](06_translated_prosody_stateful_onnx_export.md)

**Status:** ◐ implemented & smoke-tested (2026-11-15) — model, trainer, streaming driver, equivalence check and gate tooling are in and unit-tested; the multi-day training run, the `De→En` CTC ablation and the gate table are pending on the Stage 01 corpora, a GPU slot and the Stage 03 student / vocoder checkpoints (see [Verification](#verification-2026-11-15-windows--code--smoke-runs-only))
**Depends on:** Stage 01 (paired manifests, SentencePiece models, `eval_s2ut.py`), Stage 03 (multilingual student `latents` + target $K=1000$ units + vocoder), Stage 04 (`fvml/waitk_policy.py`)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §2 S2UT landscape & multi-task CTC · §4 latency & Wait-*k* at 50 Hz

## Objective

Train the **bidirectional `En↔{Es,Fr,De}` causal Conformer-Transformer S2UT
translator** (`python_training/fvml/translator.py`, `train_s2ut.py`) that maps
streaming 50 Hz Stage 1 continuous `latents` `[B, T, 128]` + `<src_lang>` /
`<tgt_lang>` tags into 50 Hz target discrete units `[B, T', 1000]`.

Then run the **Milestone 3 policy & architecture decision gate** across all 6
language directions (`es→en`, `fr→en`, `de→en`, `en→es`, `en→fr`, `en→de`),
comparing:
1. **Strict Wait-*k*** ($k \in \{2, 3, 4, 5\}$ chunks of 80 ms) with stacked 4-frame
   unit output;
2. **CTC-Adaptive Wait-*k*** (StreamSpeech-style CTC alignment readiness trigger
   allowing 0/1/2 target chunks per step);
3. **Deduplicated (reduced) units + duration predictor** vs. **stacked 4-frame
   units**.

## Context & references

- **Why multi-task CTC supervision is mandatory (research §2.3):** Direct S2UT
  without text supervision scores 2.46 ASR-BLEU on `De→En` (CVSS-C), whereas
  StreamSpeech's auxiliary source-ASR CTC + target-text S2TT CTC supervision
  raises `De→En` to 19.72 BLEU and `Fr/Es→En` to 27.58 / 26.16 BLEU.
- **Hibiki & SeamlessStreaming spike (research §2.2):** Hibiki (1B–2B params,
  Mimi 8–16 RVQ streams, Fr→En only) and SeamlessStreaming (2.3B params,
  CC-BY-NC-4.0) exceed the single-core CPU and 8 GB training GPU budgets by
  25–50×. Their kill-analysis table is recorded in research §2.2 and closed here.
- **Model sizing for single-core CPU real-time execution:**
  - Target ~25–38M parameters for the run-time translator (`translator_enc` +
    `translator_dec` excluding training-only CTC vocabulary projections):
    - **Causal Chunked Conformer Encoder (`translator_enc`):** 4-frame chunk
      stacking (`[B, T/4, 4*128]` → linear $d_{\text{model}} = 320$),
      `<src_lang>` embedding, 6 Conformer blocks (4 heads, FFN 1280, causal
      depthwise conv kernel 7 with explicit left-context cache, sliding-window
      self-attention $W_{\text{src}} = 128$ chunks = 10.24 s).
    - **Causal Transformer Decoder (`translator_dec`):** takes previous 4-frame
      stacked target unit embeddings + `<tgt_lang>` embedding, 6 Transformer
      decoder blocks ($d_{\text{model}} = 320$, 4 heads, sliding-window causal
      self-attention $W_{\text{tgt}} = 128$, cross-attention over encoder memory
      masked to $\le t_{\text{src}}(t_{\text{tgt}})$ per the Wait-*k* schedule),
      followed by a **stacked 4-frame unit head** (`Linear(320, 4 * 1003)`) +
      `eos_prob` head.

## Tasks

- [x] **Architecture (`python_training/fvml/translator.py`):**
  - [x] Implement `CausalTranslatorEncoder` and `CausalTranslatorDecoder` with
        both full-sequence `forward()` (using causal / Wait-*k* cross-attention
        masks and ring-window masks) and per-chunk stateful `step()` sharing exact
        weights and math.
  - [x] Training-only multi-task auxiliary heads:
    - `asr_ctc_head`: `Linear(320, 4 * V_spm_src)` on top of encoder layer 4
    - `s2tt_ctc_head`: `Linear(320, 4 * V_spm_tgt)` on top of encoder layer 6
    - `readiness_head`: `Linear(320, 1)` predicting expected new target chunk
      availability from CTC non-blank mass (kept for the CTC-Adaptive candidate).
  - [x] Reduced-unit + duration variant (`--unit-mode reduced_dur`): predicts
        deduplicated unit IDs + integer frame duration $d \in [1..16]$ per step
        for comparison against the default `--unit-mode stacked_4`.
- [ ] **Multi-task training (`python_training/train_s2ut.py`)** *(trainer implemented and smoke-tested on a synthetic corpus; the real run is pending)*:
  - [x] Joint loss:
        $$\mathcal{L} = \mathcal{L}_{\text{unit CE}} + 0.3\,\mathcal{L}_{\text{ASR CTC}} + 0.3\,\mathcal{L}_{\text{S2TT CTC}} + 0.1\,\mathcal{L}_{\text{eos BCE}} + 0.1\,\mathcal{L}_{\text{readiness}}$$
  - [x] **Prefix / Wait-*k* curriculum:**
    - Warm up for 30% of steps with full-context cross-attention so the encoder
      and CTC heads converge quickly across all 6 directions.
    - Anneal to randomized simultaneous cross-attention masks ($k \sim
      \text{Uniform}\{1..8\}$ chunks) for the remaining 70% of steps so one model
      checkpoint handles any user-selected $k \in [1..10]$ at inference time.
  - [x] **Silence-tail training examples:** append 400–800 ms of silence latents
        to 30% of training utterances and supervise the decoder to finish emitting
        the remaining target units and predict `<EOS>` during the silence tail.
  - [x] Temperature-balanced multi-pair sampling across `es↔en`, `fr↔en`, `de↔en`;
        bf16 autocast, gradient accumulation sized for 8 GB VRAM, checkpoints
        every 30 min.
- [x] **Streaming equivalence check:** verify `step()` (chunk by chunk with
      ring-buffer KV caches and `WaitKPolicy`) matches masked `forward()` over
      60 s concatenated utterances (unit argmax agreement $\ge 99.9\%$, max
      $|\Delta\text{logit}| \le 10^{-5}$).
- [ ] **Decision gate (`python_training/gate_s2ut.py`)** *(conversion, scoring glue, rules, selection and report implemented and tested on fixtures / fakes; no candidate has been evaluated: there is no trained checkpoint)*:
  - [ ] Evaluate each candidate policy (`waitk-2`, `waitk-3`, `waitk-4`,
        `waitk-5`, `ctc-adaptive`) and unit mode (`stacked_4` vs. `reduced_dur`)
        on all 6 `test_*_200.tsv` lists + `reordering_torture_60.tsv` +
        `silence_hallucination_20.tsv` using Stage 03's multilingual vocoder.
  - [ ] Score with `eval_s2ut.py`: ASR-BLEU, ASR-chrF++, onset $T-B$, SimulEval
        `AL` / `AL_CA`, UTMOS, ECAPA `tgt_cos`, and single-core CPU per-chunk
        $p_{95}$ compute time.
  - [ ] Record the full gate table and select the default policy + per-pair
        default $k$ ( written into `langs.json`).

## Decision gate

**Eligible** when *all* of these hold across the 6 directions:
- **ASR-BLEU:** $\ge 22.0$ on `Fr→En` and `Es→En`; $\ge 16.0$ on `De→En`; $\ge
  18.0$ on `En→Es` and `En→Fr`; $\ge 14.0$ on `En→De`.
- **Naturalness:** UTMOS $\ge 3.0$ in every target language.
- **Latency:** projected onset $T - B \le 500\text{ ms}$; SimulEval $\text{AL}
  \le 1200\text{ ms}$ (`En↔{Es,Fr}`) and $\le 1500\text{ ms}$ (`En↔De`).
- **Compute:** single-core CPU $p_{95}$ for (`encoder s4` + `translator_enc` +
  `translator_dec` + `vocoder s4`) $\le 40\text{ ms}$ (50% of an 80 ms chunk).
- **Zero silence babble:** 0 hallucinated words on `silence_hallucination_20.tsv`.

| Candidate | Unit mode | Onset $T-B$ (ms) | Mean AL (ms) | `Es→En` BLEU | `Fr→En` BLEU | `De→En` BLEU | `En→Es` BLEU | `En→Fr` BLEU | `En→De` BLEU | Compute $p_{95}$ (ms) | UTMOS | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Strict Wait-2 | `stacked_4` | ~250 | — | — | — | — | — | — | — | — | — | — |
| Strict Wait-3 | `stacked_4` | ~330 | — | — | — | — | — | — | — | — | — | — |
| Strict Wait-4 | `stacked_4` | ~410 | — | — | — | — | — | — | — | — | — | — |
| Strict Wait-5 | `stacked_4` | ~490 | — | — | — | — | — | — | — | — | — | — |
| CTC-Adaptive | `stacked_4` | ~300 | — | — | — | — | — | — | — | — | — | — |
| Strict Wait-3 | `reduced_dur` | ~330 | — | — | — | — | — | — | — | — | — | — |

## Implementation notes

- **Why random-$k$ training matters.** A model trained only at fixed $k=3$
  degrades when the user drags the Milestone 4 Wait-*k* slider to $k=1$ or $k=8$.
  Sampling $k \sim \text{Uniform}\{1..8\}$ per batch trains a single omnidirectional
  checkpoint that adapts cleanly to any live slider position.
- **Ablation check on `De↔En`.** Run a short 20k-step ablation with vs. without
  the auxiliary CTC losses on `De→En` to confirm the StreamSpeech convergence gap
  before committing the full multi-day training run.

## As built (what Stage 06 exports)

These are the definitions the code implements where the task list leaves room.
`fvml/translator.py`'s module docstring is the reference.

- **Size.** `stacked_4`: **26,265,710** run-time parameters (encoder 15,604,353
  incl. the 12 memory projections and the 321-parameter `readiness_head`; decoder
  10,661,357). `reduced_dur`: 25,003,197. Training-only CTC projections:
  61,632,000 (2 heads × 4 languages × `Linear(320, 4 × 6000)`). The Conformer
  block is the macaron form (½ FFN → self-attention → conv → ½ FFN → LayerNorm);
  a single-FFN block would land at ~21M, below the 25M floor. The conv module
  normalises with LayerNorm, not BatchNorm (causal, identical in train and eval).
- **No absolute positions.** Every attention adds a learned per-head bias indexed
  by a distance $d \in [0, W)$ and a key is visible iff $0 \le d < W$:
  self-attention $d = i - j$; cross-attention $d = g - 1 - j$, where $g$ is the
  number of source chunks read when the target step is decoded. A stream that is
  never reset therefore behaves like one that is.
- **Wait-*k* cross-attention mask.** `forward()` takes the schedule as a tensor
  `src_visible` `[B, T_tgt]` ($g$ per decoder step); training builds
  $g(w) = \min(w + k, S)$ for write $w$ (`waitk_visible`; $k$ = `None` is full
  context). `step()` has no mask argument: it reads the memory ring with
  `src_pos`, so it sees exactly what `WaitKPolicy` let the encoder read. The
  equivalence tests replay the schedule the policy produced through `forward()`.
- **Rings.** `enc_kv_<l>` / `dec_kv_<l>` are `[1, 2H, W, Dh]` (K heads then V
  heads), `enc_conv_<l>` is `[1, 320, 6]`, `mem_k_<l>` / `mem_v_<l>` are
  `[1, H, W_src, Dh]`, one pair per **decoder** layer: the encoder owns the
  cross-attention key/value projections and writes slot `src_pos mod W`. Slot
  $s$ of a ring whose newest entry has index `last` holds distance
  `(last - s) mod W`, live iff that is `<= last`. `src_pos` / `tgt_pos` are
  int64 `[1]` tensors; slots are selected with tensor comparisons (no Python-int
  position, MEMORY.md rule 6). Masked logits are set to $-10^4$, not $-\infty$:
  a decoder step before any source chunk averages zeros instead of emitting NaN.
- **Per-language CTC heads.** `translator_enc` only receives `<src_lang>`, so a
  single S2TT head could not know the target language of an `en→X` utterance.
  Each CTC head is one `Linear(320, 4 × V_spm)` per language (ASR picked by
  source, S2TT by target language); the CTC blank is the SentencePiece `<pad>` id.
- **Readiness target.** Per source chunk, the detached S2TT CTC non-blank mass,
  rescaled so an utterance's chunks sum to its number of target chunks, clamped
  to $[0, 1]$; BCE on `readiness_logit`.
- **Targets.** `stacked_4`: frames + `<EOS>`, `<PAD>` to whole chunks (a whole
  number of chunks still gets its own `<EOS>` step). Under Wait-*k* the policy
  keeps writing while the source lasts ($S - k + 1$ writes), so a target that
  ends early is followed by all-`<PAD>` steps with eos target 1; with a silence
  tail this is exactly "finish, say `<EOS>`, stay silent".
- **`reduced_dur`.** One run-length token per decoder step: unit head
  `Linear(320, 1003)` + duration head `Linear(320, 16)` (class $c$ = $c + 1$
  frames, runs longer than 16 split), input = unit embedding + duration
  embedding. A token starting at target frame $F$ is decoded during write
  $F // 4$, i.e. $g = \min(F // 4 + k, S)$; at run time one policy write runs
  decoder steps until the emitted durations reach 4 more frames (surplus carries
  over). Its `step()` (`prev_dur` in, `duration` out, variable frames per write)
  is **not** contract v2: if it won the gate, the contract and the C++ FIFO
  feed would have to change.
- **Streaming driver** (`fvml/s2ut_stream.py`): `StreamingTranslator` runs the
  canonical per-chunk order of `fvml/waitk_policy.py` around a `TorchRuntime`
  (`step()`) or an `OrtRuntime` (the exported graphs), zeroes the translator
  state when the policy completes a flush, and keeps it on a mid-flush
  resumption. `export_step_graphs()` writes contract-v2 `translator_enc.onnx` /
  `translator_dec.onnx` (`prosody_ctx` is an input already; `has_prosody_output`
  is false until Stage 06).

Findings that affect the gate:

- **CTC-Adaptive readiness is a `[0, 2]` expected-chunk count** *(BUG-029,
  fixed 2026-11-18)*. Stage 04 widened `WaitKPolicy.on_source_readiness` to
  `[0, 2]`; `translator.py::readiness_targets()` now clamps to `[0, 2]` and
  `s2ut_loss()` supervises `ready / 2` with BCE, while `push_chunk()` feeds
  `2 * sigmoid(readiness_logit)` into the policy (`s2ut_stream
  .expected_chunks_from_logit`; a logit of 0 means "one expected chunk"). The
  pre-fix note — that readiness clamped to 1.0 and the candidate could only
  delay writes, never catch up (0 two-write steps in 4M random steps at
  constant $k$) — described the `[0, 1]$ probability mapping this replaced.
  The Stage 07/08 C++ runner must apply the same `2.0f * sigmoidf(logit)`
  mapping before `on_source_readiness()`.
- **Compute budget is tight.** See the ORT measurement below: the translator
  alone costs 23 ms p95 per write step of the 40 ms allowed for all four stages.
- **`<EOS>` then more speech.** If `<EOS>` arrives in a pause shorter than the
  tail flush and the user resumes, the decoder continues after its own `<EOS>`,
  which single-utterance training never shows it. Left to Stage 08 (reset the
  decoder there) or to concatenated-utterance training.

## Verification (2026-11-15, Windows — code + smoke runs only)

Windows 11, CPython 3.12.13, torch 2.11.0+cu128, onnxruntime 1.31.0, run from
`python_training/` with `PYTHONIOENCODING=utf-8`. No corpus, no Stage 03
checkpoint and no trained translator exist on this machine: **nothing below is a
quality result, and the gate table above is untouched.**

- **Unit tests** (`.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
  tests/test_translator.py tests/test_translator_export.py tests/test_train_s2ut.py
  tests/test_gate_s2ut.py`): **54 passed, 1 skipped** (the 60 s run, below).
  Whole suite (`-m pytest -q -p no:cacheprovider`, 6 min 47 s): 313 passed, 7
  skipped, 1 failed; the failure is `test_target_voice.py::test_extract_units_perturbation`
  against Stage 03's in-progress `extract_units.py` (a new `include_noncommercial`
  argument), not a Stage 05 file.
- **Streaming equivalence, Definition of done** (`FLUXVOICE_SLOW_TESTS=1 ...
  pytest tests/test_translator.py -k 60s_full_size -s`, ~75 s): full-size model
  (random weights), 750 chunks = 60.0 s of concatenated 4–5.6 s utterances with
  160 ms pauses, `WaitKPolicy(k=3)`, 128-chunk rings wrapping 5.8 times, 748
  decoder steps = 2,992 unit frames, `step()` vs masked `forward()`:

  | Precision | max $|\Delta\text{logit}|$ | Unit argmax agreement |
  |---|---:|---:|
  | float64 | $3.3 \times 10^{-15}$ | 100.0000 % |
  | float32 | $1.8 \times 10^{-6}$ | 100.0000 % |

  Both meet $\le 10^{-5}$ / $\ge 99.9\%$. The default run keeps a 280-chunk
  (22.4 s, 2.2× the window) full-size float64 variant plus tiny-model variants
  for flush + reset, mid-flush resumption, adaptive writes, `reduced_dur` and
  free running (all $\le 10^{-10}$ in float64).
- **ONNX Runtime** (`tests/test_translator_export.py`): the dynamo exporter
  writes both graphs with no fallback; they pass `fvml.export.finalize` (strict
  shape inference, full checker, contract v2 round trip),
  `validate_s2ut_shared_io` and `validate_s2ut_bundle` with the Stage 02 fixture
  encoder + vocoder. ORT vs PyTorch `step()` through the policy over 92 decoder
  steps (5 ring wraps, then 3 flush/reset cycles): max $|\Delta\text{logit}|$
  $6.6 \times 10^{-7}$, identical argmax, `translator_enc` states within $10^{-5}$.
- **Translator-only ORT cost** (`gate_s2ut.py --bench-translator --exporter
  dynamo`; full-size, random weights, one intra-op thread, 400 write steps, laptop
  shared with other jobs):

  | Step | p50 (ms) | p95 (ms) | max (ms) |
  |---|---:|---:|---:|
  | READ: `translator_enc` | 10.0 | 14.1 | 16.2 |
  | `translator_dec` | 6.5 | 9.5 | 10.8 |
  | WRITE: `translator_enc` + `translator_dec` | 17.5 | 23.2 | 25.5 |

  That leaves ~17 ms p95 for the Stage 1 encoder s4 + vocoder s4 inside the 40 ms
  rule. A step streams all 105 MB of FP32 weights through the cache once, so the
  cost is memory-bound; `reduced_dur` runs 2–3 decoder steps per write and will
  be slower.
- **Trainer smoke** (`train_s2ut.py --smoke <dir> --steps 60 --ckpt-steps 30`,
  CPU, tiny model, synthetic 96-utterance corpus over the 6 directions): total
  loss 10.80 → 5.13 (unit CE 3.41 → 2.80, ASR CTC 12.04 → 3.79, S2TT CTC 12.22 →
  3.67), simultaneous share 0 → 1 across the 30 % boundary, silence tails on
  26–34 % of utterances; `--steps 90 --resume auto` resumed at step 60 and
  reached 4.93.
- **8 GB sizing** (one capped probe on the RTX 4060, full-size model + CTC heads,
  bf16, AdamW step): the default `--max-chunks 2400` (192 s of padded source per
  micro-batch) peaks at **2.35–2.94 GiB** (16 × 150, 48 × 50, 8 × 250 chunks), so
  `--accum 4` gives ~770 s of source per update with headroom for a shared card.
- **Gate** (`gate_s2ut.py --self-test`: 84 checks, also under `python -I` with the
  standard library only; `tests/test_gate_s2ut.py`): every threshold at and past
  its limit, ORT-clock requirement (BUG-015), missing metrics, selection and
  per-pair *k*; and the whole chain on the fixture bundle (fixture
  `encoder_s4.onnx` → tiny translator in ORT through `WaitKPolicy` → FIFO →
  fixture `vocoder_s4.onnx`, traces scored by `eval_s2ut`'s Average Lagging, a
  fake Whisper/UTMOS scorer).

### Pending runs (in order)

```powershell
cd python_training
# 0. Stage 01: corpora, SentencePiece, the 7 missing eval lists. Stage 03: codebook,
#    student, vocoder, and a bundle with encoder_s4.onnx / vocoder_s4.onnx / voices.json.
# 1. Features per paired manifest <key> (Stage 03's extract_units.py):
.venv\Scripts\python.exe extract_units.py --tokenizer student-L1 --checkpoint <student.pt> --save-latents --side src --manifest s2ut/<key>.jsonl --out-dir data/units/student/<key>.src
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ --prosody --side tgt --manifest s2ut/<key>.jsonl --out-dir data/units/mhubert/<key>.tgt
#    plus a silence bank: any [T, 128] .npy of the student's latents on silence / room tone.
# 2. De->En ablation, 20k updates with and without the CTC losses:
.venv\Scripts\python.exe train_s2ut.py --manifest s2ut/cvss-c-de_en.jsonl --src-latents data/units/student/cvss-c-de_en.src --tgt-units data/units/mhubert/cvss-c-de_en.tgt --spm-dir data/spm --silence-latents <silence.npy> --val-split dev --steps 20000 --run-name s2ut_de_en_ctc
.venv\Scripts\python.exe train_s2ut.py <same> --ctc-weight 0 --run-name s2ut_de_en_noctc
# 3. The two full runs (repeat --manifest / --src-latents / --tgt-units per corpus; --resume auto to continue):
.venv\Scripts\python.exe train_s2ut.py --manifest <all 7 paired manifests> --src-latents <...> --tgt-units <...> --spm-dir data/spm --silence-latents <silence.npy> --val-split dev --run-name s2ut_v1
.venv\Scripts\python.exe train_s2ut.py <same> --unit-mode reduced_dur --run-name s2ut_reduced_v1
# 4. The gate (fills the table above, writes langs.json's default_wait_k), then the 60 s check on the trained weights:
.venv\Scripts\python.exe gate_s2ut.py --checkpoint ..\checkpoints\s2ut_v1\final.pt --checkpoint-reduced ..\checkpoints\s2ut_reduced_v1\final.pt --bundle ..\models\<bundle> --voice <voice> --targets eval_lists\targets.tsv --out ..\runs\s2ut_gate
.venv\Scripts\python.exe gate_s2ut.py --bench-translator --checkpoint ..\checkpoints\s2ut_v1\final.pt --out ..\runs\s2ut_bench
```

The gate conditions the vocoder on neutral target-voice prosody (or a per-unit
prior, `--unit-prosody`) because the prosody head is Stage 06; its UTMOS column
is therefore a lower bound on what Stage 06 will produce.

## Files touched

- `python_training/fvml/translator.py` (new),
  `python_training/fvml/s2ut_stream.py` (new),
  `python_training/train_s2ut.py` (new),
  `python_training/gate_s2ut.py` (new),
  `python_training/tests/{test_translator,test_translator_export,test_train_s2ut,test_gate_s2ut}.py` (new),
  this file (gate table + decision),
  `Documents/Milestone 3/00_master.md` (open decisions 2, 3, 4 — untouched until the gate runs)

## Definition of done

- Streaming `step()` matches masked `forward()` over 60 s: unit argmax agreement
  $\ge 99.9\%$ and max $|\Delta\text{logit}| \le 10^{-5}$.
- The gate table above is filled for all 6 directions, and the winning policy +
  unit mode meets every eligibility threshold.
- `silence_hallucination_20.tsv` produces 0 hallucinated Whisper words and
  terminates `kFlush` within $\le 3$ chunks after the translated sentence ends.
