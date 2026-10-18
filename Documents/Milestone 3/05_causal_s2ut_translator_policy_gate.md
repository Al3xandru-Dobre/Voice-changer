# Stage 05 — Causal S2UT translator & policy decision gate

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 04](04_waitk_policy_streaming_state_machine.md) · [Stage 06 →](06_translated_prosody_stateful_onnx_export.md)

**Status:** ☐ not started
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

- [ ] **Architecture (`python_training/fvml/translator.py`):**
  - [ ] Implement `CausalTranslatorEncoder` and `CausalTranslatorDecoder` with
        both full-sequence `forward()` (using causal / Wait-*k* cross-attention
        masks and ring-window masks) and per-chunk stateful `step()` sharing exact
        weights and math.
  - [ ] Training-only multi-task auxiliary heads:
    - `asr_ctc_head`: `Linear(320, 4 * V_spm_src)` on top of encoder layer 4
    - `s2tt_ctc_head`: `Linear(320, 4 * V_spm_tgt)` on top of encoder layer 6
    - `readiness_head`: `Linear(320, 1)` predicting expected new target chunk
      availability from CTC non-blank mass (kept for the CTC-Adaptive candidate).
  - [ ] Reduced-unit + duration variant (`--unit-mode reduced_dur`): predicts
        deduplicated unit IDs + integer frame duration $d \in [1..16]$ per step
        for comparison against the default `--unit-mode stacked_4`.
- [ ] **Multi-task training (`python_training/train_s2ut.py`):**
  - [ ] Joint loss:
        $$\mathcal{L} = \mathcal{L}_{\text{unit CE}} + 0.3\,\mathcal{L}_{\text{ASR CTC}} + 0.3\,\mathcal{L}_{\text{S2TT CTC}} + 0.1\,\mathcal{L}_{\text{eos BCE}} + 0.1\,\mathcal{L}_{\text{readiness}}$$
  - [ ] **Prefix / Wait-*k* curriculum:**
    - Warm up for 30% of steps with full-context cross-attention so the encoder
      and CTC heads converge quickly across all 6 directions.
    - Anneal to randomized simultaneous cross-attention masks ($k \sim
      \text{Uniform}\{1..8\}$ chunks) for the remaining 70% of steps so one model
      checkpoint handles any user-selected $k \in [1..10]$ at inference time.
  - [ ] **Silence-tail training examples:** append 400–800 ms of silence latents
        to 30% of training utterances and supervise the decoder to finish emitting
        the remaining target units and predict `<EOS>` during the silence tail.
  - [ ] Temperature-balanced multi-pair sampling across `es↔en`, `fr↔en`, `de↔en`;
        bf16 autocast, gradient accumulation sized for 8 GB VRAM, checkpoints
        every 30 min.
- [ ] **Streaming equivalence check:** verify `step()` (chunk by chunk with
      ring-buffer KV caches and `WaitKPolicy`) matches masked `forward()` over
      60 s concatenated utterances (unit argmax agreement $\ge 99.9\%$, max
      $|\Delta\text{logit}| \le 10^{-5}$).
- [ ] **Decision gate (`python_training/gate_s2ut.py`):**
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

## Files touched

- `python_training/fvml/translator.py` (new),
  `python_training/train_s2ut.py` (new),
  `python_training/gate_s2ut.py` (new),
  this file (gate table + decision),
  `Documents/Milestone 3/00_master.md` (open decisions 2, 3, 4)

## Definition of done

- Streaming `step()` matches masked `forward()` over 60 s: unit argmax agreement
  $\ge 99.9\%$ and max $|\Delta\text{logit}| \le 10^{-5}$.
- The gate table above is filled for all 6 directions, and the winning policy +
  unit mode meets every eligibility threshold.
- `silence_hallucination_20.tsv` produces 0 hallucinated Whisper words and
  terminates `kFlush` within $\le 3$ chunks after the translated sentence ends.
