# Stage 05 — Causal vocoder & encoder decision gate

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 04](04_content_encoder_candidates.md) · [Stage 06 →](06_stateful_onnx_export.md)

**Status:** ☐ not started
**Depends on:** Stage 02 (contract inputs), Stage 03 (conditioning features), Stage 04 (unit streams)
**Research:** §5 causal vocoders · §7 evaluation · §11 control-ready conditioning · §3 latency

## Objective

Train a **causal, stateful, multi-speaker unit vocoder** whose inputs are exactly
the contract's conditioning:
- unit ids
- target-domain log-f0, voicing, aperiodicity, energy
- a speaker-embedding vector

Then run the **decision gate**: convert the frozen test set with each candidate
encoder and choose the encoder + lookahead + hop by the latency-first rule below.
Finally, fine-tune the vocoder on the winner's ids.

## Context & references

- `python_training/train_unit_vocoder.py` is a stub. Its TODO is "causal
  Unit-HiFi-GAN generator + MPD/MSD discriminators → feature-matching + mel
  losses", and it assumes a single speaker.
- **Causal HiFi-GAN** (research §5):
  - left padding `(k−1)·d`; transposed convolutions trimmed right;
  - no lookahead, so stateful chunked output equals offline output exactly;
  - V2-class generator (~0.9M params) fits one CPU core and the 8 GB GPU.
- **Speaker identity** enters through FiLM on an embedding *vector*, so Stage 09
  can blend two voices (research §5, §11). The voice controls depend on
  target-domain log-f0 (research §4.2), which is why it is fixed here and not
  bolted on later.
- **OLA.** `OverlapAddProcessor` (M1 / Stage 04) expects `chunk + overlap`
  frames. A stateful causal vocoder doesn't need it, so the contract's
  `overlap_samples` defaults to 0.

## Tasks

- [ ] **Model** (`python_training/fvml/vocoder.py`, built from Stage 04's
      `StreamingConv1d` plus a streaming transposed conv):
  - [ ] Unit embedding (vocab → 128); content at its own frame rate, upsampled
        by repetition to the 20 ms conditioning rate (×1 for 50 Hz students, ×4
        for Mimi)
  - [ ] Conditioning stack `[logf0, voicing, aperiodicity, energy]` → causal
        pre-net
  - [ ] Upsampling ×320 to 16 kHz (e.g. 8·5·4·2) with causal MRF residual
        blocks
  - [ ] FiLM (γ, β from `spk_emb`) after each upsampling stage
  - [ ] Speaker table `nn.Embedding(n_speakers, 64)`, whose rows are exported to
        `voices.json` in Stage 06
- [ ] **Training** (`train_unit_vocoder.py`, real implementation):
  - [ ] Data: LJSpeech + the chosen Hi-Fi TTS speakers at 16 kHz. Features from
        Stage 03's causal tracker on each speaker's *own* audio; units =
        teacher km100 ids
  - [ ] Losses: LSGAN adversarial + feature matching + multi-resolution mel L1
        (weight ≈ 45)
  - [ ] Discriminators: MPD + multi-resolution STFT discriminator (MRD)
  - [ ] AdamW lr 2e-4, betas (0.8, 0.99), exponential decay 0.999/epoch
  - [ ] bf16 autocast, segment 8192–16384 samples, batch sized for 8 GB
  - [ ] Resumable checkpoints every 30 min; TensorBoard
  - [ ] Per-speaker F0/energy stats computed and stored with the checkpoint
- [ ] **Streaming equivalence.** Generator `step()` (1/2/4 frames, state carried)
      vs. `forward()` on 30 s: max |Δ| and a boundary-step check.
- [ ] **Quick vocoders for the gate**, same architecture, a fixed short schedule
      (e.g. 150k steps):
  - [ ] **A:** teacher-id vocoder, shared by all student heads
  - [ ] **M:** Mimi-q0 vocoder, only if Mimi survived Stage 04's probe
- [ ] **Gate evaluation** (`python_training/gate_report.py`):
  - [ ] For every candidate (student L0/L1/L2 at frames_per_step 1/2/4 and Mimi
        if alive), convert the frozen `test_clean_200` list into 2 target voices
        with the full *streaming* chain. Source features are mapped to the
        target with Stage 03's mapping.
  - [ ] Score with `eval_vc.py`: WER, CER, target cosine, source cosine, UTMOS,
        F0 PCC.
  - [ ] Cost: single-core encoder + vocoder per-step p95 (PyTorch).
  - [ ] Projected T−B from the master budget formula using the measured compute
        p95.
  - [ ] **Export smoke test:** export a random-weight per-step graph of the
        candidate architecture and run one step in ONNX Runtime CPU. A design
        that can't export can't win.
- [ ] **Decision.** Apply the rule, then record the gate table + decision here and
      in the master's open decisions 4 and 7.
- [ ] **Post-gate fine-tune.** Continue the vocoder on the winning head's ids
      (e.g. 50k steps). Re-score and confirm the WER stays within the rule.
- [ ] **Controls readiness checks** on the fine-tuned model:
  - [ ] f0 shift of −12/−6/0/+6/+12 st, measuring output median f0
  - [ ] intonation scale 0/1/2
  - [ ] a 50/50 embedding blend of two voices

  Results feed Stage 09.

## Decision gate

**Eligible** when *all* of these hold on `test_clean_200`:
- WER ≤ 10% absolute **and** ≤ best candidate's WER + 3 pts;
- target cosine ≥ best − 0.05;
- UTMOS ≥ 3.0;
- encoder + vocoder compute p95 ≤ 50% of the hop on one core.

**Choose** the eligible candidate with the **lowest projected T−B**; break ties
on UTMOS. If nothing is eligible, try in this order:
1. soft units (open decision 2);
2. a larger generator (V1-class) if compute allows;
3. Mimi, if it is the only candidate that passes quality. Mimi accepts ≤ 160 ms
   and is recorded as a deviation.

| Candidate | Hop (ms) | L | Alg. latency (ms) | Compute p95 (ms) | Projected T−B (ms) | WER | CER | Tgt cos | Src cos | UTMOS | F0 PCC | Export OK | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| student-L0 s1 | 20 | 0 | — | — | — | — | — | — | — | — | — | — | — |
| student-L1 s1 | 20 | 1 | — | — | — | — | — | — | — | — | — | — | — |
| student-L2 s1 | 20 | 2 | — | — | — | — | — | — | — | — | — | — | — |
| student-L1 s2 | 40 | 1 | — | — | — | — | — | — | — | — | — | — | — |
| mimi-q0 | 80 | 0 | — | — | — | — | — | — | — | — | — | — | — |

## Implementation notes

- **Train on target-domain log-f0.** The vocoder always sees *each training
  speaker's own* absolute log-f0. At run time C++ maps the user's whitened
  contour into the chosen voice's range (Stage 03), and Stage 09's pitch shift and
  intonation scale are applied in that same domain. Nothing else is needed for
  pitch controls.
- **Multi-speaker from the start.** Four to eight voices make the bootstrap
  vocoder more robust and give presets something to choose from. Stage 10 adds
  the user's voice as one more row.
- **Plan for 8 GB.** If out-of-memory errors appear, reduce segment length
  before batch size, and keep the discriminators' FFT sizes moderate. Use
  gradient checkpointing only as a last resort, because it slows a laptop run
  badly.
- **Keep checkpoints portable.** Store the contract fields (frame size, vocab,
  `cond_inputs`, `spk_emb_dim`) in the checkpoint, so Stage 06 never has to
  guess.
- **Non-causal-teacher distillation stays as a lever.** It improves causal
  students (research §5); use it only if the gate shows a quality gap at L=0/1.

## Files touched

- `python_training/fvml/vocoder.py` (new), `python_training/train_unit_vocoder.py`
  (real implementation), `python_training/gate_report.py` (new), this file
  (gate table + decision), `Documents/Milestone 2/00_master.md` (open decisions
  4, 7)

## Definition of done

- Streaming equals offline for the generator: max |Δ| ≤ 1e-4 over 30 s, with no
  boundary step larger than the signal's own maximum slope.
- Resynthesis (teacher ids, held-out utterances of a training speaker) reaches
  UTMOS ≥ 3.5.
- An f0 shift of +12 st gives a median output-f0 ratio of 2.0 ± 3%, measured with
  the offline reference tracker.
- The gate table is complete for every candidate. The decision, with its
  reasoning, is written here and in the master's open decisions. The winner
  meets the eligibility rule after the post-gate fine-tune.
