# Stage 04 — Content encoder candidates

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 03](03_causal_pitch_energy.md) · [Stage 05 →](05_causal_vocoder_decision_gate.md)

**Status:** ☐ not started
**Depends on:** Stage 01 (env, LibriSpeech, MUSAN/RIR, evaluation lists)
**Research:** §2 encoder landscape · §3 latency · §7 unit probes

## Objective

Produce the candidate encoders for the Stage 05 gate, each with streaming code
and probe numbers:

1. **A HuBERT-Base layer-7 k-means(100) teacher.** It is offline and non-causal.
   It provides the training labels and the upper bound on quality.
2. **A StreamVC-style causal conv student at 50 Hz.** It predicts the teacher's
   ids and comes in lookahead variants L ∈ {0, 1, 2}.
3. **A time-boxed kill test of Mimi's semantic codebook (q0).** It either drops
   Mimi or sends it to the gate.

Replace the `extract_units.py` stub with the real unit extractor for all of
them.

## Context & references

- Research §2.3: Mimi q0 VC tests give WER 120% (Kanade) and dWER 110
  (FocalCodec-Stream), and content leaks into Mimi's acoustic codebooks at
  12.5 Hz. Research §3: Mimi's 80 ms frame alone costs 80 ms of framing.
- **StreamVC recipe** (research §2.4): SoundStream-style causal conv encoder
  (C=64, D=64), HuBERT-Base layer-7 labels, mini-batch k-means K=100, 50 Hz.
  Output frame paired with input s_{t−2} (L=2) gives 60 ms architectural latency.
- **Tooling.** `fairseq`/`textlesslib` don't install cleanly on Python 3.12. Use
  `transformers` (`facebook/hubert-base-ls960`, `output_hidden_states`) and
  `scikit-learn` MiniBatchKMeans.
- The current `python_training/extract_units.py` stub defaults to `--tokenizer
  mimi` at 12.5 Hz with `--kmeans 500`; this stage replaces it.

## Tasks

- [ ] **Teacher.**
  - [ ] Extract HuBERT-Base layer-7 features (verify the indexing:
        `hidden_states[7]` = output of transformer layer 7) at 50 Hz on about
        100 h of train-clean-100.
  - [ ] Fit MiniBatchKMeans K=100 and save `units/km100_centroids.npy` + fit
        metadata.
  - [ ] Label LibriSpeech train-clean-100/360 and every vocoder training corpus
        (LJSpeech, chosen Hi-Fi TTS speakers) → per-utterance `int16` `.npy` +
        manifest.
- [ ] **Unit probes**, in `python_training/probe_units.py`:
  - [ ] phone purity, cluster purity and PNMI against Montreal Forced Aligner
        LibriSpeech phone alignments (precomputed public alignments, or run MFA
        on dev-clean), each at the model's own frame rate with majority-phone
        frame labels
  - [ ] speaker-ID probe: a logistic regression on unit n-gram histograms per
        utterance, 100 speakers from train-clean-100 (held-out utterances).
        Report accuracy vs. the 1% chance level
- [ ] **Mimi q0 kill test (time-box: 1 week of wall time).**
  - [ ] `MimiModel.encode` (24 kHz resample on the fly), keep codebook 0 → 12.5 Hz
        ids
  - [ ] Run the probes; record the PNMI ratio vs. the teacher
  - [ ] If the PNMI ratio is ≥ 0.8, Stage 05 trains a quick Mimi vocoder and
        compares WER. Otherwise drop Mimi now and record why
- [ ] **Student** (`python_training/fvml/streaming.py`, `fvml/content_encoder.py`,
      `train_content_encoder.py`):
  - [ ] Write `StreamingConv1d` (explicit left-context cache, `step()` and
        `forward()` share weights and math) from day one. Stage 06 exports
        exactly this module.
  - [ ] Causal SoundStream-style trunk, strides to 320 samples (e.g. 2·4·5·8),
        target ≤ 5M params. **One trunk, three heads** predicting the teacher id
        at delays 0/1/2 frames gives three variants from one training run
        (cross-entropy summed over heads).
  - [ ] Augment the student input only: MUSAN noise at 5–30 dB SNR (p≈0.6), RIR
        reverb (p≈0.4), random gain ±10 dB, mild random EQ. Labels always come
        from the clean teacher pass (denoising distillation).
  - [ ] bf16 autocast, batch sized for 8 GB (start at 32 × 2 s), checkpoint every
        30 min, resumable, TensorBoard logs in `runs/`. Budget: ≤ 3 days on the
        RTX 4060.
- [ ] **Streaming equivalence.** On 60 s of concatenated dev-clean, compare
      `step()` one frame at a time against `forward()` on the whole signal: logit
      difference and argmax agreement per head.
- [ ] **Cost.** `torch.set_num_threads(1)`, CPU, per-step p95 for
      frames_per_step 1/2/4, for each candidate (student heads share the trunk
      cost; Mimi measured if it survives).
- [ ] **`extract_units.py`, real implementation.**
  - [ ] `--tokenizer {teacher-km100, student-L0, student-L1, student-L2, mimi-q0}`
  - [ ] `--manifest`, `--out-dir`
  - [ ] Writes `.npy` ids + a manifest carrying frame rate, lookahead and vocab,
        so Stage 05 can't mix incompatible unit streams.

## Implementation notes

- **Why a shared vocoder works.** Students *predict teacher ids*, so a vocoder
  trained on teacher ids (Stage 05) can be driven by any student head. That
  avoids training a vocoder per variant on an 8 GB GPU. After the gate, the
  vocoder is fine-tuned on the winning head's ids to remove the train/test
  mismatch.
- **Gradient stop comes for free.** StreamVC stresses blocking decoder gradients
  into the content encoder to prevent speaker leakage. Here the student trains
  only on cross-entropy against teacher labels, never through the vocoder.
- **What lookahead means.** Head L at step t emits the id of frame t − L. Its
  latency cost is L × 20 ms and nothing else; the contract's `lookahead_frames`
  carries it.
- **Discrete vs. soft (open decision 2).** Keep the pre-softmax logits
  available. If no discrete variant passes the WER floor in Stage 05, a
  soft-unit vocoder is the fallback. Record that it breaks direct M3
  compatibility.
- **Teacher language.** HuBERT-Base is English-only. Note the WER on any
  non-English sample the user cares about, so M3 knows whether mHuBERT-147
  (NC licence) is needed.

## Files touched

- `python_training/fvml/streaming.py` (new), `python_training/fvml/content_encoder.py`
  (new), `python_training/train_content_encoder.py` (new),
  `python_training/probe_units.py` (new), `python_training/extract_units.py`
  (real implementation), `python_training/units/` (centroids; small, committed
  if < 1 MB, otherwise documented)

## Definition of done

- **Mimi kill decision recorded** in this file and in the master's open
  decisions. Mimi is dropped if its PNMI is < 0.8× the teacher's, or (after
  Stage 05's quick vocoder) its VC WER is > 2× the best student's.
- Every student head reaches phone purity ≥ 90% of the teacher's on dev-clean.
- Streaming equals offline for every head: max |Δlogit| ≤ 1e-5 and argmax
  agreement ≥ 99.9% over 60 s.
- Speaker-ID probe accuracy and single-core per-step p95 (frames_per_step
  1/2/4) are reported for the teacher, each student head and Mimi (if tested).
