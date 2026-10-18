# Stage 04 — Content encoder candidates

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 03](03_causal_pitch_energy.md) · [Stage 05 →](05_causal_vocoder_decision_gate.md)

**Status:** ◐ in progress (2026-10-18). All the code is written, tested and smoke-run. The long runs are waiting for data and GPU time: 100 h of teacher features, the k-means fit, labelling, student training, MFA alignments and the Mimi kill test (see Verification)
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

- [ ] **Teacher** (`fvml/teacher.py`; code done, runs pending).
  - [ ] Extract HuBERT-Base layer-7 features (verify the indexing:
        `hidden_states[7]` = output of transformer layer 7) at 50 Hz on about
        100 h of train-clean-100. *Indexing verified by a forward hook on
        `encoder.layers[6]` (`test_teacher_layer7_indexing`).*
  - [ ] Fit MiniBatchKMeans K=100 and save `units/km100_centroids.npy` + fit
        metadata. *Command: `extract_units.py --fit-kmeans --manifest
        librispeech-train-clean-100.jsonl --hours 100 --out-dir units/`.*
  - [ ] Label LibriSpeech train-clean-100/360 and every vocoder training corpus
        (LJSpeech, chosen Hi-Fi TTS speakers) → per-utterance `int16` `.npy` +
        manifest. *Command: `extract_units.py --tokenizer teacher-km100
        --centroids units/ --manifest … --out-dir …`.*
- [ ] **Unit probes**, in `python_training/probe_units.py` (math in
      `fvml/unit_probes.py`; code done and tested on synthetic TextGrids; no
      MFA alignments are available locally yet):
  - [ ] phone purity, cluster purity and PNMI against Montreal Forced Aligner
        LibriSpeech phone alignments (precomputed public alignments, or run MFA
        on dev-clean), each at the model's own frame rate with majority-phone
        frame labels
  - [ ] speaker-ID probe: a logistic regression on unit n-gram histograms per
        utterance, 100 speakers from train-clean-100 (held-out utterances).
        Report accuracy vs. the 1% chance level
- [ ] **Mimi q0 kill test (time-box: 1 week of wall time).** *The tokenizer
      (`extract_units.py --tokenizer mimi-q0`, pinned `kyutai/mimi@89091b3e`) is
      done and smoke-tested; the test itself has not been run.*
  - [ ] `MimiModel.encode` (24 kHz resample on the fly), keep codebook 0 → 12.5 Hz
        ids
  - [ ] Run the probes; record the PNMI ratio vs. the teacher
  - [ ] If the PNMI ratio is ≥ 0.8, Stage 05 trains a quick Mimi vocoder and
        compares WER. Otherwise drop Mimi now and record why
- [ ] **Student** (`python_training/fvml/streaming.py`, `fvml/content_encoder.py`,
      `train_content_encoder.py`). Code done; **training not run**.
  - [x] Write `StreamingConv1d` (explicit left-context cache, `step()` and
        `forward()` share weights and math) from day one. Stage 06 exports
        exactly this module.
  - [x] Causal SoundStream-style trunk, strides to 320 samples (e.g. 2·4·5·8),
        target ≤ 5M params. **One trunk, three heads** predicting the teacher id
        at delays 0/1/2 frames gives three variants from one training run
        (cross-entropy summed over heads).
  - [x] Augment the student input only: MUSAN noise at 5–30 dB SNR (p≈0.6), RIR
        reverb (p≈0.4), random gain ±10 dB, mild random EQ. Labels always come
        from the clean teacher pass (denoising distillation). *`fvml/augment.py`;
        tested on synthetic noise/RIR files only, since MUSAN and the RIRs are
        still downloading.*
  - [x] bf16 autocast, batch sized for 8 GB (start at 32 × 2 s), checkpoint every
        30 min, resumable, TensorBoard logs in `runs/`. Budget: ≤ 3 days on the
        RTX 4060.
- [ ] **Streaming equivalence** (`stream_check.py`). Passes with random init
      and with a smoke checkpoint; it must be rerun on the trained checkpoint. On 60 s of concatenated dev-clean, compare
      `step()` one frame at a time against `forward()` on the whole signal: logit
      difference and argmax agreement per head.
- [ ] **Cost** (`probe_units.py --cost`). Measured for the untrained student,
      the teacher and Mimi; rerun on the final model. `torch.set_num_threads(1)`, CPU, per-step p95 for
      frames_per_step 1/2/4, for each candidate (student heads share the trunk
      cost; Mimi measured if it survives).
- [x] **`extract_units.py`, real implementation.**
  - [x] `--tokenizer {teacher-km100, student-L0, student-L1, student-L2, mimi-q0}`
  - [x] `--manifest`, `--out-dir`
  - [x] Writes `.npy` ids + a manifest carrying frame rate, lookahead and vocab,
        so Stage 05 can't mix incompatible unit streams. *`units_meta.json`
        also records the label space (the centroids' sha256), and Stage 05
        checks it with `fvml.units.check_compatible()`.*

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
- Added during implementation:
  - `fvml/teacher.py`, `fvml/units.py` (the units-directory format and the
    tokenizers), `fvml/augment.py`, `fvml/unit_probes.py`.
  - `stream_check.py`.
  - `tests/test_streaming.py` (26 tests) and `tests/test_content_encoder.py`
    (22 tests).
- The real km100 centroids will be about 307 KB, under the 1 MB limit, so
  they get committed.

## Definition of done

- **Mimi kill decision recorded** in this file and in the master's open
  decisions. Mimi is dropped if its PNMI is < 0.8× the teacher's, or (after
  Stage 05's quick vocoder) its VC WER is > 2× the best student's.
- Every student head reaches phone purity ≥ 90% of the teacher's on dev-clean.
- Streaming equals offline for every head: max |Δlogit| ≤ 1e-5 and argmax
  agreement ≥ 99.9% over 60 s.
- Speaker-ID probe accuracy and single-core per-step p95 (frames_per_step
  1/2/4) are reported for the teacher, each student head and Mimi (if tested).

## Design decisions

- **Student widths:**
  - Channels 32 → 64 → 128 → 256 → 256 (stem, then each block), 3 residual
    units per block (k 7, dilations 1/3/9). A 50 Hz context stack follows
    (2 units, k 3, dilations 1/2), then a projection to D = 64 and a per-frame
    LayerNorm. Its output is `latents`, which Milestone 3 uses.
  - Each head is a causal conv over frames t−L..t (64 → 256) → ELU → 1×1 →
    100.
  - Totals: **4.29 M params, 3.02 GMAC per second of audio**, 432 ms of past
    context, a state of 22 tensors / 42 694 floats.
  - `--channels` gives cheaper variants: 16…256 is 1.90 M params / 0.80 GMAC/s;
    24…256 is 2.94 M / 1.72 GMAC/s.
- **Frame alignment.** Frame i covers samples [320 i, 320 i + 320) of the
  end-zero-padded input; student frame i closes on sample 320 i + 319.
  - The teacher pads HuBERT's input by 40 samples each side. A 320·n input then
    gives exactly n frames, centred on the same sample as student frame i.
  - Student ids are stored with the head delay removed and `lookahead_frames`
    recorded. At run time, head L's ids arrive L frames late, so **Stages 05
    and 07 must run the Stage 03 pitch tracker with the same L**: its push(t)
    already returns frame t − L.
- **Teacher input is not normalised.** fairseq pretrained HuBERT-Base with
  `task.normalize: false`. HF's `preprocessor_config.json` says
  `do_normalize: true`, and it is ignored here.
  - The teacher runs one utterance per forward. The first conv's GroupNorm
    spans the utterance, so padded batches would change the features.
  - It runs in fp32 with TF32 off; CPU and GPU labels agree 100%.
  - Layers 8–12 are dropped. Revision pinned: `dba3bb02`.
- **k-means** uses fairseq `simple_kmeans` settings on a uniform reservoir
  sample, by default 500 k frames (1.5 GB).
- **Unit-stream compatibility.** `check_compatible()` compares frame rate,
  vocab, label space and lookahead. Driving a teacher-trained vocoder with a
  student head must pass `ignore=("lookahead_frames",)` explicitly.
- **Augmentation**, in order, each step optional:
  1. RIR (p 0.4): cut at the direct-path peak, 1 s max, unit energy; dry RMS
     restored afterwards.
  2. MUSAN (p 0.6, 5–30 dB SNR).
  3. One RBJ EQ biquad (p 0.5, ±6 dB).
  4. Gain ±10 dB, then clip.
- **Training defaults:**
  - AdamW 5e-4 (betas 0.9/0.98, weight decay 0.01), 2 k warmup, cosine down to
    2%, gradient clip 1.0, 300 k steps.
  - Utterances are sampled in proportion to their length.
  - Checkpoints: `last.pt` / `best.pt` / `final.pt` under `checkpoints/`.

## Verification

Measured 2026-10-18. The CPU was shared with other jobs, so p95 values varied
by up to 2× between runs.

- **Tests:** `tests/test_streaming.py` + `tests/test_content_encoder.py` give
  48 passed, about 35 s on CPU. The full `python_training` suite is 106
  passed.
  - Coverage includes chunked = offline for stride 2/4/5/8 and dilations 3/9,
    causality of the conv and of the whole model, and the head-delay label
    shift.
  - It also covers PNMI/purity on hand-made tables, TextGrid parsing,
    `check_compatible`, train → resume → extract, and HuBERT layer-7
    indexing on the cached weights.
- **Streaming equivalence** (`stream_check.py`, 60 s of dev-clean, random
  init, fps 1/2/4): max |Δlogit| **1.6–1.9e-6** for every head, argmax
  agreement **1.000**, latents ≤ 5.5e-6. A 50-step smoke checkpoint gives
  1.2–1.9e-6 and 1.000. (DoD: ≤ 1e-5 and ≥ 99.9%; the trained checkpoint is
  still to be checked.)
- **Smoke runs** (20 dev-clean utterances):
  - Teacher frame count = n for every utterance.
  - Toy k-means on 7 480 frames uses all 100 units.
  - 50 training steps, CUDA bf16, 32 × 2 s: **3.7 it/s, peak 2.43 GiB**.
    300 k steps would take about 22 h before real augmentation, inside the
    3-day budget.
  - `--resume auto` continued 50 → 60.
- **Single-core cost** (untrained student; p50 / p95 per step):

  | Candidate | fps 1 (20 ms) | fps 2 (40 ms) | fps 4 (80 ms) |
  |---|---|---|---|
  | Student, PyTorch eager (best run) | 9.0 / 12.5 ms | 12.0 / 17.4 ms | 14.5 / 18.7 ms |
  | Student, ORT 1.31, 1 thread (throwaway export) | 5.0 / 11.9 ms | 8.6 / 21.7 ms | 16.3 / 39.2 ms |
  | Mimi streaming encode, HF eager (per 80 ms frame) | 38 / 59 ms | 58 / 97 ms (2 frames) | 76 / 120 ms (4 frames) |

  - The teacher is offline only: 297 ms per second of audio on one thread.
  - The default student is about **0.2 RTF on one core under ORT**. Stage 05
    has to weigh that against the vocoder's share of the ≤ 50%-of-hop compute
    budget, and the cheaper `--channels` variants exist for that.
- **Mimi:** 3 utterances encoded at 12.5 Hz, vocab 2048, exact frame counts.
  Its streaming encode matches offline 100%.

**Open:**
- Run the teacher pipeline on train-clean-100 (now downloading) and label the
  vocoder corpora.
- Get MFA phone alignments for dev-clean and run the probes.
- Train the student (≤ 3 days) and rerun `stream_check.py` and `--cost` on it.
- Run the Mimi kill test (1-week time box).
- Stage 06 note: torch 2.11's default dynamo ONNX exporter needs
  `onnxscript`, which is not in `pyproject.toml`. `dynamo=False` works.
