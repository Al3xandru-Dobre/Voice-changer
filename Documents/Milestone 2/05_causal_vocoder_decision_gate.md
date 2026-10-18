# Stage 05 — Causal vocoder & encoder decision gate

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 04](04_content_encoder_candidates.md) · [Stage 06 →](06_stateful_onnx_export.md)

**Status:** ◐ in progress (2026-10-18). All generator, discriminator, trainer, streaming-equivalence, export-smoke, and decision-gate code is implemented, unit-tested and smoke-run. Waiting on Stage 04's teacher labelling + student training and Hi-Fi TTS download to run the full 150k-step vocoder and populate quality metrics (see Verification)
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

- [x] **Model** (`python_training/fvml/vocoder.py`, built from Stage 04's
      `StreamingConv1d` plus `StreamingConvTranspose1d` in `fvml/streaming.py`):
  - [x] Unit embedding (vocab → 128); content at its own frame rate, upsampled
        by repetition to the 20 ms conditioning rate (×1 for 50 Hz students, ×4
        for Mimi); also supports soft-unit fallback (`content_kind="soft"`)
  - [x] Conditioning stack `[logf0, voicing, aperiodicity, energy]` → causal
        pre-net (`ConditioningPreNet`, two `StreamingConv1d` layers)
  - [x] Upsampling ×320 to 16 kHz (`8·5·4·2`, kernels `16·10·8·4`) with causal
        MRF residual blocks (`CausalMRF` / `CausalResBlock1`, kernels `3/7/11`,
        dilations `1/3/5`)
  - [x] FiLM (γ, β from `spk_emb`, identity-centred `x * (1 + γ) + β`) after
        each upsampling stage
  - [x] Speaker table `nn.Embedding(n_speakers, 64)`, whose rows are exported to
        `voices.json` in Stage 06 (`UnitVocoder.voices_json()`; expandable via
        `expand_speakers()` for Stage 10)
- [x] **Training** (`train_unit_vocoder.py`, real implementation):
  - [x] Data: LJSpeech + the chosen Hi-Fi TTS speakers at 16 kHz (excluding
        held-out reference clips from `eval_lists/targets.tsv`). Features from
        Stage 03's causal tracker on each speaker's *own* audio; units =
        teacher km100 ids (or student ids with `--finetune-from`)
  - [x] Losses: LSGAN adversarial + feature matching (weight 2.0) +
        multi-resolution mel L1 (`MultiResolutionMelLoss`, weight 45.0)
  - [x] Discriminators: MPD (`periods=(2,3,5,7,11)`) + multi-resolution STFT
        discriminator (`MultiResolutionDiscriminator`, FFTs `512/1024/2048`)
  - [x] AdamW lr 2e-4, betas (0.8, 0.99), exponential decay 0.999/epoch
  - [x] bf16 autocast, segment 8320 samples (26 frames), batch 16 (~1.74 GiB
        peak on the 8 GB RTX 4060 Laptop)
  - [x] Resumable checkpoints every 30 min (`--resume auto`); TensorBoard
  - [x] Per-speaker F0/energy stats computed (`compute_speaker_stats`) and
        stored with the checkpoint + written to `voices.json`
- [x] **Streaming equivalence** (`stream_check.py --role vocoder`). Generator
      `step()` (1/2/4 frames, state carried) vs. `forward()` on 30 s: max |Δ|
      and a boundary-step check. Passes on random init and on a 30-step CUDA
      smoke checkpoint (see Verification); rerun on the 150k-step checkpoint.
- [ ] **Quick vocoders for the gate**, same architecture, a fixed short schedule
      (e.g. 150k steps): *code ready; waiting on Stage 04 teacher labels +
      Hi-Fi TTS audio.*
  - [ ] **A:** teacher-id vocoder, shared by all student heads
  - [ ] **M:** Mimi-q0 vocoder, only if Mimi survived Stage 04's probe
- [x] **Gate evaluation** (`python_training/gate_report.py`, code done; full
      evaluation run pending trained checkpoints):
  - [x] For every candidate (student L0/L1/L2 at frames_per_step 1/2/4 and Mimi
        if alive), convert the frozen `test_clean_200` list into 2 target voices
        with the full *streaming* chain (`convert_utterance_streaming`). Source
        features are mapped to the target with Stage 03's mapping.
  - [x] Score with `eval_vc.py` (`fvml.evaluate.evaluate`): WER, CER, target
        cosine, source cosine, UTMOS, F0 PCC, RTF.
  - [x] Cost: single-core encoder + vocoder per-step p95 (PyTorch eager + ORT
        1.31 CPU 1-thread).
  - [x] Projected T−B from the master budget formula using the measured compute
        p95.
  - [x] **Export smoke test:** export a random-weight per-step graph of the
        candidate architecture (`dynamo=True`) and run one step in ONNX Runtime
        CPU. A design that can't export can't win.
- [ ] **Decision.** Apply the rule, then record the gate table + decision here and
      in the master's open decisions 4 and 7.
- [ ] **Post-gate fine-tune.** Continue the vocoder on the winning head's ids
      (e.g. 50k steps, `train_unit_vocoder.py --finetune-from`). Re-score and
      confirm the WER stays within the rule.
- [ ] **Controls readiness checks** on the fine-tuned model
      (`gate_report.py --controls-check`; code implemented and tested):
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

| Candidate | Hop (ms) | L | Alg. latency (ms) | Compute p95 PyTorch / ORT (ms) | Projected T−B PyTorch / ORT (ms) | WER | CER | Tgt cos | Src cos | UTMOS | F0 PCC | Export OK | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| student-L0 s1 | 20 | 0 | 20 | 15.4 / **4.5** | 65.4 / **54.5** | — | — | — | — | — | — | ✓ | — |
| student-L1 s1 | 20 | 1 | 40 | 16.7 / **4.9** | 86.7 / **74.9** | — | — | — | — | — | — | ✓ | — |
| student-L2 s1 | 20 | 2 | 60 | 16.5 / **4.6** | 106.5 / **94.6** | — | — | — | — | — | — | ✓ | — |
| student-L1 s2 | 40 | 1 | 60 | 18.7 / **8.2** | 108.7 / **98.2** | — | — | — | — | — | — | ✓ | — |
| mimi-q0 | 80 | 0 | 80 | 32.7 / — | 147.7 / — | — | — | — | — | — | — | ✗ (pending kill test) | — |

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
- Added / extended during implementation:
  - `python_training/fvml/streaming.py` (`StreamingConvTranspose1d` with
    right-trimming `trim_right_ratio = 1.0` and explicit input-context cache).
  - `python_training/stream_check.py` (`--role vocoder`, 30 s streaming vs.
    offline equivalence + chunk-boundary step check).
  - `python_training/probe_units.py` (`--cost-vocoder`).
  - `python_training/tests/test_streaming.py` (21 new transposed-conv tests) and
    `python_training/tests/test_vocoder.py` (11 unit/integration tests,
    registered in `CMakeLists.txt` as `python_training_stage05`).

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

## Verification (2026-10-18 — code, streaming equivalence, export smoke & cost)

- **Generator architecture (`fvml/vocoder.py`):**
  - V2-class `UnitVocoder`: **1.06 M parameters, 0.89 GMAC/s**, 806 ms of past
    receptive field, **80 state tensors / 27 496 floats** (`upsample_rates =
    (8, 5, 4, 2)`, `upsample_kernels = (16, 10, 8, 4)`, `upsample_initial_channel
    = 128`, MRF kernels `3/7/11`, dilations `1/3/5`).
  - `VocoderDiscriminator` (MPD periods `2/3/5/7/11` + MRD FFTs `512/1024/2048`):
    11.06 M parameters.
- **Streaming equivalence (`stream_check.py --role vocoder --seconds 30`):**
  - Random init over 30.0 s (1 500 frames) at `fps` 1, 2, 4: max |Δ| =
    **3.09e-11**, SNR = **125.6 dB**, max boundary step **1.05e-7 ≤ 2.32e-5**
    (reference max step).
  - 30-step CUDA bf16 smoke checkpoint over 30.0 s at `fps` 1, 2, 4: max |Δ| =
    **1.79e-7** (DoD ≤ 1e-4), SNR = **136.4 dB**, max boundary step **0.0532 ≤
    0.3896** (reference max step).
- **Training smoke run (`train_unit_vocoder.py`, RTX 4060 Laptop 8 GB, CUDA bf16):**
  - Batch 16 × 8 320 samples (26 frames): **2.72 it/s, peak GPU memory
    1.74 GiB** (150k steps ≈ 15.3 h, well inside the 8 GB VRAM budget).
  - `--resume auto` verified (step 20 → 30); `--finetune-from` verified with
    student-L1 units and speaker-table expansion (`expand_speakers`).
- **ONNX export smoke test & single-core cost (`gate_report.py --skip-eval`):**
  - All four student candidates (`student-L0/L1/L2 s1`, `student-L1 s2`) export
    cleanly via `torch.onnx.export(..., dynamo=True)`, pass `onnx.checker` full
    check, `fvml.contract.load_onnx`, and `validate_pair`, and match PyTorch
    `step()` in ONNX Runtime 1.31.0 CPU (`intra_op = 1`) with exact unit match
    and vocoder max |Δ| ≤ **4.37e-11**.
  - Single-core CPU per-hop p95 (`encoder + vocoder`):
    - **ORT 1.31.0 (`intra_op = 1`):** **4.5–4.9 ms** at 20 ms hop (`s1`, ≤ 25%
      of hop; encoder 2.7–2.9 ms + vocoder 1.8–2.0 ms), **8.2 ms** at 40 ms hop
      (`s2`, 20% of hop). All `s1` and `s2` student candidates pass the ≤ 50%-of-hop
      compute budget comfortably under ORT.
    - **PyTorch eager (`torch.set_num_threads(1)`):** 15.4–16.7 ms at `s1` (Python
      dispatch across 100+ `StreamingConv1d` modules) and **18.7 ms** at `s2`
      (40 ms hop, within the 20 ms budget even in eager PyTorch).
- **Tests:** 46/46 CTest targets pass (`cmake --build build --config Release &&
  ctest --test-dir build -C Release --output-on-failure`), including
  `python_training_stage05` (`tests/test_vocoder.py`, 11 tests) and the expanded
  `tests/test_streaming.py` (47 tests).
