# Stage 05 — Decision gate, post-gate fine-tune & stateful ONNX bundle

> **Milestone 6** · [Master plan](00_master.md) · [← Stage 04](04_wave2_vocoder_corpora_and_training.md) · [Stage 06 →](06_live_mode_a_soak_and_own_voice.md)

**Status:** ☐ not started
**Depends on:** Stage 03 (`checkpoints/ce_m2_v1/best.pt`), Stage 04 (`checkpoints/voc_a/best.pt` + `16k/{ljspeech,hifitts}`)
**Closes:** [M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md), [M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md), and real-bundle verification in [M2 / Stage 07](../Milestone%202/07_ort_cpp_runtime.md)

---

## Objective

Run the objective latency-first encoder decision gate on `test_clean_200`, fine-tune the vocoder for 50k steps on the winning student head's actual discrete units, export the first real fixed-shape stateful ONNX bundle (`models/mode_a_v1/`), and verify Python ORT ↔ C++ `fv_vc_file` golden parity.

---

## Gate rules (from [M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md))

A candidate `(student-L{0,1,2}, s{1,2,4})` is **eligible** when *all* hold on `test_clean_200` (averaged over `lj` and `hifitts_6097`):
- `WER <= 10%` absolute **and** `<= best candidate WER + 3 pts`
- `tgt_cos >= best - 0.05`
- `UTMOS >= 3.0`
- Single-core ORT CPU `compute p95 <= 50%` of the hop
- Export smoke test (`dynamo=True`) passes

Among eligible candidates, pick the **lowest projected `T - B`** (Must `<= 100 ms`; Stretch `<= 80 ms`; tie-break on UTMOS).

---

## Tasks

- [ ] **1. Run the Stage 05 decision gate (`gate_report.py`).**
  *(First run caches Whisper `large-v3`, ECAPA-TDNN, and UTMOS22 under `$FLUXVOICE_DATA/cache/`, `~3.5 GB`.)*
  ```bash
  cd python_training
  .venv/bin/python gate_report.py \
      --encoder-checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --vocoder-checkpoint ../checkpoints/voc_a/best.pt \
      --voices lj,hifitts_6097 \
      --out ../runs/gate_m2_v1
  ```
  Inspect `runs/gate_m2_v1/gate_report.md` and note the winning head (call it `WIN_HEAD`, e.g. `student-L1`, and its lookahead `WIN_L`, e.g. `1`) and winning hop (`s1`/`s2`).
  *(If every candidate misses WER/UTMOS, consult the fallback order in [M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md#decision-gate): verify `train-clean-360` was included in Stage 03, check soft-unit fallback, or train a wider generator.)*

- [ ] **2. Extract the winning student head's units on `ljspeech` and `hifitts`.**
  ```bash
  cd python_training
  WIN_HEAD=student-L1   # replace with the gate winner from step 1
  .venv/bin/python extract_units.py \
      --tokenizer "$WIN_HEAD" \
      --checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --manifest ljspeech.jsonl \
      --out-dir "../data/units/$WIN_HEAD/lj"

  .venv/bin/python extract_units.py \
      --tokenizer "$WIN_HEAD" \
      --checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --manifest hifitts.jsonl \
      --out-dir "../data/units/$WIN_HEAD/hifitts"
  ```

- [ ] **3. Run the 50k-step post-gate vocoder fine-tune on the winning student's units (`voc_ft`).**
  Align `--lookahead-frames` automatically from `units_meta.json` (or pass explicitly matching `WIN_L`):
  ```bash
  cd python_training
  WIN_HEAD=student-L1   # replace with the gate winner
  .venv/bin/python train_unit_vocoder.py \
      --train-manifest ljspeech.jsonl --train-manifest hifitts.jsonl \
      --train-units "../data/units/$WIN_HEAD/lj" \
      --train-units "../data/units/$WIN_HEAD/hifitts" \
      --finetune-from ../checkpoints/voc_a/best.pt \
      --steps 50000 --ckpt-minutes 30 \
      --run-name voc_ft --resume auto
  ```

- [ ] **4. Re-score the fine-tuned vocoder + run controls readiness checks.**
  ```bash
  cd python_training
  .venv/bin/python gate_report.py \
      --encoder-checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --vocoder-checkpoint ../checkpoints/voc_ft/best.pt \
      --voices lj,hifitts_6097 \
      --controls-check \
      --out ../runs/gate_m2_v1_ft
  ```
  Confirm the winner still satisfies every gate criterion and `+12 st` pitch shift gives median output-F0 ratio `2.0 ± 3%`.

- [ ] **5. Export the fixed-shape stateful ONNX bundle (`models/mode_a_v1/`).**
  Export both roles (`s1`, `s2`, `s4`) and run the built-in 60 s ORT-vs-PyTorch parity + single-core cost + golden I/O generation:
  ```bash
  cd python_training
  WIN_HEAD=student-L1   # replace with the gate winner
  BUNDLE=../models/mode_a_v1

  .venv/bin/python export_stateful_onnx.py \
      --role encoder \
      --checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --head "$WIN_HEAD" \
      --license "CC BY 4.0 (LibriSpeech + MUSAN) / Apache 2.0 (RIRS_NOISES) / MIT (HuBERT teacher)" \
      --bundle "$BUNDLE"

  .venv/bin/python export_stateful_onnx.py \
      --role vocoder \
      --checkpoint ../checkpoints/voc_ft/best.pt \
      --gate-report ../runs/gate_m2_v1_ft/gate_report.json \
      --bundle "$BUNDLE"
  ```
  Verify `export_stateful_onnx.py` exits `0` and check `models/mode_a_v1/export_report.json`:
  - Encoder ORT vs PyTorch over 60 s: argmax agreement `>= 99.9%`
  - Vocoder ORT vs PyTorch over 60 s: SNR `>= 50 dB`
  - `models/mode_a_v1/{voices.json,MODEL_CARD.md,golden/}` all generated cleanly

- [ ] **6. Verify C++ `fv_vc_file` bit/signal parity against `models/mode_a_v1/golden/`.**
  Ensure the ORT C++ build is up to date (`cmake --workflow --preset linux-gcc-ort`), then run `fv_vc_file --no-output-chain` on the exported golden input WAV and compare with `parity.py`:
  ```bash
  ./build-linux-gcc-ort/fv_vc_file \
      --model-dir models/mode_a_v1 \
      --no-output-chain \
      --dump-units /tmp/cpp_golden_units.npy \
      models/mode_a_v1/golden/input.wav \
      /tmp/cpp_golden_out.wav

  cd python_training
  .venv/bin/python parity.py \
      ../models/mode_a_v1/golden/audio_s1.wav \
      /tmp/cpp_golden_out.wav \
      --min-snr 50 --trim
  ```
  *(Use the golden filename matching the exported default variant in `models/mode_a_v1/golden/`.)*

- [ ] **7. Record results back into Milestone 2 docs.**
  Fill in the gate table & winning decision in [M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md), update [M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md) parity numbers, and tick Stages 05 and 06 in [M2 / 00_master.md](../Milestone%202/00_master.md).

---

## Verification & measurements *(fill in)*

- Gate winner (`head` / `frames_per_step` / projected `T - B` ms): `___`
- Post-fine-tune gate metrics (`WER` / `CER` / `tgt_cos` / `src_cos` / `UTMOS` / `F0 PCC`): `___`
- Exported bundle size (`du -sh models/mode_a_v1`): `___ MB`
- Export parity — encoder argmax `%` / vocoder SNR dB / exporter (`dynamo` vs fallback): `___`
- C++ `fv_vc_file` vs Python ORT golden SNR (dB) and unit match `%`: `___`

---

## Definition of done

- `runs/gate_m2_v1_ft/gate_report.json` shows an eligible winner passing all M2 Stage 05 thresholds.
- `export_stateful_onnx.py` exits `0` for both `encoder` and `vocoder` into `models/mode_a_v1/` with `voices.json`, `MODEL_CARD.md`, and `golden/`.
- `build-linux-gcc-ort/fv_vc_file --no-output-chain` matches `models/mode_a_v1/golden/` within parity thresholds (`SNR >= 50 dB` and unit parity).
- Milestone 2 Stage 05 and Stage 06 docs are updated with the measured tables.
