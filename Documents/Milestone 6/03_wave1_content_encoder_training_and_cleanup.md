# Stage 03 — Wave 1: Content encoder training & disk reclaim

> **Milestone 6** · [Master plan](00_master.md) · [← Stage 02](02_wave1_encoder_corpora_and_teacher_units.md) · [Stage 04 →](04_wave2_vocoder_corpora_and_training.md)

**Status:** ☐ not started
**Depends on:** Stage 02 (Wave 1 manifests + `teacher-km100` unit labels)
**Closes:** [M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md)

---

## Objective

Train the 50 Hz causal content encoder (`L ∈ {0, 1, 2}` delay heads sharing one causal conv trunk) on the local RTX 4060 Laptop, prove streaming-vs-offline equivalence and single-core CPU cost, and **reclaim Wave 1 training audio** so the drive has `>= 115 GB` free before Wave 2 downloads Hi-Fi TTS.

---

## Training recipe & hardware pacing (RTX 4060 Laptop 8 GB)

- `train_content_encoder.py` uses `bf16` autocast by default, `batch_size=32`, `crop_seconds=2.0`, `steps=300_000`, `lr=5e-4`, cosine schedule with `warmup=2000`, and denoising distillation (`MUSAN` noise `p=0.6` + `RIR` reverb `p=0.4` + EQ + gain on input only; clean teacher labels).
- Checkpoints write to `checkpoints/ce_m2_v1/last.pt` every 30 minutes (`--ckpt-minutes 30`) and `best.pt` on mean validation accuracy (`--val-every 2000`).
- **Pause/resume anytime:** hit `Ctrl+C` cleanly whenever you need the laptop, then re-run the identical command with `--resume auto`.
- **Two-gear option:**
  - *Quick proof run (`100k` steps)* first if you want to confirm convergence and inspect curves in TensorBoard (`tensorboard --logdir runs/ce_m2_v1`) before committing a full `300k`-step run.
  - *Full run (`300k` steps)* on `tc100` + `tc360` (repeatable `--train-manifest` / `--train-units`).

---

## Tasks

- [ ] **1. Run a 2,000-step thermal & throughput check first.**
  ```bash
  cd python_training
  .venv/bin/python train_content_encoder.py \
      --train-manifest librispeech-train-clean-100.jsonl \
      --train-units ../data/units/teacher-km100/tc100 \
      --val-manifest librispeech-dev-clean.jsonl \
      --val-units ../data/units/teacher-km100/dev_clean \
      --musan-manifest musan.jsonl \
      --rir-manifest rirs.jsonl \
      --steps 2000 --val-every 1000 --log-every 100 \
      --run-name ce_m2_smoke
  ```
  Check:
  - Peak GPU VRAM stays well under `8 GiB` (if OOM ever appears, lower `--batch-size 16`).
  - Record steps/sec to estimate total wall time for `150k–300k` steps.
  - Check GPU temperature (`nvidia-smi`); remove `checkpoints/ce_m2_smoke` once verified.

- [ ] **2. Launch the full resumable student training run.**
  *(Omit the second `--train-manifest` / `--train-units` pair if you skipped `train-clean-360` in Stage 02.)*
  ```bash
  cd python_training
  .venv/bin/python train_content_encoder.py \
      --train-manifest librispeech-train-clean-100.jsonl \
      --train-manifest librispeech-train-clean-360.jsonl \
      --train-units ../data/units/teacher-km100/tc100 \
      --train-units ../data/units/teacher-km100/tc360 \
      --val-manifest librispeech-dev-clean.jsonl \
      --val-units ../data/units/teacher-km100/dev_clean \
      --musan-manifest musan.jsonl \
      --rir-manifest rirs.jsonl \
      --steps 300000 --val-every 2000 --ckpt-minutes 30 \
      --run-name ce_m2_v1 --resume auto
  ```
  Monitor via `python_training/.venv/bin/tensorboard --logdir runs/ce_m2_v1`. Watch per-head validation accuracy (`L0`, `L1`, `L2`) plateau; stop early only once validation accuracy has clearly converged.

- [ ] **3. Run mandatory post-training verification on `checkpoints/ce_m2_v1/best.pt`.**
  - **Streaming equivalence** (60 s of `dev-clean`, `fps ∈ {1, 2, 4}`):
    ```bash
    cd python_training
    .venv/bin/python stream_check.py \
        --checkpoint ../checkpoints/ce_m2_v1/best.pt
    ```
    Must pass: `max |Δlogit| <= 1e-5` and argmax agreement `>= 99.9%` on all three heads (`L0`, `L1`, `L2`).
  - **Single-core CPU cost:**
    ```bash
    cd python_training
    .venv/bin/python probe_units.py --cost \
        --checkpoint ../checkpoints/ce_m2_v1/best.pt \
        --out ../runs/probes/ce_m2_v1_cost.json
    ```
  - **Student unit probe (`dev-clean` / `tc100`):** extract `student-L0`, `student-L1`, `student-L2` on `dev-clean` and run `probe_units.py` to compare against the teacher baseline from Stage 02.

- [ ] **4. Back up Wave 1 outputs before deleting any raw audio.**
  Confirm the following small, irreplaceable artifacts exist and back them up (external USB drive or a free private Hugging Face Hub repo — see Stage 07):
  - `checkpoints/ce_m2_v1/best.pt` and `final.pt` (`< 100 MB`)
  - `data/units/km100/km100_centroids.npy` and `km100_meta.json` (`< 1 MB`)
  - `runs/ce_m2_v1/` and `runs/probes/`

- [ ] **5. Reclaim Wave 1 raw audio for Wave 2 (`>= 115 GB` free target).**
  Once `checkpoints/ce_m2_v1/best.pt` passes step 3, the vocoder (Stage 04) does **not** read LibriSpeech training audio, MUSAN, or RIRs.

  | Path | Action | Space freed |
  |---|---|---|
  | `data/raw/LibriSpeech/train-clean-360/` + `.state/librispeech-train-clean-360.*` | **Delete** | `~23.1 GB` |
  | `data/raw/LibriSpeech/train-clean-100/` + `.state/librispeech-train-clean-100.*` | **Delete** *(optional if already `>= 100 GB` free)* | `~6.4 GB` |
  | `data/raw/musan/` + `.state/musan.*` | **Delete** *(note: keep `eval_lists/s2ut/silence_hallucination_20.tsv` untouched — it is committed in git; only delete MUSAN if disk is tight)* | `~11.1 GB` |
  | `data/raw/RIRS_NOISES/` + `.state/rirs.*` | **Delete** *(optional)* | `~1.4 GB` |
  | **`data/raw/LibriSpeech/dev-clean/` & `test-clean/`** | **KEEP ALWAYS** (needed by `stream_check.py`, `gate_report.py`, `export_stateful_onnx.py`) | — |
  | **`data/units/km100/`** | **KEEP ALWAYS** (needed to label Wave 2 `ljspeech` + `hifitts`!) | — |
  | **`checkpoints/ce_m2_v1/`** | **KEEP ALWAYS** | — |

  Verify `df -h .` reports **`>= 110 GB` free** before starting Stage 04.

---

## Verification & measurements *(fill in)*

- Final step / wall-clock hours / peak VRAM: `___ steps` / `___ h` / `___ GiB`
- Held-out `dev-clean` top-1 / top-5 accuracy — `L0`: `___` · `L1`: `___` · `L2`: `___`
- `stream_check.py` max `|Δlogit|` & argmax agreement: `___`
- Single-core CPU cost p95 (`s1` / `s2` / `s4`): `___ ms`
- Disk free after Wave 1 cleanup: `___ GB`

---

## Definition of done

- `checkpoints/ce_m2_v1/best.pt` is trained to convergence and passes `stream_check.py` (`max |Δlogit| <= 1e-5`, argmax `>= 99.9%`).
- [M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md) verification section is updated with the trained student numbers and marked `☑ done`.
- `data/units/km100/` and `dev-clean` / `test-clean` remain intact, and `/` has `>= 100 GB` free for Wave 2.
