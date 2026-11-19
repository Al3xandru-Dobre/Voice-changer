# Stage 04 — Wave 2: Vocoder corpora & 150k causal vocoder

> **Milestone 6** · [Master plan](00_master.md) · [← Stage 03](03_wave1_content_encoder_training_and_cleanup.md) · [Stage 05 →](05_decision_gate_finetune_and_onnx_bundle.md)

**Status:** ☐ not started
**Depends on:** Stage 01 (`--delete-raw`), Stage 02 (`data/units/km100/`), Stage 03 (`>= 100 GB` free disk)
**Feeds:** [M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md)

---

## Objective

Download and resample the Wave 2 multi-speaker studio target corpora (`ljspeech` + 4 default `hifitts` readers) one at a time using `--delete-archives --delete-raw`, label them with the frozen `teacher-km100` codebook, and train the 150k-step causal, stateful, multi-speaker Unit-HiFi-GAN vocoder on the RTX 4060 Laptop.

---

## Disk budget for Wave 2

| Step | Download / Archive | Peak transient disk | Resting (`16k/` 24-bit FLAC) |
|---|---|---|---|
| 1. `ljspeech` (1 speaker, ~24 h) | `2.75 GB` | `~7 GB` | **`~1.5 GB`** |
| 2. `hifitts` (4 readers: `92,8051,6097,9017`, ~115 h) | `41.35 GB` | `~62 GB` *(41.4 GB archive + ~16 GB 44.1 kHz raw + ~6–8 GB 16k)* | **`~6–8 GB`** |
| 3. Teacher units (`lj` + `hifitts`) | — | `< 0.1 GB` | **`< 0.1 GB`** |
| **Wave 2 Total** | `44.1 GB` | **`~62 GB` peak** | **`~8–10 GB` resting** |

> **Why Stage 01's `--delete-raw` matters here:** without deleting the archive and the 44.1 kHz raw FLACs after resampling, `hifitts` alone holds `~65 GB` permanently. With `--delete-archives --delete-raw`, all 5 studio target voices (`lj` + 4 `hifitts` readers) settle at **`~8–10 GB` total** on disk!

---

## Tasks

- [ ] **1. Verify free disk before pulling Hi-Fi TTS.**
  ```bash
  df -h .
  ```
  Confirm **`>= 90 GB` free** before fetching `hifitts`.

- [ ] **2. Prepare `ljspeech` first (fast sanity check for `--delete-archives --delete-raw`).**
  ```bash
  cd python_training
  .venv/bin/python prepare_data.py \
      --datasets ljspeech \
      --delete-archives --delete-raw
  ```
  Verify `data/manifests/ljspeech.jsonl` exists, `data/16k/ljspeech/` is populated, and `data/archives/LJSpeech-1.1.tar.bz2` + `data/raw/LJSpeech-1.1/` were removed.

- [ ] **3. Prepare `hifitts` (default 4 readers: `92,8051,6097,9017`).**
  ```bash
  cd python_training
  .venv/bin/python prepare_data.py \
      --datasets hifitts \
      --delete-archives --delete-raw
  ```
  Commit any new SHA256 pin recorded in `python_training/datasets.lock.json`.

- [ ] **4. Verify all held-out target clips in `eval_lists/targets.tsv` resolve on disk.**
  ```bash
  cd python_training
  .venv/bin/python -c "from fvml import evallist, paths; \
    rows = evallist.read_targets(paths.EVAL_LISTS / 'targets.tsv'); \
    missing = [r['path'] for r in rows if not paths.resolve(r['path']).is_file()]; \
    print(f'targets: {len(rows)} rows, {len(missing)} missing'); \
    assert not missing, missing[:5]"
  ```
  Expect: `100 rows, 0 missing` (`20` clips each for `lj`, `hifitts_92`, `hifitts_8051`, `hifitts_6097`, `hifitts_9017`).

- [ ] **5. Extract `teacher-km100` unit labels for `ljspeech` and `hifitts`.**
  *(Must use the exact same `data/units/km100/` centroids fitted in Stage 02!)*
  ```bash
  cd python_training
  .venv/bin/python extract_units.py \
      --tokenizer teacher-km100 \
      --centroids ../data/units/km100 \
      --manifest ljspeech.jsonl \
      --out-dir ../data/units/teacher-km100/lj

  .venv/bin/python extract_units.py \
      --tokenizer teacher-km100 \
      --centroids ../data/units/km100 \
      --manifest hifitts.jsonl \
      --out-dir ../data/units/teacher-km100/hifitts
  ```

- [ ] **6. Run a short 1,000-step vocoder sanity check on the RTX 4060 Laptop.**
  Confirm `bf16` peak VRAM (~1.7–3.5 GiB at `segment_samples=12800`, `batch_size=16`), finite generator/discriminator losses, and steps/sec:
  ```bash
  cd python_training
  .venv/bin/python train_unit_vocoder.py \
      --train-manifest ljspeech.jsonl --train-manifest hifitts.jsonl \
      --train-units ../data/units/teacher-km100/lj \
      --train-units ../data/units/teacher-km100/hifitts \
      --steps 1000 --log-every 100 --run-name voc_a_smoke
  ```
  *(If VRAM pressure appears on 8 GB, reduce `--segment-samples 9600` before lowering `--batch-size`, per [M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md#implementation-notes).)* Delete `checkpoints/voc_a_smoke` once verified.

- [ ] **7. Train the 150k-step multi-speaker causal Unit-HiFi-GAN gate vocoder (`voc_a`).**
  *(Held-out clips in `eval_lists/targets.tsv` are excluded automatically via `--exclude-targets` default. Resumable every 30 minutes via `--resume auto`.)*
  ```bash
  cd python_training
  .venv/bin/python train_unit_vocoder.py \
      --train-manifest ljspeech.jsonl --train-manifest hifitts.jsonl \
      --train-units ../data/units/teacher-km100/lj \
      --train-units ../data/units/teacher-km100/hifitts \
      --steps 150000 --ckpt-minutes 30 \
      --run-name voc_a --resume auto
  ```
  Monitor audio samples and mel L1 / discriminator losses in TensorBoard (`python_training/.venv/bin/tensorboard --logdir runs/voc_a`).

- [ ] **8. Verify vocoder streaming equivalence & teacher-unit resynthesis quality.**
  ```bash
  cd python_training
  .venv/bin/python stream_check.py --role vocoder --seconds 30 \
      --checkpoint ../checkpoints/voc_a/best.pt
  ```
  Must pass: `max |Δ| <= 1e-4` and no chunk-boundary discontinuity exceeding the reference signal's own max sample-to-sample slope.
  *(Check held-out resynthesis UTMOS ≥ 3.5 before entering the Stage 05 gate; if UTMOS is still climbing steeply at 150k steps, continue training before gating.)*

---

## Verification & measurements *(fill in)*

- Disk free after `ljspeech` + `hifitts` prep: `___ GB`
- `voc_a` final step / wall-clock hours / peak VRAM: `___ steps` / `___ h` / `___ GiB`
- `stream_check.py --role vocoder` max `|Δ|` and SNR (dB): `___`
- Held-out resynthesis UTMOS (target `>= 3.5`): `___`

---

## Definition of done

- `ljspeech` and 4 `hifitts` speakers are resampled into `data/16k/`, all 100 rows of `eval_lists/targets.tsv` resolve on disk, and `data/units/teacher-km100/{lj,hifitts}` are extracted.
- `checkpoints/voc_a/best.pt` (150k steps) passes `stream_check.py --role vocoder` (`max |Δ| <= 1e-4`) and reaches held-out resynthesis UTMOS `>= 3.5`.
- `checkpoints/voc_a/` and `data/16k/{ljspeech,hifitts}` are kept intact for Stage 05's post-gate fine-tune; `/` remains `>= 80 GB` free.
