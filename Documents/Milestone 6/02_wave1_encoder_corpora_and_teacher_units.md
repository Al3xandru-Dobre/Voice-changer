# Stage 02 — Wave 1: Encoder corpora & teacher units

> **Milestone 6** · [Master plan](00_master.md) · [← Stage 01](01_local_preflight_cuda_and_disk_hygiene.md) · [Stage 03 →](03_wave1_content_encoder_training_and_cleanup.md)

**Status:** ☐ not started
**Depends on:** Stage 01 (CUDA venv, `--delete-raw`, `>= 130 GB` free)
**Feeds:** [M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md)

---

## Objective

Download **only** the Wave 1 corpora needed for the content encoder, fit the HuBERT-Base layer-7 `km100` teacher codebook, extract per-utterance `int16` unit labels, and run the Stage 04 unit probes (including the time-boxed Mimi `q0` kill test).

---

## Disk budget for Wave 1

> **Do NOT run `--datasets core`** — that pulls Hi-Fi TTS (`41.4 GB`) and 5 LibriTTS-R splits (`12.7 GB`) at the same time. Pull Wave 1 explicitly by key.

| Dataset key | Archive | Resting on disk (`raw/`, used in place) | Required? |
|---|---|---|---|
| `librispeech-train-clean-100` | `6.39 GB` | `~6.4 GB` | **Required** (teacher k-means + student baseline) |
| `musan` | `11.09 GB` | `~11.1 GB` | **Required** (denoising distillation noise/music) |
| `rirs` | `1.31 GB` | `~1.4 GB` | **Required** (reverb augmentation) |
| `librispeech-train-clean-360` | `23.05 GB` | `~23.1 GB` | **Recommended** (adds 360 h; fits easily in free space) |
| **Wave 1 Subtotal (lean / full)** | `18.8 GB` / `41.8 GB` | **`~19 GB` / `~42 GB`** | Peak during extract + `--delete-archives`: `~30 GB` (lean) or `~65 GB` (full) |
| Extracted teacher units (`int16` `.npy`) | — | **`< 0.25 GB`** | 50 Hz `int16` ≈ `360 KB/h`; 460 h is `~170 MB` |

With `~135 GB` free entering Stage 02, even **full Wave 1 (`460 h` LibriSpeech + MUSAN + RIRs)** leaves **`>= 70 GB` free** at peak and **`~90 GB` free** at rest.

---

## Tasks

- [ ] **1. Fetch Wave 1 core (lean pass first: ~19 GB).**
  ```bash
  cd python_training
  .venv/bin/python prepare_data.py \
      --datasets librispeech-train-clean-100,musan,rirs \
      --delete-archives
  ```
  Verify `df -h .` and `.venv/bin/python prepare_data.py --summary`.

- [ ] **2. Optional but recommended: fetch `librispeech-train-clean-360` (+23 GB).**
  If `df -h .` shows `>= 90 GB` free after step 1, fetch the 360 h split for a substantially stronger student encoder:
  ```bash
  cd python_training
  .venv/bin/python prepare_data.py \
      --datasets librispeech-train-clean-360 \
      --delete-archives
  ```
  *(If you want the fastest possible end-to-end proof of concept first, you can skip `train-clean-360` here and train on `train-clean-100` alone, then add `360` later.)*

- [ ] **3. Commit any newly pinned SHA256 hashes in `python_training/datasets.lock.json`.**
  Check `git diff python_training/datasets.lock.json`; if TOFU pinned any new archive hash, commit the lockfile update.

- [ ] **4. Confirm M2 Stage 03 pitch priors (already done — sanity check only).**
  Stage 03 priors (`logf0`: mean `5.028`, std `0.421`; `log_energy`: mean `-8.486`, std `3.817`) are already baked into `fvml/features.py` and `tests/data/pitch/`. No rerun required unless you intentionally change tracker constants.

- [ ] **5. Fit the HuBERT-Base layer-7 `km100` teacher codebook on `train-clean-100`.**
  *(First run caches `facebook/hubert-base-ls960` under `$FLUXVOICE_DATA/cache/`, `~0.4 GB`.)*
  ```bash
  cd python_training
  .venv/bin/python extract_units.py --fit-kmeans \
      --manifest librispeech-train-clean-100.jsonl \
      --hours 100 --max-frames 500000 --seed 0 \
      --out-dir ../data/units/km100
  ```
  Output: `data/units/km100/km100_centroids.npy` + `km100_meta.json`.

- [ ] **6. Extract `teacher-km100` labels for Wave 1 + eval splits.**
  *(Resumable: rerunning into the same `--out-dir` skips completed utterances.)*
  ```bash
  cd python_training
  for pair in \
      "librispeech-dev-clean.jsonl:dev_clean" \
      "librispeech-test-clean.jsonl:test_clean" \
      "librispeech-train-clean-100.jsonl:tc100"; do
    mf="${pair%%:*}"; tag="${pair##*:}"
    .venv/bin/python extract_units.py \
        --tokenizer teacher-km100 \
        --centroids ../data/units/km100 \
        --manifest "$mf" \
        --out-dir "../data/units/teacher-km100/$tag"
  done
  ```
  If `train-clean-360` was downloaded in step 2:
  ```bash
  cd python_training
  .venv/bin/python extract_units.py \
      --tokenizer teacher-km100 \
      --centroids ../data/units/km100 \
      --manifest librispeech-train-clean-360.jsonl \
      --out-dir ../data/units/teacher-km100/tc360
  ```

- [ ] **7. Run the teacher speaker-ID probe & Mimi `q0` kill test (`dev-clean`).**
  - Extract Mimi `q0` on `dev-clean` (`~5.4 h`, fast on GPU):
    ```bash
    cd python_training
    .venv/bin/python extract_units.py \
        --tokenizer mimi-q0 \
        --manifest librispeech-dev-clean.jsonl \
        --out-dir ../data/units/mimi-q0/dev_clean
    ```
  - Run `probe_units.py` speaker-ID leakage probe on `teacher-km100/tc100` (and phone/PNMI probe on `dev_clean` if precomputed LibriSpeech MFA TextGrids are placed under `data/alignments/dev-clean/`):
    ```bash
    cd python_training
    .venv/bin/python probe_units.py \
        --units ../data/units/teacher-km100/tc100 \
        --speaker-probe --out ../runs/probes/teacher_spk.json
    ```
  - If MFA TextGrids are available, compare `teacher-km100/dev_clean` vs `mimi-q0/dev_clean` with `--alignments`:
    - If Mimi's PNMI ratio vs. teacher is `< 0.80`, **drop Mimi now** and record the decision in [M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md) (saves an entire extra vocoder training run in Stage 04!).

---

## Verification & measurements *(fill in)*

- Disk free after Wave 1 fetch + unit extraction: `___ GB`
- `data/units/km100/km100_centroids.npy` SHA256: `___`
- Teacher extraction throughput (utterances/sec on RTX 4060): `___`
- Teacher speaker-ID probe accuracy (vs. `1%` chance over 100 speakers): `___`
- Mimi `q0` kill decision (dropped / kept + PNMI ratio if MFA run): `___`

---

## Definition of done

- `data/units/km100/km100_centroids.npy` and `km100_meta.json` exist and are backed up.
- `data/units/teacher-km100/{dev_clean,test_clean,tc100}` (and `tc360` if fetched) are 100% labelled with valid `units_meta.json`.
- Teacher speaker-ID probe result and Mimi kill-test outcome are recorded here and in [M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md).
- `/` has `>= 70 GB` free.
