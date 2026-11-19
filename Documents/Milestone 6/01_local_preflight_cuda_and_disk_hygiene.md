# Stage 01 — Local preflight: CUDA venv & disk hygiene

> **Milestone 6** · [Master plan](00_master.md) · [Stage 02 →](02_wave1_encoder_corpora_and_teacher_units.md)

**Status:** ☐ not started
**Depends on:** nothing
**Estimated scope:** Local environment sync, disk cleanup, and a small Python-only enhancement (`--delete-raw`) in `python_training/`

---

## Objective

Make the local Linux laptop genuinely ready for GPU training and disk-safe dataset staging before downloading a single gigabyte:

1. Restore the locked CUDA 12.8 PyTorch build (`torch 2.11.0+cu128`) in `python_training/.venv` so training runs on the RTX 4060 Laptop instead of silently falling back to CPU.
2. Reclaim immediate wasted disk space (stale non-preset `build-*` trees, already-extracted archives in `data/archives/`, and duplicate 24 kHz raw files).
3. Add a safe `--delete-raw` flag to `prepare_data.py` / `fvml/datasets.py` so non-16 kHz corpora (`ljspeech`, `hifitts`, `libritts-r-*`) delete their raw originals immediately after resampling to `16k/` and writing a verified manifest.

---

## Context & verified starting point (2026-11-19)

- `python_training/.venv/bin/python` currently imports **`torch 2.11.0+cpu`** (`cuda: False`), whereas `python_training/uv.lock` already pins `torch==2.11.0+cu128` and `torchaudio==2.11.0+cu128` for Linux (`sys_platform == 'linux'`).
- `nvidia-smi` confirms **NVIDIA GeForce RTX 4060 Laptop (8188 MiB VRAM)** with driver `595.84` / CUDA `13.2` — fully compatible with `cu128` wheels (requires driver `>= 525`).
- `/` has `133 GB` free. Quick reclaimable waste verified on disk:
  - `~1.9 GB` in 20 non-preset ad-hoc `build-*` directories (keep only the four preset trees from `AGENTS.md` rule 4: `build-linux-gcc`, `build-linux-gcc-ort`, `build-linux-clang`, `build-linux-clang-ort`).
  - `2.0 GB` in `data/archives/` (`dev-clean.tar.gz`, `test-clean.tar.gz`, `dev_clean.tar.gz`), all already verified and extracted.
  - `1.1 GB` in `data/raw/LibriTTS_R/` whose 16 kHz FLACs already exist in `data/16k/libritts_r/`.
- In `python_training/fvml/datasets.py`, `--delete-archives` deletes `data/archives/<file>` after extraction, but `data/raw/<corpus>` is never deleted even when `build_*` resamples every clip into `data/16k/<dataset>/` (`ljspeech`, `hifitts`, `libritts-r-*`, `vctk`). Without `--delete-raw`, extracting 4 Hi-Fi TTS readers in Stage 04 leaves ~16 GB of unneeded 44.1 kHz raw audio sitting beside `data/16k/hifitts/`.

---

## Tasks

- [ ] **1. Restore CUDA PyTorch in `python_training/.venv`.**
  ```bash
  cd python_training
  uv sync --frozen
  .venv/bin/python -c "import torch, torchaudio, onnxruntime as ort; \
    print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), \
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); \
    print('bf16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); \
    print('ort:', ort.__version__)"
  ```
  Expect: `torch: 2.11.0+cu128`, `cuda: True`, `NVIDIA GeForce RTX 4060 Laptop GPU`, `bf16: True`, `ort: 1.31.0`.
  *(If `uv sync --frozen` still resolves a cached CPU wheel, run `uv sync --frozen --reinstall-package torch --reinstall-package torchaudio`.)*

- [ ] **2. Run the existing Python training test suite as a baseline.**
  ```bash
  cd python_training
  .venv/bin/pytest -q
  ```
  Confirm every existing unit test passes on the synced venv before touching any code.

- [ ] **3. Reclaim wasted local disk space (`~5 GB` immediate gain).**
  - Remove only the non-preset ad-hoc `build-*` directories (keep `build-linux-gcc`, `build-linux-gcc-ort`, `build-linux-clang`, `build-linux-clang-ort`).
  - Remove already-extracted archives in `data/archives/*.tar.gz` (their `.state/*.verified.json` and `.extracted.json` stamps and `datasets.lock.json` pins remain intact).
  - Optionally remove `data/raw/LibriTTS_R/` once `data/manifests/libritts-r-dev-clean.jsonl` is confirmed to point exclusively into `16k/libritts_r/`.
  - Check `df -h .` and record free GB.

- [ ] **4. Implement `--delete-raw` in `prepare_data.py` & `fvml/datasets.py`.**
  - Add `delete_raw: bool = False` to `fvml.datasets.Context` and `--delete-raw` to `prepare_data.py`.
  - **Safety invariant:** delete raw files **only** after `write_jsonl(manifest, ...)` succeeds AND **only** when every manifest row's `path` points outside `raw/` (i.e., into `16k/...`). Corpora used in place (`librispeech-*`, `musan`, `rirs`) point into `raw/` and MUST NEVER be touched by `--delete-raw`.
  - Keep `.state/<key>.extracted.json` and `.state/<key>.manifest.json` intact so reruns skip already-prepared datasets without re-downloading.
  - For subset corpora sharing a raw parent (e.g. `LibriTTS_R/<subset>`), remove only that dataset's own subset directory, never sibling subsets.
  - Add unit tests in `python_training/tests/` verifying:
    - resampled corpora (`ljspeech`, `hifitts`, `libritts-r-*`) remove their raw files when `delete_raw=True` while preserving `16k/` + manifest + idempotent rerun;
    - in-place corpora (`librispeech-*`, `musan`, `rirs`) keep their `raw/` files untouched even when `delete_raw=True`.

- [ ] **5. Verify dataset registry and disk report.**
  ```bash
  cd python_training
  .venv/bin/python prepare_data.py --list
  ```
  Confirm free space is `>= 135 GB` and existing eval manifests (`librispeech-dev-clean`, `librispeech-test-clean`, `libritts-r-dev-clean`) show `manifest`.

- [ ] **6. Quick GPU smoke benchmark on RTX 4060 Laptop.**
  Verify `bf16` CUDA kernels work cleanly (no driver/cuDNN errors) via existing smoke paths or pytest CUDA check.

---

## Verification & measurements *(fill in)*

- `df -h .` before / after cleanup: `133 GB` → `___ GB` free
- `torch.__version__` / CUDA device: `___`
- `pytest -q` result: `___`
- CTest `python_training_*` suite (`ctest --preset linux-gcc-training`): `___`

---

## Definition of done

- `python_training/.venv/bin/python` imports `torch 2.11.0+cu128` with `torch.cuda.is_available() == True` and `bf16 == True`.
- `--delete-raw` is implemented in `prepare_data.py` / `fvml/datasets.py` with unit tests proving in-place corpora are never deleted and resampled corpora remain idempotent on rerun.
- `python_training/.venv/bin/pytest` passes completely, and `/` has `>= 130 GB` free.
