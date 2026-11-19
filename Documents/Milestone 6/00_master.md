# Milestone 6 — Local-First Dataset Execution & Model Training Runbook

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md) ·
> Dataset plan: [DATASET/README.md](../../DATASET/README.md) ·
> Closes ML training gaps in: [Milestone 2](../Milestone%202/00_master.md) ·
> Bridges to: [Milestone 3](../Milestone%203/00_master.md)

**Status:** ☐ not started — execution runbook written 2026-11-19
**Goal:** *Go from today's local Linux laptop (`133 GB` free NVMe, RTX 4060 Laptop 8 GB VRAM) to a fully trained, gated, parity-verified Mode A voice-conversion ONNX bundle running live in C++ — without buying cloud storage or blowing the local disk budget — and then graduate cleanly to cheap/free cloud compute for `DATASET/README.md` P1/P2 scale-up and Milestone 3 S2UT.*

---

## Why this milestone

Milestones 2 and 3 already have their Python training, evaluation, export and C++ ONNX Runtime code implemented and unit/fixture-tested, **but zero trained neural bundles exist in `models/`**. The remaining bottleneck is operational:

- **Disk ceiling:** the local root partition (`/dev/nvme0n1p7`) has **`133 GB` free**, while `prepare_data.py --datasets core` currently queues `~98 GB` of archives (LibriSpeech + LJSpeech + Hi-Fi TTS + MUSAN + RIRs + 5 LibriTTS-R splits) and `fvml/datasets.py` keeps `data/raw/` originals alongside resampled `data/16k/` FLACs. Running `--datasets core` blindly would fill the drive mid-download.
- **Environment drift:** `python_training/.venv` currently has **`torch 2.11.0+cpu`** (`torch.cuda.is_available() == False`) even though `python_training/uv.lock` pins `torch 2.11.0+cu128` and the host has an RTX 4060 Laptop with driver `595.84` (CUDA `13.2`).
- **Scope creep trap:** `DATASET/README.md` Phases P1–P3 (`~600+ GB`) are proposals whose `build_*` loaders are not wired into `fvml/datasets.py` yet, and Milestone 3 S2UT data prep still has two open blockers (`BUG-027` archive plumbing and the `europarl-st` CC BY-NC 4.0 licence mismatch).

Splitting Mode A training into **two disk-budgeted local waves** (Wave 1: encoder corpora → train student → reclaim audio; Wave 2: vocoder corpora → train & fine-tune vocoder → export ONNX) fits comfortably inside `133 GB` free space on the RTX 4060 Laptop at **\$0 cost**, produces the first real shipped bundle, and establishes clean artifact backups before any cloud scale-up.

---

## Verified starting point (2026-11-19)

| Item | State today | Notes / action |
|---|---|---|
| Local disk (`/`) | `133 GB` free of `938 GB` (`86%` used) | Target `>= 20 GB` free headroom at every step |
| GPU & RAM | NVIDIA GeForce RTX 4060 Laptop (`8 GiB` VRAM, driver `595.84`), `31 GiB` RAM | Stages 04–05 are already sized for 8 GB VRAM (`bf16` autocast; vocoder peaks at `1.74 GiB`) |
| Python training env | `python_training/.venv` has **`torch 2.11.0+cpu`** (`cuda: False`) | Re-sync with `cd python_training && uv sync --frozen` (Stage 01) |
| Existing `data/` | `5.2 GB` (`archives/` `2.0 GB`, `raw/` `2.2 GB`, `16k/` `1.1 GB`) | Has `librispeech-{dev,test}-clean` and `libritts-r-dev-clean`; archives + duplicate `raw/LibriTTS_R` can be pruned |
| Ad-hoc `build-*` trees | `2.8 GB` across 24 directories (only 4 are preset trees) | Prune the 20 stale non-preset directories in Stage 01 (`~1.9 GB` back) |
| C++ ORT runtime | `build-linux-gcc-ort/` & `build-linux-clang-ort/` built; `third_party/onnxruntime/` present | Ready for `fv_vc_file` golden check (Stage 05) and `fluxvoice --vc` live soak (Stage 06) |
| Eval lists & priors | `eval_lists/test_clean_200.tsv` & `targets.tsv` committed; Stage 03 YIN priors already fitted on `train-clean-100` | No need to redo Stage 03 pitch priors (`logf0`: 5.028 / 0.421; `log_energy`: -8.486 / 3.817) |
| Mode A code | M2 Stages 01–10 code implemented & unit-tested | Waiting only on trained checkpoints + gate report + real bundle |
| Mode B / M3 blockers | `BUG-027` open (`s2ut_data.py` archive plumbing); `europarl-st` mislabelled CC BY 4.0 instead of CC BY-NC 4.0 | Must be resolved before cloud M3 data prep (Stage 07) |

---

## Disk-budget architecture (why Mode A fits in 133 GB)

```text
Start: ~133 GB free  ──(Stage 01: uv cu128 + prune stale builds/archives + add --delete-raw)──► ~135 GB free
                                                                                                    │
  WAVE 1 (Encoder)                                                                                  ▼
  Stage 02: fetch train-clean-100 (6.4 GB) + musan (11.1 GB) + rirs (1.3 GB)          Peak: ~32 GB  │
            [optional: + train-clean-360 (+23.1 GB) -> resting ~42 GB]                Rest: ~19–42 GB
            fit km100 + extract teacher units (.npy < 200 MB)                                       │
  Stage 03: train causal 50 Hz student (L0/L1/L2) -> stream_check + cost                            │
            keep checkpoints/, units/, eval splits; DELETE large Wave 1 raw audio                   ▼
                                                                                      Free: >= 115 GB
  WAVE 2 (Vocoder)                                                                                  │
  Stage 04: fetch ljspeech (2.8 GB -> ~1.3 GB 16k) with --delete-archives --delete-raw              │
            fetch hifitts  (41.4 GB archive, 4 readers -> ~6–8 GB 16k)                Peak: ~62 GB  │
            extract teacher units for lj + hifitts -> train 150k causal vocoder       Rest: ~12 GB  │
  Stage 05: gate_report --controls-check -> extract winning student units                           │
            -> 50k vocoder fine-tune -> export_stateful_onnx (models/<bundle>/)                     │
            -> C++ fv_vc_file + parity.py golden check                                              ▼
  Stage 06: live PipeWire latency + 30-min soak + optional own-voice ("me")           Rest: ~15–20 GB
  Stage 07: push tiny artifacts to private HF Hub (free) -> cloud scale-up (P1/P2 + M3 S2UT)
```

> **Rule of thumb:** never pass `--datasets core` locally. Always pass explicit dataset keys with `--delete-archives --delete-raw` so peak disk stays under `~65 GB` (leaving `>= 65 GB` safety margin at all times).

---

## Placement & dependencies

```text
M6 Stage 01 (CUDA venv + --delete-raw + disk cleanup)
      │
      ▼
M6 Stage 02 (Wave 1 data + km100 teacher units + Mimi probe)   ──► closes M2 Stage 04 prep
      │
      ▼
M6 Stage 03 (Student training + stream_check + Wave 1 reclaim) ──► closes M2 Stage 04
      │
      ▼
M6 Stage 04 (Wave 2 lj/hifitts + 150k causal vocoder)          ──► advances M2 Stage 05
      │
      ▼
M6 Stage 05 (Gate + 50k fine-tune + ONNX export + C++ parity)  ──► closes M2 Stages 05, 06, 07(real)
      │
      ▼
M6 Stage 06 (PipeWire latency + 30-min soak + optional "me")   ──► closes M2 Stages 08–11
      │
      ▼
M6 Stage 07 (BUG-027 & Europarl-ST fix + HF sync + Cloud M3)   ──► unblocks M3 training & P1/P2
```

---

## Scope

**In scope**
- Restoring CUDA PyTorch (`cu128`) in `python_training/.venv` and verifying RTX 4060 Laptop `bf16` throughput
- Adding safe `--delete-raw` cleanup (with unit tests) to `prepare_data.py` / `fvml/datasets.py` and reclaiming stale build/archive disk space
- Executing Mode A training end-to-end on the local laptop in two disk-safe waves (`Wave 1` encoder, `Wave 2` vocoder)
- Running the Stage 05 decision gate, post-gate vocoder fine-tune, stateful ONNX bundle export, and C++ `fv_vc_file` golden verification
- Live Linux PipeWire latency measurement, 30-minute soak, and optional Stage 10 own-voice ("me") fine-tune
- Backing up portable training artifacts (`checkpoints/`, `units/`, `datasets.lock.json`, `models/<bundle>/`) and defining the exact low-cost/free cloud escalation playbook for `DATASET/README.md` P1/P2 and Milestone 3 S2UT

**Out of scope**
- Changing model architectures, ONNX contract schemas, or C++ real-time DSP (those belong in Milestones 2–5)
- Downloading `DATASET/README.md` P1–P3 corpora locally before P0 passes its gate
- Training Milestone 3 S2UT locally on the 8 GB RTX 4060 Laptop (deferred to Stage 07 cloud handoff)

---

## Stages

| # | Stage | Disk peak / rest | GPU time (RTX 4060) | Closes / feeds | Status |
|---|---|---|---|---|---|
| 01 | [Local preflight: CUDA venv & disk hygiene](01_local_preflight_cuda_and_disk_hygiene.md) | `~135 GB` free | `< 5 min` | Unblocks all local ML runs | ☐ |
| 02 | [Wave 1: Encoder corpora & teacher units](02_wave1_encoder_corpora_and_teacher_units.md) | Peak `~32–50 GB` / Rest `~19–42 GB` | `1–4 h` | M2 Stage 04 teacher + probes | ☐ |
| 03 | [Wave 1: Content encoder training & disk reclaim](03_wave1_content_encoder_training_and_cleanup.md) | Reclaims down to `~5 GB` | `12–48 h` (resumable) | M2 Stage 04 student | ☐ |
| 04 | [Wave 2: Vocoder corpora & 150k causal vocoder](04_wave2_vocoder_corpora_and_training.md) | Peak `~62 GB` / Rest `~12 GB` | `24–48 h` (resumable) | M2 Stage 05 quick vocoder | ☐ |
| 05 | [Decision gate, fine-tune & stateful ONNX bundle](05_decision_gate_finetune_and_onnx_bundle.md) | Rest `~15 GB` | `8–16 h` | M2 Stages 05, 06, 07 (real) | ☐ |
| 06 | [Live Mode A verification, soak & own-voice ("me")](06_live_mode_a_soak_and_own_voice.md) | `+1–2 GB` | `0 h` (or `~6–10 h` for "me") | M2 Stages 08–11 exit | ☐ |
| 07 | [Cloud scale-up & Milestone 3 S2UT handoff](07_cloud_scaleup_and_m3_handoff.md) | `0 GB` local audio | Cloud (\$0–\$15) | P1/P2 corpora + M3 training | ☐ |

---

## Acceptance criteria

- [ ] `python_training/.venv/bin/python` reports `torch 2.11.0+cu128` and `cuda: True` on the RTX 4060 Laptop, and `/` stays `>= 20 GB` free throughout every stage
- [ ] `prepare_data.py --delete-archives --delete-raw` is implemented, tested, and verified not to re-download or break manifest generation
- [ ] Trained Mode A causal content encoder (`best.pt`) and fine-tuned multi-speaker causal Unit-HiFi-GAN (`best.pt`) pass `stream_check.py` and the M2 Stage 05 decision gate (`WER <= 10%`, `UTMOS >= 3.0`, `compute p95 <= 50%` of hop, projected `T - B <= 100 ms`)
- [ ] Exported stateful ONNX bundle (`models/<bundle>/`) passes `export_stateful_onnx.py` parity (`>= 99.9%` encoder unit agreement, `>= 50 dB` vocoder SNR) and matches C++ `fv_vc_file --no-output-chain` against `golden/`
- [ ] Live Linux PipeWire test (`tools/pw_latency_check.sh --modes vc --marker speech`) meets `T - B <= 100 ms`, and the 30-minute soak passes `tools/analyze_soak.py --vc` with zero deadline misses after warm-up
- [ ] `datasets.lock.json` is updated with every newly verified archive SHA256, and portable artifacts (`checkpoints/`, `units/`, `models/<bundle>/`, gate/export reports) are backed up off-disk before deleting any large raw corpora
- [ ] Before any cloud spend on Milestone 3, `BUG-027` and the `europarl-st` CC BY-NC 4.0 licence misclassification are resolved

---

## Key constraints

1. **Never run `prepare_data.py --datasets core` locally.** That group pulls `~98 GB` of archives at once (including 5 LibriTTS-R splits). Always specify explicit dataset keys per wave.
2. **Always keep the evaluation corpora intact.** Never delete `data/raw/LibriSpeech/dev-clean`, `data/raw/LibriSpeech/test-clean`, `data/manifests/`, `data/units/`, or `checkpoints/` during Wave 1 cleanup.
3. **Every long training run must be resumable.** Always pass `--run-name <fixed> --resume auto --ckpt-minutes 30` so laptop sleep, thermal pauses, or power interruptions lose at most 30 minutes of compute.
4. **Never commit audio, private voices, checkpoints, or ONNX weights.** All live under git-ignored `/data/`, `/checkpoints/`, `/runs/`, and `models/<bundle>/`. Commit only code, `python_training/datasets.lock.json`, and updated milestone docs/tables.
5. **Respect laptop thermals.** Keep the laptop on AC power with elevated airflow, monitor `nvidia-smi` temperatures during overnight runs, and pause cleanly with `Ctrl+C` (next run resumes from `last.pt`).
6. **Freeze cloud scope until Mode A ships locally.** Only escalate to cloud compute (Stage 07) once local Mode A proves the end-to-end pipeline and M3's `BUG-027` / licence guardrails are fixed.
