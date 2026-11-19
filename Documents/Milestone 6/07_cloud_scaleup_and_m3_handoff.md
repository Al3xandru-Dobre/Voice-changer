# Stage 07 — Cloud scale-up & Milestone 3 S2UT handoff

> **Milestone 6** · [Master plan](00_master.md) · [← Stage 06](06_live_mode_a_soak_and_own_voice.md) ·
> Dataset plan: [DATASET/README.md](../../DATASET/README.md) ·
> Feeds: [Milestone 3](../Milestone%203/00_master.md)

**Status:** ☐ not started
**Depends on:** Stage 05 (Mode A proven locally); mandatory pre-cloud code fixes below before spending any cloud compute on M3

---

## Objective

Graduate cleanly from the local laptop (`133 GB` free, RTX 4060 Laptop 8 GB) to a **\$0 or ultra-cheap ephemeral cloud workflow** once you are ready to:
1. Broaden Mode A with [DATASET/README.md](../../DATASET/README.md) **Phases P1–P2** (conversational/accented robustness: Common Voice, CV Spontaneous, LibriTTS-R, M-AILABS, FSD50K, AMI — `~115–200 GB` disk), and/or
2. Train **Milestone 3 Mode B** (bidirectional `En↔{Es,Fr,De}` `mhubert-vp` km1000 encoder, multilingual vocoder, and Wait-*k* Conformer-Transformer S2UT — `~150–300 GB` transient audio + multi-day 24 GB GPU runs).

---

## When to move to the cloud (graduation criteria)

Stay local through Stage 06. Move to cloud **only when** one of these triggers hits:
- You want to train on **P2 diversity corpora** (`~120 GB` download / `~90 GB` disk) or **P3 scale** (`~800 GB`), OR
- You are starting **Milestone 3 S2UT** (`CVSS-C` + `CoVoST 2` + `CSS10` + `CML-TTS` + Piper target synthesis + Conformer-Transformer training).

> **Key architectural win:** `train_s2ut.py` reads **zero audio**—only pre-extracted student `.latents.npy`, teacher `.npy` units, and SentencePiece models (`~5–15 GB` total). And `models/<bundle>/` is `< 100 MB`. You never need to pay for long-term cloud audio storage.

---

## Mandatory pre-cloud blockers (fix locally BEFORE renting any GPU)

Do not spin up cloud compute for Milestone 3 until both repo bugs are closed locally (otherwise you will burn credits on broken downloads or unusable non-commercial audio):

- [ ] **1. Fix [`BUG-027`](../Bug%20Report/BUG-027_s2ut_data_prep_archive_plumbing_broken.md)** — broken archive download/extract/hash plumbing in `python_training/fvml/s2ut_data.py`.
- [ ] **2. Fix the `europarl-st` licence discrepancy** ([DATASET/README.md §6](../../DATASET/README.md#6-discrepancies-found-during-research--action-required)):
  - `europarl-st-en_{es,fr,de}` is licensed **CC BY-NC 4.0** (not CC BY 4.0).
  - File a `BUG-033` report in `Documents/Bug Report/`, move `europarl-st` behind `--include-noncommercial` (`data/nc/`), and decide the commercial replacement for `En→{Es,Fr}` paired data (e.g. reverse-CoVoST / commercial-clean parallel synthesis) before running Piper synthesis.
- [ ] **3. Implement `build_*` loaders for any P1/P2 corpora you intend to train on** in `fvml/datasets.py` (with `--delete-archives --delete-raw` and unit tests) before paying for cloud hours.

---

## Artifact hub pattern (keeps idle cloud cost at \$0)

Create one **private Hugging Face Hub** repository (e.g. `<user>/fluxvoice-artifacts` — **100 GB free private storage** on the free tier) to hold ONLY the small, high-value outputs:
- `checkpoints/<run>/{best.pt,last.pt,final.pt}` (`~50–300 MB` each)
- `units/km100/`, `units/mhubert/`, and pre-extracted `.npy` / `.latents.npy` packs
- `models/<bundle>/` (exported ONNX graphs, `voices.json`, `MODEL_CARD.md`, `golden/`)
- `runs/` gate/eval reports & TensorBoard logs

**Never upload raw public audio corpora to private storage** — datacenter networks re-download OpenSLR / Hugging Face archives at `200–1000 MB/s` in minutes. **Never upload private own-voice recordings (`data/private/`) to any third-party cloud.**

---

## Cheap / free cloud playbooks (ranked by cost)

| Tier | Provider | Estimated cost | Storage & GPU | Best workflow |
|---|---|---|---|---|
| **Tier 1 (Free cloud)** | **Modal Starter** + private HF Hub | **\$0/mo** *(set hard \$30 spend cap)* | **\$30/mo** free compute credits (~37 h L4 or ~27 h A10G/mo) + persistent Modal Volume *(verify current Starter Volume allowance on `modal.com/pricing`)* | Wrap `prepare_data.py`, `extract_units.py`, `train_*.py` in a small Modal runner; use Volume v2 or unpack tarred shards onto fast local scratch (`/tmp`) to avoid small-file slowdowns; sync `best.pt` to HF Hub. Zero idle storage penalty. |
| **Tier 1b (Free if student)** | **Azure for Students** / **GCP Free Trial** / **University HPC** | **\$0** | Azure: **\$100/yr** (no card, `.edu`/university email)<br>GCP: **\$300 / 90 days** (card + GPU quota request)<br>Uni HPC: free A100/V100 + TB scratch | Standard persistent Linux SSH box — run `python_training/` CLIs verbatim. Deallocate VMs and delete managed disks immediately after syncing to HF Hub. |
| **Tier 2 (Cheapest fast SSH)** | **Vast.ai** *(or RunPod Community)* + private HF Hub | **~\$8–\$20 per full sprint** | RTX 3090 (`~\$0.10–\$0.22/hr`) or RTX 4090 (`~\$0.27–\$0.48/hr`); `250–350 GB` disk (`~\$0.05–\$0.15/GB/mo` prorated = `< \$2` for a weekend) | Zero wrapper code: SSH in, `git clone`, `uv sync --frozen`, download corpora at gigabit speed, train M3 encoder + vocoder + S2UT in 1–2 days, push artifacts to HF Hub, **DESTROY instance immediately**. |

### Cloud cost-control rules (never skip)
1. **Always set a hard billing cap / credit alert** before starting any cloud job.
2. **Never leave a Vast.ai or RunPod pod "stopped" with a large disk** (RunPod charges *double*—`\$0.20/GB/mo`—for stopped volumes). Sync checkpoints/units to HF Hub, verify the upload, and **destroy** the instance.
3. **Checkpoint every 30 minutes + periodic HF Hub sync** (`--resume auto --ckpt-minutes 30`) so spot/preemptible interruptions lose almost no work.
4. **Always run `export_stateful_onnx.py` and pull only `models/<bundle>/` (`< 100 MB`) back to the laptop** for local C++ `fv_vc_file` / `fv_s2ut_file` verification and live PipeWire testing.

---

## Tasks (execute when ready to graduate to cloud)

- [ ] **1. Back up local Mode A artifacts** (`checkpoints/ce_m2_v1/best.pt`, `checkpoints/voc_ft/best.pt`, `data/units/km100/`, `models/mode_a_v1/`) to your private HF Hub repo (or offline backup drive).
- [ ] **2. Close the pre-cloud M3 blockers** (`BUG-027` + `BUG-033` `europarl-st` CC BY-NC reclassification) and pass `ctest --preset linux-gcc-training`.
- [ ] **3. Choose Tier 1 (Modal / student credits) or Tier 2 (Vast.ai weekend sprint)** and add a lightweight bootstrap script (e.g. `tools/cloud_bootstrap.sh` or `python_training/modal_runner.py`, keeping secrets out of git).
- [ ] **4. Execute M3 training wave in the cloud** following [Milestone 3 Stages 01, 03, 05, 06](../Milestone%203/00_master.md), pulling `models/mode_b_v1/` back locally for Stages 07–11 C++ & live verification.

---

## Definition of done

- Local Mode A checkpoints, `km100` codebook, and `models/mode_a_v1/` are safely backed up off-laptop.
- `BUG-027` and the `europarl-st` licence bug are resolved before any cloud M3 prep begins.
- A reproducible, spend-capped cloud workflow produces the next trained bundle (`P1/P2` Mode A upgrade or `Milestone 3` Mode B) with **\$0 recurring idle storage cost**.
