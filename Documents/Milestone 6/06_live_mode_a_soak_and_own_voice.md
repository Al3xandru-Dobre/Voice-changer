# Stage 06 — Live Mode A verification, soak & optional own-voice ("me")

> **Milestone 6** · [Master plan](00_master.md) · [← Stage 05](05_decision_gate_finetune_and_onnx_bundle.md) · [Stage 07 →](07_cloud_scaleup_and_m3_handoff.md)

**Status:** ☐ not started
**Depends on:** Stage 05 (`models/mode_a_v1/` verified with `fv_vc_file`), `build-linux-gcc-ort` built
**Closes:** [M2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) (live), [M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) (offline & live DoD), [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) (optional), [M2 / Stage 11](../Milestone%202/11_verification_exit.md)

---

## Objective

Prove the trained Mode A bundle works live on this Linux machine (PipeWire latency budget, controls DoD, 30-minute idle & loaded soak), and optionally add your own private voice (`--voice me`) before closing Milestone 2.

---

## Tasks

- [ ] **1. Run the Stage 09 offline voice-controls DoD against the shipped `fv_vc_file`.**
  ```bash
  cd python_training
  FLUXVOICE_VC_FILE=../build-linux-gcc-ort/fv_vc_file \
  .venv/bin/python eval_controls.py \
      --bundle ../models/mode_a_v1 \
      --voice lj --voice-b hifitts_6097 \
      --out ../runs/eval/controls_mode_a_v1
  ```
  Verify `runs/eval/controls_mode_a_v1/report.md` shows `PASS` for `pitch`, `intonation`, `blend`, `click`, and `presets`.

- [ ] **2. Measure real Mode A live latency on Linux PipeWire (`tools/pw_latency_check.sh`).**
  ```bash
  FLUXVOICE_BIN=./build-linux-gcc-ort/fluxvoice \
  FV_PROBE_BIN=./build-linux-gcc-ort/fv_probe \
  tools/pw_latency_check.sh \
      --modes "passthrough queue vc" \
      --marker speech \
      --model-dir models/mode_a_v1 \
      --period-ms 10 --count 60 \
      --keep-logs runs/latency_mode_a_v1
  ```
  Repeat with `--period-ms 20` for the Stage 11 latency matrix.
  Confirm added `T - B <= 100 ms` (stretch `<= 80 ms`) and check in-process `lat_ms` telemetry against measured `T - B`.

- [ ] **3. Run the 30-minute Mode A soak (idle + loaded) and analyze with `analyze_soak.py`.**
  - *Idle soak* (`1800 s`, `--stats 10`, routed over PipeWire null sinks or loopback):
    ```bash
    ./build-linux-gcc-ort/fluxvoice --vc \
        --model-dir models/mode_a_v1 \
        --voice lj --duration 1800 --stats 10 \
        | tee runs/soak_mode_a_idle.log

    python3 tools/analyze_soak.py --vc \
        --latency-target-ms 100 --hop-ms 20 \
        runs/soak_mode_a_idle.log
    ```
  - *Loaded soak:* repeat while running a parallel CPU build (`cmake --build --preset linux-clang -j` or `stress-ng` on non-audio cores).
  Confirm: **0 deadline misses after warm-up**, flat underruns, bounded `queued`, and `RSS growth <= 5 MB`.

- [ ] **4. Smoke-test the desktop GUI with the real bundle.**
  ```bash
  ./build-linux-gcc-ort/frontend/fluxvoice_gui
  ```
  Load `models/mode_a_v1`, test voice switching, presets (`deep`, `robot`, etc.), pitch/intonation/blend sliders, and live latency telemetry.

- [ ] **5. Optional — Own-voice ("me") fine-tune ([M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md)).**
  *(Requires `1–3 h` of clean 48 kHz / 24-bit WAV takes + 30 s room tone per session; uses `< 2 GB` extra disk. Output stays strictly private under `data/private/me/` and `models/mode_a_v1_me/` — NEVER commit.)*
  ```bash
  cd python_training
  # 5a. Clean, VAD-segment, SNR-filter (>= 30 dB), and hold out 5 min
  .venv/bin/python prepare_target_voice.py \
      --input /path/to/raw_me_takes \
      --speaker me --display "Me (private)" --lang en

  # 5b. Extract winning student units with 50% speaker perturbation
  WIN_HEAD=student-L1   # match Stage 05 winner
  .venv/bin/python extract_units.py \
      --tokenizer "$WIN_HEAD" \
      --checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --perturb-fraction 0.5 \
      --manifest ../data/private/me/train.jsonl \
      --out-dir "../data/units/$WIN_HEAD/me"

  # 5c. 50k-step mixed fine-tune (50% "me" / 50% public voices) with UTMOS early stopping
  .venv/bin/python train_unit_vocoder.py \
      --train-manifest ljspeech.jsonl --train-manifest hifitts.jsonl \
      --train-manifest ../data/private/me/train.jsonl \
      --train-units "../data/units/$WIN_HEAD/lj" \
      --train-units "../data/units/$WIN_HEAD/hifitts" \
      --train-units "../data/units/$WIN_HEAD/me" \
      --exclude-targets eval_lists/targets.tsv,../data/private/me/me_targets.tsv \
      --finetune-from ../checkpoints/voc_ft/best.pt \
      --new-speaker-init nearest \
      --mix-speakers me --mix-ratio 0.5 \
      --speaker-metadata ../data/private/me/speaker_metadata.json \
      --early-stop-patience 3 --val-utmos \
      --steps 50000 --ckpt-minutes 30 \
      --run-name voc_me --resume auto

  # 5d. Export PRIVATE bundle (verify PRIVATE banner in MODEL_CARD.md)
  .venv/bin/python export_stateful_onnx.py \
      --role encoder --checkpoint ../checkpoints/ce_m2_v1/best.pt \
      --head "$WIN_HEAD" --bundle ../models/mode_a_v1_me \
      --license "private — do not distribute"
  .venv/bin/python export_stateful_onnx.py \
      --role vocoder --checkpoint ../checkpoints/voc_me/best.pt \
      --bundle ../models/mode_a_v1_me
  ```
  Score with `eval_target_voice.py` / `eval_vc.py --targets ../data/private/me/me_targets.tsv --voice me`.

- [ ] **6. Close Milestone 2 documentation.**
  Fill in [M2 / Stage 11](../Milestone%202/11_verification_exit.md) tables (latency, quality, soak), update [M2 / 00_master.md](../Milestone%202/00_master.md) acceptance criteria, and update `Documents/README.md` + root `README.md` §3/§4 stale Mimi/RMVPE/worker-priority notes per Stage 11's checklist.

---

## Verification & measurements *(fill in)*

- `eval_controls.py` verdict (`pitch` / `intonation` / `blend` / `click` / `presets`): `___`
- Measured PipeWire `T - B` median / p95 (`10 ms` period): `___ / ___ ms`
- 30-min soak (idle / loaded) misses, underruns, RSS delta: `___`
- Own-voice ("me") fine-tune run? (`yes` / `deferred`; if yes, `WER` / `tgt_cos` / `UTMOS`): `___`

---

## Definition of done

- `eval_controls.py` exits `0` (`PASS` on all 5 checks) on `models/mode_a_v1`.
- Measured PipeWire `T - B <= 100 ms` on `models/mode_a_v1`, and 30-min idle + loaded soak pass `analyze_soak.py --vc`.
- [M2 / Stage 11](../Milestone%202/11_verification_exit.md) exit report is filled in and Milestone 2 is closed.
