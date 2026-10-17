# Stage 01 — Python toolchain, datasets & evaluation harness

> **Milestone 2** · [Master plan](00_master.md) · Next: [Stage 02 →](02_model_contract_fixtures.md)

**Status:** ☐ not started
**Depends on:** —
**Research:** §6 data & recording · §7 evaluation · §12 licences
**Estimated scope:** Python only; no C++ changes

## Objective

Make every later ML stage reproducible and measurable. That needs four things:
- a pinned training environment that installs the same way on any machine;
- scripted dataset preparation with checksums;
- frozen evaluation lists;
- a single `eval_vc.py` that scores any converted audio the same way.

Every decision in this milestone, including the Stage 05 gate, is made with the
numbers this harness produces.

## Context & references

- `python_training/requirements.txt` is unpinned ("Pin exact versions once the
  toolchain is chosen in stage plans"). The three scripts there are argparse stubs
  that raise `NotImplementedError`.
- **Dev machine:** Windows 11, RTX 4060 Laptop (8 GB), i7-13700HX, only Python
  3.14 installed. Python 3.14 is too new for parts of the ML stack (speech
  toolkits, some CUDA wheels), so the training env pins **3.12** and lets `uv`
  manage the interpreter.
- `.gitignore` covers `models/*.onnx`, `.venv/` and `__pycache__/`. It doesn't
  cover datasets, checkpoints, logs or ONNX external-data files.
- **Metric choices (research §7):**
  - Whisper large-v3 WER/CER with Whisper's text normaliser
  - ECAPA-TDNN cosine similarity to the target *and* to the source (leakage)
  - UTMOS
  - F0 Pearson correlation

## Tasks

- [ ] **Environment.** `python_training/pyproject.toml` (`requires-python =
      "==3.12.*"`) + committed `uv.lock`.
  - [ ] CUDA PyTorch wheels from the PyTorch index that matches the driver
        (`[tool.uv.sources]`). Pin `torch`/`torchaudio` to the newest stable
        release at stage start (≥ 2.9, where the dynamo ONNX exporter is the
        default).
  - [ ] Core dependencies, all pinned in the lock: `numpy`, `soundfile`,
        `soxr`, `onnx`, `onnxruntime` (CPU; same minor version as the C++ SDK
        pin in Stage 07), `transformers`, `scikit-learn`, `speechbrain`,
        `openai-whisper` (or `transformers` Whisper), a UTMOS package,
        `pyloudnorm`, `tensorboard`.
  - [ ] `uv python install 3.12` documented; `uv sync --frozen` is the only
        install command.
  - [ ] Open decision 6: generate `requirements.txt` from the lock with
        `uv export --no-hashes` and note this at the top of the file.
- [ ] **Package layout.** `python_training/fvml/` holds shared modules (later
      stages add `contract.py`, `features.py`, `streaming.py`, models). Top-level
      scripts stay thin CLIs.
- [ ] **Data root.** `FLUXVOICE_DATA` env var, default `<repo>/data/`. Add to
      `.gitignore`: `data/`, `checkpoints/`, `runs/`, `*.pt`, `*.ckpt`,
      `*.onnx.data`.
- [ ] **`prepare_data.py`.** Download + SHA256 verify + resample to 16 kHz mono
      float + write JSONL manifests (path, duration, speaker, dataset).
  - [ ] LibriSpeech train-clean-100, train-clean-360, dev-clean, test-clean
  - [ ] LJSpeech; Hi-Fi TTS, 2–4 speakers chosen for gender/pitch spread (the
        rest optional)
  - [ ] MUSAN (noise/music/babble) and RIRs (OpenSLR 26/28) for augmentation
  - [ ] VCTK: optional, for accent evaluation
  - [ ] Reports hours per dataset and speaker; idempotent (skips verified files)
- [ ] **Frozen evaluation lists**, committed under `python_training/eval_lists/`:
  - [ ] `test_clean_200.tsv`: 200 test-clean utterances, ≥ 20 speakers,
        balanced by gender, 3–15 s, with reference transcripts
  - [ ] `targets.tsv`: held-out reference clips per target voice, used for
        target-similarity scoring
- [ ] **`eval_vc.py`.** Input is a directory of converted WAVs + a list + target
      references. Output is per-utterance CSV + summary JSON + a Markdown table.
  - [ ] WER/CER: Whisper large-v3 (fp16 on GPU), Whisper `EnglishTextNormalizer`
        on both sides, corpus-level aggregation
  - [ ] Speaker: ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) cosine to the
        target references (mean) and to the source utterance (leakage); EER
        helper
  - [ ] Naturalness: UTMOS (record model + version in the JSON)
  - [ ] Prosody: F0 PCC between source and output on frames voiced in both,
        using an offline (non-causal is fine) reference tracker
  - [ ] RTF of the system under test, when it reports timing
- [ ] **`parity.py`.** Compares two signals or tensors: max |Δ|, SNR, argmax
      agreement, and the first index that diverges. Later stages use it for
      streaming-vs-offline and PyTorch-vs-ORT checks.
- [ ] **Baseline run.** Score the *unconverted* test-clean audio: ground-truth
      WER, same-speaker vs. different-speaker cosine, UTMOS. Record the numbers
      in this file's Verification section.
- [ ] **`python_training/DATASETS.md`.** One row per dataset/model: source URL,
      licence, attribution text, what it is used for. The MUSAN and RIR
      licences are checked per subset.

## Implementation notes

- **Keep the evaluation stack independent of the training stack.** `eval_vc.py`
  must score files produced by C++ (`fv_vc_file`, live recordings) exactly as it
  scores PyTorch outputs. Its only input is WAV files.
- **Resample once, at preparation time.** All training and evaluation reads
  16 kHz mono float. If the Mimi kill test survives (Stage 04), resample to
  24 kHz on the fly inside that test only.
- **Use Whisper large-v3 even though it is slow.** It is the reference ASR in
  current VC papers. With 200 utterances, a full evaluation should stay under
  30 min on the RTX 4060. If it doesn't, batch with the `transformers` pipeline
  before reaching for a smaller model.
- **Similarity to the source is the leakage signal.** A conversion that sounds
  like the target but keeps source cosine high is leaking timbre through units
  or F0.
- **Cloud GPU stays optional.** The same `uv.lock` must reproduce on a Linux CUDA
  box, so check the lock with `uv sync --frozen` on both OSes if a cloud run
  happens.

## Files touched

- `python_training/pyproject.toml` (new), `python_training/uv.lock` (new),
  `python_training/requirements.txt` (generated), `python_training/fvml/__init__.py`
  (new), `python_training/prepare_data.py` (new), `python_training/eval_vc.py`
  (new), `python_training/parity.py` (new), `python_training/eval_lists/*` (new),
  `python_training/DATASETS.md` (new), `.gitignore`

## Definition of done

- `uv sync --frozen` on a clean clone reproduces the environment;
  `torch.cuda.is_available()` is true on the dev machine.
- `prepare_data.py` checksums pass for every dataset, and the manifest summary
  reports hours per dataset.
- Ground-truth test-clean WER (Whisper large-v3, normalised) ≤ 3.5% on
  `test_clean_200`.
- ECAPA mean same-speaker cosine exceeds mean different-speaker cosine by ≥ 0.3
  on the frozen list.
- A full `eval_vc.py` run on 200 utterances finishes in ≤ 30 min on the RTX 4060.
