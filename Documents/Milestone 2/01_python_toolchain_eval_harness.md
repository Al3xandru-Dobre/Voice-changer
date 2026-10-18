# Stage 01 — Python toolchain, datasets & evaluation harness

> **Milestone 2** · [Master plan](00_master.md) · Next: [Stage 02 →](02_model_contract_fixtures.md)

**Status:** ◐ in progress — environment, data tooling, evaluation lists, harness and baseline
done (2026-10-18); the training corpora are still downloading (see Verification)
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

- [x] **Environment.** `python_training/pyproject.toml` (`requires-python =
      "==3.12.*"`) + committed `uv.lock`.
  - [x] CUDA PyTorch wheels from the PyTorch index that matches the driver
        (`[tool.uv.sources]`). Pin `torch`/`torchaudio` to the newest stable
        release at stage start (≥ 2.9, where the dynamo ONNX exporter is the
        default).
        *Pinned: torch 2.11.0 + torchaudio 2.11.0 from the `cu128` index. torch
        itself is at 2.14, but torchaudio was discontinued after 2.11, and
        SpeechBrain and SpeechMOS import it, so 2.11.0 is the newest matched
        pair. cu128 runs on any driver ≥ 525 (cu130 would need ≥ 580 on cloud
        boxes too). macOS falls back to the PyPI build.*
  - [x] Core dependencies, all pinned in the lock: `numpy`, `soundfile`,
        `soxr`, `onnx`, `onnxruntime` (CPU; same minor version as the C++ SDK
        pin in Stage 07), `transformers`, `scikit-learn`, `speechbrain`,
        `openai-whisper` (or `transformers` Whisper), a UTMOS package,
        `pyloudnorm`, `tensorboard`.
        *Notes:*
        - *onnx 1.23.2 and onnxruntime 1.31.0.*
        - *Whisper runs through `transformers` 5.20.1: `openai-whisper` is
          sdist-only and drags in numba and tiktoken.*
        - *UTMOS22-strong comes from SpeechMOS (git tag v1.2.0, commit locked).
          Its weights are checked by SHA256 in `fvml/mos.py`.*
        - *Added: `jiwer` 4.0.0, `scipy`, and `praat-parselmouth` 0.4.7 for the
          reference F0 tracker.*
  - [x] `uv python install 3.12` documented; `uv sync --frozen` is the only
        install command.
  - [x] Open decision 6: generate `requirements.txt` from the lock with
        `uv export --no-hashes` and note this at the top of the file.
        *The header gives the exact command. pip users also need
        `--extra-index-url https://download.pytorch.org/whl/cu128`, because the
        export emits no index lines.*
- [x] **Package layout.** `python_training/fvml/` holds shared modules (later
      stages add `contract.py`, `features.py`, `streaming.py`, models). Top-level
      scripts stay thin CLIs.
      *Stage 01 modules: `paths`, `audio`, `datasets`, `evallist`, `parity`,
      `asr`, `speaker`, `mos`, `f0_ref`, `evaluate`.*
- [x] **Data root.** `FLUXVOICE_DATA` env var, default `<repo>/data/`. Add to
      `.gitignore`: `/data/`, `/checkpoints/`, `/runs/`, `*.pt`, `*.ckpt`,
      `*.onnx.data` (root-anchored: a bare `data/` would also ignore `tests/data/`).
      *Notes:*
      - *The entries are anchored (`/data/`, `/checkpoints/`, `/runs/`, plus
        `python_training/{checkpoints,runs}/`). An unanchored `data/` would
        also hide `tests/data/` ([BUG-008](../Bug%20Report/BUG-008_gitignore_data_swallows_tests_data.md)).*
      - *The data root also ignores itself (`data/.gitignore`).*
      - *Model weights fetched for evaluation are cached under
        `$FLUXVOICE_DATA/cache/` unless `HF_HOME` is set
        (`fvml.paths.configure_model_caches`).*
- [x] **`prepare_data.py`.** Download + SHA256 verify + resample to 16 kHz mono
      float + write JSONL manifests (path, duration, speaker, dataset).
      *How it works:*
      - *Verification: OpenSLR and DataShare publish MD5 only, and Hi-Fi TTS
        publishes nothing. Each archive is therefore checked against its
        published MD5 and the SHA256 in `datasets.lock.json`, which is seeded
        from torchaudio's dataset tables. An archive without a pin gets one on
        its first verified download.*
      - *Only non-16 kHz corpora are resampled, and the copies are stored as
        24-bit FLAC: a third of the disk of float WAV, with quantisation noise
        near −144 dB.*
      - *Mirrors are ranked by measured speed. From here the ELDA mirror is
        about 10× faster than www.openslr.org.*
      - *Manifest rows also carry `utt_id`, `subset`, `gender` and `text` where
        the corpus has them.*
  - [x] LibriSpeech train-clean-100, train-clean-360, dev-clean, test-clean
        *(test/dev-clean prepared; train-clean-100/360 downloading)*
  - [x] LJSpeech; Hi-Fi TTS, 2–4 speakers chosen for gender/pitch spread (the
        rest optional)
        *Status:*
        - *LJSpeech is prepared.*
        - *Hi-Fi TTS readers 92 (F), 8051 (F), 6097 (M) and 9017 (M) are
          extracted from the single 41 GB archive: the three clean readers plus
          the strongest "other" female. Their audio is downloading. The
          manifests were streamed from the head of the archive so that
          `targets.tsv` could be built first.*
        - *The pitch spread is to be confirmed with the Stage 03 tracker.*
  - [x] MUSAN (noise/music/babble) and RIRs (OpenSLR 26/28) for augmentation
        *(downloading; OpenSLR 28 contains the OpenSLR 26 simulated RIRs)*
  - [x] VCTK: optional, for accent evaluation *(registered: `--datasets vctk`; not downloaded)*
  - [x] Reports hours per dataset and speaker; idempotent (skips verified files)
- [x] **Frozen evaluation lists**, committed under `python_training/eval_lists/`:
  - [x] `test_clean_200.tsv`: 200 test-clean utterances, ≥ 20 speakers,
        balanced by gender, 3–15 s, with reference transcripts
        *(5 from each of the 40 speakers, 100 F / 100 M, 24.9 min; built
        deterministically by `make_eval_lists.py`)*
  - [x] `targets.tsv`: held-out reference clips per target voice, used for
        target-similarity scoring
        *(20 clips of 3–20 s per voice: `lj` from LJSpeech chapter LJ050 and
        `hifitts_{92,8051,6097,9017}` from each reader's test split. Reader
        9017 has only 17 test clips under 15 s, hence 20 s. Stage 05 must keep
        these out of training.)*
- [x] **`eval_vc.py`.** Input is a directory of converted WAVs + a list + target
      references. Output is per-utterance CSV + summary JSON + a Markdown table.
      *The engine is `fvml/evaluate.py` (`evaluate()`, importable for Stage 05).*
      *Additions:*
      - *`--baseline`;*
      - *`--ref-text source`, Whisper on the source as the reference, for the
        Stage 11 live set;*
      - *`--targets <private.tsv>`, for the Stage 10 "me" voice;*
      - *a per-utterance F0 median and log-F0 std, for Stage 09.*
  - [x] WER/CER: Whisper large-v3 (fp16 on GPU), Whisper `EnglishTextNormalizer`
        on both sides, corpus-level aggregation
        *(revision pinned. Decoding uses `generate()`: the transformers 5.20 ASR
        pipeline re-appends the cross-attention cache every step, which ran an
        8 GB GPU out of memory.)*
  - [x] Speaker: ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) cosine to the
        target references (mean) and to the source utterance (leakage); EER
        helper
  - [x] Naturalness: UTMOS (record model + version in the JSON)
  - [x] Prosody: F0 PCC between source and output on frames voiced in both,
        using an offline (non-causal is fine) reference tracker
        *(the tracker is Praat's autocorrelation pitch via parselmouth, 50–1000
        Hz, in `fvml/f0_ref.py`, shared with Stages 03/05/09. The PCC is on
        log-F0, after removing the output's delay with an envelope
        cross-correlation of ±200 ms.)*
  - [x] RTF of the system under test, when it reports timing
        *(`--timing` CSV `utt_id,seconds`, or `<converted>/timing.csv`)*
- [x] **`parity.py`.** Compares two signals or tensors: max |Δ|, SNR, argmax
      agreement, and the first index that diverges. Later stages use it for
      streaming-vs-offline and PyTorch-vs-ORT checks.
      *(`fvml/parity.py`, plus max sample-to-sample step and the 99.9th
      percentile step for click checks, and `fvml.audio.concat` for long
      dev-clean streams)*
- [x] **Baseline run.** Score the *unconverted* test-clean audio: ground-truth
      WER, same-speaker vs. different-speaker cosine, UTMOS. Record the numbers
      in this file's Verification section.
- [x] **`python_training/DATASETS.md`.** One row per dataset/model: source URL,
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

## Verification (2026-10-18, Windows — partial)

**Environment.** Windows 11, RTX 4060 Laptop (driver 592.82), uv 0.11.21:

- `uv python install 3.12` gave CPython 3.12.13.
- `uv lock` resolved 98 packages, and `uv sync --frozen` reproduced the
  environment.
- `torch 2.11.0+cu128` reports `torch.cuda.is_available() == True`.

The Linux side of the same lock was only resolved (universal lock), not
installed.

**Data** (`prepare_data.py --datasets eval`, then `--datasets ljspeech`). Every
archive passed its published MD5 (where there is one) and torchaudio's SHA256:

| Dataset | Rows | Hours | Speakers | Check |
|---|---|---|---|---|
| librispeech-test-clean | 2 620 | 5.40 | 40 | MD5 ✓ SHA256 ✓ |
| librispeech-dev-clean | 2 703 | 5.39 | 40 | MD5 ✓ SHA256 ✓ |
| ljspeech | 13 100 | 23.92 | 1 | SHA256 ✓ (resampled 22.05 → 16 kHz) |

**Remaining data.** train-clean-100, RIRs, MUSAN, train-clean-360 and the four
Hi-Fi TTS readers were started at 15:30 in one resumable run (`--datasets
librispeech-train-clean-100,rirs,musan,librispeech-train-clean-360,hifitts
--delete-archives`, 83 GB at ~4 MB/s). Their row in this table and the
"checksums pass for every dataset" DoD item are still open.

**Baseline** (`eval_vc.py --baseline`, unconverted `test_clean_200`):

| WER % | CER % | UTMOS | Same-spk cos | Diff-spk cos | Gap | EER % | Eval time |
|---|---|---|---|---|---|---|---|
| 1.54 | 0.45 | 4.14 | 0.706 (400 pairs) | 0.086 (19 500 pairs) | 0.620 | 1.27 | 148 s |

- The six utterances above 15% WER are spelling variants of the reference:
  "et cetera" → "etc", "mainhall" → "main hall", "lindens" → "lyndon is".
- UTMOS spans 2.99–4.50 (median 4.18).
- Evaluation time by stage: Whisper 114 s, ECAPA 13 s, UTMOS 16 s, F0 3 s.

**Tests.**

- `python_training/tests/` (pytest, uv env) passes 32 tests:
  - the contract cases;
  - fixture behaviour in ORT 1.31;
  - every `prepare_data` builder (LibriSpeech, LJSpeech, Hi-Fi TTS speaker
    filter, MUSAN, RIRs, VCTK) on miniature fake corpora through `file://`
    URLs, including checksum failures and idempotent reruns;
  - parity, eval lists, corpus-level WER and EER.
- The reference tracker reads a synthetic +12 st shift as an F0 ratio of
  2.00 ± 3 % and recovers a 40 ms delay exactly.
- These run under CTest as `python_training_stage01_02` when
  `python_training/.venv` exists.

**Notes.**

- The first full run hit a CUDA out-of-memory error inside the transformers
  5.20 ASR pipeline:
  - more than 20 GiB was allocated on an 8 GB card, and a 12-utterance smoke
    run took 921 s;
  - the cause is the cross-attention cache being concatenated at every
    decoding step;
  - `fvml/asr.py` calls `generate()` directly instead, at batch 8 (< 5 GiB
    peak).
- `make_eval_lists.py --check` rebuilds both lists identically.

## Files touched

- `python_training/pyproject.toml` (new), `python_training/uv.lock` (new),
  `python_training/requirements.txt` (generated), `python_training/fvml/__init__.py`
  (new), `python_training/prepare_data.py` (new), `python_training/eval_vc.py`
  (new), `python_training/parity.py` (new), `python_training/eval_lists/*` (new),
  `python_training/DATASETS.md` (new), `.gitignore`
- Added: `python_training/fvml/{paths,audio,datasets,evallist,parity,asr,speaker,mos,f0_ref,evaluate}.py`,
  `python_training/make_eval_lists.py`, `python_training/datasets.lock.json`,
  `python_training/tests/{test_prepare_data,test_eval_tools}.py`,
  `data/.gitignore` (generated), `CMakeLists.txt` (CTest `python_training_stage01_02`)

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

Met except the dataset checksums (see Verification):
- the environment reproduces with CUDA;
- WER is 1.54 %;
- the cosine gap is 0.620;
- the evaluation takes 148 s;
- checksums pass for test-clean, dev-clean and LJSpeech, and the training
  corpora are still downloading.
