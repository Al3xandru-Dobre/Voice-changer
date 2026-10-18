# Stage 01 — Bidirectional datasets & SimulEval harness

> **Milestone 3** · [Master plan](00_master.md) · Next: [Stage 02 →](02_translator_contract_v2_fixtures.md)

**Status:** ☐ not started
**Depends on:** [Milestone 2 / Stage 01](../Milestone%202/01_python_toolchain_eval_harness.md) (pinned Python 3.12 `uv` environment, `eval_vc.py`, `DATASETS.md`)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §6 datasets · §7 evaluation · §10 licences
**Estimated scope:** Python only; no C++ changes

## Objective

Extend the Milestone 2 Python data and evaluation infrastructure to cover
bidirectional simultaneous speech-to-speech translation across all 6
`En↔{Es,Fr,De}` directions (`es→en`, `fr→en`, `de→en`, `en→es`, `en→fr`,
`en→de`) and monolingual `{es,fr,de}` vocoder corpora:
- scripted, checksummed dataset preparation producing unified JSONL manifests;
- canonical target-speech synthesis for `En→{Es,Fr,De}` pairs (mirroring
  CVSS-C's construction for `{Es,Fr,De}→En`);
- frozen 200-utterance evaluation lists per direction plus a word-order
  reordering torture set;
- a single `eval_s2ut.py` that scores any translated WAV directory on
  **ASR-BLEU**, **ASR-chrF++**, **SimulEval Average Lagging (`AL` and `AL_CA`)**,
  **ECAPA speaker cosine**, **UTMOS**, and **tail-flush hallucination rate**.

## Context & references

- [M2 / Stage 01](../Milestone%202/01_python_toolchain_eval_harness.md) established
  `python_training/pyproject.toml`, `uv.lock`, `prepare_data.py`, `eval_vc.py`
  and `python_training/DATASETS.md` for English VC.
- **Asymmetry in public S2ST corpora (research §6):**
  - **CVSS-C** (`google/cvss`, CC BY 4.0) provides parallel `(src_audio,
    src_text, tgt_text, tgt_audio)` for `Fr→En` (264 h src / 174 h tgt), `De→En`
    (184 h / 112 h), and `Es→En` (113 h / 70 h) in a single clean canonical voice.
  - **CoVoST 2** (`facebook/covost2`, CC0) provides `En→De` (289k train utts,
    ~364 h) with human English source audio and German target text, while
    **Europarl-ST** (CC BY 4.0) and reversed CoVoST 2 pairs provide parallel
    `(en_audio, en_text, {es,fr,de}_text)`. Synthesizing those target transcripts
    with a single clean canonical voice per language (`de`, `es`, `fr`) produces a
    symmetric "reverse CVSS-C" corpus for `En→{Es,Fr,De}`.
- **Multilingual vocoder waveforms (Stage 03):**
  - **CSS10** (`kyubyong/css10`, Apache 2.0 / CC0 LibriVox): studio
    single-speaker datasets for `es` (~24 h), `fr` (~19 h), `de` (~16 h).
  - **CML-TTS** (`ylacombe/cml-tts`, OpenSLR 146, CC BY 4.0): 24 kHz LibriVox
    TTS splits; 2–3 high-SNR speakers per language (~15 h each) add multi-speaker
    diversity.
- **Commercial licence guardrail (research §10):** VoxPopuli (`facebook/voxpopuli`)
  is CC BY-NC 4.0. Keep it behind an explicit `--include-noncommercial` flag for
  research comparisons only; never include it in the default bundle manifests.

## Tasks

- [ ] **Dependencies.** Update `python_training/pyproject.toml` + `uv.lock` (and
      regenerate `requirements.txt` via `uv export --no-hashes`):
  - [ ] `sacrebleu` (pinned, with `13a` and `chrf` scorers)
  - [ ] `sentencepiece` (for Stage 05's 6,000-subword auxiliary CTC vocabularies)
  - [ ] `piper-tts` or `coqui-tts` / `vits` checkpoint runner for offline
        canonical `En→{Es,Fr,De}` target-audio generation during data prep
- [ ] **`python_training/prepare_s2ut_data.py`:**
  - [ ] Download, verify SHA256, resample all audio to **16 kHz mono float32**,
        and write paired JSONL manifests (`id`, `src_lang`, `tgt_lang`,
        `src_audio_path`, `src_duration_s`, `src_text`, `tgt_audio_path`,
        `tgt_duration_s`, `tgt_text`, `speaker_id`, `dataset`, `license`):
    - [ ] **CVSS-C** (`fr_en`, `de_en`, `es_en`) joined with CoVoST 2 source
          audio and transcripts
    - [ ] **CoVoST 2 `en_de`** + **Europarl-ST `en_{es,fr,de}`**: synthesize
          canonical 16 kHz target WAVs using one fixed studio TTS voice per
          target language (`es`, `fr`, `de`); verify synthesis quality by running
          Whisper ASR on the generated target WAVs and rejecting clips with
          synthesis CER > 5%
    - [ ] **CSS10** (`es`, `fr`, `de`) and curated **CML-TTS** speakers (2 male,
          2 female per language, filtered by SNR $\ge 30\text{ dB}$ and
          duration 2–12 s) → monolingual vocoder manifests
    - [ ] Optional `--include-noncommercial`: prepares **VoxPopuli** S2S /
          transcribed splits into `data/nc/` so they cannot be mixed into
          default training runs accidentally
  - [ ] Fit and save shared 6,000-token **SentencePiece** unigram models per
        language (`data/spm/spm_{en,es,fr,de}.model`) for Stage 05's auxiliary
        CTC supervision heads.
- [ ] **Frozen evaluation lists** under `python_training/eval_lists/s2ut/`:
  - [ ] `test_{es_en,fr_en,de_en,en_es,en_fr,en_de}_200.tsv`: 200 held-out test
        utterances per direction (3–12 s, $\ge 20$ source speakers, balanced
        gender, normalized reference source & target text, duration ratio
        $\gamma = T_{\text{tgt}} / T_{\text{src}}$)
  - [ ] `reordering_torture_60.tsv`: 60 utterances (10 per direction) specifically
        selected for cross-lingual syntactic reordering (German subordinate
        verb-final `weil/dass ... [verb]` clauses, Spanish/French post-nominal
        adjective chains and clitic object pronouns)
  - [ ] `silence_hallucination_20.tsv`: 10 room-tone/noise-only clips + 10
        utterances followed by 5 s of trailing silence (tests `kFlush`
        termination)
- [ ] **`python_training/eval_s2ut.py`:**
  - [ ] Inputs: directory of translated WAVs + optional per-utterance emission
        timestamp JSONL (`[{step, t_src_ms, t_tgt_frames_emitted}]`) + eval TSV +
        target speaker reference clips.
  - [ ] **Translation accuracy:** transcribe each output WAV with Whisper
        `large-v3` (`language=tgt_lang`, `task="transcribe"`), apply text
        normalisation (Whisper normaliser + lowercase punctuation strip), and
        compute corpus **ASR-BLEU** (`sacrebleu`, `tokenize="13a"`) and
        **ASR-chrF++** (`word_order=2`).
  - [ ] **Simultaneous latency (SimulEval formula, research §4 & §7):**
    - [ ] First-audio onset delay (ms): first voiced/non-silence output sample
          timestamp minus source speech onset timestamp
    - [ ] Algorithmic **Average Lagging (`AL`, ms)** and **Computation-Aware
          Average Lagging (`AL_CA`, ms)** using word/frame emission timestamps
  - [ ] **Voice & acoustic quality:** ECAPA-TDNN cosine similarity to target
        voice references (`tgt_cos`) and to source speaker (`src_cos` leakage),
        plus **UTMOS**.
  - [ ] **Flush & hallucination audit:** trailing non-silent duration after the
        expected utterance end, plus Whisper token count on the
        `silence_hallucination_20.tsv` noise clips (must be 0 words).
- [ ] **Oracle baseline run.** Score the ground-truth target audio on all 6
      `test_*_200.tsv` lists with `eval_s2ut.py` and record the oracle ASR-BLEU,
      ASR-chrF++ and UTMOS ceilings in this file's Verification section.
- [ ] **Update `python_training/DATASETS.md`** with CVSS-C, CoVoST 2,
      Europarl-ST, CSS10, CML-TTS, `mhubert_base_vp_en_es_fr_it3`, and the
      NC-isolated datasets (`mHuBERT-147`, VoxPopuli).

## Implementation notes

- **Why `eval_s2ut.py` takes WAVs + a simple timestamp JSONL.** Just like
  `eval_vc.py` in M2, `eval_s2ut.py` must score outputs from PyTorch
  (`gate_s2ut.py`), Python ORT, C++ offline (`fv_s2ut_file`), and live virtual-cable
  recordings (`fv_probe record`) with the exact same code path.
- **Filter synthetic `En→{Es,Fr,De}` target clips by round-trip CER.** TTS
  occasionally skips numbers or mispronounces proper nouns. Rejecting any clip
  whose Whisper transcript differs from `tgt_text` by $> 5\%$ CER keeps the
  target unit labels as clean as CVSS-C.
- **Whisper language forcing.** Always pass `language=tgt_lang` to Whisper when
  scoring S2UT outputs; otherwise short utterances with a foreign accent or mixed
  proper nouns can trigger language-ID flips.

## Files touched

- `python_training/pyproject.toml`, `python_training/uv.lock`,
  `python_training/requirements.txt` (regenerated),
  `python_training/prepare_s2ut_data.py` (new),
  `python_training/eval_s2ut.py` (new),
  `python_training/eval_lists/s2ut/*.tsv` (new),
  `python_training/DATASETS.md`

## Definition of done

- `uv sync --frozen` installs cleanly; `prepare_s2ut_data.py` verifies checksums,
  writes 16 kHz manifests for all 6 `En↔{Es,Fr,De}` pairs plus CSS10/CML-TTS, and
  trains the 4 SentencePiece models.
- Every synthesized `En→{Es,Fr,De}` training clip passes the $\le 5\%$ Whisper
  round-trip CER filter.
- Oracle ground-truth `eval_s2ut.py` run on the six `test_*_200.tsv` lists yields
  oracle ASR-BLEU $\ge 70$ across all 6 directions and UTMOS $\ge 3.6$, recorded
  in this file.
- Full 6-direction `eval_s2ut.py` run (1,200 utterances total) completes in $\le
  45\text{ min}$ on the RTX 4060 Laptop GPU.
