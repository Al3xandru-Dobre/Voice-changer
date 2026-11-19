---
paths:
  - "python_training/**"
  - "models/**"
---

# Speech & voice-conversion ML rules (`@ml-speech-trainer`)

Scope: the Python 3.12 training & evaluation pipeline, causal pitch/energy extraction, HuBERT km100 teacher extraction, StreamVC-style causal student, Mimi `q0` kill test, causal multi-speaker Unit-HiFi-GAN vocoder, target-voice fine-tuning, M3 Wait-*k* S2UT models.
Rationale, by section: [`Documents/BEST_PRACTICES.md`](../../Documents/BEST_PRACTICES.md) §4.1 (causality audit), §4.2 (Python↔C++ golden parity), §4.3 (conditioning-domain controls), §4.6 (Mode B training/streaming, BUG-027..032), §4.7 (gates measure what ships). Authoritative specs: `Documents/Milestone 2/` and `Documents/Milestone 3/` stage files.

1. **Strict Causality First:** Every streaming model and feature extractor must have bounded, explicit lookahead `L`. Never use non-causal pitch trackers (SwiftF0 has 176 ms lookahead; RMVPE and FCPE are non-causal) or non-causal conv padding/normalizations.
2. **Mode A Architecture (M2):**
   - **Pitch/Energy:** Causal YIN on a 40 ms past window (20 ms frame hop, 70 Hz high-pass pre-filter) emitting F0 (`thr=0.10`), CMND aperiodicity, soft voicing (`mean(0.05, 0.10, 0.15)`), and log-energy. Whiten log-F0/log-energy with EMA running stats and map to the target speaker's domain: `logf0_tgt = mu_tgt + scale * sigma_tgt * z + shift_st * ln(2) / 12`.
   - **Content Encoder Bake-off:** Train the 50 Hz (20 ms frame) StreamVC-style causal conv student (`L in {0, 1, 2}`) against HuBERT-Base (`facebook/hubert-base-ls960`, layer 7, km100). Block decoder gradients into the content encoder. Time-box the Mimi `q0` (12.5 Hz, 80 ms) kill test (PNMI + speaker probe) before spending GPU days on a Mimi vocoder.
   - **Causal Unit-Vocoder:** V2-class HiFi-GAN with causal left-padding `(k - 1) * d` and right-trimmed transposed convs (`trim_right_ratio = 1.0`), conditioned on discrete unit IDs, target-domain `[logf0_tgt, aperiodicity, voicing, log_energy]`, and FiLM speaker embeddings (`e_spk`). Apply NANSY-style pitch/formant/EQ perturbation before unit extraction during training.
3. **Mode B Simultaneous S2UT (M3):**
   - Feed Stage 2 with Stage 1's pre-quantization continuous `latents` (`[1, frames_per_step, D]` f32) and train bidirectional `En<->{Es,Fr,De}` with multi-task auxiliary source-ASR + target-S2TT CTC supervision (dropped at ONNX export).
   - Predict target-frame prosody (`f0_z, voicing, aperiodicity, energy_z`) in `translator_dec` and evaluate with `eval_s2ut.py` (Whisper ASR-BLEU/chrF++, SimulEval Average Lagging `AL`/`AL_CA`, ECAPA cosine, UTMOS).
4. **Objective Evaluation (`eval_vc.py` / `eval_s2ut.py`):** Evaluate all candidates on frozen manifests using Whisper large-v3 WER/CER/ASR-BLEU, SpeechBrain ECAPA-TDNN target/source cosine similarity & EER, UTMOS, F0 PCC, PNMI, speaker-leakage probe, and single-core CPU RTF (`torch.set_num_threads(1)` / ORT `intra_op = 1`).
5. **Licence & Environment Hygiene:** Pin dependencies with `uv.lock` (`uv sync --frozen`) and export `requirements.txt`. Keep default shipped bundles commercial-clean (MIT `HuBERT-Base`, MIT `mhubert_base_vp_en_es_fr_it3`, CVSS-C, CoVoST 2, Europarl-ST; keep NC `mHuBERT-147` / VoxPopuli behind opt-in research flags). Propagate `noncommercial: true` through every manifest/checkpoint/contract hop (BUG-031).
6. **Gates and parity checks measure the shipped runtime and can fail (MEMORY.md rule 11):** ORT stateful multi-hop timings, not PyTorch eager; relative tolerance on a non-silent output; every exporter fallback visible in the report; exit code independent of the console codepage (force UTF-8 on stdout/stderr). No bare `except Exception` fallbacks.
7. **Python↔C++ twins move together:** `fvml/waitk_policy.py` ↔ `src/pipeline/WaitKPolicy.hpp` + `UnitProsodyFifo.hpp` (rerun `make_waitk_golden.py`), `s2ut_golden.py` ↔ the C++ Mode B runner, `fvml/contract.py` ↔ `ModelContract.hpp`, `voices.json` validation ↔ both C++ `VoiceCatalog` parsers. Change a rule on both sides in the same change and regenerate the golden.
8. **Seed Praat before parselmouth resynthesis (MEMORY.md rule 14):** `parselmouth.praat.run("random_initializeWithSeedUnsafelyButPredictably(<int>)")` with a per-utterance seed (see `fvml/augment.py::perturb_speaker`).
