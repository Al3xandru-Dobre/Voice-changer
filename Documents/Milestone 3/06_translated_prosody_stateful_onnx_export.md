# Stage 06 — Translated prosody predictor & stateful ONNX export

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 05](05_causal_s2ut_translator_policy_gate.md) · [Stage 07 →](07_cpp_translator_runtime_offline_runner.md)

**Status:** ☐ not started
**Depends on:** Stage 02 (contract v2), Stage 05 (gated S2UT translator)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §5 prosody transfer across translation · §8 stateful ONNX export for encoder-decoder cross-attention

## Objective

Attach and fine-tune the **target-frame prosody predictor head** on the winning
Stage 05 translator so Mode B generates natural target-language pitch, voicing,
aperiodicity, and energy contours conditioned on the speaker's live vocal energy
and pitch expressiveness (`prosody_ctx`).

Then export the complete Mode A + Mode B model bundle (`encoder_s{1,2,4}.onnx`,
`translator_enc.onnx`, `translator_dec.onnx`, `vocoder_s{1,2,4}.onnx`,
`voices.json`, `langs.json`, `MODEL_CARD.md`) as **fixed-shape stateful ONNX
graphs** with ring-buffer KV caches, and prove 60-second multi-utterance parity
between ONNX Runtime and PyTorch.

## Context & references

- **Why Mode B needs its own prosody head (research §5.1):**
  - Target words have different syllable counts and stress patterns from source
    words spoken $k$ chunks earlier.
  - During `kFlush` (> 400 ms after the user stops speaking), live microphone
    audio is unvoiced silence (`voicing = 0`, low energy). Feeding live mic
    conditioning to the vocoder would whisper or mute the end of every translated
    sentence.
- **How source expressiveness still transfers (`prosody_ctx`):**
  - C++ computes a running 2-scalar summary of the user's recent voiced speech:
    `prosody_ctx = [utterance_mean_z_f0, utterance_mean_z_energy]`.
  - `translator_dec`'s prosody head takes the decoder hidden state + embedded
    target units + `prosody_ctx` and predicts per-20 ms-target-frame normalized
    prosody:
    `prosody` `[1, chunk_frames, 4]` = `[z_f0, voicing, aperiodicity, z_energy]`.
  - C++ maps `z_f0` and `z_energy` into the chosen target voice's range using
    `voices.json` ($\mu_{\text{tgt}}, \sigma_{\text{tgt}}$) + Stage 09's
    `pitch_shift` and `intonation` controls — the exact same equation as Mode A!
- **Ring-buffer KV export pitfalls (research §8.2):**
  - Never use Python-int slice indices for `src_pos` or `tgt_pos`; pass them as
    `int64` `[1]` state tensors and update ring slots via tensor indexing /
    `scatter` so `torch.onnx.export` traces dynamic position updates accurately.
  - Parity tests must run for $\ge 60\text{ s}$ (exceeding $5\times$ the
    $128 \times 80\text{ ms} = 10.24\text{ s}$ ring window) and include at least 4
    inter-utterance tail flushes (`reset()`).

## Tasks

- [ ] **Prosody predictor head (`fvml/translator.py`, `train_s2ut.py`):**
  - [ ] Add a 2-layer causal conv/linear prosody head on `translator_dec` taking
        `(dec_hidden, embed(pred_units), prosody_ctx)` → `[B, chunk_frames, 4]`.
  - [ ] Train/fine-tune (frozen or low-lr backbone, 30k steps) against the
        ground-truth target audio's Stage 03 YIN features (`z_f0` smooth-L1 on
        voiced frames, `voicing` BCE, `aperiodicity` L1, `z_energy` L1).
  - [ ] Verify expressiveness transfer: shifting `prosody_ctx` by $+1.0\,\sigma$
        raises the synthesized target speech's median log-F0 and energy
        monotonically without hurting ASR-BLEU.
- [ ] **Extend `python_training/export_stateful_onnx.py` for v2 bundles:**
  - [ ] Support `--role {encoder,translator_enc,translator_dec,vocoder}`.
  - [ ] Strip training-only `asr_ctc_head` and `s2tt_ctc_head` before export (keep
        `readiness_head` if CTC-Adaptive won Stage 05).
  - [ ] Wrap `translator_enc` and `translator_dec` so all self-attention KV rings
        (`enc_kv_*`, `dec_kv_*`), cross-attention memory rings (`mem_k_*`,
        `mem_v_*`), conv left-context states, and position tensors (`src_pos`,
        `tgt_pos`) are explicit `state_in.*` / `state_out.*` graph tensors with
        static shapes.
  - [ ] Run `onnx.shape_inference` + `onnx.checker.check_model(full_check=True)`.
  - [ ] Embed `metadata_props["fluxvoice.contract"]` (v2) and validate the full
        quartet with `fvml.contract.validate_s2ut_bundle`.
  - [ ] Write `langs.json`, updated `voices.json`, and `MODEL_CARD.md` into
        `models/<bundle>/`.
- [ ] **60-second multi-utterance parity (`python_training/parity.py` extended):**
  - [ ] Construct a 65-second multi-utterance stream (6 utterances separated by
        600 ms silence gaps, forcing 6 `kFlush` + `reset()` cycles and wrapping
        the 10.24 s KV ring when tested without reset).
  - [ ] Compare PyTorch streaming `step()` vs. ONNX Runtime CPU chunked execution:
    - `translator_enc` memory states: max $|\Delta| \le 10^{-5}$;
    - `translator_dec` teacher-forced unit logits: max $|\Delta| \le 10^{-4}$,
      unit argmax agreement $\ge 99.8\%$, prosody max $|\Delta| \le 10^{-4}$;
    - end-to-end free-running S2UT chain (`audio → encoder → translator_enc/dec →
      vocoder`): report unit agreement, waveform SNR on teacher-forced units, and
      ASR-BLEU parity on `test_es_en_200` and `test_en_de_200`.
- [ ] **Golden I/O vectors for C++ (Stage 07):**
  - [ ] Save a 15-second 2-utterance input WAV (with a 600 ms pause in the middle)
        for `es→en` and `en→de` plus the exact Python ORT step-by-step decisions,
        emitted unit IDs, prosody frames, and output WAV in
        `models/<bundle>/golden_s2ut/`.
- [ ] **Single-core ORT-Python timing:** measure per-step $p_{50}/p_{95}/\max$ on
      1 CPU thread (`intra_op_num_threads=1`) for `kRead` (`enc + trans_enc`) and
      `kWrite` (`enc + trans_enc + trans_dec + voc`).

## Implementation notes

- **Why test both ring-wrap (> 25 s continuous speech) and multi-utterance
  reset.** A user reading a long paragraph may speak for 25 seconds without a
  400 ms pause, wrapping the 128-chunk (10.24 s) sliding window twice. Parity must
  hold across the wrap boundary at $t = 10.24\text{ s}$ and $t = 20.48\text{ s}$
  with zero spike in $|\Delta\text{logit}|$.
- **Keep `mem_k_*` / `mem_v_*` layout identical between `translator_enc` and
  `translator_dec`.** Both graphs declare `[1, num_heads, max_src_steps, head_dim]`
  f32 so C++ binds the exact same `KVStateCache::Buffer` pointer as output of
  `translator_enc` and input of `translator_dec`.

## Files touched

- `python_training/fvml/translator.py`, `python_training/train_s2ut.py`,
  `python_training/export_stateful_onnx.py`, `python_training/parity.py`,
  `models/<bundle>/**` (generated, git-ignored), `models/README.md`

## Definition of done

- All exported ONNX files (`encoder_s{1,2,4}`, `translator_enc`, `translator_dec`,
  `vocoder_s{1,2,4}`) pass `onnx.checker` full check and
  `validate_s2ut_bundle`.
- Over both a 30 s continuous utterance (ring wrap $\times 3$) and a 65 s
  multi-utterance stream (6 flushes):
  - `translator_enc` max $|\Delta| \le 10^{-5}$;
  - `translator_dec` unit argmax agreement $\ge 99.8\%$ and prosody max $|\Delta|
    \le 10^{-4}$;
  - ORT-Python ASR-BLEU on `test_es_en_200` and `test_en_de_200` is within $\pm
    0.2$ BLEU of PyTorch.
- Single-core ORT-Python $p_{95}$ for a full `kWrite` step (`enc + trans_enc +
  trans_dec + voc`) is $\le 40\text{ ms}$ (50% of the 80 ms chunk).
