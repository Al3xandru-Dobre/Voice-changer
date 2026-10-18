# Milestone 3 — Wait-k S2UT Mode (Simultaneous Translation + VC)

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md) ·
> Research basis: [Simultaneous S2UT research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md)

**Status:** ☐ not started — stage files written 2026-10-18
**Goal from README:** *Train/fine-tune causal S2UT transformer on discrete
units. Implement `WaitKPolicy` in C++ and connect between encoder and
vocoder.*

## Why this milestone

Mode B adds simultaneous speech-to-speech translation on top of Mode A's real-time
voice conversion: the user speaks in one language (`en`, `es`, `fr`, or `de`),
and after a short Wait-*k* buffering delay (~250–490 ms depending on $k$), the
listener hears the translated speech in the target language (`en`, `es`, `fr`, or
`de`) rendered in the chosen target voice (including the user's own "Me" voice).

Research for this milestone ([`M3_WAITK_S2UT_RESEARCH.md`](../Research_documents/M3_WAITK_S2UT_RESEARCH.md))
settled four architectural choices that refine the earlier draft and the root
README:

1. **Direct S2UT on shared Stage 1 continuous latents (`En↔{Es,Fr,De}`).**
   Instead of feeding hard-quantized source unit IDs into the translator (which
   discards cross-lingual phonetic nuance) or running a separate acoustic
   encoder in Mode B, Milestone 3 upgrades the M2 50 Hz causal conv encoder to
   emit both discrete `units` (for Mode A) and speaker-disentangled continuous
   `latents` `[1, frames_per_step, D]` f32 (for Mode B). One joint translator
   model covers all 6 bidirectional pairs (`es↔en`, `fr↔en`, `de↔en`) via
   `<src_lang>` and `<tgt_lang>` tokens.
2. **Multi-task CTC supervision during training, 100% textless at run time.**
   Published CVSS-C benchmarks show that single-pass S2UT trained only on unit
   cross-entropy collapses on high word-order reordering (`De→En` scores 2.46
   ASR-BLEU), whereas adding auxiliary source-ASR and target-text CTC heads during
   training (StreamSpeech recipe, research §2.3) raises `De→En` to 19.72 BLEU and
   `Fr/Es→En` to 26–28 BLEU. The auxiliary text heads are dropped (or reduced to a
   1-scalar readiness score) at ONNX export.
3. **80 ms translation chunks ($C = 4$ frames at 50 Hz) and a two-graph split
   (`translator_enc.onnx` + `translator_dec.onnx`).**
   Stepping `WaitKPolicy` every 80 ms chunk (4 frames of 20 ms) keeps `k ∈
   [1..10]` equal to 80–800 ms of initial wait, shortens transformer KV caches by
   $4\times$, and lets `kRead` steps run only `translator_enc` while `kFlush`
   steps run only `translator_dec` over static ring-buffer KV caches.
4. **Translator-predicted prosody + tail-flush gate override.**
   In Mode B, translated words do not align frame-for-frame with the live
   microphone, and during `kFlush` (> 400 ms after the user stops speaking) the
   live microphone is silent while the translator emits the end of the translated
   sentence. `translator_dec` therefore predicts per-target-frame normalized
   prosody `(f0_z, voicing, aperiodicity, energy_z)` mapped into the target
   voice's domain (`voices.json`), and the output noise gate stays open during
   `kFlush` until the translated tail finishes draining.

## Placement & dependencies

```text
ML track:   01 (data/eval) ─► 03 (multilingual enc/voc) ─► 05 (S2UT & policy gate) ─► 06 (prosody & export) ────────► 10 (x-lingual "Me")
                                                                                           │                              ▲
C++ track:  02 (contract v2) ─► 04 (WaitKPolicy & FIFO) ─► 07 (C++ S2UT runner) ───────────┴─► 08 (RT Mode B) ─► 09 ──────┴─► 11 (verify)
```

- **Depends on Milestone 2:**
  - [M2 / Stage 02](../Milestone%202/02_model_contract_fixtures.md) (`ModelContract.hpp` / `fvml/contract.py`, extended here to contract v2);
  - [M2 / Stage 03](../Milestone%202/03_causal_pitch_energy.md) (`PitchTracker`, `RunningNorm`, target-domain F0/energy mapping);
  - [M2 / Stage 04–05](../Milestone%202/05_causal_vocoder_decision_gate.md) (50 Hz causal conv encoder + causal multi-speaker Unit-HiFi-GAN vocoder);
  - [M2 / Stage 07–09](../Milestone%202/08_realtime_mode_a_pipeline.md) (`OrtModel`, `InferencePipeline`, `LatencyGovernor`, `VoiceParams`).
- **Two parallel tracks within M3:** Stages 02, 04 and 07 (contract v2,
  `WaitKPolicy` state machine, two-session `OrtModel` plumbing, and deterministic
  fixture models) need no trained S2UT weights and can land in C++ while Stages
  01, 03, 05 and 06 prepare data and train on GPU.
- **Feeds Milestone 4:** Stage 09's S2UT parameter descriptors (`translate`,
  `wait_k`, `src_lang`, `tgt_lang`) and `langs.json` drive the Milestone 4 ImGui
  translation toggle, Wait-*k* slider, and language dropdowns without further
  engine changes.

## Scope

**In scope**
- Bidirectional `En↔{Es,Fr,De}` (6 directions) training & evaluation pipeline
  (CVSS-C, CoVoST 2 + canonical TTS, Europarl-ST, CSS10, CML-TTS)
- `eval_s2ut.py`: Whisper large-v3 ASR-BLEU, ASR-chrF++, SimulEval Average
  Lagging (`AL` and `AL_CA` in ms), ECAPA speaker cosine, UTMOS, tail-flush
  hallucination checks
- Model contract v2 (`translator_enc`, `translator_dec`, `langs.json`, ring-KV
  state tensors, prosody outputs) + deterministic ONNX translator fixtures
- Multilingual unit codebook ($K=1000$ on `mhubert_base_vp_en_es_fr_it3` MIT vs.
  `mHuBERT-147` probe), dual-output 50 Hz causal student (`latents` f32 + `units`
  i64), and multilingual `{en,es,fr,de}` causal Unit-HiFi-GAN vocoder
- Upgraded `WaitKPolicy` (50 Hz chunk ratio, bounded tail flush, `<EOS>` early
  stop, mid-flush speech resume, adaptive readiness hook) + unit/prosody FIFO
- Causal Conformer-Transformer S2UT model trained with multi-task ASR/S2TT CTC
  supervision; decision gate comparing Strict Wait-*k* vs. CTC-Adaptive policy
  (plus recorded Hibiki kill analysis)
- Translator prosody predictor head + fixed-shape ring-KV stateful ONNX export
  (`translator_enc.onnx`, `translator_dec.onnx`) + 60 s multi-utterance parity
- Two-session C++ runtime in `OrtModel` / `InferencePipeline` + `fv_s2ut_file`
  offline WAV→WAV runner
- Real-time Mode B (`fluxvoice --s2ut`), `kRead` vocoder bypass, `kFlush` gate
  override, Mode B `LatencyGovernor` burst cushion, S2UT telemetry
- Live CLI/console controls for Mode A↔B switching, `--wait-k`, `--src-lang`,
  `--tgt-lang`
- Cross-lingual "Me" voice validation and fine-tuning across all 6 directions
- End-to-end latency, Average Lagging, quality and 30-min soak verification

**Out of scope**
- GPU execution providers (CUDA, WinML/DirectML, CoreML), FP16/INT8 quantization,
  and Dear ImGui controls → [Milestone 4](../Milestone%204/00_master.md)
- Unwritten languages or pairs outside `En↔{Es,Fr,De}` in the default bundle
  (the architecture and `--src-lang`/`--tgt-lang` contract remain open to adding
  more language tags later)
- Multi-speaker diarization / overlapping cross-talk translation

## Stages

| # | Stage | Research | Output | Status |
|---|---|---|---|---|
| 01 | [Bidirectional datasets & SimulEval harness](01_multilingual_data_simuleval_harness.md) | §6 §7 §10 | `prepare_s2ut_data.py`; 6-pair eval lists; `eval_s2ut.py` (ASR-BLEU, AL) | ☐ |
| 02 | [Translator contract v2 & fixture models](02_translator_contract_v2_fixtures.md) | §4 §8 | Contract v2 in `ModelContract.hpp` & `fvml/contract.py`; translator fixtures | ☐ |
| 03 | [Multilingual encoder & vocoder upgrade](03_multilingual_encoder_vocoder_upgrade.md) | §3 | `mhubert-vp` km1000; dual-output student (`latents` + `units`); `{en,es,fr,de}` vocoder | ☐ |
| 04 | [`WaitKPolicy` & streaming unit/prosody FIFO](04_waitk_policy_streaming_state_machine.md) | §4 §9 | Upgraded `WaitKPolicy.hpp` + `UnitProsodyFifo.hpp`; `test_waitk_policy` | ☐ |
| 05 | [Causal S2UT translator & policy decision gate](05_causal_s2ut_translator_policy_gate.md) | §2 §4 | Multi-task CTC Conformer-Transformer; Wait-*k* vs. Adaptive gate table | ☐ |
| 06 | [Translated prosody & stateful ONNX export](06_translated_prosody_stateful_onnx_export.md) | §5 §8 | Prosody head; `translator_enc.onnx` + `translator_dec.onnx` + `langs.json`; parity | ☐ |
| 07 | [C++ translator runtime & `fv_s2ut_file`](07_cpp_translator_runtime_offline_runner.md) | §8 §9 | Shared `mem_kv` `OrtModel` execution; `fv_s2ut_file` WAV→WAV runner | ☐ |
| 08 | [Real-time Mode B pipeline & governor](08_realtime_mode_b_pipeline.md) | §4 §5 §9 | `fluxvoice --s2ut` live; `kRead` bypass; `kFlush` gate override; telemetry | ☐ |
| 09 | [S2UT controls, language switching & Mode A↔B toggle](09_s2ut_controls_language_switching.md) | §9 | `VoiceParams` S2UT extensions; live `mode`/`lang`/`waitk` console commands | ☐ |
| 10 | [Cross-lingual target-voice transfer ("Me" voice)](10_cross_lingual_target_voice.md) | §3 §5 | `--voice me` speaking `Es/Fr/De` and `En` across all 6 directions | ☐ |
| 11 | [Verification & exit report](11_verification_exit.md) | §4 §7 | Measured onset $T-B$, AL, 6-pair ASR-BLEU, 30-min soak; README updates | ☐ |

## Latency budget (Mode B)

Below is the FluxVoice-added first-audio onset latency $T - B$ in ms (speech
onset at input → first translated audio at output, excluding the virtual cable)
at 16 kHz with 10 ms device periods and 80 ms translation chunks ($C = 4 \times
20\text{ ms}$ frames). Simultaneous Average Lagging ($\text{AL}$) across full
utterances is tracked separately in Stages 05 and 11 (research §4).

| Component | $k=2$ (160 ms wait) | $k=3$ (240 ms wait, default) | $k=4$ (320 ms wait) | $k=5$ (400 ms wait, De↔En) |
|---|---:|---:|---:|---:|
| I/O floor (passthrough $T - B$) | 25 | 25 | 25 | 25 |
| Encoder lookahead ($L = 1 \times 20\text{ ms}$) | 20 | 20 | 20 | 20 |
| Wait-*k* buffering ($k \times 80\text{ ms}$, incl. framing) | 160 | 240 | 320 | 400 |
| Compute $p_{99}$ (Enc + Trans Enc/Dec + Voc, $\le 50\%$ of 80 ms) | 35 | 35 | 35 | 35 |
| Governor jitter margin | 10 | 10 | 10 | 10 |
| **FluxVoice Mode B onset $T - B$** | **250** | **330** | **410** | **490** |
| Algorithmic onset latency ($k \times 80 + L \times 20$) | 180 | 260 | 340 | 420 |

- **Must (onset $T - B$):** $\le 500\text{ ms}$ at the default $k$ ($k \in \{3,
  4\}$), matching the README's ~300–500 ms target.
- **Must (compute $p_{99}$):** Combined Stage 1 (4 frames) + Stage 2
  (`translator_enc` + `translator_dec`) + Stage 3 (4 frames) $p_{99} \le
  40\text{ ms}$ on one CPU core.
- **Must (Average Lagging $\text{AL}$):** SimulEval speech $\text{AL} \le
  1200\text{ ms}$ on `En↔{Es,Fr}` at $k=3$ and $\le 1500\text{ ms}$ on
  `En↔De` at $k=5$.

## Acceptance criteria

- [ ] `fluxvoice --s2ut --src-lang <L> --tgt-lang <L>` translates live microphone
      speech across all 6 `En↔{Es,Fr,De}` directions with measured onset $T - B
      \le 500\text{ ms}$ at default $k$
- [ ] Offline translation quality on the frozen 200-utterance test sets meets the
      Stage 05 gate floors (ASR-BLEU $\ge 22$ on `Fr/Es→En`, $\ge 16$ on
      `De→En`, $\ge 18$ on `En→{Es,Fr}`, $\ge 14$ on `En→De`; UTMOS $\ge 3.0$),
      and live-cable ASR-BLEU is within $-2.0$ BLEU of offline
- [ ] Stateful chunked ORT (`translator_enc` + `translator_dec`) reproduces
      PyTorch streaming over $\ge 60\text{ s}$ multi-utterance streams (unit
      argmax agreement $\ge 99.5\%$, prosody max $|\Delta| \le 10^{-4}$), and
      `fv_s2ut_file` matches Python ORT
- [ ] Tail-flush (`> 400\text{ ms}` silence) drains remaining translated speech
      completely without muting or clipping the sentence tail, then resets
      cleanly on `<EOS>` or `max_flush_steps` with zero silence-hallucination
      babble
- [ ] Mode A (`--vc`) on the upgraded multilingual encoder + vocoder bundle does
      not regress: English test-clean WER remains $\le 10\%$ and within $+1.0$ pt
      of the M2 exit number
- [ ] The user's own voice (`--voice me`) works across all 6 translation
      directions, preserving target speaker cosine similarity within $0.05$ of
      same-language English VC
- [ ] Live switching between Mode A and Mode B (`mode vc|s2ut`), language pairs
      (`lang <src> <tgt>`), and `waitk <1..10>` executes without audio clicks,
      allocations, or worker deadline misses
- [ ] 30-min live Mode B soak (`analyze_soak.py --s2ut`): 0 deadline misses after
      warm-up, underruns flat, `queued` bounded within target + burst cushion,
      RSS growth $\le 5\text{ MB}$
- [ ] ORT-less builds and all M1 / M1.5 / M2 CTest suites remain green

## Key constraints

1. **All Milestone 1 & 2 real-time rules still hold.** Zero allocations, locks, or
   logging in the audio callback; zero steady-state heap allocations in FluxVoice
   worker code; global ORT thread pool with `intra_op = 1`; session warm-up
   before `engine.start()`.
2. **Static shapes via ring-buffer KV caches.** Both `translator_enc.onnx` and
   `translator_dec.onnx` use fixed-capacity ring KV caches ($W_{\text{src}} =
   W_{\text{tgt}} = 128$ chunks = 10.24 s) indexed by tensor state counters
   (`src_pos`, `tgt_pos`). No dynamic sequence growth in ONNX Runtime.
3. **Zero wasted vocoder compute during `kRead`.** When `WaitKPolicy::decide()`
   returns `kRead`, only Stage 1 and `translator_enc` run; `translator_dec` and
   Stage 3 (`vocoder`) are skipped and the pipeline emits silence directly.
4. **Output gate follows translated speech, not live mic energy, in Mode B.**
   During `kWrite` and `kFlush`, the output noise gate is driven by predicted
   target voicing/energy and active unit emission so translated tails spoken
   during user pauses are never gated off.
5. **Commercial-clean default bundle.** The default shipped bundle uses MIT /
   CC-BY-4.0 / CC0 teachers and datasets (`mhubert-vp`, CVSS-C, CoVoST 2,
   Europarl-ST, CSS10, CML-TTS). Non-commercial assets (`mHuBERT-147`,
   VoxPopuli) are opt-in research flags only.

## Open decisions

| # | Decision | Recommended default | Needed by |
|---|---|---|---|
| 1 | Multilingual unit teacher | `mhubert_base_vp_en_es_fr_it3` L11 km1000 (MIT) with `{en,es,fr,de}` balanced centroids; `mHuBERT-147` (NC) probed for comparison | Stage 03 |
| 2 |Simultaneous policy winner | Chosen by Stage 05 gate: Strict Wait-*k* vs. CTC-Adaptive Wait-*k* (both supported by `WaitKPolicy`) | Stage 05 |
| 3 | Target unit emission format | Stacked $C = 4$ frames (80 ms) per decoder step (default) vs. deduplicated units + duration predictor | Stage 05 |
| 4 | Default Wait-*k* setting | $k = 3$ (240 ms wait) for `En↔{Es,Fr}`; $k = 4$ or $5$ (320–400 ms) for `En↔De` | Stage 05 |
| 5 | `En→{Es,Fr,De}` target training audio | Canonical multi-speaker TTS synthesis on CoVoST 2 / Europarl-ST text (matching CVSS-C's construction) | Stage 01 |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Single-pass S2UT fails to converge on `De↔En` verb reordering | Multi-task auxiliary ASR CTC + target text S2TT CTC losses during training (StreamSpeech recipe, Stage 05) |
| Target Romance/German phrases have 15–25% more syllables than English source | Evaluate Adaptive CTC-Wait-*k* (0/1/2 target chunks per step) and learned duration scaling in Stage 05; drain remainder in `kFlush` |
| Tail flush is muted because user stopped speaking into the mic | Gate override in Mode B: output gate tracks `WaitKPolicy` active output + predicted target energy, not live mic energy (Stage 08) |
| Translator hallucinates or loops at end of utterance during `kFlush` | Supervise `<EOS>` token + `eos_prob` head on silence-padded training tails; hard cap `max_flush_steps` in `WaitKPolicy` (Stages 04–06) |
| Ring-buffer KV cache wraps or leaks stale tokens across utterances | Explicit `src_pos` / `tgt_pos` tensor masks tested over > 60 s (6× window) in Stage 06; full state zeroing on `WaitKPolicy` flush completion |
| Combined Stage 1 + 2 + 3 exceeds single-core 80 ms hop budget | Run Stage 1 + 3 in `s4` (80 ms batch) in Mode B; skip decoder + vocoder on `kRead`; cap translator at ~30–40M params (`d_model=256/384`) |
| Multilingual vocoder slightly degrades English Mode A WER | Oversample English (40% of batches) and check Mode A WER regression ($\le +1.0$ pt) in Stage 03 before proceeding |

## References

- [Simultaneous S2UT research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) — policy landscape (§2), multilingual units (§3), 50 Hz latency budget (§4), prosody transfer (§5), datasets (§6), evaluation (§7), ring-KV export (§8), C++ runtime (§9), licences (§10).
- `RESEARCH_GUIDE.md` §1 & §5 — StreamSpeech, Hibiki, SeamlessStreaming, Fairseq Direct S2UT, CVSS, VoxPopuli.
- [Milestone 2](../Milestone%202/00_master.md) — Model contract v1, `OrtModel`, `PitchTracker`, `InferencePipeline`, `LatencyGovernor`, `VoiceParams`.
- `src/pipeline/WaitKPolicy.hpp`, `src/pipeline/InferencePipeline.cpp`: existing M3 state-machine implementation and pipeline hooks.
