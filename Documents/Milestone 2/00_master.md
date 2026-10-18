# Milestone 2 — Textless VC Mode (Encoder + Unit-Vocoder)

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md) ·
> Research basis: [Textless VC research](../Research_documents/M2_TEXTLESS_VC_RESEARCH.md)

**Status:** ◐ in progress — Stage 02 done; Stage 01 harness and baseline done, training
corpora downloading (2026-10-18)
**Goal from README:** *Export causal encoder + custom Unit-HiFi-GAN to ONNX.
Run Mode A (voice changer without translation) in C++ with stateful KV/CNN
padding caches.*

## Why this milestone

Mode A is the product's first audible result: the user speaks, and the output is
the same words, with the user's intonation, in another voice, with as little
delay as possible. The goal is a **fluent stream**: low latency and no dropouts.

Research for this plan changed two assumptions in the earlier draft and the
README (research §2–§3):

- **The default encoder.** Mimi's semantic codebook runs at a fixed 80 ms frame.
  That alone uses most of the ~60–100 ms Mode A budget. Published
  voice-conversion tests also show that codebook loses most of the words on its
  own (WER > 100%). A StreamVC-style causal encoder at 50 Hz (20 ms frames)
  keeps hops at 20–40 ms. M2 therefore runs a **bake-off with a latency-first
  decision gate** (Stage 05) instead of assuming Mimi.
- **The pitch tracker.** SwiftF0, RMVPE and FCPE are not causal (SwiftF0 needs
  176 ms of lookahead). A causal YIN tracker replaces them.

The user also asked for **voice controls and presets**. The engine-side
parameter API, presets and CLI/console control are M2 work (Stage 09). The
ImGui sliders that drive them are Milestone 4.

## Placement & dependencies

```text
ML track:  01 ─► 03 ─► 04 ─► 05 (gate) ─► 06 ────────────────► 10 (own voice)
                 │                         │                      ▲
C++ track: 02 (contract) ─► 07 (ORT C++) ──┴─► 08 (RT) ─► 09 ─────┴─► 11 (verify)
```

- **Two tracks.** The model contract (02) and the ONNX Runtime plumbing (07) need
  no trained weights; they are built and tested against tiny fixture models. C++
  work can therefore proceed while the GPU trains (03–06).
- **Entry condition for Stage 08's live runs:** the
  [M1 / Stage 06](../Milestone%201/06_latency_verification.md) latency table is
  filled, or a documented cable workaround exists
  ([BUG-001](../Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)).
  The offline file runner (Stage 07) covers everything that needs no device.
- **Milestone 1.5 runs in parallel.** The neural pipeline only sees the SPSC
  queues. M2 provides `InferencePipeline::reset()` and the `LatencyGovernor`
  (Stage 08), and
  [M1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md)
  reuses them. Whichever lands second adapts.
- **Feeds Milestone 3:** the gate fixes the unit rate and vocabulary the
  translator must emit.
- **Feeds Milestone 4:** Stage 09's descriptor table is the spec for the voice
  sliders.

## Scope

**In scope**
- Pinned Python 3.12 training toolchain, datasets, evaluation harness
- Model contract v1 (ONNX metadata) + fixture models + validators (Python and C++)
- Causal pitch & energy features (YIN), Python reference + C++ port
- Content-encoder bake-off: HuBERT km100 teacher, StreamVC-style causal student
  (lookahead 0/1/2), Mimi q0 kill test
- Causal, stateful, multi-speaker, controllable unit vocoder; decision gate
- Fixed-shape stateful ONNX export + parity; `voices.json`
- ONNX Runtime C++ runtime (CPU EP), offline file runner
- Real-time Mode A: low-latency I/O, warm-up, latency governor, telemetry, worker priority
- Voice controls & presets: engine API, CLI, console commands
- Own-voice recording + fine-tune
- Latency / quality / soak verification

**Out of scope**
- GPU execution providers (CUDA, DirectML/WinML, CoreML), FP16/INT8, ImGui
  sliders, 24/48 kHz output or bandwidth extension, noise suppression, monitor
  mix → [Milestone 4](../Milestone%204/00_master.md)
- Hot-plug recovery, device switching, drift resampling, CI jobs that need ORT →
  [Milestone 1.5](../Milestone%201.5/00_master.md) (M2 keeps ORT-less builds green)
- Translator, Wait-k, multilingual units → [Milestone 3](../Milestone%203/00_master.md)
- Zero-shot cloning from reference audio; formant DSP; model packaging/installer

## Stages

| # | Stage | Research | Output | Status |
|---|---|---|---|---|
| 01 | [Python toolchain, datasets & evaluation harness](01_python_toolchain_eval_harness.md) | §6 §7 | `uv sync --frozen` env; manifests; `eval_vc.py` baseline numbers | ◐ env, lists, harness, baseline (WER 1.54 %) done; training corpora downloading |
| 02 | [Model contract & fixture models](02_model_contract_fixtures.md) | §8 §9 | `ModelContract.hpp` + `fvml/contract.py`; fixtures in `tests/data/` | ☑ |
| 03 | [Causal pitch & energy features](03_causal_pitch_energy.md) | §4 | YIN + whitening, Python ↔ C++ golden parity | ◐ code, golden vectors & dev-clean accuracy done; train-clean-100 priors pending |
| 04 | [Content encoder candidates](04_content_encoder_candidates.md) | §2 §7 | km100 teacher; causal student L∈{0,1,2}; Mimi kill decision | ◐ code, probes & streaming checks done; long GPU runs pending |
| 05 | [Causal vocoder & decision gate](05_causal_vocoder_decision_gate.md) | §5 §7 §11 | Multi-speaker vocoder; gate table; encoder decision | ◐ vocoder, trainer, export smoke & gate harness done; 150k-step training & quality gate pending |
| 06 | [Stateful ONNX export & parity](06_stateful_onnx_export.md) | §8 | Model bundle (s1/s2/s4) + `voices.json`; parity report | ☐ |
| 07 | [ONNX Runtime C++ runtime](07_ort_cpp_runtime.md) | §9 | `OrtModel`, CMake import + DLL copy, `fv_vc_file` | ☐ |
| 08 | [Real-time Mode A pipeline](08_realtime_mode_a_pipeline.md) | §3 §10 | `fluxvoice --vc` live; governor; telemetry | ☐ |
| 09 | [Voice controls & presets](09_voice_controls_presets.md) | §11 | `VoiceParams` descriptor table; presets; console | ☐ |
| 10 | [Target-voice recording & fine-tune](10_target_voice_finetune.md) | §6 | "Me" voice in the bundle | ☐ |
| 11 | [Verification & exit report](11_verification_exit.md) | §3 §7 | Measured latency/quality/soak; README pointers | ☐ |

## Latency budget

Below is fluxvoice-added T−B in ms (input → output, excluding the virtual cable)
at 16 kHz with 10 ms device periods. The values are **estimates** until Stage
08/11 measure them (research §3).

| Component | A0: 50 Hz, L=0, hop 20 | A1: L=1, hop 20 | A2: L=2, hop 20 | A1-40: L=1, hop 40 | M: Mimi q0, hop 80 @24 kHz |
|---|---|---|---|---|---|
| I/O floor (passthrough T−B) | 25 | 25 | 25 | 25 | 25 |
| Framing = hop | 20 | 20 | 20 | 40 | 80 |
| Encoder lookahead (L × 20) | 0 | 20 | 40 | 20 | 0 |
| Vocoder lookahead + OLA overlap (off) | 0 | 0 | 0 | 0 | 0 |
| Compute p99 (≤ 50% of hop) | 10 | 10 | 10 | 20 | 40 |
| Governor jitter margin | 5 | 5 | 5 | 5 | 10 |
| **fluxvoice-added T−B** | **60** | **80** | **100** | **110** | **155** |
| Algorithmic only (framing + lookahead) | 20 | 40 | 60 | 60 | 80 |

- Add about 20 ms if device periods stay at 20 ms. Add the overlap if OLA is
  enabled.
- **Must:** T−B ≤ 100 ms for the chosen configuration.
- **Stretch:** ≤ 80 ms.
- **Mimi:** ≤ 160 ms, and only if it is the sole candidate that passes the quality
  floor. That would be a recorded deviation from the README target.

## Acceptance criteria

- [ ] `fluxvoice --vc` converts live mic speech to a target voice, and measured
      T−B meets the budget's **Must** line for the gated configuration
- [ ] ORT chunked inference matches PyTorch offline (Stage 06 parity thresholds),
      and the C++ runner matches Python ORT (Stage 07)
- [ ] 30-min live VC soak: 0 deadline misses after warm-up, underruns flat,
      `queued` bounded, RSS growth ≤ 5 MB
- [ ] Converted speech intelligibility: offline WER ≤ 10% on the frozen list;
      live WER within +2 pts of offline
- [ ] Pitch shift, intonation scale, timbre blend, gain, gate and the built-in
      presets work live, each meeting its Stage 09 DoD
- [ ] The user's own recorded voice is selectable (`--voice me`)
- [ ] ORT-less builds and all existing CTest cases still pass (CI from M1.5 / Stage 01)

## Key constraints

1. **Milestone 1 real-time rules still hold.** The audio callback never runs
   inference, allocates, locks or logs. All inference runs on the worker thread.
2. **Preallocate everything.** That covers model I/O and state buffers, the
   IoBindings and the telemetry histograms. Steady-state hops allocate nothing in
   FluxVoice code.
3. **Warm up before `engine.start()`.** A slow first `Run` would become permanent
   latency (M1 / Stage 06 finding).
4. **One duplex rate.** Encoder input rate = vocoder output rate = device client
   rate (`cfg.sampleRate` is shared). The contract enforces it.
5. **`FLUXVOICE_WITH_ORT` guards every ONNX Runtime include and call.** The
   ORT-less build is the CI default.
6. **CPU execution provider, FP32 only.** GPU and quantization are M4.
7. **The training and run-time feature code is the same algorithm.** YIN,
   whitening and the F0 mapping are kept identical by golden vectors.
8. **Voice controls act on model conditioning, not on audio** (research §11).
   That way they add no latency.

## Open decisions

| # | Decision | Recommended default | Needed by |
|---|---|---|---|
| 1 | Gate rule | Latency-first rule as written in Stage 05: eligible on WER/similarity/UTMOS/compute, then lowest projected T−B | Stage 05 |
| 2 | Content kind | Discrete km100 ids; soft units only if discrete fails the WER floor | Stage 04 |
| 3 | Teacher | HuBERT-Base (English, MIT); mHuBERT-147 deferred to M3 (NC licence) | Stage 04 |
| 4 | Hop / lookahead | Chosen by the gate (expected: L=1 at 20 ms hops) | Stage 05 |
| 5 | Role of "own voice" | A target row ("Me") in the multi-speaker vocoder | Stage 10 |
| 6 | `python_training/requirements.txt` | Generated from `uv.lock` (`uv export`) for non-uv users — **adopted** (Stage 01) | Stage 01 |
| 7 | Encoder decision (filled by the gate) | — | Stage 05 |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Mimi q0 consumes weeks before failing | Kill test time-boxed to one week; PNMI + speaker probe before any vocoder training |
| Causal student too weak for intelligible VC | Lookahead variants L=1/2 in the gate; soft units as fallback; non-causal-teacher distillation noted for later |
| 8 GB laptop GPU | V2-class generator, bf16, resumable runs, one shared vocoder trained on teacher ids |
| Laptop thermal throttling on multi-day runs | Checkpoint every 30 min; cloud GPU optional, same lockfile |
| Residual speaker info in units | Leakage probe (Stage 04); perturbation augmentation (Stage 10); source-similarity metric in every evaluation |
| Windows loads System32's older `onnxruntime.dll` | DLL copied next to every exe; version logged and checked at load (Stage 07) |
| Slow first `Run`, or compute spikes, raise latency for good | Warm-up before start; `LatencyGovernor` gives excess back during silence (Stage 08) |
| Virtual-cable rig blocked (BUG-001/002) | In-process latency accounting + `fv_vc_file` algorithmic check; live numbers once a route works |
| Timbre blending between few speakers sounds unnatural | Marked experimental, with a Stage 09 DoD; presets use trained voices first |
| Stale claims (Mimi default, fixed 80 ms, RMVPE, "Normal" worker priority) mislead readers | Stub docstrings fixed in Stage 02; root README / RESEARCH_GUIDE pointers in Stage 11 |

## References

- [Textless VC research](../Research_documents/M2_TEXTLESS_VC_RESEARCH.md): encoder evidence (§2), latency (§3), pitch (§4), vocoder (§5), data (§6), evaluation (§7), export (§8), ORT (§9), scheduling (§10), controls (§11), licences (§12).
- `RESEARCH_GUIDE.md` §2–§4: Mimi, speech-resynthesis, StreamVoice/X-VC, sherpa-onnx stateful `Ort::Value` handling.
- [Milestone 1](../Milestone%201/00_master.md): real-time rules; [Stage 06](../Milestone%201/06_latency_verification.md) latency method and the "underruns are permanent" finding.
- `src/pipeline/InferencePipeline.cpp`, `src/pipeline/KVStateCache.hpp`, `python_training/*.py`: existing M2 placeholders (TODO(M2)).
