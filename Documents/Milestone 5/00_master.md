# Milestone 5 — Personal Voice Profiles (on-device enrolment from user recordings)

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md) ·
> Builds on: [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) (own-voice fine-tune) ·
> [M4 / Stage 03](../Milestone%204/03_voice_studio_translate_catalogue_panels.md) (Voice Studio)

**Status:** ☐ planned — stage files written 2026-11-18; nothing implemented
**Goal:** *A user hands the app a few recordings of a voice they have the right
to use. Entirely on their own machine, the app derives a small voice profile
from them and offers it as a target voice in Mode A (and in Mode B where the
bundle supports it). The recordings never leave the machine, the profile can be
inspected, exported and erased, and audio converted into a personal voice is
marked as synthetic.*

## Why this milestone

A personal voice exists today only as a developer recipe.
[M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) needs 1–3 hours
of studio-grade recordings, the Python training toolchain, a GPU fine-tune of
the vocoder and a re-export of a private bundle. The M2 master lists
"zero-shot cloning from reference audio" as out of scope. This milestone is
that scope: minutes of ordinary recordings, no training toolchain, CPU only,
seconds to minutes of processing.

Reading the code on 2026-11-18 produced two findings that shape the plan:

- **At run time a voice is already just data.** The vocoder takes `spk_emb`
  as a graph input (`[1, spk_emb_dim]` f32, `spk_emb_dim = 64`), and
  `voices.json` carries one embedding and four statistics per voice
  (`logf0_mean`, `logf0_std`, `energy_mean`, `energy_std`). A personal voice
  therefore needs no per-user model and no re-export: it is 64 numbers, four
  statistics and a record of where they came from.
- **Nothing can produce those 64 numbers from audio.** They are rows of a
  learned table with five speakers (`DEFAULT_SPEAKERS` in
  `python_training/fvml/vocoder.py`). ECAPA is used for evaluation and for
  choosing an initial row, never as conditioning. A five-point speaker space
  cannot represent a voice it was not trained on.

So the machine-learning work is one thing: a speaker encoder and a vocoder
trained over many speakers (Stage 04). Everything else is engineering around a
small file, and most of that engineering is about handling it responsibly.

**On "local and GDPR-compliant".** Local processing and data-protection
controls are necessary, and this plan makes both testable. They are not
sufficient. The person most at risk from this feature is not the user: it is
someone whose voice is enrolled without their agreement. The plan therefore
treats the consent of the *voice's owner* as the central requirement
(Stage 01), ships own-voice enrolment first, and marks the output (Stage 09).
It also says plainly what these controls cannot do: the application runs on
the user's machine from published source, so a determined user can remove
them. They set the default behaviour and create accountability; they do not
make misuse impossible.

## Verified starting point (2026-11-18)

| Item | State today | Where |
|---|---|---|
| Voice conditioning | `spk_emb` is a vocoder graph input; per-voice F0/energy mapping happens outside the graph | `python_training/fvml/contract.py` (`expected_inputs`), `src/pipeline/InferencePipeline.cpp` |
| Voice store | One `voices.json` (schema 1) per bundle; strict reader, unknown keys rejected | `src/pipeline/VoiceCatalog.hpp` |
| Speaker space | Learned table `nn.Embedding(5, 64)`; a new voice is a new row trained by fine-tuning | `fvml/vocoder.py` (`speaker_table`, `expand_speakers`) |
| Audio → embedding | ☐ none. ECAPA (192-d, VoxCeleb weights) is evaluation only | `fvml/speaker.py` |
| Own-voice ingestion | Python CLI, WAV only, room tone required, 60-minute minimum | `python_training/prepare_target_voice.py` |
| C++ audio file reading | Two hand-written RIFF parsers; no FLAC/MP3, no resampler. miniaudio's decoders are compiled into the implementation TU but unused | `frontend/core/VoiceStudio.cpp`, `tools/fv_vc_file.cpp`, `src/audio/MiniaudioBackend.cpp` |
| Causal YIN in C++ | ☑ per-frame tracker; no file-level statistics | `src/dsp/PitchTracker.hpp` |
| GUI flow | Voice Studio six-step wizard for the Stage 10 protocol; consent is a checkbox and a free-text note; no enrolment, no delete command | `frontend/core/VoiceStudio.*`, `frontend/core/AppModel.cpp` |
| User data location | Settings and Studio drafts under the config directory (`%APPDATA%` on Windows, which roams) | `frontend/core/Paths.cpp` |
| Network | No runtime network calls; `SDL_OpenURL` opens the system browser on a user click | `frontend/imgui/GuiApp.cpp` |
| Privacy controls | ☐ no consent record, export, erasure, encryption, marking or speaker check anywhere | — |
| Trained bundle | ☐ none yet (M2 Stages 04–06 long runs pending); `models/` holds a README | [M2 master](../Milestone%202/00_master.md) |

## Design in one page

**A voice profile is data, not a model.** Enrolment reads the user's files,
measures them, and writes one file:

```text
recordings (stay where the user keeps them; read-only, never copied)
   │ decode + resample → mono 16 kHz            RAM only
   │ quality checks, speech detection           RAM only
   ├─► speaker encoder (ORT, offline)  ──► embedding [64]
   └─► causal YIN (same code as live)  ──► logf0 / energy mean + std
                                             │
<local data dir>/fluxvoice/voices/<id>/profile.json
   embedding · statistics · consent record · provenance (hashes, QC numbers)
                                             │
bundle voices.json (public, immutable)  +  compatible profiles
                                             ▼
                       one voice table → Mode A / Mode B, unchanged graphs
```

The bundle stays public, immutable and commercially clean. Private data never
enters `models/`. A profile is tied to the bundle family that can interpret it
by a `speaker_space` identifier (Stage 02).

Three levels of fidelity share that one file format:

| Tier | Input | What runs on the device | Place in this milestone |
|---|---|---|---|
| **Instant** | ≥ 30 s of speech; 2–5 min recommended | decode → checks → speaker encoder → statistics | The product path: Stages 04, 06, 07 |
| **Refined** | 3–10 min | Instant, then the 64 values are optimised against the user's own clips with the vocoder frozen | Stage 05: a time-boxed experiment with an accept/reject rule |
| **Studio** | 1–3 h, GPU, Python toolchain | M2 Stage 10 fine-tune into a private bundle | Exists; M5 only gives it the same consent record |

**What a profile reproduces, and what it does not.** It approximates timbre
and pitch range. Words, rhythm, accent and speaking style come from the person
speaking into the microphone. The user interface says this in those terms and
never claims an identical copy.

## Placement & dependencies

```text
Policy:    01 privacy, consent & misuse ─────────► normative for 02, 07–10
                    │
ML track:  03 eval lists ─► 04 speaker encoder + vocoder (gate) ─► 05 refine (gate)
                                        │                               │
                                        └────────► 06 export & parity ◄─┘
                                                        │
C++ track: 02 profile contract & store ─► 07 enrolment engine + fv_enroll ─► 08 runtime
                                                        │                       │
                                          09 marking (gate) ───────────────────►┤
                                                                                ▼
                                                    10 GUI flow & Privacy centre ─► 11 exit
```

- **Two tracks again.** Stages 01, 02, 07 and 08 need no trained weights; they
  are built against fixture models and fakes, as M2 Stages 02 and 07 were.
- **The ML track waits for a trained M2 bundle.** Stage 03's baselines and
  Stage 04's training start from the M2 Stage 05 gate winner. Until then the
  harness is built and tested with fixtures and fakes.
- **Stage 10 needs the GUI to run Mode A**
  ([M4 / Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md)) and a green
  build gate ([M4 / Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md)).
  `fv_enroll` and `fluxvoice --vc --voice <profile>` do not.
- **Mode B** uses a profile only if the multilingual vocoder
  ([M3 / Stage 03](../Milestone%203/03_multilingual_encoder_vocoder_upgrade.md))
  is trained with the Stage 04 recipe. Recorded as a stretch goal.

## Scope

**In scope**
- Data inventory, consent model, misuse analysis and a data-protection impact
  record, written before code
- `profile.json` (schema 1), the `speaker_space` identifier, a `speaker_enc`
  model role, and a local profile store with export and erasure
- Unseen-speaker evaluation lists, a nearest-voice baseline, and a duration
  and degradation study
- A speaker encoder and a speaker-conditioned vocoder over many speakers, with
  a decision gate
- A time-boxed experiment on on-device refinement of the embedding
- `speaker_enc.onnx` export, Python ↔ C++ enrolment parity on synthetic
  fixtures
- C++ enrolment engine (file decoding, resampling, checks, statistics) and the
  `fv_enroll` command-line tool
- Profiles as selectable voices in `fluxvoice`, `fv_vc_file` and the GUI
- A decision on real-time marking of converted audio, a detector tool, and
  disclosure in the interface
- Quick Voice flow and a Privacy centre in the GUI
- A privacy audit with recorded evidence

**Out of scope**
- Any cloud processing, account, telemetry upload, crash upload or update
  check. A future network feature needs a new inventory row and its own
  consent
- Sharing, publishing or importing other people's personal profiles; a
  marketplace or catalogue of personal voices
- Speaker identification or recognition as a feature (one comparison, "is the
  person at the microphone the voice in these files", is the only use)
- Text-to-speech in a personal voice
- In-app recording of a full Stage 10 dataset; the Studio flow stays as it is
- Fixing the open Studio defects (BUG-021, BUG-022, BUG-025) →
  [M4 / Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md)
- GPU providers and quantised variants of the speaker encoder → follow
  [M4 / Stages 08–09](../Milestone%204/00_master.md) once those land

## Stages

| # | Stage | Owner persona | Output | Status |
|---|---|---|---|---|
| 01 | [Privacy, consent & misuse architecture](01_privacy_consent_misuse_architecture.md) | `@qa-verifier` + legal reviewer | Data inventory, consent model, threat table, impact record | ☐ |
| 02 | [Voice-profile contract & local store](02_voice_profile_contract_store.md) | `@onnx-ort-runtime` | `profile.json` schema 1, `speaker_space`, `ProfileStore`, fixtures | ☐ |
| 03 | [Unseen-speaker evaluation harness](03_unseen_speaker_eval_harness.md) | `@ml-speech-trainer` | Frozen enrolment lists, `eval_enroll.py`, baseline numbers | ☐ |
| 04 | [Speaker encoder & vocoder decision gate](04_speaker_encoder_vocoder_gate.md) | `@ml-speech-trainer` | Many-speaker vocoder, `SpeakerEncoder`, gate table | ☐ |
| 05 | [On-device embedding refinement](05_on_device_embedding_refinement.md) | `@ml-speech-trainer`, `@onnx-ort-runtime` | Accept/reject decision for the Refined tier | ☐ |
| 06 | [Speaker-encoder export & enrolment parity](06_speaker_encoder_export_parity.md) | `@onnx-ort-runtime` | `speaker_enc.onnx`, golden enrolment vectors | ☐ |
| 07 | [C++ enrolment engine & `fv_enroll`](07_cpp_enrolment_engine_cli.md) | `@rt-audio-dsp`, `@onnx-ort-runtime` | `src/enroll/**`, `AudioFileReader`, CLI | ☐ |
| 08 | [Runtime integration](08_runtime_profile_integration.md) | `@onnx-ort-runtime` | Merged voice table, `--voice <profile>`, add/erase while running | ☐ |
| 09 | [Synthetic-audio marking & disclosure](09_synthetic_audio_marking_disclosure.md) | `@onnx-ort-runtime`, `@ml-speech-trainer` | Marking decision, `fv_wm_detect`, disclosure spec | ☐ |
| 10 | [GUI: Quick Voice & Privacy centre](10_gui_quick_voice_privacy_centre.md) | `@ui-telemetry` | Enrolment flow, consent screens, export/erase | ☐ |
| 11 | [Verification, privacy audit & exit](11_verification_privacy_audit_exit.md) | `@qa-verifier` | Measured exit criteria, audit evidence | ☐ |

## Budgets

Targets until Stages 03, 07 and 09 measure them.

| Quantity | Must | Stretch |
|---|---|---|
| Speech needed for an Instant profile | 30 s accepted; quality stated honestly | Knee of the Stage 03 duration curve |
| Speech used at most | 10 min (longer input is sampled evenly) | — |
| Instant enrolment time, 3 min of audio, one core | ≤ 30 s | ≤ 10 s |
| Refined enrolment time (if accepted), four cores | ≤ 10 min, cancellable | ≤ 3 min |
| Profile size on disk | ≤ 16 KiB | — |
| Mode A cost of using a profile | 0: same graphs, same inputs | — |
| Mode A cost of marking | per-hop p99 ≤ 15 % of the hop | ≤ 5 % |
| Added latency | 0 ms: the M2 **Must** line (T−B ≤ 100 ms) still holds with marking on | — |
| Bytes sent over a network | 0 | — |

## Acceptance criteria

- [ ] A user creates a profile from their own recordings with `fv_enroll` and
      with the GUI, selects it, and converts live speech; the bundle on disk
      is byte-identical before and after
- [ ] On unseen speakers the Instant profile beats the nearest-voice baseline
      by ≥ 0.05 mean target cosine, no subgroup is below the baseline, WER
      ≤ 10 % and UTMOS ≥ 3.0, measured on the shipped runtime (Stage 04 rule,
      re-measured in Stage 11)
- [ ] The M2 latency budget and soak criteria hold with a profile selected and
      marking on; enrolling while converting causes 0 deadline misses
- [ ] Enrolment and conversion pass inside an empty network namespace, and the
      first-party binaries import no socket API
- [ ] Enrolment writes only inside the profile store, creates no temporary
      files, and leaves the source recordings unmodified
- [ ] No code path creates a profile without a recorded consent; a profile of
      another person requires evidence and is always marked
- [ ] Export returns every stored field as JSON; erasure leaves no residue in
      the application's directories; each is at most two actions from the
      profile
- [ ] Audio converted into a personal profile carries a detectable mark, or
      the recorded fallback applies (own verified voice only)
- [ ] The impact record is complete and the legal review row is PASS; until
      then the feature stays off in release builds
- [ ] ORT-less builds and all existing CTest cases pass on the four native legs

## Key constraints

1. **Real-time rules are unchanged.** Enrolment never runs on the audio
   callback or on the inference worker. The voice table changes only while the
   worker is quiesced.
2. **Local only.** No network code in first-party sources. This is checked by
   tests, not by intention.
3. **A profile is data.** Bundles are never modified. Private data never
   enters `models/` or a `voices.json`.
4. **Minimum data.** Recordings are read in place. Only `profile.json` is
   persisted. No temporary files. Buffers are released when the job ends.
5. **Consent before processing.** Recorded with its text version, as easy to
   withdraw as to give.
6. **No silent substitution.** An incompatible profile, an erased voice or a
   failed marker is an error the user sees. Marking fails closed.
7. **One algorithm, two implementations.** The Python and C++ enrolment paths
   are pinned by golden vectors, like YIN and the Wait-*k* policy.
8. **No real voices in the repository.** Fixtures and golden vectors are
   synthetic signals.
9. **Commercial-clean by default.** The shipped speaker encoder is trained
   only on CC0 / CC BY corpora recorded with the speakers' participation. No
   VoxCeleb-derived weights in a default bundle.
10. **Decisions are made on the shipped runtime**, with the acceptance rule
    written before the measurement.
11. **Honest labels.** An approximation is called an approximation. A
    nearest-voice fallback is never called the user's voice.

## Open decisions

| # | Decision | Recommended default | Needed by |
|---|---|---|---|
| 1 | Other people's voices in the first release | Own voice only. Another adult's voice only after Stage 09 accepts a marker, with evidence of consent, always marked | Stage 01 |
| 2 | Contract version for the `speaker_enc` role | `contract_version 3` = v2 plus the role and the `speaker_space` key (and a `marker` role if Stage 09 accepts one); v1 and v2 bundles load unchanged and report "no speaker encoder" | Stage 02 |
| 3 | Encryption of the profile store at rest | First release: per-user, non-roaming, non-synced directory with owner-only permissions. Passphrase-protected export as a stretch. Revisit if the legal review asks | Stage 02 |
| 4 | Input formats | WAV and FLAC required; MP3 on (decoder already compiled). Opus and AAC (phone voice memos) are a stretch: new dependencies and a patent/licence review | Stage 07 |
| 5 | Speaker-encoder design | Jointly trained StreamVC-style encoder with learned pooling; a frozen ECAPA projection only as a research comparison | Stage 04 |
| 6 | Generator width if the 1 M-parameter vocoder cannot hold many speakers | Decided by the Stage 04 gate against the compute budget | Stage 04 |
| 7 | How refinement runs on the device | Decided in Stage 05; ONNX Runtime's training packages are ruled out (deprecated) | Stage 05 |
| 8 | What is marked | All converted output if the cost allows; at minimum every personal profile. Legal review decides for bundle voices | Stage 09 |
| 9 | Retention | A review reminder after 12 months; never automatic deletion without the user | Stage 02, 10 |
| 10 | "Verified own voice" badge | Ships only if the encoder's verification error is ≤ 5 % EER; threshold calibrated at ≤ 1 % false accepts | Stage 04, 10 |
| 11 | Voices of minors | Not supported as third-party voices; wording reviewed by the legal reviewer | Stage 01 |
| 12 | Mode B with a profile | Stretch; needs the multilingual vocoder trained with the Stage 04 recipe | Stage 08 |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| A voice is enrolled without its owner's agreement | Own-voice default, live match for the verified badge, evidence for third parties, marking, no profile sharing. Residual risk stated in Stage 01; not claimed as solved |
| Zero-shot similarity from a small causal vocoder is poor | Gate against a baseline and a same-speaker ceiling; wider generator as an arm; refinement experiment; fallback is an honestly labelled "closest built-in voice" |
| No trained M2 bundle when the ML track is ready to start | Harness and C++ built on fixtures; baseline and gate tasks stay open rather than estimated |
| Many-speaker training exceeds the 8 GB laptop GPU schedule | Measure wall-clock per 10 k steps first; resumable runs; same lockfile on a rented GPU |
| Real user files are noisy, compressed, multi-speaker | Degradation study sets the thresholds; augmentation on the encoder input; consistency check warns about a second voice |
| C++ and Python enrolment drift apart | One reference function, golden vectors, parity threshold in CTest |
| A hostile audio file exploits a decoder | Bounded streaming decode, size and duration caps, malformed-header corpus, optional fuzz target |
| A marker costs too much or adds delay | Gate with the rule written first; "evaluated and rejected" is valid and triggers the own-voice-only fallback |
| A marker is removed or does not survive the call codec | Robustness measured through a VoIP codec; limits documented; never presented as a security boundary |
| Profile file copied by malware or another account | Owner-only permissions, non-roaming location, sync-folder warning; encryption decision 3 |
| A dependency starts making network calls | Import-table check and a no-network run in CI |
| Legal position is misjudged | Engineering reading is labelled as such; a named reviewer signs the impact record before release; the feature stays off until then |
| A bundle upgrade invalidates every profile | `speaker_space` mismatch is reported, never ignored; re-enrolment from the same files, verified by hash |

## References

Regulatory texts are summarised as an engineering reading, not legal advice;
the legal reviewer in Stage 01 confirms them before release.

- GDPR (Regulation (EU) 2016/679): Art. 2(2)(c) and Recital 18 (personal or
  household activity), Art. 4(14) (biometric data), Art. 5 (principles),
  Art. 7(3) (withdrawal), Art. 9 (special categories; explicit consent),
  Art. 15, 17, 20 (access, erasure, portability), Art. 25 and Recital 78
  (data protection by design; producers of applications), Art. 32 (security),
  Art. 35 (impact assessment).
- EU AI Act (Regulation (EU) 2024/1689), Art. 50(2) (providers mark synthetic
  audio in a machine-readable, detectable way) and Art. 50(4) (deployers
  disclose deep fakes); Art. 3(60) (definition). Applicable from 2 August
  2026; the Digital Omnibus on AI (in force 27 July 2026) defers Art. 50(2)
  to 2 December 2026 only for systems already on the market before 2 August
  2026. Sources checked 2026-11-18:
  [Sidley](https://datamatters.sidley.com/2026/06/24/eu-ai-act-transparency-obligations-preparing-for-compliance-by-2-august-2026/),
  [Lewis Silkin](https://www.lewissilkin.com/insights/2026/07/27/the-digital-omnibus-on-ai-enters-into-force-today),
  [CSA research note](https://labs.cloudsecurityalliance.org/research/csa-research-note-eu-ai-act-article-50-transparency-20260729/).
  Re-verify against the Official Journal before release.
- StreamVC (Yang et al., [arXiv 2401.03078](https://arxiv.org/abs/2401.03078)):
  speaker encoder with a learned pooling layer, decoder FiLM from the speaker
  latent, whitened F0. The model for Stage 04's recommended design.
- AudioSeal ([PyPI](https://pypi.org/project/audioseal/),
  [model card](https://huggingface.co/facebook/audioseal)): code and weights
  MIT since April 2024; streaming generator since 0.2. No independent latency
  figure found: Stage 09 measures it.
- ONNX Runtime on-device training
  ([docs](https://onnxruntime.ai/docs/get-started/training-on-device.html)):
  the training packages are marked deprecated in the v1.20.1 release notes.
  Not used.
- Internal: [M2 master](../Milestone%202/00_master.md) (budget, gate rule,
  constraints), [M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md)
  (export discipline), [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md)
  (protocol, preprocessing), [M4 master](../Milestone%204/00_master.md)
  (frontend constraints), [`DATASET/README.md`](../../DATASET/README.md)
  (licence policy), [`BEST_PRACTICES.md`](../BEST_PRACTICES.md).
