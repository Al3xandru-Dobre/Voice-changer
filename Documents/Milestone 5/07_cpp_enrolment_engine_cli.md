# Stage 07 — C++ enrolment engine & `fv_enroll`

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 06](06_speaker_encoder_export_parity.md) · [Stage 08 →](08_runtime_profile_integration.md)

**Status:** ☐ not started
**Depends on:** [Stage 02](02_voice_profile_contract_store.md) (`VoiceProfile`, `ProfileStore`),
[Stage 06](06_speaker_encoder_export_parity.md) (`enroll_algo = 1`, golden vectors, fixture `speaker_enc`),
[M2 / Stage 07](../Milestone%202/07_ort_cpp_runtime.md) (`OrtModel`),
[Stage 01](01_privacy_consent_misuse_architecture.md) (inventory rules)
**Owner persona:** `@rt-audio-dsp` (file reader, checks, DSP) with `@onnx-ort-runtime` (model step, job)

## Objective

Turn a list of audio files into a `profile.json` in C++, with the same result
as the Python reference, without writing anything except that file, and never
touching the audio callback or the inference worker. Provide a command-line
tool so the feature is usable and testable before the GUI exists.

## Context & verified starting point (2026-11-18)

- C++ reads audio files in two places, both hand-written RIFF parsers:
  `analyze_wav_file` in `frontend/core/VoiceStudio.cpp` and `read_mono_wav`
  in `tools/fv_vc_file.cpp`. Neither resamples; `fv_vc_file` asks the user to
  resample first. [M4 / Stage 03](../Milestone%204/03_voice_studio_translate_catalogue_panels.md)
  records that a damaged header can request a multi-gigabyte allocation and
  that non-finite samples pass as clean.
- miniaudio's WAV, FLAC and MP3 decoders are compiled into
  `src/audio/MiniaudioBackend.cpp` (no `MA_NO_DECODING`) and unused.
  `<miniaudio.h>` must not leak outside `src/audio/`, and each binary has
  exactly one `MINIAUDIO_IMPLEMENTATION` translation unit.
- `src/dsp/PitchTracker.hpp` is the causal YIN used live; it yields
  per-frame values and no file-level statistics.
- `OrtModel` loads a contract-validated model; `ModelStep` is the interface
  the tests fake (`tests/FakeModelStep.hpp`).

## Design

```text
src/audio/AudioFileReader.{hpp,cpp}   decode + convert, miniaudio hidden
src/enroll/AudioQc.hpp                per-file measurements and verdicts
src/enroll/SpeechSelect.hpp           speech detection (twin of the Python rules)
src/enroll/Enroll.{hpp,cpp}           enroll_algo = 1: windows, pooling, statistics
src/enroll/SpeakerEncoderStep.hpp     interface; OrtModel-backed and fake
src/enroll/EnrollJob.{hpp,cpp}        thread, progress, cancel, consent check
tools/fv_enroll.cpp                   command-line front end
```

- **`AudioFileReader`**: streaming decode in fixed-size blocks to mono f32
  at the requested rate. Typed errors (`Unreadable`, `Unsupported`,
  `TooLong`, `TooLarge`, `NonFinite`). It never trusts a header-derived size:
  it allocates per block and stops at the caps. Decode-only binaries use a
  separate implementation unit built with `MA_NO_DEVICE_IO`, so `fv_enroll`
  links no audio-device code.
- **Caps:** per file ≤ 60 minutes and ≤ 2 GiB; sample rate 8–192 kHz;
  ≤ 8 channels; at most 64 files per job. A cap hit is a visible error.
- **`AudioQc`**: the measurement code moves down from
  `frontend/core/VoiceStudio.cpp` so the GUI and the CLI share one
  implementation. Per file: duration, speech seconds, peak, RMS, noise
  floor, SNR estimate, clipped samples, verdict.
- **Thresholds** come from the Stage 03 degradation study, not from the
  Studio protocol: minimum SNR and minimum speech are the measured values;
  until then 20 dB and 30 s are placeholders marked as such in the code.
- **Second-voice warning:** from the consistency scores of `enroll_algo = 1`.
  A file whose mean is far from the others, or a spread of window scores
  beyond a calibrated limit, produces "these recordings may contain more
  than one voice" with the file names. It warns; the user decides.
- **`EnrollJob`**: owns a normal-priority thread. Inputs: file list, a
  consent record, a name, a bundle. It refuses to start without a valid
  consent record. Progress and cancellation are atomics. On completion it
  returns `Result<VoiceProfile>`; the caller saves it through `ProfileStore`.
  On cancel or error nothing is written. It has its own `Ort::Session`
  options (one or two threads) and never shares state with the worker.

### `fv_enroll`

```text
fv_enroll analyze <files…>                                   checks only; writes nothing
fv_enroll create  --model-dir <bundle> --name <name> [--display <text>]
                  --subject self|other --accept-consent <text-version>
                  [--consent-evidence <file>] [--profile-dir <dir>] <files…>
fv_enroll list    [--profile-dir <dir>] [--json]
fv_enroll show    <name> [--json]
fv_enroll check   <name> --model-dir <bundle>               speaker_space compatibility
fv_enroll export  <name> <directory>
fv_enroll erase   <name> [--yes]
fv_enroll --show-consent <text-version>
```

- `create` without `--subject` and `--accept-consent` exits non-zero and
  writes nothing. `--subject other` requires `--consent-evidence` and is
  refused while open decision 1 says own voice only.
- The verified badge needs a microphone and is granted in the GUI
  (Stage 10); a CLI profile is unverified and therefore `marking_required`.
- Exit codes are distinct per failure class and independent of the console
  code page. Output never includes embedding values unless `show --json` is
  asked for.

## Tasks

- [ ] `AudioFileReader` over `ma_decoder`, plus the decode-only
      implementation unit; WAV and FLAC required, MP3 enabled (open
      decision 4). Opus and AAC are refused with a message naming the
      formats that work.
- [ ] **Resampler tolerance.** Compare embeddings from files resampled by
      the C++ path and by the Python path (soxr) on the Stage 03 list:
      cosine ≥ 0.995 and a `tgt_cos` change ≤ 0.005. If it fails, add a
      windowed-sinc resampler in `src/dsp/` (offline use only) and repeat.
- [ ] Move the measurement code to `src/enroll/AudioQc.hpp`; `VoiceStudio`
      calls it. Fix the two recorded defects on the way: bounded sizes and
      non-finite samples rejected.
- [ ] `SpeechSelect.hpp`, `Enroll.{hpp,cpp}` as ports of the Stage 06
      reference. File-level statistics from `PitchTracker` with the live
      configuration.
- [ ] `SpeakerEncoderStep` with an `OrtModel` implementation (guarded by
      `FLUXVOICE_WITH_ORT`, `Ort::Exception` caught inside) and
      `tests/FakeSpeakerEncoder.hpp`.
- [ ] `ModelBundle`: locate and validate the optional `speaker_enc.onnx`;
      `kNoSpeakerEncoderMessage` when absent; `speaker_space` equality.
- [ ] `EnrollJob` and `tools/fv_enroll.cpp`; CMake targets
      `fluxvoice_enroll` and `fv_enroll`; platform libraries per OS in CMake.
- [ ] Tests, hardware-free, in CTest:
  - [ ] `test_audio_file_reader`: formats, channel mixing, rates; a corpus
        of malformed files (truncated, absurd sizes, zero channels, wrong
        chunk order) each ending in a typed error with bounded memory
        (`AllocationCounter` peak);
  - [ ] `test_enroll_qc`, `test_speech_select`: golden twins of the Python
        rules;
  - [ ] `test_enroll`: every golden vector from Stage 06 with the fake and,
        in the ORT build, with the fixture `speaker_enc`: embedding cosine
        ≥ 0.9999, statistics within 1e-6 relative, identical refusals;
  - [ ] `test_enroll_job`: no consent → refused; cancel → nothing written;
        the directory tree outside the profile store is identical before and
        after (snapshot under a temporary home); source files unchanged by
        hash;
  - [ ] `check_enroll_cli.cmake`: every error path of `fv_enroll`, and the
        ORT-less build reporting `kBuiltWithoutOrtMessage`.
- [ ] Optional: a libFuzzer target for `AudioFileReader` on the Clang leg,
      not part of the required gate.
- [ ] Measure and record: enrolment time for 3 minutes of audio on one core;
      peak memory; a run of Mode A with an enrolment in progress (deadline
      misses must stay 0).

## Implementation notes

- **Not real-time code, but next to it.** The job may allocate and block. It
  must not take the worker's thread priority, share its session or touch its
  queues. The concurrency check above is the proof.
- **No temporary files.** Decoding and resampling are streamed through
  memory. Ten minutes of mono 16 kHz float audio is under 40 MB.
- **One measurement implementation.** The Studio wizard and enrolment must
  not disagree about whether a file clips.
- **Header hygiene.** New headers own their includes and compile in
  `fluxvoice_header_checks`; OS headers stay behind platform guards.

## Files touched

- `src/audio/AudioFileReader.{hpp,cpp}`, `src/audio/MiniaudioDecodeImpl.cpp`
  (new), `src/enroll/**` (new), `src/pipeline/ModelBundle.{hpp,cpp}`,
  `frontend/core/VoiceStudio.{hpp,cpp}`, `tools/fv_enroll.cpp` (new),
  `tests/test_audio_file_reader.cpp`, `test_enroll_qc.cpp`,
  `test_speech_select.cpp`, `test_enroll.cpp`, `test_enroll_job.cpp`,
  `tests/FakeSpeakerEncoder.hpp`, `tests/check_enroll_cli.cmake` (new),
  `tests/data/audio_malformed/**` (new), `CMakeLists.txt`,
  `cmake/CheckHeaders.cmake`

## Definition of done

- `fv_enroll create` produces a valid profile from the synthetic fixtures and
  matches every Stage 06 golden vector within the thresholds.
- With a trained bundle, C++ enrolment matches Python enrolment on the
  Stage 03 list within the resampler tolerance (or the row is marked pending:
  no trained bundle).
- The footprint, cancel, consent-refusal and malformed-file tests pass and
  each fails when its rule is removed.
- Enrolment time, peak memory and the concurrent-conversion result are
  recorded.
- Warnings-as-errors build and CTest pass ORT-less and with ORT, recorded as
  PASS / FAIL / NOT RUN per native leg.
