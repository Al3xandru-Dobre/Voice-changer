# Stage 02 — Voice-profile contract & local profile store

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 01](01_privacy_consent_misuse_architecture.md) · [Stage 03 →](03_unseen_speaker_eval_harness.md)

**Status:** ☐ not started
**Depends on:** [Stage 01](01_privacy_consent_misuse_architecture.md) (inventory, consent model),
[M2 / Stage 02](../Milestone%202/02_model_contract_fixtures.md) (model contract, `voices.json` rules).
Needs no trained weights.
**Owner persona:** `@onnx-ort-runtime` (contract, store) with `@qa-verifier` (fixtures, twins)

## Objective

Define the file a personal voice is stored in, the identifier that ties it to
the models able to interpret it, the model role for the speaker encoder, and a
small store that saves, lists, exports and erases profiles safely. Python and
C++ validate the same fixtures.

## Context & verified starting point (2026-11-18)

- `voices.json` (schema 1) is read strictly: only `schema` and `voices` at the
  top level, unknown keys rejected, embedding length checked against the
  vocoder's `spk_emb_dim` (`src/pipeline/VoiceCatalog.hpp`,
  `fvml.contract.validate_voices`). It is produced by the exporter and belongs
  to the bundle.
- The contract has versions 1 and 2 and four roles: `encoder`,
  `translator_enc`, `translator_dec`, `vocoder`
  (`src/pipeline/ModelContract.hpp`, `python_training/fvml/contract.py`).
  Nothing identifies *which* speaker space an embedding belongs to; today the
  only embeddings are the rows exported with the vocoder itself.
- User data paths come from `frontend/core/Paths.cpp`, which the CLI tools do
  not link. Earlier parsing defects in this area are the reference for what
  to avoid: a boolean accepted as a number
  ([BUG-006](../Bug%20Report/BUG-006_voices_schema_accepts_true.md)) and a
  wrong-typed field terminating the GUI
  ([BUG-022](../Bug%20Report/BUG-022_wrong_typed_json_field_terminates_gui.md)).

## Design

### `profile.json` (schema 1)

```json
{ "schema": 1,
  "id": "pv_5c0e9a1f4b7d4e2a8f3b6c1d0e9a7b42",
  "name": "me", "display": "Me",
  "created_utc": "2026-11-18T10:22:31Z",
  "speaker_space": "fvss1-9f2c41d07be35a68",
  "tier": "instant",
  "embedding": [0.0123, -0.0456],
  "logf0_mean": 4.87, "logf0_std": 0.21,
  "energy_mean": -3.9, "energy_std": 1.1,
  "license": "private — do not distribute",
  "source": "on-device enrolment (M5)",
  "consent": { "subject": "self", "granted_utc": "2026-11-18T10:20:02Z",
               "text_version": "consent-en-1", "scopes": ["enroll", "voice_match"],
               "evidence": [],
               "verified_self": { "method": "live_match", "score": 0.71,
                                  "threshold": 0.55, "utc": "2026-11-18T10:24:10Z" } },
  "provenance": { "enroll_algo": 1, "bundle": "fluxvoice-en-v2",
                  "app_version": "0.5.0", "speech_seconds": 184.2,
                  "files": [ { "sha256": "…", "seconds": 61.0,
                               "speech_seconds": 48.3, "snr_db": 34.1,
                               "verdict": "clean" } ] },
  "marking_required": true,
  "review_after_utc": "2027-11-18T00:00:00Z",
  "disabled": false }
```

(The example embedding is shortened; a real one has `spk_emb_dim` values.)

Rules, identical in both validators:

- `schema` is the integer `1`. Unknown keys are rejected at every level.
- `id` is `pv_` plus 32 lowercase hex digits from a random source. It is not
  derived from the audio or from the name.
- `name` follows the `voices.json` rule (`[a-z0-9_-]+`) and the Studio's
  `is_safe_project_name` rule; `display` is a non-empty string.
- `embedding` is a list of finite f32 numbers; its length must equal the
  loaded vocoder's `spk_emb_dim`. The four statistics are finite numbers and
  both standard deviations are positive. Booleans are never numbers.
- `tier` is `instant` or `refined`. `license` and `source` are non-empty.
- `consent.subject` is `self` or `other_adult`; `scopes` contains `enroll`;
  `text_version` names a known consent text. `other_adult` requires at least
  one `evidence` entry (`kind` + `sha256`) and forbids `verified_self`.
- `provenance.files` holds hashes and quality numbers. **No file paths.**
- `marking_required` is `true` unless the subject is `self` and
  `verified_self` is present. The loader recomputes it; a file cannot lower
  it.
- Timestamps are UTC, `YYYY-MM-DDTHH:MM:SSZ`.
- A file larger than 1 MiB is rejected before parsing.

### `speaker_space`

An embedding only has meaning for the vocoder that was trained with the
encoder that produced it. `speaker_space` is the string
`fvss1-<first 16 hex of SHA-256(speaker-encoder weights ‖ vocoder FiLM weights)>`,
computed at export and written into both contracts and into every profile.

- The bundle validator requires the vocoder's and the speaker encoder's
  `speaker_space` to be equal.
- A profile whose `speaker_space` differs from the loaded bundle's is listed
  as "needs re-enrolment for this bundle" and cannot be selected. It is never
  used anyway and never silently dropped.
- A bundle without a speaker encoder has no `speaker_space`; enrolment is
  unavailable with a fixed message (`kNoSpeakerEncoderMessage`, mirroring
  `kNoTranslatorModelsMessage`).

### Contract v3 (recommended, open decision 2)

`contract_version 3` = version 2 plus:

- role `speaker_enc`: input `audio` f32 `[1, window_samples]`, output
  `spk_emb` f32 `[1, spk_emb_dim]`, no states; keys `sample_rate`,
  `window_samples`, `window_hop_samples`, `min_speech_samples`,
  `spk_emb_dim`, `embedding_scale`, `speaker_space`, `enroll_algo`, `license`;
- vocoder key `speaker_space` (optional; required when the bundle contains a
  `speaker_enc.onnx`).

Version 1 and 2 bundles load exactly as today.

### Store layout and location

```text
<local data dir>/fluxvoice/voices/
  <id>/profile.json        one directory per profile; nothing else is written
```

`default_profile_dir()` lives in `src/voice/ProfilePaths.hpp` so the CLI
tools and the GUI share it:

| OS | Location | Why |
|---|---|---|
| Windows | `%LOCALAPPDATA%\fluxvoice\voices` | `%APPDATA%` roams to a server in managed environments |
| Linux | `$XDG_DATA_HOME/fluxvoice/voices`, else `~/.local/share/fluxvoice/voices` | Data, not configuration |
| macOS | `~/Library/Application Support/fluxvoice/voices` | Not synced by default |

`--profile-dir` overrides it. A chosen path with a component that looks like
a sync root (OneDrive, Dropbox, iCloud Drive / Mobile Documents, Google
Drive, Nextcloud) produces a warning that the profile would leave the device.

### `ProfileStore` (`src/voice/ProfileStore.{hpp,cpp}`)

Toolkit-free, no ONNX Runtime. Every function returns a typed result and
never throws across the boundary.

- `list()` → entries with state `Ok`, `Invalid(reason)`, `Disabled`.
- `load(id)`, `save(profile)`: write to a sibling temporary name inside the
  profile directory, flush, then rename over the target. Directory mode
  `0700`, file mode `0600` on POSIX; inherited per-user ACL on Windows.
- `erase(id)` → `EraseReport`: files removed, settings keys cleared, presets
  that still name the voice (reported, not rewritten), Studio drafts with the
  same name (reported), and anything that could not be removed.
- `export_to(id, directory)`: `profile.json` byte-identical to the stored
  one, plus `README.txt` describing each field in plain words.
- `set_disabled(id, bool)`.

Erasure is an unlink. On SSDs and copy-on-write file systems overwriting does
not guarantee destruction, so the documentation recommends full-disk
encryption and the application never claims a "secure wipe".

## Tasks

- [ ] `python_training/fvml/profile.py`: dataclass, `validate_profile`,
      `write_profile`, `read_profile`; the same error messages as the C++
      reader for the shared fixtures.
- [ ] `src/voice/VoiceProfile.hpp` (struct + `read_voice_profile`),
      `ProfileStore.{hpp,cpp}`, `ProfilePaths.hpp`. Environment access goes
      through an injected lookup so the path logic is unit-tested on every
      leg and does not repeat
      [BUG-018](../Bug%20Report/BUG-018_frontend_getenv_breaks_msvc_build.md).
- [ ] Contract v3 in `fvml/contract.py` and `ModelContract.hpp`: role, keys,
      `expected_inputs` / `expected_outputs`, bundle-level `speaker_space`
      equality. Valid and invalid JSON under `tests/data/contracts/`.
- [ ] Fixtures `tests/data/profiles/valid/*.json` and `invalid/*.json`, one
      invalid file per rule above, each with the expected message.
- [ ] `tools/make_fixture_models.py`: a tiny `speaker_enc` fixture and a v3
      fixture bundle with matching `speaker_space`; one bundle with a
      deliberate mismatch.
- [ ] Tests (`tests/test_voice_profile.cpp`, `tests/test_profile_store.cpp`,
      `python_training/tests/test_profile.py`), registered in CTest:
  - [ ] every fixture gives the same verdict in Python and C++;
  - [ ] a wrong-typed field yields `Invalid`, never a crash or an exception;
  - [ ] `marking_required` cannot be lowered by editing the file;
  - [ ] an interrupted save leaves the old file or the new one, never a
        partial one (inject a failure between write and rename);
  - [ ] after `erase`, a recursive scan of the store finds no file containing
        the id, and `list()` no longer returns it;
  - [ ] `export_to` round-trips byte-identically;
  - [ ] POSIX modes are `0700` / `0600`;
  - [ ] Windows resolves to `LOCALAPPDATA`, not `APPDATA`; the sync-root
        warning fires for each listed provider name;
  - [ ] a 2 MiB file is rejected before parsing.
- [ ] Header checks: the new headers compile in `fluxvoice_header_checks`.

## Implementation notes

- **Why not extend `voices.json`.** It is produced by the exporter, belongs to
  a public bundle and is validated strictly on both sides. Writing private
  data into it would make every bundle directory personal data and would mix
  two owners in one file.
- **Why hashes instead of paths.** Re-enrolment after a bundle upgrade asks
  the user to select the files again; the hashes confirm it is the same
  material. The profile never needs to know where the files live.
- **Why the loader recomputes `marking_required`.** The file is editable. The
  rule has to live in code.
- **No backup copies and no "recently deleted" folder.** Erase means erase.

## Files touched

- `src/voice/VoiceProfile.hpp`, `ProfileStore.{hpp,cpp}`, `ProfilePaths.hpp`
  (new), `src/pipeline/ModelContract.hpp`, `python_training/fvml/profile.py`
  (new), `python_training/fvml/contract.py`, `tools/make_fixture_models.py`,
  `tests/data/profiles/**`, `tests/data/contracts/**`,
  `tests/test_voice_profile.cpp`, `tests/test_profile_store.cpp` (new),
  `tests/test_model_contract.cpp`, `python_training/tests/test_profile.py`
  (new), `CMakeLists.txt`, `cmake/CheckHeaders.cmake`

## Definition of done

- Python and C++ agree on every profile and contract fixture, with the same
  messages.
- Save is atomic, erase leaves nothing, export is complete; each has a test
  that fails when the rule is broken.
- Version 1 and 2 bundles and all existing contract tests are unaffected.
- Warnings-as-errors build and CTest pass on the legs available, recorded as
  PASS / FAIL / NOT RUN per leg.
