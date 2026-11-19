# BUG-032 — C++ `VoiceCatalog` parsers do not validate or retain the Stage 03 `voices.json` `"lang"` field

> [Bug index](00_index.md) · Found by: [REVIEW-M3-S01-S05_2026-11-15](REVIEW-M3-S01-S05_2026-11-15.md)

| | |
|---|---|
| **Tag** | `BUG-032` |
| **Name** | *Dropped voice language* — C++ accepts ill-typed `"lang"` in `voices.json` and discards per-voice language metadata |
| **Reported** | 2026-11-15 |
| **Severity** | Medium (breaks Python/C++ `voices.json` validation parity and loses per-voice language tags in Stage 03 multilingual bundles) |
| **Status** | ◐ fixed in tree (uncommitted; CI legs pending) — [fix plan](FIX-PLAN-BUG-029-032_2026-11-18.md) |
| **Component** | `frontend/core/VoiceCatalog.cpp:58, 135-160, 380`, `frontend/core/VoiceCatalog.hpp:29-42`, `src/pipeline/VoiceCatalog.hpp:59-91` |
| **Blocks** | [Milestone 3 / Stage 02](../Milestone%203/02_translator_contract_v2_fixtures.md) & [Stage 03](../Milestone%203/03_multilingual_encoder_vocoder_upgrade.md) C++ `voices.json` parity; Milestone 4 multilingual voice filtering |

## Summary

Milestone 3 Stage 02 (`02_translator_contract_v2_fixtures.md:185`) and Stage 03 (`03_multilingual_encoder_vocoder_upgrade.md:101, 218-221`) extended `voices.json` (schema 1) with an optional per-voice `"lang"` metadata field (`[a-z-]{2,8}`, e.g. `"en"`, `"es"`, `"fr"`, `"de"`).

In Python, `fvml/contract.py::validate_voices()` (`lines 804, 839-846`) validates that `"lang"`, when present, is a non-empty string matching `_is_lang_code`. On the C++ side:

1. **`frontend/core/VoiceCatalog.cpp` whitelists `"lang"` without reading, validating, or storing it.**
   Line 58 adds `"lang"` to `kVoiceOptionalKeys[] = {"attribution", "lang"}` so the unknown-key loop (`line 136`) accepts it, **but `parse_voices_json()` (`lines 158-160`) checks only `"attribution"` and never touches `"lang"`**. Consequently:
   - Invalid entries such as `"lang": 123`, `"lang": null`, `"lang": ""`, or `"lang": "English"` (explicitly rejected by `test_contract.py:264` and `test_multilingual.py:886`) pass `frontend::parse_voices_json` silently.
   - `frontend::VoiceEntry` (`frontend/core/VoiceCatalog.hpp:29-42`) has no `lang` member, so the parsed voice language is lost.
   - `VoiceCatalog::search()` (`frontend/core/VoiceCatalog.cpp:380`) assigns `it.languages = b.languages` (populated solely from `langs.json`). For a Stage 03 multilingual Mode-A bundle (`encoder_s*.onnx` + `vocoder_s*.onnx` + `voices.json` with 12–16 `{en,es,fr,de}` voices and no `langs.json`), `b.languages` is empty, so filtering by language in the catalogue returns **0 voices**; for a Mode-B bundle with `langs.json`, every voice is tagged with all 4 bundle languages rather than its own `"lang"`.
2. **`src/pipeline/VoiceCatalog.hpp` (`read_voice_catalog`, `lines 59-91`) validates neither unknown per-voice keys nor optional `"attribution"` / `"lang"` fields.**
   Unlike `fvml/contract.py` and `frontend::parse_voices_json`, the engine's `read_voice_catalog()` accepts arbitrary unknown keys on `voices[i]` as well as non-string/invalid `"lang"` and `"attribution"` values.

## Environment

- Source audit + `zig c++` (Clang 20.1.2) CTest build on Windows 11 x64, 2026-11-15.

## Steps to reproduce

Compare `fvml/contract.py:839-846` with `frontend/core/VoiceCatalog.cpp:58, 135-160` and `src/pipeline/VoiceCatalog.hpp:59-91` on a `voices.json` containing `{"lang": "English"}` or `{"lang": 42}`:

- Python (`fvml.contract.validate_voices`): raises `ContractError("voice 'lj': invalid language code 'English' (expected lowercase [a-z-]{2,8})")`.
- C++ (`fluxvoice::frontend::parse_voices_json` and `fluxvoice::read_voice_catalog`): returns `ok() == true` and discards `"lang"`.

## Expected

- Both C++ `voices.json` parsers (`src/pipeline/VoiceCatalog.hpp` and `frontend/core/VoiceCatalog.cpp`) enforce the same schema-1 per-voice key whitelist and `"lang"` format rule (`lowercase [a-z-]{2,8}`) as `fvml/contract.py::validate_voices`.
- `VoiceEntry` retains `lang` so `frontend::VoiceCatalog::search()` and `all_languages()` surface each voice's `"lang"` tag even when `langs.json` is absent.

## Actual

`"lang"` is ignored after passing `known_voice_key()`, accepting malformed values and dropping the language metadata from `VoiceEntry`.

## Evidence

| File | Lines | Detail |
|---|---|---|
| `python_training/fvml/contract.py` | `839-846` | Validates `lang` is a non-empty string matching `_is_lang_code` (`[a-z-]{2,8}`) |
| `frontend/core/VoiceCatalog.cpp` | `58, 158-160` | `"lang"` in `kVoiceOptionalKeys`, but only `"attribution"` is validated/stored |
| `frontend/core/VoiceCatalog.cpp` | `380, 446-452` | `search()` and `all_languages()` read only `b.languages` (`langs.json`), never per-voice `lang` |
| `src/pipeline/VoiceCatalog.hpp` | `59-91` | No unknown per-voice key check and no `"lang"`/`"attribution"` validation |

## Root cause

When Stage 02 added `"lang"` to `kVoiceOptionalKeys` in `frontend/core/VoiceCatalog.cpp` so v2/Stage 03 `voices.json` files would not fail the unknown-key check, the corresponding field validation, `VoiceEntry::lang` storage, and `src/pipeline/VoiceCatalog.hpp` mirror were omitted.

## Suggested fix

1. Add `std::string lang` to `fluxvoice::frontend::VoiceEntry` (and `fluxvoice::VoiceEntry` in `src/pipeline/VoiceCatalog.hpp`), validate `"lang"` with `contract_detail::is_lang_code` (`[a-z-]{2,8}`) in both C++ parsers, and enforce the per-voice key whitelist + optional `"attribution"` string check in `src/pipeline/VoiceCatalog.hpp`.
2. In `frontend::VoiceCatalog::search()` and `all_languages()`, populate `it.languages` from `{v.lang}` when `!v.lang.empty()` (falling back to `b.languages` when `v.lang` is unset, or exposing bundle S2UT target languages separately from voice native language).
3. Add negative `"lang"` cases (`"EN"`, `""`, `42`) and a Stage 03 `voices.json` language-filter test in `tests/test_voice_catalog.cpp` and `tests/test_frontend_model.cpp`.

## Workaround

Validate `voices.json` through `fvml.contract.validate_voices` (or `export_stateful_onnx.py`) before loading bundles in C++.

## Resolution

- `src/pipeline/VoiceCatalog.hpp`: `VoiceEntry` gains `lang` and `attribution`; `read_voice_catalog` validates both optional keys like `fvml.contract.validate_voices` (non-empty string, then `lang` must be lowercase `[a-z-]{2,8}`) with the twin's wording: `voice 'X': lang must be a non-empty string` and `voice 'X': invalid language code 'Y' (expected lowercase [a-z-]{2,8})`.
- `frontend/core/VoiceCatalog.{hpp,cpp}`: `VoiceEntry::lang` added; `parse_voices_json` reads and validates it with the same rule and messages. Unknown-key handling is unchanged (the src reader still ignores unknown per-voice keys; the frontend reader and the twin reject them).
- Tests: `tests/test_voice_catalog.cpp` (lang round-trip, `zh-hans`, absent stays empty, invalid codes and non-string values rejected with the exact messages) and `test_voice_catalog` in `tests/test_frontend_model.cpp` (same cases through `parse_voices_json`). Built GCC Release with warnings-as-errors and the GUI; MSVC / Clang / macOS legs not run.
