# Stage 01 — Privacy, consent & misuse architecture

> **Milestone 5** · [Master plan](00_master.md) · [Stage 02 →](02_voice_profile_contract_store.md)

**Status:** ☐ not started
**Depends on:** nothing. This stage is written before any Milestone 5 code and
is normative for Stages 02, 07, 08, 09 and 10.
**Owner persona:** `@qa-verifier`, with a named human legal reviewer for the
sign-off rows

## Objective

Decide, in writing, what personal data this feature touches, why, where it
lives, for how long, and who can do what with it. Fix the consent model and
the misuse analysis. Every later stage implements rows from this file, and
Stage 11 audits each row.

This file is an engineering design. Its reading of the law is marked as such
and is confirmed by the legal reviewer before release.

## Context & verified starting point (2026-11-18)

- The application has no runtime network code, no account and no telemetry
  upload. The publisher receives nothing from a user's machine.
- Consent today is a checkbox and a free-text note in the Voice Studio
  (`StudioDraft::consent`, `consent_note`). It is not versioned, not
  enforced outside the wizard, and the CLI has no equivalent.
- No command exports or erases a voice. Studio drafts keep absolute paths to
  the user's takes and the consent note under the config directory, which on
  Windows is `%APPDATA%` and may roam to a server.
- The project documents contain no data-protection text
  ([M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) has two
  sentences on consent and on keeping the private bundle out of git).

## Roles and legal basis (engineering reading)

- **The user's machine does all processing.** The publisher has no access to
  recordings or profiles and does not determine what a given user enrols.
- **A user who enrols their own voice for their own use** acts within a
  personal or household activity (GDPR Art. 2(2)(c)).
- **A user who enrols another person's voice, or uses the feature in a
  professional context,** decides the purpose and means for that person's
  data and carries the obligations for it. The application must give them the
  tools: a consent record, export, erasure.
- **Data protection by design is adopted as an engineering requirement
  regardless of role** (Art. 25, Recital 78). It is what makes "local and
  privacy-respecting" a true statement, and it keeps the analysis valid if a
  networked feature is ever proposed.
- **Classification.** Voice recordings are personal data. A speaker embedding
  is personal data, and biometric data in the sense of Art. 4(14) where it is
  used to confirm identity. The "verified own voice" check is such a use.
  The whole profile store is therefore handled to the special-category
  standard: explicit, separate consent (Art. 9(2)(a)) and the strictest
  handling rules below.

## Data inventory

| # | Item | Where it exists | Lifetime | Removed by |
|---|---|---|---|---|
| 1 | Source recordings | The user's own folders. Opened read-only; never copied, moved or modified | The user's decision | The user; the app never deletes them |
| 2 | Decoded, resampled audio | RAM, enrolment job only | Until the job ends or is cancelled | Buffer release; no swap-avoidance claim is made |
| 3 | Speech/silence decisions, window embeddings | RAM, enrolment job only | Same | Same |
| 4 | Speaker embedding (64 f32) | `profile.json` | Until erased | Erase |
| 5 | F0 and energy statistics (4 numbers) | `profile.json` | Until erased | Erase |
| 6 | Quality numbers per file (duration, speech seconds, SNR, verdict) and the file's SHA-256 | `profile.json` → `provenance` | Until erased | Erase |
| 7 | Consent record (subject, scopes, text version, time) | `profile.json` → `consent` | Until erased; the export keeps a copy for the user | Erase |
| 8 | Evidence of a third party's consent | A file the user keeps; the profile stores its SHA-256 and kind only | The user's decision | The user |
| 9 | Live verification sample | RAM only; score, threshold and time are stored | Seconds | Buffer release |
| 10 | Refinement features (Stage 05) | RAM only | Job | Buffer release |
| 11 | References to a profile name | User presets, `settings.json` | Until edited | Reported by erase; settings cleared |
| 12 | Studio drafts (existing): take paths, consent note | `studio/<voice>/` | Until deleted | Listed in the Privacy centre; delete action (Stage 10) |
| 13 | Logs and `[stats]` lines | Console, user-chosen log files | — | Must never contain items 2–10 or absolute source paths |
| 14 | Converted audio | The live stream; files the user writes with `fv_vc_file` | The user's decision | The user |

Rules that follow from the table:

- **File paths are not stored in `profile.json`.** Paths often contain names.
  Hashes are enough to recognise the same material at re-enrolment.
- **No temporary files** anywhere in the enrolment path.
- **The store is in a non-roaming, non-synced location** (Stage 02).
- **Nothing in the inventory is ever transmitted.** There is no code to do so.

## Principles mapped to engineering rules

| Principle (Art. 5) | Rule in this milestone | Checked in |
|---|---|---|
| Lawfulness, fairness, transparency | Plain-language consent screen and an offline privacy notice; honest description of what a profile reproduces | Stage 10, 11 |
| Purpose limitation | A profile is used for voice conversion and for the one own-voice comparison. No identification, clustering or analytics features | Master constraint; Stage 11 review |
| Data minimisation | Read in place; persist derived values only; cap the speech used at 10 minutes | Stage 07 footprint test |
| Accuracy | Re-enrolment replaces a profile; quality numbers are shown, not hidden | Stage 10 |
| Storage limitation | Review reminder (open decision 9); nothing is kept after erase | Stage 02, 11 |
| Integrity and confidentiality | Owner-only permissions, atomic writes, strict parsing, no network | Stage 02, 11 |
| Accountability | This file, the impact record below, and tests that fail when a rule is removed | Stage 11 |

## Rights of the person whose voice it is

| Right | In the application |
|---|---|
| Access, portability (Art. 15, 20) | **Export**: `profile.json` plus a readable summary, to a folder the user chooses |
| Erasure, withdrawal of consent (Art. 17, 7(3)) | **Delete**: removes the profile, clears references, reports anything left. At most two actions from the profile; no harder than creating it |
| Rectification | **Re-enrol** from corrected material |
| Restriction, objection | **Disable**: the profile stays on disk but cannot be selected |
| Information | The consent screen and the notice state what is stored, where, and that nothing is sent anywhere |

## Consent model

- **Subject.** `self` or `other_adult`. Third-party minors are not supported
  (open decision 11).
- **Scopes are separate and unticked by default:**
  1. `enroll` — process my recordings on this device to build a voice profile.
  2. `voice_match` — compare a short live sample with the recordings to mark
     the profile as verified. Optional; refusing it does not block enrolment.
- **Versioned text.** Each consent text is an asset with an identifier
  (`consent-en-1`) and a hash. The record stores the identifier, the scopes
  and the time. A changed text is a new version; old records stay valid for
  the text they name.
- **Own voice.** The verified badge requires the live match. An unverified
  profile is usable but labelled, and is always marked (Stage 09).
- **Another adult's voice** (only if open decision 1 allows it):
  - the user attests that the person gave explicit, informed, documented
    consent for this purpose;
  - an evidence file is required (a signed document or a recorded spoken
    statement); the profile stores its kind and hash;
  - where the evidence is a spoken statement, the app compares its voice with
    the enrolment files and stores the score;
  - the profile cannot be verified, is always marked, and carries a visible
    badge.
- **Refusals stated in the text:** no voice without its owner's agreement; no
  impersonation to deceive; the user is responsible for disclosure where a
  listener could be misled.
- **Every entry point enforces it.** GUI, `fv_enroll` and the library API all
  refuse to write a profile without a consent record. There is no default
  that implies consent.

## Misuse and threat analysis

| Threat | Who is harmed | Controls | What remains |
|---|---|---|---|
| Enrolling someone else's voice to impersonate them | The voice's owner; people deceived | Own-voice default; live match for verification; evidence for third parties; marking; no sharing or import of personal profiles; disclosure prompts | A user can modify the source or replay a recording into the microphone. The controls set defaults and leave a record; they do not prevent a determined person |
| Copying a profile file to impersonate the user | The user | Owner-only permissions, non-roaming location, sync-folder warning, export only on request | Malware running as the user can read it. Encryption (open decision 3) raises the cost |
| Using a converted voice against voice-authentication systems | The voice's owner | Marking; documentation warning | A marker helps only systems that check for it |
| A hostile audio file | The user | Bounded streaming decode, caps, strict header checks, limited decoder set (Stage 07) | Decoder bugs not yet found |
| Disclosure through logs, crash dumps, screenshots or bug reports | The user | Log hygiene rule and test; profile shown by name, never by values | Screenshots of the Privacy centre show metadata |
| A dependency adds network traffic | The user | Import-table check; no-network run in CI | Checked per release, not continuously |
| Voices in the training corpora | The corpus speakers | Only CC0 / CC BY corpora recorded with the speakers' participation; no scraped or "in the wild" sets for the speaker encoder; attribution in `MODEL_CARD.md` | A 64-value embedding space carries little about any one training speaker; not measured |

## Regulatory checkpoints

- **GDPR** — the rows above. An impact assessment (Art. 35) is prepared
  because biometric-grade data and a new technology are involved, even though
  the publisher may not be the controller.
- **EU AI Act, Art. 50** — providers must mark synthetic audio in a
  machine-readable, detectable way (50(2)); deployers must disclose deep
  fakes (50(4)). In application since 2 August 2026; a product first released
  after that date complies at release. Stage 09 implements the engineering
  side. Whether conversion into one's own voice, or into a bundle voice, is
  in scope is a question for the reviewer; the default is to mark.
- **National law on voice and likeness** varies by country and is the user's
  responsibility for third-party voices; the notice says so.

## Tasks

- [ ] Review and freeze the data inventory with the owners of Stages 02, 07,
      08 and 10. Each row names the test or manual check that proves it.
- [ ] Write the consent texts `consent-en-1` (own voice) and
      `consent-third-party-en-1`, the privacy notice and the "what a profile
      can and cannot do" text as assets under `frontend/assets/privacy/`.
      Plain language; no pre-ticked boxes; no bundled scopes.
- [ ] Write the impact record: purposes, necessity, risks from the threat
      table with likelihood and severity, controls, residual risk, decision.
- [ ] Decide open decisions 1 and 11 with the reviewer and record the reason.
- [ ] Legal review of this file, the texts and the impact record. Record the
      reviewer, date and outcome. A review that did not happen is NOT RUN.
- [ ] Add the handling rules to `Documents/BEST_PRACTICES.md` (new section
      "Personal voice data") and the condensed form to `MEMORY.md`.
- [ ] File a bug report for the Studio drafts stored under the roaming config
      directory (inventory row 12).

## Results *(fill in)*

| Row | Outcome | Evidence |
|---|---|---|
| Inventory frozen | | |
| Consent texts and notice written | | |
| Impact record complete | | |
| Open decision 1 (third-party voices) | | |
| Open decision 11 (minors) | | |
| Legal review | PASS / FAIL / NOT RUN — reviewer, date | |

## Files touched

- This file (results), `frontend/assets/privacy/*.md` (new),
  `Documents/Milestone 5/impact_record.md` (new),
  `Documents/BEST_PRACTICES.md`, `MEMORY.md`,
  `Documents/Bug Report/BUG-NNN_studio_drafts_in_roaming_config.md` (new)

## Definition of done

- Every inventory row has an owner stage and a named check.
- The consent texts, the notice and the impact record exist and are reviewed
  by the stage owners.
- Open decisions 1 and 11 are recorded with reasons.
- The legal review row is filled. If it is NOT RUN, the master's release rule
  applies: the feature stays off in release builds.
