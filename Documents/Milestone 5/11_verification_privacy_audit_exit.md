# Stage 11 — Verification, privacy audit & exit report

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 10](10_gui_quick_voice_privacy_centre.md)

**Status:** ☐ not started
**Depends on:** Stages 01–10 (Stage 05 as decided; Stage 10 only if M4 Stages 04–05 have landed),
[M2 / Stage 11](../Milestone%202/11_verification_exit.md) (latency, quality and soak numbers to compare against)
**Owner persona:** `@qa-verifier`

## Objective

Measure the milestone's acceptance criteria and audit its privacy claims with
recorded evidence. "Processing stays local" and "erase means erase" are
tested here, not asserted.

## Tasks

### Build and regression

- [ ] **Build gate, four native legs.** Warnings-as-errors all-target build
      plus CTest, ORT-less and with ORT, for Windows MSVC, Linux GCC, Linux
      Clang and macOS AppleClang. Record the commit, compiler versions and
      PASS / FAIL / NOT RUN per leg with the CI run. A leg that did not run
      is NOT RUN.
- [ ] **Nothing regressed.** Every pre-existing CTest case passes; version 1
      and 2 bundles load and convert as before; the bundle directory is
      byte-identical before and after enrolling and converting.

### Quality (shipped runtime: `fv_enroll` + `fv_vc_file`)

- [ ] `eval_enroll.py` main, duration and degraded conditions for Instant
      (and Refined, if accepted) against B0. Fill the table below.
- [ ] Curated voices on `test_clean_200`: no regression against the M2
      numbers beyond the Stage 04 rule.
- [ ] **A real case.** The developer's own voice from 60 seconds of ordinary
      recordings (laptop or phone microphone), scored against held-out
      recordings of the same voice. If the M2 Stage 10 "me" dataset exists,
      also compare with the fine-tuned voice.
- [ ] Enrolment time and peak memory for 60 s and 180 s on the reference
      machine.

### Real time

- [ ] `fv_probe latency`: `T − B` with a bundle voice and with a profile,
      marking on. The difference between the two is within ±2 ms, and the
      M2 **Must** line holds.
- [ ] 30-minute live soak with a profile and marking: 0 deadline misses
      after warm-up, underruns flat, `queued` bounded, RSS growth ≤ 5 MB
      (`tools/analyze_soak.py`).
- [ ] An enrolment (and a refinement, if accepted) started during live
      conversion: 0 deadline misses.
- [ ] Add, disable and erase a profile while converting: the stream resumes
      after each swap; erasing the active voice stops it with the message.

### Privacy audit

Each row is recorded as PASS / FAIL / NOT RUN with its evidence.

- [ ] **No network, by construction.** The import tables of `fluxvoice`,
      `fv_enroll`, `fv_vc_file`, `fv_wm_detect` and the first-party objects
      of `fluxvoice_gui` contain no socket or name-resolution symbols
      (`nm` / `dumpbin` script in CTest, per OS).
- [ ] **No network, by observation.** On Linux, the full sequence (enrol →
      list → convert a file → export → erase) passes inside a network
      namespace with no interfaces. On Windows and macOS, a manual run with
      a traffic monitor; record the tool and the result.
- [ ] **Footprint.** Under a temporary home directory, a snapshot before
      and after enrolment differs only inside the profile store. No
      temporary files appear during the job (watch the temp directories).
      Source recordings have the same hashes afterwards.
- [ ] **Erasure.** After erase, a recursive search of the application's
      data and config directories finds neither the profile id, the name in
      a profile file, nor the embedding's bytes. The voice is gone from
      every list. The `EraseReport` matches what is on disk.
- [ ] **Export.** The exported file equals the stored one byte for byte and
      contains every field of schema 1.
- [ ] **Location and permissions.** The store resolves to the non-roaming
      location on each OS; POSIX modes are `0700` / `0600`; on Windows the
      ACL is the user's only (manual check).
- [ ] **Log hygiene.** With the most verbose logging, no log line or
      `[stats]` line contains embedding values, statistics, or an absolute
      path of a source recording.
- [ ] **Consent cannot be bypassed.** For the GUI, the CLI and the library
      API: remove the consent check in a scratch build and confirm the
      tests fail. Record which test caught each.
- [ ] **Marking.** Output for a `marking_required` profile is detected by
      `fv_wm_detect` directly, after the virtual-cable loopback, and after
      an Opus round trip. A forced marker failure stops conversion. If
      Stage 09 rejected every marker, confirm the fallback instead:
      third-party and unverified profiles cannot be selected.
- [ ] **Repository.** No real voice recording, embedding of a real person
      or personal profile is tracked by git (fixtures are synthetic;
      `check_gitignore.cmake` extended to the new paths).
- [ ] **Inventory review.** Walk the Stage 01 inventory row by row against
      the code as built. Any new data item found is added and assessed.
- [ ] **Impact record and legal review.** Complete; reviewer, date and
      outcome recorded.

### Documentation

- [ ] Update the [master plan](00_master.md) status and stage table,
      `Documents/README.md`, the root `README.md` milestone table,
      `frontend/README.md`, `models/README.md` and `python_training/DATASETS.md`.
- [ ] `MEMORY.md` and `Documents/BEST_PRACTICES.md`: the invariants this
      milestone adds (profiles never enter a bundle; no temporary files; no
      network code; consent before reading; marking fails closed; fixtures
      are synthetic).
- [ ] `AGENTS.md`: Milestone 5 in the roster's milestone column and the
      owned paths (`src/enroll/**`, `src/voice/Profile*`,
      `Documents/Milestone 5/**`).
- [ ] File every defect found as `BUG-NNN`.

## Results *(fill in)*

### Quality

| Method | Condition | Enrol (s) | WER | `tgt_cos` | B0 | Ceiling | Norm. sim. | Subgroup min Δ | UTMOS | Enrol time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Instant | main | 60 | | | | | | | | |
| Instant | duration | 10 / 30 / 60 / 180 | | | | | | | | |
| Instant | degraded | 60 | | | | | | | | |
| Refined *(if accepted)* | main | 180 | | | | | | | | |
| Real case | own voice | 60 | | | | | | | | |

### Real time

| Check | Bundle voice | Profile + marking | Limit | Result |
|---|---|---|---|---|
| `T − B` (ms) | | | within ±2 ms; ≤ 100 ms | |
| Soak misses after warm-up | | | 0 | |
| RSS growth over 30 min (MB) | | | ≤ 5 | |
| Misses during enrolment | — | | 0 | |

### Privacy audit

| Row | Linux GCC | Linux Clang | Windows MSVC | macOS AppleClang | Evidence |
|---|---|---|---|---|---|
| No network: import tables | | | | | |
| No network: observed run | | | | | |
| Footprint | | | | | |
| Erasure | | | | | |
| Export | | | | | |
| Location and permissions | | | | | |
| Log hygiene | | | | | |
| Consent cannot be bypassed | | | | | |
| Marking / fallback | | | | | |
| Repository holds no real voice | | | | | |
| Inventory review | | | | | |
| Impact record and legal review | | | | | |

## Release rule

Personal voice enrolment stays **off in release builds** (build option and
setting) until every privacy-audit row is PASS on the released platforms and
the legal review row is PASS. Until then it is available in developer
builds only, for the developer's own voice.

## Definition of done

- Every acceptance criterion in the [master plan](00_master.md) has a
  measured row here, or an explicit, dated deviation with its reason.
- No row claims a platform, a measurement or a review that was not run.
- Open defects are filed and linked; none is of High severity; no privacy
  audit row is FAIL.
