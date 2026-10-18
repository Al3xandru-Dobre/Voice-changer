# Bug reports — index

> Parent: [Documents index](../README.md) · [Engineering best practices](../BEST_PRACTICES.md) · [Project memory](../../MEMORY.md)

Bugs in FluxVoice itself, its test rig, or its external dependencies —
one file per bug, tagged `BUG-NNN` with a short name. External-dependency
bugs are tracked here because they constrain our milestone work; file them
upstream separately when reproducible. Once a bug is diagnosed, its root cause
and prevention rules are compiled into [`../BEST_PRACTICES.md`](../BEST_PRACTICES.md).

| Tag | Name | File | Component | Status | First seen |
|---|---|---|---|---|---|
| BUG-001 | Dead pipes | [BUG-001_voicemeeter_vaio_routing_wedge.md](BUG-001_voicemeeter_vaio_routing_wedge.md) | Voicemeeter Banana 2.1.2.2 (external) | ◐ diagnosed | 2026-10-17 |
| BUG-002 | Read-only handshake | [BUG-002_voicemeeter_remote_write_lock.md](BUG-002_voicemeeter_remote_write_lock.md) | VoicemeeterRemote64.dll (external) | ◐ workaround | 2026-10-17 |
| BUG-003 | Unbounded geometry | [BUG-003_contract_extreme_geometry_accepted.md](BUG-003_contract_extreme_geometry_accepted.md) | `contract.py` / `ModelContract.hpp` (M2 S02) | ☑ fixed | 2026-10-18 |
| BUG-004 | Leaky errors | [BUG-004_contract_py_leaks_non_contracterror.md](BUG-004_contract_py_leaks_non_contracterror.md) | `fvml/contract.py` (M2 S02) | ☑ fixed | 2026-10-18 |
| BUG-005 | List vs tuple | [BUG-005_contract_list_fields_unhashable.md](BUG-005_contract_list_fields_unhashable.md) | `fvml/contract.py` (M2 S02) | ☑ fixed | 2026-10-18 |
| BUG-006 | Truthy schema | [BUG-006_voices_schema_accepts_true.md](BUG-006_voices_schema_accepts_true.md) | `fvml/contract.py` `validate_voices` (M2 S02) | ☑ fixed | 2026-10-18 |
| BUG-007 | Shapeless state | [BUG-007_shapeless_state_passes_as_static.md](BUG-007_shapeless_state_passes_as_static.md) | `contract.py` / `ModelContract.hpp` (M2 S02) | ☑ fixed | 2026-10-18 |
| BUG-008 | Data-root trap | [BUG-008_gitignore_data_swallows_tests_data.md](BUG-008_gitignore_data_swallows_tests_data.md) | `.gitignore` / M2 S01 data root | ☑ fixed | 2026-10-18 |
| BUG-009 | Silent gate | [BUG-009_m2_tests_not_in_ctest.md](BUG-009_m2_tests_not_in_ctest.md) | `CMakeLists.txt`, CI, `tools/make_fixture_models.py` (M2 S02) | ☑ fixed | 2026-10-18 |
| BUG-010 | Deaf liveness | [BUG-010_liveness_ignores_callback_progress.md](BUG-010_liveness_ignores_callback_progress.md) | `AudioController` (M1.5 S05) | ☑ fixed | 2026-10-18 |
| BUG-011 | Retry bypass | [BUG-011_preopen_retryable_failure_bypasses_recovery.md](BUG-011_preopen_retryable_failure_bypasses_recovery.md) | `AudioController` (M1.5 S05) | ☑ fixed | 2026-10-18 |
| BUG-012 | Fresh stale | [BUG-012_unavailable_refresh_left_snapshot_fresh.md](BUG-012_unavailable_refresh_left_snapshot_fresh.md) | `AudioDeviceRegistry` (M1.5 S03/05) | ☑ fixed | 2026-10-18 |

## Reviews

| Date | Scope | File |
|---|---|---|
| 2026-10-18 | Milestone 2 stages 01–02: test run, definition-of-done check, mutation-test gaps, suggestions | [REVIEW-M2-S01-S02_2026-10-18.md](REVIEW-M2-S01-S02_2026-10-18.md) |
| 2026-10-18 | Milestone 1.5 stages 05–06: controller/backend review, cmake gate, Windows WASAPI + loopback + feedback-guard verification; BUG-010/011/012 found & fixed | [REVIEW-M15-S05-S06_2026-10-18.md](REVIEW-M15-S05-S06_2026-10-18.md) |

## Conventions

- **Tag:** `BUG-NNN`, assigned in order; never reused.
- **Name:** two-or-three-word handle for talking about the bug.
- **File:** `BUG-NNN_<snake_case_slug>.md` in this directory.
- **Status:** `☐ open` → `◐ diagnosed` → `◐ workaround` → `☑ fixed` /
  `☑ wontfix (external)` → closed.
- Every bug file carries: environment, repro steps, expected vs actual,
  evidence, workaround, and which milestone stage it blocks.
