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
| BUG-013 | Stale golden | [BUG-013_stale_golden_survives_reexport.md](BUG-013_stale_golden_survives_reexport.md) | `export_stateful_onnx.py` bundle lifecycle (M2 S06) | ☑ fixed | 2026-10-18 |
| BUG-014 | Traceback exits | [BUG-014_export_cli_traceback_exits.md](BUG-014_export_cli_traceback_exits.md) | `export_stateful_onnx.py` / `fvml/export.py` error boundary (M2 S06) | ☑ fixed | 2026-10-18 |
| BUG-015 | Wrong clock | [BUG-015_gate_decides_on_pytorch_clock.md](BUG-015_gate_decides_on_pytorch_clock.md) | `gate_report.py` eligibility/projected T−B (M2 S05) | ☑ fixed | 2026-10-18 |
| BUG-016 | Silent fallback | [BUG-016_gate_export_silent_exporter_fallback.md](BUG-016_gate_export_silent_exporter_fallback.md) | `gate_report.py` export smoke test / console encoding (M2 S05) | ☑ fixed | 2026-10-18 |
| BUG-017 | Quiet tolerance | [BUG-017_vocoder_checks_below_noise_floor.md](BUG-017_vocoder_checks_below_noise_floor.md) | `vocoder.py` init, `stream_check.py`, `gate_report.py`, `test_vocoder.py` (M2 S05) | ☑ fixed | 2026-10-18 |
| BUG-018 | Red Windows | [BUG-018_frontend_getenv_breaks_msvc_build.md](BUG-018_frontend_getenv_breaks_msvc_build.md) | `frontend/core/Paths.cpp`, CI `windows-msvc` (M4 S01) | ◐ fixed in tree, native MSVC leg pending | 2026-11-14 |
| BUG-019 | Two-arg QoS | [BUG-019_macos_qos_call_wrong_arity.md](BUG-019_macos_qos_call_wrong_arity.md) | `src/util/WorkerPriority.hpp`, CI `macos-clang` (M2 S08) | ◐ fixed in tree, native legs pending | 2026-11-14 |
| BUG-020 | Quiet default | [BUG-020_missing_remembered_device_becomes_default.md](BUG-020_missing_remembered_device_becomes_default.md) | `frontend/core/AppModel.cpp` device policy restore (M4 S02) | ◐ fixed in tree, native legs pending | 2026-11-14 |
| BUG-021 | Prefix folders | [BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md](BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md) | Voice Studio draft persistence (M4 S03) | ◐ fixed in tree, native legs pending (`project_dir` residual fixed 2026-11-18) | 2026-11-14 |
| BUG-022 | Null ETA | [BUG-022_wrong_typed_json_field_terminates_gui.md](BUG-022_wrong_typed_json_field_terminates_gui.md) | `frontend/core/VoiceStudio.cpp` JSON readers (M4 S03) | ◐ fixed in tree, native legs pending (guard made recoverable and checked number conversions added 2026-11-19, M4 S04) | 2026-11-14 |
| BUG-023 | Oldest wins | [BUG-023_gui_param_queue_drops_newest_snapshot.md](BUG-023_gui_param_queue_drops_newest_snapshot.md) | `frontend/core/EngineHost.cpp` `push_params` (M4 S03/S05) | ◐ fixed in tree, native legs pending ("applied = accepted": worker publishes the accepted generation, 2026-11-19, M4 S05) | 2026-11-14 |
| BUG-024 | Sticky residue | [BUG-024_gui_queued_latency_not_rebaselined.md](BUG-024_gui_queued_latency_not_rebaselined.md) | `frontend/core/EngineHost.cpp` `telemetry()` (M4 S06) | ◐ fixed in tree, native legs pending (Mode A uses the full identity from `ModeAHost::stats()`, 2026-11-19) | 2026-11-14 |
| BUG-025 | Phantom script | [BUG-025_studio_commands_not_runnable.md](BUG-025_studio_commands_not_runnable.md) | Voice Studio job commands (M4 S03 / M2 S10) | ◐ fixed in tree (unavailable steps labelled), native legs pending | 2026-11-14 |
| BUG-026 | Double lookahead | [BUG-026_latency_target_counts_lookahead_twice.md](BUG-026_latency_target_counts_lookahead_twice.md) | `LatencyGovernor` default target, `PipelineGeometry` (M2 S08) | ☑ fixed | 2026-11-14 |
| BUG-027 | Broken archive plumbing | [BUG-027_s2ut_data_prep_archive_plumbing_broken.md](BUG-027_s2ut_data_prep_archive_plumbing_broken.md) | `fvml/s2ut_data.py` archive download/extract/hash (M3 S01) | ☐ open | 2026-11-15 |
| BUG-028 | Stranded tail frames | [BUG-028_fifo_strands_partial_eos_chunk_frames.md](BUG-028_fifo_strands_partial_eos_chunk_frames.md) | `gate_s2ut.py`, `UnitProsodyFifo` partial `<EOS>` chunks (M3 S04/S05) | ◐ fixed in C++ runtime + Python FIFO twin (`gate_s2ut.py` still open) | 2026-11-15 |
| BUG-029 | Half-wired readiness | [BUG-029_ctc_adaptive_probability_never_catches_up.md](BUG-029_ctc_adaptive_probability_never_catches_up.md) | `s2ut_stream.py`, `translator.py` `CTC-Adaptive` readiness (M3 S05) | ◐ fixed in tree | 2026-11-15 |
| BUG-030 | Blind tail audit | [BUG-030_silence_tail_babble_passes_gate_and_eval.md](BUG-030_silence_tail_babble_passes_gate_and_eval.md) | `gate_s2ut.py`, `s2ut_eval.py`, `silence_hallucination_20.tsv` (M3 S01/S05) | ◐ fixed in tree | 2026-11-15 |
| BUG-031 | Laundered non-commercial taint | [BUG-031_s2ut_pipeline_drops_noncommercial_taint.md](BUG-031_s2ut_pipeline_drops_noncommercial_taint.md) | `units.py`, `train_s2ut.py`, `s2ut_stream.py` NC guard (M3 S03/S05) | ◐ fixed in tree | 2026-11-15 |
| BUG-032 | Dropped voice language | [BUG-032_cpp_voice_catalog_ignores_voice_lang_field.md](BUG-032_cpp_voice_catalog_ignores_voice_lang_field.md) | `frontend/core/VoiceCatalog.cpp`, `src/pipeline/VoiceCatalog.hpp` (M3 S02/S03) | ◐ fixed in tree | 2026-11-15 |

## Reviews

| Date | Scope | File |
|---|---|---|
| 2026-10-18 | Milestone 2 stages 01–02: test run, definition-of-done check, mutation-test gaps, suggestions | [REVIEW-M2-S01-S02_2026-10-18.md](REVIEW-M2-S01-S02_2026-10-18.md) |
| 2026-10-18 | Milestone 1.5 stages 05–06: controller/backend review, cmake gate, Windows WASAPI + loopback + feedback-guard verification; BUG-010/011/012 found & fixed | [REVIEW-M15-S05-S06_2026-10-18.md](REVIEW-M15-S05-S06_2026-10-18.md) |
| 2026-10-18 | Milestone 2 stage 06: stateful ONNX export static review (no shell); DoD check, gate/conditioning/golden verification; BUG-013/014 found | [REVIEW-M2-S06_2026-10-18.md](REVIEW-M2-S06_2026-10-18.md) |
| 2026-10-18 | Milestone 2 stage 05: causal vocoder + decision gate; fresh `/WX` build 46/46, 41-mutant test audit (28 survive), gate re-run with the stage file's numbers; BUG-015/016/017 found | [REVIEW-M2-S05_2026-10-18.md](REVIEW-M2-S05_2026-10-18.md) |
| 2026-11-14 | Milestone 4 status + desktop frontend audit: master-plan verification, CI evidence (Windows leg red since the frontend merge), specification-vs-code tables, test gaps; no local build (no toolchain); BUG-018, BUG-020…025 found | [REVIEW-M4-FRONTEND_2026-11-14.md](REVIEW-M4-FRONTEND_2026-11-14.md) |
| 2026-11-14 | Milestone 2 status + stages 07–09: stage claims vs. tree, real-time rule check, contract v2 spot check, test gaps; static review + CI evidence + 21 contract tests run; BUG-019, BUG-026 found | [REVIEW-M2-S07-S09_2026-11-14.md](REVIEW-M2-S07-S09_2026-11-14.md) |
| 2026-11-15 | Milestone 3 stages 01–05 validation: zig-Clang `-Werror` build + CTest, 145 M3 pytest + 1,000-trajectory live C++/Python `WaitKPolicy` parity + 60 s full-size translator equivalence; BUG-027…032 found | [REVIEW-M3-S01-S05_2026-11-15.md](REVIEW-M3-S01-S05_2026-11-15.md) |

## Conventions

- **Tag:** `BUG-NNN`, assigned in order; never reused.
- **Name:** two-or-three-word handle for talking about the bug.
- **File:** `BUG-NNN_<snake_case_slug>.md` in this directory.
- **Status:** `☐ open` → `◐ diagnosed` → `◐ workaround` → `☑ fixed` /
  `☑ wontfix (external)` → closed.
- Every bug file carries: environment, repro steps, expected vs actual,
  evidence, workaround, and which milestone stage it blocks.
