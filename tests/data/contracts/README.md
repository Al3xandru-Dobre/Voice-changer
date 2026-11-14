# Shared model-contract cases (Milestone 2 / Stage 02 & Milestone 3 / Stage 02)

Run by both validators, so they accept and reject the same contracts with the
same messages:

- C++: `tests/test_model_contract.cpp` (`src/pipeline/ModelContract.hpp`)
- Python: `python_training/tests/test_contract.py` (`python_training/fvml/contract.py`)

Each file is one case:

```json
{
  "description": "what the case is about",
  "error": "substring of the expected message",   // invalid/ only
  "algorithmic_latency_samples": 640,              // valid/ Mode A pairs, optional
  "wait_k": 3,                                     // valid/ Mode B quartets, optional
  "s2ut_onset_latency_samples": 4160,              // valid/ Mode B quartets, optional
  "models": [
    {"contract": {...} | "raw metadata string",
     "graph": {"inputs": [{"name", "dtype", "shape"}], "outputs": [...]}}   // optional
  ]
}
```

A case parses every model's contract, checks its graph I/O when `graph` is
given, validates a Mode A pair when there are two models (`encoder`, `vocoder`),
and validates a Mode B S2UT quartet (plus cross-graph `shared_mem_states` I/O
when both translator graphs are given) when there are four models (`encoder`,
`translator_enc`, `translator_dec`, `vocoder`). The first error must contain
`error`. Dtypes are `f32`, `i64`, `f16` or `other`, and `-1` in a shape is a
dynamic dimension.

The `valid/pair_*` cases mirror the Milestone 2 master plan's latency table
(algorithmic only): A0 20 ms, A1 40 ms, A2 60 ms, A1-40 60 ms, Mimi q0 80 ms.
The `valid/quartet_*_v2` cases mirror the Milestone 3 master plan's Mode B onset
latency formula (`wait_k * chunk_frames * frame_samples + (L_enc + L_voc) *
frame_samples + overlap_samples`).
