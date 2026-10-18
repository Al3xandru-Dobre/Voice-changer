# Shared model-contract cases (Milestone 2 / Stage 02)

Run by both validators, so they accept and reject the same contracts with the
same messages:

- C++: `tests/test_model_contract.cpp` (`src/pipeline/ModelContract.hpp`)
- Python: `python_training/tests/test_contract.py` (`python_training/fvml/contract.py`)

Each file is one case:

```json
{
  "description": "what the case is about",
  "error": "substring of the expected message",   // invalid/ only
  "algorithmic_latency_samples": 640,              // valid/ pairs, optional
  "models": [
    {"contract": {...} | "raw metadata string",
     "graph": {"inputs": [{"name", "dtype", "shape"}], "outputs": [...]}}   // optional
  ]
}
```

A case parses every model's contract, checks its graph I/O when `graph` is
given, and validates the pair when there are two models (encoder first). The
first error must contain `error`. Dtypes are `f32`, `i64`, `f16` or `other`, and
`-1` in a shape is a dynamic dimension.

The `valid/pair_*` cases mirror the master plan's latency table (algorithmic
only): A0 20 ms, A1 40 ms, A2 60 ms, A1-40 60 ms, Mimi q0 80 ms.
