# BUG-003 — Contract v1 accepts geometry whose derived sizes overflow or demand gigabytes

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-003` |
| **Name** | *Unbounded geometry* — every field is range-checked, the quantities computed from them are not |
| **Reported** | 2026-10-18 |
| **Severity** | Medium (latent: needs a hostile or corrupt model file, or a typo in an exporter) |
| **Status** | ☑ fixed — verified 2026-10-18 15:30, see [Resolution](#resolution-verified-2026-10-18) |
| **Component** | `python_training/fvml/contract.py` and `src/pipeline/ModelContract.hpp` (same rules in both) |
| **Blocks** | [Stage 02](../Milestone%202/02_model_contract_fixtures.md) ("both languages agree"); Stage 07 `KVStateCache` / `OrtModel` preallocation |

## Summary

`contract.py` states: *"Every integer must fit int32 so both languages agree on
the range."* That holds for the stored fields. It does not hold for what the
engine computes from them: `hop = frames_per_step × frame_samples`,
`cond_frames = hop / cond_frame_samples`, `output_samples = hop + overlap`, and
`algorithmic_latency`. Nothing bounds those, nor the individual fields to
physically meaningful values (a `frame_samples` of 16 000 000 at 16 kHz is a
1000-second frame).

## Steps to reproduce

From the repo root, with any Python 3:

```python
import sys; sys.path.insert(0, "python_training")
from fvml.contract import ModelContract, algorithmic_latency, validate_pair

I32 = 2**31 - 1
enc = ModelContract(role="encoder", sample_rate=16000, frame_samples=I32, frames_per_step=4,
                    lookahead_frames=I32, content_kind="discrete", vocab=100, license="x")
voc = ModelContract(role="vocoder", sample_rate=16000, frame_samples=I32, frames_per_step=4,
                    lookahead_frames=I32, content_kind="discrete", vocab=100,
                    cond_frame_samples=1, cond_inputs=("logf0",), license="x")
enc.validate(); voc.validate(); validate_pair(enc, voc)       # all accepted
print(enc.hop, voc.cond_frames, algorithmic_latency(enc, voc))
```

## Expected

A contract this far outside any real model is rejected with a specific message.

## Actual

Accepted. The script prints:

```
8589934588 8589934588 9223372036854775806
```

That is `hop` and `cond_frames`, both above the int32 maximum of 2 147 483 647,
and a latency of 2⁶³ − 2 samples.

A second example with ordinary-looking numbers: `frame_samples=16 000 000` at
16 kHz (a 1000-second frame), `frames_per_step=1`, `cond_frame_samples=1`
validates, and the `logf0` input would be `[1, 16 000 000]` floats = **61 MiB**.
Four conditioning inputs come to about 244 MiB, which the Stage 07 design
preallocates at load time.

C++ twin: `ModelContract::hop()` is already widened to `int64_t`, so `hop`
itself is safe. `algorithmic_latency_samples()` is not. Take `frame_samples =
2³¹−1`, `frames_per_step = 4`, both lookaheads `2³¹−1`, and `overlap_samples =
2³¹−1` (valid, since overlap ≤ hop): the sum is `9 223 372 039 002 259 453`,
which exceeds `INT64_MAX` by 2 147 483 646. That is signed overflow, undefined
behaviour in C++. (Arithmetic checked in Python; the C++ test suite has no case
for it and I did not run a C++ reproduction.)

## Root cause

`_check_types` / `get_int` bound each integer to int32 and `_positive` bounds it
below, but no rule bounds a field above in physical terms, and derived values
are computed without checks. The Python side hides the problem because its
integers are unbounded.

## Fix

Add upper bounds to both validators (same rules, same messages, one shared
invalid case each), for example:

| Field | Suggested bound |
|---|---|
| `sample_rate` | 8 000 … 192 000 |
| `frame_samples` | ≤ `sample_rate` (a frame ≤ 1 s) |
| `hop` (= `frames_per_step × frame_samples`) | ≤ `sample_rate` |
| `lookahead_frames` | ≤ 64 |
| `vocab`, `content_dim`, `spk_emb_dim`, `cond_frame_samples` | ≤ 2²⁰ / 2¹⁶ / 2¹² / `frame_samples` |

Pick the real limits from Stage 07's memory budget. With these in place every
derived value fits `int32`, so `ModelContract` could also drop the `int64_t`
widening.

## Workaround

None needed today: no model in the repo is near these values, and the loader
(Stage 07) does not exist yet. Until the bounds exist, treat a contract from an
untrusted `.onnx` as unvalidated for sizing.

## Resolution (verified 2026-10-18)

Fixed by the session implementing Stages 01–02, in both languages with the same
limits and a message of the form `X must be between lo and hi (got v)`:
`sample_rate` 8 000–192 000, `frame_samples` 1–192 000, `hop` ≤ `sample_rate`
(`hop (N samples) is longer than one second at sample_rate R`), `lookahead_frames`
0–64, `vocab` 1–2²⁰, `content_dim` 1–2¹⁶, `spk_emb_dim` 0–4 096.

How it was checked:

| Check | Result |
|---|---|
| Repro above | raises `ContractError: frame_samples must be between 1 and 192000 (got 2147483647)` |
| Constants, `contract.py` vs `ModelContract.hpp` | read side by side: identical |
| Shared cases in C++ and Python | `model_contract` and `model_contract_py` pass (CTest 42/42, `/W4 /WX`); the set now includes an at-limits valid case and `extreme_frame`, `*_too_large`, `sample_rate_too_low` invalid cases |
| Mutation test (limits moved one past or short, hop check `>=` / removed, lower bounds loosened) | **21 of 21 mutants killed** |
| Derived sizes | largest `hop` 192 000; largest latency 192 000 + 128 × 192 000 + 192 000 = 24 960 000, which fits int32 — the int64 overflow is no longer reachable |

Not verified: GCC/Clang builds (CI runs them; I ran MSVC only).

