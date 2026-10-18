# BUG-006 — `validate_voices` accepts `"schema": true` and `"schema": 1.0`

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-006` |
| **Name** | *Truthy schema* — Python's `True == 1` defeats the version check |
| **Reported** | 2026-10-18 |
| **Severity** | Low |
| **Status** | ☑ fixed — verified 2026-10-18 15:30, see [Resolution](#resolution-verified-2026-10-18) |
| **Component** | `python_training/fvml/contract.py` (`validate_voices`) |
| **Blocks** | Stage 08 `--list-voices` (the C++ reader will be stricter; the two will then disagree) |

## Summary

`validate_voices` checks the schema with `data.get("schema") != 1`. In Python
`True == 1` and `1.0 == 1`, so a `voices.json` whose `schema` is the boolean
`true` or the float `1.0` is accepted. Everywhere else in the module integers
go through `_is_int`, which rejects booleans and floats (cases `int_as_bool`,
`int_as_float`). This is the one place the rule is not applied.

## Steps to reproduce

```python
import sys; sys.path.insert(0, "python_training")
from fvml.contract import ModelContract, validate_voices

voc = ModelContract(role="vocoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
                    lookahead_frames=0, content_kind="discrete", vocab=100, cond_frame_samples=320,
                    cond_inputs=("spk_emb",), spk_emb_dim=2, license="x")
voice = {"name": "lj", "display": "LJ", "embedding": [0.5, -0.5], "logf0_mean": 5.3, "logf0_std": 0.2,
         "energy_mean": -3.0, "energy_std": 1.1, "license": "PD", "source": "LJSpeech 1.1"}
for schema in (1, True, 1.0, 2, "1"):
    try: validate_voices({"schema": schema, "voices": [voice]}, voc); print(repr(schema), "accepted")
    except Exception as e: print(repr(schema), "rejected:", e)
```

## Expected

Only the integer `1` is accepted.

## Actual

```
1 accepted
True accepted
1.0 accepted
2 rejected: unsupported voices.json schema 2
'1' rejected: unsupported voices.json schema '1'
```

## Root cause

`!=` compares by value across `bool`/`int`/`float`. The module's own `_is_int`
helper exists for this and isn't used here.

## Fix

```python
schema = data.get("schema")
if not _is_int(schema) or schema != 1:
    raise ContractError(f"unsupported voices.json schema {schema!r}")
```

Add `True` and `1.0` to `test_voices_validation`. While there, note the mutation
run in [REVIEW-M2-S01-S02](REVIEW-M2-S01-S02_2026-10-18.md#3-test-gaps-found-by-mutation-testing):
`validate_voices` has 12 rules no test pins.

## Workaround

Write `"schema": 1` literally; nothing in the repo generates anything else.

## Resolution (verified 2026-10-18)

`validate_voices` now checks `_is_int(schema) and schema == 1`.

| Check | Result |
|---|---|
| Repro above, re-run | `1` accepted; `True`, `1.0`, `2`, `'1'` all rejected with `unsupported voices.json schema …` |
| Mutation test, schema | removing the `_is_int` guard is caught |
| Mutation test, the 12 previously unpinned `validate_voices` rules (unknown top-level key, missing key, uppercase name, non-ASCII name, `energy_std == 0`, non-number `logf0_mean` / `energy_mean`, embedding element, `display`/`license`/`source` type, `attribution` type, NaN, non-object voice) | **12 of 12 mutants killed** |

