# BUG-004 — `fvml.contract` raises `TypeError`, `KeyError`, `RecursionError` and `UnicodeEncodeError` where it promises `ContractError`

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-004` |
| **Name** | *Leaky errors* — callers that catch `ContractError` crash on malformed input instead |
| **Reported** | 2026-10-18 |
| **Severity** | Low (Python tooling only; the C++ twin never throws) |
| **Status** | ☑ fixed — verified 2026-10-18 15:30, see [Resolution](#resolution-verified-2026-10-18) |
| **Component** | `python_training/fvml/contract.py` |
| **Blocks** | Stage 06 exporter (`to_json`, `embed`) and any tool that loads third-party `.onnx` files via `load_onnx` |

## Summary

Every public function documents `Raises ContractError`, and the loader pattern
is `except ContractError`. Five inputs get past that and surface as other
exception types. A throwaway fuzz (300 random single-key mutations of every
contract in `tests/data/contracts/`) found no stray exception type from
`from_json` on well-formed JSON; these five come from deeply nested text,
code-built contracts, malformed graph dicts and an unencodable string.

## Steps to reproduce

From the repo root with the project's venv (the last line needs `onnx`):

```python
import sys; sys.path.insert(0, "python_training")
from fvml.contract import ModelContract, TensorSpec, ContractError, embed
import onnx

base = dict(role="vocoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
            lookahead_frames=0, content_kind="discrete", vocab=100, cond_frame_samples=320,
            license="x")
def run(label, fn):
    try: fn(); print(label, "-> no error")
    except ContractError: print(label, "-> ContractError (ok)")
    except Exception as e: print(label, "->", type(e).__name__)

run("deeply nested JSON ", lambda: ModelContract.from_json("[" * 100000))
run("cond_inputs=5      ", lambda: ModelContract(**base, cond_inputs=5).validate())
run("states=None        ", lambda: ModelContract(**base, states=None).validate())
run("TensorSpec no dtype", lambda: TensorSpec.from_dict({"name": "a", "shape": [1]}))
run("lone surrogate     ", lambda: embed(onnx.ModelProto(),
        ModelContract(**{**base, "license": "\ud800", "cond_inputs": ()}).to_json()))
```

## Expected

Every line prints `ContractError (ok)` (or, for the surrogate case, `to_json()`
refuses to produce text that cannot be stored).

## Actual

```
deeply nested JSON  -> RecursionError
cond_inputs=5       -> TypeError
states=None        -> TypeError
TensorSpec no dtype -> KeyError
lone surrogate      -> UnicodeEncodeError
```

(`TypeError` is "object is not iterable"; `UnicodeEncodeError` is "'utf-8' codec
can't encode character '\ud800'".)

## Root cause

- `_check_types` does `all(isinstance(s, str) for s in v)` on a value it has not
  yet shown to be iterable.
- `from_json` catches `(ValueError, TypeError)` around `json.loads`;
  `RecursionError` is a `RuntimeError`.
- `TensorSpec.from_dict` indexes `d["dtype"]` directly.
- `to_json` uses `ensure_ascii=False`, so a lone surrogate survives into a
  `str` that is not valid UTF-8; `embed` then fails when protobuf encodes it.

## Fix

- `_check_types`: test `isinstance(v, (list, tuple))` before iterating.
- `from_json`: also catch `RecursionError` (or cap nesting depth first).
- `TensorSpec.from_dict`: wrap in a check that raises `ContractError` naming the
  missing key.
- `to_json`: `text.encode("utf-8")` once inside `try`, raising `ContractError`
  ("license is not valid UTF-8"), or validate string fields with
  `s.encode("utf-8")` in `_check_types`.
- Add one case for each to `test_contracts_built_in_code_are_validated`.

## Workaround

Build contracts with tuples of `str`, and don't feed `from_json` text of unknown
provenance without a size limit.

## Resolution (verified 2026-10-18)

`_check_types` now checks for a list or tuple before iterating and requires every
string to be UTF-8 encodable; `from_json` also catches `RecursionError`;
`TensorSpec.from_dict` raises `ContractError` for a malformed tensor dict.

| Check | Result |
|---|---|
| Repro above, re-run on the current tree | all five lines print `-> ContractError (ok)` |
| Mutation test | removing the `RecursionError` catch, the list/tuple check, the UTF-8 check, or the `dtype` check in `from_dict` is each caught by the tests |
| Survivor | removing the `name` type check in `TensorSpec.from_dict` is **not** caught (one untested branch; negligible) |

One thing I noticed afterwards: `TensorSpec.from_dict` treated a missing
`"shape"` key as a scalar `()`, so a manifest entry that simply omitted `shape`
was read as rank 0 rather than rejected. The fixing session has since made
`shape` and a string `name` required, with tests. I re-checked: a missing shape
and a non-string name now raise `ContractError`, `shape: null` is read as unknown
rank, and a full shape is accepted. That also covers the one mutant that
survived above (the `name` check).

