# BUG-005 — A contract built with lists validates, then breaks equality and hashing

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-005` |
| **Name** | *List vs tuple* — `ModelContract` accepts `list` fields it cannot compare or hash |
| **Reported** | 2026-10-18 |
| **Severity** | Low |
| **Status** | ☑ fixed — verified 2026-10-18 15:30, see [Resolution](#resolution-verified-2026-10-18) |
| **Component** | `python_training/fvml/contract.py` (`ModelContract`, `_check_types`) |
| **Blocks** | Stage 06 exporter (it builds contracts in code, which is what `_check_types` is for) |

## Summary

`ModelContract` is `@dataclass(frozen=True)` with tuple-typed `cond_inputs` and
`states`. `_check_types` accepts any non-`str` iterable of strings, including a
`list`. Such a contract passes `validate()` and `to_json()`, but:

- `from_json(c.to_json()) == c` is **False** (the parsed copy holds tuples, the
  original lists), so the canonical round-trip property the tests rely on
  silently fails for it;
- `hash(c)` raises `TypeError: unhashable type: 'list'` (a frozen, eq-comparing
  dataclass generates `__hash__` from its fields), so it cannot go in a `set`
  or be a dict key.

## Steps to reproduce

```python
import sys; sys.path.insert(0, "python_training")
from fvml.contract import ModelContract

c = ModelContract(role="encoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
                  lookahead_frames=0, content_kind="soft", content_dim=4,
                  states=["conv", "kv"], license="x")       # lists, as an exporter might pass
c.validate()                                                # accepted
print("round trip equal:", ModelContract.from_json(c.to_json()) == c)
hash(c)
```

## Expected

Either the lists are normalised to tuples (so equality and hashing work), or
`validate()` rejects them.

## Actual

```
round trip equal: False
TypeError: unhashable type: 'list'
```

## Root cause

`from_dict` converts with `tuple(v)` (`_get_str_list`), but the constructor and
`_check_types` do not. The frozen dataclass has no `__post_init__` to
normalise.

## Fix

Add to `ModelContract`:

```python
def __post_init__(self):
    object.__setattr__(self, "cond_inputs", tuple(self.cond_inputs))
    object.__setattr__(self, "states", tuple(self.states))
```

guarded so a non-iterable still reaches `_check_types` as a `ContractError`
(see [BUG-004](BUG-004_contract_py_leaks_non_contracterror.md)). Add
`ModelContract(..., states=["a"])` to `test_contracts_built_in_code_are_validated`
asserting round-trip equality.

## Workaround

Pass tuples. `tools/make_fixture_models.py` already does.

## Resolution (verified 2026-10-18)

`ModelContract.__post_init__` converts `cond_inputs` and `states` lists to tuples,
as proposed.

| Check | Result |
|---|---|
| Repro above, re-run | prints `round trip equal: True`, and `hash(c)` no longer raises (the script exits 0) |
| Mutation test | removing the conversion is caught by the tests |

