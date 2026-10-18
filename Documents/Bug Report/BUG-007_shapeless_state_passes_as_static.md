# BUG-007 — A state tensor with no shape information is accepted as "static"

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-007` |
| **Name** | *Shapeless state* — unknown rank and scalar are the same thing to the validator |
| **Reported** | 2026-10-18 |
| **Severity** | Low–medium (a model like this can't be preallocated; the loader would fail late) |
| **Status** | ☑ fixed — verified 2026-10-18 15:30 in both validators, see [Resolution](#resolution-verified-2026-10-18) |
| **Component** | `python_training/fvml/contract.py` (`_value_spec`, `_is_static`), `src/pipeline/ModelContract.hpp` (`validate_io`) |
| **Blocks** | Stage 07 `KVStateCache` (sizes every state buffer from the graph at load time) |

## Summary

`validate_io` rejects a state with a dynamic dimension (`state '…' has a dynamic
shape [?,…]`) because `KVStateCache` must size the buffer once, at load. A state
whose ONNX `ValueInfo` carries **no shape at all** (unknown rank) is more
dynamic than that, yet it passes: `_value_spec` turns "no shape" into `()`, and
`_is_static(())` is `all([])`, which is `True`. The same path also accepts a
genuine rank-0 scalar state, which is a different, legitimate case; the
validator can't tell them apart.

## Steps to reproduce

Needs `onnx` (the project's venv):

```python
import sys; sys.path.insert(0, "python_training")
from onnx import TensorProto, helper
from fvml import contract as fc

c = fc.ModelContract(role="encoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
                     lookahead_frames=0, content_kind="soft", content_dim=4,
                     states=("s",), license="x")
F = TensorProto.FLOAT
graph = helper.make_graph(
    [helper.make_node("Identity", ["state_in.s"], ["state_out.s"])], "g",
    [helper.make_tensor_value_info("audio", F, [1, 320]),
     helper.make_tensor_value_info("state_in.s", F, None)],      # no shape at all
    [helper.make_tensor_value_info("content", F, [1, 1, 4]),
     helper.make_tensor_value_info("state_out.s", F, None)])
ins, outs = fc.graph_io(helper.make_model(graph))
print(ins[1])
print(fc.check_model(c.to_json(), ins, outs).role)
```

## Expected

`ContractError: state 's' has a dynamic shape …` (unknown rank), or at minimum
a distinct message, since a scalar state would be the only legitimate `()`.

## Actual

```
TensorSpec(name='state_in.s', dtype='f32', shape=())
encoder
```

The model is accepted.

## Root cause

`_value_spec` builds the shape from `t.shape.dim` and never checks
`t.HasField("shape")`, so "no shape" and "rank 0" both become `()`. The C++
twin takes its shapes from the same representation (`std::vector<int64_t>`,
empty for both) and `any_of` over an empty vector is `false`, so it would
behave the same once Stage 07 fills `TensorSpec` from ORT (the C++ path is
read from the code; I did not run it).

## Fix

- Python: in `_value_spec`, return `shape=None` (or a sentinel such as `(-1,)`)
  when `not t.HasField("shape")`, and have `_is_static` reject it. Keep `()` for
  a real scalar.
- C++ (Stage 07): when converting ORT's `GetShape()`, map "unknown rank" to a
  `{-1}` shape so the existing `d < 1` test fires; add the case to
  `tests/data/contracts/invalid/` so both languages run it.
- Add a fixture to `broken/` ("state without a shape") to `make_fixture_models.py`.

Also related and also unpinned by tests: `_is_static` uses `d >= 1`, so a state
with a zero-size dimension is reported as "dynamic". The message is
misleading but the rejection is right.

## Workaround

None needed for the repo's own fixtures: every state they use has a full static
shape.

## Resolution (verified 2026-10-18)

Python: `TensorSpec.shape` is `None` when the graph gives no shape and `()` only
for a real scalar; `validate_io` rejects `state '…' has no shape in the graph`
and, separately, `has an empty dimension […]` for a zero-size dim (the old
"dynamic" message was misleading for that case). C++: `TensorSpec` gained
`has_shape`, with the same messages; `format_shape` prints `unknown`.

| Check | Result |
|---|---|
| Repro above, re-run | `TensorSpec(name='state_in.s', dtype='f32', shape=None)`, then `ContractError: state 's' has no shape in the graph` |
| C++ | the shared cases `io_state_no_shape`, `io_state_out_no_shape`, `io_input_no_shape` pass in `model_contract`; I read the C++ rejection paths but did not mutate them |
| Mutation test | state_in without shape, state_out without shape, `_value_spec` ignoring a missing shape, dynamic dim, zero-size dim, and three variants of the graph-tensor shape comparison: **all killed** |

A correction to the severity above: the session that fixed this pointed out that
the ONNX checker already rejects graph I/O with no `shape` field, and I confirmed
it on my own repro model: `onnx.checker.check_model` (onnx 1.23.2, with and
without `full_check`) raises `ValidationError: Field 'shape' of 'type' is
required but missing.` A checker-valid `.onnx` therefore can't trigger this,
which is why there is no broken `.onnx` fixture for it and the cases are
JSON-only. My repro built the `ModelProto` without running the checker. The guard
is still right defence for a model that skips the checker; the original
"low–medium" was too high.

