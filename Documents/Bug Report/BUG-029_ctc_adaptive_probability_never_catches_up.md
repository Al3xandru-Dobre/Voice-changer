# BUG-029 — `CTC-Adaptive` in `translator.py` and `s2ut_stream.py` still feeds `(0, 1)` probabilities, preventing 2-chunk catch-up steps

> [Bug index](00_index.md) · Found by: [REVIEW-M3-S01-S05_2026-11-15](REVIEW-M3-S01-S05_2026-11-15.md)

| | |
|---|---|
| **Tag** | `BUG-029` |
| **Name** | *Half-wired readiness* — Stage 04 widened readiness to `[0, 2]`, but Stage 05 still trains & feeds `(0, 1)` probabilities |
| **Reported** | 2026-11-15 |
| **Severity** | Medium–High (invalidates the `ctc-adaptive` candidate in Stage 05's decision gate: 0 two-write steps at constant `k`, monotonic lag ratchet) |
| **Status** | ☐ open |
| **Component** | `python_training/fvml/s2ut_stream.py:361-362`, `python_training/fvml/translator.py:770-778, 843-845` |
| **Blocks** | [Milestone 3 / Stage 05](../Milestone%203/05_causal_s2ut_translator_policy_gate.md) policy decision gate (`Strict Wait-k` vs. `CTC-Adaptive`) |

## Summary

Stage 05 (`05_causal_s2ut_translator_policy_gate.md:21-22`, `00_master.md:224`) pits `Strict Wait-k` against `CTC-Adaptive Wait-k` (*"allowing 0/1/2 target chunks per step"* so target languages with 15–25% more syllables than English can catch up without accumulating lag).

During Stage 05 development (`05_causal_s2ut_translator_policy_gate.md:191-197`), it was noted that clamping readiness to `[0, 1]` produced `0` two-write steps across 4,000,000 random steps at constant `k`. Stage 04 (`WaitKPolicy.hpp:126-134`, `waitk_policy.py:226-236`) was updated on 2026-11-15 to accept `expected_chunks` in `[0, 2]`, with `04_waitk_policy_streaming_state_machine.md:147-149` explicitly warning:

> *"A caller that feeds a plain probability still gets the delay-only behaviour: the readiness head has to be mapped to an expected count for the catch-up path to be used."*

Stage 05's model training and streaming driver were never updated to match:

1. **Training target clamping (`translator.py:770-778, 843-845`).** `readiness_targets()` rescales CTC non-blank mass so an utterance sums to `target_chunks`, **then clamps each source chunk to `[0.0, 1.0]`** and trains `readiness_head` with `binary_cross_entropy_with_logits`. Whenever `target_chunks > src_chunks` (`En -> Es/Fr/De`), chunks carrying `> 1` target chunk are clipped to `1.0` (`test_translator.py:704` even asserts `target_chunks=8` over `4` source chunks clamps to `[1, 1, 1, 1]`, discarding half the target chunks).
2. **Inference sigmoid cap (`s2ut_stream.py:361-362`).** `StreamingTranslator.push_chunk()` calls:
   ```python
   policy.on_source_readiness(1.0 / (1.0 + math.exp(-readiness)))
   ```
   passing `x in (0, 1)` into `WaitKPolicy`.
3. **State-machine invariant consequence.** In `WaitKPolicy`, `max_writes_this_step()` returns `2` only when `lag > k and (readiness >= 2.0 or lag > k + adaptive_extra_wait)`. At constant `k`, `lag` can only exceed `k` immediately after a step that returned `Decision.READ` — which implies `readiness < 1.0` and `lag <= k + adaptive_extra_wait` on the next step. Adding `x = sigmoid(logit) < 1.0` to `readiness < 1.0` yields `readiness < 2.0`, and `lag > k + adaptive_extra_wait` is false (`lag <= k + adaptive_extra_wait`). Therefore:
   - `max_writes_this_step()` **never returns `2`** at constant `k` for non-saturated logits;
   - every held-back `READ` step permanently increases utterance lag by `+80 ms` (ratcheting `ctc-adaptive@2` up to `waitk-4` after two low-readiness chunks, with no ability to catch back up during speech).

## Environment

- Windows 11 x64, `python_training/.venv` (CPython 3.12.13), 2026-11-15.

## Steps to reproduce

```powershell
cd python_training
.\.venv\Scripts\python.exe -c "
import math, random
from fvml.waitk_policy import WaitKPolicy, WaitKConfig

rng = random.Random(0)
two_writes = 0
for _ in range(500):
    p = WaitKPolicy(WaitKConfig(k=2, adaptive_mode=True, adaptive_extra_wait=2))
    for _ in range(100):
        p.on_chunk_energy(True, 80)
        p.notify_source_unit()
        logit = rng.uniform(-10.0, 10.0)
        p.on_source_readiness(1.0 / (1.0 + math.exp(-logit)))
        n = p.max_writes_this_step()
        two_writes += (n == 2)
        for _ in range(n):
            p.notify_target_unit(False)
print('2-write steps out of 50,000:', two_writes)
"
```

## Expected

`CTC-Adaptive` can emit `0`, `1`, or `2` target chunks per step at constant `k` when `readiness_head` predicts `> 1` expected target chunks on dense/expanded phrases, matching both Stage 04's `[0, 2]` readiness contract and Stage 05's gate specification.

## Actual

```text
2-write steps out of 50,000: 0
```

`test_translator.py:404-424` confirms this: `assert {r.writes for r in results} == {0, 1}`, requiring an explicit `st.policy.set_k(1)` mid-utterance to observe a 2-write step.

## Evidence

| File | Lines | Detail |
|---|---|---|
| `Documents/Milestone 3/04_waitk_policy_streaming_state_machine.md` | `144-149` | Documents widening `on_source_readiness` to `[0, 2]` on 2026-11-15 and notes callers must map readiness to `[0, 2]` |
| `Documents/Milestone 3/05_causal_s2ut_translator_policy_gate.md` | `191-197` | Still contains the pre-update note claiming `WaitKPolicy` clamps readiness to `1.0` |
| `python_training/fvml/translator.py` | `777, 844` | `.clamp(0.0, 1.0)` + `binary_cross_entropy_with_logits` |
| `python_training/fvml/s2ut_stream.py` | `362` | `1.0 / (1.0 + math.exp(-readiness))` in `(0, 1)` |

## Root cause

Stage 04 widened `WaitKPolicy::on_source_readiness` to `[0, 2]` after Stage 05 had already been written against the `[0, 1]` probability signature, and the corresponding training/inference mapping in `translator.py` and `s2ut_stream.py` was not updated before training.

## Suggested fix

1. In `translator.py::readiness_targets()`, clamp expected target chunks per source chunk to `[0.0, 2.0]` (matching `WaitKPolicy`'s `[0, 2]` cap), and supervise either `2.0 * torch.sigmoid(readiness_logit)` with smooth-L1 / MSE (or BCE on `ready / 2.0` paired with `2.0 * sigmoid(readiness_logit)` at inference — noting that under `2.0 * sigmoid(...)` a neutral `0` logit maps to `1.0`, so initialize `readiness_head` bias or threshold appropriately).
2. In `s2ut_stream.py:362` (and Stage 07/08 C++ Mode B runner), apply the exact same `[0, 2]` mapping (`2.0 * sigmoid(readiness_logit)` or `softplus` clamped to `2.0`) before calling `policy.on_source_readiness(...)`.
3. Update `05_causal_s2ut_translator_policy_gate.md:191-197` and `test_translator.py:403-415` to verify `2` writes occur at constant `k` without mutating `set_k()`.

## Workaround

None for `ctc-adaptive` — running `gate_s2ut.py` today evaluates a delay-only variant rather than true 0/1/2-chunk CTC-Adaptive Wait-*k*.
