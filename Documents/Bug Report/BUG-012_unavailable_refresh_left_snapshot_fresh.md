# BUG-012 — `refresh()` Unavailable path left a retained snapshot marked fresh

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-012` |
| **Name** | *Fresh stale* — one failure path forgot to mark the retained snapshot `stale` |
| **Reported** | 2026-10-18 |
| **Severity** | Low (cosmetic inconsistency; unreachable with `MiniaudioBackend` today but reachable for fakes/future backends) |
| **Status** | ☑ fixed — one-line `stale_` marking added, matching every other failure exit |
| **Component** | `src/audio/AudioDeviceRegistry.cpp` (`refresh()` `Unavailable` branch) |
| **Blocks** | [Milestone 1.5 / Stage 03](../Milestone%201.5/03_device_registry.md) — "on refresh failure keep the previous snapshot, marked stale" |

## Summary

`AudioDeviceRegistry::refresh()` has three failure exits. Two of them mark the
previously published snapshot `stale` before returning:

```cpp
if (st.state == BackendState::NullOnly) {
    if (published_.generation != 0) stale_ = true;   // marked
    ...
}
auto r = backend_.enumerate();
if (!is_ok(r)) {
    if (published_.generation != 0) stale_ = true;   // marked
    ...
}
```

The first one did not:

```cpp
if (st.state == BackendState::Unavailable)
    return AudioError{AudioErrorCategory::BackendUnavailable, ...};   // not marked
```

A backend transitioning `Ready → Unavailable` mid-session (e.g. a fake
backend simulating server death, or a future backend whose context can be
torn down) would leave the retained catalog presented as fresh — the exact
hazard the `stale` flag exists to prevent.

## Steps to reproduce

Fake backend returning a snapshot, then switching `status()` to
`Unavailable`, then `refresh()` → `stale()` still reads `false`.

## Expected

Every failure exit marks the retained snapshot stale, per the stage 03 rule.

## Actual

The `Unavailable` exit skipped the marking.

## Fix (2026-10-18)

```cpp
if (st.state == BackendState::Unavailable) {
    if (published_.generation != 0) stale_ = true;
    return AudioError{AudioErrorCategory::BackendUnavailable, ...};
}
```

## Workaround (pre-fix)

None needed — unreachable with `MiniaudioBackend` (its init result is fixed
at construction); consistency fix for the seam contract.
