# BUG-019 — `WorkerPriority` calls `pthread_get_qos_class_np` with the wrong signature (macOS build broken)

> [Bug index](00_index.md) · Found by: [REVIEW-M2-S07-S09_2026-11-14](REVIEW-M2-S07-S09_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-019` |
| **Name** | *Two-arg QoS* — the macOS branch of `WorkerPriority` does not compile |
| **Reported** | 2026-11-14 |
| **Severity** | High for the build gate (macOS leg red on `master`); no effect on Windows or Linux binaries |
| **Status** | ◐ fixed in tree 2026-11-15 — native legs pending, see [Resolution](#resolution) |
| **Component** | `src/util/WorkerPriority.hpp:36` (M2 Stage 08) |
| **Blocks** | [Milestone 2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) ("Worker priority … native Windows/macOS execution pending"), [Milestone 4 / Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md) |

## Summary

The `__APPLE__` branch reads the thread's current QoS class like this:

```cpp
old_qos_ = pthread_get_qos_class_np(pthread_self(), &old_relative_);   // WorkerPriority.hpp:36
```

The SDK declares the function with three parameters and an `int` return
value (the class is an out-parameter):

```c
int pthread_get_qos_class_np(pthread_t __pthread,
                             qos_class_t * _Nullable __restrict __qos_class,
                             int * _Nullable __restrict __relative_priority);
```

AppleClang rejects the call twice: no matching function for two arguments,
and `int` cannot be assigned to `qos_class_t`. Every translation unit that
includes the header fails, which takes down `fluxvoice` and
`test_worker_priority` on macOS.

## Environment

- GitHub Actions `macos-latest`, AppleClang 21.0.0, Xcode 26.6 SDK,
  `-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0`, warnings as errors.
- Introduced by `c299e9f` (M2 Stage 08). The macOS leg passed on the two
  previous commits (`70b000b`, `86ad15b`).
- Local reproduction: NOT RUN (no macOS host).

## Steps to reproduce

Build any target that includes `util/WorkerPriority.hpp` with AppleClang:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
cmake --build build --config Release --parallel
```

## Expected

The header compiles on all three OS branches.

## Actual (CI run 42829208493, job `macos-clang`)

```text
src/util/WorkerPriority.hpp:36:20: error: no matching function for call to 'pthread_get_qos_class_np'
   36 |         old_qos_ = pthread_get_qos_class_np(pthread_self(), &old_relative_);
pthread/qos.h:186:1: note: candidate function not viable: requires 3 arguments, but 2 were provided
src/util/WorkerPriority.hpp:36:20: error: assigning to 'qos_class_t' from incompatible type 'int'
make[2]: *** [CMakeFiles/test_worker_priority.dir/tests/test_worker_priority.cpp.o] Error 1
make[2]: *** [CMakeFiles/fluxvoice.dir/src/main.cpp.o] Error 1
```

The line is unchanged in the working tree on 2026-11-14.

## Root cause

The macOS branch was written from memory of the API and never compiled:
Stage 08 was developed on Windows and Linux, and its stage file records the
macOS execution as pending. The header is behind `#elif defined(__APPLE__)`,
so the Windows and Linux legs and `fluxvoice_header_checks` on those hosts
cannot see the error.

## Suggested fix

```cpp
#elif defined(__APPLE__)
        method_ = "QoS user-interactive";
        if (pthread_get_qos_class_np(pthread_self(), &old_qos_, &old_relative_) != 0)
            old_qos_ = QOS_CLASS_UNSPECIFIED;   // restore is skipped below
        result_ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        elevated_ = result_ == 0;
```

Also add `<pthread/qos.h>` to the `__APPLE__` include block: the header that
declares both functions should be included by name rather than reached
through `<pthread.h>` (rule 6, own every include).

## Regression guard

`tests/test_worker_priority.cpp` already includes the header, so the macOS
CI leg is the guard once it is required for merging. No new test is needed
for the compile error. For behaviour, extend the test on macOS to assert that
the destructor restores the QoS class that was read before elevation.

## Workaround

None on macOS. Windows and Linux are unaffected.

## Resolution

**2026-11-15: fix in tree, macOS leg pending.**

- `src/util/WorkerPriority.hpp`: the class is read through the out-parameter,
  `pthread_get_qos_class_np(pthread_self(), &old_qos_, &old_relative_)`. On a
  non-zero return `old_qos_` stays `QOS_CLASS_UNSPECIFIED`, so the destructor
  skips the restore. `<pthread/qos.h>` is included by name (rule 6).
- `tests/test_worker_priority.cpp` (`__APPLE__` only) runs on a scratch thread
  set to `QOS_CLASS_UTILITY`. It checks that elevation reports
  `QoS user-interactive`, that the class is `USER_INTERACTIVE` while the guard
  lives, and that the destructor restores the previous class and relative
  priority. The main thread's QoS is never changed.

| Leg | Evidence |
|---|---|
| Windows / zig 0.15.2 (Clang 20), Ninja, strict warnings + `-Werror`, `FLUXVOICE_BUILD_GUI=OFF` | Full build incl. `fluxvoice_header_checks`: pass. CTest 82/82 pass (`build-zig-bug1921`). |
| macOS / AppleClang | **NOT RUN** (no macOS host). Only the `macos-clang` CI job can verify the `__APPLE__` branch. |
| Windows / MSVC, Linux / GCC, Linux / Clang | NOT RUN locally. The header's branches for these platforms are unchanged. |
