# BUG-018 — `std::getenv` in `frontend/core/Paths.cpp` breaks the Windows MSVC build

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-018` |
| **Name** | *Red Windows* — the frontend has never built on native Windows/MSVC |
| **Reported** | 2026-11-14 |
| **Severity** | High (the Windows leg of the build gate is red on `master`; `fluxvoice_gui` and `test_frontend_model` are not built or tested on the primary target OS) |
| **Status** | ☐ open |
| **Component** | `frontend/core/Paths.cpp` (`env_or_empty`), CI leg `windows-msvc` |
| **Blocks** | [Milestone 4 / Stage 01](../Milestone%204/01_frontend_shell_tray_mini_panel.md) (marked ☑), [Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md) |

## Summary

`frontend/core/Paths.cpp:21` calls `std::getenv`. MSVC reports the call as
C4996 ("may be unsafe"), and the project gate compiles with `/W4 /WX`, so the
warning is an error. `fluxvoice_frontend_core` fails to compile. MSBuild then
skips every target that depends on it, so `fluxvoice_gui` and
`test_frontend_model` are never compiled, linked or run on Windows.

The `windows-msvc` CI leg has failed on every run since the frontend landed:
the pull request itself, its merge commit and the Stage 08 commit.

## Environment

- GitHub Actions `windows-latest`, Visual Studio generator, Release,
  `-DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON`.
- Local reproduction was **NOT RUN**: this machine has no C++ toolchain as of
  2026-11-14 (the Visual Studio 18 directory is empty since 2026-11-12, and
  `cmake`, `cl`, `ninja`, `g++` and `clang++` are not installed).

## Steps to reproduce

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON
cmake --build build --config Release --parallel
```

## Expected

All targets build with warnings as errors, and CTest runs.

## Actual

```text
frontend\core\Paths.cpp(21,26): warning C4996: 'getenv': This function or variable may be unsafe.
    Consider using _dupenv_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS.
frontend\core\Paths.cpp(21,26): error C2220: the following warning is treated as an error
    [build\frontend\fluxvoice_frontend_core.vcxproj]
```

The Build step exits 1 and the Test step never runs.

## Evidence

| Run | Commit | Event | `windows-msvc` | Other legs |
|---|---|---|---|---|
| 42824977880 | `70b000b` (frontend PR #1) | pull_request | **failure** (Build) | linux-gcc, linux-clang, macos-clang, python-m2-contract: success |
| 42826469000 | `86ad15b` (merge of PR #1) | push `master` | **failure** | all others success |
| 42829208493 | `c299e9f` (M2 Stage 08) | push `master` | **failure** (Build) | macos-clang also fails — [BUG-019](BUG-019_macos_qos_call_wrong_arity.md) |

From the log of run 42829208493 (Windows job):

- Targets that built: `fluxvoice.exe`, `fv_probe.exe`, `fv_vc_file.exe`,
  `ola_sine_test.exe`, `fluxvoice_header_checks`, `fluxvoice_imgui`,
  `SDL3-static`, and 19 `test_*.exe` (including `test_voice_params`,
  `test_mode_a_worker`, `test_inference_pipeline`).
- Targets that did **not** build: `fluxvoice_frontend_core`, `fluxvoice_gui`,
  `test_frontend_model`. No file under `frontend/imgui/`, `frontend/platform/`
  or `frontend/app/` appears in the compiler output.
- The line is unchanged in the working tree on 2026-11-14
  (`frontend/core/Paths.cpp:21`); `git diff` shows no edit to the file.

## Consequences

1. **Hidden errors.** MSVC has never compiled `GuiApp.cpp`, `SdlTray.cpp`, the
   nine `View*.cpp` files, `Widgets.cpp`, `Theme.cpp`, `app/main.cpp` or
   `tests/test_frontend_model.cpp`. Fixing this line may reveal further
   `/W4` diagnostics in those files.
2. **No Windows test run for three commits.** The 19 engine tests built, but
   CTest did not run because the Build step failed.
3. **The documents overstate the state.** `Documents/Milestone 4/00_master.md`
   marks stages 01–03 ☑, and `frontend/README.md` §10 lists the Windows leg as
   "NOT RUN here (CI)". CI did run, and it failed. The pull request was merged
   with that check red.

## Root cause

A CRT function that MSVC deprecates was used in first-party code that was
only compiled with GCC and Clang before merging. The cross-platform workflow
(`AGENTS.md` rule 7, BEST_PRACTICES §5.6) requires native evidence per leg; the
red leg was not treated as blocking.

## Suggested fix

Do not define `_CRT_SECURE_NO_WARNINGS` and do not disable C4996 for the
target (rule 7: fix the warning at its source). Replace the call on Windows:

```cpp
#if defined(_WIN32)
// Wide read: %APPDATA% may contain characters outside the ANSI code page.
std::filesystem::path env_path(const wchar_t* name) {
    wchar_t* value = nullptr;
    std::size_t len = 0;
    if (_wdupenv_s(&value, &len, name) != 0 || value == nullptr) return {};
    std::filesystem::path p(value);
    std::free(value);
    return p;
}
#else
std::filesystem::path env_path(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::filesystem::path(v) : std::filesystem::path();
}
#endif
```

The wide variant also fixes a second problem in the current code: a narrow
`std::getenv("APPDATA")` returns the path in the ANSI code page, so a user
profile name that the code page cannot represent yields a wrong config
directory.

Then: rebuild the Windows leg, fix whatever the newly compiled GUI files
report, and require a green `windows-msvc` check before merging.

## Regression guard

- The CI leg is the guard; it already exists and already fails. Make the
  `build` checks required on `master` so a red leg cannot be merged.
- Add `default_config_dir()` cases to `tests/test_frontend_model.cpp` that set
  the environment variable and assert the result on each OS, so the function
  is compiled and executed by every leg.

## Workaround

None for a `/WX` build. A developer build without `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`
compiles the file with a warning.
