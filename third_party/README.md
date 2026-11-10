# third_party/

External dependencies live here in one of two forms:

| Dependency | How it arrives | Needed from |
|---|---|---|
| `miniaudio/` | Auto-fetched by CMake `FetchContent` (pinned `0.11.25`); manual vendoring optional | M1 |
| `readerwriterqueue/` | Auto-fetched by CMake `FetchContent` (pinned `v1.0.7`); manual vendoring optional | M1 |
| `nlohmann_json/` | Auto-fetched by CMake `FetchContent` (release tarball `v3.12.0`, pinned by SHA256); manual vendoring optional | M2 |
| `onnxruntime/` | **Manual or fetched** — pass `-DONNXRUNTIME_ROOT=<dir>` or `-DFLUXVOICE_FETCH_ORT=ON` (pinned `v1.31.0`, SHA256-verified) | M2 |
| `imgui/` | Auto-fetched by CMake `FetchContent` (pinned `v1.92.9`, shallow); compiled into `fluxvoice_imgui` with the SDL3 + OpenGL 3 backends; manual vendoring optional | M4 |
| `SDL3/` | Auto-fetched by CMake `FetchContent` (release tarball `3.4.18`, pinned by SHA256; static, trimmed subsystems); manual vendoring optional | M4 |

Contents of these directories are git-ignored; only this and per-directory
`README.md` files are tracked.
