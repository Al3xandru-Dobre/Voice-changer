# third_party/

External dependencies live here in one of two forms:

| Dependency | How it arrives | Needed from |
|---|---|---|
| `miniaudio/` | Auto-fetched by CMake `FetchContent` (pinned `0.11.25`); manual vendoring optional | M1 |
| `readerwriterqueue/` | Auto-fetched by CMake `FetchContent` (pinned `v1.0.7`); manual vendoring optional | M1 |
| `nlohmann_json/` | Auto-fetched by CMake `FetchContent` (release tarball `v3.12.0`, pinned by SHA256); manual vendoring optional | M2 |
| `onnxruntime/` | **Manual** — download a prebuilt SDK from the onnxruntime releases page and pass `-DONNXRUNTIME_ROOT=<dir>` to CMake | M2 |
| `imgui/` | **Manual** — vendor Dear ImGui (+ backend sources) when building the dashboard | M4 |

Contents of these directories are git-ignored; only this and per-directory
`README.md` files are tracked.
