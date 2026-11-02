# imgui/

Dear ImGui — the widget library of the Milestone 4 frontend (`fluxvoice_gui`),
compiled into the `fluxvoice_imgui` static library together with its SDL3
platform backend and OpenGL 3 renderer backend (own loader; no system GL
headers needed).

Auto-fetched by CMake `FetchContent` (git tag `v1.92.9`, shallow). With
`-DFLUXVOICE_FETCH_DEPS=OFF` clone the same tag here so that
`third_party/imgui/imgui.cpp` and `third_party/imgui/backends/` exist.
Upstream: <https://github.com/ocornut/imgui>
