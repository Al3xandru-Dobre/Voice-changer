# SDL3/

Simple DirectMedia Layer 3 — window, OpenGL context, events, native file
dialogs and the system tray for the Milestone 4 frontend (`fluxvoice_gui`).

Auto-fetched by CMake `FetchContent` (release tarball `SDL3-3.4.18.tar.gz`,
pinned by SHA-256) and built static with audio, joystick, haptic, sensor,
camera, GPU, render and HIDAPI subsystems off. With `-DFLUXVOICE_FETCH_DEPS=OFF`
unpack the same release here so that `third_party/SDL3/CMakeLists.txt` exists.

Linux build/runtime notes: X11 headers (`libx11-dev libxext-dev`) are enough to
build; add `libwayland-dev libxkbcommon-dev libegl1-mesa-dev libdecor-0-dev`
for native Wayland. The tray needs `libayatana-appindicator3-1` (or
`libappindicator3-1`) at runtime; GNOME also needs the AppIndicator extension.
Upstream: <https://github.com/libsdl-org/SDL>
