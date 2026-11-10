# onnxruntime/

Prebuilt ONNX Runtime C++ SDK (CPU Execution Provider) — required from Milestone 2
(Stage 07) for stateful ONNX inference of the encoder / translator / vocoder.

**Pinned version:** `1.31.0` (`v1.31.0`, matching `onnxruntime==1.31.0` in
`python_training/pyproject.toml`). `OrtModel` checks `Ort::GetVersionString()`
against this pin at load time and refuses mismatched libraries (preventing
`C:\Windows\System32\onnxruntime.dll` shadowing on Windows 11).

Not vendored in git. Either pass `-DFLUXVOICE_FETCH_ORT=ON` to CMake (fetches and
verifies the matching archive via `FetchContent`), or download the archive for
your platform from
<https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0>, unpack it here
(or anywhere), and configure with `-DONNXRUNTIME_ROOT=<path>`:

| Platform / Arch | Archive | URL | SHA256 |
|---|---|---|---|
| Windows x64 | `onnxruntime-win-x64-1.31.0.zip` | <https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip> | `3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195` |
| Windows arm64 | `onnxruntime-win-arm64-1.31.0.zip` | <https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-arm64-1.31.0.zip> | `ef6206fa8d6975878d93a97035d280240ce5764c126a7430e5427e18923316d5` |
| Linux x64 | `onnxruntime-linux-x64-1.31.0.tgz` | <https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-1.31.0.tgz> | `cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57` |
| Linux aarch64 | `onnxruntime-linux-aarch64-1.31.0.tgz` | <https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-aarch64-1.31.0.tgz> | `c5b8b3cca31f3d643a3b313b8f42e2d0f16a4fcefd02cda343d97bd5306afaab` |
| macOS arm64 | `onnxruntime-osx-arm64-1.31.0.tgz` | <https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-osx-arm64-1.31.0.tgz> | `1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc` |

> Note: ONNX Runtime `v1.31.0` publishes `osx-arm64` for macOS (no `osx-x86_64` or
> `osx-universal2` archive is published upstream for 1.31.0).

For GPU inference in Milestone 4, prefer a build with CUDA (Linux/Windows),
DirectML/WinML (Windows), or CoreML (macOS) execution providers — see Milestone 4
optimization notes.

