# onnxruntime/

Prebuilt ONNX Runtime C++ SDK — required from Milestone 2 for stateful ONNX
inference of the encoder / translator / vocoder.

Not vendored in git. Download the release matching your platform
(`onnxruntime-<os>-<arch>-<ver>`) from
<https://github.com/microsoft/onnxruntime/releases>, unpack it here (or
anywhere), and configure with `-DONNXRUNTIME_ROOT=<path>`.

For GPU inference prefer a build with CUDA (Linux/Windows), DirectML (Windows),
or CoreML (macOS) execution providers — see Milestone 4 optimization notes.
