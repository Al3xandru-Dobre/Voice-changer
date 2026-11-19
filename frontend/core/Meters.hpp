// Meters — peak / RMS measurement and the dBFS conversion the UI displays.
//
// Toolkit-free and allocation-free. Used by the Bypass worker for its input
// and output meters; the Mode A meters come from fluxvoice::WorkerMeters
// (linear values) and go through the same to_dbfs() so both modes read alike.
#pragma once

#include <cmath>
#include <cstddef>

namespace fluxvoice::frontend::meters {

/// Linear amplitude -> dBFS. At or below 1e-6 (-120 dBFS) reports -120, the
/// value the UI treats as "silent / not running".
inline float to_dbfs(float lin) {
    return lin <= 1e-6f ? -120.0f : 20.0f * std::log10(lin);
}

/// Linear peak and RMS of a block, without modifying it.
inline void measure(const float* buf, std::size_t n, float& peak, float& rms) {
    peak = 0.0f;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = std::fabs(buf[i]);
        peak = a > peak ? a : peak;
        sum_sq += static_cast<double>(buf[i]) * static_cast<double>(buf[i]);
    }
    rms = n == 0 ? 0.0f : static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));
}

} // namespace fluxvoice::frontend::meters
