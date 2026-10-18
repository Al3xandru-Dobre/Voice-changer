// Biquad — 2nd-order IIR section, transposed direct form II, double state.
//
// Same structure and coefficient formula as the Python reference
// (python_training/fvml/features.py: highpass_biquad + scipy.signal.lfilter),
// so the Stage 03 pitch tracker's pre-filter matches it to rounding error.
//
// Allocation-free; safe on the inference worker thread.
#pragma once

#include <cmath>

struct BiquadCoefficients {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;   // numerator
    double a1 = 0.0, a2 = 0.0;             // denominator, a0 normalised to 1

    /// RBJ audio-EQ-cookbook high-pass at `fc` Hz.
    static BiquadCoefficients high_pass(double fc, double sample_rate, double q) {
        constexpr double kPi = 3.14159265358979323846;
        const double w0 = 2.0 * kPi * fc / sample_rate;
        const double cosw = std::cos(w0);
        const double alpha = std::sin(w0) / (2.0 * q);
        const double a0 = 1.0 + alpha;
        BiquadCoefficients c;
        c.b0 = (1.0 + cosw) / 2.0 / a0;
        c.b1 = -(1.0 + cosw) / a0;
        c.b2 = (1.0 + cosw) / 2.0 / a0;
        c.a1 = -2.0 * cosw / a0;
        c.a2 = (1.0 - alpha) / a0;
        return c;
    }
};

class Biquad {
public:
    Biquad() = default;
    explicit Biquad(const BiquadCoefficients& c) : c_(c) {}

    double process(double x) {
        const double y = c_.b0 * x + z1_;
        z1_ = c_.b1 * x + z2_ - c_.a1 * y;
        z2_ = c_.b2 * x - c_.a2 * y;
        return y;
    }

    void reset() { z1_ = z2_ = 0.0; }

    const BiquadCoefficients& coefficients() const { return c_; }

private:
    BiquadCoefficients c_;
    double z1_ = 0.0;
    double z2_ = 0.0;
};
