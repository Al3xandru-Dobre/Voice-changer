// RunningNorm — exponential moving mean / standard deviation for whitening.
//
// Port of RunningNorm in python_training/fvml/features.py (Milestone 2 /
// Stage 03): starts from population priors, includes the current value before
// normalising it (still causal), and never lets the std drop below `floor`.
//
// Allocation-free; safe on the inference worker thread.
#pragma once

#include <algorithm>
#include <cmath>

class RunningNorm {
public:
    RunningNorm() = default;
    RunningNorm(double prior_mean, double prior_std, double alpha, double floor)
        : alpha_(alpha), floor_(floor), prior_mean_(prior_mean), prior_var_(prior_std * prior_std),
          mean_(prior_mean), var_(prior_std * prior_std) {}

    void update(double x) {
        const double delta = x - mean_;
        mean_ += alpha_ * delta;
        var_ = (1.0 - alpha_) * (var_ + alpha_ * delta * delta);
    }

    double mean() const { return mean_; }
    double std() const { return std::max(std::sqrt(var_), floor_); }
    double z(double x) const { return (x - mean_) / std(); }

    /// Back to the priors (stream restart).
    void reset() {
        mean_ = prior_mean_;
        var_ = prior_var_;
    }

private:
    double alpha_ = 0.0;
    double floor_ = 0.0;
    double prior_mean_ = 0.0;
    double prior_var_ = 1.0;
    double mean_ = 0.0;
    double var_ = 1.0;
};
