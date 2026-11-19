// LatencyProbe — marker signal and detector for measuring an audio route's
// delay (Milestone 1 / Stage 06). Used by tools/fv_probe.cpp.
//
// The probe plays a short chirp every `interval` samples and records what comes
// back on the same duplex stream, so playback and recording share one sample
// clock. Marker k starts at sample first + k*interval; the detector looks for
// it in the recording's window [first + k*interval, first + (k+1)*interval)
// and reports the lag of the best normalized cross-correlation, refined to a
// fraction of a sample by parabolic interpolation.
//
// So the route's delay must stay below `interval - marker length`, or a marker
// lands in the next window. The chirp (500 Hz – 3.5 kHz) passes FluxVoice's
// 16 kHz path; the resampler's filter delay is then part of the measured
// latency, as it should be.
//
// Three markers, one per kind of route:
//   chirp   20 ms waveform, matched by cross-correlation: routes that keep the
//           waveform (cables, passthrough, the queue loopback).
//   speech  150 ms /pa/ burst, matched by its log-energy envelope: Mode A voice
//           conversion, which replaces pitch and timbre but keeps the timing.
//   phrase  1.2 s speech-like phrase, measured by its energy edges
//           (PhraseDetector): Mode B simultaneous translation (Milestone 3 /
//           Stage 11), where the output is different speech that starts k
//           chunks late and keeps going after the input stops. Two delays per
//           trial: the onset delay and the flush tail lag.
//
// Pure DSP, no audio device: unit-tested in tests/test_latency_probe.cpp.
// Allocation-free after construction; the detectors run on the consumer
// thread, never in an audio callback.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace probe {

constexpr double kPi = 3.141592653589793;
constexpr double kSpeechBurstSeconds = 0.150;
constexpr double kPhraseSeconds = 1.200;  // 15 translation chunks of 80 ms

enum class MarkerKind {
    Chirp,   // 20 ms linear chirp matched by waveform cross-correlation
    Speech,  // 150 ms plosive speech burst matched by log-energy envelope correlation
    Phrase,  // 1.2 s speech-like phrase measured by its energy edges (PhraseDetector)
};

namespace detail {

struct Resonator {
    double b0 = 0.0, a1 = 0.0, a2 = 0.0, y1 = 0.0, y2 = 0.0;

    static Resonator make(double rate, double freq_hz, double bw_hz) noexcept {
        const double f = std::clamp(freq_hz, 60.0, 0.45 * rate);
        const double bw = std::clamp(bw_hz, 30.0, 0.25 * rate);
        const double r = std::exp(-kPi * bw / rate);
        Resonator res;
        res.a1 = -2.0 * r * std::cos(2.0 * kPi * f / rate);
        res.a2 = r * r;
        res.b0 = 1.0 - r;
        return res;
    }

    double step(double x) noexcept {
        const double y = b0 * x - a1 * y1 - a2 * y2;
        y2 = y1;
        y1 = y;
        return y;
    }

    void reset() noexcept { y1 = y2 = 0.0; }

    /// Move the resonance without touching the filter state (formant glides).
    void retune(double rate, double freq_hz, double bw_hz) noexcept {
        const Resonator r = make(rate, freq_hz, bw_hz);
        b0 = r.b0;
        a1 = r.a1;
        a2 = r.a2;
    }
};

inline double raised_gate(double t, double t0, double t1, double rise, double fall) noexcept {
    if (t <= t0 || t >= t1) return 0.0;
    if (rise > 0.0 && t < t0 + rise)
        return 0.5 * (1.0 - std::cos(kPi * (t - t0) / rise));
    if (fall > 0.0 && t > t1 - fall)
        return 0.5 * (1.0 - std::cos(kPi * (t1 - t) / fall));
    return 1.0;
}

} // namespace detail

/// When the markers are played, in samples of the probe's stream.
struct MarkerSchedule {
    std::uint64_t first    = 0;  // start of marker 0
    std::uint64_t interval = 0;  // between marker starts; also the search window
    std::size_t   count    = 0;

    std::uint64_t start(std::size_t k) const { return first + k * interval; }
    /// Samples the probe must play and record to cover every window.
    std::uint64_t total() const { return first + count * interval; }
};

/// Hann-windowed linear chirp from f0 to f1 Hz, `len` samples at `rate`.
inline std::vector<float> make_chirp(double rate, std::size_t len, double f0, double f1, double amp) {
    std::vector<float> c(len);
    const double dur = static_cast<double>(len) / rate;
    for (std::size_t i = 0; i < len; ++i) {
        const double t = static_cast<double>(i) / rate;
        const double phase = 2.0 * kPi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t);
        const double w = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                               static_cast<double>(len - 1)));
        c[i] = static_cast<float>(amp * w * std::sin(phase));
    }
    return c;
}

/// Deterministic ~150 ms plosive-vowel speech burst (/pa/) with a 20 ms silent
/// pre-onset margin, crisp 8 ms aspirated plosive attack, 100 ms voiced 3-formant
/// vowel body, and 25 ms silent post-margin so the silence-to-speech onset sits
/// cleanly inside [0, len) without sub-20-ms gaps that a 50 Hz VC model smears.
inline std::vector<float> make_speech_burst(double rate, double amp = 0.5) {
    const std::size_t len = std::max<std::size_t>(
        64, static_cast<std::size_t>(std::llround(rate * kSpeechBurstSeconds)));
    std::vector<float> out(len, 0.0f);

    auto burst1 = detail::Resonator::make(rate, 1850.0, 750.0);
    auto f1 = detail::Resonator::make(rate, 730.0, 90.0);
    auto f2 = detail::Resonator::make(rate, 1150.0, 115.0);
    auto f3 = detail::Resonator::make(rate, 2480.0, 170.0);

    std::uint32_t rng = 0x5A17C3E9u;
    auto white = [&]() noexcept {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<double>(rng >> 8) / 8388608.0 - 1.0;
    };

    const double max_harm_hz = std::min(3800.0, 0.44 * rate);
    double phase = 0.0;
    double peak = 0.0;

    for (std::size_t i = 0; i < len; ++i) {
        const double t = static_cast<double>(i) / rate;
        double sample = 0.0;

        if (t >= 0.020 && t < 0.030) {
            const double env = detail::raised_gate(t, 0.020, 0.030, 0.001, 0.004) *
                               std::exp(-(t - 0.020) / 0.0035);
            sample += 0.85 * env * burst1.step(white());
        }
        if (t >= 0.023 && t < 0.125) {
            const double u = (t - 0.023) / (0.125 - 0.023);
            const double f0 = 132.0 + 18.0 * std::sin(kPi * u);
            phase = std::fmod(phase + 2.0 * kPi * f0 / rate, 2.0 * kPi);
            const int n_harm = std::min(24, std::max(1, static_cast<int>(max_harm_hz / f0)));
            double glottal = 0.0;
            for (int h = 1; h <= n_harm; ++h)
                glottal += std::sin(static_cast<double>(h) * phase) / (1.0 + 0.10 * h * h);
            const double voiced = 1.00 * f1.step(glottal) +
                                  0.72 * f2.step(glottal) +
                                  0.36 * f3.step(glottal);
            const double env = detail::raised_gate(t, 0.023, 0.125, 0.004, 0.014) *
                               (1.0 - 0.35 * u);
            sample += env * voiced;
        }

        out[i] = static_cast<float>(sample);
        peak = std::max(peak, std::abs(sample));
    }

    const double scale = peak > 1e-12 ? amp / peak : 0.0;
    for (float& v : out) v = static_cast<float>(static_cast<double>(v) * scale);
    out.front() = 0.0f;
    out.back() = 0.0f;
    return out;
}

/// Deterministic speech-like phrase for Mode B (Milestone 3 / Stage 11): a
/// glottal harmonic source with a falling F0 through three gliding formants
/// (/a i u e o/, one vowel per 200 ms syllable), levelled so every vowel is
/// equally loud, with syllabic amplitude modulation that never drops below
/// 60 % (no gap a VAD could take for a pause) and 2 ms edges. The phrase fills
/// the whole vector: its onset is sample 0 and its offset is `size()`. At the
/// default 1.2 s it spans 15 chunks of 80 ms, long enough for WaitKPolicy to
/// reach kWrite at any k in [1..10].
inline std::vector<float> make_phrase(double rate, double seconds = kPhraseSeconds,
                                      double amp = 0.5) {
    const std::size_t len = std::max<std::size_t>(
        64, static_cast<std::size_t>(std::llround(rate * seconds)));
    // 30 ms of lead-in and lead-out around the phrase: the formant filters
    // have settled at sample 0, and the levelling window below is complete at
    // both ends. The edges are then as sharp as the 2 ms gate.
    const std::size_t margin = std::max<std::size_t>(
        2, static_cast<std::size_t>(std::llround(rate * 0.030)));
    const std::size_t half = margin / 2;  // levelling window: +/- 15 ms

    struct Vowel { double f1, f2, f3; };
    constexpr Vowel vowels[] = {
        {730.0, 1090.0, 2440.0},  // a
        {270.0, 2290.0, 3010.0},  // i
        {300.0,  870.0, 2240.0},  // u
        {530.0, 1840.0, 2480.0},  // e
        {570.0,  840.0, 2410.0},  // o
    };
    constexpr std::size_t n_vowels = sizeof(vowels) / sizeof(vowels[0]);
    constexpr double syllable_s = 0.200;

    auto f1 = detail::Resonator::make(rate, vowels[0].f1, 90.0);
    auto f2 = detail::Resonator::make(rate, vowels[0].f2, 120.0);
    auto f3 = detail::Resonator::make(rate, vowels[0].f3, 170.0);

    const double dur = static_cast<double>(len) / rate;
    const double max_harm_hz = std::min(3800.0, 0.44 * rate);
    const std::size_t total = len + 2 * margin;
    std::vector<double> voiced(total);
    std::vector<double> power(total + 1, 0.0);  // running sum of squares
    double phase = 0.0;

    auto syllable_pos = [&](double t) {  // position inside the current syllable, [0, 1)
        const double syl = std::clamp(t, 0.0, dur) / syllable_s;
        return syl - std::floor(syl);
    };

    for (std::size_t n = 0; n < total; ++n) {
        const double t = std::clamp(
            (static_cast<double>(n) - static_cast<double>(margin)) / rate, 0.0, dur);
        const auto idx = static_cast<std::size_t>(t / syllable_s);
        const double u = syllable_pos(t);
        // Glide to the next vowel over the last 30 % of each syllable.
        const double g = u < 0.7 ? 0.0 : 0.5 * (1.0 - std::cos(kPi * (u - 0.7) / 0.3));
        const Vowel& a = vowels[idx % n_vowels];
        const Vowel& b = vowels[(idx + 1) % n_vowels];
        f1.retune(rate, a.f1 + g * (b.f1 - a.f1), 90.0);
        f2.retune(rate, a.f2 + g * (b.f2 - a.f2), 120.0);
        f3.retune(rate, a.f3 + g * (b.f3 - a.f3), 170.0);

        // 135 -> 105 Hz declination with a small accent on each syllable.
        const double f0 = (135.0 - 30.0 * t / dur) * (1.0 + 0.08 * std::sin(kPi * u));
        phase = std::fmod(phase + 2.0 * kPi * f0 / rate, 2.0 * kPi);
        const int n_harm = std::min(24, std::max(1, static_cast<int>(max_harm_hz / f0)));
        double glottal = 0.0;
        for (int h = 1; h <= n_harm; ++h)
            glottal += std::sin(static_cast<double>(h) * phase) / (1.0 + 0.10 * h * h);
        voiced[n] = 1.00 * f1.step(glottal) + 0.70 * f2.step(glottal) + 0.35 * f3.step(glottal);
        power[n + 1] = power[n] + voiced[n] * voiced[n];
    }

    std::vector<float> out(len, 0.0f);
    double peak = 0.0;
    for (std::size_t i = 0; i < len; ++i) {
        const std::size_t n = i + margin;
        // A formant sitting on a harmonic (/i/, /u/) is ~13 dB louder than one
        // between harmonics (/a/): divide by the local RMS so the onset is not
        // the quietest part of the phrase.
        const double local = std::sqrt((power[n + half + 1] - power[n - half]) /
                                       static_cast<double>(2 * half + 1));
        const double t = static_cast<double>(i) / rate;
        const double sy = std::sin(kPi * syllable_pos(t));
        const double env = (0.6 + 0.4 * sy * sy) * detail::raised_gate(t, 0.0, dur, 0.002, 0.002);
        const double sample = local > 1e-9 ? env * voiced[n] / local : 0.0;
        out[i] = static_cast<float>(sample);
        peak = std::max(peak, std::abs(sample));
    }

    const double scale = peak > 1e-12 ? amp / peak : 0.0;
    for (float& v : out) v = static_cast<float>(static_cast<double>(v) * scale);
    return out;
}

/// The marker stream at sample `n`: the chirp inside a marker, 0 elsewhere.
/// Real-time safe (no allocation, no locks).
inline float marker_sample(const std::vector<float>& marker, const MarkerSchedule& s, std::uint64_t n) {
    if (n < s.first) return 0.0f;
    const std::uint64_t k = (n - s.first) / s.interval;
    const std::uint64_t off = (n - s.first) % s.interval;
    return (k < s.count && off < marker.size()) ? marker[static_cast<std::size_t>(off)] : 0.0f;
}

struct Detection {
    bool   found = false;
    double lag   = 0.0;  // samples after the marker's start (fractional)
    double score = 0.0;  // normalized cross-correlation at the lag, -1 … 1
};

/// Finds each scheduled marker in a recording fed in arbitrary-sized pieces.
class MarkerDetector {
public:
    /// `min_score`: weakest correlation accepted as the marker. Noise alone
    /// stays far below 0.5 for a 20 ms chirp and below 0.2 for the regularized
    /// speech log-energy envelope.
    MarkerDetector(std::vector<float> marker, MarkerSchedule sched, double min_score = 0.5,
                   MarkerKind kind = MarkerKind::Chirp, double rate = 48000.0)
        : marker_(std::move(marker)), sched_(sched), min_score_(min_score), kind_(kind),
          win_(static_cast<std::size_t>(sched.interval)),
          detections_(sched.count) {
        if (marker_.empty() || sched_.interval < marker_.size() || rate <= 0.0)
            throw std::invalid_argument("invalid MarkerDetector marker/schedule/rate");
        if (kind_ == MarkerKind::Phrase)
            throw std::invalid_argument("phrase markers are measured by PhraseDetector");

        if (kind_ == MarkerKind::Chirp) {
            score_.resize(static_cast<std::size_t>(sched.interval) - marker_.size() + 1);
            for (float m : marker_) marker_energy_ += static_cast<double>(m) * m;
        } else {
            env_step_ = std::max<std::size_t>(
                1, static_cast<std::size_t>(std::llround(rate * 0.001)));
            env_win_ = std::clamp<std::size_t>(
                static_cast<std::size_t>(std::llround(rate * 0.020)),
                4, std::max<std::size_t>(4, marker_.size() / 4));
            if (marker_.size() < env_win_)
                throw std::invalid_argument("speech marker shorter than envelope window");

            hann_.resize(env_win_);
            double wsum = 0.0;
            for (std::size_t j = 0; j < env_win_; ++j) {
                hann_[j] = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(j) /
                                                 static_cast<double>(env_win_ - 1)));
                wsum += hann_[j];
            }
            for (double& w : hann_) w /= wsum;

            ref_frames_ = (marker_.size() - env_win_) / env_step_ + 1;
            const std::size_t win_frames = (win_.size() - env_win_) / env_step_ + 1;
            env_diff_ = std::clamp<std::size_t>(10, 1, std::max<std::size_t>(1, ref_frames_ / 4));
            if (win_frames < ref_frames_ || ref_frames_ <= env_diff_)
                throw std::invalid_argument("search interval too short for speech envelope");

            slice_log_.resize(ref_frames_);
            ref_env_.resize(ref_frames_ - env_diff_);
            win_energy_.resize(win_frames);
            score_.resize(win_frames - ref_frames_ + 1);

            double ref_max = 0.0;
            for (std::size_t f = 0; f < ref_frames_; ++f) {
                slice_log_[f] = frame_energy(marker_.data() + f * env_step_);
                ref_max = std::max(ref_max, slice_log_[f]);
            }
            const double floor_e = std::max(1e-7, ref_max * 0.02);
            for (double& e : slice_log_) e = std::log(std::max(e, floor_e));
            for (std::size_t f = 0; f < ref_env_.size(); ++f) {
                const double flux = std::max(0.0, slice_log_[f + env_diff_] - slice_log_[f]);
                ref_env_[f] = flux;
                marker_energy_ += flux * flux;
            }
        }
    }

    /// Recorded samples, consecutive from sample 0 of the stream.
    void push(const float* x, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i, ++pos_) {
            if (next_ >= sched_.count) return;
            const std::uint64_t ws = sched_.start(next_);
            if (pos_ < ws) continue;
            win_[static_cast<std::size_t>(pos_ - ws)] = x[i];
            if (pos_ - ws + 1 == sched_.interval)
                detections_[next_++] =
                    (kind_ == MarkerKind::Chirp) ? analyse_waveform() : analyse_envelope();
        }
    }

    /// True once every marker's window has been analysed.
    bool done() const { return next_ >= sched_.count; }
    std::size_t analysed() const { return next_; }
    const std::vector<Detection>& detections() const { return detections_; }

private:
    double frame_energy(const float* p) const noexcept {
        double acc = 0.0;
        for (std::size_t j = 0; j < env_win_; ++j) {
            const double v = static_cast<double>(p[j]);
            acc += hann_[j] * v * v;
        }
        return acc;
    }

    Detection analyse_waveform() {
        const std::size_t len = marker_.size();
        const std::size_t lags = score_.size();

        double energy = 0.0;  // of win_[d, d + len), slid along with d
        for (std::size_t j = 0; j < len; ++j) energy += static_cast<double>(win_[j]) * win_[j];

        std::size_t best = 0;
        for (std::size_t d = 0; d < lags; ++d) {
            double c = 0.0;
            for (std::size_t j = 0; j < len; ++j) c += static_cast<double>(win_[d + j]) * marker_[j];
            score_[d] = energy > 1e-12 ? c / std::sqrt(energy * marker_energy_) : 0.0;
            if (score_[d] > score_[best]) best = d;
            if (d + 1 < lags) {
                energy += static_cast<double>(win_[d + len]) * win_[d + len] -
                          static_cast<double>(win_[d]) * win_[d];
                energy = std::max(energy, 0.0);
            }
        }

        Detection r;
        r.score = score_[best];
        r.found = r.score >= min_score_;
        r.lag = static_cast<double>(best);
        if (best > 0 && best + 1 < lags) {
            const double y0 = score_[best - 1], y1 = score_[best], y2 = score_[best + 1];
            const double denom = y0 - 2.0 * y1 + y2;
            if (denom < 0.0) r.lag += 0.5 * (y0 - y2) / denom;
        }
        return r;
    }

    Detection analyse_envelope() {
        const std::size_t win_frames = win_energy_.size();
        for (std::size_t f = 0; f < win_frames; ++f)
            win_energy_[f] = frame_energy(win_.data() + f * env_step_);

        const std::size_t lags = score_.size();
        const std::size_t flux_len = ref_env_.size();
        const std::size_t pre_end = std::max<std::size_t>(1, std::min<std::size_t>(8, ref_frames_ / 10));
        const std::size_t body_a  = std::min<std::size_t>(28, ref_frames_ - 1);
        const std::size_t body_b  = std::max<std::size_t>(body_a + 1, std::min<std::size_t>(95, ref_frames_));
        // Positive log-energy onset flux correlates the rising speech onset
        // while ignoring trailing noise-gate hold or causal vocoder ring-out;
        // the sustained-body check rejects isolated 1 ms transient clicks.
        const double mod_reg = static_cast<double>(flux_len) * 0.015;
        std::size_t best = 0;

        for (std::size_t d = 0; d < lags; ++d) {
            double slice_max = 0.0;
            for (std::size_t f = 0; f < ref_frames_; ++f)
                slice_max = std::max(slice_max, win_energy_[d + f]);
            if (slice_max < 1e-7 || marker_energy_ <= 1e-12) {
                score_[d] = 0.0;
                continue;
            }
            const double floor_e = std::max(1e-7, slice_max * 0.02);
            for (std::size_t f = 0; f < ref_frames_; ++f)
                slice_log_[f] = std::log(std::max(win_energy_[d + f], floor_e));

            double pre_mean = 0.0, body_mean = 0.0;
            for (std::size_t f = 0; f < pre_end; ++f) pre_mean += slice_log_[f];
            pre_mean /= static_cast<double>(pre_end);
            for (std::size_t f = body_a; f < body_b; ++f) body_mean += slice_log_[f];
            body_mean /= static_cast<double>(body_b - body_a);
            if (body_mean - pre_mean < 1.5) {  // require sustained vowel body above -17 dB clamp
                score_[d] = 0.0;
                continue;
            }

            double dot = 0.0, w_energy = 0.0;
            for (std::size_t f = 0; f < flux_len; ++f) {
                const double wf = std::max(0.0, slice_log_[f + env_diff_] - slice_log_[f]);
                dot += wf * ref_env_[f];
                w_energy += wf * wf;
            }
            const double denom = std::sqrt(marker_energy_ * (w_energy + mod_reg));
            score_[d] = denom > 1e-12 ? dot / denom : 0.0;
            if (score_[d] > score_[best]) best = d;
        }

        Detection r;
        r.score = score_[best];
        r.found = r.score >= min_score_;
        double best_frame = static_cast<double>(best);
        if (best > 0 && best + 1 < lags) {
            const double y0 = score_[best - 1], y1 = score_[best], y2 = score_[best + 1];
            const double denom = y0 - 2.0 * y1 + y2;
            if (denom < 0.0) best_frame += 0.5 * (y0 - y2) / denom;
        }
        r.lag = best_frame * static_cast<double>(env_step_);
        return r;
    }

    std::vector<float> marker_;
    double             marker_energy_ = 0.0;
    MarkerSchedule     sched_;
    double             min_score_;
    MarkerKind         kind_ = MarkerKind::Chirp;
    std::size_t        env_step_ = 1;
    std::size_t        env_win_ = 4;
    std::size_t        env_diff_ = 1;
    std::size_t        ref_frames_ = 0;
    std::vector<double> hann_;
    std::vector<double> ref_env_;
    std::vector<double> slice_log_;
    std::vector<double> win_energy_;
    std::vector<float> win_;    // current marker's window
    std::vector<double> score_; // correlation per lag
    std::vector<Detection> detections_;
    std::uint64_t pos_  = 0;         // stream index of the next pushed sample
    std::size_t   next_ = 0;         // marker whose window is being filled
};

/// Why a phrase trial produced no measurement.
enum class PhraseMiss : std::uint8_t {
    None,       // measured
    Silent,     // nothing above -80 dBFS came back
    LowSnr,     // the output is less than 16 dB above the window's noise floor
    NoSpeech,   // only clicks: no 20 ms run, or under 100 ms of output in total
    Overlap,    // output already running at the window start (previous tail, or a
                // delay below -pre_roll): raise --interval-ms
    Truncated,  // output still running at the window end: raise --interval-ms
};

inline const char* to_string(PhraseMiss m) noexcept {
    switch (m) {
    case PhraseMiss::None:      return "none";
    case PhraseMiss::Silent:    return "silent";
    case PhraseMiss::LowSnr:    return "low_snr";
    case PhraseMiss::NoSpeech:  return "no_speech";
    case PhraseMiss::Overlap:   return "overlap";
    case PhraseMiss::Truncated: return "truncated";
    }
    return "unknown";
}

/// One phrase trial. Delays are in samples and relative to the played phrase:
/// `onset_delay` is the output's first energy edge minus the phrase's onset
/// (Mode B: route delay + Wait-k buffering), `flush_tail` is the output's last
/// energy edge minus the phrase's offset (route delay + the translated tail
/// drained after the speaker stopped). A pure delay gives the same value twice.
struct PhraseDetection {
    bool       found = false;
    PhraseMiss miss = PhraseMiss::Silent;
    double onset_delay = 0.0;
    double flush_tail  = 0.0;
    double duration    = 0.0;   // output offset edge - output onset edge
    double peak_dbfs   = -200.0;  // loudest 4 ms frame of the window
    double floor_dbfs  = -200.0;  // 10th-percentile frame: the route's noise floor
};

/// Thresholds of the edge detector. The defaults suit a digital cable route.
struct PhraseConfig {
    double below_peak_db   = 30.0;  // an edge is where the level crosses peak - 30 dB ...
    double above_floor_db  = 10.0;  // ... but never less than 10 dB above the noise floor
    double min_headroom_db = 6.0;   // and the threshold must stay 6 dB under the peak
    double silent_dbfs     = -80.0;
    double frame_ms        = 4.0;   // energy window
    double step_ms         = 1.0;   // coarse search step (edges are refined to one sample)
    double min_run_ms      = 20.0;  // shorter bursts are clicks, not speech
    double bridge_ms       = 5.0;   // dips this short do not split a run
    double min_active_ms   = 100.0; // total output needed for a measurement
    double pre_roll_ms     = 50.0;  // each window opens this long before its phrase
};

/// Measures the onset delay and flush tail lag of each scheduled phrase in a
/// recording fed in arbitrary-sized pieces.
///
/// A translator answers the phrase with different speech, so nothing can be
/// correlated: the detector works on short-time energy alone. Per trial it
/// takes the window [start(k) - pre_roll, start(k) - pre_roll + interval),
/// puts a threshold 30 dB under the window's peak (at least 10 dB over its
/// noise floor), and reports the start of the first and the end of the last
/// run of at least 20 ms above it. Pauses inside the output do not matter;
/// clicks shorter than 20 ms are ignored. Both edges are refined to one
/// sample, and calibrated against the same analysis of the clean phrase, so a
/// pure delay of D samples reads D for both figures.
///
/// The window must hold the whole answer plus silence on both sides: a trial
/// whose output touches either end of its window is reported as a miss
/// (Overlap / Truncated) instead of a wrong number.
class PhraseDetector {
public:
    PhraseDetector(std::vector<float> phrase, MarkerSchedule sched, double rate,
                   const PhraseConfig& config = {})
        : sched_(sched), cfg_(config), detections_(sched.count) {
        if (phrase.empty() || rate <= 0.0 || sched_.interval == 0)
            throw std::invalid_argument("invalid PhraseDetector phrase/schedule/rate");
        if (!(cfg_.below_peak_db > cfg_.min_headroom_db) || !(cfg_.above_floor_db >= 0.0) ||
            !(cfg_.frame_ms > 0.0) || !(cfg_.step_ms > 0.0) || !(cfg_.pre_roll_ms >= 0.0))
            throw std::invalid_argument("invalid PhraseDetector thresholds");

        auto samples = [rate](double ms) {
            return std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(rate * ms / 1000.0)));
        };
        win_len_ = samples(cfg_.frame_ms);
        step_    = samples(cfg_.step_ms);
        pre_     = static_cast<std::size_t>(std::llround(rate * cfg_.pre_roll_ms / 1000.0));
        min_run_ = std::max<std::size_t>(1, samples(cfg_.min_run_ms) / step_);
        bridge_  = samples(cfg_.bridge_ms) / step_;
        min_active_ = std::max<std::size_t>(1, samples(cfg_.min_active_ms) / step_);
        silent_e_ = std::pow(10.0, cfg_.silent_dbfs / 10.0);

        const std::size_t interval = static_cast<std::size_t>(sched_.interval);
        // Room for the phrase with silence on both sides, or no window could
        // ever hold a complete answer.
        if (interval < phrase.size() + 2 * pre_ + 2 * win_len_ + step_)
            throw std::invalid_argument("phrase interval too short for the phrase and its margins");
        if (sched_.first < pre_)
            throw std::invalid_argument("first phrase must start after the detector's pre-roll");

        win_.assign(interval, 0.0f);
        const std::size_t frames = (interval - win_len_) / step_ + 1;
        energy_.resize(frames);
        scratch_.resize(frames);

        // Calibration: the same analysis of the clean phrase gives the edges a
        // zero-delay route would report.
        std::copy(phrase.begin(), phrase.end(), win_.begin() + static_cast<std::ptrdiff_t>(pre_));
        const Edges ref = analyse();
        if (ref.miss != PhraseMiss::None)
            throw std::invalid_argument("phrase has no detectable onset/offset");
        ref_onset_  = ref.onset;
        ref_offset_ = ref.offset;
        std::fill(win_.begin(), win_.end(), 0.0f);
    }

    /// Recorded samples, consecutive from sample 0 of the stream.
    void push(const float* x, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i, ++pos_) {
            if (next_ >= sched_.count) return;
            const std::uint64_t ws = sched_.start(next_) - pre_;
            if (pos_ < ws) continue;
            win_[static_cast<std::size_t>(pos_ - ws)] = x[i];
            if (pos_ - ws + 1 == sched_.interval) detections_[next_++] = measure();
        }
    }

    /// True once every trial's window has been analysed.
    bool done() const { return next_ >= sched_.count; }
    std::size_t analysed() const { return next_; }
    const std::vector<PhraseDetection>& detections() const { return detections_; }

    /// Samples the recording must cover: the last window closes `pre_roll`
    /// before the schedule's total().
    std::uint64_t required_samples() const { return sched_.total() - pre_; }
    /// Longest flush tail a window can hold, in samples: the room between the
    /// phrase's own offset and the window end.
    double max_flush_tail() const {
        return static_cast<double>(sched_.interval) - ref_offset_ -
               static_cast<double>(win_len_ + step_);
    }

private:
    struct Edges {
        PhraseMiss miss = PhraseMiss::Silent;
        double onset = 0.0, offset = 0.0;  // samples from the window start
        double peak = 0.0, floor = 0.0;    // mean-square
    };

    double mean_square(std::size_t at) const noexcept {
        double acc = 0.0;
        for (std::size_t j = 0; j < win_len_; ++j) {
            const double v = static_cast<double>(win_[at + j]);
            acc += v * v;
        }
        return acc / static_cast<double>(win_len_);
    }

    Edges analyse() {
        Edges r;
        const std::size_t frames = energy_.size();
        for (std::size_t f = 0; f < frames; ++f) {
            energy_[f] = mean_square(f * step_);
            r.peak = std::max(r.peak, energy_[f]);
        }
        std::copy(energy_.begin(), energy_.end(), scratch_.begin());
        const auto tenth = scratch_.begin() + static_cast<std::ptrdiff_t>(frames / 10);
        std::nth_element(scratch_.begin(), tenth, scratch_.end());
        r.floor = *tenth;

        if (!(r.peak > silent_e_)) return r;  // PhraseMiss::Silent

        // Runs of frames above the threshold; dips up to bridge_ frames stay
        // inside a run, runs shorter than min_run_ are clicks. The first pass
        // takes the threshold from the window's peak; the second from the
        // peak of the runs it found, so a loud click cannot lift the
        // threshold and move the edges of quieter speech.
        constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
        const double floor_thr = r.floor * std::pow(10.0, cfg_.above_floor_db / 10.0);
        const double below = std::pow(10.0, -cfg_.below_peak_db / 10.0);
        const double headroom = std::pow(10.0, -cfg_.min_headroom_db / 10.0);
        double peak = r.peak, thr = 0.0;
        std::size_t first_on = none, last_off = 0, active = 0;
        for (int pass = 0; pass < 2; ++pass) {
            thr = std::max(peak * below, floor_thr);
            if (thr > peak * headroom) {
                r.miss = PhraseMiss::LowSnr;
                return r;
            }
            std::size_t run_start = none, last_active = 0;
            double run_peak = 0.0, speech_peak = 0.0;
            first_on = none;
            last_off = 0;
            active = 0;
            auto close_run = [&]() {
                const std::size_t len = last_active - run_start + 1;
                if (len < min_run_) return;
                if (first_on == none) first_on = run_start;
                last_off = last_active;
                active += len;
                speech_peak = std::max(speech_peak, run_peak);
            };
            for (std::size_t f = 0; f < frames; ++f) {
                if (!(energy_[f] > thr)) continue;
                if (run_start == none) {
                    run_start = f;
                    run_peak = 0.0;
                } else if (f - last_active - 1 > bridge_) {
                    close_run();
                    run_start = f;
                    run_peak = 0.0;
                }
                last_active = f;
                run_peak = std::max(run_peak, energy_[f]);
            }
            if (run_start != none) close_run();
            if (first_on == none || active < min_active_) {
                r.miss = PhraseMiss::NoSpeech;
                return r;
            }
            if (speech_peak >= peak) break;  // the peak was speech: nothing to redo
            peak = speech_peak;
        }
        if (first_on == 0) {
            r.miss = PhraseMiss::Overlap;
            return r;
        }
        if (last_off + 1 >= frames) {
            r.miss = PhraseMiss::Truncated;
            return r;
        }

        // Refine to one sample. Frame first_on - 1 is below the threshold and
        // frame first_on above it, so the first window to cross lies between
        // them; its last sample is where the output's energy arrived.
        std::size_t p = first_on * step_;
        for (std::size_t q = (first_on - 1) * step_ + 1; q < first_on * step_; ++q)
            if (mean_square(q) > thr) { p = q; break; }
        r.onset = static_cast<double>(p + win_len_ - 1);
        // Likewise the last window still above the threshold starts on the
        // output's final audible sample.
        p = last_off * step_;
        for (std::size_t q = (last_off + 1) * step_ - 1; q > last_off * step_; --q)
            if (mean_square(q) > thr) { p = q; break; }
        r.offset = static_cast<double>(p + 1);
        r.miss = PhraseMiss::None;
        return r;
    }

    PhraseDetection measure() {
        const Edges e = analyse();
        PhraseDetection d;
        d.miss = e.miss;
        d.found = e.miss == PhraseMiss::None;
        d.peak_dbfs  = 10.0 * std::log10(e.peak + 1e-20);
        d.floor_dbfs = 10.0 * std::log10(e.floor + 1e-20);
        if (d.found) {
            d.onset_delay = e.onset - ref_onset_;
            d.flush_tail  = e.offset - ref_offset_;
            d.duration    = e.offset - e.onset;
        }
        return d;
    }

    MarkerSchedule sched_;
    PhraseConfig   cfg_;
    std::size_t win_len_ = 1, step_ = 1, pre_ = 0;
    std::size_t min_run_ = 1, bridge_ = 0, min_active_ = 1;
    double silent_e_ = 0.0;
    double ref_onset_ = 0.0, ref_offset_ = 0.0;  // clean phrase's edges, from the window start
    std::vector<float>  win_;      // current trial's window
    std::vector<double> energy_;   // mean-square per coarse frame
    std::vector<double> scratch_;  // percentile work area
    std::vector<PhraseDetection> detections_;
    std::uint64_t pos_  = 0;  // stream index of the next pushed sample
    std::size_t   next_ = 0;  // trial whose window is being filled
};

/// Statistics over the found markers, in samples.
struct Summary {
    std::size_t found = 0, missed = 0;
    double min = 0, median = 0, p95 = 0, max = 0;
    /// Least-squares change of the lag per minute of stream time: a route
    /// whose clocks drift apart shows a steady slope (or a sawtooth).
    double slope_per_min = 0;
};

namespace detail {

/// Median / p95 / range / drift slope of `lags`, measured at `t` minutes.
inline Summary lag_stats(std::vector<double> lags, const std::vector<double>& t) {
    Summary r;
    r.found = lags.size();
    if (lags.empty()) return r;

    if (lags.size() >= 2) {
        double mt = 0, ml = 0;
        for (std::size_t i = 0; i < lags.size(); ++i) { mt += t[i]; ml += lags[i]; }
        mt /= static_cast<double>(lags.size());
        ml /= static_cast<double>(lags.size());
        double num = 0, den = 0;
        for (std::size_t i = 0; i < lags.size(); ++i) {
            num += (t[i] - mt) * (lags[i] - ml);
            den += (t[i] - mt) * (t[i] - mt);
        }
        r.slope_per_min = den > 0 ? num / den : 0;
    }

    std::sort(lags.begin(), lags.end());
    const std::size_t n = lags.size();
    r.min    = lags.front();
    r.max    = lags.back();
    r.median = n % 2 ? lags[n / 2] : 0.5 * (lags[n / 2 - 1] + lags[n / 2]);
    // Nearest-rank percentile.
    r.p95    = lags[static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1];
    return r;
}

} // namespace detail

inline Summary summarize(const std::vector<Detection>& dets, const MarkerSchedule& s, double rate) {
    std::size_t missed = 0;
    std::vector<double> lags, t;
    for (std::size_t k = 0; k < dets.size(); ++k) {
        if (!dets[k].found) { ++missed; continue; }
        lags.push_back(dets[k].lag);
        t.push_back(static_cast<double>(s.start(k)) / rate / 60.0);
    }
    Summary r = detail::lag_stats(std::move(lags), t);
    r.missed = missed;
    return r;
}

/// Statistics over the measured phrase trials, in samples. `found` / `missed`
/// are repeated in each member.
struct PhraseSummary {
    std::size_t found = 0, missed = 0;
    Summary onset;     // onset_delay
    Summary tail;      // flush_tail
    Summary duration;  // output length
};

inline PhraseSummary summarize_phrase(const std::vector<PhraseDetection>& dets,
                                      const MarkerSchedule& s, double rate) {
    PhraseSummary r;
    std::vector<double> onset, tail, duration, t;
    for (std::size_t k = 0; k < dets.size(); ++k) {
        if (!dets[k].found) { ++r.missed; continue; }
        onset.push_back(dets[k].onset_delay);
        tail.push_back(dets[k].flush_tail);
        duration.push_back(dets[k].duration);
        t.push_back(static_cast<double>(s.start(k)) / rate / 60.0);
    }
    r.found = onset.size();
    r.onset    = detail::lag_stats(std::move(onset), t);
    r.tail     = detail::lag_stats(std::move(tail), t);
    r.duration = detail::lag_stats(std::move(duration), t);
    r.onset.missed = r.tail.missed = r.duration.missed = r.missed;
    return r;
}

} // namespace probe
