// VoiceParams — the single-source parameter descriptor table and the value
// type the control thread posts to the inference worker.
//
// Milestone 2 / Stage 09 defines the voice controls (pitch, intonation, timbre
// blend, output chain, stretch-tier extras); Milestone 3 / Stage 09 adds the
// translation controls (translate, wait_k, adaptive_policy) and the language
// ids. One table, four consumers: CLI flags, console commands, preset JSON keys
// and the Milestone 4 UI sliders all read `kParams`. Adding a parameter is one
// row here plus its apply code in InferencePipeline; nothing else duplicates a
// range or a default (AGENTS.md, @ui-telemetry rule 1).
//
// Controls act on the vocoder's *conditioning*, never on the audio path, except
// the output chain (gain -> gate -> limiter). There is deliberately no dry/wet
// mix: the wet path lags the microphone by the pipeline latency, so mixing
// comb-filters and lets the real voice through (M2 research §11).
//
// `VoiceParams` is trivially copyable so a full snapshot can travel through a
// preallocated moodycamel::ReaderWriterQueue<VoiceParams> without allocation.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace fluxvoice {

enum class ParamId : std::uint8_t {
    PitchShift,
    Intonation,
    TimbreBlend,
    Gain,
    GateThreshold,
    Limiter,
    VibratoRate,
    VibratoDepth,
    Whisper,
    Brightness,
    Translate,
    WaitK,
    AdaptivePolicy,
    Count
};

inline constexpr std::size_t kParamCount = static_cast<std::size_t>(ParamId::Count);

constexpr std::size_t param_index(ParamId id) { return static_cast<std::size_t>(id); }

/// Continuous: any float in [min, max]. Integer: integral steps in [min, max].
/// Toggle: exactly 0 or 1 (min 0, max 1).
enum class ParamKind : std::uint8_t { Continuous, Integer, Toggle };

/// MVP parameters are always shown; Stretch parameters sit under "Advanced"
/// in the UI and may be unimplemented by a given bundle.
enum class ParamTier : std::uint8_t { Mvp, Stretch };

/// UI grouping only; carries no engine semantics.
enum class ParamGroup : std::uint8_t { Pitch, Timbre, Output, Translation };

/// How the worker transitions when the value changes (stage-file column
/// "Smoothing" / "Transition behaviour").
enum class Smoothing : std::uint8_t {
    None,              // applied at the next hop
    PerFrameRamp,      // ramped per 20 ms frame (conditioning values)
    PerSampleOnePole,  // ~20 ms one-pole per sample (gain)
    Envelope,          // attack/hold/release envelope (gate)
    OutputRamp,        // 20 ms cosine fade-out/in across a state reset (translate)
};

struct ParamDesc {
    ParamId          id;
    std::string_view name;   // preset key, CLI/console name: lowercase [a-z0-9_]
    std::string_view label;  // UI label
    std::string_view unit;   // "" for dimensionless
    float            min;
    float            max;
    float            def;
    ParamKind        kind;
    Smoothing        smoothing;
    ParamTier        tier;
    ParamGroup       group;
    std::string_view help;
};

// Ranges follow the stage files verbatim. Stage 09's DoD ("range honesty")
// may narrow them once readiness checks show where quality falls apart.
inline constexpr std::array<ParamDesc, kParamCount> kParams = {{
    {ParamId::PitchShift, "pitch_shift", "Pitch shift", "st", -12.0f, 12.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerFrameRamp, ParamTier::Mvp, ParamGroup::Pitch,
     "Offset on the target-domain log-F0 (formants come from the voice, so no chipmunk effect)."},
    {ParamId::Intonation, "intonation", "Intonation", "x", 0.0f, 2.0f, 1.0f,
     ParamKind::Continuous, Smoothing::PerFrameRamp, ParamTier::Mvp, ParamGroup::Pitch,
     "Scales the whitened pitch contour. 0 is a robot monotone, 1 is natural, 2 exaggerates."},
    {ParamId::TimbreBlend, "timbre_blend", "Timbre blend (A to B)", "", 0.0f, 1.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerFrameRamp, ParamTier::Mvp, ParamGroup::Timbre,
     "spk_emb = (1 - a) * e_A + a * e_B. Experimental: midpoints between trained voices may sound less natural."},
    {ParamId::Gain, "gain", "Output gain", "dB", -24.0f, 12.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerSampleOnePole, ParamTier::Mvp, ParamGroup::Output,
     "Applied to output samples with a ~20 ms one-pole ramp."},
    {ParamId::GateThreshold, "gate_threshold", "Gate threshold", "dBFS", -80.0f, -20.0f, -55.0f,
     ParamKind::Continuous, Smoothing::Envelope, ParamTier::Mvp, ParamGroup::Output,
     "Output gate on input energy; also tells the latency governor when output may be dropped."},
    {ParamId::Limiter, "limiter", "Safety limiter", "", 0.0f, 1.0f, 1.0f,
     ParamKind::Toggle, Smoothing::None, ParamTier::Mvp, ParamGroup::Output,
     "Soft clip at -1 dBFS with no lookahead."},
    {ParamId::VibratoRate, "vibrato_rate", "Vibrato rate", "Hz", 0.0f, 8.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerFrameRamp, ParamTier::Stretch, ParamGroup::Pitch,
     "LFO rate on the target log-F0."},
    {ParamId::VibratoDepth, "vibrato_depth", "Vibrato depth", "cents", 0.0f, 100.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerFrameRamp, ParamTier::Stretch, ParamGroup::Pitch,
     "LFO depth on the target log-F0."},
    {ParamId::Whisper, "whisper", "Whisper", "", 0.0f, 1.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerFrameRamp, ParamTier::Stretch, ParamGroup::Timbre,
     "Scales the voicing input towards 0."},
    {ParamId::Brightness, "brightness", "Brightness", "dB", -6.0f, 6.0f, 0.0f,
     ParamKind::Continuous, Smoothing::PerSampleOnePole, ParamTier::Stretch, ParamGroup::Output,
     "First-order tilt EQ on the output."},
    {ParamId::Translate, "translate", "Translate (Mode B)", "", 0.0f, 1.0f, 0.0f,
     ParamKind::Toggle, Smoothing::OutputRamp, ParamTier::Mvp, ParamGroup::Translation,
     "0 = Mode A voice changer, 1 = Mode B simultaneous translation. Switching resets translator state only."},
    {ParamId::WaitK, "wait_k", "Wait-k", "chunks", 1.0f, 10.0f, 3.0f,
     ParamKind::Integer, Smoothing::None, ParamTier::Mvp, ParamGroup::Translation,
     "Source chunks (80 ms each) buffered before translation starts. Takes effect on the next chunk, no state reset."},
    {ParamId::AdaptivePolicy, "adaptive_policy", "Adaptive policy", "", 0.0f, 1.0f, 0.0f,
     ParamKind::Toggle, Smoothing::None, ParamTier::Stretch, ParamGroup::Translation,
     "0 = strict Wait-k, 1 = CTC-adaptive readiness."},
}};

constexpr const ParamDesc& param_desc(ParamId id) { return kParams[param_index(id)]; }

/// Descriptor by preset/CLI name, or nullptr.
constexpr const ParamDesc* find_param(std::string_view name) {
    for (const ParamDesc& d : kParams)
        if (d.name == name) return &d;
    return nullptr;
}

/// Bring `v` into the descriptor's domain: NaN -> default, then clamp, then
/// round for Integer, then snap to {0, 1} for Toggle.
constexpr float clamp_param(const ParamDesc& d, float v) {
    if (v != v) return d.def;  // NaN compares unequal to itself
    if (v < d.min) v = d.min;
    if (v > d.max) v = d.max;
    switch (d.kind) {
        case ParamKind::Continuous: return v;
        case ParamKind::Integer:    return static_cast<float>(static_cast<long>(v + (v >= 0.0f ? 0.5f : -0.5f)));
        case ParamKind::Toggle:     return v >= 0.5f ? 1.0f : 0.0f;
    }
    return v;
}

namespace detail {

constexpr bool is_lower_ident(std::string_view s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return true;
}

constexpr bool table_is_well_formed() {
    for (std::size_t i = 0; i < kParamCount; ++i) {
        const ParamDesc& d = kParams[i];
        if (param_index(d.id) != i) return false;                 // table order == enum order
        if (!is_lower_ident(d.name)) return false;
        if (!(d.min < d.max)) return false;
        if (!(d.min <= d.def && d.def <= d.max)) return false;
        if (d.kind == ParamKind::Toggle && (d.min != 0.0f || d.max != 1.0f)) return false;
        if (clamp_param(d, d.def) != d.def) return false;        // default is representable
        for (std::size_t j = 0; j < i; ++j)
            if (kParams[j].name == d.name) return false;          // unique names
    }
    return true;
}

constexpr std::array<float, kParamCount> default_values() {
    std::array<float, kParamCount> v{};
    for (std::size_t i = 0; i < kParamCount; ++i) v[i] = kParams[i].def;
    return v;
}

} // namespace detail

static_assert(detail::table_is_well_formed(), "kParams: order, names, ranges or defaults are inconsistent");

/// The full control snapshot. Voice and language selections are indices into
/// the loaded bundle's voices.json / langs.json tables (names are resolved on
/// the control thread; the worker only sees indices).
struct VoiceParams {
    std::array<float, kParamCount> values = detail::default_values();
    std::uint16_t voice_a     = 0;  // bundle default voice
    std::uint16_t voice_b     = 0;  // blend endpoint B (= voice_a until chosen)
    std::uint16_t src_lang_id = 0;  // langs.json id (default table: en)
    std::uint16_t tgt_lang_id = 1;  // langs.json id (default table: es)

    constexpr float get(ParamId id) const { return values[param_index(id)]; }
    constexpr void  set(ParamId id, float v) { values[param_index(id)] = clamp_param(param_desc(id), v); }
    constexpr bool  enabled(ParamId id) const { return get(id) >= 0.5f; }

    /// Re-apply every descriptor's domain (after a bulk load).
    constexpr void clamp() {
        for (std::size_t i = 0; i < kParamCount; ++i) values[i] = clamp_param(kParams[i], values[i]);
    }

    friend constexpr bool operator==(const VoiceParams&, const VoiceParams&) = default;
};

static_assert(std::is_trivially_copyable_v<VoiceParams>,
              "VoiceParams travels through a lock-free SPSC queue by value");

} // namespace fluxvoice
