// VoiceParams / Presets — descriptor table and preset loader tests
// (Milestone 2 / Stage 09, Milestone 3 / Stage 09). ORT-free, hardware-free.
//
// Usage: test_voice_params [presets-dir]
#include "voice/ParamChannel.hpp"
#include "voice/ParamSmoothing.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceOutputChain.hpp"
#include "voice/VoiceParams.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "AllocationCounter.hpp"
#include "FakeModelStep.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace fluxvoice;
namespace fs = std::filesystem;

bool has_warning_containing(const PresetLoadResult& r, std::string_view needle) {
    for (const std::string& w : r.warnings)
        if (w.find(needle) != std::string::npos) return true;
    return false;
}

void test_descriptor_table() {
    std::set<std::string_view> names;
    for (std::size_t i = 0; i < kParamCount; ++i) {
        const ParamDesc& d = kParams[i];
        CHECK(param_index(d.id) == i);
        CHECK(names.insert(d.name).second);
        CHECK(d.min < d.max);
        CHECK(d.min <= d.def && d.def <= d.max);
        if (d.kind == ParamKind::Toggle) {
            CHECK(d.min == 0.0f && d.max == 1.0f);
            CHECK(d.def == 0.0f || d.def == 1.0f);
        }
        if (d.kind == ParamKind::Integer) {
            CHECK(std::floor(d.min) == d.min && std::floor(d.max) == d.max && std::floor(d.def) == d.def);
        }
        CHECK(!d.label.empty() && !d.help.empty());
    }
    CHECK(names.size() == kParamCount);

    // Rows pinned to the stage files.
    CHECK(param_desc(ParamId::PitchShift).min == -12.0f && param_desc(ParamId::PitchShift).max == 12.0f);
    CHECK(param_desc(ParamId::Intonation).def == 1.0f && param_desc(ParamId::Intonation).max == 2.0f);
    CHECK(param_desc(ParamId::Gain).min == -24.0f && param_desc(ParamId::Gain).max == 12.0f);
    CHECK(param_desc(ParamId::GateThreshold).def == -55.0f);
    CHECK(param_desc(ParamId::Limiter).def == 1.0f);
    CHECK(param_desc(ParamId::WaitK).min == 1.0f && param_desc(ParamId::WaitK).max == 10.0f &&
          param_desc(ParamId::WaitK).def == 3.0f && param_desc(ParamId::WaitK).kind == ParamKind::Integer);
    CHECK(param_desc(ParamId::Translate).def == 0.0f && param_desc(ParamId::Translate).kind == ParamKind::Toggle);

    // Lookup by name.
    CHECK(find_param("pitch_shift") == &param_desc(ParamId::PitchShift));
    CHECK(find_param("wait_k") == &param_desc(ParamId::WaitK));
    CHECK(find_param("Pitch_Shift") == nullptr);
    CHECK(find_param("") == nullptr);
    CHECK(find_param("dry_wet") == nullptr);  // deliberately excluded (comb-filter hazard)
}

void test_value_type() {
    VoiceParams p;
    for (std::size_t i = 0; i < kParamCount; ++i) CHECK(p.values[i] == kParams[i].def);
    CHECK(p.voice_a == 0 && p.voice_b == 0 && p.src_lang_id == 0 && p.tgt_lang_id == 1);

    p.set(ParamId::PitchShift, 40.0f);
    CHECK(p.get(ParamId::PitchShift) == 12.0f);
    p.set(ParamId::PitchShift, -40.0f);
    CHECK(p.get(ParamId::PitchShift) == -12.0f);
    p.set(ParamId::WaitK, 4.6f);
    CHECK(p.get(ParamId::WaitK) == 5.0f);  // integer rounding
    p.set(ParamId::WaitK, 0.0f);
    CHECK(p.get(ParamId::WaitK) == 1.0f);
    p.set(ParamId::Translate, 0.7f);
    CHECK(p.get(ParamId::Translate) == 1.0f && p.enabled(ParamId::Translate));
    p.set(ParamId::Translate, 0.2f);
    CHECK(!p.enabled(ParamId::Translate));
    p.set(ParamId::Gain, std::numeric_limits<float>::quiet_NaN());
    CHECK(p.get(ParamId::Gain) == 0.0f);  // NaN -> default

    // Bulk clamp after an unchecked write.
    p.values[param_index(ParamId::Intonation)] = 9.0f;
    p.clamp();
    CHECK(p.get(ParamId::Intonation) == 2.0f);

    VoiceParams q;
    CHECK(q != p);
    q = p;
    CHECK(q == p);
}

void test_preset_round_trip() {
    VoiceParams p;
    p.set(ParamId::PitchShift, -3.5f);
    p.set(ParamId::Intonation, 1.25f);
    p.set(ParamId::WaitK, 5.0f);
    p.set(ParamId::Translate, 1.0f);
    const Preset saved = preset_from_params("Round trip", p, "alice", "bob", "es", "en");
    const std::string text = preset_to_json(saved);

    const PresetLoadResult r = parse_preset(text);
    CHECK(r.ok());
    CHECK(r.warnings.empty());
    if (!r.ok()) return;
    const Preset& back = *r.preset;
    CHECK(back.name == "Round trip");
    CHECK(back.voice_a && *back.voice_a == "alice");
    CHECK(back.voice_b && *back.voice_b == "bob");
    CHECK(back.src_lang && *back.src_lang == "es");
    CHECK(back.tgt_lang && *back.tgt_lang == "en");
    for (std::size_t i = 0; i < kParamCount; ++i) {
        CHECK(back.params[i].has_value());
        if (back.params[i]) CHECK(*back.params[i] == p.values[i]);
    }

    VoiceParams applied;
    apply_preset_params(back, applied);
    CHECK(applied.values == p.values);
}

void test_preset_loader_rules() {
    // Clamping with a warning.
    {
        auto r = parse_preset(R"({"schema":1,"name":"x","params":{"pitch_shift":40}})");
        CHECK(r.ok());
        CHECK(r.preset && r.preset->has(ParamId::PitchShift) &&
              *r.preset->params[param_index(ParamId::PitchShift)] == 12.0f);
        CHECK(has_warning_containing(r, "pitch_shift"));
        CHECK(has_warning_containing(r, "outside"));
    }
    // Unknown keys (top level and params) warn, never reject.
    {
        auto r = parse_preset(R"({"schema":1,"name":"x","dry_wet":0.5,"params":{"formant":1,"gain":2}})");
        CHECK(r.ok());
        CHECK(has_warning_containing(r, "unknown key 'dry_wet'"));
        CHECK(has_warning_containing(r, "unknown parameter 'formant'"));
        CHECK(r.preset && r.preset->has(ParamId::Gain));
    }
    // Wrong types inside params warn and skip.
    {
        auto r = parse_preset(R"({"schema":1,"name":"x","params":{"gain":"loud"}})");
        CHECK(r.ok());
        CHECK(r.preset && !r.preset->has(ParamId::Gain));
        CHECK(has_warning_containing(r, "must be a number"));
    }
    // Booleans are accepted for toggles.
    {
        auto r = parse_preset(R"({"schema":1,"name":"x","params":{"translate":true,"limiter":false}})");
        CHECK(r.ok());
        CHECK(r.preset && *r.preset->params[param_index(ParamId::Translate)] == 1.0f);
        CHECK(r.preset && *r.preset->params[param_index(ParamId::Limiter)] == 0.0f);
    }
    // A boolean for a non-toggle is still read as 0/1 but warns.
    {
        auto r = parse_preset(R"({"schema":1,"name":"x","params":{"gain":true}})");
        CHECK(r.ok());
        CHECK(r.preset && *r.preset->params[param_index(ParamId::Gain)] == 1.0f);
        CHECK(has_warning_containing(r, "not a toggle"));
    }
    // Schema: only the integer 1.
    CHECK(!parse_preset(R"({"schema":2,"name":"x"})").ok());
    CHECK(!parse_preset(R"({"schema":true,"name":"x"})").ok());
    CHECK(!parse_preset(R"({"schema":1.0,"name":"x"})").ok());
    CHECK(!parse_preset(R"({"schema":"1","name":"x"})").ok());
    CHECK(!parse_preset(R"({"name":"x"})").ok());
    // Name required; malformed documents rejected.
    CHECK(!parse_preset(R"({"schema":1})").ok());
    CHECK(!parse_preset(R"({"schema":1,"name":""})").ok());
    CHECK(!parse_preset("[1,2,3]").ok());
    CHECK(!parse_preset("{not json").ok());
    CHECK(!parse_preset("").ok());
    const auto bad = parse_preset(R"({"schema":2,"name":"x"})");
    CHECK(!bad.error.empty());
}

void test_voice_only_preset_keeps_translation_state() {
    // M3 / Stage 09: switching Neutral -> Deep while translating must keep
    // translate, src/tgt and wait_k.
    VoiceParams p;
    p.set(ParamId::Translate, 1.0f);
    p.set(ParamId::WaitK, 6.0f);
    p.src_lang_id = 1;
    p.tgt_lang_id = 0;
    auto deep = parse_preset(R"({"schema":1,"name":"Deep","params":{"pitch_shift":-4,"intonation":0.9}})");
    CHECK(deep.ok());
    if (!deep.ok()) return;
    apply_preset_params(*deep.preset, p);
    CHECK(p.get(ParamId::PitchShift) == -4.0f);
    CHECK(p.get(ParamId::Intonation) == 0.9f);
    CHECK(p.enabled(ParamId::Translate));
    CHECK(p.get(ParamId::WaitK) == 6.0f);
    CHECK(p.src_lang_id == 1 && p.tgt_lang_id == 0);
    CHECK(!deep.preset->voice_a && !deep.preset->src_lang && !deep.preset->tgt_lang);
}

void test_builtin_presets(const fs::path& dir) {
    const auto files = list_preset_files(dir);
    CHECK(files.size() == 5);
    std::set<std::string> names;
    for (const fs::path& f : files) {
        const PresetLoadResult r = load_preset_file(f);
        CHECK(r.ok());
        if (!r.ok()) {
            std::printf("  %s: %s\n", f.string().c_str(), r.error.c_str());
            continue;
        }
        CHECK(r.warnings.empty());
        for (const std::string& w : r.warnings) std::printf("  %s: %s\n", f.string().c_str(), w.c_str());
        CHECK(r.preset->path == f);
        names.insert(r.preset->name);
        // Built-ins are voice-only: they never pin a voice or a language.
        CHECK(!r.preset->voice_a && !r.preset->voice_b && !r.preset->src_lang && !r.preset->tgt_lang);
        CHECK(!r.preset->has(ParamId::Translate));
    }
    for (const char* n : {"Neutral", "Deep", "Bright", "Robot", "Whisper"}) CHECK(names.count(n) == 1);

    // Spot checks pinned to the stage file.
    auto deep = load_preset_file(dir / "deep.json");
    CHECK(deep.ok() && deep.preset->has(ParamId::PitchShift) &&
          *deep.preset->params[param_index(ParamId::PitchShift)] == -4.0f);
    auto robot = load_preset_file(dir / "robot.json");
    CHECK(robot.ok() && robot.preset->has(ParamId::Intonation) &&
          *robot.preset->params[param_index(ParamId::Intonation)] == 0.0f);
    auto neutral = load_preset_file(dir / "neutral.json");
    if (neutral.ok()) {
        VoiceParams p;
        p.set(ParamId::PitchShift, 7.0f);
        apply_preset_params(*neutral.preset, p);
        CHECK(p.get(ParamId::PitchShift) == 0.0f);
    } else {
        CHECK(false);
    }

    CHECK(list_preset_files(dir / "does-not-exist").empty());
}

void test_save_and_reload(const fs::path& scratch) {
    VoiceParams p;
    p.set(ParamId::Gain, -6.0f);
    const Preset saved = preset_from_params("Saved", p, "identity");
    std::string err;
    const fs::path file = scratch / "fluxvoice_test_presets" / "saved.json";
    CHECK(save_preset_file(file, saved, err));
    CHECK(err.empty());
    const auto back = load_preset_file(file);
    CHECK(back.ok());
    CHECK(back.ok() && back.preset->voice_a && *back.preset->voice_a == "identity");
    CHECK(back.ok() && *back.preset->params[param_index(ParamId::Gain)] == -6.0f);
    std::error_code ec;
    fs::remove_all(file.parent_path(), ec);

    // A directory in place of the file reports an error instead of throwing.
    std::string err2;
    CHECK(!save_preset_file(scratch, saved, err2));
    CHECK(!err2.empty());
}

void test_descriptor_json() {
    const nlohmann::json doc = nlohmann::json::parse(descriptor_table_json(), nullptr, false);
    CHECK(!doc.is_discarded());
    if (doc.is_discarded()) return;
    CHECK(doc.at("schema") == 1);
    CHECK(doc.at("params").size() == kParamCount);
    for (std::size_t i = 0; i < kParamCount; ++i) {
        const auto& e = doc.at("params").at(i);
        CHECK(e.at("name") == std::string(kParams[i].name));
        CHECK(e.at("min").get<float>() == kParams[i].min);
        CHECK(e.at("max").get<float>() == kParams[i].max);
        CHECK(e.at("default").get<float>() == kParams[i].def);
        CHECK(e.contains("kind") && e.contains("tier") && e.contains("group") && e.contains("unit"));
    }
}

// ---------------------------------------------------------------------------
// Stage 09 engine tests: ramps, param channel, output chain, pipeline controls
// ---------------------------------------------------------------------------

constexpr double kLn2 = 0.69314718055994530942;
constexpr double kTwoPi = 6.28318530717958647692;

std::vector<float> make_sine(double freq, double amp, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<float>(amp * std::sin(kTwoPi * freq * static_cast<double>(i) / 16000.0));
    return v;
}

double rms_of(std::span<const float> v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s / static_cast<double>(v.size()));
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1,
                      static_cast<std::size_t>(p / 100.0 * static_cast<double>(v.size() - 1)))];
}

void test_frame_ramp() {
    FrameRamp r(kParamRampFrames);
    r.snap(0.0f);
    CHECK(r.value() == 0.0f && r.target() == 0.0f && !r.ramping());
    r.set_target(10.0f);
    float prev = r.value();
    for (std::uint32_t i = 0; i < kParamRampFrames; ++i) {
        const float v = r.advance();
        CHECK(v > prev);
        prev = v;
    }
    CHECK(r.value() == 10.0f);       // exact landing, no float residue
    CHECK(r.advance() == 10.0f);     // afterwards a no-op, never overshoots
    CHECK(!r.ramping());

    // Retarget mid-ramp: step recomputed from the current value.
    FrameRamp m(kParamRampFrames);
    m.snap(0.0f);
    m.set_target(10.0f);
    CHECK(m.advance() == 2.0f);      // 10/5
    m.set_target(20.0f);             // step = (20-2)/5 = 3.6
    for (std::uint32_t i = 0; i < kParamRampFrames; ++i) m.advance();
    CHECK(m.value() == 20.0f);       // still lands exactly
    CHECK(!m.ramping());
}

void test_resolve_preset(const fs::path& presets_dir) {
    const auto dirs = preset_search_dirs({}, {});
    // The compile-definition dir should list the shipped presets/.
    CHECK(std::find(dirs.begin(), dirs.end(),
                    fs::weakly_canonical(presets_dir)) != dirs.end());

    // By file stem, and case-insensitive by preset name.
    auto by_stem = resolve_preset("deep", preset_search_dirs({}, {}));
    CHECK(by_stem.ok() && by_stem.preset->name == "Deep");
    auto by_name = resolve_preset("ROBOT", preset_search_dirs({}, {}));
    CHECK(by_name.ok() && by_name.preset->name == "Robot");

    // Explicit path resolution.
    auto by_path = resolve_preset((presets_dir / "neutral.json").string(),
                                  preset_search_dirs({}, {}));
    CHECK(by_path.ok() && by_path.preset->name == "Neutral");

    // A user dir shadows the built-ins (same stem, different contents).
    const fs::path user_dir = fs::temp_directory_path() / "fv_preset_shadow";
    std::error_code ec;
    fs::create_directories(user_dir, ec);
    {
        std::ofstream f(user_dir / "deep.json");
        f << R"({"schema":1,"name":"Custom Deep","params":{"pitch_shift":1}})" << '\n';
        std::ofstream f2(user_dir / "oddname.json");
        f2 << R"({"schema":1,"name":"Hidden Gem","params":{"gain":-3}})" << '\n';
        std::ofstream f3(user_dir / "CUSTOM.JSON");
        f3 << R"({"schema":1,"name":"Uppercase Json","params":{"gain":-1}})" << '\n';
    }
    auto dirs2 = preset_search_dirs(user_dir, {});
    auto shadow = resolve_preset("deep", dirs2);
    CHECK(shadow.ok() && shadow.preset->name == "Custom Deep");
    CHECK(shadow.ok() && *shadow.preset->params[param_index(ParamId::PitchShift)] == 1.0f);
    // Name match even when the file stem differs.
    auto hidden = resolve_preset("hidden gem", dirs2);
    CHECK(hidden.ok() && hidden.preset->name == "Hidden Gem");
    // A path ending in .JSON (any case) resolves as a file, not a name.
    auto upper = resolve_preset((user_dir / "CUSTOM.JSON").string(), dirs2);
    CHECK(upper.ok() && upper.preset->name == "Uppercase Json");
    fs::remove_all(user_dir, ec);

    // Unknown -> error naming it plus the available list.
    auto unknown = resolve_preset("definitely-not-a-preset", preset_search_dirs({}, {}));
    CHECK(!unknown.ok());
    CHECK(unknown.error.find("unknown preset 'definitely-not-a-preset'") != std::string::npos);
    CHECK(unknown.error.find("Neutral") != std::string::npos);
}

void test_parse_param_assignment() {
    const ParamDesc* d = nullptr;
    float v = 0.0f;
    std::string warning, error;

    CHECK(parse_param_assignment("pitch_shift=3.5", d, v, warning, error));
    CHECK(d->id == ParamId::PitchShift && v == 3.5f && warning.empty());
    CHECK(parse_param_assignment("intonation=0", d, v, warning, error) && v == 0.0f);
    // Clamped with a warning.
    CHECK(parse_param_assignment("gain=40", d, v, warning, error));
    CHECK(v == 12.0f);
    CHECK(warning.find("gain") != std::string::npos && warning.find("outside") != std::string::npos);
    // Toggle words, and a toggle given as 0/1 is exact: no warning.
    CHECK(parse_param_assignment("limiter=off", d, v, warning, error) && v == 0.0f);
    CHECK(parse_param_assignment("limiter=on", d, v, warning, error) && v == 1.0f);
    CHECK(parse_param_assignment("limiter=1", d, v, warning, error));
    CHECK(v == 1.0f && warning.empty());
    // One leading '+' is accepted (std::from_chars rejects it).
    CHECK(parse_param_assignment("pitch_shift=+3", d, v, warning, error));
    CHECK(v == 3.0f && warning.empty());
    CHECK(parse_param_assignment("pitch_shift=+7.5", d, v, warning, error));
    CHECK(v == 7.5f && warning.empty());
    // In-range rounding of an integer kind says "rounded to", not "outside".
    CHECK(parse_param_assignment("wait_k=4.6", d, v, warning, error));
    CHECK(v == 5.0f);
    CHECK(warning.find("rounded to 5") != std::string::npos);
    CHECK(warning.find("outside") == std::string::npos);
    // Unknown name.
    CHECK(!parse_param_assignment("bogus=1", d, v, warning, error));
    CHECK(error.find("unknown parameter 'bogus'") != std::string::npos);
    // voice_a gets a hint toward --voice.
    CHECK(!parse_param_assignment("voice_a=2", d, v, warning, error));
    CHECK(error.find("--voice") != std::string::npos);
    // Missing '=' or an empty side.
    CHECK(!parse_param_assignment("pitch_shift", d, v, warning, error));
    CHECK(error.find("expected name=value") != std::string::npos);
    CHECK(!parse_param_assignment("pitch_shift=", d, v, warning, error));
    CHECK(error.find("expected name=value") != std::string::npos);
    // Trailing junk and non-finite values rejected.
    CHECK(!parse_param_assignment("gain=1x", d, v, warning, error));
    CHECK(!parse_param_assignment("gain=nan", d, v, warning, error));
}

void test_parse_finite_float() {
    double v = 0.0;
    CHECK(!parse_finite_float("", v));
    CHECK(!parse_finite_float("1e400", v));   // ERANGE
    CHECK(!parse_finite_float("nan", v));
    CHECK(!parse_finite_float("inf", v));
    CHECK(!parse_finite_float(" 3", v));      // leading whitespace
    CHECK(!parse_finite_float("3x", v));      // trailing junk
    CHECK(!parse_finite_float("x3", v));
    CHECK(!parse_finite_float("++3", v));
    CHECK(parse_finite_float("+3", v) && v == 3.0);
    CHECK(parse_finite_float("-2.5", v) && v == -2.5);
    CHECK(parse_finite_float("0.125", v) && v == 0.125);
    CHECK(parse_finite_float("1e-3", v) && v == 1e-3);
}

// -- through the fake pipeline ----------------------------------------------

PipelineOptions chain_off() {
    PipelineOptions o;
    o.output_chain = false;
    return o;
}

void test_f0_mapping_closed_form() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    constexpr int kHops = 12;
    const auto in = make_sine(220.0, 0.4, 320 * kHops);
    std::vector<float> base(kHops), mod(kHops);
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, stats, emb, chain_off());
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            base[k] = voc.input_f32("logf0")[0];
        }
    }
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, stats, emb, chain_off());
        VoiceParams np;
        np.set(ParamId::PitchShift, 4.0f);
        np.set(ParamId::Intonation, 0.5f);
        CHECK(p.set_params(np));
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            mod[k] = voc.input_f32("logf0")[0];
        }
    }
    // Same input -> the tracker reproduces the same whitened z per frame.
    for (int k = kParamRampFrames; k < kHops; ++k) {
        const double z = (base[k] - 4.8) / 0.25;
        const double expected = 4.8 + 0.5 * 0.25 * z + 4.0 * kLn2 / 12.0;
        CHECK(std::fabs(mod[k] - expected) < 2e-4);
    }
}

void test_pitch_ramp_timing() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    constexpr int kHops = 8;
    const auto in = make_sine(220.0, 0.4, 320 * kHops);
    std::vector<float> base(kHops), mod(kHops);
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, stats, emb, chain_off());
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            base[k] = voc.input_f32("logf0")[0];
        }
    }
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, stats, emb, chain_off());
        VoiceParams np;
        np.set(ParamId::PitchShift, 5.0f);
        CHECK(p.set_params(np));
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            mod[k] = voc.input_f32("logf0")[0];
        }
    }
    // Frame k contributes pitch (k+1)/5 * 5 st, landing on frame index 4.
    double prev = -1.0;
    for (int k = 0; k < kHops; ++k) {
        const double steps = std::min(k + 1, static_cast<int>(kParamRampFrames));
        const double expected = steps * (5.0 / kParamRampFrames) * kLn2 / 12.0;
        const double diff = mod[k] - base[k];
        CHECK(std::fabs(diff - expected) < 2e-4);
        CHECK(diff >= prev - 1e-6);  // monotonic during the ramp
        prev = diff;
    }
    CHECK(std::fabs((mod[4] - base[4]) - 5.0 * kLn2 / 12.0) < 2e-4);  // landed exactly
}

void test_timbre_blend_arithmetic() {
    const TargetVoiceStats stats_a{4.8, 0.25, -4.0, 1.5};
    const TargetVoiceStats stats_b{5.0, 0.50, -3.0, 1.0};
    const std::array<float, 4> emb_a{0.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, 4> emb_b{1.0f, 1.0f, 1.0f, 1.0f};
    const PipelineVoice voices[2] = {{stats_a, emb_a}, {stats_b, emb_b}};
    constexpr int kHops = 8;
    const auto in = make_sine(220.0, 0.4, 320 * kHops);
    std::vector<float> base(kHops);
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, voices, chain_off());
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            base[k] = voc.input_f32("logf0")[0];
        }
    }
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    PipelineOptions o = chain_off();
    o.params.set(ParamId::TimbreBlend, 0.25f);
    o.params.voice_b = 1;
    InferencePipeline p(enc, voc, voices, o);
    for (int k = 0; k < kHops; ++k) {
        std::vector<float> out(320);
        CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
        if (k >= 4) {
            const std::span<float> spk = voc.input_f32("spk_emb");
            // spk_emb = (1-a)*e_A + a*e_B = 0.25 on every element.
            for (float x : spk) CHECK(std::fabs(x - 0.25f) < 1e-6f);
            // And the stats blend feeds the logf0 mapping.
            const double z = (base[k] - 4.8) / 0.25;
            const double mean = 4.8 + 0.25 * (5.0 - 4.8);   // 4.85
            const double std = 0.25 + 0.25 * (0.5 - 0.25);  // 0.3125
            CHECK(std::fabs(voc.input_f32("logf0")[0] - (mean + std * z)) < 2e-4);
        }
    }
}

void test_voice_switch_crossfade() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb_a{0.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, 4> emb_b{1.0f, 1.0f, 1.0f, 1.0f};
    const PipelineVoice voices[2] = {{stats, emb_a}, {stats, emb_b}};
    constexpr int kHops = 7;
    const auto in = make_sine(220.0, 0.4, 320 * kHops);
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, voices, chain_off());
    std::vector<float> out(320);
    CHECK(p.process(std::span(in).first(320), out).ok);
    CHECK(std::fabs(voc.input_f32("spk_emb")[0] - 0.0f) < 1e-6f);

    // Switching voice_a at timbre_blend = 0 must still fade, not jump.
    VoiceParams np;
    np.voice_a = 1;
    CHECK(p.set_params(np));
    float prev = 0.0f;
    for (int k = 1; k < kHops; ++k) {
        CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
        const float e = voc.input_f32("spk_emb")[0];
        CHECK(e >= prev - 1e-6f);                                   // monotonic
        CHECK(e - prev <= 1.0f / kVoiceCrossfadeFrames + 1e-6f);    // bounded step
        prev = e;
    }
    CHECK(std::fabs(prev - 1.0f) < 1e-6f);                          // ends at e_B
}

void test_invalid_voice_index_rejected() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, stats, emb, chain_off());
    CHECK(p.voice_count() == 1);
    VoiceParams np;
    np.voice_a = 3;
    CHECK(!p.set_params(np));
    CHECK(p.params().voice_a == 0);
    np.voice_a = 0;
    np.voice_b = 7;
    CHECK(!p.set_params(np));
    CHECK(p.params().voice_b == 0);
}

void test_vibrato_and_whisper() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    constexpr int kHops = 10;
    const auto in = make_sine(220.0, 0.4, 320 * kHops);
    std::vector<float> base(kHops), voicing(kHops);
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, stats, emb, chain_off());
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            base[k] = voc.input_f32("logf0")[0];
            voicing[k] = voc.input_f32("voicing")[0];
        }
    }
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    PipelineOptions o = chain_off();
    o.params.set(ParamId::VibratoRate, 4.0f);
    o.params.set(ParamId::VibratoDepth, 50.0f);  // cents
    o.params.set(ParamId::Whisper, 0.5f);
    InferencePipeline p(enc, voc, stats, emb, o);
    for (int k = 0; k < kHops; ++k) {
        std::vector<float> out(320);
        CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
        // vibrato = depth_cents/1200 * ln2 * sin(phase); phase advances after use.
        const double vib = 50.0 / 1200.0 * kLn2 * std::sin(kTwoPi * 4.0 * 0.020 * k);
        CHECK(std::fabs((voc.input_f32("logf0")[0] - base[k]) - vib) < 2e-4);
        CHECK(std::fabs(voc.input_f32("voicing")[0] - voicing[k] * 0.5f) < 1e-5f);
    }
}

void test_vibrato_rate_latch() {
    // vibrato_rate -> 0 while depth != 0 must leave NO residual pitch offset:
    // the LFO contribution is scaled by min(1, rate / 0.5 Hz), so rate 0 gives
    // exactly zero rather than a frozen sin(phase).
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    constexpr int kHops = 18;
    const auto in = make_sine(220.0, 0.4, 320 * kHops);
    std::vector<float> base(kHops);
    {
        test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
        test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
        InferencePipeline p(enc, voc, stats, emb, chain_off());
        for (int k = 0; k < kHops; ++k) {
            std::vector<float> out(320);
            CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
            base[k] = voc.input_f32("logf0")[0];
        }
    }
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    PipelineOptions o = chain_off();
    o.params.set(ParamId::VibratoRate, 5.0f);
    o.params.set(ParamId::VibratoDepth, 50.0f);
    InferencePipeline p(enc, voc, stats, emb, o);
    for (int k = 0; k < kHops; ++k) {
        if (k == 4) {
            VoiceParams np = p.params();
            np.set(ParamId::VibratoRate, 0.0f);
            CHECK(p.set_params(np));
        }
        std::vector<float> out(320);
        CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
        if (k >= 10) {   // ramp lands exactly on 0 at the 5th frame after k=4
            CHECK(voc.input_f32("logf0")[0] == base[k]);
        }
    }
}

// -- output chain unit tests --------------------------------------------------

void test_chain_gate_envelope() {
    VoiceOutputChain c;
    c.configure(16000);
    c.set_targets(0.0f, -55.0f, false, 0.0f);
    c.reset();
    CHECK(!c.gate_open());
    const std::vector<float> loud{0.0f};   // well above threshold
    const std::vector<float> low{-62.0f};  // below threshold - 6 dB hysteresis
    std::vector<float> buf(320, 1.0f);
    c.process(buf, loud, 320);
    CHECK(c.gate_open());
    CHECK(buf[0] < 0.05f);              // envelope just started (2 ms = 32 samples)
    CHECK(buf[31] > 0.95f);
    CHECK(buf[319] == 1.0f);            // saturated: identity passthrough
    // Hysteresis: -58 > -61 keeps it open, hold refreshed.
    const std::vector<float> warm{-58.0f};
    std::fill(buf.begin(), buf.end(), 1.0f);
    c.process(buf, warm, 320);
    CHECK(c.gate_open());
    // 5-frame (100 ms) hold below the hysteresis band, closes on the 6th.
    for (int k = 0; k < 5; ++k) {
        c.process(buf, low, 320);
        CHECK(c.gate_open());
    }
    c.process(buf, low, 320);
    CHECK(!c.gate_open());
    // Release: 50 ms = 800 samples. The closing frame already fell to ~0.6;
    // this frame continues 0.6 -> 0.2 and a third reaches exact zeros.
    std::fill(buf.begin(), buf.end(), 1.0f);
    c.process(buf, low, 320);
    CHECK(buf[0] > 0.5f && buf[0] < 0.7f && buf[319] < 0.3f);
    std::vector<float> buf2(960, 1.0f);
    c.process(buf2, low, 320);
    CHECK(buf2.back() == 0.0f);
    CHECK(std::all_of(buf2.begin() + 800, buf2.end(), [](float x) { return x == 0.0f; }));
}

void test_chain_gain_and_limiter() {
    VoiceOutputChain c;
    c.configure(16000);
    const std::vector<float> loud{0.0f};
    // Gain settles: +6.0206 dB ≈ x2 amplitude.
    c.set_targets(6.0206f, -55.0f, false, 0.0f);
    c.reset();
    std::vector<float> buf(320, 0.25f);
    c.process(buf, loud, 320);
    CHECK(buf[319] > 0.49f && buf[319] <= 0.5001f);
    // Limiter: ceiling 0.891251, unity below the knee.
    c.set_targets(0.0f, -55.0f, true, 0.0f);
    c.reset();
    std::vector<float> big(320, 2.0f);
    c.process(big, loud, 320);
    for (float x : big) CHECK(x <= 0.891251f + 1e-4f);          // never above the ceiling
    for (std::size_t i = 40; i < big.size(); ++i) CHECK(big[i] > 0.8f);  // settled: ~0.8912
    std::vector<float> small(320, 0.5f);
    c.process(small, loud, 320);
    for (std::size_t i = 40; i < small.size(); ++i) CHECK(small[i] == 0.5f);  // below knee: exact
    // Toggle crossfade: turning the limiter on under a loud signal ramps in.
    c.set_targets(0.0f, -55.0f, false, 0.0f);
    c.reset();
    std::vector<float> hot(320, 1.5f);
    c.process(hot, loud, 320);
    CHECK(hot[319] > 1.4f);   // limiter off: passes
    c.set_targets(0.0f, -55.0f, true, 0.0f);
    float prev = hot[319];
    for (int f = 0; f < 10; ++f) {
        std::vector<float> step(320, 1.5f);
        c.process(step, loud, 320);
        for (std::size_t i = 1; i < step.size(); ++i)
            CHECK(std::fabs(step[i] - step[i - 1]) < 0.01f);   // crossfade, no click
        CHECK(step[319] <= prev + 1e-6f);                      // approaching the ceiling
        prev = step[319];
    }
    CHECK(prev < 0.95f);
}

// Closed form of the specified tilt: with lp = H(f)·x (one-pole at the 1 kHz
// pivot, a = 1-exp(-2pi*1000/sr)) and hp = x - lp, the output is
// y = hi·x + (1/hi - hi)·H(f)·x, hi = 10^(t/40). A one-pole LP still passes
// ~21 % at 6 kHz, so the real high-band gain is ~+2.1 dB, not a full +3 dB
// shelf — the assert is against this formula, not the approximation.
double expected_tilt_ratio(double freq, double tilt_db) {
    const double hi = std::pow(10.0, tilt_db / 40.0);
    const double a = 1.0 - std::exp(-kTwoPi * 1000.0 / 16000.0);
    const std::complex<double> z1 =
        std::exp(std::complex<double>(0.0, -kTwoPi * freq / 16000.0));
    const std::complex<double> h = a / (1.0 - (1.0 - a) * z1);
    return std::abs(hi + h * (1.0 / hi - hi));
}

void test_chain_brightness() {
    VoiceOutputChain c;
    c.configure(16000);
    const std::vector<float> loud{0.0f};
    c.set_targets(0.0f, -55.0f, false, 0.0f);
    c.reset();
    const auto s6k = make_sine(6000.0, 0.2, 320 * 20);
    const auto s100 = make_sine(100.0, 0.2, 320 * 20);
    // t = 0 reproduces the input within 1e-6 (actually bit-exact here).
    {
        std::vector<float> buf(s6k.begin(), s6k.end());
        c.process(buf, loud, 320);
        for (std::size_t i = 640; i < buf.size(); ++i)
            CHECK(std::fabs(buf[i] - s6k[i]) <= 1e-6f);
    }
    // +6 dB tilt: 6 kHz above the pivot gains, 100 Hz below it loses; both
    // within +-0.7 dB of the formula's closed-form ratio.
    const double lo = std::pow(10.0, -0.7 / 20.0), hi = std::pow(10.0, 0.7 / 20.0);
    c.set_targets(0.0f, -55.0f, false, 6.0f);
    {
        std::vector<float> buf(s6k.begin(), s6k.end());
        c.process(buf, loud, 320);
        const double ratio = rms_of(std::span(buf).subspan(3200)) / rms_of(std::span(s6k).subspan(3200));
        const double want = expected_tilt_ratio(6000.0, 6.0);
        CHECK(ratio > want * lo && ratio < want * hi);
    }
    c.reset();
    c.set_targets(0.0f, -55.0f, false, 6.0f);
    {
        std::vector<float> buf(s100.begin(), s100.end());
        c.process(buf, loud, 320);
        const double ratio = rms_of(std::span(buf).subspan(3200)) / rms_of(std::span(s100).subspan(3200));
        const double want = expected_tilt_ratio(100.0, 6.0);
        CHECK(ratio > want * lo && ratio < want * hi);
    }
}

void test_chain_brightness_snap_returns_bit_exact() {
    VoiceOutputChain c;
    c.configure(16000);
    const std::vector<float> loud{0.0f};   // 0 dBFS >> -55: the gate stays open
    c.set_targets(0.0f, -55.0f, true, 6.0f);
    c.reset();
    const auto in = make_sine(997.0, 0.3, 320 * 30);
    std::vector<float> buf(in.begin(), in.end());
    // 60 ms of +6 dB tilt (the gate's 2 ms attack lands env at exactly 1.0),
    // then back to 0 dB for the rest (~200 ms tail).
    c.process(std::span(buf).subspan(0, 320 * 3), std::span(loud), 320);
    c.set_targets(0.0f, -55.0f, true, 0.0f);
    c.process(std::span(buf).subspan(320 * 3), std::span(loud), 320);
    // The tilt one-pole (tau 20 ms) snaps to 0 once |target - tilt| < 1e-6,
    // ~16 tau ≈ 310 ms after +6 dB; a 0.3-amplitude signal below the limiter
    // knee then passes bit-exact.
    for (std::size_t i = 320 * 22; i < buf.size(); ++i)
        CHECK(buf[i] == in[i]);
}

// -- output chain through the pipeline (click safety) -------------------------

void test_pipeline_gain_step_and_rms() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    const auto in = make_sine(440.0, 0.3, 320 * 24);
    PipelineOptions o;  // default: full chain; 0.3 -> 0.6 stays below the knee
    o.params.set(ParamId::GateThreshold, -80.0f);
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, stats, emb, o);
    std::vector<std::vector<float>> outs(24, std::vector<float>(320));
    for (int k = 0; k < 24; ++k) {
        if (k == 10) {
            VoiceParams np;
            np.set(ParamId::GateThreshold, -80.0f);
            np.set(ParamId::Gain, 6.0206f);
            CHECK(p.set_params(np));
        }
        CHECK(p.process(std::span(in).subspan(k * 320, 320), outs[k]).ok);
    }
    const auto diffs = [](const std::vector<std::vector<float>>& hops, int a, int b) {
        std::vector<double> d;
        for (int k = a; k < b; ++k)
            for (std::size_t i = 1; i < 320; ++i)
                d.push_back(std::fabs(hops[k][i] - hops[k][i - 1]));
        return d;
    };
    // Click bound: the 20 ms gain ramp must not create a bigger adjacent-sample
    // step than the steady-state signal itself.
    const double before = percentile(diffs(outs, 5, 10), 99.9);
    const double after = percentile(diffs(outs, 18, 24), 99.9);
    const double during = percentile(diffs(outs, 10, 18), 100.0);
    CHECK(during <= std::max(before, after) * 1.05);
    // +6.0206 dB ≈ x2 RMS, settled well before 200 ms.
    const double ratio = rms_of(outs[23]) / rms_of(outs[5]);
    CHECK(ratio > 1.98 && ratio < 2.02);
}

void test_pipeline_gate_and_limiter() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    // Burst: 6 hops of 0.5 sine then silence; gate at -30 dBFS.
    auto in = make_sine(440.0, 0.5, 320 * 24);
    std::fill(in.begin() + 320 * 6, in.end(), 0.0f);
    PipelineOptions o;
    o.params.set(ParamId::GateThreshold, -30.0f);
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, stats, emb, o);
    bool quiet_seen = false;
    for (int k = 0; k < 24; ++k) {
        std::vector<float> out(320);
        const auto res = p.process(std::span(in).subspan(k * 320, 320), out);
        CHECK(res.ok);
        if (k < 4) {
            // Gate opens on the burst: signal comes through.
            CHECK(rms_of(out) > 0.05);
        }
        if (k >= 13) {
            // Hold (5 frames) + release (50 ms) -> exact zeros, reported quiet.
            CHECK(std::all_of(out.begin(), out.end(), [](float x) { return x == 0.0f; }));
        }
        quiet_seen = quiet_seen || res.quiet;
    }
    CHECK(quiet_seen);

    // Limiter through the pipeline: a 2.0 sine never exceeds the ceiling, and
    // a 0.5 sine below the knee passes bit-exact.
    const auto hot = make_sine(440.0, 2.0, 320 * 6);
    test::FakeModelStep enc2(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc2(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p2(enc2, voc2, stats, emb, PipelineOptions{});
    for (int k = 0; k < 6; ++k) {
        std::vector<float> out(320);
        CHECK(p2.process(std::span(hot).subspan(k * 320, 320), out).ok);
        if (k >= 2)
            for (float x : out) CHECK(std::fabs(x) <= 0.891251f + 1e-4f);
    }
    const auto mild = make_sine(440.0, 0.5, 320 * 4);
    test::FakeModelStep enc3(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc3(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p3(enc3, voc3, stats, emb, PipelineOptions{});
    for (int k = 0; k < 4; ++k) {
        std::vector<float> out(320);
        CHECK(p3.process(std::span(mild).subspan(k * 320, 320), out).ok);
        if (k >= 1)
            for (std::size_t i = 0; i < out.size(); ++i)
                CHECK(out[i] == mild[k * 320 + i]);  // below knee: bit-exact
    }
}

void test_pipeline_brightness_identity() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    PipelineOptions o;
    o.params.set(ParamId::GateThreshold, -80.0f);
    o.params.set(ParamId::Limiter, 0.0f);
    o.params.set(ParamId::Brightness, 0.0f);
    const auto in = make_sine(997.0, 0.4, 320 * 4);
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, stats, emb, o);
    for (int k = 0; k < 4; ++k) {
        std::vector<float> out(320);
        CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
        if (k >= 1)
            for (std::size_t i = 0; i < out.size(); ++i)
                CHECK(std::fabs(out[i] - in[k * 320 + i]) <= 1e-6f);
    }
}

void test_reset_equals_fresh() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    PipelineOptions o;
    o.params.set(ParamId::Gain, 3.0f);
    o.params.set(ParamId::PitchShift, 2.0f);
    const auto in = make_sine(330.0, 0.3, 320 * 8);
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, stats, emb, o);
    std::vector<float> out(320);
    for (int k = 0; k < 4; ++k) CHECK(p.process(std::span(in).subspan(k * 320, 320), out).ok);
    p.reset();
    test::FakeModelStep enc2(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc2(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline fresh(enc2, voc2, stats, emb, o);
    for (int k = 0; k < 8; ++k) {
        std::vector<float> a(320), b(320);
        CHECK(p.process(std::span(in).subspan(k * 320, 320), a).ok);
        CHECK(fresh.process(std::span(in).subspan(k * 320, 320), b).ok);
        CHECK(a == b);
    }
}

// -- ParamChannel -------------------------------------------------------------

void test_param_channel() {
    ParamChannel ch;
    VoiceParams out;
    CHECK(!ch.drain(out));
    VoiceParams a, b, c;
    a.set(ParamId::PitchShift, 1.0f);
    b.set(ParamId::PitchShift, 2.0f);
    c.set(ParamId::PitchShift, 3.0f);
    const auto g1 = ch.push(a);
    const auto g2 = ch.push(b);
    const auto g3 = ch.push(c);
    CHECK(g3 > g2 && g2 > g1);
    CHECK(ch.pushed_generation() == g3);
    // Newest snapshot wins; older ones are still enqueued but the drain keeps
    // the last dequeued value.
    CHECK(ch.drain(out));
    CHECK(out.get(ParamId::PitchShift) == 3.0f);
    CHECK(ch.applied_generation() == g3);
    CHECK(!ch.drain(out));

    // Fill the queue (capacity 64), then confirm pending + flush delivery.
    ParamChannel full;
    for (int i = 0; i < 64; ++i) full.push(a);
    const auto gp = full.push(c);
    // Push may have left the newest pending if the queue filled exactly.
    VoiceParams got;
    std::size_t drained = 0;
    while (full.drain(got)) ++drained;
    full.flush();
    while (full.drain(got)) ++drained;
    CHECK(drained > 0);
    CHECK(got.get(ParamId::PitchShift) == 3.0f);
    CHECK(full.applied_generation() == gp || full.pushed_generation() == gp);
}

// -- allocation-free hop path -------------------------------------------------

void test_zero_allocations() {
    const TargetVoiceStats stats{4.8, 0.25, -4.0, 1.5};
    const std::array<float, 4> emb{};
    test::FakeModelStep enc(test::fake_contract(ModelRole::kEncoder));
    test::FakeModelStep voc(test::fake_contract(ModelRole::kVocoder));
    InferencePipeline p(enc, voc, stats, emb, PipelineOptions{});
    CHECK(p.warm_up(5));
    ParamChannel ch;
    const auto in = make_sine(220.0, 0.3, 320);
    std::vector<float> out(320);
    VoiceParams np;
    const long before = test::allocation_count();
    for (int i = 0; i < 100; ++i) {
        np.set(ParamId::PitchShift, static_cast<float>(i % 7) - 3.0f);
        np.set(ParamId::Gain, static_cast<float>(i % 9) - 4.0f);
        ch.push(np);
        ch.flush();
        VoiceParams got;
        if (ch.drain(got)) CHECK(p.set_params(got));
        CHECK(p.process(in, out).ok);
    }
    if constexpr (test::kAllocationCounting) CHECK(test::allocation_count() == before);
}

} // namespace

int main(int argc, char** argv) {
    const fs::path presets_dir = argc > 1 ? fs::path(argv[1]) : fs::path("presets");
    test_descriptor_table();
    test_value_type();
    test_preset_round_trip();
    test_preset_loader_rules();
    test_voice_only_preset_keeps_translation_state();
    test_builtin_presets(presets_dir);
    test_save_and_reload(fs::temp_directory_path());
    test_descriptor_json();
    // Stage 09 engine
    test_frame_ramp();
    test_resolve_preset(presets_dir);
    test_parse_param_assignment();
    test_f0_mapping_closed_form();
    test_pitch_ramp_timing();
    test_timbre_blend_arithmetic();
    test_voice_switch_crossfade();
    test_invalid_voice_index_rejected();
    test_vibrato_and_whisper();
    test_vibrato_rate_latch();
    test_parse_finite_float();
    test_chain_gate_envelope();
    test_chain_gain_and_limiter();
    test_chain_brightness();
    test_chain_brightness_snap_returns_bit_exact();
    test_pipeline_gain_step_and_rms();
    test_pipeline_gate_and_limiter();
    test_pipeline_brightness_identity();
    test_reset_equals_fresh();
    test_param_channel();
    test_zero_allocations();
    return test::finish("voice_params");
}
