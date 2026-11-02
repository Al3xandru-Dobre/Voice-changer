// VoiceParams / Presets — descriptor table and preset loader tests
// (Milestone 2 / Stage 09, Milestone 3 / Stage 09). ORT-free, hardware-free.
//
// Usage: test_voice_params [presets-dir]
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <string_view>

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
    return test::finish("voice_params");
}
