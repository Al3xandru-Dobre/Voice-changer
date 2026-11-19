// Frontend model layer — headless tests (Milestone 4 frontend, no display, no
// audio hardware). Covers frontend/README.md §10:
//   device rows and selection policy, catalogue loading and validation,
//   translation guard, studio wizard and WAV statistics, tray menu spec and
//   icon, settings round-trip, output-chain click bound, and the AppModel
//   driving an AudioController over a fake backend.
//
// Usage: test_frontend_model [repo-root [stats-sample-file]]
// With a second argument the GUI-produced [stats] lines of a Bypass run and a
// Mode A run (fake models) are written there for tools/analyze_soak.py --check-format.
#include "core/AppModel.hpp"
#include "core/DeviceSelectorModel.hpp"
#include "core/JsonRead.hpp"
#include "core/LoaderThread.hpp"
#include "core/Meters.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "core/Telemetry.hpp"
#include "core/TranslationModel.hpp"
#include "core/TrayModel.hpp"
#include "core/VoiceCatalog.hpp"
#include "core/VoiceStudio.hpp"
#include "pipeline/ModeAHost.hpp"
#include "pipeline/ModelBundle.hpp"
#include "pipeline/OrtModel.hpp"
#include "voice/VoiceParams.hpp"
#include "FakeAudioBackend.hpp"
#include "FakeModelStep.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace {

using namespace fluxvoice;
using namespace fluxvoice::frontend;
using fluxvoice::testing::FakeAudioBackend;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

fs::path g_repo = ".";
fs::path g_tmp;
fs::path g_stats_sample;  // optional output of write_stats_sample()

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool contains(std::string_view hay, std::string_view needle) { return hay.find(needle) != std::string_view::npos; }

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

// --------------------------------------------------------------------------
// WAV writer for the Studio tests (RIFF/WAVE PCM 16/24-bit, little-endian).
// --------------------------------------------------------------------------
void put32(std::string& s, std::uint32_t v) { for (int i = 0; i < 4; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF)); }
void put16(std::string& s, std::uint16_t v) { for (int i = 0; i < 2; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF)); }

void write_wav(const fs::path& path, const std::vector<float>& mono, std::uint32_t sr, std::uint16_t bits,
               std::uint16_t channels = 1) {
    const std::uint16_t bytes = static_cast<std::uint16_t>(bits / 8);
    std::string data;
    for (float x : mono) {
        for (std::uint16_t c = 0; c < channels; ++c) {
            const double v = x < -1.0f ? -1.0 : (x > 1.0f ? 1.0 : static_cast<double>(x));
            if (bits == 16) {
                put16(data, static_cast<std::uint16_t>(static_cast<std::int16_t>(std::lround(v * 32767.0))));
            } else {
                const auto i = static_cast<std::int32_t>(std::lround(v * 8388607.0));
                data.push_back(static_cast<char>(i & 0xFF));
                data.push_back(static_cast<char>((i >> 8) & 0xFF));
                data.push_back(static_cast<char>((i >> 16) & 0xFF));
            }
        }
    }
    std::string s;
    s += "RIFF";
    put32(s, static_cast<std::uint32_t>(36 + data.size()));
    s += "WAVEfmt ";
    put32(s, 16);
    put16(s, 1);
    put16(s, channels);
    put32(s, sr);
    put32(s, sr * channels * bytes);
    put16(s, static_cast<std::uint16_t>(channels * bytes));
    put16(s, bits);
    s += "data";
    put32(s, static_cast<std::uint32_t>(data.size()));
    s += data;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << s;
}

std::vector<float> sine(double seconds, std::uint32_t sr, float amp, double hz = 220.0) {
    std::vector<float> v(static_cast<std::size_t>(seconds * sr));
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = amp * static_cast<float>(std::sin(2.0 * 3.14159265358979 * hz * static_cast<double>(i) / sr));
    return v;
}

/// Alternating 100 ms blocks of "speech" (amp_speech) and "quiet" (amp_quiet).
std::vector<float> alternating(double seconds, std::uint32_t sr, float amp_speech, float amp_quiet) {
    std::vector<float> v(static_cast<std::size_t>(seconds * sr));
    const std::size_t block = sr / 10;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const float amp = ((i / block) % 2 == 0) ? amp_speech : amp_quiet;
        v[i] = amp * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 220.0 * static_cast<double>(i) / sr));
    }
    return v;
}

// --------------------------------------------------------------------------
// Device rows & selection policy
// --------------------------------------------------------------------------
void test_device_selector() {
    FakeAudioBackend fb;
    const auto mic   = fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true);
    const auto cout_ = fb.make_ep(EndpointDirection::Capture, 2, "CABLE Output (VB-Audio Virtual Cable)");
    const auto spk   = fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true);
    const auto cin   = fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input (VB-Audio Virtual Cable)");
    fb.capture  = {mic, cout_};
    fb.playback = {spk, cin};
    const auto snap = std::get<DeviceSnapshot>(fb.enumerate());

    const DeviceRows rows = build_rows(snap, EndpointDirection::Capture, false);
    CHECK(rows.rows.size() == 3);
    CHECK(rows.rows[0].follow_default);
    CHECK(rows.rows[0].label == "Follow system default");
    CHECK(contains(rows.rows[1].label, "(default)"));
    CHECK(rows.generation == snap.generation && rows.backend == "wasapi" && !rows.stale);

    // Row -> policy carries key + generation, never the index.
    CHECK(selection_for_row(rows, 2).has_value());
    const DeviceSelection sel = selection_for_row(rows, 2).value_or(DeviceSelection{FollowSystemDefault{}});
    CHECK(std::holds_alternative<ExplicitEndpoint>(sel));
    if (const auto* e = std::get_if<ExplicitEndpoint>(&sel)) {
        CHECK(e->key == cout_.key);
        CHECK(e->generation == snap.generation);
    }
    CHECK(selection_for_row(rows, 0) && std::holds_alternative<FollowSystemDefault>(*selection_for_row(rows, 0)));
    // Stage 02: an out-of-range row is an error, never "Follow system default".
    CHECK(!selection_for_row(rows, 3).has_value());
    CHECK(!selection_for_row(rows, 99).has_value());
    CHECK(!selection_for_row(DeviceRows{}, 0).has_value());

    // Policy -> row round trip; a vanished key maps to nothing (never another row).
    CHECK(row_for_selection(rows, sel) == std::size_t{2});
    CHECK(row_for_selection(rows, FollowSystemDefault{}) == std::size_t{0});
    const DeviceSelection gone = ExplicitEndpoint{EndpointKey{EndpointDirection::Capture, 1, 999}, snap.generation};
    CHECK(!row_for_selection(rows, gone).has_value());
    CHECK(describe_selection(gone, snap, EndpointDirection::Capture) == "(device no longer listed)");
    CHECK(describe_selection(sel, snap, EndpointDirection::Capture) == cout_.name);
    CHECK(resolved_name(FollowSystemDefault{}, snap, EndpointDirection::Playback) == "Speakers");

    // Loopback capture rows come from the playback list.
    const DeviceRows lb = build_rows(snap, EndpointDirection::Capture, true, CaptureMode::SystemPlaybackLoopback);
    CHECK(lb.rows.size() == 3 && lb.stale);
    CHECK(contains(lb.rows[1].label, "Speakers"));

    // Feedback guard preview.
    CHECK(feedback_preview(snap, ExplicitEndpoint{cout_.key, snap.generation}, ExplicitEndpoint{cin.key, snap.generation},
                           CaptureMode::OrdinaryMicrophone).has_value());
    CHECK(!feedback_preview(snap, ExplicitEndpoint{mic.key, snap.generation}, ExplicitEndpoint{cin.key, snap.generation},
                            CaptureMode::OrdinaryMicrophone).has_value());
    CHECK(feedback_preview(snap, ExplicitEndpoint{spk.key, snap.generation}, FollowSystemDefault{},
                           CaptureMode::SystemPlaybackLoopback).has_value());

    AudioError err{AudioErrorCategory::DeviceNotFound, "open capture", 0, "wasapi", "USB Mic is gone."};
    const std::string text = explain_error(err);
    CHECK(contains(text, "device not found") && contains(text, "Choose the device again"));
}

// --------------------------------------------------------------------------
// Voice catalogue
// --------------------------------------------------------------------------
void test_voice_catalog() {
    const fs::path fixtures = g_repo / "tests" / "data" / "fixtures";
    const auto parsed = parse_voices_json(slurp(fixtures / "identity" / "voices.json"));
    CHECK(parsed.ok());
    CHECK(parsed.voices.size() == 2);
    if (parsed.voices.size() == 2) {
        CHECK(parsed.voices[0].name == "identity" && parsed.voices[1].name == "identity_b");
        CHECK(parsed.voices[0].embedding.size() == 4);
        CHECK(near(parsed.voices[0].logf0_mean, 4.8f));
        CHECK(parsed.voices[0].license_class() == LicenseClass::CommercialClean);
    }

    // Rejections mirror fvml.contract.validate_voices (BUG-006 included).
    const char* voice = R"({"name":"lj","display":"LJ","embedding":[0.5,-0.5],"logf0_mean":5.3,"logf0_std":0.2,"energy_mean":-3.0,"energy_std":1.1,"license":"PD","source":"LJSpeech"})";
    auto doc = [&](const char* schema, const char* extra = "") {
        return std::string("{\"schema\":") + schema + ",\"voices\":[" + voice + "]" + extra + "}";
    };
    CHECK(parse_voices_json(doc("1")).ok());
    CHECK(!parse_voices_json(doc("true")).ok());
    CHECK(!parse_voices_json(doc("1.0")).ok());
    CHECK(!parse_voices_json(doc("2")).ok());
    CHECK(!parse_voices_json(doc("1", ",\"extra\":1")).ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[1]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"LJ","display":"x","embedding":[0.1],"logf0_mean":1,"logf0_std":1,"energy_mean":1,"energy_std":1,"license":"x","source":"x"}]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"lj","display":"x","embedding":[0.1],"logf0_mean":1,"logf0_std":0,"energy_mean":1,"energy_std":1,"license":"x","source":"x"}]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"lj","display":"x","embedding":[true],"logf0_mean":1,"logf0_std":1,"energy_mean":1,"energy_std":1,"license":"x","source":"x"}]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"lj","display":"x","embedding":[0.1],"logf0_mean":true,"logf0_std":1,"energy_mean":1,"energy_std":1,"license":"x","source":"x"}]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"lj","display":"","embedding":[0.1],"logf0_mean":1,"logf0_std":1,"energy_mean":1,"energy_std":1,"license":"x","source":"x"}]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"lj","embedding":[0.1],"logf0_mean":1,"logf0_std":1,"energy_mean":1,"energy_std":1,"license":"x","source":"x"}]})").ok());
    CHECK(!parse_voices_json(R"({"schema":1,"voices":[{"name":"lj","display":"x","embedding":[0.1],"logf0_mean":1,"logf0_std":1,"energy_mean":1,"energy_std":1,"license":"x","source":"x","colour":"red"}]})").ok());
    {
        const std::string dup = std::string("{\"schema\":1,\"voices\":[") + voice + "," + voice + "]}";
        const auto r = parse_voices_json(dup);
        CHECK(!r.ok() && contains(r.error, "duplicate"));
    }
    CHECK(!parse_voices_json("[]").ok());
    CHECK(!parse_voices_json("nope").ok());

    // BUG-032: optional per-voice "lang" round-trips / is validated like the Python twin.
    {
        const std::string base = voice;
        const auto with = [&](const std::string& extra) {
            return std::string("{\"schema\":1,\"voices\":[") + base.substr(0, base.size() - 1) + extra + "}]}";
        };
        const auto none = parse_voices_json(with(""));
        CHECK(none.ok() && none.voices.size() == 1 && none.voices[0].lang.empty());
        const auto en = parse_voices_json(with(",\"lang\":\"en\",\"attribution\":\"x\""));
        CHECK(en.ok() && en.voices.size() == 1 && en.voices[0].lang == "en" && en.voices[0].attribution == "x");
        const auto bad = parse_voices_json(with(",\"lang\":\"EN\""));
        CHECK(!bad.ok() && contains(bad.error, "voice 'lj': invalid language code 'EN' (expected lowercase [a-z-]{2,8})"));
        const auto empty = parse_voices_json(with(",\"lang\":\"\""));
        CHECK(!empty.ok() && contains(empty.error, "lang must be a non-empty string"));
        CHECK(!parse_voices_json(with(",\"lang\":5")).ok());
    }

    CHECK(classify_license("CC BY 4.0") == LicenseClass::CommercialClean);
    CHECK(classify_license("CC BY-NC-SA 4.0") == LicenseClass::ResearchOnly);
    CHECK(classify_license("private - do not distribute") == LicenseClass::Private);
    CHECK(classify_license("") == LicenseClass::Unknown);
    CHECK(is_valid_voice_name("me-2") && !is_valid_voice_name("Me") && !is_valid_voice_name("") && !is_valid_voice_name("a b"));

    // Scan the fixture root: Mode A v1 `identity/` + Mode B v2 `s2ut/` bundles.
    VoiceCatalog cat;
    CHECK(cat.scan_models_root(fixtures) == 2);
    CHECK(cat.scan_models_root(g_repo / "does-not-exist") == 0);
    CHECK(cat.bundles().size() == 2);
    const BundleInfo* b = cat.bundle("identity");
    CHECK(b != nullptr);
    if (b) {
        CHECK(b->valid() && b->voices.size() == 2);
        CHECK(b->has_encoder && b->has_vocoder && !b->has_translator && !b->has_langs);
        CHECK(b->onnx_files.size() == 6);
    }
    const BundleInfo* s2ut = cat.bundle("s2ut");
    CHECK(s2ut != nullptr);
    if (s2ut) {
        CHECK(s2ut->valid() && s2ut->voices.size() == 2);
        CHECK(s2ut->has_encoder && s2ut->has_vocoder && s2ut->has_translator && s2ut->has_langs);
        CHECK(s2ut->onnx_files.size() == 8);
        CHECK(s2ut->languages.size() == 4);
    }
    CHECK(cat.find_voice("identity", "identity_b") != nullptr);
    CHECK(cat.find_voice("identity", "nope") == nullptr);
    CHECK(cat.voice_names("identity").size() == 2);
    CHECK(cat.installed_voice_count() == 4);

    // An invalid bundle is listed with its diagnostic, not skipped.
    const fs::path bad = g_tmp / "bad-bundle";
    fs::create_directories(bad);
    { std::ofstream f(bad / "voices.json"); f << R"({"schema":2,"voices":[]})"; }
    std::string err;
    CHECK(cat.add_bundle_dir(bad, &err));
    CHECK(cat.bundle("bad-bundle") && !cat.bundle("bad-bundle")->valid());
    CHECK(!cat.add_bundle_dir(g_tmp, &err) && contains(err, "no voices.json"));

    // Catalogue index from the shipped sample.
    CHECK(cat.load_catalogue_index(g_repo / "frontend" / "assets" / "catalogue" / "index.json", &err));
    CHECK(cat.indices().size() == 1);
    if (!cat.indices().empty()) CHECK(cat.indices()[0].entries.size() == 3);
    const auto bad_idx = parse_catalogue_index(R"({"schema":1,"entries":[{"id":"a"}]})");
    CHECK(!bad_idx.ok());
    CHECK(!parse_catalogue_index(R"({"schema":1,"entries":[{"id":"a","display":"A"},{"id":"a","display":"B"}]})").ok());

    VoiceCatalog::Filter f;
    auto all = cat.search(f);
    CHECK(all.size() == 7);  // 4 installed voices (identity + s2ut) + 3 catalogue entries
    bool fixture_installed = false;
    for (const auto& it : all)
        if (it.id == "fixture-identity") fixture_installed = it.installed;
    CHECK(fixture_installed);
    f.installed_only = true;
    CHECK(cat.search(f).size() == 5);  // 4 voices + the catalogue entry whose bundle is installed
    f = {};
    f.license = LicenseClass::ResearchOnly;
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.query = "HI-FI";
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.language = "de";
    CHECK(cat.search(f).size() == 3);  // 2 voices in s2ut (en/es/fr/de) + 1 catalogue entry
    f = {};
    f.tag = "test";
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.sort = VoiceCatalog::Sort::License;
    auto sorted = cat.search(f);
    CHECK(!sorted.empty() && sorted.front().license_class == LicenseClass::CommercialClean);
    CHECK(cat.all_languages().size() == 4);
    CHECK(!cat.all_tags().empty());

    // BUG-032: per-voice "lang" tags drive language filtering and
    // all_languages() without a langs.json (a Stage 03 multilingual Mode A
    // bundle of tagged voices).
    const fs::path tagged = g_tmp / "tagged-bundle";
    fs::create_directories(tagged);
    {
        const std::string vopen = std::string(voice).substr(0, std::string(voice).size() - 1);
        std::ofstream f2(tagged / "voices.json");
        f2 << "{\"schema\":1,\"voices\":[" << vopen << ",\"lang\":\"es\"},"
           << vopen << ",\"name\":\"lj2\",\"display\":\"LJ 2\",\"lang\":\"it\"}]}";
    }
    CHECK(cat.add_bundle_dir(tagged, &err));
    f = {};
    f.language = "it";
    const auto its = cat.search(f);
    CHECK(its.size() == 1 && its[0].id == "tagged-bundle/lj2");
    f.language = "es";
    bool tagged_es = false;
    for (const auto& it : cat.search(f)) tagged_es = tagged_es || it.id == "tagged-bundle/lj";
    CHECK(tagged_es);                       // alongside the s2ut bundle's langs.json voices
    const auto langs_now = cat.all_languages();
    CHECK(langs_now.size() == 5 && std::find(langs_now.begin(), langs_now.end(), "it")
              != langs_now.end());
}

// --------------------------------------------------------------------------
// Translation
// --------------------------------------------------------------------------
void test_translation() {
    const LangTable builtin = builtin_lang_table();
    CHECK(builtin.ok() && builtin.langs.size() == 4);
    const LangTable asset = parse_langs_json(slurp(g_repo / "frontend" / "assets" / "langs.default.json"));
    CHECK(asset.ok());
    CHECK(asset.langs.size() == builtin.langs.size());
    for (std::size_t i = 0; i < asset.langs.size() && i < builtin.langs.size(); ++i) {
        CHECK(asset.langs[i].code == builtin.langs[i].code && asset.langs[i].id == builtin.langs[i].id);
    }
    CHECK(asset.default_wait_k == builtin.default_wait_k);
    CHECK(builtin.default_k("de", "en") == 4 && builtin.default_k("es", "en") == 3 && builtin.default_k("xx", "yy") == 3);
    CHECK(builtin.by_code("fr") && builtin.by_code("fr")->id == 2 && builtin.by_id(3) && builtin.by_id(3)->code == "de");

    CHECK(!parse_langs_json(R"({"schema":true,"langs":[{"code":"en","id":0}]})").ok());
    CHECK(!parse_langs_json(R"({"schema":1,"langs":[]})").ok());
    CHECK(!parse_langs_json(R"({"schema":1,"langs":[{"code":"EN","id":0}]})").ok());
    CHECK(!parse_langs_json(R"({"schema":1,"langs":[{"code":"en","id":0},{"code":"es","id":0}]})").ok());
    CHECK(!parse_langs_json(R"({"schema":1,"langs":[{"code":"en","id":0}],"default_wait_k":{"en->es":11}})").ok());

    TranslationModel tm;
    CHECK(!tm.translator_available());
    PairValidation v = tm.validate_pair("en", "en");
    CHECK(!v.ok && contains(v.message, "Mode A"));
    v = tm.validate_pair("en", "xx");
    CHECK(!v.ok && contains(v.message, "unknown target"));
    v = tm.validate_pair("es", "en");
    CHECK(v.ok && !v.message.empty());  // advisory: no bundle / translator yet
    tm.set_translator_available(true);
    CHECK(tm.validate_pair("es", "en").message.empty());
    // Wait-k range and default come from the wait_k descriptor (Stage 03).
    const ParamDesc& wk = param_desc(ParamId::WaitK);
    CHECK(TranslationModel::wait_k_min() == static_cast<int>(wk.min) && TranslationModel::wait_k_max() == static_cast<int>(wk.max));
    CHECK(TranslationModel::clamp_wait_k(0) == static_cast<int>(wk.min));
    CHECK(TranslationModel::clamp_wait_k(99) == static_cast<int>(wk.max));
    CHECK(TranslationModel::clamp_wait_k(5) == 5);
    CHECK(builtin.default_k("xx", "yy") == static_cast<int>(wk.def));
    const std::string over = R"({"schema":1,"langs":[{"code":"en","id":0}],"default_wait_k":{"en->es":)" +
                             std::to_string(static_cast<int>(wk.max) + 1) + "}}";
    const std::string at_max = R"({"schema":1,"langs":[{"code":"en","id":0}],"default_wait_k":{"en->es":)" +
                               std::to_string(static_cast<int>(wk.max)) + "}}";
    CHECK(!parse_langs_json(over).ok() && parse_langs_json(at_max).ok());

    // Translator capabilities are unknown until the bundle's contract is read:
    // no onset figure, no adaptive control.
    CHECK(!tm.translator_caps().known && !tm.onset_estimate_ms(3).has_value() && !tm.adaptive_policy_available());
    tm.set_translator_caps(TranslatorCaps{true, 80.0, false});
    CHECK(tm.onset_estimate_ms(3) && near(static_cast<float>(*tm.onset_estimate_ms(3)), 240.0f));
    CHECK(!tm.adaptive_policy_available());
    tm.set_translator_caps(TranslatorCaps{true, 40.0, true});
    CHECK(tm.onset_estimate_ms(3) && near(static_cast<float>(*tm.onset_estimate_ms(3)), 120.0f));
    CHECK(tm.adaptive_policy_available());
    tm.set_translator_available(false);
    CHECK(!tm.adaptive_policy_available());  // declared, but no translator to use it
}

// --------------------------------------------------------------------------
// Tray
// --------------------------------------------------------------------------
const TrayItem* find_item(const std::vector<TrayItem>& items, std::string_view id) {
    for (const TrayItem& it : items) {
        if (it.id == id) return &it;
        if (const TrayItem* c = find_item(it.children, id)) return c;
    }
    return nullptr;
}

void test_tray_model() {
    TrayInputs in;
    in.presets = {"Neutral", "Deep"};
    in.voices  = {"identity", "identity_b"};
    in.language_pairs = {"es -> en"};
    TraySpec idle = build_tray_spec(in);
    CHECK(idle.state == TrayState::Idle && contains(idle.tooltip, "Idle"));
    {   // BUG-020: an unresolved remembered device puts the icon in the error state.
        TrayInputs unresolved = in;
        unresolved.device_unresolved = true;
        unresolved.error_text = "DeviceNotFound: the remembered playback device 'CABLE Input' is not listed";
        const TraySpec u = build_tray_spec(unresolved);
        CHECK(u.state == TrayState::Error && contains(u.tooltip, "CABLE Input"));
    }
    CHECK(find_item(idle.items, "show") && find_item(idle.items, "show")->label == "Hide panel");
    CHECK(find_item(idle.items, "engine") && !find_item(idle.items, "engine")->checked);
    CHECK(find_item(idle.items, "mute") && !find_item(idle.items, "mute")->enabled);
    CHECK(find_item(idle.items, "preset:Deep") && find_item(idle.items, "preset:Deep")->kind == TrayItem::Kind::Radio);
    CHECK(find_item(idle.items, "langs") && !find_item(idle.items, "langs")->enabled);  // no translator
    CHECK(find_item(idle.items, "mode:s2ut") && !find_item(idle.items, "mode:s2ut")->enabled);
    CHECK(find_item(idle.items, "quit"));

    in.stream = StreamState::Running;
    in.engine_on = true;
    in.preset = "Deep";
    in.queued_ms = 42.0;
    in.window_visible = false;
    TraySpec running = build_tray_spec(in);
    CHECK(running.state == TrayState::Running);
    CHECK(contains(running.tooltip, "Running") && contains(running.tooltip, "Deep") && contains(running.tooltip, "42 ms"));
    CHECK(find_item(running.items, "show")->label == "Show panel");
    CHECK(find_item(running.items, "engine")->checked);
    CHECK(find_item(running.items, "preset:Deep")->checked && !find_item(running.items, "preset:Neutral")->checked);
    CHECK(running.hash() != idle.hash());
    {
        TrayInputs tip = in;
        tip.queued_ms = 43.0;  // tooltip changes, menu does not
        const TraySpec other = build_tray_spec(tip);
        CHECK(other.hash() != running.hash() && other.menu_hash() == running.menu_hash());
    }

    in.muted = true;
    CHECK(build_tray_spec(in).state == TrayState::Muted);
    in.muted = false;
    in.stream = StreamState::Recovering;
    in.retry_attempt = 2;
    CHECK(build_tray_spec(in).state == TrayState::Recovering && contains(build_tray_spec(in).tooltip, "attempt 2"));
    in.stream = StreamState::AwaitingUser;
    in.error_text = "device not found: USB Mic";
    const TraySpec err = build_tray_spec(in);
    CHECK(err.state == TrayState::Error && contains(err.tooltip, "USB Mic"));

    // Actions
    CHECK(action_for_item("engine") && action_for_item("engine")->type == TrayAction::Type::ToggleEngine);
    CHECK(action_for_item("preset:Deep") && action_for_item("preset:Deep")->arg == "Deep");
    CHECK(action_for_item("voice:identity_b") && action_for_item("voice:identity_b")->type == TrayAction::Type::SetVoice);
    CHECK(action_for_item("lang:es -> en") && action_for_item("lang:es -> en")->arg == "es -> en");
    CHECK(action_for_item("mode:s2ut") && action_for_item("mode:s2ut")->type == TrayAction::Type::SetModeTranslate);
    CHECK(!action_for_item("presets").has_value());
    CHECK(!action_for_item("preset:").has_value());
    CHECK(!action_for_item("").has_value());

    // Icon: a white bar pixel at the centre, transparent corner, muted slash.
    const TrayIconPixels run = render_tray_icon(TrayState::Running);
    constexpr int N = TrayIconPixels::kSize;
    auto px = [&](const TrayIconPixels& p, int x, int y) { return p.rgba[static_cast<std::size_t>(y * N + x)]; };
    CHECK(px(run, 16, 15) == 0xFFFFFFFFu);
    CHECK((px(run, 0, 0) >> 24) == 0u);
    CHECK((px(run, 3, 16) & 0xFFu) == 0x2Eu);  // green tile R channel, left of the first bar
    CHECK(px(run, 4, 16) == 0xFFFFFFFFu);        // first bar starts at x = 4
    const TrayIconPixels muted = render_tray_icon(TrayState::Muted);
    CHECK(px(muted, 16, 15) != 0xFFFFFFFFu);
    CHECK(px(render_tray_icon(TrayState::Error), 3, 16) != px(run, 3, 16));
}

// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------
void test_settings() {
    Settings s;
    s.start_minimized = true;
    s.theme           = Theme::Light;
    s.capture_policy  = "name:USB Mic";
    s.wait_k          = 5;
    s.extras["future_key"] = "{\"a\":1}";
    std::vector<std::string> warnings;
    const Settings back = Settings::from_json(s.to_json(), &warnings);
    CHECK(warnings.empty());
    CHECK(back == s);
    CHECK(back.extras.count("future_key") == 1);

    // Stage 05 model-host fields: defaults, round trip and typed rejection.
    {
        const Settings d;
        CHECK(d.frames_per_step == 0 && d.latency_target_ms == 0);
        Settings m;
        m.frames_per_step   = 2;
        m.latency_target_ms = 120;
        std::vector<std::string> w;
        const Settings mb = Settings::from_json(m.to_json(), &w);
        CHECK(w.empty() && mb == m && mb.frames_per_step == 2 && mb.latency_target_ms == 120);
        CHECK(m.to_json().find("\"frames_per_step\": 2") != std::string::npos);
        CHECK(m.to_json().find("\"latency_target_ms\": 120") != std::string::npos);
        for (int ok : {0, 1, 2, 4}) {
            Settings v;
            v.frames_per_step = ok;
            CHECK(Settings::from_json(v.to_json()).frames_per_step == ok);
        }
        // Not 0/1/2/4, wrong type, fractional, negative, too large: the default, with a warning each.
        for (const char* doc : {R"({"frames_per_step":3})", R"({"frames_per_step":"2"})", R"({"frames_per_step":2.5})",
                                R"({"frames_per_step":-1})", R"({"frames_per_step":8})", R"({"frames_per_step":null})",
                                R"({"frames_per_step":99999999999})"}) {
            std::vector<std::string> ww;
            const Settings bs = Settings::from_json(doc, &ww);
            CHECK(bs.frames_per_step == 0);
            CHECK(ww.size() == 1);
        }
        for (const char* doc : {R"({"latency_target_ms":-5})", R"({"latency_target_ms":"80"})",
                                R"({"latency_target_ms":5001})", R"({"latency_target_ms":18446744073709551615})",
                                R"({"latency_target_ms":true})"}) {
            std::vector<std::string> ww;
            const Settings bs = Settings::from_json(doc, &ww);
            CHECK(bs.latency_target_ms == 0);
            CHECK(ww.size() == 1);
        }
        CHECK(Settings::from_json(R"({"latency_target_ms":1000.0})").latency_target_ms == 1000);  // integral float
        CHECK(Settings::from_json(R"({"latency_target_ms":99.5})").latency_target_ms == 0);       // fractional
        CHECK(Settings::from_json(R"({"latency_target_ms":5000})").latency_target_ms == 5000);
        // The new keys are known: never parked in extras.
        CHECK(Settings::from_json(R"({"frames_per_step":1,"latency_target_ms":90})").extras.empty());
    }

    // Stage 06 stats-log fields: defaults, round trip, typed rejection, known keys.
    {
        const Settings d;
        CHECK(!d.write_stats_log && d.stats_log_path.empty());
        Settings m;
        m.write_stats_log = true;
        m.stats_log_path  = "/var/tmp/fv \xC3\xA9/stats.log";  // UTF-8 text, kept verbatim
        std::vector<std::string> w;
        const Settings mb = Settings::from_json(m.to_json(), &w);
        CHECK(w.empty() && mb == m && mb.write_stats_log && mb.stats_log_path == m.stats_log_path);
        CHECK(m.to_json().find("\"write_stats_log\": true") != std::string::npos);
        CHECK(mb.extras.empty());
        for (const char* doc : {R"({"write_stats_log":"yes"})", R"({"write_stats_log":1})", R"({"write_stats_log":null})"}) {
            std::vector<std::string> ww;
            CHECK(!Settings::from_json(doc, &ww).write_stats_log);
            CHECK(ww.size() == 1);
        }
        for (const char* doc : {R"({"stats_log_path":5})", R"({"stats_log_path":null})", R"({"stats_log_path":["a"]})"}) {
            std::vector<std::string> ww;
            CHECK(Settings::from_json(doc, &ww).stats_log_path.empty());
            CHECK(ww.size() == 1);
        }
        std::vector<std::string> nul_w;
        CHECK(Settings::from_json(R"({"stats_log_path":"a\u0000b"})", &nul_w).stats_log_path.empty());  // a NUL would truncate the path
        CHECK(nul_w.size() == 1);
    }

    const Settings bad = Settings::from_json(R"({"schema":1,"wait_k":42,"theme":"neon","start_minimized":"yes","ui_scale":9})", &warnings);
    CHECK(bad.wait_k == 3 && bad.theme == Theme::Dark && !bad.start_minimized && near(bad.ui_scale, 1.0f));
    CHECK(warnings.size() == 4);
    warnings.clear();
    CHECK(Settings::from_json("garbage", &warnings) == Settings{});
    CHECK(warnings.size() == 1);

    Theme t = Theme::Dark;
    CHECK(theme_from_string("system", t) && t == Theme::System);
    CHECK(!theme_from_string("blue", t));

    const fs::path file = g_tmp / "settings" / "settings.json";
    std::string err;
    CHECK(s.save(file, err));
    CHECK(Settings::load(file) == s);
    CHECK(Settings::load(g_tmp / "nope.json") == Settings{});
}

// --------------------------------------------------------------------------
// Meters and the loader-thread plumbing (M4 Stage 05, toolkit-free parts)
// --------------------------------------------------------------------------
void test_meters() {
    CHECK(near(meters::to_dbfs(1.0f), 0.0f) && near(meters::to_dbfs(0.5f), -6.0206f, 1e-3f));
    CHECK(meters::to_dbfs(0.0f) == -120.0f && meters::to_dbfs(1e-7f) == -120.0f);
    const std::vector<float> block = {0.5f, -0.5f, 0.5f, -0.5f};
    float peak = -1.0f, rms = -1.0f;
    meters::measure(block.data(), block.size(), peak, rms);
    CHECK(near(peak, 0.5f) && near(rms, 0.5f));
    meters::measure(block.data(), 0, peak, rms);
    CHECK(peak == 0.0f && rms == 0.0f);
}

void test_loader_thread() {
    LoaderThread<int> lt;
    CHECK(!lt.busy() && !lt.poll());

    // The job runs off the calling thread, the result is queued until drained.
    std::mutex m;
    std::condition_variable cv;
    bool release = false;
    std::thread::id job_thread;
    CHECK(lt.start([&] {
        job_thread = std::this_thread::get_id();
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return release; });
        return 42;
    }));
    CHECK(lt.busy());
    CHECK(!lt.start([] { return 1; }));  // one load at a time
    CHECK(!lt.poll());                   // still running
    {
        const std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();
    std::optional<int> got;
    for (int i = 0; i < 1000 && !got; ++i) {
        got = lt.poll();
        if (!got) std::this_thread::sleep_for(2ms);
    }
    CHECK(got && *got == 42);
    CHECK(job_thread != std::this_thread::get_id());
    CHECK(!lt.busy() && !lt.poll());  // drained exactly once

    // A finished job whose result is not drained still blocks the next start.
    CHECK(lt.start([] { return 7; }));
    std::this_thread::sleep_for(50ms);  // the job has certainly finished by now
    CHECK(!lt.start([] { return 8; }));
    std::optional<int> second;
    for (int i = 0; i < 1000 && !second; ++i) {
        second = lt.poll();
        if (!second) std::this_thread::sleep_for(2ms);
    }
    CHECK(second && *second == 7);
    CHECK(lt.start([] { return 9; }));  // free again

    // An exception escaping the job is rethrown by poll(), once, and the loader is reusable.
    LoaderThread<int> bad;
    CHECK(bad.start([]() -> int { throw std::runtime_error("boom"); }));
    bool threw = false;
    for (int i = 0; i < 1000 && !threw; ++i) {
        try {
            if (bad.poll()) break;
        } catch (const std::runtime_error& e) {
            threw = std::string_view(e.what()) == "boom";
        }
        if (!threw) std::this_thread::sleep_for(2ms);
    }
    CHECK(threw && !bad.busy());
    CHECK(bad.start([] { return 1; }));

    // The destructor joins a running job (no detached thread outlives its owner).
    std::atomic<bool> finished{false};
    {
        LoaderThread<int> scoped;
        CHECK(scoped.start([&] {
            std::this_thread::sleep_for(30ms);
            finished.store(true);
            return 0;
        }));
    }
    CHECK(finished.load());
}

// --------------------------------------------------------------------------
// Telemetry text and history
// --------------------------------------------------------------------------
void test_telemetry_text() {
    // The Bypass output stage is fluxvoice::VoiceOutputChain (tested in test_voice_params);
    // its host-level behaviour is covered by test_bypass_* below.
    TelemetrySnapshot t;
    t.frames = 10; t.output_underruns = 3; t.input_overruns = 1; t.output_drops = 1; t.queued_samples = 1;
    t.sample_rate = 16000; t.rss_bytes = 2048;
    t.stats.frames = 10; t.stats.underruns = 3; t.stats.overruns = 1; t.stats.drops = 1; t.stats.queued = 1;
    t.stats.rss = 2048;
    // The legacy line (vc false), spelled out independently of format_stats_line().
    CHECK(t.stats_line(1.0) == "[stats] t=1.0 frames=10 overruns=1 underruns=3 drops=1 queued=1 skipped=0 prefilled=0 rss_kb=2");
    CHECK(t.stats_line(1.0) == fluxvoice::format_stats_line(1.0, t.stats));
    CHECK(near(static_cast<float>(t.queued_ms()), 0.0625f));

    History<4> h;
    for (float v : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}) h.push(v);
    const auto o = h.ordered();
    CHECK(h.size() == 4 && o[0] == 2.0f && o[3] == 5.0f && h.latest() == 5.0f && h.max() == 5.0f);
}

// --------------------------------------------------------------------------
// Voice Studio
// --------------------------------------------------------------------------
void test_voice_studio() {
    const std::uint32_t sr = 16000;
    // Levels: -6 dBFS sine, no clipping.
    const auto s6 = sine(3.0, sr, 0.5f);
    WavStats st = analyze_samples(s6.data(), s6.size(), 1, sr);
    CHECK(st.ok && near(st.peak_dbfs, -6.02f, 0.1f) && !st.clipped() && near(static_cast<float>(st.seconds), 3.0f, 1e-3f));
    auto clipped = s6;
    clipped[100] = 0.9995f;
    st = analyze_samples(clipped.data(), clipped.size(), 1, sr);
    CHECK(st.clipped_samples == 1 && judge_take(st) == TakeVerdict::Clipped);

    // SNR from alternating blocks: clean vs. too noisy.
    const auto clean = alternating(4.0, sr, 0.3f, 0.001f);
    const WavStats clean_stats = analyze_samples(clean.data(), clean.size(), 1, sr);
    st = clean_stats;
    CHECK(st.snr_db > 45.0f && judge_take(st) == TakeVerdict::Clean);
    const auto noisy = alternating(4.0, sr, 0.3f, 0.1f);
    st = analyze_samples(noisy.data(), noisy.size(), 1, sr);
    CHECK(st.snr_db < 12.0f && judge_take(st) == TakeVerdict::LowSnr);
    CHECK(judge_take(analyze_samples(s6.data(), 8000, 1, sr)) == TakeVerdict::TooShort);
    std::vector<float> stereo(clean.size() * 2);
    for (std::size_t i = 0; i < clean.size(); ++i) stereo[2 * i] = stereo[2 * i + 1] = clean[i];
    CHECK(judge_take(analyze_samples(stereo.data(), clean.size(), 2, sr)) == TakeVerdict::NotMono);
    CHECK(!analyze_samples(stereo.data(), 0, 0, sr).ok);

    // WAV files: 16-bit and 24-bit agree with the in-memory analysis; bad files are reported.
    // Named <voice>_sNN_TTTT.wav, the layout prepare_target_voice.py reads (BUG-025).
    const fs::path w16 = g_tmp / "me_s01_0001.wav", w24 = g_tmp / "me_s02_0002.wav", w2 = g_tmp / "me_s01_0003.wav";
    write_wav(w16, clean, sr, 16);
    write_wav(w24, clean, sr, 24);
    write_wav(w2, clean, sr, 16, 2);
    const WavStats f16 = analyze_wav_file(w16), f24 = analyze_wav_file(w24), f2 = analyze_wav_file(w2);
    CHECK(f16.ok && f16.bits == 16 && f16.channels == 1 && near(static_cast<float>(f16.seconds), 4.0f, 1e-3f));
    CHECK(f24.ok && f24.bits == 24 && near(f24.snr_db, clean_stats.snr_db, 6.0f) && judge_take(f24) == TakeVerdict::Clean);
    CHECK(near(f16.snr_db, clean_stats.snr_db, 6.0f));
    CHECK(near(f16.peak_dbfs, f24.peak_dbfs, 0.05f));
    CHECK(f2.ok && f2.channels == 2 && judge_take(f2) == TakeVerdict::NotMono);
    { std::ofstream f(g_tmp / "not.wav"); f << "RIFFxxxxWAVEjunk"; }
    CHECK(!analyze_wav_file(g_tmp / "not.wav").ok);
    CHECK(!analyze_wav_file(g_tmp / "missing.wav").ok);

    // Wizard.
    VoiceStudioModel m;
    m.requirements().minimum_minutes = 0.1;  // 6 s for the test; the product default is 60 min
    std::string why;
    CHECK(m.step() == StudioStep::NameConsent && !m.can_advance(&why) && contains(why, "Voice name"));
    m.draft().voice_name = "Me";
    CHECK(m.name_problem(nullptr).has_value());
    m.draft().voice_name = "me";
    CHECK(!m.name_problem(nullptr).has_value());
    VoiceCatalog cat;
    cat.scan_models_root(g_repo / "tests" / "data" / "fixtures");
    m.draft().voice_name = "identity";
    CHECK(m.name_problem(&cat).has_value());
    m.draft().voice_name = "me";
    m.draft().display_name = "Me";
    CHECK(!m.can_advance(&why) && contains(why, "right to use"));
    m.draft().consent = true;
    m.draft().is_own_voice = false;
    CHECK(!m.can_advance(&why) && contains(why, "consent"));
    m.draft().is_own_voice = true;
    CHECK(m.can_advance() && m.next() && m.step() == StudioStep::Setup);
    CHECK(!m.next());
    for (bool& b : m.draft().setup) b = true;
    CHECK(m.next() && m.step() == StudioStep::Takes);
    CHECK(!m.can_advance(&why) && contains(why, "clean minutes"));
    const Take& t1 = m.add_take(w16, "s01");
    CHECK(t1.verdict == TakeVerdict::Clean);
    const Take& t2 = m.add_take(w2, "s01");
    CHECK(t2.verdict == TakeVerdict::NotMono);
    CHECK(near(static_cast<float>(m.clean_minutes()), 4.0f / 60.0f, 1e-3f));
    CHECK(near(static_cast<float>(m.excluded_minutes()), 4.0f / 60.0f, 1e-3f));
    m.add_take(w24, "s02");
    CHECK(m.clean_minutes() >= 0.1 && near(m.progress_to_minimum(), 1.0f));
    CHECK(m.next() && m.step() == StudioStep::RoomTone);
    CHECK(!m.can_advance(&why) && contains(why, "room tone"));
    const auto quiet = sine(30.0, sr, 0.0005f);
    const fs::path room = g_tmp / "me_s01_roomtone.wav";
    write_wav(room, quiet, sr, 16);
    CHECK(m.add_room_tone(room).verdict == TakeVerdict::Clean);
    CHECK(m.room_tone_seconds() >= 29.0);
    m.add_live_mic(w16);
    CHECK(near(static_cast<float>(m.live_mic_minutes()), 4.0f / 60.0f, 1e-3f));
    CHECK(m.next() && m.step() == StudioStep::Prepare);

    const fs::path studio = g_tmp / "studio", models = g_tmp / "models";
    const StudioCommands posix = m.commands(g_repo, studio, models, Shell::Posix);
    // BUG-025: only real scripts and options; steps the CLIs cannot express yet
    // are labelled unavailable instead of emitting invented flags.
    CHECK(posix.prepare_unavailable.empty());
    CHECK(contains(posix.prepare, "prepare_target_voice.py") && contains(posix.prepare, "--speaker=me"));
    CHECK(posix.finetune.empty() && contains(posix.finetune_unavailable, "unavailable until M2 Stage 10 lands"));
    CHECK(posix.export_.empty() && !posix.export_unavailable.empty());
    CHECK(contains(posix.prepare, ".venv/bin/python"));
    const StudioCommands ps = m.commands(g_repo, studio, models, Shell::PowerShell);
    CHECK(contains(ps.prepare, "python.exe") && contains(ps.prepare, "'"));
    CHECK(shell_quote(fs::path("it's"), Shell::Posix) == "'it'\\''s'");
    CHECK(shell_quote(fs::path("it's"), Shell::PowerShell) == "'it''s'");

    std::string err;
    CHECK(!m.job_written(studio));
    CHECK(m.write_job(g_repo, studio, models, err));
    CHECK(err.empty() && m.job_written(studio));
    for (const char* f : {"job.json", "takes.tsv", "room_tone.tsv", "run_finetune.sh", "run_finetune.ps1"})
        CHECK(fs::is_regular_file(studio / "me" / f));
    const nlohmann::json job = nlohmann::json::parse(slurp(studio / "me" / "job.json"), nullptr, false);
    CHECK(!job.is_discarded() && job.at("voice") == "me" && job.at("takes").size() == 3 && job.at("license") == "private - do not distribute");
    CHECK(contains(m.takes_tsv(), "not mono"));
    CHECK(m.next() && m.step() == StudioStep::Install);
    CHECK(!m.can_advance() && !m.next());
    CHECK(!m.installed(cat));
    m.prev();
    CHECK(m.step() == StudioStep::Prepare);

    // Draft persistence round trip keeps takes and step.
    CHECK(m.save_draft(studio, err));
    VoiceStudioModel back;
    back.draft().voice_name = "me";
    CHECK(back.load_draft(studio));
    CHECK(back.step() == StudioStep::Prepare && back.draft().takes.size() == 3 && back.draft().room_tone.size() == 1);
    CHECK(back.draft().takes[1].verdict == TakeVerdict::NotMono && back.draft().consent);
    CHECK(!back.load_draft_json("{\"schema\":2}", &err));

    // status.json polling.
    CHECK(!m.poll_status(studio).has_value());
    { std::ofstream f(studio / "me" / "status.json"); f << R"({"stage":"finetune","progress":0.4,"message":"step 4000","eta_s":120})"; }
    const auto status = m.poll_status(studio);
    CHECK(status && status->stage == "finetune" && near(status->progress, 0.4f) && near(static_cast<float>(status->eta_s), 120.0f));
}

// --------------------------------------------------------------------------
// AppModel over the fake backend
// --------------------------------------------------------------------------
AppPaths test_paths() {
    AppPaths p;
    p.config_dir         = g_tmp / "config";
    p.repo_root          = g_repo;
    p.models_dir         = g_repo / "tests" / "data" / "fixtures";
    p.builtin_preset_dir = g_repo / "presets";
    p.user_preset_dir    = g_tmp / "config" / "presets";
    p.studio_dir         = g_tmp / "config" / "studio";
    p.assets_dir         = g_repo / "frontend" / "assets";
    return p;
}

void test_app_model() {
    FakeAudioBackend fb;
    const auto mic   = fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true);
    const auto cout_ = fb.make_ep(EndpointDirection::Capture, 2, "CABLE Output (VB-Audio Virtual Cable)");
    const auto spk   = fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true);
    const auto cin   = fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input (VB-Audio Virtual Cable)");
    fb.capture  = {mic, cout_};
    fb.playback = {spk, cin};

    const auto t0 = AppModel::Clock::now();
    AppModel app(fb, test_paths(), Settings{});
    app.initialise(t0);

    CHECK(app.active_bundle() == "identity");
    CHECK(app.voice_names().size() == 2 && app.voice_a() == "identity" && app.voice_b() == "identity");
    CHECK(app.presets().size() == 5 && app.preset_names().front() == "Neutral");
    CHECK(app.preset_warnings().empty());
    CHECK(!app.translation().translator_available());
    CHECK(app.catalog().indices().size() == 1);
    CHECK(app.section() == Section::Live);
    CHECK(app.backend_usable() && !app.engine_on());
    CHECK(app.status_summary().pill == "Idle");
    CHECK(app.snapshot().capture.size() == 2);

    // Devices: mic -> cable, no feedback; apply starts the stream.
    CHECK(app.capture_rows().rows.size() == 3 && app.playback_rows().rows.size() == 3);
    app.select_capture_row(2);
    app.select_playback_row(2);
    CHECK(app.feedback_warning().has_value());
    CHECK(!app.start_engine() && !app.engine_on() && contains(app.engine_message(), "feed back"));
    app.select_capture_row(1);
    CHECK(!app.feedback_warning().has_value());
    CHECK(app.apply_devices());
    CHECK(app.engine_on() && app.engine().status().state == StreamState::Running);
    CHECK(fb.open_calls == 1 && !fb.last_capture_was_default && !fb.last_playback_was_default);
    CHECK(app.settings().capture_policy == "name:USB Mic");
    CHECK(app.settings().playback_policy == "name:CABLE Input (VB-Audio Virtual Cable)");
    CHECK(fs::is_regular_file(test_paths().settings_file()));
    CHECK(app.capture_name() == "USB Mic" && app.playback_name() == cin.name);
    CHECK(app.status_summary().pill == "Running" && contains(app.status_summary().detail, "USB Mic"));

    // Worker: pump audio through the callback and wait for a hop.
    {
        std::vector<float> in(1600, 0.25f), out(1600, 0.0f);
        fb.pump_callback(in.data(), out.data(), in.size());
        bool hopped = false;
        for (int i = 0; i < 400 && !hopped; ++i) {
            std::this_thread::sleep_for(2ms);
            hopped = app.engine().worker_hops() >= 1;
        }
        CHECK(hopped);
        fb.pump_callback(in.data(), out.data(), in.size());
        float peak = 0.0f;
        for (float v : out) peak = std::max(peak, std::fabs(v));
        CHECK(peak > 0.2f);  // output arrived
        const TelemetrySnapshot t = app.telemetry();
        CHECK(t.frames == 3200 && t.sample_rate == 16000);
        CHECK(t.input_peak_dbfs > -13.0f);
    }

    // Parameters & presets.
    const std::uint64_t pushed_before = app.telemetry().params_pushed;
    app.set_param(ParamId::Gain, -6.0f);
    CHECK(near(app.params().get(ParamId::Gain), -6.0f) && app.telemetry().params_pushed == pushed_before + 1);
    CHECK(near(app.settings().gain_db, -6.0f));
    CHECK(app.load_preset("Deep"));
    CHECK(near(app.params().get(ParamId::PitchShift), -4.0f) && near(app.params().get(ParamId::Intonation), 0.9f));
    CHECK(app.current_preset() == "Deep" && !app.preset_modified());
    app.set_param(ParamId::PitchShift, -5.0f);
    CHECK(app.preset_modified());
    CHECK(!app.load_preset("Nope"));
    CHECK(app.save_preset_as("My voice"));
    CHECK(app.presets().size() == 6 && app.current_preset() == "My voice");
    CHECK(fs::is_regular_file(test_paths().user_preset_dir / "My_voice.json"));
    {   // Stage 03: "Save as..." writes voice parameters only by default.
        const std::string text = slurp(test_paths().user_preset_dir / "My_voice.json");
        CHECK(contains(text, "pitch_shift") && contains(text, "gain"));
        CHECK(!contains(text, "\"translate\"") && !contains(text, "wait_k") && !contains(text, "adaptive_policy"));
        CHECK(!contains(text, "src_lang") && !contains(text, "tgt_lang"));
    }
    CHECK(app.set_voice_b("identity_b") && app.params().voice_b == 1);
    CHECK(!app.set_voice_a("ghost"));
    app.reset_params();
    CHECK(near(app.params().get(ParamId::PitchShift), 0.0f) && app.params().voice_b == 1);

    // Translation: unavailable for a Mode A bundle; pair rules still apply.
    CHECK(!app.set_translate(true) && contains(app.translation_message(), "translator"));
    CHECK(!app.translate());
    CHECK(!app.set_language_pair("en", "en") && contains(app.translation_message(), "Mode A"));
    CHECK(app.set_language_pair("de", "en") && app.wait_k() == 4 && app.language_pair_label() == "de -> en");
    CHECK(app.set_language_pair("es", "en") && app.wait_k() == 3);
    app.swap_languages();
    CHECK(app.src_lang() == "en" && app.tgt_lang() == "es");
    app.set_wait_k(42);
    CHECK(app.wait_k() == 10 && app.settings().wait_k == 10);
    CHECK(app.language_pair_options().size() == 12);

    // Tray reflects the model and drives it.
    TraySpec spec = app.tray_spec();
    CHECK(spec.state == TrayState::Running);
    CHECK(find_item(spec.items, "engine") && find_item(spec.items, "engine")->checked);
    CHECK(find_item(spec.items, "voice:identity") && find_item(spec.items, "voice:identity")->checked);
    app.handle_tray_action(*action_for_item("mute"));
    CHECK(app.muted() && app.tray_spec().state == TrayState::Muted && app.status_summary().pill == "Running (muted)");
    app.handle_tray_action(*action_for_item("mute"));
    CHECK(!app.muted());
    app.handle_tray_action(*action_for_item("preset:Bright"));
    CHECK(near(app.params().get(ParamId::PitchShift), 4.0f));
    app.handle_tray_action(*action_for_item("lang:de -> en"));
    CHECK(app.src_lang() == "de" && app.tgt_lang() == "en");
    app.handle_tray_action(*action_for_item("voice:identity_b"));
    CHECK(app.voice_a() == "identity_b");
    app.handle_tray_action(*action_for_item("mode:s2ut"));
    CHECK(!app.translate());
    app.handle_tray_action(*action_for_item("show"));
    CHECK(app.take_show_panel_request() && !app.take_show_panel_request());
    app.handle_tray_action(*action_for_item("mini"));
    CHECK(app.take_mini_toggle_request());
    app.handle_tray_action(*action_for_item("quit"));
    CHECK(app.take_quit_request() && !app.take_quit_request());

    // The explicitly selected microphone disappears: the controller stops the
    // route and waits for the user; the model never substitutes a device.
    fb.capture = {cout_};
    app.tick(t0 + 1100ms);
    app.tick(t0 + 1200ms);
    CHECK(app.engine().status().state == StreamState::AwaitingUser);
    CHECK(!app.engine_on());
    CHECK(fb.open_calls == 1);
    CHECK(app.tray_spec().state == TrayState::Error);
    CHECK(contains(app.status_summary().detail, "device not found"));
    bool error_notice = false;
    for (const Notice& n : app.notices())
        if (n.level == Notice::Level::Error && contains(n.text, "Stream stopped")) error_notice = true;
    CHECK(error_notice);
    CHECK(!app.capture_rows().stale && app.capture_rows().rows.size() == 2);
    CHECK(app.capture_name() == "(device no longer listed)");

    // Telemetry history ticks once per second.
    app.tick(t0 + 2300ms);
    app.tick(t0 + 3400ms);
    CHECK(app.underrun_history().size() >= 1);

    app.stop_engine();
    CHECK(app.engine().status().state == StreamState::Idle && app.status_summary().pill == "Idle");

    // Probe on demand only, through the registry.
    const auto info = app.probe(ExplicitEndpoint{cin.key, app.snapshot().generation}, EndpointDirection::Playback);
    CHECK(is_ok(info) && std::get<EndpointInfo>(info).native_formats.size() == 1);
    const auto missing = app.probe(ExplicitEndpoint{mic.key, 0}, EndpointDirection::Capture);
    CHECK(!is_ok(missing));

    std::string err;
    CHECK(app.save_settings(&err));
    const Settings saved = Settings::load(test_paths().settings_file());
    CHECK(saved.last_preset == "Bright" && saved.voice_a == "identity_b" && saved.src_lang == "de");

    // Studio hooks through the model.
    app.studio().draft().voice_name = "me";
    CHECK(app.studio_write_job(err) && err.empty());
    CHECK(app.studio().draft().base_bundle == "identity");
    CHECK(app.studio_save_draft());
}

void test_app_model_restores_policy() {
    // A remembered explicit device that is gone: no substitution, user is asked.
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Other Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    Settings s;
    s.capture_policy  = "name:USB Mic";
    s.playback_policy = "default";
    s.start_engine_on_launch = true;
    AppModel app(fb, test_paths(), s);
    app.initialise();
    CHECK(!app.engine_on());
    CHECK(app.section() == Section::Devices);
    CHECK(std::holds_alternative<FollowSystemDefault>(app.capture_selection()));
    bool asked = false;
    for (const Notice& n : app.notices()) asked = asked || contains(n.text, "Choose it again");
    CHECK(asked);

    // BUG-020: the placeholder selection is never opened. Every Start path
    // (Live / Devices / tray / Mini / hotkey) refuses and names the device.
    CHECK(app.unresolved_capture() && *app.unresolved_capture() == "USB Mic" && !app.unresolved_playback());
    CHECK(!app.start_engine() && fb.open_calls == 0 && !app.engine_on());
    CHECK(contains(app.engine_message(), "USB Mic") && contains(app.engine_message(), "never substitutes"));
    app.handle_tray_action(TrayAction{TrayAction::Type::ToggleEngine, {}});
    CHECK(fb.open_calls == 0 && !app.engine_on());
    CHECK(app.tray_spec().state == TrayState::Error && contains(app.tray_spec().tooltip, "USB Mic"));
    CHECK(contains(app.capture_name(), "USB Mic") && contains(app.capture_name(), "not listed"));
    CHECK(contains(app.status_summary().detail, "USB Mic"));
    // Apply keeps the remembered policy instead of saving the substitution.
    CHECK(!app.apply_devices() && fb.open_calls == 0);
    CHECK(app.settings().capture_policy == "name:USB Mic");
    // The device comes back: a refresh restores the explicit endpoint.
    const auto usb = fb.make_ep(EndpointDirection::Capture, 3, "USB Mic");
    fb.capture.push_back(usb);
    app.refresh_devices();
    CHECK(!app.device_unresolved());
    CHECK(std::holds_alternative<ExplicitEndpoint>(app.capture_selection()) &&
          std::get<ExplicitEndpoint>(app.capture_selection()).key == usb.key);
    CHECK(app.start_engine() && fb.open_calls == 1 && !fb.last_capture_was_default);
    app.stop_engine();

    // Ambiguous name (two identical cables): refused with that reason until
    // the user picks a row; "Follow system default" chosen explicitly clears it.
    {
        FakeAudioBackend fa;
        fa.capture  = {fa.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
        fa.playback = {fa.make_ep(EndpointDirection::Playback, 10, "Speakers", true),
                       fa.make_ep(EndpointDirection::Playback, 11, "CABLE Input"),
                       fa.make_ep(EndpointDirection::Playback, 12, "CABLE Input")};
        Settings sa;
        sa.playback_policy = "name:CABLE Input";
        sa.start_engine_on_launch = true;
        AppModel amb(fa, test_paths(), sa);
        amb.initialise();
        CHECK(!amb.engine_on() && fa.open_calls == 0 && amb.unresolved_playback());
        CHECK(!amb.start_engine() && contains(amb.engine_message(), "more than one"));
        amb.select_playback_row(0);  // "Follow system default", explicitly
        CHECK(!amb.device_unresolved() && std::holds_alternative<FollowSystemDefault>(amb.playback_selection()));
        CHECK(amb.start_engine() && fa.open_calls == 1 && fa.last_playback_was_default);
        amb.stop_engine();
    }

    // Resolvable policy + start on launch -> running with the named device.
    FakeAudioBackend fb2;
    const auto mic = fb2.make_ep(EndpointDirection::Capture, 1, "USB Mic");
    fb2.capture  = {fb2.make_ep(EndpointDirection::Capture, 2, "Other Mic", true), mic};
    fb2.playback = {fb2.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    AppModel app2(fb2, test_paths(), s);
    app2.initialise();
    CHECK(app2.engine_on() && app2.engine().status().state == StreamState::Running);
    CHECK(!fb2.last_capture_was_default && fb2.last_playback_was_default);
    CHECK(app2.capture_name() == "USB Mic");

    // Null backend: offline mode is reported, the engine refuses to start.
    FakeAudioBackend fb3;
    fb3.backend_status = BackendStatus{BackendState::NullOnly, "null", 0};
    AppModel app3(fb3, test_paths(), Settings{});
    app3.initialise();
    CHECK(!app3.backend_usable() && !app3.start_engine() && contains(app3.engine_message(), "null backend"));
    CHECK(contains(app3.status_summary().detail, "offline"));
}

// Stage 02: nothing in frontend/ produces FollowSystemDefault except the user
// choosing it or a stored "default" policy.
void test_device_policy_rules() {
    // Malformed stored policies are unresolved explicit choices, never "default".
    for (const char* bad : {"Other Mic", "Name:Other Mic", "name:", "defaults"}) {
        FakeAudioBackend fb;
        fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Other Mic", true)};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
        Settings s;
        s.capture_policy         = bad;
        s.start_engine_on_launch = true;
        AppModel app(fb, test_paths(), s);
        app.initialise();
        CHECK(app.unresolved_capture().has_value());
        CHECK(!app.engine_on() && fb.open_calls == 0);
        CHECK(!app.start_engine() && fb.open_calls == 0);
        CHECK(app.tray_spec().state == TrayState::Error);
        CHECK(!app.apply_devices() && app.settings().capture_policy == bad);
    }
    {   // A prefix-less policy is matched later only as that exact device name.
        FakeAudioBackend fb;
        fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Other Mic", true)};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
        Settings s;
        s.capture_policy = "USB Mic";
        AppModel app(fb, test_paths(), s);
        app.initialise();
        CHECK(app.unresolved_capture() && *app.unresolved_capture() == "USB Mic");
        const auto usb = fb.make_ep(EndpointDirection::Capture, 2, "USB Mic");
        fb.capture.push_back(usb);
        app.refresh_devices();
        CHECK(!app.device_unresolved() && std::holds_alternative<ExplicitEndpoint>(app.capture_selection()));
        CHECK(app.start_engine() && !fb.last_capture_was_default);
        app.stop_engine();
    }
    {   // An out-of-range row changes nothing and keeps the BUG-020 block.
        FakeAudioBackend fb;
        fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Other Mic", true)};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
        Settings s;
        s.capture_policy = "name:USB Mic";
        AppModel app(fb, test_paths(), s);
        app.initialise();
        const std::size_t rows = app.capture_rows().rows.size();
        CHECK(!app.select_capture_row(rows) && !app.select_playback_row(99));
        CHECK(app.unresolved_capture() && *app.unresolved_capture() == "USB Mic");
        CHECK(!app.start_engine() && fb.open_calls == 0);
        bool error_notice = false;
        for (const Notice& n : app.notices())
            error_notice = error_notice || (n.level == Notice::Level::Error && contains(n.text, "not in the device list"));
        CHECK(error_notice);
        CHECK(app.select_capture_row(1) && !app.device_unresolved());
    }
    {   // A chosen endpoint that vanished before Apply is never saved by a
        // placeholder name such as "(device no longer listed)".
        FakeAudioBackend fb;
        const auto mic = fb.make_ep(EndpointDirection::Capture, 1, "USB Mic");
        fb.capture  = {fb.make_ep(EndpointDirection::Capture, 2, "Other Mic", true), mic};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
        Settings s;
        s.capture_policy = "name:Other Mic";
        AppModel app(fb, test_paths(), s);
        app.initialise();
        CHECK(app.select_capture_row(2));  // USB Mic
        fb.capture = {fb.capture.front()};
        app.refresh_devices();
        CHECK(app.capture_name() == "(device no longer listed)");
        CHECK(!app.apply_devices() && fb.open_calls == 0);
        CHECK(app.settings().capture_policy == "name:Other Mic");
        CHECK(app.settings().playback_policy == "default");
        CHECK(!contains(slurp(test_paths().settings_file()), "no longer listed"));
    }
}

// Stage 03: preset save/restore rules, Wait-k persistence, translator caps.
void test_preset_and_translation_rules() {
    AppPaths paths = test_paths();
    paths.config_dir      = g_tmp / "config_presets";
    paths.user_preset_dir = paths.config_dir / "presets";
    paths.studio_dir      = paths.config_dir / "studio";
    std::error_code ec;
    fs::create_directories(paths.user_preset_dir, ec);
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};

    {   // Built-in names are refused at save time (any case); nothing is written.
        AppModel app(fb, paths, Settings{});
        app.initialise();
        CHECK(app.preset_name_problem("Deep") && app.preset_name_problem("deep") && app.preset_name_problem(""));
        CHECK(!app.save_preset_as("deep") && !fs::exists(paths.user_preset_dir / "deep.json"));
        CHECK(!app.preset_name_problem("Deep 2"));
        // Two names that share one file name: the second is refused.
        CHECK(app.save_preset_as("My voice") && app.save_preset_as("My voice"));  // re-saving the same name is fine
        CHECK(app.preset_name_problem("My_voice") && !app.save_preset_as("My_voice"));
        const PresetLoadResult mine = load_preset_file(paths.user_preset_dir / "My_voice.json");
        CHECK(mine.ok() && mine.preset->name == "My voice");
    }
    {   // A hand-written user file named like a built-in is reported and hidden.
        std::ofstream(paths.user_preset_dir / "mine_deep.json")
            << R"({"schema":1,"name":"DEEP","params":{"pitch_shift":7}})";
        AppModel app(fb, paths, Settings{});
        app.initialise();
        int deep = 0;
        for (const std::string& n : app.preset_names()) deep += contains(n, "eep") ? 1 : 0;
        CHECK(deep == 1);
        bool warned = false;
        for (const std::string& w : app.preset_warnings()) warned = warned || (contains(w, "mine_deep.json") && contains(w, "built-in"));
        CHECK(warned);
        CHECK(app.load_preset("Deep") && near(app.params().get(ParamId::PitchShift), -4.0f));
        fs::remove(paths.user_preset_dir / "mine_deep.json", ec);
    }
    {   // The last preset is re-applied at start, not only its label.
        Settings s;
        s.last_preset = "Deep";
        s.gain_db     = -2.0f;
        AppModel app(fb, paths, s);
        app.initialise();
        CHECK(app.current_preset() == "Deep" && !app.preset_modified());  // Deep carries no gain
        CHECK(near(app.params().get(ParamId::PitchShift), -4.0f) && near(app.params().get(ParamId::Intonation), 0.9f));
        CHECK(near(app.params().get(ParamId::Gain), -2.0f));
        Settings gone;
        gone.last_preset = "Vanished";
        AppModel app2(fb, paths, gone);
        app2.initialise();
        CHECK(app2.current_preset().empty() && app2.settings().last_preset.empty());
    }
    {   // A saved Wait-k survives a restart for every pair (not only en -> es).
        Settings s;
        s.src_lang = "de";
        s.tgt_lang = "en";
        s.wait_k   = 7;  // de -> en default is 4
        AppModel app(fb, paths, s);
        app.initialise();
        CHECK(app.src_lang() == "de" && app.tgt_lang() == "en" && app.wait_k() == 7 && app.settings().wait_k == 7);
        Settings same;
        same.wait_k = 5;  // the en -> es default pair
        AppModel app2(fb, paths, same);
        app2.initialise();
        CHECK(app2.wait_k() == 5);
        // A user-chosen pair change still applies the pair default.
        CHECK(app.set_language_pair("es", "en") && app.wait_k() == 3);
    }
    {   // Translation state goes into a preset only on request; a preset's own
        // wait_k wins over the pair default, a pair-only preset gets the default.
        AppModel app(fb, paths, Settings{});
        app.initialise();
        CHECK(app.set_language_pair("es", "en"));
        app.set_wait_k(6);
        CHECK(app.save_preset_as("Trip", true));
        const auto trip = load_preset_file(paths.user_preset_dir / "Trip.json");
        CHECK(trip.ok() && trip.preset->has(ParamId::WaitK) && trip.preset->has(ParamId::Translate));
        CHECK(trip.ok() && trip.preset->src_lang == std::optional<std::string>("es"));
        CHECK(app.set_language_pair("en", "fr") && app.wait_k() == 3);
        CHECK(app.load_preset("Trip"));
        CHECK(app.src_lang() == "es" && app.tgt_lang() == "en" && app.wait_k() == 6 && app.settings().wait_k == 6);
        std::ofstream(paths.user_preset_dir / "pair_only.json")
            << R"({"schema":1,"name":"PairOnly","params":{},"src_lang":"de","tgt_lang":"en"})";
        app.reload_presets();
        CHECK(app.load_preset("PairOnly") && app.src_lang() == "de" && app.wait_k() == 4);
        // A voice-only preset leaves Wait-k and the pair alone.
        app.set_wait_k(9);
        CHECK(app.load_preset("Deep") && app.wait_k() == 9 && app.src_lang() == "de");
    }
    {   // Translator caps: unknown at start and after a bundle change; a bundle
        // without readiness output switches adaptive_policy off.
        AppModel app(fb, paths, Settings{});
        app.initialise();
        CHECK(app.set_active_bundle("s2ut") && app.translation().translator_available());
        CHECK(!app.translation().translator_caps().known && !app.translation().adaptive_policy_available());
        app.set_translator_caps(TranslatorCaps{true, 80.0, true});
        CHECK(app.translation().adaptive_policy_available());
        app.set_param(ParamId::AdaptivePolicy, 1.0f);
        CHECK(app.params().enabled(ParamId::AdaptivePolicy));
        app.set_translator_caps(TranslatorCaps{true, 80.0, false});
        CHECK(!app.params().enabled(ParamId::AdaptivePolicy) && !app.translation().adaptive_policy_available());
        app.set_translator_caps(TranslatorCaps{true, 80.0, true});
        CHECK(app.set_active_bundle("identity") && !app.translation().translator_caps().known);
    }
}

// Stage 03: WAV analysis runs off the UI thread; AppModel::tick applies the
// result (in any section) and saves the draft.
void test_studio_analysis_drained_by_tick() {
    AppPaths paths = test_paths();
    paths.config_dir      = g_tmp / "config_async";
    paths.user_preset_dir = paths.config_dir / "presets";
    paths.studio_dir      = paths.config_dir / "studio";
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    const auto t0 = AppModel::Clock::now();
    AppModel app(fb, paths, Settings{});
    app.initialise(t0);
    CHECK(app.section() != Section::Studio);
    app.studio().draft().voice_name = "async";
    const fs::path wav = g_tmp / "async_s01_0001.wav";
    write_wav(wav, sine(2.0, 16000, 0.3f), 16000, 16);
    app.studio().request_take_analysis(TakeKind::Take, wav, "s01");
    CHECK(app.studio().analysis_pending() && app.studio().draft().takes.empty());
    bool done = false;
    for (int i = 1; i <= 1000 && !done; ++i) {
        std::this_thread::sleep_for(2ms);
        app.tick(t0 + std::chrono::milliseconds(i));
        done = !app.studio().analysis_pending();
    }
    CHECK(done && app.studio().draft().takes.size() == 1);
    CHECK(fs::is_regular_file(paths.studio_dir / "async" / "draft.json"));
}

std::vector<std::string> dir_names(const fs::path& root) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(root, ec))
        if (e.is_directory(ec)) out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

void test_studio_drafts() {
    // BUG-021: the draft directory is the voice name; it must be validated,
    // unique per project and never written over another project.
    AppPaths p = test_paths();
    p.studio_dir = g_tmp / "studio21" / "root";
    const fs::path outside = g_tmp / "studio21";
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};

    {
        AppModel app(fb, p, Settings{});
        app.initialise();
        CHECK(app.studio_project().empty());

        // Simulated edits a, al, ale, alex (each saved) produce one directory.
        app.studio().draft().display_name = "Alex";
        for (const char* name : {"a", "al", "ale", "alex"}) {
            app.studio().draft().voice_name = name;
            CHECK(app.studio_save_draft());
        }
        CHECK(dir_names(p.studio_dir) == std::vector<std::string>{"alex"});
        CHECK(app.studio_project() == "alex");

        // Unsafe names never build a path, inside or outside the root.
        const std::vector<std::string> before = dir_names(outside);
        for (const std::string& bad : {std::string("../x"), std::string("..\\..\\x"), (outside / "abs").string(),
                                       std::string("con"), std::string("Alex"), std::string("")}) {
            app.studio().draft().voice_name = bad;
            CHECK(!app.studio_save_draft());
            std::string err;
            CHECK(!app.studio_write_job(err) && !err.empty());
        }
        CHECK(dir_names(outside) == before && dir_names(p.studio_dir) == std::vector<std::string>{"alex"});
        CHECK(!fs::exists(outside / "x") && !fs::exists(outside / "abs"));
        VoiceStudioModel raw;
        raw.draft().voice_name = "../x";
        std::string err;
        CHECK(!raw.save_draft(p.studio_dir, err) && !err.empty());
        CHECK(raw.project_dir(p.studio_dir).empty());  // an unsafe name never maps to a folder (BUG-021)
        CHECK(!is_safe_project_name("com1") && !is_safe_project_name("nul") && is_safe_project_name("com10"));

        // Reopening the project by name reloads its draft.
        app.studio().draft() = StudioDraft{};
        app.studio().draft().voice_name = "alex";
        CHECK(app.studio_open_project("alex") && app.studio().draft().display_name == "Alex");
    }
    {   // A newer "zed" project on disk; "alex" is older.
        VoiceStudioModel z;
        z.draft().voice_name   = "zed";
        z.draft().display_name = "Zed";
        std::string err;
        CHECK(z.save_draft(p.studio_dir, err));
        std::error_code ec;
        fs::last_write_time(p.studio_dir / "alex" / "draft.json", fs::file_time_type::clock::now() - std::chrono::hours(1), ec);
        CHECK(!ec);
    }
    {
        AppModel app(fb, p, Settings{});
        app.initialise();
        // Startup restores the newer of two projects (not "exactly one").
        CHECK(app.studio_project() == "zed" && app.studio().draft().display_name == "Zed");

        // Typing an existing project's name into a fresh wizard does not touch it.
        const std::string alex_before = slurp(p.studio_dir / "alex" / "draft.json");
        app.studio().draft() = StudioDraft{};
        app.studio().draft().voice_name = "alex";
        CHECK(app.studio_name_conflict().has_value());
        CHECK(!app.studio_save_draft());
        std::string err;
        CHECK(!app.studio_write_job(err) && contains(err, "already exists"));
        CHECK(slurp(p.studio_dir / "alex" / "draft.json") == alex_before);
        CHECK(fs::is_regular_file(p.studio_dir / "zed" / "draft.json"));
        CHECK(!fs::exists(p.studio_dir / "alex" / "job.json"));

        // "Open existing project" loads it instead.
        CHECK(app.studio_open_project("alex") && app.studio_project() == "alex");
        CHECK(app.studio().draft().display_name == "Alex" && !app.studio_name_conflict());

        // A project with job files keeps them (and its draft) when renamed.
        CHECK(app.studio_write_job(err) && fs::is_regular_file(p.studio_dir / "alex" / "job.json"));
        app.studio().draft().voice_name = "alex2";
        CHECK(app.studio_save_draft() && app.studio_project() == "alex2");
        CHECK(fs::is_regular_file(p.studio_dir / "alex" / "draft.json"));
        CHECK(fs::is_regular_file(p.studio_dir / "alex" / "job.json"));
        CHECK(dir_names(p.studio_dir) == (std::vector<std::string>{"alex", "alex2", "zed"}));
    }
}

// --------------------------------------------------------------------------
// BUG-022: wrong-typed JSON fields must keep defaults, never throw
// --------------------------------------------------------------------------
void test_json_wrong_types() {
    // Typed readers: missing or wrong-typed keys keep the default.
    const nlohmann::json j = nlohmann::json::parse(
        R"({"s":"x","n":1.5,"i":3,"b":true,"nil":null})", nullptr, false);
    CHECK(read_string(j, "s", "d") == "x" && read_string(j, "missing", "d") == "d");
    CHECK(read_string(j, "nil", "d") == "d" && read_string(j, "n", "d") == "d");
    CHECK(near(static_cast<float>(read_number(j, "n", 0.0)), 1.5f));
    CHECK(read_number(j, "i", 0.0) == 3.0 && read_number(j, "s", 2.0) == 2.0 && read_number(j, "nil", 2.0) == 2.0);
    CHECK(read_int(j, "i", 0) == 3 && read_int(j, "n", 0) == 0 && read_int(j, "s", 7) == 7);
    CHECK(read_bool(j, "b", false) && !read_bool(j, "nil", false) && read_bool(j, "i", true));
    std::vector<std::string> rw;
    CHECK(read_string(j, "nil", "d", &rw) == "d" && rw.size() == 1);
    CHECK(contains(rw.front(), "wrong type"));

    // status.json written by a training script that does not know a value yet.
    VoiceStudioModel m;
    m.draft().voice_name = "bug022";
    const fs::path studio = g_tmp / "studio-b22";
    fs::create_directories(studio / "bug022");
    struct BadStatus { const char* text; const char* stage; float progress; double eta; };
    const BadStatus bad[] = {
        {R"({"stage":"finetune","progress":0.4,"eta_s":null})", "finetune", 0.4f, -1.0},
        {R"({"stage":"finetune","progress":"40%","eta_s":120})", "finetune", 0.0f, 120.0},
        {R"({"stage":null,"progress":[1],"eta_s":"soon","message":7})", "unknown", 0.0f, -1.0},
    };
    for (const BadStatus& b : bad) {
        { std::ofstream f(studio / "bug022" / "status.json", std::ios::trunc); f << b.text; }
        const auto s = m.poll_status(studio);  // must not throw (BUG-022)
        CHECK(s.has_value());
        if (s) {
            CHECK(s->stage == b.stage && near(s->progress, b.progress) && s->eta_s == b.eta && s->message.empty());
        }
    }

    // draft.json: a wrong type keeps the default instead of terminating (BUG-022).
    std::string err;
    CHECK(!m.load_draft_json(R"({"schema":"1"})", &err));  // schema must be the integer 1
    CHECK(m.load_draft_json(R"({"schema":1,"step":"2","schema_extra":0})"));
    CHECK(m.step() == StudioStep::NameConsent);
    CHECK(m.load_draft_json(R"({"schema":1,"voice_name":7,"display_name":false,"is_own_voice":"yes",)"
                            R"("consent":"yes","consent_note":[],"base_bundle":{},"target_bundle":null})"));
    CHECK(m.draft().voice_name.empty() && m.draft().display_name.empty());
    CHECK(m.draft().is_own_voice && !m.draft().consent && m.draft().consent_note.empty());
    CHECK(m.load_draft_json(R"({"schema":1,"takes":[{"path":1,"ok":"yes","seconds":"3","verdict":9,"clipped_samples":null}]})"));
    CHECK(m.draft().takes.size() == 1);
    if (!m.draft().takes.empty()) {
        const Take& t = m.draft().takes.front();
        CHECK(t.path.empty() && t.session.empty() && !t.stats.ok && t.stats.seconds == 0.0);
        CHECK(t.stats.clipped_samples == 0 && t.verdict == TakeVerdict::Unreadable);
    }

    // The same table for Settings and the strict loaders (they reject, not throw).
    std::vector<std::string> sw;
    const Settings ns = Settings::from_json(
        R"({"schema":null,"start_minimized":1,"theme":null,"ui_scale":"2","wait_k":null,)"
        R"("translate":"yes","gain_db":null,"voice_a":[]})", &sw);
    CHECK(!ns.start_minimized && ns.theme == Theme::Dark && near(ns.ui_scale, 1.0f));
    CHECK(ns.wait_k == 3 && !ns.translate && near(ns.gain_db, 0.0f) && ns.voice_a.empty());
    CHECK(!sw.empty());
    CHECK(!parse_voices_json(R"({"schema":null,"voices":[]})").ok());
    CHECK(!parse_langs_json(R"({"schema":1,"langs":[{"code":"en","id":null}]})").ok());
    CHECK(!parse_catalogue_index(R"({"schema":1,"entries":[{"id":null,"display":"A"}]})").ok());
}

// --------------------------------------------------------------------------
// BUG-022 regression table: every loader, every field, every JSON type. Each
// leaf and container of a valid document is replaced in turn by null, a
// string, a boolean, a float, an object and an array; the loader may reject
// or keep the default, but it must never throw.
// --------------------------------------------------------------------------
void collect_pointers(const nlohmann::json& j, const nlohmann::json::json_pointer& at,
                      std::vector<nlohmann::json::json_pointer>& out) {
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            const auto p = at / it.key();
            out.push_back(p);
            collect_pointers(*it, p, out);
        }
    } else if (j.is_array()) {
        for (std::size_t i = 0; i < j.size(); ++i) {
            const auto p = at / i;
            out.push_back(p);
            collect_pointers(j[i], p, out);
        }
    }
}

/// Runs `load` on `valid` with each node replaced by each hostile value, and on
/// hostile roots. Returns the number of loader calls that threw (reported).
template <typename Load>
int hostile_table(const char* name, const nlohmann::json& valid, Load load, std::size_t* calls = nullptr,
                  const std::vector<nlohmann::json>* values = nullptr) {
    const std::vector<nlohmann::json> types = {nullptr, "x", true, 1.5, nlohmann::json::object(), nlohmann::json::array()};
    const std::vector<nlohmann::json>& hostile = values ? *values : types;
    std::vector<nlohmann::json::json_pointer> paths;
    collect_pointers(valid, nlohmann::json::json_pointer(), paths);
    int threw = 0;
    std::size_t n = 0;
    auto run = [&](const std::string& text, const std::string& where) {
        ++n;
        try {
            load(text);
        } catch (const std::exception& e) {
            ++threw;
            std::printf("[%s] threw at %s: %s\n", name, where.c_str(), e.what());
        } catch (...) {
            ++threw;
            std::printf("[%s] threw at %s\n", name, where.c_str());
        }
    };
    for (const auto& p : paths)
        for (const nlohmann::json& h : hostile) {
            nlohmann::json m = valid;
            m[p] = h;
            run(m.dump(), p.to_string() + " := " + h.dump());
        }
    if (!values) {
        for (const nlohmann::json& h : hostile) run(h.dump(), "root := " + h.dump());
        run("", "empty text");
        run("{\"unterminated\":", "truncated text");
    }
    if (calls) *calls = n;
    return threw;
}

std::size_t hostile_documents(const std::vector<nlohmann::json>* values) {
    const fs::path assets = g_repo / "frontend" / "assets";
    const fs::path fixtures = g_repo / "tests" / "data" / "fixtures";
    auto parse = [](const std::string& text) { return nlohmann::json::parse(text); };
    std::size_t calls = 0, total = 0;

    // voices.json, langs.json, catalogue index.json
    CHECK(hostile_table("voices.json", parse(slurp(fixtures / "identity" / "voices.json")),
                        [](const std::string& t) { static_cast<void>(parse_voices_json(t)); }, &calls, values) == 0);
    total += calls;
    CHECK(hostile_table("langs.json", parse(slurp(assets / "langs.default.json")),
                        [](const std::string& t) { static_cast<void>(parse_langs_json(t)); }, &calls, values) == 0);
    total += calls;
    CHECK(hostile_table("index.json", parse(slurp(assets / "catalogue" / "index.json")),
                        [](const std::string& t) { static_cast<void>(parse_catalogue_index(t)); }, &calls, values) == 0);
    total += calls;

    // settings.json, with an unknown key so the extras path is covered too
    nlohmann::json settings = parse(Settings{}.to_json());
    settings["mini_position"] = nlohmann::json::array({10, 20});
    settings["future_key"]    = nlohmann::json::object({{"a", 1}});
    CHECK(hostile_table("settings.json", settings,
                        [](const std::string& t) { static_cast<void>(Settings::from_json(t).to_json()); }, &calls, values) == 0);
    total += calls;

    // draft.json as the model writes it (three take lists), and status.json
    const fs::path studio = g_tmp / "studio-b22-table";
    std::error_code ec;
    fs::create_directories(studio, ec);
    const fs::path wav = studio / "t.wav";
    write_wav(wav, sine(1.0, 16000, 0.3f), 16000, 16);
    VoiceStudioModel src;
    src.draft().voice_name   = "b22table";
    src.draft().display_name = "Table";
    src.add_take(wav, "s01");
    src.add_room_tone(wav);
    src.add_live_mic(wav);
    std::string err;
    CHECK(src.save_draft(studio, err));
    CHECK(hostile_table("draft.json", parse(slurp(src.project_dir(studio) / "draft.json")),
                        [](const std::string& t) { VoiceStudioModel m; static_cast<void>(m.load_draft_json(t)); }, &calls, values) == 0);
    total += calls;

    VoiceStudioModel poll;
    poll.draft().voice_name = "b22table";
    const nlohmann::json status = parse(R"({"stage":"finetune","progress":0.4,"eta_s":120,"message":"step 3"})");
    CHECK(hostile_table("status.json", status, [&](const std::string& t) {
        { std::ofstream f(poll.project_dir(studio) / "status.json", std::ios::binary | std::ios::trunc); f << t; }
        static_cast<void>(poll.poll_status(studio));
    }, &calls, values) == 0);
    total += calls;
    return total;
}

void test_json_every_field_every_type() {
    // Every node replaced by each JSON type, then by each hostile number (text-expressible rows).
    const std::size_t types = hostile_documents(nullptr);
    CHECK(types > 1000);  // the table really ran (hundreds of nodes per document)
    const std::vector<nlohmann::json> numbers = {
        -1, 1e300, -1e300, 1e-300, 0.5, -0.5, 3.0, 18446744073709551616.0,  // 2^64 as a float
        static_cast<std::uint64_t>(18446744073709551615ULL), static_cast<std::uint64_t>(9223372036854775808ULL),
        static_cast<std::int64_t>(-9223372036854775807LL - 1), static_cast<std::uint64_t>(4294967297ULL), 2147483648LL};
    const std::size_t nums = hostile_documents(&numbers);
    CHECK(nums > 1000);
}

// --------------------------------------------------------------------------
// Hostile numbers land as the default or a rejected record, never as a cast
// result (UB for out-of-range doubles, a wrong wrapped value for wide integers).
// --------------------------------------------------------------------------
void test_json_hostile_numbers() {
    using nlohmann::json;
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();

    // The checked readers, with values the parser cannot produce from text (NaN, inf).
    CHECK(!json_to_integral<int>(json(nan)) && !json_to_integral<int>(json(inf)) && !json_to_integral<int>(json(-inf)));
    CHECK(!json_to_float(json(nan)) && !json_to_float(json(inf)) && !json_to_float(json(-inf)));
    CHECK(!json_to_integral<std::uint32_t>(json(-1)) && !json_to_integral<std::uint32_t>(json(4294967296ULL)));
    CHECK(json_to_integral<std::uint32_t>(json(4294967295ULL)) == 4294967295u);
    CHECK(!json_to_integral<std::size_t>(json(-1)) && !json_to_integral<std::size_t>(json(-1e300)));
    CHECK(!json_to_integral<std::size_t>(json(1e300)) && !json_to_integral<std::size_t>(json(18446744073709551616.0)));
    CHECK(json_to_integral<std::size_t>(json(18446744073709551615ULL)) == static_cast<std::size_t>(18446744073709551615ULL));
    CHECK(!json_to_integral<int>(json(0.5)) && !json_to_integral<int>(json(-0.5)) && json_to_integral<int>(json(3.0)) == 3);
    CHECK(!json_to_integral<int>(json(2147483648LL)) && json_to_integral<int>(json(-2147483647LL - 1)) == -2147483647 - 1);
    CHECK(!json_to_integral<int>(json("3")) && !json_to_integral<int>(json(true)) && !json_to_integral<int>(json(nullptr)));
    CHECK(!json_to_integral<int>(json(5), 6, 9) && json_to_integral<int>(json(6), 6, 9) == 6 && !json_to_integral<int>(json(10), 6, 9));
    CHECK(!json_to_float(json(1e300)) && !json_to_float(json(-1e300)) && json_to_float(json(1e-300)) == 0.0f);
    CHECK(json_to_float(json(1.5)) == 1.5f && !json_to_float(json("1")) && !json_to_float(json(false)));
    const json doc = json::parse(R"({"n":-1,"big":1e300})");
    CHECK(read_integral<std::uint16_t>(doc, "n", 7) == 7 && read_integral<std::uint16_t>(doc, "big", 8) == 8);
    CHECK(read_integral<std::uint16_t>(doc, "missing", 9) == 9);
    CHECK(read_float(doc, "big", 2.0f) == 2.0f && read_float(doc, "n", 2.0f) == -1.0f);
    std::vector<std::string> warn;
    CHECK(read_integral<int>(doc, "big", 4, 0, 10, &warn) == 4 && warn.size() == 1);

    // Studio take: every integral and float statistic keeps its default.
    VoiceStudioModel m;
    CHECK(m.load_draft_json(R"({"schema":1,"step":4294967297,"takes":[{"path":"a.wav","ok":true,"sample_rate":4294967297,)"
                            R"("channels":65537,"bits":-1,"clipped_samples":-1,"non_finite_samples":1e300,)"
                            R"("peak_dbfs":1e300,"rms_dbfs":-1e300,"noise_floor_dbfs":1e39,"speech_dbfs":3.5e38,"snr_db":-1e39},)"
                            R"({"path":"b.wav","sample_rate":16000,"channels":2,"bits":24,"clipped_samples":5,"peak_dbfs":-3.5}]})"));
    CHECK(m.step() == StudioStep::NameConsent);  // 4294967297 wraps to 1 through a plain int cast
    CHECK(m.draft().takes.size() == 2);
    if (m.draft().takes.size() == 2) {
        const Take& t = m.draft().takes[0];
        CHECK(t.stats.sample_rate == 0 && t.stats.channels == 0 && t.stats.bits == 0);
        CHECK(t.stats.clipped_samples == 0 && t.stats.non_finite_samples == 0);
        CHECK(t.stats.peak_dbfs == -120.0f && t.stats.rms_dbfs == -120.0f && t.stats.noise_floor_dbfs == -120.0f);
        CHECK(t.stats.speech_dbfs == -120.0f && t.stats.snr_db == 0.0f);
        const Take& g = m.draft().takes[1];  // valid values still load
        CHECK(g.stats.sample_rate == 16000 && g.stats.channels == 2 && g.stats.bits == 24);
        CHECK(g.stats.clipped_samples == 5 && g.stats.peak_dbfs == -3.5f);
    }
    CHECK(m.load_draft_json(R"({"schema":1,"takes":[{"sample_rate":16000.5,"channels":2.0}]})"));
    if (!m.draft().takes.empty()) {  // fractional rejected, an integral float accepted
        CHECK(m.draft().takes[0].stats.sample_rate == 0 && m.draft().takes[0].stats.channels == 2);
    }

    // status.json progress beyond float range is the default, not an overflow cast.
    const fs::path studio = g_tmp / "studio-numbers";
    fs::create_directories(studio / "numbers");
    VoiceStudioModel poll;
    poll.draft().voice_name = "numbers";
    { std::ofstream f(studio / "numbers" / "status.json", std::ios::trunc); f << R"({"stage":"x","progress":1e300,"eta_s":1e300})"; }
    const auto st = poll.poll_status(studio);
    CHECK(st.has_value() && st->progress == 0.0f);

    // Settings: out-of-range numbers keep the defaults (4294967299 wraps to 3 through get<int>).
    std::vector<std::string> sw;
    const Settings d{};
    const Settings s = Settings::from_json(R"({"window_w":1099511627776,"window_h":-4294967296,"wait_k":4294967299,)"
                                           R"("ui_scale":1e300,"gain_db":-1e300,"mini_position":[4294967296,0]})", &sw);
    CHECK(s.window_w == d.window_w && s.window_h == d.window_h && s.wait_k == d.wait_k);
    CHECK(near(s.ui_scale, 1.0f) && s.gain_db == 0.0f && !s.mini_has_position);
    CHECK(sw.size() >= 5);
    CHECK(!Settings::from_json(R"({"mini_x":4294967297,"mini_y":5})").mini_has_position);
    const Settings ok = Settings::from_json(R"({"window_w":900,"ui_scale":1.5,"mini_position":[-20,30]})");
    CHECK(ok.window_w == 900 && near(ok.ui_scale, 1.5f) && ok.mini_has_position && ok.mini_x == -20 && ok.mini_y == 30);

    // voices.json: a number beyond float range is a rejected record.
    auto voices = [](const std::string& emb, const std::string& mean) {
        return std::string(R"({"schema":1,"voices":[{"name":"lj","display":"LJ","embedding":[0.5,)") + emb +
               R"(],"logf0_mean":)" + mean + R"(,"logf0_std":0.2,"energy_mean":-3.0,"energy_std":1.1,"license":"PD","source":"S"}]})";
    };
    CHECK(parse_voices_json(voices("-0.5", "5.3")).ok());
    CHECK(!parse_voices_json(voices("1e300", "5.3")).ok());
    CHECK(!parse_voices_json(voices("-0.5", "-1e300")).ok());
    CHECK(!parse_voices_json(voices("-0.5", "3.5e38")).ok());
}

// --------------------------------------------------------------------------
// BUG-023: the newest parameter snapshot must always reach the worker
// --------------------------------------------------------------------------
void test_param_queue_latest_wins() {
    FakeAudioBackend fb;
    const auto mic = fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true);
    const auto cin = fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input", true);
    fb.capture  = {mic};
    fb.playback = {cin};

    AppModel app(fb, test_paths(), Settings{});
    app.initialise();
    CHECK(!app.engine_on());

    // Engine stopped: nothing drains the channel. Push far past the queue
    // capacity (64), then make the final gain the model's state too.
    VoiceParams snap = app.params();
    for (int i = 1; i <= 200; ++i) {
        snap.set(ParamId::Gain, -0.2f * static_cast<float>(i));  // 0 dB .. -40 dB
        static_cast<void>(app.engine().push_params(snap));
    }
    app.set_param(ParamId::Gain, -40.0f);  // the newest snapshot wins
    CHECK(app.apply_devices());

    bool caught_up = false;
    for (int i = 0; i < 500 && !caught_up; ++i) {
        app.tick();
        std::this_thread::sleep_for(2ms);
        const TelemetrySnapshot t = app.telemetry();
        caught_up = t.params_applied == t.params_pushed && t.params_pushed > 200;
    }
    CHECK(caught_up);  // the parked newest snapshot was flushed and applied

    // The audio level must match the newest gain, not the stale queued one
    // (the 64th push held about -12.8 dB; 0.25 * 10^(-40/20) is about 0.0025).
    std::vector<float> in(1600, 0.25f), out(1600, 0.0f);
    for (int i = 0; i < 8; ++i) {
        fb.pump_callback(in.data(), out.data(), in.size());
        std::this_thread::sleep_for(25ms);  // worker hops + gain ramp settle
    }
    float peak = 0.0f;
    for (float v : out) peak = std::max(peak, std::fabs(v));
    CHECK(peak < 0.02f);
    const TelemetrySnapshot t = app.telemetry();
    CHECK(t.params_applied == t.params_pushed);
}

// --------------------------------------------------------------------------
// BUG-024: the queued-latency figure rebaselines after queue discards
// --------------------------------------------------------------------------
void test_queued_latency_rebaselined() {
    FakeAudioBackend fb;
    const auto mic = fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true);
    const auto cin = fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input", true);
    fb.capture  = {mic};
    fb.playback = {cin};

    AppModel app(fb, test_paths(), Settings{});
    app.initialise();
    CHECK(app.apply_devices());

    std::vector<float> in(1600, 0.25f), out(1600, 0.0f);
    auto wait_hop = [&app](std::uint64_t at_least) {
        for (int i = 0; i < 400 && app.engine().worker_hops() < at_least; ++i) std::this_thread::sleep_for(2ms);
    };
    fb.pump_callback(in.data(), out.data(), in.size());
    wait_hop(1);
    const TelemetrySnapshot t1 = app.telemetry();
    CHECK(t1.queued_samples == 1600 && t1.queued_valid);  // one empty-output callback = 80 ms queued
    CHECK(t1.output_underruns == 1600 && t1.input_overruns == 0 && t1.output_drops == 0);

    // Stop: the queues are discarded, so the readout starts from zero again.
    app.stop_engine();
    const TelemetrySnapshot t2 = app.telemetry();
    CHECK(t2.queued_samples == 0 && t2.output_underruns == 0 && t2.output_drops == 0);
    CHECK(t2.lifetime_output_underruns == 1600);  // lifetime totals survive

    // Start: still zero before the first callback of the new stream.
    CHECK(app.start_engine());
    CHECK(app.telemetry().queued_samples == 0);

    // Reconfigure ("Apply and restart" in Devices) rebaselines the same way.
    fb.pump_callback(in.data(), out.data(), in.size());
    wait_hop(2);
    CHECK(app.telemetry().queued_samples == 1600);
    CHECK(app.apply_devices());
    CHECK(app.telemetry().queued_samples == 0);

    // Device-loss recovery: an unexpected stop discards the queues too.
    fb.pump_callback(in.data(), out.data(), in.size());
    CHECK(app.telemetry().queued_samples == 1600);
    fb.fire_notification(NotificationBits::kStopped);
    bool recovered = false;
    for (int i = 0; i < 2000 && !recovered; ++i) {
        app.tick();
        std::this_thread::sleep_for(5ms);
        recovered = app.engine().status().state == StreamState::Running;
    }
    CHECK(recovered);
    const TelemetrySnapshot t4 = app.telemetry();
    CHECK(t4.queued_samples == 0);  // fresh stream baseline, not 1600 + 1600 + 1600
    CHECK(t4.lifetime_output_underruns == 4800);

    // Text form: loopback mode has independently clocked callbacks (n/a).
    TelemetrySnapshot ts;
    ts.sample_rate = 16000;
    ts.queued_samples = 160;
    CHECK(ts.queued_ms_text() == "10 ms");
    ts.queued_valid = false;
    CHECK(ts.queued_ms_text() == "n/a (independent clocks)");
}

// --------------------------------------------------------------------------
// Mute reaches the audio output: EngineHost over the fake backend with a
// non-silent input (Stage 04 section D). Removing the worker's
// chain_.set_muted(...) leaves every other flag intact and fails only here.
// --------------------------------------------------------------------------
void test_mute_silences_output() {
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    EngineHost host(fb);
    CHECK(is_ok(host.controller().refresh()));
    CHECK(is_ok(host.start(FollowSystemDefault{}, FollowSystemDefault{}, CaptureMode::OrdinaryMicrophone)));

    // 440 Hz at -12 dBFS, phase-continuous across callbacks. Every callback
    // hands 1600 samples to the worker and takes 1600 from the output queue.
    constexpr std::size_t kBlock = 1600;
    std::vector<float> in(kBlock), out(kBlock, 0.0f);
    std::size_t sample_index = 0;
    auto pump_block = [&]() {
        for (float& x : in)
            x = 0.25f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(sample_index++) / 16000.0));
        const std::uint64_t hops_before = host.worker_hops();
        fb.pump_callback(in.data(), out.data(), in.size());
        for (int i = 0; i < 100 && host.worker_hops() == hops_before; ++i) std::this_thread::sleep_for(2ms);
        std::this_thread::sleep_for(10ms);  // let the hop reach the output queue
        float peak = 0.0f;
        for (float v : out) peak = std::max(peak, std::fabs(v));
        return peak;
    };
    // Peak of the last `tail` blocks after running `blocks` of them: the earlier
    // ones cover the queue's backlog and the 20 ms gain ramp (about 3 hops to -60 dB).
    auto settled_peak = [&](int blocks, int tail) {
        float peak = 0.0f;
        for (int i = 0; i < blocks; ++i) {
            const float p = pump_block();
            if (i >= blocks - tail) peak = std::max(peak, p);
        }
        return peak;
    };
    const float kMinus60dBFS = 0.001f;

    const float live = settled_peak(8, 3);
    CHECK(live > 0.1f);  // the signal passes through before muting

    host.set_muted(true);
    CHECK(host.muted());
    const float muted = settled_peak(8, 3);
    CHECK(muted < kMinus60dBFS);
    CHECK(host.telemetry().output_peak_dbfs < -60.0f);

    host.set_muted(false);
    const float back = settled_peak(8, 3);
    CHECK(back > 0.1f);  // and returns after unmuting
    host.stop();
}

// --------------------------------------------------------------------------
// Bypass (no model loaded): labelled, never "applied", output stage only
// --------------------------------------------------------------------------
void test_bypass_mode() {
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input", true)};

    AppModel app(fb, test_paths(), Settings{});
    app.initialise();
    CHECK(!app.mode_a() && !app.engine().mode_a());
    CHECK(app.mode_label() == "Bypass - no model loaded");
    CHECK(app.status_summary().mode == "Bypass - no model loaded");
    CHECK(contains(app.tray_spec().tooltip, "Bypass - no model loaded"));
    CHECK(app.param_apply() == ParamApply::BypassInactive);
    CHECK(app.param_apply_text() == "Bypass - model controls inactive");

    CHECK(app.start_engine());
    CHECK(contains(app.tray_spec().tooltip, "Running") && contains(app.tray_spec().tooltip, "Bypass - no model loaded"));
    app.set_param(ParamId::PitchShift, 3.0f);
    bool delivered = false;
    for (int i = 0; i < 500 && !delivered; ++i) {
        app.tick();
        std::this_thread::sleep_for(2ms);
        const TelemetrySnapshot t = app.telemetry();
        delivered = t.params_applied == t.params_pushed && t.params_pushed > 0;
    }
    CHECK(delivered);  // delivery to the Bypass worker works ...
    const TelemetrySnapshot t = app.telemetry();
    CHECK(!t.mode_a && t.params_accepted == 0 && t.params_rejected == 0);
    CHECK(app.param_apply() == ParamApply::BypassInactive);  // ... and is still never "applied"
    CHECK(contains(app.status_summary().mode, "Bypass"));
    app.stop_engine();
    CHECK(app.param_apply() == ParamApply::BypassInactive);

    // Translation needs a model with translator models: rejected here, state unchanged.
    CHECK(!app.set_translate(true));
    CHECK(!app.translate() && contains(app.translation_message(), "no model loaded"));
    app.set_param(ParamId::Translate, 1.0f);
    CHECK(!app.translate());

    // Voices are chosen by name. B follows A until chosen explicitly; an unknown
    // name is rejected with a message naming the choices and the previous voice stays.
    CHECK(app.voice_a() == "identity" && app.voice_b() == "identity");
    CHECK(app.set_voice_a("identity_b") && app.voice_b() == "identity_b" && app.params().voice_a == 1 && app.params().voice_b == 1);
    CHECK(app.set_voice_b("identity") && app.params().voice_b == 0);
    CHECK(app.set_voice_a("identity") && app.set_voice_a("identity_b"));
    CHECK(app.voice_b() == "identity" && app.params().voice_b == 0 && app.params().voice_a == 1);  // B no longer follows
    const std::size_t notices = app.notices().size();
    CHECK(!app.set_voice_a("ghost") && app.voice_a() == "identity_b" && app.params().voice_a == 1);
    CHECK(!app.set_voice_b("ghost_b") && app.voice_b() == "identity" && app.params().voice_b == 0);
    CHECK(app.notices().size() == notices + 2 && contains(app.notices().back().text, "ghost_b") &&
          contains(app.notices().back().text, "identity_b"));

    // A saved pair with B different from A is explicit; a saved pair with B == A follows.
    {
        Settings saved;
        saved.voice_a = "identity_b";
        saved.voice_b = "identity";
        AppModel a2(fb, test_paths(), saved);
        a2.initialise();
        CHECK(a2.voice_a() == "identity_b" && a2.voice_b() == "identity");
        CHECK(a2.set_voice_a("identity") && a2.set_voice_a("identity_b") && a2.voice_b() == "identity");
        Settings follow;
        follow.voice_a = "identity_b";
        follow.voice_b = "identity_b";
        AppModel a3(fb, test_paths(), follow);
        a3.initialise();
        CHECK(a3.set_voice_a("identity") && a3.voice_b() == "identity");
    }
}

// The output stage that does run in Bypass is VoiceOutputChain: the gate decides on the
// input level and gain applies. (Mute has its own test above.)
void test_bypass_output_stage() {
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    EngineHost host(fb);
    CHECK(is_ok(host.controller().refresh()));
    CHECK(is_ok(host.start(FollowSystemDefault{}, FollowSystemDefault{}, CaptureMode::OrdinaryMicrophone)));

    constexpr std::size_t kBlock = 1600;
    std::vector<float> in(kBlock), out(kBlock, 0.0f);
    std::size_t sample_index = 0;
    auto settled_peak = [&](float amp, int blocks) {
        float peak = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            for (float& x : in)
                x = amp * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(sample_index++) / 16000.0));
            const std::uint64_t hops_before = host.worker_hops();
            fb.pump_callback(in.data(), out.data(), in.size());
            for (int i = 0; i < 100 && host.worker_hops() < hops_before + 5; ++i) std::this_thread::sleep_for(2ms);
            std::this_thread::sleep_for(10ms);
            if (b >= blocks - 2) {
                peak = 0.0f;
                for (float v : out) peak = std::max(peak, std::fabs(v));
            }
        }
        return peak;
    };

    VoiceParams p;
    p.set(ParamId::GateThreshold, -30.0f);
    static_cast<void>(host.push_params(p));
    CHECK(settled_peak(0.25f, 8) > 0.2f);  // -15 dBFS rms: above the gate, unity gain, under the limiter knee
    CHECK(settled_peak(0.01f, 12) < 0.001f);  // -43 dBFS rms: below the gate threshold, so the gate closes

    p.set(ParamId::GateThreshold, -80.0f);
    p.set(ParamId::Gain, -12.0f);
    static_cast<void>(host.push_params(p));
    const float quiet = settled_peak(0.25f, 12);
    CHECK(quiet > 0.04f && quiet < 0.08f);  // 0.25 * 10^(-12/20) = 0.063

    p.set(ParamId::Gain, 0.0f);
    static_cast<void>(host.push_params(p));
    const float limited = settled_peak(1.2f, 12);
    CHECK(limited > 0.8f && limited < 0.9f);  // the -1 dBFS soft limiter holds the ceiling
    host.stop();
}

// --------------------------------------------------------------------------
// Mode A (M4 Stage 05): AppModel + EngineHost over fluxvoice::ModeAHost, with
// fake models (tests/FakeModelStep.hpp) and the fake audio backend. No ONNX
// Runtime: the real loader is only checked for its ORT-less message.
// --------------------------------------------------------------------------

/// Two ModelSteps and a voice table, the way HostModels hands them to a host.
class FakeModels final : public fluxvoice::HostModels {
public:
    FakeModels(int fps, std::vector<fluxvoice::VoiceEntry> voice_list, std::function<void()> on_destroy)
        : enc(test::fake_contract(ModelRole::kEncoder, fps)),
          voc(test::fake_contract(ModelRole::kVocoder, fps)),
          voices_(std::move(voice_list)),
          on_destroy_(std::move(on_destroy)) {}
    ~FakeModels() override {
        if (on_destroy_) on_destroy_();
    }
    ModelStep& encoder() noexcept override { return enc; }
    ModelStep& vocoder() noexcept override { return voc; }
    const std::vector<fluxvoice::VoiceEntry>& voices() const noexcept override { return voices_; }

    test::FakeModelStep enc, voc;

private:
    std::vector<fluxvoice::VoiceEntry> voices_;
    std::function<void()> on_destroy_;
};

/// HostModelLoader returning FakeModels. Voices follow the bundle directory
/// name: identity -> {identity, identity_b}; s2ut -> {identity, s2ut_only}.
class FakeLoader final : public fluxvoice::HostModelLoader {
public:
    struct Request {
        std::string dir;
        int         fps = 0;
        bool        with_translator = false;
    };

    bool has_translator(const fs::path&, int) const override { return false; }

    std::unique_ptr<fluxvoice::HostModels> load(const fs::path& dir, int frames_per_step, bool with_translator,
                                                std::string& error) const override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            gate_cv_.wait(lock, [&] { return !gate_closed_; });  // lets a test hold the loader thread
            requests_.push_back({dir.filename().string(), frames_per_step, with_translator});
            if (fail_dirs_.count(dir.filename().string()) != 0) {
                error = "fake: corrupt bundle " + dir.filename().string();
                return nullptr;
            }
        }
        const std::string leaf = dir.filename().string();
        std::vector<fluxvoice::VoiceEntry> voices;
        const std::vector<std::string> names = leaf == "identity" ? std::vector<std::string>{"identity", "identity_b"}
                                               : leaf == "s2ut"   ? std::vector<std::string>{"identity", "s2ut_only"}
                                                                  : std::vector<std::string>{leaf};
        for (const std::string& n : names) {
            fluxvoice::VoiceEntry v;
            v.name      = n;
            v.display   = n;
            v.embedding = {0.1f, 0.2f, 0.3f, 0.4f};  // spk_emb_dim of the fake vocoder
            v.stats     = {4.8, 0.25, -4.0, 1.5};
            v.license   = "CC0-1.0";
            voices.push_back(std::move(v));
        }
        auto models = std::make_unique<FakeModels>(frames_per_step == 0 ? 1 : frames_per_step, std::move(voices), [this] {
            const std::lock_guard<std::mutex> lock(mutex_);
            --alive_;
            if (on_destroy) on_destroy();
        });
        const std::lock_guard<std::mutex> lock(mutex_);
        ++alive_;
        last_ = models.get();
        return models;
    }

    void fail(const std::string& leaf) { const std::lock_guard<std::mutex> l(mutex_); fail_dirs_.insert(leaf); }
    void heal(const std::string& leaf) { const std::lock_guard<std::mutex> l(mutex_); fail_dirs_.erase(leaf); }
    void close_gate() { const std::lock_guard<std::mutex> l(mutex_); gate_closed_ = true; }
    void open_gate() {
        { const std::lock_guard<std::mutex> l(mutex_); gate_closed_ = false; }
        gate_cv_.notify_all();
    }
    int alive() const { const std::lock_guard<std::mutex> l(mutex_); return alive_; }
    std::size_t loads() const { const std::lock_guard<std::mutex> l(mutex_); return requests_.size(); }
    Request request(std::size_t i) const { const std::lock_guard<std::mutex> l(mutex_); return requests_.at(i); }
    FakeModels* last() const { const std::lock_guard<std::mutex> l(mutex_); return last_; }

    mutable std::function<void()> on_destroy;  // called under the loader's lock-free section, after alive_ dropped

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable gate_cv_;
    mutable bool gate_closed_ = false;
    mutable std::vector<Request> requests_;
    mutable int alive_ = 0;
    mutable FakeModels* last_ = nullptr;
    std::set<std::string> fail_dirs_;
};

struct ModelRig {
    FakeAudioBackend fb;
    std::shared_ptr<FakeLoader> loader = std::make_shared<FakeLoader>();
    std::unique_ptr<AppModel> app;

    explicit ModelRig(Settings settings = {}, bool initialise = true) {
        fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true),
                       fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input")};
        app = std::make_unique<AppModel>(fb, test_paths(), std::move(settings), loader);
        if (initialise) app->initialise();
    }
    fs::path bundle(const char* name) const { return g_repo / "tests" / "data" / "fixtures" / name; }

    /// Tick until the loader thread's result was drained and installed.
    bool wait_load(int ms = 10000) {
        for (int i = 0; i < ms / 2; ++i) {
            app->tick();
            if (!app->bundle_loading()) return true;
            std::this_thread::sleep_for(2ms);
        }
        return false;
    }
    /// One capture block of a 440 Hz tone through the backend callback.
    void pump(int blocks, float amp = 0.25f, std::size_t n = 320) {
        std::vector<float> in(n), out(n, 0.0f);
        for (int b = 0; b < blocks; ++b) {
            for (float& x : in)
                x = amp * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(phase_++) / 16000.0));
            fb.pump_callback(in.data(), out.data(), n);
            std::this_thread::sleep_for(1ms);
        }
    }
    /// Pump until `done()` (ticking the model) or the time runs out.
    bool pump_until(const std::function<bool()>& done, int max_blocks = 600) {
        for (int i = 0; i < max_blocks; ++i) {
            pump(1);
            app->tick();
            if (done()) return true;
        }
        return false;
    }

private:
    std::size_t phase_ = 0;
};

bool settle(const std::function<bool()>& cond, int ms = 3000) {
    for (int i = 0; i < ms / 2; ++i) {
        if (cond()) return true;
        std::this_thread::sleep_for(2ms);
    }
    return cond();
}

void test_mode_a_start_and_accounting() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(!app.mode_a());
    CHECK(app.load_bundle(r.bundle("identity")));
    CHECK(app.bundle_loading() && app.loading_bundle() == "identity");
    CHECK(!app.load_bundle(r.bundle("s2ut")));  // one load at a time
    CHECK(!app.start_engine() && !app.engine_on());  // Start is refused while loading
    CHECK(r.wait_load());
    CHECK(app.mode_a() && app.loaded_bundle() == "identity" && app.mode_label() == "Mode A - identity");
    CHECK(app.status_summary().mode == "Mode A - identity" && contains(app.tray_spec().tooltip, "Mode A - identity"));
    CHECK(app.active_bundle() == "identity" && app.settings().active_bundle == "identity");
    CHECK(r.loader->loads() == 1 && r.loader->request(0).fps == 0 && !r.loader->request(0).with_translator);  // kNever

    const auto rep = app.take_bundle_report();
    CHECK(rep && rep->ok && rep->rate == 16000 && rep->hop == 320 && rep->voices == 2 && rep->name == "identity");
    CHECK(!app.take_bundle_report());

    // The host wires the controller: contract rate, faded underruns, no queue cap, hop from the geometry.
    const fluxvoice::ModeAHost* host = app.engine().host();
    CHECK(host && host->loaded() && host->attached() && host->contract_rate() == 16000);
    CHECK(app.engine().controller().config().sample_rate == host->contract_rate());
    CHECK(app.engine().controller().config().fade_underruns && app.engine().controller().config().max_queued_samples == 0);
    CHECK(host->geometry().hop == 320);
    CHECK(app.param_apply() == ParamApply::Stopped);

    const std::size_t steps_before = r.loader->last()->enc.steps;  // the warm-up ran inside load()
    CHECK(steps_before == 20);
    CHECK(app.start_engine());
    CHECK(r.fb.last_request.sample_rate == 16000);
    CHECK(r.pump_until([&] { return app.engine().worker_hops() >= 25; }));
    // Quiet: stop feeding, let the worker finish its backlog, then read the accounting.
    std::uint64_t hops = 0;
    CHECK(settle([&] { const auto h = app.engine().worker_hops(); const bool same = h == hops; hops = h; std::this_thread::sleep_for(15ms); return same; }));
    const fluxvoice::WorkerSnapshot w = host->snapshot();
    CHECK(w.prefilled > 0);  // the startup silence the governor queued is part of the identity
    CHECK(w.queued == static_cast<std::int64_t>(w.prefilled + w.underruns - w.overruns - w.drops - w.skipped));
    const TelemetrySnapshot t = app.telemetry();
    CHECK(t.mode_a && t.queued_valid);
    CHECK(t.queued_samples == w.queued);  // Mode A shows the full identity, not the Bypass subset
    CHECK(t.queued_samples == static_cast<std::int64_t>(app.engine().controller().router().output_queued_approx()));
    CHECK(t.queued_samples > 0 && t.output_underruns == w.underruns && t.input_overruns == w.overruns);

    app.stop_engine();
    CHECK(r.loader->last()->enc.steps > steps_before + 20);  // hops reached the fake encoder
    CHECK(r.loader->last()->voc.steps > steps_before + 20);
    // Bypass keeps its own (rebaselined) identity; Mode A keeps the stream's own after stop too.
    CHECK(app.telemetry().queued_samples == 0 && app.telemetry().output_underruns == 0);
}

void test_mode_a_params_and_indicator() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(app.start_engine());
    CHECK(app.param_apply() != ParamApply::BypassInactive);

    // A push is accepted within a hop: accepted == pushed, and only then it reads "applied".
    app.set_param(ParamId::PitchShift, 3.0f);
    CHECK(r.pump_until([&] { const auto t = app.telemetry(); return t.params_accepted == t.params_pushed && t.params_pushed > 0; }, 100));
    CHECK(app.param_apply() == ParamApply::Applied && app.param_apply_text() == "applied by the pipeline");
    CHECK(app.telemetry().params_rejected == 0 && app.telemetry().params_applied == app.telemetry().params_pushed);

    // A snapshot the pipeline refuses (translate = 1 without translator models, pushed past the
    // control-thread check) is delivered but not accepted: the indicator must say so.
    VoiceParams bad = app.params();
    bad.set(ParamId::Translate, 1.0f);
    const std::uint64_t bad_gen = app.engine().push_params(bad);
    CHECK(r.pump_until([&] { return app.engine().rejected_params() >= 1; }, 100));
    const TelemetrySnapshot t = app.telemetry();
    CHECK(t.params_pushed == bad_gen && t.params_applied == bad_gen && t.params_accepted < bad_gen);  // delivered, not accepted
    CHECK(app.param_apply() == ParamApply::Rejected && app.param_apply_text() == "rejected by the pipeline");
    // The next accepted snapshot clears it.
    app.set_param(ParamId::Gain, -3.0f);
    CHECK(r.pump_until([&] { return app.param_apply() == ParamApply::Applied; }, 100));

    // Stopped engine: applied at start.
    app.stop_engine();
    CHECK(app.param_apply() == ParamApply::Stopped);
    // The pipeline really holds what was accepted (the worker is joined: readable).
    CHECK(near(app.engine().host()->pipeline().params().get(ParamId::PitchShift), 3.0f));
    CHECK(near(app.engine().host()->pipeline().params().get(ParamId::Gain), -3.0f));
}

void test_mode_a_param_queue_latest_wins() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    // Engine stopped: the worker does not drain. 200 pushes overflow the 64-slot channel and
    // park the newest; the final state must be the one the pipeline ends up with (BUG-023).
    VoiceParams snap = app.params();
    for (int i = 1; i <= 200; ++i) {
        snap.set(ParamId::Gain, -0.1f * static_cast<float>(i));
        static_cast<void>(app.engine().push_params(snap));
    }
    app.set_param(ParamId::Gain, -20.0f);
    CHECK(app.start_engine());
    bool caught_up = r.pump_until([&] {
        const TelemetrySnapshot t = app.telemetry();
        return t.params_accepted == t.params_pushed && t.params_pushed > 200;  // the parked snapshot was flushed by tick()
    }, 300);
    CHECK(caught_up);
    app.stop_engine();
    CHECK(near(app.engine().host()->pipeline().params().get(ParamId::Gain), -20.0f));
}

void test_mode_a_voices_by_name() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(app.voice_names() == std::vector<std::string>({"identity", "identity_b"}));
    CHECK(app.voice_a() == "identity" && app.voice_b() == "identity");

    CHECK(app.set_voice_a("identity_b") && app.params().voice_a == 1 && app.voice_b() == "identity_b" && app.params().voice_b == 1);
    CHECK(app.set_voice_b("identity") && app.params().voice_b == 0);
    CHECK(app.set_voice_a("identity") && app.set_voice_a("identity_b") && app.params().voice_b == 0);  // B no longer follows

    // An unknown name is rejected on the control thread, the host's message names the choices,
    // and the previous voice stays.
    const std::size_t notices = app.notices().size();
    CHECK(!app.set_voice_a("ghost") && app.voice_a() == "identity_b" && app.params().voice_a == 1);
    CHECK(app.notices().size() == notices + 1 && contains(app.notices().back().text, "ghost") &&
          contains(app.notices().back().text, "identity_b"));
    CHECK(!app.set_voice_b("nobody") && app.voice_b() == "identity" && app.params().voice_b == 0);

    // The pipeline sees the indices, not names.
    CHECK(app.start_engine());
    CHECK(r.pump_until([&] { return app.param_apply() == ParamApply::Applied; }, 100));
    app.stop_engine();
    CHECK(app.engine().host()->pipeline().params().voice_a == 1 && app.engine().host()->pipeline().params().voice_b == 0);

    // translate = 1 is rejected before it reaches the worker, with the stage's message.
    CHECK(!app.set_translate(true) && app.translation_message() == fluxvoice::kNoTranslatorModelsMessage);
    CHECK(!app.translate() && !app.translation().translator_available());
    app.set_param(ParamId::Translate, 1.0f);
    CHECK(!app.translate() && app.telemetry().params_rejected == 0);
}

void test_mode_a_bundle_swap() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(app.set_voice_a("identity_b"));
    CHECK(app.start_engine() && app.engine_on());
    CHECK(r.pump_until([&] { return app.engine().worker_hops() >= 10; }));
    const std::uint64_t opens = r.fb.open_calls;
    CHECK(r.fb.last_capture_was_default && r.fb.last_playback_was_default);

    // Swap while Running: stop -> load -> restart with the same policy; the old models are
    // destroyed only after the stream is stopped and the old host is detached.
    bool destroyed_while_running = false;
    int  destroyed = 0;
    r.loader->on_destroy = [&] {
        ++destroyed;
        destroyed_while_running = destroyed_while_running || r.fb.stream_running;
    };
    CHECK(app.activate_bundle("s2ut", "s2ut_only"));
    CHECK(!r.fb.stream_running && !app.engine_on() && app.bundle_loading());  // stopped before the load
    CHECK(app.mode_a() && app.loaded_bundle() == "identity");                   // the old model is still the active one
    CHECK(r.wait_load());
    CHECK(app.mode_a() && app.loaded_bundle() == "s2ut" && app.active_bundle() == "s2ut");
    CHECK(app.mode_label() == "Mode A - s2ut");
    CHECK(destroyed == 1 && !destroyed_while_running && r.loader->alive() == 1);
    CHECK(app.engine_on() && r.fb.stream_running && r.fb.open_calls == opens + 1);  // restarted
    CHECK(r.fb.last_capture_was_default && r.fb.last_playback_was_default);         // same device policy
    CHECK(app.voice_a() == "s2ut_only" && app.params().voice_a == 1);               // chosen with the bundle, by name
    CHECK(app.voice_names() == std::vector<std::string>({"identity", "s2ut_only"}));
    const auto rep = app.take_bundle_report();
    CHECK(rep && rep->ok && rep->name == "s2ut");
    CHECK(r.pump_until([&] { return app.param_apply() == ParamApply::Applied; }, 200));  // the new host takes parameters

    // Muting survives the swap: the new host starts muted.
    app.set_muted(true);
    CHECK(app.engine().host()->muted());
    CHECK(app.reload_bundle() && r.wait_load());
    CHECK(app.engine().host()->muted() && app.muted());
    CHECK(r.loader->alive() == 1);

    // A stopped engine stays stopped across a swap.
    app.stop_engine();
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(!app.engine_on() && !r.fb.stream_running && app.loaded_bundle() == "identity");
}

void test_mode_a_failed_load_keeps_previous() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(app.start_engine());
    r.loader->fail("s2ut");
    const std::uint64_t opens = r.fb.open_calls;
    CHECK(app.load_bundle(r.bundle("s2ut")) && !app.engine_on());
    CHECK(r.wait_load());

    // The previous bundle stays active (never Bypass), the error is a notice, the stream restarts.
    CHECK(app.mode_a() && app.loaded_bundle() == "identity" && app.mode_label() == "Mode A - identity");
    CHECK(app.active_bundle() == "identity" && app.engine().host() && app.engine().host()->attached());
    CHECK(app.engine_on() && r.fb.stream_running && r.fb.open_calls == opens + 1);
    CHECK(!app.notices().empty() && app.notices().back().level == Notice::Level::Error);
    CHECK(contains(app.notices().back().text, "fake: corrupt bundle s2ut") && contains(app.notices().back().text, "previous bundle"));
    const auto rep = app.take_bundle_report();
    CHECK(rep && !rep->ok && contains(rep->message, "corrupt"));
    CHECK(r.pump_until([&] { return app.engine().worker_hops() >= 5; }));  // and it still converts
    CHECK(app.voice_a() == "identity" && app.set_voice_a("identity_b"));

    // Healed: the same request now installs.
    r.loader->heal("s2ut");
    CHECK(app.load_bundle(r.bundle("s2ut")) && r.wait_load() && app.loaded_bundle() == "s2ut");

    // With nothing loaded, a failed load leaves Bypass, says so, and does not start a stopped engine.
    ModelRig r2;
    r2.loader->fail("identity");
    CHECK(r2.app->load_bundle(r2.bundle("identity")) && r2.wait_load());
    CHECK(!r2.app->mode_a() && r2.app->mode_label() == "Bypass - no model loaded" && !r2.app->engine_on());
    CHECK(contains(r2.app->notices().back().text, "Bypass") && contains(r2.app->notices().back().text, "corrupt"));
}

void test_mode_a_settings_feed_the_request() {
    Settings s;
    s.frames_per_step   = 2;
    s.latency_target_ms = 100;
    ModelRig r(s);
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(r.loader->request(0).fps == 2);
    const fluxvoice::ModeAHost* host = app.engine().host();
    CHECK(host->geometry().frames_per_step == 2 && host->geometry().hop == 640);
    CHECK(host->governor_target_samples() == 1600);  // 100 ms at 16 kHz
    CHECK(app.take_bundle_report()->hop == 640);

    // The defaults ask the bundle for its own variant and the master-budget latency.
    ModelRig d;
    CHECK(d.app->load_bundle(d.bundle("identity")) && d.wait_load());
    CHECK(d.loader->request(0).fps == 0);
    CHECK(d.app->engine().host()->governor_target_samples() != 1600);

    // A target that cannot fit the audio queue is a load error, not a crash.
    Settings big;
    big.latency_target_ms = 5000;
    ModelRig b(big);
    CHECK(b.app->load_bundle(b.bundle("identity")) && b.wait_load());
    CHECK(!b.app->mode_a() && b.app->take_bundle_report()->message.find("queue") != std::string::npos);
}

void test_mode_a_device_loss_keeps_bundle() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    CHECK(app.start_engine());
    CHECK(r.pump_until([&] { return app.engine().worker_hops() >= 8; }));

    r.fb.fire_notification(NotificationBits::kStopped);
    bool recovered = false;
    for (int i = 0; i < 2000 && !recovered; ++i) {
        app.tick();
        std::this_thread::sleep_for(2ms);
        recovered = app.engine().status().state == StreamState::Running && r.fb.stream_running;
    }
    CHECK(recovered);
    CHECK(app.mode_a() && app.loaded_bundle() == "identity" && app.engine().host()->attached());
    CHECK(r.loader->loads() == 1 && r.loader->alive() == 1);  // recovery never reloads
    // prepare() ran again for the new stream: the governor's startup silence is queued once, not stacked.
    const fluxvoice::WorkerSnapshot w = app.engine().host()->snapshot();
    CHECK(w.prefilled == app.engine().host()->governor_target_samples());
    CHECK(app.telemetry().queued_samples <= static_cast<std::int64_t>(w.prefilled));
    CHECK(r.pump_until([&] { return app.engine().worker_hops() >= 3; }, 300));  // converting again
    CHECK(app.param_apply() != ParamApply::BypassInactive);
}

void test_mode_a_fatal_error_stops_stream() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    r.loader->last()->voc.always_fail = true;  // the worker is not running: no race
    CHECK(app.start_engine() && app.engine_on());
    CHECK(r.pump_until([&] { return app.pipeline_failed(); }, 400));

    // The stream is stopped and the pipeline's message is shown; nothing restarts by itself.
    CHECK(!app.engine_on() && app.engine().status().state == StreamState::Idle && !r.fb.stream_running);
    CHECK(contains(app.pipeline_failure(), "fake Run failure"));
    CHECK(contains(app.engine_message(), "fake Run failure") && app.notices().back().level == Notice::Level::Error);
    CHECK(app.status_summary().pill == "Pipeline error" && contains(app.status_summary().detail, "fake Run failure"));
    const std::uint64_t opens = r.fb.open_calls;
    for (int i = 0; i < 50; ++i) { app.tick(); std::this_thread::sleep_for(2ms); }
    CHECK(!app.engine_on() && r.fb.open_calls == opens && app.pipeline_failed());  // no automatic restart loop

    // "Choose another bundle" opens the Catalogue; "Retry" starts again with the same bundle and devices.
    app.choose_another_bundle();
    CHECK(app.section() == Section::Catalogue);
    r.loader->last()->voc.always_fail = false;
    CHECK(app.retry_after_failure());
    CHECK(app.engine_on() && !app.pipeline_failed() && r.fb.stream_running && app.loaded_bundle() == "identity");
    CHECK(r.pump_until([&] { return app.engine().worker_hops() >= 5; }, 300));
    for (int i = 0; i < 20; ++i) { app.tick(); std::this_thread::sleep_for(2ms); }
    CHECK(!app.pipeline_failed() && app.engine_on());  // the latch cleared with prepare(): no stale second failure
}

void test_mode_a_startup_bundle() {
    // --bundle (set_startup_bundle) loads on the loader thread; "start on launch" waits for it.
    Settings s;
    s.start_engine_on_launch = true;
    ModelRig r(s, false);
    r.app->set_startup_bundle(r.bundle("identity"));
    r.app->initialise();
    CHECK(r.app->bundle_loading() && !r.app->engine_on());
    CHECK(r.wait_load());
    CHECK(r.app->mode_a() && r.app->engine_on() && r.fb.stream_running);
    CHECK(r.app->take_bundle_report()->ok);

    // A failed startup load does not start Bypass behind the user's back.
    ModelRig f(s, false);
    f.loader->fail("identity");
    f.app->set_startup_bundle(f.bundle("identity"));
    f.app->initialise();
    CHECK(f.wait_load());
    CHECK(!f.app->mode_a() && !f.app->engine_on() && f.fb.open_calls == 0);
    const auto rep = f.app->take_bundle_report();
    CHECK(rep && !rep->ok && contains(rep->message, "corrupt"));

    // The saved active bundle is the startup bundle when --bundle is absent; nothing saved means no load.
    Settings saved;
    saved.active_bundle = "s2ut";
    ModelRig sv(saved);
    CHECK(sv.app->bundle_loading() && sv.wait_load() && sv.app->loaded_bundle() == "s2ut");
    ModelRig none;
    CHECK(!none.app->bundle_loading() && none.loader->loads() == 0);
}

void test_mode_a_destructor_joins_loader() {
    auto loader = std::make_shared<FakeLoader>();
    loader->close_gate();
    {
        FakeAudioBackend fb;
        fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
        fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
        AppModel app(fb, test_paths(), Settings{}, loader);
        app.initialise();
        CHECK(app.load_bundle(g_repo / "tests" / "data" / "fixtures" / "identity"));
        std::thread release([&] {
            std::this_thread::sleep_for(60ms);
            loader->open_gate();
        });
        // Leaving the scope destroys the AppModel while the loader thread is blocked in load():
        // the destructor must wait for it, then destroy the undrained host.
        release.detach();
    }
    CHECK(settle([&] { return loader->alive() == 0; }));
    CHECK(loader->loads() == 1);
}

// --------------------------------------------------------------------------
// M4 Stage 06: live latency and per-stage telemetry
// --------------------------------------------------------------------------

/// Pure derivations of TelemetrySnapshot (no engine): stage rows, the identity
/// verdict and the latency card.
void test_telemetry_derived() {
    TelemetrySnapshot t;
    t.sample_rate = 16000;
    t.mode_a = true;
    t.stream_running = true;
    t.stats.vc = true;
    t.stats.sample_rate = 16000;
    t.stats.hop = 320;  // 20 ms
    t.stats.timing = {100, 4000, 8000, 12000, 3};
    t.stats.stages[0] = {100, 1000, 2000, 3000, 0};
    CHECK(near(static_cast<float>(t.hop_ms()), 20.0f));

    // One row per PipelineStage in enum order, then the whole hop.
    const auto rows = t.stage_rows();
    CHECK(rows.size() == fluxvoice::kPipelineStageCount + 1);
    for (std::size_t i = 0; i < fluxvoice::kPipelineStageCount; ++i)
        CHECK(std::string(rows[i].name) == fluxvoice::to_string(static_cast<fluxvoice::PipelineStage>(i)));
    const StageRow& hop = rows[fluxvoice::kPipelineStageCount];
    CHECK(std::string(hop.name) == "hop" && hop.timing.samples == 100);
    CHECK(near(static_cast<float>(hop.p50_ms), 4.0f) && near(static_cast<float>(hop.p95_ms), 8.0f) &&
          near(static_cast<float>(hop.max_ms), 12.0f));
    CHECK(near(static_cast<float>(hop.share), 0.4f));  // p95 / hop duration = 8 / 20
    CHECK(near(static_cast<float>(rows[0].p95_ms), 2.0f) && near(static_cast<float>(rows[0].share), 0.1f));
    CHECK(rows[1].timing.samples == 0 && rows[1].share == 0.0);

    // Identity check: consistent while |difference| <= tolerance, MISMATCH beyond it either way.
    t.stats.queued = 560;
    t.stats.occupancy_tolerance = 480;
    auto verdict = [&](std::int64_t occupancy) {
        t.stats.occupancy = occupancy;
        return t.identity_check();
    };
    CHECK(verdict(560).verdict == IdentityVerdict::Consistent && verdict(560).difference == 0);
    CHECK(verdict(560 + 480).verdict == IdentityVerdict::Consistent);   // the tolerance itself is allowed
    CHECK(verdict(560 + 481).verdict == IdentityVerdict::Mismatch);
    CHECK(verdict(560 + 481).difference == 481 && verdict(560 + 481).tolerance == 480);
    CHECK(verdict(560 - 480).verdict == IdentityVerdict::Consistent);
    CHECK(verdict(560 - 481).verdict == IdentityVerdict::Mismatch);     // a missing term is negative too
    CHECK(std::string(to_string(verdict(1200).verdict)) == "MISMATCH");
    CHECK(verdict(-1).verdict == IdentityVerdict::NotApplicable);       // occupancy unknown
    t.stats.occupancy = 560;
    t.stream_running = false;
    CHECK(t.identity_check().verdict == IdentityVerdict::NotApplicable);
    t.stream_running = true;
    t.queued_valid = false;  // system-playback loopback: the identity does not hold
    CHECK(t.identity_check().verdict == IdentityVerdict::NotApplicable && contains(t.identity_check().why_na, "loopback"));
    t.queued_valid = true;

    // Latency card: the CLI's lat_ms, its terms, compute p95 and the algorithmic latency beside it.
    fluxvoice::PipelineGeometry g;
    g.sample_rate = 16000;
    g.frame_samples = 320;
    g.hop = 320;
    g.lookahead_frames = 1;
    g.overlap_samples = 0;
    StreamInfo si;
    si.capture_internal_rate = si.playback_internal_rate = 48000;
    si.capture_internal_period = si.playback_internal_period = 480;  // 10 ms each
    t.stats.latency = fluxvoice::estimate_latency(g, 800, si);       // 50 ms queued + 20 ms model + 20 ms devices
    t.stats.lat_ms = t.stats.latency.total_ms;
    t.stats.algorithmic_samples = g.algorithmic_samples();            // 640 samples = 40 ms
    const LatencyCard c = t.latency_card();
    CHECK(near(static_cast<float>(c.total_ms), 90.0f) && c.total_ms == t.stats.lat_ms);
    CHECK(near(static_cast<float>(c.capture_device_ms), 10.0f) && near(static_cast<float>(c.playback_device_ms), 10.0f));
    CHECK(near(static_cast<float>(c.queued_ms), 50.0f) && near(static_cast<float>(c.model_delay_ms), 20.0f));
    CHECK(std::fabs(c.terms_sum_ms() - c.total_ms) < 1e-9);               // the terms are the whole sum
    CHECK(near(static_cast<float>(c.compute_p95_ms), 8.0f) && near(static_cast<float>(c.algorithmic_ms), 40.0f));
    CHECK(std::fabs(c.total_ms - (90.0 + c.compute_p95_ms)) > 1.0);       // compute is not a term
    CHECK(std::fabs(c.total_ms - (90.0 + c.algorithmic_ms)) > 1.0);       // neither is the algorithmic latency
    CHECK(contains(t.stats_line(1.0), "lat_ms=90.0"));                    // the same number the CLI prints
    CHECK(!c.bypass && c.valid);
}

/// Bypass: the legacy `[stats]` snapshot, a measured occupancy that matches the
/// identity, no model terms in the latency card, no stage timings.
void test_bypass_stats() {
    ModelRig r;  // no bundle loaded: Bypass
    AppModel& app = *r.app;
    CHECK(!app.mode_a());
    {
        const TelemetrySnapshot idle = app.telemetry();
        CHECK(idle.identity_check().verdict == IdentityVerdict::NotApplicable && !idle.stream_running);
    }
    CHECK(app.start_engine());
    r.pump(10);
    CHECK(settle([&] {
        const TelemetrySnapshot s = app.telemetry();
        return s.stats.queued > 0 && s.stats.occupancy == s.stats.queued;
    }));
    const TelemetrySnapshot t = app.telemetry();
    CHECK(!t.mode_a && !t.stats.vc && t.stats.skipped == 0 && t.stats.prefilled == 0);
    CHECK(t.stream_running && t.queued_valid);
    // The per-stream counters and the HostStats are one set of numbers.
    CHECK(t.stats.queued == t.queued_samples && t.stats.overruns == t.input_overruns &&
          t.stats.underruns == t.output_underruns && t.stats.drops == t.output_drops && t.stats.frames == t.frames);
    CHECK(t.queued_samples == static_cast<std::int64_t>(t.output_underruns) - static_cast<std::int64_t>(t.input_overruns) -
                                  static_cast<std::int64_t>(t.output_drops));
    // Occupancy: queues plus the samples the worker holds; the tolerance is one device period (480 frames at
    // 48 kHz = 160 samples at 16 kHz) plus one 20 ms block.
    CHECK(t.stats.occupancy == t.queued_samples);
    CHECK(t.stats.occupancy_tolerance == 160 + 320);
    const IdentityCheck id = t.identity_check();
    CHECK(id.verdict == IdentityVerdict::Consistent && id.difference == 0);

    // The legacy line, spelled out field by field.
    const std::string line = t.stats_line(3.0);
    const std::string expect = "[stats] t=3.0 frames=" + std::to_string(t.frames) + " overruns=" + std::to_string(t.input_overruns) +
                               " underruns=" + std::to_string(t.output_underruns) + " drops=" + std::to_string(t.output_drops) +
                               " queued=" + std::to_string(t.queued_samples) + " skipped=0 prefilled=0 rss_kb=" +
                               std::to_string(t.rss_bytes / 1024);
    CHECK(line == expect);
    CHECK(!contains(line, "inf_p50_us") && !contains(line, "lat_ms") && !contains(line, "s2ut_"));
    CHECK(app.stats_line().rfind("[stats] t=", 0) == 0);

    // Latency card: queued + device terms only, labelled Bypass; no model delay, no compute.
    const LatencyCard c = t.latency_card();
    CHECK(c.bypass && c.valid && c.model_delay_ms == 0.0 && c.compute_p95_ms == 0.0 && c.algorithmic_ms == 0.0);
    CHECK(near(static_cast<float>(c.capture_device_ms), 10.0f) && near(static_cast<float>(c.playback_device_ms), 10.0f));
    CHECK(near(static_cast<float>(c.queued_ms), static_cast<float>(1000.0 * static_cast<double>(t.queued_samples) / 16000.0)));
    CHECK(std::fabs(c.terms_sum_ms() - c.total_ms) < 1e-9);
    for (const StageRow& row : t.stage_rows()) CHECK(row.timing.samples == 0);  // nothing is timed in Bypass

    // Stopping rebaselines both sides; the check is n/a until a stream runs again.
    app.stop_engine();
    const TelemetrySnapshot stopped = app.telemetry();
    CHECK(stopped.stats.queued == 0 && stopped.identity_check().verdict == IdentityVerdict::NotApplicable);
}

/// One model tick at `t0 + sec` seconds of synthetic time. A capture callback
/// runs first while the engine is on: the controller's liveness check treats a
/// clock that jumps ahead of the callback counter as a stalled device.
void tick_at(ModelRig& r, AppModel::TimePoint t0, int sec) {
    if (r.app->engine_on()) r.pump(1);
    r.app->tick(t0 + std::chrono::seconds(sec));
}

std::vector<std::string> read_lines(const fs::path& p) {
    std::vector<std::string> out;
    std::ifstream f(p, std::ios::binary);
    for (std::string line; std::getline(f, line);) out.push_back(line);
    return out;
}

/// "Write stats log": one line per second of stream time, opened once per
/// stream, `t` rebased per stream, a failure turns it off with a notice.
void test_stats_log() {
    const fs::path dir = g_tmp / "statslog";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path file = path_from_utf8(path_to_utf8(dir) + "/journal_\xC3\xA9.log");  // non-ASCII name: the UTF-8 path helpers
    fs::remove(file, ec);
    auto has_error_notice = [](const AppModel& a) {
        for (const Notice& n : a.notices())
            if (n.level == Notice::Level::Error && contains(n.text, "Stats log disabled")) return true;
        return false;
    };

    // Off by default: nothing is created, whatever runs.
    {
        fs::remove(g_tmp / "config" / "stats.log", ec);
        ModelRig r;
        const auto t0 = AppModel::Clock::now();
        CHECK(r.app->start_engine());
        for (int k = 0; k <= 3; ++k) tick_at(r, t0, k);
        CHECK(!fs::exists(g_tmp / "config" / "stats.log") && !r.app->stats_log_active() && r.app->stats_log_opens() == 0);
        CHECK(r.app->stats_log_file() == g_tmp / "config" / "stats.log");  // the default path
    }

    Settings on;
    on.write_stats_log = true;
    on.stats_log_path  = path_to_utf8(file);
    ModelRig r(on);
    AppModel& app = *r.app;
    const auto t0 = AppModel::Clock::now();
    auto at = [&](int sec) { tick_at(r, t0, sec); };
    auto t_of = [](const std::string& line) {  // "[stats] t=<x> frames=..."
        CHECK(line.rfind("[stats] t=", 0) == 0);
        return std::stod(line.substr(10));
    };

    CHECK(app.start_engine());
    at(0);  // the stream's epoch begins here
    r.pump(4);  // never r.pump_until here: it ticks with the real clock, and the test drives the model clock itself
    CHECK(settle([&] { return app.engine().worker_hops() >= 2; }));
    CHECK(read_lines(file).empty() && app.stats_log_opens() == 0);  // the first line is due one second in
    for (int k = 1; k <= 3; ++k) {
        at(k);
        CHECK(read_lines(file).size() == static_cast<std::size_t>(k));  // one line per second, flushed
    }
    auto lines = read_lines(file);
    CHECK(lines.size() == 3 && t_of(lines[0]) == 1.0 && t_of(lines[1]) == 2.0 && t_of(lines[2]) == 3.0);
    CHECK(app.stats_log_opens() == 1 && app.stats_log_lines() == 3 && app.stats_log_active());  // opened once, not per line
    for (const std::string& l : lines) CHECK(contains(l, " skipped=0 prefilled=0 rss_kb="));  // Bypass: the legacy line

    // The stream ends: the file is closed and nothing is appended while stopped.
    app.stop_engine();
    at(4);
    at(5);
    CHECK(!app.stats_log_active() && read_lines(file).size() == 3);

    // A new stream is a new epoch: t restarts at 0, the file is opened again (append).
    CHECK(app.start_engine());
    at(6);
    at(7);
    at(8);
    lines = read_lines(file);
    CHECK(lines.size() == 5 && t_of(lines[3]) == 1.0 && t_of(lines[4]) == 2.0);
    CHECK(app.stats_log_opens() == 2);

    // A reconfigure while running (Devices > Apply) rebaselines the counters too: another epoch.
    CHECK(app.apply_devices());
    at(9);
    at(10);
    lines = read_lines(file);
    CHECK(lines.size() == 6 && t_of(lines[5]) == 1.0 && app.stats_log_opens() == 3);
    CHECK(app.stream_seconds() == 1.0);

    // Switching it off closes the file at the next line due.
    app.settings().write_stats_log = false;
    at(11);
    CHECK(!app.stats_log_active() && read_lines(file).size() == 6);
    app.stop_engine();

    // The default path: an empty setting writes <config dir>/stats.log.
    {
        Settings d;
        d.write_stats_log = true;
        ModelRig rd(d);
        const auto d0 = AppModel::Clock::now();
        CHECK(rd.app->start_engine());
        tick_at(rd, d0, 0);
        tick_at(rd, d0, 1);
        CHECK(read_lines(g_tmp / "config" / "stats.log").size() == 1);
        fs::remove(g_tmp / "config" / "stats.log", ec);
    }

    // An unwritable path: logging turns itself off, once, with an Error notice.
    {
        Settings bad;
        bad.write_stats_log = true;
        bad.stats_log_path  = path_to_utf8(g_tmp / "no_such_dir" / "stats.log");  // a user-chosen path never creates directories
        ModelRig rb(bad);
        const auto b0 = AppModel::Clock::now();
        CHECK(rb.app->start_engine());
        tick_at(rb, b0, 0);
        CHECK(!has_error_notice(*rb.app));
        tick_at(rb, b0, 1);
        CHECK(has_error_notice(*rb.app) && !rb.app->settings().write_stats_log && !rb.app->stats_log_active());
        CHECK(!fs::exists(g_tmp / "no_such_dir"));
        const std::size_t notices = rb.app->notices().size();
        for (int k = 2; k <= 4; ++k) tick_at(rb, b0, k);
        CHECK(rb.app->notices().size() == notices && rb.app->stats_log_lines() == 0 && rb.app->stats_log_opens() == 0);
        CHECK(rb.app->engine_on());  // the audio never depends on the log
    }
#if defined(__linux__)
    // A write error mid-run (the device accepts the open, then refuses the data).
    if (fs::exists("/dev/full")) {
        Settings full;
        full.write_stats_log = true;
        full.stats_log_path  = "/dev/full";
        ModelRig rf(full);
        const auto f0 = AppModel::Clock::now();
        CHECK(rf.app->start_engine());
        tick_at(rf, f0, 0);
        tick_at(rf, f0, 1);
        CHECK(has_error_notice(*rf.app) && !rf.app->settings().write_stats_log && !rf.app->stats_log_active());
        CHECK(rf.app->stats_log_lines() == 0);
    }
#endif
}

/// Mode A: the telemetry is the host's HostStats, with the full identity (a
/// governor skip and the startup prefill are both in `queued`), the CLI's
/// `[stats]` line, per-stage rows, a consistent identity check and the latency
/// card as the CLI computes it.
void test_mode_a_stats() {
    ModelRig r;
    AppModel& app = *r.app;
    CHECK(app.load_bundle(r.bundle("identity")) && r.wait_load());
    const fluxvoice::ModeAHost* host = app.engine().host();
    CHECK(host != nullptr);
    if (!host) return;
    CHECK(app.start_engine());
    r.pump(14);
    // A silent burst the governor must trim: 8 hops in one callback drain the
    // prefill (underruns), refill the queue past the target and force a skip.
    {
        std::vector<float> zin(8 * 320, 0.0f), zout(8 * 320, 0.0f);
        r.fb.pump_callback(zin.data(), zout.data(), zin.size());
    }
    // At rest: the worker consumed everything, the counters stopped moving, and the
    // measured occupancy has caught up with the identity.
    auto at_rest = [&] {
        TelemetrySnapshot last = app.telemetry();
        int stable = 0;
        return settle([&] {
            const TelemetrySnapshot now = app.telemetry();
            const bool same = now.stats.queued == last.stats.queued && now.stats.skipped == last.stats.skipped &&
                              now.stats.underruns == last.stats.underruns && now.stats.timing.samples == last.stats.timing.samples;
            last   = now;
            stable = (same && now.stats.skipped > 0 && now.stats.occupancy == now.stats.queued) ? stable + 1 : 0;
            std::this_thread::sleep_for(3ms);
            return stable >= 5;
        }, 5000);
    };
    CHECK(at_rest());

    const TelemetrySnapshot t = app.telemetry();
    const fluxvoice::WorkerSnapshot w = host->snapshot();
    CHECK(t.mode_a && t.stats.vc && t.stream_running && t.queued_valid);
    CHECK(t.stats.prefilled > 0 && t.stats.skipped > 0 && t.stats.prefilled != t.stats.skipped);  // the subset formula is wrong here
    // queued is the worker's full identity, not underruns - overruns - drops.
    CHECK(t.queued_samples == w.queued && t.stats.queued == w.queued);
    CHECK(t.queued_samples == static_cast<std::int64_t>(w.prefilled + w.underruns - w.overruns - w.drops - w.skipped));
    CHECK(t.queued_samples != static_cast<std::int64_t>(t.output_underruns) - static_cast<std::int64_t>(t.input_overruns) -
                                  static_cast<std::int64_t>(t.output_drops));
    CHECK(t.stats.skipped == w.skipped && t.stats.prefilled == w.prefilled);
    CHECK(t.output_underruns == w.underruns && t.input_overruns == w.overruns && t.output_drops == w.drops);

    // The `[stats]` line is exactly the CLI's (format_stats_line over the host's HostStats). At rest the two
    // snapshots agree; RSS may move by a page between them, so retry until the whole line matches.
    {
        bool same = false;
        std::string gui, cli;
        for (int i = 0; i < 50 && !same; ++i) {
            const fluxvoice::HostStats hs = host->stats();
            const TelemetrySnapshot ts = app.telemetry();
            gui  = ts.stats_line(7.0);
            cli  = fluxvoice::format_stats_line(7.0, hs);
            same = gui == cli;
        }
        CHECK(same);
        if (!same) std::printf("  gui: %s\n  cli: %s\n", gui.c_str(), cli.c_str());
    }
    const std::string line = t.stats_line(7.0);
    for (const char* field : {" skipped=", " prefilled=", " rss_kb=", " inf_p50_us=", " inf_p95_us=", " inf_max_us=", " misses=",
                              " errors=", " lat_ms="})
        CHECK(contains(line, field));
    CHECK(!contains(line, "s2ut_"));  // no translator models in this bundle
    CHECK(contains(line, " skipped=" + std::to_string(w.skipped) + " prefilled=" + std::to_string(w.prefilled) + " "));
    CHECK(contains(line, " queued=" + std::to_string(w.queued) + " "));

    // Stage rows: every PipelineStage in order, then the whole hop; the stages Mode A runs have samples.
    const auto rows = t.stage_rows();
    CHECK(rows.size() == fluxvoice::kPipelineStageCount + 1);
    for (std::size_t i = 0; i < fluxvoice::kPipelineStageCount; ++i)
        CHECK(std::string(rows[i].name) == fluxvoice::to_string(static_cast<fluxvoice::PipelineStage>(i)));
    CHECK(std::string(rows.back().name) == "hop" && rows.back().timing.samples > 0 && rows.back().timing.samples == t.stats.timing.samples);
    using fluxvoice::PipelineStage;
    for (PipelineStage st : {PipelineStage::kFeatures, PipelineStage::kEncoder, PipelineStage::kVocoder, PipelineStage::kOutputChain})
        CHECK(rows[static_cast<std::size_t>(st)].timing.samples > 0);
    CHECK(rows[static_cast<std::size_t>(PipelineStage::kTranslatorEnc)].timing.samples == 0);   // no translator: never ran
    CHECK(rows[static_cast<std::size_t>(PipelineStage::kTranslatorDec)].timing.samples == 0);
    CHECK(near(static_cast<float>(t.hop_ms()), 20.0f));
    CHECK(near(static_cast<float>(rows.back().share), static_cast<float>(rows.back().p95_ms / 20.0)));

    // Governor: target and hard cap in ms, from the host.
    CHECK(t.stats.governor_target == host->governor_target_samples() && t.stats.governor_hard_cap > t.stats.governor_target);
    CHECK(near(static_cast<float>(t.governor_target_ms()),
               static_cast<float>(1000.0 * static_cast<double>(t.stats.governor_target) / 16000.0)));
    CHECK(t.governor_hard_cap_ms() > t.governor_target_ms());

    // The identity check: measured occupancy agrees with the full identity, skip included.
    const IdentityCheck id = t.identity_check();
    CHECK(id.verdict == IdentityVerdict::Consistent && id.difference == 0 && id.queued == w.queued);
    CHECK(id.tolerance == 160 + 320 && id.occupancy == id.queued);

    // Latency card: the CLI's lat_ms with its terms; compute and the algorithmic latency are shown, not added.
    const LatencyCard c = t.latency_card();
    CHECK(!c.bypass && c.valid && c.total_ms == t.stats.lat_ms && c.total_ms == t.stats.latency.total_ms);
    CHECK(std::fabs(c.terms_sum_ms() - c.total_ms) < 1e-9);
    CHECK(near(static_cast<float>(c.capture_device_ms), 10.0f) && near(static_cast<float>(c.playback_device_ms), 10.0f));
    CHECK(near(static_cast<float>(c.queued_ms), static_cast<float>(1000.0 * static_cast<double>(w.queued) / 16000.0)));
    CHECK(near(static_cast<float>(c.algorithmic_ms), 20.0f));  // the framing hop of the fake models (no lookahead)
    CHECK(near(static_cast<float>(c.compute_p95_ms), static_cast<float>(static_cast<double>(t.stats.timing.p95_us) / 1000.0)));
    CHECK(contains(line, " lat_ms=" + [&] {
        char b[32];
        std::snprintf(b, sizeof b, "%.1f", c.total_ms);
        return std::string(b);
    }()));

    // The compute sparkline gets one whole-hop p95 point per second.
    const auto t0 = AppModel::Clock::now();
    for (int k = 0; k <= 3; ++k) tick_at(r, t0, k);
    CHECK(app.compute_history().size() == 3);

    // Stopping rebaselines: the check is n/a until a stream runs again.
    app.stop_engine();
    CHECK(app.telemetry().identity_check().verdict == IdentityVerdict::NotApplicable);
    CHECK(app.telemetry().queued_samples == 0);
}

/// Argument 2: write the GUI-produced `[stats]` lines of one Bypass run and one
/// Mode A run (fake models) to a file, `t` increasing over the whole file, for
/// `tools/analyze_soak.py --check-format` (CTest gui_stats_parse). The lines
/// come from the real TelemetrySnapshot::stats_line path.
void write_stats_sample() {
    if (g_stats_sample.empty()) return;
    std::vector<std::string> lines;
    double t = 0.0;
    auto sample = [&](ModelRig& r, int count, auto&& settle_hops) {
        for (int k = 0; k < count; ++k) {
            r.pump(6);
            settle_hops();
            r.app->tick();
            t += 1.0;
            lines.push_back(r.app->telemetry().stats_line(t));
        }
    };
    {  // Bypass: the legacy line
        ModelRig r;
        CHECK(!r.app->mode_a() && r.app->start_engine());
        sample(r, 4, [&] { std::this_thread::sleep_for(20ms); });
        r.app->stop_engine();
    }
    {  // Mode A over the fake models: the full VC line, with a governor skip in it
        ModelRig r;
        CHECK(r.app->load_bundle(r.bundle("identity")) && r.wait_load() && r.app->mode_a());
        CHECK(r.app->start_engine());
        for (int k = 0; k < 5; ++k) {
            if (k == 2) {
                std::vector<float> zin(8 * 320, 0.0f), zout(8 * 320, 0.0f);
                r.fb.pump_callback(zin.data(), zout.data(), zin.size());
            }
            sample(r, 1, [&] { std::this_thread::sleep_for(60ms); });
        }
        r.app->stop_engine();
    }
    std::error_code ec;
    fs::create_directories(g_stats_sample.parent_path(), ec);
    std::ofstream out(g_stats_sample, std::ios::binary | std::ios::trunc);
    CHECK(static_cast<bool>(out));
    for (const std::string& l : lines) out << l << '\n';
    out.flush();
    CHECK(static_cast<bool>(out));
    CHECK(lines.size() >= 9);
    std::size_t vc_lines = 0;
    for (const std::string& l : lines) {
        CHECK(l.rfind("[stats] t=", 0) == 0);
        if (contains(l, " inf_p50_us=")) ++vc_lines;
    }
    CHECK(vc_lines == 5 && vc_lines < lines.size());  // both kinds of line are in the file
}

void test_engine_host_modes() {
    // EngineHost alone: install -> Mode A, remove -> Bypass with the Bypass config and hooks back.
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    auto loader = std::make_shared<FakeLoader>();
    EngineHost host(fb, ControllerConfig{}, loader);
    CHECK(host.mode() == EngineHost::Mode::Bypass && !host.mode_a() && host.mode_label() == "Bypass - no model loaded");
    const ControllerConfig bypass = host.controller().config();

    auto make = [&] {
        auto h = std::make_unique<fluxvoice::ModeAHost>(loader);
        fluxvoice::HostError err;
        fluxvoice::HostLoadRequest req;
        req.model_dir = "identity";
        CHECK(h->load(req, {}, err));
        return h;
    };
    std::string error;
    CHECK(host.install_host(make(), "identity", error) && host.mode_a() && host.mode_label() == "Mode A - identity");
    CHECK(host.controller().config().fade_underruns && host.controller().config().max_queued_samples == 0);
    CHECK(host.host() && host.host()->attached());
    CHECK(is_ok(host.controller().refresh()));

    // A running stream refuses a model change; nothing is replaced.
    CHECK(is_ok(host.start(FollowSystemDefault{}, FollowSystemDefault{}, CaptureMode::OrdinaryMicrophone)));
    CHECK(!host.install_host(make(), "other", error) && contains(error, "stopped") && host.mode_label() == "Mode A - identity");
    CHECK(!host.remove_host(error) && host.mode_a());
    host.stop();
    CHECK(!host.install_host(nullptr, "x", error));

    CHECK(host.remove_host(error) && !host.mode_a() && host.host() == nullptr);
    CHECK(host.controller().config().sample_rate == bypass.sample_rate);
    CHECK(host.controller().config().fade_underruns == bypass.fade_underruns);
    CHECK(host.controller().config().max_queued_samples == bypass.max_queued_samples);
    CHECK(loader->alive() == 0 || loader->alive() == 1);  // the unattached `make()` leftovers are gone with their hosts
    // Bypass works again (its hooks are back): the worker runs and processes hops.
    CHECK(is_ok(host.start(FollowSystemDefault{}, FollowSystemDefault{}, CaptureMode::OrdinaryMicrophone)));
    std::vector<float> in(320, 0.25f), out(320, 0.0f);
    for (int i = 0; i < 5; ++i) { fb.pump_callback(in.data(), out.data(), in.size()); std::this_thread::sleep_for(5ms); }
    CHECK(settle([&] { return host.worker_hops() >= 3; }));
    host.stop();
}

void test_real_loader_without_ort() {
    // The shipped loader in a build without ONNX Runtime: the control thread reports the
    // message, nothing crashes, the engine stays where it was.
    if (fluxvoice::ort_available()) return;  // the ORT trees cover the real load in gui_smoke_bundle
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    AppModel app(fb, test_paths(), Settings{});  // default loader: fluxvoice::ort_bundle_loader()
    app.initialise();
    CHECK(app.load_bundle(g_repo / "tests" / "data" / "fixtures" / "identity"));
    bool done = false;
    for (int i = 0; i < 5000 && !done; ++i) {
        app.tick();
        done = !app.bundle_loading();
        if (!done) std::this_thread::sleep_for(2ms);
    }
    CHECK(done && !app.mode_a());
    const auto rep = app.take_bundle_report();
    CHECK(rep && !rep->ok && contains(rep->message, "built without ONNX Runtime"));
    CHECK(app.notices().back().level == Notice::Level::Error && contains(app.notices().back().text, "built without ONNX Runtime"));
    CHECK(app.mode_label() == "Bypass - no model loaded");

    // Same through ModeAHost directly: a typed error, never an exception.
    fluxvoice::ModeAHost host;
    fluxvoice::HostError err;
    fluxvoice::HostLoadRequest req;
    req.model_dir = g_repo / "tests" / "data" / "fixtures" / "identity";
    CHECK(!host.load(req, {}, err) && err.kind == fluxvoice::HostErrorKind::kBuiltWithoutOrt);
}

// --------------------------------------------------------------------------
// BUG-022 guard: an unexpected exception stops the stream and reports, and
// the model keeps working (the GUI loop continues).
// --------------------------------------------------------------------------
void test_unexpected_exception_recovery() {
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input", true)};

    AppModel app(fb, test_paths(), Settings{});
    app.initialise();
    app.set_param(ParamId::Gain, -6.0f);
    CHECK(app.start_engine());
    CHECK(app.engine_on() && app.engine().stream_live());
    const std::string bundle = app.active_bundle(), voice = app.voice_a();
    const DeviceSelection cap = app.capture_selection(), play = app.playback_selection();
    CHECK(!bundle.empty() && !voice.empty());

    app.on_unexpected_exception("field 'gain' has the wrong type");
    CHECK(!app.engine_on());
    CHECK(!app.engine().stream_live() && app.engine().status().state == StreamState::Idle);
    CHECK(!fb.stream_running);
    CHECK(!app.notices().empty() && app.notices().back().level == Notice::Level::Error);
    CHECK(contains(app.notices().back().text, "field 'gain' has the wrong type"));
    CHECK(contains(app.engine_message(), "field 'gain' has the wrong type"));

    // Everything else is intact.
    CHECK(near(app.params().get(ParamId::Gain), -6.0f));
    CHECK(app.active_bundle() == bundle && app.voice_a() == voice);
    CHECK(app.capture_selection() == cap && app.playback_selection() == play);
    CHECK(!app.device_unresolved());

    // The same failure on the next frames collapses into one notice.
    const std::size_t notices = app.notices().size();
    app.on_unexpected_exception("field 'gain' has the wrong type");
    app.on_unexpected_exception("field 'gain' has the wrong type");
    CHECK(app.notices().size() == notices);
    app.on_unexpected_exception("a different failure");
    CHECK(app.notices().size() == notices + 1 && contains(app.notices().back().text, "a different failure"));

    // Calling it with no stream open is harmless, and the loop's next frame works: tick, then restart.
    CHECK(!app.engine_on());
    app.tick();
    CHECK(app.start_engine());
    CHECK(app.engine_on() && app.engine().stream_live());
    app.on_unexpected_exception("again");
    CHECK(!app.engine().stream_live());
    app.on_unexpected_exception("with the stream already stopped");
    CHECK(!app.engine_on());
}

void test_paths_helpers() {
    CHECK(!default_config_dir().empty());
    const auto roots = resource_search_roots(g_repo / "build" / "bin", g_repo);
    CHECK(!roots.empty() && roots.front() == g_repo);
    CHECK(find_resource(roots, fs::path("presets") / "neutral.json") == g_repo / "presets" / "neutral.json");
    CHECK(find_resource(roots, "no-such-resource.bin").empty());
    CHECK(!system_font_candidates().empty());
}

// --------------------------------------------------------------------------
// UTF-8 paths: SDL dialog strings, settings.json and Studio drafts keep a
// non-ASCII directory intact (Stage 04 section C). On Linux this pins the
// behaviour only; the ANSI-code-page hazard itself exists on Windows and is
// verified only by the MSVC leg.
// --------------------------------------------------------------------------
void test_utf8_paths() {
    // Escapes, not literal characters: independent of the compiler's source charset.
    const std::u8string u8name = u8"voces_\u00f1_\u65e5\u672c";
    const std::string   utf8(u8name.begin(), u8name.end());
    CHECK(utf8.size() == 6 + 2 + 1 + 3 + 3);  // "voces_", n-tilde (2 bytes), "_", two CJK characters (3 bytes each)
    CHECK(path_to_utf8(path_from_utf8(utf8)) == utf8);
    CHECK(path_from_utf8(utf8) == fs::path(u8name));
    CHECK(path_to_utf8(path_from_utf8("plain_ascii")) == "plain_ascii" && path_from_utf8("").empty());
    CHECK(path_to_utf8_generic(path_from_utf8(utf8) / "a") == utf8 + "/a");

    const fs::path root   = g_tmp / "utf8";
    const fs::path bundle = root / path_from_utf8(utf8);
    const fs::path cfg    = bundle / "config";  // settings.json itself lives under a non-ASCII directory
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(cfg, ec);
    fs::copy_file(g_repo / "tests" / "data" / "fixtures" / "identity" / "voices.json", bundle / "voices.json", ec);
    CHECK(!ec && fs::is_directory(bundle));

    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 11, "CABLE Input", true)};

    AppPaths paths       = test_paths();
    paths.config_dir     = cfg;
    paths.models_dir     = root / "empty_models";   // the bundle enters only through the dialog path
    paths.studio_dir     = bundle / "studio";
    {
        AppModel app(fb, paths, Settings{});
        app.initialise();
        CHECK(app.catalog().bundle(utf8) == nullptr);

        // What GuiApp::dialog_callback does with SDL's UTF-8 result, then the command the view calls.
        const std::string from_sdl = path_to_utf8(bundle);
        CHECK(app.add_bundle_dir(path_from_utf8(from_sdl)));
        const BundleInfo* b = app.catalog().bundle(utf8);
        CHECK(b != nullptr && b->valid() && b->dir == bundle);
        CHECK(app.active_bundle() == utf8);

        app.settings().models_dir = path_to_utf8(root);   // as the Settings "Browse..." callback stores it
        std::string err;
        CHECK(app.save_settings(&err));
        CHECK(err.empty());
    }
    CHECK(contains(slurp(paths.settings_file()), utf8));  // raw UTF-8 bytes in the JSON, not escapes or ANSI bytes

    const Settings loaded = Settings::load(paths.settings_file());
    CHECK(loaded.active_bundle == utf8);
    CHECK(loaded.models_dir == path_to_utf8(root));
    CHECK(path_from_utf8(loaded.models_dir) == root && fs::is_directory(path_from_utf8(loaded.models_dir)));

    {   // Restart: the saved models dir and active bundle resolve to the same directory.
        paths.models_dir = path_from_utf8(loaded.models_dir);
        AppModel app(fb, paths, loaded);
        app.initialise();
        const BundleInfo* b = app.catalog().bundle(utf8);
        CHECK(b != nullptr && b->valid() && b->dir == bundle);
        CHECK(app.active_bundle() == utf8);
    }

    {   // Studio draft: a take under a non-ASCII directory survives save and load.
        const fs::path wav = bundle / "take.wav";
        write_wav(wav, sine(1.0, 16000, 0.3f), 16000, 16);
        VoiceStudioModel m;
        m.draft().voice_name = "utf8voice";
        m.add_take(wav, "s01");
        std::string err;
        CHECK(m.save_draft(paths.studio_dir, err));
        CHECK(contains(slurp(m.project_dir(paths.studio_dir) / "draft.json"), utf8));
        VoiceStudioModel m2;
        m2.draft().voice_name = "utf8voice";
        CHECK(m2.load_draft(paths.studio_dir));
        CHECK(m2.draft().takes.size() == 1);
        if (m2.draft().takes.size() == 1) CHECK(m2.draft().takes[0].path == wav);
        CHECK(contains(m.takes_tsv(), utf8));
    }
}

} // namespace

int main(int argc, char** argv) {
    g_repo = argc > 1 ? fs::path(argv[1]) : fs::current_path();
    if (argc > 2) g_stats_sample = fs::path(argv[2]);
    g_tmp  = fs::temp_directory_path() / "fluxvoice_frontend_model_test";
    std::error_code ec;
    fs::remove_all(g_tmp, ec);
    fs::create_directories(g_tmp, ec);

    test_device_selector();
    test_voice_catalog();
    test_translation();
    test_tray_model();
    test_settings();
    test_meters();
    test_loader_thread();
    test_telemetry_text();
    test_voice_studio();
    test_app_model();
    test_app_model_restores_policy();
    test_device_policy_rules();
    test_preset_and_translation_rules();
    test_studio_analysis_drained_by_tick();
    test_studio_drafts();
    test_json_wrong_types();
    test_json_every_field_every_type();
    test_json_hostile_numbers();
    test_param_queue_latest_wins();
    test_queued_latency_rebaselined();
    test_mute_silences_output();
    test_bypass_mode();
    test_bypass_output_stage();
    test_mode_a_start_and_accounting();
    test_mode_a_params_and_indicator();
    test_mode_a_param_queue_latest_wins();
    test_mode_a_voices_by_name();
    test_mode_a_bundle_swap();
    test_mode_a_failed_load_keeps_previous();
    test_mode_a_settings_feed_the_request();
    test_mode_a_device_loss_keeps_bundle();
    test_mode_a_fatal_error_stops_stream();
    test_mode_a_startup_bundle();
    test_mode_a_destructor_joins_loader();
    test_telemetry_derived();
    test_bypass_stats();
    test_mode_a_stats();
    test_stats_log();
    write_stats_sample();
    test_engine_host_modes();
    test_real_loader_without_ort();
    test_unexpected_exception_recovery();
    test_paths_helpers();
    test_utf8_paths();

    fs::remove_all(g_tmp, ec);
    return test::finish("frontend_model");
}
