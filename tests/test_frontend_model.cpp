// Frontend model layer — headless tests (Milestone 4 frontend, no display, no
// audio hardware). Covers frontend/README.md §10:
//   device rows and selection policy, catalogue loading and validation,
//   translation guard, studio wizard and WAV statistics, tray menu spec and
//   icon, settings round-trip, output-chain click bound, and the AppModel
//   driving an AudioController over a fake backend.
//
// Usage: test_frontend_model [repo-root]
#include "core/AppModel.hpp"
#include "core/DeviceSelectorModel.hpp"
#include "core/OutputChain.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "core/Telemetry.hpp"
#include "core/TranslationModel.hpp"
#include "core/TrayModel.hpp"
#include "core/VoiceCatalog.hpp"
#include "core/VoiceStudio.hpp"
#include "voice/VoiceParams.hpp"
#include "FakeAudioBackend.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
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
    const DeviceSelection sel = selection_for_row(rows, 2);
    CHECK(std::holds_alternative<ExplicitEndpoint>(sel));
    if (const auto* e = std::get_if<ExplicitEndpoint>(&sel)) {
        CHECK(e->key == cout_.key);
        CHECK(e->generation == snap.generation);
    }
    CHECK(std::holds_alternative<FollowSystemDefault>(selection_for_row(rows, 0)));
    CHECK(std::holds_alternative<FollowSystemDefault>(selection_for_row(rows, 99)));

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
    CHECK(feedback_preview(snap, selection_for_row(rows, 2), ExplicitEndpoint{cin.key, snap.generation},
                           CaptureMode::OrdinaryMicrophone).has_value());
    CHECK(!feedback_preview(snap, selection_for_row(rows, 1), ExplicitEndpoint{cin.key, snap.generation},
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

    CHECK(classify_license("CC BY 4.0") == LicenseClass::CommercialClean);
    CHECK(classify_license("CC BY-NC-SA 4.0") == LicenseClass::ResearchOnly);
    CHECK(classify_license("private - do not distribute") == LicenseClass::Private);
    CHECK(classify_license("") == LicenseClass::Unknown);
    CHECK(is_valid_voice_name("me-2") && !is_valid_voice_name("Me") && !is_valid_voice_name("") && !is_valid_voice_name("a b"));

    // Scan the fixture root: one bundle, encoder + vocoder, no translator.
    VoiceCatalog cat;
    CHECK(cat.scan_models_root(fixtures) == 1);
    CHECK(cat.scan_models_root(g_repo / "does-not-exist") == 0);
    CHECK(cat.bundles().size() == 1);
    const BundleInfo* b = cat.bundle("identity");
    CHECK(b != nullptr);
    if (b) {
        CHECK(b->valid() && b->voices.size() == 2);
        CHECK(b->has_encoder && b->has_vocoder && !b->has_translator && !b->has_langs);
        CHECK(b->onnx_files.size() == 6);
    }
    CHECK(cat.find_voice("identity", "identity_b") != nullptr);
    CHECK(cat.find_voice("identity", "nope") == nullptr);
    CHECK(cat.voice_names("identity").size() == 2);
    CHECK(cat.installed_voice_count() == 2);

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
    CHECK(all.size() == 5);  // 2 installed voices + 3 catalogue entries
    bool fixture_installed = false;
    for (const auto& it : all)
        if (it.id == "fixture-identity") fixture_installed = it.installed;
    CHECK(fixture_installed);
    f.installed_only = true;
    CHECK(cat.search(f).size() == 3);  // 2 voices + the catalogue entry whose bundle is installed
    f = {};
    f.license = LicenseClass::ResearchOnly;
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.query = "HI-FI";
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.language = "de";
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.tag = "test";
    CHECK(cat.search(f).size() == 1);
    f = {};
    f.sort = VoiceCatalog::Sort::License;
    auto sorted = cat.search(f);
    CHECK(!sorted.empty() && sorted.front().license_class == LicenseClass::CommercialClean);
    CHECK(cat.all_languages().size() == 4);
    CHECK(!cat.all_tags().empty());
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
    CHECK(TranslationModel::clamp_wait_k(0) == 1 && TranslationModel::clamp_wait_k(99) == 10);
    CHECK(near(static_cast<float>(TranslationModel::onset_estimate_ms(3)), 240.0f));
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
// Output chain & telemetry
// --------------------------------------------------------------------------
void test_output_chain() {
    OutputChain c;
    c.configure(16000);
    std::vector<float> buf(1600, 0.5f);
    c.process(buf.data(), buf.size());
    CHECK(near(c.last_peak(), 0.5f, 0.02f));

    // A -24 dB step must ramp: no sample-to-sample jump above the click bound.
    c.set_gain_db(-24.0f);
    std::fill(buf.begin(), buf.end(), 0.5f);
    c.process(buf.data(), buf.size());
    float max_delta = 0.0f;
    for (std::size_t i = 1; i < buf.size(); ++i) max_delta = std::max(max_delta, std::fabs(buf[i] - buf[i - 1]));
    CHECK(max_delta < 0.01f);
    CHECK(near(buf.back(), 0.5f * 0.0631f, 0.01f));  // reached -24 dB after 100 ms

    c.set_muted(true);
    for (int i = 0; i < 3; ++i) { std::fill(buf.begin(), buf.end(), 0.5f); c.process(buf.data(), buf.size()); }
    CHECK(c.last_peak() < 1e-3f);

    // Limiter keeps |y| under the ceiling for hot input.
    OutputChain lim;
    lim.configure(16000);
    std::vector<float> hot(256, 2.0f);
    lim.process(hot.data(), hot.size());
    CHECK(lim.last_peak() < 0.9f && lim.last_peak() > 0.8f);
    lim.set_limiter(false);
    std::fill(hot.begin(), hot.end(), 2.0f);
    lim.process(hot.data(), hot.size());
    CHECK(lim.last_peak() > 1.9f);
    CHECK(near(OutputChain::to_dbfs(1.0f), 0.0f) && OutputChain::to_dbfs(0.0f) == -120.0f);

    TelemetrySnapshot t;
    t.frames = 10; t.output_underruns = 3; t.input_overruns = 1; t.output_drops = 1; t.queued_samples = 1;
    t.sample_rate = 16000; t.rss_bytes = 2048;
    CHECK(t.stats_line(1.0) == "[stats] t=1.0 frames=10 overruns=1 underruns=3 drops=1 queued=1 rss_kb=2");
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
    const fs::path w16 = g_tmp / "take16.wav", w24 = g_tmp / "take24.wav", w2 = g_tmp / "stereo.wav";
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
    const fs::path room = g_tmp / "room.wav";
    write_wav(room, quiet, sr, 16);
    CHECK(m.add_room_tone(room).verdict == TakeVerdict::Clean);
    CHECK(m.room_tone_seconds() >= 29.0);
    m.add_live_mic(w16);
    CHECK(near(static_cast<float>(m.live_mic_minutes()), 4.0f / 60.0f, 1e-3f));
    CHECK(m.next() && m.step() == StudioStep::Prepare);

    const fs::path studio = g_tmp / "studio", models = g_tmp / "models";
    const StudioCommands posix = m.commands(g_repo, studio, models, Shell::Posix);
    CHECK(contains(posix.prepare, "prepare_target_voice.py") && contains(posix.prepare, "--voice me"));
    CHECK(contains(posix.finetune, "train_unit_vocoder.py") && contains(posix.finetune, "--fine-tune"));
    CHECK(contains(posix.export_, "export_stateful_onnx.py") && contains(posix.export_, "--role vocoder") &&
          contains(posix.export_, "private"));
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

void test_paths_helpers() {
    CHECK(!default_config_dir().empty());
    const auto roots = resource_search_roots(g_repo / "build" / "bin", g_repo);
    CHECK(!roots.empty() && roots.front() == g_repo);
    CHECK(find_resource(roots, fs::path("presets") / "neutral.json") == g_repo / "presets" / "neutral.json");
    CHECK(find_resource(roots, "no-such-resource.bin").empty());
    CHECK(!system_font_candidates().empty());
}

} // namespace

int main(int argc, char** argv) {
    g_repo = argc > 1 ? fs::path(argv[1]) : fs::current_path();
    g_tmp  = fs::temp_directory_path() / "fluxvoice_frontend_model_test";
    std::error_code ec;
    fs::remove_all(g_tmp, ec);
    fs::create_directories(g_tmp, ec);

    test_device_selector();
    test_voice_catalog();
    test_translation();
    test_tray_model();
    test_settings();
    test_output_chain();
    test_voice_studio();
    test_app_model();
    test_app_model_restores_policy();
    test_paths_helpers();

    fs::remove_all(g_tmp, ec);
    return test::finish("frontend_model");
}
