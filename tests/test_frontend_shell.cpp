// Frontend shell — headless tests (Milestone 4 Stage 01 residuals, no display,
// no audio hardware): platform paths and environment reads (BUG-018),
// settings load/save robustness (malformed file kept as settings.json.bad,
// atomic save), Mini panel position persistence, web-URL policy for the
// catalogue, and the GUI worker's priority request.
//
// Usage: test_frontend_shell [repo-root]
#include "core/EngineHost.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "core/UrlPolicy.hpp"
#include "voice/VoiceParams.hpp"
#include "FakeAudioBackend.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using namespace fluxvoice;
using namespace fluxvoice::frontend;
using fluxvoice::testing::FakeAudioBackend;
namespace fs = std::filesystem;

[[maybe_unused]] fs::path g_repo = ".";
fs::path g_tmp;

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void spit(const fs::path& p, std::string_view text) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << text;
}

bool contains(std::string_view hay, std::string_view needle) { return hay.find(needle) != std::string_view::npos; }

fs::path fresh_dir(const char* name) {
    const fs::path d = g_tmp / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d, ec);
    return d;
}

// --------------------------------------------------------------------------
void test_paths_env() {
    CHECK(env_path("FLUXVOICE_SURELY_UNSET_VARIABLE_12345").empty());
#if defined(_WIN32)
    CHECK(!env_path("PATH").empty());
    CHECK(!env_path("SystemRoot").empty());
#else
    const fs::path home = env_path("HOME");
    const char*    raw  = std::getenv("HOME");
    CHECK((raw == nullptr || *raw == '\0') ? home.empty() : home == fs::path(raw));
    CHECK(!env_path("PATH").empty());
#endif
    CHECK(!default_config_dir().empty());
    CHECK(default_config_dir().filename() == "fluxvoice" || default_config_dir().filename() == ".fluxvoice");
}

// --------------------------------------------------------------------------
void test_settings_malformed() {
    const fs::path d    = fresh_dir("malformed");
    const fs::path file = d / "settings.json";
    const std::string original = "{ \"theme\": \"light\", oops this is broken";
    spit(file, original);

    std::vector<std::string> warnings;
    const Settings s = Settings::load(file, &warnings);
    CHECK(s == Settings{});
    CHECK(warnings.size() == 1);
    CHECK(!warnings.empty() && contains(warnings[0], "settings.json.bad"));
    CHECK(!fs::exists(file));
    CHECK(slurp(d / "settings.json.bad") == original);

    // An older .bad is replaced by the newer one.
    spit(file, "[1,2,3]");   // valid JSON but not an object
    warnings.clear();
    static_cast<void>(Settings::load(file, &warnings));
    CHECK(!fs::exists(file) && slurp(d / "settings.json.bad") == "[1,2,3]");
    CHECK(warnings.size() == 1);

    // Missing file: defaults, silently.
    warnings.clear();
    const Settings first = Settings::load(d / "nope.json", &warnings);
    CHECK(first == Settings{} && warnings.empty());

    // After the move-aside a save creates a fresh file.
    std::string err;
    CHECK(Settings{}.save(file, err));
    CHECK(fs::exists(file) && slurp(d / "settings.json.bad") == "[1,2,3]");
}

void test_settings_save() {
    const fs::path d    = fresh_dir("save");
    const fs::path file = d / "sub" / "settings.json";
    Settings s;
    s.theme   = Theme::Light;
    s.gain_db = -3.5f;
    s.extras["future_key"] = "{\"a\":[1,2]}";
    std::string err;
    CHECK(s.save(file, err));
    CHECK(err.empty());
    CHECK(!fs::exists(file.parent_path() / "settings.json.tmp"));

    std::vector<std::string> warnings;
    const Settings back = Settings::load(file, &warnings);
    CHECK(warnings.empty());
    CHECK(back == s);
    CHECK(back.extras.count("future_key") == 1);
    CHECK(back.extras.at("future_key") == "{\"a\":[1,2]}");
    // Unknown keys survive a second save as real JSON.
    CHECK(back.save(file, err));
    const auto doc = nlohmann::json::parse(slurp(file));
    CHECK(doc.contains("future_key") && doc["future_key"]["a"][1] == 2);
    CHECK(!fs::exists(file.parent_path() / "settings.json.tmp"));

    // Save over a malformed file preserves it as .bad (never silently destroys it).
    spit(file, "not json at all");
    CHECK(s.save(file, err));
    CHECK(slurp(fs::path(file) += ".bad") == "not json at all");
    CHECK(Settings::load(file, nullptr) == s);

    // Cannot move aside (a directory sits at the .bad path) -> refuse, keep original.
    const fs::path d2    = fresh_dir("save_refuse");
    const fs::path file2 = d2 / "settings.json";
    spit(file2, "garbage");
    std::error_code ec;
    fs::create_directories(d2 / "settings.json.bad" / "child", ec);
    err.clear();
    CHECK(!s.save(file2, err));
    CHECK(!err.empty());
    CHECK(slurp(file2) == "garbage");
}

void test_settings_mini_position() {
    // Round trip incl. negative coordinates (monitor left of / above the primary).
    Settings s;
    CHECK(!s.mini_has_position);
    CHECK(!contains(s.to_json(), "mini_position"));
    CHECK(!contains(s.to_json(), "mini_x"));
    s.mini_has_position = true;
    s.mini_x            = -1920;
    s.mini_y            = -40;
    const std::string text = s.to_json();
    CHECK(contains(text, "mini_position"));
    CHECK(!contains(text, "\"mini_x\"") && !contains(text, "\"mini_y\""));
    const Settings back = Settings::from_json(text);
    CHECK(back.mini_has_position && back.mini_x == -1920 && back.mini_y == -40);

    // Absent -> no position.
    CHECK(!Settings::from_json("{}").mini_has_position);

    // Legacy -1,-1 -> no position; the legacy keys are not re-emitted.
    const Settings legacy_none = Settings::from_json("{\"mini_x\":-1,\"mini_y\":-1}");
    CHECK(!legacy_none.mini_has_position);
    CHECK(legacy_none.extras.empty());
    CHECK(!contains(legacy_none.to_json(), "mini_x"));

    // Legacy positive pair -> position kept, migrated to the new key.
    const Settings legacy = Settings::from_json("{\"mini_x\":300,\"mini_y\":120}");
    CHECK(legacy.mini_has_position && legacy.mini_x == 300 && legacy.mini_y == 120);
    CHECK(legacy.extras.empty());
    const std::string migrated = legacy.to_json();
    CHECK(contains(migrated, "mini_position") && !contains(migrated, "\"mini_x\""));

    // The new key wins over legacy ones; a legacy half-pair gives no position.
    const Settings both = Settings::from_json("{\"mini_position\":[5,-6],\"mini_x\":9,\"mini_y\":9}");
    CHECK(both.mini_has_position && both.mini_x == 5 && both.mini_y == -6);
    CHECK(!Settings::from_json("{\"mini_x\":50}").mini_has_position);

    // A malformed new key warns and gives no position.
    std::vector<std::string> warnings;
    CHECK(!Settings::from_json("{\"mini_position\":[1]}", &warnings).mini_has_position);
    CHECK(warnings.size() == 1);
}

void test_settings_wait_k() {
    const ParamDesc& wk = param_desc(ParamId::WaitK);
    const int lo = static_cast<int>(wk.min), hi = static_cast<int>(wk.max), def = static_cast<int>(wk.def);
    std::vector<std::string> warnings;
    CHECK(Settings::from_json("{\"wait_k\":" + std::to_string(hi + 1) + "}", &warnings).wait_k == def);
    CHECK(warnings.size() == 1);
    CHECK(Settings::from_json("{\"wait_k\":" + std::to_string(lo - 1) + "}").wait_k == def);
    CHECK(Settings::from_json("{\"wait_k\":" + std::to_string(lo) + "}").wait_k == lo);
    CHECK(Settings::from_json("{\"wait_k\":" + std::to_string(hi) + "}").wait_k == hi);
    warnings.clear();
    CHECK(Settings::from_json("{\"wait_k\":" + std::to_string(lo) + "}", &warnings).wait_k == lo);
    CHECK(warnings.empty());
}

// --------------------------------------------------------------------------
void test_url_policy() {
    const char* accept[] = {"http://example.com", "https://example.com/path?q=1#frag", "HTTPS://x", "HtTp://localhost:8080/",
                            "https://huggingface.co/org/model"};
    for (const char* u : accept) CHECK(is_web_url(u));
    const char* reject[] = {"",
                            "https://",
                            "http://",
                            "https:// x",
                            "https://exa mple.com",
                            "https://a\\b",
                            "https://a\nb",
                            "https://a\tb",
                            "https://x\x7f",
                            "javascript:alert(1)",
                            "file:///etc/passwd",
                            "data:text/html,<b>x</b>",
                            "ftp://example.com",
                            "example.com",
                            "/usr/bin/x",
                            "//example.com",
                            "https:/example.com",
                            "https:///path",
                            "xhttps://example.com",
                            " https://example.com"};
    for (const char* u : reject) CHECK(!is_web_url(u));
    CHECK(!is_web_url(std::string_view("https://a\0b", 11)));
}

// --------------------------------------------------------------------------
void test_worker_priority() {
    FakeAudioBackend fb;
    fb.capture  = {fb.make_ep(EndpointDirection::Capture, 1, "USB Mic", true)};
    fb.playback = {fb.make_ep(EndpointDirection::Playback, 10, "Speakers", true)};
    EngineHost host(fb);
    CHECK(std::string_view(host.priority_method()) == "not requested");
    CHECK(!host.priority_elevated());
    CHECK(is_ok(host.controller().refresh()));
    const auto res = host.start(FollowSystemDefault{}, FollowSystemDefault{}, CaptureMode::OrdinaryMicrophone);
    CHECK(is_ok(res));
    for (int i = 0; i < 500 && std::string_view(host.priority_method()) == "not requested"; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(std::string_view(host.priority_method()) != "not requested");
    host.stop();
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_repo = argv[1];
    std::error_code ec;
    g_tmp = fs::temp_directory_path(ec) /
            ("fv_frontend_shell_" + std::to_string(static_cast<unsigned long long>(
                                         std::chrono::steady_clock::now().time_since_epoch().count())));
    fs::create_directories(g_tmp, ec);

    test_paths_env();
    test_settings_malformed();
    test_settings_save();
    test_settings_mini_position();
    test_settings_wait_k();
    test_url_policy();
    test_worker_priority();

    fs::remove_all(g_tmp, ec);
    return test::finish("frontend_shell");
}
