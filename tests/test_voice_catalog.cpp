#include "pipeline/VoiceCatalog.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    using namespace fluxvoice;
    using nlohmann::json;
    std::vector<VoiceEntry> voices;
    std::string error;
    if (argc == 2) {
        CHECK(read_voice_catalog(argv[1], 4, voices, error));
        CHECK(find_voice(voices, "identity_b", error) != nullptr);
        const auto* voice = find_voice(voices, "identity", error);
        CHECK(voice != nullptr);
        if (voice) CHECK(voice->embedding == std::vector<float>(4, 0.0f));
        return test::finish("voice_catalog_fixture");
    }
    const auto path = std::filesystem::temp_directory_path() /
        ("fluxvoice-voices-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    if (std::filesystem::exists(path)) {
        CHECK(false);
        return test::finish("voice_catalog");
    }
    const json good = {{"schema", 1}, {"voices", json::array({
        {{"name", "test"}, {"display", "Test"}, {"license", "CC0"}, {"source", "fake"},
         {"embedding", {0.25, -0.5}}, {"logf0_mean", 5.0}, {"logf0_std", 0.25},
         {"energy_mean", -8.0}, {"energy_std", 2.0}}})}};
    const auto check = [&](const json& data, bool expected, std::optional<std::size_t> dim = 2) {
        { std::ofstream f(path); f << data.dump(); }
        CHECK(read_voice_catalog(path, dim, voices, error) == expected);
    };
    check(good, true);
    CHECK(find_voice(voices, "missing", error) == nullptr);
    CHECK(error.find("available: test") != std::string::npos);
    check(good, false, 4);
    check(good, true, std::nullopt); // listing needs no model/runtime
    json bad = good;
    bad["schema"] = true;
    check(bad, false);
    bad = good;
    bad["voices"].push_back(bad["voices"][0]);
    check(bad, false);
    for (const char* field : {"name", "display", "license", "source", "energy_std", "embedding"}) {
        bad = good;
        bad["voices"][0].erase(field);
        check(bad, false);
    }
    for (const auto& value : {json(0), json(-0.5), json(true), json("1")}) {
        bad = good;
        bad["voices"][0]["logf0_std"] = value;
        check(bad, false);
    }
    bad = good;
    bad["voices"][0]["embedding"][0] = 1e100; // finite double, not finite f32
    check(bad, false);
    bad["voices"][0]["embedding"][0] = true;
    check(bad, false);
    bad = good;
    bad["voices"][0]["name"] = "Bad Voice";
    check(bad, false);
    std::filesystem::remove(path);
    return test::finish("voice_catalog");
}
