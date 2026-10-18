// ModelContract — contract v1 parser/validator tests (Milestone 2 / Stage 02).
//
// Runs the cases in tests/data/contracts/ that python_training/tests/
// test_contract.py also runs, so the two validators accept and reject the same
// contracts with the same messages. Then checks the fixture manifest written by
// tools/make_fixture_models.py and prints the algorithmic latency of every
// fixture pair. No ONNX Runtime: the manifest carries each fixture's contract
// string and graph I/O.
//
// Usage: test_model_contract [tests/data]
#include "pipeline/ModelContract.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef FLUXVOICE_TEST_DATA_DIR
#define FLUXVOICE_TEST_DATA_DIR "tests/data"
#endif

namespace fs = std::filesystem;
using nlohmann::json;
using namespace fluxvoice;

namespace {

fs::path g_data_dir = FLUXVOICE_TEST_DATA_DIR;

// Hand-computed in Documents/Milestone 2/02_model_contract_fixtures.md
// (Verification): hop + (L_enc + L_voc) x frame_samples + overlap_samples.
const std::map<std::string, int64_t> kFixtureLatency = {
    {"identity_s1", 320}, {"identity_s2", 640}, {"identity_s4", 1280}, {"delay_s1", 640}};

json read_json(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    json j = json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) std::printf("%s: not valid JSON\n", path.string().c_str());
    return j;
}

std::vector<fs::path> json_files(const fs::path& dir) {
    std::vector<fs::path> files;
    if (fs::is_directory(dir))
        for (const auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".json") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<TensorSpec> specs(const json& items) {
    std::vector<TensorSpec> out;
    for (const auto& t : items) {
        TensorSpec s{t.at("name").get<std::string>(), dtype_from_string(t.at("dtype").get<std::string>()), {}};
        if (t.at("shape").is_null()) s.has_shape = false;  // the graph gives no shape (unknown rank)
        else s.shape = t["shape"].get<std::vector<int64_t>>();
        out.push_back(std::move(s));
    }
    return out;
}

/// Parse every model, cross-check graphs, validate the pair. Returns false with
/// the first error; `latency` is set for pairs.
bool run_case(const json& c, std::string& error, int64_t& latency) {
    std::vector<ModelContract> contracts;
    for (const auto& m : c.at("models")) {
        const json& raw = m.at("contract");
        const std::string text = raw.is_string() ? raw.get<std::string>() : raw.dump();
        ModelContract mc;
        if (!parse_contract(text, mc, error)) return false;
        if (m.contains("graph") &&
            !validate_io(mc, specs(m["graph"].at("inputs")), specs(m["graph"].at("outputs")), error))
            return false;
        contracts.push_back(mc);
    }
    latency = -1;
    if (contracts.size() == 2) {
        if (!validate_pair(contracts[0], contracts[1], error)) return false;
        latency = algorithmic_latency_samples(contracts[0], contracts[1]);
    }
    return true;
}

void test_valid_cases() {
    const auto files = json_files(g_data_dir / "contracts" / "valid");
    CHECK(!files.empty());
    for (const auto& path : files) {
        const json c = read_json(path);
        std::string error;
        int64_t latency = -1;
        const bool ok = run_case(c, error, latency);
        if (!ok) std::printf("%s: rejected: %s\n", path.filename().string().c_str(), error.c_str());
        CHECK(ok);
        if (ok && c.contains("algorithmic_latency_samples")) {
            const auto want = c["algorithmic_latency_samples"].get<int64_t>();
            if (latency != want)
                std::printf("%s: latency %lld, want %lld\n", path.filename().string().c_str(),
                            static_cast<long long>(latency), static_cast<long long>(want));
            CHECK(latency == want);
        }
    }
    std::printf("valid cases:   %zu\n", files.size());
}

void test_invalid_cases() {
    const auto files = json_files(g_data_dir / "contracts" / "invalid");
    CHECK(!files.empty());
    for (const auto& path : files) {
        const json c = read_json(path);
        const auto want = c.at("error").get<std::string>();
        std::string error;
        int64_t latency = -1;
        const bool ok = run_case(c, error, latency);
        const bool matched = !ok && error.find(want) != std::string::npos;
        if (!matched)
            std::printf("%s: %s, want \"%s\"\n", path.filename().string().c_str(),
                        ok ? "accepted" : ("got \"" + error + "\"").c_str(), want.c_str());
        CHECK(matched);
    }
    std::printf("invalid cases: %zu\n", files.size());
}

void test_fixture_manifest() {
    const fs::path dir = g_data_dir / "fixtures";
    const json manifest = read_json(dir / "manifest.json");
    CHECK(manifest.is_object());
    if (!manifest.is_object()) return;

    std::map<std::string, ModelContract> by_file;
    for (const auto& m : manifest.at("models")) {
        const auto file = m.at("file").get<std::string>();
        CHECK(fs::is_regular_file(dir / file));
        std::string text, error;
        if (m.at("contract").is_string()) text = m["contract"].get<std::string>();
        ModelContract mc;
        const bool ok = check_model(m["contract"].is_string() ? &text : nullptr, specs(m.at("graph").at("inputs")),
                                    specs(m.at("graph").at("outputs")), mc, error);
        if (m.at("error").is_null()) {
            if (!ok) std::printf("%s: rejected: %s\n", file.c_str(), error.c_str());
            CHECK(ok);
            by_file[file] = mc;
        } else {
            const auto want = m["error"].get<std::string>();
            if (ok || error != want) std::printf("%s: got \"%s\", want \"%s\"\n", file.c_str(), error.c_str(), want.c_str());
            CHECK(!ok && error == want);
        }
    }

    std::printf("fixture pairs (algorithmic latency):\n");
    size_t checked = 0;
    for (const auto& p : manifest.at("pairs")) {
        const auto name = p.at("name").get<std::string>();
        const auto enc = by_file.find(p.at("encoder").get<std::string>());
        const auto voc = by_file.find(p.at("vocoder").get<std::string>());
        CHECK(enc != by_file.end() && voc != by_file.end());
        if (enc == by_file.end() || voc == by_file.end()) continue;
        std::string error;
        CHECK(validate_pair(enc->second, voc->second, error));
        const int64_t latency = algorithmic_latency_samples(enc->second, voc->second);
        const auto hand = kFixtureLatency.find(name);
        std::printf("  %-12s %-26s + %-26s %5lld samples = %5.1f ms @ %d Hz\n", name.c_str(), enc->first.c_str(),
                    voc->first.c_str(), static_cast<long long>(latency),
                    1000.0 * static_cast<double>(latency) / enc->second.sample_rate, enc->second.sample_rate);
        CHECK(latency == p.at("algorithmic_latency_samples").get<int64_t>());
        CHECK(hand != kFixtureLatency.end() && hand->second == latency);
        ++checked;
    }
    CHECK(checked == kFixtureLatency.size());
}

void test_geometry_accessors() {
    ModelContract voc;
    voc.role = ModelRole::kVocoder;
    voc.sample_rate = 24000;
    voc.frame_samples = 1920;
    voc.frames_per_step = 2;
    voc.vocab = 2048;
    voc.cond_frame_samples = 480;
    voc.cond_inputs = {"logf0", "spk_emb"};
    voc.spk_emb_dim = 64;
    voc.overlap_samples = 240;
    voc.license = "test";
    std::string error;
    CHECK(validate_contract(voc, error));
    CHECK(voc.hop() == 3840);
    CHECK(voc.cond_frames() == 8);
    CHECK(voc.output_samples() == 4080);
    const auto in = voc.expected_inputs();
    CHECK(in.size() == 3);
    CHECK(in[0].name == "units" && in[0].dtype == TensorDtype::kI64 && in[0].shape == std::vector<int64_t>({1, 2}));
    CHECK(in[1].name == "logf0" && in[1].shape == std::vector<int64_t>({1, 8}));
    CHECK(in[2].name == "spk_emb" && in[2].shape == std::vector<int64_t>({1, 64}));
    CHECK(voc.expected_outputs()[0].shape == std::vector<int64_t>({1, 4080}));
    CHECK(format_shape({1, -1, 320}) == "[1,?,320]");
    CHECK(format_shape(TensorSpec{"s", TensorDtype::kF32, {}, false}) == "unknown");
    CHECK(format_shape(TensorSpec{"s", TensorDtype::kF32, {}, true}) == "[]");  // a scalar
}

void test_contracts_built_in_code() {
    ModelContract enc;
    enc.sample_rate = 16000;
    enc.frame_samples = 320;
    enc.frames_per_step = 1;
    enc.vocab = 100;
    enc.license = "test";
    std::string error;
    CHECK(validate_contract(enc, error));
    enc.overlap_samples = 160;  // a vocoder-only field on an encoder
    CHECK(!validate_contract(enc, error) && error == "key 'overlap_samples' is not valid for role 'encoder'");
    enc.overlap_samples = 0;
    enc.content_dim = 8;
    CHECK(!validate_contract(enc, error) && error == "key 'content_dim' is not valid for content_kind 'discrete'");
}

void test_missing_metadata() {
    ModelContract c;
    std::string error;
    CHECK(!check_model(nullptr, {}, {}, c, error));
    CHECK(error == "model has no 'fluxvoice.contract' metadata");
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_data_dir = argv[1];
    std::printf("test data: %s\n", g_data_dir.string().c_str());
    test_valid_cases();
    test_invalid_cases();
    test_fixture_manifest();
    test_geometry_accessors();
    test_contracts_built_in_code();
    test_missing_metadata();
    return test::finish("model_contract");
}
