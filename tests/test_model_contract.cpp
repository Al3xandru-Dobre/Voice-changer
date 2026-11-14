// ModelContract — contract v1 & v2 parser/validator tests
// (Milestone 2 / Stage 02 & Milestone 3 / Stage 02).
//
// Runs the shared cases in tests/data/contracts/ that python_training/tests/
// test_contract.py also runs, so the two validators accept and reject the same
// contracts with the same messages. Then checks the fixture manifest written by
// tools/make_fixture_models.py (Mode A v1 pairs + Mode B v2 S2UT quartets +
// langs.json), and verifies hand-computed latency formulas. No ONNX Runtime.
//
// Usage: test_model_contract [tests/data]
#include "pipeline/ModelContract.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

// Hand-computed Mode A latencies in Documents/Milestone 2/02_model_contract_fixtures.md:
// hop + (L_enc + L_voc) x frame_samples + overlap_samples.
const std::map<std::string, std::int64_t> kFixtureLatency = {
    {"identity_s1", 320}, {"identity_s2", 640}, {"identity_s4", 1280}, {"delay_s1", 640}};

// Hand-computed Mode B onset latencies in Documents/Milestone 3/02_translator_contract_v2_fixtures.md:
// wait_k * chunk_frames * frame_samples + (L_enc + L_voc) * frame_samples + overlap_samples
// With chunk_frames = 4, frame_samples = 320, L_enc = 0, L_voc = 0, overlap = 0:
// k=1 -> 1280 (80 ms), k=3 -> 3840 (240 ms), k=5 -> 6400 (400 ms), k=10 -> 12800 (800 ms).
const std::map<int, std::int64_t> kFixtureS2utOnset = {
    {1, 1280}, {3, 3840}, {5, 6400}, {10, 12800}};

std::string slurp(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

json read_json(const fs::path& path) {
    json j = json::parse(slurp(path), nullptr, /*allow_exceptions=*/false);
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
        if (t.at("shape").is_null()) s.has_shape = false;
        else s.shape = t["shape"].get<std::vector<std::int64_t>>();
        out.push_back(std::move(s));
    }
    return out;
}

/// Parse every model, cross-check graphs, validate Mode A pair (2 models) or
/// Mode B S2UT quartet (4 models). Returns false with the first error.
bool run_case(const json& c, std::string& error, std::int64_t& latency, std::int64_t& s2ut_onset) {
    std::vector<ModelContract> contracts;
    std::vector<std::vector<TensorSpec>> all_ins, all_outs;
    std::vector<bool> has_graph;
    for (const auto& m : c.at("models")) {
        const json& raw = m.at("contract");
        const std::string text = raw.is_string() ? raw.get<std::string>() : raw.dump();
        ModelContract mc;
        if (!parse_contract(text, mc, error)) return false;
        std::vector<TensorSpec> ins, outs;
        const bool g = m.contains("graph");
        if (g) {
            ins = specs(m["graph"].at("inputs"));
            outs = specs(m["graph"].at("outputs"));
            if (!validate_io(mc, ins, outs, error)) return false;
        }
        contracts.push_back(std::move(mc));
        all_ins.push_back(std::move(ins));
        all_outs.push_back(std::move(outs));
        has_graph.push_back(g);
    }
    latency = -1;
    s2ut_onset = -1;
    if (contracts.size() == 2) {
        if (!validate_pair(contracts[0], contracts[1], error)) return false;
        latency = algorithmic_latency_samples(contracts[0], contracts[1]);
    } else if (contracts.size() == 4) {
        if (!validate_s2ut_bundle(contracts[0], contracts[1], contracts[2], contracts[3], error)) return false;
        if (has_graph[1] && has_graph[2] &&
            !validate_s2ut_shared_io(contracts[1], all_outs[1], contracts[2], all_ins[2], error))
            return false;
        const int wait_k = c.contains("wait_k") ? c["wait_k"].get<int>() : 3;
        s2ut_onset = s2ut_onset_latency_samples(contracts[0], contracts[1], contracts[2], contracts[3], wait_k);
    }
    return true;
}

void test_valid_cases() {
    const auto files = json_files(g_data_dir / "contracts" / "valid");
    CHECK(!files.empty());
    for (const auto& path : files) {
        const json c = read_json(path);
        std::string error;
        std::int64_t latency = -1, s2ut_onset = -1;
        const bool ok = run_case(c, error, latency, s2ut_onset);
        if (!ok) std::printf("%s: rejected: %s\n", path.filename().string().c_str(), error.c_str());
        CHECK(ok);
        if (ok && c.contains("algorithmic_latency_samples")) {
            const auto want = c["algorithmic_latency_samples"].get<std::int64_t>();
            if (latency != want)
                std::printf("%s: latency %lld, want %lld\n", path.filename().string().c_str(),
                            static_cast<long long>(latency), static_cast<long long>(want));
            CHECK(latency == want);
        }
        if (ok && c.contains("s2ut_onset_latency_samples")) {
            const auto want = c["s2ut_onset_latency_samples"].get<std::int64_t>();
            if (s2ut_onset != want)
                std::printf("%s: s2ut onset %lld, want %lld\n", path.filename().string().c_str(),
                            static_cast<long long>(s2ut_onset), static_cast<long long>(want));
            CHECK(s2ut_onset == want);
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
        std::int64_t latency = -1, s2ut_onset = -1;
        const bool ok = run_case(c, error, latency, s2ut_onset);
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
    std::map<std::string, std::pair<std::vector<TensorSpec>, std::vector<TensorSpec>>> graphs_by_file;
    for (const auto& m : manifest.at("models")) {
        const auto file = m.at("file").get<std::string>();
        CHECK(fs::is_regular_file(dir / file));
        std::string text, error;
        if (m.at("contract").is_string()) text = m["contract"].get<std::string>();
        const auto ins = specs(m.at("graph").at("inputs"));
        const auto outs = specs(m.at("graph").at("outputs"));
        graphs_by_file[file] = {ins, outs};
        ModelContract mc;
        const bool ok = check_model(m["contract"].is_string() ? &text : nullptr, ins, outs, mc, error);
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

    std::printf("fixture Mode A pairs (algorithmic latency):\n");
    std::size_t checked = 0;
    for (const auto& p : manifest.at("pairs")) {
        const auto name = p.at("name").get<std::string>();
        const auto enc = by_file.find(p.at("encoder").get<std::string>());
        const auto voc = by_file.find(p.at("vocoder").get<std::string>());
        CHECK(enc != by_file.end() && voc != by_file.end());
        if (enc == by_file.end() || voc == by_file.end()) continue;
        std::string error;
        CHECK(validate_pair(enc->second, voc->second, error));
        const std::int64_t latency = algorithmic_latency_samples(enc->second, voc->second);
        const auto hand = kFixtureLatency.find(name);
        std::printf("  %-12s %-26s + %-26s %5lld samples = %5.1f ms @ %d Hz\n", name.c_str(), enc->first.c_str(),
                    voc->first.c_str(), static_cast<long long>(latency),
                    1000.0 * static_cast<double>(latency) / enc->second.sample_rate, enc->second.sample_rate);
        CHECK(latency == p.at("algorithmic_latency_samples").get<std::int64_t>());
        CHECK(hand != kFixtureLatency.end() && hand->second == latency);
        ++checked;
    }
    CHECK(checked == kFixtureLatency.size());

    std::printf("fixture Mode B S2UT quartets (onset latency for k in {1, 3, 5, 10}):\n");
    CHECK(manifest.contains("s2ut_quartets") && !manifest["s2ut_quartets"].empty());
    for (const auto& q : manifest.at("s2ut_quartets")) {
        const auto name = q.at("name").get<std::string>();
        const auto enc = by_file.find(q.at("encoder").get<std::string>());
        const auto t_enc = by_file.find(q.at("translator_enc").get<std::string>());
        const auto t_dec = by_file.find(q.at("translator_dec").get<std::string>());
        const auto voc = by_file.find(q.at("vocoder").get<std::string>());
        CHECK(enc != by_file.end() && t_enc != by_file.end() && t_dec != by_file.end() && voc != by_file.end());
        if (enc == by_file.end() || t_enc == by_file.end() || t_dec == by_file.end() || voc == by_file.end()) continue;
        std::string error;
        CHECK(validate_s2ut_bundle(enc->second, t_enc->second, t_dec->second, voc->second, error));
        CHECK(validate_s2ut_shared_io(t_enc->second, graphs_by_file[t_enc->first].second, t_dec->second,
                                      graphs_by_file[t_dec->first].first, error));
        for (const auto& [k, want] : kFixtureS2utOnset) {
            const std::int64_t got =
                s2ut_onset_latency_samples(enc->second, t_enc->second, t_dec->second, voc->second, k);
            CHECK(got == want);
            CHECK(got == q.at("onset_latency_samples").at(std::to_string(k)).get<std::int64_t>());
        }
        std::printf("  %-12s k=1:%5lld  k=3:%5lld  k=5:%5lld  k=10:%6lld samples\n", name.c_str(),
                    static_cast<long long>(kFixtureS2utOnset.at(1)), static_cast<long long>(kFixtureS2utOnset.at(3)),
                    static_cast<long long>(kFixtureS2utOnset.at(5)), static_cast<long long>(kFixtureS2utOnset.at(10)));
    }

    for (const auto& b : manifest.at("bundles")) {
        if (b.contains("langs")) {
            const auto langs_file = b["langs"].get<std::string>();
            const auto trans_file = b.at("translator").get<std::string>();
            std::string error;
            CHECK(validate_langs_json(slurp(dir / langs_file), &by_file.at(trans_file), error));
        }
    }
}

void test_s2ut_onset_with_lookahead_and_overlap() {
    // Master-plan Mode B algorithmic onset table (00_master.md §Latency budget):
    // L_enc = 1 (320 samples), L_voc = 0, overlap = 0, chunk_frames = 4:
    // k=1 -> 1600 (100 ms), k=3 -> 4160 (260 ms), k=5 -> 6720 (420 ms), k=10 -> 13120 (820 ms).
    ModelContract enc;
    enc.contract_version = 2;
    enc.role = ModelRole::kEncoder;
    enc.sample_rate = 16000;
    enc.frame_samples = 320;
    enc.frames_per_step = 4;
    enc.lookahead_frames = 1;
    enc.vocab = 1000;
    enc.latent_dim = 128;
    enc.license = "MIT";

    ModelContract t_enc;
    t_enc.contract_version = 2;
    t_enc.role = ModelRole::kTranslatorEnc;
    t_enc.sample_rate = 16000;
    t_enc.frame_samples = 320;
    t_enc.chunk_frames = 4;
    t_enc.latent_dim = 128;
    t_enc.max_src_steps = 128;
    t_enc.shared_mem_states = {"mem_k_0", "mem_v_0"};
    t_enc.langs = {{"en", 0}, {"es", 1}, {"fr", 2}, {"de", 3}};
    t_enc.states = {"src_pos", "mem_k_0", "mem_v_0"};
    t_enc.license = "MIT";

    ModelContract t_dec;
    t_dec.contract_version = 2;
    t_dec.role = ModelRole::kTranslatorDec;
    t_dec.sample_rate = 16000;
    t_dec.frame_samples = 320;
    t_dec.chunk_frames = 4;
    t_dec.vocab = 1000;
    t_dec.special_units = {1000, 1001, 1002};
    t_dec.max_src_steps = 128;
    t_dec.max_tgt_steps = 128;
    t_dec.shared_mem_states = {"mem_k_0", "mem_v_0"};
    t_dec.langs = t_enc.langs;
    t_dec.has_prosody_output = true;
    t_dec.states = {"tgt_pos"};
    t_dec.license = "MIT";

    ModelContract voc;
    voc.contract_version = 2;
    voc.role = ModelRole::kVocoder;
    voc.sample_rate = 16000;
    voc.frame_samples = 320;
    voc.frames_per_step = 4;
    voc.lookahead_frames = 0;
    voc.vocab = 1000;
    voc.cond_frame_samples = 320;
    voc.overlap_samples = 0;
    voc.license = "MIT";

    std::string error;
    CHECK(validate_contract(enc, error));
    CHECK(validate_contract(t_enc, error));
    CHECK(validate_contract(t_dec, error));
    CHECK(validate_contract(voc, error));
    CHECK(validate_s2ut_bundle(enc, t_enc, t_dec, voc, error));

    CHECK(s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, 1) == 1600);
    CHECK(s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, 3) == 4160);
    CHECK(s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, 5) == 6720);
    CHECK(s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, 10) == 13120);

    // Frontend's shipped langs.default.json must agree with this 4-language v2 contract!
    const fs::path default_langs = g_data_dir.parent_path().parent_path() / "frontend" / "assets" / "langs.default.json";
    if (fs::is_regular_file(default_langs)) {
        CHECK(validate_langs_json(slurp(default_langs), &t_enc, error));
    }
    CHECK(!validate_langs_json(R"({"schema":1,"langs":[{"code":"en","id":0,"name":"English"}]})", &t_enc, error));
    CHECK(error == "langs.json does not match translator contract langs");
    CHECK(!validate_langs_json(
        R"({"schema":1,"langs":[{"code":"en","id":0,"name":"English"},{"code":"es","id":1,"name":"Spanish"}],"default_wait_k":{"en->en":3}})",
        nullptr, error));
    CHECK(error.find("invalid language pair 'en->en'") != std::string::npos);
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
    CHECK(in[0].name == "units" && in[0].dtype == TensorDtype::kI64 && in[0].shape == std::vector<std::int64_t>({1, 2}));
    CHECK(in[1].name == "logf0" && in[1].shape == std::vector<std::int64_t>({1, 8}));
    CHECK(in[2].name == "spk_emb" && in[2].shape == std::vector<std::int64_t>({1, 64}));
    CHECK(voc.expected_outputs()[0].shape == std::vector<std::int64_t>({1, 4080}));
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
    enc.content_dim = 0;
    enc.latent_dim = 64;
    CHECK(!validate_contract(enc, error) && error == "key 'latent_dim' requires contract_version 2");
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
    test_s2ut_onset_with_lookahead_and_overlap();
    test_geometry_accessors();
    test_contracts_built_in_code();
    test_missing_metadata();
    return test::finish("model_contract");
}
