// test_ort_model — ONNX Runtime C++ wrapper unit tests
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
//
// Exercises OrtModel against the deterministic Stage 02 fixture models:
//   * runtime version check + deliberate version-mismatch refusal
//   * shared library staged next to the executable
//   * identity_s{1,2,4}: bit-exact passthrough (both direct and via InferencePipeline)
//   * delay_s1: one-hop delay across steps (state round-trip)
//   * accumulator_s1: matches closed form over 1000 steps (ping-pong parity)
//   * broken/*.onnx: every broken-metadata fixture rejected with its exact manifest error
//   * reset() and warm_up(): restore the model to freshly loaded behaviour
//
// Usage: test_ort_model [tests/data/fixtures]
#include "pipeline/OrtModel.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/ModeAWorker.hpp"
#include "util/ProcessMemory.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef FLUXVOICE_FIXTURES_DIR
#define FLUXVOICE_FIXTURES_DIR "tests/data/fixtures"
#endif

namespace fs = std::filesystem;
using nlohmann::json;
using namespace fluxvoice;

namespace {

fs::path g_fixtures_dir = FLUXVOICE_FIXTURES_DIR;
fs::path g_exe_dir;

json read_json(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    json j = json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) std::printf("%s: not valid JSON\n", path.string().c_str());
    return j;
}

void test_version_and_staged_library() {
    CHECK(ort_available());
    const std::string ver = ort_runtime_version();
    std::printf("ONNX Runtime loaded version: %s (pinned: %s)\n", ver.c_str(), kPinnedOrtVersion);
    CHECK(ver == kPinnedOrtVersion);

    std::string err;
    CHECK(check_ort_version(ver, kPinnedOrtVersion, err));
    CHECK(!check_ort_version("1.17.1", kPinnedOrtVersion, err));
    CHECK(err.find("version mismatch") != std::string::npos);
    CHECK(err.find("System32") != std::string::npos);

    // Deliberately mismatched expected version must be refused by OrtModel::load().
    OrtModel m;
    const OrtStatus st = m.load(g_fixtures_dir / "identity" / "encoder_s1.onnx", "0.0.0-mismatch");
    CHECK(st == OrtStatus::kVersionMismatch);
    CHECK(!m.is_loaded());
    CHECK(m.last_error().find("ONNX Runtime version mismatch") != std::string::npos);
    CHECK(m.last_error().find("0.0.0-mismatch") != std::string::npos);

    // Verify the ONNX Runtime shared library was staged next to this executable.
    if (!g_exe_dir.empty()) {
#if defined(_WIN32)
        const fs::path shlib = g_exe_dir / "onnxruntime.dll";
#elif defined(__APPLE__)
        const fs::path shlib = g_exe_dir / "libonnxruntime.dylib";
#else
        const fs::path shlib = g_exe_dir / "libonnxruntime.so.1";
#endif
        if (!fs::exists(shlib)) {
            std::printf("missing staged ORT library next to executable: %s\n", shlib.string().c_str());
        }
        CHECK(fs::exists(shlib));
    }
}

void test_identity_bundle() {
    for (int fps : {1, 2, 4}) {
        const std::string tag = "s" + std::to_string(fps);
        OrtModel enc, voc;
        CHECK(enc.load(g_fixtures_dir / "identity" / ("encoder_" + tag + ".onnx")) == OrtStatus::kOk);
        CHECK(voc.load(g_fixtures_dir / "identity" / ("vocoder_" + tag + ".onnx")) == OrtStatus::kOk);
        CHECK(enc.is_loaded() && voc.is_loaded());

        std::string err;
        CHECK(validate_pair(enc.contract(), voc.contract(), err));
        CHECK(enc.warm_up(4) == OrtStatus::kOk);
        CHECK(voc.warm_up(4) == OrtStatus::kOk);
        CHECK(enc.active_bank() == 0);
        CHECK(voc.active_bank() == 0);

        const std::size_t hop = static_cast<std::size_t>(enc.contract().hop());
        const std::size_t cond_frames = static_cast<std::size_t>(voc.contract().cond_frames());
        CHECK(enc.audio_in().size() == hop);
        CHECK(voc.audio_out().size() == hop);
        CHECK(enc.content_out().size() == hop);
        CHECK(voc.content_in().size() == hop);
        CHECK(voc.logf0_in().size() == cond_frames);
        CHECK(voc.voicing_in().size() == cond_frames);
        CHECK(voc.aperiodicity_in().size() == cond_frames);
        CHECK(voc.energy_in().size() == cond_frames);
        CHECK(voc.spk_emb_in().size() == static_cast<std::size_t>(voc.contract().spk_emb_dim));

        std::fill(voc.spk_emb_in().begin(), voc.spk_emb_in().end(), 1.0f);
        std::fill(voc.logf0_in().begin(), voc.logf0_in().end(), 4.8f);
        std::fill(voc.voicing_in().begin(), voc.voicing_in().end(), 1.0f);
        std::fill(voc.aperiodicity_in().begin(), voc.aperiodicity_in().end(), 0.05f);
        std::fill(voc.energy_in().begin(), voc.energy_in().end(), -4.0f);

        std::vector<float> expected(hop);
        for (int step = 0; step < 8; ++step) {
            for (std::size_t i = 0; i < hop; ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>((step + 1) * 1315423911u + i * 2654435761u);
                expected[i] = static_cast<float>(static_cast<std::int32_t>(bits % 65536u) - 32768) / 32768.0f;
            }
            std::copy(expected.begin(), expected.end(), enc.audio_in().begin());
            CHECK(enc.step());
            std::copy(enc.content_out().begin(), enc.content_out().end(), voc.content_in().begin());
            CHECK(voc.step());
            CHECK(std::memcmp(voc.audio_out().data(), expected.data(), hop * sizeof(float)) == 0);
        }

        // Also verify OrtModel satisfies ModelStep inside InferencePipeline.
        const TargetVoiceStats voice{4.8, 0.25, -4.0, 1.5};
        const std::vector<float> spk(static_cast<std::size_t>(voc.contract().spk_emb_dim), 0.5f);
        InferencePipeline pipe(enc, voc, voice, spk);
        CHECK(pipe.warm_up(2));
        std::vector<float> out(hop, 0.0f);
        const auto res = pipe.process(expected, out);
        CHECK(res.ok && !res.stop);
        CHECK(std::memcmp(out.data(), expected.data(), hop * sizeof(float)) == 0);

        // Exercise the same callback/worker plumbing as --vc, without opening
        // hardware. Prefill is the only delay for these transparent models.
        DuplexRouter router;
        ModeAWorker worker(router, pipe);
        CHECK(worker.prepare());
        const auto target = pipe.governor().target_samples();
        for (std::size_t step = 0; step < 10; ++step) {
            router.process(out.data(), expected.data(), hop);
            CHECK(worker.pump());
            for (std::size_t i = 0; i < hop; ++i) {
                const auto sample = step * hop + i;
                CHECK(out[i] == (sample < target ? 0.0f : expected[(sample - target) % hop]));
            }
            const auto s = worker.snapshot();
            CHECK(s.underruns == 0 && s.overruns == 0 && s.drops == 0 && s.skipped == 0);
            CHECK(s.queued == static_cast<std::int64_t>(target));
        }
    }
    std::printf("identity bundle s1/s2/s4: bit-exact passthrough verified\n");
}

void test_delay_state_roundtrip() {
    OrtModel voc;
    CHECK(voc.load(g_fixtures_dir / "delay_s1.onnx") == OrtStatus::kOk);
    CHECK(voc.contract().lookahead_frames == 1);
    CHECK(voc.contract().states == std::vector<std::string>{"buf"});

    const std::size_t hop = static_cast<std::size_t>(voc.contract().hop());
    CHECK(hop == 320);
    std::vector<float> prev(hop, 0.0f);
    std::vector<float> curr(hop, 0.0f);

    for (int step = 0; step < 12; ++step) {
        CHECK(voc.active_bank() == static_cast<std::size_t>(step & 1));
        for (std::size_t i = 0; i < hop; ++i) {
            curr[i] = static_cast<float>((step + 1) * 1000 + static_cast<int>(i));
        }
        std::copy(curr.begin(), curr.end(), voc.content_in().begin());
        CHECK(voc.step_status() == OrtStatus::kOk);
        CHECK(voc.active_bank() == static_cast<std::size_t>((step + 1) & 1));
        CHECK(std::memcmp(voc.audio_out().data(), prev.data(), hop * sizeof(float)) == 0);
        prev = curr;
    }
    std::printf("delay_s1: one-hop state round-trip verified over 12 steps\n");
}

void test_accumulator_1000_steps() {
    OrtModel enc;
    CHECK(enc.load(g_fixtures_dir / "accumulator_s1.onnx") == OrtStatus::kOk);
    CHECK(enc.warm_up(4) == OrtStatus::kOk);
    CHECK(enc.active_bank() == 0);

    // 320 samples of 0.25f -> exact float32 sum of 80.0f per step.
    auto audio = enc.audio_in();
    CHECK(audio.size() == 320);
    std::fill(audio.begin(), audio.end(), 0.25f);

    const float* const bank_a_ptr = enc.state_bank(0).find("acc")->as_f32();
    const float* const bank_b_ptr = enc.state_bank(1).find("acc")->as_f32();
    CHECK(bank_a_ptr != nullptr && bank_b_ptr != nullptr && bank_a_ptr != bank_b_ptr);

    for (int step = 1; step <= 1000; ++step) {
        CHECK(enc.step_status() == OrtStatus::kOk);
        const float want = 80.0f * static_cast<float>(step);
        CHECK(enc.content_out().size() == 1);
        CHECK(enc.content_out()[0] == want);
        CHECK(enc.active_bank() == static_cast<std::size_t>(step & 1));
        CHECK(enc.active_state_bank().find("acc")->span_f32()[0] == want);
        // State buffer addresses must never reallocate across 1000 ping-pong hops.
        CHECK(enc.state_bank(0).find("acc")->as_f32() == bank_a_ptr);
        CHECK(enc.state_bank(1).find("acc")->as_f32() == bank_b_ptr);
    }
    std::printf("accumulator_s1: 1000-step closed form verified (final = %.1f)\n",
                enc.content_out()[0]);
}

void test_reset_equivalence() {
    OrtModel fresh_acc, used_acc;
    CHECK(fresh_acc.load(g_fixtures_dir / "accumulator_s1.onnx") == OrtStatus::kOk);
    CHECK(used_acc.load(g_fixtures_dir / "accumulator_s1.onnx") == OrtStatus::kOk);

    // Dirty used_acc for an odd number of steps so parity_ == 1 and both banks are non-zero.
    std::fill(used_acc.audio_in().begin(), used_acc.audio_in().end(), 1.5f);
    for (int i = 0; i < 7; ++i) CHECK(used_acc.step());
    CHECK(used_acc.active_bank() == 1);
    CHECK(used_acc.state_bank(0).find("acc")->span_f32()[0] != 0.0f);
    CHECK(used_acc.state_bank(1).find("acc")->span_f32()[0] != 0.0f);

    used_acc.reset();
    CHECK(used_acc.active_bank() == 0);
    CHECK(used_acc.state_bank(0).find("acc")->span_f32()[0] == 0.0f);
    CHECK(used_acc.state_bank(1).find("acc")->span_f32()[0] == 0.0f);

    for (int step = 0; step < 25; ++step) {
        const float val = 0.125f * static_cast<float>((step % 5) + 1);
        std::fill(fresh_acc.audio_in().begin(), fresh_acc.audio_in().end(), val);
        std::fill(used_acc.audio_in().begin(), used_acc.audio_in().end(), val);
        CHECK(fresh_acc.step() && used_acc.step());
        CHECK(fresh_acc.content_out()[0] == used_acc.content_out()[0]);
    }

    // Same check on delay_s1 after odd number of steps.
    OrtModel fresh_del, used_del;
    CHECK(fresh_del.load(g_fixtures_dir / "delay_s1.onnx") == OrtStatus::kOk);
    CHECK(used_del.load(g_fixtures_dir / "delay_s1.onnx") == OrtStatus::kOk);
    std::fill(used_del.content_in().begin(), used_del.content_in().end(), 99.0f);
    for (int i = 0; i < 3; ++i) CHECK(used_del.step());
    used_del.reset();
    CHECK(used_del.active_bank() == 0);

    for (int step = 0; step < 6; ++step) {
        std::fill(fresh_del.content_in().begin(), fresh_del.content_in().end(), static_cast<float>(step + 1));
        std::fill(used_del.content_in().begin(), used_del.content_in().end(), static_cast<float>(step + 1));
        CHECK(fresh_del.step() && used_del.step());
        CHECK(std::memcmp(fresh_del.audio_out().data(), used_del.audio_out().data(),
                          fresh_del.audio_out().size() * sizeof(float)) == 0);
    }
    std::printf("reset() equivalence verified on stateful encoder and vocoder\n");
}

void test_broken_fixtures_rejected() {
    const json manifest = read_json(g_fixtures_dir / "manifest.json");
    CHECK(manifest.is_object());
    if (!manifest.is_object()) return;

    std::size_t broken_count = 0;
    for (const auto& m : manifest.at("models")) {
        if (m.at("error").is_null()) continue;
        const std::string file = m.at("file").get<std::string>();
        const std::string want = m.at("error").get<std::string>();

        OrtModel model;
        const OrtStatus st = model.load(g_fixtures_dir / file);
        const bool matched = (st == OrtStatus::kContractError) && !model.is_loaded() &&
                             (model.last_error() == want);
        if (!matched) {
            std::printf("%s: status=%s, got \"%.*s\", want \"%s\"\n",
                        file.c_str(), to_string(st),
                        static_cast<int>(model.last_error().size()), model.last_error().data(),
                        want.c_str());
        }
        CHECK(matched);
        ++broken_count;
    }
    CHECK(broken_count >= 10);

    // Unloaded model and missing file handling.
    OrtModel unloaded;
    CHECK(unloaded.step_status() == OrtStatus::kNotLoaded);
    CHECK(unloaded.warm_up(1) == OrtStatus::kNotLoaded);
    CHECK(unloaded.load(g_fixtures_dir / "does_not_exist.onnx") == OrtStatus::kOrtError);
    std::printf("broken fixtures rejected: %zu\n", broken_count);
}

void test_rss_stability_10000_steps() {
    OrtModel enc, voc;
    CHECK(enc.load(g_fixtures_dir / "identity" / "encoder_s1.onnx") == OrtStatus::kOk);
    CHECK(voc.load(g_fixtures_dir / "delay_s1.onnx") == OrtStatus::kOk);

    const TargetVoiceStats voice{4.8, 0.25, -4.0, 1.5};
    InferencePipeline pipe(enc, voc, voice, {});
    CHECK(pipe.warm_up(20));

    const std::size_t hop = static_cast<std::size_t>(enc.contract().hop());
    std::vector<float> in(hop, 0.125f);
    std::vector<float> out(hop, 0.0f);

    const std::size_t rss_before = resident_bytes();
    for (int step = 0; step < 10000; ++step) {
        in[0] = static_cast<float>((step % 17) - 8) * 0.05f;
        const auto res = pipe.process(in, out);
        CHECK(res.ok);
    }
    const std::size_t rss_after = resident_bytes();
    const std::int64_t delta_bytes =
        static_cast<std::int64_t>(rss_after) - static_cast<std::int64_t>(rss_before);
    std::printf("10,000-step RSS: before=%zu KB, after=%zu KB, delta=%+lld KB\n",
                rss_before / 1024, rss_after / 1024,
                static_cast<long long>(delta_bytes / 1024));
    // Stage 07 DoD: RSS grows < 1 MB over 10,000 steps after warm-up.
    CHECK(delta_bytes < 1024 * 1024);
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 0 && argv[0] && argv[0][0] != '\0') {
        std::error_code ec;
        const fs::path exe = fs::weakly_canonical(fs::path(argv[0]), ec);
        if (!ec) g_exe_dir = exe.parent_path();
    }
    if (argc > 1) g_fixtures_dir = argv[1];
    std::printf("fixtures dir: %s\n", g_fixtures_dir.string().c_str());

    test_version_and_staged_library();
    test_identity_bundle();
    test_delay_state_roundtrip();
    test_accumulator_1000_steps();
    test_reset_equivalence();
    test_broken_fixtures_rejected();
    test_rss_stability_10000_steps();
    return test::finish("ort_model");
}
