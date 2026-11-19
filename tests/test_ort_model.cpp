// test_ort_model — ONNX Runtime C++ wrapper unit tests
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md;
// Milestone 3 / Stage 07, Documents/Milestone 3/07_cpp_translator_runtime_offline_runner.md).
//
// Exercises OrtModel against the deterministic Stage 02 fixture models:
//   * runtime version check + deliberate version-mismatch refusal
//   * shared library staged next to the executable
//   * identity_s{1,2,4}: bit-exact passthrough (both direct and via InferencePipeline)
//   * delay_s1: one-hop delay across steps (state round-trip)
//   * accumulator_s1: matches closed form over 1000 steps (ping-pong parity)
//   * broken/*.onnx: every broken-metadata fixture rejected with its exact manifest error
//   * reset() and warm_up(): restore the model to freshly loaded behaviour
//   * s2ut/encoder_s{1,2,4}: v2 dual outputs (units + latents) match the fixture arithmetic
//   * s2ut/ translator pair: zero-copy mem_*/src_pos sharing through the 2 x 2
//     [translator_enc bank][translator_dec parity] IoBinding table, the k = 3
//     schedule of python_training/tests/test_fixtures.py, reset() and
//     warm_up_translator_pair() reproducibility, reload detection, RSS
//   * s2ut_mismatch/: valid models that break the quartet (manifest "s2ut_mismatch")
//
// Usage: test_ort_model [tests/data/fixtures]
#include "pipeline/OrtModel.hpp"
#include "pipeline/InferencePipeline.hpp"
#include "pipeline/ModelBundle.hpp"
#include "pipeline/ModeAWorker.hpp"
#include "util/ProcessMemory.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
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
        PipelineOptions pipe_options;
        pipe_options.output_chain = false; // bit-exact ORT parity check
        InferencePipeline pipe(enc, voc, voice, spk, pipe_options);
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
    PipelineOptions pipe_options;
    pipe_options.output_chain = false; // bit-exact delay-model plumbing
    InferencePipeline pipe(enc, voc, voice, {}, pipe_options);
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

// ---------------------------------------------------------------------------
// Milestone 3 / Stage 07: contract v2 dual encoder and the two-graph S2UT
// translator (tests/data/fixtures/s2ut/, tools/make_fixture_models.py).
// ---------------------------------------------------------------------------

constexpr std::size_t kS2utLatent = 8;        // latent_dim
constexpr std::size_t kS2utChunkFrames = 4;   // chunk_frames (80 ms)
constexpr std::size_t kS2utRing = 4;          // max_src_steps: ring slots of mem_k_0 / mem_v_0
constexpr std::size_t kS2utFrame = 320;       // frame_samples
constexpr std::int64_t kS2utVocab = 16;
constexpr std::int64_t kS2utBos = 16;         // special_units.bos
constexpr std::int64_t kSrcLangEs = 1;        // langs["es"]
constexpr std::int64_t kTgtLangDe = 3;        // langs["de"]
constexpr float kProsodyCtx0 = 0.5f;
constexpr float kProsodyCtx1 = -0.25f;
constexpr int kSourceChunks = 10;             // 2.5 x the 4-slot ring

using Ring = std::array<std::array<float, kS2utLatent>, kS2utRing>;

/// Constant value of source chunk i (python_training/tests/test_fixtures.py).
float chunk_value(int i) { return 0.125f * static_cast<float>(i + 1); }

bool has_text(std::string_view text, std::string_view part) { return text.find(part) != std::string_view::npos; }

/// translator_enc's shared cross-attention memory plus its position counter.
std::vector<std::string> shared_names_for(const OrtModel& trans_enc) {
    std::vector<std::string> names = trans_enc.contract().shared_mem_states;
    names.push_back("src_pos");
    return names;
}

std::int64_t bank_i64(const OrtModel& m, std::size_t bank, std::string_view name) {
    const KVStateCache::Buffer* b = m.state_bank(bank).find(name);
    return b && b->span_i64().size() == 1 ? b->span_i64()[0] : -1;
}

std::span<const float> bank_f32(const OrtModel& m, std::size_t bank, std::string_view name) {
    const KVStateCache::Buffer* b = m.state_bank(bank).find(name);
    return b ? b->span_f32() : std::span<const float>{};
}

/// Raw bytes of every state tensor in both banks (contract order).
std::vector<std::uint8_t> bank_bytes(const OrtModel& m) {
    std::vector<std::uint8_t> bytes;
    for (std::size_t bank = 0; bank < 2; ++bank) {
        for (const std::string& name : m.contract().states) {
            const KVStateCache::Buffer* b = m.state_bank(bank).find(name);
            if (!b) continue;
            const auto raw = b->span_bytes();
            bytes.insert(bytes.end(), raw.begin(), raw.end());
        }
    }
    return bytes;
}

bool all_zero(const std::vector<std::uint8_t>& bytes) {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t x) { return x == 0; });
}

bool outputs_zero(const OrtModel& m) {
    for (const TensorSpec& spec : m.contract().expected_outputs()) {
        for (float x : m.output_f32(spec.name))
            if (x != 0.0f) return false;
        for (std::int64_t x : m.output_i64(spec.name))
            if (x != 0) return false;
    }
    return true;
}

/// Own (non-shared) inputs are all zero; shared inputs read as empty spans.
bool inputs_zero(const OrtModel& m) {
    for (const TensorSpec& spec : m.contract().expected_inputs()) {
        for (float x : m.input_f32(spec.name))
            if (x != 0.0f) return false;
        for (std::int64_t x : m.input_i64(spec.name))
            if (x != 0) return false;
    }
    return true;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

/// translator_enc bank `bank` holds exactly src_pos, enc_conv (all `conv`) and
/// the expected mem_k_0 / mem_v_0 rings.
bool enc_bank_holds(const OrtModel& tenc, std::size_t bank, std::int64_t src_pos, float conv,
                    const Ring& ring_k, const Ring& ring_v) {
    const auto enc_conv = bank_f32(tenc, bank, "enc_conv");
    const auto mem_k = bank_f32(tenc, bank, "mem_k_0");
    const auto mem_v = bank_f32(tenc, bank, "mem_v_0");
    bool ok = bank_i64(tenc, bank, "src_pos") == src_pos && enc_conv.size() == kS2utLatent &&
              mem_k.size() == kS2utRing * kS2utLatent && mem_v.size() == mem_k.size();
    for (std::size_t j = 0; ok && j < kS2utLatent; ++j) ok = enc_conv[j] == conv;
    for (std::size_t s = 0; ok && s < kS2utRing; ++s)
        for (std::size_t j = 0; ok && j < kS2utLatent; ++j)
            ok = mem_k[s * kS2utLatent + j] == ring_k[s][j] && mem_v[s * kS2utLatent + j] == ring_v[s][j];
    if (!ok) {
        std::printf("translator_enc bank %zu: src_pos %lld (want %lld) or enc_conv/mem_* differ\n", bank,
                    static_cast<long long>(bank_i64(tenc, bank, "src_pos")), static_cast<long long>(src_pos));
    }
    return ok;
}

/// translator_dec bank `bank` holds exactly tgt_pos and the dec_kv_0 ring.
bool dec_bank_holds(const OrtModel& tdec, std::size_t bank, std::int64_t tgt_pos, const Ring& dec_kv) {
    const auto kv = bank_f32(tdec, bank, "dec_kv_0");
    bool ok = bank_i64(tdec, bank, "tgt_pos") == tgt_pos && kv.size() == kS2utRing * kS2utLatent;
    for (std::size_t s = 0; ok && s < kS2utRing; ++s)
        for (std::size_t j = 0; ok && j < kS2utLatent; ++j) ok = kv[s * kS2utLatent + j] == dec_kv[s][j];
    if (!ok) {
        std::printf("translator_dec bank %zu: tgt_pos %lld (want %lld) or dec_kv_0 differs\n", bank,
                    static_cast<long long>(bank_i64(tdec, bank, "tgt_pos")), static_cast<long long>(tgt_pos));
    }
    return ok;
}

/// The s2ut/ fixture quartet. bind() shares translator_enc's mem_* and src_pos
/// with translator_dec; a bound quartet is never moved (bind_shared_inputs()).
struct S2utQuartet {
    OrtModel enc, tenc, tdec, voc;

    bool load(int fps = 4) {
        const fs::path dir = g_fixtures_dir / "s2ut";
        const std::string tag = "s" + std::to_string(fps);
        return enc.load(dir / ("encoder_" + tag + ".onnx")) == OrtStatus::kOk &&
               tenc.load(dir / "translator_enc.onnx") == OrtStatus::kOk &&
               tdec.load(dir / "translator_dec.onnx") == OrtStatus::kOk &&
               voc.load(dir / ("vocoder_" + tag + ".onnx")) == OrtStatus::kOk;
    }

    bool bind() { return tdec.bind_shared_inputs(tenc, shared_names_for(tenc)) == OrtStatus::kOk; }
};

/// Everything a quartet run emitted, for bit-exact run-to-run comparisons.
struct QuartetTrace {
    std::vector<float> readiness;           // translator_enc, one per source chunk
    std::vector<std::int64_t> units;        // translator_dec, chunk_frames per decoder step
    std::vector<float> prosody;             // translator_dec, chunk_frames x 4 per decoder step
    std::vector<float> eos;                 // translator_dec, one per decoder step
    std::vector<float> audio;               // vocoder, one hop per decoder step
    std::vector<float> decoded;             // v-mean of the slot read by each non-final step
    std::vector<float> flush_eos;           // eos_prob of the kFlush steps
    std::array<std::array<int, 2>, 2> combos{};  // decoder steps per [translator_enc bank][translator_dec parity]
};

bool same_trace(const QuartetTrace& a, const QuartetTrace& b) {
    return same_bits(a.readiness, b.readiness) && a.units == b.units && same_bits(a.prosody, b.prosody) &&
           same_bits(a.eos, b.eos) && same_bits(a.audio, b.audio) && same_bits(a.decoded, b.decoded) &&
           same_bits(a.flush_eos, b.flush_eos) && a.combos == b.combos;
}

/// python_training/tests/test_fixtures.py's schedule on the C++ runtime, for a
/// freshly loaded or reset bound quartet: 10 source chunks of constant
/// 0.125 * (i + 1) through encoder_s4 -> translator_enc; translator_dec +
/// vocoder whenever src_pos - tgt_pos >= wait_k (kWrite; otherwise kRead skips
/// both); then kFlush decoder steps (src_pos fixed) until eos_prob == 1. Every
/// tensor is checked against the fixture arithmetic on the way.
QuartetTrace run_quartet(S2utQuartet& q, int wait_k) {
    QuartetTrace t;
    q.tenc.input_i64("src_lang")[0] = kSrcLangEs;
    q.tdec.input_i64("tgt_lang")[0] = kTgtLangDe;
    const auto ctx = q.tdec.input_f32("prosody_ctx");
    const auto prev_units = q.tdec.input_i64("prev_units");
    const auto latents_in = q.tenc.input_f32("latents");
    const auto dec_units = q.tdec.output_i64("units");
    const auto prosody = q.tdec.output_f32("prosody");
    const auto eos_prob = q.tdec.output_f32("eos_prob");
    const auto voc_units = q.voc.units_in();
    const auto audio = q.voc.audio_out();
    const bool sizes_ok = ctx.size() == 2 && prev_units.size() == kS2utChunkFrames &&
                          latents_in.size() == kS2utChunkFrames * kS2utLatent &&
                          q.enc.latents_out().size() == latents_in.size() &&
                          dec_units.size() == kS2utChunkFrames && prosody.size() == kS2utChunkFrames * 4 &&
                          eos_prob.size() == 1 && voc_units.size() == kS2utChunkFrames &&
                          q.voc.voicing_in().size() == kS2utChunkFrames &&
                          audio.size() == kS2utChunkFrames * kS2utFrame;
    CHECK(sizes_ok);
    if (!sizes_ok) return t;
    ctx[0] = kProsodyCtx0;
    ctx[1] = kProsodyCtx1;
    std::fill(prev_units.begin(), prev_units.end(), kS2utBos);
    const auto spk_emb = q.voc.spk_emb_in();
    std::fill(spk_emb.begin(), spk_emb.end(), 0.0f);

    Ring ring_k{}, ring_v{}, dec_kv{};
    std::int64_t unit = kS2utBos;  // every frame of prev_units holds the same id

    const auto decoder_step = [&]() -> float {
        const std::size_t e = q.tenc.active_bank();
        const std::size_t d = q.tdec.active_bank();
        ++t.combos[e][d];
        const std::int64_t src = bank_i64(q.tenc, e, "src_pos");
        const std::int64_t tgt = bank_i64(q.tdec, d, "tgt_pos");
        // The fixture reads ring slot min(tgt_pos, max(src_pos - 1, 0)) mod 4
        // of the newest translator_enc bank: (src_pos - k) mod 4 in steady wait-k.
        const auto slot = static_cast<std::size_t>(std::min(tgt, std::max<std::int64_t>(src - 1, 0)) % 4);
        const std::vector<std::uint8_t> shared_before = bank_bytes(q.tenc);
        CHECK(q.tdec.step_status() == OrtStatus::kOk);
        CHECK(bank_bytes(q.tenc) == shared_before);  // read-only for the decoder
        CHECK(q.tenc.active_bank() == e && q.tdec.active_bank() == (d ^ 1u));

        unit = (unit + kTgtLangDe) % kS2utVocab;  // (prev_units + tgt_lang) mod 16
        for (std::int64_t u : dec_units) CHECK(u == unit);
        for (std::size_t f = 0; f < kS2utChunkFrames; ++f) {
            CHECK(prosody[f * 4 + 0] == ring_k[slot][0] + kProsodyCtx0);  // mean(k_read) + ctx0
            CHECK(prosody[f * 4 + 1] == 0.9f);
            CHECK(prosody[f * 4 + 2] == 0.1f);
            CHECK(prosody[f * 4 + 3] == ring_v[slot][0] + kProsodyCtx1);  // mean(v_read) + ctx1
        }
        const float eos = eos_prob[0];
        CHECK(eos == (tgt + 1 > src ? 1.0f : 0.0f));
        dec_kv[static_cast<std::size_t>(tgt % 4)] = ring_v[slot];
        CHECK(dec_bank_holds(q.tdec, d ^ 1u, tgt + 1, dec_kv));
        t.units.insert(t.units.end(), dec_units.begin(), dec_units.end());
        t.prosody.insert(t.prosody.end(), prosody.begin(), prosody.end());
        t.eos.push_back(eos);
        if (eos == 0.0f) t.decoded.push_back(prosody[3] - kProsodyCtx1);

        // Autoregressive feedback, then the vocoder on the emitted units + prosody.
        std::copy(dec_units.begin(), dec_units.end(), prev_units.begin());
        std::copy(dec_units.begin(), dec_units.end(), voc_units.begin());
        for (std::size_t f = 0; f < kS2utChunkFrames; ++f) {
            q.voc.logf0_in()[f] = prosody[f * 4 + 0];
            q.voc.voicing_in()[f] = prosody[f * 4 + 1];
            q.voc.aperiodicity_in()[f] = prosody[f * 4 + 2];
            q.voc.energy_in()[f] = prosody[f * 4 + 3];
        }
        CHECK(q.voc.step_status() == OrtStatus::kOk);
        bool audio_ok = true;  // tile((units / 16) * voicing) per 320-sample frame
        for (std::size_t f = 0; f < kS2utChunkFrames; ++f) {
            const float want = static_cast<float>(dec_units[f]) * (1.0f / 16.0f) * prosody[f * 4 + 1];
            for (std::size_t s = 0; s < kS2utFrame; ++s)
                audio_ok = audio_ok && std::fabs(audio[f * kS2utFrame + s] - want) <= 1e-6f;
        }
        CHECK(audio_ok);
        t.audio.insert(t.audio.end(), audio.begin(), audio.end());
        return eos;
    };

    for (int i = 0; i < kSourceChunks; ++i) {
        const float val = chunk_value(i);
        const auto audio_in = q.enc.audio_in();
        std::fill(audio_in.begin(), audio_in.end(), val);
        CHECK(q.enc.step_status() == OrtStatus::kOk);
        const auto latents = q.enc.latents_out();
        const auto enc_units = q.enc.units_out();
        CHECK(std::all_of(latents.begin(), latents.end(), [val](float x) { return x == val; }));
        // |sum of a frame's 8 latents| * 16 = 16 (i + 1) -> unit 0 on every frame.
        CHECK(std::all_of(enc_units.begin(), enc_units.end(), [](std::int64_t u) { return u == 0; }));
        std::copy(latents.begin(), latents.end(), latents_in.begin());

        const std::size_t read_bank = q.tenc.active_bank();
        const Ring ring_k_before = ring_k, ring_v_before = ring_v;
        CHECK(q.tenc.step_status() == OrtStatus::kOk);
        const std::size_t new_bank = q.tenc.active_bank();
        CHECK(new_bank == (read_bank ^ 1u));
        const std::size_t slot = static_cast<std::size_t>(i) % kS2utRing;
        ring_k[slot].fill(val + 1.0f);  // k_slot = chunk mean + float(src_lang)
        ring_v[slot].fill(val);         // v_slot = chunk mean
        t.readiness.push_back(q.tenc.output_f32("readiness_logit")[0]);
        CHECK(t.readiness.back() == val);
        // The newest bank holds the ring after chunk i, the bank it read the ring before.
        CHECK(enc_bank_holds(q.tenc, new_bank, i + 1, val, ring_k, ring_v));
        CHECK(enc_bank_holds(q.tenc, read_bank, i, i > 0 ? chunk_value(i - 1) : 0.0f, ring_k_before, ring_v_before));

        // Wait-k steady state: kWrite once the source leads the target by k chunks.
        const std::int64_t src = bank_i64(q.tenc, new_bank, "src_pos");
        const std::int64_t tgt = bank_i64(q.tdec, q.tdec.active_bank(), "tgt_pos");
        if (src - tgt >= wait_k) (void)decoder_step();
    }
    for (int f = 0; f < 8; ++f) {  // kFlush: src_pos stays put; only the decoder (+ vocoder) runs
        const float eos = decoder_step();
        t.flush_eos.push_back(eos);
        if (eos == 1.0f) break;
    }
    return t;
}

void test_dual_encoder_v2() {
    for (int fps : {1, 2, 4}) {
        OrtModel enc;
        CHECK(enc.load(g_fixtures_dir / "s2ut" / ("encoder_s" + std::to_string(fps) + ".onnx")) == OrtStatus::kOk);
        const ModelContract& c = enc.contract();
        CHECK(c.contract_version == kContractVersionV2 && c.role == ModelRole::kEncoder);
        CHECK(c.content_kind == ContentKind::kDiscrete && c.vocab == kS2utVocab);
        CHECK(c.latent_dim == static_cast<int>(kS2utLatent) && c.frames_per_step == fps);

        const auto n = static_cast<std::size_t>(fps);
        const auto audio = enc.audio_in();
        const auto units = enc.units_out();
        const auto latents = enc.latents_out();
        CHECK(audio.size() == n * kS2utFrame && units.size() == n && latents.size() == n * kS2utLatent);
        CHECK(latents.data() == std::as_const(enc).output_f32("latents").data());
        const TensorSpec* lat_spec = contract_detail::find_spec(enc.graph_outputs(), "latents");
        const TensorSpec* unit_spec = contract_detail::find_spec(enc.graph_outputs(), "units");
        CHECK(lat_spec && lat_spec->dtype == TensorDtype::kF32 &&
              lat_spec->shape == (std::vector<std::int64_t>{1, fps, static_cast<std::int64_t>(kS2utLatent)}));
        CHECK(unit_spec && unit_spec->dtype == TensorDtype::kI64 &&
              unit_spec->shape == (std::vector<std::int64_t>{1, fps}));
        CHECK(enc.graph_inputs().size() == 1 && enc.graph_inputs()[0].name == "audio");
        if (audio.size() != n * kS2utFrame || units.size() != n || latents.size() != n * kS2utLatent) continue;
        CHECK(enc.warm_up(2) == OrtStatus::kOk);

        constexpr std::size_t kBin = kS2utFrame / kS2utLatent;  // 40 samples per latent bin
        for (int step = 0; step < 6; ++step) {
            // Every 40-sample bin holds one value m / 64 with m in [-20, 20]: the
            // bin mean (sum x float32(1/40)) and every frame sum stay exact.
            std::vector<int> m(n * kS2utLatent);
            for (std::size_t i = 0; i < m.size(); ++i)
                m[i] = static_cast<int>((i * 37 + static_cast<std::size_t>(step) * 11) % 41) - 20;
            for (std::size_t i = 0; i < m.size(); ++i)
                std::fill_n(audio.begin() + static_cast<std::ptrdiff_t>(i * kBin), kBin, static_cast<float>(m[i]) / 64.0f);
            CHECK(enc.step_status() == OrtStatus::kOk);
            for (std::size_t i = 0; i < m.size(); ++i) CHECK(latents[i] == static_cast<float>(m[i]) / 64.0f);
            for (std::size_t f = 0; f < n; ++f) {
                int sum = 0;
                for (std::size_t b = 0; b < kS2utLatent; ++b) sum += m[f * kS2utLatent + b];
                // units = Cast_i64(|sum of the frame's latents| * 16) mod 16 = (|sum m| / 4) mod 16.
                CHECK(units[f] == static_cast<std::int64_t>((std::abs(sum) / 4) % 16));
            }
        }
    }
    std::printf("s2ut dual encoder s1/s2/s4: units + latents match the fixture arithmetic\n");
}

void test_shared_binding_rejections() {
    const fs::path dir = g_fixtures_dir / "s2ut";
    OrtModel tenc, tdec, wide_tenc, unloaded;
    CHECK(tenc.load(dir / "translator_enc.onnx") == OrtStatus::kOk);
    CHECK(tdec.load(dir / "translator_dec.onnx") == OrtStatus::kOk);
    CHECK(wide_tenc.load(g_fixtures_dir / "s2ut_mismatch" / "translator_enc_latent16.onnx") == OrtStatus::kOk);
    const std::vector<std::string> good = shared_names_for(tenc);

    struct Case {
        const OrtModel* producer;
        std::vector<std::string> names;
        const char* error;
    };
    const Case cases[] = {
        {&tenc, {}, "bind_shared_inputs: no input names given"},
        {&tenc, {"bogus"}, "bind_shared_inputs: 'bogus' is not a graph input of this translator_dec model"},
        {&tenc, {"state_in.tgt_pos"},
         "bind_shared_inputs: 'state_in.tgt_pos' is a state tensor; only plain graph inputs can be shared"},
        {&tenc, {"mem_k_0", "mem_v_0", "mem_k_0"}, "bind_shared_inputs: duplicate name 'mem_k_0'"},
        {&tenc, {"mem_k_0", "mem_v_0", "prev_units"},
         "bind_shared_inputs: 'prev_units' is not a state of the producer translator_enc model"},
        {&tenc, {"mem_k_0", "src_pos"},
         "bind_shared_inputs: graph input 'mem_v_0' has no buffer of its own; add it to the shared names"},
        {&wide_tenc, good,
         "bind_shared_inputs: 'mem_k_0' shape mismatch: producer state_in.mem_k_0 is [1,4,16], "
         "this model's input is [1,4,8]"},
        {&unloaded, good, "bind_shared_inputs: the producer model is not loaded"},
        {&tdec, good, "bind_shared_inputs: a model cannot share inputs with itself"},
    };
    for (const Case& c : cases) {
        const OrtStatus st = tdec.bind_shared_inputs(*c.producer, c.names);
        const bool ok = st == OrtStatus::kContractError && tdec.last_error() == c.error && tdec.is_loaded() &&
                        !tdec.has_shared_inputs() && tdec.shared_producer() == nullptr;
        if (!ok) {
            std::printf("bind_shared_inputs: status=%s, got \"%.*s\", want \"%s\"\n", to_string(st),
                        static_cast<int>(tdec.last_error().size()), tdec.last_error().data(), c.error);
        }
        CHECK(ok);
    }
    CHECK(unloaded.bind_shared_inputs(tenc, good) == OrtStatus::kNotLoaded);

    // A failed call drops an earlier binding: loaded, unbound, own src_pos view back.
    CHECK(tdec.bind_shared_inputs(tenc, good) == OrtStatus::kOk && tdec.has_shared_inputs());
    CHECK(tdec.input_i64("src_pos").empty());
    CHECK(tdec.bind_shared_inputs(tenc, std::vector<std::string>{"bogus"}) == OrtStatus::kContractError);
    CHECK(tdec.is_loaded() && !tdec.has_shared_inputs() && tdec.input_i64("src_pos").size() == 1);
    CHECK(tdec.step_status() == OrtStatus::kContractError);

    // Sharing only the memory keeps translator_dec's own src_pos input in use.
    CHECK(tdec.bind_shared_inputs(tenc, tenc.contract().shared_mem_states) == OrtStatus::kOk);
    CHECK(tdec.input_i64("src_pos").size() == 1 && tdec.input_f32("mem_v_0").empty());
    tdec.input_i64("src_pos")[0] = 1;
    CHECK(tdec.step());
    CHECK(tdec.output_f32("eos_prob")[0] == 0.0f);  // state_out.tgt_pos 1 > src_pos 1 is false
    std::printf("bind_shared_inputs: %zu rejections verified\n", std::size(cases) + 2);
}

void test_translator_shared_binding_table() {
    S2utQuartet q;
    CHECK(q.load());

    // A translator_dec that was never bound fails with a clear error, not an ORT exception.
    CHECK(!q.tdec.has_shared_inputs() && q.tdec.shared_producer() == nullptr);
    CHECK(q.tdec.input_i64("src_pos").size() == 1);
    CHECK(q.tdec.step_status() == OrtStatus::kContractError);
    CHECK(has_text(q.tdec.last_error(), "call bind_shared_inputs() first"));
    CHECK(q.tdec.active_bank() == 0);
    CHECK(q.tdec.warm_up(1) == OrtStatus::kContractError);
    CHECK(q.tdec.step_with_producer_bank(0) == OrtStatus::kContractError);
    CHECK(has_text(q.tdec.last_error(), "no shared inputs are bound"));
    CHECK(q.tenc.warm_up(2) == OrtStatus::kOk);  // a producer needs no binding

    // graph_inputs()/graph_outputs() as inspected at load(): the cross-graph check a bundle runs.
    const TensorSpec* mem_in = contract_detail::find_spec(q.tdec.graph_inputs(), "mem_k_0");
    const TensorSpec* mem_out = contract_detail::find_spec(q.tenc.graph_outputs(), "state_out.mem_k_0");
    CHECK(mem_in && mem_out && mem_in->dtype == TensorDtype::kF32 &&
          mem_in->shape == (std::vector<std::int64_t>{1, 4, 8}) && mem_out->shape == mem_in->shape);
    std::string err;
    CHECK(validate_s2ut_shared_io(q.tenc.contract(), q.tenc.graph_outputs(), q.tdec.contract(),
                                  q.tdec.graph_inputs(), err));

    CHECK(q.bind());
    CHECK(q.tdec.has_shared_inputs() && q.tdec.shared_producer() == &q.tenc);
    // Shared inputs live in translator_enc's banks: no own views any more.
    CHECK(q.tdec.input_i64("src_pos").empty() && std::as_const(q.tdec).input_i64("src_pos").empty());
    CHECK(q.tdec.input_f32("mem_k_0").empty() && q.tdec.input_f32("mem_v_0").empty());
    CHECK(q.tdec.input_i64("prev_units").size() == kS2utChunkFrames);

    // Distinct rings in the two producer banks: after 5 chunks bank 1 holds
    // src_pos 5 (slot 0 rewritten by chunk 4), bank 0 holds src_pos 4 (chunks 0..3).
    q.tenc.input_i64("src_lang")[0] = kSrcLangEs;
    const auto latents = q.tenc.input_f32("latents");
    for (int i = 0; i < 5; ++i) {
        std::fill(latents.begin(), latents.end(), chunk_value(i));
        CHECK(q.tenc.step());
    }
    CHECK(q.tenc.active_bank() == 1);
    CHECK(bank_i64(q.tenc, 1, "src_pos") == 5 && bank_i64(q.tenc, 0, "src_pos") == 4);

    // With tgt_pos = 4 the decoder reads slot min(4, src_pos - 1) mod 4 and
    // raises eos iff 5 > src_pos, so every table entry shows which bank it read:
    //   producer bank 1 (src_pos 5) -> slot 0 = chunk 4, eos 0
    //   producer bank 0 (src_pos 4) -> slot 3 = chunk 3, eos 1
    const auto prosody = q.tdec.output_f32("prosody");
    const auto eos = q.tdec.output_f32("eos_prob");
    const auto ctx = q.tdec.input_f32("prosody_ctx");
    CHECK(prosody.size() == kS2utChunkFrames * 4 && eos.size() == 1 && ctx.size() == 2);
    if (prosody.size() != kS2utChunkFrames * 4 || eos.size() != 1 || ctx.size() != 2) return;
    ctx[0] = kProsodyCtx0;
    ctx[1] = kProsodyCtx1;
    const auto set_tgt_pos = [&q](std::size_t bank, std::int64_t pos) {
        q.tdec.state_bank(bank).find("tgt_pos")->span_i64()[0] = pos;
        q.tdec.state_bank(bank).find("dec_kv_0")->zero();
    };
    for (std::size_t bank = 0; bank < 2; ++bank) {
        for (std::size_t parity = 0; parity < 2; ++parity) {
            q.tdec.reset();
            if (parity == 1) CHECK(q.tdec.step_with_producer_bank(bank) == OrtStatus::kOk);
            CHECK(q.tdec.active_bank() == parity);
            set_tgt_pos(parity, 4);
            const std::vector<std::uint8_t> shared_before = bank_bytes(q.tenc);
            CHECK(q.tdec.step_with_producer_bank(bank) == OrtStatus::kOk);
            const float v = chunk_value(bank == 1 ? 4 : 3);
            bool ok = bank_bytes(q.tenc) == shared_before && q.tenc.active_bank() == 1 &&
                      q.tdec.active_bank() == (parity ^ 1u) && eos[0] == (bank == 1 ? 0.0f : 1.0f);
            for (std::size_t f = 0; f < kS2utChunkFrames; ++f) {
                ok = ok && prosody[f * 4 + 0] == v + 1.0f + kProsodyCtx0 && prosody[f * 4 + 3] == v + kProsodyCtx1;
            }
            Ring kv{};
            kv[0].fill(v);  // tgt_pos 4 -> dec_kv_0 slot 0 = v_read
            ok = ok && dec_bank_holds(q.tdec, parity ^ 1u, 5, kv);
            if (!ok) std::printf("table[producer bank %zu][parity %zu] read the wrong bank\n", bank, parity);
            CHECK(ok);
        }
    }

    // step() runs table[producer.active_bank()][parity].
    q.tdec.reset();
    set_tgt_pos(0, 4);
    CHECK(q.tdec.step());
    CHECK(prosody[3] == chunk_value(4) + kProsodyCtx1 && eos[0] == 0.0f);

    // Zero copy: a write into translator_enc's bank reaches the next decoder step, no rebinding.
    const auto v_slot0 = q.tenc.state_bank(1).find("mem_v_0")->span_f32().first(kS2utLatent);
    std::fill(v_slot0.begin(), v_slot0.end(), 3.0f);
    q.tdec.reset();
    set_tgt_pos(0, 4);
    CHECK(q.tdec.step() && prosody[3] == 3.0f + kProsodyCtx1);
    q.tenc.state_bank(1).find("src_pos")->span_i64()[0] = 2;  // the shared src_pos, too
    q.tdec.reset();
    set_tgt_pos(0, 4);
    CHECK(q.tdec.step() && prosody[3] == chunk_value(1) + kProsodyCtx1 && eos[0] == 1.0f);  // slot min(4, 1)

    // warm_up() of a bound consumer steps through the table and zeroes only its own tensors.
    const std::vector<std::uint8_t> shared_before = bank_bytes(q.tenc);
    CHECK(q.tdec.warm_up(3) == OrtStatus::kOk);
    CHECK(q.tdec.active_bank() == 0 && all_zero(bank_bytes(q.tdec)));
    CHECK(outputs_zero(q.tdec) && inputs_zero(q.tdec));
    CHECK(bank_bytes(q.tenc) == shared_before);

    // Reloading the producer releases the tensors the table points at: detected, never run.
    CHECK(q.tenc.load(g_fixtures_dir / "s2ut" / "translator_enc.onnx") == OrtStatus::kOk);
    CHECK(q.tdec.has_shared_inputs());
    const std::size_t parity = q.tdec.active_bank();
    CHECK(q.tdec.step_status() == OrtStatus::kContractError);
    CHECK(q.tdec.last_error() == kSharedProducerReloadedMessage);
    CHECK(q.tdec.step_with_producer_bank(1) == OrtStatus::kContractError);
    CHECK(q.tdec.active_bank() == parity);
    CHECK(q.bind() && q.tdec.step());

    // Moving the producer is caught the same way (the moved-from object is still alive here).
    OrtModel moved_enc = std::move(q.tenc);
    CHECK(q.tdec.step_status() == OrtStatus::kContractError);
    CHECK(q.tdec.last_error() == kSharedProducerReloadedMessage);
    CHECK(q.tdec.bind_shared_inputs(moved_enc, shared_names_for(moved_enc)) == OrtStatus::kOk);
    CHECK(q.tdec.step());

    // Moving the consumer keeps its binding: the table and the buffers it binds move along.
    OrtModel moved_dec = std::move(q.tdec);
    CHECK(moved_dec.has_shared_inputs() && moved_dec.shared_producer() == &moved_enc);
    CHECK(moved_dec.step());

    // Reloading the consumer drops its binding.
    CHECK(moved_dec.load(g_fixtures_dir / "s2ut" / "translator_dec.onnx") == OrtStatus::kOk);
    CHECK(!moved_dec.has_shared_inputs() && moved_dec.shared_producer() == nullptr);
    CHECK(moved_dec.input_i64("src_pos").size() == 1);
    CHECK(moved_dec.step_status() == OrtStatus::kContractError);
    std::printf("translator_dec 2 x 2 table: every [enc bank][dec parity] entry, zero copy, reload/move rules\n");
}

void test_translator_quartet_waitk() {
    std::vector<float> chunks;
    for (int i = 0; i < kSourceChunks; ++i) chunks.push_back(chunk_value(i));

    // k = 3 mirrors python_training/tests/test_fixtures.py: 3 source chunks
    // before the first write, 8 writes, 3 flush steps ending in eos.
    S2utQuartet q3;
    CHECK(q3.load() && q3.bind());
    const QuartetTrace t3 = run_quartet(q3, 3);
    CHECK(same_bits(t3.decoded, chunks));  // every chunk read back exactly once, in order
    CHECK(t3.flush_eos == (std::vector<float>{0.0f, 0.0f, 1.0f}));
    const std::int64_t want_units[] = {3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1};  // 3n mod 16
    CHECK(t3.units.size() == std::size(want_units) * kS2utChunkFrames);
    for (std::size_t n = 0; n < std::size(want_units) && n * kS2utChunkFrames < t3.units.size(); ++n)
        CHECK(t3.units[n * kS2utChunkFrames] == want_units[n]);
    // Odd k: while writing the parities stay out of phase (1,0)/(0,1); the
    // flush steps (decoder only) add (0,0).
    CHECK(t3.combos[0][0] == 2 && t3.combos[0][1] == 5 && t3.combos[1][0] == 4 && t3.combos[1][1] == 0);

    // k = 2: even k keeps the parities in phase (0,0)/(1,1).
    S2utQuartet q2;
    CHECK(q2.load() && q2.bind());
    const QuartetTrace t2 = run_quartet(q2, 2);
    CHECK(same_bits(t2.decoded, chunks));
    CHECK(t2.flush_eos == (std::vector<float>{0.0f, 1.0f}));
    CHECK(t2.combos[0][0] == 6 && t2.combos[0][1] == 1 && t2.combos[1][0] == 0 && t2.combos[1][1] == 4);

    // Together the two schedules run all four (translator_enc bank, translator_dec parity) bindings.
    for (std::size_t e = 0; e < 2; ++e)
        for (std::size_t d = 0; d < 2; ++d) CHECK(t3.combos[e][d] + t2.combos[e][d] > 0);

    // k = 1 reads the slot translator_enc has just written (and eos compares
    // with the newest src_pos): the schedule where reading the other producer
    // bank would change every write. For k >= 2 both banks agree on the slot.
    S2utQuartet q1;
    CHECK(q1.load() && q1.bind());
    const QuartetTrace t1 = run_quartet(q1, 1);
    CHECK(same_bits(t1.decoded, chunks));
    CHECK(t1.flush_eos == (std::vector<float>{1.0f}));
    CHECK(t1.combos[0][0] == 1 && t1.combos[0][1] == 5 && t1.combos[1][0] == 5 && t1.combos[1][1] == 0);
    std::printf("s2ut quartet: k=3 (%zu decoder steps, flush eos %.0f/%.0f/%.0f) and k=2 cover all 4 bindings\n",
                t3.eos.size(), t3.flush_eos.size() > 0 ? t3.flush_eos[0] : -1.0f,
                t3.flush_eos.size() > 1 ? t3.flush_eos[1] : -1.0f, t3.flush_eos.size() > 2 ? t3.flush_eos[2] : -1.0f);
}

void test_translator_reset_and_pair_warm_up() {
    S2utQuartet q;
    CHECK(q.load() && q.bind());
    const QuartetTrace reference = run_quartet(q, 3);
    CHECK(!reference.eos.empty());

    // reset() of both translators: src_pos, tgt_pos and every ring bank back to zero.
    CHECK(!all_zero(bank_bytes(q.tenc)) && !all_zero(bank_bytes(q.tdec)));
    CHECK(q.tdec.active_bank() == 1);  // 11 decoder steps
    q.tenc.reset();
    q.tdec.reset();
    CHECK(q.tenc.active_bank() == 0 && q.tdec.active_bank() == 0);
    CHECK(all_zero(bank_bytes(q.tenc)) && all_zero(bank_bytes(q.tdec)));
    CHECK(outputs_zero(q.tenc) && outputs_zero(q.tdec));
    CHECK(same_trace(reference, run_quartet(q, 3)));

    // warm_up_translator_pair() on a dirty pair (k = 2 run plus one more
    // translator_enc step: both parities odd) restores fresh-load behaviour.
    S2utQuartet w;
    CHECK(w.load() && w.bind());
    (void)run_quartet(w, 2);
    CHECK(w.tenc.step());
    CHECK(w.tenc.active_bank() == 1 && w.tdec.active_bank() == 1);
    CHECK(warm_up_translator_pair(w.tenc, w.tdec, 4) == OrtStatus::kOk);
    CHECK(w.tenc.active_bank() == 0 && w.tdec.active_bank() == 0);
    CHECK(all_zero(bank_bytes(w.tenc)) && all_zero(bank_bytes(w.tdec)));
    CHECK(outputs_zero(w.tenc) && outputs_zero(w.tdec));
    CHECK(inputs_zero(w.tenc) && inputs_zero(w.tdec));
    CHECK(same_trace(reference, run_quartet(w, 3)));

    // Failures propagate as OrtStatus.
    S2utQuartet u;
    CHECK(u.load());
    CHECK(warm_up_translator_pair(u.tenc, u.tdec, 2) == OrtStatus::kContractError);  // never bound
    CHECK(has_text(u.tdec.last_error(), "bind_shared_inputs()"));
    CHECK(u.bind());
    CHECK(warm_up_translator_pair(u.tdec, u.tenc, 2) == OrtStatus::kContractError);  // roles swapped
    OrtModel unloaded;
    CHECK(warm_up_translator_pair(unloaded, u.tdec, 2) == OrtStatus::kNotLoaded);
    CHECK(u.tenc.load(g_fixtures_dir / "s2ut" / "translator_enc.onnx") == OrtStatus::kOk);
    CHECK(warm_up_translator_pair(u.tenc, u.tdec, 2) == OrtStatus::kContractError);  // producer reloaded
    CHECK(u.tdec.last_error() == kSharedProducerReloadedMessage);
    CHECK(u.bind() && warm_up_translator_pair(u.tenc, u.tdec, 2) == OrtStatus::kOk);
    std::printf("translator reset() and warm_up_translator_pair(): bit-identical reruns\n");
}

void test_s2ut_mismatch_fixtures() {
    const json manifest = read_json(g_fixtures_dir / "manifest.json");
    CHECK(manifest.is_object() && manifest.contains("s2ut_mismatch") && manifest.contains("s2ut_quartets"));
    if (!manifest.is_object() || !manifest.contains("s2ut_mismatch") || !manifest.contains("s2ut_quartets")) return;

    const char* const roles[] = {"encoder", "translator_enc", "translator_dec", "vocoder"};
    std::vector<std::string> errors;
    for (const auto& mm : manifest.at("s2ut_mismatch")) {
        const std::string file = mm.at("file").get<std::string>();
        const std::string role = mm.at("replaces").get<std::string>();
        const std::string want = mm.at("error").get<std::string>();
        const json* quartet = nullptr;
        for (const auto& q : manifest.at("s2ut_quartets"))
            if (q.at("name") == mm.at("quartet")) quartet = &q;
        CHECK(quartet != nullptr);
        if (!quartet) continue;

        std::map<std::string, OrtModel> models;  // node-based: addresses stay put for binding
        for (const char* r : roles) CHECK(models[r].load(g_fixtures_dir / quartet->at(r).get<std::string>()) == OrtStatus::kOk);
        OrtModel replacement;
        CHECK(replacement.load(g_fixtures_dir / file) == OrtStatus::kOk);  // valid on its own
        CHECK(replacement.is_loaded() && to_string(replacement.contract().role) == role);
        if (!replacement.is_loaded()) continue;

        std::map<std::string, const ModelContract*> c;
        for (const char* r : roles) c[r] = &models[r].contract();
        c[role] = &replacement.contract();
        std::string error;
        const bool accepted = validate_s2ut_bundle(*c["encoder"], *c["translator_enc"], *c["translator_dec"],
                                                   *c["vocoder"], error);
        if (accepted || error != want) std::printf("%s: got \"%s\", want \"%s\"\n", file.c_str(), error.c_str(), want.c_str());
        CHECK(!accepted && error == want);
        errors.push_back(error);

        OrtModel& tenc = models["translator_enc"];
        OrtModel& tdec = models["translator_dec"];
        if (role == "translator_enc") {
            // Its [1,4,16] ring cannot feed translator_dec's [1,4,8] memory inputs.
            CHECK(tdec.bind_shared_inputs(replacement, shared_names_for(replacement)) == OrtStatus::kContractError);
            CHECK(has_text(tdec.last_error(), "'mem_k_0' shape mismatch"));
            CHECK(!validate_s2ut_shared_io(replacement.contract(), replacement.graph_outputs(), tdec.contract(),
                                           tdec.graph_inputs(), error));
            CHECK(error == "shared_mem_state 'mem_k_0' shape mismatch: translator_enc [1,4,16], translator_dec [1,4,8]");
            const auto lat = replacement.input_f32("latents");
            CHECK(lat.size() == kS2utChunkFrames * 16);
            std::fill(lat.begin(), lat.end(), 0.25f);
            CHECK(replacement.step() && replacement.output_f32("readiness_logit")[0] == 0.25f);  // runs alone
        } else if (replacement.contract().shared_mem_states != tenc.contract().shared_mem_states) {
            // Renamed memory: no binding under either naming.
            std::vector<std::string> own = replacement.contract().shared_mem_states;
            own.push_back("src_pos");
            CHECK(replacement.bind_shared_inputs(tenc, own) == OrtStatus::kContractError);
            CHECK(replacement.last_error() ==
                  "bind_shared_inputs: 'mem_k_1' is not a state of the producer translator_enc model");
            CHECK(replacement.bind_shared_inputs(tenc, shared_names_for(tenc)) == OrtStatus::kContractError);
            CHECK(replacement.last_error() ==
                  "bind_shared_inputs: 'mem_k_0' is not a graph input of this translator_dec model");
            CHECK(replacement.is_loaded() && !replacement.has_shared_inputs());
            CHECK(replacement.step_status() == OrtStatus::kContractError);
        } else {
            // Only the language table differs: tensors bind, so the bundle check is the gate.
            CHECK(replacement.bind_shared_inputs(tenc, shared_names_for(tenc)) == OrtStatus::kOk);
            CHECK(replacement.step());
        }
    }
    std::sort(errors.begin(), errors.end());
    CHECK(errors == (std::vector<std::string>{"langs mismatch between translator_enc and translator_dec",
                                              "latent_dim mismatch: encoder 8, translator_enc 16",
                                              "shared_mem_states mismatch between translator_enc and translator_dec"}));
    std::printf("s2ut_mismatch fixtures: %zu quartet mismatches rejected\n", errors.size());
}

void test_translator_rss_10000_steps() {
    S2utQuartet q;
    CHECK(q.load() && q.bind());
    CHECK(warm_up_translator_pair(q.tenc, q.tdec, 20) == OrtStatus::kOk);
    q.tenc.input_i64("src_lang")[0] = kSrcLangEs;
    q.tdec.input_i64("tgt_lang")[0] = kTgtLangDe;
    const auto latents = q.tenc.input_f32("latents");
    const auto prev_units = q.tdec.input_i64("prev_units");
    const auto units = q.tdec.output_i64("units");
    CHECK(!latents.empty() && prev_units.size() == units.size());
    if (latents.empty() || prev_units.size() != units.size()) return;
    std::fill(prev_units.begin(), prev_units.end(), kS2utBos);

    std::array<std::array<int, 2>, 2> combos{};
    bool ok = true;
    const std::size_t rss_before = resident_bytes();
    for (std::size_t step = 0; step < 10000; ++step) {
        latents[step % latents.size()] = static_cast<float>(static_cast<int>(step % 17) - 8) * 0.05f;
        ok = q.tenc.step() && ok;
        // An occasional catch-up write drifts the parities, so all 4 table entries run.
        const int writes = step % 7 == 0 ? 2 : 1;
        for (int w = 0; w < writes; ++w) {
            ++combos[q.tenc.active_bank()][q.tdec.active_bank()];
            ok = q.tdec.step() && ok;
            std::copy(units.begin(), units.end(), prev_units.begin());
        }
    }
    const std::size_t rss_after = resident_bytes();
    CHECK(ok);
    for (const auto& row : combos)
        for (int n : row) CHECK(n > 0);
    const std::int64_t delta_bytes = static_cast<std::int64_t>(rss_after) - static_cast<std::int64_t>(rss_before);
    std::printf("10,000 translator pair steps RSS: before=%zu KB, after=%zu KB, delta=%+lld KB\n",
                rss_before / 1024, rss_after / 1024, static_cast<long long>(delta_bytes / 1024));
    CHECK(delta_bytes < 1024 * 1024);  // M3 Stage 07 DoD: < 1 MB after warm-up
}

// ---------------------------------------------------------------------------
// Milestone 3 / Stages 07-08: the whole Mode B stack (S2utBundle ->
// InferencePipeline -> translator quartet) on the s2ut/ fixture bundle, the
// language of fv_s2ut_file. Counter values come from the Python policy twin
// (python_training/fvml/waitk_policy.py): 10 speech chunks then 15 silent
// chunks at k = 3.
// ---------------------------------------------------------------------------

std::vector<PipelineVoice> pipeline_voices_of(const S2utBundle& bundle) {
    std::vector<PipelineVoice> voices;
    for (const VoiceEntry& v : bundle.voices)
        voices.push_back({v.stats, std::span<const float>(v.embedding.data(), v.embedding.size())});
    return voices;
}

/// Fresh temporary directory holding a copy of every file of `from`.
fs::path copy_bundle_dir(const fs::path& from, const std::string& tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir = fs::temp_directory_path() / ("fv_s2ut_bundle_" + tag + "_" + std::to_string(stamp));
    std::error_code ec;
    fs::create_directories(dir, ec);
    for (const auto& e : fs::directory_iterator(from))
        if (e.is_regular_file()) fs::copy_file(e.path(), dir / e.path().filename(), fs::copy_options::overwrite_existing);
    return dir;
}

void test_s2ut_bundle_pipeline() {
    const fs::path bundle_dir = g_fixtures_dir / "s2ut";
    constexpr std::size_t kChunk = 1280;  // 4 frames = 80 ms
    constexpr std::size_t kSpeechChunks = 10, kSilentChunks = 15;
    constexpr std::size_t kTotal = (kSpeechChunks + kSilentChunks) * kChunk;

    for (const int fps : {1, 2, 4}) {
        auto bundle = std::make_unique<S2utBundle>();
        std::string error;
        const bool loaded = bundle->load(bundle_dir, fps, error);
        if (!loaded) std::printf("S2utBundle::load(s%d): %s\n", fps, error.c_str());
        CHECK(loaded);
        if (!loaded) continue;
        CHECK(!bundle->voices.empty() && bundle->langs.find("es") && bundle->langs.find("de"));
        CHECK(bundle->langs.default_k("es", "de") >= 1);
        const LangEntry* es = bundle->langs.find("es");
        const LangEntry* de = bundle->langs.find("3");
        CHECK(es && de && es->id == kSrcLangEs && de->id == kTgtLangDe && de->code == "de");
        if (!es || !de || bundle->voices.empty()) continue;

        const std::vector<PipelineVoice> voices = pipeline_voices_of(*bundle);
        S2utOptions s2ut;
        s2ut.policy.k = 3;
        s2ut.src_lang_id = es->id;
        s2ut.tgt_lang_id = de->id;
        PipelineOptions options;
        options.output_chain = false;
        options.params.voice_a = 0;
        options.params.voice_b = 0;
        InferencePipeline pipeline(bundle->encoder, bundle->translator_enc, bundle->translator_dec,
                                   bundle->vocoder, voices, s2ut, options);
        CHECK(pipeline.warm_up(20));
        CHECK(pipeline.mode() == PipelineMode::kS2UT);

        const std::size_t hop = static_cast<std::size_t>(bundle->encoder.contract().hop());
        CHECK(hop == static_cast<std::size_t>(fps) * kS2utFrame && kTotal % hop == 0);
        std::vector<float> in(hop), out;
        out.reserve(kTotal);
        std::vector<float> hop_out(hop);
        constexpr double kTwoPi = 6.283185307179586;
        for (std::size_t pos = 0; pos < kTotal; pos += hop) {
            for (std::size_t i = 0; i < hop; ++i) {
                const std::size_t n = pos + i;
                in[i] = n < kSpeechChunks * kChunk
                            ? 0.25f * static_cast<float>(std::sin(kTwoPi * 200.0 * static_cast<double>(n) / 16000.0))
                            : 0.0f;
            }
            const PipelineStepResult r = pipeline.process(in, hop_out);
            CHECK(r.ok);
            out.insert(out.end(), hop_out.begin(), hop_out.end());
        }

        const S2utTelemetry t = pipeline.s2ut_telemetry();
        std::printf("s2ut pipeline s%d: reads=%llu writes=%llu flushes=%llu utt=%llu eos=%llu cap=%llu dec=%llu voc=%llu\n",
                    fps, static_cast<unsigned long long>(t.reads), static_cast<unsigned long long>(t.writes),
                    static_cast<unsigned long long>(t.flushes), static_cast<unsigned long long>(t.utterances),
                    static_cast<unsigned long long>(t.eos_ends), static_cast<unsigned long long>(t.cap_ends),
                    static_cast<unsigned long long>(t.dec_steps), static_cast<unsigned long long>(t.voc_steps));
        CHECK(t.reads == 9 && t.writes == 13 && t.flushes == 3);
        CHECK(t.utterances == 1 && t.eos_ends == 1 && t.cap_ends == 0);
        CHECK(t.dec_steps == 16);
        CHECK(t.voc_steps == 16u * 4u / static_cast<std::uint64_t>(fps));
        CHECK(pipeline.s2ut_idle());
        CHECK(pipeline.errors() == 0);

        // Translated audio: per decoder step n = 1..16 the unit is 3n mod 16 on
        // all 4 frames (BOS 16 + 3 first), a 320-sample frame of (unit / 16) * 0.9;
        // unit 0 (the 16th step) and every silent hop are exactly zero.
        CHECK(out.size() == kTotal);
        std::vector<float> want_frames;
        std::int64_t unit = kS2utBos;
        for (int n = 0; n < 16; ++n) {
            unit = (unit + kTgtLangDe) % kS2utVocab;
            for (std::size_t f = 0; f < kS2utChunkFrames; ++f)
                if (unit != 0) want_frames.push_back(static_cast<float>(unit) * (1.0f / 16.0f) * 0.9f);
        }
        std::vector<float> got_frames;
        bool frames_constant = true;
        for (std::size_t f = 0; f + kS2utFrame <= out.size(); f += kS2utFrame) {
            const float first = out[f];
            for (std::size_t s = 0; s < kS2utFrame; ++s) frames_constant = frames_constant && out[f + s] == first;
            if (first != 0.0f) got_frames.push_back(first);
        }
        CHECK(frames_constant);
        CHECK(got_frames.size() == want_frames.size());
        bool frames_match = got_frames.size() == want_frames.size();
        for (std::size_t i = 0; frames_match && i < want_frames.size(); ++i)
            frames_match = std::fabs(got_frames[i] - want_frames[i]) <= 1e-5f;
        CHECK(frames_match);
        // Nothing is spoken before the policy has read k = 3 chunks (the first write is at chunk 3).
        CHECK(std::all_of(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(2 * kChunk),
                          [](float x) { return x == 0.0f; }));
    }

    // ---- negative loads --------------------------------------------------
    {
        S2utBundle v1;
        std::string error;
        CHECK(!v1.load(g_fixtures_dir / "identity", 4, error));
        CHECK(error.rfind(kNoTranslatorModelsMessage, 0) == 0);
    }
    {
        const fs::path dir = copy_bundle_dir(bundle_dir, "nolangs");
        fs::remove(dir / "langs.json");
        S2utBundle b;
        std::string error;
        CHECK(!b.load(dir, 4, error));
        CHECK(error.rfind("Mode B bundle has no langs.json", 0) == 0);
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    {
        const json manifest = read_json(g_fixtures_dir / "manifest.json");
        CHECK(manifest.is_object() && manifest.contains("s2ut_mismatch"));
        std::size_t rejected = 0;
        if (manifest.is_object() && manifest.contains("s2ut_mismatch")) {
            for (const auto& mm : manifest.at("s2ut_mismatch")) {
                const std::string file = mm.at("file").get<std::string>();
                const std::string replaces = mm.at("replaces").get<std::string>();
                const std::string want = mm.at("error").get<std::string>();
                const fs::path dir = copy_bundle_dir(bundle_dir, "mismatch");
                fs::copy_file(g_fixtures_dir / file, dir / (replaces + ".onnx"), fs::copy_options::overwrite_existing);
                S2utBundle b;
                std::string error;
                const bool accepted = b.load(dir, 4, error);
                if (accepted || error != want)
                    std::printf("%s: got \"%s\", want \"%s\"\n", file.c_str(), error.c_str(), want.c_str());
                CHECK(!accepted && error == want);
                rejected += accepted ? 0 : 1;
                std::error_code ec;
                fs::remove_all(dir, ec);
            }
        }
        CHECK(rejected == 3);
    }

    // ---- RSS: 10,000 hops (fps 1) after warm-up grow < 1 MB ----------------
    {
        auto bundle = std::make_unique<S2utBundle>();
        std::string error;
        const bool loaded = bundle->load(bundle_dir, 1, error);
        CHECK(loaded);
        if (loaded && !bundle->voices.empty()) {
            const std::vector<PipelineVoice> voices = pipeline_voices_of(*bundle);
            S2utOptions s2ut;
            s2ut.policy.k = 3;
            s2ut.src_lang_id = kSrcLangEs;
            s2ut.tgt_lang_id = kTgtLangDe;
            PipelineOptions options;
            options.output_chain = false;
            InferencePipeline pipeline(bundle->encoder, bundle->translator_enc, bundle->translator_dec,
                                       bundle->vocoder, voices, s2ut, options);
            CHECK(pipeline.warm_up(20));
            std::vector<float> in(kS2utFrame), out(kS2utFrame);
            bool ok = true;
            const std::size_t rss_before = resident_bytes();
            for (std::size_t h = 0; h < 10000; ++h) {
                // Repeating 25-chunk utterance: 10 speech chunks, 15 silent (4 hops each).
                const bool speech = (h / 4) % 25 < kSpeechChunks;
                for (std::size_t i = 0; i < in.size(); ++i)
                    in[i] = speech ? 0.25f * static_cast<float>(std::sin(6.283185307179586 * 200.0 *
                                                                         static_cast<double>(h * in.size() + i) / 16000.0))
                                   : 0.0f;
                ok = pipeline.process(in, out).ok && ok;
            }
            const std::size_t rss_after = resident_bytes();
            CHECK(ok);
            CHECK(pipeline.s2ut_telemetry().utterances > 10);
            const std::int64_t delta_bytes = static_cast<std::int64_t>(rss_after) - static_cast<std::int64_t>(rss_before);
            std::printf("10,000 s2ut pipeline hops RSS: before=%zu KB, after=%zu KB, delta=%+lld KB\n",
                        rss_before / 1024, rss_after / 1024, static_cast<long long>(delta_bytes / 1024));
            CHECK(delta_bytes < 1024 * 1024);
        }
    }
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
    test_dual_encoder_v2();
    test_shared_binding_rejections();
    test_translator_shared_binding_table();
    test_translator_quartet_waitk();
    test_translator_reset_and_pair_warm_up();
    test_s2ut_mismatch_fixtures();
    test_translator_rss_10000_steps();
    test_s2ut_bundle_pipeline();
    return test::finish("ort_model");
}
