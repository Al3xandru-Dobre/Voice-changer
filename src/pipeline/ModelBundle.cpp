#include "pipeline/ModelBundle.hpp"

#include "pipeline/ModelContract.hpp"

#include <cstddef>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fluxvoice {

bool locate_bundle(const std::filesystem::path& directory, int fps,
                    BundlePaths& out, std::string& error) {
    if (fps != 0 && fps != 1 && fps != 2 && fps != 4) {
        error = "frames-per-step must be 1, 2 or 4";
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        error = "model bundle directory not found: " + directory.string();
        return false;
    }
    for (int candidate : {1, 2, 4}) {
        if (fps != 0 && fps != candidate) continue;
        const auto suffix = "s" + std::to_string(candidate) + ".onnx";
        BundlePaths p{directory / ("encoder_" + suffix), directory / ("vocoder_" + suffix),
                      directory / "voices.json", candidate};
        if (std::filesystem::is_regular_file(p.encoder, ec) &&
            std::filesystem::is_regular_file(p.vocoder, ec)) {
            out = std::move(p);
            error.clear();
            return true;
        }
    }
    error = "no complete encoder/vocoder bundle variant" +
        (fps ? " s" + std::to_string(fps) : std::string{}) + " in " + directory.string();
    return false;
}

bool locate_s2ut_bundle(const std::filesystem::path& directory, int fps,
                        S2utBundlePaths& out, std::string& error) {
    BundlePaths base;
    if (!locate_bundle(directory, fps, base, error)) return false;
    S2utBundlePaths p{base.encoder, directory / "translator_enc.onnx",
                      directory / "translator_dec.onnx", base.vocoder, base.voices,
                      directory / "langs.json", base.frames_per_step};
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p.translator_enc, ec) ||
        !std::filesystem::is_regular_file(p.translator_dec, ec)) {
        error = std::string(kNoTranslatorModelsMessage) + " in " + directory.string();
        return false;
    }
    if (!std::filesystem::is_regular_file(p.langs, ec)) {
        error = "Mode B bundle has no langs.json in " + directory.string();
        return false;
    }
    out = std::move(p);
    error.clear();
    return true;
}

bool S2utBundle::load(const std::filesystem::path& directory, int fps, std::string& error) {
    try {
        if (!locate_s2ut_bundle(directory, fps, paths, error)) return false;
        struct Slot {
            OrtModel& model;
            const std::filesystem::path& path;
            const char* name;
        };
        const Slot slots[] = {{encoder, paths.encoder, "encoder"},
                              {translator_enc, paths.translator_enc, "translator_enc"},
                              {translator_dec, paths.translator_dec, "translator_dec"},
                              {vocoder, paths.vocoder, "vocoder"}};
        for (const Slot& s : slots) {
            if (s.model.load(s.path) != OrtStatus::kOk) {
                error = std::string("Failed to load ") + s.name + ": " + s.model.last_error_str();
                return false;
            }
        }
        if (!validate_s2ut_bundle(encoder.contract(), translator_enc.contract(),
                                  translator_dec.contract(), vocoder.contract(), error) ||
            !validate_s2ut_shared_io(translator_enc.contract(), translator_enc.graph_outputs(),
                                     translator_dec.contract(), translator_dec.graph_inputs(),
                                     error))
            return false;
        if (encoder.contract().frames_per_step != paths.frames_per_step) {
            error = "bundle filename variant disagrees with model frames_per_step";
            return false;
        }
        std::vector<std::string> shared = translator_enc.contract().shared_mem_states;
        shared.emplace_back("src_pos");
        if (translator_dec.bind_shared_inputs(translator_enc, shared) != OrtStatus::kOk) {
            error = "Failed to bind translator_dec to translator_enc: " +
                    translator_dec.last_error_str();
            return false;
        }
        if (!read_lang_catalog(paths.langs, &translator_enc.contract(), langs, error)) return false;
        return read_voice_catalog(paths.voices,
            static_cast<std::size_t>(vocoder.contract().spk_emb_dim), voices, error);
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool ModelBundle::load(const std::filesystem::path& directory, int fps, std::string& error) {
    try {
        if (!locate_bundle(directory, fps, paths, error)) return false;
        if (encoder.load(paths.encoder) != OrtStatus::kOk) {
            error = "Failed to load encoder: " + std::string(encoder.last_error());
            return false;
        }
        if (vocoder.load(paths.vocoder) != OrtStatus::kOk) {
            error = "Failed to load vocoder: " + std::string(vocoder.last_error());
            return false;
        }
        if (!validate_pair(encoder.contract(), vocoder.contract(), error)) return false;
        if (encoder.contract().frames_per_step != paths.frames_per_step) {
            error = "bundle filename variant disagrees with model frames_per_step";
            return false;
        }
        return read_voice_catalog(paths.voices,
            static_cast<std::size_t>(vocoder.contract().spk_emb_dim), voices, error);
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

} // namespace fluxvoice
