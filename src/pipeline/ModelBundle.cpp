#include "pipeline/ModelBundle.hpp"

#include <exception>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <system_error>
#include <utility>

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
