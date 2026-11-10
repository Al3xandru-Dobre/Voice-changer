#pragma once

#include "pipeline/OrtModel.hpp"
#include "pipeline/VoiceCatalog.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace fluxvoice {

struct BundlePaths {
    std::filesystem::path encoder, vocoder, voices;
    int frames_per_step = 0;
};

bool locate_bundle(const std::filesystem::path& directory, int frames_per_step,
                    BundlePaths& out, std::string& error);

// Models must outlive InferencePipeline. Load/validate before opening devices.
struct ModelBundle {
    OrtModel encoder, vocoder;
    BundlePaths paths;
    std::vector<VoiceEntry> voices;

    bool load(const std::filesystem::path& directory, int frames_per_step, std::string& error);
};

} // namespace fluxvoice
