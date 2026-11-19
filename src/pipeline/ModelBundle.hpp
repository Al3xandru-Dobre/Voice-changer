#pragma once

#include "pipeline/LangCatalog.hpp"
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

// ---- Mode B (Milestone 3 / Stages 07–08) --------------------------------
// Contract v2 layout (Documents/Milestone 3/02_translator_contract_v2_fixtures.md):
// encoder_s{1,2,4}.onnx + vocoder_s{1,2,4}.onnx (as Mode A), translator_enc.onnx,
// translator_dec.onnx, voices.json and langs.json.
struct S2utBundlePaths {
    std::filesystem::path encoder, translator_enc, translator_dec, vocoder, voices, langs;
    int frames_per_step = 0;
};

inline constexpr const char* kNoTranslatorModelsMessage =
    "bundle has no translator models (translator_enc.onnx / translator_dec.onnx)";

bool locate_s2ut_bundle(const std::filesystem::path& directory, int frames_per_step,
                        S2utBundlePaths& out, std::string& error);

// Load order: the four sessions, validate_s2ut_bundle() + validate_s2ut_shared_io()
// on the real graph I/O, translator_dec's `src_pos` + `mem_*` bound to
// translator_enc's two state banks (OrtModel::bind_shared_inputs: zero copy,
// zero rebinding), then langs.json (checked against the translator contract)
// and voices.json. translator_dec keeps a pointer to translator_enc, so the
// bundle is neither copyable nor movable: hold it by unique_ptr and keep it
// alive (unmoved) for as long as any pipeline uses it.
struct S2utBundle {
    OrtModel encoder, translator_enc, translator_dec, vocoder;
    S2utBundlePaths paths;
    std::vector<VoiceEntry> voices;
    LangCatalog langs;

    S2utBundle() = default;
    S2utBundle(const S2utBundle&) = delete;
    S2utBundle& operator=(const S2utBundle&) = delete;

    bool load(const std::filesystem::path& directory, int frames_per_step, std::string& error);
};

} // namespace fluxvoice
