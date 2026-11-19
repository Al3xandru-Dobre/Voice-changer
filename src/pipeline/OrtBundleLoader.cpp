// ort_bundle_loader() — ModelBundle / S2utBundle behind HostModelLoader
// (Milestone 4 / Stage 05; see ModeAHost.hpp). Compiles with and without ONNX
// Runtime: without it every load fails with kBuiltWithoutOrtMessage and the
// host reports HostErrorKind::kBuiltWithoutOrt.
#include "pipeline/ModeAHost.hpp"

#include "pipeline/ModelBundle.hpp"
#include "pipeline/OrtModel.hpp"

#include <exception>

namespace fluxvoice {

namespace {

// Both bundles are non-movable (translator_dec points into translator_enc), so
// each lives on the heap behind a unique_ptr for the HostModels' lifetime.
class PlainModels final : public HostModels {
public:
    explicit PlainModels(std::unique_ptr<ModelBundle> bundle) : bundle_(std::move(bundle)) {}
    ModelStep& encoder() noexcept override { return bundle_->encoder; }
    ModelStep& vocoder() noexcept override { return bundle_->vocoder; }
    const std::vector<VoiceEntry>& voices() const noexcept override { return bundle_->voices; }

private:
    std::unique_ptr<ModelBundle> bundle_;
};

class TranslatorModels final : public HostModels {
public:
    explicit TranslatorModels(std::unique_ptr<S2utBundle> bundle) : bundle_(std::move(bundle)) {}
    ModelStep& encoder() noexcept override { return bundle_->encoder; }
    ModelStep& vocoder() noexcept override { return bundle_->vocoder; }
    ModelStep* translator_enc() noexcept override { return &bundle_->translator_enc; }
    ModelStep* translator_dec() noexcept override { return &bundle_->translator_dec; }
    const std::vector<VoiceEntry>& voices() const noexcept override { return bundle_->voices; }
    const LangCatalog* langs() const noexcept override { return &bundle_->langs; }

private:
    std::unique_ptr<S2utBundle> bundle_;
};

class OrtBundleLoader final : public HostModelLoader {
public:
    bool has_translator(const std::filesystem::path& dir, int frames_per_step) const override {
        S2utBundlePaths paths;
        std::string error;
        return locate_s2ut_bundle(dir, frames_per_step, paths, error);
    }

    std::unique_ptr<HostModels> load(const std::filesystem::path& dir, int frames_per_step,
                                     bool with_translator, std::string& error) const override {
        try {
            if (!ort_available()) {
                error = kBuiltWithoutOrtMessage;
                return nullptr;
            }
            if (with_translator) {
                auto bundle = std::make_unique<S2utBundle>();
                if (!bundle->load(dir, frames_per_step, error)) return nullptr;
                return std::make_unique<TranslatorModels>(std::move(bundle));
            }
            auto bundle = std::make_unique<ModelBundle>();
            if (!bundle->load(dir, frames_per_step, error)) return nullptr;
            return std::make_unique<PlainModels>(std::move(bundle));
        } catch (const std::exception& e) {
            error = e.what();
            return nullptr;
        }
    }
};

} // namespace

std::shared_ptr<const HostModelLoader> ort_bundle_loader() {
    static const std::shared_ptr<const HostModelLoader> instance = std::make_shared<OrtBundleLoader>();
    return instance;
}

} // namespace fluxvoice
