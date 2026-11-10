// VoiceCatalog — installed voice bundles (models/<bundle>/voices.json) and
// browsable catalogue indices (index.json), with search, filters and licence
// classification for the Catalogue and Voices panels.
//
// voices.json is read with the same rules as python_training/fvml/contract.py
// `validate_voices` (schema 1), minus the spk_emb_dim cross-check that needs
// the vocoder contract. A bundle that fails validation stays listed as invalid
// with its diagnostic — never silently skipped (frontend/README.md §9.1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

/// Licence hygiene badge (M2 research §12): default shipped bundles must be
/// commercial-clean; NC material stays behind research flags; a user's own
/// voice is private.
enum class LicenseClass : std::uint8_t { CommercialClean, ResearchOnly, Private, Unknown };

const char* to_string(LicenseClass c);
LicenseClass classify_license(std::string_view license_text);

struct VoiceEntry {
    std::string name;     // lowercase [a-z0-9_-]+, unique within the bundle
    std::string display;
    std::string license;
    std::string source;
    std::string attribution;  // optional
    std::vector<float> embedding;
    float logf0_mean  = 0.0f;
    float logf0_std   = 0.0f;
    float energy_mean = 0.0f;
    float energy_std  = 0.0f;

    // Filled by the catalogue, not by the file.
    std::string bundle;
    std::filesystem::path bundle_dir;

    LicenseClass license_class() const { return classify_license(license); }
};

struct VoicesParseResult {
    std::vector<VoiceEntry> voices;
    std::string error;  // non-empty => rejected
    bool ok() const { return error.empty(); }
};

/// Strict schema-1 parse of a voices.json document.
VoicesParseResult parse_voices_json(std::string_view text);

/// A name is acceptable for a new voice when it matches the voices.json rule.
bool is_valid_voice_name(std::string_view name);

struct BundleInfo {
    std::string           name;  // directory name
    std::filesystem::path dir;
    std::vector<VoiceEntry> voices;
    std::vector<std::string> languages;  // codes from langs.json when present
    std::vector<std::string> onnx_files; // file names found in the bundle
    bool has_encoder    = false;
    bool has_vocoder    = false;
    bool has_translator = false;  // translator_enc.onnx + translator_dec.onnx
    bool has_langs      = false;  // langs.json present and valid
    std::string error;            // voices.json / langs.json diagnostic; voices may be empty

    bool valid() const { return error.empty(); }
};

struct CatalogueEntry {
    std::string id;
    std::string display;
    std::string bundle;
    std::string license;
    std::string attribution;
    std::string url;     // archive to obtain (opened in the browser, never downloaded)
    std::string page;    // information page
    std::string sample;  // preview audio URL
    std::vector<std::string> languages;
    std::vector<std::string> tags;
    double size_mb = 0.0;
};

struct CatalogueIndex {
    std::string title;
    std::string description;
    std::vector<CatalogueEntry> entries;
    std::filesystem::path path;
    std::string error;  // non-empty => rejected
    bool ok() const { return error.empty(); }
};

/// Schema-1 index: `id` and `display` required per entry; other keys optional
/// with type checks; unknown keys ignored.
CatalogueIndex parse_catalogue_index(std::string_view text);

class VoiceCatalog {
public:
    enum class Sort : std::uint8_t { Name, Language, License };

    struct Filter {
        std::string query;                    // case-insensitive substring over text fields
        std::string language;                 // "" = any
        std::optional<LicenseClass> license;  // nullopt = any
        std::string tag;                      // "" = any
        bool installed_only = false;
        Sort sort = Sort::Name;
    };

    /// One row of the Catalogue panel: an installed voice, a catalogue entry,
    /// or both when they refer to the same bundle.
    struct Item {
        std::string id;        // "<bundle>/<voice>" or the catalogue id
        std::string display;
        std::string bundle;
        std::string license;
        std::string attribution;
        std::string source;
        std::string page;
        std::string url;
        std::vector<std::string> languages;
        std::vector<std::string> tags;
        LicenseClass license_class = LicenseClass::Unknown;
        bool   installed = false;
        double size_mb   = 0.0;
        const VoiceEntry*     voice = nullptr;  // engaged for installed voices
        const CatalogueEntry* entry = nullptr;  // engaged for catalogue entries
    };

    void clear();

    /// Load every `<root>/*/voices.json` (and `<root>/voices.json` itself when
    /// the root is a bundle). Returns the number of bundle directories seen,
    /// valid or not. A missing root is simply 0.
    std::size_t scan_models_root(const std::filesystem::path& root);

    /// Load one bundle directory. False (with `error`) when it has no
    /// voices.json at all; an invalid voices.json still registers the bundle
    /// with its diagnostic and returns true.
    bool add_bundle_dir(const std::filesystem::path& dir, std::string* error = nullptr);

    bool load_catalogue_index(const std::filesystem::path& file, std::string* error = nullptr);

    const std::vector<BundleInfo>& bundles() const { return bundles_; }
    const BundleInfo* bundle(std::string_view name) const;
    const std::vector<CatalogueIndex>& indices() const { return indices_; }

    const VoiceEntry* find_voice(std::string_view bundle, std::string_view name) const;
    std::vector<std::string> voice_names(std::string_view bundle) const;
    std::size_t installed_voice_count() const;

    std::vector<Item> search(const Filter& f) const;
    std::vector<std::string> all_languages() const;
    std::vector<std::string> all_tags() const;

private:
    std::vector<BundleInfo>     bundles_;
    std::vector<CatalogueIndex> indices_;
};

} // namespace fluxvoice::frontend
