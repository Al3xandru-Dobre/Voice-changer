// VoiceCatalog — voices.json / index.json parsing, bundle scanning, search.
#include "core/VoiceCatalog.hpp"

#include "core/TranslationModel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fluxvoice::frontend {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool contains_ci(std::string_view hay, const std::string& needle_lower) {
    if (needle_lower.empty()) return true;
    return lower(hay).find(needle_lower) != std::string::npos;
}

std::string read_file(const fs::path& p, bool& ok) {
    std::ifstream f(p, std::ios::binary);
    ok = static_cast<bool>(f);
    if (!ok) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// A finite JSON number that is not a boolean (fvml.contract._is_number).
bool is_number(const json& j) {
    if (!j.is_number()) return false;  // booleans are not numbers in nlohmann
    const double v = j.get<double>();
    return std::isfinite(v);
}

constexpr const char* kVoiceKeys[] = {"name", "display", "embedding", "logf0_mean", "logf0_std",
                                      "energy_mean", "energy_std", "license", "source"};
constexpr const char* kVoiceOptionalKeys[] = {"attribution"};

bool known_voice_key(const std::string& k) {
    for (const char* n : kVoiceKeys)
        if (k == n) return true;
    for (const char* n : kVoiceOptionalKeys)
        if (k == n) return true;
    return false;
}

std::vector<std::string> string_list(const json& j) {
    std::vector<std::string> out;
    if (!j.is_array()) return out;
    for (const json& e : j)
        if (e.is_string() && !e.get_ref<const std::string&>().empty()) out.push_back(e.get<std::string>());
    return out;
}

} // namespace

const char* to_string(LicenseClass c) {
    switch (c) {
        case LicenseClass::CommercialClean: return "commercial-clean";
        case LicenseClass::ResearchOnly:    return "research-only";
        case LicenseClass::Private:         return "private";
        case LicenseClass::Unknown:         return "unknown";
    }
    return "unknown";
}

LicenseClass classify_license(std::string_view license_text) {
    const std::string l = lower(license_text);
    if (l.empty() || l == "unknown") return LicenseClass::Unknown;
    if (l.find("private") != std::string::npos || l.find("do not distribute") != std::string::npos)
        return LicenseClass::Private;
    if (l.find("-nc") != std::string::npos || l.find(" nc") != std::string::npos ||
        l.find("non-commercial") != std::string::npos || l.find("noncommercial") != std::string::npos ||
        l.find("research only") != std::string::npos || l.find("research-only") != std::string::npos)
        return LicenseClass::ResearchOnly;
    return LicenseClass::CommercialClean;
}

bool is_valid_voice_name(std::string_view name) {
    if (name.empty()) return false;
    for (char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
    return true;
}

VoicesParseResult parse_voices_json(std::string_view text) {
    VoicesParseResult r;
    const json doc = json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        r.error = "voices.json must be a JSON object";
        return r;
    }
    const auto schema = doc.find("schema");
    if (schema == doc.end() || !schema->is_number_integer() || schema->get<long long>() != 1) {
        r.error = "unsupported voices.json schema " + (schema == doc.end() ? std::string("(missing)") : schema->dump());
        return r;
    }
    for (auto it = doc.begin(); it != doc.end(); ++it) {
        if (it.key() != "schema" && it.key() != "voices") {
            r.error = "voices.json: unknown key '" + it.key() + "'";
            return r;
        }
    }
    const auto voices = doc.find("voices");
    if (voices == doc.end() || !voices->is_array() || voices->empty()) {
        r.error = "voices.json: 'voices' must be a non-empty list";
        return r;
    }
    std::set<std::string> names;
    std::size_t i = 0;
    for (const json& v : *voices) {
        const std::string where = "voices[" + std::to_string(i++) + "]";
        if (!v.is_object()) { r.error = where + " must be an object"; return r; }
        for (auto it = v.begin(); it != v.end(); ++it)
            if (!known_voice_key(it.key())) { r.error = where + ": unknown key '" + it.key() + "'"; return r; }
        for (const char* k : kVoiceKeys)
            if (!v.contains(k)) { r.error = where + ": missing key '" + k + "'"; return r; }

        VoiceEntry e;
        const json& name = v["name"];
        if (!name.is_string() || !is_valid_voice_name(name.get_ref<const std::string&>())) {
            r.error = where + ": name must be lowercase [a-z0-9_-]+ (got " + name.dump() + ")";
            return r;
        }
        e.name = name.get<std::string>();
        if (!names.insert(e.name).second) { r.error = "duplicate voice '" + e.name + "'"; return r; }

        auto req_str = [&](const char* k, std::string& out) {
            const json& j = v[k];
            if (!j.is_string() || j.get_ref<const std::string&>().empty()) {
                r.error = "voice '" + e.name + "': " + k + " must be a non-empty string";
                return false;
            }
            out = j.get<std::string>();
            return true;
        };
        if (!req_str("display", e.display) || !req_str("license", e.license) || !req_str("source", e.source)) return r;
        if (v.contains("attribution") && !req_str("attribution", e.attribution)) return r;

        auto req_num = [&](const char* k, float& out) {
            const json& j = v[k];
            if (!is_number(j)) {
                r.error = "voice '" + e.name + "': " + k + " must be a finite number";
                return false;
            }
            out = j.get<float>();
            return true;
        };
        if (!req_num("logf0_mean", e.logf0_mean) || !req_num("logf0_std", e.logf0_std) ||
            !req_num("energy_mean", e.energy_mean) || !req_num("energy_std", e.energy_std))
            return r;
        if (e.logf0_std <= 0.0f) { r.error = "voice '" + e.name + "': logf0_std must be > 0"; return r; }
        if (e.energy_std <= 0.0f) { r.error = "voice '" + e.name + "': energy_std must be > 0"; return r; }

        const json& emb = v["embedding"];
        if (!emb.is_array() || emb.empty()) {
            r.error = "voice '" + e.name + "': embedding must be a list of numbers";
            return r;
        }
        for (const json& x : emb) {
            if (!is_number(x)) { r.error = "voice '" + e.name + "': embedding must be a list of numbers"; return r; }
            e.embedding.push_back(x.get<float>());
        }
        r.voices.push_back(std::move(e));
    }
    return r;
}

CatalogueIndex parse_catalogue_index(std::string_view text) {
    CatalogueIndex idx;
    const json doc = json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        idx.error = "index.json must be a JSON object";
        return idx;
    }
    const auto schema = doc.find("schema");
    if (schema == doc.end() || !schema->is_number_integer() || schema->get<long long>() != 1) {
        idx.error = "unsupported catalogue schema (expected the integer 1)";
        return idx;
    }
    if (auto t = doc.find("title"); t != doc.end() && t->is_string()) idx.title = t->get<std::string>();
    if (auto d = doc.find("description"); d != doc.end() && d->is_string()) idx.description = d->get<std::string>();
    const auto entries = doc.find("entries");
    if (entries == doc.end() || !entries->is_array()) {
        idx.error = "catalogue: 'entries' must be a list";
        return idx;
    }
    std::set<std::string> ids;
    for (const json& e : *entries) {
        if (!e.is_object()) { idx.error = "catalogue: every entry must be an object"; return idx; }
        const auto id = e.find("id");
        const auto display = e.find("display");
        if (id == e.end() || !id->is_string() || id->get_ref<const std::string&>().empty() ||
            display == e.end() || !display->is_string() || display->get_ref<const std::string&>().empty()) {
            idx.error = "catalogue: every entry needs non-empty 'id' and 'display' strings";
            return idx;
        }
        if (!ids.insert(id->get<std::string>()).second) {
            idx.error = "catalogue: duplicate id '" + id->get<std::string>() + "'";
            return idx;
        }
        CatalogueEntry c;
        c.id      = id->get<std::string>();
        c.display = display->get<std::string>();
        auto opt_str = [&](const char* k, std::string& out) {
            if (auto it = e.find(k); it != e.end() && it->is_string()) out = it->get<std::string>();
        };
        opt_str("bundle", c.bundle);
        opt_str("license", c.license);
        opt_str("attribution", c.attribution);
        opt_str("url", c.url);
        opt_str("page", c.page);
        opt_str("sample", c.sample);
        if (auto it = e.find("languages"); it != e.end()) c.languages = string_list(*it);
        if (auto it = e.find("tags"); it != e.end()) c.tags = string_list(*it);
        if (auto it = e.find("size_mb"); it != e.end() && is_number(*it)) c.size_mb = it->get<double>();
        idx.entries.push_back(std::move(c));
    }
    return idx;
}

void VoiceCatalog::clear() {
    bundles_.clear();
    indices_.clear();
}

std::size_t VoiceCatalog::scan_models_root(const fs::path& root) {
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return 0;
    std::size_t seen = 0;
    if (fs::is_regular_file(root / "voices.json", ec)) {
        if (add_bundle_dir(root)) ++seen;
    }
    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(root, ec))
        if (e.is_directory(ec) && fs::is_regular_file(e.path() / "voices.json", ec)) dirs.push_back(e.path());
    std::sort(dirs.begin(), dirs.end());
    for (const fs::path& d : dirs)
        if (add_bundle_dir(d)) ++seen;
    return seen;
}

bool VoiceCatalog::add_bundle_dir(const fs::path& dir, std::string* error) {
    std::error_code ec;
    const fs::path voices_path = dir / "voices.json";
    if (!fs::is_regular_file(voices_path, ec)) {
        if (error) *error = "no voices.json in " + dir.string();
        return false;
    }
    BundleInfo b;
    b.dir  = dir;
    b.name = dir.filename().string();
    if (b.name.empty()) b.name = dir.parent_path().filename().string();
    // Replace an earlier load of the same directory.
    bundles_.erase(std::remove_if(bundles_.begin(), bundles_.end(),
                                  [&](const BundleInfo& x) { return x.dir == dir; }),
                   bundles_.end());

    bool ok = false;
    const std::string text = read_file(voices_path, ok);
    if (!ok) {
        b.error = "cannot read " + voices_path.string();
    } else {
        VoicesParseResult r = parse_voices_json(text);
        if (!r.ok()) {
            b.error = r.error;
        } else {
            b.voices = std::move(r.voices);
            for (VoiceEntry& v : b.voices) {
                v.bundle     = b.name;
                v.bundle_dir = dir;
            }
        }
    }

    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string fn = e.path().filename().string();
        if (e.path().extension() == ".onnx") b.onnx_files.push_back(fn);
        if (fn.rfind("encoder_", 0) == 0 && e.path().extension() == ".onnx") b.has_encoder = true;
        if (fn.rfind("vocoder_", 0) == 0 && e.path().extension() == ".onnx") b.has_vocoder = true;
    }
    std::sort(b.onnx_files.begin(), b.onnx_files.end());
    b.has_translator = fs::is_regular_file(dir / "translator_enc.onnx", ec) &&
                       fs::is_regular_file(dir / "translator_dec.onnx", ec);
    if (fs::is_regular_file(dir / "langs.json", ec)) {
        bool lok = false;
        const LangTable t = parse_langs_json(read_file(dir / "langs.json", lok));
        if (lok && t.ok()) {
            b.has_langs = true;
            for (const Lang& l : t.langs) b.languages.push_back(l.code);
        } else if (b.error.empty()) {
            b.error = t.error.empty() ? "cannot read langs.json" : t.error;
        }
    }
    bundles_.push_back(std::move(b));
    std::sort(bundles_.begin(), bundles_.end(),
              [](const BundleInfo& a, const BundleInfo& c) { return a.name < c.name; });
    if (error) error->clear();
    return true;
}

bool VoiceCatalog::load_catalogue_index(const fs::path& file, std::string* error) {
    bool ok = false;
    const std::string text = read_file(file, ok);
    if (!ok) {
        if (error) *error = "cannot read " + file.string();
        return false;
    }
    CatalogueIndex idx = parse_catalogue_index(text);
    idx.path = file;
    indices_.erase(std::remove_if(indices_.begin(), indices_.end(),
                                  [&](const CatalogueIndex& x) { return x.path == file; }),
                   indices_.end());
    const bool valid = idx.ok();
    if (!valid && error) *error = idx.error;
    indices_.push_back(std::move(idx));
    return valid;
}

const BundleInfo* VoiceCatalog::bundle(std::string_view name) const {
    for (const BundleInfo& b : bundles_)
        if (b.name == name) return &b;
    return nullptr;
}

const VoiceEntry* VoiceCatalog::find_voice(std::string_view bundle_name, std::string_view name) const {
    const BundleInfo* b = bundle(bundle_name);
    if (!b) return nullptr;
    for (const VoiceEntry& v : b->voices)
        if (v.name == name) return &v;
    return nullptr;
}

std::vector<std::string> VoiceCatalog::voice_names(std::string_view bundle_name) const {
    std::vector<std::string> out;
    if (const BundleInfo* b = bundle(bundle_name))
        for (const VoiceEntry& v : b->voices) out.push_back(v.name);
    return out;
}

std::size_t VoiceCatalog::installed_voice_count() const {
    std::size_t n = 0;
    for (const BundleInfo& b : bundles_) n += b.voices.size();
    return n;
}

std::vector<VoiceCatalog::Item> VoiceCatalog::search(const Filter& f) const {
    std::vector<Item> items;
    for (const BundleInfo& b : bundles_) {
        for (const VoiceEntry& v : b.voices) {
            Item it;
            it.id            = b.name + "/" + v.name;
            it.display       = v.display;
            it.bundle        = b.name;
            it.license       = v.license;
            it.attribution   = v.attribution;
            it.source        = v.source;
            it.languages     = b.languages;
            it.license_class = v.license_class();
            it.installed     = true;
            it.voice         = &v;
            items.push_back(std::move(it));
        }
    }
    for (const CatalogueIndex& idx : indices_) {
        if (!idx.ok()) continue;
        for (const CatalogueEntry& e : idx.entries) {
            Item it;
            it.id            = e.id;
            it.display       = e.display;
            it.bundle        = e.bundle;
            it.license       = e.license;
            it.attribution   = e.attribution;
            it.page          = e.page;
            it.url           = e.url;
            it.languages     = e.languages;
            it.tags          = e.tags;
            it.size_mb       = e.size_mb;
            it.license_class = classify_license(e.license);
            it.installed     = !e.bundle.empty() && bundle(e.bundle) != nullptr;
            it.entry         = &e;
            items.push_back(std::move(it));
        }
    }

    const std::string q = lower(f.query);
    std::vector<Item> out;
    for (Item& it : items) {
        if (f.installed_only && !it.installed) continue;
        if (f.license && it.license_class != *f.license) continue;
        if (!f.language.empty() &&
            std::find(it.languages.begin(), it.languages.end(), f.language) == it.languages.end())
            continue;
        if (!f.tag.empty() && std::find(it.tags.begin(), it.tags.end(), f.tag) == it.tags.end()) continue;
        if (!q.empty()) {
            bool hit = contains_ci(it.id, q) || contains_ci(it.display, q) || contains_ci(it.bundle, q) ||
                       contains_ci(it.license, q) || contains_ci(it.attribution, q) || contains_ci(it.source, q);
            for (const std::string& l : it.languages) hit = hit || contains_ci(l, q);
            for (const std::string& t : it.tags) hit = hit || contains_ci(t, q);
            if (!hit) continue;
        }
        out.push_back(std::move(it));
    }

    auto lang_key = [](const Item& it) { return it.languages.empty() ? std::string("~") : it.languages.front(); };
    std::stable_sort(out.begin(), out.end(), [&](const Item& a, const Item& b) {
        switch (f.sort) {
            case Sort::Language: {
                const std::string la = lang_key(a), lb = lang_key(b);
                if (la != lb) return la < lb;
                break;
            }
            case Sort::License:
                if (a.license_class != b.license_class) return a.license_class < b.license_class;
                break;
            case Sort::Name:
                break;
        }
        return lower(a.display) < lower(b.display);
    });
    return out;
}

std::vector<std::string> VoiceCatalog::all_languages() const {
    std::set<std::string> s;
    for (const BundleInfo& b : bundles_) s.insert(b.languages.begin(), b.languages.end());
    for (const CatalogueIndex& i : indices_)
        for (const CatalogueEntry& e : i.entries) s.insert(e.languages.begin(), e.languages.end());
    return {s.begin(), s.end()};
}

std::vector<std::string> VoiceCatalog::all_tags() const {
    std::set<std::string> s;
    for (const CatalogueIndex& i : indices_)
        for (const CatalogueEntry& e : i.entries) s.insert(e.tags.begin(), e.tags.end());
    return {s.begin(), s.end()};
}

} // namespace fluxvoice::frontend
