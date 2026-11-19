// LangCatalog — a Mode B bundle's `langs.json` (Milestone 3 / Stages 07–08).
//
// `langs.json` (schema 1, Documents/Milestone 3/02_translator_contract_v2_fixtures.md):
//   {"schema": 1,
//    "langs": [{"code": "en", "id": 0, "name": "English"}, ...],
//    "default_wait_k": {"es->en": 3, "de->en": 4, ...}}
//
// The document is validated by validate_langs_json() (ModelContract.hpp, same
// rules and messages as python_training/fvml/contract.py), optionally against
// a translator contract's `langs` map, then copied into plain values. ORT-free
// and allocation only at load time, so the CLI can resolve --src-lang /
// --tgt-lang / --wait-k and serve --list-langs before any model session or
// audio device is opened.
#pragma once

#include "pipeline/ModelContract.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fluxvoice {

/// Wait-k used for a pair absent from `default_wait_k` (Stage 08 CLI default).
inline constexpr int kDefaultWaitK = 3;

struct LangEntry {
    std::string code;   // lowercase ISO 639-1 tag, e.g. "es"
    int id = 0;         // translator src_lang / tgt_lang token id
    std::string name;   // display name, e.g. "Spanish"
};

struct LangPairDefault {
    std::string src;
    std::string tgt;
    int wait_k = kDefaultWaitK;
};

struct LangCatalog {
    std::vector<LangEntry> langs;               // file order
    std::vector<LangPairDefault> default_wait_k; // file order of the object keys

    /// A language by code ("es") or by its decimal id ("1"); nullptr if unknown.
    const LangEntry* find(std::string_view code_or_id) const noexcept {
        for (const auto& l : langs)
            if (l.code == code_or_id) return &l;
        if (code_or_id.empty() || code_or_id.size() > 5) return nullptr;
        int id = 0;
        for (char c : code_or_id) {
            if (c < '0' || c > '9') return nullptr;
            id = id * 10 + (c - '0');
        }
        for (const auto& l : langs)
            if (l.id == id) return &l;
        return nullptr;
    }

    /// default_wait_k["<src>-><tgt>"] (codes), or `fallback` when absent.
    int default_k(std::string_view src_code, std::string_view tgt_code,
                  int fallback = kDefaultWaitK) const noexcept {
        for (const auto& d : default_wait_k)
            if (d.src == src_code && d.tgt == tgt_code) return d.wait_k;
        return fallback;
    }
};

/// Parse + validate a langs.json document. With `translator` set, its
/// code -> id map must equal the translator contract's `langs`. On failure
/// `out` is left unchanged and `error` names the problem.
inline bool parse_lang_catalog(std::string_view text, const ModelContract* translator,
                               LangCatalog& out, std::string& error) {
    if (!validate_langs_json(text, translator, error)) return false;
    try {
        const auto doc = nlohmann::json::parse(text.begin(), text.end());
        LangCatalog cat;
        for (const auto& e : doc.at("langs")) {
            cat.langs.push_back({e.at("code").get<std::string>(), e.at("id").get<int>(),
                                 e.at("name").get<std::string>()});
        }
        const auto dk = doc.find("default_wait_k");
        if (dk != doc.end()) {
            for (auto it = dk->begin(); it != dk->end(); ++it) {
                const std::string& key = it.key();
                const std::size_t arrow = key.find("->");
                cat.default_wait_k.push_back(
                    {key.substr(0, arrow), key.substr(arrow + 2), it.value().get<int>()});
            }
        }
        out = std::move(cat);
        error.clear();
        return true;
    } catch (const std::exception& e) {
        // validate_langs_json() accepted the document, so this is unreachable
        // in practice; keep the no-throw contract anyway.
        error = std::string("langs.json: ") + e.what();
        return false;
    }
}

inline bool read_lang_catalog(const std::filesystem::path& path, const ModelContract* translator,
                              LangCatalog& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot read " + path.string();
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // validate_langs_json() messages already name langs.json.
    return parse_lang_catalog(text, translator, out, error);
}

} // namespace fluxvoice
