// TranslationModel — langs.json (Milestone 3 / Stage 02 contract v2) and the
// Mode B control rules from M3 / Stage 09: validated language pairs, the
// same-language guard, per-pair default Wait-k, and the "bundle has no
// translator models" state.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

struct Lang {
    std::string   code;  // "en"
    std::string   name;  // "English"
    std::uint16_t id = 0;
};

struct LangTable {
    std::vector<Lang>          langs;
    std::map<std::string, int> default_wait_k;  // "es->en" -> 3
    std::string                error;           // non-empty => the document was rejected
    bool                       from_bundle = false;

    bool ok() const { return error.empty() && !langs.empty(); }
    const Lang* by_code(std::string_view code) const;
    const Lang* by_id(std::uint16_t id) const;
    /// Pair default, or 3 (the stage-file default) when the pair is not listed.
    int default_k(std::string_view src, std::string_view tgt) const;
};

/// Strict parse: integer schema 1, non-empty `langs` with unique lowercase
/// codes and unique ids, optional `default_wait_k` of integers in [1, 10].
LangTable parse_langs_json(std::string_view text);

/// The table shipped in frontend/assets/langs.default.json, embedded so the
/// Translate panel works without a bundle. A test checks the two agree.
LangTable builtin_lang_table();

struct PairValidation {
    bool        ok = false;
    std::string message;  // reason when !ok, or an advisory note when ok
};

class TranslationModel {
public:
    TranslationModel();

    void set_table(LangTable table);
    const LangTable& table() const { return table_; }

    /// Whether the active bundle has translator_enc.onnx + translator_dec.onnx.
    void set_translator_available(bool available, std::string reason_when_not = {});
    bool translator_available() const { return available_; }
    const std::string& unavailable_reason() const { return reason_; }

    /// Unknown codes and src == tgt are rejected; the same-language message
    /// recommends Mode A (60–100 ms vs. ~330 ms).
    PairValidation validate_pair(std::string_view src, std::string_view tgt) const;

    static int clamp_wait_k(int k);
    /// First-audio onset added by Wait-k alone: k x 80 ms translation chunks.
    static double onset_estimate_ms(int k) { return 80.0 * k; }

private:
    LangTable   table_;
    bool        available_ = false;
    std::string reason_    = "no bundle loaded";
};

} // namespace fluxvoice::frontend
