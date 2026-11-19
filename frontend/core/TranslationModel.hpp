// TranslationModel — langs.json (Milestone 3 / Stage 02 contract v2) and the
// Mode B control rules from M3 / Stage 09: validated language pairs, the
// same-language guard, per-pair default Wait-k, and the "bundle has no
// translator models" state.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

/// What the loaded translator declares in its model contract (M3 Stage 02
/// contract v2): `chunk_ms` = translator_enc chunk_samples / sample_rate,
/// `adaptive_policy` = translator_enc has_readiness_output (CTC-adaptive
/// Wait-k). The contract is embedded in the .onnx files, so the GUI learns it
/// only when the engine host loads the bundle (M4 Stage 05 / 07); until then
/// `known` is false and nothing is assumed.
struct TranslatorCaps {
    bool   known           = false;
    double chunk_ms        = 0.0;
    bool   adaptive_policy = false;
};

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
    /// Pair default, or the `wait_k` descriptor default when the pair is not listed.
    int default_k(std::string_view src, std::string_view tgt) const;
};

/// Strict parse: integer schema 1, non-empty `langs` with unique lowercase
/// codes and unique ids, optional `default_wait_k` of integers inside the
/// `wait_k` descriptor range (src/voice/VoiceParams.hpp).
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

    /// Contract-declared translator capabilities (see TranslatorCaps). The
    /// engine host sets them after loading the bundle; a bundle change resets
    /// them to unknown.
    void set_translator_caps(const TranslatorCaps& caps) { caps_ = caps; }
    const TranslatorCaps& translator_caps() const { return caps_; }
    /// The adaptive-policy control is offered only when the bundle declares it.
    bool adaptive_policy_available() const { return available_ && caps_.known && caps_.adaptive_policy; }

    /// Clamp to the `wait_k` descriptor range.
    static int clamp_wait_k(int k);
    static int wait_k_min();
    static int wait_k_max();
    /// First-audio onset added by Wait-k alone: k x the contract chunk length;
    /// nullopt while the chunk length is unknown.
    std::optional<double> onset_estimate_ms(int k) const;

private:
    LangTable      table_;
    bool           available_ = false;
    std::string    reason_    = "no bundle loaded";
    TranslatorCaps caps_{};
};

} // namespace fluxvoice::frontend
