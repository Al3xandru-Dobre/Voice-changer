// TranslationModel — langs.json parsing and pair validation.
#include "core/TranslationModel.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace fluxvoice::frontend {

using nlohmann::json;

namespace {

bool is_lower_code(std::string_view s) {
    if (s.size() < 2 || s.size() > 8) return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || c == '-')) return false;
    return true;
}

} // namespace

const Lang* LangTable::by_code(std::string_view code) const {
    for (const Lang& l : langs)
        if (l.code == code) return &l;
    return nullptr;
}

const Lang* LangTable::by_id(std::uint16_t id) const {
    for (const Lang& l : langs)
        if (l.id == id) return &l;
    return nullptr;
}

int LangTable::default_k(std::string_view src, std::string_view tgt) const {
    const auto it = default_wait_k.find(std::string(src) + "->" + std::string(tgt));
    return it == default_wait_k.end() ? 3 : it->second;
}

LangTable parse_langs_json(std::string_view text) {
    LangTable t;
    const json doc = json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        t.error = "langs.json must be a JSON object";
        return t;
    }
    const auto schema = doc.find("schema");
    if (schema == doc.end() || !schema->is_number_integer() || schema->get<long long>() != 1) {
        t.error = "unsupported langs.json schema (expected the integer 1)";
        return t;
    }
    const auto langs = doc.find("langs");
    if (langs == doc.end() || !langs->is_array() || langs->empty()) {
        t.error = "langs.json: 'langs' must be a non-empty list";
        return t;
    }
    std::set<std::string> codes;
    std::set<long long> ids;
    for (const json& e : *langs) {
        if (!e.is_object()) { t.error = "langs.json: every language must be an object"; return t; }
        const auto code = e.find("code");
        const auto id   = e.find("id");
        if (code == e.end() || !code->is_string() || !is_lower_code(code->get_ref<const std::string&>())) {
            t.error = "langs.json: 'code' must be a lowercase language tag";
            return t;
        }
        if (id == e.end() || !id->is_number_integer() || id->get<long long>() < 0 || id->get<long long>() > 65535) {
            t.error = "langs.json: 'id' must be an integer in [0, 65535] for '" + code->get<std::string>() + "'";
            return t;
        }
        if (!codes.insert(code->get<std::string>()).second) {
            t.error = "langs.json: duplicate code '" + code->get<std::string>() + "'";
            return t;
        }
        if (!ids.insert(id->get<long long>()).second) {
            t.error = "langs.json: duplicate id for '" + code->get<std::string>() + "'";
            return t;
        }
        Lang l;
        l.code = code->get<std::string>();
        l.id   = static_cast<std::uint16_t>(id->get<long long>());
        const auto name = e.find("name");
        l.name = (name != e.end() && name->is_string() && !name->get_ref<const std::string&>().empty())
                     ? name->get<std::string>() : l.code;
        t.langs.push_back(std::move(l));
    }
    const auto dk = doc.find("default_wait_k");
    if (dk != doc.end()) {
        if (!dk->is_object()) { t.error = "langs.json: 'default_wait_k' must be an object"; return t; }
        for (auto it = dk->begin(); it != dk->end(); ++it) {
            if (!it->is_number_integer() || it->get<long long>() < 1 || it->get<long long>() > 10) {
                t.error = "langs.json: default_wait_k['" + it.key() + "'] must be an integer in [1, 10]";
                return t;
            }
            t.default_wait_k[it.key()] = static_cast<int>(it->get<long long>());
        }
    }
    return t;
}

LangTable builtin_lang_table() {
    static constexpr std::string_view kBuiltin = R"({
  "schema": 1,
  "langs": [
    { "code": "en", "id": 0, "name": "English" },
    { "code": "es", "id": 1, "name": "Spanish" },
    { "code": "fr", "id": 2, "name": "French" },
    { "code": "de", "id": 3, "name": "German" }
  ],
  "default_wait_k": { "es->en": 3, "fr->en": 3, "de->en": 4, "en->es": 3, "en->fr": 3, "en->de": 4 }
})";
    return parse_langs_json(kBuiltin);
}

TranslationModel::TranslationModel() : table_(builtin_lang_table()) {}

void TranslationModel::set_table(LangTable table) {
    if (table.ok()) table_ = std::move(table);
}

void TranslationModel::set_translator_available(bool available, std::string reason_when_not) {
    available_ = available;
    reason_    = available ? std::string() : std::move(reason_when_not);
    if (!available && reason_.empty())
        reason_ = "bundle has no translator models (translator_enc.onnx / translator_dec.onnx)";
}

PairValidation TranslationModel::validate_pair(std::string_view src, std::string_view tgt) const {
    PairValidation v;
    const Lang* s = table_.by_code(src);
    const Lang* t = table_.by_code(tgt);
    if (!s) { v.message = "unknown source language '" + std::string(src) + "'"; return v; }
    if (!t) { v.message = "unknown target language '" + std::string(tgt) + "'"; return v; }
    if (s->code == t->code) {
        v.message = "Source and target are the same language. Use the voice changer (Mode A): "
                    "about 60-100 ms added latency instead of about 330 ms for translation.";
        return v;
    }
    v.ok = true;
    if (!available_) v.message = reason_;
    return v;
}

int TranslationModel::clamp_wait_k(int k) { return k < 1 ? 1 : (k > 10 ? 10 : k); }

} // namespace fluxvoice::frontend
