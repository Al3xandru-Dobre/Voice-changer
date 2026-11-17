// JsonRead — typed readers for JSON that comes from outside the process
// (BUG-022). nlohmann's `value(key, default)` returns the default only when
// the key is *absent*; a present key with another type throws type_error.302,
// and nothing under frontend/ catches — one `"eta_s": null` in status.json
// was enough to terminate the GUI. These readers treat "missing" and "wrong
// type" the same: keep the default, optionally append a warning, never throw.
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace json_detail {

inline void wrong_type(const char* key, std::vector<std::string>* warnings) {
    if (warnings) warnings->push_back(std::string("'") + key + "' has the wrong type; using the default");
}

} // namespace json_detail

/// String field; default when absent or not a string.
inline std::string read_string(const nlohmann::json& j, const char* key, std::string def,
                               std::vector<std::string>* warnings = nullptr) {
    const auto it = j.find(key);
    if (it == j.end()) return def;
    if (!it->is_string()) {
        json_detail::wrong_type(key, warnings);
        return def;
    }
    return it->get<std::string>();
}

/// Number field (integer or float); default when absent or not a number.
/// Booleans do not qualify — nlohmann keeps the types distinct.
inline double read_number(const nlohmann::json& j, const char* key, double def,
                          std::vector<std::string>* warnings = nullptr) {
    const auto it = j.find(key);
    if (it == j.end()) return def;
    if (!it->is_number()) {
        json_detail::wrong_type(key, warnings);
        return def;
    }
    return it->get<double>();
}

/// Integer field; default when absent, fractional or not a number at all.
inline long long read_int(const nlohmann::json& j, const char* key, long long def,
                          std::vector<std::string>* warnings = nullptr) {
    const auto it = j.find(key);
    if (it == j.end()) return def;
    if (!it->is_number_integer()) {
        json_detail::wrong_type(key, warnings);
        return def;
    }
    return it->get<long long>();
}

/// Boolean field; default when absent or not a boolean.
inline bool read_bool(const nlohmann::json& j, const char* key, bool def,
                      std::vector<std::string>* warnings = nullptr) {
    const auto it = j.find(key);
    if (it == j.end()) return def;
    if (!it->is_boolean()) {
        json_detail::wrong_type(key, warnings);
        return def;
    }
    return it->get<bool>();
}

} // namespace fluxvoice::frontend
