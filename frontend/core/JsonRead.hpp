// JsonRead — typed readers for JSON that comes from outside the process
// (BUG-022). nlohmann's `value(key, default)` returns the default only when
// the key is *absent*; a present key with another type throws type_error.302,
// and nothing under frontend/ catches — one `"eta_s": null` in status.json
// was enough to terminate the GUI. These readers treat "missing" and "wrong
// type" the same: keep the default, optionally append a warning, never throw.
#pragma once

#include <nlohmann/json.hpp>

#include <cmath>
#include <cfloat>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
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

/// Checked numeric conversions. Casting an out-of-range or non-finite double
/// to an integral type (or a double beyond FLT_MAX to float) is undefined
/// behaviour, and a modular wrap of a wide integer silently yields a wrong
/// value, so every JSON number that lands in a narrower C++ type goes through
/// these: a hostile value becomes the default or a rejected record.

/// `j` as a `T` in [lo, hi]; nullopt when it is not a number, is non-finite,
/// has a fractional part, or lies outside the range. A float-typed value such
/// as 3.0 is accepted when integral and below 2^63 in magnitude.
template <typename T>
std::optional<T> json_to_integral(const nlohmann::json& j, T lo = std::numeric_limits<T>::lowest(),
                                  T hi = std::numeric_limits<T>::max()) {
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>, "integral, non-bool target");
    auto in_range = [&](auto v) -> std::optional<T> {
        if (std::cmp_less(v, lo) || std::cmp_greater(v, hi)) return std::nullopt;
        return static_cast<T>(v);
    };
    if (j.is_number_unsigned()) return in_range(j.get<unsigned long long>());
    if (j.is_number_integer()) return in_range(j.get<long long>());
    if (j.is_number_float()) {
        const double d = j.get<double>();
        if (!std::isfinite(d) || d != std::trunc(d) || std::fabs(d) >= 9223372036854775808.0) return std::nullopt;
        return in_range(static_cast<long long>(d));
    }
    return std::nullopt;
}

/// `j` as a float; nullopt when it is not a number, non-finite, or beyond FLT_MAX.
inline std::optional<float> json_to_float(const nlohmann::json& j) {
    if (!j.is_number()) return std::nullopt;
    const double d = j.get<double>();
    if (!std::isfinite(d) || std::fabs(d) > static_cast<double>(FLT_MAX)) return std::nullopt;
    return static_cast<float>(d);
}

/// Integral field in [lo, hi]; default when absent, wrong-typed, fractional or out of range.
template <typename T>
T read_integral(const nlohmann::json& j, const char* key, T def, T lo = std::numeric_limits<T>::lowest(),
                T hi = std::numeric_limits<T>::max(), std::vector<std::string>* warnings = nullptr) {
    const auto it = j.find(key);
    if (it == j.end()) return def;
    if (const auto v = json_to_integral<T>(*it, lo, hi)) return *v;
    json_detail::wrong_type(key, warnings);
    return def;
}

/// Float field; default when absent, not a number, non-finite or beyond FLT_MAX.
inline float read_float(const nlohmann::json& j, const char* key, float def,
                        std::vector<std::string>* warnings = nullptr) {
    const auto it = j.find(key);
    if (it == j.end()) return def;
    if (const auto v = json_to_float(*it)) return *v;
    json_detail::wrong_type(key, warnings);
    return def;
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
