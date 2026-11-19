// UrlPolicy — which strings the shell may hand to the OS URL opener. Catalogue
// `page`/`url` fields come from index.json files (possibly user supplied), so
// only plain web links are allowed. Toolkit-free and header-only.
#pragma once

#include <cstddef>
#include <string_view>

namespace fluxvoice::frontend {

/// True only for `http://` or `https://` (scheme case-insensitive) followed by
/// a non-empty host; false for control characters, spaces, backslashes and
/// every other scheme (`file:`, `javascript:`, `data:`), relative paths, empty.
inline bool is_web_url(std::string_view url) {
    for (const char c : url) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20 || u == 0x7F || c == ' ' || c == '\\') return false;
    }
    const auto ieq_prefix = [&](std::string_view prefix) {
        if (url.size() < prefix.size()) return false;
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            char c = url[i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != prefix[i]) return false;
        }
        return true;
    };
    std::size_t start = 0;
    if (ieq_prefix("https://")) start = 8;
    else if (ieq_prefix("http://")) start = 7;
    else return false;
    // The host must start with something other than a path/query/fragment separator.
    if (start >= url.size()) return false;
    const char h = url[start];
    return h != '/' && h != '?' && h != '#' && h != ':' && h != '@';
}

} // namespace fluxvoice::frontend
