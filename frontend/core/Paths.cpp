// Paths — config directory, resource roots and font discovery.
#include "core/Paths.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifndef FLUXVOICE_SOURCE_DIR
#define FLUXVOICE_SOURCE_DIR ""
#endif

namespace fluxvoice::frontend {

namespace fs = std::filesystem;

namespace {

std::string from_u8(const std::u8string& u) { return std::string(u.begin(), u.end()); }

} // namespace

fs::path path_from_utf8(std::string_view utf8) {
    return fs::path(std::u8string(utf8.begin(), utf8.end()));
}

std::string path_to_utf8(const fs::path& p) { return from_u8(p.u8string()); }

std::string path_to_utf8_generic(const fs::path& p) { return from_u8(p.generic_u8string()); }

fs::path env_path(const char* name) {
#if defined(_WIN32)
    std::wstring wname;
    for (const char* c = name; *c != '\0'; ++c) wname.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*c)));
    wchar_t*    buffer = nullptr;
    std::size_t len    = 0;
    if (_wdupenv_s(&buffer, &len, wname.c_str()) != 0 || buffer == nullptr) return {};
    fs::path result(buffer);   // wide string: non-ASCII values stay intact
    std::free(buffer);
    return result;
#else
    const char* v = std::getenv(name);
    return v != nullptr ? fs::path(v) : fs::path();
#endif
}

fs::path default_config_dir() {
#if defined(_WIN32)
    const fs::path appdata = env_path("APPDATA");
    if (!appdata.empty()) return appdata / "fluxvoice";
    const fs::path profile = env_path("USERPROFILE");
    if (!profile.empty()) return profile / "AppData" / "Roaming" / "fluxvoice";
#elif defined(__APPLE__)
    const fs::path home = env_path("HOME");
    if (!home.empty()) return home / "Library" / "Application Support" / "fluxvoice";
#else
    const fs::path xdg = env_path("XDG_CONFIG_HOME");
    if (!xdg.empty()) return xdg / "fluxvoice";
    const fs::path home = env_path("HOME");
    if (!home.empty()) return home / ".config" / "fluxvoice";
#endif
    return fs::current_path() / ".fluxvoice";
}

std::vector<fs::path> resource_search_roots(const fs::path& exe_dir, const fs::path& override_root) {
    std::vector<fs::path> roots;
    if (!override_root.empty()) roots.push_back(override_root);
    fs::path p = exe_dir;
    for (int i = 0; i < 5 && !p.empty(); ++i) {
        roots.push_back(p);
        const fs::path parent = p.parent_path();
        if (parent == p) break;
        p = parent;
    }
    const std::string src = FLUXVOICE_SOURCE_DIR;
    if (!src.empty()) roots.emplace_back(src);
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec) roots.push_back(cwd);
    return roots;
}

fs::path find_resource(const std::vector<fs::path>& roots, const fs::path& relative) {
    std::error_code ec;
    for (const fs::path& r : roots) {
        const fs::path candidate = r / relative;
        if (fs::exists(candidate, ec)) return candidate;
    }
    return {};
}

std::vector<fs::path> system_font_candidates() {
    std::vector<fs::path> c;
#if defined(_WIN32)
    fs::path windir = env_path("WINDIR");
    if (windir.empty()) windir = "C:\\Windows";
    c.emplace_back(windir / "Fonts" / "segoeui.ttf");
    c.emplace_back(windir / "Fonts" / "tahoma.ttf");
    c.emplace_back(windir / "Fonts" / "arial.ttf");
#elif defined(__APPLE__)
    c.emplace_back("/System/Library/Fonts/SFNS.ttf");
    c.emplace_back("/System/Library/Fonts/Supplemental/Arial.ttf");
    c.emplace_back("/Library/Fonts/Arial.ttf");
#else
    c.emplace_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    c.emplace_back("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf");
    c.emplace_back("/usr/share/fonts/noto/NotoSans-Regular.ttf");
    c.emplace_back("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf");
    c.emplace_back("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf");
    c.emplace_back("/usr/share/fonts/TTF/DejaVuSans.ttf");
    c.emplace_back("/usr/share/fonts/dejavu/DejaVuSans.ttf");
    c.emplace_back("/usr/share/fonts/truetype/freefont/FreeSans.ttf");
#endif
    return c;
}

fs::path find_system_font() {
    std::error_code ec;
    for (const fs::path& p : system_font_candidates())
        if (fs::is_regular_file(p, ec)) return p;
    return {};
}

} // namespace fluxvoice::frontend
