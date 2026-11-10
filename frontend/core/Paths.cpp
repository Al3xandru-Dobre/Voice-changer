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

std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

} // namespace

fs::path default_config_dir() {
#if defined(_WIN32)
    const std::string appdata = env_or_empty("APPDATA");
    if (!appdata.empty()) return fs::path(appdata) / "fluxvoice";
    const std::string profile = env_or_empty("USERPROFILE");
    if (!profile.empty()) return fs::path(profile) / "AppData" / "Roaming" / "fluxvoice";
#elif defined(__APPLE__)
    const std::string home = env_or_empty("HOME");
    if (!home.empty()) return fs::path(home) / "Library" / "Application Support" / "fluxvoice";
#else
    const std::string xdg = env_or_empty("XDG_CONFIG_HOME");
    if (!xdg.empty()) return fs::path(xdg) / "fluxvoice";
    const std::string home = env_or_empty("HOME");
    if (!home.empty()) return fs::path(home) / ".config" / "fluxvoice";
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
    std::string windir = env_or_empty("WINDIR");
    if (windir.empty()) windir = "C:\\Windows";
    c.emplace_back(fs::path(windir) / "Fonts" / "segoeui.ttf");
    c.emplace_back(fs::path(windir) / "Fonts" / "tahoma.ttf");
    c.emplace_back(fs::path(windir) / "Fonts" / "arial.ttf");
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
