// Paths — per-OS configuration directory, asset discovery and system-font
// candidates. Uses only the environment (std::getenv) and <filesystem>, so it
// needs no OS headers and compiles identically on every leg.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

/// `%APPDATA%\fluxvoice` on Windows, `~/Library/Application Support/fluxvoice`
/// on macOS, `$XDG_CONFIG_HOME/fluxvoice` or `~/.config/fluxvoice` elsewhere.
/// Falls back to the current directory when no home is known.
std::filesystem::path default_config_dir();

/// Candidate directories that may hold a repository-relative resource such as
/// `presets/` or `frontend/assets/`, checked in order: `--assets-root`
/// override, the executable's directory and its parents (up to 4 levels), the
/// compile-time source directory, the current directory.
std::vector<std::filesystem::path> resource_search_roots(const std::filesystem::path& exe_dir,
                                                         const std::filesystem::path& override_root = {});

/// First `root / relative` that exists among `resource_search_roots`, or empty.
std::filesystem::path find_resource(const std::vector<std::filesystem::path>& roots,
                                    const std::filesystem::path& relative);

/// Candidate TTF files for the UI font, in preference order for this OS.
std::vector<std::filesystem::path> system_font_candidates();

/// First existing candidate, or empty (callers use the embedded font).
std::filesystem::path find_system_font();

} // namespace fluxvoice::frontend
