// Paths — per-OS configuration directory, asset discovery and system-font
// candidates. Uses only the environment (getenv / _wdupenv_s on Windows) and
// <filesystem>, so it needs no OS headers and compiles identically on every leg.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

/// UTF-8 text (SDL dialog results, argv under SDL_main, JSON strings, ImGui
/// buffers) as a path. Built on `std::u8string`: `fs::path(std::string)` reads
/// the bytes in the ANSI code page on Windows and garbles non-ASCII names.
/// On POSIX the bytes are kept verbatim, so any byte string round-trips.
std::filesystem::path path_from_utf8(std::string_view utf8);

/// A path as UTF-8 text for JSON, ImGui, SDL and messages; never use
/// `path.string()` for these (ANSI code page on Windows). The `_generic`
/// variant uses '/' separators (POSIX shell commands, file:// URLs).
std::string path_to_utf8(const std::filesystem::path& p);
std::string path_to_utf8_generic(const std::filesystem::path& p);

/// Value of an environment variable as a path; empty when unset or empty.
/// Windows reads the wide value (`_wdupenv_s`) so non-ASCII `%APPDATA%` survives
/// and MSVC's C4996 deprecation of `std::getenv` is avoided (BUG-018).
std::filesystem::path env_path(const char* name);

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
