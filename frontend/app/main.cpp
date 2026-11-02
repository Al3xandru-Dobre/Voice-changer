// fluxvoice_gui — entry point of the desktop frontend (frontend/README.md).
//
// Argument handling runs before any SDL or audio initialisation, so the
// `--help` / bad-argument paths are testable without a display or a sound
// card. Like src/main.cpp, this is the one application TU that includes the
// miniaudio backend header; everything else speaks AudioTypes.
#include "audio/AudioTypes.hpp"
#include "audio/MiniaudioBackend.hpp"
#include "core/AppModel.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "imgui/GuiApp.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace fluxvoice::frontend;

void print_usage(const char* argv0) {
    std::printf(
        "FluxVoice desktop frontend (Milestone 4)\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "  --minimized              Start hidden in the system tray (minimised window when no tray)\n"
        "  --mini                   Start as the compact Mini panel\n"
        "  --section <name>         Open on a section: live, devices, voices, studio, translate,\n"
        "                           catalogue, telemetry, settings\n"
        "  --config-dir <dir>       Settings location (default: per-OS config dir / fluxvoice)\n"
        "  --models-dir <dir>       Voice bundle root (default: Settings, else ./models, else the fixture bundle)\n"
        "  --preset-dir <dir>       User preset directory (default: <config>/presets)\n"
        "  --studio-dir <dir>       Voice Studio projects (default: <config>/studio)\n"
        "  --assets-root <dir>      Repository root to find presets/ and frontend/assets/\n"
        "  --smoke-frames <n>       Render n frames, then exit (verification without interaction)\n"
        "  --screenshot <file.ppm>  With --smoke-frames: dump the last frame as a binary PPM\n"
        "  -h, --help               This help\n",
        argv0);
}

bool parse_int(std::string_view s, int lo, int hi, int& out) {
    int v = 0;
    const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || v < lo || v > hi) return false;
    out = v;
    return true;
}

fs::path exe_dir(const char* argv0) {
    const char* base = SDL_GetBasePath();
    if (base && *base) return fs::path(base);
    std::error_code ec;
    const fs::path p = fs::absolute(fs::path(argv0 ? argv0 : "."), ec);
    return ec ? fs::current_path() : p.parent_path();
}

bool has_bundle(const fs::path& root) {
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return false;
    if (fs::is_regular_file(root / "voices.json", ec)) return true;
    for (const auto& e : fs::directory_iterator(root, ec))
        if (e.is_directory(ec) && fs::is_regular_file(e.path() / "voices.json", ec)) return true;
    return false;
}

} // namespace

int main(int argc, char* argv[]) {
    GuiOptions opts;
    std::string config_dir, models_dir, preset_dir, studio_dir, assets_root, section;

    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto value = [&](std::string& out) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", argv[i]);
                return false;
            }
            out = argv[++i];
            return true;
        };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "--minimized")   opts.start_minimized = true;
        else if (a == "--mini")        opts.start_mini = true;
        else if (a == "--section")     { if (!value(section)) return 1; }
        else if (a == "--config-dir")  { if (!value(config_dir)) return 1; }
        else if (a == "--models-dir")  { if (!value(models_dir)) return 1; }
        else if (a == "--preset-dir")  { if (!value(preset_dir)) return 1; }
        else if (a == "--studio-dir")  { if (!value(studio_dir)) return 1; }
        else if (a == "--assets-root") { if (!value(assets_root)) return 1; }
        else if (a == "--screenshot")  { if (!value(opts.screenshot_path)) return 1; }
        else if (a == "--smoke-frames") {
            std::string v;
            if (!value(v) || !parse_int(v, 1, 100000, opts.smoke_frames)) {
                std::fprintf(stderr, "Invalid value for --smoke-frames (expected an integer in [1, 100000])\n");
                return 1;
            }
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    static const char* const kSections[] = {"live", "devices", "voices", "studio", "translate", "catalogue", "telemetry", "settings"};
    int section_index = -1;
    if (!section.empty()) {
        for (int i = 0; i < static_cast<int>(Section::Count); ++i)
            if (section == kSections[i]) section_index = i;
        if (section_index < 0) {
            std::fprintf(stderr, "Unknown section '%s'\n", section.c_str());
            return 1;
        }
    }

    // Paths and settings.
    AppPaths paths;
    paths.config_dir = config_dir.empty() ? default_config_dir() : fs::path(config_dir);
    std::vector<std::string> warnings;
    Settings settings = Settings::load(paths.settings_file(), &warnings);
    for (const std::string& w : warnings) std::fprintf(stderr, "%s\n", w.c_str());

    const std::vector<fs::path> roots = resource_search_roots(exe_dir(argv[0]), assets_root.empty() ? fs::path() : fs::path(assets_root));
    paths.builtin_preset_dir = find_resource(roots, "presets");
    paths.assets_dir         = find_resource(roots, fs::path("frontend") / "assets");
    paths.repo_root          = paths.assets_dir.empty() ? fs::path() : paths.assets_dir.parent_path().parent_path();

    if (!models_dir.empty())              paths.models_dir = fs::path(models_dir);
    else if (!settings.models_dir.empty()) paths.models_dir = fs::path(settings.models_dir);
    else                                   paths.models_dir = find_resource(roots, "models");
    if (!has_bundle(paths.models_dir)) {
        // Development fallback: the generated identity bundle keeps the UI usable
        // (and testable) before any trained bundle is exported.
        const fs::path fixtures = find_resource(roots, fs::path("tests") / "data" / "fixtures");
        if (has_bundle(fixtures)) paths.models_dir = fixtures;
    }
    paths.user_preset_dir = !preset_dir.empty() ? fs::path(preset_dir)
                            : !settings.user_preset_dir.empty() ? fs::path(settings.user_preset_dir)
                                                                : paths.config_dir / "presets";
    paths.studio_dir = !studio_dir.empty() ? fs::path(studio_dir)
                       : !settings.studio_dir.empty() ? fs::path(settings.studio_dir)
                                                      : paths.config_dir / "studio";

    // Engine composition: the one miniaudio context, the controller inside
    // AppModel/EngineHost. A backend that failed to initialise still opens the
    // window, which then explains the failure instead of exiting silently.
    MiniaudioBackend backend;
    AppModel app(backend, paths, settings);
    app.initialise();
    if (app.catalog().bundles().empty())
        app.notify(fluxvoice::frontend::Notice::Level::Warning,
                   "No voice bundle found. Set the models directory in Settings or add a bundle folder in the Catalogue.");

    if (section_index >= 0) app.navigate(static_cast<Section>(section_index));

    GuiApp gui(app, opts);
    return gui.run();
}
