// Views — one draw function per section (frontend/README.md §4). Views read
// AppModel and call its commands; anything that needs the window shell
// (native dialogs, clipboard, URLs, hide/quit, theme) goes through
// ShellServices, implemented by GuiApp.
#pragma once

#include "core/AppModel.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

class ShellServices {
public:
    virtual ~ShellServices() = default;

    using FilesCallback = std::function<void(const std::vector<std::string>& paths)>;

    /// Native dialogs are asynchronous: `cb` runs later, on the UI thread,
    /// with zero paths when cancelled.
    virtual void pick_files(const char* filter_name, const char* filter_pattern, bool allow_many,
                            FilesCallback cb) = 0;
    virtual void pick_folder(FilesCallback cb) = 0;

    virtual void open_url(const std::string& url) = 0;
    virtual void open_path(const std::filesystem::path& p) = 0;
    virtual void set_clipboard(const std::string& text) = 0;

    virtual void request_hide_to_tray() = 0;
    virtual void request_quit() = 0;
    virtual void toggle_mini_panel() = 0;

    virtual bool tray_available() const = 0;
    virtual const std::string& tray_unavailable_reason() const = 0;
    virtual std::string about_text() const = 0;

    /// Re-apply theme/scale after a Settings change.
    virtual void apply_appearance() = 0;
};

struct ViewContext {
    AppModel&      app;
    ShellServices& shell;
    float          scale = 1.0f;  // display scale x user scale; fonts already include it
};

void draw_live(ViewContext& ctx);
void draw_devices(ViewContext& ctx);
void draw_voices(ViewContext& ctx);
void draw_studio(ViewContext& ctx);
void draw_translate(ViewContext& ctx);
void draw_catalogue(ViewContext& ctx);
void draw_telemetry(ViewContext& ctx);
void draw_settings(ViewContext& ctx);

/// Contents of the compact always-on-top window (the shell owns the window).
void draw_mini_panel(ViewContext& ctx);

void draw_section(ViewContext& ctx, Section section);

} // namespace fluxvoice::frontend
