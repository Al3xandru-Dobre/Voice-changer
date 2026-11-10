// GuiApp — the window shell: SDL3 window + OpenGL context + Dear ImGui
// lifecycle, the main loop (UI thread = control thread), the tray, the Mini
// panel (the same window morphed into a compact always-on-top strip), native
// dialogs marshalled back to the UI thread, and the smoke-test / screenshot
// mode used for headless verification (frontend/README.md §8).
#pragma once

#include "core/AppModel.hpp"
#include "imgui/Views.hpp"
#include "platform/Tray.hpp"

#include <SDL3/SDL.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ImFont;

namespace fluxvoice::frontend {

struct GuiOptions {
    bool        start_minimized = false;  // overrides Settings when true
    bool        start_mini      = false;  // open directly as the Mini panel
    int         smoke_frames    = 0;      // > 0: render this many frames, then exit
    std::string screenshot_path;          // with smoke_frames: dump the last frame (binary PPM)
};

class GuiApp final : public ShellServices {
public:
    GuiApp(AppModel& app, GuiOptions opts);
    ~GuiApp() override;

    GuiApp(const GuiApp&)            = delete;
    GuiApp& operator=(const GuiApp&) = delete;

    /// 0 on a clean exit, 1 when SDL/OpenGL initialisation failed.
    int run();

    // ---- ShellServices ----
    void pick_files(const char* filter_name, const char* filter_pattern, bool allow_many, FilesCallback cb) override;
    void pick_folder(FilesCallback cb) override;
    void open_url(const std::string& url) override;
    void open_path(const std::filesystem::path& p) override;
    void set_clipboard(const std::string& text) override;
    void request_hide_to_tray() override;
    void request_quit() override { running_ = false; }
    void toggle_mini_panel() override;
    bool tray_available() const override { return tray_ && tray_->available(); }
    const std::string& tray_unavailable_reason() const override;
    std::string about_text() const override;
    void apply_appearance() override;

private:
    using Clock = std::chrono::steady_clock;

    struct PendingDialog {
        GuiApp*       self = nullptr;
        FilesCallback cb;
        std::string   filter_name;
        std::string   filter_pattern;
        SDL_DialogFileFilter filter{};
    };
    struct DialogResult {
        FilesCallback            cb;
        std::vector<std::string> paths;
    };

    static void SDLCALL dialog_callback(void* userdata, const char* const* filelist, int filter);
    static SDL_HitTestResult SDLCALL mini_hit_test(SDL_Window* win, const SDL_Point* area, void* data);

    bool init();
    void shutdown();
    void load_fonts();
    void handle_event(const SDL_Event& e);
    void process_intents();
    void process_dialog_results();
    void update_tray(Clock::time_point now);
    void render_frame();
    void draw_main_window();
    void draw_menu_bar();
    void draw_rail();
    void draw_status_bar();
    void draw_toast();
    void handle_shortcuts();
    void show_main();
    void hide_main();
    void enter_mini();
    void leave_mini();
    void save_screenshot();
    bool resolve_dark() const;
    float display_scale() const;

    AppModel&   app_;
    GuiOptions  opts_;
    SDL_Window*   window_ = nullptr;
    SDL_GLContext gl_     = nullptr;
    std::unique_ptr<TrayHost> tray_;
    std::string no_tray_reason_ = "tray not initialised";

    bool  running_   = true;
    bool  hidden_    = false;
    bool  minimized_ = false;
    bool  mini_      = false;
    int   main_x_ = 0, main_y_ = 0, main_w_ = 0, main_h_ = 0;  // restored when leaving the Mini panel
    float scale_ = 1.0f;
    int   frames_ = 0;
    Clock::time_point last_tray_update_{};

    std::vector<std::unique_ptr<PendingDialog>> pending_dialogs_;
    std::mutex                dialog_mutex_;
    std::deque<DialogResult>  dialog_results_;
};

} // namespace fluxvoice::frontend
