// GuiApp — SDL3 + OpenGL 3 + Dear ImGui shell implementation.
#include "imgui/GuiApp.hpp"

#include "core/AppModel.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "core/UrlPolicy.hpp"
#include "core/TrayModel.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Views.hpp"
#include "imgui/Widgets.hpp"
#include "platform/SdlTray.hpp"
#include "platform/Tray.hpp"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef FLUXVOICE_VERSION
#define FLUXVOICE_VERSION "dev"
#endif

namespace fluxvoice::frontend {

namespace {

// The few OpenGL entry points the shell calls itself (clear, viewport,
// screenshot). Resolved through SDL so no system GL headers are needed on any
// platform; Dear ImGui's renderer backend has its own loader.
#if defined(_WIN32)
#define FV_GLAPIENTRY __stdcall
#else
#define FV_GLAPIENTRY
#endif
struct GlFns {
    void (FV_GLAPIENTRY* Viewport)(int, int, int, int)                             = nullptr;
    void (FV_GLAPIENTRY* ClearColor)(float, float, float, float)                   = nullptr;
    void (FV_GLAPIENTRY* Clear)(unsigned int)                                      = nullptr;
    void (FV_GLAPIENTRY* ReadPixels)(int, int, int, int, unsigned int, unsigned int, void*) = nullptr;
    void (FV_GLAPIENTRY* Finish)()                                                 = nullptr;

    template <typename F>
    static bool load_one(F& out, const char* name) {
        SDL_FunctionPointer p = SDL_GL_GetProcAddress(name);
        out = reinterpret_cast<F>(p);
        return out != nullptr;
    }
    bool load() {
        return load_one(Viewport, "glViewport") && load_one(ClearColor, "glClearColor") &&
               load_one(Clear, "glClear") && load_one(ReadPixels, "glReadPixels") && load_one(Finish, "glFinish");
    }
};
constexpr unsigned int kGlColorBufferBit = 0x4000;
constexpr unsigned int kGlRgba           = 0x1908;
constexpr unsigned int kGlUnsignedByte   = 0x1401;

GlFns g_gl;

const char* section_label(Section s) { return to_string(s); }

} // namespace

GuiApp::GuiApp(AppModel& app, GuiOptions opts) : app_(app), opts_(std::move(opts)) {}

GuiApp::~GuiApp() { shutdown(); }

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool GuiApp::init() {
    SDL_SetAppMetadata("FluxVoice", FLUXVOICE_VERSION, "dev.fluxvoice.gui");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }

#if defined(__APPLE__)
    const char* glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    const char* glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);

    scale_ = display_scale();
    const Settings& s = app_.settings();
    const int w = static_cast<int>(static_cast<float>(s.window_w > 640 ? s.window_w : 1180) * scale_);
    const int h = static_cast<int>(static_cast<float>(s.window_h > 480 ? s.window_h : 760) * scale_);
    const SDL_WindowFlags flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    window_ = SDL_CreateWindow("FluxVoice", w, h, flags);
    if (!window_) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
    SDL_SetWindowMinimumSize(window_, static_cast<int>(720.0f * scale_), static_cast<int>(480.0f * scale_));
    gl_ = SDL_GL_CreateContext(window_);
    if (!gl_) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return false;
    }
    SDL_GL_MakeCurrent(window_, gl_);
    SDL_GL_SetSwapInterval(1);
    if (!g_gl.load()) {
        std::fprintf(stderr, "OpenGL entry points unavailable\n");
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // layout is fixed; nothing to persist
    load_fonts();
    apply_appearance();
    if (!ImGui_ImplSDL3_InitForOpenGL(window_, gl_) || !ImGui_ImplOpenGL3_Init(glsl_version)) {
        std::fprintf(stderr, "Dear ImGui backend initialisation failed\n");
        return false;
    }

    auto sdl_tray = std::make_unique<SdlTray>();
    if (sdl_tray->available()) {
        tray_ = std::move(sdl_tray);
    } else {
        no_tray_reason_ = sdl_tray->unavailable_reason();
        tray_ = std::make_unique<NullTray>(no_tray_reason_);
    }

    const bool start_hidden = (opts_.start_minimized || s.start_minimized) && opts_.smoke_frames == 0;
    if (start_hidden && tray_->available()) {
        hidden_ = true;
        app_.set_window_visible(false);
    } else {
        SDL_ShowWindow(window_);
        if (start_hidden) SDL_MinimizeWindow(window_);  // no tray: fall back to a minimised window
        app_.set_window_visible(true);
    }
    if (opts_.start_mini) enter_mini();
    return true;
}

void GuiApp::shutdown() {
    tray_.reset();
    if (ImGui::GetCurrentContext()) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }
    if (gl_) {
        SDL_GL_DestroyContext(gl_);
        gl_ = nullptr;
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

float GuiApp::display_scale() const {
    const float sys = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    const float user = app_.settings().ui_scale;
    const float s = (sys > 0.1f ? sys : 1.0f) * (user > 0.1f ? user : 1.0f);
    return s;
}

void GuiApp::load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    const std::filesystem::path font = find_system_font();
    ImFont* f = nullptr;
    if (!font.empty()) f = io.Fonts->AddFontFromFileTTF(path_to_utf8(font).c_str(), tokens::kBaseFont);
    if (!f) f = io.Fonts->AddFontDefaultVector();
    io.FontDefault = f;
}

bool GuiApp::resolve_dark() const {
    switch (app_.settings().theme) {
        case Theme::Dark:   return true;
        case Theme::Light:  return false;
        case Theme::System: return SDL_GetSystemTheme() != SDL_SYSTEM_THEME_LIGHT;
    }
    return true;
}

void GuiApp::apply_appearance() {
    scale_ = display_scale();
    apply_theme(resolve_dark(), scale_);
    ImGui::GetStyle().FontScaleDpi = scale_;
}

int GuiApp::run() {
    if (!init()) {
        shutdown();
        return 1;
    }
    auto last = Clock::now();
    const auto run_started = last;
    // Last line of defence (BUG-022): the typed JSON readers make malformed
    // external files non-fatal, but anything else unexpected must not
    // terminate the process with the stream still open. One failing frame
    // stops the stream in order, shows the error and keeps the loop going; a
    // deterministic per-frame failure would otherwise spin forever, so after
    // kMaxConsecutiveFrameErrors failing frames in a row the loop leaves
    // through the orderly shutdown below. Lines start with "[imgui-error]":
    // the headless smoke run expects none.
    constexpr int kMaxConsecutiveFrameErrors = 30;
    int consecutive_errors = 0;
    int exit_code          = 0;
    auto frame_failed = [&](std::string what) {
        std::fprintf(stderr, "[imgui-error] unexpected exception: %s\n", what.c_str());
        if (frame_open_) {
            ImGui::EndFrame();  // recovers an unbalanced ImGui stack so the next NewFrame is valid
            frame_open_ = false;
        }
        app_.on_unexpected_exception(what);
        if (++consecutive_errors >= kMaxConsecutiveFrameErrors) {
            std::fprintf(stderr, "[imgui-error] %d consecutive failing frames, shutting down\n", consecutive_errors);
            app_.notify(Notice::Level::Error, "Repeated unexpected errors, shutting down: " + what);
            running_  = false;
            exit_code = 2;
        }
    };
    while (running_) try {
        SDL_Event e;
        if (hidden_ || minimized_) {
            while (SDL_WaitEventTimeout(&e, 20)) handle_event(e);
        } else {
            while (SDL_PollEvent(&e)) handle_event(e);
        }
        const auto now = Clock::now();
        app_.tick(now);
        if (opts_.report_bundle) report_bundle_result();
        tray_->pump();
        process_dialog_results();
        while (auto a = tray_->poll_action()) app_.handle_tray_action(*a);
        process_intents();
        update_tray(now);

        if (!hidden_ && !minimized_) {
            render_frame();
            ++frames_;
            // Without vsync (headless, some drivers) keep the UI near 60 Hz.
            const auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - now).count();
            if (spent < 8) SDL_Delay(static_cast<Uint32>(8 - spent));
        }
        if (opts_.smoke_frames > 0 && frames_ >= opts_.smoke_frames) {
            // With --bundle the run does not end before the load result was drained
            // (bounded: a hung load must not hang the smoke run).
            const bool waiting_for_bundle =
                opts_.report_bundle && !bundle_reported_ && now - run_started < std::chrono::seconds(10);
            if (!waiting_for_bundle) {
                if (opts_.report_bundle && !bundle_reported_)
                    std::fprintf(stderr, "[bundle] error: no load result within 10 s\n");
                if (!opts_.screenshot_path.empty()) save_screenshot();
                running_ = false;
            }
        }
        last = now;
        consecutive_errors = 0;
    } catch (const std::exception& e) {
        frame_failed(e.what());
    } catch (...) {
        frame_failed("unknown exception type");
    }
    (void)last;
    app_.stop_engine();
    {
        std::string save_error;
        if (!app_.save_settings(&save_error)) std::fprintf(stderr, "settings not saved: %s\n", save_error.c_str());
    }
    shutdown();
    return exit_code;
}

// ---------------------------------------------------------------------------
// Events, intents, tray
// ---------------------------------------------------------------------------

void GuiApp::report_bundle_result() {
    const std::optional<BundleReport> r = app_.take_bundle_report();
    if (!r) return;
    if (r->ok)
        std::fprintf(stderr, "[bundle] loaded '%s' rate=%u hop=%zu voices=%zu\n", path_to_utf8(r->dir).c_str(),
                     static_cast<unsigned>(r->rate), r->hop, r->voices);
    else
        std::fprintf(stderr, "[bundle] error: %s\n", r->message.c_str());
    bundle_reported_ = true;
}

void GuiApp::handle_event(const SDL_Event& e) {
    ImGui_ImplSDL3_ProcessEvent(&e);
    switch (e.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            if (mini_) {
                leave_mini();
            } else if (app_.settings().close_to_tray && tray_->available()) {
                hide_main();
            } else {
                running_ = false;
            }
            break;
        case SDL_EVENT_WINDOW_MINIMIZED:
            minimized_ = true;
            break;
        case SDL_EVENT_WINDOW_RESTORED:
        case SDL_EVENT_WINDOW_SHOWN:
        case SDL_EVENT_WINDOW_MAXIMIZED:
            minimized_ = false;
            break;
        case SDL_EVENT_WINDOW_RESIZED:
            if (!mini_) {
                Settings& s = app_.settings();
                int w = 0, h = 0;
                SDL_GetWindowSize(window_, &w, &h);
                s.window_w = static_cast<int>(static_cast<float>(w) / (scale_ > 0.0f ? scale_ : 1.0f));
                s.window_h = static_cast<int>(static_cast<float>(h) / (scale_ > 0.0f ? scale_ : 1.0f));
            }
            break;
        case SDL_EVENT_WINDOW_MOVED:
            if (mini_) {
                int x = 0, y = 0;
                SDL_GetWindowPosition(window_, &x, &y);
                Settings& s = app_.settings();
                s.mini_has_position = true;
                s.mini_x            = x;
                s.mini_y            = y;
            }
            break;
        default:
            break;
    }
}

void GuiApp::process_intents() {
    if (app_.take_quit_request()) running_ = false;
    if (app_.take_show_panel_request()) {
        if (hidden_) show_main(); else hide_main();
    }
    if (app_.take_mini_toggle_request()) toggle_mini_panel();
}

void GuiApp::update_tray(Clock::time_point now) {
    if (!tray_->available()) return;
    if (now - last_tray_update_ < std::chrono::milliseconds(500)) return;
    last_tray_update_ = now;
    tray_->update(app_.tray_spec());
}

void GuiApp::show_main() {
    hidden_ = false;
    SDL_ShowWindow(window_);
    SDL_RestoreWindow(window_);
    SDL_RaiseWindow(window_);
    app_.set_window_visible(true);
}

void GuiApp::hide_main() {
    if (!tray_->available()) {
        SDL_MinimizeWindow(window_);
        return;
    }
    if (mini_) leave_mini();
    hidden_ = true;
    SDL_HideWindow(window_);
    app_.set_window_visible(false);
}

void GuiApp::request_hide_to_tray() { hide_main(); }

const std::string& GuiApp::tray_unavailable_reason() const {
    return tray_ ? tray_->unavailable_reason() : no_tray_reason_;
}

// ---------------------------------------------------------------------------
// Mini panel: the same window, morphed.
// ---------------------------------------------------------------------------

SDL_HitTestResult SDLCALL GuiApp::mini_hit_test(SDL_Window* /*win*/, const SDL_Point* area, void* data) {
    auto* self = static_cast<GuiApp*>(data);
    const int strip = static_cast<int>(28.0f * (self ? self->scale_ : 1.0f));
    // The header strip (status text) drags the borderless window; everything
    // below it holds controls and stays interactive.
    return area && area->y < strip && area->x < static_cast<int>(static_cast<float>(tokens::kMiniWidth) * 0.7f * (self ? self->scale_ : 1.0f))
               ? SDL_HITTEST_DRAGGABLE : SDL_HITTEST_NORMAL;
}

namespace {

/// True when (x, y) lies inside some display's bounds; a position saved on a
/// monitor that is gone is then left to the window manager.
bool point_on_a_display(int x, int y) {
    int count = 0;
    SDL_DisplayID* displays = SDL_GetDisplays(&count);
    if (displays == nullptr) return false;
    bool inside = false;
    for (int i = 0; i < count && !inside; ++i) {
        SDL_Rect r{};
        if (SDL_GetDisplayBounds(displays[i], &r) && x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.h) inside = true;
    }
    SDL_free(displays);
    return inside;
}

} // namespace

void GuiApp::enter_mini() {
    if (mini_) return;
    if (hidden_) show_main();
    SDL_GetWindowPosition(window_, &main_x_, &main_y_);
    SDL_GetWindowSize(window_, &main_w_, &main_h_);
    mini_ = true;
    app_.set_mini_panel(true);
    SDL_SetWindowResizable(window_, false);
    SDL_SetWindowBordered(window_, false);
    SDL_SetWindowAlwaysOnTop(window_, app_.settings().mini_panel_on_top);
    const int mw = static_cast<int>(tokens::kMiniWidth * scale_), mh = static_cast<int>(tokens::kMiniHeight * scale_);
    SDL_SetWindowMinimumSize(window_, mw, mh);
    SDL_SetWindowSize(window_, mw, mh);
    const Settings& s = app_.settings();
    if (s.mini_has_position && point_on_a_display(s.mini_x, s.mini_y)) SDL_SetWindowPosition(window_, s.mini_x, s.mini_y);
    SDL_SetWindowHitTest(window_, &GuiApp::mini_hit_test, this);
}

void GuiApp::leave_mini() {
    if (!mini_) return;
    mini_ = false;
    app_.set_mini_panel(false);
    SDL_SetWindowHitTest(window_, nullptr, nullptr);
    SDL_SetWindowAlwaysOnTop(window_, false);
    SDL_SetWindowBordered(window_, true);
    SDL_SetWindowResizable(window_, true);
    SDL_SetWindowMinimumSize(window_, static_cast<int>(720.0f * scale_), static_cast<int>(480.0f * scale_));
    if (main_w_ > 0 && main_h_ > 0) SDL_SetWindowSize(window_, main_w_, main_h_);
    SDL_SetWindowPosition(window_, main_x_, main_y_);
}

void GuiApp::toggle_mini_panel() {
    if (mini_) leave_mini(); else enter_mini();
}

// ---------------------------------------------------------------------------
// Dialogs, URLs, clipboard
// ---------------------------------------------------------------------------

void SDLCALL GuiApp::dialog_callback(void* userdata, const char* const* filelist, int /*filter*/) {
    auto* pd = static_cast<PendingDialog*>(userdata);
    if (!pd || !pd->self) return;
    DialogResult r;
    r.cb = pd->cb;
    if (filelist) {
        for (const char* const* p = filelist; *p; ++p) r.paths.push_back(path_from_utf8(*p));
    }
    // May run on another thread (SDL documents this): only queue here.
    std::lock_guard<std::mutex> lock(pd->self->dialog_mutex_);
    pd->self->dialog_results_.push_back(std::move(r));
}

void GuiApp::pick_files(const char* filter_name, const char* filter_pattern, bool allow_many, FilesCallback cb) {
    auto pd            = std::make_unique<PendingDialog>();
    pd->self           = this;
    pd->cb             = std::move(cb);
    pd->filter_name    = filter_name ? filter_name : "All files";
    pd->filter_pattern = filter_pattern ? filter_pattern : "*";
    pd->filter.name    = pd->filter_name.c_str();
    pd->filter.pattern = pd->filter_pattern.c_str();
    SDL_ShowOpenFileDialog(&GuiApp::dialog_callback, pd.get(), window_, &pd->filter, 1, nullptr, allow_many);
    pending_dialogs_.push_back(std::move(pd));
}

void GuiApp::pick_folder(FilesCallback cb) {
    auto pd  = std::make_unique<PendingDialog>();
    pd->self = this;
    pd->cb   = std::move(cb);
    SDL_ShowOpenFolderDialog(&GuiApp::dialog_callback, pd.get(), window_, nullptr, false);
    pending_dialogs_.push_back(std::move(pd));
}

void GuiApp::process_dialog_results() {
    std::deque<DialogResult> ready;
    {
        std::lock_guard<std::mutex> lock(dialog_mutex_);
        ready.swap(dialog_results_);
    }
    for (DialogResult& r : ready)
        if (r.cb) r.cb(r.paths);
    // Pending descriptors are kept until shutdown: SDL gives no completion
    // signal we can match to a descriptor, and they are tiny.
}

void GuiApp::open_url(const std::string& url) {
    if (!is_web_url(url)) {
        app_.notify(Notice::Level::Warning, "Refused to open a link that is not an http(s) address.");
        return;
    }
    if (!SDL_OpenURL(url.c_str())) app_.notify(Notice::Level::Warning, std::string("Could not open: ") + SDL_GetError());
}

void GuiApp::open_path(const std::filesystem::path& p) {
    std::string s = path_to_utf8_generic(std::filesystem::absolute(p));
    if (!s.empty() && s.front() != '/') s = "/" + s;  // Windows drive letters
    const std::string target = "file://" + s;
    if (!SDL_OpenURL(target.c_str())) app_.notify(Notice::Level::Warning, std::string("Could not open: ") + SDL_GetError());
}

void GuiApp::set_clipboard(const std::string& text) {
    if (!SDL_SetClipboardText(text.c_str())) app_.notify(Notice::Level::Warning, "Clipboard unavailable.");
    else app_.notify(Notice::Level::Info, "Copied to the clipboard.");
}

std::string GuiApp::about_text() const {
    const int v = SDL_GetVersion();
    char buf[256];
    std::snprintf(buf, sizeof buf, "FluxVoice %s - SDL %d.%d.%d (%s) - Dear ImGui %s - video driver %s",
                  FLUXVOICE_VERSION, SDL_VERSIONNUM_MAJOR(v), SDL_VERSIONNUM_MINOR(v), SDL_VERSIONNUM_MICRO(v),
                  SDL_GetRevision(), IMGUI_VERSION, SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "-");
    return buf;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void GuiApp::render_frame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    frame_open_ = true;

    handle_shortcuts();
    ViewContext ctx{app_, *this, scale_};
    if (mini_) {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f * scale_, 6.0f * scale_));
        if (ImGui::Begin("##mini", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus)) {
            draw_mini_panel(ctx);
        }
        ImGui::End();
        ImGui::PopStyleVar();
    } else {
        draw_main_window();
        draw_toast();
    }

    ImGui::Render();
    frame_open_ = false;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    g_gl.Viewport(0, 0, w, h);
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    g_gl.ClearColor(bg.x, bg.y, bg.z, 1.0f);
    g_gl.Clear(kGlColorBufferBit);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window_);
}

void GuiApp::handle_shortcuts() {
    const ImGuiInputFlags route = ImGuiInputFlags_RouteGlobal;
    constexpr ImGuiKey keys[] = {ImGuiKey_1, ImGuiKey_2, ImGuiKey_3, ImGuiKey_4, ImGuiKey_5, ImGuiKey_6, ImGuiKey_7, ImGuiKey_8};
    for (int i = 0; i < static_cast<int>(Section::Count) && i < 8; ++i)
        if (ImGui::Shortcut(ImGuiMod_Ctrl | keys[i], route)) app_.navigate(static_cast<Section>(i));
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_M, route)) toggle_mini_panel();
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_H, route)) hide_main();
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_Q, route)) running_ = false;
    if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::Shortcut(ImGuiKey_M, route)) app_.toggle_mute();
    }
}

void GuiApp::draw_main_window() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_MenuBar |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                   ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::Begin("##main", nullptr, flags)) {
        draw_menu_bar();
        const float status_h = tokens::kStatusHeight * scale_;
        const float rail_w   = tokens::kRailWidth * scale_;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, theme_colors().rail_bg);
        if (ImGui::BeginChild("rail", ImVec2(rail_w, -status_h), ImGuiChildFlags_AlwaysUseWindowPadding)) draw_rail();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18.0f * scale_, 14.0f * scale_));
        if (ImGui::BeginChild("content", ImVec2(0.0f, -status_h), ImGuiChildFlags_AlwaysUseWindowPadding)) {
            ViewContext ctx{app_, *this, scale_};
            draw_section(ctx, app_.section());
            // Space toggles the engine only while the Live panel has keyboard
            // focus (frontend/README.md §4.10): never in the Mini panel, never
            // while a text field is active or a widget is being operated.
            if (app_.section() == Section::Live && !mini_ && !ImGui::GetIO().WantTextInput && !ImGui::IsAnyItemActive() &&
                ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Space, false))
                app_.toggle_engine();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        draw_status_bar();
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void GuiApp::draw_menu_bar() {
    if (!ImGui::BeginMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Mini panel", "Ctrl+M")) toggle_mini_panel();
        if (ImGui::MenuItem("Hide to tray", "Ctrl+H", false, tray_->available())) hide_main();
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", "Ctrl+Q")) running_ = false;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        for (int i = 0; i < static_cast<int>(Section::Count); ++i) {
            const auto s = static_cast<Section>(i);
            char shortcut[8];
            std::snprintf(shortcut, sizeof shortcut, "Ctrl+%d", i + 1);
            if (ImGui::MenuItem(section_label(s), shortcut, app_.section() == s)) app_.navigate(s);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Engine")) {
        if (ImGui::MenuItem(app_.engine_on() ? "Stop" : "Start", "Space", false, app_.backend_usable())) app_.toggle_engine();
        if (ImGui::MenuItem("Mute", "M", app_.muted(), app_.engine_on())) app_.toggle_mute();
        ImGui::Separator();
        if (ImGui::MenuItem("Voice changer (Mode A)", nullptr, !app_.translate())) static_cast<void>(app_.set_translate(false));
        if (ImGui::MenuItem("Translate (Mode B)", nullptr, app_.translate(), app_.translation().translator_available()))
            static_cast<void>(app_.set_translate(true));
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        ImGui::TextUnformatted(about_text().c_str());
        ImGui::Separator();
        ImGui::TextUnformatted("Ctrl+1..8 sections, Space start/stop (Live), M mute, Ctrl+M mini panel, Ctrl+H hide.");
        ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
}

void GuiApp::draw_rail() {
    ImGui::PushFont(nullptr, tokens::kTitleFont);
    ImGui::TextColored(theme_colors().accent, "FluxVoice");
    ImGui::PopFont();
    ImGui::PushFont(nullptr, tokens::kBaseFont * 0.8f);
    widgets::dim_text("voice changer & translator");
    ImGui::PopFont();
    ImGui::Spacing();
    ImGui::Spacing();
    for (int i = 0; i < static_cast<int>(Section::Count); ++i) {
        const auto s = static_cast<Section>(i);
        const bool selected = app_.section() == s;
        if (selected) ImGui::PushStyleColor(ImGuiCol_Header, theme_colors().accent);
        if (selected) ImGui::PushStyleColor(ImGuiCol_Text, theme_colors().accent_text);
        if (ImGui::Selectable(section_label(s), selected, 0, ImVec2(0.0f, 34.0f * scale_))) app_.navigate(s);
        if (selected) ImGui::PopStyleColor(2);
    }
    // Bottom actions.
    const float bottom = ImGui::GetWindowHeight() - 3.0f * ImGui::GetFrameHeightWithSpacing();
    if (ImGui::GetCursorPosY() < bottom) ImGui::SetCursorPosY(bottom);
    if (ImGui::Button("Mini panel", ImVec2(-1.0f, 0.0f))) toggle_mini_panel();
    ImGui::BeginDisabled(!tray_->available());
    if (ImGui::Button("Hide to tray", ImVec2(-1.0f, 0.0f))) hide_main();
    ImGui::EndDisabled();
}

void GuiApp::draw_status_bar() {
    const StatusSummary s = app_.status_summary();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme_colors().status_bg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f * scale_, 5.0f * scale_));
    if (ImGui::BeginChild("status", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AlwaysUseWindowPadding)) {
        const TrayState ts = tray_state_for([&] {
            TrayInputs in;
            in.stream = s.state;
            in.muted  = app_.muted();
            in.device_unresolved = app_.device_unresolved();
            return in;
        }());
        widgets::status_dot(state_color(ts), 5.0f * scale_);
        ImGui::SameLine();
        ImGui::TextUnformatted(s.pill.c_str());
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Worker priority: %s (%s)", app_.engine().priority_method(),
                              app_.engine().priority_elevated() ? "elevated" : "normal");
        ImGui::SameLine();
        widgets::dim_text("|");
        ImGui::SameLine();
        ImGui::TextUnformatted(s.mode.c_str());  // "Bypass - no model loaded" / "Mode A - <bundle>"
        ImGui::SameLine();
        widgets::dim_text("|");
        ImGui::SameLine();
        widgets::text_ellipsis(s.detail, ImGui::GetContentRegionAvail().x);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void GuiApp::draw_toast() {
    const auto& notices = app_.notices();
    if (notices.empty()) return;
    const Notice& n = notices.back();
    if (app_.uptime_seconds() - n.at_seconds > 6.0) return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 16.0f * scale_,
                                   vp->WorkPos.y + vp->WorkSize.y - (tokens::kStatusHeight + 16.0f) * scale_),
                            ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(420.0f * scale_, 200.0f * scale_));
    const ImVec4 c = n.level == Notice::Level::Error ? theme_colors().danger
                     : n.level == Notice::Level::Warning ? theme_colors().warning : theme_colors().accent;
    ImGui::PushStyleColor(ImGuiCol_Border, c);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, theme_colors().card_bg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f * scale_);
    if (ImGui::Begin("##toast", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs)) {
        ImGui::PushTextWrapPos(400.0f * scale_);
        ImGui::TextColored(c, "%s", n.level == Notice::Level::Error ? "Error" : n.level == Notice::Level::Warning ? "Warning" : "Info");
        ImGui::TextUnformatted(n.text.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void GuiApp::save_screenshot() {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    if (w <= 0 || h <= 0) return;
    std::vector<unsigned char> px(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u);
    g_gl.Finish();
    g_gl.ReadPixels(0, 0, w, h, kGlRgba, kGlUnsignedByte, px.data());
    std::ofstream f(path_from_utf8(opts_.screenshot_path), std::ios::binary | std::ios::trunc);  // argv is UTF-8
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", opts_.screenshot_path.c_str());
        return;
    }
    f << "P6\n" << w << ' ' << h << "\n255\n";
    for (int y = h - 1; y >= 0; --y) {  // OpenGL rows are bottom-up
        const unsigned char* row = px.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(w) * 4u;
        for (int x = 0; x < w; ++x) f.write(reinterpret_cast<const char*>(row + static_cast<std::size_t>(x) * 4u), 3);
    }
    std::fprintf(stdout, "screenshot: %s (%dx%d)\n", opts_.screenshot_path.c_str(), w, h);
}

} // namespace fluxvoice::frontend
