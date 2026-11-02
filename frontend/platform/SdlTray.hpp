// SdlTray — TrayHost over SDL3's tray API (SDL_tray.h): Shell_NotifyIcon on
// Windows, AppIndicator/GTK on Linux, a status item on macOS. Clicks arrive on
// the main thread inside SDL's event pump; they are queued and drained by the
// application loop, so model mutations never happen inside an SDL callback.
#pragma once

#include "core/TrayModel.hpp"
#include "platform/Tray.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

class SdlTray final : public TrayHost {
public:
    /// Requires SDL_Init(SDL_INIT_VIDEO). `available()` is false when the
    /// platform refused to create a tray (reason from SDL_GetError()).
    SdlTray();
    ~SdlTray() override;

    SdlTray(const SdlTray&)            = delete;
    SdlTray& operator=(const SdlTray&) = delete;

    bool available() const override { return tray_ != nullptr; }
    const std::string& unavailable_reason() const override { return reason_; }
    void update(const TraySpec& spec) override;
    void pump() override;
    std::optional<TrayAction> poll_action() override;

private:
    struct EntryContext {
        SdlTray*    self = nullptr;
        std::string id;
    };

    static void SDLCALL on_entry(void* userdata, SDL_TrayEntry* entry);
    void set_icon(TrayState state);
    void rebuild_menu(const TraySpec& spec);
    void add_items(SDL_TrayMenu* menu, const std::vector<TrayItem>& items);

    SDL_Tray*     tray_ = nullptr;
    SDL_Surface*  icon_ = nullptr;
    TrayIconPixels pixels_{};  // must outlive icon_: SDL_CreateSurfaceFrom does not copy
    std::string   reason_;
    std::uint64_t last_menu_hash_ = 0;
    std::string   last_tooltip_;
    bool          has_state_ = false;
    TrayState     last_state_ = TrayState::Idle;
    std::vector<std::unique_ptr<EntryContext>> contexts_;
    std::deque<TrayAction> actions_;
};

} // namespace fluxvoice::frontend
