// SdlTray — SDL3 system tray implementation.
#include "platform/SdlTray.hpp"

#include "core/TrayModel.hpp"

#include <SDL3/SDL.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fluxvoice::frontend {

SdlTray::SdlTray() {
    pixels_ = render_tray_icon(TrayState::Idle);
    icon_   = SDL_CreateSurfaceFrom(TrayIconPixels::kSize, TrayIconPixels::kSize, SDL_PIXELFORMAT_RGBA32,
                                    pixels_.rgba.data(), TrayIconPixels::kSize * 4);
    tray_   = SDL_CreateTray(icon_, "FluxVoice");
    if (!tray_) {
        const char* err = SDL_GetError();
        reason_ = err && *err ? err : "the platform did not provide a system tray";
#if defined(__linux__)
        reason_ += " (Linux: install libayatana-appindicator3-1; GNOME also needs the AppIndicator extension)";
#endif
        if (icon_) {
            SDL_DestroySurface(icon_);
            icon_ = nullptr;
        }
        return;
    }
    SDL_CreateTrayMenu(tray_);
    has_state_  = true;
    last_state_ = TrayState::Idle;
}

SdlTray::~SdlTray() {
    if (tray_) SDL_DestroyTray(tray_);
    if (icon_) SDL_DestroySurface(icon_);
}

void SdlTray::set_icon(TrayState state) {
    if (has_state_ && state == last_state_) return;
    pixels_ = render_tray_icon(state);
    SDL_Surface* fresh = SDL_CreateSurfaceFrom(TrayIconPixels::kSize, TrayIconPixels::kSize, SDL_PIXELFORMAT_RGBA32,
                                               pixels_.rgba.data(), TrayIconPixels::kSize * 4);
    if (!fresh) return;
    SDL_SetTrayIcon(tray_, fresh);
    if (icon_) SDL_DestroySurface(icon_);
    icon_       = fresh;
    last_state_ = state;
    has_state_  = true;
}

void SdlTray::update(const TraySpec& spec) {
    if (!tray_) return;
    set_icon(spec.state);
    if (spec.tooltip != last_tooltip_) {
        SDL_SetTrayTooltip(tray_, spec.tooltip.c_str());
        last_tooltip_ = spec.tooltip;
    }
    const std::uint64_t h = spec.menu_hash();
    if (h != last_menu_hash_) {
        rebuild_menu(spec);
        last_menu_hash_ = h;
    }
}

void SdlTray::pump() {
    if (tray_) SDL_UpdateTrays();
}

std::optional<TrayAction> SdlTray::poll_action() {
    if (actions_.empty()) return std::nullopt;
    TrayAction a = actions_.front();
    actions_.pop_front();
    return a;
}

void SDLCALL SdlTray::on_entry(void* userdata, SDL_TrayEntry* /*entry*/) {
    auto* ctx = static_cast<EntryContext*>(userdata);
    if (!ctx || !ctx->self) return;
    if (auto a = action_for_item(ctx->id)) ctx->self->actions_.push_back(*a);
}

void SdlTray::add_items(SDL_TrayMenu* menu, const std::vector<TrayItem>& items) {
    for (const TrayItem& it : items) {
        if (it.kind == TrayItem::Kind::Separator) {
            SDL_InsertTrayEntryAt(menu, -1, nullptr, 0);
            continue;
        }
        SDL_TrayEntryFlags flags = 0;
        switch (it.kind) {
            case TrayItem::Kind::Button:    flags = SDL_TRAYENTRY_BUTTON; break;
            case TrayItem::Kind::Check:
            case TrayItem::Kind::Radio:     flags = SDL_TRAYENTRY_CHECKBOX | (it.checked ? SDL_TRAYENTRY_CHECKED : 0u); break;
            case TrayItem::Kind::Submenu:   flags = SDL_TRAYENTRY_SUBMENU; break;
            case TrayItem::Kind::Separator: break;
        }
        if (!it.enabled) flags |= SDL_TRAYENTRY_DISABLED;
        SDL_TrayEntry* entry = SDL_InsertTrayEntryAt(menu, -1, it.label.c_str(), flags);
        if (!entry) continue;
        if (it.kind == TrayItem::Kind::Submenu) {
            if (SDL_TrayMenu* sub = SDL_CreateTraySubmenu(entry)) add_items(sub, it.children);
            continue;
        }
        auto ctx  = std::make_unique<EntryContext>();
        ctx->self = this;
        ctx->id   = it.id;
        SDL_SetTrayEntryCallback(entry, &SdlTray::on_entry, ctx.get());
        contexts_.push_back(std::move(ctx));
    }
}

void SdlTray::rebuild_menu(const TraySpec& spec) {
    SDL_TrayMenu* menu = SDL_GetTrayMenu(tray_);
    if (!menu) menu = SDL_CreateTrayMenu(tray_);
    if (!menu) return;
    int count = 0;
    const SDL_TrayEntry** entries = SDL_GetTrayEntries(menu, &count);
    // Remove from the back; SDL_GetTrayEntries' array is invalidated by removal.
    while (entries && count > 0) {
        SDL_RemoveTrayEntry(const_cast<SDL_TrayEntry*>(entries[count - 1]));
        entries = SDL_GetTrayEntries(menu, &count);
    }
    contexts_.clear();  // callbacks of removed entries can no longer fire
    add_items(menu, spec.items);
}

} // namespace fluxvoice::frontend
