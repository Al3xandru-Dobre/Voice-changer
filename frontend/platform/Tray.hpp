// Tray — the platform seam for the system tray. The model produces a
// declarative TraySpec (core/TrayModel.hpp); a TrayHost renders it and hands
// clicks back as TrayActions. NullTray is the fallback when the platform has
// no tray (and the test double): the application then treats "close" as quit
// and explains why in Settings (frontend/README.md §6).
#pragma once

#include "core/TrayModel.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace fluxvoice::frontend {

class TrayHost {
public:
    virtual ~TrayHost() = default;

    virtual bool available() const = 0;
    virtual const std::string& unavailable_reason() const = 0;

    /// Apply icon state, tooltip and menu. Implementations rebuild the native
    /// menu only when `spec.menu_hash()` changed.
    virtual void update(const TraySpec& spec) = 0;

    /// Platform housekeeping once per frame (SDL_UpdateTrays on SDL).
    virtual void pump() = 0;

    /// Next queued click, if any. Drained on the UI thread.
    virtual std::optional<TrayAction> poll_action() = 0;
};

class NullTray final : public TrayHost {
public:
    explicit NullTray(std::string reason) : reason_(std::move(reason)) {}
    bool available() const override { return false; }
    const std::string& unavailable_reason() const override { return reason_; }
    void update(const TraySpec&) override {}
    void pump() override {}
    std::optional<TrayAction> poll_action() override { return std::nullopt; }

private:
    std::string reason_;
};

} // namespace fluxvoice::frontend
