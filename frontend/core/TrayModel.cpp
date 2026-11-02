// TrayModel — menu spec, tooltip, actions and the procedural icon.
#include "core/TrayModel.hpp"

#include "audio/AudioTypes.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice::frontend {

const char* to_string(TrayState s) {
    switch (s) {
        case TrayState::Idle:       return "idle";
        case TrayState::Running:    return "running";
        case TrayState::Muted:      return "muted";
        case TrayState::Recovering: return "recovering";
        case TrayState::Error:      return "error";
    }
    return "?";
}

namespace {

void hash_str(std::uint64_t& h, std::string_view s) {
    for (char ch : s) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
    h ^= 0xFFu; h *= 1099511628211ULL;  // separator
}

void hash_item(std::uint64_t& h, const TrayItem& it) {
    hash_str(h, it.id);
    hash_str(h, it.label);
    h ^= static_cast<std::uint64_t>(it.kind) | (it.checked ? 0x10u : 0u) | (it.enabled ? 0x20u : 0u);
    h *= 1099511628211ULL;
    for (const TrayItem& c : it.children) hash_item(h, c);
}

TrayItem button(std::string id, std::string label, bool enabled = true) {
    TrayItem t;
    t.id = std::move(id); t.label = std::move(label); t.kind = TrayItem::Kind::Button; t.enabled = enabled;
    return t;
}

TrayItem check(std::string id, std::string label, bool checked, bool enabled = true) {
    TrayItem t = button(std::move(id), std::move(label), enabled);
    t.kind = TrayItem::Kind::Check;
    t.checked = checked;
    return t;
}

TrayItem radio(std::string id, std::string label, bool checked, bool enabled = true) {
    TrayItem t = check(std::move(id), std::move(label), checked, enabled);
    t.kind = TrayItem::Kind::Radio;
    return t;
}

TrayItem separator() {
    TrayItem t;
    t.kind = TrayItem::Kind::Separator;
    return t;
}

TrayItem submenu(std::string id, std::string label, std::vector<TrayItem> children, bool enabled = true) {
    TrayItem t = button(std::move(id), std::move(label), enabled);
    t.kind = TrayItem::Kind::Submenu;
    t.children = std::move(children);
    return t;
}

std::string fmt_ms(double ms) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f ms", ms);
    return buf;
}

} // namespace

std::uint64_t TraySpec::hash() const {
    std::uint64_t h = 1469598103934665603ULL;
    h ^= static_cast<std::uint64_t>(state);
    hash_str(h, tooltip);
    for (const TrayItem& it : items) hash_item(h, it);
    return h;
}

std::uint64_t TraySpec::menu_hash() const {
    std::uint64_t h = 1469598103934665603ULL;
    for (const TrayItem& it : items) hash_item(h, it);
    return h;
}

TrayState tray_state_for(const TrayInputs& in) {
    switch (in.stream) {
        case StreamState::Error:
        case StreamState::AwaitingUser:
            return TrayState::Error;
        case StreamState::Opening:
        case StreamState::Recovering:
        case StreamState::Stopping:
            return TrayState::Recovering;
        case StreamState::Running:
            return in.muted ? TrayState::Muted : TrayState::Running;
        case StreamState::Idle:
            break;
    }
    return TrayState::Idle;
}

TraySpec build_tray_spec(const TrayInputs& in) {
    TraySpec s;
    s.state = tray_state_for(in);

    // Tooltip: "FluxVoice · <state> · <preset> · <queued> ms"
    s.tooltip = "FluxVoice";
    switch (s.state) {
        case TrayState::Idle:       s.tooltip += " - Idle"; break;
        case TrayState::Running:
            s.tooltip += " - Running";
            if (!in.preset.empty()) s.tooltip += " - " + in.preset;
            s.tooltip += " - " + fmt_ms(in.queued_ms);
            break;
        case TrayState::Muted:      s.tooltip += " - Muted"; break;
        case TrayState::Recovering:
            s.tooltip += " - Recovering";
            if (in.retry_attempt > 0) s.tooltip += " (attempt " + std::to_string(in.retry_attempt) + ")";
            break;
        case TrayState::Error:
            s.tooltip += " - " + (in.error_text.empty() ? std::string("stopped, needs attention") : in.error_text);
            break;
    }

    s.items.push_back(button("show", in.window_visible ? "Hide panel" : "Show panel"));
    s.items.push_back(button("mini", "Mini panel"));
    s.items.push_back(separator());
    s.items.push_back(check("engine", "Engine on", in.engine_on));
    s.items.push_back(check("mute", "Mute", in.muted, in.engine_on));
    s.items.push_back(separator());

    s.items.push_back(submenu("mode", "Mode", {
        radio("mode:vc", "Voice changer", !in.translate),
        radio("mode:s2ut", "Translate", in.translate, in.translator_available),
    }));

    std::vector<TrayItem> presets;
    for (const std::string& p : in.presets) presets.push_back(radio("preset:" + p, p, p == in.preset));
    s.items.push_back(submenu("presets", "Preset", std::move(presets), !in.presets.empty()));

    std::vector<TrayItem> voices;
    for (const std::string& v : in.voices) voices.push_back(radio("voice:" + v, v, v == in.voice));
    s.items.push_back(submenu("voices", "Voice", std::move(voices), !in.voices.empty()));

    std::vector<TrayItem> pairs;
    for (const std::string& p : in.language_pairs) pairs.push_back(radio("lang:" + p, p, p == in.language_pair));
    s.items.push_back(submenu("langs", "Languages", std::move(pairs),
                              in.translator_available && !in.language_pairs.empty()));

    s.items.push_back(separator());
    s.items.push_back(button("quit", "Quit"));
    return s;
}

std::optional<TrayAction> action_for_item(std::string_view id) {
    using T = TrayAction::Type;
    auto make = [](T t, std::string_view arg = {}) {
        TrayAction a;
        a.type = t;
        a.arg  = std::string(arg);
        return a;
    };
    if (id == "show")      return make(T::ShowPanel);
    if (id == "mini")      return make(T::ToggleMiniPanel);
    if (id == "engine")    return make(T::ToggleEngine);
    if (id == "mute")      return make(T::ToggleMute);
    if (id == "mode:vc")   return make(T::SetModeVoiceChanger);
    if (id == "mode:s2ut") return make(T::SetModeTranslate);
    if (id == "quit")      return make(T::Quit);
    constexpr std::string_view kPreset = "preset:", kVoice = "voice:", kLang = "lang:";
    if (id.starts_with(kPreset) && id.size() > kPreset.size()) return make(T::LoadPreset, id.substr(kPreset.size()));
    if (id.starts_with(kVoice) && id.size() > kVoice.size())   return make(T::SetVoice, id.substr(kVoice.size()));
    if (id.starts_with(kLang) && id.size() > kLang.size())     return make(T::SetLanguagePair, id.substr(kLang.size()));
    return std::nullopt;
}

namespace {

constexpr std::uint32_t pack(std::uint32_t r, std::uint32_t g, std::uint32_t b, std::uint32_t a) {
    return r | (g << 8) | (b << 16) | (a << 24);
}

std::uint32_t tile_colour(TrayState s) {
    switch (s) {
        case TrayState::Idle:       return pack(0x6B, 0x72, 0x80, 0xFF);  // grey
        case TrayState::Running:    return pack(0x2E, 0xA8, 0x6B, 0xFF);  // green
        case TrayState::Muted:      return pack(0x2E, 0xA8, 0x6B, 0xFF);  // green + slash
        case TrayState::Recovering: return pack(0xD9, 0x8E, 0x1F, 0xFF);  // amber
        case TrayState::Error:      return pack(0xC9, 0x3C, 0x3C, 0xFF);  // red
    }
    return pack(0x6B, 0x72, 0x80, 0xFF);
}

} // namespace

TrayIconPixels render_tray_icon(TrayState state) {
    TrayIconPixels px;
    constexpr int N = TrayIconPixels::kSize;
    const std::uint32_t tile  = tile_colour(state);
    const std::uint32_t white = pack(0xFF, 0xFF, 0xFF, 0xFF);
    const std::uint32_t clear = pack(0, 0, 0, 0);

    // Rounded tile: inset 2 px, corner radius 7 px.
    constexpr int inset = 2, radius = 7;
    auto inside_tile = [&](int x, int y) {
        if (x < inset || y < inset || x >= N - inset || y >= N - inset) return false;
        const int cx = x < inset + radius ? inset + radius : (x >= N - inset - radius ? N - inset - radius - 1 : x);
        const int cy = y < inset + radius ? inset + radius : (y >= N - inset - radius ? N - inset - radius - 1 : y);
        const int dx = x - cx, dy = y - cy;
        return dx * dx + dy * dy <= radius * radius;
    };

    // Five waveform bars, heights symmetric around the centre line.
    constexpr std::array<int, 5> bar_half = {4, 8, 11, 7, 3};
    constexpr int bar_w = 3, gap = 2;
    constexpr int total_w = 5 * bar_w + 4 * gap;  // 23
    constexpr int x0 = (N - total_w) / 2;
    auto inside_bar = [&](int x, int y) {
        for (int i = 0; i < 5; ++i) {
            const int bx = x0 + i * (bar_w + gap);
            if (x >= bx && x < bx + bar_w && y >= N / 2 - bar_half[static_cast<std::size_t>(i)] &&
                y < N / 2 + bar_half[static_cast<std::size_t>(i)])
                return true;
        }
        return false;
    };

    // Muted: a 2 px diagonal slash from bottom-left to top-right.
    auto inside_slash = [&](int x, int y) {
        if (state != TrayState::Muted) return false;
        const int d = (x + y) - (N - 1);
        return d >= -1 && d <= 1 && x >= 6 && x < N - 6;
    };

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            std::uint32_t c = clear;
            if (inside_tile(x, y)) {
                c = tile;
                if (inside_bar(x, y)) c = white;
                if (inside_slash(x, y)) c = pack(0x1A, 0x1A, 0x1A, 0xFF);
            }
            px.rgba[static_cast<std::size_t>(y * N + x)] = c;
        }
    }
    return px;
}

} // namespace fluxvoice::frontend
