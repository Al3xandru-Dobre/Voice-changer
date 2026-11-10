// Widgets — implementation.
#include "imgui/Widgets.hpp"

#include "imgui/Theme.hpp"
#include "voice/VoiceParams.hpp"

#include <imgui.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace fluxvoice::frontend::widgets {

bool begin_card(const char* id, const char* title, float height) {
    const ThemeColors& t = theme_colors();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.card_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, t.card_border);
    const ImGuiChildFlags flags = ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding |
                                  (height <= 0.0f ? ImGuiChildFlags_AutoResizeY : 0);
    // BeginChild() returning false only means the child is clipped this frame;
    // ImGui still requires EndChild(), and submitting content is harmless, so
    // callers always draw and always call end_card().
    ImGui::BeginChild(id, ImVec2(0.0f, height), flags);
    ImGui::PopStyleColor(2);
    if (title) {
        ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.1f);
        ImGui::TextUnformatted(title);
        ImGui::PopFont();
        ImGui::Spacing();
    }
    return true;
}

void end_card() { ImGui::EndChild(); }

void heading(const char* text) {
    ImGui::PushFont(nullptr, tokens::kTitleFont);
    ImGui::TextUnformatted(text);
    ImGui::PopFont();
    ImGui::Spacing();
}

void dim_text(const char* text) { ImGui::TextColored(theme_colors().text_dim, "%s", text); }

void dim_text_wrapped(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, theme_colors().text_dim);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

void status_dot(const ImVec4& color, float radius_px) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight();
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + radius_px, p.y + h * 0.5f), radius_px,
                                                ImGui::GetColorU32(color));
    ImGui::Dummy(ImVec2(radius_px * 2.0f, h));
}

void pill(const char* text, const ImVec4& color) {
    const ImVec2 pad(8.0f, 2.0f);
    const ImVec2 size = ImGui::CalcTextSize(text);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const ImVec2 box(size.x + pad.x * 2.0f, size.y + pad.y * 2.0f);
    ImVec4 bg = color;
    bg.w = 0.22f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + box.x, p.y + box.y), ImGui::GetColorU32(bg), box.y * 0.5f);
    dl->AddText(ImVec2(p.x + pad.x, p.y + pad.y), ImGui::GetColorU32(color), text);
    ImGui::Dummy(box);
}

void level_meter(const char* id, float dbfs, float width, float height) {
    const ThemeColors& t = theme_colors();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + width, p.y + height), ImGui::GetColorU32(t.meter_bg), 3.0f);
    float frac = (dbfs + 60.0f) / 60.0f;
    frac = frac < 0.0f ? 0.0f : (frac > 1.0f ? 1.0f : frac);
    const float fill_w = width * frac;
    if (fill_w > 0.0f) {
        dl->AddRectFilled(p, ImVec2(p.x + fill_w, p.y + height),
                          ImGui::GetColorU32(dbfs > -3.0f ? t.meter_hot : t.meter_fill), 3.0f);
    }
    // -12 dBFS tick
    const float tick_x = p.x + width * (48.0f / 60.0f);
    dl->AddLine(ImVec2(tick_x, p.y), ImVec2(tick_x, p.y + height), ImGui::GetColorU32(t.card_border));
    ImGui::InvisibleButton(id, ImVec2(width, height));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%.1f dBFS", static_cast<double>(dbfs));
}

bool toggle_button(const char* on_label, const char* off_label, bool on, const ImVec2& size,
                   const ImVec4& on_color) {
    const ThemeColors& t = theme_colors();
    if (on) {
        ImVec4 hover = on_color;
        hover.x *= 1.1f; hover.y *= 1.1f; hover.z *= 1.1f;
        ImGui::PushStyleColor(ImGuiCol_Button, on_color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, on_color);
        ImGui::PushStyleColor(ImGuiCol_Text, t.accent_text);
    }
    const bool clicked = ImGui::Button(on ? on_label : off_label, size);
    if (on) ImGui::PopStyleColor(4);
    return clicked;
}

bool combo_strings(const char* label, int& index, const std::vector<std::string>& items, const char* empty_hint) {
    const char* preview = (index >= 0 && static_cast<std::size_t>(index) < items.size())
                              ? items[static_cast<std::size_t>(index)].c_str() : empty_hint;
    bool changed = false;
    if (ImGui::BeginCombo(label, preview, ImGuiComboFlags_HeightLarge)) {
        for (std::size_t i = 0; i < items.size(); ++i) {
            const bool selected = static_cast<int>(i) == index;
            if (ImGui::Selectable(items[i].c_str(), selected)) {
                index   = static_cast<int>(i);
                changed = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool param_control(const ParamDesc& d, float& value) {
    bool changed = false;
    ImGui::PushID(d.name.data(), d.name.data() + d.name.size());
    const std::string label(d.label);
    switch (d.kind) {
        case ParamKind::Toggle: {
            bool on = value >= 0.5f;
            if (ImGui::Checkbox(label.c_str(), &on)) {
                value   = on ? 1.0f : 0.0f;
                changed = true;
            }
            break;
        }
        case ParamKind::Integer: {
            int v = static_cast<int>(std::lround(value));
            if (ImGui::SliderInt(label.c_str(), &v, static_cast<int>(d.min), static_cast<int>(d.max),
                                 d.unit.empty() ? "%d" : ("%d " + std::string(d.unit)).c_str(),
                                 ImGuiSliderFlags_AlwaysClamp)) {
                value   = static_cast<float>(v);
                changed = true;
            }
            break;
        }
        case ParamKind::Continuous: {
            const std::string fmt = d.unit.empty() ? "%.2f" : ("%.2f " + std::string(d.unit));
            if (ImGui::SliderFloat(label.c_str(), &value, d.min, d.max, fmt.c_str(), ImGuiSliderFlags_AlwaysClamp))
                changed = true;
            break;
        }
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && value != d.def) {
        value   = d.def;
        changed = true;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
        ImGui::TextUnformatted(d.help.data(), d.help.data() + d.help.size());
        ImGui::Separator();
        ImGui::Text("Range %.2g to %.2g %s, default %.2g. Double-click to reset.", static_cast<double>(d.min),
                    static_cast<double>(d.max), std::string(d.unit).c_str(), static_cast<double>(d.def));
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
    ImGui::PopID();
    return changed;
}

void help_marker(const char* text) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

bool kv_table_begin(const char* id) {
    // BeginTable() returns false when the table is clipped; EndTable() must
    // then NOT be called, so callers wrap rows + kv_table_end() in the result.
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) return false;
    ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 15.0f);
    ImGui::TableSetupColumn("v", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

void kv_row(const char* key, const std::string& value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    dim_text(key);
    ImGui::TableNextColumn();
    ImGui::TextWrapped("%s", value.c_str());
}

void kv_table_end() { ImGui::EndTable(); }

void notice(const char* text, const ImVec4& color) {
    ImVec4 bg = color;
    bg.w = 0.12f;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
    ImGui::PushStyleColor(ImGuiCol_Border, color);
    if (ImGui::BeginChild(text, ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        ImGui::TextWrapped("%s", text);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

void text_ellipsis(const std::string& text, float max_width) {
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    if (size.x <= max_width) {
        ImGui::TextUnformatted(text.c_str());
        return;
    }
    std::string shown = text;
    while (shown.size() > 3 && ImGui::CalcTextSize((shown + "...").c_str()).x > max_width) shown.pop_back();
    shown += "...";
    ImGui::TextUnformatted(shown.c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", text.c_str());
}

} // namespace fluxvoice::frontend::widgets
