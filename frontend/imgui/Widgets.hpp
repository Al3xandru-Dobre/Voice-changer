// Widgets — small reusable ImGui pieces shared by the views: cards, status
// pills, level meters, a toggle button, string combos, generated parameter
// sliders and key/value rows.
#pragma once

#include "core/TrayModel.hpp"
#include "voice/VoiceParams.hpp"

#include <imgui.h>

#include <cstddef>
#include <string>
#include <vector>

namespace fluxvoice::frontend::widgets {

/// A bordered, padded child window. Always returns true; always call end_card().
bool begin_card(const char* id, const char* title = nullptr, float height = 0.0f);
void end_card();

void heading(const char* text);          // section heading (title font)
void dim_text(const char* text);         // secondary text
void dim_text_wrapped(const char* text);
void status_dot(const ImVec4& color, float radius_px);
void pill(const char* text, const ImVec4& color);

/// Horizontal level meter over -60..0 dBFS; red above -3 dBFS.
void level_meter(const char* id, float dbfs, float width, float height);

/// A large on/off button; returns true when clicked.
bool toggle_button(const char* on_label, const char* off_label, bool on, const ImVec2& size,
                   const ImVec4& on_color);

/// Combo over strings; `index` may be -1 for "nothing selected".
bool combo_strings(const char* label, int& index, const std::vector<std::string>& items,
                   const char* empty_hint = "(none)");

/// A slider/checkbox generated from a descriptor. Returns true when changed;
/// double-click resets to the default (also reported as a change).
bool param_control(const ParamDesc& d, float& value);

void help_marker(const char* text);
/// Two-column key/value table. Only when it returns true: add kv_row()s and
/// call kv_table_end() (ImGui forbids EndTable() after a false BeginTable()).
bool kv_table_begin(const char* id);
void kv_row(const char* key, const std::string& value);
void kv_table_end();

/// Inline notice block in a semantic colour.
void notice(const char* text, const ImVec4& color);

/// Fits long device/voice names: full text, tooltip when hovered.
void text_ellipsis(const std::string& text, float max_width);

} // namespace fluxvoice::frontend::widgets
