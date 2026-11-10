// Voices — bundle/voice pickers, generated parameter sliders, preset bar
// (frontend/README.md §4.3). Every slider comes from kParams; nothing here
// hard-codes a range or a default.
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/VoiceCatalog.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"
#include "voice/Presets.hpp"
#include "voice/VoiceParams.hpp"

#include <imgui.h>

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace {

int index_of(const std::vector<std::string>& items, const std::string& value) {
    for (std::size_t i = 0; i < items.size(); ++i)
        if (items[i] == value) return static_cast<int>(i);
    return -1;
}

char g_save_name[64] = "";

void draw_voice_badge(const VoiceEntry* v) {
    if (!v) return;
    const LicenseClass lc = v->license_class();
    const ImVec4 colour = lc == LicenseClass::Private ? theme_colors().warning
                          : lc == LicenseClass::ResearchOnly ? theme_colors().danger
                          : lc == LicenseClass::CommercialClean ? theme_colors().success : theme_colors().neutral;
    widgets::pill(to_string(lc), colour);
    ImGui::SameLine();
    widgets::dim_text(v->license.c_str());
    if (!v->attribution.empty()) widgets::dim_text_wrapped(v->attribution.c_str());
}

void draw_group(ViewContext& ctx, ParamGroup group, ParamTier tier, const char* title) {
    AppModel& app = ctx.app;
    bool any = false;
    for (const ParamDesc& d : kParams)
        if (d.group == group && d.tier == tier) any = true;
    if (!any) return;
    ImGui::SeparatorText(title);
    for (const ParamDesc& d : kParams) {
        if (d.group != group || d.tier != tier) continue;
        float v = app.params().get(d.id);
        ImGui::SetNextItemWidth(-200.0f * ctx.scale);
        if (widgets::param_control(d, v)) app.set_param(d.id, v);
    }
}

} // namespace

void draw_voices(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const ThemeColors& t = theme_colors();
    widgets::heading("Voices");

    if (widgets::begin_card("bundle", "Bundle & voices")) {
        std::vector<std::string> bundles;
        for (const BundleInfo& b : app.catalog().bundles())
            if (b.valid() && !b.voices.empty()) bundles.push_back(b.name);
        int bi = index_of(bundles, app.active_bundle());
        ImGui::SetNextItemWidth(300.0f * ctx.scale);
        if (widgets::combo_strings("Bundle", bi, bundles, "(no bundle with voices found)"))
            static_cast<void>(app.set_active_bundle(bundles[static_cast<std::size_t>(bi)]));
        ImGui::SameLine();
        if (ImGui::SmallButton("Browse catalogue")) app.navigate(Section::Catalogue);
        if (const BundleInfo* b = app.active_bundle_info()) {
            ImGui::SameLine();
            widgets::dim_text(b->has_translator ? "Mode A + B bundle" : "Mode A bundle (no translator)");
        }
        if (bundles.empty())
            widgets::notice("No voice bundle found. Set the models directory in Settings or add a bundle folder in the Catalogue.", t.warning);

        const std::vector<std::string> voices = app.voice_names();
        int va = index_of(voices, app.voice_a());
        int vb = index_of(voices, app.voice_b());
        ImGui::SetNextItemWidth(300.0f * ctx.scale);
        if (widgets::combo_strings("Voice A", va, voices)) static_cast<void>(app.set_voice_a(voices[static_cast<std::size_t>(va)]));
        draw_voice_badge(app.voice_entry(app.voice_a()));
        ImGui::SetNextItemWidth(300.0f * ctx.scale);
        if (widgets::combo_strings("Voice B (blend target)", vb, voices)) static_cast<void>(app.set_voice_b(voices[static_cast<std::size_t>(vb)]));
        draw_voice_badge(app.voice_entry(app.voice_b()));
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("presets", "Presets")) {
        const std::vector<std::string> presets = app.preset_names();
        int pi = index_of(presets, app.current_preset());
        ImGui::SetNextItemWidth(260.0f * ctx.scale);
        if (widgets::combo_strings("##preset", pi, presets, "(custom)"))
            static_cast<void>(app.load_preset(presets[static_cast<std::size_t>(pi)]));
        ImGui::SameLine();
        if (app.preset_modified()) { widgets::pill("modified", t.warning); ImGui::SameLine(); }
        if (ImGui::Button("Save as...")) {
            std::snprintf(g_save_name, sizeof g_save_name, "%s", app.current_preset().c_str());
            ImGui::OpenPopup("Save preset");
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset to defaults")) app.reset_params();
        ImGui::SameLine();
        if (ImGui::Button("Reload")) app.reload_presets();
        ImGui::SameLine();
        widgets::help_marker("Voice presets never change the translation mode or languages: applying Deep while "
                             "translating es -> en keeps translating es -> en.");

        if (ImGui::BeginPopupModal("Save preset", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted("Preset name");
            ImGui::SetNextItemWidth(300.0f * ctx.scale);
            const bool enter = ImGui::InputText("##name", g_save_name, sizeof g_save_name, ImGuiInputTextFlags_EnterReturnsTrue);
            widgets::dim_text_wrapped(app.paths().user_preset_dir.string().c_str());
            const bool ok = ImGui::Button("Save", ImVec2(120.0f * ctx.scale, 0.0f)) || enter;
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120.0f * ctx.scale, 0.0f))) ImGui::CloseCurrentPopup();
            if (ok && g_save_name[0] != '\0') {
                static_cast<void>(app.save_preset_as(g_save_name));
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        for (const std::string& w : app.preset_warnings()) {
            ImGui::PushStyleColor(ImGuiCol_Text, t.warning);
            ImGui::TextWrapped("%s", w.c_str());
            ImGui::PopStyleColor();
        }
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("params", "Controls")) {
        const TelemetrySnapshot tel = app.telemetry();
        if (!app.engine_on()) widgets::pill("engine stopped: applied at start", t.neutral);
        else if (tel.params_applied == tel.params_pushed) widgets::pill("applied by worker", t.success);
        else widgets::pill("pending", t.warning);
        ImGui::SameLine();
        widgets::dim_text("Controls act on the vocoder conditioning and the output chain; there is no dry/wet mix.");

        draw_group(ctx, ParamGroup::Pitch, ParamTier::Mvp, "Pitch");
        draw_group(ctx, ParamGroup::Timbre, ParamTier::Mvp, "Timbre");
        draw_group(ctx, ParamGroup::Output, ParamTier::Mvp, "Output");
        if (ImGui::CollapsingHeader("Advanced (stretch tier)")) {
            widgets::dim_text_wrapped("These controls are defined by the descriptor table but may not be implemented by the loaded bundle yet.");
            draw_group(ctx, ParamGroup::Pitch, ParamTier::Stretch, "Pitch");
            draw_group(ctx, ParamGroup::Timbre, ParamTier::Stretch, "Timbre");
            draw_group(ctx, ParamGroup::Output, ParamTier::Stretch, "Output");
        }
        widgets::end_card();
    }
}

} // namespace fluxvoice::frontend
