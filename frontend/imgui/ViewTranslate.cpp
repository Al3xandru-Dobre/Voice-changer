// Translate — Mode B controls: language pair, same-language guard, Wait-k,
// translation toggle, lag telemetry placeholder (frontend/README.md §4.5).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/TranslationModel.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"
#include "voice/VoiceParams.hpp"

#include <imgui.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace {

int index_of_code(const LangTable& t, const std::string& code) {
    for (std::size_t i = 0; i < t.langs.size(); ++i)
        if (t.langs[i].code == code) return static_cast<int>(i);
    return -1;
}

std::vector<std::string> lang_labels(const LangTable& t) {
    std::vector<std::string> out;
    for (const Lang& l : t.langs) out.push_back(l.name + " (" + l.code + ")");
    return out;
}

} // namespace

void draw_translate(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const ThemeColors& t = theme_colors();
    const TranslationModel& tm = app.translation();
    widgets::heading("Translate");

    // A Mode A only bundle: every control below stays readable but disabled
    // (frontend/README.md §4.5); the tray refuses translation for the same reason.
    const bool available = tm.translator_available();
    if (!available)
        widgets::notice(("Translation is unavailable: " + tm.unavailable_reason() +
                         ". The controls stay readable; they activate when a bundle with translator_enc.onnx and "
                         "translator_dec.onnx is loaded.").c_str(), t.warning);
    ImGui::BeginDisabled(!available);

    if (widgets::begin_card("pair", "Languages")) {
        const LangTable& table = tm.table();
        const std::vector<std::string> labels = lang_labels(table);
        int si = index_of_code(table, app.src_lang());
        int ti = index_of_code(table, app.tgt_lang());
        ImGui::SetNextItemWidth(220.0f * ctx.scale);
        if (widgets::combo_strings("Source", si, labels)) static_cast<void>(app.set_language_pair(table.langs[static_cast<std::size_t>(si)].code, app.tgt_lang()));
        ImGui::SameLine();
        if (ImGui::Button("<->")) app.swap_languages();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(220.0f * ctx.scale);
        if (widgets::combo_strings("Target", ti, labels)) static_cast<void>(app.set_language_pair(app.src_lang(), table.langs[static_cast<std::size_t>(ti)].code));
        widgets::dim_text(table.from_bundle ? "Language table from the bundle's langs.json" : "Built-in language table (bundle has no langs.json)");
        if (!app.translation_message().empty() && app.translation_message() != tm.unavailable_reason()) {
            const PairValidation v = tm.validate_pair(app.src_lang(), app.tgt_lang());
            widgets::notice(app.translation_message().c_str(), v.ok ? t.text_dim : t.warning);
        }
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("waitk", "Wait-k")) {
        // Range, default and help come from the wait_k descriptor (kParams);
        // the chunk length from the translator contract once it is loaded.
        const ParamDesc& wk = param_desc(ParamId::WaitK);
        float k = app.params().get(ParamId::WaitK);
        ImGui::SetNextItemWidth(320.0f * ctx.scale);
        if (widgets::param_control(wk, k)) app.set_param(ParamId::WaitK, k);
        const std::optional<double> onset = tm.onset_estimate_ms(app.wait_k());
        const std::optional<double> chunk = tm.onset_estimate_ms(1);
        if (onset) ImGui::Text("about %.0f ms extra onset latency (%d x %.0f ms chunks)", *onset, app.wait_k(), *chunk);
        else widgets::dim_text("Onset latency: k x the translator's chunk length, known once the bundle is loaded.");
        widgets::dim_text_wrapped("Lower k answers faster, higher k resolves word order better. Changing k takes "
                                  "effect on the next chunk without resetting translator state. Choosing a language "
                                  "pair applies its default from langs.json; a preset's own Wait-k wins.");
        if (tm.adaptive_policy_available()) {  // hidden unless the bundle declares readiness (M3 Stage 09)
            const ParamDesc& ad = param_desc(ParamId::AdaptivePolicy);
            float adaptive = app.params().get(ParamId::AdaptivePolicy);
            if (widgets::param_control(ad, adaptive)) app.set_param(ParamId::AdaptivePolicy, adaptive);
        }
        widgets::end_card();
    }
    ImGui::Spacing();

    if (widgets::begin_card("mode", "Mode")) {
        const bool on = app.translate();
        if (widgets::toggle_button("Translation ON (Mode B)", "Translation OFF (Mode A)", on,
                                   ImVec2(280.0f * ctx.scale, 42.0f * ctx.scale), t.accent))
            static_cast<void>(app.set_translate(!on));
        ImGui::SameLine();
        widgets::dim_text_wrapped("Switching mode or languages resets only translator state; the encoder and "
                                  "vocoder stay continuous, with a 20 ms output ramp.");
        widgets::end_card();
    }
    ImGui::EndDisabled();
    ImGui::Spacing();

    if (widgets::begin_card("lag", "Lag telemetry")) {
        widgets::dim_text_wrapped("reads / writes / flushes and lag_ms are published by the Mode B pipeline "
                                  "(Milestone 3 / Stage 08) and appear here while translation runs.");
        widgets::end_card();
    }
}

} // namespace fluxvoice::frontend
