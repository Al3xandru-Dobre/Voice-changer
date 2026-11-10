// Catalogue — installed voices and catalogue entries with search, filters,
// sort, a detail pane and actions (frontend/README.md §4.6).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/VoiceCatalog.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace {

char g_query[128] = "";
int  g_lang = 0;     // 0 = any
int  g_lic  = 0;     // 0 = any, then LicenseClass order
int  g_tag  = 0;
bool g_installed_only = false;
int  g_sort = 0;
std::string g_selected_id;

ImVec4 licence_colour(LicenseClass c) {
    switch (c) {
        case LicenseClass::CommercialClean: return theme_colors().success;
        case LicenseClass::ResearchOnly:    return theme_colors().danger;
        case LicenseClass::Private:         return theme_colors().warning;
        case LicenseClass::Unknown:         break;
    }
    return theme_colors().neutral;
}

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + v[i];
    return out;
}

void draw_detail(ViewContext& ctx, const VoiceCatalog::Item& it) {
    AppModel& app = ctx.app;
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.25f);
    ImGui::TextWrapped("%s", it.display.c_str());
    ImGui::PopFont();
    widgets::pill(to_string(it.license_class), licence_colour(it.license_class));
    ImGui::SameLine();
    if (it.installed) widgets::pill("installed", theme_colors().accent);
    ImGui::Spacing();
    if (widgets::kv_table_begin("detail")) {
        widgets::kv_row("Id", it.id);
        widgets::kv_row("Bundle", it.bundle.empty() ? "-" : it.bundle);
        widgets::kv_row("Languages", it.languages.empty() ? "-" : join(it.languages));
        widgets::kv_row("Licence", it.license.empty() ? "unknown" : it.license);
        if (!it.attribution.empty()) widgets::kv_row("Attribution", it.attribution);
        if (!it.source.empty()) widgets::kv_row("Source", it.source);
        if (!it.tags.empty()) widgets::kv_row("Tags", join(it.tags));
        if (it.size_mb > 0.0) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f MB", it.size_mb);
            widgets::kv_row("Size", buf);
        }
        if (it.voice) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "%zu dims; log-f0 %.2f +/- %.2f; energy %.2f +/- %.2f", it.voice->embedding.size(),
                          static_cast<double>(it.voice->logf0_mean), static_cast<double>(it.voice->logf0_std),
                          static_cast<double>(it.voice->energy_mean), static_cast<double>(it.voice->energy_std));
            widgets::kv_row("Embedding / stats", buf);
            widgets::kv_row("Folder", it.voice->bundle_dir.string());
        }
        widgets::kv_table_end();
    }
    ImGui::Spacing();

    if (it.voice) {
        const bool active = app.active_bundle() == it.bundle;
        if (ImGui::Button("Set as Voice A")) {
            if (!active) static_cast<void>(app.set_active_bundle(it.bundle));
            static_cast<void>(app.set_voice_a(it.voice->name));
        }
        ImGui::SameLine();
        if (ImGui::Button("Set as Voice B")) {
            if (!active) static_cast<void>(app.set_active_bundle(it.bundle));
            static_cast<void>(app.set_voice_b(it.voice->name));
        }
        ImGui::SameLine();
        if (ImGui::Button("Open folder")) ctx.shell.open_path(it.voice->bundle_dir);
    }
    if (it.entry) {
        if (!it.page.empty() && ImGui::Button("Open page")) ctx.shell.open_url(it.page);
        if (!it.url.empty()) {
            ImGui::SameLine();
            if (ImGui::Button("Open download link")) ctx.shell.open_url(it.url);
        }
        ImGui::SameLine();
        if (ImGui::Button("Add bundle folder...")) {
            ctx.shell.pick_folder([&app](const std::vector<std::string>& paths) {
                if (!paths.empty()) static_cast<void>(app.add_bundle_dir(paths.front()));
            });
        }
        widgets::dim_text_wrapped("FluxVoice never downloads. Obtain the pack with your browser, unpack it, then add its folder.");
    }
}

} // namespace

void draw_catalogue(ViewContext& ctx) {
    AppModel& app = ctx.app;
    const ThemeColors& t = theme_colors();
    widgets::heading("Catalogue");

    const std::vector<std::string> langs = app.catalog().all_languages();
    const std::vector<std::string> tags  = app.catalog().all_tags();

    // Toolbar: search + actions on the first row, filters on the second.
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 250.0f * ctx.scale);
    ImGui::InputTextWithHint("##q", "Search name, bundle, tag, licence...", g_query, sizeof g_query);
    ImGui::SameLine();
    if (ImGui::Button("Rescan")) app.rescan_models();
    ImGui::SameLine();
    if (ImGui::Button("Add bundle folder...")) {
        ctx.shell.pick_folder([&app](const std::vector<std::string>& paths) {
            if (!paths.empty()) static_cast<void>(app.add_bundle_dir(paths.front()));
        });
    }
    std::vector<std::string> lang_items = {"Any language"};
    lang_items.insert(lang_items.end(), langs.begin(), langs.end());
    ImGui::SetNextItemWidth(150.0f * ctx.scale);
    widgets::combo_strings("##lang", g_lang, lang_items);
    ImGui::SameLine();
    const std::vector<std::string> lic_items = {"Any licence", "commercial-clean", "research-only", "private", "unknown"};
    ImGui::SetNextItemWidth(170.0f * ctx.scale);
    widgets::combo_strings("##lic", g_lic, lic_items);
    ImGui::SameLine();
    std::vector<std::string> tag_items = {"Any tag"};
    tag_items.insert(tag_items.end(), tags.begin(), tags.end());
    ImGui::SetNextItemWidth(140.0f * ctx.scale);
    widgets::combo_strings("##tag", g_tag, tag_items);
    ImGui::SameLine();
    ImGui::Checkbox("Installed only", &g_installed_only);
    ImGui::SameLine();
    const std::vector<std::string> sort_items = {"Sort: name", "Sort: language", "Sort: licence"};
    ImGui::SetNextItemWidth(150.0f * ctx.scale);
    widgets::combo_strings("##sort", g_sort, sort_items);

    VoiceCatalog::Filter f;
    f.query = g_query;
    if (g_lang > 0 && static_cast<std::size_t>(g_lang) < lang_items.size()) f.language = lang_items[static_cast<std::size_t>(g_lang)];
    if (g_lic > 0) f.license = static_cast<LicenseClass>(g_lic - 1);
    if (g_tag > 0 && static_cast<std::size_t>(g_tag) < tag_items.size()) f.tag = tag_items[static_cast<std::size_t>(g_tag)];
    f.installed_only = g_installed_only;
    f.sort = static_cast<VoiceCatalog::Sort>(g_sort);
    const std::vector<VoiceCatalog::Item> items = app.catalog().search(f);

    for (const BundleInfo& b : app.catalog().bundles())
        if (!b.valid()) widgets::notice(("Bundle '" + b.name + "' is invalid: " + b.error).c_str(), t.danger);
    for (const CatalogueIndex& i : app.catalog().indices())
        if (!i.ok()) widgets::notice(("Catalogue index " + i.path.string() + ": " + i.error).c_str(), t.danger);

    const float list_w = ImGui::GetContentRegionAvail().x * 0.46f;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.card_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, t.card_border);
    if (ImGui::BeginChild("list", ImVec2(list_w, 0.0f), ImGuiChildFlags_Borders)) {
        if (items.empty()) widgets::dim_text_wrapped("No voices match. Add a bundle folder or clear the filters.");
        for (const VoiceCatalog::Item& it : items) {
            ImGui::PushID(it.id.c_str());
            const bool selected = it.id == g_selected_id;
            if (ImGui::Selectable("##row", selected, 0, ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 2.0f)))
                g_selected_id = it.id;
            ImGui::SameLine(8.0f);
            ImGui::BeginGroup();
            ImGui::TextUnformatted(it.display.c_str());
            widgets::pill(to_string(it.license_class), licence_colour(it.license_class));
            ImGui::SameLine();
            if (it.installed) { widgets::pill("installed", t.accent); ImGui::SameLine(); }
            widgets::dim_text((it.bundle.empty() ? std::string("-") : it.bundle).c_str());
            if (!it.languages.empty()) { ImGui::SameLine(); widgets::dim_text(join(it.languages).c_str()); }
            ImGui::EndGroup();
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();
    if (ImGui::BeginChild("detail", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        const VoiceCatalog::Item* sel = nullptr;
        for (const VoiceCatalog::Item& it : items)
            if (it.id == g_selected_id) sel = &it;
        if (sel) draw_detail(ctx, *sel);
        else widgets::dim_text_wrapped("Select a voice to see its licence, attribution and actions.");
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

} // namespace fluxvoice::frontend
