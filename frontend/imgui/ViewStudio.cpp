// Voice Studio — the six-step custom-voice wizard (frontend/README.md §4.4).
#include "imgui/Views.hpp"

#include "core/AppModel.hpp"
#include "core/VoiceStudio.hpp"
#include "imgui/Theme.hpp"
#include "imgui/Widgets.hpp"

#include <imgui.h>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace fluxvoice::frontend {

namespace {

/// Keep a fixed ImGui text buffer in sync with a model string.
struct TextField {
    char        buf[256] = "";
    std::string mirror;

    void sync(const std::string& model_value) {
        if (model_value != mirror) {
            std::snprintf(buf, sizeof buf, "%s", model_value.c_str());
            mirror = model_value;
        }
    }
    /// Draw; on edit copy back into `model_value`.
    bool draw(const char* label, std::string& model_value, const char* hint = nullptr) {
        sync(model_value);
        const bool changed = hint ? ImGui::InputTextWithHint(label, hint, buf, sizeof buf)
                                  : ImGui::InputText(label, buf, sizeof buf);
        if (changed) {
            model_value = buf;
            mirror      = model_value;
        }
        return changed;
    }
};

TextField g_name, g_display, g_consent_note, g_session, g_target_bundle;

void step_indicator(const VoiceStudioModel& m) {
    const ThemeColors& t = theme_colors();
    for (int i = 0; i < static_cast<int>(StudioStep::Count); ++i) {
        const auto s = static_cast<StudioStep>(i);
        const bool current = s == m.step();
        const bool done = i < static_cast<int>(m.step());
        char label[64];
        std::snprintf(label, sizeof label, "%d. %s", i + 1, to_string(s));
        widgets::pill(label, current ? t.accent : (done ? t.success : t.neutral));
        if (i + 1 < static_cast<int>(StudioStep::Count)) ImGui::SameLine();
    }
    ImGui::Spacing();
}

void takes_table(ViewContext& ctx, std::vector<Take>& takes, bool removable) {
    const ThemeColors& t = theme_colors();
    if (takes.empty()) { widgets::dim_text("No files yet."); return; }
    if (ImGui::BeginTable("takes", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("File", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn("Session");
        ImGui::TableSetupColumn("Length");
        ImGui::TableSetupColumn("Format");
        ImGui::TableSetupColumn("Peak");
        ImGui::TableSetupColumn("SNR");
        ImGui::TableSetupColumn("Verdict", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableHeadersRow();
        std::optional<std::size_t> remove;
        for (std::size_t i = 0; i < takes.size(); ++i) {
            const Take& tk = takes[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(i));
            widgets::text_ellipsis(tk.path.filename().string(), ImGui::GetContentRegionAvail().x);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(tk.session.c_str());
            ImGui::TableNextColumn(); ImGui::Text("%.1f s", tk.stats.seconds);
            ImGui::TableNextColumn();
            if (tk.stats.ok) ImGui::Text("%u Hz %u-bit %s", tk.stats.sample_rate, tk.stats.bits, tk.stats.channels == 1 ? "mono" : "multi");
            else ImGui::TextUnformatted("-");
            ImGui::TableNextColumn(); ImGui::Text("%.1f dBFS", static_cast<double>(tk.stats.peak_dbfs));
            ImGui::TableNextColumn(); ImGui::Text("%.0f dB", static_cast<double>(tk.stats.snr_db));
            ImGui::TableNextColumn();
            const ImVec4 c = tk.verdict == TakeVerdict::Clean ? t.success : t.danger;
            ImGui::TextColored(c, "%s", tk.stats.ok ? to_string(tk.verdict) : tk.stats.error.c_str());
            if (removable) {
                ImGui::SameLine();
                if (ImGui::SmallButton("x")) remove = i;
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
        if (remove) {
            takes.erase(takes.begin() + static_cast<std::ptrdiff_t>(*remove));
            ctx.app.studio_save_draft();
        }
    }
}

void pick_wavs(ViewContext& ctx, bool many, const std::function<void(const std::vector<std::string>&)>& on_files) {
    ctx.shell.pick_files("WAV audio", "wav", many, on_files);
}

void draw_step_name(ViewContext& ctx) {
    AppModel& app = ctx.app;
    VoiceStudioModel& m = app.studio();
    StudioDraft& d = m.draft();
    widgets::dim_text_wrapped("A custom voice is a new speaker row fine-tuned into the multi-speaker vocoder. It needs "
                              "1-3 hours of clean recordings and the right to use the voice.");
    ImGui::SetNextItemWidth(300.0f * ctx.scale);
    if (g_name.draw("Voice name", d.voice_name, "lowercase letters, digits, - or _ (e.g. me)")) app.studio_save_draft();
    if (const auto problem = m.name_problem(&app.catalog())) {
        if (!d.voice_name.empty()) widgets::notice(problem->c_str(), theme_colors().warning);
    }
    ImGui::SetNextItemWidth(300.0f * ctx.scale);
    if (g_display.draw("Display name", d.display_name, "as shown in the voice list")) app.studio_save_draft();
    if (ImGui::Checkbox("This is my own voice", &d.is_own_voice)) app.studio_save_draft();
    if (ImGui::Checkbox("I have the right to use this voice and consent to recording it", &d.consent)) app.studio_save_draft();
    if (!d.is_own_voice) {
        ImGui::SetNextItemWidth(-1.0f);
        if (g_consent_note.draw("Where the written consent is kept", d.consent_note, "required for another person's voice"))
            app.studio_save_draft();
    }
    ImGui::SetNextItemWidth(300.0f * ctx.scale);
    const std::string bundle_hint = (d.voice_name.empty() ? std::string("voice") : d.voice_name) + "-bundle";
    if (g_target_bundle.draw("Target bundle folder name", d.target_bundle, bundle_hint.c_str())) app.studio_save_draft();
}

void draw_step_setup(ViewContext& ctx) {
    AppModel& app = ctx.app;
    StudioDraft& d = app.studio().draft();
    widgets::dim_text_wrapped("Bad audio cannot be fixed in training; denoisers leave artefacts the vocoder learns. "
                              "Re-record rather than repair.");
    for (std::size_t i = 0; i < kSetupItemCount; ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Checkbox(kSetupItems[i], &d.setup[i])) app.studio_save_draft();
        ImGui::PopID();
    }
}

void draw_step_takes(ViewContext& ctx) {
    AppModel& app = ctx.app;
    VoiceStudioModel& m = app.studio();
    StudioDraft& d = m.draft();
    const ThemeColors& t = theme_colors();
    char overlay[96];
    std::snprintf(overlay, sizeof overlay, "%.1f clean minutes of %.0f minimum (%.0f ideal)", m.clean_minutes(),
                  m.requirements().minimum_minutes, VoiceStudioModel::kIdealMinutes);
    ImGui::ProgressBar(m.progress_to_minimum(), ImVec2(-1.0f, 0.0f), overlay);
    if (m.excluded_minutes() > 0.0) {
        char ex[96];
        std::snprintf(ex, sizeof ex, "%.1f minutes excluded (clipping, SNR below 30 dB, not mono, or too short)", m.excluded_minutes());
        ImGui::TextColored(t.warning, "%s", ex);
    }
    ImGui::SetNextItemWidth(120.0f * ctx.scale);
    static std::string session_text = "s01";
    g_session.draw("Session", session_text, "s01");
    ImGui::SameLine();
    if (ImGui::Button("Add WAV takes...")) {
        AppModel* ap = &app;
        const std::string session = session_text;
        pick_wavs(ctx, true, [ap, session](const std::vector<std::string>& paths) {
            for (const std::string& p : paths) ap->studio().add_take(p, session);
            ap->studio_save_draft();
        });
    }
    ImGui::SameLine();
    widgets::help_marker("Each file is read and checked: duration, format, peak, RMS, clipping (any sample >= 0.999 FS) "
                         "and an SNR estimate from the quietest 10 % of 10 ms blocks vs. the loudest 10 %. Files "
                         "that fail are kept in the list but excluded from the total.");
    takes_table(ctx, d.takes, true);
    ImGui::Spacing();
    ImGui::SeparatorText("Material covered");
    for (std::size_t i = 0; i < kMaterialItemCount; ++i) {
        ImGui::PushID(static_cast<int>(100 + i));
        if (ImGui::Checkbox(kMaterialItems[i], &d.material[i])) app.studio_save_draft();
        ImGui::PopID();
    }
}

void draw_step_roomtone(ViewContext& ctx) {
    AppModel& app = ctx.app;
    VoiceStudioModel& m = app.studio();
    StudioDraft& d = m.draft();
    ImGui::SeparatorText("Room tone (30 s per session)");
    ImGui::Text("%.0f s collected", m.room_tone_seconds());
    ImGui::SameLine();
    if (ImGui::Button("Add room tone...")) {
        AppModel* ap = &app;
        pick_wavs(ctx, true, [ap](const std::vector<std::string>& paths) {
            for (const std::string& p : paths) ap->studio().add_room_tone(p);
            ap->studio_save_draft();
        });
    }
    takes_table(ctx, d.room_tone, true);
    ImGui::Spacing();
    ImGui::SeparatorText("Live-mic source set (10 min through the everyday microphone)");
    widgets::dim_text_wrapped("Not training data: the only evaluation input that matches your real microphone and room "
                              "(Stage 11). Optional for the fine-tune itself.");
    ImGui::Text("%.1f min collected", m.live_mic_minutes());
    ImGui::SameLine();
    if (ImGui::Button("Add live-mic recordings...")) {
        AppModel* ap = &app;
        pick_wavs(ctx, true, [ap](const std::vector<std::string>& paths) {
            for (const std::string& p : paths) ap->studio().add_live_mic(p);
            ap->studio_save_draft();
        });
    }
    takes_table(ctx, d.live_mic, true);
}

void draw_command(ViewContext& ctx, const char* title, const std::string& cmd) {
    ImGui::TextUnformatted(title);
    ImGui::SameLine();
    ImGui::PushID(title);
    if (ImGui::SmallButton("Copy")) ctx.shell.set_clipboard(cmd);
    ImGui::PopID();
    ImGui::PushStyleColor(ImGuiCol_Text, theme_colors().text_dim);
    ImGui::TextWrapped("%s", cmd.c_str());
    ImGui::PopStyleColor();
}

void draw_step_prepare(ViewContext& ctx) {
    AppModel& app = ctx.app;
    VoiceStudioModel& m = app.studio();
    const ThemeColors& t = theme_colors();
#if defined(_WIN32)
    const Shell shell = Shell::PowerShell;
#else
    const Shell shell = Shell::Posix;
#endif
    const StudioCommands c = m.commands(app.paths().repo_root, app.paths().studio_dir, app.paths().models_dir, shell);
    widgets::dim_text_wrapped("The pipeline is textless: no transcripts are needed. Prepare normalises and segments the "
                              "takes, fine-tune adds the speaker row (50/50 own/public batches with formant/pitch/EQ "
                              "perturbation), export writes a new bundle version with a private licence.");
    draw_command(ctx, "1. Prepare", c.prepare);
    draw_command(ctx, "2. Fine-tune", c.finetune);
    draw_command(ctx, "3. Export", c.export_);
    ImGui::Spacing();
    if (ImGui::Button("Write job files")) {
        std::string err;
        if (app.studio_write_job(err)) app.notify(Notice::Level::Info, "Wrote job.json, takes.tsv, room_tone.tsv and run_finetune scripts.");
        else app.notify(Notice::Level::Error, err);
    }
    ImGui::SameLine();
    if (ImGui::Button("Open project folder")) ctx.shell.open_path(m.project_dir(app.paths().studio_dir));
    ImGui::SameLine();
    widgets::dim_text(m.job_written(app.paths().studio_dir) ? "job.json written" : "job.json not written yet");

    if (const auto st = app.studio_status()) {
        ImGui::Spacing();
        ImGui::SeparatorText("Job status");
        char overlay[128];
        std::snprintf(overlay, sizeof overlay, "%s: %s", st->stage.c_str(), st->message.c_str());
        ImGui::ProgressBar(st->progress, ImVec2(-1.0f, 0.0f), overlay);
        if (st->eta_s >= 0.0) ImGui::Text("about %.0f s remaining", st->eta_s);
        if (st->stage == "error") widgets::notice(st->message.c_str(), t.danger);
    } else {
        widgets::dim_text("No status.json yet; the training side writes it while the job runs.");
    }
}

void draw_step_install(ViewContext& ctx) {
    AppModel& app = ctx.app;
    VoiceStudioModel& m = app.studio();
    const ThemeColors& t = theme_colors();
    if (m.installed(app.catalog())) {
        widgets::notice(("Voice '" + m.draft().voice_name + "' is installed. It is private: the bundle is git-ignored and must not be distributed.").c_str(), t.success);
        for (const BundleInfo& b : app.catalog().bundles())
            for (const VoiceEntry& v : b.voices)
                if (v.name == m.draft().voice_name) {
                    if (ImGui::Button("Use as Voice A")) {
                        static_cast<void>(app.set_active_bundle(b.name));
                        static_cast<void>(app.set_voice_a(v.name));
                        app.navigate(Section::Voices);
                    }
                }
    } else {
        widgets::dim_text_wrapped("When the export has finished, the new bundle's voices.json contains the voice. "
                                  "Rescan the models directory (or add the bundle folder) to install it.");
        if (ImGui::Button("Rescan models")) app.rescan_models();
    }
}

} // namespace

void draw_studio(ViewContext& ctx) {
    AppModel& app = ctx.app;
    VoiceStudioModel& m = app.studio();
    widgets::heading("Voice Studio");
    step_indicator(m);

    if (widgets::begin_card("step", to_string(m.step()))) {
        switch (m.step()) {
            case StudioStep::NameConsent: draw_step_name(ctx); break;
            case StudioStep::Setup:       draw_step_setup(ctx); break;
            case StudioStep::Takes:       draw_step_takes(ctx); break;
            case StudioStep::RoomTone:    draw_step_roomtone(ctx); break;
            case StudioStep::Prepare:     draw_step_prepare(ctx); break;
            case StudioStep::Install:     draw_step_install(ctx); break;
            case StudioStep::Count:       break;
        }
        widgets::end_card();
    }
    ImGui::Spacing();

    std::string why;
    const bool can = m.can_advance(&why);
    ImGui::BeginDisabled(m.step() == StudioStep::NameConsent);
    if (ImGui::Button("Back", ImVec2(100.0f * ctx.scale, 0.0f))) { m.prev(); app.studio_save_draft(); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!can || m.step() == StudioStep::Install);
    if (ImGui::Button("Next", ImVec2(100.0f * ctx.scale, 0.0f))) { m.next(); app.studio_save_draft(); }
    ImGui::EndDisabled();
    if (!can && !why.empty() && m.step() != StudioStep::Install) {
        ImGui::SameLine();
        widgets::dim_text(why.c_str());
    }
}

} // namespace fluxvoice::frontend
