// Presets — JSON preset files for VoiceParams (Milestone 2 / Stage 09,
// translation keys from Milestone 3 / Stage 09).
//
// Format (presets/*.json):
//   { "schema": 1, "name": "Deep", "description": "...",
//     "voice_a": "<voice name>", "voice_b": "<voice name>",
//     "params": { "pitch_shift": -4, "intonation": 0.9, "translate": 1, "wait_k": 3 },
//     "src_lang": "es", "tgt_lang": "en" }
//
// Every key except `schema`, `name` and `params` is optional. Omitted keys
// leave the current value unchanged, so a voice-only preset (Deep, Bright, ...)
// never turns translation off or switches languages (M3 / Stage 09 rule).
// The loader clamps out-of-range values with a warning, warns on unknown keys
// and rejects any schema other than the integer 1 (a boolean `true` or a float
// `1.0` is not 1 — BUG-006 lesson carried over from the Python validator).
//
// Voice names and language codes are kept as strings here; resolving them to
// bundle indices needs the loaded voices.json / langs.json and happens on the
// control thread (frontend AppModel or the CLI).
#pragma once

#include "voice/VoiceParams.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxvoice {

struct Preset {
    std::string name;
    std::string description;
    std::optional<std::string> voice_a;
    std::optional<std::string> voice_b;
    /// Only the keys present in the file are engaged.
    std::array<std::optional<float>, kParamCount> params{};
    std::optional<std::string> src_lang;
    std::optional<std::string> tgt_lang;

    // Runtime bookkeeping, never serialized.
    std::filesystem::path path;
    bool builtin = false;

    bool has(ParamId id) const { return params[param_index(id)].has_value(); }
};

struct PresetLoadResult {
    std::optional<Preset>    preset;    // engaged on success
    std::vector<std::string> warnings;  // clamped values, unknown keys, skipped values
    std::string              error;     // non-empty when rejected

    bool ok() const { return preset.has_value(); }
};

/// Parse one preset document. Never throws.
PresetLoadResult parse_preset(std::string_view json_text);

/// Read + parse; sets `preset->path`. Never throws.
PresetLoadResult load_preset_file(const std::filesystem::path& path);

/// Pretty JSON (2-space indent, sorted keys inside "params").
std::string preset_to_json(const Preset& p);

/// Write `preset_to_json(p)`; false with `error` set on I/O failure.
bool save_preset_file(const std::filesystem::path& path, const Preset& p, std::string& error);

/// Snapshot every descriptor into a preset (all params explicit), e.g. for
/// "Save as...". `voice_a`/`voice_b`/langs are recorded only when non-empty.
Preset preset_from_params(std::string name, const VoiceParams& params,
                          std::string_view voice_a = {}, std::string_view voice_b = {},
                          std::string_view src_lang = {}, std::string_view tgt_lang = {});

/// Apply the numeric parameters a preset carries; everything else is left as
/// it is. Voice names and language codes are the caller's job.
void apply_preset_params(const Preset& p, VoiceParams& params);

/// `*.json` files directly inside `dir`, sorted by filename. Missing dir -> {}.
std::vector<std::filesystem::path> list_preset_files(const std::filesystem::path& dir);

/// The descriptor table as JSON (what `fluxvoice --list-params --json` prints):
/// {"schema":1,"params":[{"name","label","unit","min","max","default","kind",
/// "smoothing","tier","group","help"}, ...]}
std::string descriptor_table_json();

const char* to_string(ParamKind k);
const char* to_string(ParamTier t);
const char* to_string(ParamGroup g);
const char* to_string(Smoothing s);

} // namespace fluxvoice
