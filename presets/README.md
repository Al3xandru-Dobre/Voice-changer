# presets/

Built-in voice presets (Milestone 2 / Stage 09). One JSON file per preset; the
format and loader rules are documented in `src/voice/Presets.hpp`. Voice-only
presets deliberately omit `voice_a`, `voice_b`, `translate`, `src_lang` and
`tgt_lang`, so applying **Deep** while translating `es -> en` keeps translating
`es -> en` (Milestone 3 / Stage 09 rule).

| File | Preset | Parameters |
|---|---|---|
| `neutral.json` | Neutral | every MVP parameter at its descriptor default |
| `deep.json` | Deep | `pitch_shift -4`, `intonation 0.9` |
| `bright.json` | Bright | `pitch_shift +4`, `intonation 1.1` |
| `robot.json` | Robot | `intonation 0` |
| `whisper.json` | Whisper | `whisper 1`, `gain +3` (stretch tier) |

User presets live in the directory given by `--preset-dir` (GUI: Settings ->
User preset directory) and use the same format.
