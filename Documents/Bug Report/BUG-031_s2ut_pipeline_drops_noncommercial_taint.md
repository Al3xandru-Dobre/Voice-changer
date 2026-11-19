# BUG-031 — Non-commercial dataset and teacher taint is dropped across `audio_rows`, `train_s2ut.py`, and `export_step_graphs`

> [Bug index](00_index.md) · Found by: [REVIEW-M3-S01-S05_2026-11-15](REVIEW-M3-S01-S05_2026-11-15.md)

| | |
|---|---|
| **Tag** | `BUG-031` |
| **Name** | *Laundered non-commercial taint* — VoxPopuli manifests and NC-trained S2UT checkpoints lose `noncommercial: true` and export as `"MIT"` |
| **Reported** | 2026-11-15 |
| **Severity** | Medium (violates `00_master.md` Key Constraint 5 and `@ml-speech-trainer` commercial-clean guardrail) |
| **Status** | ◐ fixed in tree 2026-11-18 ([fix plan](FIX-PLAN-BUG-029-032_2026-11-18.md); GCC/Clang + pytest green, native MSVC/AppleClang legs pending) |
| **Component** | `python_training/fvml/units.py:101-129`, `python_training/extract_units.py:275-285`, `python_training/train_s2ut.py:201-232`, `python_training/fvml/s2ut_stream.py:170-200`, `python_training/gate_s2ut.py:718-728` |
| **Blocks** | [Milestone 3 / Stage 03](../Milestone%203/03_multilingual_encoder_vocoder_upgrade.md) & [Stage 05](../Milestone%203/05_causal_s2ut_translator_policy_gate.md) licence isolation; [Stage 06](../Milestone%203/06_translated_prosody_stateful_onnx_export.md) bundle export |

## Summary

Milestone 3 Key Constraint 5 (`00_master.md:204-207`) and `03_multilingual_encoder_vocoder_upgrade.md:170-178` require strict end-to-end taint propagation for non-commercial assets (`VoxPopuli` CC BY-NC 4.0 and `utter-project/mHuBERT-147` CC-BY-NC-SA-4.0) so nothing trained on them can run without `--include-noncommercial` or be exported with a clean `"MIT"` contract licence.

Three gaps in Stages 03 and 05 drop that taint:

1. **Manifest-level NC taint (`VoxPopuli`) is ignored by `units.audio_rows()`, `extract_units.py`, and `train_s2ut.py`.**
   `s2ut_data.py:768` stamps every VoxPopuli row with `"noncommercial": True` and `"license": "CC BY-NC 4.0"`. However:
   - `fvml/units.py::audio_rows()` (`lines 101-129`) does not copy `"noncommercial"` onto converted monolingual rows, and `extract_units.py` (`lines 279-285`) checks only `tok.meta["noncommercial"]` (the teacher's licence, not the corpus's). Labelling NC VoxPopuli audio with the MIT `mhubert-vp` teacher therefore succeeds **without `--include-noncommercial`** and outside any `nc/` directory, writing `units_meta.json` with `"noncommercial": false` and `"license": "MIT ..."`.
   - `train_s2ut.py::join_items()` (`lines 201-204`) checks `units.check_license` only on `unit_metas` and `lat_metas`, never inspecting `r.get("noncommercial")` (or `"NC"` in `r.get("license")`) on the paired manifest rows.
2. **`train_s2ut.py` strips `noncommercial` and `license` when saving checkpoints (`train_s2ut.py:232, 366`).**
   Unlike `train_content_encoder.py:286-288` and `train_unit_vocoder.py:661-663` (both of which preserve `license` and `noncommercial` in `ckpt["labels_meta"]`), `train_s2ut.py:232` returns only `{"vocab": vocab, "label_space": label_space}`. A translator trained with `--include-noncommercial` on `mHuBERT-147` target units or an NC-distilled student writes a checkpoint with zero record of its non-commercial provenance.
3. **`export_step_graphs()` and `gate_s2ut.py` stamp exported translator ONNX files as `"MIT"` without checking `labels_meta` (`s2ut_stream.py:170`, `translator.py:78`, `gate_s2ut.py:718-728`).**
   `S2utTranslator.from_checkpoint` ignores `ck["labels_meta"]`, `gate_s2ut.py` has no `--include-noncommercial` guard, and `export_step_graphs()` defaults to `license=tr.LICENSE` (`"MIT"`), embedding `"license": "MIT"` into `translator_enc.onnx` and `translator_dec.onnx`.

## Environment

- Windows 11 x64, `python_training/.venv` (CPython 3.12.13), 2026-11-15.

## Steps to reproduce

```powershell
cd python_training
.\.venv\Scripts\python.exe -c "
import json, tempfile
from fvml import units, translator as tr, s2ut_stream as ss
import train_s2ut as ts

vp_row = {'id': 'vp_01', 'split': 'train', 'src_lang': 'en', 'tgt_lang': 'es',
          'src_audio_path': 's.flac', 'src_duration_s': 3.0, 'src_text': 'hi',
          'tgt_audio_path': 't.flac', 'tgt_duration_s': 3.0, 'tgt_text': 'hola',
          'speaker_id': 'vp1', 'dataset': 'voxpopuli-s2ut', 'license': 'CC BY-NC 4.0',
          'noncommercial': True}
print('audio_rows keeps noncommercial:', 'noncommercial' in units.audio_rows([vp_row], 'src')[0])

with tempfile.TemporaryDirectory() as tmp:
    cfg = tr.tiny_config()
    data = ts.make_synthetic(tmp, cfg, n_per_direction=2)
    # Mark target units as non-commercial mHuBERT-147:
    mp = f\"{data['tgt_units'][0]}/units_meta.json\"
    meta = json.loads(open(mp, encoding='utf-8').read())
    meta.update(noncommercial=True, license='CC-BY-NC-SA-4.0')
    open(mp, 'w', encoding='utf-8').write(json.dumps(meta))
    args = ts.parse_args(['--manifest', data['manifest'][0], '--src-latents', data['src_latents'][0],
                          '--tgt-units', data['tgt_units'][0], '--spm-dir', data['spm_dir'],
                          '--include-noncommercial', '--silence-tail-p', '0', '--run-name', 'nc'])
    _, saved_meta = ts.join_items(args, cfg, 'train', log=lambda *_: None)
    print('train_s2ut saved labels_meta:', saved_meta)
    rec = ss.export_step_graphs(tr.S2utTranslator(cfg).eval(), tmp, exporter='torchscript')
    print('exported ONNX contract license:', rec['encoder_contract'].license)
"
```

## Expected

- `units.audio_rows` preserves `noncommercial: True`, and `extract_units.py` / `train_content_encoder.py` / `train_unit_vocoder.py` / `train_s2ut.py` refuse non-commercial manifest rows unless `--include-noncommercial` (and an `nc/` output path for extracted artefacts) is supplied.
- `train_s2ut.py` records `noncommercial: True` and `license` in `ckpt["labels_meta"]` whenever any input manifest, source-latents directory, or target-units directory is non-commercial.
- `gate_s2ut.py` and `export_step_graphs()` (and Stage 06's `export_stateful_onnx.py`) refuse NC-tainted translator checkpoints without `--include-noncommercial` and append `export_stateful_onnx.NC_NOTE` (`[trained on non-commercial units: research only, not for distribution]`) when opted in.

## Actual

```text
audio_rows keeps noncommercial: False
train_s2ut saved labels_meta: {'vocab': 24, 'label_space': 'synthetic'}
exported ONNX contract license: MIT
```

## Root cause

Stage 03 implemented NC taint propagation along the teacher -> units -> student/vocoder path, while Stage 01's manifest-row `noncommercial` flag and Stage 05's new `train_s2ut.py` / `s2ut_stream.py` checkpoint-to-ONNX path were wired only at the `join_items` input check and omitted from output metadata.

## Suggested fix

1. Preserve `"noncommercial"` in `units.audio_rows()` (`COPY_FIELDS` too), and check row-level `r.get("noncommercial")` in `extract_units.py`, `train_content_encoder.py`, `train_unit_vocoder.py`, and `train_s2ut.py::join_items()` via `units.check_license`.
2. In `train_s2ut.py::join_items()`, propagate `noncommercial` and `license` from `(*unit_metas, *lat_metas, manifest_rows)` into the returned `labels_meta` saved in checkpoints.
3. Reuse `export_stateful_onnx.noncommercial_license()` in `s2ut_stream.export_step_graphs()` and `gate_s2ut.run_gate()`, adding `--include-noncommercial` to `gate_s2ut.py` and regression coverage in `test_train_s2ut.py` / `test_multilingual.py`.

## Workaround

Keep `data/nc/` physically separate and pass explicit `--license "... [non-commercial, research only]"` whenever exporting research checkpoints.
