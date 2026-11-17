# BUG-027 — `prepare_s2ut_data.py` archive plumbing cannot download or extract S2UT corpora

> [Bug index](00_index.md) · Found by: [REVIEW-M3-S01-S05_2026-11-15](REVIEW-M3-S01-S05_2026-11-15.md)

| | |
|---|---|
| **Tag** | `BUG-027` |
| **Name** | *Broken archive plumbing* — `size=0` and `urls=()` disable download, pre-extraction, and CSS10/VoxPopuli extraction |
| **Reported** | 2026-11-15 |
| **Severity** | High (blocks running Milestone 3 Stage 01 corpus preparation as documented; `--delete-archives` also breaks multi-pair Europarl-ST) |
| **Status** | ☐ open |
| **Component** | `python_training/fvml/s2ut_data.py:92-212, 294-318, 781-822`, `python_training/tests/test_prepare_s2ut_data.py:68-73` |
| **Blocks** | [Milestone 3 / Stage 01](../Milestone%203/01_multilingual_data_simuleval_harness.md) external corpus downloads, [Stage 03](../Milestone%203/03_multilingual_encoder_vocoder_upgrade.md), [Stage 05](../Milestone%203/05_causal_s2ut_translator_policy_gate.md) |

## Summary

`Documents/Milestone 3/01_multilingual_data_simuleval_harness.md:59` marks `[x] Download, verify SHA256, resample all audio to 16 kHz mono float32` complete, and `prepare_s2ut_data.py`'s docstring states that it downloads and verifies archives against `datasets.lock.json`. In reality, four archive-handling defects in `python_training/fvml/s2ut_data.py` prevent the tool from downloading or extracting corpora as documented:

1. **Downloads are dead code (`size=0` everywhere).** Every `fd.Archive(...)` entry in `S2UT_DATASETS` (`s2ut_data.py:99-203` — `cvss-c-*`, `covost2-en_de`, `europarl-st-*`, `cml-tts-*`, `css10-*`, `voxpopuli-s2ut`) is declared with `size=0`. In `prepare_s2ut()` (`s2ut_data.py:800`), `if a.size > 0: path = fd.download(bctx, a)` is therefore unreachable in production, even for datasets with configured HTTPS/OpenSLR mirror URLs.
2. **Pre-extracted `raw/` directories are rejected when `urls` is non-empty.** When `a.size == 0` and `<data_root>/archives/<a.file>` is absent, `prepare_s2ut()` (`s2ut_data.py:806-808`) aborts with `PrepareError: place <file> in <archives> or extract into <dest>`. Following the second half of that message — extracting directly into `data/raw/<dir>` — still raises the exact same `PrepareError` because `if downloadable and not extracted:` runs unconditionally without checking whether `dest` is already populated.
3. **Manual archives in `data/archives/` are ignored when `urls=()`.** `css10-{es,fr,de}` (`s2ut_data.py:144`) and `voxpopuli-s2ut` (`s2ut_data.py:203`) set `urls=()`. `prepare_s2ut()` filters `downloadable = [a for a in ds.archives if a.urls]` (`line 792`), which evaluates to `[]`. Placing `css10_es.zip` or `voxpopuli_s2ut.tar.gz` into `data/archives/` skips extraction and SHA-256 verification/pinning entirely, failing later inside the builder (`transcript.txt not found`).
4. **Shared Europarl archive collides with `--delete-archives`, and Piper TTS skips checksums.** `europarl-st-en_{es,fr,de}` share `europarl-st-v1.1.tar.gz` (`s2ut_data.py:126`) and `raw_dir="europarl_st"`, but write per-dataset stamps (`europarl-st-en_es.extracted.json`). Running `prepare_s2ut_data.py --datasets paired --delete-archives` deletes `archives/europarl-st-v1.1.tar.gz` after `en_es` (`lines 819-821`) and fails on `en_fr` and `en_de` (and without `--delete-archives` re-extracts the tarball three times). Separately, `PiperCanonicalSynthesizer._get_voice` (`lines 311-316`) downloads six `.onnx`/`.onnx.json` model files via `urllib.request.urlretrieve` with no size or SHA-256 verification.

## Environment

- Windows 11 x64, `python_training/.venv` (CPython 3.12.13), working tree on 2026-11-15 (`4f6d225` + uncommitted M3 Stages 03–05).

## Steps to reproduce

```powershell
cd python_training
.\.venv\Scripts\python.exe -c "
import pathlib, tempfile
from fvml import datasets as fd, s2ut_data as sd
with tempfile.TemporaryDirectory() as tmp:
    sctx = sd.S2utContext(root=pathlib.Path(tmp) / 'data', log=lambda *_: None)
    for fn in (
        lambda: sd.prepare_s2ut(sctx, sd.S2UT_DATASETS['cml-tts-es']),
        lambda: ((sctx.root / 'raw/europarl_st/v1.1/en/es/train').mkdir(parents=True, exist_ok=True),
                 sd.prepare_s2ut(sctx, sd.S2UT_DATASETS['europarl-st-en_es'])),
        lambda: ((sctx.root / 'archives').mkdir(parents=True, exist_ok=True),
                 (sctx.root / 'archives/css10_es.zip').write_bytes(b'PK'),
                 sd.prepare_s2ut(sctx, sd.S2UT_DATASETS['css10-es'])),
    ):
        try: fn()
        except fd.PrepareError as e: print(e)
"
```

## Expected

- Datasets with public URLs (`cvss-c-*`, `covost2-en_de`, `europarl-st-*`, `cml-tts-*`) download, verify/pin SHA-256 in `datasets.lock.json`, and extract once per archive.
- Manual-download datasets (`css10-*`, `voxpopuli-s2ut`) extract and pin SHA-256 when placed in `<data_root>/archives/`, or succeed when already extracted under `<data_root>/raw/<raw_dir>`.
- Canonical Piper voice checkpoints verify SHA-256 against `datasets.lock.json`.

## Actual

```text
cml-tts-es: place cml_tts_dataset_spanish_v0.1.tar.bz2 in ...\data\archives or extract into ...\data\raw\cml_tts\es
europarl-st-en_es: place europarl-st-v1.1.tar.gz in ...\data\archives or extract into ...\data\raw\europarl_st
css10-es: transcript.txt not found under ...\data\raw\css10\es
```

## Evidence

| Check | Location | Observation |
|---|---|---|
| Archive sizes in registry | `s2ut_data.py:99-203` | All 15 `fd.Archive` entries have `size=0`; `datasets.lock.json` contains 0 Stage 01 entries |
| Download branch guard | `s2ut_data.py:800` | `if a.size > 0:` is never entered for any entry in `S2UT_DATASETS` |
| Why unit tests missed it | `test_prepare_s2ut_data.py:68-73` | `_local_archive()` replaces every archive with `fd.Archive(a.file, (p.as_uri(),), len(data), md5)`, supplying both non-empty `urls` and `size > 0` |
| Europarl stamp collision | `s2ut_data.py:791, 819-821` | Stamp is keyed by `ds.key` (`europarl-st-en_es`), while deletion unlinks shared `europarl-st-v1.1.tar.gz` |
| Unverified TTS weights | `s2ut_data.py:311-316` | `urllib.request.urlretrieve(url, tmp)` with no hash check |

## Root cause

`S2UT_DATASETS` was authored with placeholder `size=0` values while `fd._fetch` / `fd.verify` require exact byte sizes, and the unit test fixture `_local_archive` mutated the production `S2utDataset` descriptors instead of testing `S2UT_DATASETS` as declared.

## Suggested fix

1. Populate real byte sizes (and published MD5s where available) on downloadable `S2UT_DATASETS` archives, or support unknown-size streaming download + first-use SHA-256 pinning in `fd.download` / `prepare_s2ut`.
2. Iterate all `ds.archives` (not only `[a for a in ds.archives if a.urls]`): if `archives/<a.file>` exists on disk, verify/pin its SHA-256 and extract it even when `a.urls == ()`; if neither archive nor URLs are available, allow an already-populated `dest` directory rather than failing before the builder runs.
3. Key extraction stamps by `archive.file` + `dest` (not `ds.key`), and defer `--delete-archives` cleanup until all selected datasets sharing an archive have extracted it.
4. Route `PiperCanonicalSynthesizer` downloads through `datasets.lock.json` SHA-256 verification.
5. Add a test in `test_prepare_s2ut_data.py` that exercises manual `archives/css10_es.zip` extraction, pre-extracted `raw/` fallback, and shared-archive `--delete-archives` without rewriting `S2UT_DATASETS` entries via `_local_archive`.

## Workaround

Manually download every archive into `<data_root>/archives/` (keeping `css10` and `voxpopuli` manually extracted into `<data_root>/raw/css10/<lang>` and `<data_root>/nc/raw/voxpopuli`), and never pass `--delete-archives` when preparing multiple `europarl-st-*` pairs.
