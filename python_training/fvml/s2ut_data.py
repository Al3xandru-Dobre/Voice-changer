"""Bidirectional En<->{Es,Fr,De} S2UT & multilingual vocoder dataset preparation
(Milestone 3 / Stage 01: prepare_s2ut_data.py).

Extends Milestone 2's dataset & checksum lock infrastructure (`fvml.datasets`)
with:
  1. Paired S2UT manifests (`$FLUXVOICE_DATA/manifests/s2ut/<key>.jsonl`):
       id, src_lang, tgt_lang, src_audio_path, src_duration_s, src_text,
       tgt_audio_path, tgt_duration_s, tgt_text, speaker_id, dataset, license
       (plus split, gender, synth_cer where applicable)
     covering all 6 directions (`es_en`, `fr_en`, `de_en`, `en_es`, `en_fr`,
     `en_de`) from CVSS-C + CoVoST 2 (`{es,fr,de}->en`) and CoVoST 2 `en_de` +
     Europarl-ST `en_{es,fr,de}` synthesized with a canonical studio Piper VITS
     voice per target language and filtered by Whisper round-trip CER <= 5%.
  2. Monolingual `{es,fr,de}` vocoder manifests (`$FLUXVOICE_DATA/manifests/`):
       CSS10 (`es`, `fr`, `de`) and curated CML-TTS (top 2 male + 2 female
       speakers per language filtered by SNR >= 30 dB and duration 2-12 s).
  3. Commercial licence guardrail (`--include-noncommercial`):
       VoxPopuli (CC BY-NC 4.0) is refused unless `--include-noncommercial` is
       explicitly passed, and is always isolated under `$FLUXVOICE_DATA/nc/` so
       default training runs cannot mix non-commercial data accidentally.
  4. Shared 6,000-token SentencePiece unigram models (`$FLUXVOICE_DATA/spm/
     spm_{en,es,fr,de}.{model,vocab}`) for Stage 05's auxiliary ASR/S2TT CTC
     supervision heads.
"""
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import json
import os
import pathlib
import tempfile

import jiwer
import numpy as np

from . import asr, audio, datasets as fd, paths

DIRECTIONS = ("es_en", "fr_en", "de_en", "en_es", "en_fr", "en_de")
S2UT_LANGS = ("en", "es", "fr", "de")
PAIRED_REQUIRED_KEYS = (
    "id", "src_lang", "tgt_lang", "src_audio_path", "src_duration_s", "src_text",
    "tgt_audio_path", "tgt_duration_s", "tgt_text", "speaker_id", "dataset", "license",
)

# Commercial-clean single-speaker Piper VITS checkpoints (rhasspy/piper-voices, MIT code).
PIPER_CANONICAL_VOICES = {
    "es": {
        "voice_id": "es_ES-davefx-medium",
        "speaker_id": "piper_es_davefx",
        "license": "CC BY 4.0 (Piper es_ES-davefx-medium)",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
    },
    "fr": {
        "voice_id": "fr_FR-siwis-medium",
        "speaker_id": "piper_fr_siwis",
        "license": "CC BY 4.0 (SIWIS French Speech Synthesis Database / Piper)",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
    },
    "de": {
        "voice_id": "de_DE-thorsten-high",
        "speaker_id": "piper_de_thorsten",
        "license": "CC0-1.0 (Thorsten-Voice / Piper)",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json",
    },
}

MAX_SYNTH_CER = 0.05
CML_MIN_SNR_DB = 30.0
CML_MIN_DUR_S = 2.0
CML_MAX_DUR_S = 12.0
CML_SPEAKERS_PER_GENDER = 2
SPM_VOCAB_SIZE = 6000


@dataclass(frozen=True)
class S2utDataset:
    key: str
    title: str
    group: str                  # "paired" | "vocoder" | "nc"
    archives: tuple             # tuple[fd.Archive, ...]
    builder: str                # build_* function name below
    raw_dir: str
    licence: str
    src_lang: str = ""
    tgt_lang: str = ""
    noncommercial: bool = False


def _cvss_c(src_lang):
    pair = f"{src_lang}_en"
    return S2utDataset(
        key=f"cvss-c-{pair}",
        title=f"CVSS-C {src_lang}->en (joined with CoVoST 2 / Common Voice {src_lang})",
        group="paired",
        archives=(
            fd.Archive(
                f"cvss_c_{src_lang}_en_v1.0.tar.gz",
                (f"https://commondatastorage.googleapis.com/cvss/cvss_c_{src_lang}_en_v1.0.tar.gz",),
                size=0,
            ),
            fd.Archive(
                f"covost_v2.{src_lang}_en.tsv.tar.gz",
                (f"https://dl.fbaipublicfiles.com/covost/covost_v2.{src_lang}_en.tsv.tar.gz",),
                size=0,
            ),
        ),
        builder="build_cvss_c",
        raw_dir=f"cvss_c/{pair}",
        licence="CC BY 4.0",
        src_lang=src_lang,
        tgt_lang="en",
    )


def _europarl(tgt_lang):
    pair = f"en_{tgt_lang}"
    return S2utDataset(
        key=f"europarl-st-{pair}",
        title=f"Europarl-ST en->{tgt_lang} + canonical Piper TTS target audio",
        group="paired",
        archives=(
            fd.Archive(
                "europarl-st-v1.1.tar.gz",
                ("https://www.mllp.upv.es/europarl-st/v1.1.tar.gz",),
                size=0,
            ),
        ),
        builder="build_europarl_st",
        raw_dir="europarl_st",
        licence="CC BY 4.0",
        src_lang="en",
        tgt_lang=tgt_lang,
    )


def _css10(lang, archive_name):
    return S2utDataset(
        key=f"css10-{lang}",
        title=f"CSS10 studio single-speaker ({lang})",
        group="vocoder",
        archives=(fd.Archive(archive_name, (), size=0),),
        builder="build_css10",
        raw_dir=f"css10/{lang}",
        licence="Apache-2.0 / CC0-1.0 (LibriVox)",
        src_lang=lang,
        tgt_lang=lang,
    )


def _cml_tts(lang, archive_file):
    return S2utDataset(
        key=f"cml-tts-{lang}",
        title=f"CML-TTS OpenSLR 146 ({lang}, curated 2M+2F SNR>=30dB)",
        group="vocoder",
        archives=(fd.Archive(archive_file, fd.openslr(146, archive_file), size=0),),
        builder="build_cml_tts",
        raw_dir=f"cml_tts/{lang}",
        licence="CC BY 4.0",
        src_lang=lang,
        tgt_lang=lang,
    )


S2UT_DATASETS = {
    d.key: d
    for d in (
        _cvss_c("es"),
        _cvss_c("fr"),
        _cvss_c("de"),
        S2utDataset(
            key="covost2-en_de",
            title="CoVoST 2 en->de + canonical Piper TTS target audio",
            group="paired",
            archives=(
                fd.Archive(
                    "covost_v2.en_de.tsv.tar.gz",
                    ("https://dl.fbaipublicfiles.com/covost/covost_v2.en_de.tsv.tar.gz",),
                    size=0,
                ),
            ),
            builder="build_covost2_en_tgt",
            raw_dir="covost2/en_de",
            licence="CC0-1.0",
            src_lang="en",
            tgt_lang="de",
        ),
        _europarl("es"),
        _europarl("fr"),
        _europarl("de"),
        _css10("es", "css10_es.zip"),
        _css10("fr", "css10_fr.zip"),
        _css10("de", "css10_de.zip"),
        _cml_tts("es", "cml_tts_dataset_spanish_v0.1.tar.bz2"),
        _cml_tts("fr", "cml_tts_dataset_french_v0.1.tar.bz2"),
        _cml_tts("de", "cml_tts_dataset_german_v0.1.tar.bz2"),
        S2utDataset(
            key="voxpopuli-s2ut",
            title="VoxPopuli En<->{Es,Fr,De} (NON-COMMERCIAL CC BY-NC 4.0 — isolated in data/nc/)",
            group="nc",
            archives=(fd.Archive("voxpopuli_s2ut.tar.gz", (), size=0),),
            builder="build_voxpopuli_nc",
            raw_dir="voxpopuli",
            licence="CC BY-NC 4.0",
            src_lang="multi",
            tgt_lang="multi",
            noncommercial=True,
        ),
    )
}
S2UT_GROUPS = {
    "paired": ("paired",),
    "vocoder": ("vocoder",),
    "all": ("paired", "vocoder"),
}


def select_s2ut(spec, include_noncommercial=False):
    """Resolve group or comma-separated dataset keys, enforcing the commercial
    licence guardrail."""
    if spec in S2UT_GROUPS:
        groups = S2UT_GROUPS[spec] + (("nc",) if spec == "all" and include_noncommercial else ())
        return [d for d in S2UT_DATASETS.values() if d.group in groups]
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in S2UT_DATASETS]
    if unknown:
        raise fd.PrepareError(f"unknown S2UT dataset(s): {', '.join(unknown)} (see --list)")
    chosen = [S2UT_DATASETS[k] for k in keys]
    nc = [d.key for d in chosen if d.noncommercial]
    if nc and not include_noncommercial:
        raise fd.PrepareError(
            f"dataset(s) {', '.join(nc)} are CC BY-NC 4.0 (non-commercial); pass "
            f"--include-noncommercial to prepare them into data/nc/"
        )
    return chosen


@dataclass
class S2utContext:
    root: pathlib.Path
    common_voice_root: pathlib.Path | None = None
    include_noncommercial: bool = False
    max_synth_cer: float = MAX_SYNTH_CER
    cml_min_snr_db: float = CML_MIN_SNR_DB
    jobs: int = max(1, (os.cpu_count() or 2) // 2)
    delete_archives: bool = False
    force_manifests: bool = False
    device: str = "cuda"
    batch_size: int = 8
    lock_file: pathlib.Path = fd.LOCK_FILE
    synthesizer: object = None      # callable(text, lang, dst_path) -> duration_s
    cer_transcriber: object = None  # callable(waves, language) -> [transcript]
    log: object = print
    _base_ctx: fd.Context = field(default=None, repr=False)
    _nc_ctx: fd.Context = field(default=None, repr=False)

    def base_ctx(self, noncommercial=False):
        if noncommercial:
            if not self.include_noncommercial:
                raise fd.PrepareError("non-commercial context requested without --include-noncommercial")
            if self._nc_ctx is None:
                nc_root = self.root / "nc"
                nc_root.mkdir(parents=True, exist_ok=True)
                self._nc_ctx = fd.Context(
                    root=nc_root, jobs=self.jobs, delete_archives=self.delete_archives,
                    force_manifests=self.force_manifests, lock_file=self.lock_file, log=self.log,
                )
            return self._nc_ctx
        if self._base_ctx is None:
            self._base_ctx = fd.Context(
                root=self.root, jobs=self.jobs, delete_archives=self.delete_archives,
                force_manifests=self.force_manifests, lock_file=self.lock_file, log=self.log,
            )
        return self._base_ctx


# -- Acoustic SNR & canonical TTS helpers -------------------------------------
def estimate_snr_db(wave, frame_ms=20.0):
    """Frame-energy percentile SNR estimate (95th-pct speech RMS over 10th-pct
    noise-floor RMS) in dB."""
    x = np.asarray(wave, dtype=np.float32)
    frame_len = max(1, int(round(audio.SAMPLE_RATE * frame_ms / 1000.0)))
    if len(x) < 2 * frame_len:
        return 0.0
    frames = x[: (len(x) // frame_len) * frame_len].reshape(-1, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-16)
    p95 = float(np.percentile(rms, 95))
    p10 = max(float(np.percentile(rms, 10)), 1e-7)
    return float(min(80.0, max(0.0, 20.0 * np.log10(max(p95, 1e-7) / p10))))


class PiperCanonicalSynthesizer:
    """Lazily downloads and runs the pinned single-speaker Piper VITS ONNX model
    per target language (`es`, `fr`, `de`), resampling to 16 kHz mono 24-bit FLAC."""

    def __init__(self, cache_root):
        self.cache_root = pathlib.Path(cache_root)
        self._voices = {}

    def _get_voice(self, lang):
        if lang not in self._voices:
            import urllib.request
            from piper.voice import PiperVoice
            meta = PIPER_CANONICAL_VOICES[lang]
            vdir = self.cache_root / "piper" / lang
            vdir.mkdir(parents=True, exist_ok=True)
            onnx_path = vdir / f"{meta['voice_id']}.onnx"
            json_path = vdir / f"{meta['voice_id']}.onnx.json"
            for url, dst in ((meta["onnx_url"], onnx_path), (meta["json_url"], json_path)):
                if not dst.is_file():
                    tmp = dst.with_name(dst.name + ".part")
                    urllib.request.urlretrieve(url, tmp)
                    tmp.replace(dst)
            self._voices[lang] = PiperVoice.load(str(onnx_path), config_path=str(json_path))
        return self._voices[lang]

    def __call__(self, text, lang, dst_path):
        dst_path = pathlib.Path(dst_path)
        if dst_path.is_file():
            return audio.info(dst_path)[0]
        voice = self._get_voice(lang)
        sr_native = int(voice.config.sample_rate)
        chunks = []
        for chunk in voice.synthesize(text):
            arr = getattr(chunk, "audio_float_array", None)
            if arr is None:
                raw = getattr(chunk, "audio_int16_bytes", b"")
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            chunks.append(np.asarray(arr, dtype=np.float32))
        wav = np.concatenate(chunks) if chunks else np.zeros((audio.SAMPLE_RATE // 10,), np.float32)
        wav_16k = audio.resample(wav, sr_native, audio.SAMPLE_RATE)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_path.with_name(dst_path.stem + ".tmp.flac")
        audio.save(tmp, wav_16k, subtype="PCM_24")
        tmp.replace(dst_path)
        return audio.info(dst_path)[0]


def paired_row(bctx, uid, src_lang, tgt_lang, src_path, src_dur, src_text,
               tgt_path, tgt_dur, tgt_text, speaker_id, dataset, licence,
               split="train", gender=None, synth_cer=None):
    r = {
        "id": uid,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_audio_path": bctx.rel(src_path),
        "src_duration_s": round(float(src_dur), 3),
        "src_text": src_text,
        "tgt_audio_path": bctx.rel(tgt_path),
        "tgt_duration_s": round(float(tgt_dur), 3),
        "tgt_text": tgt_text,
        "speaker_id": speaker_id,
        "dataset": dataset,
        "license": licence,
        "split": split,
    }
    if gender is not None:
        r["gender"] = gender
    if synth_cer is not None:
        r["synth_cer"] = round(float(synth_cer), 4)
    return r


def verify_synthesis_roundtrip_cer(sctx, rows_with_paths, tgt_lang):
    """Run Whisper ASR (`language=tgt_lang`) on synthesized target clips and
    reject any clip whose round-trip CER against `tgt_text` exceeds `max_synth_cer`
    (default 5%). Returns `(accepted_rows, report_dict)`."""
    if not rows_with_paths:
        return [], {"synthesized": 0, "accepted": 0, "rejected_cer": 0, "mean_cer": 0.0}

    waves = [audio.load(tgt_abs) for _, tgt_abs in rows_with_paths]
    if sctx.cer_transcriber is not None:
        hyps = sctx.cer_transcriber(waves, language=tgt_lang)
        norm_fn = lambda t: asr.normalize_text(t, language=tgt_lang)  # noqa: E731
    else:
        w = asr.Whisper(sctx.device, sctx.batch_size)
        try:
            hyps = w.transcribe(waves, language=tgt_lang, silence_dbfs=-55.0)
            norm_fn = lambda t: w.normalize(t, language=tgt_lang)  # noqa: E731
            refs_n = [norm_fn(r["tgt_text"]) for r, _ in rows_with_paths]
            hyps_n = [norm_fn(h) for h in hyps]
        finally:
            w.close()
        norm_fn = None

    if norm_fn is not None:
        refs_n = [norm_fn(r["tgt_text"]) for r, _ in rows_with_paths]
        hyps_n = [norm_fn(h) for h in hyps]

    accepted = []
    cers = []
    rejected = 0
    for (row, _), ref_n, hyp_n in zip(rows_with_paths, refs_n, hyps_n):
        if not ref_n.strip():
            rejected += 1
            continue
        cer = float(jiwer.cer(ref_n, hyp_n)) if hyp_n.strip() else 1.0
        cers.append(cer)
        if cer <= sctx.max_synth_cer:
            row["synth_cer"] = round(cer, 4)
            accepted.append(row)
        else:
            rejected += 1
    report = {
        "synthesized": len(rows_with_paths),
        "accepted": len(accepted),
        "rejected_cer": rejected,
        "max_synth_cer": sctx.max_synth_cer,
        "mean_cer": round(float(np.mean(cers)), 4) if cers else 0.0,
    }
    return accepted, report


# -- Dataset builders ---------------------------------------------------------
def _find_cv_clip(cv_root, lang, clip_name):
    stem = pathlib.Path(clip_name).stem
    for base in (cv_root / lang / "clips", cv_root / "clips", cv_root / lang):
        for ext in ("", ".mp3", ".wav", ".flac"):
            cand = base / (clip_name if not ext else f"{stem}{ext}")
            if cand.is_file():
                return cand
    return None


def _read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def build_cvss_c(sctx, ds):
    """Join CVSS-C (`{es,fr,de}->en`) target speech with Common Voice source
    clips and CoVoST 2 source/target transcripts."""
    bctx = sctx.base_ctx()
    raw = bctx.raw / ds.raw_dir
    cv_root = sctx.common_voice_root or (bctx.raw / "common_voice")

    covost_meta = {}
    for tsv in sorted(raw.rglob(f"covost_v2.{ds.src_lang}_en*.tsv")):
        for r in _read_tsv(tsv):
            clip = pathlib.Path(r["path"]).stem
            covost_meta[clip] = r

    # Also read Common Voice validated.tsv if present for speaker_id & gender.
    cv_meta = {}
    for vtsv in sorted(cv_root.rglob("*.tsv")) if cv_root.is_dir() else []:
        if vtsv.name in ("validated.tsv", "train.tsv", "dev.tsv", "test.tsv"):
            for r in _read_tsv(vtsv):
                if "path" in r:
                    cv_meta[pathlib.Path(r["path"]).stem] = r

    src_out = bctx.dir("16k", f"cvss_src_{ds.src_lang}")
    tgt_out = bctx.dir("16k", f"cvss_tgt_{ds.src_lang}_en")
    items = []
    for split in ("train", "dev", "test"):
        split_tsvs = sorted(raw.rglob(f"{split}.tsv"))
        for stsv in split_tsvs:
            with open(stsv, encoding="utf-8", newline="") as f:
                # CVSS-C TSVs are either headerless `clip\ttgt_text` or have headers.
                lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
            if not lines:
                continue
            first_cols = lines[0].split("\t")
            start = 1 if first_cols[0] in ("path", "clip", "id") else 0
            for ln in lines[start:]:
                parts = ln.split("\t")
                clip_ref = parts[0]
                tgt_text = parts[1] if len(parts) > 1 else ""
                stem = pathlib.Path(clip_ref).stem
                tgt_wav = None
                for cand in (stsv.parent / split / f"{clip_ref}.wav",
                             stsv.parent / split / clip_ref,
                             stsv.parent / f"{stem}.wav"):
                    if cand.is_file():
                        tgt_wav = cand
                        break
                src_clip = _find_cv_clip(cv_root, ds.src_lang, clip_ref)
                if tgt_wav is None or src_clip is None:
                    continue
                meta = covost_meta.get(stem, {})
                cvm = cv_meta.get(stem, {})
                src_text = meta.get("sentence") or cvm.get("sentence") or ""
                tgt_text = tgt_text or meta.get("translation", "")
                spk = cvm.get("client_id") or meta.get("client_id") or f"{ds.src_lang}_spk_{stem[:8]}"
                gender_raw = (cvm.get("gender") or meta.get("gender") or "").lower()
                gender = "F" if gender_raw.startswith("f") else ("M" if gender_raw.startswith("m") else None)
                items.append((
                    f"cvss_{ds.src_lang}_en_{stem}", split, src_clip, src_out / f"{stem}.flac",
                    src_text, tgt_wav, tgt_out / f"{stem}.flac", tgt_text, spk, gender,
                ))

    if not items:
        raise fd.PrepareError(
            f"{ds.key}: no joined (source, target) pairs found under {raw} and {cv_root}"
        )
    src_durs = fd.resample_all(bctx, [(s, d) for _, _, s, d, _, _, _, _, _, _ in items])
    tgt_durs = fd.resample_all(bctx, [(s, d) for _, _, _, _, _, s, d, _, _, _ in items])
    return [
        paired_row(
            bctx, uid, ds.src_lang, "en", s_dst, sd, s_txt, t_dst, td, t_txt,
            spk, ds.key, ds.licence, split=split, gender=gender,
        )
        for (uid, split, _, s_dst, s_txt, _, t_dst, t_txt, spk, gender), sd, td
        in zip(items, src_durs, tgt_durs)
    ]


def _synthesize_reverse_pairs(sctx, ds, raw_items):
    """Shared reverse-CVSS pipeline for CoVoST 2 `en_de` and Europarl-ST
    `en_{es,fr,de}`: resample source English audio, synthesize canonical target
    WAVs via Piper VITS, and filter by Whisper round-trip CER <= 5%."""
    bctx = sctx.base_ctx()
    synth = sctx.synthesizer or PiperCanonicalSynthesizer(paths.cache_dir())
    src_out = bctx.dir("16k", f"{ds.key}_src")
    tgt_out = bctx.dir("16k", f"{ds.key}_tgt")

    src_pairs = [(it["src_file"], src_out / f"{it['id']}.flac") for it in raw_items]
    src_durs = fd.resample_all(bctx, src_pairs)

    rows_with_paths = []
    for it, (_, s_dst), s_dur in zip(raw_items, src_pairs, src_durs):
        t_dst = tgt_out / f"{it['id']}.flac"
        t_dur = synth(it["tgt_text"], ds.tgt_lang, t_dst)
        row = paired_row(
            bctx, it["id"], ds.src_lang, ds.tgt_lang, s_dst, s_dur, it["src_text"],
            t_dst, t_dur, it["tgt_text"], it["speaker_id"], ds.key, ds.licence,
            split=it["split"], gender=it.get("gender"),
        )
        rows_with_paths.append((row, t_dst))

    accepted, report = verify_synthesis_roundtrip_cer(sctx, rows_with_paths, ds.tgt_lang)
    (bctx.dir(".state") / f"{ds.key}.synth_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    sctx.log(
        f"  [{ds.key}] synthesis CER gate (<= {sctx.max_synth_cer * 100:.1f}%): "
        f"{report['accepted']}/{report['synthesized']} kept, "
        f"{report['rejected_cer']} rejected (mean CER {report['mean_cer'] * 100:.2f}%)"
    )
    return accepted


def build_covost2_en_tgt(sctx, ds):
    bctx = sctx.base_ctx()
    raw = bctx.raw / ds.raw_dir
    cv_root = sctx.common_voice_root or (bctx.raw / "common_voice")
    raw_items = []
    for split in ("train", "dev", "test"):
        for tsv in sorted(raw.rglob(f"*{split}*.tsv")):
            for r in _read_tsv(tsv):
                clip_ref = r.get("path", "")
                stem = pathlib.Path(clip_ref).stem
                src_file = _find_cv_clip(cv_root, "en", clip_ref)
                src_text = r.get("sentence", "").strip()
                tgt_text = r.get("translation", "").strip()
                if src_file is None or not src_text or not tgt_text:
                    continue
                gender_raw = (r.get("gender") or "").lower()
                gender = "F" if gender_raw.startswith("f") else ("M" if gender_raw.startswith("m") else None)
                raw_items.append({
                    "id": f"covost2_en_{ds.tgt_lang}_{stem}",
                    "split": split,
                    "src_file": src_file,
                    "src_text": src_text,
                    "tgt_text": tgt_text,
                    "speaker_id": r.get("client_id") or f"en_spk_{stem[:8]}",
                    "gender": gender,
                })
    if not raw_items:
        raise fd.PrepareError(f"{ds.key}: no CoVoST 2 items found under {raw} and {cv_root}")
    return _synthesize_reverse_pairs(sctx, ds, raw_items)


def build_europarl_st(sctx, ds):
    """Europarl-ST v1.1 `en -> {es,fr,de}` builder."""
    bctx = sctx.base_ctx()
    pair_root = bctx.raw / ds.raw_dir / "v1.1" / "en" / ds.tgt_lang
    if not pair_root.is_dir():
        pair_root = bctx.raw / ds.raw_dir / "en" / ds.tgt_lang
    raw_items = []
    for split in ("train", "dev", "test"):
        sdir = pair_root / split
        seg_lst = sdir / "segments.lst"
        src_txt_file = sdir / "segments.en"
        tgt_txt_file = sdir / f"segments.{ds.tgt_lang}"
        if not (seg_lst.is_file() and src_txt_file.is_file() and tgt_txt_file.is_file()):
            continue
        segs = seg_lst.read_text(encoding="utf-8").splitlines()
        srcs = src_txt_file.read_text(encoding="utf-8").splitlines()
        tgts = tgt_txt_file.read_text(encoding="utf-8").splitlines()
        for idx, (seg_line, s_txt, t_txt) in enumerate(zip(segs, srcs, tgts)):
            parts = seg_line.split()
            wav_id = parts[0]
            # Either pre-segmented `<wav_id>.wav` or `<wav_id>_<idx>.wav` under speeches/ or wavs/
            src_wav = None
            for cand in (
                sdir / "wavs" / f"{wav_id}_{idx:04d}.wav",
                sdir / "wavs" / f"{wav_id}.wav",
                pair_root.parent / "speeches" / f"{wav_id}.wav",
            ):
                if cand.is_file():
                    src_wav = cand
                    break
            if src_wav is None or not s_txt.strip() or not t_txt.strip():
                continue
            spk = parts[1] if len(parts) > 1 else wav_id.split("_")[0]
            raw_items.append({
                "id": f"europarl_en_{ds.tgt_lang}_{split}_{idx:05d}",
                "split": split,
                "src_file": src_wav,
                "src_text": s_txt.strip(),
                "tgt_text": t_txt.strip(),
                "speaker_id": f"ep_{spk}",
                "gender": None,
            })
    if not raw_items:
        raise fd.PrepareError(f"{ds.key}: no Europarl-ST segments found under {pair_root}")
    return _synthesize_reverse_pairs(sctx, ds, raw_items)


def build_css10(sctx, ds):
    """CSS10 studio single-speaker dataset (`es`, `fr`, `de`) -> M2-compatible
    vocoder manifest (`utt_id`, `path`, `duration`, `speaker`, `dataset`,
    `subset`, `lang`, `text`, `license`)."""
    bctx = sctx.base_ctx()
    raw = bctx.raw / ds.raw_dir
    trans_files = sorted(raw.rglob("transcript.txt"))
    if not trans_files:
        raise fd.PrepareError(f"{ds.key}: transcript.txt not found under {raw}")
    tfile = trans_files[0]
    base = tfile.parent
    out = bctx.dir("16k", ds.key)
    entries = []
    for line in tfile.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        rel_wav = parts[0].strip()
        text = (parts[2] if len(parts) > 2 else parts[1]).strip()
        src_wav = base / rel_wav
        if src_wav.is_file():
            utt_id = f"css10_{ds.src_lang}_{pathlib.PurePosixPath(rel_wav).stem}"
            entries.append((utt_id, src_wav, out / f"{utt_id}.flac", text))
    if not entries:
        raise fd.PrepareError(f"{ds.key}: no audio files referenced in {tfile}")
    durs = fd.resample_all(bctx, [(s, d) for _, s, d, _ in entries])
    # Hold out the final 40 clips deterministically as `test` for targets.tsv / resynthesis.
    n_total = len(entries)
    n_test = min(40, max(1, n_total // 20)) if n_total >= 4 else 0
    rows = []
    for idx, ((utt_id, _, dst, text), dur) in enumerate(zip(entries, durs)):
        subset = "test" if idx >= n_total - n_test else "train"
        rows.append(
            fd.row(
                bctx, utt_id, dst, dur, f"css10_{ds.src_lang}", ds.key, subset,
                lang=ds.src_lang, text=text, license=ds.licence,
            )
        )
    return rows


def build_cml_tts(sctx, ds):
    """Curated CML-TTS builder: filters clips to 2-12 s and SNR >= 30 dB, then
    deterministically selects top 2 male + 2 female speakers by clean hours."""
    bctx = sctx.base_ctx()
    raw = bctx.raw / ds.raw_dir
    genders = {}
    for spk_meta in sorted(raw.rglob("*speaker*.*")) + sorted(raw.rglob("META*")):
        if spk_meta.is_file():
            for line in spk_meta.read_text(encoding="utf-8", errors="replace").splitlines():
                sep = "|" if "|" in line else "\t"
                cols = [c.strip() for c in line.split(sep)]
                if len(cols) >= 2 and cols[1].upper() in ("M", "F", "MALE", "FEMALE"):
                    genders[cols[0]] = "F" if cols[1].upper().startswith("F") else "M"

    candidates_by_spk = defaultdict(list)
    for split in ("train", "dev", "test"):
        for csv_path in sorted(raw.rglob(f"{split}.csv")):
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="|")
                for r in reader:
                    rel_wav = r.get("wav_filename") or r.get("path") or ""
                    src_wav = csv_path.parent / rel_wav
                    if not src_wav.is_file():
                        src_wav = raw / rel_wav
                    if not src_wav.is_file():
                        continue
                    dur, _, _ = audio.info(src_wav)
                    if not (CML_MIN_DUR_S <= dur <= CML_MAX_DUR_S):
                        continue
                    wav = audio.load(src_wav)
                    snr = estimate_snr_db(wav)
                    if snr < sctx.cml_min_snr_db:
                        continue
                    spk_raw = str(r.get("client_id") or r.get("speaker_id") or pathlib.PurePosixPath(rel_wav).parts[0])
                    g_raw = (r.get("gender") or genders.get(spk_raw) or "").upper()
                    gender = "F" if g_raw.startswith("F") else ("M" if g_raw.startswith("M") else "?")
                    genders[spk_raw] = gender
                    candidates_by_spk[spk_raw].append((split, src_wav, dur, snr, (r.get("transcript") or "").strip()))

    if not candidates_by_spk:
        raise fd.PrepareError(
            f"{ds.key}: no clips met 2-12 s and SNR >= {sctx.cml_min_snr_db} dB under {raw}"
        )

    def _rank_speakers(target_gender):
        pool = [
            (sum(c[2] for c in clips), float(np.mean([c[3] for c in clips])), spk)
            for spk, clips in candidates_by_spk.items()
            if genders.get(spk) == target_gender
        ]
        pool.sort(key=lambda t: (-round(t[0], 3), -round(t[1], 2), t[2]))
        return [spk for _, _, spk in pool[:CML_SPEAKERS_PER_GENDER]]

    chosen = _rank_speakers("F") + _rank_speakers("M")
    if not chosen:
        raise fd.PrepareError(f"{ds.key}: no gendered speakers found in {raw}")

    out = bctx.dir("16k", ds.key)
    selected_items = []
    for spk in chosen:
        for split, src_wav, _, snr, text in candidates_by_spk[spk]:
            utt_id = f"cml_{ds.src_lang}_{spk}_{src_wav.stem}"
            selected_items.append((utt_id, spk, genders[spk], split, src_wav,
                                   out / spk / f"{src_wav.stem}.flac", snr, text))

    durs = fd.resample_all(bctx, [(s, d) for _, _, _, _, s, d, _, _ in selected_items])
    return [
        fd.row(
            bctx, utt_id, dst, dur, f"cml_{ds.src_lang}_{spk}", ds.key, split,
            gender=gender, lang=ds.src_lang, snr_db=round(snr, 2), text=text, license=ds.licence,
        )
        for (utt_id, spk, gender, split, _, dst, snr, text), dur in zip(selected_items, durs)
    ]


def build_voxpopuli_nc(sctx, ds):
    """Non-commercial VoxPopuli builder — strictly isolated inside `$FLUXVOICE_DATA/nc/`."""
    nc_ctx = sctx.base_ctx(noncommercial=True)
    raw = nc_ctx.raw / ds.raw_dir
    out = nc_ctx.dir("16k", "voxpopuli")
    items = []
    for tsv in sorted(raw.rglob("*.tsv")):
        for r in _read_tsv(tsv):
            uid = r.get("id") or pathlib.Path(r.get("src_audio", "x")).stem
            src_lang = r.get("src_lang", "en")
            tgt_lang = r.get("tgt_lang", "de")
            src_f = tsv.parent / r.get("src_audio", "")
            tgt_f = tsv.parent / r.get("tgt_audio", "")
            if not (src_f.is_file() and tgt_f.is_file()):
                continue
            items.append((
                f"vp_{uid}", r.get("split", "train"), src_lang, tgt_lang,
                src_f, out / "src" / f"{uid}.flac", r.get("src_text", ""),
                tgt_f, out / "tgt" / f"{uid}.flac", r.get("tgt_text", ""),
                r.get("speaker_id", f"vp_{src_lang}"),
            ))
    if not items:
        raise fd.PrepareError(f"{ds.key}: no VoxPopuli paired clips found under {raw}")
    s_durs = fd.resample_all(nc_ctx, [(s, d) for _, _, _, _, s, d, _, _, _, _, _ in items])
    t_durs = fd.resample_all(nc_ctx, [(s, d) for _, _, _, _, _, _, _, s, d, _, _ in items])
    rows = []
    for (uid, split, sl, tl, _, sd_p, st, _, td_p, tt, spk), sd, td in zip(items, s_durs, t_durs):
        r = paired_row(
            nc_ctx, uid, sl, tl, sd_p, sd, st, td_p, td, tt, spk, ds.key, ds.licence, split=split
        )
        r["noncommercial"] = True
        rows.append(r)
    return rows


# -- Driver & SentencePiece training ------------------------------------------
def manifest_path_for(sctx, ds):
    bctx = sctx.base_ctx(noncommercial=ds.noncommercial)
    if ds.group in ("paired", "nc"):
        return bctx.dir("manifests", "s2ut") / f"{ds.key}.jsonl"
    return bctx.dir("manifests") / f"{ds.key}.jsonl"


def prepare_s2ut(sctx, ds):
    """Download (when archive URLs are configured), verify SHA256/MD5, extract,
    resample to 16 kHz mono, and write the manifest for `ds`."""
    if ds.noncommercial and not sctx.include_noncommercial:
        raise fd.PrepareError(
            f"{ds.key} is CC BY-NC 4.0; pass --include-noncommercial to prepare into data/nc/"
        )
    bctx = sctx.base_ctx(noncommercial=ds.noncommercial)
    sctx.log(f"[{ds.key}] {ds.title}")
    state = bctx.dir(".state")
    stamp = state / f"{ds.key}.extracted.json"
    downloadable = [a for a in ds.archives if a.urls]
    pins = {a.file: bctx.lock().get(a.file) for a in downloadable}
    extracted = downloadable and stamp.is_file() and json.loads(stamp.read_text(encoding="utf-8")) == {
        "archives": pins
    }
    if downloadable and not extracted:
        dest = bctx.raw / ds.raw_dir if ds.raw_dir else bctx.raw
        for a in downloadable:
            if a.size > 0:
                path = fd.download(bctx, a)
                fd.verify(bctx, a, path)
            else:
                path = bctx.dir("archives") / a.file
                if not path.is_file():
                    raise fd.PrepareError(
                        f"{ds.key}: place {a.file} in {bctx.dir('archives')} or extract into {dest}"
                    )
                _, sha = fd.file_hashes(path)
                pinned = bctx.lock().get(a.file)
                if pinned and pinned != sha:
                    raise fd.PrepareError(f"{a.file}: SHA256 {sha} != pinned {pinned}")
                if not pinned:
                    bctx.pin(a.file, sha)
            sctx.log(f"  extracting {a.file}")
            fd.extract(path, dest)
        pins = {a.file: bctx.lock().get(a.file) for a in downloadable}
        stamp.write_text(json.dumps({"archives": pins}), encoding="utf-8")
        if bctx.delete_archives:
            for a in downloadable:
                (bctx.root / "archives" / a.file).unlink(missing_ok=True)

    manifest = manifest_path_for(sctx, ds)
    mstamp = state / f"{ds.key}.manifest.json"
    if manifest.is_file() and mstamp.is_file() and not bctx.force_manifests:
        sctx.log(f"  manifest up to date: {bctx.rel(manifest)}")
        return manifest

    rows = globals()[ds.builder](sctx, ds)
    if not rows:
        raise fd.PrepareError(f"{ds.key}: builder produced 0 rows")
    sort_key = (lambda r: (r.get("split", ""), r["id"])) if ds.group in ("paired", "nc") \
        else (lambda r: (r["subset"], r["utt_id"]))
    fd.write_jsonl(manifest, sorted(rows, key=sort_key))
    mstamp.write_text(json.dumps({"rows": len(rows)}), encoding="utf-8")
    dur_key = "src_duration_s" if ds.group in ("paired", "nc") else "duration"
    hours = sum(r[dur_key] for r in rows) / 3600.0
    sctx.log(f"  {len(rows)} rows, {hours:.2f} h -> {bctx.rel(manifest)}")
    return manifest


def train_spm_models(sctx, vocab_size=SPM_VOCAB_SIZE, langs=S2UT_LANGS):
    """Fit and save shared SentencePiece unigram models (`spm_{en,es,fr,de}.model`)
    under `$FLUXVOICE_DATA/spm/` from commercial-clean training manifests."""
    import sentencepiece as spm

    bctx = sctx.base_ctx(noncommercial=False)
    spm_dir = bctx.dir("spm")
    texts_by_lang = defaultdict(list)

    # Paired commercial S2UT manifests (never reads data/nc/!).
    paired_dir = bctx.root / "manifests" / "s2ut"
    if paired_dir.is_dir():
        for mf in sorted(paired_dir.glob("*.jsonl")):
            for r in fd.read_jsonl(mf):
                if r.get("split", "train") == "test":
                    continue
                if r.get("src_text"):
                    texts_by_lang[r["src_lang"]].append(
                        asr.normalize_multilingual(r["src_text"])
                    )
                if r.get("tgt_text"):
                    texts_by_lang[r["tgt_lang"]].append(
                        asr.normalize_multilingual(r["tgt_text"])
                    )

    # Monolingual vocoder / LibriSpeech manifests for additional clean coverage.
    mono_dir = bctx.root / "manifests"
    if mono_dir.is_dir():
        for mf in sorted(mono_dir.glob("*.jsonl")):
            for r in fd.read_jsonl(mf):
                if r.get("subset", "").endswith("test"):
                    continue
                txt = r.get("text")
                lang = r.get("lang") or ("en" if r.get("dataset") in ("librispeech", "ljspeech", "hifitts") else None)
                if txt and lang in langs:
                    texts_by_lang[lang].append(asr.normalize_multilingual(txt))

    written = {}
    for lang in langs:
        lines = [ln for ln in texts_by_lang.get(lang, []) if ln]
        if not lines:
            continue
        prefix = spm_dir / f"spm_{lang}"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tf:
            tf.write("\n".join(lines) + "\n")
            corpus_file = pathlib.Path(tf.name)
        try:
            spm.SentencePieceTrainer.train(
                input=str(corpus_file),
                model_prefix=str(prefix),
                vocab_size=vocab_size,
                model_type="unigram",
                character_coverage=1.0,
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=3,
                hard_vocab_limit=False,
                train_extremely_large_corpus=False,
            )
        finally:
            corpus_file.unlink(missing_ok=True)
        written[lang] = prefix.with_suffix(".model")
        sctx.log(f"  [spm] trained {bctx.rel(written[lang])} on {len(lines)} lines")
    return written


def summarize_s2ut(sctx):
    """Summarize paired S2UT manifests into `manifests/s2ut/summary.json`."""
    bctx = sctx.base_ctx(noncommercial=False)
    paired_dir = bctx.root / "manifests" / "s2ut"
    out = {}
    if paired_dir.is_dir():
        for path in sorted(paired_dir.glob("*.jsonl")):
            rows = fd.read_jsonl(path)
            src_h = sum(r["src_duration_s"] for r in rows) / 3600.0
            tgt_h = sum(r["tgt_duration_s"] for r in rows) / 3600.0
            spks = {r["speaker_id"] for r in rows}
            out[path.stem] = {
                "rows": len(rows),
                "src_hours": round(src_h, 3),
                "tgt_hours": round(tgt_h, 3),
                "speakers": len(spks),
                "license": rows[0]["license"] if rows else "",
            }
        (paired_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
