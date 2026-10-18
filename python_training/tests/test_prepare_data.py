"""prepare_data's pipeline on miniature fake corpora (file:// URLs, tmp dirs).

Each fake archive copies the real corpus layout (fvml/datasets.py docstrings,
Stage 01 research), so every builder runs end to end: download -> verify ->
extract -> resample -> manifest, then a rerun that must skip finished work.
"""
import dataclasses
import hashlib
import io
import json
import tarfile
import zipfile

import numpy as np
import pytest
import soundfile as sf

from fvml import datasets as fd


def tone(sr, seconds=0.5, f=220.0):
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def audio_bytes(sr, fmt, channels=1, seconds=0.5):
    buf = io.BytesIO()
    x = tone(sr, seconds)
    sf.write(buf, np.stack([x] * channels, axis=1) if channels > 1 else x, sr, format=fmt)
    return buf.getvalue()


def make_tar(path, files, mode="w:gz"):
    with tarfile.open(path, mode) as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def fake_dataset(ds, archive_path, md5=True):
    data = archive_path.read_bytes()
    a = fd.Archive(ds.archives[0].file, (archive_path.as_uri(),), len(data),
                   hashlib.md5(data).hexdigest() if md5 else None)
    return dataclasses.replace(ds, archives=(a,))


@pytest.fixture
def ctx(tmp_path):
    return fd.Context(root=tmp_path / "data", jobs=2, lock_file=tmp_path / "lock.json",
                      log=lambda *_: None)


def run(ctx, ds):
    manifest = fd.prepare(ctx, ds)
    return fd.read_jsonl(manifest)


def test_librispeech(ctx, tmp_path):
    files = {
        "LibriSpeech/SPEAKERS.TXT": b";ID  |SEX| SUBSET    |MINUTES| NAME\n19   | F | test-clean | 25.19 | Kara\n",
        "LibriSpeech/test-clean/19/198/19-198.trans.txt": b"19-198-0000 HELLO WORLD\n19-198-0001 AGAIN\n",
        "LibriSpeech/test-clean/19/198/19-198-0000.flac": audio_bytes(16000, "FLAC"),
        "LibriSpeech/test-clean/19/198/19-198-0001.flac": audio_bytes(16000, "FLAC", seconds=1.0),
    }
    make_tar(tmp_path / "test-clean.tar.gz", files)
    ds = fake_dataset(fd.DATASETS["librispeech-test-clean"], tmp_path / "test-clean.tar.gz")
    rows = run(ctx, ds)
    assert [r["utt_id"] for r in rows] == ["19-198-0000", "19-198-0001"]
    assert rows[0] == {"utt_id": "19-198-0000", "path": "raw/LibriSpeech/test-clean/19/198/19-198-0000.flac",
                       "duration": 0.5, "speaker": "19", "dataset": "librispeech",
                       "subset": "test-clean", "gender": "F", "text": "HELLO WORLD"}
    # First download of an unpinned archive records its SHA256 (TOFU).
    lock = json.loads(ctx.lock_file.read_text())
    assert lock["sha256"]["test-clean.tar.gz"] == hashlib.sha256(
        (tmp_path / "test-clean.tar.gz").read_bytes()).hexdigest()
    # The data root ignores itself, and a rerun is a no-op.
    assert (ctx.root / ".gitignore").read_text().strip().endswith("*")
    logs = []
    ctx.log = logs.append
    run(ctx, ds)
    assert any("already extracted" in m for m in logs) and any("up to date" in m for m in logs)


def test_checksum_mismatch_is_fatal(ctx, tmp_path):
    make_tar(tmp_path / "test-clean.tar.gz", {"LibriSpeech/README.TXT": b"x"})
    ds = fake_dataset(fd.DATASETS["librispeech-test-clean"], tmp_path / "test-clean.tar.gz")
    ds = dataclasses.replace(ds, archives=(dataclasses.replace(ds.archives[0], md5="0" * 32),))
    with pytest.raises(fd.PrepareError, match="MD5"):
        fd.prepare(ctx, ds)
    assert (ctx.root / "archives" / "test-clean.tar.gz.bad").is_file()
    ctx.pin("test-clean.tar.gz", "f" * 64)                   # wrong pin, right MD5
    ds = fake_dataset(fd.DATASETS["librispeech-test-clean"], tmp_path / "test-clean.tar.gz")
    with pytest.raises(fd.PrepareError, match="SHA256"):
        fd.prepare(ctx, ds)


def test_ljspeech_is_resampled(ctx, tmp_path):
    files = {"LJSpeech-1.1/metadata.csv": b"LJ001-0001|Printing, in the|Printing, in the\n",
             "LJSpeech-1.1/wavs/LJ001-0001.wav": audio_bytes(22050, "WAV")}
    make_tar(tmp_path / "LJSpeech-1.1.tar.bz2", files, mode="w:bz2")
    rows = run(ctx, fake_dataset(fd.DATASETS["ljspeech"], tmp_path / "LJSpeech-1.1.tar.bz2", md5=False))
    assert rows[0]["path"] == "16k/ljspeech/LJ001-0001.flac" and rows[0]["speaker"] == "lj"
    info = sf.info(str(ctx.root / rows[0]["path"]))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_24")
    assert abs(rows[0]["duration"] - 0.5) < 1e-3


def test_hifitts_extracts_only_chosen_speakers(ctx, tmp_path):
    def manifest(spk, quality, n):
        return "".join(json.dumps({"audio_filepath": f"audio/{spk}_{quality}/100/clip_{i:04d}.flac",
                                   "text": "t", "duration": 0.5, "text_normalized": f"Clip {i}."}) + "\n"
                       for i in range(n)).encode()
    files = {"hi_fi_tts_v0/92_manifest_clean_test.json": manifest("92", "clean", 2),
             "hi_fi_tts_v0/6670_manifest_other_test.json": manifest("6670", "other", 1)}
    for i in range(2):
        files[f"hi_fi_tts_v0/audio/92_clean/100/clip_{i:04d}.flac"] = audio_bytes(44100, "FLAC")
    files["hi_fi_tts_v0/audio/6670_other/100/clip_0000.flac"] = audio_bytes(44100, "FLAC")
    make_tar(tmp_path / "hi_fi_tts_v0.tar.gz", files)
    ctx.hifitts_speakers = ("92",)
    rows = run(ctx, fake_dataset(fd.DATASETS["hifitts"], tmp_path / "hi_fi_tts_v0.tar.gz", md5=False))
    assert [r["utt_id"] for r in rows] == ["hifitts_92_100_clip_0000", "hifitts_92_100_clip_0001"]
    assert rows[0]["path"] == "16k/hifitts/92_clean/100/clip_0000.flac"
    assert rows[0]["subset"] == "clean_test" and rows[0]["gender"] == "F"
    assert not (ctx.root / "raw" / "hi_fi_tts_v0" / "audio" / "6670_other").exists()


def test_musan_rirs_vctk(ctx, tmp_path):
    make_tar(tmp_path / "musan.tar.gz", {
        "musan/music/fma/LICENSE": b"music-fma-0000.wav CC BY 3.0\n",
        "musan/music/fma/music-fma-0000.wav": audio_bytes(16000, "WAV"),
        "musan/noise/free-sound/noise-free-sound-0000.wav": audio_bytes(16000, "WAV")})
    rows = run(ctx, fake_dataset(fd.DATASETS["musan"], tmp_path / "musan.tar.gz"))
    assert {r["subset"] for r in rows} == {"music", "noise"}
    assert rows[0]["licence_file"] == "raw/musan/music/fma/LICENSE"

    make_zip(tmp_path / "rirs_noises.zip", {
        "RIRS_NOISES/simulated_rirs/smallroom/Room001/Room001-00001.wav": audio_bytes(16000, "WAV"),
        "RIRS_NOISES/real_rirs_isotropic_noises/RVB2014_type1_rir_largeroom1_near_angla.wav":
            audio_bytes(16000, "WAV", channels=8),
        "RIRS_NOISES/real_rirs_isotropic_noises/RVB2014_type1_noise_largeroom1_1.wav":
            audio_bytes(16000, "WAV", channels=8),
        "RIRS_NOISES/pointsource_noises/noise-free-sound-0001.wav": audio_bytes(16000, "WAV")})
    rows = {r["subset"]: r for r in run(ctx, fake_dataset(fd.DATASETS["rirs"], tmp_path / "rirs_noises.zip"))}
    assert set(rows) == {"simulated_rir", "real_rir", "isotropic_noise", "pointsource_noise"}
    assert rows["simulated_rir"]["utt_id"] == "simulated_rirs/smallroom/Room001/Room001-00001"
    assert rows["real_rir"]["channels"] == 8

    make_zip(tmp_path / "VCTK-Corpus-0.92.zip", {
        "speaker-info.txt": b"ID  AGE  GENDER  ACCENTS  REGION COMMENTS\np225  23  F    English    Southern  England\n",
        "txt/p225/p225_001.txt": b"Please call Stella.\n",
        "wav48_silence_trimmed/p225/p225_001_mic1.flac": audio_bytes(48000, "FLAC"),
        "wav48_silence_trimmed/p225/p225_001_mic2.flac": audio_bytes(48000, "FLAC")})
    rows = run(ctx, fake_dataset(fd.DATASETS["vctk"], tmp_path / "VCTK-Corpus-0.92.zip"))
    assert rows == [{"utt_id": "p225_001", "path": "16k/vctk/p225/p225_001.flac", "duration": 0.5,
                     "speaker": "p225", "dataset": "vctk", "subset": "mic1", "gender": "F",
                     "accent": "English", "text": "Please call Stella."}]
    assert not (ctx.root / "raw" / "VCTK-Corpus-0.92" / "wav48_silence_trimmed" / "p225" /
                "p225_001_mic2.flac").exists()

    summary = fd.summarize(ctx)
    assert summary["vctk"]["speaker_hours"] == {"p225": round(0.5 / 3600, 3)}


def test_select_groups():
    assert [d.key for d in fd.select("eval")] == ["librispeech-test-clean", "librispeech-dev-clean"]
    assert "vctk" not in [d.key for d in fd.select("core")]
    assert "vctk" in [d.key for d in fd.select("all")]
    with pytest.raises(fd.PrepareError, match="unknown"):
        fd.select("librispeech,bogus")
