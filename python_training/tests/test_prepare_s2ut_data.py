"""End-to-end offline tests for Milestone 3 / Stage 01 data preparation,
synthesis round-trip CER filtering, CML-TTS SNR/speaker curation, commercial
licence guardrails, SentencePiece training, and deterministic S2UT eval-list
construction."""
import dataclasses
import hashlib
import io
import json
import tarfile
import zipfile

import numpy as np
import pytest
import sentencepiece as spm
import soundfile as sf

from fvml import audio, datasets as fd, evallist, s2ut_data as sd
import make_s2ut_eval_lists
import prepare_s2ut_data


def speech_like_wave(sr=16000, seconds=3.5, snr_db=45.0, f0=180.0):
    """ Voiced burst with 20% quiet pauses so percentile SNR is controllable."""
    n = int(round(sr * seconds))
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.25 * np.sin(2.0 * np.pi * f0 * t)
    # Silence the first and last 12% so 10th-percentile frame RMS measures the noise floor.
    env = ((t >= 0.12 * seconds) & (t <= 0.88 * seconds)).astype(np.float32)
    noise_amp = 0.25 * (10.0 ** (-snr_db / 20.0))
    rng = np.random.default_rng(42)
    noise = noise_amp * rng.standard_normal(n).astype(np.float32)
    return (sig * env + noise).astype(np.float32)


def wav_bytes(sr=22050, seconds=3.5, snr_db=45.0, fmt="WAV"):
    buf = io.BytesIO()
    sf.write(buf, speech_like_wave(sr, seconds, snr_db), sr, format=fmt)
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


@pytest.fixture
def sctx(tmp_path):
    cv_root = tmp_path / "common_voice"
    cv_root.mkdir()
    return sd.S2utContext(
        root=tmp_path / "data",
        common_voice_root=cv_root,
        jobs=1,
        lock_file=tmp_path / "datasets.lock.json",
        log=lambda *_: None,
    )


def _local_archive(ds, archive_paths):
    arcs = []
    for a, p in zip(ds.archives, archive_paths):
        data = p.read_bytes()
        arcs.append(fd.Archive(a.file, (p.as_uri(),), len(data), hashlib.md5(data).hexdigest()))
    return dataclasses.replace(ds, archives=tuple(arcs))


def test_cvss_c_join_and_schema(sctx, tmp_path):
    (sctx.common_voice_root / "es" / "clips").mkdir(parents=True)
    (sctx.common_voice_root / "es" / "clips" / "common_voice_es_001.wav").write_bytes(
        wav_bytes(sr=48000, seconds=4.0)
    )
    (sctx.common_voice_root / "es" / "validated.tsv").write_text(
        "client_id\tpath\tsentence\tgender\nspk_es_1\tcommon_voice_es_001.mp3\tHola mundo amigo\tfemale\n",
        encoding="utf-8",
    )
    make_tar(tmp_path / "cvss.tar.gz", {
        "train.tsv": b"common_voice_es_001.mp3\tHello world friend\n",
        "train/common_voice_es_001.mp3.wav": wav_bytes(sr=22050, seconds=3.5),
    })
    make_tar(tmp_path / "covost.tar.gz", {
        "covost_v2.es_en.train.tsv": (
            b"path\tsentence\ttranslation\tclient_id\n"
            b"common_voice_es_001.mp3\tHola mundo amigo\tHello world friend\tspk_es_1\n"
        ),
    })
    ds = _local_archive(sd.S2UT_DATASETS["cvss-c-es_en"],
                        [tmp_path / "cvss.tar.gz", tmp_path / "covost.tar.gz"])
    mf = sd.prepare_s2ut(sctx, ds)
    rows = fd.read_jsonl(mf)
    assert len(rows) == 1
    r = rows[0]
    for k in sd.PAIRED_REQUIRED_KEYS:
        assert k in r, k
    assert (r["src_lang"], r["tgt_lang"], r["gender"], r["license"]) == ("es", "en", "F", "CC BY 4.0")
    assert sf.info(str(sctx.root / r["src_audio_path"])).samplerate == 16000
    assert sf.info(str(sctx.root / r["tgt_audio_path"])).samplerate == 16000


def test_europarl_and_covost_synth_cer_gate_rejects_above_5_percent(sctx, tmp_path):
    """Clips with Whisper round-trip CER <= 5% pass; clips > 5% are rejected."""
    # Fake synthesizer writes deterministic 16 kHz FLACs and embeds clip index in frequency.
    synth_texts = []

    def fake_synth(text, lang, dst_path):
        idx = len(synth_texts)
        synth_texts.append((lang, text))
        wav = speech_like_wave(16000, seconds=3.2, f0=150.0 + 30.0 * idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        audio.save(dst_path, wav, subtype="PCM_24")
        return audio.info(dst_path)[0]

    # Utterance 0: exact match (CER = 0.0)
    # Utterance 1: 1 char typo in 28 chars (CER = 3.57% <= 5% -> kept)
    # Utterance 2: mangled numbers/words (CER > 30% -> rejected!)
    fake_hyps = [
        "das parlament ist heute offen",
        "wir müssen zusammen arbeitem",     # 1 char vs "wir müssen zusammen arbeiten" (1/28 = 3.57%)
        "komplett falscher satz ohne zahlen",
    ]
    sctx.synthesizer = fake_synth
    sctx.cer_transcriber = lambda waves, language: fake_hyps[: len(waves)]

    ep_files = {
        "v1.1/en/de/train/segments.lst": b"ep01 spkA\nep01 spkB\nep01 spkC\n",
        "v1.1/en/de/train/segments.en": (
            "The parliament is open today.\n"
            "We must work together.\n"
            "In 1998 there were 42 members.\n"
        ).encode("utf-8"),
        "v1.1/en/de/train/segments.de": (
            "Das Parlament ist heute offen.\n"
            "Wir müssen zusammen arbeiten.\n"
            "Im Jahr 1998 gab es 42 Mitglieder.\n"
        ).encode("utf-8"),
        "v1.1/en/de/train/wavs/ep01_0000.wav": wav_bytes(16000, 3.4),
        "v1.1/en/de/train/wavs/ep01_0001.wav": wav_bytes(16000, 3.4),
        "v1.1/en/de/train/wavs/ep01_0002.wav": wav_bytes(16000, 3.4),
    }
    make_tar(tmp_path / "europarl.tar.gz", ep_files)
    ds = _local_archive(sd.S2UT_DATASETS["europarl-st-en_de"], [tmp_path / "europarl.tar.gz"])
    mf = sd.prepare_s2ut(sctx, ds)
    rows = fd.read_jsonl(mf)
    assert len(rows) == 2
    assert all(r["synth_cer"] <= 0.05 for r in rows)
    report = json.loads((sctx.root / ".state" / "europarl-st-en_de.synth_report.json").read_text("utf-8"))
    assert report["synthesized"] == 3 and report["accepted"] == 2 and report["rejected_cer"] == 1


def test_css10_and_cml_tts_curation_and_m2_summary_compatibility(sctx, tmp_path):
    # CSS10 Spanish
    make_zip(tmp_path / "css10_es.zip", {
        "transcript.txt": (
            "0001.wav|Uno dos tres.|uno dos tres|3.5\n"
            "0002.wav|Cuatro cinco.|cuatro cinco|3.5\n"
        ).encode("utf-8"),
        "0001.wav": wav_bytes(22050, 3.5),
        "0002.wav": wav_bytes(22050, 3.5),
    })
    ds_css = _local_archive(sd.S2UT_DATASETS["css10-es"], [tmp_path / "css10_es.zip"])
    sd.prepare_s2ut(sctx, ds_css)

    # CML-TTS French: 3 F + 3 M clean speakers + 1 noisy speaker (SNR ~ 10 dB) + out-of-range durations
    cml_files = {}
    csv_lines = ["wav_filename|client_id|gender|transcript"]
    for g, spks in (("F", ("f1", "f2", "f3")), ("M", ("m1", "m2", "m3"))):
        for rank, spk in enumerate(spks):
            # Give f1, f2 and m1, m2 longer clean duration than f3, m3 so top-2 ranking is deterministic
            dur = 5.0 - rank * 1.0
            rel = f"wavs/{spk}_ok.wav"
            cml_files[rel] = wav_bytes(24000, seconds=dur, snr_db=45.0)
            csv_lines.append(f"{rel}|{spk}|{g}|Bonjour tout le monde {spk}")
    # Noisy speaker (SNR ~ 8 dB < 30 dB -> rejected) and too-short clip (1.0 s < 2.0 s -> rejected)
    cml_files["wavs/f_noisy.wav"] = wav_bytes(24000, seconds=6.0, snr_db=8.0)
    csv_lines.append("wavs/f_noisy.wav|f_noisy|F|Trop de bruit")
    cml_files["wavs/f1_short.wav"] = wav_bytes(24000, seconds=1.0, snr_db=45.0)
    csv_lines.append("wavs/f1_short.wav|f1|F|Court")
    cml_files["train.csv"] = ("\n".join(csv_lines) + "\n").encode("utf-8")

    make_tar(tmp_path / "cml_fr.tar.bz2", cml_files, mode="w:bz2")
    ds_cml = _local_archive(sd.S2UT_DATASETS["cml-tts-fr"], [tmp_path / "cml_fr.tar.bz2"])
    mf_cml = sd.prepare_s2ut(sctx, ds_cml)
    rows = fd.read_jsonl(mf_cml)
    kept_speakers = sorted({r["speaker"] for r in rows})
    assert kept_speakers == ["cml_fr_f1", "cml_fr_f2", "cml_fr_m1", "cml_fr_m2"]
    assert all(r["snr_db"] >= 30.0 and 2.0 <= r["duration"] <= 12.0 for r in rows)

    # Milestone 2's fd.summarize() must still succeed cleanly on $FLUXVOICE_DATA/manifests/*.jsonl!
    m2_summary = fd.summarize(sctx.base_ctx())
    assert "css10-es" in m2_summary and "cml-tts-fr" in m2_summary


def test_noncommercial_guardrail_and_spm_isolation(sctx, tmp_path):
    # Default mode must block VoxPopuli and omit it from `all`.
    assert all(not d.noncommercial for d in sd.select_s2ut("all", include_noncommercial=False))
    with pytest.raises(fd.PrepareError, match="CC BY-NC 4.0"):
        sd.select_s2ut("voxpopuli-s2ut", include_noncommercial=False)
    assert prepare_s2ut_data.main(["--data-root", str(sctx.root), "--datasets", "voxpopuli-s2ut"]) == 1

    # With --include-noncommercial, VoxPopuli lands strictly under data/nc/ and never data/manifests/.
    make_tar(tmp_path / "vp.tar.gz", {
        "pairs.tsv": (
            b"id\tsplit\tsrc_lang\ttgt_lang\tsrc_audio\ttgt_audio\tsrc_text\ttgt_text\tspeaker_id\n"
            b"01\ttrain\ten\tes\ten01.wav\tes01.wav\texclusive noncommercial token zzznc\thola zzznc\tvp1\n"
        ),
        "en01.wav": wav_bytes(16000, 3.2),
        "es01.wav": wav_bytes(16000, 3.2),
    })
    sctx.include_noncommercial = True
    ds_vp = _local_archive(sd.S2UT_DATASETS["voxpopuli-s2ut"], [tmp_path / "vp.tar.gz"])
    mf_vp = sd.prepare_s2ut(sctx, ds_vp)
    assert "nc/manifests/s2ut/voxpopuli-s2ut.jsonl" in mf_vp.as_posix()
    assert not (sctx.root / "manifests" / "s2ut" / "voxpopuli-s2ut.jsonl").exists()

    # Add commercial paired transcripts for all 4 languages and fit SentencePiece models.
    comm_dir = sctx.root / "manifests" / "s2ut"
    comm_dir.mkdir(parents=True, exist_ok=True)
    fd.write_jsonl(comm_dir / "comm.jsonl", [
        {"id": "1", "split": "train", "src_lang": "en", "tgt_lang": "es",
         "src_audio_path": "a", "src_duration_s": 3.5, "src_text": "Hello bright morning sun",
         "tgt_audio_path": "b", "tgt_duration_s": 3.6, "tgt_text": "Hola sol brillante de la mañana",
         "speaker_id": "s1", "dataset": "test", "license": "CC BY 4.0"},
        {"id": "2", "split": "train", "src_lang": "fr", "tgt_lang": "de",
         "src_audio_path": "c", "src_duration_s": 3.5, "src_text": "Bonjour les amis français",
         "tgt_audio_path": "d", "tgt_duration_s": 3.6, "tgt_text": "Guten Morgen liebe Freunde aus München",
         "speaker_id": "s2", "dataset": "test", "license": "CC BY 4.0"},
    ])
    models = sd.train_spm_models(sctx, vocab_size=64)
    assert set(models) == {"en", "es", "fr", "de"}
    for lang, mpath in models.items():
        proc = spm.SentencePieceProcessor(model_file=str(mpath))
        assert (proc.unk_id(), proc.bos_id(), proc.eos_id(), proc.pad_id()) == (0, 1, 2, 3)
        # Ensure non-commercial data/nc/ token "zzznc" never leaked into the commercial SPM vocab!
        pieces = [proc.id_to_piece(i) for i in range(proc.get_piece_size())]
        assert not any("zzznc" in p for p in pieces), lang


def test_deterministic_eval_list_builders_and_reordering_score(tmp_path):
    # German subordinate verb-final clause must outscore flat SVO sentence.
    high_de = make_s2ut_eval_lists.reordering_score(
        "de", "en",
        "Ich weiß, weil mein Bruder gestern Abend das schwere Buch gelesen hat.",
        "I know because my brother read the heavy book last night.",
    )
    low_de = make_s2ut_eval_lists.reordering_score("de", "en", "Das Haus ist groß.", "The house is big.")
    assert high_de > low_de >= 0

    # Spanish clitic + post-nominal adjective outscores bare greeting.
    high_es = make_s2ut_eval_lists.reordering_score(
        "en", "es", "She gave it to the red car owner.", "Ella se lo dio al dueño del coche rojo."
    )
    assert high_es > make_s2ut_eval_lists.reordering_score("en", "es", "Hello.", "Hola.")

    # Build synthetic 6-direction manifests (>= 20 speakers, 200+ test clips each) and check --check!
    paired_dir = tmp_path / "data" / "manifests" / "s2ut"
    paired_dir.mkdir(parents=True)
    for pair in sd.DIRECTIONS:
        src_l, tgt_l = pair.split("_")
        rows = []
        for i in range(220):
            spk = f"spk_{i % 24:02d}"
            gender = "F" if (i % 24) < 12 else "M"
            reord_txt = (
                "weil der sprecher heute das wichtige dokument endlich unterschrieben hat"
                if "de" in pair and i < 12 else f"sample {src_l} sentence number {i}"
            )
            rows.append({
                "id": f"{pair}_{i:04d}",
                "split": "test",
                "src_lang": src_l,
                "tgt_lang": tgt_l,
                "src_audio_path": f"16k/{pair}_src_{i:04d}.flac",
                "src_duration_s": 4.0 + (i % 5),
                "src_text": reord_txt,
                "tgt_audio_path": f"16k/{pair}_tgt_{i:04d}.flac",
                "tgt_duration_s": 4.4 + (i % 5),
                "tgt_text": reord_txt if "de" in pair and tgt_l == "de" else f"translated {tgt_l} {i}",
                "speaker_id": spk,
                "gender": gender,
                "dataset": "synthetic",
                "license": "CC BY 4.0",
            })
        fd.write_jsonl(paired_dir / f"{pair}.jsonl", rows)

    out_lists = tmp_path / "eval_lists_s2ut"
    assert make_s2ut_eval_lists.main([
        "--only", "paired", "--data-root", str(tmp_path / "data"), "--out-dir", str(out_lists)
    ]) == 0
    assert make_s2ut_eval_lists.main([
        "--only", "paired", "--data-root", str(tmp_path / "data"), "--out-dir", str(out_lists), "--check"
    ]) == 0

    es_en_rows = evallist.read_s2ut(out_lists / "test_es_en_200.tsv")
    assert len(es_en_rows) == 200
    assert sum(1 for r in es_en_rows if r["gender"] == "F") == 100
    assert sum(1 for r in es_en_rows if r["gender"] == "M") == 100
    torture_rows = evallist.read_s2ut(out_lists / "reordering_torture_60.tsv")
    assert len(torture_rows) == 60
