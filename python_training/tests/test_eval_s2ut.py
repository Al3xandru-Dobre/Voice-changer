"""Mutation-proof unit & CLI tests for Milestone 3 / Stage 01 S2UT evaluation
(`fvml/s2ut_eval.py`, `eval_s2ut.py`, `fvml/asr.py`):
  * SimulEval Average Lagging (`AL` & `AL_CA`) exact Ma et al. (2020) formulas
    including algorithmic cutoff `tau` preservation under slow compute
  * Forced-language Whisper transcription & diacritic-preserving normalisation
  * SacreBLEU `13a` ASR-BLEU & `chrF++` (`word_order=2`) scoring
  * Waveform first-audio onset delay & trailing non-silent flush duration
  * Silence/noise hallucination audit (`0` words on clean silence, non-zero on babble)
"""
import json
import math

import numpy as np
import pytest

import eval_s2ut
from fvml import asr, audio, evallist, s2ut_eval


def voiced_segment(start_s, end_s, total_s, sr=16000, f0=200.0):
    n = int(round(total_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    env = ((t >= start_s) & (t < end_s)).astype(np.float32)
    return (0.25 * np.sin(2.0 * np.pi * f0 * t) * env).astype(np.float32)


def test_simuleval_average_lagging_waitk_closed_form():
    # 10 chunks of 80 ms (T_src = 800 ms), 1 emitted unit per chunk (|Y| = 10, gamma = 1/80).
    # Under Ma et al. (2019/2020), Wait-k emitting 1 unit at t_src = (k + i) * 80 ms
    # (capped at T_src = 800 ms during tail flush) has exact AL = k * 80 ms!
    for k in (1, 3, 5):
        steps = []
        for i in range(10):
            t_src_ms = min(800.0, float((k + i) * 80))
            steps.append({"step": i, "t_src_ms": t_src_ms, "t_tgt_frames_emitted": 1,
                          "t_wall_ms": t_src_ms + 25.0})
        al, al_ca = s2ut_eval.average_lagging_ms(800.0, steps)
        assert al == pytest.approx(k * 80.0)
        assert al_ca == pytest.approx(k * 80.0 + 25.0)


def test_simuleval_al_clamps_tail_beyond_source():
    """An offline runner that keeps counting the zero tail it feeds after EOF
    (fv_s2ut_file --dump-trace) must score exactly like a producer that caps
    t_src_ms at T_src: the reader cannot have read past the source."""
    # T_src = 760 ms is not a whole number of 80 ms chunks, so the first delay
    # reaching T_src is 800 ms uncapped vs 760 ms capped: without the clamp the
    # cutoff term inflates AL by 40 / tau ms.
    t_src_total = 760.0
    for k in (1, 3, 5):
        capped, uncapped = [], []
        for i in range(10):
            t = float((k + i) * 80)
            capped.append({"step": i, "t_src_ms": min(t_src_total, t), "t_tgt_frames_emitted": 1})
            uncapped.append({"step": i, "t_src_ms": t, "t_tgt_frames_emitted": 1})
        want = s2ut_eval.average_lagging_ms(t_src_total, capped)[0]
        assert s2ut_eval.average_lagging_ms(t_src_total, uncapped)[0] == pytest.approx(want)


def test_simuleval_al_ca_uses_algorithmic_tau_not_wallclock_tau():
    """Ma et al. (EMNLP 2020 §2.2): AL_CA must use the algorithmic cutoff `tau`
    defined by `d_j >= T_src`, NOT `d_j_ca >= T_src`. Otherwise a slow first
    step where `d_1_ca >= T_src` would truncate `tau` to 1 and ignore the rest
    of the utterance."""
    # T_src = 400 ms, 4 units (gamma = 4 / 400 = 0.01 frames/ms, ideal = [0, 100, 200, 300]).
    # Algorithmic delays: [160, 240, 320, 400] -> tau = 4, AL = mean([160, 140, 120, 100]) = 130 ms.
    # Suppose step 0 had a huge 300 ms stall so wall clock is [460, 540, 620, 700] (all > 400 ms!).
    # With true algorithmic tau = 4: AL_CA = mean([460-0, 540-100, 620-200, 700-300]) = 430 ms.
    # A buggy wall-clock tau would stop at j=1 and return 460 - 0 = 460 ms!
    steps = [
        {"step": 0, "t_src_ms": 160.0, "t_tgt_frames_emitted": 1, "t_wall_ms": 460.0},
        {"step": 1, "t_src_ms": 240.0, "t_tgt_frames_emitted": 1, "t_wall_ms": 540.0},
        {"step": 2, "t_src_ms": 320.0, "t_tgt_frames_emitted": 1, "t_wall_ms": 620.0},
        {"step": 3, "t_src_ms": 400.0, "t_tgt_frames_emitted": 1, "t_wall_ms": 700.0},
    ]
    al, al_ca = s2ut_eval.average_lagging_ms(400.0, steps)
    assert al == pytest.approx(130.0)
    assert al_ca == pytest.approx(430.0)


def test_multilingual_normalisation_and_sacrebleu():
    assert asr.normalize_multilingual("¡Hola, Señor Muñoz!") == "hola señor muñoz"
    assert asr.normalize_multilingual("C'est l'été à Paris !") == "c est l été à paris"
    assert asr.normalize_multilingual("Übermäßige Größe, straße?") == "übermäßige größe straße"

    exact = asr.translation_scores(
        ["hola señor muñoz cómo está usted hoy", "wir müssen heute zusammen arbeiten"],
        ["hola señor muñoz cómo está usted hoy", "wir müssen heute zusammen arbeiten"],
    )
    assert exact["asr_bleu"] == pytest.approx(100.0, abs=1e-3)
    assert exact["asr_chrfpp"] == pytest.approx(100.0, abs=1e-3)

    # Empty reference rows (noise-only clips) must be skipped rather than dividing by zero.
    mixed = asr.translation_scores(["", "the cat sat on the mat"], ["", "the cat sat on the mat"])
    assert mixed["skipped"] == 1 and mixed["asr_bleu"] == pytest.approx(100.0, abs=1e-3)
    assert math.isnan(mixed["per_bleu"][0])


def test_onset_and_trailing_non_silent_detectors():
    src = voiced_segment(0.10, 1.50, 2.00)
    out = voiced_segment(0.42, 1.98, 2.50)   # onset delayed by +320 ms
    assert s2ut_eval.first_audio_onset_delay_ms(src, out) == pytest.approx(320.0, abs=1e-3)
    # Expected end at 1.80 s -> active speech [1.80, 1.98) = 180 ms trailing non-silent audio.
    assert s2ut_eval.trailing_non_silent_duration_ms(out, 1.80) == pytest.approx(180.0, abs=1e-3)
    assert s2ut_eval.trailing_non_silent_duration_ms(np.zeros(16000, np.float32), 0.0) == 0.0


def test_eval_s2ut_cli_oracle_translated_and_hallucination_audit(tmp_path, monkeypatch):
    """Exercise eval_s2ut.py end-to-end with deterministic fake ASR/ECAPA/UTMOS
    models, verifying language forcing, trace AL/AL_CA, and noise hallucination
    detection."""
    seen_languages = []

    class FakeWhisper:
        def __init__(self, device="cpu", batch_size=8):
            pass

        def transcribe(self, waves, language="en", silence_dbfs=None):
            seen_languages.append(language)
            out = []
            for w in waves:
                if silence_dbfs is not None and asr.is_silent(w, silence_dbfs):
                    out.append("")
                elif language == "de":
                    out.append("Das Haus ist groß.")
                elif language == "es":
                    out.append("Hola mundo amigo.")
                else:
                    out.append("Hello world friend.")
            return out

        def normalize(self, text, language="en"):
            return asr.normalize_multilingual(text)

        def close(self):
            pass

    class FakeEcapa:
        def __init__(self, device="cpu", batch_size=8):
            pass

        def embed(self, waves):
            embs = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(waves), 1))
            return embs

        def close(self):
            pass

    class FakeUtmos:
        def __init__(self, device="cpu"):
            pass

        def score(self, waves):
            return [4.1] * len(waves)

        def close(self):
            pass

    monkeypatch.setattr(s2ut_eval.asr, "Whisper", FakeWhisper)
    monkeypatch.setattr(s2ut_eval.speaker, "Ecapa", FakeEcapa)
    monkeypatch.setattr(s2ut_eval.mos, "Utmos", FakeUtmos)
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))

    # Create source, oracle target, converted WAVs, target voice ref, and emission trace.
    audio.save(tmp_path / "src_es.wav", voiced_segment(0.10, 1.10, 1.50))
    audio.save(tmp_path / "src_en.wav", voiced_segment(0.10, 1.10, 1.50))
    audio.save(tmp_path / "tgt_en.wav", voiced_segment(0.05, 1.05, 1.40))
    audio.save(tmp_path / "tgt_de.wav", voiced_segment(0.05, 1.05, 1.40))
    audio.save(tmp_path / "ref_voice.wav", voiced_segment(0.0, 1.0, 1.0))

    conv_dir = tmp_path / "converted"
    conv_dir.mkdir()
    audio.save(conv_dir / "u1.wav", voiced_segment(0.34, 1.34, 1.60))   # +240 ms onset
    audio.save(conv_dir / "u2.wav", voiced_segment(0.42, 1.42, 1.70))   # +320 ms onset

    trace_path = conv_dir / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"utt_id": "u1", "steps": [
            {"step": 0, "t_src_ms": 240.0, "t_tgt_frames_emitted": 4, "elapsed_ms": 20.0},
            {"step": 1, "t_src_ms": 1500.0, "t_tgt_frames_emitted": 4, "elapsed_ms": 20.0},
        ]}) + "\n" +
        json.dumps({"utt_id": "u2", "steps": [
            {"step": 0, "t_src_ms": 320.0, "t_tgt_frames_emitted": 4, "elapsed_ms": 25.0},
            {"step": 1, "t_src_ms": 1500.0, "t_tgt_frames_emitted": 4, "elapsed_ms": 25.0},
        ]}) + "\n",
        encoding="utf-8",
    )
    (conv_dir / "timing.csv").write_text("utt_id,seconds\nu1,0.30\nu2,0.45\n", encoding="utf-8")

    eval_tsv = tmp_path / "eval.tsv"
    evallist.write(eval_tsv, [
        {"utt_id": "u1", "src_lang": "es", "tgt_lang": "en", "speaker": "s1", "gender": "F",
         "src_duration": 1.5, "tgt_duration": 1.4, "gamma": 1.4 / 1.5,
         "src_path": "src_es.wav", "tgt_path": "tgt_en.wav",
         "src_text": "hola mundo amigo", "tgt_text": "hello world friend"},
        {"utt_id": "u2", "src_lang": "en", "tgt_lang": "de", "speaker": "s2", "gender": "M",
         "src_duration": 1.5, "tgt_duration": 1.4, "gamma": 1.4 / 1.5,
         "src_path": "src_en.wav", "tgt_path": "tgt_de.wav",
         "src_text": "the house is big", "tgt_text": "das haus ist groß"},
    ], evallist.S2UT_COLUMNS)

    targets_tsv = tmp_path / "targets.tsv"
    evallist.write(targets_tsv, [
        {"voice": "lj", "clip_id": "c1", "speaker": "lj", "gender": "F",
         "duration": 1.0, "path": "ref_voice.wav", "dataset": "ljspeech"},
    ], evallist.TARGET_COLUMNS)

    # 1) Oracle mode
    assert eval_s2ut.main([
        "--list", str(eval_tsv), "--oracle", "--device", "cpu", "--out", str(tmp_path / "out_oracle")
    ]) == 0
    oracle_sum = json.loads((tmp_path / "out_oracle" / "summary.json").read_text("utf-8"))
    assert oracle_sum["asr_bleu"] == pytest.approx(100.0, abs=1e-3)
    assert oracle_sum["utmos"] == pytest.approx(4.1)
    assert set(seen_languages) == {"en", "de"}   # Whisper forced per target language!

    # 2) Translated mode with traces, speaker refs, and timing
    assert eval_s2ut.main([
        "--list", str(eval_tsv), "--converted", str(conv_dir), "--targets", str(targets_tsv),
        "--voice", "lj", "--device", "cpu", "--out", str(tmp_path / "out_conv")
    ]) == 0
    conv_sum = json.loads((tmp_path / "out_conv" / "summary.json").read_text("utf-8"))
    assert conv_sum["onset_delay_ms"] == pytest.approx(280.0, abs=1e-3)   # mean(240, 320)
    assert conv_sum["al_ms"] is not None and conv_sum["al_ca_ms"] > conv_sum["al_ms"]
    assert conv_sum["tgt_cos"] == pytest.approx(1.0)
    assert conv_sum["rtf"] == pytest.approx(0.75 / 3.0)
    assert set(conv_sum["per_pair"]) == {"es->en", "en->de"}

    # 3) Silence hallucination audit: 0 words when gated silent vs >0 words when babbling
    sil_tsv = tmp_path / "sil.tsv"
    evallist.write(sil_tsv, [
        {"utt_id": "n1", "kind": "noise_only", "src_lang": "en", "tgt_lang": "es",
         "src_duration": 2.0, "speech_end_s": 0.0, "src_path": "src_en.wav",
         "src_text": "", "tgt_text": ""},
    ], evallist.SILENCE_COLUMNS)

    clean_sil_dir = tmp_path / "sil_clean"
    clean_sil_dir.mkdir()
    audio.save(clean_sil_dir / "n1.wav", np.zeros(32000, dtype=np.float32))
    assert eval_s2ut.main([
        "--list", str(sil_tsv), "--converted", str(clean_sil_dir),
        "--metrics", "flush", "--device", "cpu", "--out", str(tmp_path / "out_sil_ok")
    ]) == 0
    ok_sum = json.loads((tmp_path / "out_sil_ok" / "summary.json").read_text("utf-8"))
    assert ok_sum["hallucinated_words"] == 0 and ok_sum["noise_hallucination_rate"] == 0.0

    babble_dir = tmp_path / "sil_babble"
    babble_dir.mkdir()
    audio.save(babble_dir / "n1.wav", voiced_segment(0.2, 1.0, 2.0))
    assert eval_s2ut.main([
        "--list", str(sil_tsv), "--converted", str(babble_dir),
        "--metrics", "flush", "--device", "cpu", "--out", str(tmp_path / "out_sil_bad")
    ]) == 0
    bad_sum = json.loads((tmp_path / "out_sil_bad" / "summary.json").read_text("utf-8"))
    assert bad_sum["hallucinated_words"] == 3 and bad_sum["noise_hallucination_rate"] == 1.0
