"""Causal Unit-HiFi-GAN vocoder, trainer, and Stage 05 decision gate tests."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

import gate_report
import stream_check
import train_unit_vocoder as tuv
from fvml import contract, evallist, features, units, vocoder
from fvml import content_encoder as ce

FS = vocoder.COND_FRAME_SAMPLES


@pytest.fixture(scope="module")
def small_cfg():
    return vocoder.VocoderConfig(
        vocab=32,
        unit_dim=32,
        cond_hidden=16,
        n_speakers=3,
        spk_emb_dim=64,
        upsample_initial_channel=32,
    )


@pytest.fixture(scope="module")
def small_vocoder(small_cfg):
    torch.manual_seed(0)
    return vocoder.UnitVocoder(small_cfg).eval()


def test_vocoder_size_shapes_and_contract(small_vocoder, small_cfg):
    v2 = vocoder.UnitVocoder()
    assert 500_000 <= v2.num_params() <= 1_500_000
    summary = v2.summary()
    assert summary["sample_rate"] == 16000 and summary["frame_samples"] == 320
    assert summary["state_tensors"] == len(v2.state_layout()) > 0
    assert v2.past_context_samples() > 0

    u = torch.randint(0, small_cfg.vocab, (2, 7))
    lf = torch.full((2, 7), 5.0)
    vo = torch.ones(2, 7)
    ap = torch.full((2, 7), 0.1)
    en = torch.full((2, 7), -6.0)
    spk_ids = torch.tensor([0, 2], dtype=torch.int64)
    spk_vecs = small_vocoder.speaker_table(spk_ids)

    with torch.no_grad():
        y_id = small_vocoder(u, lf, vo, ap, en, spk_ids)
        y_vec = small_vocoder(u, lf, vo, ap, en, spk_vecs)
    assert y_id.shape == (2, 7 * FS)
    torch.testing.assert_close(y_id, y_vec)

    for fps in (1, 2, 4):
        vc = small_vocoder.contract(fps)
        ec = contract.ModelContract(
            role="encoder", sample_rate=16000, frame_samples=320,
            frames_per_step=fps, lookahead_frames=1, content_kind="discrete",
            vocab=small_cfg.vocab, states=("s0",), license="MIT",
        )
        contract.validate_pair(ec, vc)
        assert contract.algorithmic_latency(ec, vc) == (fps + 1) * 320

    # Soft-unit variant and 12.5 Hz Mimi frame-rate variant
    soft_v = vocoder.UnitVocoder(vocoder.VocoderConfig(
        content_kind="soft", vocab=0, content_dim=16, unit_dim=16,
        cond_hidden=8, upsample_initial_channel=32,
    )).eval()
    with torch.no_grad():
        y_soft = soft_v(torch.randn(1, 4, 16), lf[:1, :4], vo[:1, :4],
                        ap[:1, :4], en[:1, :4], torch.tensor([0]))
    assert y_soft.shape == (1, 4 * FS)
    assert soft_v.contract(2).content_kind == "soft"

    mimi_v = vocoder.UnitVocoder(vocoder.VocoderConfig(
        frame_samples=1280, cond_frame_samples=320, vocab=64,
        unit_dim=16, cond_hidden=8, upsample_initial_channel=32,
    )).eval()
    assert mimi_v.cfg.unit_repeat == 4
    with torch.no_grad():
        y_mimi = mimi_v(torch.randint(0, 64, (1, 2)),
                        torch.full((1, 8), 5.0), torch.ones(1, 8),
                        torch.full((1, 8), 0.1), torch.full((1, 8), -6.0),
                        torch.tensor([0]))
    assert y_mimi.shape == (1, 2 * 1280)


def test_config_round_trip_and_validation():
    cfg = vocoder.VocoderConfig(vocab=50, upsample_initial_channel=64)
    assert vocoder.VocoderConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(ValueError, match="cond_frame_samples"):
        vocoder.VocoderConfig(upsample_rates=(8, 5, 4, 3))
    with pytest.raises(ValueError, match="multiple"):
        vocoder.VocoderConfig(frame_samples=500, cond_frame_samples=320)
    with pytest.raises(ValueError, match="content_dim"):
        vocoder.VocoderConfig(content_kind="soft", content_dim=0)


@pytest.mark.parametrize("fps", [1, 2, 4])
def test_vocoder_step_equals_forward_and_boundary_slope(small_vocoder, fps):
    rng = np.random.default_rng(fps)
    wav = (0.2 * np.sin(2 * np.pi * 180.0 * np.arange(16 * FS) / 16000.0)).astype(np.float32)
    wav += (0.02 * rng.standard_normal(len(wav))).astype(np.float32)
    rep = stream_check.check_vocoder(small_vocoder, wav, fps_list=(fps,), seed=fps)
    assert rep["pass"]
    r = rep["fps"][fps]
    assert r["max_abs"] <= 1e-4
    assert r["max_boundary_step"] <= r["max_step_ref"] + 1e-6


def test_vocoder_whole_model_is_causal(small_cfg):
    torch.manual_seed(4)
    m = vocoder.UnitVocoder(small_cfg).double().eval()
    n = 10
    u = torch.randint(0, small_cfg.vocab, (1, n))
    lf = torch.full((1, n), 5.0, dtype=torch.float64)
    vo = torch.ones((1, n), dtype=torch.float64)
    ap = torch.full((1, n), 0.1, dtype=torch.float64)
    en = torch.full((1, n), -5.0, dtype=torch.float64)
    spk = m.speaker_table.weight[:1].detach().clone()

    with torch.no_grad():
        ref = m(u, lf, vo, ap, en, spk)
        for t in (0, 4, 8):
            u2 = u.clone()
            u2[:, t + 1] = (u2[:, t + 1] + 7) % small_cfg.vocab
            lf2 = lf.clone()
            lf2[:, t + 1] += 0.5
            out = m(u2, lf2, vo, ap, en, spk)
            diff = (out - ref).abs()[0]
            assert diff[:(t + 1) * FS].max().item() == 0.0
            assert diff[(t + 1) * FS:(t + 2) * FS].max().item() > 0.0


def test_film_blend_and_expand_speakers(small_cfg):
    torch.manual_seed(5)
    m = vocoder.UnitVocoder(small_cfg).eval()
    e0 = m.speaker_table.weight[0:1].detach().clone()
    e1 = m.speaker_table.weight[1:2].detach().clone()
    eb = 0.5 * e0 + 0.5 * e1
    for film in m.films:
        x = torch.ones(1, film.proj.out_features // 2, 4)
        y0, y1, yb = film(x, e0), film(x, e1), film(x, eb)
        torch.testing.assert_close(yb, 0.5 * y0 + 0.5 * y1, atol=1e-6, rtol=0)

    old_w = m.speaker_table.weight.detach().clone()
    m.expand_speakers(5, init_from_row=1)
    assert m.cfg.n_speakers == 5
    torch.testing.assert_close(m.speaker_table.weight[:3], old_w)
    torch.testing.assert_close(m.speaker_table.weight[3], old_w[1])
    torch.testing.assert_close(m.speaker_table.weight[4], old_w[1])


def test_discriminators_and_losses():
    torch.manual_seed(6)
    disc = vocoder.VocoderDiscriminator(
        periods=(2, 3),
        resolutions=((256, 64, 256), (512, 128, 512)),
    )
    mel_fn = vocoder.MultiResolutionMelLoss()
    real = torch.randn(2, 4 * FS) * 0.1
    fake = (torch.randn(2, 4 * FS) * 0.1).requires_grad_(True)

    zero_mel, _ = mel_fn(real, real)
    assert zero_mel.item() == pytest.approx(0.0, abs=1e-6)
    mel_l, per_mel = mel_fn(real, fake)
    assert mel_l.item() > 0.0 and len(per_mel) == 3

    rs, rf = disc(real)
    fs, ff = disc(fake)
    assert len(rs) == len(fs) == 4
    d_loss = vocoder.discriminator_loss(rs, [s.detach() for s in fs])
    g_loss = vocoder.generator_adv_loss(fs) + 2.0 * vocoder.feature_matching_loss(rf, ff) + 45.0 * mel_l
    g_loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert torch.isfinite(d_loss)


def test_train_resume_finetune_and_targets_exclusion(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    rng = np.random.default_rng(0)
    (tmp_path / "audio").mkdir()
    u_teacher = tmp_path / "units_teacher"
    u_student = tmp_path / "units_student"
    u_teacher.mkdir()
    u_student.mkdir()

    rows, urows_t, urows_s = [], [], []
    speakers = ["lj", "hifitts_92", "lj", "hifitts_92", "lj"]
    for i, spk in enumerate(speakers):
        n = 12 + 2 * i
        t = np.arange(n * FS) / 16000.0
        f0 = 180.0 if spk == "lj" else 220.0
        x = (0.2 * np.sin(2 * np.pi * f0 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        rel = f"audio/u{i}.flac"
        sf.write(tmp_path / rel, x, 16000)
        ids = rng.integers(0, 16, size=n, dtype=np.int16)
        np.save(u_teacher / f"u{i}.npy", ids)
        np.save(u_student / f"u{i}.npy", ids)
        row = {"utt_id": f"u{i}", "path": rel, "duration": len(x) / 16000.0,
               "speaker": spk, "dataset": "fake", "subset": "clean", "gender": "F"}
        rows.append(row)
        urows_t.append({"utt_id": f"u{i}", "units": f"u{i}.npy", "n_frames": n, "path": rel})
        urows_s.append({"utt_id": f"u{i}", "units": f"u{i}.npy", "n_frames": n, "path": rel})

    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (u_teacher / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in urows_t), encoding="utf-8")
    (u_student / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in urows_s), encoding="utf-8")
    meta_t = units.base_meta("teacher-km100", 320, None, 16, "km16:fake", False, "", {})
    meta_s = units.base_meta("student-L1", 320, 1, 16, "km16:fake", True, "", {})
    (u_teacher / units.META_FILE).write_text(json.dumps(meta_t), encoding="utf-8")
    (u_student / units.META_FILE).write_text(json.dumps(meta_s), encoding="utf-8")

    # Mark u4 in targets.tsv and verify join_vocoder_items excludes it.
    targets_tsv = tmp_path / "targets.tsv"
    evallist.write(targets_tsv, [{
        "voice": "lj", "clip_id": "u4", "speaker": "lj", "gender": "F",
        "duration": rows[4]["duration"], "path": rows[4]["path"], "dataset": "fake",
    }], evallist.TARGET_COLUMNS)
    items, _, spk_list = tuv.join_vocoder_items([str(manifest)], [str(u_teacher)], str(targets_tsv))
    assert len(items) == 4 and spk_list == ["lj", "hifitts_92"]

    common = [
        "--train-manifest", str(manifest),
        "--train-units", str(u_teacher),
        "--val-manifest", str(manifest),
        "--val-units", str(u_teacher),
        "--val-every", "2",
        "--exclude-targets", str(targets_tsv),
        "--upsample-initial-channel", "32",
        "--segment-samples", "1280",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
        "--log-every", "1",
        "--run-name", "vtest",
        "--runs-dir", str(tmp_path / "runs"),
        "--ckpt-dir", str(tmp_path / "ckpt"),
    ]
    assert tuv.main(common + ["--steps", "2"]) == 0
    ck = torch.load(tmp_path / "ckpt" / "vtest" / "last.pt", weights_only=True)
    assert ck["step"] == 2
    assert ck["speakers"] == ["lj", "hifitts_92"]
    contract.ModelContract.from_json(ck["contract"])
    voices_disk = json.loads((tmp_path / "ckpt" / "vtest" / "voices.json").read_text(encoding="utf-8"))
    voc_loaded, _ = vocoder.UnitVocoder.from_checkpoint(tmp_path / "ckpt" / "vtest" / "last.pt")
    contract.validate_voices(voices_disk, voc_loaded.contract(1))

    # Resume to step 3
    assert tuv.main(common + ["--steps", "3", "--resume", "auto"]) == 0
    ck3 = torch.load(tmp_path / "ckpt" / "vtest" / "final.pt", weights_only=True)
    assert ck3["step"] == 3

    # Post-gate fine-tune on student-L1 units AND with an added speaker ("me")
    rows_me = rows + [{**rows[0], "utt_id": "u_me", "speaker": "me"}]
    urows_me = urows_s + [{**urows_s[0], "utt_id": "u_me"}]
    m_me = tmp_path / "m_me.jsonl"
    m_me.write_text("".join(json.dumps(r) + "\n" for r in rows_me), encoding="utf-8")
    (u_student / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in urows_me), encoding="utf-8")

    ft_args = [
        "--train-manifest", str(m_me),
        "--train-units", str(u_student),
        "--exclude-targets", str(targets_tsv),
        "--upsample-initial-channel", "32",
        "--segment-samples", "1280",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
        "--steps", "2",
        "--finetune-from", str(tmp_path / "ckpt" / "vtest" / "final.pt"),
        "--run-name", "vft",
        "--runs-dir", str(tmp_path / "runs"),
        "--ckpt-dir", str(tmp_path / "ckpt"),
    ]
    assert tuv.main(ft_args) == 0
    ck_ft = torch.load(tmp_path / "ckpt" / "vft" / "final.pt", weights_only=True)
    assert ck_ft["speakers"] == ["lj", "hifitts_92", "me"]


def test_gate_rules_and_markdown_table():
    rows = [
        {"candidate": "student-L0 s1", "hop_ms": 20.0, "lookahead_frames": 0, "alg_latency_ms": 20.0,
         "compute_p95_ms": 8.0, "projected_t_minus_b_ms": 58.0,
         "wer": 0.085, "cer": 0.03, "tgt_cos": 0.72, "src_cos": 0.15, "utmos": 3.4,
         "f0_pcc": 0.82, "export_ok": True},
        {"candidate": "student-L1 s1", "hop_ms": 20.0, "lookahead_frames": 1, "alg_latency_ms": 40.0,
         "compute_p95_ms": 8.0, "projected_t_minus_b_ms": 78.0,
         "wer": 0.045, "cer": 0.015, "tgt_cos": 0.75, "src_cos": 0.14, "utmos": 3.7,
         "f0_pcc": 0.86, "export_ok": True},
        {"candidate": "student-L2 s1", "hop_ms": 20.0, "lookahead_frames": 2, "alg_latency_ms": 60.0,
         "compute_p95_ms": 8.0, "projected_t_minus_b_ms": 98.0,
         "wer": 0.042, "cer": 0.014, "tgt_cos": 0.76, "src_cos": 0.14, "utmos": 3.75,
         "f0_pcc": 0.87, "export_ok": True},
        {"candidate": "student-L1 s2", "hop_ms": 40.0, "lookahead_frames": 1, "alg_latency_ms": 60.0,
         "compute_p95_ms": 12.0, "projected_t_minus_b_ms": 102.0,
         "wer": 0.045, "cer": 0.015, "tgt_cos": 0.75, "src_cos": 0.14, "utmos": 3.7,
         "f0_pcc": 0.86, "export_ok": True},
    ]
    # student-L0 s1 has WER 8.5% > best (4.2%) + 3.0 pts (7.2%), so it is ineligible;
    # student-L1 s1 (projected T-B 78 ms) beats student-L2 s1 (98 ms) and student-L1 s2 (102 ms).
    annotated, decision = gate_report.apply_gate_rules(rows)
    by_name = {r["candidate"]: r for r in annotated}
    assert not by_name["student-L0 s1"]["eligible"]
    assert by_name["student-L1 s1"]["eligible"]
    assert decision["winner"] == "student-L1 s1"
    md = gate_report.format_gate_markdown(annotated, decision)
    assert "| student-L1 s1 |" in md and "Selected student-L1 s1" in md

    # Tie-break on UTMOS when projected T-B is equal
    ties = [
        {**rows[1], "candidate": "a", "projected_t_minus_b_ms": 75.0, "utmos": 3.5},
        {**rows[1], "candidate": "b", "projected_t_minus_b_ms": 75.0, "utmos": 3.8},
    ]
    _, dec_tie = gate_report.apply_gate_rules(ties)
    assert dec_tie["winner"] == "b"

    # Fallback when nothing is eligible (e.g. compute > 50% of hop)
    over_budget = [{**rows[1], "compute_p95_ms": 15.0}]
    _, dec_fb = gate_report.apply_gate_rules(over_budget)
    assert dec_fb["winner"] is None and len(dec_fb["fallback"]) == 3


def test_export_smoke_and_streaming_chain(tmp_path, small_vocoder):
    torch.manual_seed(0)
    enc = ce.ContentEncoder(ce.EncoderConfig(
        channels=(8, 16, 16, 32, 32), latent_dim=16, head_hidden=32,
        vocab=small_vocoder.cfg.vocab,
    )).eval()
    for spec in ("student-L0:s1", "student-L1:s2"):
        cand = gate_report.parse_candidate_spec(spec)
        smoke = gate_report.export_smoke_test(cand, enc=enc, voc=small_vocoder)
        assert smoke["ok"], smoke
        assert smoke["encoder_units_match"]
        assert smoke["vocoder_max_abs"] <= 1e-4

    voice_entry = {
        "name": "lj",
        "display": "LJ",
        "embedding": small_vocoder.speaker_table.weight[0].detach().tolist(),
        "logf0_mean": features.LOGF0_PRIOR_MEAN,
        "logf0_std": features.LOGF0_PRIOR_STD,
        "energy_mean": features.ENERGY_PRIOR_MEAN,
        "energy_std": features.ENERGY_PRIOR_STD,
        "license": "Public domain",
        "source": "test",
    }
    wav = (0.1 * np.sin(2 * np.pi * 200.0 * np.arange(8 * FS) / 16000.0)).astype(np.float32)
    cand = gate_report.parse_candidate_spec("student-L1:s1")
    y, elapsed = gate_report.convert_utterance_streaming(wav, cand, enc, small_vocoder, voice_entry)
    assert y.shape == (8 * FS,) and np.isfinite(y).all() and elapsed > 0.0

    voices_dict = {"schema": 1, "voices": [voice_entry, {**voice_entry, "name": "v2"}]}
    ctrl = gate_report.controls_readiness_check(
        [wav], cand, enc, small_vocoder, voices_dict,
        shifts_st=(-12.0, 0.0, 12.0), scales=(0.0, 1.0),
    )
    assert ctrl["blend_50_50"]["finite"]
    assert set(ctrl["pitch_shift"]) == {"-12.0", "0.0", "12.0"}
