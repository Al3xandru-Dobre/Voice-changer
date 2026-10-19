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
    m = vocoder.UnitVocoder(small_cfg).eval()
    # Default training init (std=0.01 across ~100 convs) peaks at ~4e-5; boost
    # the test fixture's weight gain so equivalence/parity checks run at O(0.1)
    # amplitude above the 1e-4 absolute tolerance (BUG-017).
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.2)
    return m


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
    assert rep["ref_peak"] > 1e-2 and rep["ref_rms"] > 1e-3
    assert rep["pass"]
    r = rep["fps"][fps]
    assert r["max_abs"] <= 1e-4
    assert r["snr_db"] >= 90.0
    assert r["max_boundary_step"] <= r["max_step_ref"] * 1.001 + 1e-6


def test_vocoder_check_catches_broken_caches(small_vocoder, monkeypatch):
    """BUG-017: dropping transposed-conv or dilated-conv caches must fail check_vocoder."""
    wav = (0.2 * np.sin(2 * np.pi * 180.0 * np.arange(12 * FS) / 16000.0)).astype(np.float32)
    orig_t = vocoder.StreamingConvTranspose1d.step

    def drop_t_cache(self, x, cache):
        y, c = orig_t(self, x, cache)
        return y, torch.zeros_like(c)

    monkeypatch.setattr(vocoder.StreamingConvTranspose1d, "step", drop_t_cache)
    rep_t = stream_check.check_vocoder(small_vocoder, wav, fps_list=(1,), seed=0)
    assert not rep_t["pass"]
    assert rep_t["fps"][1]["snr_db"] < 90.0
    monkeypatch.setattr(vocoder.StreamingConvTranspose1d, "step", orig_t)

    orig_c = vocoder.StreamingConv1d.step

    def drop_dilated_cache(self, x, cache):
        y, c = orig_c(self, x, cache)
        if self.dilation[0] > 1:
            return y, torch.zeros_like(c)
        return y, c

    monkeypatch.setattr(vocoder.StreamingConv1d, "step", drop_dilated_cache)
    rep_d = stream_check.check_vocoder(small_vocoder, wav, fps_list=(1,), seed=0)
    assert not rep_d["pass"]
    assert rep_d["fps"][1]["snr_db"] < 90.0


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

        # Empirical past receptive field: perturbing frame 0 cannot affect samples
        # at or beyond frame_samples + past_context_samples().
        ctx = m.past_context_samples()
        assert ctx == 9696
        n_long = 35
        u_long = torch.zeros((1, n_long), dtype=torch.int64)
        lf_long = torch.full((1, n_long), 5.0, dtype=torch.float64)
        vo_long = torch.ones((1, n_long), dtype=torch.float64)
        ap_long = torch.full((1, n_long), 0.1, dtype=torch.float64)
        en_long = torch.full((1, n_long), -5.0, dtype=torch.float64)
        ref_long = m(u_long, lf_long, vo_long, ap_long, en_long, spk)
        lf_pert = lf_long.clone()
        lf_pert[:, 0] += 1.0
        out_long = m(u_long, lf_pert, vo_long, ap_long, en_long, spk)
        delta = (out_long - ref_long).abs()[0]
        # With dilations (1, 3, 5) the perturbation reaches past sample 8000 (> d=1 reach)
        # and vanishes identically after FS + ctx = 10016 samples.
        assert delta[8000:FS + ctx].max().item() > 0.0
        assert delta[FS + ctx:].max().item() == 0.0


def test_generator_wiring_conditioning_film_mrf_tanh_and_unit_repeat(small_vocoder, small_cfg):
    """Pins generator rules that previously survived mutation testing (F-1):
    speaker embedding reaches audio (FiLM gamma and beta), all four conditioning
    channels are wired in order, MRF averages branches, output tanh bounds
    amplitude, and unit_repeat interleaves rather than tiles."""
    u = torch.randint(0, small_cfg.vocab, (1, 6))
    lf = torch.full((1, 6), 5.0)
    vo = torch.full((1, 6), 0.8)
    ap = torch.full((1, 6), 0.2)
    en = torch.full((1, 6), -5.0)
    e0 = small_vocoder.speaker_table.weight[0:1].detach().clone()
    e1 = small_vocoder.speaker_table.weight[1:2].detach().clone()

    with torch.no_grad():
        y0 = small_vocoder(u, lf, vo, ap, en, e0)
        y1 = small_vocoder(u, lf, vo, ap, en, e1)
        # Speaker embedding must change the generated waveform (kills FiLM bypass).
        assert (y0 - y1).abs().max().item() > 1e-3

        # FiLM beta term: on zero input x=0, FiLM(0, spk) == beta != 0.
        film0 = small_vocoder.films[0]
        x_zero = torch.zeros(1, film0.proj.out_features // 2, 4)
        assert film0(x_zero, e0).abs().max().item() > 1e-4

        # Each of the 4 frame conditioning channels must affect output and map to its
        # designated column in FRAME_COND_INPUTS = (logf0, voicing, aperiodicity, energy).
        y_lf = small_vocoder(u, lf + 1.5, vo, ap, en, e0)
        y_vo = small_vocoder(u, lf, vo * 0.0, ap, en, e0)
        y_ap = small_vocoder(u, lf, vo, ap + 0.7, en, e0)
        y_en = small_vocoder(u, lf, vo, ap, en + 3.0, e0)
        for y_pert in (y_lf, y_vo, y_ap, y_en):
            assert (y_pert - y0).abs().max().item() > 1e-4

        # Verify exact column order [logf0, voicing, aperiodicity, energy] entering cond_net.
        probe_v = vocoder.UnitVocoder(small_cfg).eval()
        for col_idx in range(4):
            probe_v.cond_net.conv1.weight.zero_()
            probe_v.cond_net.conv1.bias.zero_()
            probe_v.cond_net.conv1.weight[:, col_idx, :] = 1.0
            args_base = [torch.zeros(1, 4) for _ in range(4)]
            c_base = probe_v.cond_net(torch.stack(args_base, dim=1))
            for test_idx in range(4):
                args_mod = [t.clone() for t in args_base]
                args_mod[test_idx] = torch.ones(1, 4)
                y_base = probe_v(u[:, :4], *args_base, e0)
                y_mod = probe_v(u[:, :4], *args_mod, e0)
                changed = (y_mod - y_base).abs().max().item() > 0.0
                assert changed is (test_idx == col_idx), f"col {col_idx} vs input {test_idx}"
            assert c_base.abs().max().item() == 0.0

        # CausalMRF averages across branches: with zero conv weights, each ResBlock is
        # identity (x + 0), so MRF(x) must equal x (not len(blocks) * x).
        mrf = vocoder.CausalMRF(8, kernels=(3, 7, 11), dilations=((1, 3, 5),) * 3).eval()
        for p in mrf.parameters():
            p.zero_()
        x_in = torch.ones(1, 8, 16)
        torch.testing.assert_close(mrf(x_in), x_in)

        # Output tanh bounds big post-conv activations to (-1, 1].
        sat_v = vocoder.UnitVocoder(small_cfg).eval()
        sat_v.post.bias.fill_(50.0)
        y_sat = sat_v(u[:, :2], lf[:, :2], vo[:, :2], ap[:, :2], en[:, :2], e0)
        assert y_sat.max().item() <= 1.0 and y_sat.min().item() > 0.99

        # unit_repeat=4 (Mimi 12.5 Hz -> 50 Hz): frame 0 repeats 4x then frame 1 repeats 4x.
        mimi_v = vocoder.UnitVocoder(vocoder.VocoderConfig(
            frame_samples=1280, cond_frame_samples=320, vocab=16,
            unit_dim=8, cond_hidden=8, upsample_initial_channel=16,
        )).eval()
        emb = mimi_v._embed_content(torch.tensor([[1, 2]], dtype=torch.int64))
        assert emb.shape == (1, 8, 8)
        for k in range(4):
            torch.testing.assert_close(emb[0, :, k], mimi_v.unit_emb.weight[1])
            torch.testing.assert_close(emb[0, :, 4 + k], mimi_v.unit_emb.weight[2])


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

    # LSGAN target values: real target is 1.0 and fake target is 0.0 for D, 1.0 for G.
    ones = [torch.ones(2, 10)]
    zeros = [torch.zeros(2, 10)]
    assert vocoder.discriminator_loss(ones, zeros).item() == pytest.approx(0.0, abs=1e-7)
    assert vocoder.discriminator_loss(zeros, zeros).item() == pytest.approx(1.0, abs=1e-6)
    assert vocoder.discriminator_loss(ones, ones).item() == pytest.approx(1.0, abs=1e-6)
    assert vocoder.generator_adv_loss(ones).item() == pytest.approx(0.0, abs=1e-7)
    assert vocoder.generator_adv_loss(zeros).item() == pytest.approx(1.0, abs=1e-6)

    # MRD STFT magnitude must include the imaginary component (spec.real^2 + spec.imag^2):
    # a 90-degree phase shift (sin vs cos at an exact FFT bin) has equal STFT magnitude.
    rd = disc.mrd.discriminators[0]
    t_idx = torch.arange(4 * FS, dtype=torch.float32)
    freq_bin = 16.0 / rd.n_fft
    cos_w = torch.cos(2 * np.pi * freq_bin * t_idx)[None]
    sin_w = torch.sin(2 * np.pi * freq_bin * t_idx)[None]
    mag_cos = rd._mag(cos_w)
    mag_sin = rd._mag(sin_w)
    torch.testing.assert_close(mag_cos[:, :, :, 2:-2], mag_sin[:, :, :, 2:-2], atol=1e-4, rtol=1e-4)


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
    assert tuv.main(common + ["--steps", "2", "--steps-per-epoch", "10", "--seed", "7"]) == 0
    ck = torch.load(tmp_path / "ckpt" / "vtest" / "last.pt", weights_only=True)
    assert ck["step"] == 2
    assert ck["speakers"] == ["lj", "hifitts_92"]
    # LR scheduler must decay with step s: lr * 0.999^(s / steps_per_epoch).
    assert ck["scheduler_g"]["_last_lr"][0] == pytest.approx(2e-4 * (0.999 ** (2.0 / 10.0)), rel=1e-6)
    assert ck["scheduler_g"]["_last_lr"][0] < 2e-4
    contract.ModelContract.from_json(ck["contract"])
    voices_disk = json.loads((tmp_path / "ckpt" / "vtest" / "voices.json").read_text(encoding="utf-8"))
    voc_loaded, _ = vocoder.UnitVocoder.from_checkpoint(tmp_path / "ckpt" / "vtest" / "last.pt")
    contract.validate_voices(voices_disk, voc_loaded.contract(1))
    # Row i of voices.json must carry speaker_table.weight[i] and distinct logf0/energy stats.
    for idx, spk_name in enumerate(["lj", "hifitts_92"]):
        v_entry = voices_disk["voices"][idx]
        assert v_entry["name"] == spk_name
        np.testing.assert_allclose(
            v_entry["embedding"],
            voc_loaded.speaker_table.weight[idx].detach().numpy(),
            rtol=0, atol=1e-6,
        )
        assert v_entry["logf0_std"] == pytest.approx(ck["speaker_stats"][spk_name]["logf0_std"])
        assert v_entry["energy_std"] == pytest.approx(ck["speaker_stats"][spk_name]["energy_std"])
        assert v_entry["logf0_std"] != v_entry["energy_std"]
    assert voices_disk["voices"][0]["embedding"] != voices_disk["voices"][1]["embedding"]

    # Mel loss weight (45.0) must contribute to generator parameter updates (F-1).
    assert tuv.main(common + ["--run-name", "vtest_nomel", "--steps", "1", "--seed", "7", "--mel-weight", "0.0"]) == 0
    ck_nomel = torch.load(tmp_path / "ckpt" / "vtest_nomel" / "last.pt", weights_only=True)
    assert tuv.main(common + ["--run-name", "vtest_mel45", "--steps", "1", "--seed", "7", "--mel-weight", "45.0"]) == 0
    ck_mel45 = torch.load(tmp_path / "ckpt" / "vtest_mel45" / "last.pt", weights_only=True)
    assert not torch.allclose(ck_mel45["model"]["post.weight"], ck_nomel["model"]["post.weight"])

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


def test_gate_rules_and_markdown_table(tmp_path, monkeypatch):
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

    # Pin individual eligibility thresholds (max_wer_abs, max_tgt_cos_drop, min_utmos)
    # and projected_t_minus_b_ms / alg_latency_ms arithmetic (F-1).
    base_row = {
        "candidate": "student-L1 s1", "hop_ms": 20.0, "lookahead_frames": 1,
        "alg_latency_ms": 40.0, "compute_p95_ms": 5.0, "projected_t_minus_b_ms": 75.0,
        "wer": 0.04, "cer": 0.01, "tgt_cos": 0.80, "src_cos": 0.10, "utmos": 3.5,
        "f0_pcc": 0.85, "export_ok": True,
    }
    # tgt_cos drop > 0.05 is rejected, <= 0.05 is accepted.
    ann_cos, _ = gate_report.apply_gate_rules([
        base_row,
        {**base_row, "candidate": "cos_ok", "tgt_cos": 0.751},
        {**base_row, "candidate": "cos_bad", "tgt_cos": 0.749},
    ])
    by_c = {r["candidate"]: r for r in ann_cos}
    assert by_c["cos_ok"]["eligible"] and not by_c["cos_bad"]["eligible"]

    # UTMOS < 3.0 is rejected, >= 3.0 is accepted.
    ann_mos, _ = gate_report.apply_gate_rules([
        {**base_row, "candidate": "mos_ok", "utmos": 3.0},
        {**base_row, "candidate": "mos_bad", "utmos": 2.99},
    ])
    by_m = {r["candidate"]: r for r in ann_mos}
    assert by_m["mos_ok"]["eligible"] and not by_m["mos_bad"]["eligible"]

    # BUG-015: when both PyTorch and ORT clocks are present, ORT decides eligibility and T-B.
    c_l0 = gate_report.parse_candidate_spec("student-L0:s1")
    c_l1 = gate_report.parse_candidate_spec("student-L1:s1")
    c_s2 = gate_report.parse_candidate_spec("student-L1:s2")
    assert c_l0.alg_latency_ms == 20.0 and c_l1.alg_latency_ms == 40.0 and c_s2.alg_latency_ms == 60.0
    assert gate_report.projected_t_minus_b_ms(c_l0, 4.5, io_floor_ms=25.0) == pytest.approx(54.5)
    assert gate_report.projected_t_minus_b_ms(c_l0, 9.5, io_floor_ms=25.0) == pytest.approx(59.5)
    assert gate_report.projected_t_minus_b_ms(c_l0, 4.5, io_floor_ms=30.0) == pytest.approx(59.5)
    dual_rows = [
        {**base_row, "candidate": "student-L0 s1", "hop_ms": 20.0, "lookahead_frames": 0,
         "alg_latency_ms": 20.0, "compute_pytorch_p95_ms": 15.4, "compute_ort_p95_ms": 4.5,
         "compute_p95_ms": 15.4, "projected_t_minus_b_pytorch_ms": 65.4,
         "projected_t_minus_b_ort_ms": 54.5, "projected_t_minus_b_ms": 65.4},
        {**base_row, "candidate": "student-L1 s2", "hop_ms": 40.0, "lookahead_frames": 1,
         "alg_latency_ms": 60.0, "compute_pytorch_p95_ms": 18.7, "compute_ort_p95_ms": 8.2,
         "compute_p95_ms": 18.7, "projected_t_minus_b_pytorch_ms": 108.7,
         "projected_t_minus_b_ort_ms": 98.2, "projected_t_minus_b_ms": 108.7},
    ]
    ann_dual, dec_dual = gate_report.apply_gate_rules(dual_rows)
    assert dec_dual["winner"] == "student-L0 s1"
    assert dec_dual["compute_clock"] == "ort"
    assert dec_dual["projected_t_minus_b_ms"] == pytest.approx(54.5)
    md_dual = gate_report.format_gate_markdown(ann_dual, dec_dual)
    assert "15.4 / **4.5**" in md_dual and "65.4 / **54.5**" in md_dual and "ORT clock" in md_dual

    # BUG-015 item 4: run_gate-level test where PyTorch cost exceeds 50% of hop (15.4 ms > 10 ms)
    # while ORT smoke cost is within budget (4.5 ms <= 10 ms).
    pt_map = {"student-L0 s1": 15.4, "student-L1 s2": 18.7}
    ort_map = {"student-L0 s1": 4.5, "student-L1 s2": 8.2}
    monkeypatch.setattr(gate_report, "export_smoke_test", lambda cand, **kw: {
        "ok": True, "encoder_exporter": "dynamo", "vocoder_exporter": "dynamo",
        "fallback_reason": None, "ort_total_p95_ms": ort_map[cand.name],
        "ort_total_p50_ms": ort_map[cand.name] * 0.8, "ort_total_p99_ms": ort_map[cand.name] * 1.1,
    })
    monkeypatch.setattr(gate_report, "measure_candidate_cost", lambda cand, **kw: {
        "total_p95_ms": pt_map[cand.name], "hop_ms": cand.hop_ms,
    })
    gate_out = tmp_path / "gate_run"
    assert gate_report.main([
        "--skip-eval", "--candidates", "student-L0:s1,student-L1:s2", "--out", str(gate_out)
    ]) == 0
    rep_json = json.loads((gate_out / "gate_report.json").read_text(encoding="utf-8"))
    by_cand = {r["candidate"]: r for r in rep_json["candidates"]}
    assert by_cand["student-L0 s1"]["compute_p95_ms"] == pytest.approx(4.5)
    assert by_cand["student-L0 s1"]["compute_pytorch_p95_ms"] == pytest.approx(15.4)
    assert by_cand["student-L0 s1"]["compute_ort_p95_ms"] == pytest.approx(4.5)
    assert by_cand["student-L0 s1"]["projected_t_minus_b_ms"] == pytest.approx(54.5)
    assert by_cand["student-L0 s1"]["projected_t_minus_b_pytorch_ms"] == pytest.approx(65.4)
    rep_md = (gate_out / "gate_report.md").read_text(encoding="utf-8")
    assert "15.4 / **4.5**" in rep_md and "65.4 / **54.5**" in rep_md


def test_trainer_stats_crops_preroll_validation_and_guards(tmp_path):
    """Pins compute_speaker_stats, VocoderCropDataset unit_repeat scaling,
    extract_cond_crop pre-roll, speaker-stratified validate(), upfront speaker-name
    validation, and LR decay (F-1..F-5)."""
    # 1. Closed-form check of compute_speaker_stats (voiced-only logf0, all-frame energy).
    sr = 16000
    t1 = np.arange(15 * FS) / sr
    # 15 frames at 160 Hz, 15 frames at 240 Hz, 15 frames of near-silence (unvoiced).
    w_160 = (0.3 * np.sin(2 * np.pi * 160.0 * t1)).astype(np.float32)
    w_240 = (0.1 * np.sin(2 * np.pi * 240.0 * t1)).astype(np.float32)
    w_sil = np.zeros(15 * FS, dtype=np.float32)
    full_w = np.concatenate([w_160, w_240, w_sil])
    p_wav = tmp_path / "stats_test.wav"
    sf.write(p_wav, full_w, sr, subtype="FLOAT")

    feats = features.extract(full_w)
    voiced = feats[:, 0] > 0.0
    assert voiced.sum() > 10 and (~voiced).sum() >= 10
    expected_lf_mean = float(np.mean(np.log(feats[voiced, 0])))
    expected_lf_std = float(np.std(np.log(feats[voiced, 0]), ddof=0))
    expected_en_mean = float(np.mean(feats[:, 3]))
    expected_en_std = float(np.std(feats[:, 3], ddof=0))
    voiced_only_en_mean = float(np.mean(feats[voiced, 3]))
    assert abs(expected_en_mean - voiced_only_en_mean) > 1.0

    stats = tuv.compute_speaker_stats([(str(p_wav), "u.npy", len(feats), 0, "lj")], ["lj"])
    assert stats["lj"]["logf0_mean"] == pytest.approx(expected_lf_mean, abs=1e-5)
    assert stats["lj"]["logf0_std"] == pytest.approx(expected_lf_std, abs=1e-5)
    assert stats["lj"]["energy_mean"] == pytest.approx(expected_en_mean, abs=1e-5)
    assert stats["lj"]["energy_std"] == pytest.approx(expected_en_std, abs=1e-5)

    # 2. extract_cond_crop causal pre-roll changes frame 25 vs preroll=0 and matches full extract.
    _, lf_pre, _, _, _ = tuv.extract_cond_crop(str(p_wav), start_cond_frame=20, n_cond_frames=5, preroll=20)
    _, lf_nopre, _, _, _ = tuv.extract_cond_crop(str(p_wav), start_cond_frame=20, n_cond_frames=5, preroll=0)
    np.testing.assert_allclose(lf_pre, feats[20:25, 4].astype(np.float32), atol=1e-6)
    assert np.max(np.abs(lf_pre - lf_nopre)) > 1e-3

    # 3. VocoderCropDataset with unit_repeat=4 scales start_c = start_u * 4.
    ramp_frames = 24
    ramp = np.repeat(np.arange(ramp_frames, dtype=np.float32) / 100.0, FS)
    p_ramp = tmp_path / "ramp.wav"
    sf.write(p_ramp, ramp, sr)
    p_u = tmp_path / "u_mimi.npy"
    np.save(p_u, np.arange(6, dtype=np.int16))
    ds_mimi = tuv.VocoderCropDataset([(str(p_ramp), str(p_u), 6, 0, "lj")], crop_unit_frames=2, unit_repeat=4)
    ds_mimi.rng = np.random.default_rng(1)   # picks start_u > 0
    wav_crop, u_crop, *_ = ds_mimi[0]
    start_u = int(u_crop[0].item())
    assert start_u > 0
    expected_wav = ramp[start_u * 4 * FS:(start_u + 2) * 4 * FS]
    np.testing.assert_allclose(wav_crop.numpy(), expected_wav, atol=1e-4)

    # 4. Speaker-name validation upfront (F-4) and speaker-stratified validation subset (F-3).
    with pytest.raises(ValueError, match="invalid speaker name"):
        tuv.validate_speaker_name("cori samuel")
    with pytest.raises(ValueError, match="invalid speaker name"):
        tuv.validate_speaker_name("john.doe")

    unbalanced = [(str(p_wav), str(p_u), 6, 0, "lj")] * 10 + [(str(p_wav), str(p_u), 6, 1, "hifitts_92")] * 4
    strat = tuv.select_val_items(unbalanced, limit=4)
    assert {it[4] for it in strat} == {"lj", "hifitts_92"}

    # 5. validate() scores pred vs wav (positive loss on untrained model, not 0).
    gen = vocoder.UnitVocoder(vocoder.VocoderConfig(
        vocab=16, unit_dim=8, cond_hidden=8, n_speakers=2, spk_emb_dim=8, upsample_initial_channel=16
    )).eval()
    mel_fn = vocoder.MultiResolutionMelLoss()
    per_spk = {}
    val_loss = tuv.validate(gen, mel_fn, strat, unit_repeat=1, device=torch.device("cpu"),
                            limit=2, per_speaker_out=per_spk)
    assert val_loss > 0.1 and set(per_spk) == {"lj", "hifitts_92"}


def test_export_smoke_and_streaming_chain(tmp_path, small_vocoder, monkeypatch):
    torch.manual_seed(0)
    enc = ce.ContentEncoder(ce.EncoderConfig(
        channels=(8, 16, 16, 32, 32), latent_dim=16, head_hidden=32,
        vocab=small_vocoder.cfg.vocab,
    )).eval()
    for spec in ("student-L0:s1", "student-L1:s2"):
        cand = gate_report.parse_candidate_spec(spec)
        smoke = gate_report.export_smoke_test(cand, enc=enc, voc=small_vocoder,
                                              exporter="torchscript", cost_steps=5, cost_warmup=2)
        assert smoke["ok"], smoke
        assert smoke["smoke_hops"] >= 4
        assert smoke["encoder_units_match"]
        assert smoke["vocoder_max_abs"] <= 1e-4
        assert smoke["vocoder_ref_peak"] > 1e-2

    # BUG-016: when dynamo fails, default exporter='dynamo' must fail (not silently pass),
    # while exporter='auto' records the fallback reason.
    real_export = torch.onnx.export

    def fail_dynamo(*args, dynamo=True, **kw):
        if dynamo:
            raise RuntimeError("simulated dynamo failure")
        return real_export(*args, dynamo=False, **kw)

    monkeypatch.setattr(torch.onnx, "export", fail_dynamo)
    cand_s1 = gate_report.parse_candidate_spec("student-L0:s1")
    smoke_strict = gate_report.export_smoke_test(cand_s1, enc=enc, voc=small_vocoder,
                                                 exporter="dynamo", cost_steps=2, cost_warmup=1)
    assert not smoke_strict["ok"]
    smoke_auto = gate_report.export_smoke_test(cand_s1, enc=enc, voc=small_vocoder,
                                               exporter="auto", cost_steps=2, cost_warmup=1)
    assert smoke_auto["ok"]
    assert smoke_auto["encoder_exporter"] == "torchscript"
    assert "simulated dynamo failure" in smoke_auto["fallback_reason"]
    monkeypatch.setattr(torch.onnx, "export", real_export)

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
    t_step = np.arange(8 * FS) / 16000.0
    f0_traj = np.where(np.arange(8 * FS) < 4 * FS, 150.0, 260.0)
    wav = (0.2 * np.sin(2 * np.pi * np.cumsum(f0_traj) / 16000.0)).astype(np.float32) + 0.0 * t_step.astype(np.float32)
    cand = gate_report.parse_candidate_spec("student-L1:s1")
    seen_la = []
    orig_tracker = features.PitchTracker

    def spy_tracker(cfg=None):
        seen_la.append(cfg.lookahead_frames if cfg is not None else 0)
        return orig_tracker(cfg)

    monkeypatch.setattr(features, "PitchTracker", spy_tracker)
    y0, elapsed = gate_report.convert_utterance_streaming(wav, cand, enc, small_vocoder, voice_entry, shift_st=0.0)
    assert seen_la == [1]
    monkeypatch.setattr(features, "PitchTracker", orig_tracker)
    y12, _ = gate_report.convert_utterance_streaming(wav, cand, enc, small_vocoder, voice_entry, shift_st=12.0)
    assert y0.shape == (8 * FS,) and np.isfinite(y0).all() and elapsed > 0.0
    # Pitch shift must alter the conditioning and therefore the converted waveform (F-1).
    assert np.max(np.abs(y12 - y0)) > 1e-4

    voices_dict = {"schema": 1, "voices": [voice_entry, {**voice_entry, "name": "v2"}]}
    # Pin +12 st ratio tolerance (2.0 +/- 0.06) in controls_readiness_check (F-1).
    monkeypatch.setattr(gate_report.f0_ref, "track",
                        lambda audio_arr, *a, **k: np.full(10, 360.0 if np.allclose(audio_arr, y12) else 180.0))
    ctrl_pass = gate_report.controls_readiness_check(
        [wav], cand, enc, small_vocoder, voices_dict,
        shifts_st=(-12.0, 0.0, 12.0), scales=(0.0, 1.0),
    )
    assert ctrl_pass["blend_50_50"]["finite"]
    assert set(ctrl_pass["pitch_shift"]) == {"-12.0", "0.0", "12.0"}
    assert ctrl_pass["plus12st_ratio"] == pytest.approx(2.0)
    assert ctrl_pass["plus12st_dod_pass"] is True

    monkeypatch.setattr(gate_report.f0_ref, "track",
                        lambda audio_arr, *a, **k: np.full(10, 380.0 if np.allclose(audio_arr, y12) else 180.0))
    ctrl_fail = gate_report.controls_readiness_check(
        [wav], cand, enc, small_vocoder, voices_dict,
        shifts_st=(0.0, 12.0), scales=(1.0,),
    )
    assert ctrl_fail["plus12st_dod_pass"] is False
