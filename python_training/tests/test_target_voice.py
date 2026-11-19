"""Stage 10 target-voice tooling tests — CPU only, synthetic audio, fakes."""
import json
import math
import pathlib
import types

import numpy as np
import pytest
import soundfile as sf
import torch

import extract_units
import export_stateful_onnx as ex
import prepare_target_voice as ptv
import train_unit_vocoder as tuv
from fvml import audio, augment, evallist, target_voice as tv, units

SR48 = 48000


def _burst(seconds, f0=150.0, amp=0.3, sr=SR48, harmonics=6, seed=0):
    """Harmonic 'speech' burst + tiny noise (well above a room-tone floor)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))
    x = x / np.max(np.abs(x)) * amp
    return (x + 0.0005 * rng.standard_normal(len(x))).astype(np.float32)


def _noise(seconds, amp=0.001, sr=SR48, seed=1):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def _take(bursts=((0.0, 3.0), (4.0, 3.0)), gap_noise=0.001, f0=150.0, amp=0.3,
          sr=SR48, seed=0):
    """Bursts [(start_s, dur)] on a low-noise bed, total = last end + 0.5 s."""
    end = max(s + d for s, d in bursts) + 0.5
    x = _noise(end, gap_noise, sr, seed)
    for i, (s, d) in enumerate(bursts):
        a = int(s * sr)
        x[a:a + int(d * sr)] += _burst(d, f0, amp, sr, seed=seed + i)
    return np.clip(x, -1, 1).astype(np.float32)


def _write_session(root, speaker, session, takes, tone_seconds=4.0, tone_amp=0.001):
    for name, x in takes:
        sf.write(root / f"{speaker}_{session}_{name}.wav", x, SR48, subtype="PCM_24")
    sf.write(root / f"{speaker}_{session}_roomtone.wav",
             _noise(tone_seconds, tone_amp, SR48, seed=9), SR48, subtype="PCM_24")


# -- fvml.target_voice ----------------------------------------------------------
def test_parse_take_and_room_tone():
    assert tv.parse_take("me_s01_0001.wav", "me") == ("s01", "0001")
    assert tv.parse_take("a/b/me_s12_9876.wav", "me") == ("s12", "9876")
    assert tv.parse_room_tone("me_s01_roomtone.wav", "me") == "s01"
    with pytest.raises(ValueError, match="expected"):
        tv.parse_take("whatever.wav", "me")
    with pytest.raises(ValueError, match="expected"):
        tv.parse_room_tone("me_s01_0001.wav", "me")
    # Configurable patterns for rigs that rename files.
    assert tv.parse_take("s03_0007.wav", "me",
                         pattern=__import__("re").compile(r"^(s\d{2})_(\d{4})$")) \
        == ("s03", "0007")


def test_highpass_and_loudness(tmp_path):
    sr = SR48
    t = np.arange(sr) / sr
    x = _burst(1.0, 150.0, 0.3, sr) + 0.2 * np.sin(2 * np.pi * 30 * t)
    y = tv.highpass(x, sr, 65.0)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    i30 = int(np.argmin(np.abs(freqs - 30.0)))
    i150 = int(np.argmin(np.abs(freqs - 150.0)))
    spec_x, spec_y = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y))
    assert spec_y[i30] < 0.1 * spec_x[i30]          # 30 Hz component gone
    assert spec_y[i150] > 0.5 * spec_x[i150]        # 150 Hz kept
    yn, loud, gain_db = tv.loudness_normalize(_take(), sr, -23.0)
    import pyloudnorm
    assert pyloudnorm.Meter(sr).integrated_loudness(yn.astype(np.float64)) \
        == pytest.approx(-23.0, abs=0.3)
    with pytest.raises(ValueError, match="silent"):
        tv.loudness_normalize(np.zeros(sr, np.float32), sr)


def test_vad_merge_split_minmax():
    sr = SR48
    # Two bursts 0.5 s apart stay separate; 0.2 s apart merge (min_pause 0.3).
    x = _take(((0.0, 3.0), (3.7, 3.0)), seed=2)
    floor = tv.noise_floor_db(_noise(4.0, 0.001, sr), sr)
    segs = tv.vad_segments(x, sr, floor, min_s=2.0, max_s=12.0)
    assert len(segs) == 2
    for a, b in segs:
        assert 2.0 * sr <= b - a <= 12.0 * sr
    x = _take(((0.0, 3.0), (3.2, 3.0)), seed=3)
    segs = tv.vad_segments(x, sr, floor, min_s=2.0, max_s=12.0)
    assert len(segs) == 1
    # A 30 s region splits into <= max_s pieces at low-energy points.
    x = _take(((0.0, 30.0),), gap_noise=0.02, seed=4)
    segs = tv.vad_segments(x, sr, tv.noise_floor_db(_noise(4, 0.001, sr), sr),
                           min_s=2.0, max_s=12.0)
    assert len(segs) >= 3
    assert all(2.0 * sr <= b - a <= 12.0 * sr + 1 for a, b in segs)
    # A 1 s blip is dropped (min_s = 2).
    x = _take(((0.0, 1.0),), seed=5)
    assert tv.vad_segments(x, sr, floor, min_s=2.0) == []
    # Dead zone (max_s < len < 2*min_s): no low-energy point leaves >= min_s
    # on both sides -> equal-part fallback; parts below min_s are dropped so
    # no kept region is ever over max_s.
    x = _take(((0.0, 9.5),), seed=6)
    segs = tv.vad_segments(x, sr, floor, min_s=5.0, max_s=6.0)
    assert all(b - a <= 6.0 * sr + 1 for a, b in segs)


def test_clip_dod_boundary():
    # A peak just under the threshold rounds to "-0.0" dB in peak_dbfs but
    # must not count as clipped — the DoD compares the unrounded linear peak.
    assert ptv.any_peak_at_or_over([0.99895], 0.999) is False
    assert ptv.any_peak_at_or_over([0.999], 0.999) is True
    assert ptv.any_peak_at_or_over([], 0.999) is False


def test_snr_and_clipping():
    sr = SR48
    floor = tv.noise_floor_db(_noise(4.0, 0.001, sr), sr)
    good = _take(((0.0, 3.0),), seed=6)
    assert tv.estimate_snr_db(good, sr, floor) > 30
    bad = _noise(3.0, 0.004, sr)                  # barely above the floor
    assert tv.estimate_snr_db(bad, sr, floor) < 30
    assert tv.is_clipped(np.array([0.0, 0.999, -0.5]))
    assert not tv.is_clipped(np.array([0.0, 0.99, -0.5]))


def test_select_heldout_deterministic():
    clips = [{"utt_id": f"u{i}", "duration": 10.0,
              "session": "s01", "take": f"{i:04d}", "seg": 0} for i in range(40)]
    a = tv.select_heldout(clips, seconds=100.0)
    assert a == tv.select_heldout(clips, 100.0)
    assert a["mode"] == "take" and not a["guard"]
    assert sum(10.0 for c in clips if c["utt_id"] in a["heldout"]) >= 100.0
    idx = sorted(int(u[1:]) for u in a["heldout"])
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1          # evenly spread


def test_select_heldout_whole_takes():
    # Many short takes -> "take" mode: a held-out take contributes ALL its
    # clips — no take may appear in both train and held-out.
    clips = [{"utt_id": f"u{t}_{s}", "duration": 3.0,
              "session": "s01", "take": f"t{t:02d}", "seg": s}
             for t in range(20) for s in range(2)]
    sel = tv.select_heldout(clips, seconds=12.0)
    assert sel["mode"] == "take" and not sel["guard"]
    held = sel["heldout"]
    take_of = {c["utt_id"]: c["take"] for c in clips}
    held_takes = {take_of[u] for u in held}
    assert all(c["utt_id"] in held for c in clips if c["take"] in held_takes)
    assert sum(c["duration"] for c in clips if c["utt_id"] in held) >= 12.0
    train = {c["utt_id"] for c in clips} - held
    assert not {take_of[u] for u in train} & held_takes


def test_select_heldout_take_tail_single_long_take():
    # One long take: whole-take holdout would hold out everything -> take-tail.
    clips = [{"utt_id": f"u0_{s}", "duration": 5.0,
              "session": "s01", "take": "t00", "seg": s} for s in range(20)]
    sel = tv.select_heldout(clips, seconds=30.0)
    assert sel == tv.select_heldout(clips, 30.0)    # deterministic
    assert sel["mode"] == "take-tail"
    held, guard = sel["heldout"], sel["guard"]
    assert sum(c["duration"] for c in clips if c["utt_id"] in held) >= 30.0
    assert held and guard and not held & guard
    seg_of = {c["utt_id"]: c["seg"] for c in clips}
    # Held-out = contiguous tail of highest segs; guard = the clip just before.
    held_segs = sorted(seg_of[u] for u in held)
    assert held_segs == list(range(held_segs[0], 20))
    assert [seg_of[u] for u in guard] == [held_segs[0] - 1]
    train = {c["utt_id"] for c in clips} - held - guard
    assert train and max(seg_of[u] for u in train) < min(seg_of[u] for u in guard)


def test_select_heldout_take_tail_many_takes_deterministic():
    # 3 takes x 40 s, target 50 s: whole-take holdout picks 80 s > 50 % of the
    # 120 s total -> take-tail splits the target over evenly spaced takes,
    # one guard clip before each held-out tail.
    clips = [{"utt_id": f"u{t}_{s}", "duration": 5.0,
              "session": "s01", "take": f"t{t:02d}", "seg": s}
             for t in range(3) for s in range(8)]
    sel = tv.select_heldout(clips, seconds=50.0)
    assert sel == tv.select_heldout(clips, 50.0)
    assert sel["mode"] == "take-tail"
    held, guard = sel["heldout"], sel["guard"]
    assert sum(c["duration"] for c in clips if c["utt_id"] in held) >= 50.0
    assert not held & guard
    take_of = {c["utt_id"]: c["take"] for c in clips}
    seg_of = {c["utt_id"]: c["seg"] for c in clips}
    held_takes = {take_of[u] for u in held}
    for take in held_takes:
        hs = sorted(seg_of[u] for u in held if take_of[u] == take)
        assert hs == list(range(hs[0], 8))           # contiguous tail
        assert hs[0] - 1 in {seg_of[u] for u in guard}
    train = {c["utt_id"] for c in clips} - held - guard
    assert train


def test_select_heldout_too_few_segments():
    one = [{"utt_id": "u0", "duration": 5.0,
            "session": "s01", "take": "t00", "seg": 0}]
    with pytest.raises(ValueError, match="at least 2"):
        tv.select_heldout(one, seconds=1.0)
    with pytest.raises(ValueError, match="at least 2"):
        tv.select_heldout([], seconds=1.0)


# -- prepare_target_voice end-to-end ----------------------------------------------
def _make_input(tmp_path, speaker="me"):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_session(raw, speaker, "s01", [("0001", _take(seed=10)),
                                         ("0002", _take(seed=11))])
    clipped = _take(seed=12)
    clipped[1000:2000] = 1.0                      # raw full-scale burst
    _write_session(raw, speaker, "s02", [("0001", clipped),
                                         # SNR ~20 dB vs the 0.001 room tone
                                         ("0002", _noise(7.0, 0.01, SR48, seed=13))])
    return raw


def test_prepare_target_voice_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    raw = _make_input(tmp_path)
    out = tmp_path / "out"
    rc = ptv.main(["--input", str(raw), "--speaker", "me", "--out", str(out),
                   "--min-minutes", "0.05", "--heldout-s", "3", "--min-clip-s", "2"])
    assert rc == 0
    train = (out / "train.jsonl").read_text().strip().splitlines()
    held = (out / "heldout.jsonl").read_text().strip().splitlines()
    rejected = [json.loads(l) for l in (out / "rejected.jsonl").read_text().splitlines()]
    assert train and held
    reasons = {r["reason"] for r in rejected}
    assert "clipped" in reasons and "low_snr" in reasons
    import pyloudnorm
    for line in train + held:
        r = json.loads(line)
        x, sr = sf.read(out / "clips" / f"{r['utt_id']}.flac")
        assert sr == audio.SAMPLE_RATE
        assert 2.0 <= len(x) / sr <= 12.5
        assert pyloudnorm.Meter(sr).integrated_loudness(x.astype(np.float64)) \
            == pytest.approx(-23.0, abs=2.0)
        assert r["dataset"] == "own_voice" and r["speaker"] == "me"
    targets = evallist.read_targets(out / "me_targets.tsv", "me")
    assert targets and {r["clip_id"] for r in targets} == \
        {json.loads(l)["utt_id"] for l in held}
    meta = json.loads((out / "speaker_metadata.json").read_text())
    assert meta["me"]["license"].startswith("private")
    report = json.loads((out / "report.json").read_text())
    assert all(d["pass"] for d in report["dod"])
    assert sum(r["duration"] for r in targets) >= 3.0


def test_prepare_target_voice_dod_exit_code_and_live_mic(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    raw = _make_input(tmp_path)
    out = tmp_path / "out2"
    # Too few kept minutes and too little held-out -> DoD fails without --allow-short.
    rc = ptv.main(["--input", str(raw), "--speaker", "me", "--out", str(out),
                   "--min-minutes", "60", "--heldout-s", "300", "--min-clip-s", "2"])
    assert rc == 1
    rc = ptv.main(["--input", str(raw), "--speaker", "me", "--out", str(out),
                   "--min-minutes", "60", "--heldout-s", "300", "--min-clip-s", "2",
                   "--allow-short"])
    assert rc == 0
    live = tmp_path / "live"
    rc = ptv.main(["--input", str(raw), "--speaker", "me", "--out", str(live),
                   "--live-mic", "--min-clip-s", "2"])
    assert rc == 0
    rows = evallist.read_sources(live / "live_mic.tsv")
    assert rows and all(r["text"] == "" for r in rows)
    assert not (live / "heldout.jsonl").exists()
    # Noisy take kept under --live-mic (no SNR rejection) but clipped still dropped.
    rejected = [json.loads(l) for l in (live / "rejected.jsonl").read_text().splitlines()]
    assert all(r["reason"].startswith("clipped") for r in rejected)


# -- perturbation ------------------------------------------------------------------
def test_should_perturb_fraction_and_rng():
    n = sum(augment.should_perturb(f"u{i}", 0.5, seed=0) for i in range(400))
    assert 400 * 0.42 <= n <= 400 * 0.58
    assert augment.should_perturb("u0", 1.0)
    assert not augment.should_perturb("u0", 0.0)
    r1, r2 = augment.perturb_rng("u5", 0), augment.perturb_rng("u5", 0)
    assert r1.random() == r2.random()


def test_perturb_speaker_length_determinism_and_pitch():
    x = _burst(2.0, 160.0, 0.3, 16000)
    cfg = augment.PerturbConfig()
    y1, p1 = augment.perturb_speaker(x, augment.perturb_rng("u1", 0), 16000, cfg)
    y2, p2 = augment.perturb_speaker(x, augment.perturb_rng("u1", 0), 16000, cfg)
    assert len(y1) == len(x) and y1.dtype == np.float32
    np.testing.assert_array_equal(y1, y2)
    assert p1 == p2
    from fvml import f0_ref
    med_out, _ = f0_ref.stats(f0_ref.track(y1))
    measured_st = 12.0 * math.log2(med_out / p1["source_median_hz"])
    assert measured_st == pytest.approx(p1["pitch_st"], abs=0.5)
    assert abs(p1["formant_ratio"] - 1.0) >= 0.10


def test_extract_units_perturbation(tmp_path, monkeypatch):
    class FakeTok:
        meta = units.base_meta("fake", 320, 0, 16, "fake:0", True, "", {})

        def encode(self, wav):
            return np.arange(len(wav) // 320, dtype=np.int16), None

    monkeypatch.setattr(units, "make_tokenizer", lambda *a, **k: FakeTok())
    wav = _burst(1.0, 150.0, 0.3, 16000)
    rows = []
    for i in range(6):
        p = tmp_path / f"a{i}.wav"
        sf.write(p, wav, 16000)
        rows.append({"utt_id": f"u{i}", "path": str(p), "duration": 1.0,
                     "speaker": "me", "dataset": "own_voice", "subset": "train"})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "units"
    args = types.SimpleNamespace(
        out_dir=str(out), tokenizer="fake", centroids=None, checkpoint=None,
        device="cpu", save_logits=False, limit=None,
        perturb_fraction=1.0, perturb_seed=0, perturb_formant="0.10,0.15",
        perturb_pitch_st=2.0)
    assert extract_units.extract(args, rows, [manifest]) == 0
    urows = [json.loads(l) for l in (out / units.MANIFEST_FILE).read_text().splitlines()]
    assert all(r.get("perturbed") is True and r.get("perturb") for r in urows)
    assert all(r["path"].endswith(".wav") for r in urows)   # clean path kept
    meta = json.loads((out / units.META_FILE).read_text())
    assert meta["perturbation"]["conditioning"] == "clean"
    # Rerun with a different perturbation config -> UnitsMismatch.
    args.perturb_fraction = 0.0
    with pytest.raises(units.UnitsMismatch):
        extract_units.extract(args, rows, [manifest])


# -- trainer fine-tune mode ----------------------------------------------------------
def test_mixed_batch_sampler_homogeneity_and_ratio():
    weights = [1.0] * 20
    mask = [i < 6 for i in range(20)]
    s = tuv.MixedBatchSampler(weights, mask, batch_size=4, ratio=0.5, seed=0)
    it = iter(s)
    in_batches = 0
    for _ in range(400):
        b = next(it)
        assert all(mask[i] for i in b) or all(not mask[i] for i in b)
        in_batches += all(mask[i] for i in b)
    assert 400 * 0.42 <= in_batches <= 400 * 0.58
    with pytest.raises(ValueError):
        tuv.MixedBatchSampler(weights, [True] * 20, 4, 0.5, 0)
    with pytest.raises(ValueError):
        tuv.MixedBatchSampler(weights, [False] * 20, 4, 0.5, 0)


def test_nearest_speaker(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    rng = np.random.default_rng(0)
    waves = {}
    for i, (name, f0) in enumerate((("lj", 180.0), ("hifitts_92", 220.0), ("me", 225.0))):
        t = np.arange(16000) / 16000.0
        x = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.001 * rng.standard_normal(16000)
        p = tmp_path / f"{name}.wav"
        sf.write(p, x.astype(np.float32), 16000)
        waves[name] = str(p)

    def fake_embed(ws):
        out = []
        for w in ws:
            zcr = float(np.mean(np.abs(np.diff(np.sign(w)))))
            out.append([zcr * 50.0, float(np.std(w))])
        e = np.asarray(out, np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    by_spk = {"lj": [(waves["lj"],)], "hifitts_92": [(waves["hifitts_92"],)]}
    name, cosines = tuv.nearest_speaker([(waves["me"],)], by_spk, fake_embed)
    assert name == "hifitts_92"
    assert cosines["hifitts_92"] > cosines["lj"]
    # An empty new speaker is a clear error naming it.
    with pytest.raises(ValueError, match="'me'"):
        tuv.nearest_speaker([], by_spk, fake_embed, speaker_name="me")
    # Existing speakers with no items are skipped; if none remain it errors.
    skipped = {"lj": [], "hifitts_92": [(waves["hifitts_92"],)]}
    name2, cos2 = tuv.nearest_speaker([(waves["me"],)], skipped, fake_embed)
    assert name2 == "hifitts_92" and "lj" not in cos2
    with pytest.raises(ValueError, match="no existing speaker"):
        tuv.nearest_speaker([(waves["me"],)], {"lj": []}, fake_embed)


def test_early_stopper():
    s = tuv.EarlyStopper(2)
    assert s.update(1.0)          # mel improves
    assert s.update(1.5) is False and s.since == 1
    assert s.update(0.9)          # mel improves again -> reset
    assert s.update(1.0) is False and s.since == 1
    assert s.update(1.0) is False and s.since == 2 and s.stop
    s = tuv.EarlyStopper(2)
    s.update(1.0, 3.0)
    assert s.update(1.1, 3.02)    # utmos improves by > 0.01 despite worse mel
    assert s.update(1.1, 3.02) is False and s.since == 1
    assert tuv.EarlyStopper(0).stop is False     # patience 0 = off


def _mini_vocoder_run(tmp_path, monkeypatch, extra_args, steps):
    """Two-speaker base checkpoint + 'me' fine-tune, modelled on
    test_train_resume_finetune_and_targets_exclusion."""
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    rng = np.random.default_rng(0)
    (tmp_path / "audio").mkdir(exist_ok=True)
    udir = tmp_path / "units"
    udir.mkdir(exist_ok=True)
    rows, urows = [], []
    speakers = ["lj", "hifitts_92", "lj", "hifitts_92", "me"]
    for i, spk in enumerate(speakers):
        n = 16
        t = np.arange(n * 320) / 16000.0
        x = (0.2 * np.sin(2 * np.pi * (180 + 20 * i) * t)
             + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        rel = f"audio/u{i}.flac"
        sf.write(tmp_path / rel, x, 16000)
        np.save(udir / f"u{i}.npy", rng.integers(0, 16, size=n).astype(np.int16))
        row = {"utt_id": f"u{i}", "path": rel, "duration": len(x) / 16000.0,
               "speaker": spk, "dataset": "fake", "subset": "clean", "gender": "F"}
        rows.append(row)
        urows.append({"utt_id": f"u{i}", "units": f"u{i}.npy", "n_frames": n, "path": rel})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (udir / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in urows))
    (udir / units.META_FILE).write_text(
        json.dumps(units.base_meta("fake", 320, 0, 16, "fake:0", True, "", {})))

    def run(name, spk_rows, extra):
        m = tmp_path / f"m_{name}.jsonl"
        m.write_text("".join(json.dumps(r) + "\n" for r in spk_rows))
        return tuv.main([
            "--train-manifest", str(m), "--train-units", str(udir),
            "--val-manifest", str(m), "--val-units", str(udir),
            "--exclude-targets", "",
            "--upsample-initial-channel", "32", "--segment-samples", "1280",
            "--batch-size", "2", "--workers", "0", "--device", "cpu",
            "--steps", str(steps), "--val-every", "1", "--log-every", "1",
            "--run-name", name, "--runs-dir", str(tmp_path / "runs"),
            "--ckpt-dir", str(tmp_path / "ckpt"), *extra])

    assert run("base", [r for r in rows if r["speaker"] != "me"], []) == 0
    return rows, manifest


def test_finetune_new_speaker_row_mix_metadata_and_utmos(tmp_path, monkeypatch):
    rows, _ = _mini_vocoder_run(tmp_path, monkeypatch, [], steps=2)
    meta_file = tmp_path / "meta.json"
    meta_file.write_text(json.dumps(
        {"me": {"display": "Me (private)", "license": "private — do not distribute",
                "source": "own recordings (M2 Stage 10)"}}))
    monkeypatch.setattr(tuv, "UTMOS_FN", lambda waves: [3.5] * len(waves))
    monkeypatch.setattr(tuv, "EMBED_FN",
                        lambda waves: np.tile(np.array([1.0, 0.0], np.float32),
                                              (len(waves), 1)))
    rc = tuv.main([
        "--train-manifest", str(tmp_path / "m.jsonl"),
        "--train-units", str(tmp_path / "units"),
        "--val-manifest", str(tmp_path / "m.jsonl"),
        "--val-units", str(tmp_path / "units"),
        "--exclude-targets", "",
        "--upsample-initial-channel", "32", "--segment-samples", "1280",
        "--batch-size", "2", "--workers", "0", "--device", "cpu",
        "--steps", "0", "--val-every", "1",
        "--finetune-from", str(tmp_path / "ckpt" / "base" / "final.pt"),
        "--new-speaker-init", "hifitts_92",
        "--mix-speakers", "me", "--mix-ratio", "0.5",
        "--speaker-metadata", str(meta_file),
        "--early-stop-patience", "1", "--val-utmos",
        "--run-name", "ft0", "--runs-dir", str(tmp_path / "runs"),
        "--ckpt-dir", str(tmp_path / "ckpt")])
    assert rc == 0
    ck = torch.load(tmp_path / "ckpt" / "ft0" / "final.pt", weights_only=True)
    assert ck["speakers"] == ["lj", "hifitts_92", "me"]
    assert ck["speaker_init"]["me"]["from"] == "hifitts_92"
    base_w = torch.load(tmp_path / "ckpt" / "base" / "final.pt",
                        weights_only=True)["model"]["speaker_table.weight"]
    torch.testing.assert_close(ck["model"]["speaker_table.weight"][2],
                               base_w[1])          # me row = hifitts_92 row at init
    voices = json.loads((tmp_path / "ckpt" / "ft0" / "voices.json").read_text())
    assert voices["voices"][2]["name"] == "me"
    assert voices["voices"][2]["license"] == "private — do not distribute"
    assert ck["speaker_metadata"]["me"]["display"] == "Me (private)"

    # A short real run with UTMOS validation + early-stop machinery.
    rc = tuv.main([
        "--train-manifest", str(tmp_path / "m.jsonl"),
        "--train-units", str(tmp_path / "units"),
        "--val-manifest", str(tmp_path / "m.jsonl"),
        "--val-units", str(tmp_path / "units"),
        "--exclude-targets", "",
        "--upsample-initial-channel", "32", "--segment-samples", "1280",
        "--batch-size", "2", "--workers", "0", "--device", "cpu",
        "--steps", "2", "--val-every", "1",
        "--finetune-from", str(tmp_path / "ckpt" / "base" / "final.pt"),
        "--new-speaker-init", "nearest",
        "--speaker-metadata", str(meta_file),
        "--early-stop-patience", "1", "--val-utmos",
        "--run-name", "ft2", "--runs-dir", str(tmp_path / "runs"),
        "--ckpt-dir", str(tmp_path / "ckpt")])
    assert rc == 0
    ck2 = torch.load(tmp_path / "ckpt" / "ft2" / "final.pt", weights_only=True)
    assert "best_utmos" in ck2 and "stopped_early" in ck2
    assert "early_stop_since" in ck2
    assert ck2["speaker_init"]["me"]["from"] in ("lj", "hifitts_92")


def test_resume_restores_early_stop_and_speaker_init(tmp_path, monkeypatch):
    rows, _ = _mini_vocoder_run(tmp_path, monkeypatch, [], steps=2)
    monkeypatch.setattr(tuv, "UTMOS_FN", lambda waves: [3.5] * len(waves))

    def args_for(run_name, steps, extra):
        return [
            "--train-manifest", str(tmp_path / "m.jsonl"),
            "--train-units", str(tmp_path / "units"),
            "--val-manifest", str(tmp_path / "m.jsonl"),
            "--val-units", str(tmp_path / "units"),
            "--exclude-targets", "",
            "--upsample-initial-channel", "32", "--segment-samples", "1280",
            "--batch-size", "2", "--workers", "0", "--device", "cpu",
            "--steps", str(steps), "--val-every", "1",
            "--early-stop-patience", "5", "--val-utmos",
            "--run-name", run_name, "--runs-dir", str(tmp_path / "runs"),
            "--ckpt-dir", str(tmp_path / "ckpt"), *extra]

    assert tuv.main(args_for("ft", 2, [
        "--finetune-from", str(tmp_path / "ckpt" / "base" / "final.pt"),
        "--new-speaker-init", "hifitts_92"])) == 0
    ck = torch.load(tmp_path / "ckpt" / "ft" / "final.pt", weights_only=True)
    assert ck["speaker_init"]["me"]["from"] == "hifitts_92"
    since_before = ck["early_stop_since"]

    # Resume adds no rows, so speaker_init must survive from the prior
    # checkpoint and the EarlyStopper state is restored, not reset.
    assert tuv.main(args_for("ft", 4, ["--resume", "auto"])) == 0
    ck2 = torch.load(tmp_path / "ckpt" / "ft" / "final.pt", weights_only=True)
    assert ck2["speaker_init"]["me"]["from"] == "hifitts_92"
    assert ck2["best_utmos"] == ck["best_utmos"]
    assert ck2["early_stop_since"] >= since_before - 1  # restored, not reset to 0


# -- export private licence ----------------------------------------------------------
def test_export_private_voice_banner_and_flag(tmp_path):
    report = {"roles": {}, "generated": "2026-11-14T00:00:00+00:00", "pairs": {},
              "versions": {},
              "voices": [{"name": "me", "display": "Me (private)",
                          "source": "own recordings (M2 Stage 10)",
                          "license": "private — do not distribute"}]}
    assert ex.voices_private(report)
    card = ex.model_card(tmp_path / "bundle", report, None)
    assert "PRIVATE — contains a personal voice; not for distribution" in card
    report["voices"][0]["license"] = "MIT"
    assert not ex.voices_private(report)
    card = ex.model_card(tmp_path / "bundle", report, None)
    assert "PRIVATE" not in card


# -- Milestone 3 / Stage 10: the voice's language (--lang) ---------------------------
def _prep_args(raw, out, *extra):
    return ["--input", str(raw), "--speaker", "me", "--out", str(out),
            "--min-minutes", "0.05", "--heldout-s", "3", "--min-clip-s", "2", *extra]


def test_prepare_target_voice_lang_tags_rows_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    raw = _make_input(tmp_path)
    out = tmp_path / "out_es"
    assert ptv.main(_prep_args(raw, out, "--lang", "es")) == 0
    rows = [json.loads(line) for name in ("train.jsonl", "heldout.jsonl")
            for line in (out / name).read_text().splitlines()]
    assert rows and all(r["lang"] == "es" for r in rows)
    meta = json.loads((out / "speaker_metadata.json").read_text())
    assert meta["me"]["lang"] == "es"
    assert meta["me"]["license"] == "private — do not distribute"
    assert json.loads((out / "report.json").read_text())["lang"] == "es"
    assert "- language: es" in (out / "report.md").read_text()
    # Default English; the live-mic rows are tagged too.
    out_en = tmp_path / "out_en"
    assert ptv.main(_prep_args(raw, out_en)) == 0
    assert json.loads((out_en / "speaker_metadata.json").read_text())["me"]["lang"] == "en"
    live = tmp_path / "live"
    assert ptv.main(["--input", str(raw), "--speaker", "me", "--out", str(live),
                     "--live-mic", "--min-clip-s", "2", "--lang", "fr"]) == 0
    assert all(json.loads(line)["lang"] == "fr"
               for line in (live / "live_mic.jsonl").read_text().splitlines())
    # The voices.json rule (lowercase [a-z-]{2,8}) is enforced up front.
    for bad in ("English", "EN", "e", "en_us"):
        assert ptv.main(_prep_args(raw, tmp_path / f"bad_{bad}", "--lang", bad)) == 1
        assert not (tmp_path / f"bad_{bad}" / "train.jsonl").exists()


def test_target_voice_lang_reaches_the_exported_voices_json(tmp_path, monkeypatch):
    """prepare_target_voice --lang -> train_unit_vocoder fine-tune -> checkpoint
    -> the Stage 06 export's voices.json: the "me" entry carries its language."""
    from fvml import export as fx, vocoder
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 2))        # tiny model: do not thrash a busy box
    try:
        _mini_vocoder_run(tmp_path, monkeypatch, [], steps=2)
        prep = tmp_path / "prep"
        assert ptv.main(_prep_args(_make_input(tmp_path), prep)) == 0
        train_rows = [json.loads(line) for line in (prep / "train.jsonl").read_text().splitlines()]
        udir = tmp_path / "units_me"
        udir.mkdir()
        rng = np.random.default_rng(3)
        urows = []
        for r in train_rows:              # stand-in units: one id per 20 ms frame
            n = math.ceil(round(r["duration"] * audio.SAMPLE_RATE) / 320)
            np.save(udir / f"{r['utt_id']}.npy", rng.integers(0, 16, size=n).astype(np.int16))
            urows.append({"utt_id": r["utt_id"], "units": f"{r['utt_id']}.npy", "n_frames": n})
        (udir / units.MANIFEST_FILE).write_text("".join(json.dumps(u) + "\n" for u in urows))
        (udir / units.META_FILE).write_text(
            json.dumps(units.base_meta("fake", 320, 0, 16, "fake:0", True, "", {})))
        assert tuv.main([
            "--train-manifest", f"{tmp_path / 'm_base.jsonl'},{prep / 'train.jsonl'}",
            "--train-units", f"{tmp_path / 'units'},{udir}", "--exclude-targets", "",
            "--upsample-initial-channel", "32", "--segment-samples", "1280",
            "--batch-size", "2", "--workers", "0", "--device", "cpu", "--steps", "0",
            "--finetune-from", str(tmp_path / "ckpt" / "base" / "final.pt"),
            "--new-speaker-init", "hifitts_92", "--mix-speakers", "me",
            "--speaker-metadata", str(prep / "speaker_metadata.json"),
            "--run-name", "ft_lang", "--runs-dir", str(tmp_path / "runs"),
            "--ckpt-dir", str(tmp_path / "ckpt")]) == 0
    finally:
        torch.set_num_threads(threads)
    final = tmp_path / "ckpt" / "ft_lang" / "final.pt"
    model, ck = vocoder.UnitVocoder.from_checkpoint(final)
    assert ck["speakers"] == ["lj", "hifitts_92", "me"]
    assert ck["speaker_metadata"]["me"]["lang"] == "en"
    exported = {v["name"]: v for v in fx.voices_json(model, ck)["voices"]}
    assert exported["me"]["lang"] == "en"
    assert exported["me"]["license"] == "private — do not distribute"
    assert "lang" not in exported["lj"]             # untagged M2 corpora stay untagged
    on_disk = json.loads((final.parent / "voices.json").read_text())
    assert [v.get("lang") for v in on_disk["voices"]] == [None, None, "en"]
