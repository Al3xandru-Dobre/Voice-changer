"""Causal pitch & energy feature tests (Milestone 2 / Stage 03), Python side.

The C++ twin is tests/test_pitch_tracker.cpp; both check the golden vectors in
tests/data/pitch/ (written by make_pitch_golden.py).

    uv run pytest tests/test_features.py          # from python_training/
"""
import json
import math
import pathlib

import numpy as np
import pytest

from fvml import features as F
from fvml import paths

GOLDEN = paths.REPO_ROOT / "tests" / "data" / "pitch"
SR = 16000


def golden_header():
    return json.loads((GOLDEN / "header.json").read_text(encoding="utf-8"))


def golden_input(name):
    return np.fromfile(GOLDEN / f"{name}.pcm16", dtype="<i2").astype(np.float64) / 32768.0


def tone(freq, seconds=1.0, amp=0.5):
    return amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR)


def test_header_matches_the_code():
    head = golden_header()
    head.pop("cases")
    head.pop("libri_utterance")
    assert head == json.loads(json.dumps(F.header())), \
        "constants changed: run make_pitch_golden.py and update src/dsp/PitchTracker.hpp"


@pytest.mark.parametrize("case", golden_header()["cases"])
def test_golden_vectors(case):
    name, la = case.rsplit("_L", 1)
    want = np.fromfile(GOLDEN / f"{case}.f32", dtype="<f4").reshape(-1, len(F.FEATURES))
    got = F.extract(golden_input(name), F.PitchConfig(lookahead_frames=int(la))).astype(np.float32)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("la", [0, 1, 2])
def test_streaming_equals_whole_array(la):
    cfg = F.PitchConfig(lookahead_frames=la)
    x = golden_input("libri")
    whole = F.extract(x, cfg)
    tracker = F.PitchTracker(cfg)
    hop = cfg.frame_samples
    padded = np.zeros((len(whole) + la) * hop)
    padded[: len(x)] = x
    frames = [tracker.push(padded[i * hop:(i + 1) * hop]).as_tuple() for i in range(len(whole) + la)]
    np.testing.assert_array_equal(np.asarray(frames[la:]), whole)


@pytest.mark.parametrize("la", [0, 1, 2])
def test_causal(la):
    """Frame t depends on nothing after the end of frame t + L."""
    cfg = F.PitchConfig(lookahead_frames=la)
    x = golden_input("libri")
    base = F.extract(x, cfg)
    t = 40
    y = x.copy()
    y[(t + 1 + la) * cfg.frame_samples:] = np.random.default_rng(0).uniform(-1, 1, len(y) - (t + 1 + la) * 320)
    changed = F.extract(y, cfg)
    np.testing.assert_array_equal(changed[: t + 1], base[: t + 1])
    assert not np.array_equal(changed[t + 1:], base[t + 1:])


@pytest.mark.parametrize("freq", [60.0, 100.0, 220.0, 440.0, 900.0])
def test_tone_f0(freq):
    o = F.extract(tone(freq))
    steady = o[5:]
    assert (steady[:, 0] > 0).all()
    assert np.abs(steady[:, 0] / freq - 1).max() < 0.005
    assert (steady[:, 2] == 1.0).all()
    assert steady[:, 1].max() < 0.05


def test_silence_and_noise_are_unvoiced():
    o = F.extract(np.zeros(SR))
    assert (o[:, 0] == 0).all() and (o[:, 1] == 1.0).all() and (o[:, 2] == 0).all()
    assert np.allclose(o[:, 3], math.log(F.PitchConfig().energy_eps))
    noise = 0.1 * np.random.default_rng(1).standard_normal(SR)
    o = F.extract(noise)
    assert (o[5:, 0] == 0).mean() > 0.95
    assert o[5:, 1].min() > 0.4


def test_held_logf0_relaxes_to_the_running_mean():
    cfg = F.PitchConfig()
    x = np.concatenate([tone(300.0, 0.5), np.zeros(SR)])
    o = F.extract(x, cfg)
    voiced_end = np.flatnonzero(o[:, 0] > 0).max()
    held = o[voiced_end + 1:, 4]
    assert held[0] < math.log(300.0) + 1e-9
    assert np.all(np.diff(held) < 0)            # moving monotonically towards the mean
    assert held[-1] - held[-2] > -1e-3          # and slowing down near it


def test_energy_gain_invariance():
    """Whitened energy doesn't depend on mic gain once the EMA has adapted."""
    x = np.tile(golden_input("libri"), 30)      # ~55 s
    z1 = F.extract(x)[:, 6]
    z2 = F.extract(0.1 * x)[:, 6]               # -20 dB
    tail = slice(-1000, None)                   # last 20 s: > 8 EMA time constants in
    assert np.abs(z1[tail] - z2[tail]).max() < 0.02


def test_whitening_converges_on_ar1():
    mu, sd, phi = math.log(120.0), 0.2, 0.9
    sigma_e = sd * math.sqrt(1 - phi ** 2)
    cfg = F.PitchConfig()
    norm = F.RunningNorm(cfg.logf0_prior_mean, cfg.logf0_prior_std, cfg.norm_alpha, cfg.logf0_std_floor)
    rng = np.random.default_rng(7)
    x = mu
    for _ in range(1000):                       # 20 s: the prior is forgotten
        x = mu + phi * (x - mu) + sigma_e * rng.standard_normal()
        norm.update(x)
    assert abs(norm.mean - mu) < 0.1
    means, stds, zs = [], [], []
    for _ in range(10000):                      # 200 s
        x = mu + phi * (x - mu) + sigma_e * rng.standard_normal()
        norm.update(x)
        means.append(norm.mean)
        stds.append(norm.std)
        zs.append(norm.z(x))
    assert abs(np.mean(means) - mu) < 0.02
    assert abs(np.mean(stds) - sd) < 0.15 * sd
    assert abs(np.sqrt(np.mean(np.square(zs))) - 1.0) < 0.15


def test_std_floor():
    cfg = F.PitchConfig()
    norm = F.RunningNorm(1.0, 0.0, cfg.norm_alpha, cfg.logf0_std_floor)
    for _ in range(100):
        norm.update(1.0)
    assert norm.std == cfg.logf0_std_floor


def test_map_to_target_closed_form():
    voice = {"logf0_mean": math.log(200.0), "logf0_std": 0.15, "energy_mean": -6.0, "energy_std": 2.0}
    lf, e = F.map_to_target(0.0, 1.5, voice)
    assert lf == pytest.approx(math.log(200.0), abs=1e-12)
    assert e == pytest.approx(-6.0 + 2.0 * 1.5, abs=1e-12)
    assert math.exp(F.map_to_target(0.0, 0.0, voice, shift_st=12.0)[0]) == pytest.approx(400.0)
    assert math.exp(F.map_to_target(0.0, 0.0, voice, shift_st=-12.0)[0]) == pytest.approx(100.0)
    lf, _ = F.map_to_target(2.0, 0.0, voice, scale=0.5, shift_st=3.0)
    assert lf == pytest.approx(math.log(200.0) + 0.5 * 0.15 * 2.0 + 3.0 * math.log(2) / 12, abs=1e-12)
    flat = F.map_to_target(np.array([-1.7, 2.3]), np.zeros(2), voice, scale=0.0)[0]
    assert flat[0] == flat[1]


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        F.PitchConfig(lookahead_frames=-1)
    with pytest.raises(ValueError):
        F.PitchConfig(f0_threshold=0.12)
    with pytest.raises(ValueError):
        F.PitchConfig(window_samples=200)
    with pytest.raises(ValueError):
        F.PitchTracker().push(np.zeros(100))


def test_regenerates_identically():
    """make_pitch_golden.py --check, when the LibriSpeech input is prepared."""
    import make_pitch_golden as mpg
    if not pathlib.Path(paths.resolve(mpg.LIBRI_PATH)).is_file():
        pytest.skip("LibriSpeech test-clean not prepared")
    files = mpg.build()
    for name, data in files.items():
        assert mpg._read(GOLDEN / name) == data, f"stale golden file {name}"
