"""Milestone 2 / Stage 09: fv_vc_file voice controls on the identity fixture.

- ``--no-output-chain`` must reproduce the input bit-exactly (raw vocoder PCM
  parity, same as the golden test's chain).
- ``--param gain=-6.0206 --param gate_threshold=-80`` must scale a sine's RMS
  by 0.5 (+-1 %) once the ~20 ms one-pole has settled.
- ``--preset robot`` and a mid-run ``--change`` are accepted.

Skipped when ``FLUXVOICE_VC_FILE`` is not set or not an ORT-enabled binary.
"""
import os
import pathlib
import subprocess

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("soxr")

from fvml import audio  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
IDENTITY = REPO / "tests" / "data" / "fixtures" / "identity"
SR = audio.SAMPLE_RATE


def _vc_file_bin():
    env_path = os.environ.get("FLUXVOICE_VC_FILE")
    if not env_path:
        pytest.skip("FLUXVOICE_VC_FILE not set (run via ORT-enabled CTest)")
    p = pathlib.Path(env_path)
    if not p.is_file():
        pytest.skip(f"fv_vc_file not found at {p}")
    return p


def _sine(seconds, freq=440.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _run(vc_file, *args, expect_ok=True):
    res = subprocess.run([str(vc_file), *args], capture_output=True, text=True)
    if expect_ok:
        assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_no_output_chain_is_bit_exact_identity(tmp_path):
    vc_file = _vc_file_bin()
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    x = _sine(1.0)
    audio.save(src, x)
    _run(vc_file, "--model-dir", str(IDENTITY), "--voice", "identity",
         "--no-output-chain", str(src), str(out))
    y = audio.load(out)
    assert np.array_equal(y[: len(x)], x), "identity bundle must pass PCM through"


def test_gain_and_gate_params(tmp_path):
    vc_file = _vc_file_bin()
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    x = _sine(1.0, amp=0.4)
    audio.save(src, x)
    _run(vc_file, "--model-dir", str(IDENTITY), "--voice", "identity",
         "--param", "gain=-6.0206", "--param", "gate_threshold=-80",
         str(src), str(out))
    y = audio.load(out)
    skip = int(0.05 * SR)  # wait out the ~20 ms gain one-pole
    ratio = np.sqrt(np.mean(y[skip:] ** 2)) / np.sqrt(np.mean(x[skip:] ** 2))
    assert ratio == pytest.approx(0.5, rel=0.01)


def test_preset_and_change(tmp_path):
    vc_file = _vc_file_bin()
    src = tmp_path / "in.wav"
    x = _sine(1.5, amp=0.4)
    audio.save(src, x)
    # --preset robot resolves by name from the shipped presets/ dir.
    out1 = tmp_path / "preset.wav"
    _run(vc_file, "--model-dir", str(IDENTITY), "--voice", "identity",
         "--preset", "robot", str(src), str(out1))
    assert audio.info(out1)[0] > 0.0
    # --change applies mid-run at the first hop at/after the given second.
    out2 = tmp_path / "change.wav"
    _run(vc_file, "--model-dir", str(IDENTITY), "--voice", "identity",
         "--param", "gate_threshold=-80",
         "--change", "0.5:gain=-6", str(src), str(out2))
    y = audio.load(out2)
    first = np.sqrt(np.mean(y[int(0.2 * SR):int(0.45 * SR)] ** 2))
    second = np.sqrt(np.mean(y[int(0.8 * SR):int(1.4 * SR)] ** 2))
    assert second / first == pytest.approx(0.5, rel=0.05)


def test_param_and_change_rejections(tmp_path):
    vc_file = _vc_file_bin()
    src = tmp_path / "in.wav"
    audio.save(src, _sine(0.1))
    res = _run(vc_file, "--model-dir", str(IDENTITY), "--voice", "identity",
               "--param", "bogus=1", str(src), str(tmp_path / "o.wav"), expect_ok=False)
    assert res.returncode != 0
    assert "unknown parameter" in res.stderr
    res = _run(vc_file, "--model-dir", str(IDENTITY), "--voice", "identity",
               "--change", "abc", str(src), str(tmp_path / "o.wav"), expect_ok=False)
    assert res.returncode != 0
    assert "Invalid --change" in res.stderr
