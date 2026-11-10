"""Milestone 2 / Stage 07 golden parity check: C++ `fv_vc_file` vs. Stage 06
Python ONNX Runtime `golden/` outputs (units exact, audio max |Δ| <= 1e-5, and
RSS growth < 1 MB over 10,000 steps after warm-up).

Skipped when `FLUXVOICE_VC_FILE` is not set or does not point to an ORT-enabled
`fv_vc_file` binary.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import export_stateful_onnx as cli  # noqa: E402
from fvml import audio  # noqa: E402
from fvml import content_encoder as ce  # noqa: E402
from fvml import parity as fp  # noqa: E402
from fvml import vocoder as vc  # noqa: E402

FS = ce.FRAME_SAMPLES
TINY_ENC = ce.EncoderConfig(channels=(8, 8, 16, 16, 16), latent_dim=16, head_hidden=16, vocab=32)
TINY_VOC = vc.VocoderConfig(vocab=32, unit_dim=16, cond_hidden=16, n_speakers=2, spk_emb_dim=4,
                            upsample_initial_channel=16, resblock_kernels=(3,),
                            resblock_dilations=((1, 3),))
SPEAKERS = ["lj", "hifitts_92"]
STATS = {"lj": {"logf0_mean": 5.3, "logf0_std": 0.2, "energy_mean": -8.0, "energy_std": 3.0},
         "hifitts_92": {"logf0_mean": 5.1, "logf0_std": 0.25, "energy_mean": -7.5,
                        "energy_std": 2.5}}


def _vc_file_bin():
    env_path = os.environ.get("FLUXVOICE_VC_FILE")
    if not env_path:
        pytest.skip("FLUXVOICE_VC_FILE not set (run via ORT-enabled CTest)")
    p = pathlib.Path(env_path)
    if not p.is_file():
        pytest.skip(f"fv_vc_file not found at {p}")
    return p


def _speech(seconds, seed=0, sr=16000):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 35 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 * (1 + np.sin(2 * np.pi * 1.3 * t))
    return (0.12 * x * env + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def test_fv_vc_file_reproduces_stage06_golden_and_rss_budget(tmp_path):
    vc_file = _vc_file_bin()

    torch.manual_seed(0)
    enc = ce.ContentEncoder(TINY_ENC).eval()
    torch.manual_seed(1)
    voc = vc.UnitVocoder(TINY_VOC).eval()
    with torch.no_grad():
        for p in voc.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.3)

    enc_ckpt = tmp_path / "enc.pt"
    voc_ckpt = tmp_path / "voc.pt"
    torch.save({"model": enc.state_dict(), "config": json.dumps(enc.cfg.to_dict()), "step": 10,
                "labels_meta": json.dumps({"tokenizer": "hubert-km32"})}, enc_ckpt)
    torch.save({"model": voc.state_dict(), "config": json.dumps(voc.cfg.to_dict()), "step": 10,
                "contract": voc.contract(1).to_json(), "speakers": SPEAKERS,
                "speaker_stats": STATS}, voc_ckpt)

    utt_path = tmp_path / "utt.wav"
    audio.save(utt_path, _speech(12.0))

    bundle = tmp_path / "bundle"
    common = ["--bundle", str(bundle), "--frames-per-step", "1", "2", "4",
              "--exporter", "torchscript", "--parity-audio", str(utt_path),
              "--parity-seconds", "10", "--golden-seconds", "10", "--cost-steps", "10"]
    assert cli.main(["--role", "encoder", "--checkpoint", str(enc_ckpt), "--head", "student-L1",
                     "--license", "CC BY 4.0", *common]) == 0
    assert cli.main(["--role", "vocoder", "--checkpoint", str(voc_ckpt), "--gate-variant", "1",
                     *common]) == 0

    golden_dir = bundle / "golden"
    manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    voice = manifest["voice"]

    for fps in (1, 2, 4):
        out_wav = tmp_path / f"cpp_s{fps}.wav"
        out_units = tmp_path / f"cpp_units_s{fps}.npy"
        res = subprocess.run(
            [str(vc_file), "--model-dir", str(bundle), "--voice", voice,
             "--frames-per-step", str(fps), "--dump-units", str(out_units),
             str(golden_dir / "input.wav"), str(out_wav)],
            check=True, capture_output=True, text=True,
        )
        assert "ONNX Runtime       : 1.31.0" in res.stdout

        want_u = np.load(golden_dir / f"units_s{fps}.npy")
        got_u = np.load(out_units)
        assert np.array_equal(want_u, got_u), f"s{fps} units differ from golden"

        want_a = audio.load(golden_dir / f"audio_s{fps}.wav")
        got_a = audio.load(out_wav)
        assert np.sqrt(np.mean(want_a ** 2)) > 1e-2
        rep = fp.compare(want_a, got_a)
        assert rep.max_abs <= 1e-5, f"s{fps} max_abs {rep.max_abs} > 1e-5"
        assert rep.snr_db >= 90.0, f"s{fps} snr_db {rep.snr_db} < 90 dB"

    # 10,000-step leak check (500 steps/pass * 20 repeats = 10,000 steps)
    rep_res = subprocess.run(
        [str(vc_file), "--model-dir", str(bundle), "--voice", voice,
         "--frames-per-step", "1", "--repeat", "20",
         str(golden_dir / "input.wav"), str(tmp_path / "repeat.wav")],
        check=True, capture_output=True, text=True,
    )
    assert "Steps              : 10000" in rep_res.stdout
    m = re.search(r"delta=([+-]?\d+) KB", rep_res.stdout)
    assert m is not None, rep_res.stdout
    delta_kb = int(m.group(1))
    assert delta_kb < 1024, f"RSS grew by {delta_kb} KB (>= 1 MB) over 10,000 steps"
