"""Stateful ONNX export & parity (Milestone 2 / Stage 06): step-graph state
names, contract embedding and the onnx checks, dynamo with a recorded
TorchScript fallback, ORT chunked vs. PyTorch offline parity (encoder, vocoder,
chain), the conditioning alignment, voices.json and the bundle CLI.

Tiny random-weight models keep it fast; most exports use the TorchScript
exporter (about 1 s per graph), and one test per role exercises dynamo.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import export_stateful_onnx as cli  # noqa: E402
from fvml import audio  # noqa: E402
from fvml import contract as fc  # noqa: E402
from fvml import content_encoder as ce  # noqa: E402
from fvml import export as fx  # noqa: E402
from fvml import features as ft  # noqa: E402
from fvml import parity  # noqa: E402
from fvml import vocoder as vc  # noqa: E402

FS = ce.FRAME_SAMPLES
HEAD = 1                                     # lookahead 1: exercises the delay alignment
TINY_ENC = ce.EncoderConfig(channels=(4, 4, 8, 8, 8), latent_dim=8, head_hidden=8, vocab=16)
TINY_VOC = vc.VocoderConfig(vocab=16, unit_dim=8, cond_hidden=8, n_speakers=2, spk_emb_dim=4,
                            upsample_initial_channel=16, resblock_kernels=(3,),
                            resblock_dilations=((1, 3),))
SPEAKERS = ["lj", "hifitts_92"]
STATS = {"lj": {"logf0_mean": 5.3, "logf0_std": 0.2, "energy_mean": -8.0, "energy_std": 3.0},
         "hifitts_92": {"logf0_mean": 5.1, "logf0_std": 0.25, "energy_mean": -7.5,
                        "energy_std": 2.5}}


def speech(seconds, seed=0, sr=16000):
    """Voiced, gliding, amplitude-modulated harmonics plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 110 + 30 * seed + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 * (1 + np.sin(2 * np.pi * 1.3 * t))
    return (0.1 * x * env + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def tiny_encoder(seed=0):
    torch.manual_seed(seed)
    return ce.ContentEncoder(TINY_ENC).eval()


def tiny_vocoder(seed=1):
    torch.manual_seed(seed)
    m = vc.UnitVocoder(TINY_VOC).eval()
    with torch.no_grad():                   # the training init is near-silent; parity on
        for p in m.parameters():            # silence would prove nothing
            if p.dim() > 1:
                p.normal_(0.0, 0.3)
    return m


@pytest.fixture(scope="module")
def encoder():
    return tiny_encoder()


@pytest.fixture(scope="module")
def vocoder():
    return tiny_vocoder()


@pytest.fixture(scope="module")
def voice(vocoder):
    return vocoder.voices_json(SPEAKERS, STATS)["voices"][0]


@pytest.fixture(scope="module")
def exported(encoder, vocoder, tmp_path_factory):
    """{(role, fps): (path, contract, record)} for s1/s2, TorchScript-exported."""
    d = tmp_path_factory.mktemp("bundle")
    out = {}
    for fps in (1, 2):
        c = fx.encoder_contract(encoder, HEAD, fps, "test fixture")
        g = fx.encoder_graph(encoder, HEAD, fps)
        p = d / f"encoder_s{fps}.onnx"
        rec = fx.export_graph(g, p, "torchscript")
        fx.finalize(p, c, drop_outputs=g.debug_outputs)
        out[("encoder", fps)] = (p, c, rec)
        c = fx.vocoder_contract(vocoder, {}, fps)
        p = d / f"vocoder_s{fps}.onnx"
        rec = fx.export_graph(fx.vocoder_graph(vocoder, c), p, "torchscript")
        fx.finalize(p, c)
        out[("vocoder", fps)] = (p, c, rec)
    return out


# -- step graphs -----------------------------------------------------------------
def test_encoder_graph_holds_the_trunk_and_one_head(encoder):
    layout = [n for n, _, _ in encoder.state_layout()]
    g = fx.encoder_graph(encoder, HEAD, 2)
    assert g.states == [n for n in layout if n.startswith(("trunk.", f"heads.{HEAD}."))]
    assert f"heads.{HEAD}.conv" in g.states
    assert not any(n.startswith(("heads.0.", "heads.2.")) for n in g.states)
    # head 0 (L = 0) is a k=1 conv: no state of its own
    assert fx.encoder_graph(encoder, 0, 1).states == [n for n in layout if n.startswith("trunk.")]
    assert g.input_names() == ["audio"] + [f"state_in.{n}" for n in g.states]
    assert g.output_names() == ["units", "logits"] + [f"state_out.{n}" for n in g.states]
    assert g.debug_outputs == ["logits"]
    c = fx.encoder_contract(encoder, HEAD, 2, "x")
    assert list(c.states) == g.states and c.lookahead_frames == 1 and c.hop == 2 * FS
    assert [tuple(t.shape) for t in g.example()[:1]] == [(1, 2 * FS)]


def test_vocoder_graph_matches_the_contract(vocoder):
    c = fx.vocoder_contract(vocoder, {"license": "override"}, 4)
    g = fx.vocoder_graph(vocoder, c)
    assert c.license == "override" and c.frames_per_step == 4
    assert g.states == [n for n, _, _ in vocoder.state_layout()] == list(c.states)
    assert [t.name for t in g.inputs] == ["units", *fc.COND_INPUTS]
    assert g.output_names()[0] == "audio" and g.debug_outputs == []


def test_head_index(encoder):
    assert fx.head_index(encoder, "student-L1") == 1
    assert fx.head_index(encoder, "2") == 2 and fx.head_index(encoder, 0) == 0
    for bad in ("student-L5", 3, "-1"):
        with pytest.raises(fx.ExportError):
            fx.head_index(encoder, bad)


# -- exported files ----------------------------------------------------------------
def test_exported_files_carry_a_valid_contract(exported):
    for (role, fps), (path, c, rec) in exported.items():
        loaded, model = fc.load_onnx(path)
        assert loaded == c, path.name
        assert rec["exporter"] == "torchscript" and rec["fallback_reason"] is None
        assert [o.name for o in model.graph.output if o.name == "logits"] == []
        assert model.ir_version <= fx.MAX_IR_VERSION
        assert fx.default_opset(model) <= fx.MAX_OPSET
        onnx.checker.check_model(model, full_check=True)
        keys = [p.key for p in model.metadata_props]
        assert keys == sorted(keys) and fc.METADATA_KEY in keys
    for fps in (1, 2):
        fc.validate_pair(exported[("encoder", fps)][1], exported[("vocoder", fps)][1])
    with pytest.raises(fc.ContractError, match="frames_per_step mismatch"):
        fc.validate_pair(exported[("encoder", 1)][1], exported[("vocoder", 2)][1])


@pytest.mark.parametrize("fps", [1, 2])
def test_encoder_parity_chunked_vs_offline(encoder, exported, fps):
    r = fx.encoder_parity(encoder, HEAD, exported[("encoder", fps)][0], speech(3.0))
    assert r["frames"] == 150 and r["seconds"] == pytest.approx(3.0)
    assert r["unit_agreement"] == 1.0 and r["pass"] is True
    assert r["max_abs_logit"] < 1e-4


def test_states_carry_history(encoder, exported):
    """The graph really reads state_in.*: the same hop gives different logits
    on a fresh and on a used state, and a stream that drops its states (the
    frozen-cache bug) fails parity."""
    path, c, _ = exported[("encoder", 1)]
    stream = fx.OrtStream(path, extra_outputs=[fx.logits_spec(c)])
    wav = speech(2.0)
    hop = {"audio": wav[None, 5 * FS:6 * FS]}
    first = stream.step(hop)["logits"]
    second = stream.step(hop)["logits"]
    assert np.abs(first - second).max() > 1e-3

    ref = fx.encoder_offline(encoder, HEAD, wav)
    stateless = []
    for i in range(len(wav) // FS):
        stream.reset()
        stateless.append(stream.step({"audio": wav[None, i * FS:(i + 1) * FS]})["logits"][0])
    assert np.abs(np.concatenate(stateless) - ref).max() > 1e-2
    units, logits = fx.encoder_chunked(stream, wav)       # reset + fed back: exact again
    assert np.abs(logits - ref).max() < 1e-4 and units.dtype == np.int64


def test_conditioning_is_the_streaming_tracker_mapped_to_the_voice(exported, voice):
    """Push t of the tracker (lookahead L) describes frame t - L, the frame the
    encoder's id at t describes: conditioning[t] = extract()[t - L], mapped."""
    enc_c, voc_c = exported[("encoder", 1)][1], exported[("vocoder", 1)][1]
    wav = speech(2.0, seed=1)
    cond = fx.conditioning(wav, voice, enc_c, voc_c)
    L = enc_c.lookahead_frames
    assert L == 1 and set(cond) == {"logf0", "voicing", "aperiodicity", "energy"}
    ref = ft.extract(wav, ft.PitchConfig(lookahead_frames=L))
    col = {n: ref[:, i] for i, n in enumerate(ft.FEATURES)}
    logf0, energy = ft.map_to_target(col["z_logf0"], col["z_energy"], voice)
    n = len(wav) // FS
    assert all(len(v) == n and v.dtype == np.float32 for v in cond.values())
    np.testing.assert_allclose(cond["logf0"][L:], logf0[:n - L], rtol=0, atol=1e-5)
    np.testing.assert_allclose(cond["energy"][L:], energy[:n - L], rtol=0, atol=1e-5)
    np.testing.assert_allclose(cond["voicing"][L:], col["voicing"][:n - L], rtol=0, atol=1e-6)
    np.testing.assert_allclose(cond["aperiodicity"][L:], col["aperiodicity"][:n - L],
                               rtol=0, atol=1e-6)
    assert voice["logf0_mean"] == STATS["lj"]["logf0_mean"]   # the voice's domain, not the source's


@pytest.mark.parametrize("fps", [1, 2])
def test_vocoder_parity_chunked_vs_offline(vocoder, exported, voice, fps):
    enc_path, enc_c, _ = exported[("encoder", fps)]
    voc_path, voc_c, _ = exported[("vocoder", fps)]
    wav = speech(3.0)
    units, _ = fx.encoder_chunked(fx.OrtStream(enc_path), wav)
    full = fx.vocoder_inputs(voc_c, units, fx.conditioning(wav, voice, enc_c, voc_c),
                             voice["embedding"])
    assert full["units"].dtype == np.int64 and full["spk_emb"].shape == (4,)
    ref = fx.vocoder_offline(vocoder, voc_c, full)
    assert np.sqrt(np.mean(ref ** 2)) > 1e-2              # not a parity of silence
    r = fx.vocoder_parity(vocoder, voc_path, full)
    assert r["samples"] == len(wav) and r["pass"] is True
    assert r["snr_db"] is None or r["snr_db"] >= fx.MIN_VOCODER_SNR_DB
    # units one frame late (an alignment slip) fail the gate
    late = dict(full, units=np.concatenate([full["units"][:1], full["units"][:-1]]))
    test = fx.vocoder_chunked(fx.OrtStream(voc_path), late)
    assert parity.snr_db(ref, test) < fx.MIN_VOCODER_SNR_DB


def test_chain_parity(encoder, vocoder, exported, voice):
    r = fx.chain_parity(encoder, HEAD, vocoder, exported[("encoder", 2)][0],
                        exported[("vocoder", 2)][0], speech(3.0), voice)
    assert r["unit_agreement"] == 1.0
    assert r["snr_db"] is None or r["snr_db"] >= fx.MIN_VOCODER_SNR_DB


def test_step_cost_reports_percentiles(exported):
    stream = fx.OrtStream(exported[("encoder", 1)][0])
    feeds = ([f] for f in fx.encoder_feeds(stream.contract, speech(1.0)))
    k = fx.step_cost([stream], feeds, warmup=5)
    assert k["steps"] == 45 and k["hop_ms"] == 20.0
    assert 0 < k["p50_ms"] <= k["p95_ms"] <= k["p99_ms"] <= k["max_ms"]
    assert k["within_budget"] is (k["p95_ms"] <= 10.0)
    few = list(fx.encoder_feeds(stream.contract, speech(0.1)))
    with pytest.raises(fx.ExportError, match="warm-up"):
        fx.step_cost([stream], ([f] for f in few), warmup=5)


# -- exporters -------------------------------------------------------------------
@pytest.mark.parametrize("role", ["encoder", "vocoder"])
def test_dynamo_export(encoder, vocoder, voice, tmp_path, role):
    pytest.importorskip("onnxscript")
    if role == "encoder":
        c, g = fx.encoder_contract(encoder, HEAD, 1, "x"), fx.encoder_graph(encoder, HEAD, 1)
    else:
        c = fx.vocoder_contract(vocoder, {}, 1)
        g = fx.vocoder_graph(vocoder, c)
    path = tmp_path / f"{role}_s1.onnx"
    rec = fx.export_graph(g, path, "auto")
    assert rec["exporter"] == "dynamo" and rec["fallback_reason"] is None
    fx.finalize(path, c, drop_outputs=g.debug_outputs, metadata={"fluxvoice.exporter": "dynamo"})
    assert fc.load_onnx(path)[0] == c
    wav = speech(2.0)
    if role == "encoder":
        assert fx.encoder_parity(encoder, HEAD, path, wav)["pass"]
    else:
        units = np.random.default_rng(0).integers(0, 16, len(wav) // FS)
        enc_c = fx.encoder_contract(encoder, HEAD, 1, "x")
        full = fx.vocoder_inputs(c, units, fx.conditioning(wav, voice, enc_c, c),
                                 voice["embedding"])
        assert fx.vocoder_parity(vocoder, path, full)["pass"]


def test_fallback_to_torchscript_is_recorded(encoder, tmp_path, monkeypatch):
    real = torch.onnx.export

    def fake(*args, dynamo=True, **kw):
        if dynamo:
            raise RuntimeError("capture failed: data-dependent slice\nlong trace")
        return real(*args, dynamo=False, **kw)

    monkeypatch.setattr(torch.onnx, "export", fake)
    g = fx.encoder_graph(encoder, HEAD, 1)
    rec = fx.export_graph(g, tmp_path / "e.onnx", "auto")
    assert rec["exporter"] == "torchscript"
    assert rec["fallback_reason"] == "RuntimeError: capture failed: data-dependent slice"
    fx.finalize(tmp_path / "e.onnx", fx.encoder_contract(encoder, HEAD, 1, "x"),
                drop_outputs=g.debug_outputs)
    with pytest.raises(fx.ExportError, match="dynamo export failed: RuntimeError: capture"):
        fx.export_graph(g, tmp_path / "f.onnx", "dynamo")
    with pytest.raises(ValueError):
        fx.export_graph(g, tmp_path / "g.onnx", "onnxscript")


def _identity_encoder(path, opset, ir):
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Unsqueeze", ["audio", "axes"], ["content"])], "identity",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, FS])],
        [helper.make_tensor_value_info("content", TensorProto.FLOAT, [1, 1, FS])],
        [helper.make_tensor("axes", TensorProto.INT64, [1], [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir
    onnx.save(model, str(path))
    return fc.ModelContract(role="encoder", sample_rate=16000, frame_samples=FS, frames_per_step=1,
                            lookahead_frames=0, content_kind="soft", content_dim=FS, license="x")


def test_finalize_caps_ir_and_rejects_new_opsets(tmp_path):
    c = _identity_encoder(tmp_path / "ok.onnx", 17, 14)
    assert fx.finalize(tmp_path / "ok.onnx", c, metadata={"fluxvoice.exporter": "test"}) == (17, 13)
    model = onnx.load(str(tmp_path / "ok.onnx"))
    assert model.ir_version == 13
    assert {p.key: p.value for p in model.metadata_props}["fluxvoice.exporter"] == "test"
    c = _identity_encoder(tmp_path / "new.onnx", 27, 13)
    with pytest.raises(fx.ExportError, match="opset 27 > 26"):
        fx.finalize(tmp_path / "new.onnx", c)
    # a contract the graph does not honour is caught before shipping
    _identity_encoder(tmp_path / "bad.onnx", 17, 10)
    wrong = fc.ModelContract(role="encoder", sample_rate=16000, frame_samples=FS,
                             frames_per_step=2, lookahead_frames=0, content_kind="soft",
                             content_dim=FS, license="x")
    with pytest.raises(fx.ExportError, match="graph input 'audio' has shape"):
        fx.finalize(tmp_path / "bad.onnx", wrong)
    with pytest.raises(fx.ExportError, match="no tensor named 'logits'"):
        fx.with_outputs(model, [fx.TensorSpec("logits", "f32", (1, 1, 16))])


# -- voices.json -----------------------------------------------------------------
def test_voices_json_from_the_checkpoint(vocoder):
    data = fx.voices_json(vocoder, {"speakers": SPEAKERS, "speaker_stats": STATS})
    fc.validate_voices(data, fx.vocoder_contract(vocoder, {}, 1))
    assert [v["name"] for v in data["voices"]] == SPEAKERS
    np.testing.assert_allclose(data["voices"][1]["embedding"],
                               vocoder.speaker_table.weight[1].detach().numpy(), rtol=0, atol=0)
    assert "attribution" in data["voices"][1] and data["voices"][1]["license"] == "CC BY 4.0"
    as_text = fx.voices_json(vocoder, {"speakers": json.dumps(SPEAKERS),
                                       "speaker_stats": json.dumps(STATS)})
    assert as_text == data
    with pytest.raises(fx.ExportError, match="needs 'speakers'"):
        fx.voices_json(vocoder, {"speaker_stats": STATS})
    with pytest.raises(fx.ExportError, match="voices.json"):
        fx.voices_json(vocoder, {"speakers": SPEAKERS, "speaker_stats": {"lj": STATS["lj"]}})


# -- CLI -------------------------------------------------------------------------
def _checkpoints(tmp_path):
    enc, voc = tiny_encoder(), tiny_vocoder()
    torch.save({"model": enc.state_dict(), "config": json.dumps(enc.cfg.to_dict()), "step": 7,
                "labels_meta": json.dumps({"tokenizer": "hubert-km16"})}, tmp_path / "enc.pt")
    torch.save({"model": voc.state_dict(), "config": json.dumps(voc.cfg.to_dict()), "step": 9,
                "contract": voc.contract(1).to_json(), "speakers": SPEAKERS,
                "speaker_stats": STATS}, tmp_path / "voc.pt")
    files = []
    for i in range(3):
        files.append(tmp_path / f"utt{i}.wav")
        audio.save(files[-1], speech(2.0, seed=i))
    return tmp_path / "enc.pt", tmp_path / "voc.pt", [str(f) for f in files]


def test_cli_builds_a_bundle(tmp_path):
    enc_ckpt, voc_ckpt, wavs = _checkpoints(tmp_path)
    bundle = tmp_path / "models" / "tiny"
    common = ["--bundle", str(bundle), "--frames-per-step", "1", "2", "--exporter", "torchscript",
              "--parity-audio", *wavs, "--parity-seconds", "5", "--golden-seconds", "2",
              "--cost-steps", "5"]
    assert cli.main(["--role", "encoder", "--checkpoint", str(enc_ckpt), "--head", "student-L1",
                     "--license", "CC BY 4.0 (LibriSpeech)", *common]) == 0
    golden = json.loads((bundle / "golden" / "manifest.json").read_text(encoding="utf-8"))
    assert set(golden["variants"]) == {"s1", "s2"} and "cond" not in golden   # no vocoder yet

    assert cli.main(["--role", "vocoder", "--checkpoint", str(voc_ckpt), "--gate-variant", "2",
                     *common]) == 0
    for name in ("encoder_s1.onnx", "encoder_s2.onnx", "vocoder_s1.onnx", "vocoder_s2.onnx",
                 "voices.json", "MODEL_CARD.md", "export_report.json"):
        assert (bundle / name).exists(), name
    assert not (bundle / "encoder_s4.onnx").exists()
    report = json.loads((bundle / "export_report.json").read_text(encoding="utf-8"))
    assert report["gate_variant"] == 2 and set(report["pairs"]) == {"s1", "s2"}
    for role in fc.ROLES:
        r = report["roles"][role]
        assert set(r["files"]) == set(r["parity"]) == set(r["cost"]) == {"s1", "s2"}
        assert all(p["pass"] for p in r["parity"].values())
    assert report["roles"]["encoder"]["head"] == 1
    assert report["roles"]["vocoder"]["parity"]["s1"]["units"] == "encoder_s1.onnx"
    assert report["chain"]["s2"]["unit_agreement"] == 1.0
    assert set(report["chain_cost"]) == {"s1", "s2"}
    assert [v["name"] for v in report["voices"]] == SPEAKERS
    voices = json.loads((bundle / "voices.json").read_text(encoding="utf-8"))
    fc.validate_voices(voices, fx.read_contract(bundle / "vocoder_s1.onnx"))

    # golden/: what Stage 07's fv_vc_file reproduces
    g = bundle / "golden"
    golden = json.loads((g / "manifest.json").read_text(encoding="utf-8"))
    assert golden["voice"] == "lj" and golden["cond_columns"] == list(fc.COND_INPUTS[:4])
    clip = audio.load(g / "input.wav")
    assert len(clip) == 2 * 16000
    u1, u2 = np.load(g / "units_s1.npy"), np.load(g / "units_s2.npy")
    assert u1.dtype == np.int64 and np.array_equal(u1, u2) and len(u1) == 100
    assert np.load(g / "cond.npy").shape == (100, 4)
    a1, a2 = audio.load(g / "audio_s1.wav"), audio.load(g / "audio_s2.wav")
    assert len(a1) == len(clip) and np.abs(a1 - a2).max() < 1e-5
    units, cond, again = fx.chain_outputs(bundle / "encoder_s1.onnx", bundle / "vocoder_s1.onnx",
                                          clip, voices["voices"][0])
    assert np.array_equal(units, u1) and np.abs(again - a1).max() == 0

    card = (bundle / "MODEL_CARD.md").read_text(encoding="utf-8")
    for section in ("## Files", "## Training data and licences", "## Decision gate",
                    "## Exporter", "## Versions", "## Parity", "## Cost", "## Golden I/O",
                    "## Contracts"):
        assert section in card, section
    assert "CC BY 4.0 (LibriSpeech)" in card and "Cori Samuel" in card and "torchscript" in card

    # A different encoder exported at s1 only: its stale s2 file goes away.
    torch.save({"model": tiny_encoder(seed=5).state_dict(),
                "config": json.dumps(TINY_ENC.to_dict()), "step": 1}, tmp_path / "enc2.pt")
    assert cli.main(["--role", "encoder", "--checkpoint", str(tmp_path / "enc2.pt"),
                     "--head", "1", "--license", "x", "--bundle", str(bundle),
                     "--frames-per-step", "1", "--exporter", "torchscript", "--skip-parity"]) == 0
    assert not (bundle / "encoder_s2.onnx").exists()
    report = json.loads((bundle / "export_report.json").read_text(encoding="utf-8"))
    assert set(report["roles"]["encoder"]["files"]) == {"s1"} and set(report["pairs"]) == {"s1"}


def test_gate_report_drives_the_verdict():
    gate = {"decision": {"winner": "student-L1 s2", "lookahead_frames": 1, "reason": "r"},
            "candidates": [{"candidate": "student-L1 s1", "frames_per_step": 1},
                           {"candidate": "student-L1 s2", "frames_per_step": 2}]}
    assert cli.gate_variant(gate) == 2
    assert cli.gate_variant({"frames_per_step": 4}) == 4
    assert cli.gate_variant(None) is None and cli.gate_variant({"decision": {}}) is None
    fast, slow = {"within_budget": True}, {"within_budget": False}
    report = {"roles": {"encoder": {"lookahead_frames": 1, "parity": {"s2": {"pass": True}}}},
              "gate": gate, "gate_variant": 2, "chain_cost": {"s1": slow, "s2": fast}}
    assert all(ok for _, ok in cli.verdicts(report))       # s1 is over budget, but not chosen
    report["chain_cost"]["s2"] = slow
    assert [w for w, ok in cli.verdicts(report) if not ok] == ["chain s2 compute budget"]
    report["chain_cost"]["s2"] = fast
    report["roles"]["encoder"]["lookahead_frames"] = 2      # exported the wrong head
    assert [w for w, ok in cli.verdicts(report) if not ok] == [
        "encoder head (lookahead 2) matches the gate winner (lookahead 1)"]
    del report["chain_cost"]                                # not measured: no verdict
    report["roles"]["encoder"]["lookahead_frames"] = 1
    assert cli.verdicts(report) == [("encoder s2 parity", True), (
        "encoder head (lookahead 1) matches the gate winner (lookahead 1)", True)]


def test_cli_rejects_bad_input(tmp_path, capsys):
    enc_ckpt, _, wavs = _checkpoints(tmp_path)
    with pytest.raises(SystemExit):
        cli.parse_args(["--role", "encoder", "--checkpoint", "x", "--bundle", "b"])   # no --head
    with pytest.raises(SystemExit):
        cli.parse_args(["--role", "vocoder", "--checkpoint", "x", "--bundle", "b", "--head", "1"])
    bundle = str(tmp_path / "b")
    assert cli.main(["--role", "encoder", "--checkpoint", str(enc_ckpt), "--head", "1",
                     "--bundle", bundle, "--skip-parity"]) == 2      # no licence anywhere
    assert "pass --license" in capsys.readouterr().err
    assert cli.main(["--role", "encoder", "--checkpoint", str(enc_ckpt), "--head", "1",
                     "--license", "x", "--bundle", bundle, "--frames-per-step", "1",
                     "--exporter", "torchscript", "--parity-audio", wavs[0],
                     "--parity-seconds", "30"]) == 2      # 2 s of audio, 30 s asked
    assert "parity audio" in capsys.readouterr().err
