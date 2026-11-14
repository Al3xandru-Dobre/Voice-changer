"""Stage 02 fixture models (v1 Mode A & v2 Mode B S2UT) run in ONNX Runtime
the way Stage 07's C++ tests expect.

Needs onnx + onnxruntime (the uv env): uv run pytest tests/test_fixtures.py
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from fvml import contract as fc  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "data" / "fixtures"


def read_manifest():
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def session(rel):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(str(FIXTURES / rel), so, providers=["CPUExecutionProvider"])


def load_generator():
    spec = importlib.util.spec_from_file_location("make_fixture_models",
                                                  REPO / "tools" / "make_fixture_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_fixture_models"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_regenerates_byte_identically():
    gen = load_generator()
    assert gen.check(FIXTURES, gen.build_all()) == 0


def test_contract_metadata_reaches_ort():
    sess = session("identity/encoder_s1.onnx")
    text = sess.get_modelmeta().custom_metadata_map[fc.METADATA_KEY]
    c = fc.ModelContract.from_json(text)
    assert (c.role, c.hop) == ("encoder", 320)


@pytest.mark.parametrize("fps", [1, 2, 4])
def test_identity_bundle_is_bit_exact(fps):
    enc, voc = session(f"identity/encoder_s{fps}.onnx"), session(f"identity/vocoder_s{fps}.onnx")
    c = fc.ModelContract.from_json(voc.get_modelmeta().custom_metadata_map[fc.METADATA_KEY])
    rng = np.random.default_rng(fps)
    for _ in range(5):
        audio = rng.standard_normal((1, c.hop), dtype=np.float32)
        (content,) = enc.run(None, {"audio": audio})
        feeds = {"content": content, "spk_emb": np.ones((1, c.spk_emb_dim), np.float32)}
        feeds.update({n: np.zeros((1, c.cond_frames), np.float32)
                      for n in c.cond_inputs if n != "spk_emb"})
        (out,) = voc.run(["audio"], feeds)
        assert out.shape == audio.shape and np.array_equal(out, audio)


def test_delay_is_one_hop():
    voc = session("delay_s1.onnx")
    state = np.zeros((1, 320), np.float32)
    previous = state.copy()
    for step in range(10):
        content = np.full((1, 1, 320), step + 1, np.float32)
        audio, state = voc.run(["audio", "state_out.buf"], {"content": content, "state_in.buf": state})
        assert np.array_equal(audio, previous)
        previous = content.reshape(1, 320)


def test_accumulator_matches_closed_form():
    enc = session("accumulator_s1.onnx")
    state = np.zeros((1, 1), np.float32)
    audio = np.full((1, 320), 0.25, np.float32)
    for step in range(1, 1001):
        content, state = enc.run(["content", "state_out.acc"], {"audio": audio, "state_in.acc": state})
        assert content.item() == state.item() == 80.0 * step


def test_s2ut_v2_quartet_read_write_flush_and_ring_wrap():
    """Verify the Stage 02 v2 S2UT quartet over > 2x the 4-slot ring window
    (10 source chunks = 2.5x wrap), followed by 3 tail-flush steps."""
    enc = session("s2ut/encoder_s4.onnx")
    t_enc = session("s2ut/translator_enc.onnx")
    t_dec = session("s2ut/translator_dec.onnx")
    voc = session("s2ut/vocoder_s4.onnx")

    wait_k = 3
    src_lang = np.array([1], dtype=np.int64)   # "es" = 1
    tgt_lang = np.array([3], dtype=np.int64)   # "de" = 3
    prosody_ctx = np.array([[0.5, -0.25]], dtype=np.float32)

    src_pos = np.zeros((1,), dtype=np.int64)
    enc_conv = np.zeros((1, 8), dtype=np.float32)
    mem_k = np.zeros((1, 4, 8), dtype=np.float32)
    mem_v = np.zeros((1, 4, 8), dtype=np.float32)

    tgt_pos = np.zeros((1,), dtype=np.int64)
    dec_kv = np.zeros((1, 4, 8), dtype=np.float32)
    prev_units = np.array([[16, 16, 16, 16]], dtype=np.int64)  # bos = 16

    chunk_vals = [0.125 * (i + 1) for i in range(10)]
    decoded_v_means = []

    for i, val in enumerate(chunk_vals):
        audio = np.full((1, 1280), val, dtype=np.float32)
        units_a, latents = enc.run(["units", "latents"], {"audio": audio})
        assert units_a.shape == (1, 4) and latents.shape == (1, 4, 8)
        np.testing.assert_allclose(latents, val, rtol=0, atol=1e-6)

        readiness, src_pos, enc_conv, mem_k, mem_v = t_enc.run(
            ["readiness_logit", "state_out.src_pos", "state_out.enc_conv",
             "state_out.mem_k_0", "state_out.mem_v_0"],
            {"latents": latents, "src_lang": src_lang, "state_in.src_pos": src_pos,
             "state_in.enc_conv": enc_conv, "state_in.mem_k_0": mem_k, "state_in.mem_v_0": mem_v},
        )
        assert src_pos.item() == i + 1
        assert np.isclose(readiness.item(), val, atol=1e-6)
        assert np.isclose(mem_v[0, i % 4, 0], val, atol=1e-6)
        assert np.isclose(mem_k[0, i % 4, 0], val + 1.0, atol=1e-6)

        # WaitKPolicy: first (wait_k - 1) steps are kRead; step >= wait_k - 1 is kWrite.
        if src_pos.item() - tgt_pos.item() >= wait_k:
            expected_units = (prev_units + tgt_lang) % 16
            units_b, prosody, eos_prob, tgt_pos, dec_kv = t_dec.run(
                ["units", "prosody", "eos_prob", "state_out.tgt_pos", "state_out.dec_kv_0"],
                {"prev_units": prev_units, "tgt_lang": tgt_lang, "prosody_ctx": prosody_ctx,
                 "src_pos": src_pos, "state_in.tgt_pos": tgt_pos,
                 "mem_k_0": mem_k, "mem_v_0": mem_v, "state_in.dec_kv_0": dec_kv},
            )
            assert np.array_equal(units_b, expected_units)
            assert eos_prob.item() == 0.0
            # Decoder reads slot (src_pos - wait_k) mod 4 == chunk (i - wait_k + 1).
            delayed_val = chunk_vals[i - wait_k + 1]
            decoded_v_means.append(prosody[0, 0, 3] - prosody_ctx[0, 1])
            assert np.isclose(decoded_v_means[-1], delayed_val, atol=1e-6)
            assert np.isclose(prosody[0, 0, 0], delayed_val + 1.0 + prosody_ctx[0, 0], atol=1e-6)
            prev_units = units_b

            # Feed emitted target units + prosody into the v2 discrete vocoder.
            (wav,) = voc.run(["audio"], {
                "units": units_b,
                "logf0": prosody[:, :, 0],
                "voicing": prosody[:, :, 1],
                "aperiodicity": prosody[:, :, 2],
                "energy": prosody[:, :, 3],
                "spk_emb": np.zeros((1, 4), dtype=np.float32),
            })
            assert wav.shape == (1, 1280)

    # Tail flush: src_pos stays at 10; 3 flush steps drain remaining buffered chunks
    # and raise eos_prob = 1.0 on the 3rd flush step.
    flush_eos = []
    for f_step in range(3):
        units_b, prosody, eos_prob, tgt_pos, dec_kv = t_dec.run(
            ["units", "prosody", "eos_prob", "state_out.tgt_pos", "state_out.dec_kv_0"],
            {"prev_units": prev_units, "tgt_lang": tgt_lang, "prosody_ctx": prosody_ctx,
             "src_pos": src_pos, "state_in.tgt_pos": tgt_pos,
             "mem_k_0": mem_k, "mem_v_0": mem_v, "state_in.dec_kv_0": dec_kv},
        )
        flush_eos.append(eos_prob.item())
        if f_step < 2:
            decoded_v_means.append(prosody[0, 0, 3] - prosody_ctx[0, 1])
        prev_units = units_b

    assert flush_eos == [0.0, 0.0, 1.0]
    np.testing.assert_allclose(decoded_v_means, chunk_vals, rtol=0, atol=1e-6)


def test_broken_fixtures_load_but_break_the_contract():
    manifest = read_manifest()
    broken = [m for m in manifest["models"] if m["error"]]
    assert len(broken) >= 14
    for m in broken:
        session(m["file"])  # valid ONNX: only the contract check rejects it
        with pytest.raises(fc.ContractError) as e:
            fc.load_onnx(FIXTURES / m["file"])
        assert str(e.value) == m["error"], m["file"]


def test_embed_replaces_a_stale_contract_and_sorts_keys():
    import onnx
    model = onnx.load(str(FIXTURES / "identity" / "encoder_s1.onnx"))
    fc.embed(model, '{"old": 1}')
    for key in ("zeta", "author"):
        p = model.metadata_props.add()
        p.key, p.value = key, "x"
    fc.embed(model, "NEW")
    assert [(q.key, q.value) for q in model.metadata_props] == [
        ("author", "x"), (fc.METADATA_KEY, "NEW"), ("zeta", "x")]


def test_graph_io_reads_dtypes_and_missing_shapes():
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["state_in.kv"], ["state_out.kv"]),
         helper.make_node("Identity", ["audio"], ["units_f"])], "g",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "T"]),
         helper.make_tensor_value_info("state_in.kv", TensorProto.FLOAT16, [2, 8])],
        [helper.make_tensor_value_info("units_f", TensorProto.INT32, [1, 1]),
         helper.make_tensor_value_info("state_out.kv", TensorProto.FLOAT16, None)])
    ins, outs = fc.graph_io(helper.make_model(graph))
    assert ins == [fc.TensorSpec("audio", "f32", (1, -1)), fc.TensorSpec("state_in.kv", "f16", (2, 8))]
    assert outs == [fc.TensorSpec("units_f", "other", (1, 1)), fc.TensorSpec("state_out.kv", "f16", None)]


def test_manifest_hashes_match_files():
    import hashlib
    manifest = read_manifest()
    for m in manifest["models"]:
        assert hashlib.sha256((FIXTURES / m["file"]).read_bytes()).hexdigest() == m["sha256"]
