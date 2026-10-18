"""Stage 02 fixture models run in ONNX Runtime the way Stage 07's C++ tests expect.

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
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


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
    # 1000 hops of a constant 0.25 (sum 80 per hop): every partial sum is exact
    # in float32, whatever order ReduceSum adds in.
    enc = session("accumulator_s1.onnx")
    state = np.zeros((1, 1), np.float32)
    audio = np.full((1, 320), 0.25, np.float32)
    for step in range(1, 1001):
        content, state = enc.run(["content", "state_out.acc"], {"audio": audio, "state_in.acc": state})
        assert content.item() == state.item() == 80.0 * step


def test_broken_fixtures_load_but_break_the_contract():
    broken = [m for m in MANIFEST["models"] if m["error"]]
    assert len(broken) >= 8
    for m in broken:
        session(m["file"])  # valid ONNX: only the contract check rejects it
        with pytest.raises(fc.ContractError) as e:
            fc.load_onnx(FIXTURES / m["file"])
        assert str(e.value) == m["error"], m["file"]


def test_embed_replaces_a_stale_contract_and_sorts_keys():
    import onnx
    model = onnx.load(str(FIXTURES / "identity" / "encoder_s1.onnx"))
    fc.embed(model, '{"old": 1}')
    p = model.metadata_props.add()
    p.key, p.value = "author", "x"
    fc.embed(model, "NEW")
    assert [(q.key, q.value) for q in model.metadata_props] == [("author", "x"), (fc.METADATA_KEY, "NEW")]


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
    for m in MANIFEST["models"]:
        assert hashlib.sha256((FIXTURES / m["file"]).read_bytes()).hexdigest() == m["sha256"]
