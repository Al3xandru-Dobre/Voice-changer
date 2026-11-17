"""translator_enc / translator_dec step graphs in ONNX Runtime (Milestone 3 /
Stage 05; the export itself is Stage 06's job, this pins that step() is
exportable to exactly the contract v2 signatures and that the ORT clock the
decision gate needs exists).

  * the exported files carry and satisfy contract v2 (I/O names, dtypes, static
    shapes, tensor position counters, shared memory rings) and form a valid
    quartet with the Stage 02 fixture encoder + vocoder
  * ORT stepped through WaitKPolicy == PyTorch step() for 5x the ring window,
    across tail flushes and state resets (a frozen position would diverge at
    the second chunk)
  * the reduced_dur comparison graphs run too (no contract: not v2)
"""
import json
import pathlib

import numpy as np
import pytest
import torch

from fvml import contract as fc
from fvml import s2ut_stream as ss
from fvml import translator as tr
from fvml.waitk_policy import WaitKConfig

from test_translator import make_model, random_latents, unit_teacher, utterance_stream

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "fixtures"
# The Stage 02 fixture bundle's geometry: latent_dim 8, vocab 16 (specials 16-18).
CFG = dict(latent_dim=8, vocab=16)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model = make_model(tr.tiny_config(**CFG), dtype=torch.float32, seed=21)
    record = ss.export_step_graphs(model, tmp_path_factory.mktemp("s2ut_onnx"), exporter="dynamo")
    return model, record


def specs(tensors):
    return {t.name: (t.dtype, tuple(t.shape)) for t in tensors}


def test_step_graphs_are_contract_v2(exported):
    model, rec = exported
    cfg = model.cfg
    # BUG-016: the exporter that produced the files is on record, with no silent fallback.
    assert (rec["encoder_exporter"], rec["decoder_exporter"], rec["fallback_reason"]) == \
        ("dynamo", "dynamo", None)
    enc_c, enc_proto = fc.load_onnx(rec["encoder"])        # parses + validate_io
    dec_c, dec_proto = fc.load_onnx(rec["decoder"])
    assert enc_c == model.encoder.contract() and dec_c == model.decoder.contract()
    assert (enc_c.role, dec_c.role, enc_c.contract_version) == ("translator_enc",
                                                                "translator_dec", 2)
    assert dict(dec_c.special_units) == {"bos": 16, "eos": 17, "pad": 18}
    assert enc_c.has_readiness_output and not dec_c.has_prosody_output

    h, w, dh = cfg.heads, cfg.max_src_steps, cfg.head_dim
    enc_in, enc_out = (specs(x) for x in fc.graph_io(enc_proto))
    assert enc_in["latents"] == ("f32", (1, 4, 8)) and enc_in["src_lang"] == ("i64", (1,))
    assert enc_in["state_in.src_pos"] == enc_out["state_out.src_pos"] == ("i64", (1,))
    assert enc_out["readiness_logit"] == ("f32", (1, 1))
    assert enc_in["state_in.enc_kv_0"] == ("f32", (1, 2 * h, w, dh))
    assert enc_in["state_in.enc_conv_1"] == ("f32", (1, cfg.d_model, cfg.conv_kernel - 1))
    dec_in, dec_out = (specs(x) for x in fc.graph_io(dec_proto))
    assert dec_in["prev_units"] == ("i64", (1, 4)) and dec_in["tgt_lang"] == ("i64", (1,))
    assert dec_in["prosody_ctx"] == ("f32", (1, 2)) and dec_in["src_pos"] == ("i64", (1,))
    assert dec_in["state_in.tgt_pos"] == dec_out["state_out.tgt_pos"] == ("i64", (1,))
    assert dec_out["units"] == ("i64", (1, 4)) and dec_out["eos_prob"] == ("f32", (1, 1))
    assert "logits" not in dec_out, "the debug output must not ship"
    # The decoder reads the very tensors the encoder writes: same names, dtype, shape.
    for name in enc_c.shared_mem_states:
        assert enc_out["state_out." + name] == dec_in[name] == ("f32", (1, h, w, dh))
        assert "state_out." + name not in dec_out
    assert list(enc_c.shared_mem_states) == ["mem_k_0", "mem_v_0", "mem_k_1", "mem_v_1"]
    fc.validate_s2ut_shared_io(enc_c, fc.graph_io(enc_proto)[1], dec_c, fc.graph_io(dec_proto)[0])

    # A valid Mode B quartet with the Stage 02 fixture encoder + vocoder and langs.json.
    fix_enc, _ = fc.load_onnx(FIXTURES / "s2ut" / "encoder_s4.onnx")
    fix_voc, _ = fc.load_onnx(FIXTURES / "s2ut" / "vocoder_s4.onnx")
    fc.validate_s2ut_bundle(fix_enc, enc_c, dec_c, fix_voc)
    fc.validate_langs(json.loads((FIXTURES / "s2ut" / "langs.json").read_text("utf-8")), enc_c)
    assert fc.s2ut_onset_latency_samples(fix_enc, enc_c, dec_c, fix_voc, 3) == 3 * 4 * 320


def test_ort_equals_torch_across_ring_wraps_and_resets(exported):
    """40 chunks of continuous speech (5x the 8-slot rings), then utterances
    with flushing pauses: same policy, same forced units, both runtimes."""
    model, rec = exported
    cfg = model.cfg
    logits = fc.TensorSpec("logits", "f32", (1, 4, cfg.n_classes))
    lat_a = random_latents(cfg, 40, seed=1, dtype=torch.float32)
    lat_b, speech_b = utterance_stream(cfg, [9, 13, 6], gap_chunks=12, seed=2, dtype=torch.float32)
    lat = torch.cat([lat_a, lat_b])
    speech = [True] * 40 + speech_b
    policy = WaitKConfig(k=3, max_flush_steps=5)
    streams = []
    for runtime in (ss.TorchRuntime(model, "fr", "en"),
                    ss.OrtRuntime(rec["encoder"], rec["decoder"], cfg, "fr", "en",
                                  debug_outputs=[logits])):
        st = ss.StreamingTranslator(runtime, policy, teacher=unit_teacher(cfg), record=True)
        readiness = []
        for i, sp in enumerate(speech):
            chunk = lat[i * 4:(i + 1) * 4]
            st.push_chunk(chunk if runtime.clock == "torch" else chunk.numpy(), sp)
            if i == 39:                                 # before any reset: 5 ring wraps
                assert runtime.src_pos == 40 and st.segment == 0
                mid = ({n: t.numpy().copy() for n, t in runtime.enc_state.items()}
                       if runtime.clock == "torch" else
                       {n[len("state_in."):]: a.copy() for n, a in runtime.enc_state.items()})
        streams.append((st, mid))
    (pt, pt_mid), (ort, ort_mid) = streams
    assert (pt.runtime.clock, ort.runtime.clock) == ("torch", "ort")
    assert pt.policy.flushes_completed == ort.policy.flushes_completed == 3
    assert [c["decision"] for c in pt.chunk_log] == [c["decision"] for c in ort.chunk_log]
    assert [d["src_visible"] for d in pt.dec_log] == [d["src_visible"] for d in ort.dec_log]
    a = np.stack([d["logits"] for d in pt.dec_log])
    b = np.stack([d["logits"] for d in ort.dec_log])
    delta = float(np.abs(a - b).max())
    print(f"\nORT vs PyTorch step(): {len(a)} decoder steps, max |dlogit| {delta:.3e}")
    assert len(a) > 60 and delta <= 1e-4
    assert np.array_equal(a.argmax(-1), b.argmax(-1))
    assert float(np.abs(a).max()) > 0.1, "parity on a silent output proves nothing (BUG-017)"
    # translator_enc state after 40 chunks, memory rings included.
    assert ort_mid["src_pos"].tolist() == [40]
    for name, ref in pt_mid.items():
        assert np.abs(ort_mid[name] - ref).max() <= 1e-5, name
    assert np.abs(pt_mid["mem_k_0"]).max() > 0.01


def test_reduced_dur_graphs_run_in_ort(tmp_path):
    cfg = tr.tiny_config(unit_mode="reduced_dur", max_duration=6, **CFG)
    model = make_model(cfg, dtype=torch.float32, seed=22)
    rec = ss.export_step_graphs(model, tmp_path, exporter="dynamo")
    assert rec["decoder_contract"] is None and rec["decoder_exporter"] == "dynamo"
    lat = random_latents(cfg, 30, seed=3, dtype=torch.float32)
    logs = []
    for runtime in (ss.TorchRuntime(model, "en", "de"),
                    ss.OrtRuntime(rec["encoder"], rec["decoder"], cfg, "en", "de")):
        st = ss.StreamingTranslator(runtime, WaitKConfig(k=2), eos_threshold=1.0, record=True)
        frames = []
        for i in range(30):
            chunk = lat[i * 4:(i + 1) * 4]
            frames += st.push_chunk(chunk if runtime.clock == "torch" else chunk.numpy()).frames
        logs.append((st.dec_log, frames))
    (pt, pt_frames), (ort, ort_frames) = logs
    assert len(pt) == len(ort) > 3 * cfg.max_tgt_steps      # free running, rings wrap
    assert [(d["units"].tolist(), d["duration"]) for d in pt] == \
        [(d["units"].tolist(), d["duration"]) for d in ort]
    assert pt_frames == ort_frames
    assert max(abs(x["eos_prob"] - y["eos_prob"]) for x, y in zip(pt, ort)) <= 1e-5
