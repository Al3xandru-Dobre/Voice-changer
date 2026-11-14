#!/usr/bin/env python3
"""Generate the tiny ONNX fixture models that honour model contracts v1 and v2.

Milestone 2 / Stage 02 (Mode A v1) and Milestone 3 / Stage 02 (Mode B v2). The
fixtures let the C++ runtime (M2 Stages 07-08, M3 Stages 07-08) be built and
tested without trained weights:

  identity/encoder_s{1,2,4}.onnx   v1 soft encoder: audio [1,hop] -> content [1,N,320]
  identity/vocoder_s{1,2,4}.onnx   v1 soft vocoder: content -> audio [1,hop] (passthrough)
  identity/voices.json             two voices for --voice / blending tests
  delay_s1.onnx                    v1 vocoder: audio(t) = content(t-1) via state_in/out.buf
  accumulator_s1.onnx              v1 encoder: state_out.acc = state_in.acc + sum(audio)
  s2ut/encoder_s{1,2,4}.onnx       v2 dual encoder: emits discrete units + f32 latents
  s2ut/translator_enc.onnx         v2 ring-KV encoder: writes latents summary to mem_k_0/
                                   mem_v_0 at (src_pos mod 4) and increments src_pos
  s2ut/translator_dec.onnx         v2 ring-KV decoder: reads delayed mem_k_0/mem_v_0 slot,
                                   emits (prev_units + tgt_lang) mod 16, prosody, and
                                   raises eos_prob = 1.0 after 3 flush steps
  s2ut/vocoder_s{1,2,4}.onnx       v2 discrete vocoder (vocab 16)
  s2ut/voices.json, s2ut/langs.json
  broken/*.onnx                    valid ONNX whose metadata or I/O breaks v1 or v2;
                                   manifest.json records the exact rejection message
  manifest.json                    every file's SHA256, contract string, graph I/O and
                                   expected error, plus Mode A pairs and Mode B quartets

Built with onnx.helper only (no torch), opset 17 / IR 8 so any ONNX Runtime
>= 1.13 loads them, with a fixed producer and no timestamps: the bytes
regenerate identically *for the onnx/protobuf versions pinned in
python_training/uv.lock*.

Usage:
    make_fixture_models.py [--out tests/data/fixtures]   # write (needs onnx)
    make_fixture_models.py --check      # rebuild in memory, compare with disk (needs onnx)
    make_fixture_models.py --self-test  # committed files vs. manifest: SHA256, embedded
                                        # contracts, expected errors, pairs/quartets (stdlib only)
"""
import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python_training"))
from fvml import contract as fc  # noqa: E402  (standard library only)

onnx = helper = None             # imported by _import_onnx() when generating


def _import_onnx():
    global onnx, helper
    import onnx as _onnx
    from onnx import helper as _helper
    onnx, helper = _onnx, _helper


OPSET = 17
IR_VERSION = 8
PRODUCER = "fluxvoice-fixtures"
PRODUCER_VERSION = "1"
RATE = 16000
FRAME = 320                      # 20 ms @ 16 kHz
SPK_DIM = 4
LICENSE = "CC0-1.0 (generated fixture, no trained weights)"
COND = ("logf0", "voicing", "aperiodicity", "energy", "spk_emb")
F32, I64 = 1, 7                  # onnx.TensorProto.FLOAT / INT64 (fixed by the ONNX spec)

# Contract v2 tiny fixture geometry (keeps total fixture directory well under 150 KB).
S2UT_VOCAB = 16
S2UT_SPECIAL = {"bos": 16, "eos": 17, "pad": 18}
S2UT_LATENT_DIM = 8              # 320 = 8 * 40
S2UT_CHUNK_FRAMES = 4            # 80 ms @ 50 Hz
S2UT_RING = 4                    # 4-slot ring memory for wrap-around unit tests
S2UT_SHARED_MEM = ("mem_k_0", "mem_v_0")
S2UT_LANGS = {"en": 0, "es": 1, "fr": 2, "de": 3}
S2UT_WAIT_K_TEST = (1, 3, 5, 10)


# -- graph building blocks ---------------------------------------------------
def value(name, elem, shape):
    """shape None = no shape information at all (unknown rank)."""
    return helper.make_tensor_value_info(name, elem, None if shape is None else list(shape))


def const_i64(name, values, shape=None):
    vals = list(values)
    return helper.make_tensor(name, I64, [len(vals)] if shape is None else list(shape), vals)


def const_f32(name, values, shape=None):
    vals = [float(v) for v in values]
    return helper.make_tensor(name, F32, [len(vals)] if shape is None else list(shape), vals)


def reshape(src, dst, shape, name):
    """Reshape node plus its shape initializer."""
    return (helper.make_node("Reshape", [src, name + "_shape"], [dst], name=name),
            const_i64(name + "_shape", shape))


def make(graph_name, inputs, outputs, nodes, inits, contract_text):
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=inits)
    model = helper.make_model(graph, producer_name=PRODUCER, producer_version=PRODUCER_VERSION,
                              opset_imports=[helper.make_opsetid("", OPSET)],
                              ir_version=IR_VERSION)
    if contract_text is not None:
        fc.embed(model, contract_text)
    onnx.checker.check_model(model, full_check=True)
    return model


def encoder_contract(fps, content_dim=FRAME, states=(), **overrides):
    fields = dict(role="encoder", sample_rate=RATE, frame_samples=FRAME, frames_per_step=fps,
                  lookahead_frames=0, content_kind="soft", content_dim=content_dim,
                  states=tuple(states), license=LICENSE)
    return fc.ModelContract(**{**fields, **overrides})


def vocoder_contract(fps, cond=COND, lookahead=0, states=(), **overrides):
    fields = dict(
        role="vocoder", sample_rate=RATE, frame_samples=FRAME, frames_per_step=fps,
        lookahead_frames=lookahead, content_kind="soft", content_dim=FRAME,
        cond_frame_samples=FRAME, cond_inputs=tuple(cond),
        spk_emb_dim=SPK_DIM if "spk_emb" in cond else 0, overlap_samples=0,
        states=tuple(states), license=LICENSE)
    return fc.ModelContract(**{**fields, **overrides})


def dual_encoder_contract(fps, states=(), **overrides):
    fields = dict(
        contract_version=2, role="encoder", sample_rate=RATE, frame_samples=FRAME,
        frames_per_step=fps, lookahead_frames=0, content_kind="discrete",
        vocab=S2UT_VOCAB, latent_dim=S2UT_LATENT_DIM, states=tuple(states), license=LICENSE)
    return fc.ModelContract(**{**fields, **overrides})


def translator_enc_contract(states=("src_pos", "enc_conv", "mem_k_0", "mem_v_0"), **overrides):
    fields = dict(
        contract_version=2, role="translator_enc", sample_rate=RATE, frame_samples=FRAME,
        chunk_frames=S2UT_CHUNK_FRAMES, latent_dim=S2UT_LATENT_DIM, max_src_steps=S2UT_RING,
        shared_mem_states=S2UT_SHARED_MEM, langs=S2UT_LANGS, has_readiness_output=True,
        states=tuple(states), license=LICENSE)
    return fc.ModelContract(**{**fields, **overrides})


def translator_dec_contract(states=("tgt_pos", "dec_kv_0"), **overrides):
    fields = dict(
        contract_version=2, role="translator_dec", sample_rate=RATE, frame_samples=FRAME,
        chunk_frames=S2UT_CHUNK_FRAMES, vocab=S2UT_VOCAB, special_units=S2UT_SPECIAL,
        max_src_steps=S2UT_RING, max_tgt_steps=S2UT_RING, shared_mem_states=S2UT_SHARED_MEM,
        langs=S2UT_LANGS, has_prosody_output=True, states=tuple(states), license=LICENSE)
    return fc.ModelContract(**{**fields, **overrides})


def s2ut_vocoder_contract(fps, cond=COND, **overrides):
    fields = dict(
        contract_version=2, role="vocoder", sample_rate=RATE, frame_samples=FRAME,
        frames_per_step=fps, lookahead_frames=0, content_kind="discrete", vocab=S2UT_VOCAB,
        cond_frame_samples=FRAME, cond_inputs=tuple(cond),
        spk_emb_dim=SPK_DIM if "spk_emb" in cond else 0, overlap_samples=0,
        states=(), license=LICENSE)
    return fc.ModelContract(**{**fields, **overrides})


# -- v1 fixtures -------------------------------------------------------------
def identity_encoder(fps, text=None, audio_name="audio", hop=None, out_name="content",
                     out_elem=F32):
    c = encoder_contract(fps)
    hop = c.hop if hop is None else hop
    n = (hop // FRAME) if hop > 0 else -1
    node, init = reshape(audio_name, out_name, [1, n, FRAME], "to_frames")
    return make(f"identity_encoder_s{fps}",
                [value(audio_name, F32, [1, hop if hop > 0 else "T"])],
                [value(out_name, out_elem, [1, n if n > 0 else "N", FRAME])],
                [node], [init], c.to_json() if text is None else text)


def identity_vocoder(fps):
    c = vocoder_contract(fps)
    node, init = reshape("content", "audio", [1, c.hop], "to_audio")
    inputs = [value("content", F32, [1, fps, FRAME])]
    inputs += [value(name, F32, [1, SPK_DIM if name == "spk_emb" else c.cond_frames])
               for name in c.cond_inputs]
    return make(f"identity_vocoder_s{fps}", inputs, [value("audio", F32, [1, c.hop])],
                [node], [init], c.to_json())


def delay_vocoder(text=None, with_state_out=True):
    """audio = previous hop's content; the state carries one hop."""
    c = vocoder_contract(1, cond=(), lookahead=1, states=("buf",))
    nodes = [helper.make_node("Identity", ["state_in.buf"], ["audio"], name="emit_previous")]
    inits = []
    outputs = [value("audio", F32, [1, FRAME])]
    if with_state_out:
        node, init = reshape("content", "state_out.buf", [1, FRAME], "keep_current")
        nodes.append(node)
        inits.append(init)
        outputs.append(value("state_out.buf", F32, [1, FRAME]))
    return make("delay_vocoder_s1",
                [value("content", F32, [1, 1, FRAME]), value("state_in.buf", F32, [1, FRAME])],
                outputs, nodes, inits, c.to_json() if text is None else text)


def accumulator_encoder():
    c = encoder_contract(1, content_dim=1, states=("acc",))
    sum_node = helper.make_node("ReduceSum", ["audio", "axes"], ["hop_sum"], keepdims=1,
                                name="sum_hop")
    add_node = helper.make_node("Add", ["state_in.acc", "hop_sum"], ["state_out.acc"],
                                name="accumulate")
    out_node, out_init = reshape("state_out.acc", "content", [1, 1, 1], "to_content")
    return make("accumulator_encoder_s1",
                [value("audio", F32, [1, FRAME]), value("state_in.acc", F32, [1, 1])],
                [value("content", F32, [1, 1, 1]), value("state_out.acc", F32, [1, 1])],
                [sum_node, add_node, out_node], [const_i64("axes", [1]), out_init], c.to_json())


# -- v2 S2UT fixtures --------------------------------------------------------
def dual_encoder(fps, text=None, with_latents=True):
    """v2 dual-output encoder: emits discrete `units` [1,fps] i64 and continuous
    `latents` [1,fps,8] f32 (each latent bin is the exact mean of 40 audio samples)."""
    c = dual_encoder_contract(fps)
    sub = FRAME // S2UT_LATENT_DIM
    n_4d, i_4d = reshape("audio", "frames_4d", [1, fps, S2UT_LATENT_DIM, sub], "to_4d")
    lat_out = "latents" if with_latents else "latents_internal"
    nodes = [
        n_4d,
        helper.make_node("ReduceSum", ["frames_4d", "ax3"], ["lat_sum"], keepdims=0, name="sum_sub"),
        helper.make_node("Mul", ["lat_sum", "inv_sub"], [lat_out], name="mean_sub"),
        helper.make_node("ReduceSum", [lat_out, "ax2"], ["frame_sum"], keepdims=0, name="sum_lat"),
        helper.make_node("Abs", ["frame_sum"], ["frame_abs"], name="abs_lat"),
        helper.make_node("Mul", ["frame_abs", "vocab_f"], ["frame_scaled"], name="scale_vocab"),
        helper.make_node("Cast", ["frame_scaled"], ["frame_i64"], to=I64, name="cast_units"),
        helper.make_node("Mod", ["frame_i64", "vocab_i"], ["units"], fmod=0, name="mod_units"),
    ]
    inits = [
        i_4d,
        const_i64("ax3", [3]),
        const_f32("inv_sub", [1.0 / sub]),
        const_i64("ax2", [2]),
        const_f32("vocab_f", [float(S2UT_VOCAB)]),
        const_i64("vocab_i", [S2UT_VOCAB]),
    ]
    outputs = [value("units", I64, [1, fps])]
    if with_latents:
        outputs.append(value("latents", F32, [1, fps, S2UT_LATENT_DIM]))
    return make(f"dual_encoder_s{fps}", [value("audio", F32, [1, c.hop])], outputs,
                nodes, inits, c.to_json() if text is None else text)


def fixture_translator_enc(text=None, with_src_pos=True):
    """Writes chunk latents summary into 4-slot ring mem_k_0 / mem_v_0 at
    (src_pos mod 4) and increments src_pos."""
    c = translator_enc_contract()
    n_lang, i_lang = reshape("src_lang_f", "src_lang_3d", [1, 1, 1], "lang_3d")
    n_conv, i_conv = reshape("v_slot", "v_2d", [1, S2UT_LATENT_DIM], "conv_2d")
    n_mask, i_mask = reshape("slot_eq", "slot_mask", [1, S2UT_RING, 1], "mask_3d")
    pos_in = "state_in.src_pos" if with_src_pos else "zero_pos"
    nodes = [
        helper.make_node("ReduceSum", ["latents", "ax1"], ["lat_sum"], keepdims=1, name="sum_chunk"),
        helper.make_node("Mul", ["lat_sum", "inv_chunk"], ["v_slot"], name="mean_chunk"),
        helper.make_node("Cast", ["src_lang"], ["src_lang_f"], to=F32, name="cast_lang"),
        n_lang,
        helper.make_node("Add", ["v_slot", "src_lang_3d"], ["k_slot"], name="add_lang"),
        n_conv,
        helper.make_node("Mul", ["state_in.enc_conv", "zero_f"], ["conv_keep"], name="touch_conv"),
        helper.make_node("Add", ["v_2d", "conv_keep"], ["state_out.enc_conv"], name="next_conv"),
        helper.make_node("Mod", [pos_in, "ring_n"], ["slot"], fmod=0, name="ring_slot"),
        helper.make_node("Equal", ["slots_0_3", "slot"], ["slot_eq"], name="eq_slot"),
        n_mask,
        helper.make_node("Where", ["slot_mask", "k_slot", "state_in.mem_k_0"],
                         ["state_out.mem_k_0"], name="write_mem_k"),
        helper.make_node("Where", ["slot_mask", "v_slot", "state_in.mem_v_0"],
                         ["state_out.mem_v_0"], name="write_mem_v"),
        helper.make_node("ReduceSum", ["v_2d", "ax1"], ["r_sum"], keepdims=1, name="sum_ready"),
        helper.make_node("Mul", ["r_sum", "inv_lat"], ["readiness_logit"], name="mean_ready"),
    ]
    inits = [
        i_lang, i_conv, i_mask,
        const_i64("ax1", [1]),
        const_f32("inv_chunk", [1.0 / S2UT_CHUNK_FRAMES]),
        const_f32("inv_lat", [1.0 / S2UT_LATENT_DIM]),
        const_f32("zero_f", [0.0]),
        const_i64("ring_n", [S2UT_RING]),
        const_i64("slots_0_3", list(range(S2UT_RING))),
    ]
    inputs = [
        value("latents", F32, [1, S2UT_CHUNK_FRAMES, S2UT_LATENT_DIM]),
        value("src_lang", I64, [1]),
    ]
    outputs = [value("readiness_logit", F32, [1, 1])]
    if with_src_pos:
        nodes.append(helper.make_node("Add", ["state_in.src_pos", "one_i64"],
                                      ["state_out.src_pos"], name="inc_src_pos"))
        inits.append(const_i64("one_i64", [1]))
        inputs.append(value("state_in.src_pos", I64, [1]))
        outputs.append(value("state_out.src_pos", I64, [1]))
    else:
        inits.append(const_i64("zero_pos", [0]))
    inputs += [
        value("state_in.enc_conv", F32, [1, S2UT_LATENT_DIM]),
        value("state_in.mem_k_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]),
        value("state_in.mem_v_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]),
    ]
    outputs += [
        value("state_out.enc_conv", F32, [1, S2UT_LATENT_DIM]),
        value("state_out.mem_k_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]),
        value("state_out.mem_v_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]),
    ]
    return make("fixture_translator_enc", inputs, outputs, nodes, inits,
                c.to_json() if text is None else text)


def fixture_translator_dec(text=None, with_mem_v=True):
    """Reads mem_k_0 / mem_v_0 at delayed slot (min(tgt_pos, max(src_pos-1, 0)) mod 4),
    emits deterministic units ((prev_units + tgt_lang) mod 16), prosody carrying
    the read slot's summary + prosody_ctx, increments tgt_pos, and raises
    eos_prob = 1.0 once state_out.tgt_pos > src_pos (3rd flush step at wait_k=3)."""
    c = translator_dec_contract()
    n_tgt, i_tgt = reshape("tgt_lang", "tgt_2d", [1, 1], "tgt_lang_2d")
    n_pf, i_pf = reshape("p_frame", "p_frame_3d", [1, 1, 4], "prosody_3d")
    n_mask, i_mask = reshape("tgt_eq", "tgt_mask", [1, S2UT_RING, 1], "tgt_mask_3d")
    n_eos, i_eos = reshape("eos_1d", "eos_prob", [1, 1], "eos_2d")
    v_source = "mem_v_0" if with_mem_v else "mem_k_0"
    nodes = [
        helper.make_node("Sub", ["src_pos", "one_i64"], ["last_src"], name="sub_src"),
        helper.make_node("Max", ["last_src", "zero_i64"], ["valid_src"], name="clamp_lo"),
        helper.make_node("Min", ["state_in.tgt_pos", "valid_src"], ["read_pos"], name="clamp_hi"),
        helper.make_node("Mod", ["read_pos", "ring_n"], ["read_slot"], fmod=0, name="mod_read"),
        helper.make_node("Gather", ["mem_k_0", "read_slot"], ["k_read"], axis=1, name="gather_k"),
        helper.make_node("Gather", [v_source, "read_slot"], ["v_read"], axis=1, name="gather_v"),
        n_tgt,
        helper.make_node("Add", ["prev_units", "tgt_2d"], ["u_sum"], name="add_lang"),
        helper.make_node("Mod", ["u_sum", "vocab_i"], ["units"], fmod=0, name="mod_units"),
        helper.make_node("ReduceSum", ["k_read", "ax2"], ["k_sum"], keepdims=0, name="sum_k"),
        helper.make_node("Mul", ["k_sum", "inv_lat"], ["k_mean"], name="mean_k"),
        helper.make_node("ReduceSum", ["v_read", "ax2"], ["v_sum"], keepdims=0, name="sum_v"),
        helper.make_node("Mul", ["v_sum", "inv_lat"], ["v_mean"], name="mean_v"),
        helper.make_node("Gather", ["prosody_ctx", "zero_i64"], ["ctx_f0"], axis=1, name="ctx_0"),
        helper.make_node("Gather", ["prosody_ctx", "one_i64"], ["ctx_en"], axis=1, name="ctx_1"),
        helper.make_node("Add", ["k_mean", "ctx_f0"], ["p_f0"], name="pros_f0"),
        helper.make_node("Add", ["v_mean", "ctx_en"], ["p_en"], name="pros_en"),
        helper.make_node("Concat", ["p_f0", "voicing_c", "aper_c", "p_en"], ["p_frame"],
                         axis=1, name="pack_prosody"),
        n_pf,
        helper.make_node("Add", ["p_frame_3d", "pros_zeros"], ["prosody"], name="tile_prosody"),
        helper.make_node("Mod", ["state_in.tgt_pos", "ring_n"], ["tgt_slot"], fmod=0,
                         name="mod_tgt"),
        helper.make_node("Equal", ["slots_0_3", "tgt_slot"], ["tgt_eq"], name="eq_tgt"),
        n_mask,
        helper.make_node("Where", ["tgt_mask", "v_read", "state_in.dec_kv_0"],
                         ["state_out.dec_kv_0"], name="write_dec_kv"),
        helper.make_node("Add", ["state_in.tgt_pos", "one_i64"], ["state_out.tgt_pos"],
                         name="inc_tgt_pos"),
        helper.make_node("Greater", ["state_out.tgt_pos", "src_pos"], ["is_eos"], name="check_eos"),
        helper.make_node("Cast", ["is_eos"], ["eos_1d"], to=F32, name="cast_eos"),
        n_eos,
    ]
    inits = [
        i_tgt, i_pf, i_mask, i_eos,
        const_i64("zero_i64", [0]),
        const_i64("one_i64", [1]),
        const_i64("ring_n", [S2UT_RING]),
        const_i64("vocab_i", [S2UT_VOCAB]),
        const_i64("ax2", [2]),
        const_i64("slots_0_3", list(range(S2UT_RING))),
        const_f32("inv_lat", [1.0 / S2UT_LATENT_DIM]),
        const_f32("voicing_c", [0.9], shape=[1, 1]),
        const_f32("aper_c", [0.1], shape=[1, 1]),
        const_f32("pros_zeros", [0.0] * (S2UT_CHUNK_FRAMES * 4), shape=[1, S2UT_CHUNK_FRAMES, 4]),
    ]
    inputs = [
        value("prev_units", I64, [1, S2UT_CHUNK_FRAMES]),
        value("tgt_lang", I64, [1]),
        value("prosody_ctx", F32, [1, 2]),
        value("src_pos", I64, [1]),
        value("state_in.tgt_pos", I64, [1]),
        value("mem_k_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]),
    ]
    if with_mem_v:
        inputs.append(value("mem_v_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]))
    inputs.append(value("state_in.dec_kv_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]))
    outputs = [
        value("units", I64, [1, S2UT_CHUNK_FRAMES]),
        value("prosody", F32, [1, S2UT_CHUNK_FRAMES, 4]),
        value("eos_prob", F32, [1, 1]),
        value("state_out.tgt_pos", I64, [1]),
        value("state_out.dec_kv_0", F32, [1, S2UT_RING, S2UT_LATENT_DIM]),
    ]
    return make("fixture_translator_dec", inputs, outputs, nodes, inits,
                c.to_json() if text is None else text)


def s2ut_vocoder(fps):
    """v2 discrete vocoder: maps (units / 16.0) * voicing to 320-sample frames."""
    c = s2ut_vocoder_contract(fps)
    n_3d, i_3d = reshape("u_voiced", "u_3d", [1, fps, 1], "u_to_3d")
    n_out, i_out = reshape("a_3d", "audio", [1, c.hop], "to_audio")
    nodes = [
        helper.make_node("Cast", ["units"], ["u_f"], to=F32, name="cast_u"),
        helper.make_node("Mul", ["u_f", "inv_vocab"], ["u_norm"], name="norm_u"),
        helper.make_node("Mul", ["u_norm", "voicing"], ["u_voiced"], name="apply_voicing"),
        n_3d,
        helper.make_node("Add", ["u_3d", "frame_zeros"], ["a_3d"], name="tile_frame"),
        n_out,
    ]
    inits = [
        i_3d, i_out,
        const_f32("inv_vocab", [1.0 / S2UT_VOCAB]),
        const_f32("frame_zeros", [0.0] * FRAME, shape=[1, 1, FRAME]),
    ]
    inputs = [value("units", I64, [1, fps])]
    inputs += [value(name, F32, [1, SPK_DIM if name == "spk_emb" else c.cond_frames])
               for name in c.cond_inputs]
    return make(f"s2ut_vocoder_s{fps}", inputs, [value("audio", F32, [1, c.hop])],
                nodes, inits, c.to_json())


def voices_json():
    def voice(name, display, emb):
        return {"name": name, "display": display, "embedding": emb,
                "logf0_mean": 4.8, "logf0_std": 0.25, "energy_mean": -4.0, "energy_std": 1.5,
                "license": LICENSE, "source": "tools/make_fixture_models.py"}
    return {"schema": 1, "voices": [voice("identity", "Identity A (passthrough)", [0.0] * SPK_DIM),
                                    voice("identity_b", "Identity B (passthrough)", [1.0] * SPK_DIM)]}


def langs_json():
    return {
        "schema": 1,
        "langs": [
            {"code": "en", "id": 0, "name": "English"},
            {"code": "es", "id": 1, "name": "Spanish"},
            {"code": "fr", "id": 2, "name": "French"},
            {"code": "de", "id": 3, "name": "German"},
        ],
        "default_wait_k": {
            "es->en": 3, "fr->en": 3, "de->en": 4,
            "en->es": 3, "en->fr": 3, "en->de": 4,
        },
    }


def broken_models():
    """(file, model, expected error) — each is valid ONNX but breaks v1 or v2."""
    s1 = encoder_contract(1)
    no_rate = {k: v for k, v in s1.to_dict().items() if k != "sample_rate"}
    v3 = {**s1.to_dict(), "contract_version": 3}
    discrete = encoder_contract(1, content_kind="discrete", content_dim=0, vocab=FRAME)
    t_enc_no_src_pos = {**translator_enc_contract().to_dict(),
                        "states": ["enc_conv", "mem_k_0", "mem_v_0"]}
    t_dec_bad_special = {**translator_dec_contract().to_dict(),
                         "special_units": {"bos": 5, "eos": 17, "pad": 18}}
    dump = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))  # noqa: E731
    return [
        ("broken/no_contract.onnx", _without_contract(identity_encoder(1)),
         fc.MISSING_METADATA),
        ("broken/not_json.onnx", identity_encoder(1, text='{"contract_version": 1, "role": '),
         "contract is not valid JSON"),
        ("broken/unknown_version.onnx", identity_encoder(1, text=dump(v3)),
         "unsupported contract_version 3 (this build reads versions 1 and 2)"),
        ("broken/missing_key.onnx", identity_encoder(1, text=dump(no_rate)),
         "missing key 'sample_rate'"),
        ("broken/wrong_input_name.onnx", identity_encoder(1, audio_name="wav"),
         "missing graph input 'audio'"),
        ("broken/wrong_shape.onnx", identity_encoder(1, hop=2 * FRAME),
         "graph input 'audio' has shape [1,640], contract expects [1,320]"),
        ("broken/dynamic_shape.onnx", identity_encoder(1, hop=-1),
         "graph input 'audio' has shape [1,?], contract expects [1,320]"),
        ("broken/wrong_dtype.onnx",
         identity_encoder(1, text=discrete.to_json(), out_name="units"),
         "graph output 'units' has dtype f32, contract expects i64"),
        ("broken/unpaired_state.onnx", delay_vocoder(with_state_out=False),
         "unpaired state 'buf': state_in.buf has no state_out.buf"),
        ("broken/unlisted_state.onnx",
         delay_vocoder(text=vocoder_contract(1, cond=(), lookahead=1).to_json()),
         "graph input 'state_in.buf' is not a state listed in the contract"),
        # Contract v2 negative fixtures (for Stage 07 OrtModel loader tests):
        ("broken/v2_enc_missing_latents.onnx", dual_encoder(4, with_latents=False),
         "missing graph output 'latents'"),
        ("broken/v2_trans_enc_missing_src_pos.onnx",
         fixture_translator_enc(text=dump(t_enc_no_src_pos), with_src_pos=False),
         "translator_enc states must include 'src_pos'"),
        ("broken/v2_trans_dec_missing_mem.onnx", fixture_translator_dec(with_mem_v=False),
         "missing graph input 'mem_v_0'"),
        ("broken/v2_trans_dec_bad_special_units.onnx",
         fixture_translator_dec(text=dump(t_dec_bad_special)),
         "special_units['bos'] (5) must be >= vocab (16)"),
    ]


def _without_contract(model):
    del model.metadata_props[:]
    return model


def build_all():
    """{relative path: bytes} for every generated file."""
    _import_onnx()
    models = []
    for fps in (1, 2, 4):
        models.append((f"identity/encoder_s{fps}.onnx", identity_encoder(fps), None))
        models.append((f"identity/vocoder_s{fps}.onnx", identity_vocoder(fps), None))
    models.append(("delay_s1.onnx", delay_vocoder(), None))
    models.append(("accumulator_s1.onnx", accumulator_encoder(), None))
    for fps in (1, 2, 4):
        models.append((f"s2ut/encoder_s{fps}.onnx", dual_encoder(fps), None))
        models.append((f"s2ut/vocoder_s{fps}.onnx", s2ut_vocoder(fps), None))
    models.append(("s2ut/translator_enc.onnx", fixture_translator_enc(), None))
    models.append(("s2ut/translator_dec.onnx", fixture_translator_dec(), None))
    models += broken_models()

    files, entries, contracts, graph_by_file = {}, [], {}, {}
    for rel, model, want_error in models:
        data = model.SerializeToString(deterministic=True)
        files[rel] = data
        inputs, outputs = fc.graph_io(model)
        graph_by_file[rel] = (inputs, outputs)
        text = fc.read_contract_text(model)
        try:
            contracts[rel] = fc.check_model(text, inputs, outputs)
            error = None
        except fc.ContractError as e:
            error = str(e)
        if (want_error is None) != (error is None) or (want_error and error != want_error):
            raise SystemExit(f"{rel}: expected error {want_error!r}, got {error!r}")
        entries.append({"file": rel, "sha256": hashlib.sha256(data).hexdigest(), "contract": text,
                        "graph": {"inputs": [t.to_dict() for t in inputs],
                                  "outputs": [t.to_dict() for t in outputs]},
                        "error": error})

    pairs = []
    for name, enc, voc in [(f"identity_s{fps}", f"identity/encoder_s{fps}.onnx",
                            f"identity/vocoder_s{fps}.onnx") for fps in (1, 2, 4)] + \
            [("delay_s1", "identity/encoder_s1.onnx", "delay_s1.onnx")]:
        fc.validate_pair(contracts[enc], contracts[voc])
        pairs.append({"name": name, "encoder": enc, "vocoder": voc,
                      "algorithmic_latency_samples":
                          fc.algorithmic_latency(contracts[enc], contracts[voc])})

    s2ut_quartets = []
    for fps in (1, 2, 4):
        enc = f"s2ut/encoder_s{fps}.onnx"
        t_enc = "s2ut/translator_enc.onnx"
        t_dec = "s2ut/translator_dec.onnx"
        voc = f"s2ut/vocoder_s{fps}.onnx"
        fc.validate_s2ut_bundle(contracts[enc], contracts[t_enc], contracts[t_dec], contracts[voc])
        fc.validate_s2ut_shared_io(contracts[t_enc], graph_by_file[t_enc][1],
                                   contracts[t_dec], graph_by_file[t_dec][0])
        onset = {str(k): fc.s2ut_onset_latency_samples(
            contracts[enc], contracts[t_enc], contracts[t_dec], contracts[voc], k)
            for k in S2UT_WAIT_K_TEST}
        s2ut_quartets.append({
            "name": f"s2ut_s{fps}", "encoder": enc, "translator_enc": t_enc,
            "translator_dec": t_dec, "vocoder": voc, "onset_latency_samples": onset,
        })

    voices = voices_json()
    fc.validate_voices(voices, contracts["identity/vocoder_s1.onnx"])
    fc.validate_voices(voices, contracts["s2ut/vocoder_s4.onnx"])
    langs = langs_json()
    fc.validate_langs(langs, contracts["s2ut/translator_enc.onnx"])
    files["identity/voices.json"] = _json_bytes(voices)
    files["s2ut/voices.json"] = _json_bytes(voices)
    files["s2ut/langs.json"] = _json_bytes(langs)

    manifest = {
        "generator": "tools/make_fixture_models.py",
        "opset": OPSET,
        "ir_version": IR_VERSION,
        "models": entries,
        "pairs": pairs,
        "s2ut_quartets": s2ut_quartets,
        "bundles": [
            {"dir": "identity", "vocoder": "identity/vocoder_s1.onnx",
             "voices": "identity/voices.json"},
            {"dir": "s2ut", "vocoder": "s2ut/vocoder_s4.onnx",
             "voices": "s2ut/voices.json", "translator": "s2ut/translator_enc.onnx",
             "langs": "s2ut/langs.json"},
        ],
    }
    files["manifest.json"] = _json_bytes(manifest)
    return files


def _json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def check(out, files):
    problems = []
    for rel, data in sorted(files.items()):
        path = out / rel
        if not path.is_file():
            problems.append(f"missing  {rel}")
            continue
        disk = path.read_bytes()
        if rel.endswith(".json"):  # git may check text out with CRLF on Windows
            disk = disk.replace(b"\r\n", b"\n")
        if disk != data:
            problems.append(f"differs  {rel}")
    problems += [f"stale    {rel}" for rel in stale_files(out, files)]
    for p in problems:
        print(p)
    print(f"{len(files)} files checked: " + ("up to date" if not problems else
                                              f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def stale_files(out, expected):
    """Files under `out` that the generator does not produce (README.md is hand-written)."""
    return [p.relative_to(out).as_posix() for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "README.md" and p.relative_to(out).as_posix() not in expected]


def self_test(out):
    """Standard library only: the committed fixtures agree with manifest.json
    (hashes, embedded contract strings, expected errors, pairs, S2UT quartets,
    voices, langs) and nothing stale lies around. Regenerating (--check) needs
    onnx; this does not."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    failures = []

    def check_that(ok, name, why):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f" -> {why}"))
        if not ok:
            failures.append(name)

    contracts, graphs = {}, {}
    total_bytes = 0
    for m in manifest["models"]:
        rel = m["file"]
        data = (out / rel).read_bytes() if (out / rel).is_file() else b""
        total_bytes += len(data)
        check_that(hashlib.sha256(data).hexdigest() == m["sha256"], f"{rel} sha256", "file differs")
        check_that(m["contract"] is None or m["contract"].encode("utf-8") in data,
                   f"{rel} contract", "embedded string not found")
        ins = [fc.TensorSpec.from_dict(t) for t in m["graph"]["inputs"]]
        outs = [fc.TensorSpec.from_dict(t) for t in m["graph"]["outputs"]]
        graphs[rel] = (ins, outs)
        try:
            contracts[rel] = fc.check_model(m["contract"], ins, outs)
            error = None
        except fc.ContractError as e:
            error = str(e)
        check_that(error == m["error"], f"{rel} verdict", f"got {error!r}, want {m['error']!r}")
    for p in manifest["pairs"]:
        enc, voc = contracts.get(p["encoder"]), contracts.get(p["vocoder"])
        ok = enc is not None and voc is not None
        if ok:
            fc.validate_pair(enc, voc)
            ok = fc.algorithmic_latency(enc, voc) == p["algorithmic_latency_samples"]
        check_that(ok, f"pair {p['name']}", "invalid pair or latency differs")
    for q in manifest.get("s2ut_quartets", []):
        enc = contracts.get(q["encoder"])
        t_enc = contracts.get(q["translator_enc"])
        t_dec = contracts.get(q["translator_dec"])
        voc = contracts.get(q["vocoder"])
        try:
            fc.validate_s2ut_bundle(enc, t_enc, t_dec, voc)
            fc.validate_s2ut_shared_io(t_enc, graphs[q["translator_enc"]][1],
                                       t_dec, graphs[q["translator_dec"]][0])
            ok = all(fc.s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, int(k)) == v
                     for k, v in q["onset_latency_samples"].items())
            check_that(ok, f"quartet {q['name']}", "onset latency differs")
        except (fc.ContractError, AttributeError, KeyError) as e:
            check_that(False, f"quartet {q['name']}", str(e))
    json_files_set = {"manifest.json"}
    for b in manifest["bundles"]:
        json_files_set.add(b["voices"])
        try:
            fc.validate_voices(json.loads((out / b["voices"]).read_text(encoding="utf-8")),
                               contracts[b["vocoder"]])
            check_that(True, b["voices"], "")
        except (fc.ContractError, OSError, KeyError) as e:
            check_that(False, b["voices"], str(e))
        if "langs" in b:
            json_files_set.add(b["langs"])
            try:
                fc.validate_langs(json.loads((out / b["langs"]).read_text(encoding="utf-8")),
                                  contracts[b["translator"]])
                check_that(True, b["langs"], "")
            except (fc.ContractError, OSError, KeyError) as e:
                check_that(False, b["langs"], str(e))
    for jf in json_files_set:
        if (out / jf).is_file():
            total_bytes += (out / jf).stat().st_size
    check_that(total_bytes <= 150 * 1024, f"fixture size ({total_bytes} B <= 153600 B)",
               f"{total_bytes} bytes exceeds 150 KB cap")
    expected = {m["file"] for m in manifest["models"]} | json_files_set
    stale = stale_files(out, expected)
    check_that(not stale, "no stale files", ", ".join(stale))
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(REPO / "tests" / "data" / "fixtures"),
                   help="output directory (default tests/data/fixtures)")
    p.add_argument("--check", action="store_true",
                   help="compare a fresh build with the files on disk; write nothing")
    p.add_argument("--self-test", action="store_true",
                   help="check the committed fixtures against manifest.json (no onnx needed)")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    if args.self_test:
        return self_test(out)
    files = build_all()
    if args.check:
        return check(out, files)
    for rel, data in sorted(files.items()):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    total = sum(len(d) for d in files.values())
    print(f"wrote {len(files)} files to {out} ({total} bytes, onnx {onnx.__version__}, "
          f"opset {OPSET}, IR {IR_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
