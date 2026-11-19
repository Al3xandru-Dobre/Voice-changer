"""Milestone 3 / Stage 07 golden parity: s2ut_golden.py (Python ORT twin of the
C++ Mode B pipeline, src/pipeline/InferencePipeline.cpp) on the v2 fixture
bundle, and fv_s2ut_file against it when FLUXVOICE_S2UT_FILE points to the
binary (skipped otherwise).

Fixture math (tools/make_fixture_models.py): encoder latents are 40-sample bin
means; translator_dec emits units (prev_units + tgt_lang) % 16 from <BOS> = 16,
prosody voicing 0.9 and eos_prob = (tgt_pos + 1 > src_pos), i.e. on the third
flush step at k = 3; the vocoder renders unit / 16 * voicing per 320-sample
frame. es -> de (tgt id 3), so an utterance's n-th write is unit 3n % 16 and the
first translated hop is 3 / 16 * 0.9.

Input: 1.2 s of a 0.25-amplitude 200 Hz sine (15 chunks), 1.0 s of silence (a
pause > 400 ms: tail flush), 0.8 s of sine, 0.3 s of silence. Every edge is on a
20 ms frame boundary, so every frame's VAD level is -15 dBFS or -160 dBFS.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

import s2ut_golden as cli  # noqa: E402
from fvml import s2ut_eval  # noqa: E402
from fvml import s2ut_golden as G  # noqa: E402
from fvml import waitk_policy as W  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAINING = REPO / "python_training"
BUNDLE = REPO / "tests" / "data" / "fixtures" / "s2ut"
SR, FRAME, CHUNK, VOCAB = 16000, 320, 4, 16
BOS, EOS, PAD = 16, 17, 18
TGT_ID = 3                                  # de
VOICING = np.float32(0.9)
# Utterance 1: 15 speech chunks + 5 pause chunks (<= 400 ms) are encoded; the
# first k - 1 = 2 READ, the other 18 WRITE; the 6th pause chunk starts the tail
# flush, <EOS> on its 3rd step. 4 idle READ (pause, not encoded). Utterance 2:
# 11 speech + 5 pause chunks -> 2 READ + 14 WRITE, then 3 FLUSH.
DECISIONS = "RR" + "W" * 18 + "FFF" + "RRRR" + "RR" + "W" * 14 + "FFF"
# fv_s2ut_file renders one more zero hop after the pipeline is idle when the
# last output sample is audible (|x| > 0.001), as the fixture's final <EOS>
# write (unit 3) is; with chunk-sized hops that hop completes one idle chunk.
AUDIBLE_TAIL = "R"
UTTERANCE_ENDS = [22, 45]
IDLE_STEPS = [23, 24, 25, 26]
INPUT_SAMPLES = 52800                       # 3.3 s
EOS_CHUNK = [5, EOS, PAD, PAD]              # what the patched decoder emits on <EOS>


def two_utterances():
    def sine(seconds):
        t = np.arange(int(round(seconds * SR))) / SR
        return (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    def silence(seconds):
        return np.zeros(int(round(seconds * SR)), np.float32)

    return np.concatenate([sine(1.2), silence(1.0), sine(0.8), silence(0.3)])


@pytest.fixture(scope="module")
def in_wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("s2ut_in") / "in.wav"
    x = two_utterances()
    assert x.size == INPUT_SAMPLES
    G.write_wav_f32(path, SR, x)
    return path


def patch_bundle(out, eos_units=False, eos_at_lag_one=False, no_eos=False, unit_gain=False,
                 readiness_logit=None, eos_from_src=None):
    """A copy of the fixture bundle with graph edits (same I/O, same contracts):
    eos_units       translator_dec emits [5, EOS, PAD, PAD] on its <EOS> step;
    eos_at_lag_one  eos_prob = (tgt_pos + 1 >= src_pos): with k = 1 every write
                    reports <EOS>, so the first pause write marks the utterance
                    caught up and the pause completes it without a flush step;
    no_eos          eos_prob = 0: flushes run to the bound ceil(1.5 * t_src);
    eos_from_src    eos_prob = (src_pos >= this): every write once the source
                    has this many chunks reports <EOS>;
    unit_gain       vocoders ignore voicing (audio = unit / 16);
    readiness_logit translator_enc's readiness_logit is this constant."""
    shutil.copytree(BUNDLE, out)
    if readiness_logit is not None:
        enc = onnx.load(str(BUNDLE / "translator_enc.onnx"))
        (node,) = [n for n in enc.graph.node if n.name == "mean_ready"]
        node.input[1], node.output[0] = "zero_gain", "readiness_zero"
        enc.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.asarray([0.0], np.float32), "zero_gain"),
            onnx.numpy_helper.from_array(np.asarray([readiness_logit], np.float32), "logit_c")])
        enc.graph.node.append(onnx.helper.make_node(
            "Add", ["readiness_zero", "logit_c"], ["readiness_logit"], name="const_ready"))
        onnx.checker.check_model(enc, full_check=True)
        onnx.save(enc, str(out / "translator_enc.onnx"))
    dec = onnx.load(str(BUNDLE / "translator_dec.onnx"))
    if no_eos:
        (node,) = [n for n in dec.graph.node if n.name == "cast_eos"]
        node.input[0] = "never"
        dec.graph.initializer.append(onnx.numpy_helper.from_array(np.asarray([False]), "never"))
    if eos_units:
        (node,) = [n for n in dec.graph.node if "units" in n.output]
        node.output[list(node.output).index("units")] = "units_raw"
        dec.graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray([EOS_CHUNK], np.int64), "eos_units"))
        dec.graph.node.append(onnx.helper.make_node(
            "Where", ["is_eos", "eos_units", "units_raw"], ["units"], name="eos_chunk"))
    if eos_at_lag_one:
        (node,) = [n for n in dec.graph.node if n.name == "check_eos"]
        node.op_type = "GreaterOrEqual"
    if eos_from_src is not None:
        (node,) = [n for n in dec.graph.node if n.name == "check_eos"]
        node.input[0], node.input[1] = "src_pos", "eos_from"     # src_pos > eos_from - 1
        dec.graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray([eos_from_src - 1], np.int64), "eos_from"))
    onnx.checker.check_model(dec, full_check=True)
    onnx.save(dec, str(out / "translator_dec.onnx"))
    for fps in (1, 2, 4) if unit_gain else ():
        voc = onnx.load(str(BUNDLE / f"vocoder_s{fps}.onnx"))
        (mul,) = [n for n in voc.graph.node if n.name == "apply_voicing"]
        mul.input[list(mul.input).index("voicing")] = "unit_gain"
        voc.graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray([1.0], np.float32), "unit_gain"))
        onnx.checker.check_model(voc, full_check=True)
        onnx.save(voc, str(out / f"vocoder_s{fps}.onnx"))
    return out


@pytest.fixture(scope="module")
def eos_units_bundle(tmp_path_factory):
    """Partial <EOS> chunks (s2/s4 hops need BUG-028 pads) rendered by a vocoder
    on which a pad frame (the last unit, voicing 0) is audible unless the
    pipeline zeroes it."""
    return patch_bundle(tmp_path_factory.mktemp("s2ut_eos_bundle") / "bundle",
                        eos_units=True, unit_gain=True)


@pytest.fixture(scope="module")
def early_eos_bundle(tmp_path_factory):
    return patch_bundle(tmp_path_factory.mktemp("s2ut_early_eos") / "bundle",
                        eos_at_lag_one=True)


@pytest.fixture(scope="module")
def no_eos_bundle(tmp_path_factory):
    return patch_bundle(tmp_path_factory.mktemp("s2ut_no_eos") / "bundle", no_eos=True)


@pytest.fixture(scope="module")
def ready_hi_bundle(tmp_path_factory):
    """readiness 1.0 per source chunk: the bank never empties."""
    return patch_bundle(tmp_path_factory.mktemp("s2ut_ready_hi") / "bundle",
                        readiness_logit=20.0)


@pytest.fixture(scope="module")
def ready_lo_bundle(tmp_path_factory):
    """readiness 2e-9 < 0.5: every chunk is "blank"."""
    return patch_bundle(tmp_path_factory.mktemp("s2ut_ready_lo") / "bundle",
                        readiness_logit=-20.0)


# The double-write scenario: blank readiness (adaptive holds to lag k + 2), and
# <EOS> on every write once 16 source chunks are in - the first pause chunk of
# utterance 1 (chunk 15) and the last pause chunk of utterance 2 (chunk 42).
DOUBLE_WRITE_K, DOUBLE_WRITE_CHANGE = 5, (1.2, "wait_k", 1.0)   # k 5 -> 1 at chunk 15


@pytest.fixture(scope="module")
def dbl_bundle(tmp_path_factory):
    return patch_bundle(tmp_path_factory.mktemp("s2ut_dbl") / "bundle", readiness_logit=-20.0,
                        eos_from_src=16)


_GOLDENS = {}


@pytest.fixture(scope="module")
def tail_wav(tmp_path_factory):
    """0.5 s of silence, then 40 ms of sine: the input ends two frames into a
    speech chunk while the policy is still idle."""
    path = tmp_path_factory.mktemp("s2ut_tail") / "tail.wav"
    t = np.arange(2 * FRAME) / SR
    G.write_wav_f32(path, SR, np.concatenate([np.zeros(25 * FRAME, np.float32),
                                              (0.25 * np.sin(2 * np.pi * 200 * t))
                                              .astype(np.float32)]))
    return path


def golden(tmp_path_factory, in_wav, fps, bundle=BUNDLE, wait_k=3, **options):
    """make() once per configuration: (dir, meta, steps, output float32, log)."""
    key = (str(in_wav), str(bundle), fps, wait_k, tuple(sorted(options.items())))
    if key not in _GOLDENS:
        out = tmp_path_factory.mktemp(f"golden_s{fps}")
        opts = G.GoldenOptions(src_lang="es", tgt_lang="de", wait_k=wait_k, frames_per_step=fps,
                               **options)
        log = []
        meta = G.make(bundle, in_wav, out, opts, log=log.append)
        _, y = G.read_wav(out / "output.wav")
        _GOLDENS[key] = (out, meta, G.read_trace(out / "trace.jsonl"),
                         np.frombuffer(y.tobytes(), np.float32).copy(), log)
    return _GOLDENS[key]


def unit_audio(u, voicing=VOICING):
    """The fixture vocoder in float32: Cast, * (1/16), * voicing."""
    return np.float32(u) * np.float32(1.0 / VOCAB) * np.float32(voicing)


def replay_output(steps, fps, n_samples, voicing=VOICING):
    """Re-derive the vocoder output from the trace alone: each write queues its
    units up to the first special unit plus BUG-028 pads to a whole hop (pads
    render as silence); every hop pops fps frames after its chunk step."""
    hops_per_chunk = CHUNK // fps
    by_hop = {(s["step"] + 1) * hops_per_chunk - 1: s for s in steps}
    queue, out = [], np.zeros(n_samples, np.float32)
    for h in range(n_samples // (fps * FRAME)):
        s = by_hop.get(h)
        for w in range(s["writes"] if s else 0):
            units = s["units"][CHUNK * w:CHUNK * (w + 1)]
            valid = next((i for i, u in enumerate(units) if not 0 <= u < VOCAB), CHUNK)
            queue += units[:valid]
            queue += [None] * (-len(queue) % fps)
        if len(queue) >= fps:
            frames, queue = queue[:fps], queue[fps:]
            for i, u in enumerate(frames):
                if u is not None:
                    start = (h * fps + i) * FRAME
                    out[start:start + FRAME] = unit_audio(u, voicing)
    assert not queue
    return out


# -- the golden itself --------------------------------------------------------------
@pytest.mark.parametrize("fps", [4, 1])
def test_make_two_utterances(tmp_path_factory, in_wav, fps):
    out_dir, meta, steps, y, log = golden(tmp_path_factory, in_wav, fps)
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(G.GOLDEN_FILES)
    assert [tuple(s) for s in steps] == [G.TRACE_KEYS] * len(steps)
    assert "".join(s["decision"][0] for s in steps) == DECISIONS + AUDIBLE_TAIL
    assert [s["decision"] for s in steps[:3]] == ["READ", "READ", "WRITE"]
    assert [s["step"] for s in steps] == list(range(len(steps)))
    assert [s["t_src_ms"] for s in steps] == [80.0 * (i + 1) for i in range(len(steps))]

    # Two utterances, each ending on <EOS> (third flush step at k = 3), not on a cap.
    ends = [s for s in steps if s["utterance_end"]]
    assert [s["step"] for s in ends] == UTTERANCE_ENDS
    assert all(s["eos"] and not s["cap_end"] and s["decision"] == "FLUSH" for s in ends)
    assert [s["eos"] for s in steps].count(True) == 2
    assert [s["segment"] for s in steps] == [0] * 23 + [1] * 23 + [2]
    assert all((s["t_src"], s["t_tgt"]) == (0, 0) for s in ends)

    # The encoder side: every speech or active chunk is encoded, never a flush
    # step or the idle pause after an utterance.
    assert [s["step"] for s in steps if not s["encoded"]] == \
        [20, 21, 22] + IDLE_STEPS + [43, 44, 45, 46]
    assert all(not s["speech"] for s in steps[15:27]) and steps[27]["speech"]
    assert all(s["t_src"] - s["t_tgt"] == 2 for s in steps if s["decision"] == "WRITE")

    # Units: an utterance's n-th write is 3n % 16 (from <BOS> = 16, +3 per step).
    for first, last in ((0, 22), (23, 45)):
        writes = [s for s in steps[first:last + 1] if s["writes"]]
        assert [s["units"] for s in writes] == \
            [[(BOS + TGT_ID * n) % VOCAB] * CHUNK for n in range(1, len(writes) + 1)]
    assert all(s["t_tgt_frames_emitted"] == CHUNK * s["writes"] and s["pad_frames"] == 0
               for s in steps)

    # Output: whole hops, the input plus the drained tail and one silent hop
    # after the audible last one, nothing capped.
    hop = fps * FRAME
    tail = {4: 5, 1: 23}[fps]
    assert y.size == meta["output"]["samples"] == (-(-INPUT_SAMPLES // hop) + tail) * hop
    assert (meta["output"]["tail_hops"], meta["output"]["tail_capped"]) == (tail, False)
    assert np.all(y[-hop:] == 0.0) and y[-hop - 1] == unit_audio(3)
    assert meta["stats"] == {"reads": 9, "writes": 32, "flushes": 6, "utterances": 2,
                             "eos_ends": 2, "cap_ends": 0, "dec_steps": 38, "voc_steps": 38 *
                             (CHUNK // fps), "fifo_stalls": 0, "pad_frames": 0,
                             "fifo_overflows": 0}
    assert meta["options"]["prosody_ctx"] is False and meta["options"]["output_chain"] is False
    assert meta["bundle"]["files"]["translator_dec.onnx"] == \
        G.sha256(BUNDLE / "translator_dec.onnx")
    assert "--raw" in meta["cpp_command"] and "--no-prosody-ctx" in meta["cpp_command"]
    assert G.read_wav(out_dir / "input.wav")[1] == G.read_wav(in_wav)[1]
    assert not log

    # The first translated hop is chunk 2's write: unit 3 -> 3/16 * 0.9.
    first = np.flatnonzero(y)[0]
    assert first == (2 * CHUNK + CHUNK - fps) * FRAME
    np.testing.assert_array_equal(y[first:first + hop], np.full(hop, unit_audio(3)))
    assert abs(float(unit_audio(3)) - 3 / 16 * 0.9) < 1e-7     # float32(3/16) * float32(0.9)
    # Audio only on hops a WRITE/FLUSH step queued; the rest is exact silence.
    np.testing.assert_array_equal(y, replay_output(steps, fps, y.size))
    produced = {s["step"] for s in steps if s["writes"]}
    for h in np.unique(np.flatnonzero(y) // hop):
        assert ((h * fps) // CHUNK) in produced or ((h * fps) // CHUNK - 1) in produced


def test_frames_per_step_variants_agree(tmp_path_factory, in_wav):
    """s1 vocodes a chunk's frames over four hops starting at the chunk's last
    hop, so its output is s4's delayed by three frames; the chunk steps match."""
    _, _, steps4, y4, _ = golden(tmp_path_factory, in_wav, 4)
    _, _, steps1, y1, _ = golden(tmp_path_factory, in_wav, 1)
    _, _, steps2, y2, _ = golden(tmp_path_factory, in_wav, 2)

    def strip(steps):
        return [{k: v for k, v in s.items() if k != "elapsed_ms"} for s in steps]

    assert strip(steps1) == strip(steps2) == strip(steps4)
    # All three stop one silent hop after their last audible sample, so the
    # lengths agree and the delay comes out of s4's trailing silence.
    assert y1.size == y2.size == y4.size and np.all(y4[-3 * FRAME:] == 0.0)
    np.testing.assert_array_equal(
        y1, np.concatenate([np.zeros(3 * FRAME, np.float32), y4])[:y1.size])
    np.testing.assert_array_equal(
        y2, np.concatenate([np.zeros(2 * FRAME, np.float32), y4])[:y2.size])


def test_trace_is_loadable_by_s2ut_eval(tmp_path_factory, in_wav):
    out_dir, _, steps, _, _ = golden(tmp_path_factory, in_wav, 4)
    traces = s2ut_eval.load_traces(str(out_dir / "trace.jsonl"), None, ["in"])
    assert list(traces) == ["in"] and traces["in"] == steps
    al, al_ca = s2ut_eval.average_lagging_ms(1000.0 * INPUT_SAMPLES / SR, traces["in"])
    assert np.isfinite(al) and np.isfinite(al_ca) and al > 0


@pytest.mark.parametrize("fps,pads", [(4, 3), (2, 1), (1, 0)])
def test_partial_eos_chunk_is_padded_to_silence(tmp_path_factory, in_wav, eos_units_bundle, fps,
                                                pads):
    """BUG-028: [5, EOS, PAD, PAD] queues one frame plus pads to a whole hop;
    the pads are silent and the next utterance starts on its own frames."""
    _, meta, steps, y, _ = golden(tmp_path_factory, in_wav, fps, bundle=eos_units_bundle)
    assert "".join(s["decision"][0] for s in steps) == DECISIONS
    ends = [s for s in steps if s["utterance_end"]]
    assert [s["step"] for s in ends] == UTTERANCE_ENDS
    for s in ends:
        assert s["units"] == EOS_CHUNK and s["eos"] and not s["cap_end"]
        assert (s["t_tgt_frames_emitted"], s["pad_frames"]) == (1, pads)
    assert meta["stats"]["pad_frames"] == 2 * pads
    np.testing.assert_array_equal(y, replay_output(steps, fps, y.size, voicing=1.0))
    hop = fps * FRAME

    def hop_start(chunk_step):                                 # the hop that runs the step
        return (chunk_step * CHUNK + CHUNK - fps) * FRAME

    for s in ends:
        start = hop_start(s["step"])
        np.testing.assert_array_equal(y[start:start + FRAME], np.full(FRAME, unit_audio(5, 1.0)))
        np.testing.assert_array_equal(y[start + FRAME:start + hop], 0.0)    # pads: zeroed
    # Nothing is left over: silence until utterance 2's first write (chunk 29),
    # which starts on its own unit 3.
    w2 = hop_start(29)
    np.testing.assert_array_equal(y[hop_start(UTTERANCE_ENDS[0]) + hop:w2], 0.0)
    np.testing.assert_array_equal(y[w2:w2 + FRAME], np.full(FRAME, unit_audio(3, 1.0)))


def test_eos_in_a_pause_completes_without_flush_steps(tmp_path_factory, in_wav,
                                                      early_eos_bundle):
    """<EOS> on the first pause write (k = 1) marks the utterance caught up: the
    rest of the pause READs (still encoded) and the chunk whose silence passes
    400 ms completes it inside on_chunk_energy(), on a READ step without any
    write. flushes_completed is captured before on_chunk_energy(), so that step
    is the utterance end and the translators are reset."""
    _, meta, steps, y, _ = golden(tmp_path_factory, in_wav, 4, bundle=early_eos_bundle,
                                  wait_k=1)
    assert "".join(s["decision"][0] for s in steps) == \
        "W" * 16 + "R" * 4 + "R" + "R" * 6 + "W" * 12 + "R" * 4 + "R"
    ends = [s for s in steps if s["utterance_end"]]
    assert [s["step"] for s in ends] == [20, 43]
    for s in ends:
        assert (s["decision"], s["writes"], s["encoded"], s["eos"], s["cap_end"]) == \
            ("READ", 0, False, False, False)
    assert all(s["eos"] for s in steps if s["writes"])          # eos_prob = 1 on every write
    assert [s["encoded"] for s in steps[16:20]] == [True] * 4    # caught up, still reading
    assert [s["segment"] for s in steps] == [0] * 21 + [1] * 23
    assert (meta["stats"]["utterances"], meta["stats"]["eos_ends"], meta["stats"]["cap_ends"],
            meta["stats"]["flushes"]) == (2, 2, 0, 0)
    # Utterance 2 starts from zeroed translators: its writes are 3n % 16 again.
    writes2 = [s for s in steps[21:] if s["writes"]]
    assert [s["units"][0] for s in writes2] == [(BOS + TGT_ID * n) % VOCAB for n in range(1, 13)]
    np.testing.assert_array_equal(y, replay_output(steps, 4, y.size))


def test_flush_bound_and_mid_flush_resumption(tmp_path_factory, in_wav, no_eos_bundle):
    """eos_prob = 0. The pause flushes toward the bound ceil(1.5 x 20) = 30, but
    speech returns on the 8th flush step (chunk 27): the same utterance resumes
    with its counters and translator state (lag -4: READ until t_src - t_tgt >=
    k). The final flush runs to its bound ceil(1.5 x 36) = 54 and ends the
    utterance without <EOS> (cap_end, BUG-030)."""
    _, meta, steps, y, _ = golden(tmp_path_factory, in_wav, 4, bundle=no_eos_bundle)
    assert "".join(s["decision"][0] for s in steps) == \
        "RR" + "W" * 18 + "F" * 7 + "R" * 7 + "W" * 9 + "F" * 20 + AUDIBLE_TAIL
    (end,) = [s for s in steps if s["utterance_end"]]
    assert end["step"] == 62 and end["cap_end"] and not end["eos"]
    assert not any(s["eos"] for s in steps) and all(s["segment"] == 0 for s in steps[:63])
    assert [(s["decision"], s["encoded"], s["t_src"], s["t_tgt"]) for s in steps[26:28]] == \
        [("FLUSH", False, 20, 25), ("READ", True, 21, 25)]
    # No reset at the resumption: the decoder's units continue the sequence.
    assert [s["units"][0] for s in steps if s["writes"]] == \
        [(BOS + TGT_ID * n) % VOCAB for n in range(1, 55)]
    assert {k: meta["stats"][k] for k in ("utterances", "eos_ends", "cap_ends", "flushes")} == \
        {"utterances": 1, "eos_ends": 0, "cap_ends": 1, "flushes": 27}
    np.testing.assert_array_equal(y, replay_output(steps, 4, y.size))


def test_speech_in_a_trailing_partial_chunk_is_translated(tmp_path_factory, tail_wav):
    """s1: after the input the policy is still idle and the FIFO empty, but the
    partial chunk saw speech (s2ut_idle() is false), so zero hops continue and
    the utterance is read, written and flushed instead of dropped."""
    _, meta, steps, y, _ = golden(tmp_path_factory, tail_wav, 1)
    assert "".join(s["decision"][0] for s in steps) == \
        "R" * 6 + "RR" + "WWWW" + "FFF" + AUDIBLE_TAIL
    assert steps[6]["speech"] and steps[6]["encoded"] and not steps[5]["encoded"]
    assert (meta["output"]["input_hops"], meta["output"]["tail_hops"]) == (27, 37)
    assert (meta["stats"]["utterances"], meta["stats"]["eos_ends"]) == (1, 1)
    np.testing.assert_array_equal(y, replay_output(steps, 1, y.size))


def test_fifo_backpressure_defers_the_write():
    """A full FIFO stalls the write (C++ fifo_stalls): no decoder step, the lag
    is kept and the next chunk step writes."""
    pipe = G.ModeBPipeline(BUNDLE, G.GoldenOptions("es", "de", wait_k=1, frames_per_step=4))
    room = pipe.fifo.capacity - CHUNK + 1
    assert pipe.fifo.push_chunk([1] * room, [0.0, 1.0, 0.0, 0.0] * room).pushed == room
    speech = (0.25 * np.sin(2 * np.pi * 200 * np.arange(CHUNK * FRAME) / SR)).astype(np.float32)
    for _ in range(2):
        pipe.process(speech)
    first, second = pipe.steps
    assert (first["decision"], first["writes"], first["units"]) == ("WRITE", 0, [])
    assert (first["t_src"], first["t_tgt"], pipe.stats["fifo_stalls"]) == (1, 0, 1)
    assert (second["decision"], second["writes"], second["t_tgt"]) == ("WRITE", 1, 1)
    assert second["units"] == [(BOS + TGT_ID) % VOCAB] * CHUNK and pipe.stats["dec_steps"] == 1


@pytest.mark.parametrize("readiness,decisions", [
    # Always ready: the bank keeps >= 1, so CTC-adaptive writes like strict wait-3.
    ("hi", DECISIONS + AUDIBLE_TAIL),
    # Always blank: hold back to lag k + adaptive_extra_wait = 5; each flush then
    # drains 4 chunks before <EOS> on its 5th step.
    ("lo", "RRRR" + "W" * 16 + "F" * 5 + "RR" + "RRRR" + "W" * 12 + "F" * 5 + AUDIBLE_TAIL),
])
def test_adaptive_readiness_gates_the_writes(request, tmp_path_factory, in_wav, readiness,
                                             decisions):
    bundle = request.getfixturevalue(f"ready_{readiness}_bundle")
    _, meta, steps, y, _ = golden(tmp_path_factory, in_wav, 4, bundle=bundle, adaptive=True)
    assert meta["options"]["adaptive"] and meta["options"]["policy"]["adaptive_mode"]
    assert "--adaptive" in meta["cpp_command"]
    assert "".join(s["decision"][0] for s in steps) == decisions
    assert all(s["writes"] == (s["decision"] != "READ") for s in steps)
    assert sum(s["utterance_end"] and s["eos"] for s in steps) == 2
    np.testing.assert_array_equal(y, replay_output(steps, 4, y.size))
    # The strict run ignores readiness altogether.
    _, _, strict, _, _ = golden(tmp_path_factory, in_wav, 4, bundle=bundle)
    assert "".join(s["decision"][0] for s in strict) == DECISIONS + AUDIBLE_TAIL


def test_adaptive_double_write_stops_at_eos_in_a_pause(tmp_path_factory, in_wav, dbl_bundle):
    """Adaptive with blank readiness at k = 5 writes at lag 7; `--change
    1.2:wait_k=1` lowers k at the first pause chunk (15), where lag 7 > k +
    adaptive_extra_wait asks for two writes. The first reports <EOS> in the
    pause: caught up (t_tgt = t_src), decide() is READ, so the second write is
    never decoded - one decoder step. The rest of the pause READs (still
    encoded) and its 6th chunk ends the utterance on a READ step."""
    _, meta, steps, y, _ = golden(tmp_path_factory, in_wav, 4, bundle=dbl_bundle,
                                  wait_k=DOUBLE_WRITE_K, adaptive=True,
                                  changes=(DOUBLE_WRITE_CHANGE,))
    assert "".join(s["decision"][0] for s in steps) == \
        "R" * 6 + "W" * 10 + "R" * 5 + "R" * 6 + "RR" + "W" * 14 + "R"
    before, step = steps[14], steps[15]
    assert step["t_src"] - before["t_tgt"] == 7 > 1 + 2    # max_writes_this_step() = 2 at k = 1
    assert (step["decision"], step["writes"], step["eos"]) == ("WRITE", 1, True)
    assert len(step["units"]) == CHUNK and (step["t_src"], step["t_tgt"]) == (16, 16)
    assert [(s["decision"], s["encoded"], s["writes"]) for s in steps[16:20]] == \
        [("READ", True, 0)] * 4
    ends = [s for s in steps if s["utterance_end"]]
    assert [s["step"] for s in ends] == [20, 43]
    assert all((s["decision"], s["writes"], s["encoded"]) == ("READ", 0, False) for s in ends)
    assert meta["stats"]["dec_steps"] == sum(s["writes"] for s in steps) == 10 + 14
    assert (meta["stats"]["eos_ends"], meta["stats"]["cap_ends"]) == (2, 0)
    assert meta["options"]["changes"] == [{"seconds": 1.2, "name": "wait_k", "value": 1.0}]
    cmd = meta["cpp_command"]
    assert cmd[cmd.index("--change") + 1] == "1.2:wait_k=1"
    np.testing.assert_array_equal(y, replay_output(steps, 4, y.size))
    # Only one chunk was queued at chunk 15: the following pause hops are silent.
    hop = CHUNK * FRAME
    assert np.any(y[15 * hop:16 * hop]) and np.all(y[16 * hop:21 * hop] == 0.0)


STREAM_CFG = types.SimpleNamespace(lang_ids={"en": 0, "es": 1, "fr": 2, "de": 3},
                                   units_per_step=CHUNK, bos=BOS, eos=EOS, vocab=VOCAB,
                                   unit_mode="stacked_4", chunk_ms=80.0, chunk_frames=CHUNK)


@pytest.mark.parametrize("kind,wait_k,adaptive,changes", [
    ("fixture", 3, False, ()),
    ("early_eos", 1, False, ()),
    ("no_eos", 3, False, ()),
    ("eos_units", 3, False, ()),
    ("ready_lo", 3, True, ()),
    ("dbl", DOUBLE_WRITE_K, True, (DOUBLE_WRITE_CHANGE,)),
])
def test_stream_translator_matches_the_golden(request, tmp_path_factory, in_wav, kind, wait_k,
                                              adaptive, changes):
    """fvml/s2ut_stream.py StreamingTranslator (the Stage 05 driver) on
    OrtRuntime takes the golden's chunk steps: decisions, encodes, writes (one
    decoder step each), counters, <EOS>, utterance ends and valid frames.
    Adaptive runs use blank readiness only: the driver now maps 2 x sigmoid
    (BUG-029) while the C++ worker and the golden map a plain sigmoid."""
    ss = pytest.importorskip("fvml.s2ut_stream")
    bundle = BUNDLE if kind == "fixture" else request.getfixturevalue(f"{kind}_bundle")
    _, _, steps, _, _ = golden(tmp_path_factory, in_wav, 4, bundle=bundle, wait_k=wait_k,
                               adaptive=adaptive, changes=changes)
    translator = ss.StreamingTranslator(
        ss.OrtRuntime(bundle / "translator_enc.onnx", bundle / "translator_dec.onnx", STREAM_CFG,
                      "es", "de"),
        W.WaitKConfig(k=wait_k, adaptive_mode=adaptive))
    encoder = G.OrtStep(bundle / "encoder_s4.onnx")
    x = np.frombuffer(G.read_wav(in_wav)[1].tobytes(), np.float32)
    hop, pending = CHUNK * FRAME, list(changes)
    for s in steps:
        c = s["step"]
        while pending and c * hop >= pending[0][0] * SR:        # fv_s2ut_file --change timing
            _, name, value = pending.pop(0)
            assert name == "wait_k"
            translator.policy.set_k(int(value))
        chunk = np.zeros(hop, np.float32)
        chunk[:max(0, min(hop, x.size - c * hop))] = x[c * hop:(c + 1) * hop]
        speech = any(G.frame_is_speech(chunk[i * FRAME:(i + 1) * FRAME].tolist(), -55.0)
                     for i in range(CHUNK))
        res = translator.push_chunk(encoder.step({"audio": chunk[None]})["latents"][0], speech)
        valid = []
        for w in range(s["writes"]):
            units = s["units"][CHUNK * w:CHUNK * (w + 1)]
            valid += units[:next((i for i, u in enumerate(units) if not 0 <= u < VOCAB), CHUNK)]
        assert (res.decision.name, res.encoded, res.writes, res.decoder_steps, res.t_src,
                res.t_tgt, res.eos, res.flushed, res.frames) == \
            (s["decision"], s["encoded"], s["writes"], s["writes"], s["t_src"], s["t_tgt"],
             s["eos"], s["utterance_end"], valid), f"chunk {c}"


# -- check ----------------------------------------------------------------------------
def test_check_against_itself_and_perturbed_copies(tmp_path_factory, in_wav, tmp_path):
    out_dir, _, steps, y, _ = golden(tmp_path_factory, in_wav, 4)
    lines = []
    passed, summary = G.check(out_dir, out_dir / "trace.jsonl", out_dir / "output.wav",
                              out=lines.append)
    assert passed and lines[-1] == "RESULT: PASS" and summary["audio"]["max_abs"] == 0.0
    assert summary["fields"]["units"] == [47, 47]
    assert cli.main(["check", "--golden", str(out_dir), "--trace", str(out_dir / "trace.jsonl"),
                     "--wav", str(out_dir / "output.wav")]) == 0

    def run(trace_steps=steps, samples=y):
        G.write_trace(tmp_path / "cpp.jsonl", trace_steps)
        G.write_wav_f32(tmp_path / "cpp.wav", SR, samples)
        lines.clear()
        return G.check(out_dir, tmp_path / "cpp.jsonl", tmp_path / "cpp.wav", out=lines.append)

    bad_units = [dict(s) for s in steps]
    bad_units[30]["units"] = [7, 7, 7, 7]
    passed, summary = run(bad_units)
    assert not passed and summary["mismatches"] == 1
    assert "units: 1 step(s) differ" in lines[-1] and any("step    30 units" in x for x in lines)
    bad_decision = [dict(s) for s in steps]
    bad_decision[2]["decision"] = "READ"
    assert not run(bad_decision)[0] and "decision" in lines[-1]
    assert not run(steps[:-1])[0] and "chunk steps 46 != 47" in lines[-1]

    shifted = y.copy()
    shifted[2560 + 7] += 1e-3
    assert not run(samples=shifted)[0] and "from sample 2567 (hop 2)" in lines[-1]
    shifted[2560 + 7] = y[2560 + 7] + 1e-6                       # below atol 1e-5
    assert run(samples=shifted)[0]
    assert not run(samples=y[:-FRAME])[0] and "output length" in lines[-1]

    G.write_trace(tmp_path / "cpp.jsonl", bad_units)
    G.write_wav_f32(tmp_path / "cpp.wav", SR, y)
    assert cli.main(["check", "--golden", str(out_dir), "--trace", str(tmp_path / "cpp.jsonl"),
                     "--wav", str(tmp_path / "cpp.wav")]) == 1


# -- CLI --------------------------------------------------------------------------------
def make_args(in_wav, out_dir, *extra, bundle=BUNDLE):
    return ["make", "--model-dir", str(bundle), "--src-lang", "es", "--tgt-lang", "de",
            *extra, str(in_wav), str(out_dir)]


def test_make_cli(in_wav, tmp_path, capsys):
    out = tmp_path / "g"
    assert cli.main(make_args(in_wav, out, "--wait-k", "3", "--frames-per-step", "4",
                              "--no-prosody-ctx")) == 0
    text = capsys.readouterr().out
    assert "47 chunk steps: 9 READ, 32 WRITE, 6 FLUSH; 2 utterance(s), 2 on <EOS>" in text
    assert "C++: fv_s2ut_file --model-dir" in text
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == G.KIND and meta["chunk_steps"] == 47

    # fv_s2ut_file's defaults: the first voice, langs.json's es->de k (absent: 3), s4.
    assert cli.main(make_args(in_wav, tmp_path / "d", "--no-prosody-ctx")) == 0
    meta = json.loads((tmp_path / "d" / "meta.json").read_text(encoding="utf-8"))
    assert (meta["options"]["voice"], meta["options"]["wait_k"],
            meta["options"]["frames_per_step"]) == ("identity", 3, 4)
    assert G.locate_bundle(BUNDLE, 0)[1] == 1          # the bundle loader's own pick: s1
    assert cli.main(make_args(in_wav, tmp_path / "e", "--no-prosody-ctx", "--voice",
                              "identity_b")) == 0
    meta = json.loads((tmp_path / "e" / "meta.json").read_text(encoding="utf-8"))
    assert meta["options"]["voice_index"] == 1
    # --src-lang by id, and the langs.json default k for en->de (4).
    args = make_args(in_wav, tmp_path / "f", "--no-prosody-ctx")
    args[args.index("es")], args[args.index("de")] = "0", "de"
    assert cli.main(args) == 0
    meta = json.loads((tmp_path / "f" / "meta.json").read_text(encoding="utf-8"))
    assert (meta["options"]["src_lang"], meta["options"]["wait_k"]) == ("en", 4)


def test_make_cli_rejects(in_wav, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(make_args(in_wav, tmp_path / "x"))             # no --no-prosody-ctx
    assert e.value.code == 2 and "--no-prosody-ctx is required" in capsys.readouterr().err
    stereo = tmp_path / "stereo.wav"
    rate8k = tmp_path / "8k.wav"
    G.write_wav_f32(rate8k, 8000, np.zeros(800, np.float32))
    blob = bytearray((rate8k).read_bytes())
    blob[22:24] = (2).to_bytes(2, "little")
    stereo.write_bytes(bytes(blob))
    cases = [
        (make_args(in_wav, tmp_path / "a", "--no-prosody-ctx", "--wait-k", "11"),
         "wait-k must be in 1..10"),
        (make_args(in_wav, tmp_path / "b", "--no-prosody-ctx", "--voice", "nobody"),
         "unknown voice 'nobody'"),
        (make_args(rate8k, tmp_path / "c", "--no-prosody-ctx"), "Sample-rate mismatch"),
        (make_args(stereo, tmp_path / "d", "--no-prosody-ctx"), "WAV must be mono"),
        (make_args(in_wav, tmp_path / "e", "--no-prosody-ctx", bundle=REPO / "tests"),
         "no complete encoder/vocoder bundle variant"),
    ]
    same = make_args(in_wav, tmp_path / "f", "--no-prosody-ctx")
    same[same.index("es")] = "de"
    cases.append((same, "src_lang and tgt_lang must differ"))
    unknown = make_args(in_wav, tmp_path / "g", "--no-prosody-ctx")
    unknown[unknown.index("es")] = "xx"
    cases.append((unknown, "unknown --src-lang 'xx'"))
    for args, text in cases:
        assert cli.main(args) == 1, text
        assert text in capsys.readouterr().err
        assert not pathlib.Path(args[-1]).exists()


def test_parse_change_mirrors_fv_s2ut_file():
    """--change <seconds>:<name>=<value> with VoiceParams clamping: wait_k is an
    integer in 1..10 (half away from zero), adaptive_policy a toggle."""
    assert G.parse_change("1.2:wait_k=1") == (1.2, "wait_k", 1.0)
    assert G.parse_change("0:wait_k=11") == (0.0, "wait_k", 10.0)
    assert G.parse_change("2:wait_k=2.5") == (2.0, "wait_k", 3.0)
    assert G.parse_change("3:wait_k=0") == (3.0, "wait_k", 1.0)
    assert G.parse_change("1:adaptive_policy=on") == (1.0, "adaptive_policy", 1.0)
    assert G.parse_change("1:adaptive_policy=0.4") == (1.0, "adaptive_policy", 0.0)
    for bad, text in (("wait_k=1", "expected seconds:name=value"), ("-1:wait_k=1", "bad seconds"),
                      ("x:wait_k=1", "bad seconds"), ("1:wait_k", "expected name=value"),
                      ("1:gain=-3", "wait_k and adaptive_policy changes only"),
                      ("1:wait_k=abc", "is not a finite number"),
                      ("1:wait_k=inf", "is not a finite number")):
        with pytest.raises(G.GoldenError, match=text):
            G.parse_change(bad)


def test_make_cli_change(in_wav, tmp_path, capsys):
    """adaptive_policy=0 at 0 s makes an --adaptive run strict (bank to 0, no
    reset): the same chunk steps as a strict golden."""
    assert cli.main(make_args(in_wav, tmp_path / "a", "--no-prosody-ctx", "--wait-k", "3",
                              "--adaptive", "--change", "0:adaptive_policy=off")) == 0
    assert cli.main(make_args(in_wav, tmp_path / "s", "--no-prosody-ctx", "--wait-k", "3")) == 0
    capsys.readouterr()
    meta = json.loads((tmp_path / "a" / "meta.json").read_text(encoding="utf-8"))
    assert meta["options"]["changes"] == [{"seconds": 0.0, "name": "adaptive_policy",
                                           "value": 0.0}]
    assert meta["cpp_command"][meta["cpp_command"].index("--change") + 1] == \
        "0.0:adaptive_policy=0"

    def strip(path):
        return [{k: v for k, v in s.items() if k != "elapsed_ms"}
                for s in G.read_trace(path / "trace.jsonl")]

    assert strip(tmp_path / "a") == strip(tmp_path / "s")
    with pytest.raises(SystemExit) as e:
        cli.main(make_args(in_wav, tmp_path / "x", "--no-prosody-ctx", "--change", "1:gain=-3"))
    assert e.value.code == 2 and "changes only" in capsys.readouterr().err


def test_tail_cap(in_wav, tmp_path):
    log = []
    opts = G.GoldenOptions(src_lang="es", tgt_lang="de", wait_k=3, frames_per_step=4,
                           max_tail_hops=1)
    meta = G.make(BUNDLE, in_wav, tmp_path, opts, log=log.append)
    assert meta["output"]["tail_capped"] and meta["output"]["hops"] == 43
    assert meta["stats"]["utterances"] == 1
    cmd = meta["cpp_command"]
    assert cmd[cmd.index("--max-tail-ms") + 1] == "80"  # fv_s2ut_file: ceil(80 / 80) = 1 hop
    assert log and "still busy (or audible) after 1 tail hops" in log[0]


def test_self_test_runs_on_the_standard_library():
    probe = ("import sys; sys.path.insert(0, '.'); from fvml import s2ut_golden as G; "
             "G.self_test(out=lambda s: None); "
             "print(sorted(m for m in ('numpy', 'onnxruntime', 'onnx', 'torch') "
             "if m in sys.modules))")
    proc = subprocess.run([sys.executable, "-I", "-c", probe], cwd=TRAINING,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"
    proc = subprocess.run([sys.executable, "-I", str(TRAINING / "s2ut_golden.py"), "--self-test"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "checks passed" in proc.stdout, proc.stderr


# -- C++ parity --------------------------------------------------------------------------
def _s2ut_file_bin():
    env = os.environ.get("FLUXVOICE_S2UT_FILE")
    if not env:
        pytest.skip("FLUXVOICE_S2UT_FILE (an ORT-enabled fv_s2ut_file) is not set")
    path = pathlib.Path(env)
    if not path.is_file():
        pytest.skip(f"fv_s2ut_file not found at {path}")
    return path


@pytest.mark.parametrize("bundle_kind,input_kind,fps,wait_k,adaptive", [
    ("fixture", "two", 4, 3, False),     # the Stage 07 reference command
    ("fixture", "two", 1, 3, False),
    ("eos_units", "two", 4, 3, False),   # BUG-028 pads (3 per <EOS>), zeroed audibly
    ("eos_units", "two", 2, 3, False),
    ("early_eos", "two", 4, 1, False),   # <EOS> in a pause: the utterance ends on a READ step
    ("no_eos", "two", 4, 3, False),      # mid-flush resumption, then a flush-bound end
    ("fixture", "tail", 1, 3, False),    # speech in the input's last, partial chunk
    ("ready_hi", "two", 4, 3, True),     # CTC-adaptive: readiness banked ...
    ("ready_lo", "two", 4, 3, True),     # ... or blank (hold back to k + 2)
])
def test_fv_s2ut_file_matches_golden(request, tmp_path_factory, tmp_path, bundle_kind,
                                     input_kind, fps, wait_k, adaptive):
    exe = _s2ut_file_bin()
    bundle = BUNDLE if bundle_kind == "fixture" else \
        request.getfixturevalue(f"{bundle_kind}_bundle")
    in_wav = request.getfixturevalue("in_wav" if input_kind == "two" else "tail_wav")
    out_dir, _, _, _, _ = golden(tmp_path_factory, in_wav, fps, bundle=bundle, wait_k=wait_k,
                                 adaptive=adaptive)
    cmd = [str(exe), "--model-dir", str(bundle), "--src-lang", "es", "--tgt-lang", "de",
           "--wait-k", str(wait_k), "--frames-per-step", str(fps)] + \
        (["--adaptive"] if adaptive else []) + \
        ["--raw", "--no-prosody-ctx", "--dump-trace", str(tmp_path / "cpp.jsonl"), str(in_wav),
         str(tmp_path / "cpp.wav")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
    lines = []
    passed, _ = G.check(out_dir, tmp_path / "cpp.jsonl", tmp_path / "cpp.wav", out=lines.append)
    assert passed, "\n".join(lines)
