#!/usr/bin/env python3
"""Generate the tiny ONNX fixture models that honour model contract v1.

Milestone 2 / Stage 02. The fixtures let the C++ runtime (Stages 07-08) be built
and tested without trained weights:

  identity/encoder_s{1,2,4}.onnx   audio [1,hop] -> content [1,N,320] (a reshape)
  identity/vocoder_s{1,2,4}.onnx   content -> audio [1,hop]; takes every
                                   conditioning input (ignored), so a full
                                   encoder -> vocoder bundle is a bit-exact
                                   passthrough of the hop
  identity/voices.json             two voices for --voice / blending tests
  delay_s1.onnx                    vocoder: audio(t) = content(t-1) through a
                                   state pair (state_in.buf -> state_out.buf)
  accumulator_s1.onnx              encoder: state_out.acc = state_in.acc +
                                   sum(audio); content = state_out.acc
  broken/*.onnx                    valid ONNX whose metadata or I/O breaks the
                                   contract; manifest.json records the error
                                   each must be rejected with
  manifest.json                    every file's SHA256, contract string, graph
                                   I/O and expected error, plus the pairs and
                                   their algorithmic latency (read by
                                   tests/test_model_contract.cpp without ORT)

Built with onnx.helper only (no torch), opset 17 / IR 8 so any ONNX Runtime
>= 1.13 loads them, with a fixed producer and no timestamps: the bytes
regenerate identically *for the onnx/protobuf versions pinned in
python_training/uv.lock*. An upgrade may change the serialisation, so
regenerate (and commit the new manifest hashes) in the same change as any
onnx bump.

Usage:
    make_fixture_models.py [--out tests/data/fixtures]   # write (needs onnx)
    make_fixture_models.py --check      # rebuild in memory, compare with disk (needs onnx)
    make_fixture_models.py --self-test  # committed files vs. manifest: SHA256, embedded
                                        # contracts, expected errors, pairs (stdlib only)

From the repo root: uv run --project python_training python tools/make_fixture_models.py

Exit status: 0 = written / up to date / self-test passed, 1 = a difference or failure.
Generating needs onnx (the spec requires onnx.helper); --self-test and --help
are standard library only, so CTest runs the self-test with any Python 3.
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


# -- graph building blocks ---------------------------------------------------
def value(name, elem, shape):
    """shape None = no shape information at all (unknown rank)."""
    return helper.make_tensor_value_info(name, elem, None if shape is None else list(shape))


def const_i64(name, values):
    return helper.make_tensor(name, I64, [len(values)], list(values))


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


def vocoder_contract(fps, cond=COND, lookahead=0, states=()):
    return fc.ModelContract(
        role="vocoder", sample_rate=RATE, frame_samples=FRAME, frames_per_step=fps,
        lookahead_frames=lookahead, content_kind="soft", content_dim=FRAME,
        cond_frame_samples=FRAME, cond_inputs=tuple(cond),
        spk_emb_dim=SPK_DIM if "spk_emb" in cond else 0, overlap_samples=0,
        states=tuple(states), license=LICENSE)


# -- fixtures ----------------------------------------------------------------
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


def voices_json():
    def voice(name, display, emb):
        return {"name": name, "display": display, "embedding": emb,
                "logf0_mean": 4.8, "logf0_std": 0.25, "energy_mean": -4.0, "energy_std": 1.5,
                "license": LICENSE, "source": "tools/make_fixture_models.py"}
    return {"schema": 1, "voices": [voice("identity", "Identity A (passthrough)", [0.0] * SPK_DIM),
                                    voice("identity_b", "Identity B (passthrough)", [1.0] * SPK_DIM)]}


def broken_models():
    """(file, model, expected error) — each is valid ONNX but breaks the contract."""
    s1 = encoder_contract(1)
    no_rate = {k: v for k, v in s1.to_dict().items() if k != "sample_rate"}
    v2 = {**s1.to_dict(), "contract_version": 2}
    discrete = encoder_contract(1, content_kind="discrete", content_dim=0, vocab=FRAME)
    dump = lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))  # noqa: E731
    return [
        ("broken/no_contract.onnx", _without_contract(identity_encoder(1)),
         fc.MISSING_METADATA),
        ("broken/not_json.onnx", identity_encoder(1, text='{"contract_version": 1, "role": '),
         "contract is not valid JSON"),
        ("broken/unknown_version.onnx", identity_encoder(1, text=dump(v2)),
         "unsupported contract_version 2"),
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
        # No "state without a shape" fixture: onnx.checker rejects graph I/O
        # with no shape field, so only unchecked files can carry one. The
        # validators still reject it (shared cases io_*_no_shape).
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
    models += broken_models()

    files, entries, contracts = {}, [], {}
    for rel, model, want_error in models:
        data = model.SerializeToString(deterministic=True)
        files[rel] = data
        inputs, outputs = fc.graph_io(model)
        text = fc.read_contract_text(model)
        try:
            contracts[rel] = fc.check_model(text, inputs, outputs)
            error = None
        except fc.ContractError as e:
            error = str(e)
        if (want_error is None) != (error is None) or (want_error and want_error not in error):
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

    voices = voices_json()
    fc.validate_voices(voices, contracts["identity/vocoder_s1.onnx"])
    files["identity/voices.json"] = _json_bytes(voices)
    manifest = {"generator": "tools/make_fixture_models.py", "opset": OPSET,
                "ir_version": IR_VERSION, "models": entries, "pairs": pairs,
                "bundles": [{"dir": "identity", "vocoder": "identity/vocoder_s1.onnx",
                             "voices": "identity/voices.json"}]}
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
    (hashes, embedded contract strings, expected errors, pairs, voices) and
    nothing stale lies around. Regenerating (--check) needs onnx; this does not."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    failures = []

    def check_that(ok, name, why):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f" -> {why}"))
        if not ok:
            failures.append(name)

    contracts = {}
    for m in manifest["models"]:
        rel = m["file"]
        data = (out / rel).read_bytes() if (out / rel).is_file() else b""
        check_that(hashlib.sha256(data).hexdigest() == m["sha256"], f"{rel} sha256", "file differs")
        # The contract is stored verbatim in the protobuf, so the bytes must contain it.
        check_that(m["contract"] is None or m["contract"].encode("utf-8") in data,
                   f"{rel} contract", "embedded string not found")
        try:
            contracts[rel] = fc.check_model(
                m["contract"], [fc.TensorSpec.from_dict(t) for t in m["graph"]["inputs"]],
                [fc.TensorSpec.from_dict(t) for t in m["graph"]["outputs"]])
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
    for b in manifest["bundles"]:
        try:
            fc.validate_voices(json.loads((out / b["voices"]).read_text(encoding="utf-8")),
                               contracts[b["vocoder"]])
            check_that(True, b["voices"], "")
        except (fc.ContractError, OSError, KeyError) as e:
            check_that(False, b["voices"], str(e))
    expected = {m["file"] for m in manifest["models"]} | {b["voices"] for b in manifest["bundles"]} | \
        {"manifest.json"}
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
