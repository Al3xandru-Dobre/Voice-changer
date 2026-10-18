#!/usr/bin/env python3
"""export_stateful_onnx.py — PyTorch → stateful ONNX model bundle (Milestone 2 /
Stage 06, Documents/Milestone 2/06_stateful_onnx_export.md).

Exports one role per run into a bundle directory, one fixed-shape graph per
frames-per-step variant (models/<bundle>/{encoder,vocoder}_s{1,2,4}.onnx). Each
graph is one hop at batch 1, and every causal-conv cache is a state_in.X →
state_out.X pair (fvml.export). Every file then gets these steps:
  1. torch.onnx.export with dynamo=True, falling back to the TorchScript
     exporter (the exporter used, and why, is recorded);
  2. IR capped at 13 and opset checked <= 26, for ONNX Runtime 1.31;
  3. contract embedded as metadata_props["fluxvoice.contract"];
  4. onnx strict shape inference + onnx.checker full check;
  5. reloaded through fvml.contract the way the C++ loader checks it.
Encoder/vocoder pairs present in the bundle must pass validate_pair.

Then, unless --skip-parity:
  - parity over >= 60 s of LibriSpeech dev-clean (one utterance per speaker
    in turn), ORT run chunk by chunk with its states fed back vs. PyTorch
    forward() offline. Encoder: unit agreement >= 99.9% and max |Δlogit|.
    Vocoder: SNR >= 50 dB on identical units and conditioning;
  - cost: ORT per-hop p50/p95/p99 on one thread (intra_op 1, sequential, no
    spinning);
  - once both roles are in the bundle: the end-to-end chain, ORT vs. PyTorch
    (reported, not gated), the chain cost against the 50%-of-hop budget, and
    golden/ (10 s input + the ORT chunked units, conditioning and audio), which
    Stage 07's fv_vc_file is checked against.
The run writes voices.json (vocoder), export_report.json and MODEL_CARD.md.

Usage:
  export_stateful_onnx.py --role encoder --checkpoint checkpoints/<run>/best.pt \\
      --head student-L1 --license "..." --bundle models/<name>/
  export_stateful_onnx.py --role vocoder --checkpoint checkpoints/<voc>/best.pt \\
      --bundle models/<name>/

Exit status: 0 = every file exported and every gate passed; 1 = a gate failed
(the files and the report are still written, so they can be inspected);
2 = bad arguments or an export/contract error.
"""
import argparse
import datetime
import json
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fvml import contract as fc  # noqa: E402
from fvml import export as fx  # noqa: E402

REPORT = "export_report.json"
CARD = "MODEL_CARD.md"
VOICES = "voices.json"
GOLDEN = "golden"
DEV_CLEAN = "manifests/librispeech-dev-clean.jsonl"
COST_WARMUP = 20


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--checkpoint", required=True, help="PyTorch checkpoint to export.")
    p.add_argument("--role", choices=fc.ROLES, required=True)
    p.add_argument("--frames-per-step", type=int, nargs="+", choices=fc.FRAMES_PER_STEP,
                   default=list(fc.FRAMES_PER_STEP), help="variants to export (default 1 2 4)")
    p.add_argument("--bundle", required=True, help="bundle directory, e.g. models/<name>/")
    p.add_argument("--head", help="encoder: delay head, index (0/1/2) or name (student-L1)")
    p.add_argument("--license", help="licence text for the contract (encoder checkpoints "
                   "do not store one; overrides the vocoder checkpoint's)")
    p.add_argument("--exporter", choices=fx.EXPORTERS, default="auto",
                   help="auto = dynamo, falling back to TorchScript (default)")
    p.add_argument("--parity-manifest", help=f"manifest (jsonl) of parity speech (default "
                   f"<data root>/{DEV_CLEAN})")
    p.add_argument("--parity-audio", nargs="+", help="audio files to use instead of a manifest")
    p.add_argument("--parity-seconds", type=float, default=fx.PARITY_SECONDS)
    p.add_argument("--golden-seconds", type=float, default=fx.GOLDEN_SECONDS)
    p.add_argument("--voice", help="voice for the chain and golden/ (default: the first)")
    p.add_argument("--cost-steps", type=int, default=1000,
                   help=f"hops timed per variant, after {COST_WARMUP} warm-up hops")
    p.add_argument("--gate-report", help="Stage 05 gate result (JSON) for the model card; "
                   "its frames_per_step, if any, is the default --gate-variant")
    p.add_argument("--gate-variant", type=int, choices=fc.FRAMES_PER_STEP,
                   help="frames-per-step the gate chose: its encoder + vocoder p95 must be "
                   "within 50%% of the hop (the other variants are reported)")
    p.add_argument("--skip-parity", action="store_true",
                   help="export and check only (no parity, cost or golden/)")
    args = p.parse_args(argv)
    args.frames_per_step = sorted(set(args.frames_per_step))
    if args.role == "encoder" and args.head is None:
        p.error("--role encoder needs --head (the gate's delay head)")
    if args.role == "vocoder" and args.head is not None:
        p.error("--head applies to --role encoder only")
    if args.parity_seconds < args.golden_seconds:
        p.error("--parity-seconds must be >= --golden-seconds")
    return args


T0 = time.perf_counter()


def log(msg):
    print(f"[export {time.perf_counter() - T0:6.1f}s] {msg}", flush=True)


# -- models ------------------------------------------------------------------------
def load_encoder(path, head, license):
    from fvml.content_encoder import ContentEncoder
    model, ckpt = ContentEncoder.from_checkpoint(path)
    model.eval()
    head = fx.head_index(model, head)
    license = license or ckpt.get("license")
    if not license:
        raise fx.ExportError("the encoder checkpoint stores no licence: pass --license")
    meta = ckpt.get("labels_meta")
    return model, {"head": head, "license": license,
                   "labels_meta": json.loads(meta) if isinstance(meta, str) else meta,
                   "step": ckpt.get("step"), "run_name": ckpt.get("run_name")}


def load_vocoder(path, license):
    from fvml.vocoder import UnitVocoder
    model, ckpt = UnitVocoder.from_checkpoint(path)
    model.eval()
    if license:
        ckpt = {**ckpt, "license": license}
    meta = ckpt.get("labels_meta")
    return model, ckpt, {"labels_meta": json.loads(meta) if isinstance(meta, str) else meta,
                         "step": ckpt.get("step"), "license": ckpt.get("license")}


def contract_for(role, model, info_or_ckpt, fps):
    if role == "encoder":
        return fx.encoder_contract(model, info_or_ckpt["head"], fps, info_or_ckpt["license"])
    return fx.vocoder_contract(model, info_or_ckpt, fps)


def graph_for(role, model, info, contract):
    if role == "encoder":
        return fx.encoder_graph(model, info["head"], contract.frames_per_step)
    return fx.vocoder_graph(model, contract)


# -- export ------------------------------------------------------------------------
def export_role(args, bundle, model, info, ckpt):
    files = {}
    for fps in args.frames_per_step:
        c = contract_for(args.role, model, info if args.role == "encoder" else ckpt, fps)
        graph = graph_for(args.role, model, info, c)
        if graph.states != list(c.states):
            raise fx.ExportError("graph states differ from the contract's")
        path = bundle / f"{args.role}_s{fps}.onnx"
        rec = fx.export_graph(graph, path, args.exporter)
        opset, ir = fx.finalize(path, c, drop_outputs=graph.debug_outputs,
                                metadata={"fluxvoice.exporter": rec["exporter"]})
        files[f"s{fps}"] = {"file": path.name, "sha256": fx.sha256(path),
                            "bytes": path.stat().st_size, "exporter": rec["exporter"],
                            "fallback_reason": rec["fallback_reason"],
                            "warnings": rec["warnings"], "export_seconds": rec["seconds"],
                            "opset": opset, "ir_version": ir, "states": len(c.states),
                            "state_floats": graph.state_floats(), "contract": c.to_dict()}
        how = rec["exporter"] + (f" (dynamo failed: {rec['fallback_reason']})"
                                 if rec["fallback_reason"] else "")
        log(f"{path.name}: {how}, opset {opset}, IR {ir}, {len(c.states)} states, "
            f"{graph.state_floats()} floats, checks passed")
    return files


def check_pairs(bundle, frames_per_step):
    pairs = {}
    for fps in frames_per_step:
        enc, voc = bundle / f"encoder_s{fps}.onnx", bundle / f"vocoder_s{fps}.onnx"
        if not (enc.exists() and voc.exists()):
            continue
        e, v = fx.read_contract(enc), fx.read_contract(voc)
        try:
            fc.validate_pair(e, v)
        except fc.ContractError as err:
            raise fx.ExportError(f"s{fps}: encoder/vocoder pair rejected: {err}") from None
        lat = fc.algorithmic_latency(e, v)
        pairs[f"s{fps}"] = {"algorithmic_latency_samples": lat,
                            "algorithmic_latency_ms": 1e3 * lat / e.sample_rate}
        log(f"s{fps}: pair OK, algorithmic latency {1e3 * lat / e.sample_rate:.0f} ms")
    return pairs


# -- parity audio ------------------------------------------------------------------
def parity_paths(args):
    from fvml import paths
    if args.parity_audio:
        return [pathlib.Path(p) for p in args.parity_audio], "files: " + ", ".join(
            pathlib.Path(p).name for p in args.parity_audio)
    manifest = pathlib.Path(args.parity_manifest) if args.parity_manifest \
        else paths.data_root() / DEV_CLEAN
    if not manifest.exists():
        raise fx.ExportError(f"no parity manifest at {manifest} (prepare dev-clean with "
                             "prepare_data.py, or pass --parity-audio)")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_speaker = {}
    for r in sorted(rows, key=lambda r: (str(r.get("speaker")), r["utt_id"])):
        by_speaker.setdefault(str(r.get("speaker")), []).append(r)
    order, depth = [], 0
    while len(order) < len(rows):            # one utterance per speaker, in turn
        for utts in by_speaker.values():
            if depth < len(utts):
                order.append(utts[depth])
        depth += 1
    return [paths.resolve(r["path"]) for r in order], f"manifest: {manifest.name}"


def parity_audio(args):
    from fvml import audio
    files, source = parity_paths(args)
    try:
        wav = audio.concat(files, args.parity_seconds)
    except ValueError as e:
        raise fx.ExportError(f"parity audio: {e}") from None
    return wav, {"source": source, "seconds": len(wav) / audio.SAMPLE_RATE}


# -- parity, cost, golden ----------------------------------------------------------------
def encoder_checks(args, bundle, model, info, wav):
    parity, cost = {}, {}
    for fps in args.frames_per_step:
        path = bundle / f"encoder_s{fps}.onnx"
        r = fx.encoder_parity(model, info["head"], path, wav)
        parity[f"s{fps}"] = r
        log(f"{path.name}: parity over {r['seconds']:.1f} s: unit agreement "
            f"{r['unit_agreement']:.5f}, max |dlogit| {r['max_abs_logit']:.2e} -> "
            f"{'PASS' if r['pass'] else 'FAIL'}")
        stream = fx.OrtStream(path)
        feeds = fx.encoder_feeds(stream.contract, wav)
        cost[f"s{fps}"] = timed(args, [stream], ([f] for f in feeds))
    return parity, cost


def vocoder_checks(args, bundle, model, voices, wav):
    """Vocoder parity on the bundle's encoder ids when the encoder is there,
    else on seeded random ids; conditioning from the parity speech."""
    parity, cost = {}, {}
    voice = pick_voice(voices, args.voice)
    for fps in args.frames_per_step:
        path = bundle / f"vocoder_s{fps}.onnx"
        stream = fx.OrtStream(path)
        c = stream.contract
        enc_path = bundle / f"encoder_s{fps}.onnx"
        if enc_path.exists():
            enc = fx.OrtStream(enc_path)
            units, _ = fx.encoder_chunked(enc, fx.whole_hops(wav, enc.contract.hop))
            enc_c, source = enc.contract, enc_path.name
        else:
            frames = len(wav) // c.frame_samples
            units = np.random.default_rng(0).integers(0, c.vocab, frames)
            enc_c, source = None, "random ids (seed 0)"
        cond = fx.conditioning(wav, voice, enc_c or _no_lookahead(c), c)
        full = fx.vocoder_inputs(c, units, cond, np.asarray(voice["embedding"], np.float32))
        r = fx.vocoder_parity(model, path, full)
        r.update(units=source, voice=voice["name"])
        parity[f"s{fps}"] = r
        log(f"{path.name}: parity over {r['seconds']:.1f} s: SNR {fmt_db(r['snr_db'])}, "
            f"max |d| {r['max_abs']:.2e} -> {'PASS' if r['pass'] else 'FAIL'}")
        steps = len(full[c.content_name]) // c.frames_per_step
        cost[f"s{fps}"] = timed(args, [stream], ([fx.vocoder_feeds(c, full, i)]
                                                 for i in range(steps)))
    return parity, cost


def _no_lookahead(c):
    return fc.ModelContract(role="encoder", sample_rate=c.sample_rate, frame_samples=c.frame_samples,
                            frames_per_step=c.frames_per_step, lookahead_frames=0,
                            content_kind=c.content_kind, vocab=c.vocab, content_dim=c.content_dim,
                            license="-")


def chain_checks(args, bundle, enc_model, head, voc_model, voices, wav):
    chain, cost = {}, {}
    voice = pick_voice(voices, args.voice)
    for fps in args.frames_per_step:
        enc_path, voc_path = bundle / f"encoder_s{fps}.onnx", bundle / f"vocoder_s{fps}.onnx"
        if not (enc_path.exists() and voc_path.exists()):
            continue
        r = fx.chain_parity(enc_model, head, voc_model, enc_path, voc_path, wav, voice)
        r["voice"] = voice["name"]
        chain[f"s{fps}"] = r
        log(f"s{fps} chain: unit agreement {r['unit_agreement']:.5f}, waveform SNR "
            f"{fmt_db(r['snr_db'])} (reported, not gated)")
        enc, voc = fx.OrtStream(enc_path), fx.OrtStream(voc_path)
        cut = fx.whole_hops(wav, enc.contract.hop)
        units, _ = fx.encoder_chunked(enc, cut)
        full = fx.vocoder_inputs(voc.contract, units,
                                 fx.conditioning(cut, voice, enc.contract, voc.contract),
                                 np.asarray(voice["embedding"], np.float32))
        feeds = ([e, fx.vocoder_feeds(voc.contract, full, i)]
                 for i, e in enumerate(fx.encoder_feeds(enc.contract, cut)))
        cost[f"s{fps}"] = timed(args, [enc, voc], feeds)
        k = cost[f"s{fps}"]
        log(f"s{fps} chain cost: p95 {k['p95_ms']:.2f} ms of a {k['hop_ms']:.0f} ms hop "
            f"({100 * k['p95_fraction_of_hop']:.0f}%) -> "
            f"{'within' if k['within_budget'] else 'OVER'} the 50% budget")
    return chain, cost


def timed(args, streams, feeds):
    def limited():
        for i, f in enumerate(feeds):
            if i >= COST_WARMUP + args.cost_steps:
                return
            yield f
    return fx.step_cost(streams, limited(), warmup=COST_WARMUP)


def write_golden(args, bundle, voices, wav):
    """golden/: input.wav + per variant the ORT chunked units (and, with a
    vocoder, the conditioning and audio) for Stage 07's fv_vc_file."""
    from fvml import audio
    import onnxruntime as ort
    out = bundle / GOLDEN
    out.mkdir(exist_ok=True)
    clip = wav[: int(round(args.golden_seconds * audio.SAMPLE_RATE))]
    audio.save(out / "input.wav", clip)
    voice = pick_voice(voices, args.voice) if voices else None
    manifest = {"schema": 1, "generator": "python_training/export_stateful_onnx.py",
                "input": "input.wav", "sample_rate": audio.SAMPLE_RATE,
                "seconds": len(clip) / audio.SAMPLE_RATE,
                "onnxruntime": ort.__version__, "voice": voice["name"] if voice else None,
                "pitch_scale": 1.0, "pitch_shift_st": 0.0,
                "alignment": "hop t of the stream: encoder output t (frame t - L) and tracker "
                             "push t (frame t - L, lookahead L) feed vocoder step t; the "
                             "first L outputs describe the zero history",
                "tolerance": {"units": "exact", "audio_max_abs": 1e-5}, "variants": {}}
    cond_written = False
    for fps in args.frames_per_step:
        enc_path, voc_path = bundle / f"encoder_s{fps}.onnx", bundle / f"vocoder_s{fps}.onnx"
        if not enc_path.exists():
            continue
        has_voc = voc_path.exists() and voice is not None
        units, cond, out_audio = fx.chain_outputs(enc_path, voc_path if has_voc else None,
                                                  clip, voice)
        v = {"encoder": enc_path.name, "encoder_sha256": fx.sha256(enc_path),
             "units": f"units_s{fps}.npy", "frames": int(len(units))}
        np.save(out / v["units"], units.astype(np.int64))
        if has_voc:
            voc_c = fx.read_contract(voc_path)
            names = [n for n in voc_c.cond_inputs if n != "spk_emb"]
            if not cond_written:
                np.save(out / "cond.npy", np.stack([cond[n] for n in names], axis=1))
                manifest.update(cond="cond.npy", cond_columns=names,
                                cond_frame_samples=voc_c.cond_frame_samples)
                cond_written = True
            v.update(vocoder=voc_path.name, vocoder_sha256=fx.sha256(voc_path),
                     audio=f"audio_s{fps}.wav", samples=int(len(out_audio)))
            audio.save(out / v["audio"], out_audio)
        manifest["variants"][f"s{fps}"] = v
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"{GOLDEN}/: {len(manifest['variants'])} variant(s), "
        f"{'units + audio' if cond_written else 'units only (no vocoder yet)'}")
    return manifest


def pick_voice(voices, name):
    vs = voices["voices"]
    if name is None:
        return vs[0]
    for v in vs:
        if v["name"] == name:
            return v
    raise fx.ExportError(f"no voice '{name}' in voices.json ({', '.join(v['name'] for v in vs)})")


def _cell(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return "—" if v is None else str(v)


def fmt_db(x):
    return "identical" if x is None else f"{x:.1f} dB"


# -- report & model card -----------------------------------------------------------
def versions():
    import onnx
    import onnxruntime
    import torch
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "torch": torch.__version__, "onnx": onnx.__version__,
         "onnxruntime": onnxruntime.__version__}
    try:
        import onnxscript
        v["onnxscript"] = onnxscript.__version__
    except ImportError:
        v["onnxscript"] = None
    return v


def load_report(bundle):
    path = bundle / REPORT
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": 1, "roles": {}}


def verdicts(report):
    """[(what, passed)] for every gate measured in the report. The compute
    budget is a gate for the gate-chosen variant only."""
    out = []
    for role, r in sorted(report["roles"].items()):
        for v, p in sorted((r.get("parity") or {}).items()):
            out.append((f"{role} {v} parity", bool(p["pass"])))
    chosen = report.get("gate_variant")
    k = (report.get("chain_cost") or {}).get(f"s{chosen}")
    if k is not None:
        out.append((f"chain s{chosen} compute budget", bool(k["within_budget"])))
    gate_L = gate_decision(report.get("gate")).get("lookahead_frames")
    enc = report["roles"].get("encoder")
    if gate_L is not None and enc is not None:
        out.append((f"encoder head (lookahead {enc['lookahead_frames']}) matches the gate "
                    f"winner (lookahead {gate_L})", enc["lookahead_frames"] == gate_L))
    return out


def gate_decision(gate):
    return (gate.get("decision") or {}) if isinstance(gate, dict) else {}


def gate_variant(gate):
    """frames_per_step of the gate's winner (gate_report.py JSON), or None."""
    if not isinstance(gate, dict):
        return None
    if gate.get("frames_per_step") in fc.FRAMES_PER_STEP:
        return gate["frames_per_step"]
    winner = gate_decision(gate).get("winner")
    for row in gate.get("candidates") or []:
        if row.get("candidate") == winner and row.get("frames_per_step") in fc.FRAMES_PER_STEP:
            return row["frames_per_step"]
    return None


def model_card(bundle, report, gate):
    roles, lines = report["roles"], []
    add = lines.append
    add(f"# Model card — `{bundle.name}`\n")
    add(f"Generated {report['generated']} by `python_training/export_stateful_onnx.py` "
        "(Milestone 2 / Stage 06). Regenerated on every export run; edit the "
        "checkpoints or the exporter, not this file.\n")
    add("## Files\n")
    add("| File | Role | Hop | Lookahead | Exporter | Opset / IR | States (floats) | SHA-256 |")
    add("|---|---|---|---|---|---|---|---|")
    for role in ("encoder", "vocoder"):
        for v, f in sorted((roles.get(role, {}).get("files") or {}).items()):
            c = f["contract"]
            hop_ms = 1e3 * c["frames_per_step"] * c["frame_samples"] / c["sample_rate"]
            add(f"| `{f['file']}` | {role} | {hop_ms:.0f} ms | {c['lookahead_frames']} | "
                f"{f['exporter']} | {f['opset']} / {f['ir_version']} | "
                f"{f['states']} ({f['state_floats']}) | `{f['sha256'][:16]}` |")
    for v, p in sorted((report.get("pairs") or {}).items()):
        add(f"\n{v}: encoder/vocoder pair valid, algorithmic latency "
            f"{p['algorithmic_latency_ms']:.0f} ms ({p['algorithmic_latency_samples']} samples).")
    add("\n## Training data and licences\n")
    for role in ("encoder", "vocoder"):
        r = roles.get(role)
        if not r:
            add(f"- **{role}**: not exported yet.")
            continue
        add(f"- **{role}**: checkpoint `{r['checkpoint_name']}` (sha256 "
            f"`{r['checkpoint_sha256'][:16]}`, step {r.get('step')}); contract licence: "
            f"{r['license']}")
        if r.get("head") is not None:
            add(f"  - delay head {r['head']} (lookahead {r['lookahead_frames']} frames)")
        if r.get("labels_meta"):
            add(f"  - labels: `{json.dumps(r['labels_meta'], sort_keys=True)}`")
    voices = report.get("voices")
    if voices:
        add("\n| Voice | Display | Source | Licence | Attribution |")
        add("|---|---|---|---|---|")
        for v in voices:
            add(f"| `{v['name']}` | {v['display']} | {v['source']} | {v['license']} | "
                f"{v.get('attribution', '')} |")
    add("\n## Decision gate (Stage 05)\n")
    decision = gate_decision(gate)
    if not gate:
        add("Not recorded: pass `--gate-report` with the Stage 05 gate result "
            "(`gate_report.py` JSON).")
    else:
        add(f"Winner: **{decision.get('winner')}** — {decision.get('reason', '')}\n")
        keys = ("hop_ms", "lookahead_frames", "compute_p95_ms", "projected_t_minus_b_ms",
                "wer", "cer", "tgt_cos", "src_cos", "utmos", "f0_pcc", "export_ok", "eligible")
        rows = gate.get("candidates") or []
        if rows:
            add("| Candidate | " + " | ".join(keys) + " |")
            add("|---" * (len(keys) + 1) + "|")
            for r in rows:
                add(f"| {r.get('candidate')} | " + " | ".join(_cell(r.get(k)) for k in keys)
                    + " |")
    add("\n## Exporter\n")
    for role in ("encoder", "vocoder"):
        for v, f in sorted((roles.get(role, {}).get("files") or {}).items()):
            why = f" — dynamo failed: {f['fallback_reason']}" if f["fallback_reason"] else ""
            add(f"- `{f['file']}`: {f['exporter']}{why}; {len(f['warnings'])} trace warning(s)")
            for w in f["warnings"]:
                add(f"  - {w}")
    add("\n## Versions\n")
    for k, v in report["versions"].items():
        add(f"- {k}: {v}")
    add("\n## Parity (ORT chunked vs. PyTorch offline)\n")
    add(f"Input: {report.get('parity_input', {}).get('source', 'not run')}, "
        f"{report.get('parity_input', {}).get('seconds', 0):.1f} s. Gates: encoder unit "
        f"agreement >= {fx.MIN_UNIT_AGREEMENT}, vocoder SNR >= {fx.MIN_VOCODER_SNR_DB:.0f} dB.\n")
    add("| Model | Variant | Seconds | Result | Gate |")
    add("|---|---|---|---|---|")
    for v, p in sorted((roles.get("encoder", {}).get("parity") or {}).items()):
        add(f"| encoder | {v} | {p['seconds']:.1f} | unit agreement {p['unit_agreement']:.5f}, "
            f"max abs Δlogit {p['max_abs_logit']:.2e} | {'pass' if p['pass'] else '**FAIL**'} |")
    for v, p in sorted((roles.get("vocoder", {}).get("parity") or {}).items()):
        add(f"| vocoder | {v} | {p['seconds']:.1f} | SNR {fmt_db(p['snr_db'])}, max abs Δ "
            f"{p['max_abs']:.2e} ({p['units']}) | {'pass' if p['pass'] else '**FAIL**'} |")
    for v, p in sorted((report.get("chain") or {}).items()):
        add(f"| chain | {v} | {p['seconds']:.1f} | unit agreement {p['unit_agreement']:.5f}, "
            f"waveform SNR {fmt_db(p['snr_db'])} (voice `{p['voice']}`) | reported |")
    add("\n## Cost (ORT-Python, one thread, per hop)\n")
    add("| Model | Variant | p50 ms | p95 ms | p99 ms | max ms | Hop ms | p95 / hop |")
    add("|---|---|---|---|---|---|---|---|")
    rows = [(role, roles.get(role, {}).get("cost") or {}) for role in ("encoder", "vocoder")]
    rows.append(("encoder + vocoder", report.get("chain_cost") or {}))
    for name, costs in rows:
        for v, k in sorted(costs.items()):
            add(f"| {name} | {v} | {k['p50_ms']:.3f} | {k['p95_ms']:.3f} | {k['p99_ms']:.3f} | "
                f"{k['max_ms']:.3f} | {k['hop_ms']:.0f} | {100 * k['p95_fraction_of_hop']:.1f}% |")
    chosen = report.get("gate_variant")
    add(f"\nBudget: encoder + vocoder p95 <= {100 * fx.COMPUTE_BUDGET:.0f}% of the hop, gated "
        + (f"for the gate's variant s{chosen}." if chosen else
           "once the Stage 05 gate names a variant (`--gate-variant`)."))
    golden = report.get("golden")
    add("\n## Golden I/O\n")
    if golden:
        add(f"`{GOLDEN}/manifest.json`: {golden['seconds']:.1f} s input, variants "
            f"{', '.join(sorted(golden['variants']))}, voice `{golden['voice']}`, "
            f"{'units + conditioning + audio' if golden.get('cond') else 'units only'}.")
    else:
        add("Not written yet.")
    add("\n## Contracts\n")
    for role in ("encoder", "vocoder"):
        for v, f in sorted((roles.get(role, {}).get("files") or {}).items()):
            add(f"`{f['file']}`:\n\n```json\n{json.dumps(f['contract'], indent=2)}\n```\n")
    return "\n".join(lines) + "\n"


# -- main --------------------------------------------------------------------------
def run(args):
    bundle = pathlib.Path(args.bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    report = load_report(bundle)
    ckpt_path = pathlib.Path(args.checkpoint).resolve()
    if args.role == "encoder":
        model, info = load_encoder(ckpt_path, args.head, args.license)
        ckpt = None
    else:
        model, ckpt, info = load_vocoder(ckpt_path, args.license)

    files = export_role(args, bundle, model, info, ckpt)
    role = {"checkpoint": str(ckpt_path), "checkpoint_name": ckpt_path.name,
            "checkpoint_sha256": fx.sha256(ckpt_path), "files": files,
            "license": next(iter(files.values()))["contract"]["license"],
            "step": info.get("step"), "labels_meta": info.get("labels_meta")}
    if args.role == "encoder":
        role.update(head=info["head"], lookahead_frames=model.cfg.lookahead[info["head"]])
    old = report["roles"].get(args.role) or {}
    same_source = all(old.get(k) == role.get(k) for k in ("checkpoint_sha256", "head", "license"))
    if same_source:                 # same model: variants exported earlier stay valid
        role["files"] = {**(old.get("files") or {}), **files}
    else:                           # a different model: drop the old one's variants
        for v in sorted(set(old.get("files") or {}) - set(files)):
            stale = bundle / old["files"][v]["file"]
            stale.unlink(missing_ok=True)
            log(f"removed {stale.name}: exported from a different checkpoint")
        for key in ("chain", "chain_cost"):
            report.pop(key, None)
    if args.role == "vocoder":
        voices = fx.voices_json(model, ckpt)
        (bundle / VOICES).write_text(json.dumps(voices, indent=2) + "\n", encoding="utf-8")
        report["voices"] = [{k: v for k, v in x.items() if k != "embedding"}
                            for x in voices["voices"]]
        log(f"{VOICES}: {len(voices['voices'])} voice(s), validated against the contract")
    report["roles"][args.role] = role
    report["pairs"] = check_pairs(bundle, fc.FRAMES_PER_STEP)

    voices = None
    if (bundle / VOICES).exists():
        voices = json.loads((bundle / VOICES).read_text(encoding="utf-8"))
        fc.validate_voices(voices, fx.read_contract(next(bundle.glob("vocoder_s*.onnx"))))

    if not args.skip_parity:
        wav, report["parity_input"] = parity_audio(args)
        if args.role == "encoder":
            parity, cost = encoder_checks(args, bundle, model, info, wav)
        else:
            parity, cost = vocoder_checks(args, bundle, model, voices, wav)
        keep = same_source and old.get("parity_input") == report["parity_input"]
        role["parity_input"] = report["parity_input"]
        role["parity"] = {**(old.get("parity") or {} if keep else {}), **parity}
        role["cost"] = {**(old.get("cost") or {} if keep else {}), **cost}
        both = all(r in report["roles"] for r in fc.ROLES)
        if both:
            enc_model, enc_info, voc_model = other_models(args, report, model, info)
            chain, chain_cost = chain_checks(args, bundle, enc_model, enc_info["head"], voc_model,
                                             voices, wav)
            report["chain"] = {**(report.get("chain") or {}), **chain}
            report["chain_cost"] = {**(report.get("chain_cost") or {}), **chain_cost}
        report["golden"] = write_golden(args, bundle, voices if both else None, wav)

    gate = json.loads(pathlib.Path(args.gate_report).read_text(encoding="utf-8")) \
        if args.gate_report else report.get("gate")
    report["gate"] = gate
    if args.gate_variant is not None:
        report["gate_variant"] = args.gate_variant
    elif gate_variant(gate) is not None:
        report["gate_variant"] = gate_variant(gate)
    report["generated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    report["versions"] = versions()
    (bundle / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (bundle / CARD).write_text(model_card(bundle, report, gate), encoding="utf-8")
    log(f"wrote {REPORT} and {CARD}")
    failed = [what for what, ok in verdicts(report) if not ok]
    log("verdict: " + ("; ".join(f"{w} FAILED" for w in failed) if failed
                       else "every gate passed" if verdicts(report) else "exported (no gates run)"))
    return 1 if failed else 0


def other_models(args, report, model, info):
    """(encoder model, encoder info, vocoder model) for the chain: this run's
    model plus the other role reloaded from the checkpoint the report names."""
    if args.role == "encoder":
        voc_model, _, _ = load_vocoder(report["roles"]["vocoder"]["checkpoint"], None)
        return model, info, voc_model
    enc = report["roles"]["encoder"]
    enc_model, enc_info = load_encoder(enc["checkpoint"], enc["head"], enc["license"])
    return enc_model, enc_info, model


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Δ / → must not crash a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, errors="backslashreplace")
    args = parse_args(argv)
    try:
        return run(args)
    except (fx.ExportError, fc.ContractError, FileNotFoundError) as e:
        print(f"[export] error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
