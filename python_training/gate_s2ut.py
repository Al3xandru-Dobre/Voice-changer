#!/usr/bin/env python3
"""Milestone 3 / Stage 05 policy & architecture decision gate.

Spec: Documents/Milestone 3/05_causal_s2ut_translator_policy_gate.md.

For every candidate (Strict Wait-k for k in 2..5 and CTC-Adaptive on the
stacked_4 checkpoint; Strict Wait-3 on the reduced_dur checkpoint) the gate

  1. translates the six test_<pair>_200.tsv lists, reordering_torture_60.tsv and
     silence_hallucination_20.tsv chunk by chunk through the real WaitKPolicy
     (fvml/s2ut_stream.py): Stage 1 encoder -> translator_enc -> translator_dec
     -> unit FIFO -> vocoder, writing <utt_id>.wav, trace.jsonl and timing.csv;
  2. scores each list with the eval_s2ut.py machinery (ASR-BLEU, ASR-chrF++,
     onset, SimulEval AL / AL_CA, UTMOS, tgt_cos, flush / hallucination audit);
  3. takes the single-core per-chunk compute p95 of the write steps
     (encoder s4 + translator_enc + translator_dec + vocoder s4) from that run;
  4. applies the eligibility thresholds, picks the default policy and the
     per-pair default k, and writes gate_s2ut.json, gate_s2ut.md (the stage
     file's table) and langs.default_wait_k.json (the `default_wait_k` block
     for the bundle's langs.json; --langs-json also writes a merged langs.json).

Stage 1 and the vocoder come from a Stage 03 bundle as ONNX files
(--bundle: encoder_s4.onnx, vocoder_s4.onnx, voices.json) and run in ONNX
Runtime. The translator runs in ONNX Runtime too by default (--runtime ort: its
step graphs are exported to <out>/onnx/). MEMORY.md rule 11 / BUG-015: the
compute rule is only ever applied to ORT timings; with --runtime torch, or any
stage not on the ORT clock, a candidate cannot be eligible. An exporter fallback
is recorded in the report (BUG-016).

Until Stage 06 adds the prosody head, the vocoder is conditioned on neutral
target-voice prosody (z_f0 = z_energy = 0, voiced) or, with --unit-prosody, on a
per-unit prior table [vocab, 4] = (z_f0, voicing, aperiodicity, z_energy).

Selection (latency first, quality by floors): the default policy is the eligible
candidate with the lowest projected onset T-B (ties: higher mean ASR-BLEU, then
stacked_4, then strict Wait-k). The default k of a pair is the smallest k whose
strict Wait-k row, in the winning unit mode, passes that pair's BLEU floor, AL
limit and target-language UTMOS plus the pair-independent rules.

  gate_s2ut.py --checkpoint checkpoints/s2ut_v1/final.pt \\
      --checkpoint-reduced checkpoints/s2ut_reduced_v1/final.pt \\
      --bundle models/m3_default --voice lj --out runs/s2ut_gate
  gate_s2ut.py --bench-translator --out runs/s2ut_bench   # ORT cost of the translator alone
  gate_s2ut.py --self-test                                 # offline, standard library only

Exit status: 0 = table written and a default policy selected; 2 = table written,
no eligible candidate; 1 = missing or invalid inputs (or a failed self-test).
"""
import argparse
import csv
import dataclasses
import json
import math
import pathlib
import sys
import tempfile
import time

# Only the standard library up here: --self-test must run without numpy/torch
# (fvml.contract is standard library too). Everything heavy is imported inside
# the functions that convert or score.
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:    # `python -I gate_s2ut.py`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PAIRS = ("es->en", "fr->en", "de->en", "en->es", "en->fr", "en->de")
TARGET_LANGS = ("en", "es", "fr", "de")
MIN_BLEU = {"fr->en": 22.0, "es->en": 22.0, "de->en": 16.0,
            "en->es": 18.0, "en->fr": 18.0, "en->de": 14.0}
MAX_AL_MS = {"es->en": 1200.0, "fr->en": 1200.0, "en->es": 1200.0, "en->fr": 1200.0,
             "de->en": 1500.0, "en->de": 1500.0}
MIN_UTMOS = 3.0
MAX_ONSET_MS = 500.0
MAX_COMPUTE_P95_MS = 40.0           # 50 % of an 80 ms chunk
MAX_HALLUCINATED_WORDS = 0
MAX_FLUSH_TAIL_CHUNKS = 3           # Definition of done (audited, reported)

IO_FLOOR_MS = 25.0                  # master latency budget (00_master.md)
GOVERNOR_MARGIN_MS = 10.0
CHUNK_MS = 80.0
FRAME_MS = 20.0
ENCODER_LOOKAHEAD_FRAMES = 1

TEST_LISTS = {pair: f"test_{pair.replace('->', '_')}_200.tsv" for pair in PAIRS}
TORTURE_LIST = "reordering_torture_60.tsv"
SILENCE_LIST = "silence_hallucination_20.tsv"
UNIT_MODES = ("stacked_4", "reduced_dur")
CHECKS = ("bleu", "utmos", "onset", "al", "compute", "silence")
PAIR_HEADERS = {"es->en": "`Es→En` BLEU", "fr->en": "`Fr→En` BLEU", "de->en": "`De→En` BLEU",
                "en->es": "`En→Es` BLEU", "en->fr": "`En→Fr` BLEU", "en->de": "`En→De` BLEU"}


# -- candidates -----------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Candidate:
    policy: str                     # "waitk" | "ctc-adaptive"
    k: int                          # Wait-k lag in chunks (the floor for ctc-adaptive)
    unit_mode: str = "stacked_4"
    extra_wait: int = 2             # ctc-adaptive: may hold back up to k + extra_wait

    @property
    def name(self):
        return f"waitk-{self.k}" if self.policy == "waitk" else "ctc-adaptive"

    @property
    def label(self):
        return f"Strict Wait-{self.k}" if self.policy == "waitk" else "CTC-Adaptive"

    @property
    def slug(self):
        return f"{self.name}_{self.unit_mode}"


DEFAULT_CANDIDATES = (
    Candidate("waitk", 2), Candidate("waitk", 3), Candidate("waitk", 4), Candidate("waitk", 5),
    Candidate("ctc-adaptive", 2), Candidate("waitk", 3, "reduced_dur"),
)


def parse_candidate(spec):
    """'waitk-3', 'waitk-3:reduced_dur', 'ctc-adaptive', 'ctc-adaptive@3:stacked_4'
    (@k = the adaptive floor, default 2)."""
    head, _, mode = spec.strip().partition(":")
    mode = mode or "stacked_4"
    if mode not in UNIT_MODES:
        raise ValueError(f"candidate '{spec}': unknown unit mode '{mode}'")
    try:
        if head.startswith("waitk-"):
            cand = Candidate("waitk", int(head[len("waitk-"):]), mode)
        elif head.split("@")[0] == "ctc-adaptive":
            cand = Candidate("ctc-adaptive", int(head.partition("@")[2] or 2), mode)
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f"candidate '{spec}': expected waitk-<k>[:mode] or "
                         f"ctc-adaptive[@k][:mode]") from None
    if not 1 <= cand.k <= 10:
        raise ValueError(f"candidate '{spec}': k must be in 1..10")
    return cand


# -- rules ----------------------------------------------------------------------
def _num(v):
    """A finite number, else None (None, NaN, bool and strings are 'missing')."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def projected_onset_ms(first_write_chunks, compute_p95_ms,
                       encoder_lookahead_frames=ENCODER_LOOKAHEAD_FRAMES,
                       io_floor_ms=IO_FLOOR_MS, governor_margin_ms=GOVERNOR_MARGIN_MS):
    """Master latency budget: T - B = I/O floor + L x 20 ms + wait x 80 ms +
    compute + governor margin. `first_write_chunks` is the mean source chunks
    read at the first write (exactly k for strict Wait-k)."""
    wait, compute = _num(first_write_chunks), _num(compute_p95_ms)
    if wait is None or compute is None:
        return None
    return (io_floor_ms + encoder_lookahead_frames * FRAME_MS + wait * CHUNK_MS
            + compute + governor_margin_ms)


def _mean(values):
    values = [v for v in (_num(x) for x in values) if v is not None]
    return sum(values) / len(values) if values else None


def check_row(row):
    """The Stage 05 eligibility rules on one candidate row. Returns
    {check: {"ok": bool, "detail": str, "pairs": {pair: bool}}} in CHECKS order;
    `pairs` says which directions a per-pair rule holds for."""
    checks = {}
    bleu = row.get("bleu") or {}
    bad, pairs = [], {}
    for pair in PAIRS:
        v = _num(bleu.get(pair))
        pairs[pair] = v is not None and v >= MIN_BLEU[pair]
        if v is None:
            bad.append(f"{pair} missing")
        elif not pairs[pair]:
            bad.append(f"{pair} {v:.2f} < {MIN_BLEU[pair]:.1f}")
    checks["bleu"] = {"ok": not bad, "detail": "; ".join(bad), "pairs": pairs}

    utmos = row.get("utmos") or {}
    bad, langs = [], {}
    for lang in TARGET_LANGS:
        v = _num(utmos.get(lang))
        langs[lang] = v is not None and v >= MIN_UTMOS
        if v is None:
            bad.append(f"{lang} missing")
        elif not langs[lang]:
            bad.append(f"{lang} {v:.2f} < {MIN_UTMOS:.1f}")
    checks["utmos"] = {"ok": not bad, "detail": "; ".join(bad),
                       "pairs": {pair: langs[pair.split("->")[1]] for pair in PAIRS}}

    clock = row.get("compute_clock")
    p95 = _num(row.get("compute_p95_ms"))
    if clock != "ort":
        detail = (f"compute p95 is on the '{clock}' clock, not ONNX Runtime" if clock
                  else "compute p95 not measured")
        checks_compute = {"ok": False, "detail": detail}
    elif p95 is None:
        checks_compute = {"ok": False, "detail": "compute p95 missing"}
    else:
        ok = p95 <= MAX_COMPUTE_P95_MS
        checks_compute = {"ok": ok, "detail": "" if ok else
                          f"p95 {p95:.1f} ms > {MAX_COMPUTE_P95_MS:.0f} ms"}

    onset = _num(row.get("projected_onset_ms")) if clock == "ort" else None
    if onset is None:
        checks["onset"] = {"ok": False, "detail": "projected onset needs the ORT compute p95"}
    else:
        ok = onset <= MAX_ONSET_MS
        checks["onset"] = {"ok": ok, "detail": "" if ok else
                           f"onset {onset:.0f} ms > {MAX_ONSET_MS:.0f} ms"}

    al = row.get("al_ms") or {}
    bad, pairs = [], {}
    for pair in PAIRS:
        v = _num(al.get(pair))
        pairs[pair] = v is not None and v <= MAX_AL_MS[pair]
        if v is None:
            bad.append(f"{pair} missing")
        elif not pairs[pair]:
            bad.append(f"{pair} {v:.0f} ms > {MAX_AL_MS[pair]:.0f} ms")
    checks["al"] = {"ok": not bad, "detail": "; ".join(bad), "pairs": pairs}
    checks["compute"] = checks_compute

    words = _num(row.get("hallucinated_words"))
    if words is None:
        checks["silence"] = {"ok": False, "detail": "silence_hallucination_20 not scored"}
    else:
        ok = words <= MAX_HALLUCINATED_WORDS
        checks["silence"] = {"ok": ok, "detail": "" if ok else
                             f"{int(words)} hallucinated word(s) on silence/noise"}
    return {name: checks[name] for name in CHECKS}


def _pair_ok(checks, pair):
    """Rules that hold for one direction: its BLEU floor, AL limit and target
    UTMOS, plus the pair-independent rules (onset, compute, silence)."""
    return all(checks[c]["pairs"][pair] for c in ("bleu", "utmos", "al")) and \
        all(checks[c]["ok"] for c in ("onset", "compute", "silence"))


def apply_gate_rules(rows):
    """Annotate candidate rows with `checks`, `eligible`, `reasons`, `mean_bleu`
    and `mean_al_ms`, and select the default policy and per-pair default k.
    Returns (rows, decision)."""
    out = []
    for row in rows:
        checks = check_row(row)
        tail = _num(row.get("flush_tail_chunks_max"))
        out.append({**row, "checks": checks,
                    "eligible": all(c["ok"] for c in checks.values()),
                    "reasons": [f"{name}: {c['detail']}" for name, c in checks.items()
                                if not c["ok"]],
                    "mean_bleu": _mean((row.get("bleu") or {}).get(p) for p in PAIRS),
                    "mean_al_ms": _mean((row.get("al_ms") or {}).get(p) for p in PAIRS),
                    "dod_flush_tail_ok": None if tail is None else tail <= MAX_FLUSH_TAIL_CHUNKS})

    eligible = [r for r in out if r["eligible"]]
    decision = {"winner": None, "policy": None, "unit_mode": None, "k": None,
                "default_wait_k": {}, "unresolved_pairs": list(PAIRS), "complete": False}
    if not eligible:
        decision["reason"] = ("No candidate is eligible: "
                              + "; ".join(f"{r['label']} ({r['unit_mode']}): "
                                          + (", ".join(r["reasons"]) or "?") for r in out))
        return out, decision

    def rank(r):
        return (round(r["projected_onset_ms"]), -(r["mean_bleu"] or 0.0),
                UNIT_MODES.index(r["unit_mode"]), r["policy"] != "waitk")

    win = min(eligible, key=rank)
    strict = sorted((r for r in out if r["policy"] == "waitk" and r["unit_mode"] == win["unit_mode"]),
                    key=lambda r: r["k"])
    default_k, unresolved = {}, []
    for pair in PAIRS:
        k = next((r["k"] for r in strict if _pair_ok(r["checks"], pair)), None)
        if k is None and win["policy"] != "waitk" and _pair_ok(win["checks"], pair):
            k = win["k"]
        if k is None:
            unresolved.append(pair)
        else:
            default_k[pair] = k
    decision.update(
        winner=win["candidate"], policy=win["policy"], unit_mode=win["unit_mode"], k=win["k"],
        default_wait_k=default_k, unresolved_pairs=unresolved, complete=not unresolved,
        projected_onset_ms=win["projected_onset_ms"], mean_bleu=win["mean_bleu"])
    decision["reason"] = (
        f"{win['label']} ({win['unit_mode']}) is the eligible candidate with the lowest projected "
        f"onset ({win['projected_onset_ms']:.0f} ms, mean ASR-BLEU {win['mean_bleu']:.2f}); "
        f"{len(eligible)} of {len(out)} candidates are eligible. Per-pair default k: "
        + ", ".join(f"{p} {k}" for p, k in default_k.items())
        + (f"; no strict Wait-k row passes for {', '.join(unresolved)}" if unresolved else "") + ".")
    return out, decision


def langs_default_wait_k(decision, langs_json=None):
    """The `default_wait_k` block for a bundle's langs.json. With an existing
    langs.json dict, returns a validated copy carrying the gate's values."""
    block = {pair: int(k) for pair, k in decision["default_wait_k"].items()}
    if langs_json is None:
        return {"policy": decision["policy"], "unit_mode": decision["unit_mode"],
                "default_wait_k": block}
    from fvml import contract as fc
    merged = {**langs_json, "default_wait_k": {**langs_json.get("default_wait_k", {}), **block}}
    fc.validate_langs(merged)
    return merged


def format_gate_markdown(rows, decision, notes=()):
    def fmt(v, digits=1):
        v = _num(v)
        return "—" if v is None else f"{v:.{digits}f}"

    lines = [
        "| Candidate | Unit mode | Onset $T-B$ (ms) | Mean AL (ms) | "
        + " | ".join(PAIR_HEADERS[p] for p in PAIRS)
        + " | Compute $p_{95}$ (ms) | UTMOS | Eligible |",
        "|---|---|---:|---:|" + "---:|" * len(PAIRS) + "---:|---:|---|",
    ]
    for r in rows:
        utmos = [_num((r.get("utmos") or {}).get(lang)) for lang in TARGET_LANGS]
        min_utmos = None if any(u is None for u in utmos) else min(utmos)
        clock = r.get("compute_clock")
        compute = fmt(r.get("compute_p95_ms"))
        if compute != "—" and clock != "ort":
            compute += f" ({clock}, not gate-valid)"
        verdict = "✓" if r.get("eligible") else "✗ " + "; ".join(r.get("reasons", []))
        lines.append(
            f"| {r['label']} | `{r['unit_mode']}` | {fmt(r.get('projected_onset_ms'), 0)} "
            f"| {fmt(r.get('mean_al_ms'), 0)} | "
            + " | ".join(fmt((r.get("bleu") or {}).get(p), 2) for p in PAIRS)
            + f" | {compute} | {fmt(min_utmos, 2)} | {verdict.strip()} |")
    lines += ["", f"**Decision:** {decision['reason']}"]
    if decision["default_wait_k"]:
        lines += ["", "| Direction | Default k |", "|---|---:|"]
        lines += [f"| `{pair}` | {k} |" for pair, k in decision["default_wait_k"].items()]
    audits = [f"{r['label']} ({r['unit_mode']}): flush tail "
              f"{'PASS' if r['dod_flush_tail_ok'] else 'FAIL'} "
              f"(max {r.get('flush_tail_chunks_max')} chunk(s) after the last unit frame)"
              for r in rows if r.get("dod_flush_tail_ok") is not None]
    if audits:
        lines += ["", f"Definition-of-done audit (flush ends within {MAX_FLUSH_TAIL_CHUNKS} "
                      "chunks of the last unit frame):"] + [f"- {a}" for a in audits]
    lines += [""] + [f"- {n}" for n in notes]
    return "\n".join(lines).rstrip() + "\n"


def flush_tail_chunks(steps):
    """Writes after the last step that emitted a unit frame (how long the
    decoder keeps going once the sentence is over), per trace."""
    last = max((i for i, s in enumerate(steps) if s.get("t_tgt_frames_emitted", 0) > 0), default=None)
    if last is None:
        return 0
    return sum(int(s.get("writes", 0)) for s in steps[last + 1:])


def percentile(values, q):
    """Linear-interpolated percentile (numpy's default), standard library only."""
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# -- row assembly ---------------------------------------------------------------
def summarize_candidate(cand, results, stats, encoder_lookahead_frames=ENCODER_LOOKAHEAD_FRAMES):
    """One gate row from the per-list eval summaries ({list file: eval_s2ut
    result}) and the conversion statistics ({write_ms, first_write_chunks,
    flush_tail_chunks, clock, exporter, ...})."""
    row = {"candidate": cand.name, "label": cand.label, "policy": cand.policy, "k": cand.k,
           "unit_mode": cand.unit_mode, "bleu": {}, "chrf": {}, "al_ms": {}, "al_ca_ms": {},
           "measured_onset_ms": {}, "utmos": {}, "tgt_cos": {}}
    by_lang = {}
    for pair, name in TEST_LISTS.items():
        res = results.get(name)
        if not res:
            continue
        row["bleu"][pair], row["chrf"][pair] = res.get("asr_bleu"), res.get("asr_chrfpp")
        row["al_ms"][pair], row["al_ca_ms"][pair] = res.get("al_ms"), res.get("al_ca_ms")
        row["measured_onset_ms"][pair] = res.get("onset_delay_ms")
        row["tgt_cos"][pair] = res.get("tgt_cos")
        if _num(res.get("utmos")) is not None:
            by_lang.setdefault(pair.split("->")[1], []).append((res["utmos"], res.get("n", 1)))
    for lang, scored in by_lang.items():
        row["utmos"][lang] = sum(u * n for u, n in scored) / sum(n for _, n in scored)
    torture = results.get(TORTURE_LIST)
    if torture:
        row["torture"] = {"asr_bleu": torture.get("asr_bleu"), "al_ms": torture.get("al_ms"),
                          "per_pair": torture.get("per_pair")}
    silence = results.get(SILENCE_LIST)
    if silence:
        row["hallucinated_words"] = silence.get("hallucinated_words")
        row["noise_hallucination_rate"] = silence.get("noise_hallucination_rate")
        row["trailing_non_silent_ms"] = silence.get("trailing_non_silent_ms")
    write_ms = stats.get("write_ms") or []
    row.update(
        compute_clock=stats.get("clock"), compute_steps=len(write_ms),
        compute_p50_ms=percentile(write_ms, 50), compute_p95_ms=percentile(write_ms, 95),
        compute_max_ms=max(write_ms) if write_ms else None,
        read_p95_ms=percentile(stats.get("read_ms") or [], 95),
        first_write_chunks=_mean(stats.get("first_write_chunks") or []),
        flush_tail_chunks_max=max(stats["flush_tail_chunks"]) if stats.get("flush_tail_chunks")
        else None,
        exporter=stats.get("exporter"), fallback_reason=stats.get("fallback_reason"),
        fifo_overflows=stats.get("fifo_overflows", 0))
    row["projected_onset_ms"] = projected_onset_ms(row["first_write_chunks"], row["compute_p95_ms"],
                                                   encoder_lookahead_frames)
    return row


# -- Stage 1 + vocoder from a bundle (ONNX Runtime) --------------------------------
class BundleFrontend:
    """encoder_s<n>.onnx of a contract v2 bundle: audio chunk -> latents [4, D]."""

    clock = "ort"

    def __init__(self, path):
        from fvml import export as fx
        self.stream = fx.OrtStream(path)
        self.contract = c = self.stream.contract
        if c.role != "encoder" or c.contract_version != 2:
            raise ValueError(f"{path}: Mode B needs a contract v2 encoder with a `latents` output")
        self.hop = c.hop

    def reset(self):
        self.stream.reset()

    def latents(self, chunk):
        import numpy as np
        chunk = np.asarray(chunk, np.float32)
        return np.concatenate([self.stream.step({"audio": chunk[None, i:i + self.hop]})["latents"][0]
                               for i in range(0, len(chunk), self.hop)])


class BundleVocoder:
    """vocoder_s<n>.onnx: units -> audio in the chosen voice. No translator
    prosody exists before Stage 06, so the conditioning is neutral target-voice
    prosody, or a per-unit prior table [vocab, 4] = (z_f0, voicing,
    aperiodicity, z_energy) mapped into the voice like Mode A does."""

    clock = "ort"
    NEUTRAL = (0.0, 1.0, 0.1, 0.0)

    def __init__(self, path, voice, unit_prosody=None):
        import numpy as np
        from fvml import export as fx
        self.stream = fx.OrtStream(path)
        self.contract = c = self.stream.contract
        if c.role != "vocoder" or c.content_kind != "discrete":
            raise ValueError(f"{path}: not a discrete-unit vocoder")
        self.voice = voice
        self.spk = np.asarray(voice["embedding"], np.float32)[None]
        self.table = None if unit_prosody is None else np.asarray(unit_prosody, np.float32)
        if self.table is not None and self.table.shape != (c.vocab, 4):
            raise ValueError(f"unit prosody table must be [{c.vocab}, 4], got {self.table.shape}")
        self.per_frame = c.frame_samples // c.cond_frame_samples

    def reset(self):
        self.stream.reset()

    def synthesize(self, units):
        import numpy as np
        c, v = self.contract, self.voice
        units = np.asarray(units, np.int64)
        pros = (self.table[units] if self.table is not None
                else np.tile(np.asarray(self.NEUTRAL, np.float32), (len(units), 1)))
        pros = np.repeat(pros, self.per_frame, axis=0)
        cond = {"logf0": v["logf0_mean"] + v["logf0_std"] * pros[:, 0], "voicing": pros[:, 1],
                "aperiodicity": pros[:, 2], "energy": v["energy_mean"] + v["energy_std"] * pros[:, 3]}
        out = []
        for i in range(0, len(units), c.frames_per_step):
            j = slice(i * self.per_frame, (i + c.frames_per_step) * self.per_frame)
            feeds = {"units": units[None, i:i + c.frames_per_step]}
            for name in c.cond_inputs:
                feeds[name] = self.spk if name == "spk_emb" else \
                    np.asarray(cond[name][j], np.float32)[None]
            out.append(self.stream.step(feeds)["audio"][0, :c.hop])
        return np.concatenate(out)


# -- conversion -----------------------------------------------------------------
def convert_utterance(wav, translator, frontend, vocoder, vad_dbfs=-55.0, max_tail_chunks=None):
    """One clip through the Mode B chain, in real-time order: every 80 ms chunk
    goes Stage 1 -> policy/translator -> unit FIFO -> vocoder (silence while the
    FIFO holds less than a chunk). After the audio ends, silent chunks keep
    coming until the policy has drained the utterance. Returns (output wav,
    trace steps for eval_s2ut, per-utterance stats)."""
    import numpy as np
    from fvml import asr
    from fvml.waitk_policy import UnitProsodyFifo
    cfg = translator.cfg
    hop = cfg.chunk_frames * cfg.frame_samples
    wav = np.asarray(wav, np.float32)
    n_src = max(1, -(-len(wav) // hop))
    pol = translator.policy.config
    if max_tail_chunks is None:
        max_tail_chunks = math.ceil(pol.tail_flush_ms / cfg.chunk_ms) + pol.max_flush_steps + 4
    padded = np.zeros((n_src + max_tail_chunks) * hop, np.float32)
    padded[:len(wav)] = wav
    src_ms = 1000.0 * len(wav) / cfg.sample_rate
    translator.reset()
    frontend.reset()
    vocoder.reset()
    fifo = UnitProsodyFifo(capacity=16 * cfg.chunk_frames, vocab=cfg.vocab, eos_unit=cfg.eos)
    out, steps, first_write = [], [], []
    for i in range(n_src + max_tail_chunks):
        chunk = padded[i * hop:(i + 1) * hop]
        speech = bool(asr.active_frames_mask(chunk, FRAME_MS, vad_dbfs).any())
        t0 = time.perf_counter()
        lat = frontend.latents(chunk)
        t1 = time.perf_counter()
        res = translator.push_chunk(lat, speech)
        if res.frames:
            fifo.push_chunk(res.frames, [0.0] * (4 * len(res.frames)))
        popped = fifo.pop_hop(cfg.chunk_frames)
        t2 = time.perf_counter()
        audio = vocoder.synthesize(popped[0]) if popped is not None else np.zeros(hop, np.float32)
        t3 = time.perf_counter()
        out.append(np.asarray(audio, np.float32)[:hop])
        if res.writes and res.first_write_t_src and (not steps or not steps[-1]["first_write"]):
            first_write.append(res.first_write_t_src)
        stage = {"frontend_ms": 1e3 * (t1 - t0), "tenc_ms": res.enc_ms, "tdec_ms": res.dec_ms,
                 "voc_ms": 1e3 * (t3 - t2) if popped is not None else 0.0}
        steps.append({"step": i, "t_src_ms": min((i + 1) * cfg.chunk_ms, src_ms),
                      "t_tgt_frames_emitted": len(res.frames),
                      "elapsed_ms": sum(stage.values()), "decision": res.decision.name,
                      "writes": res.writes, "encoded": res.encoded, "vocoded": popped is not None,
                      "first_write": res.first_write_t_src, **stage})
        if i >= n_src - 1 and not translator.policy.utterance_active() \
                and fifo.size() < cfg.chunk_frames:
            break
    stats = {"first_write_chunks": first_write, "fifo_overflows": fifo.overflows,
             "flush_tail_chunks": flush_tail_chunks(steps),
             "write_ms": [s["elapsed_ms"] for s in steps if s["writes"] and s["encoded"]
                          and s["vocoded"]],
             "read_ms": [s["elapsed_ms"] for s in steps if s["encoded"] and not s["writes"]]}
    return np.concatenate(out), steps, stats


def convert_list(list_path, make_translator, frontend, vocoder, out_dir, limit=None,
                 vad_dbfs=-55.0, log=print):
    """Translate every row of an S2UT eval list into out_dir (<utt_id>.wav,
    trace.jsonl, timing.csv). `make_translator(src_lang, tgt_lang)` returns the
    StreamingTranslator for a direction. Returns the merged conversion stats."""
    from fvml import audio, evallist, paths
    rows = evallist.read_s2ut(list_path)[:limit or None]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = {"write_ms": [], "read_ms": [], "first_write_chunks": [], "flush_tail_chunks": [],
              "fifo_overflows": 0}
    with open(out_dir / "trace.jsonl", "w", encoding="utf-8", newline="\n") as trace, \
            open(out_dir / "timing.csv", "w", encoding="utf-8", newline="") as timing:
        tw = csv.writer(timing, lineterminator="\n")
        tw.writerow(["utt_id", "seconds"])
        for n, r in enumerate(rows):
            wav = audio.load(paths.resolve(r["src_path"]))
            y, steps, stats = convert_utterance(wav, make_translator(r["src_lang"], r["tgt_lang"]),
                                                frontend, vocoder, vad_dbfs)
            audio.save(out_dir / f"{r['utt_id']}.wav", y, subtype="PCM_16")
            trace.write(json.dumps({"utt_id": r["utt_id"], "steps": steps}) + "\n")
            tw.writerow([r["utt_id"], f"{sum(s['elapsed_ms'] for s in steps) / 1e3:.4f}"])
            for key in ("write_ms", "read_ms", "first_write_chunks"):
                merged[key] += stats[key]
            if r.get("kind", "paired") != "noise_only":
                merged["flush_tail_chunks"].append(stats["flush_tail_chunks"])
            merged["fifo_overflows"] += stats["fifo_overflows"]
            if (n + 1) % 50 == 0:
                log(f"    {n + 1}/{len(rows)}")
    return merged


def make_runtime_factory(model, runtime, onnx_dir, exporter="auto"):
    """(factory(src, tgt) -> translator runtime, clock, export record)."""
    from fvml import s2ut_stream as ss
    if runtime == "torch":
        return (lambda src, tgt: ss.TorchRuntime(model, src, tgt)), "torch", None
    rec = ss.export_step_graphs(model, onnx_dir, exporter=exporter)
    return (lambda src, tgt: ss.OrtRuntime(rec["encoder"], rec["decoder"], model.cfg, src, tgt)), \
        "ort", rec


def policy_config(cand):
    from fvml.waitk_policy import WaitKConfig
    return WaitKConfig(k=cand.k, adaptive_mode=cand.policy == "ctc-adaptive",
                       adaptive_extra_wait=cand.extra_wait)


def list_metrics(name, with_voice):
    if name == SILENCE_LIST:
        return ("flush",)
    metrics = ("bleu", "latency") if name == TORTURE_LIST else ("bleu", "latency", "utmos")
    return metrics + (("spk",) if with_voice and name != TORTURE_LIST else ())


def run_candidate(cand, runtime, frontend, vocoder, lists, out_dir, *, scorer=None, voice=None,
                  targets=None, device="cuda", batch_size=8, limit=None, vad_dbfs=-55.0,
                  encoder_lookahead_frames=ENCODER_LOOKAHEAD_FRAMES, log=print):
    """Convert + score every list for one candidate and return its gate row.
    `runtime` is make_runtime_factory()'s result for the candidate's unit mode,
    `lists` maps list file name -> path; `scorer(S2utEvalConfig, out_dir)`
    defaults to fvml.s2ut_eval.evaluate_s2ut."""
    from fvml import s2ut_eval
    from fvml import s2ut_stream as ss
    scorer = scorer or s2ut_eval.evaluate_s2ut
    out_dir = pathlib.Path(out_dir)
    factory, clock, rec = runtime
    translators = {}

    def make_translator(src, tgt):
        if (src, tgt) not in translators:
            translators[src, tgt] = ss.StreamingTranslator(factory(src, tgt), policy_config(cand))
        return translators[src, tgt]

    clocks = {clock, getattr(frontend, "clock", "unknown"), getattr(vocoder, "clock", "unknown")}
    stats = {"write_ms": [], "read_ms": [], "first_write_chunks": [], "flush_tail_chunks": [],
             "fifo_overflows": 0, "clock": "ort" if clocks == {"ort"} else "+".join(sorted(clocks)),
             "exporter": rec and f"{rec['encoder_exporter']}/{rec['decoder_exporter']}",
             "fallback_reason": rec and rec["fallback_reason"]}
    results = {}
    for name, path in lists.items():
        conv = out_dir / "converted" / cand.slug / pathlib.Path(name).stem
        log(f"[gate] {cand.label} ({cand.unit_mode}): {name}")
        merged = convert_list(path, make_translator, frontend, vocoder, conv, limit, vad_dbfs, log)
        for key in ("write_ms", "read_ms", "first_write_chunks"):
            if name != SILENCE_LIST or key != "first_write_chunks":
                stats[key] += merged[key]
        if name == SILENCE_LIST:
            stats["flush_tail_chunks"] += merged["flush_tail_chunks"]
        stats["fifo_overflows"] += merged["fifo_overflows"]
        cfg = s2ut_eval.S2utEvalConfig(
            eval_list=path, converted=conv, trace=conv / "trace.jsonl",
            targets=targets if voice else None, voice=voice,
            metrics=list_metrics(name, bool(voice and targets)), device=device,
            batch_size=batch_size, timing=conv / "timing.csv", limit=limit)
        results[name] = scorer(cfg, out_dir / "eval" / cand.slug / pathlib.Path(name).stem)
    row = summarize_candidate(cand, results, stats, encoder_lookahead_frames)
    row["lists"] = sorted(results)
    return row


def resolve_lists(lists_dir, allow_missing=False):
    lists_dir = pathlib.Path(lists_dir)
    names = list(TEST_LISTS.values()) + [TORTURE_LIST, SILENCE_LIST]
    found = {n: lists_dir / n for n in names if (lists_dir / n).is_file()}
    missing = [n for n in names if n not in found]
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"missing eval list(s) in {lists_dir}: {', '.join(missing)} (build them with "
            "make_s2ut_eval_lists.py once prepare_s2ut_data.py has prepared the corpora, or pass "
            "--allow-missing-lists for a partial, non-deciding run)")
    return found, missing


def write_report(out_dir, rows, decision, meta, notes, langs_json=None):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = format_gate_markdown(rows, decision, notes)
    report = {"candidates": rows, "decision": decision,
              "thresholds": {"min_bleu": MIN_BLEU, "max_al_ms": MAX_AL_MS, "min_utmos": MIN_UTMOS,
                             "max_onset_ms": MAX_ONSET_MS, "max_compute_p95_ms": MAX_COMPUTE_P95_MS,
                             "max_hallucinated_words": MAX_HALLUCINATED_WORDS,
                             "max_flush_tail_chunks": MAX_FLUSH_TAIL_CHUNKS},
              "notes": list(notes), **meta}
    (out_dir / "gate_s2ut.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "gate_s2ut.md").write_text(md, encoding="utf-8")
    if decision["winner"]:
        (out_dir / "langs.default_wait_k.json").write_text(
            json.dumps(langs_default_wait_k(decision), indent=2) + "\n", encoding="utf-8")
        if langs_json is not None:
            (out_dir / "langs.json").write_text(
                json.dumps(langs_default_wait_k(decision, langs_json), indent=2) + "\n",
                encoding="utf-8")
    return md


def run_gate(args, log=print):
    import numpy as np
    from fvml import contract as fc
    from fvml import paths
    from fvml import translator as tr
    cands = ([parse_candidate(s) for s in args.candidates.split(",") if s.strip()]
             if args.candidates else list(DEFAULT_CANDIDATES))
    checkpoints = {"stacked_4": args.checkpoint, "reduced_dur": args.checkpoint_reduced}
    notes = []
    skipped = [c for c in cands if not checkpoints[c.unit_mode]]
    if skipped:
        notes.append("not run (no checkpoint for the unit mode): "
                     + ", ".join(f"{c.label} ({c.unit_mode})" for c in skipped))
    cands = [c for c in cands if checkpoints[c.unit_mode]]
    if not cands:
        raise ValueError("no candidate to run: give --checkpoint (stacked_4) and/or "
                         "--checkpoint-reduced (reduced_dur)")
    lists, missing = resolve_lists(args.lists_dir, args.allow_missing_lists)
    if missing:
        notes.append("lists not scored (their rules fail as 'missing'): " + ", ".join(missing))
    bundle = pathlib.Path(args.bundle)
    voices = json.loads((bundle / "voices.json").read_text(encoding="utf-8"))
    by_name = {v["name"]: v for v in voices["voices"]}
    voice = by_name.get(args.voice or voices["voices"][0]["name"])
    if voice is None:
        raise ValueError(f"voice '{args.voice}' is not in {bundle / 'voices.json'}: {sorted(by_name)}")
    frontend = BundleFrontend(bundle / "encoder_s4.onnx")
    table = np.load(args.unit_prosody) if args.unit_prosody else None
    vocoder = BundleVocoder(bundle / "vocoder_s4.onnx", voice, table)
    fc.validate_voices(voices, vocoder.contract)
    notes.append("vocoder conditioning: " + ("per-unit prosody prior table" if args.unit_prosody
                                             else "neutral target-voice prosody")
                 + " (the translator prosody head is Stage 06)")
    models = {}
    for mode in {c.unit_mode for c in cands}:
        model, _ = tr.S2utTranslator.from_checkpoint(checkpoints[mode], runtime_only=True)
        if model.cfg.unit_mode != mode:
            raise ValueError(f"{checkpoints[mode]} is a {model.cfg.unit_mode} checkpoint, "
                             f"expected {mode}")
        if mode == "stacked_4":
            fc.validate_s2ut_bundle(frontend.contract, model.encoder.contract(),
                                    model.decoder.contract(), vocoder.contract)
        log(f"[gate] {mode}: {model.param_counts()['runtime'] / 1e6:.2f}M run-time parameters, "
            f"translator runtime {args.runtime}")
        models[mode] = make_runtime_factory(model, args.runtime,
                                            pathlib.Path(args.out) / "onnx" / mode, args.exporter)
    rows = []
    for cand in cands:
        rows.append(run_candidate(
            cand, models[cand.unit_mode], frontend, vocoder, lists, args.out,
            voice=args.voice, targets=args.targets, device=args.device,
            batch_size=args.batch_size, limit=args.limit, vad_dbfs=args.vad_dbfs,
            encoder_lookahead_frames=frontend.contract.lookahead_frames, log=log))
        if rows[-1].get("fallback_reason"):
            notes.append(f"{cand.label}: exporter fell back to {rows[-1]['exporter']}: "
                         f"{rows[-1]['fallback_reason']}")
    if args.limit:
        notes.append(f"--limit {args.limit}: a smoke run, not a decision")
    rows, decision = apply_gate_rules(rows)
    langs_path = pathlib.Path(args.langs_json) if args.langs_json else bundle / "langs.json"
    langs_json = json.loads(langs_path.read_text(encoding="utf-8")) if langs_path.is_file() else None
    md = write_report(args.out, rows, decision,
                      {"checkpoints": checkpoints, "bundle": str(bundle), "voice": voice["name"],
                       "runtime": args.runtime, "limit": args.limit,
                       "lists_dir": str(paths.resolve(args.lists_dir))}, notes, langs_json)
    print(md)
    return 0 if decision["winner"] else 2


# -- translator-only ORT cost -----------------------------------------------------
def bench_translator(checkpoint=None, unit_mode="stacked_4", steps=400, warmup=40,
                     exporter="auto", out_dir=None, k=3, log=print):
    """Single-core ONNX Runtime cost of the translator alone on random latents:
    a READ step runs translator_enc, a WRITE step translator_enc +
    translator_dec. Without a checkpoint the full-size model is randomly
    initialised: the cost of a step does not depend on the weights."""
    import numpy as np
    import torch
    from fvml import s2ut_stream as ss
    from fvml import translator as tr
    from fvml.waitk_policy import WaitKConfig
    if checkpoint:
        model, _ = tr.S2utTranslator.from_checkpoint(checkpoint, runtime_only=True)
    else:
        torch.manual_seed(0)
        model = tr.S2utTranslator(tr.TranslatorConfig(unit_mode=unit_mode, dropout=0.0)).eval()
    cfg = model.cfg
    with tempfile.TemporaryDirectory() as tmp:
        rec = ss.export_step_graphs(model, out_dir or tmp, exporter=exporter)
        sizes = {n: pathlib.Path(rec[n]).stat().st_size for n in ("encoder", "decoder")}
        runtime = ss.OrtRuntime(rec["encoder"], rec["decoder"], cfg, "es", "en")
        st = ss.StreamingTranslator(runtime, WaitKConfig(k=k), eos_threshold=1.0)
        rng = np.random.default_rng(0)
        enc_ms, dec_ms = [], []
        for i in range(warmup + steps):
            res = st.push_chunk(rng.standard_normal((cfg.chunk_frames, cfg.latent_dim))
                                .astype(np.float32))
            if i >= warmup and res.writes:
                enc_ms.append(res.enc_ms)
                dec_ms.append(res.dec_ms)
        del st, runtime
    total = [a + b for a, b in zip(enc_ms, dec_ms)]

    def pct(xs):
        return {"p50_ms": percentile(xs, 50), "p95_ms": percentile(xs, 95), "max_ms": max(xs)}

    result = {"unit_mode": cfg.unit_mode, "checkpoint": checkpoint, "steps": len(total),
              "runtime_params": model.param_counts()["runtime"],
              "exporter": f"{rec['encoder_exporter']}/{rec['decoder_exporter']}",
              "fallback_reason": rec["fallback_reason"], "onnx_bytes": sizes,
              "read_translator_enc": pct(enc_ms), "translator_dec": pct(dec_ms),
              "write_translator_enc_dec": pct(total), "clock": "ort",
              "budget_ms": MAX_COMPUTE_P95_MS,
              "note": "translator only: Stage 1 encoder s4 and vocoder s4 come on top"}
    log(json.dumps(result, indent=2))
    return result


# -- self-test ----------------------------------------------------------------------
def fake_row(cand, **overrides):
    """A candidate row that passes every rule, for the self-test and unit tests."""
    row = {"candidate": cand.name, "label": cand.label, "policy": cand.policy, "k": cand.k,
           "unit_mode": cand.unit_mode,
           "bleu": {p: MIN_BLEU[p] + 2.0 for p in PAIRS},
           "al_ms": {p: MAX_AL_MS[p] - 200.0 for p in PAIRS},
           "utmos": {lang: 3.4 for lang in TARGET_LANGS},
           "compute_clock": "ort", "compute_p95_ms": 30.0, "first_write_chunks": float(cand.k),
           "hallucinated_words": 0, "flush_tail_chunks_max": 1}
    row.update(overrides)
    row.setdefault("projected_onset_ms",
                   projected_onset_ms(row["first_write_chunks"], row["compute_p95_ms"]))
    return row


def self_test():
    n = [0]

    def check(cond, what):
        n[0] += 1
        if not cond:
            raise AssertionError(what)

    def verdict(**overrides):
        rows, decision = apply_gate_rules([fake_row(Candidate("waitk", 3), **overrides)])
        return rows[0], decision

    def nested(key, sub, value):
        base = fake_row(Candidate("waitk", 3))[key]
        return {key: {**base, sub: value}}

    # Master latency table: k = 2..5 with 35 ms compute and L = 1.
    for k, onset in ((2, 250.0), (3, 330.0), (4, 410.0), (5, 490.0)):
        check(projected_onset_ms(k, 35.0) == onset, f"projected onset k={k}")
    check(projected_onset_ms(None, 35.0) is None and projected_onset_ms(3, float("nan")) is None,
          "onset without inputs")
    check(percentile([1, 2, 3, 4, 5], 95) == 4.8 and percentile([], 95) is None, "percentile")

    row, decision = verdict()
    check(row["eligible"] and not row["reasons"], "a passing row is eligible")
    check(decision["winner"] == "waitk-3" and decision["complete"], "single winner")
    check(decision["default_wait_k"] == {p: 3 for p in PAIRS}, "per-pair k")

    # Every threshold: exactly at the limit passes, just past it fails with its own reason.
    for pair in PAIRS:
        check(verdict(**nested("bleu", pair, MIN_BLEU[pair]))[0]["eligible"], f"bleu at floor {pair}")
        r = verdict(**nested("bleu", pair, MIN_BLEU[pair] - 0.01))[0]
        check(not r["eligible"] and [x.split(":")[0] for x in r["reasons"]] == ["bleu"]
              and pair in r["reasons"][0], f"bleu below floor {pair}")
        check(verdict(**nested("al_ms", pair, MAX_AL_MS[pair]))[0]["eligible"], f"AL at limit {pair}")
        r = verdict(**nested("al_ms", pair, MAX_AL_MS[pair] + 1.0))[0]
        check(not r["eligible"] and r["reasons"][0].startswith("al:"), f"AL over limit {pair}")
    check(MAX_AL_MS["en->de"] == MAX_AL_MS["de->en"] == 1500.0 and MAX_AL_MS["en->fr"] == 1200.0
          and MIN_BLEU == {"fr->en": 22.0, "es->en": 22.0, "de->en": 16.0, "en->es": 18.0,
                           "en->fr": 18.0, "en->de": 14.0}, "thresholds are the spec's")
    check(verdict(**nested("al_ms", "en->de", 1400.0))[0]["eligible"], "De pairs get 1500 ms")
    check(not verdict(**nested("al_ms", "en->es", 1400.0))[0]["eligible"], "Es pairs get 1200 ms")
    for lang in TARGET_LANGS:
        check(verdict(**nested("utmos", lang, 3.0))[0]["eligible"], f"UTMOS at floor {lang}")
        r = verdict(**nested("utmos", lang, 2.99))[0]
        check(not r["eligible"] and r["reasons"][0].startswith("utmos:"), f"UTMOS below {lang}")
    check(verdict(compute_p95_ms=40.0, projected_onset_ms=400.0)[0]["eligible"], "compute at 40 ms")
    r = verdict(compute_p95_ms=40.1, projected_onset_ms=400.0)[0]
    check(not r["eligible"] and r["reasons"][0].startswith("compute:"), "compute over 40 ms")
    check(verdict(projected_onset_ms=500.0)[0]["eligible"], "onset at 500 ms")
    r = verdict(projected_onset_ms=500.5)[0]
    check(not r["eligible"] and r["reasons"][0].startswith("onset:"), "onset over 500 ms")
    r = verdict(hallucinated_words=1)[0]
    check(not r["eligible"] and r["reasons"][0].startswith("silence:"), "one hallucinated word")

    # BUG-015: a PyTorch-clock p95 never makes a candidate eligible, however small.
    r, d = verdict(compute_clock="torch", compute_p95_ms=5.0)
    check(not r["eligible"] and d["winner"] is None, "torch clock is not gate-valid")
    check(any("not ONNX Runtime" in x for x in r["reasons"])
          and any(x.startswith("onset:") for x in r["reasons"]), "torch clock reasons")
    check("not gate-valid" in format_gate_markdown([r], d), "torch clock is flagged in the table")
    r = verdict(compute_clock=None, compute_p95_ms=None, projected_onset_ms=None)[0]
    check(not r["eligible"] and "compute p95 not measured" in " ".join(r["reasons"]), "no compute")

    # Missing or non-finite metrics fail; they are never read as passing.
    r = verdict(bleu={}, utmos={}, al_ms={}, hallucinated_words=None)[0]
    check([x.split(":")[0] for x in r["reasons"]] == ["bleu", "utmos", "al", "silence"], "missing")
    check(not verdict(**nested("bleu", "de->en", float("nan")))[0]["eligible"], "NaN BLEU")
    check(not verdict(hallucinated_words=True)[0]["eligible"], "bool is not a count")

    # Selection: lowest projected onset among the eligible; per-pair k from strict rows.
    def table(by_slug=()):
        return apply_gate_rules([fake_row(c, **dict(by_slug).get(c.slug, {}))
                                 for c in DEFAULT_CANDIDATES])

    rows, d = table()
    check(d["winner"] == "waitk-2" and d["unit_mode"] == "stacked_4"
          and d["default_wait_k"] == {p: 2 for p in PAIRS}, "all pass: Wait-2 wins")
    check(sum(r["eligible"] for r in rows) == 6, "six eligible")
    weak_de = {"bleu": {**fake_row(Candidate("waitk", 2))["bleu"], "de->en": 12.0, "en->de": 11.0}}
    weak = {"waitk-2_stacked_4": weak_de, "waitk-3_stacked_4": weak_de}
    rows, d = table(weak)
    check(d["winner"] == "ctc-adaptive", "adaptive (k floor 2) is the fastest eligible row")
    check(d["default_wait_k"] == {"es->en": 2, "fr->en": 2, "de->en": 4, "en->es": 2,
                                  "en->fr": 2, "en->de": 4}, "De pairs wait for k = 4")
    rows, d = table({**weak, "ctc-adaptive_stacked_4": {"hallucinated_words": 3}})
    check(d["winner"] == "waitk-3" and d["unit_mode"] == "reduced_dur",
          "reduced_dur wins only when it is the fastest eligible row")
    check(d["default_wait_k"] == {p: 3 for p in PAIRS}, "k comes from the winning unit mode's rows")
    slow = {"compute_p95_ms": 60.0, "projected_onset_ms": None}
    rows, d = apply_gate_rules([fake_row(c, **slow) for c in DEFAULT_CANDIDATES])
    check(d["winner"] is None and not d["complete"] and "No candidate is eligible" in d["reason"],
          "nobody eligible")
    tie = [fake_row(Candidate("waitk", 3)), fake_row(Candidate("waitk", 3, "reduced_dur"),
                                                     bleu={p: 40.0 for p in PAIRS})]
    check(apply_gate_rules(tie)[1]["unit_mode"] == "reduced_dur", "onset tie: higher BLEU wins")
    check(apply_gate_rules([fake_row(Candidate("waitk", 3, "reduced_dur")),
                            fake_row(Candidate("waitk", 3))])[1]["unit_mode"] == "stacked_4",
          "full tie: stacked_4 wins")
    never = {"bleu": {**fake_row(Candidate("waitk", 2))["bleu"], "en->de": 1.0}}
    rows, d = apply_gate_rules([fake_row(c, **never) for c in DEFAULT_CANDIDATES])
    check(d["winner"] is None, "a failing direction blocks every row")

    # Report: the stage file's columns, one line per candidate, valid langs.json block.
    rows, d = table({"waitk-2_stacked_4": weak_de})
    md = format_gate_markdown(rows, d, ["note"])
    head = md.splitlines()[0]
    check(head.startswith("| Candidate | Unit mode | Onset $T-B$ (ms) | Mean AL (ms) | `Es→En` BLEU")
          and head.endswith("| Compute $p_{95}$ (ms) | UTMOS | Eligible |"), "table header")
    check(sum(line.startswith(("| Strict Wait-", "| CTC-Adaptive")) for line in md.splitlines())
          == 6, "one line per candidate")
    check("| Strict Wait-2 | `stacked_4` | 245 |" in md and "✗ bleu: de->en 12.00 < 16.0" in md,
          "row cells")
    check("**Decision:**" in md and "| `de->en` | 3 |" in md and "flush tail PASS" in md, "decision")
    langs = {"schema": 1, "langs": [{"code": c, "id": i, "name": c.upper()}
                                    for i, c in enumerate(TARGET_LANGS)],
             "default_wait_k": {"es->en": 9}}
    merged = langs_default_wait_k(d, langs)
    check(merged["default_wait_k"] == d["default_wait_k"] and merged["langs"] == langs["langs"],
          "langs.json merge")
    check(langs_default_wait_k(d)["policy"] == "ctc-adaptive", "langs block")
    with tempfile.TemporaryDirectory() as tmp:
        write_report(tmp, rows, d, {"runtime": "ort"}, ["note"], langs)
        report = json.loads((pathlib.Path(tmp) / "gate_s2ut.json").read_text(encoding="utf-8"))
        check(report["decision"]["winner"] == d["winner"] and len(report["candidates"]) == 6
              and report["thresholds"]["max_compute_p95_ms"] == 40.0, "JSON report")
        check(all((pathlib.Path(tmp) / f).is_file() for f in
                  ("gate_s2ut.md", "langs.default_wait_k.json", "langs.json")), "report files")

    # Candidates and trace helpers.
    check([c.slug for c in DEFAULT_CANDIDATES] == [
        "waitk-2_stacked_4", "waitk-3_stacked_4", "waitk-4_stacked_4", "waitk-5_stacked_4",
        "ctc-adaptive_stacked_4", "waitk-3_reduced_dur"], "default candidates")
    check(parse_candidate("waitk-4:reduced_dur") == Candidate("waitk", 4, "reduced_dur")
          and parse_candidate("ctc-adaptive@3") == Candidate("ctc-adaptive", 3), "parse")
    for bad in ("waitk-0", "waitk-x", "greedy", "waitk-3:stacked_2", "waitk-11"):
        try:
            parse_candidate(bad)
        except ValueError:
            n[0] += 1
        else:
            raise AssertionError(f"parse_candidate accepted {bad!r}")
    steps = [{"t_tgt_frames_emitted": f, "writes": w} for f, w in
             ((0, 0), (4, 1), (4, 1), (2, 1), (0, 1), (0, 1), (0, 0))]
    check(flush_tail_chunks(steps) == 2 and flush_tail_chunks(steps[:1]) == 0, "flush tail")
    r = apply_gate_rules([fake_row(Candidate("waitk", 3), flush_tail_chunks_max=4)])[0][0]
    check(r["dod_flush_tail_ok"] is False and r["eligible"], "flush tail is audited, not gating")
    summary = summarize_candidate(
        Candidate("waitk", 3),
        {TEST_LISTS["es->en"]: {"asr_bleu": 25.0, "al_ms": 900.0, "utmos": 3.2, "n": 100},
         TEST_LISTS["fr->en"]: {"asr_bleu": 26.0, "al_ms": 950.0, "utmos": 3.6, "n": 300},
         SILENCE_LIST: {"hallucinated_words": 0}},
        {"write_ms": [10.0, 20.0, 30.0], "first_write_chunks": [3, 3], "clock": "ort",
         "flush_tail_chunks": [1, 2]})
    check(abs(summary["utmos"]["en"] - 3.5) < 1e-9 and summary["bleu"]["fr->en"] == 26.0
          and summary["compute_p95_ms"] == 29.0 and summary["flush_tail_chunks_max"] == 2
          and abs(summary["projected_onset_ms"] - (25 + 20 + 240 + 29 + 10)) < 1e-9, "summarize")
    print(f"gate_s2ut self-test: {n[0]} checks passed")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--self-test", action="store_true", help="offline rule/report checks")
    p.add_argument("--bench-translator", action="store_true",
                   help="single-core ORT cost of translator_enc/dec alone (random init without "
                        "--checkpoint)")
    p.add_argument("--bench-steps", type=int, default=400)
    p.add_argument("--checkpoint", help="stacked_4 translator checkpoint (train_s2ut.py)")
    p.add_argument("--checkpoint-reduced", help="reduced_dur translator checkpoint")
    p.add_argument("--bundle", help="Stage 03 bundle: encoder_s4.onnx, vocoder_s4.onnx, voices.json")
    p.add_argument("--voice", help="voice in the bundle's voices.json (default: the first)")
    p.add_argument("--targets", help="target voice reference TSV (enables tgt_cos with --voice)")
    p.add_argument("--unit-prosody", help="[vocab, 4] .npy per-unit prosody prior")
    p.add_argument("--lists-dir", default=None, help="default eval_lists/s2ut")
    p.add_argument("--allow-missing-lists", action="store_true")
    p.add_argument("--candidates", help="e.g. 'waitk-3,ctc-adaptive@3,waitk-3:reduced_dur'")
    p.add_argument("--runtime", choices=("ort", "torch"), default="ort",
                   help="translator runtime (torch is not gate-valid for the compute rule)")
    p.add_argument("--exporter", choices=("auto", "dynamo", "torchscript"), default="auto")
    p.add_argument("--langs-json", help="langs.json to merge default_wait_k into "
                                        "(default <bundle>/langs.json)")
    p.add_argument("--vad-dbfs", type=float, default=-55.0)
    p.add_argument("--limit", type=int, help="first N utterances per list (smoke runs)")
    p.add_argument("--device", default="cuda", help="scoring device (Whisper, ECAPA, UTMOS)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", help="output directory")
    args = p.parse_args(argv)
    if not (args.self_test or args.bench_translator):
        for name in ("bundle", "out"):
            if not getattr(args, name):
                p.error(f"--{name} is required")
        if not (args.checkpoint or args.checkpoint_reduced):
            p.error("give --checkpoint and/or --checkpoint-reduced")
    return args


def main(argv=None):
    # BUG-015/016 family: the exit code must not depend on the console codepage.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    try:
        if args.self_test:
            return self_test()
        if args.bench_translator:
            unit_mode = "reduced_dur" if args.checkpoint_reduced and not args.checkpoint \
                else "stacked_4"
            result = bench_translator(args.checkpoint or args.checkpoint_reduced, unit_mode,
                                      args.bench_steps, exporter=args.exporter)
            if args.out:
                out = pathlib.Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                (out / "bench_translator.json").write_text(json.dumps(result, indent=2) + "\n",
                                                           encoding="utf-8")
            return 0
        if args.lists_dir is None:
            from fvml import paths
            args.lists_dir = str(paths.EVAL_LISTS / "s2ut")
        return run_gate(args)
    except AssertionError as e:
        print(f"gate_s2ut self-test FAILED: {e}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"[gate_s2ut] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
