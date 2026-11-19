"""gate_s2ut.py (Milestone 3 / Stage 05): the eligibility rules and selection
(offline --self-test plus mutation checks), and the whole conversion path on
the Stage 02 fixture bundle: fixture encoder_s4.onnx -> tiny translator in ONNX
Runtime through WaitKPolicy -> unit FIFO -> fixture vocoder_s4.onnx, traces that
eval_s2ut's Average Lagging reads, the ORT compute clock and the report files.
Scoring models (Whisper, UTMOS) are replaced by a fake scorer."""
import json
import pathlib

import numpy as np
import pytest
import torch

from fvml import audio, evallist, s2ut_eval
from fvml import s2ut_stream as ss
from fvml import translator as tr
from fvml.waitk_policy import WaitKConfig
import gate_s2ut as gate

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "tests" / "data" / "fixtures" / "s2ut"


def test_self_test_passes(capsys):
    assert gate.main(["--self-test"]) == 0
    assert "checks passed" in capsys.readouterr().out


def test_export_step_graphs_guards_noncommercial_licence(tmp_path):
    """BUG-031: a translator trained on non-commercial units refuses to export
    unless opted in, and is then stamped with the NC note."""
    from fvml import export as fx

    cfg = tr.tiny_config(latent_dim=8, vocab=16)
    model = tr.S2utTranslator(cfg).eval()
    meta = {"vocab": cfg.vocab, "label_space": "synthetic", "noncommercial": True,
            "license": "CC-BY-NC-SA-4.0"}
    with pytest.raises(fx.ExportError, match="non-commercial"):
        ss.export_step_graphs(model, tmp_path / "clean", exporter="torchscript",
                              labels_meta=meta)
    rec = ss.export_step_graphs(model, tmp_path / "nc", exporter="torchscript",
                                labels_meta=meta, include_noncommercial=True)
    assert rec["encoder_contract"].license.endswith(fx.NC_NOTE)


def test_each_rule_decides_on_its_own():
    """Break one rule at a time on an otherwise passing row: exactly that check
    fails, and the row stops being eligible."""
    cand = gate.Candidate("waitk", 3)
    base = gate.fake_row(cand)
    assert gate.apply_gate_rules([base])[0][0]["eligible"]
    breakers = {
        "bleu": {"bleu": {**base["bleu"], "en->de": 13.9}},
        "utmos": {"utmos": {**base["utmos"], "fr": 2.9}},
        "onset": {"projected_onset_ms": 501.0},
        "al": {"al_ms": {**base["al_ms"], "fr->en": 1201.0}},
        "compute": {"compute_p95_ms": 41.0, "projected_onset_ms": 331.0},
        "silence": {"hallucinated_words": 2},
        "flush": {"flush_tail_chunks_max": 4},
    }
    assert set(breakers) == set(gate.CHECKS)
    for name, override in breakers.items():
        row = gate.apply_gate_rules([gate.fake_row(cand, **override)])[0][0]
        failed = [c for c, v in row["checks"].items() if not v["ok"]]
        assert failed == [name] and not row["eligible"], name


def test_pytorch_clock_can_never_be_eligible():
    row, decision = gate.apply_gate_rules([gate.fake_row(
        gate.Candidate("waitk", 2), compute_clock="torch", compute_p95_ms=1.0)])
    assert not row[0]["eligible"] and decision["winner"] is None
    assert not row[0]["checks"]["compute"]["ok"] and not row[0]["checks"]["onset"]["ok"]


def test_per_pair_default_k_is_the_smallest_passing_k():
    weak = {"de->en": {2: 10.0, 3: 15.9, 4: 16.0, 5: 19.0}, "en->de": {2: 9.0, 3: 12.0, 4: 13.0, 5: 14.5}}
    rows = []
    for k in (2, 3, 4, 5):
        row = gate.fake_row(gate.Candidate("waitk", k))
        row["bleu"] = {**row["bleu"], **{pair: by_k[k] for pair, by_k in weak.items()}}
        rows.append(row)
    rows, decision = gate.apply_gate_rules(rows)
    assert [r["eligible"] for r in rows] == [False, False, False, True]
    assert decision["winner"] == "waitk-5"
    assert decision["default_wait_k"] == {"es->en": 2, "fr->en": 2, "de->en": 4, "en->es": 2,
                                          "en->fr": 2, "en->de": 5}
    merged = gate.langs_default_wait_k(decision, json.loads((FIXTURES / "langs.json").read_text()))
    assert merged["default_wait_k"]["en->de"] == 5 and len(merged["langs"]) == 4


# -- end to end on the fixture bundle -------------------------------------------------
def speech_wav(seconds, tail=0.0, f0=180.0):
    t = np.arange(int(16000 * seconds)) / 16000.0
    return np.concatenate([0.25 * np.sin(2 * np.pi * f0 * t),
                           np.zeros(int(16000 * tail))]).astype(np.float32)


def make_rig(tmp_path, monkeypatch, babble=False):
    """A tiny stacked_4 checkpoint that fits the fixture bundle (latent_dim 8,
    vocab 16), eval lists over synthetic wavs, and a fake scorer. With
    `babble=True` the decoder never predicts <EOS> (the BUG-030 babbling
    decoder); otherwise its flushes terminate via <EOS>."""
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path / "data"))
    (tmp_path / "data" / "wav").mkdir(parents=True)
    torch.manual_seed(0)
    cfg = tr.tiny_config(latent_dim=8, vocab=16)
    model = tr.S2utTranslator(cfg)
    with torch.no_grad():                           # always regular units
        model.decoder.unit_head.bias.view(4, cfg.n_classes)[:, cfg.vocab:] = -50.0
        model.decoder.eos_head.bias.fill_(-50.0 if babble else 50.0)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "config": json.dumps(cfg.to_dict()),
                "ctc_vocab": "{}"}, ckpt)

    lists = tmp_path / "lists"
    lists.mkdir()

    def rows_for(pairs, prefix):
        rows = []
        for i, pair in enumerate(pairs):
            src, tgt = pair.split("->")
            uid = f"{prefix}_{src}_{tgt}_{i}"
            audio.save(tmp_path / "data" / "wav" / f"{uid}.wav", speech_wav(1.6 + 0.16 * i))
            rows.append({"utt_id": uid, "src_lang": src, "tgt_lang": tgt, "speaker": "s", "gender": "F",
                         "src_duration": 1.6 + 0.16 * i, "tgt_duration": 1.6, "gamma": 1.0,
                         "src_path": f"wav/{uid}.wav", "tgt_path": f"wav/{uid}.wav",
                         "src_text": "hola", "tgt_text": "hello"})
        return rows

    for pair, name in gate.TEST_LISTS.items():
        evallist.write(lists / name, rows_for([pair, pair], "t"), evallist.S2UT_COLUMNS)
    evallist.write(lists / gate.TORTURE_LIST, rows_for(["de->en", "en->de"], "r"),
                   evallist.S2UT_COLUMNS)
    audio.save(tmp_path / "data" / "wav" / "sil_speech.wav", speech_wav(1.2, tail=2.0))
    audio.save(tmp_path / "data" / "wav" / "sil_noise.wav", np.zeros(16000, np.float32))
    evallist.write(lists / gate.SILENCE_LIST, [
        {"utt_id": "sil_noise", "kind": "noise_only", "src_lang": "en", "tgt_lang": "es",
         "src_duration": 1.0, "speech_end_s": 0.0, "src_path": "wav/sil_noise.wav",
         "src_text": "", "tgt_text": ""},
        {"utt_id": "sil_speech", "kind": "trailing_silence", "src_lang": "en", "tgt_lang": "fr",
         "src_duration": 3.2, "speech_end_s": 1.2, "src_path": "wav/sil_speech.wav",
         "src_text": "hello", "tgt_text": ""}], evallist.SILENCE_COLUMNS)

    scored = []

    def fake_scorer(cfg_eval, out_dir):
        rows = evallist.read_s2ut(cfg_eval.eval_list)
        conv = pathlib.Path(cfg_eval.converted)
        traces = s2ut_eval.load_traces(cfg_eval.trace, conv, [r["utt_id"] for r in rows])
        als = []
        for r in rows:
            wav = audio.load(conv / f"{r['utt_id']}.wav")
            assert len(wav) % 1280 == 0 and len(wav) >= int(r["src_duration"] * 16000) - 1280
            al, al_ca = s2ut_eval.average_lagging_ms(
                1000.0 * float(r.get("speech_end_s", r["src_duration"])), traces[r["utt_id"]])
            if not np.isnan(al):
                als.append(al)
                assert al_ca >= al
        scored.append((pathlib.Path(cfg_eval.eval_list).name, cfg_eval.metrics))
        return {"asr_bleu": 30.0, "asr_chrfpp": 50.0, "utmos": 3.5, "n": len(rows),
                "al_ms": float(np.mean(als)) if als else None, "al_ca_ms": None,
                "onset_delay_ms": 240.0, "hallucinated_words": 0,
                "tail_hallucinated_words": 0, "trailing_non_silent_ms": 0.0}

    monkeypatch.setattr(s2ut_eval, "evaluate_s2ut", fake_scorer)
    return {"ckpt": ckpt, "lists": lists, "out": tmp_path / "gate", "scored": scored, "cfg": cfg}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return make_rig(tmp_path, monkeypatch)


def run(rig, *extra):
    return gate.main(["--checkpoint", str(rig["ckpt"]), "--bundle", str(FIXTURES),
                      "--lists-dir", str(rig["lists"]), "--out", str(rig["out"]),
                      "--exporter", "torchscript", "--candidates", "waitk-2,waitk-4,ctc-adaptive",
                      *extra])


def test_gate_runs_the_whole_chain_on_the_ort_clock(rig, capsys):
    assert run(rig) == 0
    report = json.loads((rig["out"] / "gate_s2ut.json").read_text("utf-8"))
    rows = {r["candidate"]: r for r in report["candidates"]}
    assert list(rows) == ["waitk-2", "waitk-4", "ctc-adaptive"]
    assert len(rig["scored"]) == 3 * 8
    assert (gate.SILENCE_LIST, ("flush",)) in rig["scored"]
    assert (gate.TORTURE_LIST, ("bleu", "latency")) in rig["scored"]
    for name, k in (("waitk-2", 2), ("waitk-4", 4)):
        r = rows[name]
        assert r["compute_clock"] == "ort" and r["exporter"] == "torchscript/torchscript"
        assert r["first_write_chunks"] == k              # strict Wait-k: first write at t_src = k
        assert r["compute_steps"] > 100 and 0 < r["compute_p50_ms"] <= r["compute_p95_ms"]
        # Fixture encoder has L = 0: 25 + k * 80 + p95 + 10.
        assert r["projected_onset_ms"] == pytest.approx(35.0 + 80.0 * k + r["compute_p95_ms"])
        # eval_s2ut's Average Lagging on the written traces. The ideal k * 80 ms
        # upper structure holds, but this rig's decoder emits no frames on READ
        # steps, so |Y| shrinks by (k - 1) chunks and the gamma correction pulls
        # AL well below k * 80 (the exact AL math is pinned by test_eval_s2ut).
        assert all(40.0 * k <= v <= 80.0 * k + 500.0 for v in r["al_ms"].values()), r["al_ms"]
        assert set(r["bleu"]) == set(gate.PAIRS) and set(r["utmos"]) == set(gate.TARGET_LANGS)
        assert r["hallucinated_words"] == 0 and r["fifo_overflows"] == 0
        # BUG-030: the flush audit is measured and passing for a terminating decoder.
        assert r["flush_capped"] == 0 and r["tail_hallucinated_words"] == 0, r["reasons"]
        assert r["eligible"], r["reasons"]
    assert rows["waitk-2"]["al_ms"]["es->en"] < rows["waitk-4"]["al_ms"]["es->en"]
    assert report["decision"]["winner"] in ("waitk-2", "ctc-adaptive")
    assert report["decision"]["default_wait_k"] == {p: 2 for p in gate.PAIRS}
    # This decoder ends every utterance with <EOS>: the last write carries the
    # final unit frames, so no write follows the last unit frame.
    assert rows["waitk-2"]["flush_tail_chunks_max"] == 0
    md = (rig["out"] / "gate_s2ut.md").read_text("utf-8")
    assert "| Strict Wait-2 | `stacked_4` |" in md and "neutral target-voice prosody" in md
    langs = json.loads((rig["out"] / "langs.json").read_text("utf-8"))
    assert langs["default_wait_k"]["de->en"] == 2 and langs["schema"] == 1
    assert "**Decision:**" in capsys.readouterr().out

    # Files eval_s2ut.py can score on its own.
    conv = rig["out"] / "converted" / "waitk-2_stacked_4" / "test_es_en_200"
    assert sorted(p.name for p in conv.iterdir()) == \
        ["t_es_en_0.wav", "t_es_en_1.wav", "timing.csv", "trace.jsonl"]
    trace = [json.loads(x) for x in (conv / "trace.jsonl").read_text("utf-8").splitlines()][0]
    decisions = [s["decision"] for s in trace["steps"]]
    # A terminating decoder drains on <EOS> before the tail flush opens, so the
    # utterance ends in READ steps (the babbling rig below shows the FLUSH path).
    assert decisions[:2] == ["READ", "WRITE"] and "WRITE" in decisions
    assert decisions[-1] == "READ"
    assert [s["t_tgt_frames_emitted"] for s in trace["steps"][:3]] == [0, 4, 4]
    assert trace["steps"][-1]["t_src_ms"] == 1600.0     # capped at the source duration
    # Silence in, silence out: the policy never leaves IDLE on the digital-silence clip.
    sil = rig["out"] / "converted" / "waitk-2_stacked_4" / "silence_hallucination_20"
    assert float(np.abs(audio.load(sil / "sil_noise.wav")).max()) == 0.0
    assert float(np.abs(audio.load(sil / "sil_speech.wav")).max()) > 0.0


def test_babbling_decoder_fails_the_flush_gate(tmp_path, monkeypatch):
    """BUG-030: a decoder that never predicts <EOS> and babbles across the
    trailing silence cannot be eligible - it used to score
    flush_tail_chunks = 0 (PASS) and hallucinated_words = 0."""
    rig = make_rig(tmp_path, monkeypatch, babble=True)
    assert run(rig) == 2                            # no winner: nobody eligible
    report = json.loads((rig["out"] / "gate_s2ut.json").read_text("utf-8"))
    assert report["candidates"]
    for row in report["candidates"]:
        assert not row["eligible"], (row["candidate"], row["reasons"])
        assert row["reasons"][0].startswith("flush:"), row["reasons"]
        assert row["flush_capped"] > 0, "the silence list saw a flush bound without <EOS>"
    # ... and the babble really ran through kFlush steps on the trailing-silence clip.
    trace = [json.loads(x) for x in (
        rig["out"] / "converted" / "waitk-2_stacked_4" / "silence_hallucination_20"
        / "trace.jsonl").read_text("utf-8").splitlines() if x.strip()]
    by_id = {t["utt_id"]: t["steps"] for t in trace}
    assert "FLUSH" in {s["decision"] for s in by_id["sil_speech"]}


def test_torch_runtime_is_reported_but_not_gate_valid(rig):
    assert run(rig, "--runtime", "torch", "--candidates", "waitk-3") == 2
    report = json.loads((rig["out"] / "gate_s2ut.json").read_text("utf-8"))
    row = report["candidates"][0]
    assert row["compute_clock"] == "ort+torch" and not row["eligible"]
    assert any("not ONNX Runtime" in r for r in row["reasons"])
    assert not (rig["out"] / "langs.default_wait_k.json").exists()
    assert "not gate-valid" in (rig["out"] / "gate_s2ut.md").read_text("utf-8")


def test_missing_inputs_exit_1(rig, tmp_path, capsys):
    (rig["lists"] / gate.TORTURE_LIST).unlink()
    assert run(rig) == 1
    assert "reordering_torture_60.tsv" in capsys.readouterr().err
    assert gate.main(["--checkpoint", str(tmp_path / "nope.pt"), "--bundle", str(FIXTURES),
                      "--lists-dir", str(rig["lists"]), "--out", str(rig["out"]),
                      "--allow-missing-lists"]) == 1


def test_convert_utterance_reduced_dur_paces_frames_through_the_fifo():
    """reduced_dur through the same loop with fake Stage 1 / vocoder stages:
    variable-length tokens come out as whole 4-frame hops in order."""
    cfg = tr.tiny_config(unit_mode="reduced_dur", max_duration=6)
    torch.manual_seed(1)
    model = tr.S2utTranslator(cfg).eval()
    with torch.no_grad():
        model.decoder.unit_head.bias[cfg.vocab:] = -50.0
        model.decoder.eos_head.bias.fill_(-50.0)

    class Frontend:
        clock = "fake"

        def reset(self):
            self.rng = np.random.default_rng(0)

        def latents(self, chunk):
            return self.rng.standard_normal((4, cfg.latent_dim)).astype(np.float32)

    class Vocoder:
        clock = "fake"

        def reset(self):
            self.units = []

        def synthesize(self, units):
            self.units += list(units)
            return np.full(1280, 0.1, np.float32)

    voc = Vocoder()
    st = ss.StreamingTranslator(ss.TorchRuntime(model, "en", "de"), WaitKConfig(k=2), record=True)
    wav, steps, stats = gate.convert_utterance(speech_wav(2.4), st, Frontend(), voc)
    emitted = [u for d in st.dec_log for u in [int(d["units"][0])] * d["duration"]]
    assert len(emitted) >= 4 * sum(s["writes"] for s in steps)
    assert voc.units == emitted[:len(voc.units)] and len(emitted) - len(voc.units) < 4
    assert len(wav) == 1280 * len(steps) and stats["first_write_chunks"] == [2]
    assert sum(s["t_tgt_frames_emitted"] for s in steps) == len(emitted)
