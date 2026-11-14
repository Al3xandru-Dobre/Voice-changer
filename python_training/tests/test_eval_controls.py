"""Stage 09/10 eval drivers: pure metrics, DoD rules and CLI smoke (fakes only)."""
import json
import math
import pathlib
import types

import numpy as np
import pytest
import soundfile as sf

import eval_controls as ec
import eval_target_voice as etv
from fvml import evallist, vcfile

SR = 16000


def _sine(seconds, f0=150.0, amp=0.2, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


# -- pure metrics ------------------------------------------------------------------
def test_semitone_shift_and_std_ratio():
    assert ec.semitone_shift(100.0, 200.0) == pytest.approx(12.0)
    assert ec.semitone_shift(200.0, 100.0) == pytest.approx(-12.0)
    assert ec.std_ratio(0.02, 0.01) == pytest.approx(0.5)


def test_click_metric_pass_and_fail():
    ref = _sine(2.0)
    quiet = np.zeros_like(ref)
    val, bound = ec.click_metric(ref, quiet, t=1.0, sr=SR)
    assert val == 0.0 and bound > 0.0 and val <= bound
    stepped = ref.copy()
    stepped[int(1.0 * SR):int(1.0 * SR) + 2] += 0.8       # spike at the switch
    val, bound = ec.click_metric(ref, stepped, t=1.0, sr=SR)
    assert val > bound


def test_monotonic_and_controls_dod():
    assert ec.is_monotonic_decreasing([0.8, 0.7, 0.7, 0.5])
    assert not ec.is_monotonic_decreasing([0.8, 0.6, 0.7])
    results = {"pitch": {"status": "pass", "detail": "d1"},
               "blend": {"status": "skipped", "detail": "no ml"},
               "click": {"status": "fail", "detail": "d3"}}
    dod = ec.controls_dod(results)
    assert [(n, ok) for n, ok, _ in dod] == [("pitch", True), ("click", False)]


def test_target_voice_dod_pass_and_fail():
    good = {"me": {"tgt_cos": 0.60, "wer": 0.05},
            "baseline": {"tgt_cos": 0.50, "wer": 0.05},
            "public": {"lj": {"before": {"tgt_cos": 0.70, "wer": 0.05},
                              "after": {"tgt_cos": 0.69, "wer": 0.06}}}}
    dod = etv.target_voice_dod(good)
    assert all(ok for _, ok, _ in dod)
    bad = {"me": {"tgt_cos": 0.52, "wer": 0.05},
           "baseline": {"tgt_cos": 0.50, "wer": 0.05},
           "public": {"lj": {"before": {"tgt_cos": 0.70, "wer": 0.05},
                             "after": {"tgt_cos": 0.66, "wer": 0.09}}}}
    dod = etv.target_voice_dod(bad)
    assert not any(ok for _, ok, _ in dod)
    assert len(dod) == 3


# -- eval_controls CLI smoke ---------------------------------------------------------
def _write_sources(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    rows = []
    for i in range(2):
        p = src_dir / f"u{i}.wav"
        sf.write(p, _sine(2.0, 150.0 + 30 * i), SR)
        rows.append({"utt_id": f"u{i}", "speaker": "libri", "gender": "U",
                     "duration": 2.0, "path": str(p), "text": "hello"})
    tsv = tmp_path / "sources.tsv"
    evallist.write(tsv, rows, evallist.SOURCE_COLUMNS)
    return tsv, rows


def test_eval_controls_smoke_with_fakes(tmp_path, monkeypatch):
    tsv, _ = _write_sources(tmp_path)
    monkeypatch.setattr(vcfile, "find_vc_file", lambda *a, **k: tmp_path / "fv_vc_file")

    def fake_convert_list(vc_file, bundle, rows, out_dir, voice=None, args=(), **kw):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            sf.write(out_dir / f"{r['utt_id']}.wav", _sine(float(r["duration"])), SR)
        return [{"utt_id": r["utt_id"], "seconds": 0.0} for r in rows]

    def fake_convert_one(vc_file, bundle, in_wav, out_wav, voice=None, args=()):
        # Stepped output is silence: delta inside the switch window is 0.
        sf.write(out_wav, np.zeros(int(2.0 * SR), np.float32), SR)

    monkeypatch.setattr(vcfile, "convert_list", fake_convert_list)
    monkeypatch.setattr(vcfile, "convert", fake_convert_one)

    out = tmp_path / "out"
    rc = ec.main(["--bundle", str(tmp_path / "bundle"), "--voice", "lj",
                  "--list", str(tsv), "--out", str(out),
                  "--checks", "click,blend", "--skip-ml", "--device", "cpu"])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["checks"]["click"]["status"] == "pass"
    assert report["checks"]["blend"]["status"] == "skipped"
    assert report["verdict"] == "PASS"
    assert "SKIPPED" in (out / "report.md").read_text()


def test_eval_controls_failing_check_exits_1(tmp_path, monkeypatch):
    tsv, _ = _write_sources(tmp_path)
    monkeypatch.setattr(vcfile, "find_vc_file", lambda *a, **k: tmp_path / "fv_vc_file")

    def fake_convert_list(vc_file, bundle, rows, out_dir, voice=None, args=(), **kw):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            sf.write(out_dir / f"{r['utt_id']}.wav", _sine(float(r["duration"])), SR)
        return [{"utt_id": r["utt_id"], "seconds": 0.0} for r in rows]

    def spiky(vc_file, bundle, in_wav, out_wav, voice=None, args=()):
        x = _sine(2.0)
        x[SR:SR + 2] += 0.9                          # audible click at T = 1 s
        sf.write(out_wav, x, SR)

    monkeypatch.setattr(vcfile, "convert_list", fake_convert_list)
    monkeypatch.setattr(vcfile, "convert", spiky)
    out = tmp_path / "out"
    rc = ec.main(["--bundle", str(tmp_path / "bundle"), "--voice", "lj",
                  "--list", str(tsv), "--out", str(out),
                  "--checks", "click", "--device", "cpu"])
    assert rc == 1
    assert json.loads((out / "report.json").read_text())["verdict"] == "FAIL"


# -- eval_target_voice CLI smoke ------------------------------------------------------
def _eval_args(tmp_path, tsv):
    me_targets = tmp_path / "me_targets.tsv"
    evallist.write(me_targets, [{"voice": "me", "clip_id": "m0", "speaker": "me",
                                 "gender": "U", "duration": 3.0,
                                 "path": str(tsv), "dataset": "own_voice"}],
                   evallist.TARGET_COLUMNS)
    held = tmp_path / "heldout.jsonl"
    held.write_text(json.dumps({"utt_id": "m0", "path": str(tsv), "duration": 3.0,
                                "speaker": "me"}) + "\n")
    public = tmp_path / "targets.tsv"
    evallist.write(public, [{"voice": "lj", "clip_id": "t0", "speaker": "lj",
                             "gender": "U", "duration": 3.0, "path": str(tsv),
                             "dataset": "libri"}], evallist.TARGET_COLUMNS)
    return types.SimpleNamespace(
        bundle_before=str(tmp_path / "before"), bundle_after=str(tmp_path / "after"),
        voice="me", baseline_voice="hifitts_92", checkpoint=None,
        public_voices="lj", list=str(tsv), targets=str(public),
        me_targets=str(me_targets), heldout_manifest=str(held),
        out=str(tmp_path / "out"), limit=None, device="cpu")


def _fake_eval(metrics_by_tag):
    def fake(sources, converted, targets, voice, metrics, device):
        tag = pathlib.Path(converted).name
        m = metrics_by_tag.get(tag, {})
        return {"tgt_cos": m.get("cos", 0.5), "wer": m.get("wer", 0.05),
                "utmos": m.get("utmos", 4.1)}
    return fake


def test_eval_target_voice_run_with_fakes(tmp_path, monkeypatch):
    tsv, _ = _write_sources(tmp_path)
    monkeypatch.setattr(vcfile, "find_vc_file", lambda *a, **k: tmp_path / "fv_vc_file")
    calls = []

    def fake_convert_list(vc_file, bundle, rows, out_dir, voice=None, args=(), **kw):
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        calls.append((pathlib.Path(bundle).name, voice, len(rows)))
        return [{"utt_id": r["utt_id"], "seconds": 0.0} for r in rows]

    monkeypatch.setattr(vcfile, "convert_list", fake_convert_list)
    metrics = {"before_baseline": {"cos": 0.50}, "after_me": {"cos": 0.57},
               "before_lj": {"cos": 0.70, "wer": 0.05},
               "after_lj": {"cos": 0.69, "wer": 0.06},
               "me_resynthesis": {"utmos": 4.0}}
    args = _eval_args(tmp_path, tsv)
    results, dod = etv.run(args, evaluate_fn=_fake_eval(metrics), log=lambda *a: None)
    assert all(ok for _, ok, _ in dod)
    assert results["me_resynthesis_utmos"] == pytest.approx(4.0)
    # after/me + before/baseline + public before/after + resynthesis conversions
    assert (("before", "hifitts_92") in [(b, v) for b, v, _ in calls]
            and ("after", "me") in [(b, v) for b, v, _ in calls]
            and ("after", "lj") in [(b, v) for b, v, _ in calls])
    report = json.loads((pathlib.Path(args.out) / "report.json").read_text())
    assert report["verdict"] == "PASS"
    assert (pathlib.Path(args.out) / "heldout_sources.tsv").is_file()
    # DoD failure propagates: baseline stronger than the fine-tuned voice.
    metrics["after_me"] = {"cos": 0.40}
    results2, dod2 = etv.run(args, evaluate_fn=_fake_eval(metrics), log=lambda *a: None)
    assert not all(ok for _, ok, _ in dod2)


def test_bundle_voices_reads_voices_json(tmp_path):
    b = tmp_path / "bundle"
    b.mkdir()
    (b / "voices.json").write_text(json.dumps(
        {"voices": [{"name": "lj"}, {"name": "me"}]}))
    assert etv._bundle_voices(b) == ["lj", "me"]
