"""Milestone 3 / Stage 10 cross-lingual "me" voice tooling — CPU only, fakes.

Covers fvml/s2utfile.py (fv_s2ut_file runner), train_unit_vocoder.py
--mix-public-lang-ratios and --freeze-unit-emb-after, eval_controls.py --s2ut
(the Stage 09 DoD on translated output), eval_cross_lingual_voice.py (the
Stage 10 DoD driver) and prepare_live_mic_set.py + the committed live-mic
prompt lists. No fv_s2ut_file, Whisper, ECAPA or UTMOS is needed, except the
optional real-binary test (FLUXVOICE_S2UT_FILE, ORT-enabled CTest).
"""
import contextlib
import json
import math
import os
import pathlib
import types

import numpy as np
import pytest
import soundfile as sf
import torch

import eval_controls as ec
import eval_cross_lingual_voice as ecl
import prepare_live_mic_set as plm
import train_unit_vocoder as tuv
from fvml import asr, audio, evallist, s2ut_eval, s2utfile, sampling, units

SR = audio.SAMPLE_RATE
REPO = pathlib.Path(__file__).resolve().parents[2]
S2UT_FIXTURE = REPO / "tests" / "data" / "fixtures" / "s2ut"


@contextlib.contextmanager
def _few_torch_threads(n=2):
    """The mini vocoders are tiny: two threads are as fast as every core and
    do not thrash when other suites (ctest -j) share the machine."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, n))
    try:
        yield
    finally:
        torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def _limit_threads():
    with _few_torch_threads():
        yield


def _sine(seconds, f0=200.0, amp=0.3, sr=SR):
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _fm(seconds, f0=150.0, depth_st=1.0, sr=SR, amp=0.3):
    """Sine with a 3 Hz vibrato of +-depth_st semitones (log-F0 std ~ depth)."""
    t = np.arange(int(round(seconds * sr))) / sr
    f = f0 * 2.0 ** (depth_st * np.sin(2 * np.pi * 3.0 * t) / 12.0)
    return (amp * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)


# -- fvml/s2utfile.py ---------------------------------------------------------------
def test_s2utfile_command_and_input_rules(tmp_path, monkeypatch):
    cmd = s2utfile.build_command("bin/fv_s2ut_file", "models/b", "in.wav", "out.wav", "en", "es",
                                 voice="me", extra_args=("--wait-k", 4, "--param", "gain=-3"),
                                 trace="t.jsonl")
    assert cmd == ["bin/fv_s2ut_file", "--model-dir", "models/b", "--src-lang", "en",
                   "--tgt-lang", "es", "--voice", "me", "--dump-trace", "t.jsonl",
                   "--wait-k", "4", "--param", "gain=-3", "in.wav", "out.wav"]
    assert "--voice" not in s2utfile.build_command("x", "b", "i", "o", "en", "de")
    with pytest.raises(ValueError, match="must differ"):
        s2utfile.build_command("x", "b", "i", "o", "en", "en")
    with pytest.raises(ValueError, match="source and a target"):
        s2utfile.build_command("x", "b", "i", "o", "en", "")

    tool = tmp_path / "fv_s2ut_file"
    tool.write_text("#!/bin/sh\nexit 0\n")
    flac = tmp_path / "in.flac"
    sf.write(flac, _sine(0.5), SR)
    with pytest.raises(ValueError, match="16 kHz mono WAV"):
        s2utfile.convert("b", flac, tmp_path / "o.wav", "en", "es", s2ut_file=tool)
    with pytest.raises(FileNotFoundError, match="not found"):
        s2utfile.convert("b", flac, tmp_path / "o.wav", "en", "es", s2ut_file=tmp_path / "nope")
    monkeypatch.setattr(s2utfile, "find_s2ut_file", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="FLUXVOICE_S2UT_FILE"):
        s2utfile.convert("b", flac, tmp_path / "o.wav", "en", "es")
    with pytest.raises(FileNotFoundError, match="FLUXVOICE_S2UT_FILE"):
        s2utfile.convert_list("b", [], tmp_path / "out")


def test_s2utfile_find_env_then_newest_build(tmp_path, monkeypatch):
    monkeypatch.delenv(s2utfile.ENV_VAR, raising=False)
    assert s2utfile.find_s2ut_file(tmp_path) is None
    old = tmp_path / "build-a" / "fv_s2ut_file"
    new = tmp_path / "build-b" / "Release" / "fv_s2ut_file.exe"
    obj = tmp_path / "build-b" / "CMakeFiles" / "fv_s2ut_file.dir" / "fv_s2ut_file.cpp.o"
    for p in (old, new, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    os.utime(obj, (3_000_000, 3_000_000))            # newest, but not a binary
    assert s2utfile.find_s2ut_file(tmp_path) == new
    monkeypatch.setenv(s2utfile.ENV_VAR, str(old))
    assert s2utfile.find_s2ut_file(tmp_path) == old
    monkeypatch.setenv(s2utfile.ENV_VAR, str(tmp_path / "missing"))   # not a file: search
    assert s2utfile.find_s2ut_file(tmp_path) == new


def test_s2utfile_convert_list_stages_inputs_traces_and_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    sf.write(tmp_path / "a.flac", _sine(1.0), 24000)              # needs staging
    audio.save(tmp_path / "u2.wav", _sine(1.0))                   # usable as is
    audio.save(tmp_path / "other.wav", _sine(1.0))                # 16 kHz but wrong stem
    rows = [{"utt_id": "u1", "src_lang": "en", "tgt_lang": "es", "src_path": "a.flac"},
            {"utt_id": "u2", "src_lang": "es", "tgt_lang": "en", "src_path": "u2.wav"},
            {"utt_id": "u/3", "src_lang": "fr", "tgt_lang": "en", "src_path": "other.wav"}]
    calls = []

    def runner(bundle, in_wav, out_wav, src, tgt, voice=None, extra_args=(), trace=None,
               s2ut_file=None):
        info = sf.info(str(in_wav))
        calls.append({"in": pathlib.Path(in_wav), "src": src, "tgt": tgt, "voice": voice,
                      "extra": extra_args, "trace": trace, "rate": info.samplerate,
                      "channels": info.channels})
        pathlib.Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        audio.save(out_wav, _sine(1.2))
        if trace is not None:     # fv_s2ut_file tags trace lines with the input file stem
            pathlib.Path(trace).write_text(json.dumps({"utt_id": pathlib.Path(in_wav).stem,
                                                       "step": 0, "t_src_ms": 80.0,
                                                       "t_tgt_frames_emitted": 4}) + "\n")
        return 0.25

    out = tmp_path / "out"
    timing = s2utfile.convert_list("models/b", rows, out, voice="me", extra_args=["--wait-k", 3],
                                   runner=runner, trace=True)
    assert [t[0] for t in timing] == ["u1", "u2", "u/3"]
    assert all(c["rate"] == SR and c["channels"] == 1 for c in calls)
    assert calls[0]["in"].stem == "u1" and calls[0]["in"].parent != tmp_path   # staged copy
    assert calls[1]["in"] == tmp_path / "u2.wav"                              # used directly
    assert calls[2]["in"].stem == units.safe_name("u/3")
    assert [(c["src"], c["tgt"]) for c in calls] == [("en", "es"), ("es", "en"), ("fr", "en")]
    assert all(c["voice"] == "me" and c["extra"] == ("--wait-k", 3) for c in calls)
    assert (out / "timing.csv").read_text().splitlines() == [
        "utt_id,seconds", "u1,0.2500", "u2,0.2500", "u/3,0.2500"]
    # Trace lines carry the list's utt_id, so eval_s2ut's AL lookup finds them.
    traces = s2ut_eval.load_traces(None, out, ["u1", "u2", "u/3"])
    assert set(traces) == {"u1", "u2", "u/3"}
    # src/tgt override the row languages (Mode A source lists have none).
    mode_a = [{"utt_id": "v1", "path": "u2.wav"}]
    with pytest.raises(ValueError, match="source and a target"):
        s2utfile.convert_list("b", mode_a, tmp_path / "o2", runner=runner)
    s2utfile.convert_list("b", mode_a, tmp_path / "o2", src="de", tgt="en", runner=runner)
    assert (calls[-1]["src"], calls[-1]["tgt"]) == ("de", "en") and calls[-1]["trace"] is None
    # An output that would overwrite its own source is refused.
    with pytest.raises(ValueError, match="overwrite"):
        s2utfile.convert_list("b", [{"utt_id": "u2", "src_lang": "es", "tgt_lang": "en",
                                     "src_path": "u2.wav"}], tmp_path, runner=runner)


def test_s2utfile_real_binary_on_fixture(tmp_path):
    """The shipped fv_s2ut_file on the generated S2UT fixture bundle."""
    env = os.environ.get(s2utfile.ENV_VAR)
    if not env or not pathlib.Path(env).is_file():
        pytest.skip(f"{s2utfile.ENV_VAR} not set (run via ORT-enabled CTest)")
    src = tmp_path / "u1.wav"
    audio.save(src, _sine(1.0))
    out = tmp_path / "out" / "u1.wav"
    out.parent.mkdir()
    try:
        s2utfile.convert(S2UT_FIXTURE, src, out, "en", "es", voice="identity",
                         trace=tmp_path / "out" / "u1.trace.jsonl")
    except RuntimeError as e:
        if "built without ONNX Runtime" in str(e):
            pytest.skip("fv_s2ut_file built without ONNX Runtime")
        raise
    info = sf.info(str(out))
    assert info.samplerate == SR and info.channels == 1 and info.frames > 0
    lines = (tmp_path / "out" / "u1.trace.jsonl").read_text().splitlines()
    assert lines and all(json.loads(x)["utt_id"] == "u1" for x in lines)
    with pytest.raises(RuntimeError, match="fv_s2ut_file failed"):
        s2utfile.convert(S2UT_FIXTURE, src, tmp_path / "o.wav", "en", "es", voice="nobody")


# -- train_unit_vocoder: public language split (Task 2) -----------------------------
def _units_dir(tmp_path, n_by_utt):
    udir = tmp_path / "units"
    udir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    urows = []
    for uid, n in n_by_utt.items():
        np.save(udir / f"{uid}.npy", rng.integers(0, 16, size=n).astype(np.int16))
        urows.append({"utt_id": uid, "units": f"{uid}.npy", "n_frames": n})
    (udir / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in urows))
    (udir / units.META_FILE).write_text(
        json.dumps(units.base_meta("fake", 320, 0, 16, "fake:0", True, "", {})))
    return udir


def _multilingual_items(tmp_path, monkeypatch, untagged_public=False):
    """Synthetic manifest: one public voice per language (uneven sizes and
    lengths) + an untagged 'me' group -> join_vocoder_items output."""
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    spec = {"lj": ("en", 3), "css10_es": ("es", 5), "css10_fr": ("fr", 2),
            "css10_de": ("de", 4), "me": (None, 6)}
    rows, n_by = [], {}
    for spk, (lang, count) in spec.items():
        for i in range(count):
            uid, n = f"{spk}_{i}", 8 + 3 * i
            n_by[uid] = n
            row = {"utt_id": uid, "path": f"audio/{uid}.flac", "duration": n * 320 / SR,
                   "speaker": spk, "dataset": "fake", "subset": "train"}
            if lang and not (untagged_public and spk == "css10_fr"):
                row["lang"] = lang
            rows.append(row)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = {}
    items, _, speakers = tuv.join_vocoder_items([str(manifest)], [str(_units_dir(tmp_path, n_by))],
                                                None, info=info)
    return items, speakers, info["speaker_lang"]


def test_public_language_split_realised_proportions(tmp_path, monkeypatch):
    items, speakers, speaker_lang = _multilingual_items(tmp_path, monkeypatch)
    assert speaker_lang["me"] is None                 # the mix group carries no language
    weights = [max(n, 4) for _, _, n, _, _ in items]
    sampler, in_group = tuv.mix_batch_sampler(
        items, weights, speakers, speaker_lang, {"me"}, 0.4, 4, seed=123,
        public_lang_ratios=sampling.parse_ratios("en=1,es=1,fr=1,de=1"))
    assert sampler.batch_shares() == pytest.approx(
        {"mix": 0.4, "en": 0.15, "es": 0.15, "fr": 0.15, "de": 0.15})
    group_of = ["mix" if g else speaker_lang[it[4]] for it, g in zip(items, in_group)]
    n_batches = 4000
    batches = {"mix": 0, "en": 0, "es": 0, "fr": 0, "de": 0}
    crops = dict.fromkeys(batches, 0)
    it = iter(sampler)
    for _ in range(n_batches):
        groups = {group_of[i] for i in next(it)}
        assert len(groups) == 1                       # every batch from one group
        g = groups.pop()
        batches[g] += 1
        crops[g] += 4
    for g, want in (("mix", 0.40), ("en", 0.15), ("es", 0.15), ("fr", 0.15), ("de", 0.15)):
        assert batches[g] / n_batches == pytest.approx(want, abs=0.025), (g, batches)
        assert crops[g] / (4 * n_batches) == pytest.approx(want, abs=0.025)
    # Explicit ratios.
    sampler2, _ = tuv.mix_batch_sampler(
        items, weights, speakers, speaker_lang, {"me"}, 0.4, 4, seed=5,
        public_lang_ratios=sampling.parse_ratios("en=0.5,es=0.25,fr=0.125,de=0.125"))
    it2, en = iter(sampler2), 0
    for _ in range(n_batches):
        en += all(group_of[i] == "en" for i in next(it2))
    assert en / n_batches == pytest.approx(0.6 * 0.5, abs=0.025)
    # A language share without public items, or an untagged public voice, is an error.
    with pytest.raises(ValueError, match="no training items for language"):
        tuv.mix_batch_sampler(items, weights, speakers, speaker_lang, {"me"}, 0.4, 4, seed=1,
                              public_lang_ratios=sampling.parse_ratios("en,es,fr,de,it"))
    with pytest.raises(ValueError, match="unknown speaker"):
        tuv.mix_batch_sampler(items, weights, speakers, speaker_lang, {"you"}, 0.4, 4, seed=1)


def test_public_language_split_untagged_public_voice(tmp_path, monkeypatch):
    items, speakers, speaker_lang = _multilingual_items(tmp_path, monkeypatch,
                                                        untagged_public=True)
    weights = [max(n, 4) for _, _, n, _, _ in items]
    with pytest.raises(ValueError, match="default-lang"):
        tuv.mix_batch_sampler(items, weights, speakers, speaker_lang, {"me"}, 0.4, 4, seed=1,
                              public_lang_ratios=sampling.parse_ratios("en,es,fr,de"))


def test_mixed_batch_sampler_default_stream_is_unchanged():
    """Without the public split the sampler draws exactly the M2 sequence."""
    weights = [1.0 + (i % 3) for i in range(12)]
    mask = [i < 5 for i in range(12)]
    w = torch.as_tensor(weights, dtype=torch.double)
    m = torch.as_tensor(mask)
    w_in = torch.where(m, w, torch.zeros_like(w))
    w_out = torch.where(m, torch.zeros_like(w), w)
    g = torch.Generator().manual_seed(7)
    expected = []
    for _ in range(64):                          # the M2 algorithm, verbatim
        pick = bool(torch.rand((), generator=g) < 0.5)
        expected.append(torch.multinomial(w_in if pick else w_out, 3, replacement=True,
                                          generator=g).tolist())
    it = iter(tuv.MixedBatchSampler(weights, mask, 3, 0.5, seed=7))
    assert [next(it) for _ in range(64)] == expected


# -- train_unit_vocoder: --freeze-unit-emb-after (Task 1) ---------------------------
@pytest.fixture(scope="module")
def mini_ml(tmp_path_factory):
    """One base checkpoint (lj = en, css10_es = es) shared by the fine-tune
    tests; the manifest adds two untagged 'me' rows. Paths are absolute, so no
    data root is needed. `argv(root, run, steps, *extra)` builds a CLI run
    whose checkpoints live under `root`."""
    root = tmp_path_factory.mktemp("mini_ml")
    rng = np.random.default_rng(0)
    (root / "audio").mkdir()
    spec = [("lj", "en"), ("css10_es", "es"), ("lj", "en"), ("css10_es", "es"),
            ("me", None), ("me", None)]
    rows, n_by = [], {}
    for i, (spk, lang) in enumerate(spec):
        n = 16
        t = np.arange(n * 320) / SR
        x = (0.2 * np.sin(2 * np.pi * (180 + 20 * i) * t)
             + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        path = root / "audio" / f"u{i}.flac"
        sf.write(path, x, SR)
        n_by[f"u{i}"] = n
        row = {"utt_id": f"u{i}", "path": str(path), "duration": len(x) / SR, "speaker": spk,
               "dataset": "fake", "subset": "train"}
        if lang:
            row["lang"] = lang
        rows.append(row)
    _units_dir(root, n_by)
    (root / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "m_base.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows if r["speaker"] != "me"))

    def argv(out_root, run_name, steps, *extra, manifest="m.jsonl"):
        return ["--train-manifest", str(root / manifest),
                "--train-units", str(root / "units"), "--exclude-targets", "",
                "--upsample-initial-channel", "32", "--segment-samples", "1280",
                "--batch-size", "2", "--workers", "0", "--device", "cpu",
                "--steps", str(steps), "--log-every", "1", "--run-name", run_name,
                "--runs-dir", str(out_root / "runs"), "--ckpt-dir", str(out_root / "ckpt"),
                *extra]

    with _few_torch_threads():
        assert tuv.main(argv(root, "base", 2, manifest="m_base.jsonl")) == 0
    base = root / "ckpt" / "base" / "final.pt"
    finetune = ["--finetune-from", str(base), "--new-speaker-init", "lj",
                "--mix-speakers", "me", "--mix-ratio", "0.4",
                "--mix-public-lang-ratios", "en,es", "--ckpt-steps", "1"]
    return types.SimpleNamespace(argv=argv, base=base, finetune=finetune)


def _spy_checkpoints(monkeypatch):
    """{step: snapshot} of the generator at every checkpoint call. In-loop
    saves (--ckpt-steps 1) are only recorded, not written (a checkpoint with
    the discriminator and both optimizers is ~130 MB); the end-of-run saves
    go to disk so --resume works."""
    snaps = {}
    real = tuv.save_checkpoint

    def spy(path, gen, disc, opt_g, opt_d, sched_g, sched_d, step, args, *rest, **kw):
        snaps[step] = {"unit_emb": gen.unit_emb.weight.detach().clone(),
                       "post": gen.post.weight.detach().clone(),
                       "requires_grad": gen.unit_emb.weight.requires_grad}
        if step < args.steps:
            return None
        return real(path, gen, disc, opt_g, opt_d, sched_g, sched_d, step, args, *rest, **kw)

    monkeypatch.setattr(tuv, "save_checkpoint", spy)
    return snaps


def test_freeze_unit_emb_after_half_resume_and_records(mini_ml, tmp_path, monkeypatch):
    argv, ft = mini_ml.argv, mini_ml.finetune
    base = torch.load(mini_ml.base, weights_only=True)["model"]
    snaps = _spy_checkpoints(monkeypatch)
    # half of 4 steps: unit_emb trains in steps 1-2 and is frozen before step 3.
    assert tuv.main(argv(tmp_path, "frz", 4, *ft, "--freeze-unit-emb-after", "half")) == 0
    assert sorted(snaps) == [1, 2, 3, 4]
    assert not torch.equal(snaps[1]["unit_emb"], base["unit_emb.weight"])
    assert not torch.equal(snaps[2]["unit_emb"], snaps[1]["unit_emb"])
    assert torch.equal(snaps[3]["unit_emb"], snaps[2]["unit_emb"])      # bit-identical
    assert torch.equal(snaps[4]["unit_emb"], snaps[2]["unit_emb"])      # (weight decay too)
    assert not torch.equal(snaps[4]["post"], snaps[2]["post"])          # the rest still trains
    assert not torch.equal(snaps[4]["post"], snaps[3]["post"])
    assert snaps[2]["requires_grad"] and not snaps[3]["requires_grad"]
    ck = torch.load(tmp_path / "ckpt" / "frz" / "final.pt", weights_only=True)
    assert ck["freeze_unit_emb_after"] == 2 and ck["unit_emb_frozen_at"] == 2
    assert json.loads(ck["mix_public_lang_ratios"]) == {"en": 0.5, "es": 0.5}
    assert ck["speakers"] == ["lj", "css10_es", "me"]
    assert torch.equal(ck["model"]["unit_emb.weight"], snaps[2]["unit_emb"])

    # --resume without the flag re-applies the recorded switch.
    snaps.clear()
    assert tuv.main(argv(tmp_path, "frz", 6, *ft, "--resume", "auto")) == 0
    assert sorted(snaps) == [5, 6]
    assert not snaps[5]["requires_grad"]
    assert torch.equal(snaps[6]["unit_emb"], ck["model"]["unit_emb.weight"])
    assert not torch.equal(snaps[6]["post"], ck["model"]["post.weight"])
    ck2 = torch.load(tmp_path / "ckpt" / "frz" / "final.pt", weights_only=True)
    assert ck2["step"] == 6 and ck2["unit_emb_frozen_at"] == 2
    assert ck2["freeze_unit_emb_after"] == 2


def test_freeze_flag_absent_or_never_reached_changes_nothing(mini_ml, tmp_path, monkeypatch):
    argv, ft = mini_ml.argv, mini_ml.finetune
    snaps = _spy_checkpoints(monkeypatch)
    assert tuv.main(argv(tmp_path, "nofrz", 3, *ft)) == 0
    assert not torch.equal(snaps[2]["unit_emb"], snaps[1]["unit_emb"])
    assert not torch.equal(snaps[3]["unit_emb"], snaps[2]["unit_emb"])   # trains every step
    assert all(s["requires_grad"] for s in snaps.values())
    ck = torch.load(tmp_path / "ckpt" / "nofrz" / "final.pt", weights_only=True)
    assert ck["freeze_unit_emb_after"] is None and ck["unit_emb_frozen_at"] is None
    # A switch beyond --steps never fires: bit-identical to the flag-less run.
    assert tuv.main(argv(tmp_path, "late", 3, *ft, "--freeze-unit-emb-after", "100")) == 0
    ck_late = torch.load(tmp_path / "ckpt" / "late" / "final.pt", weights_only=True)
    assert ck_late["freeze_unit_emb_after"] == 100 and ck_late["unit_emb_frozen_at"] is None
    for k, v in ck["model"].items():
        assert torch.equal(v, ck_late["model"][k]), k
    # Flag validation fails fast.
    assert tuv.main(argv(tmp_path, "bad1", 1, *ft, "--freeze-unit-emb-after", "soon")) == 1
    assert tuv.main(argv(tmp_path, "bad2", 1, "--mix-public-lang-ratios", "en,es")) == 1
    assert tuv.main(argv(tmp_path, "bad3", 1, *ft, "--lang-ratios", "en,es")) == 1   # exclusive


# -- eval_controls --s2ut (Task 5) ----------------------------------------------------
def _s2ut_controls_list(tmp_path):
    rows = []
    for i, (src, tgt) in enumerate((("en", "es"), ("en", "es"), ("es", "en"))):
        p = tmp_path / f"src{i}.wav"
        audio.save(p, _sine(2.0, 140.0 + 20 * i))
        rows.append({"utt_id": f"u{i}", "src_lang": src, "tgt_lang": tgt, "speaker": "s",
                     "gender": "F", "src_duration": 2.0, "tgt_duration": 2.2, "gamma": 1.1,
                     "src_path": str(p), "tgt_path": "", "src_text": "the house is big",
                     "tgt_text": "la casa es grande" if tgt == "es" else "the house is big"})
    tsv = tmp_path / "s2ut.tsv"
    evallist.write(tsv, rows, evallist.S2UT_COLUMNS)
    return tsv


def _controls_args(tmp_path, tsv, **kw):
    base = dict(bundle=str(tmp_path / "bundle"), voice="lj", voice_b="hifitts_92",
                list=str(tsv), targets=str(tmp_path / "targets.tsv"),
                presets_dir=str(REPO / "presets"), limit=None, out=str(tmp_path / "out"),
                checks="pitch,intonation,click,blend,presets", skip_ml=False, device="cpu",
                s2ut=True, src_lang="en", tgt_lang="es")
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_eval_controls_s2ut_runs_every_check_on_translated_output(tmp_path, monkeypatch):
    tsv = _s2ut_controls_list(tmp_path)
    monkeypatch.setattr(s2utfile, "find_s2ut_file", lambda *a, **k: tmp_path / "fv_s2ut_file")
    converted, one_calls, evals = [], [], []

    def runner_convert(dest, extra_args):
        """Translated output = the source duration + a 0.5 s flush tail; the
        fake 'tool' honours pitch_shift and intonation like the real chain."""
        dest.mkdir(parents=True, exist_ok=True)
        extra = list(extra_args)
        params = dict(extra[i + 1].split("=", 1) for i, a in enumerate(extra) if a == "--param")
        shift = float(params.get("pitch_shift", 0.0))
        scale = float(params.get("intonation", 1.0))
        rows = evallist.read_sources(tmp_path / "out" / "s2ut_sources.tsv")
        for r in rows:
            audio.save(dest / f"{r['utt_id']}.wav",
                       _fm(r["duration"] + 0.5, 150.0 * 2 ** (shift / 12.0), 1.0 * scale))
        converted.append((dest.name, tuple(extra)))
        return dest

    def runner_one(bundle, in_wav, out_wav, src, tgt, voice=None, extra_args=(), s2ut_file=None):
        one_calls.append((src, tgt, voice, tuple(extra_args), s2ut_file))
        audio.save(out_wav, np.zeros(int(2.5 * SR), np.float32))

    def evaluate_fn(src_list, cdir, targets, voice, metrics, device, language="en"):
        name = pathlib.Path(cdir).name
        evals.append((name, language, tuple(metrics), pathlib.Path(src_list)))
        if name.startswith("blend_"):
            return {"tgt_cos": 0.8 - 0.3 * float(name.split("_")[1]), "utmos": 3.4}
        return {"wer": 0.10}

    args = _controls_args(tmp_path, tsv)
    results = ec.run_checks(args, runner_convert=runner_convert, runner_one=runner_one,
                            evaluate_fn=evaluate_fn, log=lambda *a: None)
    assert {k: v["status"] for k, v in results.items()} == {
        "pitch": "pass", "intonation": "pass", "click": "pass", "blend": "pass",
        "presets": "pass"}
    assert results["pitch"]["measured_st"]["+7"] == pytest.approx(7.0, abs=0.05)
    # Only the en->es rows of the list, with their Spanish references.
    sources = evallist.read_sources(tmp_path / "out" / "s2ut_sources.tsv")
    assert [r["utt_id"] for r in sources] == ["u0", "u1"]
    assert all(r["text"] == "la casa es grande" for r in sources)
    assert converted[0] == ("baseline", ())
    assert ("pitch_p7", ("--param", "pitch_shift=7")) in converted
    assert ("blend_0.5", ("--voice-b", "hifitts_92", "--param", "timbre_blend=0.5")) in converted
    assert ("preset_deep", ("--preset", "deep")) in converted
    # The click check drives fv_s2ut_file once per (param, utterance) at T = 50 %.
    assert len(one_calls) == len(ec.CLICK_PARAMS) * 2
    assert all(c[:3] == ("en", "es", "lj") and c[4] == tmp_path / "fv_s2ut_file"
               for c in one_calls)
    assert ("--change", "1.000:pitch_shift=+7") in [c[3] for c in one_calls]
    # Scoring runs in the target language on the derived sources list.
    assert evals and all(lang == "es" and src.name == "s2ut_sources.tsv"
                         for _, lang, _, src in evals)
    assert {m for _, _, m, _ in evals} == {("spk", "utmos"), ("wer",)}


def test_eval_controls_s2ut_cli_mode_a_list_and_arg_errors(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    rows = []
    for i in range(2):
        p = src_dir / f"u{i}.wav"
        sf.write(p, _sine(2.0, 150.0 + 30 * i), SR)
        rows.append({"utt_id": f"u{i}", "speaker": "libri", "gender": "U",
                     "duration": 2.0, "path": str(p), "text": "hello there"})
    tsv = tmp_path / "sources.tsv"
    evallist.write(tsv, rows, evallist.SOURCE_COLUMNS)
    monkeypatch.setattr(s2utfile, "find_s2ut_file", lambda *a, **k: tmp_path / "fv_s2ut_file")
    calls = []

    def fake_convert_list(bundle, rows_, out_dir, src=None, tgt=None, voice=None,
                          extra_args=(), **kw):
        calls.append((src, tgt, voice, tuple(extra_args)))
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in rows_:
            sf.write(out_dir / f"{r['utt_id']}.wav", _sine(float(r["duration"]) + 0.5), SR)
        return [(r["utt_id"], 0.0) for r in rows_]

    def fake_convert(bundle, in_wav, out_wav, src, tgt, voice=None, extra_args=(), **kw):
        calls.append((src, tgt, voice, tuple(extra_args)))
        sf.write(out_wav, np.zeros(int(2.5 * SR), np.float32), SR)

    monkeypatch.setattr(s2utfile, "convert_list", fake_convert_list)
    monkeypatch.setattr(s2utfile, "convert", fake_convert)
    out = tmp_path / "out"
    common = ["--bundle", str(tmp_path / "bundle"), "--voice", "lj", "--list", str(tsv),
              "--out", str(out), "--device", "cpu"]
    assert ec.main(common + ["--s2ut", "--src-lang", "en", "--tgt-lang", "de",
                             "--checks", "click,presets"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["mode"] == "s2ut en->de" and report["verdict"] == "PASS"
    assert report["checks"]["click"]["status"] == "pass"
    # A Mode A list has no German references: presets cannot be scored.
    assert report["checks"]["presets"]["status"] == "skipped"
    assert "Mode B" in (out / "report.md").read_text()
    assert calls and all(c[:3] == ("en", "de", "lj") for c in calls)
    # Argument errors exit 1 before any conversion.
    n = len(calls)
    assert ec.main(common + ["--s2ut", "--src-lang", "en"]) == 1
    assert ec.main(common + ["--s2ut", "--src-lang", "en", "--tgt-lang", "en"]) == 1
    assert ec.main(common + ["--src-lang", "en", "--tgt-lang", "es"]) == 1     # needs --s2ut
    assert len(calls) == n
    # An S2UT list without the requested direction is an error too.
    assert ec.main(common[:4] + ["--list", str(_s2ut_controls_list(tmp_path)), "--out", str(out),
                                 "--s2ut", "--src-lang", "fr", "--tgt-lang", "en"]) == 1


# -- eval_cross_lingual_voice (Task 6) -----------------------------------------------
def _summary(bleu=None, cos=None, utmos=None, n=200, metrics=("bleu", "spk", "utmos"), voice="v"):
    return {"asr_bleu": bleu, "tgt_cos": cos, "utmos": utmos, "n": n, "metrics": list(metrics),
            "voice": voice}


def _passing_results():
    dirs = {}
    for d in ecl.DIRECTIONS:
        tgt = d.split("_")[1]
        dirs[d] = {"me": _summary(30.0, 0.62, 3.6, voice="me"),
                   "native": _summary(31.0, 0.70, 3.9, voice=f"nat_{tgt}"),
                   "zero_shot": _summary(None, 0.50, 3.5, metrics=("spk", "utmos"))}
    public = {"nat_es": {"en_es": {"before": _summary(31.2, 0.71), "after": _summary(31.0, 0.70)}},
              "nat_en": {d: {"before": _summary(28.0, 0.70), "after": _summary(28.0, 0.69)}
                         for d in ("es_en", "fr_en", "de_en")}}
    return {"directions": dirs, "public": public, "me_license": ecl.PRIVATE_LICENSE}


def _statuses(checks):
    return {c["check"]: c["status"] for c in checks}


def test_cross_lingual_dod_all_pass_and_boundaries():
    res = _passing_results()
    checks = ecl.cross_lingual_dod(res)
    st = _statuses(checks)
    assert all(s == "pass" for s in st.values()), st
    assert ecl.verdict(checks) == "PASS"
    for d in ecl.EN_TO_X:
        assert {f"{d}_bleu", f"{d}_cos_gain", f"{d}_cos_vs_{d[3:]}_en"} <= set(st)
    assert {"utmos_en", "utmos_es", "utmos_fr", "utmos_de", "coverage",
            "me_voice_private"} <= set(st)
    assert {"public_nat_es_en_es_cos", "public_nat_es_en_es_bleu",
            "public_nat_en_fr_en_cos"} <= set(st)
    # Inclusive thresholds survive float rounding.
    r = res["directions"]["en_de"]
    r["me"]["asr_bleu"], r["native"]["asr_bleu"] = 29.5, 31.0                 # exactly -1.5
    r["me"]["tgt_cos"], r["zero_shot"]["tgt_cos"] = 0.60, 0.55                # exactly +0.05
    res["directions"]["de_en"]["me"]["tgt_cos"] = 0.65                        # |gap| = 0.05
    res["directions"]["en_fr"]["me"]["utmos"] = 3.0
    res["public"]["nat_es"]["en_es"] = {"before": _summary(31.5, 0.72),
                                        "after": _summary(31.0, 0.70)}        # 0.5 / 0.02
    assert ecl.verdict(ecl.cross_lingual_dod(res)) == "PASS"


@pytest.mark.parametrize("mutation, failing", [
    ("bleu", {"en_es_bleu"}),
    ("cos_gain", {"en_fr_cos_gain"}),
    ("cos_same_lang", {"en_de_cos_vs_de_en"}),
    ("utmos", {"utmos_fr"}),
    ("utmos_en_weighted", {"utmos_en"}),
    ("silent_me", {"en_es_cos_gain", "en_es_cos_vs_es_en", "utmos_es"}),
    ("public_cos", {"public_nat_es_en_es_cos"}),
    ("public_bleu", {"public_nat_en_fr_en_bleu"}),
    ("not_private", {"me_voice_private"}),
])
def test_cross_lingual_dod_each_verdict_fails_alone(mutation, failing):
    res = _passing_results()
    dirs = res["directions"]
    if mutation == "bleu":
        dirs["en_es"]["me"]["asr_bleu"] = 29.4                        # native 31.0: -1.6
    elif mutation == "cos_gain":
        dirs["en_fr"]["me"]["tgt_cos"] = 0.54                         # zero-shot 0.50: +0.04
        dirs["fr_en"]["me"]["tgt_cos"] = 0.54                         # keep the gap check green
    elif mutation == "cos_same_lang":
        dirs["de_en"]["me"]["tgt_cos"] = 0.70                         # 0.62 vs 0.70: 0.08
    elif mutation == "utmos":
        dirs["en_fr"]["me"]["utmos"] = 2.9
    elif mutation == "utmos_en_weighted":
        # en is the utterance-weighted mean over es/fr/de->en: 2.0, 3.5, 3.5 -> 3.0 - eps
        dirs["es_en"]["me"]["utmos"], dirs["es_en"]["me"]["n"] = 2.0, 200
        dirs["fr_en"]["me"]["utmos"], dirs["de_en"]["me"]["utmos"] = 3.49, 3.5
    elif mutation == "silent_me":
        dirs["en_es"]["me"].update(tgt_cos=None, utmos=None)          # scored, but nothing to score
    elif mutation == "public_cos":
        res["public"]["nat_es"]["en_es"]["after"]["tgt_cos"] = 0.68   # 0.71 -> 0.68
    elif mutation == "public_bleu":
        res["public"]["nat_en"]["fr_en"]["after"]["asr_bleu"] = 27.4  # 28.0 -> 27.4
    elif mutation == "not_private":
        res["me_license"] = "CC BY 4.0"
    checks = ecl.cross_lingual_dod(res)
    assert {c["check"] for c in checks if c["status"] == "fail"} == failing
    assert ecl.verdict(checks) == "FAIL"


def test_cross_lingual_dod_skips_are_incomplete_not_pass():
    res = _passing_results()
    del res["directions"]["en_fr"]
    del res["directions"]["de_en"]
    res["me_license"] = None
    res["public"]["nat_it"] = {}                                     # no direction into 'it'
    for r in res["directions"].values():
        r["me"]["asr_bleu"] = None                                   # no tgt_text references
    st = _statuses(ecl.cross_lingual_dod(res))
    assert st["coverage"] == "skipped" and st["me_voice_private"] == "skipped"
    assert st["public_nat_it"] == "skipped"
    assert st["en_es_bleu"] == "skipped" and st["en_de_cos_vs_de_en"] == "skipped"
    assert "en_fr_bleu" not in st
    assert not any(s == "fail" for s in st.values())
    assert ecl.verdict(ecl.cross_lingual_dod(res)) == "INCOMPLETE"
    # Nothing scored at all (--skip-ml): every quality check is skipped.
    bare = {"directions": {d: {k: {"n": 2, "metrics": []} for k in ("me", "native", "zero_shot")}
                           for d in ecl.DIRECTIONS},
            "public": {}, "me_license": ecl.PRIVATE_LICENSE}
    st = _statuses(ecl.cross_lingual_dod(bare))
    assert {k for k, s in st.items() if s != "skipped"} == {"coverage", "me_voice_private"}


VOICE_HZ = {("after", "me"): 300.0, ("zeroshot", "me"): 200.0, ("after", "lj"): 400.0,
            ("before", "lj"): 400.0, ("after", "css10_es"): 450.0, ("before", "css10_es"): 450.0,
            ("after", "css10_fr"): 500.0, ("before", "css10_fr"): 500.0,
            ("after", "css10_de"): 550.0, ("before", "css10_de"): 550.0}
REF_TEXT = {"en": "the house is big", "es": "la casa es grande",
            "fr": "la maison est grande", "de": "das haus ist groß"}


def _freq(wave, sr=SR):
    zc = np.count_nonzero(np.diff(np.signbit(np.asarray(wave))))
    return zc * sr / (2.0 * len(wave))


def _fake_models(monkeypatch):
    class FakeWhisper:
        def __init__(self, device="cpu", batch_size=8):
            pass

        def transcribe(self, waves, language="en", silence_dbfs=None):
            return [REF_TEXT[language] for _ in waves]

        def normalize(self, text, language="en"):
            return asr.normalize_multilingual(text)

        def close(self):
            pass

    class FakeEcapa:
        """Embedding angle follows the dominant frequency: 100 Hz = 0.5 rad."""

        def __init__(self, device="cpu", batch_size=8):
            pass

        def embed(self, waves):
            th = np.array([(_freq(w) - 300.0) / 100.0 * 0.5 for w in waves])
            return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32)

        def close(self):
            pass

    class FakeUtmos:
        def __init__(self, device="cpu"):
            pass

        def score(self, waves):
            return [3.8 for _ in waves]

        def close(self):
            pass

    monkeypatch.setattr(s2ut_eval.asr, "Whisper", FakeWhisper)
    monkeypatch.setattr(s2ut_eval.speaker, "Ecapa", FakeEcapa)
    monkeypatch.setattr(s2ut_eval.mos, "Utmos", FakeUtmos)


def _write_bundle(path, voices):
    path.mkdir(parents=True, exist_ok=True)
    (path / "voices.json").write_text(json.dumps({"schema": 1, "voices": voices}))


def _cross_lingual_rig(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    lists = tmp_path / "lists"
    lists.mkdir()
    for d in ecl.DIRECTIONS:
        src, tgt = d.split("_")
        rows = []
        for i in range(2):
            rel = f"src/{d}_{i}.wav"
            (tmp_path / "src").mkdir(exist_ok=True)
            audio.save(tmp_path / rel, _sine(1.0, 150.0 + 10 * i))
            rows.append({"utt_id": f"{d}_{i}", "src_lang": src, "tgt_lang": tgt, "speaker": "s",
                         "gender": "F", "src_duration": 1.0, "tgt_duration": 1.2, "gamma": 1.2,
                         "src_path": rel, "tgt_path": "", "src_text": REF_TEXT[src],
                         "tgt_text": REF_TEXT[tgt]})
        evallist.write(lists / f"test_{d}_200.tsv", rows, evallist.S2UT_COLUMNS)
    (tmp_path / "refs").mkdir()
    me_rows, pub_rows = [], []
    for i in range(2):
        audio.save(tmp_path / f"refs/me_{i}.wav", _sine(2.0, 300.0))
        me_rows.append({"voice": "me", "clip_id": f"m{i}", "speaker": "me", "gender": "U",
                        "duration": 2.0, "path": f"refs/me_{i}.wav", "dataset": "own_voice"})
    for voice, hz in (("lj", 400.0), ("css10_es", 450.0), ("css10_fr", 500.0),
                      ("css10_de", 550.0)):
        audio.save(tmp_path / f"refs/{voice}.wav", _sine(2.0, hz))
        pub_rows.append({"voice": voice, "clip_id": f"{voice}_0", "speaker": voice,
                         "gender": "F", "duration": 2.0, "path": f"refs/{voice}.wav",
                         "dataset": "fake"})
    evallist.write(tmp_path / "me_targets.tsv", me_rows, evallist.TARGET_COLUMNS)
    evallist.write(tmp_path / "targets.tsv", pub_rows, evallist.TARGET_COLUMNS)

    def voice(name, lang, lic="CC BY 4.0"):
        return {"name": name, "display": name, "embedding": [0.0] * 4, "logf0_mean": 5.0,
                "logf0_std": 0.2, "energy_mean": -4.0, "energy_std": 1.0, "license": lic,
                "source": "fake", "lang": lang}

    public = [voice("lj", "en"), voice("css10_es", "es"), voice("css10_fr", "fr"),
              voice("css10_de", "de")]
    _write_bundle(tmp_path / "after", public + [voice("me", "en", ecl.PRIVATE_LICENSE)])
    _write_bundle(tmp_path / "zeroshot", public + [voice("me", "en", ecl.PRIVATE_LICENSE)])
    _write_bundle(tmp_path / "before", public)
    return ["--bundle", str(tmp_path / "after"), "--zero-shot-bundle", str(tmp_path / "zeroshot"),
            "--me-targets", str(tmp_path / "me_targets.tsv"),
            "--native-voices", "en=lj,es=css10_es,fr=css10_fr,de=css10_de",
            "--lists-dir", str(lists), "--targets", str(tmp_path / "targets.tsv"),
            "--device", "cpu", "--out", str(tmp_path / "out")]


def test_eval_cross_lingual_voice_end_to_end_with_fakes(tmp_path, monkeypatch):
    argv = _cross_lingual_rig(tmp_path, monkeypatch)
    _fake_models(monkeypatch)
    calls = []
    hz = dict(VOICE_HZ)

    def fake_s2ut(bundle, in_wav, out_wav, src, tgt, voice=None, extra_args=(), trace=None,
                  s2ut_file=None):
        """Deterministic 'translation': a tone whose pitch identifies the
        (bundle, voice) that produced it."""
        calls.append((pathlib.Path(bundle).name, voice, src, tgt, tuple(extra_args)))
        pathlib.Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        audio.save(out_wav, _sine(1.2, hz[(pathlib.Path(bundle).name, voice)]))
        return 0.1

    monkeypatch.setattr(s2utfile, "convert", fake_s2ut)
    monkeypatch.setattr(s2utfile, "find_s2ut_file", lambda *a, **k: tmp_path / "fv_s2ut_file")
    argv += ["--before-bundle", str(tmp_path / "before"), "--wait-k", "4"]
    assert ecl.main(argv) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["verdict"] == "PASS", [c for c in summary["checks"] if c["status"] != "pass"]
    assert summary["private"] is True and summary["me_lang"] == "en"
    en_es = summary["directions"]["en_es"]
    assert en_es["me"]["tgt_cos"] == pytest.approx(1.0, abs=1e-5)
    assert en_es["zero_shot"]["tgt_cos"] == pytest.approx(math.cos(0.5), abs=0.02)
    assert en_es["me"]["asr_bleu"] == pytest.approx(100.0) == en_es["native"]["asr_bleu"]
    assert summary["utmos_by_lang"]["en"]["directions"] == ["de_en", "es_en", "fr_en"]
    # 6 directions x (me, native, zero-shot) + the native voices on the before-bundle.
    configs = {(b, v, s, t) for b, v, s, t, _ in calls}
    for d in ecl.DIRECTIONS:
        s, t = d.split("_")
        nat = {"en": "lj", "es": "css10_es", "fr": "css10_fr", "de": "css10_de"}[t]
        assert {("after", "me", s, t), ("after", nat, s, t), ("zeroshot", "me", s, t),
                ("before", nat, s, t)} <= configs
    assert len(calls) == 6 * 4 * 2 and all(c[4] == ("--wait-k", "4") for c in calls)
    assert set(summary["public"]) == {"lj", "css10_es", "css10_fr", "css10_de"}
    assert set(summary["public"]["lj"]) == {"es_en", "fr_en", "de_en"}
    md = (tmp_path / "out" / "summary.md").read_text()
    assert "PRIVATE" in md and "**Verdict: PASS**" in md
    assert (tmp_path / "out" / "en_de" / "me" / "eval" / "summary.json").is_file()

    # A zero-shot baseline as close as the fine-tuned voice fails the cosine gain.
    hz[("zeroshot", "me")] = 300.0
    assert ecl.main(argv[:-4] + ["--out", str(tmp_path / "out2")]) == 1
    s2 = json.loads((tmp_path / "out2" / "summary.json").read_text())
    assert s2["verdict"] == "FAIL"
    assert {c["check"] for c in s2["checks"] if c["status"] == "fail"} == {
        "en_es_cos_gain", "en_fr_cos_gain", "en_de_cos_gain"}


def test_eval_cross_lingual_voice_skip_ml_and_input_errors(tmp_path, monkeypatch):
    argv = _cross_lingual_rig(tmp_path, monkeypatch)
    n_calls = []

    def runner(bundle, in_wav, out_wav, src, tgt, voice=None, extra_args=(), trace=None,
               s2ut_file=None):
        n_calls.append(voice)
        audio.save(out_wav, _sine(1.0))
        return 0.0

    def no_score(*a, **k):
        raise AssertionError("--skip-ml must not score")

    args = ecl.parse_args(argv + ["--skip-ml", "--directions", "en_es,es_en"])
    results, checks = ecl.run(args, runner=runner, score_fn=no_score, log=lambda *a: None)
    assert len(n_calls) == 2 * 3 * 2
    st = _statuses(checks)
    assert st["coverage"] == "skipped" and st["me_voice_private"] == "pass"
    assert ecl.verdict(checks) == "INCOMPLETE"
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["verdict"] == "INCOMPLETE"
    assert (tmp_path / "out" / "en_es" / "list.tsv").is_file()

    def fails(extra, match):
        with pytest.raises((ValueError, FileNotFoundError), match=match):
            ecl.run(ecl.parse_args(argv + extra), runner=runner, score_fn=no_score,
                    log=lambda *a: None)

    fails(["--native-voices", "en=lj,es=css10_es"], "no voice for target language")
    fails(["--native-voices", "en=lj,es=nobody,fr=css10_fr,de=css10_de"], "not in")
    fails(["--voice", "you"], "no reference clips")
    fails(["--zero-shot-voice", "nobody"], "--zero-shot-voice 'nobody'")
    fails(["--directions", "en_it"], "unknown direction")
    fails(["--list-pattern", "missing_{direction}.tsv"], "no en->|no es->|no fr->|no de->")
    sil = tmp_path / "lists" / "silence.tsv"
    evallist.write(sil, [{"utt_id": "n1", "kind": "noise_only", "src_lang": "en",
                          "tgt_lang": "es", "src_duration": 2.0, "speech_end_s": 0.0,
                          "src_path": "src/en_es_0.wav", "src_text": "", "tgt_text": ""}],
                   evallist.SILENCE_COLUMNS)
    fails(["--list-pattern", "silence.tsv", "--directions", "en_es"], "not a paired S2UT list")
    assert ecl.main(argv + ["--metrics", "bleu,wer"]) == 1


# -- live-mic prompts + prepare_live_mic_set (Task 7) ---------------------------------
def test_live_mic_prompts_are_committed_parallel_and_clean():
    texts = {}
    for lang in plm.LANGS:
        prompts = plm.read_prompts(plm.paths.EVAL_LISTS / "s2ut", lang)
        assert list(prompts) == [f"{lang}_{i:02d}" for i in range(1, 21)]
        assert len(set(prompts.values())) == 20
        for pid, text in prompts.items():
            words = asr.normalize_multilingual(text).split()
            assert 4 <= len(words) <= 16, (pid, text)
            assert not any(ch.isdigit() for ch in text), (pid, text)   # no digit/word ambiguity
        texts[lang] = prompts
    # Parallel sets: --references prompts finds a translation for every prompt.
    refs = plm.read_references("prompts", plm.paths.EVAL_LISTS / "s2ut", texts)
    assert len(refs) == 6 * 20
    assert refs[("en_06", "de")] == texts["de"]["de_06"]
    assert refs[("fr_19", "en")] == texts["en"]["en_19"]


def _recording(seconds=3.0, sr=48000, amp=0.2, noise=0.001, seed=0, channels=1):
    """Speech-like burst between 0.5 s of quiet head and tail."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = noise * rng.standard_normal(n)
    a, b = int(min(0.5, seconds / 4) * sr), n - int(min(0.5, seconds / 4) * sr)
    t = np.arange(b - a) / sr
    x[a:b] += amp * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = x.astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def _record_set(root, langs=("en", "es"), sr=48000):
    root.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        for i in range(1, 21):
            sf.write(root / f"{lang}_{i:02d}.wav", _recording(seed=i, channels=2 if i == 2 else 1),
                     sr, subtype="PCM_24")


def test_prepare_live_mic_set_writes_lists_and_clips(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    _record_set(tmp_path / "rec")
    sf.write(tmp_path / "rec" / "de_01.wav", _recording(), 48000)   # de not selected: ignored
    out = tmp_path / "private" / "live_mic"
    assert plm.main(["--input", str(tmp_path / "rec"), "--langs", "en,es",
                     "--references", "prompts", "--out", str(out)]) == 0
    names = sorted(p.name for p in out.glob("live_mic_*.tsv"))
    assert names == ["live_mic_en_de.tsv", "live_mic_en_es.tsv", "live_mic_en_fr.tsv",
                     "live_mic_es_en.tsv"]
    rows = evallist.read_s2ut(out / "live_mic_en_es.tsv")
    assert len(rows) == 20 and rows[0]["utt_id"] == "live_en_01"
    assert rows[0]["src_text"] == "the kitchen window was open all night"
    assert rows[0]["tgt_text"] == "la ventana de la cocina estuvo abierta toda la noche"
    assert rows[0]["src_path"] == "private/live_mic/clips/en_01.wav"    # relative to the data root
    es_en = evallist.read_s2ut(out / "live_mic_es_en.tsv")
    assert es_en[5]["tgt_text"] == "i would like a cup of tea with a little milk"
    # The Stage 10/11 driver reads them as paired lists (--list-pattern live_mic_{direction}.tsv).
    assert len(ecl.direction_rows(out / "live_mic_en_fr.tsv", "en_fr", None)) == 20
    for r in rows[:3]:
        info = sf.info(str(tmp_path / r["src_path"]))
        assert info.samplerate == SR and info.channels == 1 and info.format == "WAV"
        assert r["src_duration"] == pytest.approx(3.0, abs=0.01)
    report = json.loads((out / "report.json").read_text())
    assert report["ok"] and report["passed"] == 40
    assert report["set_checks"]["ignored"] == ["de_01"]
    assert report["recordings"]["en_02"]["channels"] == 2
    assert report["recordings"]["en_01"]["snr_db"] > 30
    # Without --references the lists carry no tgt_text (UTMOS-only scoring).
    out2 = tmp_path / "out2"
    assert plm.main(["--input", str(tmp_path / "rec"), "--langs", "es", "--out", str(out2)]) == 0
    assert all(r["tgt_text"] == "" for r in evallist.read_s2ut(out2 / "live_mic_es_en.tsv"))
    assert not (out2 / "live_mic_en_es.tsv").exists()


def test_prepare_live_mic_set_flags_every_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    rec = tmp_path / "rec"
    _record_set(rec, langs=("en",))
    sr = 48000
    (rec / "en_05.wav").unlink()                                              # missing
    sf.write(rec / "en_21.wav", _recording(), sr)                             # extra: no prompt
    sf.write(rec / "take_final.wav", _recording(), sr)                        # extra: bad name
    sf.write(rec / "en_03.flac", _recording(), sr)                            # duplicate
    clipped = _recording()
    clipped[sr] = 0.995                                                       # -0.04 dBFS
    sf.write(rec / "en_06.wav", clipped, sr, subtype="FLOAT")
    sf.write(rec / "en_07.wav", _recording(seconds=0.6), sr)                  # too short
    sf.write(rec / "en_08.wav", _recording(seconds=13.0), sr)                 # too long
    sf.write(rec / "en_09.wav", _recording(amp=0.002, noise=0.00002), sr)     # too quiet
    sf.write(rec / "en_10.wav", _recording(amp=0.05, noise=0.03), sr)         # noisy
    sf.write(rec / "en_11.wav", _recording(sr=8000), 8000)                    # below 16 kHz
    stereo = _recording(channels=2)
    stereo[:, 1] *= -1.0                                                      # phase-cancelling
    sf.write(rec / "en_12.wav", stereo, sr)
    (rec / "en_13.wav").write_bytes(b"RIFF....not a wav")                     # unreadable
    out = tmp_path / "out"
    assert plm.main(["--input", str(rec), "--langs", "en", "--out", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    sc = report["set_checks"]
    assert not sc["missing"]["pass"] and "en_05" in sc["missing"]["detail"]
    assert not sc["extra"]["pass"] and "en_21" in sc["extra"]["detail"] \
        and "take_final" in sc["extra"]["detail"]
    assert not sc["duplicates"]["pass"] and "en_03" in sc["duplicates"]["detail"]
    failed = {pid: sorted(n for n, c in r["checks"].items() if not c["pass"])
              for pid, r in report["recordings"].items() if not r["ok"]}
    # The phase-cancelled downmix is also silent, so its level check fails too.
    assert failed == {"en_06": ["clipping"], "en_07": ["duration"], "en_08": ["duration"],
                      "en_09": ["level"], "en_10": ["level"], "en_11": ["rate"],
                      "en_12": ["level", "mono"], "en_13": ["readable"]}
    rows = evallist.read_s2ut(out / "live_mic_en_es.tsv")
    kept = {r["utt_id"] for r in rows}
    assert len(rows) == 20 - 1 - len(failed)                                  # only passing clips
    assert "live_en_06" not in kept and "live_en_01" in kept
    assert "| en_06 |" in (out / "report.md").read_text()
    # A references TSV that misses a recorded prompt fails the references check.
    refs = tmp_path / "refs.tsv"
    evallist.write(refs, [{"prompt_id": "en_01", "tgt_lang": "es", "text": "hola"}],
                   plm.REFERENCE_COLUMNS)
    good = tmp_path / "good"
    _record_set(good, langs=("en",))
    assert plm.main(["--input", str(good), "--langs", "en", "--references", str(refs),
                     "--out", str(tmp_path / "out3")]) == 1
    sc3 = json.loads((tmp_path / "out3" / "report.json").read_text())["set_checks"]
    assert not sc3["references"]["pass"]
    first = evallist.read_s2ut(tmp_path / "out3" / "live_mic_en_es.tsv")[0]
    assert first["tgt_text"] == "hola"
    # Manifest input (paths relative to the manifest) works like --input.
    manifest = good / "manifest.tsv"
    evallist.write(manifest, [{"prompt_id": f"en_{i:02d}", "path": f"en_{i:02d}.wav"}
                              for i in range(1, 21)], ("prompt_id", "path"))
    assert plm.main(["--manifest", str(manifest), "--langs", "en",
                     "--out", str(tmp_path / "out4")]) == 0
    with pytest.raises(SystemExit):
        plm.main(["--langs", "en"])                                            # no input
    assert plm.main(["--input", str(good), "--langs", "it"]) == 1
