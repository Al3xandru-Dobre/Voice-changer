"""Content encoder candidates (Milestone 2 / Stage 04): the causal student, its
delay heads and loss alignment, unit-stream metadata, the unit probes, k-means,
augmentation, a tiny train/resume/extract loop and (when the weights are cached)
the HuBERT layer-7 indexing."""
import json
import math

import numpy as np
import pytest
import torch

from fvml import augment, teacher, unit_probes, units
from fvml import content_encoder as ce

FS = ce.FRAME_SAMPLES


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ce.ContentEncoder().eval()


# -- student -------------------------------------------------------------------
def test_size_and_shapes(model):
    assert model.num_params() <= 5_000_000
    assert model.cfg.lookahead == ce.LOOKAHEAD == (0, 1, 2)
    with torch.no_grad():
        out = model(torch.randn(2, 37 * FS))
    assert len(out["logits"]) == 3
    for lg in out["logits"]:
        assert lg.shape == (2, 37, 100)
    assert out["latents"].shape == (2, 37, 64)
    # LayerNorm over channels: every latent frame is normalised on its own
    torch.testing.assert_close(out["latents"].mean(-1), torch.zeros(2, 37), atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match="multiple of 320"):
        model(torch.randn(1, 1000))


@pytest.mark.parametrize("fps", [1, 2, 4])
def test_step_equals_forward(model, fps):
    torch.manual_seed(fps)
    x = 0.3 * torch.randn(1, 152 * FS)          # ~3 s
    with torch.no_grad():
        ref = model(x)
        state, got = model.init_state(1), [[] for _ in range(3)]
        lat = []
        for i in range(0, x.shape[-1], fps * FS):
            out, state = model.step(x[:, i:i + fps * FS], state)
            for h in range(3):
                got[h].append(out["logits"][h])
            lat.append(out["latents"])
    for h in range(3):
        torch.testing.assert_close(torch.cat(got[h], 1), ref["logits"][h], atol=1e-5, rtol=0)
    torch.testing.assert_close(torch.cat(lat, 1), ref["latents"], atol=1e-5, rtol=0)


def test_whole_model_is_causal(model):
    """Changing samples of frame t+1 leaves every head's outputs <= t untouched,
    and the change does reach output t+1 (no hidden delay)."""
    torch.manual_seed(3)
    x = torch.randn(1, 40 * FS, dtype=torch.float64)
    m = model.double()
    try:
        with torch.no_grad():
            ref = m(x)
            for t in (0, 10, 38):
                x2 = x.clone()
                x2[:, (t + 1) * FS:(t + 2) * FS] += torch.randn(FS, dtype=torch.float64)
                out = m(x2)
                for h in range(3):
                    d = (out["logits"][h] - ref["logits"][h]).abs().amax(-1)[0]
                    assert d[:t + 1].max().item() == 0, (h, t)
                    assert d[t + 1].item() > 0, (h, t)
    finally:
        model.float()


def test_head_delay_alignment():
    """Output t of head L is scored against label t - L: a head emitting the
    label of frame t - L at index t is perfect, an undelayed one is not."""
    n, vocab = 30, 100
    labels = torch.randint(0, vocab, (2, n))
    pairs = ce.head_pairs([torch.zeros(2, n, vocab)] * 3, labels)
    for L, (lg, tg) in zip(ce.LOOKAHEAD, pairs):
        assert lg.shape[1] == tg.shape[1] == n - L
        torch.testing.assert_close(tg, labels[:, :n - L])
    logits = []
    for L in ce.LOOKAHEAD:
        delayed = torch.full((2, n), 0)
        delayed[:, L:] = labels[:, :n - L]          # index t carries label t - L
        logits.append(torch.nn.functional.one_hot(delayed, vocab).float() * 50)
    loss, per = ce.distill_loss(logits, labels)
    assert all(p.item() < 1e-6 for p in per)
    assert [c == k for c, k in ce.head_accuracy(logits, labels)] == [True] * 3
    undelayed = [torch.nn.functional.one_hot(labels, vocab).float() * 50] * 3
    acc = ce.head_accuracy(undelayed, labels)
    assert acc[0][0] == acc[0][1] and acc[2][0] < acc[2][1]
    ignored = labels.clone()
    ignored[:, -5:] = ce.IGNORE
    assert ce.head_accuracy(logits, ignored)[0][1] == 2 * (n - 5)


def test_aligned_ids_undo_the_head_delay(model):
    x = torch.randn(1, 25 * FS - 17)                # not a whole number of frames
    with torch.no_grad():
        padded = ce.pad_to_frames(x)
        assert padded.shape[-1] == 25 * FS
        for h, L in enumerate(ce.LOOKAHEAD):
            ids, lg = ce.aligned_ids(model, x, h)
            assert ids.shape == (1, 25) and lg.shape == (1, 25, 100)
            full = model(torch.nn.functional.pad(padded, (0, L * FS)))["logits"][h]
            torch.testing.assert_close(lg, full[:, L:L + 25])


def test_head_lookahead_names():
    assert [ce.head_lookahead(f"student-L{i}") for i in range(3)] == [0, 1, 2]
    assert ce.head_lookahead(1) == 1
    with pytest.raises(ValueError):
        ce.head_lookahead("student-L3")
    with pytest.raises(ValueError):
        ce.head_lookahead("teacher-km100")


def test_config_round_trip():
    cfg = ce.EncoderConfig(channels=(16, 32, 32, 64, 64), vocab=50)
    assert ce.EncoderConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(ValueError, match="320"):
        ce.ContentEncoder(ce.EncoderConfig(strides=(2, 4, 5, 4)))


# -- unit-stream metadata ------------------------------------------------------
def meta(**kw):
    m = units.base_meta("student-L1", 320, 1, 100, "km100:abc", True, "", {})
    m.update(kw)
    return m


def test_check_compatible():
    units.check_compatible(meta(), meta())
    for key, value in (("frame_rate_hz", 12.5), ("vocab", 2048), ("lookahead_frames", 2),
                       ("label_space", "km100:other")):
        with pytest.raises(units.UnitsMismatch, match=key):
            units.check_compatible(meta(), meta(**{key: value}))
    units.check_compatible(meta(), meta(lookahead_frames=None), ignore=("lookahead_frames",))
    with pytest.raises(units.UnitsMismatch, match="vocab"):
        units.check_compatible(meta(), meta(vocab=50, lookahead_frames=0), ignore=("lookahead_frames",))
    with pytest.raises(ValueError, match="unknown"):
        units.check_compatible(meta(), meta(), ignore=("speaker",))
    assert meta()["frame_rate_hz"] == 50.0
    assert units.safe_name("a/b c") == "a__b_c"


def test_read_manifests_names_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        units.read_manifests([str(tmp_path / "nope.jsonl")])


# -- unit probes ---------------------------------------------------------------
def test_probe_metrics_by_hand():
    perfect = unit_probes.probe_scores(np.array(list("aabbcc")), np.array([5, 5, 7, 7, 1, 1]))
    assert perfect["pnmi"] == pytest.approx(1.0)
    assert perfect["phone_purity"] == perfect["cluster_purity"] == pytest.approx(1.0)
    # independent: every phone meets every unit equally often
    ind = unit_probes.probe_scores(np.array(list("aabb")), np.array([0, 1, 0, 1]))
    assert ind["pnmi"] == pytest.approx(0.0, abs=1e-12)
    assert ind["phone_purity"] == ind["cluster_purity"] == pytest.approx(0.5)
    # p(a,0) = 1/2, p(b,0) = 1/4, p(b,1) = 1/4
    s = unit_probes.probe_scores(np.array(list("aabb")), np.array([0, 0, 0, 1]))
    assert s["phone_purity"] == pytest.approx(0.75)      # max_p p(p,0) + max_p p(p,1) = .5 + .25
    assert s["cluster_purity"] == pytest.approx(0.75)    # max_u p(a,u) + max_u p(b,u) = .5 + .25
    mi = 0.5 * math.log(4 / 3) + 0.25 * math.log(2 / 3) + 0.25 * math.log(2)
    assert s["pnmi"] == pytest.approx(mi / math.log(2))
    # many units per phone: phone purity 1, cluster purity low, PNMI 1
    split = unit_probes.probe_scores(np.array(list("aaaa")), np.array([0, 1, 2, 3]))
    assert split["phone_purity"] == 1.0 and split["cluster_purity"] == 0.25


LONG_TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.12
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.12
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.035
            text = ""
        intervals [2]:
            xmin = 0.035
            xmax = 0.12
            text = "say ""hi"""
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.12
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 0.035
            text = "sil"
        intervals [2]:
            xmin = 0.035
            xmax = 0.065
            text = "AH0"
        intervals [3]:
            xmin = 0.065
            xmax = 0.1
            text = "B"
        intervals [4]:
            xmin = 0.1
            xmax = 0.12
            text = ""
'''

SHORT_TG = '''File type = "ooTextFile"
Object class = "TextGrid"

0
0.12
<exists>
1
"IntervalTier"
"phones"
0
0.12
4
0
0.035
"sil"
0.035
0.065
"AH0"
0.065
0.1
"B"
0.1
0.12
""
'''


def test_textgrid_long_and_short(tmp_path):
    tiers = unit_probes.read_textgrid(LONG_TG)
    assert list(tiers) == ["words", "phones"]
    assert tiers["words"][1] == (0.035, 0.12, 'say "hi"')
    assert tiers["phones"][1] == (0.035, 0.065, "AH0")
    assert unit_probes.read_textgrid(SHORT_TG)["phones"] == tiers["phones"]
    p = tmp_path / "u1.TextGrid"
    p.write_text(LONG_TG, encoding="utf-16")             # Praat writes UTF-16 for non-ASCII
    assert unit_probes.read_textgrid(p) == tiers
    assert unit_probes.index_alignments(tmp_path) == {"u1": p}
    assert unit_probes.phone_tier({"spk - phones": [1]}) == [1]
    with pytest.raises(ValueError, match="TextGrid"):
        unit_probes.read_textgrid("File type = \"ooTextFile\"\nObject class = \"Sound\"\n")


def test_frame_labels_majority_and_offset():
    ivs = unit_probes.read_textgrid(LONG_TG)["phones"]
    # 50 Hz, default centre offset 10 ms: windows [0,20) [20,40) [40,60) [60,80) [80,100) [100,120) [120,140) ms
    assert unit_probes.frame_labels(ivs, 7, 50.0) == ["sil", "sil", "AH", "B", "B", "sil", "sil"]
    # centre offset 0: windows [-10,10) [10,30) [30,50) [50,70) [70,90) ms
    assert unit_probes.frame_labels(ivs, 5, 50.0, centre_offset=0.0) == ["sil", "sil", "AH", "AH", "B"]
    # 12.5 Hz: one 80 ms window covers sil 35 ms, AH 30 ms, B 15 ms
    assert unit_probes.frame_labels(ivs, 1, 12.5) == ["sil"]
    assert unit_probes.normalize_phone("ER1") == "ER" and unit_probes.normalize_phone("sp") == "sil"


def test_speaker_probe_on_synthetic_speakers():
    rng = np.random.default_rng(0)
    items = []
    for spk in range(6):
        p = rng.dirichlet(np.ones(20) * 0.3)          # each speaker has its own unit distribution
        for _ in range(10):
            items.append((f"s{spk}", rng.choice(20, size=200, p=p).astype(np.int16)))
    r = unit_probes.speaker_probe(items, vocab=20, n_speakers=4, seed=1)
    assert r["speakers"] == 4 and r["chance"] == 0.25 and r["test_utts"] == 8
    assert r["accuracy"] > 0.9
    x = unit_probes.ngram_features([np.array([1, 2, 1, 2])], vocab=3)
    assert x.shape == (1, 3 + 9)
    np.testing.assert_allclose(x.toarray()[0, :3].sum(), 1.0)
    np.testing.assert_allclose(x.toarray()[0, 3:].sum(), 1.0)
    big = unit_probes.ngram_features([np.array([2047, 0, 5])], vocab=2048, max_bigram_features=1000)
    assert big.shape == (1, 2048 + 1000)


# -- teacher k-means -----------------------------------------------------------
def test_reservoir_is_bounded_uniform_and_deterministic():
    def fill(seed):
        r = teacher.Reservoir(100, 1, seed)
        for start in range(0, 10_000, 337):
            r.add(np.arange(start, min(start + 337, 10_000), dtype=np.float32)[:, None])
        return r
    r = fill(0)
    assert r.filled == 100 and r.seen == 10_000
    vals = r.sample()[:, 0]
    assert len(np.unique(vals)) == 100
    assert 2000 < vals.mean() < 8000                   # spread over the whole stream
    np.testing.assert_array_equal(fill(0).sample()[:, 0], vals)


def test_fit_kmeans_and_assign():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    feats = [centres[rng.integers(0, 3, 50)] + 0.1 * rng.standard_normal((50, 2)).astype(np.float32)
             for _ in range(10)]
    c, info = teacher.fit_kmeans(iter(feats), k=3, max_frames=300, seed=0, log=lambda s: None)
    assert c.shape == (3, 2) and c.dtype == np.float32 and info["n_frames_used"] == 300
    ids = teacher.assign(centres, c)
    assert ids.dtype == np.int16 and len(set(ids.tolist())) == 3
    np.testing.assert_array_equal(teacher.assign(torch.from_numpy(feats[0]), c), teacher.assign(feats[0], c))
    with pytest.raises(ValueError, match="at least"):
        teacher.fit_kmeans(iter([feats[0][:2]]), k=3)
    assert teacher.PAD == 40
    assert teacher.fix_length(np.zeros((9, 2)), 10).shape == (10, 2)
    with pytest.raises(ValueError):
        teacher.fix_length(np.zeros((7, 2)), 10)


def test_teacher_layer7_indexing():
    """hidden_states[7] is the output of transformer layer 7 (encoder.layers[6]),
    the truncated teacher matches the full model, and 320*n samples give n frames."""
    if not teacher.weights_cached() or not teacher.weights_cached(filename="pytorch_model.bin"):
        pytest.skip(f"{teacher.MODEL_ID}@{teacher.REVISION[:8]} not cached (run extract_units.py "
                    "--fit-kmeans once to download it)")
    t = teacher.HubertTeacher("cpu")
    full = teacher.HubertTeacher("cpu", truncate=False)
    captured = {}
    hook = full.model.encoder.layers[6].register_forward_hook(
        lambda mod, args, out: captured.__setitem__("out", out[0] if isinstance(out, tuple) else out))
    x = (0.1 * np.random.default_rng(0).standard_normal(50 * FS)).astype(np.float32)
    try:
        with torch.inference_mode():
            hs = full.model(torch.from_numpy(np.pad(x, (teacher.PAD, teacher.PAD)))[None],
                            output_hidden_states=True).hidden_states
    finally:
        hook.remove()
    assert len(hs) == 13
    assert torch.equal(captured["out"], hs[7])
    assert not torch.equal(hs[6], hs[7])
    feats = t.features(x)
    assert feats.shape == (50, 768)
    np.testing.assert_allclose(feats, hs[7][0].numpy(), atol=1e-5)


# -- augmentation ----------------------------------------------------------------
def test_augmentations():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32) * 0.1
    n = rng.standard_normal(16000).astype(np.float32)
    y = augment.add_noise(x, n, 10.0)
    assert 10 * np.log10(augment.rms(x) ** 2 / augment.rms(y - x) ** 2) == pytest.approx(10.0, abs=1e-3)
    rir = np.zeros(800)
    rir[100], rir[300] = 0.2, 1.0                      # pre-peak samples are dropped
    rir = augment.prepare_rir(rir, 400)
    assert rir[0] == pytest.approx(1.0) and np.sum(rir ** 2) == pytest.approx(1.0)
    imp = np.zeros(1000, np.float32)
    imp[500] = 1.0
    out = augment.reverb(imp, np.array([0.8, 0.6]))
    assert np.abs(out[:500]).max() < 1e-6 and len(out) == 1000   # causal (FFT round-off only)
    assert out[500] == pytest.approx(0.8)                           # no added delay
    b, a = augment.biquad("peaking", 1000, 0.0, 1.0)
    np.testing.assert_allclose(b, a)                     # 0 dB peaking EQ is the identity
    for kind in ("lowshelf", "highshelf"):
        b, a = augment.biquad(kind, 500, 0.0, 0.707)
        np.testing.assert_allclose(b, a, atol=1e-12)
    aug = augment.Augmenter(augment.AugmentConfig(gain_db=0.0, eq_p=0.0))
    np.testing.assert_allclose(aug(x, rng), x, atol=1e-7)  # no files, no EQ, 0 dB: unchanged
    loud = augment.Augmenter(augment.AugmentConfig(eq_p=0.0, gain_db=0.0))(x * 50, rng)
    assert np.abs(loud).max() <= 1.0


def test_augmenter_with_noise_and_rir_files(tmp_path):
    import soundfile as sf
    rng = np.random.default_rng(1)
    sf.write(tmp_path / "noise.wav", rng.standard_normal(3000).astype(np.float32) * 0.1, 16000)  # shorter: looped
    rir = np.zeros(4000, np.float32)
    rir[50], rir[400:] = 1.0, 0.05 * rng.standard_normal(3600)
    sf.write(tmp_path / "rir.wav", rir, 16000)
    cfg = augment.AugmentConfig(noise_p=1.0, rir_p=1.0, eq_p=1.0, gain_db=0.0)
    aug = augment.Augmenter(cfg, [tmp_path / "noise.wav"], [tmp_path / "rir.wav"])
    x = (0.2 * np.sin(np.arange(8000) * 0.05)).astype(np.float32)
    y = aug(x, rng)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert np.abs(y - x).max() > 1e-3
    assert augment.read_segment(tmp_path / "noise.wav", 8000, rng).shape == (8000,)


# -- tiny end-to-end: teacher-style units -> train -> resume -> extract ----------
def test_train_resume_and_extract(tmp_path, monkeypatch):
    import soundfile as sf

    import extract_units
    import train_content_encoder as tce
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    rng = np.random.default_rng(0)
    (tmp_path / "audio").mkdir()
    lab_dir = tmp_path / "labels"
    lab_dir.mkdir()
    rows, urows = [], []
    for i in range(4):
        n = 30 + 7 * i
        x = (0.1 * rng.standard_normal(n * FS - 100)).astype(np.float32)
        sf.write(tmp_path / "audio" / f"u{i}.flac", x, 16000)
        np.save(lab_dir / f"u{i}.npy", rng.integers(0, 20, n).astype(np.int16))
        rows.append({"utt_id": f"u{i}", "path": f"audio/u{i}.flac", "duration": len(x) / 16000,
                     "speaker": f"s{i % 2}", "dataset": "fake", "subset": "x"})
        urows.append({"utt_id": f"u{i}", "units": f"u{i}.npy", "n_frames": n, "path": rows[-1]["path"]})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (lab_dir / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in urows), encoding="utf-8")
    labels_meta = units.base_meta("teacher-km100", 320, None, 20, "km20:fake", False, "", {})
    (lab_dir / units.META_FILE).write_text(json.dumps(labels_meta), encoding="utf-8")

    common = ["--train-manifest", str(manifest), "--train-units", str(lab_dir),
              "--val-manifest", str(manifest), "--val-units", str(lab_dir), "--val-every", "3",
              "--batch-size", "2", "--crop-seconds", "0.4", "--workers", "0", "--device", "cpu",
              "--warmup", "2", "--log-every", "1", "--run-name", "t", "--runs-dir", str(tmp_path / "runs"),
              "--ckpt-dir", str(tmp_path / "ckpt")]
    assert tce.main(common + ["--steps", "3"]) == 0
    ck = torch.load(tmp_path / "ckpt" / "t" / "last.pt", weights_only=True)
    assert ck["step"] == 3 and json.loads(ck["labels_meta"])["label_space"] == "km20:fake"
    assert tce.main(common + ["--steps", "5", "--resume", "auto"]) == 0
    ck = torch.load(tmp_path / "ckpt" / "t" / "final.pt", weights_only=True)
    assert ck["step"] == 5
    assert (tmp_path / "ckpt" / "t" / "best.pt").is_file()

    out = tmp_path / "units_L1"
    args = ["--tokenizer", "student-L1", "--checkpoint", str(tmp_path / "ckpt" / "t" / "final.pt"),
            "--manifest", str(manifest), "--out-dir", str(out), "--device", "cpu", "--save-logits"]
    assert extract_units.main(args) == 0
    m, got = units.read_units_dir(out)
    assert m["tokenizer"] == "student-L1" and m["lookahead_frames"] == 1 and m["vocab"] == 20
    assert m["label_space"] == "km20:fake" and m["frame_rate_hz"] == 50.0
    for r, u in zip(sorted(got, key=lambda r: r["utt_id"]), urows):
        ids = np.load(out / r["units"])
        assert ids.dtype == np.int16 and len(ids) == u["n_frames"] == r["n_frames"]
        assert np.load(out / r["units"].replace(".npy", ".logits.npy")).shape == (len(ids), 20)
        assert r["speaker"] in ("s0", "s1")
    units.check_compatible(labels_meta, m, ignore=("lookahead_frames",))
    # a rerun skips finished utterances; another head may not share the directory
    assert extract_units.main(args) == 0
    assert extract_units.main([*args[:1], "student-L2", *args[2:]]) == 1


def test_probe_units_cli(tmp_path, capsys):
    """Phone probe end to end: a stream that copies the phones scores PNMI 1,
    and two --units give the ratio against the first."""
    import extract_units
    import probe_units
    phones = ["sil", "AH0", "B", "sil"]
    ivs = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.6)]
    body = "".join(f'intervals [{i + 1}]:\n xmin = {s}\n xmax = {e}\n text = "{p}"\n'
                   for i, ((s, e), p) in enumerate(zip(ivs, phones)))
    tg = ('File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 0.6\n'
          'tiers? <exists>\nsize = 1\nitem []:\nitem [1]:\nclass = "IntervalTier"\nname = "phones"\n'
          f'xmin = 0\nxmax = 0.6\nintervals: size = 4\n{body}')
    (tmp_path / "align" / "spk").mkdir(parents=True)
    (tmp_path / "align" / "spk" / "u0.TextGrid").write_text(tg, encoding="utf-8")
    labels = unit_probes.frame_labels(unit_probes.read_textgrid(tg)["phones"], 30, 50.0)
    code = {"sil": 7, "AH": 3, "B": 9}
    dirs = []
    for name, ids in (("copy", [code[p] for p in labels]), ("const", [1] * 30)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "u0.npy", np.array(ids, np.int16))
        (d / units.MANIFEST_FILE).write_text(json.dumps({"utt_id": "u0", "units": "u0.npy",
                                                         "n_frames": 30}) + "\n", encoding="utf-8")
        (d / units.META_FILE).write_text(json.dumps(meta()), encoding="utf-8")
        dirs.append(str(d))
    out = tmp_path / "report.json"
    assert probe_units.main(["--units", dirs[0], "--units", dirs[1], "--alignments",
                             str(tmp_path / "align"), "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))["units"]
    assert rep[dirs[0]]["phones"]["pnmi"] == pytest.approx(1.0)
    assert rep[dirs[0]]["phones"]["frames"] == 30 and rep[dirs[0]]["phones"]["phones"] == 3
    assert rep[dirs[1]]["phones"]["pnmi_ratio_vs_first"] == pytest.approx(0.0)
    capsys.readouterr()
    assert probe_units.main(["--units", dirs[0], "--alignments", str(tmp_path / "nothing")]) == 1
    assert extract_units.main(["--fit-kmeans", "--manifest", str(tmp_path / "missing.jsonl"),
                               "--out-dir", str(tmp_path / "km"), "--device", "cpu"]) == 1
    assert "missing.jsonl" in capsys.readouterr().err
