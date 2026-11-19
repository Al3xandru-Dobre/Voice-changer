"""Multilingual encoder & vocoder upgrade (Milestone 3 / Stage 03): the teacher
registry and its licence guard, the fairseq checkpoint loader, the balanced
k-means, K=1000 labelling with prosody, the dual-output student (latents +
units) and its streaming parity, student-latent extraction, language-ratio
sampling, the multilingual vocoder (lang tag, no language input), the
contract-v2 export, the multilingual probe and the scoring hooks.

Everything runs on synthetic audio, tiny models and a fake teacher: no corpus,
no pretrained weights, CPU only.
"""
import collections
import dataclasses
import inspect
import json
import pathlib
import sys
import types

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

import eval_multilingual as em  # noqa: E402
import export_stateful_onnx as cli  # noqa: E402
import extract_units  # noqa: E402
import probe_units  # noqa: E402
import stream_check  # noqa: E402
import train_content_encoder as tce  # noqa: E402
import train_unit_vocoder as tuv  # noqa: E402
from fvml import asr, audio, evaluate, evallist, sampling, teacher, unit_probes, units  # noqa: E402
from fvml import content_encoder as ce  # noqa: E402
from fvml import contract as fc  # noqa: E402
from fvml import export as fx  # noqa: E402
from fvml import vocoder as vc  # noqa: E402

FS = ce.FRAME_SAMPLES
DIM = 12                                     # the fake teacher's feature width
TINY_ENC = ce.EncoderConfig(channels=(4, 4, 8, 8, 8), latent_dim=8, head_hidden=8, vocab=16)
TINY_VOC = vc.VocoderConfig(vocab=16, unit_dim=8, cond_hidden=8, n_speakers=2, spk_emb_dim=4,
                            upsample_initial_channel=16, resblock_kernels=(3,),
                            resblock_dilations=((1, 3),))
STATS = {"logf0_mean": 5.2, "logf0_std": 0.2, "energy_mean": -8.0, "energy_std": 3.0}


def speech(seconds, seed=0, sr=16000):
    """Voiced, gliding, amplitude-modulated harmonics plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 110 + 30 * (seed % 5) + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 * (1 + np.sin(2 * np.pi * 1.3 * t))
    return (0.1 * x * env + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


class FakeTeacher:
    """Stands in for HubertTeacher: log-magnitude spectrum bins per 20 ms frame."""
    calls = 0

    def __init__(self, device="cpu", layer=None, model_id=None, revision=None, truncate=True,
                 spec=None, checkpoint=None):
        self.spec = spec or teacher.TEACHERS[teacher.DEFAULT_TEACHER]
        self.layer = self.spec.layer if layer is None else layer
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=DIM))

    def features(self, wav):
        type(self).calls += 1
        frames = np.asarray(wav, np.float32).reshape(-1, FS)
        return np.log1p(np.abs(np.fft.rfft(frames, axis=1))[:, 1:DIM + 1]).astype(np.float32)

    def info(self):
        return {"model": "fake", "revision": None, "layer": self.layer, "teacher": self.spec.name,
                "license": self.spec.license, "noncommercial": self.spec.noncommercial}


# -- teacher registry & licence guard ----------------------------------------------
def test_teacher_registry_and_license_guard():
    base, vp = teacher.TEACHERS["hubert-base"], teacher.TEACHERS["mhubert-vp"]
    # The M2 teacher is untouched: same model, layer, K and file names.
    assert (base.model_id, base.revision, base.layer, base.k) == \
        (teacher.MODEL_ID, teacher.REVISION, 7, 100)
    assert (base.centroids_file, base.meta_file, base.tokenizer) == \
        ("km100_centroids.npy", "km100_meta.json", "teacher-km100")
    # The M3 default: MIT mhubert-vp, layer 11, K=1000, the file name the stage names.
    assert teacher.MULTILINGUAL_TEACHER == "mhubert-vp" and not vp.noncommercial
    assert (vp.layer, vp.k, vp.source) == (11, 1000, "fairseq") and "MIT" in vp.license
    assert vp.centroids_file == "mhubert_vp_en_es_fr_de_km1000.npy"
    assert vp.url.endswith("mhubert_base_vp_en_es_fr_it3.pt")
    assert teacher.get_teacher("teacher-mhubert-km1000") is vp
    assert set(units.TOKENIZERS) >= {t.tokenizer for t in teacher.TEACHERS.values()}
    # mHuBERT-147 is non-commercial: every door needs the explicit opt-in.
    nc = teacher.TEACHERS["mhubert-147"]
    assert nc.noncommercial and "NC" in nc.license
    for name in ("mhubert-147", "teacher-mhubert147-km1000"):
        with pytest.raises(teacher.LicenseError, match="include-noncommercial"):
            teacher.get_teacher(name)
        assert teacher.get_teacher(name, include_noncommercial=True) is nc
    with pytest.raises(teacher.LicenseError):
        units.make_tokenizer("teacher-mhubert147-km1000", centroids="anything", device="cpu")
    with pytest.raises(ValueError, match="unknown teacher"):
        teacher.get_teacher("wavlm")
    units.check_license({"tokenizer": "teacher-km100"})
    units.check_license({"noncommercial": True}, include_noncommercial=True)
    with pytest.raises(teacher.LicenseError):
        units.check_license({"tokenizer": "student-L1", "noncommercial": True, "license": "NC"})
    # Export: refused without the flag, tagged with it; commercial licences pass through.
    with pytest.raises(fx.ExportError, match="non-commercial"):
        cli.noncommercial_license({"noncommercial": True, "license": "CC-BY-NC-SA"}, "x", False)
    assert cli.noncommercial_license({"noncommercial": True}, "x", True) == "x" + cli.NC_NOTE
    assert cli.noncommercial_license({"tokenizer": "teacher-km100"}, "MIT", False) == "MIT"
    assert cli.noncommercial_license(None, "MIT", False) == "MIT"


def to_fairseq(state):
    """transformers HubertModel state dict -> fairseq names (the inverse of the
    loader's table, written out independently)."""
    layer = {"attention.k_proj": "self_attn.k_proj", "attention.v_proj": "self_attn.v_proj",
             "attention.q_proj": "self_attn.q_proj", "attention.out_proj": "self_attn.out_proj",
             "layer_norm": "self_attn_layer_norm", "feed_forward.intermediate_dense": "fc1",
             "feed_forward.output_dense": "fc2", "final_layer_norm": "final_layer_norm"}
    out = collections.OrderedDict()
    for k, v in state.items():
        parts = k.split(".")
        if k == "masked_spec_embed":
            out["mask_emb"] = v
        elif k.startswith("feature_extractor.conv_layers."):
            out[f"feature_extractor.conv_layers.{parts[2]}."
                f"{'0' if parts[3] == 'conv' else '2'}.{parts[4]}"] = v
        elif k.startswith("feature_projection.projection."):
            out["post_extract_proj." + parts[-1]] = v
        elif k.startswith("feature_projection.layer_norm."):
            out["layer_norm." + parts[-1]] = v
        elif k == "encoder.pos_conv_embed.conv.bias":
            out["encoder.pos_conv.0.bias"] = v
        elif k.endswith(("original0", "weight_g")):
            out["encoder.pos_conv.0.weight_g"] = v
        elif k.endswith(("original1", "weight_v")):
            out["encoder.pos_conv.0.weight_v"] = v
        elif k.startswith("encoder.layer_norm."):
            out[k] = v
        else:
            out[f"encoder.layers.{parts[2]}.{layer['.'.join(parts[3:-1])]}.{parts[-1]}"] = v
    return out


def test_fairseq_checkpoint_loads_without_fairseq(tmp_path):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.HubertConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                                    intermediate_size=64, conv_dim=(8,) * 7,
                                    num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
    torch.manual_seed(0)
    ref = transformers.HubertModel(cfg).eval()
    state = to_fairseq(ref.state_dict())
    state["label_embs_concat"] = torch.zeros(4, 8)             # pretraining heads: dropped
    state["final_proj.weight"], state["final_proj.bias"] = torch.zeros(8, 32), torch.zeros(8)

    # A pickled fairseq object whose module is gone at load time, as on Python 3.12.
    class Dictionary:
        def __init__(self):
            self.symbols, self.indices = ["<s>", "a"], {"a": 1}
    Dictionary.__module__, Dictionary.__qualname__ = "fairseq.data.dictionary", "Dictionary"
    mods = {"fairseq": types.ModuleType("fairseq"), "fairseq.data": types.ModuleType("fairseq.data"),
            "fairseq.data.dictionary": types.ModuleType("fairseq.data.dictionary")}
    mods["fairseq.data.dictionary"].Dictionary = Dictionary
    sys.modules.update(mods)
    try:
        torch.save({"args": None, "model": state, "task_state": {"dictionaries": [Dictionary()]},
                    "cfg": {"model": {"extractor_mode": "default", "layer_norm_first": False},
                            "task": {"normalize": False}},
                    "extra_state": {"best": np.float64(1.5)}}, tmp_path / "fake.pt")
    finally:
        for name in mods:
            sys.modules.pop(name, None)
    with pytest.raises(Exception):                              # plain torch cannot read it
        torch.load(tmp_path / "fake.pt", weights_only=True)

    model, fairseq_cfg = teacher.load_fairseq_hubert(tmp_path / "fake.pt", config=cfg)
    assert fairseq_cfg["task"] == {"normalize": False}
    x = torch.randn(1, 4000)
    with torch.no_grad():
        want = ref(x, output_hidden_states=True).hidden_states
        got = model.eval()(x, output_hidden_states=True).hidden_states
    assert len(got) == 4 and all(torch.equal(a, b) for a, b in zip(want, got))

    # Strict: an unknown tensor, a missing tensor and another architecture all raise.
    torch.save({"model": {**state, "encoder.layers.0.mystery.weight": torch.zeros(1)}},
               tmp_path / "unknown.pt")
    with pytest.raises(ValueError, match="no mapping for encoder.layers.0.mystery.weight"):
        teacher.load_fairseq_hubert(tmp_path / "unknown.pt", config=cfg)
    torch.save({"model": {k: v for k, v in state.items() if k != "post_extract_proj.bias"}},
               tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="feature_projection.projection.bias"):
        teacher.load_fairseq_hubert(tmp_path / "missing.pt", config=cfg)
    torch.save({"model": state, "cfg": {"model": {"layer_norm_first": True}}}, tmp_path / "large.pt")
    with pytest.raises(ValueError, match="HuBERT-Base layout"):
        teacher.load_fairseq_hubert(tmp_path / "large.pt", config=cfg)
    torch.save({"weights": 1}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="no 'model' state dict"):
        teacher.load_fairseq_hubert(tmp_path / "other.pt", config=cfg)
    # A missing checkpoint names the URL; nothing is downloaded behind the user's back.
    vp = teacher.TEACHERS["mhubert-vp"]
    with pytest.raises(FileNotFoundError, match="mhubert_base_vp_en_es_fr_it3.pt"):
        teacher.HubertTeacher("cpu", spec=vp, checkpoint=tmp_path / "absent.pt")


def test_fetch_checkpoint_uses_the_pinned_archive(tmp_path, monkeypatch):
    from fvml import datasets as fd
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    seen = []
    monkeypatch.setattr(fd, "download", lambda ctx, a: seen.append(a) or tmp_path / a.file)
    monkeypatch.setattr(fd, "verify", lambda ctx, a, path: seen.append(path))
    vp = teacher.TEACHERS["mhubert-vp"]
    assert teacher.fetch_checkpoint(vp, log=lambda s: None) == tmp_path / vp.file
    assert seen[0] == fd.Archive(vp.file, (vp.url,), vp.size) and vp.size > 1e9
    assert vp.default_checkpoint() == tmp_path / "archives" / vp.file
    with pytest.raises(ValueError, match="Hugging Face"):
        teacher.fetch_checkpoint(teacher.TEACHERS["hubert-base"])


# -- balanced k-means ---------------------------------------------------------------
def test_balanced_kmeans_gives_every_language_the_same_frames():
    rng = np.random.default_rng(0)

    def stream():
        for _ in range(50):                                  # 10x more English than German
            yield "en", rng.standard_normal((40, 4)).astype(np.float32)
        for _ in range(5):
            yield "de", (rng.standard_normal((40, 4)) + 10).astype(np.float32)

    c, info = teacher.fit_kmeans_balanced(stream(), ["en", "de"], k=8, max_frames=1000, seed=0,
                                          log=lambda s: None)
    assert c.shape == (8, 4) and c.dtype == np.float32 and info["balanced"] is True
    assert info["groups"]["en"] == {"frames_seen": 2000, "frames_used": 200, "utterances": 50}
    assert info["groups"]["de"] == {"frames_seen": 200, "frames_used": 200, "utterances": 5}
    assert info["n_frames_used"] == 400 and info["n_frames_seen"] == 2200
    # Balanced frames -> the minority language keeps its share of the codebook. A plain
    # fit on the same stream gives it about 1 centroid in 11.
    assert 3 <= int((c.mean(1) > 5).sum()) <= 5
    plain, _ = teacher.fit_kmeans((f for _, f in stream()), k=8, max_frames=4000, seed=0,
                                  log=lambda s: None)
    assert int((plain.mean(1) > 5).sum()) < int((c.mean(1) > 5).sum())
    with pytest.raises(ValueError, match="no features for group"):
        teacher.fit_kmeans_balanced(stream(), ["en", "de", "fr"], k=8, log=lambda s: None)
    with pytest.raises(ValueError, match="unexpected group"):
        teacher.fit_kmeans_balanced(stream(), ["en"], k=8, log=lambda s: None)
    with pytest.raises(ValueError, match="at least k=500"):
        teacher.fit_kmeans_balanced(stream(), ["en", "de"], k=500, max_frames=1000,
                                    log=lambda s: None)
    with pytest.raises(ValueError, match="int16"):
        teacher.assign(np.zeros((1, 2), np.float32), np.zeros((40000, 2), np.float32))
    assert teacher.assign(np.zeros((3, 2), np.float32), np.zeros((1000, 2), np.float32)).dtype == np.int16


def test_import_stock_kmeans(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sklearn.cluster import MiniBatchKMeans
    import hashlib
    x = np.random.default_rng(0).standard_normal((200, DIM)).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=4, n_init=1, random_state=0).fit(x)
    path = tmp_path / "stock_L11_km4.bin"
    joblib.dump(km, path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    np.testing.assert_allclose(teacher.import_sklearn_kmeans(path, md5), km.cluster_centers_,
                               rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="refusing to unpickle"):
        teacher.import_sklearn_kmeans(path, "0" * 32)
    joblib.dump({"not": "kmeans"}, tmp_path / "other.bin")
    with pytest.raises(ValueError, match="not a fitted"):
        teacher.import_sklearn_kmeans(tmp_path / "other.bin")
    # CLI: the default expects the teacher's published MD5 and refuses anything else.
    out = tmp_path / "stock"
    base = ["--import-kmeans", str(path), "--teacher", "mhubert-vp", "--out-dir", str(out)]
    assert extract_units.main(base) == 1
    assert extract_units.main(base + ["--kmeans-md5", md5]) == 0
    c, _ = teacher.load_centroids(out / "stock_L11_km4.npy")
    assert c.shape == (4, DIM)
    meta = json.loads((out / "stock_L11_km4.json").read_text(encoding="utf-8"))
    assert meta["fit"] == "imported" and meta["teacher"] == "mhubert-vp" and meta["layer"] == 11


# -- sampling -----------------------------------------------------------------------
def test_language_ratio_sampling():
    assert sampling.parse_ratios("m3-vocoder") == {"en": 0.4, "es": 0.2, "fr": 0.2, "de": 0.2}
    assert sampling.parse_ratios("balanced") == dict.fromkeys(("en", "es", "fr", "de"), 0.25)
    assert sampling.parse_ratios("en,de") == {"en": 0.5, "de": 0.5}
    assert sampling.parse_ratios("en=3,es=1") == {"en": 0.75, "es": 0.25}
    assert sampling.parse_ratios(None) is None and sampling.parse_ratios("") is None
    for bad in ("en=0,es=1", "en=x", "en,en", "en=-1"):
        with pytest.raises(ValueError):
            sampling.parse_ratios(bad)
    # 100 h of English against 1 h of German: German still gets its 20%.
    langs = ["en"] * 100 + ["es"] * 10 + ["fr"] * 5 + ["de"]
    base = [float(i % 7 + 1) for i in range(len(langs))]
    w, summary = sampling.language_weights(base, langs, sampling.VOCODER)
    total = sum(w)
    for lang, share in sampling.VOCODER.items():
        got = sum(x for x, item_lang in zip(w, langs) if item_lang == lang) / total
        assert got == pytest.approx(share), lang
        assert summary[lang]["share"] == share
    assert summary["de"]["items"] == 1
    assert w[1] / w[0] == pytest.approx(base[1] / base[0])     # inside a language: by length
    # The draws themselves follow the shares.
    idx = torch.multinomial(torch.tensor(w, dtype=torch.double), 20000, replacement=True,
                            generator=torch.Generator().manual_seed(0)).tolist()
    drawn = collections.Counter(langs[i] for i in idx)
    assert drawn["en"] / 20000 == pytest.approx(0.4, abs=0.02)
    assert drawn["de"] / 20000 == pytest.approx(0.2, abs=0.02)
    with pytest.raises(ValueError, match="--default-lang"):
        sampling.language_weights([1, 1], ["en", None], {"en": 1.0})
    with pytest.raises(ValueError, match="no training items for language"):
        sampling.language_weights([1, 1], ["en", "en"], {"en": 0.5, "de": 0.5})


def test_audio_rows_and_subsets():
    mono = {"utt_id": "a", "path": "a.flac", "duration": 1.0, "speaker": "lj", "subset": "train"}
    pair = {"id": "p", "src_lang": "es", "tgt_lang": "en", "src_audio_path": "s.flac",
            "src_duration_s": 2.0, "src_text": "hola", "tgt_audio_path": "t.flac",
            "tgt_duration_s": 2.5, "tgt_text": "hello", "speaker_id": "cv1",
            "dataset": "cvss-c-es_en", "license": "CC BY 4.0", "split": "dev", "gender": "F"}
    assert units.audio_rows([mono]) == [mono]                     # M2 rows pass through
    assert units.audio_rows([mono], default_lang="en")[0]["lang"] == "en"
    assert units.audio_rows([{**mono, "lang": "de"}], default_lang="en")[0]["lang"] == "de"
    src, = units.audio_rows([pair], "src")
    assert src == {"utt_id": "p", "path": "s.flac", "duration": 2.0, "dataset": "cvss-c-es_en",
                   "subset": "dev", "lang": "es", "pair": "es_en", "side": "src", "speaker": "cv1",
                   "text": "hola", "license": "CC BY 4.0", "gender": "F"}
    tgt, = units.audio_rows([pair], "tgt")
    assert (tgt["path"], tgt["duration"], tgt["lang"], tgt["text"]) == ("t.flac", 2.5, "en", "hello")
    assert tgt["speaker"] == "cvss-c-es_en_tgt_en" and "gender" not in tgt
    tuv.validate_speaker_name(tgt["speaker"])
    with pytest.raises(ValueError, match="--side"):
        units.audio_rows([pair])
    # BUG-031: a non-commercial corpus flag survives the paired -> monolingual
    # conversion, and rows-level guards see it without the opt-in flag.
    nc_src, = units.audio_rows([{**pair, "noncommercial": True, "license": "CC BY-NC 4.0"}], "src")
    assert nc_src["noncommercial"] is True and nc_src["license"] == "CC BY-NC 4.0"
    assert units.check_rows_license([mono]) is None
    assert units.check_rows_license([nc_src], include_noncommercial=True) == "CC BY-NC 4.0"
    with pytest.raises(teacher.LicenseError, match="non-commercial corpus"):
        units.check_rows_license([nc_src])
    rows = [mono, {**mono, "utt_id": "b", "subset": "test"}]
    assert [r["utt_id"] for r in units.filter_subsets(rows, "test")] == ["b"]
    assert len(units.filter_subsets(rows, None)) == 2 and len(units.filter_subsets(rows, "x")) == 0


# -- the dual-output student --------------------------------------------------------
@pytest.fixture(scope="module")
def m3_student():
    torch.manual_seed(0)
    return ce.ContentEncoder(ce.MULTILINGUAL).eval()


def test_multilingual_student_is_dual_output_within_budget(m3_student):
    cfg = m3_student.cfg
    assert (cfg.latent_dim, cfg.vocab) == (128, 1000) == \
        (ce.MULTILINGUAL_LATENT_DIM, ce.MULTILINGUAL_VOCAB)
    assert m3_student.num_params() <= ce.MAX_PARAMS_MULTILINGUAL == 6_000_000
    # The M2 English default is unchanged.
    assert (ce.EncoderConfig().latent_dim, ce.EncoderConfig().vocab) == (64, 100)
    norm = m3_student.trunk.norm
    assert isinstance(norm, torch.nn.LayerNorm) and tuple(norm.normalized_shape) == (128,)
    seen = {}
    hook = norm.register_forward_hook(lambda mod, args, out: seen.__setitem__("out", out))
    try:
        with torch.no_grad():
            out = m3_student(0.3 * torch.randn(2, 20 * FS))
    finally:
        hook.remove()
    assert out["latents"].shape == (2, 20, 128) and out["latents"].dtype == torch.float32
    assert [tuple(lg.shape) for lg in out["logits"]] == [(2, 20, 1000)] * 3
    # `latents` IS the LayerNorm output, and it is exactly what every K=1000 head reads.
    assert torch.equal(out["latents"], seen["out"])
    torch.testing.assert_close(out["latents"].mean(-1), torch.zeros(2, 20), atol=1e-5, rtol=0)
    # Unit variance per frame, up to LayerNorm's eps on a random init's small activations.
    torch.testing.assert_close(out["latents"].std(-1, correction=0), torch.ones(2, 20),
                               atol=2e-2, rtol=0)
    with torch.no_grad():
        for h, head in enumerate(m3_student.heads):
            assert head.out.out_channels == 1000
            assert torch.equal(head(out["latents"].transpose(1, 2)), out["logits"][h])


@pytest.mark.parametrize("fps", [1, 2, 4])
def test_dual_output_step_equals_forward(m3_student, fps):
    """Stage 03 DoD on a short stream: latents max |d| <= 1e-5 and unit argmax
    agreement >= 99.9% (the 60 s run is stream_check.py --multilingual)."""
    torch.manual_seed(fps)
    x = 0.3 * torch.randn(1, 76 * FS)
    with torch.no_grad():
        ref = m3_student(x)
        state, lat, units_out = m3_student.init_state(1), [], [[] for _ in range(3)]
        for i in range(0, x.shape[-1], fps * FS):
            out, state = m3_student.step(x[:, i:i + fps * FS], state)
            lat.append(out["latents"])
            for h in range(3):
                units_out[h].append(out["logits"][h].argmax(-1))
    lat = torch.cat(lat, 1)
    assert lat.shape == ref["latents"].shape == (1, 76, 128)
    assert float((lat - ref["latents"]).abs().max()) <= 1e-5
    for h in range(3):
        agree = (torch.cat(units_out[h], 1) == ref["logits"][h].argmax(-1)).float().mean()
        assert float(agree) >= 0.999


def test_stream_check_gates_the_latents_port(monkeypatch):
    torch.manual_seed(1)
    model = ce.ContentEncoder(TINY_ENC).eval()
    wav = speech(2.0)
    rep = stream_check.check(model, wav, fps_list=(1, 4))
    assert rep["pass"] and rep["latent_dim"] == 8 and rep["vocab"] == 16
    for fps in (1, 4):
        assert rep["fps"][fps]["latents"]["pass"] and rep["fps"][fps]["latents"]["max_abs"] <= 1e-5
    assert stream_check.LATENT_MAX_ABS == 1e-5
    # A trunk conv that forgets its cache breaks the latents, and the check says so.
    orig = ce.StreamingConv1d.step
    proj = model.trunk.proj

    def forget(self, x, cache):
        y, new = orig(self, x, cache)
        return y, (torch.zeros_like(new) if self is proj else new)

    monkeypatch.setattr(ce.StreamingConv1d, "step", forget)
    broken = stream_check.check(model, wav, fps_list=(1,))
    assert not broken["pass"] and not broken["fps"][1]["latents"]["pass"]


def test_encode_utterance_and_topk():
    torch.manual_seed(2)
    model = ce.ContentEncoder(TINY_ENC).eval()
    x = torch.randn(1, 25 * FS - 17)
    with torch.no_grad():
        full = model(ce.pad_to_frames(x))
        for h, L in enumerate(ce.LOOKAHEAD):
            ids, lg, lat = ce.encode_utterance(model, x, h)
            assert ids.shape == (1, 25) and lg.shape == (1, 25, 16) and lat.shape == (1, 25, 8)
            # Latents carry no head delay: they are the trunk frames of the utterance itself.
            torch.testing.assert_close(lat, full["latents"], atol=1e-6, rtol=0)
            ids2, lg2 = ce.aligned_ids(model, x, h)
            assert torch.equal(ids, ids2) and torch.equal(lg, lg2)
    # Every frame ranks the units 7, 0, 1, 2, 3, 4, 5, 6; the labels sit at ranks 2, 5, 6, 1.
    logits = -torch.arange(8.0).repeat(1, 4, 1)
    logits[0, :, 7] = 9.0
    labels = torch.tensor([[0, 3, 4, 7]])
    acc1 = ce.head_topk_accuracy([logits], labels, lookahead=(0,), k=1)
    assert acc1 == ce.head_accuracy([logits], labels, lookahead=(0,)) == [(1, 4)]
    assert ce.head_topk_accuracy([logits], labels, lookahead=(0,), k=2) == [(2, 4)]
    assert ce.head_topk_accuracy([logits], labels, lookahead=(0,), k=5) == [(3, 4)]
    assert ce.head_topk_accuracy([logits], labels, lookahead=(0,), k=6) == [(4, 4)]
    assert ce.head_topk_accuracy([logits], labels, lookahead=(0,), k=99) == [(4, 4)]
    ignored = labels.clone()
    ignored[0, 0] = ce.IGNORE
    assert ce.head_topk_accuracy([logits], ignored, lookahead=(0,), k=2) == [(1, 3)]
    # With a head delay, output t is scored against label t - L.
    assert ce.head_topk_accuracy([logits], labels, lookahead=(1,), k=5) == [(2, 3)]


# -- end to end on a tiny corpus: fit -> label (+prosody) -> student -> latents -> vocoder --
def write_corpus(root):
    """en.jsonl (LJ-style rows without `lang`), es.jsonl (CSS10-style rows) and a
    paired es->en S2UT manifest, all pointing at short synthetic clips."""
    (root / "audio").mkdir(parents=True)
    (root / "s2ut").mkdir()
    seed = iter(range(1000))

    def clip(name, frames):
        x = speech(frames * FS / 16000.0, seed=next(seed))[:frames * FS - 40]
        sf.write(root / "audio" / f"{name}.flac", x, 16000)
        return f"audio/{name}.flac", len(x) / 16000.0

    en, es, pairs = [], [], []
    for i, subset in enumerate(["train", "train", "train", "test"]):
        path, dur = clip(f"en{i}", 26 + 2 * i)
        en.append({"utt_id": f"en{i}", "path": path, "duration": dur, "speaker": "lj",
                   "dataset": "ljspeech", "subset": subset, "text": f"english clip {i}"})
        path, dur = clip(f"es{i}", 27 + 2 * i)
        es.append({"utt_id": f"es{i}", "path": path, "duration": dur, "speaker": "css10_es",
                   "dataset": "css10-es", "subset": subset, "lang": "es", "text": f"frase número {i}",
                   "license": "Apache-2.0 / CC0-1.0 (LibriVox)"})
    for i, split in enumerate(["train", "train", "dev"]):
        s_path, s_dur = clip(f"p{i}_src", 28 + i)
        t_path, t_dur = clip(f"p{i}_tgt", 30 + i)
        pairs.append({"id": f"p{i}", "src_lang": "es", "tgt_lang": "en", "src_audio_path": s_path,
                      "src_duration_s": s_dur, "src_text": "hola", "tgt_audio_path": t_path,
                      "tgt_duration_s": t_dur, "tgt_text": "hello", "speaker_id": f"cv{i}",
                      "dataset": "cvss-c-es_en", "license": "CC BY 4.0", "split": split})
    files = {"en": root / "en.jsonl", "es": root / "es.jsonl", "pair": root / "s2ut" / "pair.jsonl"}
    for key, rows in (("en", en), ("es", es), ("pair", pairs)):
        files[key].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                              encoding="utf-8")
    return files


def test_multilingual_pipeline_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    monkeypatch.setattr(teacher, "HubertTeacher", FakeTeacher)
    m = write_corpus(tmp_path)
    mono = f"{m['en']},{m['es']}"
    km = tmp_path / "units"

    # 1. Balanced K-means with the multilingual teacher, under the stage's file name.
    fit = ["--fit-kmeans", "--teacher", "mhubert-vp", "--k", "8", "--max-frames", "400",
           "--manifest", mono, "--out-dir", str(km), "--device", "cpu"]
    assert extract_units.main(fit + ["--balance-langs", "en,de", "--default-lang", "en"]) == 1
    assert "no manifest rows tagged de" in capsys.readouterr().err
    assert extract_units.main(fit + ["--balance-langs", "en,es"]) == 1      # LJ rows carry no lang
    assert extract_units.main(fit + ["--balance-langs", "en,es", "--default-lang", "en"]) == 0
    centroids = np.load(km / "mhubert_vp_en_es_fr_de_km1000.npy")
    assert centroids.shape == (8, DIM) and not (km / "km100_centroids.npy").exists()
    kmeta = json.loads((km / "mhubert_vp_en_es_fr_de_km1000.json").read_text(encoding="utf-8"))
    assert kmeta["teacher"] == "mhubert-vp" and kmeta["layer"] == 11 and kmeta["balanced"] is True
    assert kmeta["groups"]["en"]["frames_used"] == kmeta["groups"]["es"]["frames_used"] > 0
    assert set(kmeta["languages"]) == {"en", "es"} and kmeta["languages"]["en"]["utterances"] == 4

    # 2. Label the vocoder corpora: int16 ids + causal prosody + per-speaker statistics.
    lab = tmp_path / "lab"
    label = ["--tokenizer", "teacher-mhubert-km1000", "--centroids", str(km), "--device", "cpu"]
    assert extract_units.main(label + ["--manifest", mono, "--default-lang", "en", "--prosody",
                                       "--out-dir", str(lab / "mono")]) == 0
    meta, rows = units.read_units_dir(lab / "mono")
    assert meta["tokenizer"] == "teacher-mhubert-km1000" and meta["vocab"] == 8
    assert meta["noncommercial"] is False and "MIT" in meta["license"]
    assert meta["prosody"]["columns"] == ["logf0", "voicing", "aperiodicity", "log_energy"]
    assert meta["latents"] is None and len(rows) == 8
    by_id = {r["utt_id"]: r for r in rows}
    assert by_id["en0"]["lang"] == "en" and by_id["es0"]["lang"] == "es"
    from fvml import features
    for r in rows:
        ids = np.load(lab / "mono" / r["units"])
        pros = np.load(lab / "mono" / r["prosody"])
        assert ids.dtype == np.int16 and ids.min() >= 0 and ids.max() < 8
        assert pros.dtype == np.float32 and pros.shape == (len(ids), 4) == (r["n_frames"], 4)
    wav = audio.load(tmp_path / by_id["es1"]["path"])
    ref = features.extract(ce.pad_to_frames(wav))
    pros = np.load(lab / "mono" / by_id["es1"]["prosody"])
    col = {name: i for i, name in enumerate(features.FEATURES)}
    for j, name in enumerate(("logf0_held", "voicing", "aperiodicity", "log_energy")):
        np.testing.assert_allclose(pros[:, j], ref[:, col[name]], rtol=0, atol=1e-5, err_msg=name)
    stats = json.loads((lab / "mono" / units.PROSODY_STATS_FILE).read_text(encoding="utf-8"))
    assert set(stats) == {"lj", "css10_es"} and stats["css10_es"]["lang"] == "es"
    voiced = np.concatenate([np.log(f[f[:, 0] > 0, 0]) for f in (
        features.extract(audio.load(tmp_path / by_id[f"es{i}"]["path"])) for i in range(4))])
    assert stats["css10_es"]["voiced_frames"] == len(voiced) > 0
    assert stats["css10_es"]["logf0_mean"] == pytest.approx(float(voiced.mean()), abs=1e-5)
    assert stats["lj"]["logf0_std"] > 0 and stats["lj"]["energy_std"] > 0

    # 3. Paired S2UT targets: --side is required; prosody can be added later without
    #    running the teacher again; a rerun that drops it is refused.
    pair_tgt = lab / "pair_tgt"
    paired = label + ["--manifest", str(m["pair"]), "--out-dir", str(pair_tgt)]
    assert extract_units.main(paired) == 1
    assert "--side" in capsys.readouterr().err
    assert extract_units.main(paired + ["--side", "tgt"]) == 0
    calls = FakeTeacher.calls
    assert extract_units.main(paired + ["--side", "tgt", "--prosody"]) == 0
    assert FakeTeacher.calls == calls
    pmeta, prows = units.read_units_dir(pair_tgt)
    assert pmeta["prosody"] and len(prows) == 3
    for r in prows:
        assert (r["side"], r["lang"], r["pair"]) == ("tgt", "en", "es_en")
        assert r["speaker"] == "cvss-c-es_en_tgt_en" and r["path"].endswith("_tgt.flac")
        assert np.load(pair_tgt / r["prosody"]).shape == (r["n_frames"], 4)
    assert extract_units.main(paired + ["--side", "tgt"]) == 1
    assert extract_units.main(paired + ["--side", "tgt", "--prosody", "--prosody-lookahead", "1"]) == 1
    assert extract_units.main(label + ["--manifest", str(m["pair"]), "--side", "src",
                                       "--out-dir", str(lab / "pair_src")]) == 0
    assert extract_units.main(paired + ["--side", "src", "--save-latents",
                                        "--out-dir", str(lab / "no")]) == 1   # teachers have no latents

    # 4. The dual-output student on balanced source speech, per-language validation.
    all_manifests = f"{mono},{m['pair']}"
    all_units = f"{lab / 'mono'},{lab / 'pair_src'}"
    train = ["--train-manifest", all_manifests, "--train-units", all_units,
             "--val-manifest", all_manifests, "--val-units", all_units, "--side", "src",
             "--default-lang", "en", "--train-subsets", "train", "--val-subsets", "test,dev",
             "--latent-dim", "16", "--channels", "4,4,8,8,8", "--val-every", "2",
             "--batch-size", "2", "--crop-seconds", "0.4", "--workers", "0", "--device", "cpu",
             "--warmup", "1", "--log-every", "1", "--no-augment", "--steps", "2",
             "--runs-dir", str(tmp_path / "runs"), "--ckpt-dir", str(tmp_path / "ckpt")]
    assert tce.main(train + ["--run-name", "bad", "--lang-ratios", "en,es,fr"]) == 1
    assert "no training items for language(s) fr" in capsys.readouterr().err
    assert tce.main(train + ["--run-name", "stu", "--lang-ratios", "en,es"]) == 0
    assert "language sampling: en 50% of draws (3 utterances), es 50% of draws (5 utterances)" \
        in capsys.readouterr().out
    ck_path = tmp_path / "ckpt" / "stu" / "final.pt"
    ck = torch.load(ck_path, weights_only=True)
    cfg = json.loads(ck["config"])
    assert (cfg["latent_dim"], cfg["vocab"]) == (16, 8)
    assert "noncommercial" not in json.loads(ck["labels_meta"])
    val = json.loads(ck["val"])
    assert set(val["by_lang"]) == {"en", "es"} and val["step"] == 2
    assert val["by_lang"]["en"]["utts"] == 1 and val["by_lang"]["es"]["utts"] == 2   # es test + dev pair
    for s in (val, *val["by_lang"].values()):
        assert len(s["acc"]) == len(s["top5"]) == 3
        assert all(0.0 <= a <= t <= 1.0 for a, t in zip(s["acc"], s["top5"]))
    assert set(val["dod"]) == {"L0", "L1", "L2"}

    # 5. Student latents for the S2UT manifests: what the streaming encoder emits.
    stu = tmp_path / "stu_units"
    latents = ["--tokenizer", "student-L1", "--checkpoint", str(ck_path), "--device", "cpu",
               "--manifest", str(m["pair"]), "--side", "src"]
    assert extract_units.main(latents + ["--save-latents", "--out-dir", str(stu / "f16")]) == 0
    assert extract_units.main(latents + ["--save-latents", "f32", "--out-dir", str(stu / "f32")]) == 0
    smeta, srows = units.read_units_dir(stu / "f16")
    assert smeta["latents"]["dim"] == 16 and smeta["latents"]["dtype"] == "float16"
    assert smeta["tokenizer"] == "student-L1" and smeta["label_space"] == meta["label_space"]
    model, _ = ce.ContentEncoder.from_checkpoint(ck_path)
    model.eval()
    for r in srows:
        wav = torch.from_numpy(ce.pad_to_frames(audio.load(tmp_path / r["path"])))[None]
        state, stream = model.init_state(1), []
        with torch.no_grad():
            for i in range(0, wav.shape[-1], FS):
                out, state = model.step(wav[:, i:i + FS], state)
                stream.append(out["latents"])
        stream = torch.cat(stream, 1)[0].numpy()
        f16 = np.load(stu / "f16" / r["latents"])
        f32 = np.load(stu / "f32" / r["latents"])
        assert f16.dtype == np.float16 and f32.dtype == np.float32
        assert f16.shape == f32.shape == (r["n_frames"], 16) == stream.shape
        assert np.abs(f32 - stream).max() <= 1e-5
        assert np.abs(f16.astype(np.float32) - stream).max() <= 5e-3
    assert extract_units.main(latents + ["--out-dir", str(stu / "f16")]) == 1   # layout is fixed

    # 6. The multilingual vocoder: language-ratio sampling, held-out subsets, `lang` per voice.
    info = {}
    items, _, spk = tuv.join_vocoder_items([mono], [str(lab / "mono")], None, default_lang="en",
                                           subsets="train", info=info)
    assert len(items) == 6 and spk == ["lj", "css10_es"]
    assert info["speaker_lang"] == {"lj": "en", "css10_es": "es"}
    assert info["speaker_meta"]["css10_es"] == {"lang": "es", "source": "css10-es",
                                                "license": "Apache-2.0 / CC0-1.0 (LibriVox)"}
    with pytest.raises(ValueError, match="several languages"):
        mixed = tmp_path / "mixed.jsonl"
        rows_en = [json.loads(line) for line in m["en"].read_text(encoding="utf-8").splitlines()]
        rows_en[0]["lang"] = "fr"
        mixed.write_text("".join(json.dumps(r) + "\n" for r in rows_en), encoding="utf-8")
        tuv.join_vocoder_items([str(mixed)], [str(lab / "mono")], None, default_lang="en")
    voc = ["--train-manifest", mono, "--train-units", str(lab / "mono"), "--exclude-targets", "",
           "--train-subsets", "train", "--upsample-initial-channel", "32",
           "--segment-samples", "1280", "--batch-size", "2", "--workers", "0", "--device", "cpu",
           "--log-every", "1", "--steps", "2", "--runs-dir", str(tmp_path / "runs"),
           "--ckpt-dir", str(tmp_path / "ckpt")]
    assert tuv.main(voc + ["--run-name", "novoc", "--lang-ratios", "en=0.4,es=0.6"]) == 1
    assert "--default-lang" in capsys.readouterr().err
    assert tuv.main(voc + ["--run-name", "voc", "--lang-ratios", "en=0.4,es=0.6",
                           "--default-lang", "en"]) == 0
    assert "language sampling: en 40% of draws (3 utterances, 1 voice(s)), es 60% of draws " \
           "(3 utterances, 1 voice(s))" in capsys.readouterr().out
    vck = torch.load(tmp_path / "ckpt" / "voc" / "final.pt", weights_only=True)
    assert json.loads(vck["lang_ratios"]) == {"en": 0.4, "es": 0.6}
    assert vck["model"]["unit_emb.weight"].shape == (8, 128)
    voices = json.loads((tmp_path / "ckpt" / "voc" / "voices.json").read_text(encoding="utf-8"))
    lj, css = voices["voices"]
    assert (lj["name"], lj["lang"]) == ("lj", "en") and (css["name"], css["lang"]) == ("css10_es", "es")
    # Row metadata fills gaps only: LJSpeech keeps its built-in display, licence and source.
    assert lj["display"] == "LJSpeech (Linda Johnson)" and lj["license"] == "Public domain"
    assert lj["source"] == "LJSpeech 1.1"
    assert css["license"] == "Apache-2.0 / CC0-1.0 (LibriVox)" and css["source"] == "css10-es"
    gen, _ = vc.UnitVocoder.from_checkpoint(tmp_path / "ckpt" / "voc" / "final.pt")
    fc.validate_voices(voices, gen.contract(1))
    assert fx.voices_json(gen, vck) == voices                     # the export reproduces `lang`

    # 7. The Stage 05 interface: train_s2ut.py reads exactly these directories (source
    #    latents --side src, target units --side tgt) plus the silence bank.
    import train_s2ut
    from fvml import translator as tr
    bank = tmp_path / "stu_units" / "silence.latents.npy"
    assert extract_units.main(["--silence-latents", str(bank), "--checkpoint", str(ck_path),
                               "--device", "cpu", "--silence-seconds", "2"]) == 0
    s2ut_args = types.SimpleNamespace(manifest=[str(m["pair"])], src_latents=[str(stu / "f16")],
                                      tgt_units=[str(pair_tgt)], max_seconds=30.0,
                                      include_noncommercial=False)
    tcfg = tr.tiny_config(latent_dim=16, vocab=8)
    items, space = train_s2ut.join_items(s2ut_args, tcfg, "train", log=lambda *_: None)
    assert [it["id"] for it in items] == ["p0", "p1"] and space["vocab"] == 8
    assert space["label_space"] == pmeta["label_space"]
    dev, _ = train_s2ut.join_items(s2ut_args, tcfg, "dev", log=lambda *_: None)
    assert [it["id"] for it in dev] == ["p2"]
    fake_tok = types.SimpleNamespace(encode=lambda text, lang: [1, 2, 3])
    ds = train_s2ut.S2utDataset(items, tcfg, fake_tok, silence=np.load(bank), tail_p=1.0)
    sample = ds[0]
    by_utt = {r["utt_id"]: r for r in srows}
    n_src = by_utt["p0"]["n_frames"]
    assert sample["latents"].shape[1] == 16 and len(sample["latents"]) > n_src   # padded + tail
    assert sample["tail_chunks"] > 0
    np.testing.assert_allclose(sample["latents"][:n_src],
                               np.load(stu / "f16" / by_utt["p0"]["latents"]).astype(np.float32))
    assert len(sample["units"]) == {r["utt_id"]: r for r in prows}["p0"]["n_frames"]
    with pytest.raises(ValueError, match="dim 16"):               # a D=64 translator refuses them
        train_s2ut.join_items(s2ut_args, tr.tiny_config(latent_dim=64, vocab=8), "train")


def save_student(path, cfg, labels_meta=None, seed=0):
    """A train_content_encoder.py-shaped checkpoint of a random-init student."""
    torch.manual_seed(seed)
    model = ce.ContentEncoder(cfg).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": json.dumps(cfg.to_dict()), "step": 0,
                "labels_meta": json.dumps(labels_meta or {"tokenizer": "teacher-mhubert-km1000"})},
               path)
    return model


def test_silence_latent_bank_for_train_s2ut(tmp_path, monkeypatch, capsys):
    """extract_units.py --silence-latents writes the [T, D] float32 bank that
    train_s2ut.py --silence-latents loads: the student's latents on room tone."""
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    cfg = dataclasses.replace(TINY_ENC, latent_dim=16)
    ck = tmp_path / "ckpt" / "stu.pt"
    model = save_student(ck, cfg)
    out = tmp_path / "units" / "student" / "silence.latents.npy"
    base = ["--silence-latents", str(out), "--checkpoint", str(ck), "--device", "cpu",
            "--silence-seconds", "4", "--silence-dbfs=-80,-50"]
    assert extract_units.main(base) == 0
    bank = np.load(out)
    # 2 s of digital silence + one 2 s noise-floor segment = 4 s = 200 frames.
    assert bank.dtype == np.float32 and bank.shape == (200, 16)
    # What train_s2ut.py checks before it cuts tails from the bank.
    assert bank.ndim == 2 and len(bank) >= 4 and np.isfinite(bank).all()
    # It is the student's own `latents` port on the same deterministic room tone.
    wav = extract_units.silence_audio(4.0, (-80.0, -50.0), np.random.default_rng(0))
    assert len(wav) == 4 * 16000 and not wav[:32000].any()
    rms_db = 20 * np.log10(np.sqrt(np.mean(wav[32000:].astype(np.float64) ** 2)))
    assert -80.0 - 1e-3 <= rms_db <= -50.0 + 1e-3
    # ... after the receptive field of digital silence: steady state, not the zero-state transient.
    warm = -(-model.past_context_samples() // FS)
    with torch.no_grad():
        ref = model(torch.from_numpy(np.concatenate([np.zeros(warm * FS, np.float32), wav]))[None])
    np.testing.assert_allclose(bank, ref["latents"][0, warm:].numpy(), rtol=0, atol=1e-6)
    with torch.no_grad():
        cold = model(torch.from_numpy(wav)[None])["latents"][0].numpy()
    assert np.abs(cold[0] - bank[0]).max() > 1e-3          # a zero state is not silence
    # Steady-state digital silence is one constant latent; the noise floor is not.
    np.testing.assert_allclose(bank[:100], np.broadcast_to(bank[0], (100, 16)), rtol=0, atol=1e-6)
    assert np.abs(bank[100:] - bank[0]).max() > 0.0     # (a random init barely hears -80 dBFS)
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert (side["frames"], side["dim"], side["levels_dbfs"]) == (200, 16, [-80.0, -50.0])
    assert side["noncommercial"] is False and side["room_tone_clips"] == 0
    assert side["warmup_frames_dropped"] == warm

    # Recorded room tone (a manifest) is appended, scaled into the same level range.
    tone = tmp_path / "tone.flac"
    sf.write(tone, 0.2 * np.random.default_rng(3).standard_normal(16000).astype(np.float32), 16000)
    (tmp_path / "tone.jsonl").write_text(json.dumps(
        {"utt_id": "tone0", "path": str(tone), "duration": 1.0, "subset": "train"}) + "\n",
        encoding="utf-8")
    out2 = tmp_path / "units" / "student" / "silence_rec.latents.npy"
    assert extract_units.main([*base[:1], str(out2), *base[2:],
                               "--manifest", str(tmp_path / "tone.jsonl")]) == 0
    assert np.load(out2).shape == (250, 16)
    assert json.loads(out2.with_suffix(".json").read_text(encoding="utf-8"))["room_tone_clips"] == 1

    # Inputs that cannot make a bank.
    capsys.readouterr()
    assert extract_units.main(["--silence-latents", str(out), "--device", "cpu"]) == 1
    assert "--checkpoint" in capsys.readouterr().err
    assert extract_units.main([*base[:1], str(tmp_path / "bank.txt"), *base[2:]]) == 1
    with pytest.raises(SystemExit):
        extract_units.main([*base[:-1], "--silence-dbfs=-10,10"])                  # levels must be below 0 dBFS
    # A student distilled from non-commercial labels: opt-in and an `nc` directory.
    nc_ck = tmp_path / "ckpt" / "nc.pt"
    save_student(nc_ck, cfg, {"tokenizer": "teacher-mhubert147-km1000", "noncommercial": True,
                              "license": "CC-BY-NC-SA-4.0"})
    nc = ["--checkpoint", str(nc_ck), "--device", "cpu", "--silence-seconds", "2"]
    assert extract_units.main(["--silence-latents", str(out), *nc]) == 1
    assert "non-commercial" in capsys.readouterr().err
    assert extract_units.main(["--silence-latents", str(out), *nc, "--include-noncommercial"]) == 1
    assert "'nc' directory" in capsys.readouterr().err
    nc_out = tmp_path / "nc" / "silence.latents.npy"
    assert extract_units.main(["--silence-latents", str(nc_out), *nc, "--include-noncommercial"]) == 0
    assert json.loads(nc_out.with_suffix(".json").read_text(encoding="utf-8"))["noncommercial"]


def test_extract_accepts_callers_that_predate_the_m3_options():
    """extract() reads the Stage 03 options through with_defaults(), so a
    namespace built for the M2 extractor still runs (and keeps its own values)."""
    old = types.SimpleNamespace(out_dir="x", tokenizer="fake", save_logits=True, limit=3)
    args = extract_units.with_defaults(old)
    assert (args.save_logits, args.limit, args.out_dir) == (True, 3, "x")
    assert args.include_noncommercial is False and args.prosody is False
    assert args.save_latents is None and args.teacher_checkpoint is None
    assert args.perturb_fraction == 0.0 and not hasattr(old, "prosody")   # the caller's is untouched
    # Every default equals the CLI's own.
    parsed = {}

    def capture(a, rows, manifests, log=print):
        parsed.update(vars(a))
        return 0

    import unittest.mock
    with unittest.mock.patch.object(extract_units, "extract", capture), \
            unittest.mock.patch.object(units, "read_manifests",
                                       lambda m: ([{"utt_id": "u", "path": "u.wav"}], m)):
        assert extract_units.main(["--tokenizer", "teacher-km100", "--manifest", "m.jsonl",
                                   "--out-dir", "o", "--device", "cpu"]) == 0
    for key, value in extract_units.EXTRACT_DEFAULTS.items():
        assert parsed[key] == value, key


def test_noncommercial_teacher_is_isolated(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLUXVOICE_DATA", str(tmp_path))
    monkeypatch.setattr(teacher, "HubertTeacher", FakeTeacher)
    m = write_corpus(tmp_path)
    fit = ["--fit-kmeans", "--teacher", "mhubert-147", "--k", "8", "--manifest", str(m["es"]),
           "--device", "cpu"]
    assert extract_units.main(fit + ["--out-dir", str(tmp_path / "nc" / "km")]) == 1
    assert "include-noncommercial" in capsys.readouterr().err
    assert extract_units.main(fit + ["--include-noncommercial",
                                     "--out-dir", str(tmp_path / "km")]) == 1
    assert "'nc' directory" in capsys.readouterr().err
    assert extract_units.main(fit + ["--include-noncommercial",
                                     "--out-dir", str(tmp_path / "nc" / "km")]) == 0
    assert (tmp_path / "nc" / "km" / "mhubert147_en_es_fr_de_km1000.npy").is_file()
    label = ["--tokenizer", "teacher-mhubert147-km1000", "--centroids", str(tmp_path / "nc" / "km"),
             "--manifest", str(m["es"]), "--device", "cpu"]
    out = tmp_path / "nc" / "units"
    assert extract_units.main(label + ["--out-dir", str(out)]) == 1
    assert extract_units.main(label + ["--include-noncommercial", "--out-dir",
                                       str(tmp_path / "plain")]) == 1
    assert extract_units.main(label + ["--include-noncommercial", "--out-dir", str(out)]) == 0
    meta = units.read_meta(out)
    assert meta["noncommercial"] is True and "NC" in meta["license"]

    # Nothing trains on, probes or mixes those units without the flag.
    train = ["--train-manifest", str(m["es"]), "--train-units", str(out), "--channels",
             "4,4,8,8,8", "--batch-size", "2", "--crop-seconds", "0.4", "--workers", "0",
             "--device", "cpu", "--steps", "1", "--warmup", "1", "--no-augment", "--run-name", "nc",
             "--runs-dir", str(tmp_path / "runs"), "--ckpt-dir", str(tmp_path / "ckpt")]
    assert tce.main(train) == 1
    assert "non-commercial" in capsys.readouterr().err
    assert tce.main(train + ["--include-noncommercial"]) == 0
    ck = torch.load(tmp_path / "ckpt" / "nc" / "final.pt", weights_only=True)
    assert json.loads(ck["labels_meta"])["noncommercial"] is True       # the taint follows the weights
    student = units.StudentTokenizer(tmp_path / "ckpt" / "nc" / "final.pt", 1)
    assert student.meta["noncommercial"] is True
    with pytest.raises(teacher.LicenseError):
        units.check_license(student.meta)
    with pytest.raises(fx.ExportError, match="non-commercial"):
        cli.load_encoder(tmp_path / "ckpt" / "nc" / "final.pt", "student-L1", "x")
    _, info = cli.load_encoder(tmp_path / "ckpt" / "nc" / "final.pt", "student-L1", "x",
                               include_noncommercial=True)
    assert info["license"].endswith(cli.NC_NOTE)
    voc = ["--train-manifest", str(m["es"]), "--train-units", str(out), "--exclude-targets", "",
           "--upsample-initial-channel", "32", "--segment-samples", "1280", "--batch-size", "2",
           "--workers", "0", "--device", "cpu", "--steps", "1", "--run-name", "ncvoc",
           "--runs-dir", str(tmp_path / "runs"), "--ckpt-dir", str(tmp_path / "ckpt")]
    assert tuv.main(voc) == 1
    assert probe_units.main(["--units", str(out), "--speaker-probe"]) == 1
    assert "non-commercial" in capsys.readouterr().err
    # A commercial and a non-commercial stream are never mixed, even with the same ids.
    clean = dict(meta, noncommercial=False)
    (tmp_path / "clean").mkdir()
    (tmp_path / "clean" / units.META_FILE).write_text(json.dumps(clean), encoding="utf-8")
    with pytest.raises(units.UnitsMismatch, match="non-commercial"):
        units.read_units_dirs([str(tmp_path / "clean"), str(out)])
    with pytest.raises(units.UnitsMismatch, match="non-commercial"):     # the vocoder's own reader
        tuv.read_all_units_dirs([str(tmp_path / "clean"), str(out)])


# -- multilingual vocoder: K=1000, lang metadata, no language input ------------------
def test_vocoder_has_no_language_input():
    cfg = vc.VocoderConfig(vocab=vc.MULTILINGUAL_VOCAB, n_speakers=14)
    gen = vc.UnitVocoder(cfg).eval()
    assert vc.MULTILINGUAL_VOCAB == 1000 and tuple(gen.unit_emb.weight.shape) == (1000, 128)
    assert 500_000 <= gen.num_params() <= 1_500_000              # still V2-class
    # No language reaches the generator: not as a tensor, a config field or a contract input.
    wanted = ["units", "logf0", "voicing", "aperiodicity", "energy", "spk_emb"]
    assert list(inspect.signature(gen.forward).parameters) == wanted
    assert list(inspect.signature(gen.step).parameters) == wanted + ["state"]
    assert not any("lang" in f.name for f in dataclasses.fields(vc.VocoderConfig))
    assert not any("lang" in name for name, _ in gen.named_parameters())
    c = gen.contract(4)
    assert c.vocab == 1000 and c.cond_inputs == fc.COND_INPUTS
    assert [t.name for t in c.expected_inputs()] == wanted
    assert not any("lang" in n for n in fc.COND_INPUTS)

    speakers = [f"v{i}" for i in range(14)]
    stats = dict.fromkeys(speakers, STATS)
    meta = {"v0": {"lang": "en"}, "v1": {"lang": "de", "license": "CC BY 4.0", "source": "cml-tts-de"}}
    data = gen.voices_json(speakers, stats, meta)
    assert data["voices"][0]["lang"] == "en" and data["voices"][1]["lang"] == "de"
    assert "lang" not in data["voices"][2]                        # M2 voices.json is unchanged
    fc.validate_voices(data, c)
    # The same units and embedding give the same audio whatever the voice's language tag.
    torch.manual_seed(0)
    u = torch.randint(0, 1000, (1, 6))
    cond = [torch.full((1, 6), 5.0), torch.ones(1, 6), torch.full((1, 6), 0.1), torch.full((1, 6), -6.0)]
    emb = torch.tensor([data["voices"][0]["embedding"]])
    with torch.no_grad():
        assert torch.equal(gen(u, *cond, emb), gen(u, *cond, torch.tensor([0])))
    with pytest.raises(fc.ContractError, match="language code"):
        fc.validate_voices(gen.voices_json(speakers, stats, {"v0": {"lang": "English"}}), c)


# -- contract-v2 export: the dual-output encoder ------------------------------------
@pytest.fixture(scope="module")
def tiny_encoder():
    torch.manual_seed(0)
    return ce.ContentEncoder(TINY_ENC).eval()


def test_contract_v2_dual_output_export(tiny_encoder, tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    head = 1
    c1 = fx.encoder_contract(tiny_encoder, head, 2, "MIT")
    c2 = fx.encoder_contract(tiny_encoder, head, 2, "MIT", contract_version=2)
    assert (c1.contract_version, c1.latent_dim) == (1, 0)          # the default is still v1
    assert (c2.contract_version, c2.latent_dim, c2.lookahead_frames) == (2, 8, 1)
    assert c2.states == c1.states
    assert [t.name for t in c2.expected_outputs()] == ["units", "latents"]
    g1 = fx.encoder_graph(tiny_encoder, head, 2)
    g2 = fx.encoder_graph(tiny_encoder, head, 2, contract_version=2)
    assert g1.output_names()[:2] == ["units", "logits"]
    assert g2.output_names()[:3] == ["units", "latents", "logits"] and g2.debug_outputs == ["logits"]
    assert g2.states == g1.states
    with pytest.raises(fx.ExportError, match="contract_version 3"):
        fx.encoder_contract(tiny_encoder, head, 2, "MIT", contract_version=3)

    path = tmp_path / "encoder_s2.onnx"
    fx.export_graph(g2, path, "torchscript")
    fx.finalize(path, c2, drop_outputs=g2.debug_outputs)
    loaded, proto = fc.load_onnx(path)
    assert loaded == c2
    assert [o.name for o in proto.graph.output][:2] == ["units", "latents"]
    _, outs = fc.graph_io(proto)
    specs = {t.name: t for t in outs}
    assert specs["units"].dtype == "i64" and tuple(specs["units"].shape) == (1, 2)
    assert specs["latents"].dtype == "f32" and tuple(specs["latents"].shape) == (1, 2, 8)

    wav = speech(3.0)
    stream = fx.OrtStream(path)
    out = fx.encoder_chunked_all(stream, wav)
    with torch.no_grad():
        ref = tiny_encoder(torch.from_numpy(wav)[None])
    assert out["latents"].shape == (150, 8) and out["units"].shape == (150,)
    assert np.abs(out["latents"] - ref["latents"][0].numpy()).max() <= 1e-5
    assert np.array_equal(out["units"], ref["logits"][head][0].argmax(-1).numpy())
    units_only, logits = fx.encoder_chunked(stream, wav)           # the M2 helper still works
    assert np.array_equal(units_only, out["units"]) and logits is None
    r = fx.encoder_parity(tiny_encoder, head, path, wav)
    assert r["pass"] and r["latents_pass"] and r["unit_agreement"] == 1.0
    assert r["max_abs_latents"] <= fx.LATENT_RTOL * r["latents_rms"] and r["latents_rms"] > 0.5
    # A drifting latents port fails parity even when every unit still agrees.
    real = fx.encoder_offline_latents
    monkeypatch.setattr(fx, "encoder_offline_latents", lambda model, w: real(model, w) + 1e-2)
    bad = fx.encoder_parity(tiny_encoder, head, path, wav)
    assert bad["unit_agreement"] == 1.0 and not bad["latents_pass"] and not bad["pass"]
    monkeypatch.undo()

    # A v1 file has no latents output and its parity report keeps the M2 keys only.
    p1 = tmp_path / "encoder_v1.onnx"
    fx.export_graph(g1, p1, "torchscript")
    fx.finalize(p1, c1, drop_outputs=g1.debug_outputs)
    assert [o.name for o in fc.load_onnx(p1)[1].graph.output if not o.name.startswith("state_out.")] \
        == ["units"]
    assert "max_abs_latents" not in fx.encoder_parity(tiny_encoder, head, p1, wav)

    # The vocoder's v2 contract changes the stamp only, and the pair/bundle rules accept it.
    torch.manual_seed(1)
    voc = vc.UnitVocoder(TINY_VOC).eval()
    v1, v2 = fx.vocoder_contract(voc, {}, 2), fx.vocoder_contract(voc, {}, 2, contract_version=2)
    assert v1.contract_version == 1 and v2 == dataclasses.replace(v1, contract_version=2)
    fc.validate_pair(c2, v2)
    fc.validate_pair(c2, v1)
    assert fc.algorithmic_latency(c2, v2) == (2 + 1) * FS


def test_cli_exports_a_v2_bundle(tiny_encoder, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.save({"model": tiny_encoder.state_dict(), "config": json.dumps(TINY_ENC.to_dict()),
                "step": 3, "labels_meta": json.dumps({"tokenizer": "teacher-mhubert-km1000"})},
               tmp_path / "enc.pt")
    torch.manual_seed(1)
    voc = vc.UnitVocoder(TINY_VOC).eval()
    with torch.no_grad():
        for p in voc.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.3)
    torch.save({"model": voc.state_dict(), "config": json.dumps(TINY_VOC.to_dict()),
                "speakers": ["lj", "css10_es"], "speaker_stats": {"lj": STATS, "css10_es": STATS},
                "speaker_metadata": {"lj": {"lang": "en"}, "css10_es": {"lang": "es"}},
                "step": 3}, tmp_path / "voc.pt")
    wavs = []
    for i in range(3):
        wavs.append(str(tmp_path / f"utt{i}.wav"))
        audio.save(wavs[-1], speech(2.0, seed=i))
    bundle = tmp_path / "models" / "m3"
    common = ["--bundle", str(bundle), "--frames-per-step", "1", "4", "--exporter", "torchscript",
              "--contract-version", "2", "--parity-audio", *wavs, "--parity-seconds", "5",
              "--golden-seconds", "2", "--cost-steps", "5"]
    assert cli.main(["--role", "encoder", "--checkpoint", str(tmp_path / "enc.pt"),
                     "--head", "student-L1", "--license", "MIT", *common]) == 0
    assert cli.main(["--role", "vocoder", "--checkpoint", str(tmp_path / "voc.pt"), *common]) == 0
    report = json.loads((bundle / "export_report.json").read_text(encoding="utf-8"))
    enc = report["roles"]["encoder"]
    assert enc["contract_version"] == 2 and report["roles"]["vocoder"]["contract_version"] == 2
    for v in ("s1", "s4"):
        c = fx.read_contract(bundle / f"encoder_{v}.onnx")
        assert (c.contract_version, c.latent_dim) == (2, 8)
        assert fx.read_contract(bundle / f"vocoder_{v}.onnx").contract_version == 2
        assert enc["parity"][v]["pass"] and enc["parity"][v]["latents_pass"]
        assert enc["files"][v]["contract"]["latent_dim"] == 8
    assert set(report["pairs"]) == {"s1", "s4"} and set(report["chain"]) == {"s1", "s4"}
    voices = json.loads((bundle / "voices.json").read_text(encoding="utf-8"))
    assert [v["lang"] for v in voices["voices"]] == ["en", "es"]
    assert "Δlatents" in (bundle / "MODEL_CARD.md").read_text(encoding="utf-8")
    # Re-exporting the same checkpoint as v1 is a different artefact: v2 variants do not linger.
    assert cli.main(["--role", "encoder", "--checkpoint", str(tmp_path / "enc.pt"), "--head", "1",
                     "--license", "MIT", "--bundle", str(bundle), "--frames-per-step", "1",
                     "--exporter", "torchscript", "--skip-parity"]) == 0
    assert not (bundle / "encoder_s4.onnx").exists()
    assert fx.read_contract(bundle / "encoder_s1.onnx").contract_version == 1


# -- multilingual probe ---------------------------------------------------------------
def textgrid(phones, step=0.1):
    body = "".join(f'intervals [{i + 1}]:\n xmin = {i * step}\n xmax = {(i + 1) * step}\n'
                   f' text = "{p}"\n' for i, p in enumerate(phones))
    end = len(phones) * step
    return ('File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\n'
            f'xmax = {end}\ntiers? <exists>\nsize = 1\nitem []:\nitem [1]:\n'
            f'class = "IntervalTier"\nname = "phones"\nxmin = 0\nxmax = {end}\n'
            f'intervals: size = {len(phones)}\n{body}')


def test_probe_reports_per_language_and_reference_ratio(tmp_path, capsys):
    (tmp_path / "align").mkdir()
    utts = [("u0", "a", "en", ["sil", "AH", "B", "sil"]), ("u1", "a", "en", ["B", "AH", "sil", "B"]),
            ("clip2", "b", "es", ["sil", "O", "R", "sil"]), ("clip3", "b", "es", ["R", "O", "O", "sil"])]
    code = {"sil": 7, "AH": 3, "B": 9, "O": 4, "R": 5}
    dirs = {}
    for name in ("balanced", "merged", "const"):
        d = tmp_path / name
        d.mkdir()
        rows = []
        for utt, spk, lang, phones in utts:
            (tmp_path / "align" / f"{utt}.TextGrid").write_text(textgrid(phones), encoding="utf-8")
            labels = unit_probes.frame_labels(unit_probes.read_textgrid(textgrid(phones))["phones"],
                                              20, 50.0)
            if name == "balanced":
                ids = [code[p] for p in labels]
            elif name == "merged":            # Spanish phones collapse into one unit
                ids = [code[p] if lang == "en" else 1 for p in labels]
            else:
                ids = [1] * 20
            # Paired-style ids: the TextGrid is found through the audio file's name.
            uid = utt if utt.startswith("u") else f"pair_{utt}"
            np.save(d / f"{uid}.npy", np.array(ids, np.int16))
            rows.append({"utt_id": uid, "units": f"{uid}.npy", "n_frames": 20, "speaker": spk,
                         "lang": lang, "path": f"16k/x/{utt}.flac"})
        (d / units.MANIFEST_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows),
                                             encoding="utf-8")
        meta = units.base_meta("teacher-mhubert-km1000", 320, None, 16, f"km16:{name}", False, "", {})
        meta.update(license="MIT (fairseq)", noncommercial=False, source={"teacher": "mhubert-vp"})
        (d / units.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        dirs[name] = str(d)
    out, md = tmp_path / "probe.json", tmp_path / "probe.md"
    args = ["--units", dirs["balanced"], "--units", dirs["merged"], "--units", dirs["const"],
            "--alignments", str(tmp_path / "align"), "--lang-reference", f"en={dirs['balanced']}",
            "--lang-reference", f"es={dirs['balanced']}", "--speaker-probe", "--speakers", "2",
            "--min-utts", "2", "--test-fraction", "0.5", "--out", str(out), "--markdown", str(md)]
    assert probe_units.main(args) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    bal, merged, const = (rep["units"][dirs[k]] for k in ("balanced", "merged", "const"))
    assert set(bal["phones"]["by_lang"]) == {"en", "es"}
    assert bal["phones"]["by_lang"]["es"]["frames"] == 40 and bal["phones"]["utterances"] == 4
    for lang in ("en", "es"):
        assert bal["phones"]["by_lang"][lang]["pnmi"] == pytest.approx(1.0)
        assert bal["phones"]["by_lang"][lang]["phone_purity_ratio_vs_lang_specific"] == pytest.approx(1.0)
    assert bal["phones"]["purity_vs_lang_specific"] == {
        "min_ratio": pytest.approx(1.0), "threshold": 0.90, "missing_languages": [], "pass": True}
    # English is intact in `merged`, Spanish is not: the per-language view is what shows it.
    assert merged["phones"]["by_lang"]["en"]["phone_purity"] == pytest.approx(1.0)
    es_ratio = merged["phones"]["by_lang"]["es"]["phone_purity_ratio_vs_lang_specific"]
    assert es_ratio == pytest.approx(merged["phones"]["by_lang"]["es"]["phone_purity"]) and es_ratio < 0.9
    assert merged["phones"]["purity_vs_lang_specific"]["pass"] is False
    assert merged["phones"]["purity_vs_lang_specific"]["min_ratio"] == pytest.approx(es_ratio)
    assert const["phones"]["by_lang"]["en"]["pnmi"] == pytest.approx(0.0, abs=1e-12)
    assert rep["lang_references"]["es"]["frames"] == 40
    assert bal["license"] == "MIT (fairseq)" and bal["teacher"] == "mhubert-vp"
    for e in (bal, merged, const):          # two speakers: chance is 50%, far above the 5% ceiling
        s = e["speaker_id"]
        assert s["max_accuracy"] == 0.05 and s["pass"] == (s["accuracy"] <= 0.05)
    assert const["speaker_id"]["pass"] is False
    table = md.read_text(encoding="utf-8")
    assert table.splitlines()[0].startswith("| Codebook | Teacher | K | Licence | en purity / PNMI | es")
    assert "| `balanced` | mhubert-vp | 16 | MIT (fairseq) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 (pass)" in table
    assert "FAIL" in table.splitlines()[3] and len(table.splitlines()) == 5
    loose = [a for a in args if a not in (str(out), "--out", str(md), "--markdown")]
    assert probe_units.main(loose + ["--max-speaker-accuracy", "1.0", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["units"][dirs["const"]]["speaker_id"]["pass"] is True
    capsys.readouterr()
    assert probe_units.main(["--units", dirs["balanced"], "--alignments", str(tmp_path / "align"),
                             "--lang-reference", "en"]) == 1          # wants LANG=DIR
    assert probe_units.main(["--units", dirs["balanced"], "--speaker-probe", "--speakers", "2",
                             "--min-utts", "2", "--lang-reference", f"en={dirs['balanced']}"]) == 1
    assert probe_units.main(["--units", dirs["balanced"], "--alignments", str(tmp_path / "align"),
                             "--lang-reference", "de=" + dirs["balanced"]]) == 1
    # A language without a reference keeps the definition of done open.
    entry = {"phones": {"by_lang": {"en": {"phone_purity": 0.9, "pnmi": 0.5}}}}
    probe_units.add_purity_ratios(entry, {"en": {"phone_purity": 1.0, "pnmi": 1.0},
                                          "de": {"phone_purity": 1.0, "pnmi": 1.0}})
    assert entry["phones"]["purity_vs_lang_specific"]["missing_languages"] == ["de"]
    assert entry["phones"]["purity_vs_lang_specific"]["pass"] is False
    entry = {"phones": {"by_lang": {"en": {"phone_purity": 0.89, "pnmi": 0.5}}}}
    probe_units.add_purity_ratios(entry, {"en": {"phone_purity": 1.0, "pnmi": 1.0}})
    assert entry["phones"]["purity_vs_lang_specific"]["pass"] is False      # 0.89 < 0.90


# -- scoring hooks --------------------------------------------------------------------
class FakeWhisper:
    """Returns ASCII-folded transcripts and records the language it was forced to."""
    languages = []

    def __init__(self, device="cpu", batch_size=8):
        self.texts = []

    def transcribe(self, waves, language="en", silence_dbfs=None):
        type(self).languages.append(language)
        return ["uber die brucke"] * len(waves)

    def normalize(self, text, language="en"):
        fold = str.maketrans("üÜ", "uU")
        return asr.normalize_text(text, language=language,
                                  english_normalizer=lambda t: t.translate(fold).lower())

    def close(self):
        pass


def write_eval_list(root, n=2, text="Über die Brücke"):
    (root / "src").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        audio.save(root / "src" / f"s{i}.wav", speech(1.0, seed=i))
        rows.append({"utt_id": f"s{i}", "speaker": f"spk{i}", "gender": "F", "duration": 1.0,
                     "path": str(root / "src" / f"s{i}.wav"), "text": text})
    evallist.write(root / "list.tsv", rows, evallist.SOURCE_COLUMNS)
    return root / "list.tsv"


def test_evaluate_scores_in_the_list_language(tmp_path, monkeypatch):
    monkeypatch.setattr(asr, "Whisper", FakeWhisper)
    FakeWhisper.languages = []
    lst = write_eval_list(tmp_path)
    cfg = evaluate.EvalConfig(sources=lst, converted=tmp_path / "src", metrics=("wer",),
                              device="cpu", language="de")
    res = evaluate.evaluate(cfg, tmp_path / "out", log=lambda s: None)
    assert FakeWhisper.languages == ["de"] and res["language"] == "de"
    # German keeps its umlauts: "über"/"brücke" != "uber"/"brucke" -> 2 of 3 words wrong.
    assert res["wer"] == pytest.approx(2 / 3)
    assert "language=de" in res["meta"]["asr"]
    en = evaluate.evaluate(dataclasses.replace(cfg, language="en"), tmp_path / "out_en",
                           log=lambda s: None)
    assert en["wer"] == 0.0 and FakeWhisper.languages[-1] == "en"   # default path unchanged
    assert evaluate.EvalConfig(sources=lst).language == "en"
    with pytest.raises(ValueError, match="unsupported evaluation language"):
        evaluate.evaluate(dataclasses.replace(cfg, language="it"), tmp_path / "x", log=lambda s: None)


def test_resynthesis_and_mode_a_verdicts():
    good = {lang: {"wer": em.RESYNTH_MAX_WER[lang], "utmos": 3.3} for lang in sampling.LANGS}
    assert em.RESYNTH_MAX_WER == {"en": 0.08, "es": 0.08, "fr": 0.10, "de": 0.10}
    assert em.resynth_verdict(good)["pass"] is True                 # the floors themselves pass
    for lang, key, value in (("en", "wer", 0.081), ("es", "wer", 0.09), ("fr", "wer", 0.101),
                             ("de", "utmos", 3.29), ("en", "wer", float("nan")),
                             ("fr", "utmos", None)):
        v = em.resynth_verdict({**good, lang: {**good[lang], key: value}})
        assert v["pass"] is False and v["languages"][lang]["pass"] is False, (lang, key, value)
        assert all(v["languages"][x]["pass"] for x in sampling.LANGS if x != lang)
    assert em.resynth_verdict({"fr": {"wer": 0.09, "utmos": 3.4}})["languages"]["fr"]["pass"]
    partial = em.resynth_verdict({k: good[k] for k in ("en", "es")})
    assert partial["pass"] is False and partial["missing_languages"] == ["fr", "de"]
    assert em.resynth_verdict({})["pass"] is False

    assert em.mode_a_verdict(0.080, 0.070)["pass"] is True          # exactly +1.0 pt
    over = em.mode_a_verdict(0.081, 0.070)
    assert over["abs_pass"] and over["delta_pass"] is False and not over["pass"]
    assert em.mode_a_verdict(0.101, 0.099)["abs_pass"] is False     # absolute 10% ceiling
    assert em.mode_a_verdict(0.05, 0.09)["pass"] is True
    open_ = em.mode_a_verdict(0.05)
    assert open_["abs_pass"] and open_["complete"] is False and open_["pass"] is False
    assert em.mode_a_verdict(float("nan"), 0.07)["pass"] is False


def test_resynth_lists_are_held_out_and_deterministic(tmp_path):
    rows = []
    for spk in ("a", "b", "c"):
        for i in range(6):
            rows.append({"utt_id": f"{spk}{i}", "path": f"x/{spk}{i}.flac", "duration": 4.0 + i,
                         "speaker": spk, "dataset": "css10-es", "lang": "es", "gender": "F",
                         "subset": "test" if i < 4 else "train", "text": f"  frase   {spk}{i} "})
    rows.append({**rows[0], "utt_id": "short", "duration": 1.0})
    rows.append({**rows[0], "utt_id": "notext", "text": "", "speaker": "x"})
    rows.append({**rows[0], "utt_id": "german", "lang": "de", "speaker": "x"})
    got = em.build_source_list(rows, "es", n=6)
    assert [r["utt_id"] for r in got] == ["a0", "b0", "c0", "a1", "b1", "c1"]   # round-robin
    assert got[0]["text"] == "frase a0" and got == em.build_source_list(list(reversed(rows)), "es", n=6)
    assert all(r["utt_id"][1] in "0123" for r in em.build_source_list(rows, "es", n=12))
    with pytest.raises(ValueError, match="only 12 held-out clips"):
        em.build_source_list(rows, "es", n=13)
    refs = em.build_target_rows(rows, "a", n=2, exclude={"a0"})
    assert [r["clip_id"] for r in refs] == ["a1", "a3"] and refs[0]["voice"] == "a"
    with pytest.raises(ValueError, match="voice 'zz'"):
        em.build_target_rows(rows, "zz", n=1)

    manifest = tmp_path / "es.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    args = ["lists", "--manifest", str(manifest), "--lang", "es", "--n", "6", "--voice-refs", "b",
            "--refs", "1", "--base-targets", "", "--out-dir", str(tmp_path / "lists")]
    assert em.main(args) == 0
    listed = evallist.read_sources(tmp_path / "lists" / "resynth_es_6.tsv")
    assert [r["utt_id"] for r in listed] == [r["utt_id"] for r in got]
    targets = evallist.read_targets(tmp_path / "lists" / "targets_m3.tsv")
    assert [(t["voice"], t["clip_id"]) for t in targets] == [("b", "b3")]   # not a source clip
    first = (tmp_path / "lists" / "resynth_es_6.tsv").read_bytes()
    assert em.main(args) == 0 and (tmp_path / "lists" / "resynth_es_6.tsv").read_bytes() == first
    assert em.main(["lists", "--manifest", str(manifest), "--lang", "fr",
                    "--out-dir", str(tmp_path / "lists")]) == 1


def test_resynth_and_mode_a_cli(tmp_path, monkeypatch, capsys):
    torch.manual_seed(0)
    enc = ce.ContentEncoder(TINY_ENC).eval()
    voc = vc.UnitVocoder(TINY_VOC).eval()
    torch.save({"model": enc.state_dict(), "config": json.dumps(TINY_ENC.to_dict())},
               tmp_path / "enc.pt")
    torch.save({"model": voc.state_dict(), "config": json.dumps(TINY_VOC.to_dict()),
                "speakers": ["lj", "css10_es"], "speaker_stats": {"lj": STATS, "css10_es": STATS}},
               tmp_path / "voc.pt")
    lst = write_eval_list(tmp_path, text="hola")
    scores = {"en": 0.05, "es": 0.07, "fr": 0.09, "de": 0.12}
    calls = []

    def fake_evaluate(cfg, out_dir, log=print):
        calls.append(cfg)
        wavs = sorted(pathlib.Path(cfg.converted).glob("*.wav"))
        assert len(wavs) == 2 and all(len(audio.load(w)) == 16000 for w in wavs)
        return {"n": 2, "wer": scores.get(cfg.language, 0.079), "cer": 0.02, "tgt_cos": 0.6,
                "src_cos": 0.1, "utmos": 3.5, "f0_pcc": 0.8, "rtf": 0.2}

    monkeypatch.setattr(evaluate, "evaluate", fake_evaluate)
    base = ["--encoder-checkpoint", str(tmp_path / "enc.pt"),
            "--vocoder-checkpoint", str(tmp_path / "voc.pt"), "--device", "cpu"]
    lists = [x for lang in ("en", "es", "fr", "de") for x in ("--list", f"{lang}={lst}")]
    out = tmp_path / "resynth"
    assert em.main(["resynth", *base, *lists, "--voice", "*=lj", "--voice", "es=css10_es",
                    "--out", str(out)]) == 1                       # German misses its 10%
    rep = json.loads((out / "resynth.json").read_text(encoding="utf-8"))
    assert [c.language for c in calls] == ["en", "es", "fr", "de"]
    assert [c.voice for c in calls] == ["lj", "css10_es", "lj", "lj"]
    assert all(c.metrics == ("wer", "spk", "utmos") and c.limit == 100 for c in calls)
    assert rep["results"]["es"]["voice"] == "css10_es" and rep["results"]["de"]["wer"] == 0.12
    assert rep["verdict"]["languages"]["de"]["pass"] is False
    assert all(rep["verdict"]["languages"][x]["pass"] for x in ("en", "es", "fr"))
    assert "| de | `lj` | 2 | 12.00 (10)" in (out / "resynth.md").read_text(encoding="utf-8")
    scores["de"] = 0.10
    assert em.main(["resynth", *base, *lists, "--voice", "*=lj", "--out", str(out)]) == 0
    assert em.main(["resynth", *base, *lists[:4], "--voice", "*=lj", "--out", str(out)]) == 1
    assert json.loads((out / "resynth.json").read_text(encoding="utf-8"))["verdict"][
        "missing_languages"] == ["fr", "de"]
    capsys.readouterr()
    assert em.main(["resynth", *base, *lists, "--voice", "*=nobody", "--out", str(out)]) == 1
    assert "voice 'nobody'" in capsys.readouterr().err
    assert em.main(["resynth", *base, *lists, "--out", str(out)]) == 1

    # Mode A: existing WAVs (e.g. fv_vc_file output) can be scored without converting.
    calls.clear()
    scores["en"] = 0.079
    conv = out / "en" / "wav"
    ma = ["mode-a", "--list", str(lst), "--voices", "lj", "--converted", f"lj={conv}",
          "--device", "cpu", "--out", str(tmp_path / "mode_a")]
    assert em.main(ma) == 1                                        # no M2 baseline: incomplete
    assert "NOT RUN" in (tmp_path / "mode_a" / "mode_a.md").read_text(encoding="utf-8")
    assert calls[0].language == "en" and calls[0].metrics == evaluate.METRICS
    assert em.main(ma + ["--baseline-wer", "0.070"]) == 0          # 7.9% vs 7.0%: +0.9 pt
    assert em.main(ma + ["--baseline-wer", "0.068"]) == 1          # +1.1 pt
    (tmp_path / "m2_eval.json").write_text(json.dumps({"wer": 0.07}), encoding="utf-8")
    (tmp_path / "m2_gate.json").write_text(json.dumps({
        "decision": {"winner": "student-L1 s1"},
        "candidates": [{"candidate": "student-L0 s1", "wer": 0.2},
                       {"candidate": "student-L1 s1", "wer": 0.072}]}), encoding="utf-8")
    assert em.baseline_wer(tmp_path / "m2_eval.json") == 0.07
    assert em.baseline_wer(tmp_path / "m2_gate.json") == 0.072
    assert em.main(ma + ["--baseline-summary", str(tmp_path / "m2_gate.json")]) == 0
    rep = json.loads((tmp_path / "mode_a" / "mode_a.json").read_text(encoding="utf-8"))
    assert rep["verdict"]["wer_delta"] == pytest.approx(0.007) and rep["verdict"]["complete"]
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    assert em.main(ma + ["--baseline-summary", str(tmp_path / "empty.json")]) == 1
