"""train_s2ut.py (Milestone 3 / Stage 05): curriculum, temperature-balanced
direction sampling, silence tails, the on-disk data join, and a CPU smoke run on
a synthetic corpus (loss goes down, checkpoints resume, a checkpoint streams)."""
import json
import random

import numpy as np
import pytest
import torch

from fvml import s2ut_stream as ss
from fvml import translator as tr
from fvml.waitk_policy import WaitKConfig
import train_s2ut as ts


def test_curriculum_is_full_context_for_30_percent_then_random_k():
    total = 1000
    assert ts.simultaneous_probability(0, total) == 0.0
    assert ts.simultaneous_probability(299, total) == 0.0
    assert ts.simultaneous_probability(325, total) == pytest.approx(0.5)
    assert ts.simultaneous_probability(350, total) == 1.0
    rng = random.Random(0)
    assert all(ts.curriculum_k(s, total, rng) is None for s in range(300))
    ks = [ts.curriculum_k(s, total, rng) for s in range(350, 1000)]
    assert set(ks) == set(range(1, 9))                      # k ~ U{1..8}, never full context
    counts = np.bincount(ks, minlength=9)[1:]
    assert counts.min() > 0.6 * counts.mean()
    ramp = [ts.curriculum_k(s, total, rng) for s in range(300, 350)]
    assert any(k is None for k in ramp) and any(k is not None for k in ramp)


def test_temperature_weights_flatten_without_ignoring_size():
    counts = {"es_en": 80_000, "fr_en": 200_000, "de_en": 120_000, "en_de": 280_000,
              "en_es": 30_000, "en_fr": 30_000}
    prop = ts.temperature_weights(counts, 1.0)
    flat = ts.temperature_weights(counts, 5.0)
    assert prop["en_de"] == pytest.approx(280 / 740)
    assert sum(flat.values()) == pytest.approx(1.0)
    assert flat["en_de"] / flat["en_es"] == pytest.approx((280 / 30) ** 0.2)
    assert prop["en_es"] < flat["en_es"] < 1 / 6 < flat["en_de"] < prop["en_de"]
    assert ts.temperature_weights(counts, 1e9)["en_es"] == pytest.approx(1 / 6, rel=1e-6)


def test_direction_batches_follow_the_weights_and_the_chunk_budget():
    items = [{"direction": "a_b", "src_chunks_max": 10 + i % 30} for i in range(900)] \
        + [{"direction": "c_d", "src_chunks_max": 40 + i % 60} for i in range(100)]
    sampler = ts.DirectionBatches(items, {"a_b": 0.5, "c_d": 0.5}, max_chunks=400, max_batch=12,
                                  seed=3)
    seen = {"a_b": 0, "c_d": 0}
    it = iter(sampler)
    first = [next(it) for _ in range(300)]
    for batch in first:
        assert 1 <= len(batch) <= 12
        assert max(items[i]["src_chunks_max"] for i in batch) * len(batch) <= 400
        for i in batch:
            seen[items[i]["direction"]] += 1
    assert 0.45 < seen["c_d"] / sum(seen.values()) < 0.55    # 10 % of the data, 50 % of the draws
    again = iter(ts.DirectionBatches(items, {"a_b": 0.5, "c_d": 0.5}, 400, 12, seed=3))
    assert [next(again) for _ in range(300)] == first        # reproducible from the seed
    other = iter(ts.DirectionBatches(items, {"a_b": 0.5, "c_d": 0.5}, 400, 12, seed=4))
    assert [next(other) for _ in range(300)] != first


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    cfg = tr.tiny_config()
    root = tmp_path_factory.mktemp("s2ut_synth")
    return cfg, root, ts.make_synthetic(root, cfg, n_per_direction=8, seed=1)


def test_dataset_appends_silence_tails_and_layout_supervises_them(corpus):
    cfg, root, data = corpus
    args = ts.parse_args(["--smoke", str(root / "unused"), "--steps", "1"])
    args.manifest, args.src_latents, args.tgt_units = \
        data["manifest"], data["src_latents"], data["tgt_units"]
    items, meta = ts.join_items(args, cfg, "train", log=lambda *_: None)
    assert len(items) == 6 * 7 and meta == {"vocab": cfg.vocab, "label_space": "synthetic"}
    assert {it["direction"] for it in items} == {"es_en", "fr_en", "de_en", "en_es", "en_fr", "en_de"}
    tok = ts.CtcTokenizer(data["spm_dir"], [c for c, _ in cfg.langs])
    assert tok.blank == {lang: 3 for lang in ("en", "es", "fr", "de")}
    silence = np.load(data["silence_latents"])

    plain = ts.S2utDataset(items, cfg, tok)[0]
    assert plain["tail_chunks"] == 0 and len(plain["latents"]) % 4 == 0
    assert 3 not in plain["src_tokens"] and len(plain["tgt_tokens"]) > 0
    tailed = ts.S2utDataset(items, cfg, tok, silence, tail_p=1.0, seed=5)
    tails = [tailed[0]["tail_chunks"] for _ in range(60)]
    assert set(tails) == set(range(5, 11))                   # 400-800 ms in 80 ms chunks
    s = tailed[0]
    assert len(s["latents"]) == len(plain["latents"]) + 4 * s["tail_chunks"]
    # The tail is silence from the bank, not speech.
    assert np.abs(s["latents"][-4 * s["tail_chunks"]:]).max() <= np.abs(silence).max() + 1e-6
    some = ts.S2utDataset(items, cfg, tok, silence, tail_p=0.3, seed=9)
    share = np.mean([some[i % 42]["tail_chunks"] > 0 for i in range(400)])
    assert 0.2 < share < 0.4

    # Wait-k keeps writing through the tail: the decoder must finish, say <EOS>, then <PAD>.
    raw = ts.collate([s, plain])
    batch = ts.make_batch(raw, 2, cfg, tok.blank, torch.device("cpu"))
    src_chunks = batch["src_chunks"].tolist()
    assert src_chunks[0] == src_chunks[1] + s["tail_chunks"]
    steps = int(batch["mask"][0].sum())
    assert steps == src_chunks[0] - 2 + 1
    target = batch["target"][0, :steps]
    eos_step = int((target == cfg.eos).any(-1).nonzero()[0])
    assert (target[eos_step + 1:] == cfg.pad).all() and steps - 1 - eos_step >= s["tail_chunks"] - 2
    assert batch["eos"][0, :steps].tolist() == [0.0] * eos_step + [1.0] * (steps - eos_step)
    assert batch["src_visible"][0, steps - 1] == src_chunks[0]


def test_join_reports_missing_and_mismatched_inputs(corpus, tmp_path):
    cfg, root, data = corpus
    args = ts.parse_args(["--smoke", str(root / "unused"), "--steps", "1"])
    args.manifest, args.src_latents, args.tgt_units = \
        data["manifest"], data["src_latents"], data["tgt_units"]
    with pytest.raises(ValueError, match="vocab"):
        ts.join_items(args, tr.tiny_config(vocab=cfg.vocab + 1), "train", log=lambda *_: None)
    with pytest.raises(ValueError, match="dim"):
        ts.join_items(args, tr.tiny_config(latent_dim=8), "train", log=lambda *_: None)
    with pytest.raises(ValueError, match="no 'test' utterance"):
        ts.join_items(args, cfg, "test", log=lambda *_: None)
    with pytest.raises(FileNotFoundError, match="spm_en.model"):
        ts.CtcTokenizer(tmp_path, ["en"])
    assert ts.main(["--manifest", "nope.jsonl", "--src-latents", str(tmp_path),
                    "--tgt-units", str(tmp_path), "--spm-dir", str(tmp_path)]) == 1


def test_smoke_training_learns_checkpoints_and_resumes(tmp_path):
    work = tmp_path / "smoke"
    assert ts.main(["--smoke", str(work), "--steps", "40", "--ckpt-steps", "20"]) == 0
    ckpt = work / "checkpoints" / "smoke"
    history = json.loads((ckpt / "history.json").read_text("utf-8"))
    assert [h["step"] for h in history] == [10, 20, 30, 40]
    assert history[-1]["total"] < 0.75 * history[0]["total"]
    assert history[-1]["unit"] < history[0]["unit"] and history[-1]["asr_ctc"] < history[0]["asr_ctc"]
    assert history[0]["simultaneous_share"] == 0.0 and history[-1]["simultaneous_share"] == 1.0
    assert 0.1 < np.mean([h["silence_tail_share"] for h in history]) < 0.5
    assert all((ckpt / n).is_file() for n in ("last.pt", "final.pt", "best.pt"))

    first = torch.load(ckpt / "final.pt", weights_only=True)
    assert first["step"] == 40 and json.loads(first["config"])["unit_mode"] == "stacked_4"
    # Resume: picks up at step 40 with the optimizer state and keeps training.
    assert ts.main(["--smoke", str(work), "--steps", "60", "--resume", "auto"]) == 0
    second = torch.load(ckpt / "final.pt", weights_only=True)
    assert second["step"] == 60
    assert second["optimizer"]["state"][0]["step"] == 60
    resumed = json.loads((ckpt / "history.json").read_text("utf-8"))
    assert [h["step"] for h in resumed] == [50, 60]
    changed = sum(not torch.equal(first["model"][k], second["model"][k]) for k in first["model"])
    assert changed > 0.9 * len(first["model"])
    # A different model config must not silently resume.
    assert ts.main(["--smoke", str(work), "--steps", "70", "--resume", "auto",
                    "--unit-mode", "reduced_dur"]) == 1

    # The checkpoint loads as a run-time translator (CTC heads stripped) and streams.
    model, ck = tr.S2utTranslator.from_checkpoint(ckpt / "final.pt", runtime_only=True)
    assert not model.asr_ctc_head and model.param_counts()["training_only"] == 0
    st = ss.StreamingTranslator(ss.TorchRuntime(model, "es", "en"), WaitKConfig(k=2))
    lat = torch.randn(40, model.cfg.latent_dim)
    results = [st.push_chunk(lat[i * 4:(i + 1) * 4]) for i in range(10)]
    assert sum(r.writes for r in results) == 9


def test_smoke_training_reduced_dur(tmp_path):
    work = tmp_path / "smoke_reduced"
    assert ts.main(["--smoke", str(work), "--steps", "20", "--unit-mode", "reduced_dur",
                    "--full-context-fraction", "0.0", "--ramp-fraction", "0.0"]) == 0
    history = json.loads((work / "checkpoints" / "smoke" / "history.json").read_text("utf-8"))
    assert history[-1]["total"] < history[0]["total"]
    assert history[0]["simultaneous_share"] == 1.0
