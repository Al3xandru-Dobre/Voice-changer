"""Causal S2UT translator (Milestone 3 / Stage 05): fvml/translator.py and the
streaming driver fvml/s2ut_stream.py.

  * run-time parameter budget (25-38M, CTC projections excluded)
  * ring bookkeeping, causality, sliding windows and the Wait-k cross-attention
    mask, each pinned by a perturbation test
  * step() driven chunk by chunk through WaitKPolicy == masked forward():
    short variants on a tiny model (rings wrap many times, flush + reset,
    mid-flush resumption, adaptive writes, reduced_dur), 22 s on the full-size
    model (128-chunk rings, > 2x the window), and the Definition-of-done run,
    60 s on the full-size model, which takes ~1-2 min and runs only with
    FLUXVOICE_SLOW_TESTS=1
  * target layouts (stacked_4, reduced_dur, silence tail) and the joint loss
"""
import math
import os

import numpy as np
import pytest
import torch

from fvml import s2ut_stream as ss
from fvml import translator as tr
from fvml.waitk_policy import Decision, WaitKConfig

F64 = torch.float64


def make_model(cfg=None, dtype=F64, seed=0, ctc_vocab=None):
    torch.manual_seed(seed)
    model = tr.S2utTranslator(cfg or tr.tiny_config(), ctc_vocab).to(dtype).eval()
    with torch.no_grad():        # zero-initialised biases would hide a wrong distance index
        for name, p in model.named_parameters():
            if name.endswith("rel_bias"):
                p.normal_(0.0, 1.0)
    return model


def random_latents(cfg, chunks, seed=0, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(chunks * cfg.chunk_frames, cfg.latent_dim, generator=g, dtype=dtype)


def unit_teacher(cfg, seed=0):
    """Deterministic random regular units (never a special) per decoder step."""
    def teacher(step):
        rng = np.random.default_rng(seed * 1_000_003 + step)
        if cfg.unit_mode == "reduced_dur":
            return int(rng.integers(0, cfg.vocab)), int(rng.integers(1, cfg.max_duration + 1))
        return rng.integers(0, cfg.vocab, cfg.units_per_step)
    return teacher


def run_stream(model, latents, policy, speech=None, teacher=None, src="es", tgt="en"):
    cfg = model.cfg
    st = ss.StreamingTranslator(ss.TorchRuntime(model, src, tgt), policy, teacher=teacher,
                                record=True)
    c = cfg.chunk_frames
    results = []
    for i in range(latents.shape[0] // c):
        results.append(st.push_chunk(latents[i * c:(i + 1) * c],
                                     None if speech is None else speech[i]))
    return st, results


@torch.no_grad()
def replay_forward(model, st, latents, src="es", tgt="en"):
    """Feed forward() exactly what the stream saw: per segment (the chunks
    between two resets), the encoded chunks, the decoder inputs and the
    recorded schedule. Returns (step logits, forward logits), each [steps, U, C]."""
    cfg = model.cfg
    c = cfg.chunk_frames
    ids = cfg.lang_ids
    step_logits, fwd_logits = [], []
    for seg in sorted({d["segment"] for d in st.dec_log}):
        chunks = [i for i, ch in enumerate(st.chunk_log) if ch["segment"] == seg and ch["encoded"]]
        steps = [d for d in st.dec_log if d["segment"] == seg]
        lat = torch.cat([latents[i * c:(i + 1) * c] for i in chunks])[None]
        enc = model.encoder(lat, torch.tensor([ids[src]]))
        prev = torch.from_numpy(np.stack([d["prev_units"] for d in steps]))[None]
        vis = torch.tensor([[d["src_visible"] for d in steps]])
        dur = (torch.tensor([[d["prev_dur"] for d in steps]])
               if cfg.unit_mode == "reduced_dur" else None)
        out = model.decoder(prev, torch.tensor([ids[tgt]]), enc["mem_k"], enc["mem_v"], vis, dur)
        fwd_logits.append(out["logits"][0].numpy())
        step_logits.append(np.stack([d["logits"] for d in steps]))
    return np.concatenate(step_logits), np.concatenate(fwd_logits)


def agreement(a, b):
    return float(np.mean(a.argmax(-1) == b.argmax(-1)))


# -- size -----------------------------------------------------------------------
def test_runtime_parameter_budget():
    ctc = {lang: 6000 for lang in ("en", "es", "fr", "de")}
    model = tr.S2utTranslator(tr.TranslatorConfig(), ctc)
    counts = model.param_counts()
    print(f"\nstacked_4 run-time parameters: {counts['runtime']:,} "
          f"(encoder {counts['encoder']:,}, decoder {counts['decoder']:,}); "
          f"training-only CTC projections: {counts['training_only']:,}")
    assert 25_000_000 <= counts["runtime"] <= 38_000_000
    assert counts["runtime"] == counts["encoder"] + counts["decoder"] == 26_265_710
    # 2 heads x 4 languages x Linear(320, 4 * 6000)
    assert counts["training_only"] == 2 * 4 * (320 * 4 * 6000 + 4 * 6000)
    assert model.decoder.unit_head.out_features == 4 * 1003
    assert model.asr_ctc_head["de"].out_features == 4 * 6000
    assert sum(p.numel() for p in model.strip_training_heads().parameters()) == counts["runtime"]

    reduced = tr.S2utTranslator(tr.TranslatorConfig(unit_mode="reduced_dur")).param_counts()
    print(f"reduced_dur run-time parameters: {reduced['runtime']:,}")
    assert 25_000_000 <= reduced["runtime"] <= 38_000_000


def test_config_roundtrip_and_special_units():
    cfg = tr.TranslatorConfig()
    assert tr.TranslatorConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.special_units == {"bos": 1000, "eos": 1001, "pad": 1002}
    assert (cfg.units_per_step, cfg.chunk_ms, cfg.head_dim) == (4, 80.0, 80)
    assert tr.TranslatorConfig(unit_mode="reduced_dur").units_per_step == 1
    with pytest.raises(ValueError):
        tr.TranslatorConfig(unit_mode="stacked_2")


# -- ring bookkeeping -------------------------------------------------------------
@pytest.mark.parametrize("window", [1, 3, 8])
def test_ring_slots_track_the_last_w_entries(window):
    """Replay a ring with plain Python and check ring_write_mask / ring_distance
    name the same slot, distance and liveness for 3+ wraps."""
    held = [None] * window
    for pos in range(3 * window + 2):
        p = torch.tensor([pos])
        write = tr.ring_write_mask(p, window)[0].tolist()
        assert write == [s == pos % window for s in range(window)]
        held[pos % window] = pos
        dist, valid = tr.ring_distance(p, window)
        for s in range(window):
            assert bool(valid[0, s]) == (held[s] is not None)
            if held[s] is not None:
                assert int(dist[0, s]) == pos - held[s]
    # An empty ring (newest index -1, e.g. cross-attention before any source) has no live slot.
    assert not tr.ring_distance(torch.tensor([-1]), window)[1].any()


def test_self_attention_window_and_step_equal_forward():
    torch.manual_seed(0)
    w, t = 5, 23
    att = tr.RingSelfAttention(16, 2, w, 0.0).double().eval()
    with torch.no_grad():
        att.rel_bias.normal_()
    x = torch.randn(2, t, 16, dtype=F64)
    with torch.no_grad():
        ref = att(x)
        kv = torch.zeros(2, 4, w, 8, dtype=F64)
        ys = []
        for i in range(t):
            before = kv.clone()
            y, new = att.step(x[:, i:i + 1], kv, torch.tensor([i, i]))
            assert torch.equal(kv, before), "step() must not mutate its input ring"
            ys.append(y)
            kv = new
        torch.testing.assert_close(torch.cat(ys, dim=1), ref, atol=1e-12, rtol=0)
        # Key j reaches query i iff 0 <= i - j < W.
        for j in (0, 7, 20):
            x2 = x.clone()
            x2[:, j] += 1.0
            changed = (att(x2) - ref).abs().amax(dim=(0, 2)) > 1e-9
            assert changed.tolist() == [0 <= i - j < w for i in range(t)]


def test_cross_attention_sees_exactly_the_scheduled_window():
    """Target step t with schedule g sees source chunks g - W .. g - 1: nothing
    the policy has not read yet (Wait-k) and nothing older than the ring."""
    torch.manual_seed(1)
    w, s, heads = 4, 12, 2
    att = tr.RingCrossAttention(16, heads, w, 0.0).double().eval()
    with torch.no_grad():
        att.rel_bias.normal_()
    x = torch.randn(1, 6, 16, dtype=F64)
    mem_k = torch.randn(1, heads, s, 8, dtype=F64)
    mem_v = torch.randn(1, heads, s, 8, dtype=F64)
    visible = torch.tensor([[1, 3, 4, 7, 12, 12]])
    with torch.no_grad():
        ref = att(x, mem_k, mem_v, visible)
        for j in range(s):
            k2, v2 = mem_k.clone(), mem_v.clone()
            k2[:, :, j] += 1.0
            v2[:, :, j] += 1.0
            changed = (att(x, k2, v2, visible) - ref).abs().amax(dim=(0, 2)) > 1e-9
            assert changed.tolist() == [g - w <= j < g for g in visible[0].tolist()], j
        # The ring form: write chunk by chunk, read with src_pos = g.
        ring_k, ring_v = torch.zeros(1, heads, w, 8, dtype=F64), torch.zeros(1, heads, w, 8, dtype=F64)
        written = 0
        for t, g in enumerate(visible[0].tolist()):
            while written < g:
                m = tr.ring_write_mask(torch.tensor([written]), w)[:, None, :, None]
                ring_k = torch.where(m, mem_k[:, :, written:written + 1], ring_k)
                ring_v = torch.where(m, mem_v[:, :, written:written + 1], ring_v)
                written += 1
            y = att.step(x[:, t:t + 1], ring_k, ring_v, torch.tensor([g]))
            torch.testing.assert_close(y, ref[:, t:t + 1], atol=1e-12, rtol=0)


# -- encoder ----------------------------------------------------------------------
def test_encoder_step_equals_forward_and_fills_the_memory_ring():
    model = make_model()
    cfg, enc = model.cfg, model.encoder
    chunks = 3 * cfg.max_src_steps + 3                     # the rings wrap three times
    lat = random_latents(cfg, chunks)[None].repeat(2, 1, 1)
    lat[1] = lat[1].flip(0)
    lang = torch.tensor([1, 3])
    with torch.no_grad():
        ref = enc(lat, lang)
        state = enc.init_state(2)
        assert list(state) == enc.state_names()
        assert state["src_pos"].dtype == torch.int64 and state["src_pos"].shape == (2,)
        memory, ready = [], []
        for i in range(chunks):
            before = {n: t.clone() for n, t in state.items()}
            out, new = enc.step(lat[:, i * 4:(i + 1) * 4], lang, state)
            assert all(torch.equal(state[n], before[n]) for n in state), "step() mutated its state"
            assert list(new) == enc.state_names()
            assert all(new[n].shape == state[n].shape and new[n].dtype == state[n].dtype
                       for n in state)
            memory.append(out["memory"])
            ready.append(out["readiness_logit"])
            state = new
        assert state["src_pos"].tolist() == [chunks, chunks]
        torch.testing.assert_close(torch.stack(memory, 1), ref["memory"], atol=1e-11, rtol=0)
        torch.testing.assert_close(torch.cat(ready, 1), ref["readiness_logit"], atol=1e-11, rtol=0)
        # Slot s of the memory ring holds the newest chunk j with j mod W == s.
        w = cfg.max_src_steps
        for layer in range(cfg.dec_layers):
            for j in range(chunks - w, chunks):
                torch.testing.assert_close(state[f"mem_k_{layer}"][:, :, j % w],
                                           ref["mem_k"][layer][:, :, j], atol=1e-11, rtol=0)
                torch.testing.assert_close(state[f"mem_v_{layer}"][:, :, j % w],
                                           ref["mem_v"][layer][:, :, j], atol=1e-11, rtol=0)
    assert not list(enc.buffers()), "recurrent state must not live in module buffers"


def test_encoder_is_causal():
    model = make_model()
    cfg = model.cfg
    lat = random_latents(cfg, 20)[None]
    lang = torch.tensor([0])
    with torch.no_grad():
        ref = model.encoder(lat, lang)["memory"]
        for j in (0, 6, 19):
            lat2 = lat.clone()
            lat2[0, j * 4 + 3] += 1.0                      # the chunk's last frame
            changed = (model.encoder(lat2, lang)["memory"] - ref).abs().amax(dim=(0, 2)) > 1e-9
            assert not changed[:j].any() and changed[j]


# -- Wait-k -------------------------------------------------------------------------
def test_waitk_schedule():
    assert tr.waitk_visible(np.arange(6), 3, 5).tolist() == [3, 4, 5, 5, 5, 5]
    assert tr.waitk_visible(np.arange(4), 1, 10).tolist() == [1, 2, 3, 4]
    assert tr.waitk_visible(np.arange(3), None, 7).tolist() == [7, 7, 7]
    assert tr.waitk_visible(np.arange(2), 8, 3).tolist() == [3, 3]     # k longer than the source


@pytest.mark.parametrize("k", [1, 3, None])
def test_waitk_mask_hides_unread_source_from_the_decoder(k):
    """Perturbing source chunk j must leave every decoder step decoded before
    chunk j was read (g(t) <= j) bit-identical, and change the first step that
    reads it."""
    cfg = tr.tiny_config(max_src_steps=64, max_tgt_steps=64)
    model = make_model(cfg)
    s, t = 14, 12
    lat = random_latents(cfg, s)[None]
    g = torch.Generator().manual_seed(3)
    prev = torch.randint(0, cfg.vocab, (1, t, 4), generator=g)
    vis_np = tr.waitk_visible(np.arange(t), k, s)
    vis = torch.from_numpy(vis_np)[None]
    src, tgt = torch.tensor([2]), torch.tensor([0])
    with torch.no_grad():
        ref = model(lat, src, prev, tgt, vis)["logits"]
        for j in (0, 4, 9, 13):
            lat2 = lat.clone()
            lat2[0, j * 4] += 1.0
            diff = (model(lat2, src, prev, tgt, vis)["logits"] - ref).abs().amax(dim=(0, 2, 3))
            hidden = vis_np <= j
            assert torch.equal(diff[hidden], torch.zeros(int(hidden.sum()), dtype=F64)), (k, j)
            if not hidden.all():                 # k = 1 never reads chunk 13 within 12 steps
                assert diff[int(np.argmax(~hidden))] > 1e-9, (k, j)
        assert (vis_np <= 13).all() == (k == 1)


def test_policy_schedule_matches_training_schedule():
    """Strict Wait-k through the real WaitKPolicy decodes target chunk w with
    exactly waitk_visible(w, k, S) source chunks (while the source lasts)."""
    model = make_model()
    cfg = model.cfg
    for k in (1, 2, 5):
        chunks = 20
        st, results = run_stream(model, random_latents(cfg, chunks), WaitKConfig(k=k),
                                 teacher=unit_teacher(cfg))
        assert [r.decision for r in results] == \
            [Decision.READ] * (k - 1) + [Decision.WRITE] * (chunks - k + 1)
        got = [d["src_visible"] for d in st.dec_log]
        assert got == tr.waitk_visible(np.arange(len(got)), k, chunks).tolist()
        assert results[-1].first_write_t_src == k


# -- streaming equivalence ----------------------------------------------------------
def check_equivalence(model, latents, policy, speech=None, min_agreement=0.999, atol=1e-5):
    st, results = run_stream(model, latents, policy, speech, unit_teacher(model.cfg))
    a, b = replay_forward(model, st, latents)
    delta = float(np.max(np.abs(a - b)))
    agree = agreement(a, b)
    assert agree >= min_agreement, f"unit argmax agreement {agree:.5f}"
    assert delta <= atol, f"max |dlogit| {delta:.3e}"
    return st, results, delta, agree


@pytest.mark.parametrize("k", [1, 3, 8])
def test_streaming_equals_forward_short(k):
    """100 chunks through 8-slot rings (12 wraps of every ring), no VAD."""
    model = make_model(seed=k)
    st, results, delta, _ = check_equivalence(model, random_latents(model.cfg, 100, seed=k),
                                              WaitKConfig(k=k), atol=1e-10)
    assert len(st.dec_log) == 100 - k + 1 and st.segment == 0
    assert st.runtime.src_pos == 100
    assert int(st.runtime.dec_state["tgt_pos"][0]) == 100 - k + 1


def test_streaming_equals_forward_float32():
    model = make_model(dtype=torch.float32)
    lat = random_latents(model.cfg, 100, dtype=torch.float32)
    _, _, delta, agree = check_equivalence(model, lat, WaitKConfig(k=3), atol=1e-4)
    print(f"\ntiny float32: max |dlogit| {delta:.3e}, agreement {agree:.5f}")


def utterance_stream(cfg, utterances, gap_chunks, seed=0, dtype=F64, lead=0):
    """Concatenated utterances: speech chunks (random latents) separated by
    `gap_chunks` of near-silent latents. Returns (latents, speech verdicts)."""
    g = torch.Generator().manual_seed(seed)
    parts, speech = [], []
    silence = 0.01 * torch.randn(cfg.latent_dim, generator=g, dtype=dtype)

    def gap(n):
        parts.append(silence.repeat(n * cfg.chunk_frames, 1)
                     + 0.001 * torch.randn(n * cfg.chunk_frames, cfg.latent_dim, generator=g,
                                           dtype=dtype))
        speech.extend([False] * n)

    gap(lead)
    for n in utterances:
        parts.append(torch.randn(n * cfg.chunk_frames, cfg.latent_dim, generator=g, dtype=dtype))
        speech.extend([True] * n)
        gap(gap_chunks)
    return torch.cat(parts), speech


def test_streaming_equals_forward_across_flush_and_reset():
    """Utterances separated by 960 ms pauses: the policy flushes each tail and
    resets, the driver zeroes the translator state, and every utterance equals
    its own forward() (leading silence is never encoded)."""
    model = make_model(seed=5)
    cfg = model.cfg
    lat, speech = utterance_stream(cfg, [9, 30, 4, 17], gap_chunks=12, lead=3)
    policy = WaitKConfig(k=3, max_flush_ratio=1.5, max_flush_steps=6)
    st, results, _, _ = check_equivalence(model, lat, policy, speech, atol=1e-10)
    assert st.policy.flushes_completed == 4 and st.segment == 4
    assert sum(r.flushed for r in results) == 4
    # 5 silent chunks (400 ms) are encoded after each utterance, then the flush starts.
    encoded = [sum(c["encoded"] for c in st.chunk_log if c["segment"] == s) for s in range(4)]
    assert encoded == [9 + 5, 30 + 5, 4 + 5, 17 + 5]
    assert not any(c["encoded"] for c in st.chunk_log[:3])
    assert any(r.decision == Decision.FLUSH for r in results)
    # After the last reset the state is zero again.
    assert st.runtime.src_pos == 0 and int(st.runtime.dec_state["tgt_pos"][0]) == 0
    assert all(float(t.abs().sum()) == 0.0 for t in st.runtime.enc_state.values())


def test_mid_flush_speech_resumption_keeps_the_translator_state():
    """640 ms pauses: the flush starts (after 400 ms) but speech resumes before
    it drains. The policy returns to ACTIVE with its counters, the driver keeps
    the rings, and the whole stream is one forward() over the encoded chunks."""
    model = make_model(seed=13)
    lat, speech = utterance_stream(model.cfg, [9, 14, 11], gap_chunks=8)
    policy = WaitKConfig(k=3, max_flush_ratio=4.0, max_flush_steps=32)
    st, results, _, _ = check_equivalence(model, lat[:-8 * 4], policy, speech[:-8], atol=1e-10)
    assert st.policy.flush_resumes == 2 and st.policy.flushes_completed == 0 and st.segment == 0
    flush_chunks = [i for i, r in enumerate(results) if r.decision == Decision.FLUSH]
    assert len(flush_chunks) == 2 * 3                      # chunks 6-8 of each pause
    assert not any(st.chunk_log[i]["encoded"] for i in flush_chunks)
    assert all(results[i].writes == 1 for i in flush_chunks)
    assert st.runtime.src_pos == len(results) - 6


def test_streaming_equals_forward_with_short_pauses_no_reset():
    """240 ms pauses do not flush: one segment, rings wrap across utterances."""
    model = make_model(seed=6)
    lat, speech = utterance_stream(model.cfg, [10, 12, 9, 14, 11], gap_chunks=3)
    st, _, _, _ = check_equivalence(model, lat, WaitKConfig(k=2), speech, atol=1e-10)
    assert st.policy.flushes_completed == 0 and st.segment == 0
    assert st.runtime.src_pos == len(speech) > 5 * model.cfg.max_src_steps


def test_streaming_equals_forward_adaptive_policy():
    """CTC-adaptive Wait-k: 0, 1 or 2 writes per chunk driven by the readiness
    head; forward() with the recorded schedule still matches."""
    model = make_model(seed=7)
    with torch.no_grad():
        model.encoder.readiness_head.weight.mul_(40.0)     # spread readiness over (0, 1)
    lat = random_latents(model.cfg, 80, seed=7)
    policy = WaitKConfig(k=2, adaptive_mode=True, adaptive_extra_wait=2)
    st, results, _, _ = check_equivalence(model, lat, policy, atol=1e-10)
    assert {r.writes for r in results} == {0, 1}
    lags = [r.t_src - r.t_tgt for r in results[4:]]
    assert max(lags) > 2, "readiness never held a write back"
    # Lowering k mid-stream leaves a backlog: the policy then writes twice per chunk.
    st = ss.StreamingTranslator(ss.TorchRuntime(model, "es", "en"),
                                WaitKConfig(k=6, adaptive_mode=True, adaptive_extra_wait=1),
                                teacher=unit_teacher(model.cfg), record=True)
    results = []
    for i in range(40):
        if i == 20:
            st.policy.set_k(1)
        results.append(st.push_chunk(lat[i * 4:(i + 1) * 4]))
    assert 2 in {r.writes for r in results[20:]}
    double = next(r for r in results[20:] if r.writes == 2)
    assert double.decoder_steps == 2 and len(double.frames) == 8
    a, b = replay_forward(model, st, lat[:160])
    assert np.max(np.abs(a - b)) <= 1e-10
    visible = [d["src_visible"] for d in st.dec_log]
    assert visible == sorted(visible) and len(set(visible)) < len(visible)


def test_streaming_equals_forward_reduced_dur():
    cfg = tr.tiny_config(unit_mode="reduced_dur", max_duration=6)
    model = make_model(cfg, seed=8)
    lat = random_latents(cfg, 90, seed=8)
    st, results, _, _ = check_equivalence(model, lat, WaitKConfig(k=3), atol=1e-10)
    # Every write tops the frame budget up by 4 frames; surplus carries over.
    assert st.frame_budget == 4 * sum(r.writes for r in results)
    assert st.frame_budget <= st.frames_emitted < st.frame_budget + cfg.max_duration
    assert sum(len(r.frames) for r in results) == st.frames_emitted
    assert int(st.runtime.dec_state["tgt_pos"][0]) == len(st.dec_log) > 3 * cfg.max_tgt_steps


def test_free_running_stream_is_reproduced_by_forward():
    """No teacher: the decoder feeds on its own argmax. forward() on the units
    the stream emitted reproduces every one of them."""
    model = make_model(seed=9)
    cfg = model.cfg
    lat = random_latents(cfg, 60, seed=9)
    st, results = run_stream(model, lat, WaitKConfig(k=3))
    a, b = replay_forward(model, st, lat)
    assert np.array_equal(a.argmax(-1), b.argmax(-1))
    assert np.max(np.abs(a - b)) <= 1e-10
    emitted = np.stack([d["units"] for d in st.dec_log])
    assert np.array_equal(emitted, b.argmax(-1))
    assert np.array_equal(st.dec_log[0]["prev_units"], np.full(4, cfg.bos))
    assert all(np.array_equal(st.dec_log[i + 1]["prev_units"], emitted[i])
               for i in range(len(emitted) - 1))


def test_special_units_truncate_a_chunk_and_eos_stops_the_utterance():
    model = make_model(seed=10)
    cfg = model.cfg
    lat, speech = utterance_stream(cfg, [6], gap_chunks=12)

    def run(eos_step):
        def teacher(step):
            if step == eos_step:
                return np.array([5, cfg.eos, cfg.pad, cfg.pad])
            return np.array([1, 2, 3, 4])
        return run_stream(model, lat, WaitKConfig(k=2), speech, teacher)

    # Wait-2 over 6 speech chunks: writes 0..4 happen on speech chunks 1..5.
    # <EOS> on write 4 (the user is still talking) is not the end of the utterance:
    # the policy keeps writing through the pause and the flush.
    st, results = run(4)
    emitted = [r.frames for r in results if r.writes]
    assert emitted[:6] == [[1, 2, 3, 4]] * 4 + [[5], [1, 2, 3, 4]]
    assert [r.eos for r in results if r.writes][:6] == [False] * 4 + [True, False]
    assert len(emitted) > 6 and st.policy.flushes_completed == 1
    # <EOS> on write 5 (first silent chunk): caught up, nothing more is decoded,
    # and the pending tail flush completes without a single flush step.
    st, results = run(5)
    emitted = [r.frames for r in results if r.writes]
    assert emitted == [[1, 2, 3, 4]] * 5 + [[5]]
    assert len(st.dec_log) == 6 and st.policy.flushes_completed == 1
    assert not any(r.decision == Decision.FLUSH for r in results)
    # The frames after the first special unit are dropped, like UnitProsodyFifo.
    assert st.dec_log[5]["units"].tolist() == [5, cfg.eos, cfg.pad, cfg.pad]


def test_eos_prob_ends_a_free_running_utterance():
    """Without a teacher, eos_prob > threshold counts as <EOS> even when no
    <EOS> unit was emitted."""
    model = make_model(seed=10)
    cfg = model.cfg
    with torch.no_grad():
        model.decoder.eos_head.bias.fill_(50.0)            # eos_prob == 1 on every step
    lat, speech = utterance_stream(cfg, [6], gap_chunks=12)
    st, results = run_stream(model, lat, WaitKConfig(k=2), speech)
    assert all(r.eos for r in results if r.writes)
    assert len(st.dec_log) == 6                            # 5 writes while talking + 1 in the pause
    assert st.policy.flushes_completed == 1
    st2 = ss.StreamingTranslator(ss.TorchRuntime(model, "es", "en"), WaitKConfig(k=2),
                                 eos_threshold=1.0, record=True)
    for i, sp in enumerate(speech):
        st2.push_chunk(lat[i * 4:(i + 1) * 4], sp)
    assert len(st2.dec_log) > 6                            # threshold 1.0: never <EOS>


# -- the full-size model: 128-chunk rings ----------------------------------------------
@pytest.fixture
def single_thread():
    """Per-chunk steps are small matmuls: one thread is ~2x faster than a pool."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def full_size_stream(chunks, seed=11):
    """`chunks` of concatenated 4-5.6 s utterances with 160 ms pauses (shorter
    than the 400 ms tail flush, so nothing resets the rings)."""
    cfg = tr.TranslatorConfig(dropout=0.0)
    lengths = [50, 62, 55, 70, 58, 64, 52, 66, 60, 56, 68, 57, 61]
    lat, speech = utterance_stream(cfg, lengths, gap_chunks=2, seed=seed)
    assert len(speech) >= chunks
    return cfg, lat[:chunks * cfg.chunk_frames], speech[:chunks]


def test_streaming_equals_forward_full_size_ring_wrap(single_thread):
    """The shipped geometry (26.3M parameters, W_src = W_tgt = 128 chunks) over
    280 chunks = 22.4 s, more than twice the sliding window (MEMORY.md rule 6)."""
    cfg, lat, speech = full_size_stream(280)
    model = make_model(cfg, seed=11)
    st, _, delta, agree = check_equivalence(model, lat, WaitKConfig(k=3), speech)
    assert st.segment == 0 and st.runtime.src_pos == 280 > 2 * cfg.max_src_steps
    assert int(st.runtime.dec_state["tgt_pos"][0]) == 278 > 2 * cfg.max_tgt_steps
    print(f"\n22.4 s full-size float64: max |dlogit| {delta:.3e}, agreement {agree:.6f}")


@pytest.mark.skipif(not os.environ.get("FLUXVOICE_SLOW_TESTS"),
                    reason="60 s full-size run (~1-2 min): set FLUXVOICE_SLOW_TESTS=1")
def test_streaming_equals_forward_60s_full_size(single_thread):
    """Stage 05 Definition of done: 60 s (750 chunks) of concatenated utterances
    through the full-size model, so every 128-chunk ring wraps 5.8 times; step()
    + WaitKPolicy vs masked forward(): unit argmax agreement >= 99.9 % and
    max |dlogit| <= 1e-5. Asserted in float64 (the math) and in float32 (the
    precision ONNX Runtime ships), where the difference is summation-order
    rounding."""
    cfg, lat, speech = full_size_stream(750)
    for dtype in (F64, torch.float32):
        model = make_model(cfg, dtype=dtype, seed=11)
        st, _, delta, agree = check_equivalence(model, lat.to(dtype), WaitKConfig(k=3), speech,
                                                min_agreement=0.999, atol=1e-5)
        assert st.segment == 0 and st.runtime.src_pos == 750 and len(st.dec_log) == 748
        print(f"\n60 s full-size {str(dtype).split('.')[-1]}: {len(st.dec_log)} decoder steps "
              f"({4 * len(st.dec_log)} unit frames), max |dlogit| {delta:.3e}, "
              f"unit argmax agreement {agree:.6f}")


# -- targets --------------------------------------------------------------------------
def test_build_targets_stacked():
    cfg = tr.tiny_config()
    b, e, p = cfg.bos, cfg.eos, cfg.pad
    lay = tr.build_targets([1, 2, 3, 4, 5, 6], src_chunks=5, k=None, cfg=cfg)
    assert lay["target"].tolist() == [[1, 2, 3, 4], [5, 6, e, p]]
    assert lay["prev_units"].tolist() == [[b, b, b, b], [1, 2, 3, 4]]
    assert lay["eos"].tolist() == [0, 1]
    assert lay["src_visible"].tolist() == [5, 5] and lay["target_chunks"] == 2
    # A whole number of chunks still gets an <EOS> step of its own.
    lay = tr.build_targets([1, 2, 3, 4], src_chunks=2, k=1, cfg=cfg)
    assert lay["target"].tolist() == [[1, 2, 3, 4], [e, p, p, p]]
    assert lay["eos"].tolist() == [0, 1] and lay["src_visible"].tolist() == [1, 2]
    with pytest.raises(ValueError):
        tr.build_targets([cfg.vocab], 3, 1, cfg)


def test_build_targets_silence_tail_extends_writes_with_pad_steps():
    """Target done after 2 writes, source (speech + silence tail) lasts 9 chunks:
    Wait-2 keeps writing 9 - 2 + 1 = 8 chunks, so 6 all-<PAD> steps with eos = 1
    follow, each seeing the growing source."""
    cfg = tr.tiny_config()
    lay = tr.build_targets([7] * 5, src_chunks=9, k=2, cfg=cfg)
    assert lay["target"].shape == (8, 4)
    assert lay["target"][1].tolist() == [7, cfg.eos, cfg.pad, cfg.pad]
    assert (lay["target"][2:] == cfg.pad).all()
    assert lay["eos"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert lay["src_visible"].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]
    assert lay["prev_units"][2].tolist() == [7, cfg.eos, cfg.pad, cfg.pad]
    # Full context has no policy writes to mirror: no extra steps.
    assert tr.build_targets([7] * 5, src_chunks=9, k=None, cfg=cfg)["target"].shape == (2, 4)


def test_reduced_dur_targets_and_roundtrip():
    cfg = tr.tiny_config(unit_mode="reduced_dur", max_duration=4)
    units = [3, 3, 3, 3, 3, 3, 9, 1, 1, 9]
    ids, durs = tr.rle_units(units, cfg.max_duration)
    assert ids.tolist() == [3, 3, 9, 1, 9] and durs.tolist() == [4, 2, 1, 2, 1]
    assert tr.expand_units(ids, durs).tolist() == units
    lay = tr.build_targets(units, src_chunks=6, k=2, cfg=cfg)
    # start frames 0, 4, 6, 7, 9 -> writes 0, 1, 1, 1, 2; <EOS> at frame 10 -> write 2;
    # Wait-2 over 6 chunks makes 5 writes, the target fills ceil(11 / 4) = 3: 2 <PAD> steps.
    assert lay["target"][:, 0].tolist() == [3, 3, 9, 1, 9, cfg.eos, cfg.pad, cfg.pad]
    assert lay["dur"].tolist() == [4, 2, 1, 2, 1, 1, 4, 4]
    assert lay["prev_units"][:, 0].tolist() == [cfg.bos, 3, 3, 9, 1, 9, cfg.eos, cfg.pad]
    assert lay["prev_dur"].tolist() == [0, 4, 2, 1, 2, 1, 1, 4]
    assert lay["src_visible"].tolist() == [2, 3, 3, 3, 4, 4, 5, 6]
    assert lay["eos"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_reduced_dur_layout_is_the_schedule_the_stream_runs():
    """Teacher-force a layout's tokens through the streaming driver under strict
    Wait-k: every decoder step sees exactly the layout's src_visible."""
    cfg = tr.tiny_config(unit_mode="reduced_dur", max_duration=6)
    model = make_model(cfg, seed=12)
    rng = np.random.default_rng(12)
    k, chunks = 3, 40
    units = np.repeat(rng.integers(0, cfg.vocab, 60), rng.integers(1, 9, 60))[:(chunks - k) * 4]
    lay = tr.build_targets(units, chunks, k, cfg)
    tokens = list(zip(lay["target"][:, 0].tolist(), lay["dur"].tolist()))

    def teacher(step):
        return tokens[step] if step < len(tokens) else (cfg.pad, 4)

    st, _ = run_stream(model, random_latents(cfg, chunks), WaitKConfig(k=k), teacher=teacher)
    n = int(np.argmax(lay["target"][:, 0] == cfg.eos)) + 1     # through <EOS>
    assert [d["src_visible"] for d in st.dec_log][:n] == lay["src_visible"][:n].tolist()
    assert [d["prev_dur"] for d in st.dec_log][:n] == lay["prev_dur"][:n].tolist()


def test_collate_pads_with_ignore():
    cfg = tr.tiny_config()
    a = tr.build_targets([1, 2, 3], 4, 1, cfg)
    b = tr.build_targets(list(range(11)), 4, None, cfg)
    batch = tr.collate_targets([a, b], cfg)
    assert batch["target"].shape == (2, 4, 4) and batch["mask"].sum(1).tolist() == [4, 3]
    assert (batch["target"][1, 3] == tr.IGNORE).all() and batch["src_visible"][1, 3] == 1
    assert batch["target_chunks"].tolist() == [1.0, 3.0]


# -- loss -----------------------------------------------------------------------------
def loss_batch(model, k, seed=0):
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    src_chunks = [9, 6, 12]
    langs = [("es", "en"), ("en", "de"), ("en", "es")]
    lat = torch.zeros(3, 12 * 4, cfg.latent_dim)
    for i, s in enumerate(src_chunks):
        lat[i, :s * 4] = torch.from_numpy(rng.standard_normal((s * 4, cfg.latent_dim))).float()
    layouts = [tr.build_targets(rng.integers(0, cfg.vocab, n), s, k, cfg)
               for n, s in zip((30, 17, 41), src_chunks)]
    batch = tr.collate_targets(layouts, cfg)
    ids = cfg.lang_ids
    batch.update(
        latents=lat, src_chunks=torch.tensor(src_chunks),
        src_lang=torch.tensor([ids[s] for s, _ in langs]),
        tgt_lang=torch.tensor([ids[t] for _, t in langs]),
        src_tokens=[rng.integers(4, 30, n).tolist() for n in (7, 4, 9)],
        tgt_tokens=[rng.integers(4, 30, n).tolist() for n in (6, 5, 8)],
        ctc_blank={lang: 3 for lang in ids})
    return batch


@pytest.mark.parametrize("mode", tr.UNIT_MODES)
def test_joint_loss_is_the_weighted_sum_and_trains_every_part(mode):
    cfg = tr.tiny_config(unit_mode=mode)
    model = make_model(cfg, dtype=torch.float32, ctc_vocab={lang: 30 for lang in cfg.lang_ids})
    model.train()
    batch = loss_batch(model, k=2)
    out = model(batch["latents"], batch["src_lang"], batch["prev_units"], batch["tgt_lang"],
                batch["src_visible"], batch.get("prev_dur"))
    terms = tr.s2ut_loss(model, out, batch)
    unit = terms["unit"] + (terms["dur"] if mode == "reduced_dur" else 0.0)
    expect = (unit + 0.3 * terms["asr_ctc"] + 0.3 * terms["s2tt_ctc"]
              + 0.1 * terms["eos"] + 0.1 * terms["readiness"])
    assert tr.LOSS_WEIGHTS == {"unit": 1.0, "asr_ctc": 0.3, "s2tt_ctc": 0.3, "eos": 0.1,
                               "readiness": 0.1}
    assert float(terms["total"].detach()) == pytest.approx(float(expect.detach()), rel=1e-6)
    assert all(math.isfinite(float(v.detach())) for v in terms.values())
    no_ctc = tr.s2ut_loss(model, out, batch, {"asr_ctc": 0.0, "s2tt_ctc": 0.0})    # the ablation
    assert float(no_ctc["total"].detach()) == pytest.approx(
        float((unit + 0.1 * terms["eos"] + 0.1 * terms["readiness"]).detach()), rel=1e-6)
    # An untrained model is near chance on units: ln(vocab + 3).
    assert float(terms["unit"].detach()) == pytest.approx(math.log(cfg.n_classes), rel=0.25)
    terms["total"].backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    for name in ("encoder.in_proj.weight", "encoder.mem_k.0.weight", "decoder.unit_head.weight",
                 "decoder.eos_head.weight", "encoder.readiness_head.weight",
                 "asr_ctc_head.es.weight", "asr_ctc_head.en.weight", "s2tt_ctc_head.de.weight",
                 "decoder.blocks.0.cross_attn.rel_bias"):
        assert grads[name] is not None and float(grads[name].abs().sum()) > 0, name
    # French appears on neither side of this batch: its heads get no gradient.
    assert grads["asr_ctc_head.fr.weight"] is None or \
        float(grads["asr_ctc_head.fr.weight"].abs().sum()) == 0


def test_readiness_targets_spread_the_target_chunks_over_the_source():
    mass = torch.tensor([[0.0, 2.0, 2.0, 0.0, 9.0], [1.0, 1.0, 1.0, 1.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    ready = tr.readiness_targets(mass, mask, torch.tensor([2.0, 8.0]))
    assert ready[0].tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0, 0.0])   # padded mass ignored
    assert ready[1].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.0])   # clamped to 1 / chunk
    ready = tr.readiness_targets(mass, mask, torch.tensor([1.0, 2.0]))
    assert ready.sum(1).tolist() == pytest.approx([1.0, 2.0])


def test_padding_does_not_change_an_utterance():
    """Batching with a longer neighbour (right padding) leaves an utterance's
    logits unchanged: every op is causal and the schedule never looks at padding."""
    cfg = tr.tiny_config()
    model = make_model(cfg, ctc_vocab={lang: 30 for lang in cfg.lang_ids})
    batch = loss_batch(model, k=3)
    lat = batch["latents"].double()
    with torch.no_grad():
        full = model(lat, batch["src_lang"], batch["prev_units"], batch["tgt_lang"],
                     batch["src_visible"])["logits"]
        s, t = int(batch["src_chunks"][1]), int(batch["mask"][1].sum())
        alone = model(lat[1:2, :s * 4], batch["src_lang"][1:2], batch["prev_units"][1:2, :t],
                      batch["tgt_lang"][1:2], batch["src_visible"][1:2, :t])["logits"]
    torch.testing.assert_close(full[1:2, :t], alone, atol=1e-11, rtol=0)
