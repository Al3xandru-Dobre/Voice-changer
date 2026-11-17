"""fvml/waitk_policy.py — the Python twin of src/pipeline/WaitKPolicy.hpp and
UnitProsodyFifo.hpp (Milestone 3 / Stage 04).

The scenario tests mirror tests/test_waitk_policy.cpp one for one, with the same
numbers. Parity with the C++ header is pinned twice:

  * tests/data/waitk/golden_trajectories.txt is replayed here and by the C++
    test in every CTest build;
  * with FLUXVOICE_WAITK_TEST pointing at the built test_waitk_policy binary
    (CMake sets it), 1,000 fresh random trajectories are replayed by both
    implementations and compared decision for decision.

Standard library + pytest only.
"""
import math
import os
import random
import subprocess
from pathlib import Path

import pytest

import make_waitk_golden
from fvml import waitk_policy as W
from fvml.waitk_policy import Decision, UnitProsodyFifo, WaitKConfig, WaitKPolicy

CHUNK_MS = 80
GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "data" / "waitk" / "golden_trajectories.txt"


def drive(p, speech, elapsed_ms=CHUNK_MS, readiness=0.0, eos=False):
    """The canonical per-chunk call order; returns (decision, writes)."""
    p.on_chunk_energy(speech, elapsed_ms)
    if p.source_wanted():
        p.notify_source_unit()
        p.on_source_readiness(readiness)
    decision = p.decide()
    before = p.flushes_completed
    writes = 0
    for w in range(p.max_writes_this_step()):
        p.notify_target_unit(eos and w == 0)
        writes += 1
        if p.flushes_completed != before:
            break
    return decision, writes


# ---------------------------------------------------------------------------
# WaitKPolicy
# ---------------------------------------------------------------------------
def test_constants_and_defaults():
    assert (W.K_MIN, W.K_MAX, W.TAIL_FLUSH_MS, W.CHUNK_FRAMES, W.FRAME_MS) == (1, 10, 400, 4, 20)
    p = WaitKPolicy()
    assert p.k == 3 and p.chunk_ms == 80 and p.wait_ms == 240
    assert p.t_src == 0 and p.t_tgt == 0
    assert p.decide() == Decision.READ and p.max_writes_this_step() == 0
    assert not p.utterance_active() and not p.flushing() and not p.flush_pending()
    c = WaitKConfig()
    assert (c.k, c.tail_flush_ms, c.max_flush_ratio, c.max_flush_steps) == (3, 400, 1.5, 32)
    assert (c.adaptive_mode, c.readiness_threshold, c.adaptive_extra_wait) == (False, 0.5, 2)


def test_config_is_sanitized():
    p = WaitKPolicy(WaitKConfig(k=99, max_flush_ratio=0.25, max_flush_steps=0,
                                readiness_threshold=7.0, adaptive_extra_wait=-3,
                                chunk_frames=0, frame_ms=0))
    c = p.config
    assert p.k == 10
    assert (c.max_flush_ratio, c.max_flush_steps, c.readiness_threshold) == (1.0, 1, 1.0)
    assert (c.adaptive_extra_wait, c.chunk_frames, c.frame_ms) == (0, 1, 1)
    q = WaitKPolicy(WaitKConfig(max_flush_ratio=math.nan, readiness_threshold=math.nan))
    assert q.config.max_flush_ratio == 1.0 and q.config.readiness_threshold == 0.0
    assert [WaitKPolicy(k).k for k in (-5, 0, 11)] == [1, 1, 10]


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_classic_waitk_sequence(k):
    p = WaitKPolicy(k)
    for step in range(20):
        decision, writes = drive(p, True)
        if step < k - 1:
            assert (decision, writes) == (Decision.READ, 0)
        else:
            assert (decision, writes) == (Decision.WRITE, 1)
            assert p.t_src - p.t_tgt == k - 1
    assert (p.t_src, p.t_tgt, p.flushes_completed) == (20, 20 - (k - 1), 0)


def test_legacy_calls_without_vad():
    p = WaitKPolicy(3)
    assert p.source_wanted()
    p.notify_source_unit()
    p.notify_source_unit()
    assert p.decide() == Decision.READ
    p.notify_source_unit()
    assert p.decide() == Decision.WRITE
    p.notify_target_unit()
    assert p.decide() == Decision.READ
    assert (p.t_src, p.t_tgt) == (3, 1) and not p.flush_pending()
    p.reset()
    assert (p.t_src, p.t_tgt) == (0, 0) and p.source_wanted()


def test_leading_silence_is_not_counted():
    p = WaitKPolicy(3)
    for _ in range(25):
        assert drive(p, False)[0] == Decision.READ
        assert not p.source_wanted()
    assert p.silence_ms == 0 and p.t_src == 0 and not p.utterance_active()
    assert drive(p, True)[0] == Decision.READ
    assert p.t_src == 1
    assert drive(p, False)[0] == Decision.READ
    assert not p.flushing() and p.silence_ms == 80
    assert drive(p, True)[0] == Decision.WRITE
    assert p.flushes_completed == 0


def test_short_pause_vs_tail_flush():
    p = WaitKPolicy(3)
    for _ in range(10):
        drive(p, True)
    assert (p.t_src, p.t_tgt) == (10, 8)

    drive(p, False, 150)
    assert drive(p, False, 150)[0] == Decision.WRITE
    assert not p.flushing() and p.silence_ms == 300
    drive(p, True)
    assert p.silence_ms == 0 and p.flushes_completed == 0
    assert (p.t_src, p.t_tgt) == (13, 11)

    drive(p, False, 150)
    drive(p, False, 150)
    assert not p.flushing() and (p.t_src, p.t_tgt) == (15, 13)
    assert drive(p, False, 150) == (Decision.FLUSH, 1)      # 450 ms > 400 ms
    assert p.flushing() and p.flush_pending() and not p.source_wanted()
    assert (p.t_src, p.t_tgt) == (15, 14)

    flush_steps = 1                                         # bound: ceil(1.5 * 15) = 23
    while p.flushing():
        assert drive(p, False)[0] == Decision.FLUSH
        flush_steps += 1
        assert flush_steps <= 10
    assert flush_steps == 10
    assert p.flushes_completed == 1 and (p.t_src, p.t_tgt) == (0, 0)
    assert not p.utterance_active()
    assert drive(p, False)[0] == Decision.READ
    assert p.flushes_completed == 1


def test_tail_flush_threshold_is_strict():
    p = WaitKPolicy(2)
    for _ in range(6):
        drive(p, True)
    for _ in range(5):
        drive(p, False)
    assert p.silence_ms == 400 and not p.flushing()
    assert drive(p, False, 1)[0] == Decision.FLUSH


def test_flush_is_bounded_by_max_flush_steps():
    p = WaitKPolicy(WaitKConfig(k=3, max_flush_ratio=4.0, max_flush_steps=3))
    for _ in range(30):
        drive(p, True)
    for _ in range(5):
        drive(p, False)
    flush_steps = sum(drive(p, False)[0] == Decision.FLUSH for _ in range(50))
    assert flush_steps == 3
    assert p.flushes_completed == 1 and (p.t_src, p.t_tgt) == (0, 0)


def test_flush_ratio_bound():
    p = WaitKPolicy(WaitKConfig(k=5, max_flush_ratio=1.0))
    for _ in range(12):
        drive(p, True)
    for _ in range(5):
        drive(p, False)
    assert (p.t_src, p.t_tgt) == (17, 13)
    flush_steps = 0
    while drive(p, False)[0] == Decision.FLUSH:
        flush_steps += 1
    assert flush_steps == 4 and p.flushes_completed == 1


def test_early_eos_terminates_flush():
    p = WaitKPolicy(3)
    for _ in range(20):
        drive(p, True)
    for _ in range(5):
        drive(p, False)
    assert drive(p, False)[0] == Decision.FLUSH
    assert drive(p, False, eos=True)[0] == Decision.FLUSH
    assert p.flushes_completed == 1 and (p.t_src, p.t_tgt) == (0, 0) and not p.flushing()
    assert drive(p, False)[0] == Decision.READ


def test_eos_while_silent_skips_the_flush():
    p = WaitKPolicy(3)
    for _ in range(10):
        drive(p, True)
    drive(p, False)
    assert drive(p, False, eos=True)[0] == Decision.WRITE
    assert p.t_tgt == p.t_src
    for _ in range(3):
        assert drive(p, False)[0] == Decision.READ
        assert p.flushes_completed == 0
    assert drive(p, False) == (Decision.READ, 0)
    assert p.flushes_completed == 1 and (p.t_src, p.t_tgt) == (0, 0)


def test_eos_while_speaking_is_ignored():
    p = WaitKPolicy(2)
    for _ in range(5):
        drive(p, True)
    assert drive(p, True, eos=True)[0] == Decision.WRITE
    assert (p.t_src, p.t_tgt) == (6, 5)
    assert drive(p, True)[0] == Decision.WRITE


def test_speech_after_eos_in_short_pause():
    p = WaitKPolicy(3)
    for _ in range(10):
        drive(p, True)
    drive(p, False)
    drive(p, False, eos=True)
    assert drive(p, False)[0] == Decision.READ
    assert p.t_src == 13
    assert drive(p, True)[0] == Decision.READ
    assert (p.t_src, p.t_tgt) == (14, 13)
    assert drive(p, True)[0] == Decision.READ
    assert drive(p, True)[0] == Decision.WRITE
    assert p.flushes_completed == 0


def test_speech_resumption_during_flush():
    p = WaitKPolicy(3)
    for _ in range(12):
        drive(p, True)
    for _ in range(5):
        drive(p, False)
    assert (p.t_src, p.t_tgt) == (17, 15)
    assert drive(p, False)[0] == Decision.FLUSH
    assert (p.t_src, p.t_tgt, p.flush_steps) == (17, 16, 1)

    assert drive(p, True)[0] == Decision.READ
    assert not p.flushing() and p.silence_ms == 0
    assert (p.t_src, p.t_tgt) == (18, 16)
    assert (p.flush_resumes, p.flushes_completed) == (1, 0)
    assert drive(p, True)[0] == Decision.WRITE
    assert (p.t_src, p.t_tgt) == (19, 17)

    q = WaitKPolicy(2)
    for _ in range(4):
        drive(q, True)
    for _ in range(5):
        drive(q, False)
    for _ in range(3):
        assert drive(q, False)[0] == Decision.FLUSH
    assert (q.t_src, q.t_tgt) == (9, 11)
    for _ in range(3):
        assert drive(q, True)[0] == Decision.READ
    assert drive(q, True)[0] == Decision.WRITE
    assert q.t_src - q.t_tgt == 1


def test_hot_set_k_clamping():
    p = WaitKPolicy(3)
    for raw, want in ((0, 1), (-7, 1), (42, 10)):
        p.set_k(raw)
        assert p.k == want
    assert p.wait_ms == 800
    p.set_k(3)
    for _ in range(8):
        drive(p, True)
    assert (p.t_src, p.t_tgt) == (8, 6)

    p.set_k(6)
    for _ in range(3):
        assert drive(p, True)[0] == Decision.READ
    assert drive(p, True)[0] == Decision.WRITE
    assert (p.t_src, p.t_tgt) == (12, 7)

    p.set_k(2)
    for _ in range(4):
        assert drive(p, True) == (Decision.WRITE, 1)
    assert (p.t_src, p.t_tgt, p.flushes_completed) == (16, 11, 0)


def test_reset_keeps_k_and_counters():
    p = WaitKPolicy(4)
    for _ in range(9):
        drive(p, True)
    for _ in range(6):
        drive(p, False)
    while p.flushing():
        drive(p, False)
    assert p.flushes_completed == 1
    for _ in range(5):
        drive(p, True)
    p.reset()
    assert (p.k, p.t_src, p.t_tgt, p.silence_ms, p.flushes_completed) == (4, 0, 0, 0, 1)
    assert not p.source_wanted()
    for _ in range(3):
        assert drive(p, True)[0] == Decision.READ
    assert drive(p, True)[0] == Decision.WRITE


def test_stray_notifications_are_ignored():
    p = WaitKPolicy(3)
    p.on_chunk_energy(False, CHUNK_MS)
    p.notify_source_unit()
    p.notify_target_unit(True)
    p.on_source_readiness(1.0)
    assert (p.t_src, p.t_tgt, p.flushes_completed) == (0, 0, 0)
    for _ in range(8):
        drive(p, True)
    for _ in range(6):
        drive(p, False)
    assert p.flushing()
    t_src = p.t_src
    p.notify_source_unit()
    assert p.t_src == t_src


def test_silence_counter_saturates():
    u32 = 0xFFFFFFFF
    p = WaitKPolicy(WaitKConfig(tail_flush_ms=u32))
    drive(p, True)
    p.on_chunk_energy(False, u32 - 5)
    p.on_chunk_energy(False, 100)
    assert p.silence_ms == u32 and not p.flushing()


def test_frame_to_chunk_aggregation():
    p = WaitKPolicy(1)
    assert [p.on_frame_energy(s) for s in (False, True, False)] == [False] * 3
    assert not p.utterance_active()
    assert p.on_frame_energy(False)
    assert p.utterance_active() and p.silence_ms == 0
    for _ in range(5):
        assert [p.on_frame_energy(False) for _ in range(4)] == [False, False, False, True]
    assert p.silence_ms == 400
    for _ in range(3):
        p.on_frame_energy(False)
    assert p.silence_ms == 400
    p.on_frame_energy(False)
    assert p.silence_ms == 0 and p.flushes_completed == 1

    q = WaitKPolicy(WaitKConfig(chunk_frames=2, frame_ms=10))
    assert q.chunk_ms == 20
    assert not q.on_frame_energy(True) and q.on_frame_energy(False)


def test_adaptive_readiness():
    def adaptive(k):
        return WaitKPolicy(WaitKConfig(k=k, adaptive_mode=True))

    p = adaptive(2)
    assert [drive(p, True)[1] for _ in range(3)] == [0, 0, 0]
    assert drive(p, True) == (Decision.WRITE, 1)            # forced at k_max = 4
    assert (p.t_src, p.t_tgt) == (4, 1)

    p = adaptive(2)
    assert drive(p, True, readiness=1.0)[1] == 0
    assert drive(p, True) == (Decision.WRITE, 1)
    assert drive(p, True)[1] == 0

    # Readiness below the threshold is "blank" and never accumulates.
    p = WaitKPolicy(WaitKConfig(k=1, adaptive_mode=True, adaptive_extra_wait=10))
    assert [drive(p, True, readiness=0.49)[1] for _ in range(3)] == [0, 0, 0]
    assert drive(p, True, readiness=0.5)[1] == 0            # 0.5 banked
    assert drive(p, True, readiness=0.5)[1] == 1            # 1.0 banked
    assert (p.t_src, p.t_tgt) == (5, 1)

    p = adaptive(3)
    drive(p, True, readiness=1.0)
    drive(p, True, readiness=1.0)
    assert p.t_tgt == 0
    drive(p, True)
    assert (p.t_src, p.t_tgt) == (3, 1)
    drive(p, True, readiness=1.0)
    assert (p.t_src, p.t_tgt) == (4, 2)
    p.set_k(1)
    assert drive(p, True, readiness=1.0) == (Decision.WRITE, 2)
    assert (p.t_src, p.t_tgt) == (5, 4)

    p = adaptive(1)
    assert drive(p, True, readiness=math.nan)[1] == 0
    assert drive(p, True, readiness=9.0)[1] == 2            # clamped to 2; lag 2 > k
    assert (p.t_src, p.t_tgt) == (2, 2)
    assert drive(p, True)[1] == 0

    # Constant k: held back, then a source chunk worth two target chunks.
    p = adaptive(2)
    drive(p, True, readiness=1.0)
    assert drive(p, True)[1] == 1
    assert drive(p, True)[1] == 0
    assert drive(p, True, readiness=2.0) == (Decision.WRITE, 2)
    assert (p.t_src, p.t_tgt, p.k) == (4, 3, 2)
    assert drive(p, True, readiness=2.0)[1] == 1            # lag == k: one write
    assert p.t_src - p.t_tgt == 1


def test_write_lag_invariant():
    rng = random.Random(7)
    for _ in range(200):
        cfg = WaitKConfig(k=rng.randint(1, 10), adaptive_mode=rng.random() < 0.5,
                          adaptive_extra_wait=rng.randint(0, 4))
        p = WaitKPolicy(cfg)
        for _ in range(150):
            p.on_chunk_energy(rng.random() < 0.875, CHUNK_MS)
            if p.source_wanted():
                p.notify_source_unit()
                p.on_source_readiness(2 * rng.randint(0, 7) / 7)
            d = p.decide()
            n = p.max_writes_this_step()
            assert 0 <= n <= 2 and (d == Decision.READ) == (n == 0)
            assert n == 1 or d != Decision.FLUSH
            assert n <= 1 or cfg.adaptive_mode
            before = p.flushes_completed
            for _ in range(n):
                if d == Decision.WRITE:
                    assert p.t_src - p.t_tgt >= p.k
                p.notify_target_unit(False)
                if p.flushes_completed != before:
                    break


# ---------------------------------------------------------------------------
# Golden trajectories + live C++ parity
# ---------------------------------------------------------------------------
def test_golden_file_is_current():
    assert GOLDEN.read_text(encoding="ascii") == make_waitk_golden.build()


def test_golden_trajectories_replay():
    cases = flushes = 0
    for line in GOLDEN.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        traj, expected = W.parse_trajectory(line)
        result = W.replay(traj)
        assert result.footer() == expected
        flushes += "F" in result.decisions
        cases += 1
    assert cases >= 200 and flushes >= 20


def test_random_trajectories_cover_every_path():
    rng = random.Random(3)
    seen = set()
    flushes = resumes = 0
    for _ in range(300):
        r = W.replay(W.random_trajectory(rng))
        seen |= set(r.decisions)
        flushes += r.flushes
        resumes += r.resumes
    assert seen == set("RWXF") and flushes > 50 and resumes > 10


def test_cpp_parity_on_1000_random_trajectories(tmp_path):
    """Decision-for-decision parity with WaitKPolicy.hpp (Stage 04 DoD)."""
    binary = os.environ.get("FLUXVOICE_WAITK_TEST")
    if not binary:
        pytest.skip("FLUXVOICE_WAITK_TEST (built test_waitk_policy binary) is not set")
    rng = random.Random(0xF1C5)
    trajectories = [W.random_trajectory(rng) for _ in range(1000)]
    script = tmp_path / "trajectories.txt"
    script.write_text("".join(t.header() + "\n" for t in trajectories), encoding="ascii",
                      newline="\n")
    proc = subprocess.run([binary, "--replay", str(script)], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    actual = proc.stdout.split("\n")[:-1]
    assert len(actual) == len(trajectories)
    for i, (traj, line) in enumerate(zip(trajectories, actual)):
        assert line.strip() == W.replay(traj).footer(), f"trajectory {i}: {traj.header()}"


# ---------------------------------------------------------------------------
# UnitProsodyFifo
# ---------------------------------------------------------------------------
VOCAB, BOS, EOS, PAD = 1000, 1000, 1001, 1002


def make_chunk(first):
    units = [(first + j) % VOCAB for j in range(4)]
    prosody = []
    for j in range(4):
        base = float(first + j)
        prosody += [base + 0.25, base + 0.5, base + 0.75, -base]
    return units, prosody


def frame_ok(i, unit, row):
    base = float(i)
    return unit == i % VOCAB and tuple(row) == (base + 0.25, base + 0.5, base + 0.75, -base)


def test_fifo_defaults():
    fifo = UnitProsodyFifo()
    assert (fifo.capacity, fifo.size(), fifo.free(), fifo.overflows) == (64, 0, 64, 0)
    assert fifo.eos_unit == 1001
    assert UnitProsodyFifo(0).capacity == 1


@pytest.mark.parametrize("hop", [1, 2, 4])
def test_fifo_rate_bridging(hop):
    fifo = UnitProsodyFifo()
    pushed = popped = 0
    for step in range(500):
        for _ in range(2 if step % 3 == 0 else 1):
            r = fifo.push_chunk(*make_chunk(pushed))
            assert (r.pushed, r.eos, r.overflow) == (4, False, False)
            pushed += 4
        while fifo.size() > 8:
            units, rows = fifo.pop_hop(hop)
            for u, row in zip(units, rows):
                assert frame_ok(popped, u, row)
                popped += 1
    while fifo.size() >= hop:
        for u, row in zip(*fifo.pop_hop(hop)):
            assert frame_ok(popped, u, row)
            popped += 1
    assert popped == pushed and fifo.size() == 0 and fifo.overflows == 0


def test_fifo_underflow_returns_none():
    fifo = UnitProsodyFifo()
    assert fifo.pop_hop(1) is None
    assert fifo.pop_hop(0) == ([], [])
    fifo.push_chunk(*make_chunk(0))
    assert fifo.pop_hop(2) is not None and fifo.size() == 2
    assert fifo.pop_hop(4) is None and fifo.size() == 2
    units, rows = fifo.pop_hop(2)
    assert frame_ok(2, units[0], rows[0]) and frame_ok(3, units[1], rows[1])


def test_fifo_skips_trailing_special_units():
    fifo = UnitProsodyFifo(64, VOCAB, EOS)
    _, prosody = make_chunk(0)

    def push(units):
        r = fifo.push_chunk(units, prosody)
        return r.pushed, r.eos, r.overflow

    assert push([10, 11, EOS, PAD]) == (2, True, False)
    assert push([12, PAD, PAD, PAD]) == (1, False, False)
    assert push([EOS, PAD, PAD, PAD]) == (0, True, False)
    assert push([13, BOS, 14, 15]) == (1, False, False)
    assert push([-1, 16, 17, 18]) == (0, False, False)
    assert push([999, 0, 1, 2]) == (4, False, False)
    units, rows = fifo.pop_hop(8)
    assert units == [10, 11, 12, 13, 999, 0, 1, 2]
    assert rows[1][0] == 1.25 and rows[2][0] == 0.25


def test_fifo_overflow_is_all_or_nothing():
    fifo = UnitProsodyFifo(6, VOCAB)
    assert fifo.push_chunk(*make_chunk(0)).pushed == 4
    r = fifo.push_chunk(*make_chunk(4))
    assert r.overflow and r.pushed == 0 and fifo.size() == 4 and fifo.overflows == 1
    assert fifo.pop_hop(2) is not None
    assert fifo.push_chunk(*make_chunk(4)).pushed == 4
    assert fifo.size() == 6 and fifo.free() == 0
    for i in range(2, 8):
        units, rows = fifo.pop_hop(1)
        assert frame_ok(i, units[0], rows[0])
    fifo.push_chunk(*make_chunk(0))
    fifo.clear()
    assert fifo.size() == 0 and fifo.free() == 6 and fifo.pop_hop(1) is None


def test_fifo_short_prosody_span():
    assert UnitProsodyFifo().push_chunk([1, 2, 3, 4], [0.0] * 9).pushed == 2
