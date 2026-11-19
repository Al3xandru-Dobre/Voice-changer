"""Wait-k policy state machine + unit/prosody FIFO (Milestone 3 / Stage 04).

Bit-exact Python twin of src/pipeline/WaitKPolicy.hpp and
src/pipeline/UnitProsodyFifo.hpp: same states, same order of operations, same
float32 arithmetic. Stage 05's SimulEval runner and Stage 06's parity tests step
this module, so they execute the policy the C++ worker runs. Change a rule in
both files at once; tests/data/waitk/golden_trajectories.txt (written by
make_waitk_golden.py, replayed by tests/test_waitk_policy.cpp and
tests/test_waitk_policy.py) pins the two together.

Standard library only (no numpy/torch), so the golden generator runs anywhere.

One policy step is one translation chunk (chunk_frames = 4 frames of 20 ms =
80 ms). The caller's order inside a step:

    policy.on_chunk_energy(speech, elapsed_ms)      # VAD verdict for the chunk
    if policy.source_wanted():                      # not idle, not flushing
        run translator_enc
        policy.notify_source_unit()
        policy.on_source_readiness(expected_target_chunks)     # adaptive only, [0, 2]
    decision = policy.decide()
    for _ in range(policy.max_writes_this_step()):  # 0 on READ
        run translator_dec (+ vocoder)
        policy.notify_target_unit(is_eos)
        if policy.flushes_completed != before: break  # utterance drained

Phases:
    IDLE      no utterance: leading silence is not counted (it can never trigger
              a flush) and source chunks are not wanted.
    ACTIVE    READ while t_src - t_tgt < k, else WRITE. Silence accumulates; a
              pause longer than tail_flush_ms starts the tail flush.
    FLUSHING  decoder-only steps until <EOS>, the flush bound
              (t_tgt >= ceil(max_flush_ratio * t_src)) or max_flush_steps, then
              reset() to IDLE. Speech during the flush returns to ACTIVE with the
              counters (and so the caller's KV caches) untouched.

A caller that never reports energy has no VAD: notify_source_unit() then opens
the utterance itself (the pre-Stage-04 behaviour).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntEnum

K_MIN = 1
K_MAX = 10
TAIL_FLUSH_MS = 400
CHUNK_FRAMES = 4
FRAME_MS = 20
_U32_MAX = 0xFFFFFFFF


def _f32(x: float) -> float:
    """Round a Python float to the nearest IEEE-754 binary32 (C++ `float`).
    Out-of-range values become +-inf, like a C++ double -> float conversion
    (struct.pack raises instead)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Decision(IntEnum):
    READ = 0
    WRITE = 1
    FLUSH = 2


class Phase(IntEnum):
    IDLE = 0
    ACTIVE = 1
    FLUSHING = 2


@dataclass
class WaitKConfig:
    """Mirror of WaitKPolicy::Config (same defaults, same clamping)."""
    k: int = 3                         # chunks of lag before the first WRITE, 1..10
    tail_flush_ms: int = TAIL_FLUSH_MS
    max_flush_ratio: float = 1.5       # flush bound: t_tgt >= ceil(ratio * t_src)
    max_flush_steps: int = 32          # hard cap on decoder-only flush steps
    adaptive_mode: bool = False
    readiness_threshold: float = 0.5   # readiness below this counts as "blank"
    adaptive_extra_wait: int = 2       # adaptive may hold back up to k + this
    chunk_frames: int = CHUNK_FRAMES
    frame_ms: int = FRAME_MS

    def sanitized(self) -> "WaitKConfig":
        ratio = _f32(self.max_flush_ratio)
        if not (ratio >= 1.0):          # also catches NaN
            ratio = 1.0
        ratio = min(ratio, 4.0)
        thr = _f32(self.readiness_threshold)
        if not (thr >= 0.0):
            thr = 0.0
        thr = min(thr, 1.0)
        return WaitKConfig(
            k=_clamp(int(self.k), K_MIN, K_MAX),
            tail_flush_ms=_clamp(int(self.tail_flush_ms), 0, _U32_MAX),
            max_flush_ratio=ratio,
            max_flush_steps=_clamp(int(self.max_flush_steps), 1, 1024),
            adaptive_mode=bool(self.adaptive_mode),
            readiness_threshold=thr,
            adaptive_extra_wait=_clamp(int(self.adaptive_extra_wait), 0, K_MAX),
            chunk_frames=_clamp(int(self.chunk_frames), 1, 16),
            frame_ms=_clamp(int(self.frame_ms), 1, 1000),
        )


class WaitKPolicy:
    def __init__(self, config: WaitKConfig | int = 3):
        if not isinstance(config, WaitKConfig):
            config = WaitKConfig(k=int(config))
        self._cfg = config.sanitized()
        self._k = self._cfg.k
        self._vad_seen = False
        self.flushes_completed = 0
        self.flush_resumes = 0
        self.flushes_without_eos = 0
        self.reset()

    # ---- configuration -------------------------------------------------
    @property
    def config(self) -> WaitKConfig:
        return self._cfg

    @property
    def k(self) -> int:
        return self._k

    def set_k(self, k: int) -> None:
        self._k = _clamp(int(k), K_MIN, K_MAX)

    def set_adaptive_mode(self, on: bool) -> None:
        """Live strict <-> CTC-adaptive switch (M3 Stage 09 `adaptive_policy`):
        takes effect on the next decide(); the banked readiness restarts from 0
        so a stale bank never releases a write in the new mode. t_src, t_tgt
        and the phase are kept (no translator reset)."""
        self._cfg.adaptive_mode = bool(on)
        self._readiness = 0.0

    @property
    def chunk_ms(self) -> int:
        return self._cfg.chunk_frames * self._cfg.frame_ms

    @property
    def wait_ms(self) -> int:
        return self._k * self.chunk_ms

    # ---- observers -----------------------------------------------------
    @property
    def t_src(self) -> int:
        return self._t_src

    @property
    def t_tgt(self) -> int:
        return self._t_tgt

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def silence_ms(self) -> int:
        return self._silence_ms

    @property
    def flush_steps(self) -> int:
        return self._flush_steps

    def flushing(self) -> bool:
        return self._phase == Phase.FLUSHING

    flush_pending = flushing            # pre-Stage-04 name

    def utterance_active(self) -> bool:
        return self._phase != Phase.IDLE

    def source_wanted(self) -> bool:
        """True when translator_enc should consume this step's source chunk."""
        if self._phase == Phase.ACTIVE:
            return True
        return self._phase == Phase.IDLE and not self._vad_seen

    # ---- inputs --------------------------------------------------------
    def on_chunk_energy(self, speech_detected: bool, elapsed_ms: int | None = None) -> None:
        if elapsed_ms is None:
            elapsed_ms = self.chunk_ms
        self._vad_seen = True
        if speech_detected:
            self._silence_ms = 0
            if self._phase == Phase.IDLE:
                self._phase = Phase.ACTIVE
            elif self._phase == Phase.FLUSHING:
                # Mid-flush resumption: keep (t_src, t_tgt); the caller keeps
                # its KV caches.
                self._phase = Phase.ACTIVE
                self._flush_steps = 0
                self.flush_resumes += 1
            if self._eos_seen:
                self._eos_seen = False
                if self._t_tgt < self._t_src:
                    self._t_tgt = self._t_src
            return
        if self._phase == Phase.IDLE:
            return                      # leading silence never counts
        self._silence_ms = min(self._silence_ms + int(elapsed_ms), _U32_MAX)
        if self._phase == Phase.ACTIVE and self._silence_ms > self._cfg.tail_flush_ms:
            self._begin_flush()

    def on_frame_energy(self, speech_detected: bool) -> bool:
        """50 Hz frame -> chunk aggregation. Returns True on the frame that
        closes a chunk (after forwarding the chunk verdict: speech if any of its
        frames had speech)."""
        self._frame_speech = self._frame_speech or bool(speech_detected)
        self._frame_count += 1
        if self._frame_count < self._cfg.chunk_frames:
            return False
        speech = self._frame_speech
        self._frame_count = 0
        self._frame_speech = False
        self.on_chunk_energy(speech, self.chunk_ms)
        return True

    def notify_source_unit(self) -> None:
        if self._phase == Phase.IDLE:
            if self._vad_seen:
                return
            self._phase = Phase.ACTIVE
        elif self._phase == Phase.FLUSHING:
            return
        self._t_src += 1

    def on_source_readiness(self, expected_chunks: float) -> None:
        """Expected number of target chunks the newest source chunk completed,
        in [0, 2]. A plain probability in [0, 1] works, but then a step can only
        hold a write back, never write twice."""
        if self._phase != Phase.ACTIVE:
            return
        x = _f32(expected_chunks)
        if not (x >= self._cfg.readiness_threshold):        # NaN-safe
            return
        x = min(x, 2.0)
        self._readiness = min(_f32(self._readiness + x), 2.0)

    def notify_target_unit(self, is_eos: bool = False) -> None:
        if self._phase == Phase.IDLE:
            return
        self._t_tgt += 1
        if self._cfg.adaptive_mode:
            self._readiness = max(_f32(self._readiness - 1.0), 0.0)
        if self._phase == Phase.FLUSHING:
            self._flush_steps += 1
            if (is_eos or self._t_tgt >= self._flush_target
                    or self._flush_steps >= self._cfg.max_flush_steps):
                self._complete(is_eos)
            return
        if is_eos and self._silence_ms > 0:
            # <EOS> while the user is already silent: caught up, so the pending
            # tail flush has nothing left to drain.
            self._eos_seen = True
            if self._t_tgt < self._t_src:
                self._t_tgt = self._t_src

    # ---- decisions -----------------------------------------------------
    def decide(self) -> Decision:
        if self._phase == Phase.FLUSHING:
            return Decision.FLUSH
        if self._phase == Phase.IDLE or self._eos_seen:
            return Decision.READ
        lag = self._t_src - self._t_tgt
        if lag < self._k:
            return Decision.READ
        if not self._cfg.adaptive_mode:
            return Decision.WRITE
        if self._readiness >= 1.0 or lag >= self._k + self._cfg.adaptive_extra_wait:
            return Decision.WRITE
        return Decision.READ

    def max_writes_this_step(self) -> int:
        d = self.decide()
        if d == Decision.READ:
            return 0
        if d == Decision.FLUSH or not self._cfg.adaptive_mode:
            return 1
        lag = self._t_src - self._t_tgt
        if lag > self._k and (self._readiness >= 2.0
                              or lag > self._k + self._cfg.adaptive_extra_wait):
            return 2
        return 1

    def reset(self) -> None:
        self._phase = Phase.IDLE
        self._t_src = 0
        self._t_tgt = 0
        self._silence_ms = 0
        self._flush_steps = 0
        self._flush_target = 0
        self._eos_seen = False
        self._readiness = 0.0
        self._frame_count = 0
        self._frame_speech = False

    # ---- internals -----------------------------------------------------
    def _begin_flush(self) -> None:
        if self._eos_seen or self._t_src == 0:
            self._complete(self._eos_seen)
            return
        self._flush_target = math.ceil(self._cfg.max_flush_ratio * self._t_src)
        if self._t_tgt >= self._flush_target:
            self._complete(False)
            return
        self._phase = Phase.FLUSHING
        self._flush_steps = 0

    def _complete(self, by_eos: bool) -> None:
        self.reset()
        self.flushes_completed += 1
        if not by_eos:
            self.flushes_without_eos += 1


# ---------------------------------------------------------------------------
# UnitProsodyFifo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PushResult:
    pushed: int        # frames queued (0 on overflow)
    eos: bool          # the chunk contained the <EOS> unit
    overflow: bool     # not enough room: nothing was queued


class UnitProsodyFifo:
    """Fixed-capacity frame ring between translator_dec (chunk_frames frames per
    step) and the vocoder (1, 2 or 4 frames per hop). A frame is
    (unit, z_logf0, voicing, aperiodicity, z_energy, pad): the contract's
    `prosody` column order (stored as float32, like the C++ Frame) plus the
    pad flag of pad_to_multiple().

    BUG-028 (stranded tail frames): push_chunk() drops everything from the
    first special unit, so an <EOS> chunk queues 1..3 frames that an s2/s4
    vocoder could never pop. After each push the Mode B pipeline calls
    pad_to_multiple(hop frames); the pad frames keep the queue hop-aligned and
    the pipeline renders them as silence."""

    PROSODY_DIMS = 4

    def __init__(self, capacity: int = 64, vocab: int = 1000, eos_unit: int | None = None):
        self.capacity = max(1, int(capacity))
        self.vocab = int(vocab)
        self.eos_unit = self.vocab + 1 if eos_unit is None else int(eos_unit)
        self._buf = [None] * self.capacity
        self._head = 0
        self._size = 0
        self.overflows = 0

    def size(self) -> int:
        return self._size

    def free(self) -> int:
        return self.capacity - self._size

    def empty(self) -> bool:
        return self._size == 0

    def clear(self) -> None:
        self._head = 0
        self._size = 0

    def push_chunk(self, units, prosody) -> PushResult:
        """`prosody` is flat, row-major [n, 4]. Frames from the first special
        unit (< 0 or >= vocab: <PAD>/<EOS>/<BOS>) onwards are dropped. All valid
        frames are queued, or none (overflow)."""
        units = [int(u) for u in units]
        prosody = [_f32(float(p)) for p in prosody]
        n = min(len(units), len(prosody) // self.PROSODY_DIMS)
        valid = n
        eos = False
        for i in range(n):
            u = units[i]
            if u == self.eos_unit:
                eos = True
            if valid == n and (u < 0 or u >= self.vocab):
                valid = i
        if valid > self.free():
            self.overflows += 1
            return PushResult(0, eos, True)
        tail = (self._head + self._size) % self.capacity
        for i in range(valid):
            self._buf[tail] = (units[i], *prosody[4 * i:4 * i + 4], False)
            tail = (tail + 1) % self.capacity
        self._size += valid
        return PushResult(valid, eos, False)

    def pad_to_multiple(self, n_frames: int) -> int:
        """Complete the queue to a multiple of `n_frames` (the vocoder hop) with
        pad frames: the newest frame's unit, z_logf0 and z_energy, voicing 0,
        aperiodicity 1, pad = True. Returns the number of pad frames appended:
        0 when already aligned, when empty or when n_frames <= 1. Needs room
        for them (keep capacity a multiple of n_frames); without it nothing is
        padded and an overflow is counted."""
        n_frames = int(n_frames)
        if n_frames <= 1 or self._size == 0:
            return 0
        rem = self._size % n_frames
        if rem == 0:
            return 0
        pads = n_frames - rem
        if pads > self.free():
            self.overflows += 1
            return 0
        last = self._buf[(self._head + self._size - 1) % self.capacity]
        filler = (last[0], last[1], 0.0, 1.0, last[4], True)
        tail = (self._head + self._size) % self.capacity
        for _ in range(pads):
            self._buf[tail] = filler
            tail = (tail + 1) % self.capacity
        self._size += pads
        return pads

    def pop_hop_with_pads(self, n_frames: int):
        """Returns (units, prosody rows, pad flags) for exactly n_frames, or
        None when fewer are queued (the caller emits silence). Rows are
        (z_logf0, voicing, aperiodicity, z_energy)."""
        if n_frames > self._size:
            return None
        units, rows, pads = [], [], []
        for _ in range(n_frames):
            f = self._buf[self._head]
            units.append(f[0])
            rows.append(f[1:5])
            pads.append(f[5])
            self._head = (self._head + 1) % self.capacity
        self._size -= n_frames
        return units, rows, pads

    def pop_hop(self, n_frames: int):
        """Returns (units, prosody rows) for exactly n_frames, or None when
        fewer are queued (the caller emits silence)."""
        popped = self.pop_hop_with_pads(n_frames)
        return None if popped is None else popped[:2]


# ---------------------------------------------------------------------------
# Trajectory replay (golden vectors + cross-language parity)
# ---------------------------------------------------------------------------
READINESS_LEVELS = 7        # step readiness = 2 * q / 7 with q in 0..7
_WRITE_CHARS = "RWX"        # READ, WRITE x1, WRITE x2 (FLUSH is "F")


@dataclass(frozen=True)
class Trajectory:
    config: WaitKConfig
    elapsed_ms: int
    steps: str              # one char per step: chr(48 + speech | q << 1 | eos << 4)
    set_k: tuple = ()       # ((step_index, k), ...) applied before that step

    def header(self) -> str:
        c = self.config
        setk = ",".join(f"{i}={k}" for i, k in self.set_k) or "-"
        return (f"{c.k} {int(c.adaptive_mode)} {c.adaptive_extra_wait} {c.tail_flush_ms} "
                f"{c.max_flush_ratio!r} {c.max_flush_steps} {c.readiness_threshold!r} "
                f"{self.elapsed_ms} {self.steps} {setk}")


@dataclass(frozen=True)
class ReplayResult:
    decisions: str
    t_src: int
    t_tgt: int
    flushes: int
    resumes: int
    checksum: int

    def footer(self) -> str:
        return (f"{self.decisions} {self.t_src} {self.t_tgt} {self.flushes} "
                f"{self.resumes} {self.checksum}")


def encode_step(speech: bool, q: int, eos: bool) -> str:
    return chr(48 + (int(bool(speech)) | (int(q) & 7) << 1 | int(bool(eos)) << 4))


def replay(traj: Trajectory) -> ReplayResult:
    """The canonical step driver (module docstring), one chunk per step char."""
    policy = WaitKPolicy(traj.config)
    set_k = dict(traj.set_k)
    out = []
    h = 0
    for i, ch in enumerate(traj.steps):
        bits = ord(ch) - 48
        speech, q, eos = bool(bits & 1), (bits >> 1) & 7, bool(bits & 16)
        if i in set_k:
            policy.set_k(set_k[i])
        policy.on_chunk_energy(speech, traj.elapsed_ms)
        if policy.source_wanted():
            policy.notify_source_unit()
            policy.on_source_readiness(_f32(2.0 * float(q) / float(READINESS_LEVELS)))
        d = policy.decide()
        n = policy.max_writes_this_step()
        before = policy.flushes_completed
        for w in range(n):
            policy.notify_target_unit(eos and w == 0)
            if policy.flushes_completed != before:
                break
        out.append("F" if d == Decision.FLUSH else _WRITE_CHARS[n])
        h = (h * 1000003 + (policy.t_src & 0xFFFF) * 257 + (policy.t_tgt & 0xFFFF) * 17
             + policy.flushes_completed) & 0xFFFFFFFF
    return ReplayResult("".join(out), policy.t_src, policy.t_tgt, policy.flushes_completed,
                        policy.flush_resumes, h)


def parse_trajectory(line: str) -> tuple[Trajectory, str]:
    """Split a golden line into (trajectory, expected footer)."""
    f = line.split()
    if len(f) != 16:
        raise ValueError(f"expected 16 fields, got {len(f)}")
    cfg = WaitKConfig(k=int(f[0]), adaptive_mode=bool(int(f[1])), adaptive_extra_wait=int(f[2]),
                      tail_flush_ms=int(f[3]), max_flush_ratio=float(f[4]),
                      max_flush_steps=int(f[5]), readiness_threshold=float(f[6]))
    set_k = () if f[9] == "-" else tuple(
        (int(a), int(b)) for a, b in (kv.split("=") for kv in f[9].split(",")))
    return Trajectory(cfg, int(f[7]), f[8], set_k), " ".join(f[10:])


def random_trajectory(rng) -> Trajectory:
    """One randomized speech/pause trajectory (`rng` is a random.Random). Speech
    comes in bursts separated by pauses on both sides of the tail-flush
    threshold, so flushes, resumptions and leading silence all occur."""
    elapsed = rng.choice((80, 80, 80, 40, 160))
    cfg = WaitKConfig(
        k=rng.randint(K_MIN, K_MAX),
        adaptive_mode=rng.random() < 0.5,
        adaptive_extra_wait=rng.randint(0, 4),
        tail_flush_ms=rng.choice((400, 400, 240, 800)),
        max_flush_ratio=rng.choice((1.0, 1.25, 1.5, 2.0)),
        max_flush_steps=rng.choice((1, 3, 8, 32)),
        readiness_threshold=rng.choice((0.25, 0.5, 0.75)),
    )
    n_steps = rng.randint(20, 120)
    steps = []
    speech = rng.random() < 0.5
    while len(steps) < n_steps:
        if speech:
            run = rng.randint(1, 30)
        else:
            run = rng.randint(1, 3 * max(1, cfg.tail_flush_ms // elapsed))
        for _ in range(run):
            eos = rng.random() < (0.02 if speech else 0.12)
            steps.append(encode_step(speech, rng.randint(0, READINESS_LEVELS), eos))
        speech = not speech
    steps = steps[:n_steps]
    set_k = tuple(sorted({rng.randrange(n_steps): rng.randint(-2, 13)
                          for _ in range(rng.choice((0, 0, 1, 2)))}.items()))
    return Trajectory(cfg, elapsed, "".join(steps), set_k)
