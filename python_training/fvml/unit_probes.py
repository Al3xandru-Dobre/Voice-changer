"""Unit probes (Milestone 2 / Stage 04): phone purity, cluster purity and PNMI
against forced-aligned phones, and a speaker-ID leakage probe.

Definitions follow the HuBERT paper (Hsu et al. 2021, §IV-C) on the joint
distribution p(phone, unit) of paired frame labels:
  phone purity   = E_unit[max_phone p(phone | unit)]  = sum_u max_p p(p, u)
  cluster purity = E_phone[max_unit p(unit | phone)]  = sum_p max_u p(p, u)
  PNMI           = I(phone; unit) / H(phone)

Frame labels come from Montreal Forced Aligner TextGrids: the phone that
overlaps each frame's window most (majority label), at the unit stream's own
frame rate. Frame i's window is centred on centre_offset + i / frame_rate
(default half a frame: [i, i+1) / frame_rate, which is where both the 50 Hz
streams and Mimi's 12.5 Hz frames sit). Stress digits are stripped (AH0 -> AH);
empty, sil, sp and <eps> become "sil".

TextGrid reading is standard library: a tokenizer that keeps quoted strings
and numbers (skipping keys and [n] indices) reads Praat's long "ooTextFile"
format, and the short one too, since both list the same values in order.
"""
import collections
import pathlib
import re

import numpy as np

SILENCE = {"", "sil", "sp", "<eps>", "silence"}
_TOKEN = re.compile(r'"((?:[^"]|"")*)"|\[\s*\d*\s*\]|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')


# -- information-theoretic probes ---------------------------------------------
def contingency(phones, units):
    """Joint count table (n_phones, n_units) plus the row/column labels."""
    phones, units = np.asarray(phones), np.asarray(units)
    if phones.shape != units.shape or phones.ndim != 1:
        raise ValueError(f"need equal-length 1-D label arrays, got {phones.shape} and {units.shape}")
    p_lab, p_idx = np.unique(phones, return_inverse=True)
    u_lab, u_idx = np.unique(units, return_inverse=True)
    table = np.zeros((len(p_lab), len(u_lab)), np.int64)
    np.add.at(table, (p_idx, u_idx), 1)
    return table, p_lab, u_lab


def phone_purity(table):
    joint = table / table.sum()
    return float(joint.max(axis=0).sum())


def cluster_purity(table):
    joint = table / table.sum()
    return float(joint.max(axis=1).sum())


def pnmi(table):
    joint = table / table.sum()
    pp, pu = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pp @ pu)[nz])).sum())
    h = float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum())
    return mi / h if h > 0 else 0.0


def probe_scores(phones, units):
    table, p_lab, u_lab = contingency(phones, units)
    return {"phone_purity": phone_purity(table), "cluster_purity": cluster_purity(table),
            "pnmi": pnmi(table), "frames": int(table.sum()), "phones": int(len(p_lab)),
            "units_used": int(len(u_lab))}


# -- TextGrid alignments -------------------------------------------------------
def read_textgrid(source):
    """Path or text of a Praat TextGrid -> {tier name: [(xmin, xmax, text)]} for
    interval tiers (point tiers -> [(time, time, mark)])."""
    is_text = isinstance(source, str) and ("\n" in source or source.lstrip().startswith("File type"))
    if not is_text:
        raw = pathlib.Path(source).read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    else:
        text = str(source)
    toks = []
    for m in _TOKEN.finditer(text):
        if m.group(1) is not None:
            toks.append(m.group(1).replace('""', '"'))
        elif m.group(2) is not None:
            toks.append(float(m.group(2)))
    if toks[:2] != ["ooTextFile", "TextGrid"]:
        raise ValueError("not a Praat TextGrid (expected File type = \"ooTextFile\", "
                         "Object class = \"TextGrid\")")
    pos = 4                                      # header, xmin, xmax
    n_tiers = int(toks[pos])
    pos += 1
    tiers = {}
    for _ in range(n_tiers):
        cls, name = toks[pos], toks[pos + 1]
        n = int(toks[pos + 4])
        pos += 5
        items = []
        if cls == "IntervalTier":
            for _ in range(n):
                items.append((float(toks[pos]), float(toks[pos + 1]), str(toks[pos + 2])))
                pos += 3
        elif cls == "TextTier":
            for _ in range(n):
                items.append((float(toks[pos]), float(toks[pos]), str(toks[pos + 1])))
                pos += 2
        else:
            raise ValueError(f"unknown tier class {cls!r}")
        tiers[name] = items
    return tiers


def normalize_phone(p):
    p = p.strip()
    if p.lower() in SILENCE:
        return "sil"
    return re.sub(r"\d+$", "", p).upper()


def frame_labels(intervals, n_frames, frame_rate, centre_offset=None):
    """Majority phone per frame: the interval that overlaps frame i's window
    (width 1/frame_rate, centred on centre_offset + i/frame_rate) the most.
    Frames no interval touches are "sil"."""
    if centre_offset is None:
        centre_offset = 0.5 / frame_rate
    ivs = [(s, e, normalize_phone(t)) for s, e, t in intervals if e > s]
    if not ivs or n_frames <= 0:
        return ["sil"] * max(n_frames, 0)
    s = np.array([v[0] for v in ivs])
    e = np.array([v[1] for v in ivs])
    c = centre_offset + np.arange(n_frames) / frame_rate
    lo, hi = c - 0.5 / frame_rate, c + 0.5 / frame_rate
    overlap = np.clip(np.minimum(hi[:, None], e[None]) - np.maximum(lo[:, None], s[None]), 0, None)
    best = overlap.argmax(1)
    return [ivs[b][2] if overlap[i, b] > 0 else "sil" for i, b in enumerate(best)]


def index_alignments(align_dir):
    """{utterance id (file stem): TextGrid path} under align_dir, recursively."""
    out = {}
    for p in pathlib.Path(align_dir).rglob("*.TextGrid"):
        out.setdefault(p.stem, p)
    return out


def phone_tier(tiers, name="phones"):
    if name in tiers:
        return tiers[name]
    for k, v in tiers.items():       # MFA 1.x names the tier "<speaker> - phones"
        if k.endswith(name):
            return v
    raise KeyError(f"no '{name}' tier (tiers: {', '.join(tiers)})")


# -- speaker-ID probe ----------------------------------------------------------
def ngram_features(seqs, vocab, max_bigram_features=1 << 17):
    """Per-utterance unigram + bigram relative frequencies (each order sums to 1)
    as a CSR matrix. Bigrams index exactly when vocab**2 fits, else they are
    hashed into max_bigram_features columns (Mimi's 2048**2 would not fit)."""
    from scipy import sparse
    n_bi = vocab * vocab if vocab * vocab <= max_bigram_features else max_bigram_features
    data, cols, ptr = [], [], [0]
    for s in seqs:
        s = np.asarray(s, np.int64)
        uni = collections.Counter(s.tolist())
        bi = collections.Counter(((s[:-1] * vocab + s[1:]) % n_bi).tolist()) if len(s) > 1 else {}
        nu, nb = max(len(s), 1), max(len(s) - 1, 1)
        for k, v in uni.items():
            cols.append(k)
            data.append(v / nu)
        for k, v in bi.items():
            cols.append(vocab + k)
            data.append(v / nb)
        ptr.append(len(cols))
    return sparse.csr_matrix((np.array(data, np.float64), np.array(cols, np.int64), np.array(ptr)),
                             shape=(len(seqs), vocab + n_bi))


def split_speakers(items, n_speakers=100, test_fraction=0.2, min_utts=5, seed=0):
    """items: [(speaker, payload)] -> (train, test) lists over n_speakers chosen
    at random among speakers with >= min_utts utterances; each speaker keeps
    round(test_fraction * count) (at least 1) utterances held out."""
    by_spk = collections.defaultdict(list)
    for spk, x in items:
        by_spk[spk].append(x)
    eligible = sorted(s for s, v in by_spk.items() if len(v) >= min_utts)
    if len(eligible) < 2:
        raise ValueError(f"need >= 2 speakers with >= {min_utts} utterances, have {len(eligible)}")
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=min(n_speakers, len(eligible)), replace=False).tolist())
    train, test = [], []
    for spk in chosen:
        utts = list(by_spk[spk])
        rng.shuffle(utts)
        k = max(1, int(round(test_fraction * len(utts))))
        test += [(spk, x) for x in utts[:k]]
        train += [(spk, x) for x in utts[k:]]
    return train, test, chosen


def speaker_probe(items, vocab, n_speakers=100, test_fraction=0.2, min_utts=5, seed=0, C=1.0):
    """Logistic regression on unit n-gram histograms. items: [(speaker, ids)].
    Returns accuracy on held-out utterances vs. the 1/N chance level."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    train, test, chosen = split_speakers(items, n_speakers, test_fraction, min_utts, seed)
    xtr = ngram_features([x for _, x in train], vocab)
    xte = ngram_features([x for _, x in test], vocab)
    scaler = StandardScaler(with_mean=False).fit(xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(xtr), [s for s, _ in train])
    pred = clf.predict(scaler.transform(xte))
    acc = float(np.mean(pred == np.array([s for s, _ in test])))
    return {"accuracy": acc, "chance": 1.0 / len(chosen), "speakers": len(chosen),
            "train_utts": len(train), "test_utts": len(test), "features": int(xtr.shape[1])}
