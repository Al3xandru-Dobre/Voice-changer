"""Model-free pieces of the evaluation stack: parity, eval lists, WER
aggregation, the reference F0 tracker and its alignment."""
import numpy as np
import pytest

from fvml import asr, evallist, f0_ref, parity, speaker

SR = 16000


def test_parity_identical_and_divergent():
    x = np.sin(np.linspace(0, 20, 1000)).astype(np.float32)
    r = parity.compare(x, x.copy())
    assert r.max_abs == 0 and r.snr_db == float("inf") and r.first_divergence is None
    y = x.copy()
    y[123] += 1e-3
    r = parity.compare(x, y, atol=1e-4)
    # signal energy ~500 (1000 samples of a unit sine) over 1e-6 of error: ~87 dB
    assert r.first_divergence == (123,) and abs(r.max_abs - 1e-3) < 1e-6 and 85 < r.snr_db < 89
    logits = np.random.default_rng(0).standard_normal((50, 10))
    r = parity.compare(logits, logits + 1e-7, argmax_axis=-1)
    assert r.argmax_agreement == 1.0
    with pytest.raises(ValueError, match="shape"):
        parity.compare(x, x[:-1])


def test_parity_step_detects_a_click():
    x = 0.1 * np.sin(np.arange(SR) * 2 * np.pi * 200 / SR)
    clicked = x.copy()
    clicked[5000] += 0.5
    assert parity.max_step(clicked) > 5 * parity.max_step(x)
    assert parity.step_percentile(x) <= parity.max_step(x)


def test_evallist_round_trip(tmp_path):
    rows = [{"utt_id": "a", "speaker": "1", "gender": "F", "duration": 3.14159, "path": "x/a.flac",
             "text": "HELLO"}]
    evallist.write(tmp_path / "l.tsv", rows, evallist.SOURCE_COLUMNS)
    back = evallist.read_sources(tmp_path / "l.tsv")
    assert back[0]["duration"] == 3.142 and back[0]["text"] == "HELLO"
    assert evallist.evenly_spaced(list(range(10)), 3) == [0, 4, 9]   # round(4.5) = 4 (banker's)
    with pytest.raises(ValueError):
        evallist.to_text([{**rows[0], "text": "a\tb"}], evallist.SOURCE_COLUMNS)


def test_error_rates_are_corpus_level():
    refs = ["the cat sat", "a b c d e f g h i j"]
    hyps = ["the cat", "a b c d e f g h i j"]
    r = asr.error_rates(refs, hyps)
    assert r["wer"] == pytest.approx(1 / 13)        # 1 error over 13 reference words
    assert r["per_wer"][0] == pytest.approx(1 / 3)
    r = asr.error_rates(["", "x y"], ["junk", "x y"])  # empty references are skipped
    assert r["skipped"] == 1 and r["wer"] == 0 and np.isnan(r["per_wer"][0])


def harmonic(f0, seconds=1.0, delay_s=0.0):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(0.2 / k * np.sin(2 * np.pi * k * np.cumsum(f0(t)) / SR) for k in range(1, 6))
    x *= (t > 0.2) & (t < 0.8)                      # on/off envelope for alignment
    shift = int(delay_s * SR)
    return np.concatenate([np.zeros(shift), x])[: len(t)].astype(np.float32)


def test_reference_tracker_and_pcc():
    glide = lambda t: 150 + 100 * t                  # noqa: E731  150 -> 250 Hz
    x = harmonic(glide)
    f = f0_ref.track(x)
    voiced = f[f > 0]
    assert len(voiced) > 40 and 150 < np.median(voiced) < 250
    assert f0_ref.pcc(f, f) == pytest.approx(1.0)
    up = f0_ref.track(harmonic(lambda t: 2 * glide(t)))    # +12 semitones
    assert np.median(up[up > 0]) / np.median(voiced) == pytest.approx(2.0, rel=0.03)


def test_alignment_recovers_a_delay():
    glide = lambda t: 150 + 100 * t                  # noqa: E731
    src, out = harmonic(glide), harmonic(glide, delay_s=0.04)
    lag = f0_ref.best_lag(src, out)
    assert lag == 4                                   # 40 ms in 10 ms frames
    f_src, f_out = f0_ref.track(src), f0_ref.track(out)
    assert f0_ref.pcc(f_src, f_out, lag) > f0_ref.pcc(f_src, f_out, 0)


def test_speaker_pair_stats_and_eer():
    rng = np.random.default_rng(1)
    centres = rng.standard_normal((4, 192))
    emb = np.repeat(centres, 5, axis=0) + 0.1 * rng.standard_normal((20, 192))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    s = speaker.pair_stats(emb, np.repeat(["a", "b", "c", "d"], 5))
    assert s["same_pairs"] == 40 and s["diff_pairs"] == 150
    assert s["same_spk_cos"] - s["diff_spk_cos"] > 0.5 and s["spk_eer"] == 0.0
