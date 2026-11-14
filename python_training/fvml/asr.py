"""Intelligibility & translation ASR: Whisper large-v3 transcripts, language-aware
text normalisation, corpus-level WER/CER (Mode A) and SacreBLEU ASR-BLEU /
ASR-chrF++ (Mode B)."""
import unicodedata

import jiwer
import numpy as np

from . import audio

MODEL = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"   # pinned 2026-10-18 (Stage 01)
SUPPORTED_LANGS = ("en", "es", "fr", "de")


def active_frames_mask(wave, frame_ms=20.0, threshold_dbfs=-55.0):
    """Boolean mask of 20 ms frames whose RMS exceeds `threshold_dbfs`."""
    x = np.asarray(wave, dtype=np.float32)
    frame_len = max(1, int(round(audio.SAMPLE_RATE * frame_ms / 1000.0)))
    if len(x) == 0:
        return np.zeros((0,), dtype=bool)
    pad = (-len(x)) % frame_len
    if pad:
        x = np.pad(x, (0, pad))
    frames = x.reshape(-1, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-16)
    dbfs = 20.0 * np.log10(rms)
    return dbfs > threshold_dbfs


def is_silent(wave, threshold_dbfs=-55.0):
    """True when no 20 ms frame exceeds `threshold_dbfs`."""
    return not bool(np.any(active_frames_mask(wave, threshold_dbfs=threshold_dbfs)))


def normalize_multilingual(text):
    """Lowercase NFKC normalisation stripping punctuation/symbols while
    preserving non-English letters and diacritics (ä, ö, ü, ß, ñ, é, è, ç, ...),
    matching Whisper's BasicTextNormalizer(remove_diacritics=False)."""
    norm = unicodedata.normalize("NFKC", text or "").lower()
    chars = [
        ch if unicodedata.category(ch)[0] in ("L", "N", "M") else " "
        for ch in norm
    ]
    return " ".join("".join(chars).split())


def normalize_text(text, language="en", english_normalizer=None):
    """Normalise a transcript or reference for `language` ('en', 'es', 'fr', 'de')."""
    if language == "en" and english_normalizer is not None:
        return english_normalizer(text or "")
    return normalize_multilingual(text)


class Whisper:
    """Batched greedy decoding with model.generate().

    Not the transformers "automatic-speech-recognition" pipeline: in
    transformers 5.20 it re-appends the cross-attention cache at every decoding
    step, so a batch of 16 needed > 20 GiB and spilled into shared memory
    (~75 s per utterance). generate() on the same model: ~0.4 s per utterance,
    < 5 GiB peak at batch 8."""

    def __init__(self, device="cuda", batch_size=8, model=MODEL, revision=REVISION):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device = "cuda:0" if device == "cuda" else device
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, revision=revision, dtype=self.dtype).to(self.device).eval()
        self.batch_size = batch_size

    def transcribe(self, waves, language="en", silence_dbfs=None):
        """16 kHz mono float arrays (each < 30 s) -> raw transcripts.

        Always forces `language` (Stage 01 note: short S2UT clips with foreign
        accents or proper nouns can otherwise flip Whisper's language ID).
        When `silence_dbfs` is set (e.g. -55.0 in eval_s2ut), pure-silence clips
        return "" without invoking greedy decoding (which otherwise hallucinates
        boilerplate tokens on all-zero spectrograms).
        """
        import torch
        if silence_dbfs is None:
            active_idx = list(range(len(waves)))
        else:
            active_idx = [i for i, w in enumerate(waves) if not is_silent(w, silence_dbfs)]
        out = [""] * len(waves)
        for s in range(0, len(active_idx), self.batch_size):
            idxs = active_idx[s:s + self.batch_size]
            batch = [np.asarray(waves[i], np.float32) for i in idxs]
            feats = self.processor.feature_extractor(batch, sampling_rate=audio.SAMPLE_RATE,
                                                     return_tensors="pt")
            with torch.inference_mode():
                ids = self.model.generate(feats.input_features.to(self.device, self.dtype),
                                          language=language, task="transcribe", max_new_tokens=440)
            decoded = [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
            for i, text in zip(idxs, decoded):
                out[i] = text
        return out

    def normalize(self, text, language="en"):
        """Whisper's EnglishTextNormalizer for English; diacritic-preserving
        lowercase punctuation strip for Es/Fr/De."""
        return normalize_text(text, language=language,
                              english_normalizer=self.processor.tokenizer.normalize)

    def close(self):
        self.model = None
        free_gpu()


def error_rates(refs, hyps):
    """Normalised reference/hypothesis lists -> corpus and per-utterance rates.

    Utterances whose reference normalises to nothing are skipped (jiwer
    cannot score them) and counted in `skipped`."""
    keep = [i for i, r in enumerate(refs) if r.strip()]
    if not keep:
        return {"wer": np.nan, "cer": np.nan, "per_wer": [np.nan] * len(refs),
                "per_cer": [np.nan] * len(refs), "skipped": len(refs)}
    r = [refs[i] for i in keep]
    h = [hyps[i] for i in keep]
    words = jiwer.process_words(r, h)
    chars = jiwer.process_characters(r, h)
    per_wer = [np.nan] * len(refs)
    per_cer = [np.nan] * len(refs)
    for i in keep:
        per_wer[i] = jiwer.wer(refs[i], hyps[i])
        per_cer[i] = jiwer.cer(refs[i], hyps[i])
    return {"wer": words.wer, "cer": chars.cer, "per_wer": per_wer, "per_cer": per_cer,
            "skipped": len(refs) - len(keep)}


def translation_scores(refs, hyps):
    """Corpus and per-utterance SacreBLEU (13a) and chrF++ (word_order=2) on
    0..100 scale. Rows with an empty reference (e.g. noise-only clips) are
    skipped from BLEU/chrF++ aggregation and counted in `skipped`."""
    from sacrebleu.metrics import BLEU, CHRF
    bleu_metric = BLEU(tokenize="13a", effective_order=True)
    corpus_bleu_metric = BLEU(tokenize="13a")
    chrf_metric = CHRF(word_order=2)

    keep = [i for i, r in enumerate(refs) if r.strip()]
    per_bleu = [np.nan] * len(refs)
    per_chrf = [np.nan] * len(refs)
    if not keep:
        return {"asr_bleu": np.nan, "asr_chrfpp": np.nan, "per_bleu": per_bleu,
                "per_chrfpp": per_chrf, "skipped": len(refs)}
    r_keep = [refs[i] for i in keep]
    h_keep = [hyps[i] for i in keep]
    corpus_bleu = float(corpus_bleu_metric.corpus_score(h_keep, [r_keep]).score)
    corpus_chrf = float(chrf_metric.corpus_score(h_keep, [r_keep]).score)
    for i in keep:
        per_bleu[i] = float(bleu_metric.sentence_score(hyps[i], [refs[i]]).score)
        per_chrf[i] = float(chrf_metric.sentence_score(hyps[i], [refs[i]]).score)
    return {
        "asr_bleu": corpus_bleu,
        "asr_chrfpp": corpus_chrf,
        "per_bleu": per_bleu,
        "per_chrfpp": per_chrf,
        "skipped": len(refs) - len(keep),
    }


def free_gpu():
    """Give a dropped model's GPU memory back before loading the next."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
