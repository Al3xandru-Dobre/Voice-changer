"""Intelligibility: Whisper large-v3 transcripts, Whisper's English text
normaliser on both sides, corpus-level WER/CER (total errors / total
reference words or characters)."""
import jiwer
import numpy as np

from . import audio

MODEL = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"   # pinned 2026-10-18 (Stage 01)


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

    def transcribe(self, waves):
        """16 kHz mono float arrays (each < 30 s) -> raw transcripts."""
        import torch
        out = []
        for s in range(0, len(waves), self.batch_size):
            batch = [np.asarray(w, np.float32) for w in waves[s:s + self.batch_size]]
            feats = self.processor.feature_extractor(batch, sampling_rate=audio.SAMPLE_RATE,
                                                     return_tensors="pt")
            with torch.inference_mode():
                ids = self.model.generate(feats.input_features.to(self.device, self.dtype),
                                          language="en", task="transcribe", max_new_tokens=440)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out

    def normalize(self, text):
        """Whisper's EnglishTextNormalizer (with the model's spelling map)."""
        return self.processor.tokenizer.normalize(text)

    def close(self):
        self.model = None
        free_gpu()


def error_rates(refs, hyps):
    """Normalised reference/hypothesis lists -> corpus and per-utterance rates.

    Utterances whose reference normalises to nothing are skipped (jiwer
    cannot score them) and counted in `skipped`."""
    keep = [i for i, r in enumerate(refs) if r.strip()]
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


def free_gpu():
    """Give a dropped model's GPU memory back before loading the next."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
