"""fvml — shared modules for FluxVoice's Milestone 2 & 3 training and evaluation.

Top-level scripts in python_training/ are thin CLIs over these modules:

  contract   model contract v1 & v2 (M2/M3 Stage 02); standard library only
  paths      repo root and the FLUXVOICE_DATA data root
  audio      16 kHz mono float I/O (soundfile + soxr)
  datasets   M2 dataset registry, download/verify, JSONL manifests (M2 Stage 01)
  s2ut_data  M3 bidirectional En<->{Es,Fr,De} data prep, canonical TTS + CER gate,
             CSS10/CML-TTS vocoder filtering, NC isolation, SentencePiece (M3 Stage 01)
  evallist   frozen evaluation lists (eval_lists/*.tsv, eval_lists/s2ut/*.tsv)
  parity     signal / tensor comparison (streaming vs offline, PyTorch vs ORT)
  asr, speaker, mos, f0_ref, evaluate
             the eval_vc.py metric stack: Whisper WER/CER, ECAPA cosine/EER,
             UTMOS, offline reference F0
  s2ut_eval  the eval_s2ut.py metric stack: forced-language Whisper ASR-BLEU &
             ASR-chrF++, SimulEval Average Lagging (AL / AL_CA), onset delay,
             ECAPA cosine, UTMOS, and tail-flush / silence hallucination audit
"""
