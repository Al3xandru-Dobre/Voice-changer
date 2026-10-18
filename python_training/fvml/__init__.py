"""fvml — shared modules for FluxVoice's Milestone 2 training and evaluation.

Top-level scripts in python_training/ are thin CLIs over these modules:

  contract   model contract v1 (Stage 02); standard library only
  paths      repo root and the FLUXVOICE_DATA data root
  audio      16 kHz mono float I/O (soundfile + soxr)
  datasets   dataset registry, download/verify, JSONL manifests (Stage 01)
  evallist   frozen evaluation lists (eval_lists/*.tsv)
  parity     signal / tensor comparison (streaming vs offline, PyTorch vs ORT)
  asr, speaker, mos, f0_ref, evaluate
             the eval_vc.py metric stack: Whisper WER/CER, ECAPA cosine/EER,
             UTMOS, offline reference F0

Later stages add features (Stage 03), streaming and models (Stages 04-06).
"""
