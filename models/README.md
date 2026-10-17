# models/

Exported stateful ONNX graphs produced by `python_training/export_stateful_onnx.py`.
Large binaries — git-ignored.

| File | Stage | Arrives in |
|---|---|---|
| `encoder_mimi.onnx` | Causal semantic tokenizer (Mimi / HuBERT+k-means) | M2 |
| `translator_s2ut_waitk.onnx` | Wait-k latent S2UT translator | M3 |
| `vocoder_custom_voice.onnx` | Causal unit vocoder in the target voice | M2 |
