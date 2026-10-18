#!/usr/bin/env python3
"""train_unit_vocoder.py — causal, stateful, multi-speaker unit vocoder trainer.

Placeholder until Milestone 2 / Stage 05
(Documents/Milestone 2/05_causal_vocoder_decision_gate.md).

Trains the Mode A decoder that maps (unit ids, target-domain log-f0, voicing,
aperiodicity, energy, speaker embedding) → 16 kHz waveform. It is
**multi-speaker**: LJSpeech plus the chosen Hi-Fi TTS readers, with a speaker
table (nn.Embedding(n_speakers, 64)) exported to voices.json. The user's own
voice becomes one more row through a Stage 10 fine-tune.

Key requirement: every convolution is *causal* (left-padded, right-trimmed
transposed convs, no lookahead) and stateful, so chunked inference at 20/40/80 ms
hops (frames_per_step 1/2/4) equals offline inference (Stage 05 DoD) without
OLA. The checkpoint stores the contract fields (frame size, vocab,
cond_inputs, spk_emb_dim; Documents/Milestone 2/02_model_contract_fixtures.md).
See StreamVoice (arXiv:2401.11053) and X-VC in RESEARCH_GUIDE.md §3.

Reference implementation: facebookresearch/speech-resynthesis (Unit-HiFi-GAN);
V2-class generator, MPD + multi-resolution STFT discriminators.
"""

import argparse
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--units-dir", required=True,
                   help="Directory of features produced by extract_units.py.")
    p.add_argument("--audio-dir", required=True,
                   help="Directory of target-speaker waveform files.")
    p.add_argument("--speaker", default="target",
                   help="Placeholder; voices become voices.json rows, not model names (Stage 05).")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--checkpoint-dir", default="checkpoints/vocoder")
    return p.parse_args(argv)


def main():
    args = parse_args()
    print(f"[train_unit_vocoder] speaker={args.speaker} epochs={args.epochs} "
          f"bs={args.batch_size} lr={args.lr}")
    # TODO(M2 Stage 05): dataset loader (units + features + waveform) → causal
    #       multi-speaker generator + MPD/MRD discriminators → LSGAN +
    #       feature-matching + multi-resolution mel losses.
    raise NotImplementedError("See Documents/Milestone 2/05_causal_vocoder_decision_gate.md.")


if __name__ == "__main__":
    sys.exit(main())
