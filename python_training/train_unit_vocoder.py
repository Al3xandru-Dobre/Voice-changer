#!/usr/bin/env python3
"""train_unit_vocoder.py — Stage 3 tooling: causal Unit-HiFi-GAN trainer.

Trains the target-voice decoder that maps (discrete units, F0, speaker
embedding) → raw waveform, using 1–3 h of clean audio from a single speaker
prepared by extract_units.py.

Key requirement: all convolutions must be *causal* (left-padded, no future
lookahead) so the exported model survives chunked 80 ms inference without
boundary artefacts — see StreamVoice (arXiv:2401.11053) and X-VC in
RESEARCH_GUIDE.md §3.

Reference implementation: facebookresearch/speech-resynthesis (Unit-HiFi-GAN).
Milestone: M2 (vocoder needed for Mode A voice conversion).
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
                   help="Speaker tag — becomes part of the exported model name.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--checkpoint-dir", default="checkpoints/vocoder")
    return p.parse_args(argv)


def main():
    args = parse_args()
    print(f"[train_unit_vocoder] speaker={args.speaker} epochs={args.epochs} "
          f"bs={args.batch_size} lr={args.lr}")
    # TODO: dataset loader (units + F0 + waveform) → causal Unit-HiFi-GAN
    #       generator + MPD/MSD discriminators → feature-matching + mel losses.
    raise NotImplementedError("See Documents/Milestone 2 for the training plan.")


if __name__ == "__main__":
    sys.exit(main())
