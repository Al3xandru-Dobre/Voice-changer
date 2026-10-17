#!/usr/bin/env python3
"""extract_units.py — Stage 1 tooling: waveform → discrete units + F0.

Converts a dataset of clean speech into the ground-truth features consumed by
the downstream training stages:

  * discrete semantic units    (Mimi semantic codebook, or causal HuBERT /
                                ContentVec layer-6/9 features + k-means)
  * F0 pitch contour           (RMVPE or SwiftF0)
  * speaker embedding e_spk    (optional, for vocoder conditioning)

Runs at 12.5 Hz (one token per 80 ms) to match the C++ streaming geometry.

References — RESEARCH_GUIDE.md §2:
  * kyutai-labs/moshi (Mimi codec)
  * facebookresearch/speech-resynthesis (units + F0 + speaker pipeline)

Milestone: M2 prerequisite (dataset preparation for the unit vocoder).
"""

import argparse
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Directory of source WAV/FLAC files (clean, dry speech).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write extracted .units/.f0/.spk features.")
    p.add_argument("--tokenizer", choices=["mimi", "hubert"], default="mimi",
                   help="Discrete-unit extractor backend (default: mimi).")
    p.add_argument("--kmeans", type=int, default=500,
                   help="Codebook size V for the HuBERT path (100/500/1000).")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--pitch", choices=["rmvpe", "swiftf0"], default="rmvpe")
    return p.parse_args(argv)


def main():
    args = parse_args()
    print(f"[extract_units] tokenizer={args.tokenizer} kmeans={args.kmeans} "
          f"pitch={args.pitch} sr={args.sample_rate}")
    print(f"  in : {args.input_dir}\n  out: {args.output_dir}")
    # TODO: resample → run tokenizer → k-means quantize (hubert path)
    #       → pitch track → persist per-utterance .npz features.
    raise NotImplementedError("See Documents/Milestone 2 for the extraction plan.")


if __name__ == "__main__":
    sys.exit(main())
