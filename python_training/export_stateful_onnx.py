#!/usr/bin/env python3
"""export_stateful_onnx.py — PyTorch → stateful ONNX export.

Exports each pipeline stage as a *stateful* ONNX graph: all recurrent state
(KV-caches `past_key_values`, causal-conv left context) is exposed as explicit
graph inputs AND outputs, so the C++ engine can feed outputs back as inputs on
the next 80 ms chunk via KVStateCache (src/pipeline/KVStateCache.hpp).

Export checklist per model:
  1. Wrap the module so every recurrent state is a named tensor in the
     signature (no hidden member buffers).
  2. torch.onnx.export with matching input/output names for state pairs.
  3. Verify numerically: ONNX Runtime output vs PyTorch on the same chunks.
  4. Smoke-test state round-trip in C++ (see sherpa-onnx online model code,
     RESEARCH_GUIDE.md §4).

Milestones: M2 (encoder + vocoder), M3 (Wait-k S2UT translator).
"""

import argparse
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="PyTorch checkpoint to export.")
    p.add_argument("--model", choices=["encoder", "translator", "vocoder"], required=True)
    p.add_argument("--output", required=True, help="Destination .onnx path (models/).")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--fp16", action="store_true",
                   help="Export FP16 weights (Milestone 4 optimization).")
    return p.parse_args(argv)


def main():
    args = parse_args()
    print(f"[export_stateful_onnx] model={args.model} opset={args.opset} "
          f"fp16={args.fp16}\n  ckpt: {args.checkpoint}\n  out : {args.output}")
    # TODO: load checkpoint → wrap with explicit state tensors →
    #       torch.onnx.export → onnx.checker → ORT parity test on 80 ms chunks.
    raise NotImplementedError("See Documents/Milestones 2–3 for export plans.")


if __name__ == "__main__":
    sys.exit(main())
