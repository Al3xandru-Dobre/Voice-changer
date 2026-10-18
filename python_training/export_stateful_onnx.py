#!/usr/bin/env python3
"""export_stateful_onnx.py — PyTorch → stateful ONNX export.

Placeholder until Milestone 2 / Stage 06 (Documents/Milestone 2/06_stateful_onnx_export.md),
which replaces this CLI with --role {encoder,vocoder} --frames-per-step {1,2,4}
--bundle models/<name>/.

Exports each model as a *stateful*, fixed-shape ONNX graph (batch 1, one hop):
every recurrent tensor (causal-conv left context, KV cache, position counter)
is an explicit state_in.X → state_out.X pair, so the C++ engine feeds outputs
back as inputs on the next hop (frames_per_step × frame_samples: 20/40/80 ms,
one graph per variant) via OrtModel / KVStateCache (Stage 07).

Export checklist per model:
  1. Wrap the module so every recurrent state is a named tensor in the
     signature (no hidden member buffers, no Python-int offsets).
  2. torch.onnx.export (dynamo=True first; record if dynamo=False was needed)
     with the contract's tensor names; onnx.shape_inference +
     onnx.checker.check_model(full_check=True).
  3. Embed metadata_props["fluxvoice.contract"] (contract v1,
     Documents/Milestone 2/02_model_contract_fixtures.md) and validate it with
     fvml.contract; the encoder/vocoder pair must pass validate_pair.
  4. Opset: the exporter's default, verified to load in the pinned ONNX
     Runtime (1.31 reads at most opset 26 / IR 13; set ir_version explicitly,
     since onnx 1.23 writes IR 14).
  5. Parity: ORT chunked vs. PyTorch offline on >= 60 s of dev-clean (longer
     than 2× any attention window), via fvml.parity.

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
    p.add_argument("--opset", type=int, default=18,
                   help="Placeholder; Stage 06 uses the exporter's default, checked against ORT.")
    p.add_argument("--fp16", action="store_true",
                   help="Export FP16 weights (Milestone 4 optimization).")
    return p.parse_args(argv)


def main():
    args = parse_args()
    print(f"[export_stateful_onnx] model={args.model} opset={args.opset} "
          f"fp16={args.fp16}\n  ckpt: {args.checkpoint}\n  out : {args.output}")
    # TODO(M2 Stage 06): load checkpoint → wrap with explicit state tensors →
    #       torch.onnx.export → onnx.checker → embed contract → ORT parity at s1/s2/s4.
    raise NotImplementedError("See Documents/Milestone 2/06_stateful_onnx_export.md.")


if __name__ == "__main__":
    sys.exit(main())
