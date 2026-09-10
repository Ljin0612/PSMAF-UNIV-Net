# PSMAF-UNIV design

The new package is deliberately head-neutral. `UNIVEncoderWrapper` normalizes encoder
outputs, `MultiscaleTaskAdapter` projects each level to a shared channel width,
`PseudoSemanticGuidance` estimates spatial confidence, and `PSMAFFusion` uses that
confidence when injecting coarse context into finer levels. `PSMAFUNIVModel` composes
those pieces and returns a fine-to-coarse tuple. Detection and segmentation adapters
only translate this tuple into the interface expected by their task framework.

## Feature and checkpoint interfaces

`MultiscaleTaskAdapter` accepts BCHW maps directly. BNC token features must include
their rectangular layout via `spatial_shape=(height, width)`, the equivalent
`grid_size` argument, or a feature dictionary such as
`{"tensor": tokens, "spatial_shape": (height, width)}`. Square-grid inference is
disabled by default and must be explicitly enabled with `allow_square_infer=True`.

Checkpoint loading keeps PyTorch's non-strict missing/unexpected-key behavior while
returning a detailed report. If source and destination `pos_embed` tensors use
compatible embedding widths and safely inferable spatial grids, only their spatial
tokens are bicubically resized; leading class/distillation tokens are preserved.
Unsafe positional layouts and all other incompatible tensor shapes are skipped and
listed in `skipped_shape_mismatch_keys` rather than causing a size-mismatch crash.

This is a Stage-3 structural baseline, not a claim that the proposed training recipe
or final architecture has been experimentally validated.
