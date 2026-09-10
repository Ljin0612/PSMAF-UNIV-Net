# Stage 2: UNIV verification and feature diagnostics

Stage 2 is an observation stage. It does **not** add detection training, a YOLO
branch, or a replacement encoder. The diagnostics instantiate the original
`convmae_convvit_base_patch16` factory directly from `UNIV-main/`, optionally load
the real UNIV checkpoint, and record evidence needed to design the later
Multi-scale Task Adapter.

## Reproducible workflow

All commands are run from the repository root and emit JSON so results can be
saved and compared between environments.

```bash
python tools/inspect_univ_model.py > model_inventory.json
python tools/check_univ_checkpoint.py /path/to/checkpoint.pth > checkpoint_report.json
python -m tools.check_univ_checkpoint --checkpoint /path/to/checkpoint.pth \
  --checkpoint-key student --device cpu --json-out checkpoint_report.json
python tools/check_univ_checkpoint.py /path/to/checkpoint.pth --build-model
python tools/inspect_univ_features.py \
  --checkpoint /path/to/checkpoint.pth \
  --image-size 224 224 > feature_report.json
```

Checkpoint inspection does not construct the original timm model unless
`--build-model` is supplied (PyTorch is still required to deserialize its
checkpoint). The model check uses the checked-in
`convmae_convvit_base_patch16` constructor, loads with `strict=False`, safely
resizes a compatible positional grid, and reports skipped shape mismatches. If
that constructor or one of its imports is unavailable, the report has
`model_load.status = "unavailable"` and names the missing import or constructor;
the required next step is to restore `UNIV-main/models/backbone/mcmae/models_convmae.py`
or install the dependency named in that error before rerunning the command. Use `--modules NAME
[NAME ...]` to override hook locations and `--device cuda` only after the CPU smoke
test passes. Unknown module names are errors rather than silently missing results.

## What the reports establish

The checkpoint report includes top-level branches, tensor and parameter counts,
dtypes, selected state branch, loaded fraction, missing/unexpected keys,
incompatible shapes, and positional-embedding interpolation.

The feature probe uses `mask_ratio=0` and `return_last_attention=True`, matching the
active UNIV adaptation call. It records shape, dtype/device, finite fraction, and
basic statistics at these defaults:

| Hook | Expected role at 224×224 | Adapter relevance |
|---|---|---|
| `patch_embed1`, `blocks1.1` | stride-4, 256-channel map | fine candidate after the last stage-1 block |
| `patch_embed2`, `blocks2.1` | stride-8, 384-channel map | intermediate candidate after the last stage-2 block |
| `patch_embed3`, `patch_embed4` | stride-16 map then 768-channel tokens | transformer transition |
| `blocks3.10`, `norm` | final transformer and normalized tokens | coarse semantic candidate |

The report labels the returned normalized latent as the semantic-anchor source used
by the original frozen RGB teacher and separately verifies last-block attention.
Expected terminal shapes are latent `[B, 196, 768]` and attention
`[B, 12, 196, 196]`.

## Interpretation constraints

The pretraining implementation hard-codes 14×14 mask reshaping and a 196-token
positional embedding, so the first execution check intentionally uses 224×224.
Arbitrary or rectangular sizes are negative compatibility tests, not evidence that
the original encoder supports them.

Stage-1 and stage-2 outputs are candidates, not currently returned UNIV features:
the upstream code computes projected versions but leaves their stage-3 additions
commented out. Diagnostics capture live block outputs without claiming a finished
feature pyramid. Tap selection and projection/fusion policy belong to the later
task-adapter experiment.

## Stage 2 exit criteria

Before detector implementation, archive JSON evidence that:

1. the original factory constructs;
2. the real checkpoint branch and key compatibility are understood;
3. canonical 224×224 inference produces finite latent features;
4. final self-attention has the expected head/token axes;
5. stage-1, stage-2, and normalized stage-3 shapes were measured; and
6. adapter taps were selected from measured live activations.
