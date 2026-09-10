# Stage 2 diagnostics

## 1. Purpose

Stage 2 verifies real UNIV checkpoint loading, model construction, and feature
extraction before downstream detection or segmentation.

## 2. Why this stage is necessary

- UNIV checkpoints may need `pos_embed` resizing for downstream resolutions.
- `strict=False` does not ignore same-key tensor shape mismatches.
- UNIV token outputs must preserve spatial shape.
- The Multi-scale Task Adapter requires correct token-grid metadata.
- Detection training should not start until feature extraction is verified.

## 3. Diagnostic commands

Run the diagnostics from the repository root. Replace the example checkpoint
and image paths with paths available in the local environment.

Inspect the student branch of a real checkpoint:

```bash
python tools/check_univ_checkpoint.py /path/to/checkpoint0400.pth --checkpoint-key student --json-out outputs/diagnostics/univ_checkpoint_student.json
```

Inspect the teacher branch of the same checkpoint:

```bash
python tools/check_univ_checkpoint.py /path/to/checkpoint0400.pth --checkpoint-key teacher --json-out outputs/diagnostics/univ_checkpoint_teacher.json
```

Inspect model construction and its candidate feature extraction points:

```bash
python tools/inspect_univ_model.py --json-out outputs/diagnostics/univ_model.json
```

Inspect RGB and infrared features at the intended downstream resolution:

```bash
python tools/inspect_univ_features.py \
  --checkpoint /path/to/checkpoint0400.pth \
  --checkpoint-key student \
  --rgb-image /path/to/sample_rgb.jpg \
  --ir-image /path/to/sample_ir.jpg \
  --input-size 512 \
  --device cuda \
  --json-out outputs/diagnostics/univ_features_512.json
```

## 4. Expected outputs

The diagnostic run should produce or document:

- a checkpoint report;
- a model report;
- a feature report;
- a `pos_embed` resize report;
- candidate feature extraction points; and
- limitations if the original model cannot be instantiated.

Keep the generated JSON reports under `outputs/diagnostics/` so checkpoint,
model, and feature results can be compared together before implementation work
continues.

## 5. Next stage after diagnostics

After Stage 2 passes:

1. Implement the real `UNIVEncoderWrapper`.
2. Connect UNIV outputs to `MultiScaleTaskAdapter`.
3. Verify P3/P4/P5 shape.
4. Then start the UNIV direct downstream baseline.
