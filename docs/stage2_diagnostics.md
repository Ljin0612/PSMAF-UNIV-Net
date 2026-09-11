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
path with a path available in the local environment, and create the report
directory before running the commands:

```bash
mkdir -p outputs/diagnostics
```

Inspect the student branch of a real checkpoint:

```bash
python tools/check_univ_checkpoint.py /path/to/checkpoint0400.pth \
  --checkpoint-key student \
  --json-out outputs/diagnostics/univ_checkpoint_student.json
```

Inspect the teacher branch of the same checkpoint:

```bash
python tools/check_univ_checkpoint.py /path/to/checkpoint0400.pth \
  --checkpoint-key teacher \
  --json-out outputs/diagnostics/univ_checkpoint_teacher.json
```

Test checkpoint loading against a constructed original UNIV model:

```bash
python tools/check_univ_checkpoint.py /path/to/checkpoint0400.pth \
  --checkpoint-key student \
  --build-model \
  --json-out outputs/diagnostics/univ_checkpoint_student_build.json
```

Inspect model construction and its candidate feature extraction points:

```bash
python tools/inspect_univ_model.py --json-out outputs/diagnostics/univ_model.json
```

Probe features at the original UNIV input resolution. The feature inspection
script prints its JSON report to standard output, so use shell redirection to
save it:

```bash
python tools/inspect_univ_features.py \
  --checkpoint /path/to/checkpoint0400.pth \
  --image-size 224 224 \
  --device cuda \
  > outputs/diagnostics/univ_features_224.json
```

The initial feature probe should use `224x224` because the original UNIV
factory may contain fixed `14x14` masks and 196-token positional embeddings.
Downstream resolutions such as 512, 640, or 1024 require explicit
resolution-adaptation support and belong to a later stage.

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
