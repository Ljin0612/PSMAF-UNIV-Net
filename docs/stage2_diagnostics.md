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
directory before running the commands.

### NumPy compatibility pre-check

The original UNIV source is known to require NumPy 1.23.x compatibility: it
uses deprecated aliases such as `np.float`, which NumPy 1.24 removed. The
recommended reproducible Stage 2 environment therefore pins
`numpy==1.23.5`. With NumPy 1.24 or newer, feature and model inspection may
fail during model construction before producing JSON unless the diagnostic
compatibility handling is enabled.

Verify the active NumPy environment before constructing the model:

```bash
python - <<'PY'
import numpy as np
print("numpy:", np.__version__)
print("has np.float:", hasattr(np, "float"))
PY
```

If necessary, install the recommended version with pip:

```bash
pip install "numpy==1.23.5"
```

or with conda:

```bash
conda install -y numpy=1.23.5
```

The diagnostic tools also apply a narrow compatibility shim for the known
legacy NumPy aliases before importing the original UNIV model. The pin remains
the recommended way to reproduce the original environment; the shim does not
hide unrelated import or model-construction errors.

### Run the diagnostics

Create the output directory after confirming NumPy compatibility:

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
