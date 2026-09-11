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

The original UNIV source uses deprecated aliases such as `np.float`, which
NumPy 1.24 removed. The recommended reproducible UNIV diagnostic environment
is therefore **Python 3.10 + NumPy 1.23.5**. Install it with the dedicated
`requirements-univ.txt` file (or use `environment.yml`):

```bash
python3.10 -m pip install -r requirements-univ.txt
```

The root `requirements.txt` remains compatible across Python versions. On
Python 3.12 or newer it installs NumPy 1.26.4 or newer, since NumPy 1.23.5 does
not support those Python versions. The diagnostic tools apply the repository's
narrow compatibility shim for the known legacy aliases before importing the
original UNIV model, allowing diagnostics and development with newer NumPy.
The Python 3.10 + NumPy 1.23.5 environment is still the most faithful choice
for original UNIV diagnostics and reproducibility.

Verify the active NumPy environment before constructing the model:

```bash
python - <<'PY'
import numpy as np
print("numpy:", np.__version__)
print("has np.float:", hasattr(np, "float"))
PY
```

To create the recommended environment with conda instead, use the repository
environment specification:

```bash
conda env create -f environment.yml
```

The compatibility shim does not hide unrelated import or model-construction
errors.

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

### UNIV to Multi-scale Task Adapter smoke test

After checkpoint and feature diagnostics pass, validate the first-version
feature policy: `blocks2.1` supplies the stride-8 spatial map (P3), while
`norm`/`output.latent` supplies stride-16 semantic tokens (P4). The tool passes
the explicit `grid_size=(14, 14)` metadata required to reshape those BNC tokens
and creates P5 by downsampling P4. `blocks1.1` is reported only as an optional
diagnostic and is not an adapter input.

```bash
python tools/inspect_univ_mta_features.py \
  --checkpoint /home/jinlei/checkpoints/UNIV/checkpoint0400.pth \
  --checkpoint-key student \
  --image-size 224 224 \
  --device cuda \
  > outputs/diagnostics/univ_mta_features_224_student.json
```

With the default `--adapter-out-channels 256`, the expected P3, P4, and P5
shapes are `[1, 256, 28, 28]`, `[1, 256, 14, 14]`, and
`[1, 256, 7, 7]`, respectively. The JSON report also records the checkpoint
load fraction, selected feature shapes and grid metadata, finite fractions,
warnings, and optional `blocks1.1` diagnostics. This is shape validation only;
it does not construct or train a downstream detector.

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

### Recorded real-server result

Stage 2 has passed on the real server for both checkpoint branches. Student and
teacher each loaded with `load_fraction=1.0`; original UNIV construction and the
224×224 feature probe succeeded; and both UNIV-to-MTA smoke tests passed
checkpoint validation. All selected features and adapter outputs were finite.
The observed outputs were P3 `[1,256,28,28]`, P4 `[1,256,14,14]`, and P5
`[1,256,7,7]`.

These diagnostics establish compatibility only; they are not detection training
or accuracy results.

After Stage 2 passes:

1. Implement the real `UNIVEncoderWrapper`.
2. Connect UNIV outputs to `MultiScaleTaskAdapter`.
3. Verify P3/P4/P5 shape.
4. Then start the UNIV direct downstream baseline.
