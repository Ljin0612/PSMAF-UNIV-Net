# Stage 6 resolution support (224, 320, and experimental 640)

Stage 6 keeps the established architecture and research direction unchanged: a
single M3FD-IR stream, a frozen UNIV student encoder, MTA, and Faster R-CNN. It
does not introduce paired RGB-IR fusion, PSG/MSAF, full PSMAF-UNIV, or YOLO.

## Confirmed baselines

The confirmed 224 baseline (200 epochs, 672,000 optimizer steps) achieved val
mAP50/mAP50:95 of **0.4648/0.2503** and test mAP50/mAP50:95/AP75 of
**0.4612/0.2575/0.2663**.

The current best confirmed 320 baseline (100 epochs) achieved val
mAP50/mAP50:95 of **0.5677/0.3198** and test mAP50/mAP50:95/AP75 of
**0.5453/0.3231/0.3267**.

The 320 result indicates that retaining more spatial detail helps, while small
and difficult classes such as motorcycle remain limited. Stage 6.4 therefore
adds 640 as an experimental resolution. The UNIV encoder grid grows from 14×14
at 224 and 20×20 at 320 to 40×40 at 640. Checkpoint `pos_embed` is reshaped from
its 14×14 grid, bicubically interpolated from `[1,196,768]` to
`[1,1600,768]`, and restored to its original dtype and device. The detection
adapter produces P3/P4/P5 maps of 80×80, 40×40, and 20×20 at 640.

## Experimental 640 commands

Feature and checkpoint inspection (no detector training):

```bash
python tools/inspect_univ_mta_features.py --checkpoint checkpoint0400.pth --image-size 640 640 --batch-size 1 --device cuda
```

One-step forward/backward smoke:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 640 --epochs 1 --batch-size 1 --max-train-steps 1 --freeze-univ true --output-dir outputs/stage6_640_smoke_1step
```

Fifty-step forward/backward smoke:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 640 --epochs 1 --batch-size 1 --max-train-steps 50 --freeze-univ true --output-dir outputs/stage6_640_smoke_50step
```

Only after both smokes pass, an optional short 5- or 10-epoch experiment can be
started explicitly (choose `--epochs 5` or `--epochs 10` and set the step cap to
the appropriate number for the local training split):

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 640 --epochs 5 --batch-size 1 --max-train-steps 16800 --freeze-univ true --output-dir outputs/stage6_640_frozen_e5
```

Evaluation continues to retain all model detections internally by setting the
Faster R-CNN box score threshold to 0.0; `--score-threshold` is applied by the
evaluation script:

```bash
python detection/scripts/eval_univ_mta_fasterrcnn_m3fd.py --checkpoint outputs/stage6_640_frozen_e5/stage4_smoke_checkpoint.pth --image-size 640 --split test --output-json outputs/stage6_640_frozen_e5/test_metrics.json
```

## Memory and compute warning

640×640 has substantially higher activation memory and compute cost than both
224×224 and 320×320. Use batch size 1 initially and do **not** begin long 640
training until feature inspection, the one-step smoke, and the 50-step smoke
all pass on the target hardware. Reduce concurrent GPU workloads if a smoke
encounters out-of-memory errors; do not silently fall back to a smaller image
size.
