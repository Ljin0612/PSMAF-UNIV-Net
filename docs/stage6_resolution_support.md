# Stage 6.5 resolution and long-training support

Stage 6.5 keeps the established single M3FD-IR stream, UNIV encoder, MTA, and
Faster R-CNN architecture. It adds training stability and experimental 1024
support only; it does not introduce PSMAF, RGB-IR fusion, PSG/MSAF, or YOLO.

## Confirmed resolution results

The confirmed frozen-UNIV test results are:

| Resolution and run | mAP50 | mAP50:95 | AP75 |
| --- | ---: | ---: | ---: |
| 320, 200 epochs | 0.5695 | 0.3314 | 0.3192 |
| 640, 20 epochs | 0.5975 | 0.3342 | 0.3276 |
| 640, 100 epochs | **0.6362** | **0.3588** | **0.3601** |

Thus, 640 is currently the strongest confirmed resolution. To continue it to
200 total epochs, including resuming optimizer and step state, run explicitly:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 640 --epochs 200 --batch-size 1 --max-train-steps 672000 --freeze-univ true --resume outputs/stage6_640_frozen_e100/last.pth --output-dir outputs/stage6_640_frozen_e200
```

Training writes `last.pth`, the backward-compatible
`stage4_smoke_checkpoint.pth`, `training_summary.json`, and
`metrics_history.jsonl`. No command is started automatically.

## Experimental 1024 protocol

At 1024, the input is `[B,3,1024,1024]`, the token grid is 64×64, and the
checkpoint positional embedding is resized from `[1,196,768]` (14×14) to
`[1,4096,768]` (64×64). MTA produces P3/P4/P5 feature maps at 128×128, 64×64,
and 32×32.

Inspect features before training:

```bash
python tools/inspect_univ_mta_features.py --checkpoint checkpoint0400.pth --image-size 1024 1024 --batch-size 1 --device cuda
```

Run a one-step smoke, then a 50-step smoke:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 1024 --epochs 1 --batch-size 1 --max-train-steps 1 --freeze-univ true --output-dir outputs/stage6_1024_smoke_1step
```

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 1024 --epochs 1 --batch-size 1 --max-train-steps 50 --freeze-univ true --output-dir outputs/stage6_1024_smoke_50step
```

Only after both smokes pass, start the five-epoch experiment explicitly:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 1024 --epochs 5 --batch-size 1 --max-train-steps 16800 --freeze-univ true --output-dir outputs/stage6_1024_frozen_e5
```

**Warning:** 1024 may be very slow or run out of memory on an 11 GB RTX 2080
Ti. Use batch size 1, reduce concurrent GPU workloads, and do not silently fall
back to a smaller resolution.

Evaluation retains all detections internally (`box_score_thresh=0.0`) and
applies `--score-threshold` only in the evaluation script, leaving AP
computation unchanged.
