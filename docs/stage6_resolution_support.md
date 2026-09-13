# Stage 6.1: 320×320 resolution support

Stage 6.1 keeps the Stage 4 architecture unchanged: a frozen, single-stream
M3FD-IR UNIV student, MTA, and Faster R-CNN. The confirmed 224 baseline after
200 epochs (672,000 optimizer steps) achieved val mAP50/mAP50:95 of
0.4648/0.2503 and test mAP50/mAP50:95/AP75 of 0.4612/0.2575/0.2663.

The 320 experiment targets better preservation of small objects, especially
motorcycles and lamps. UNIV's checkpoint positional embedding is a 14×14 grid
(`[1,196,768]`). At 320 it is reshaped to two dimensions, bicubically
interpolated to 20×20, and flattened to `[1,400,768]`. The 224 path does not
interpolate. Expected MTA maps are P3/P4/P5 = 28/14/7 at 224 and 40/20/10 at
320, each with 256 channels.

## Commands

Feature/checkpoint inspection:

```bash
python tools/inspect_univ_mta_features.py --checkpoint checkpoint0400.pth --image-size 320 320 --device cuda
```

Fifty-step forward/backward smoke:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 320 --epochs 1 --batch-size 1 --max-train-steps 50 --freeze-univ true --output-dir outputs/stage6_320_smoke
```

Frozen 20-epoch experiment (run explicitly; this is not started by setup):

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py --image-size 320 --epochs 20 --batch-size 1 --max-train-steps 67200 --freeze-univ true --output-dir outputs/stage6_320_frozen_20ep
```

Evaluation:

```bash
python detection/scripts/eval_univ_mta_fasterrcnn_m3fd.py --checkpoint outputs/stage6_320_frozen_20ep/stage4_smoke_checkpoint.pth --image-size 320 --split test --output-json outputs/stage6_320_frozen_20ep/test_metrics.json
```

320×320 consumes more accelerator memory and takes more time than 224×224;
reduce batch size if necessary. Evaluation retains an internal detector score
threshold of 0.0 and applies the requested threshold in the evaluation script.
