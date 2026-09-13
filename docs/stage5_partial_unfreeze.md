# Stage 5.1: partial UNIV unfreezing

## Baseline and scope

The completed Stage 4 baseline is the single-stream M3FD-IR UNIV student with
MTA and Faster R-CNN at 224 pixels. UNIV was frozen for 200 epochs (672,000
optimizer steps). It reached validation mAP50 `0.4648` and mAP50:95 `0.2503`;
test mAP50 `0.4612`, mAP50:95 `0.2575`, and AP75 `0.2663`.

Stage 5.1 tests whether limited adaptation of UNIV's latest semantic layers can
improve that baseline. It remains IR-only and does not add paired fusion,
PSG/MSAF, full PSMAF-UNIV, or YOLO. The frozen Stage 4 protocol remains the
default (`--freeze-univ true --unfreeze-last-n-blocks 0 --unfreeze-norm false`).

Only the last *N* modules in `blocks3` can be selected, plus the optional final
`norm`. Patch embedding and early convolutional blocks remain frozen. Detector
and MTA parameters use `--lr`; selected UNIV parameters are placed in a separate
optimizer group using `--univ-lr`. **Keep the UNIV learning rate substantially
smaller** because these weights are pretrained (`1e-5` is the default).

## 50-step smoke experiment

This example updates `blocks3.10` and final `norm` while leaving earlier UNIV
layers frozen:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py \
  --data-root /home/jinlei/database/M3FD_Detection \
  --checkpoint /home/jinlei/checkpoints/UNIV/checkpoint0400.pth \
  --checkpoint-key student --split train --val-split val \
  --epochs 1 --batch-size 1 --image-size 224 --max-train-steps 50 \
  --freeze-univ true --unfreeze-last-n-blocks 1 --unfreeze-norm true \
  --lr 0.005 --univ-lr 1e-5 \
  --output-dir outputs/stage5_1_smoke
```

## 20-epoch partial-unfreeze experiment

M3FD has 3,360 training samples, so a batch-size-one 20-epoch experiment is
bounded at 67,200 optimizer steps:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py \
  --data-root /home/jinlei/database/M3FD_Detection \
  --checkpoint /home/jinlei/checkpoints/UNIV/checkpoint0400.pth \
  --checkpoint-key student --split train --val-split val \
  --epochs 20 --batch-size 1 --image-size 224 --max-train-steps 67200 \
  --freeze-univ true --unfreeze-last-n-blocks 2 --unfreeze-norm true \
  --lr 0.005 --univ-lr 1e-5 \
  --output-dir outputs/stage5_1_blocks3_last2_norm_20ep
```

The training summary records the requested policy and learning rate, component
parameter counts, and exact unfrozen UNIV module names. Preserve these fields
with evaluation results so partial-unfreeze runs remain distinguishable from
the frozen Stage 4 baseline.
