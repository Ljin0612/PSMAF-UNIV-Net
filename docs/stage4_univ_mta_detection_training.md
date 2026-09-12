# Stage 4: UNIV + MTA detection training smoke test

## Purpose and scope

Stage 4 verifies that the original UNIV encoder and `MultiScaleTaskAdapter` are a
usable **real detection backbone**. The bounded smoke run covers the M3FD loader,
YOLO target parsing, Faster R-CNN training losses, backward propagation, one or
more optimizer steps, checkpoint output, and a lightweight validation forward.

This remains single-stream and IR-first. It deliberately does **not** implement
RGB/IR pairing, PSMAF fusion, PSG/MSAF, YOLO, or full COCO mAP. Establishing a
small, debuggable detection baseline first prevents future fusion work from
masking errors in data parsing, the UNIV feature boundary, or detector training.
Full PSMAF-UNIV begins only after this isolated boundary is reliable.

## Smoke command

The defaults are one epoch, batch size one, at most five optimizer steps, a
224-pixel input, frozen UNIV parameters, and the smoke splits:

```bash
python detection/scripts/train_univ_mta_fasterrcnn_m3fd.py \
  --data-root /home/jinlei/database/M3FD_Detection \
  --checkpoint /home/jinlei/checkpoints/UNIV/checkpoint0400.pth \
  --checkpoint-key student \
  --split smoke_train --val-split smoke_val \
  --epochs 1 --batch-size 1 --image-size 224 \
  --max-train-steps 5 --freeze-univ true \
  --output-dir outputs/stage4_smoke
```

Use `--checkpoint-key teacher` to exercise the teacher branch. Set
`--freeze-univ false` only when intentionally checking gradients through UNIV.
The script never launches a long run unless its bounded defaults are explicitly
changed.

## Expected outputs

`stage4_smoke_checkpoint.pth` contains model and optimizer state plus the number
of completed steps. `training_summary.json` records dataset counts, checkpoint
load diagnostics, final Faster R-CNN losses, and the validation scaffold:

- number of detections;
- prediction box tensor shape;
- score tensor shape; and
- whether every score is finite.

M3FD class IDs remain `0..5` in the dataset target. The training boundary shifts
them to `1..6` because torchvision reserves detector label zero for background.

## Interpreting failures

- A split/image/label error means the server tree or split stems do not match
  `meta/{split}.txt`, `ir/{stem}.png`, and `labels/{stem}.txt`.
- A class or coordinate error identifies malformed YOLO input rather than
  silently clipping it.
- A checkpoint load-fraction failure means the requested student/teacher branch
  is incompatible and the run refuses random UNIV weights.
- A missing/non-finite loss points to the Faster R-CNN training boundary.
- A backward or optimizer exception means the trainable graph is not viable.
- A non-finite validation score means training completed but inference is not
  numerically sound. Full accuracy and mAP assessment are deferred to Stage 5.
