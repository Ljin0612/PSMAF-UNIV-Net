# Experiment plan

## Stages

### Stage 1: Completed

* Source analysis.
* Research goal documentation.
* Skeleton modules.
* Checkpoint-loading hardening.
* Rectangular-grid handling.

### Stage 2: Current next step

* Diagnose a real checkpoint.
* Inspect model construction.
* Run a feature-extraction smoke test.
* Do **not** begin detection training yet.

### Stage 3: UNIV-to-multiscale adapter validation

* Pass real UNIV output through `MultiScaleTaskAdapter`.
* Verify P3, P4, and P5 shapes for 512 and 640 inputs, plus rectangular inputs
  when the underlying model supports them.
* Do not set an mAP target yet; this stage validates the adapter interface and
  feature shapes only.

### Stage 4: Original UNIV direct downstream baseline

* Build a single-stream UNIV baseline with a Mask R-CNN- or Faster R-CNN-style
  detection head.
* Start with M3FD-IR.
* Compare random initialization, student and teacher checkpoints, a frozen
  backbone, partial fine-tuning, and LoRA when available.

### Stage 5: PSMAF-UNIV full detection model

Build and evaluate the complete paired detection path:

`paired RGB-IR input -> UNIV-based encoder/wrapper -> Multi-scale Task Adapter ->`
`Pseudo-Semantic Guidance -> PSMAF Fusion -> Mask/Faster R-CNN-style detection head`

### Stage 6: Segmentation validation

* Validate with a UPerNet- or SegFormer-style head.
* Evaluate on MSRS, MFNet, and FMB.

### Stage 7: YOLO extension

* Treat YOLO as a real-time auxiliary branch only.
* Do not use it as the core proof of the proposed approach.

Across applicable stages, report accuracy, parameter count, throughput, memory,
per-class results, and mean/standard deviation across seeds. Store machine-readable
outputs outside Git and aggregate them with `tools/collect_results.py` once its
schema is finalized.

## Evaluation settings

Every experiment must select and label one of these settings and keep its metrics
separate from the other settings:

### A. Single-modality evaluation

* **Input:** RGB-only or IR-only.
* **Purpose:** evaluate single-stream UNIV representation transfer.
* **Aggregation:** report task metrics independently for RGB and IR datasets/splits;
  do not average the two into an unlabeled cross-modal score.

### B. Cross-modality transfer evaluation

* **Input/training protocol:** train or adapt on one modality and evaluate on the
  other, if the data and implementation support this protocol.
* **Purpose:** measure modality generalization and feature-space alignment.
* **Aggregation:** report each direction (RGB→IR and IR→RGB) separately, including
  the train/adaptation and evaluation split used. Any bidirectional summary must be
  clearly labeled and accompanied by both directional results.

### C. Paired RGB-IR task adaptation evaluation

* **Input:** paired RGB and IR images.
* **Model path:** `RGB/IR inputs -> UNIV-based encoder or wrappers -> Multi-scale
  Task Adapter -> Pseudo-Semantic Guidance -> PSMAF Fusion -> downstream head`.
* **Purpose:** evaluate the new PSMAF-UNIV paired fusion model.
* **Aggregation:** produce one downstream prediction per registered pair and compute
  the task metric once over the paired evaluation set; do not treat the modalities
  as two independent samples and average their separately computed metrics.

The original UNIV downstream source appears to be single-stream. Paired two-stream
fusion is therefore a planned PSMAF-UNIV extension, not an existing original UNIV
capability.
