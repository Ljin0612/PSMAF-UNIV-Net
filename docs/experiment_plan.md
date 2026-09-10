# Experiment plan

1. Verify original UNIV checkpoint loading and reproduce its segmentation baseline.
2. Freeze the encoder and train task adapters to establish an interface baseline.
3. Compare full fine-tuning, LoRA, multiscale adaptation, semantic guidance, and
   adaptive fusion under matched schedules and seeds.
4. Evaluate Mask R-CNN and Faster R-CNN on M3FD, then UPerNet on the selected
   segmentation benchmark.
5. Report accuracy, parameter count, throughput, memory, per-class results, and
   mean/standard deviation across seeds. Store machine-readable outputs outside Git
   and aggregate them with `tools/collect_results.py` once its schema is finalized.

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
  Task Adapter -> PSMAF Fusion -> downstream head`.
* **Purpose:** evaluate the new PSMAF-UNIV paired fusion model.
* **Aggregation:** produce one downstream prediction per registered pair and compute
  the task metric once over the paired evaluation set; do not treat the modalities
  as two independent samples and average their separately computed metrics.

The original UNIV downstream source appears to be single-stream. Paired two-stream
fusion is therefore a planned PSMAF-UNIV extension, not an existing original UNIV
capability.
