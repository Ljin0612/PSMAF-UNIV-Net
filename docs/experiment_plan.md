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

### Stage 4: Single-stream downstream baselines

Run these baselines on the single-stream protocol, starting with M3FD-IR:

* Random initialization + Mask R-CNN- or Faster R-CNN-style detection head.
* UNIV direct + the same detection head.
* UNIV + `MultiScaleTaskAdapter` (MTA) + the same detection head. This is a
  required downstream baseline, not just an adapter shape check, and isolates
  whether MTA helps independently of Pseudo-Semantic Guidance (PSG) and
  Multi-Scale Adaptive Fusion (MSAF).

The UNIV direct and UNIV + MTA baselines must use the same data split, input
modality, detection head, and training recipe. For the UNIV baselines, compare
student versus teacher checkpoints and frozen versus partially fine-tuned
backbones; include LoRA when available. Report each variant explicitly rather
than combining results across these choices.

### Stage 5: Paired RGB-IR controlled baselines

All methods in this stage must consume the same paired RGB-IR input and use the
same data split, detection head, and training recipe. Include:

* Paired simple-add fusion.
* Paired concatenation fusion.
* Paired UNIV + MTA without PSG.
* Paired UNIV + MTA without MSAF.
* Paired UNIV + MTA + PSG only.
* Paired UNIV + MTA + MSAF only.
* Full PSMAF-UNIV, using the complete paired detection path:

  `paired RGB-IR input -> UNIV-based encoder/wrapper -> Multi-scale Task Adapter ->`
  `Pseudo-Semantic Guidance -> PSMAF Fusion -> Mask/Faster R-CNN-style detection head`

Use comparisons within this paired-input group for strict component attribution,
especially comparisons between full PSMAF-UNIV and the paired controls. Do not
claim that full paired PSMAF-UNIV directly improves over the single-stream UNIV
direct baseline; if both results are discussed, clearly state that their input
protocols differ and do not treat the comparison as component attribution.

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

## Controlled experiment protocol

Before running the first baseline for a dataset, record the train, validation,
and test split manifests; preprocessing and augmentation; checkpoint branch;
backbone freezing policy; optimizer and scheduler; training budget; stopping and
checkpoint-selection rule; evaluation code version; hardware; and random seeds.
Reuse that protocol unchanged for every comparison in the same evaluation
setting. A component comparison is valid only when the input setting and all
non-component choices match.

Select checkpoints using the validation split only and evaluate the selected
checkpoint on the test split once per seed. Do not tune hyperparameters, choose
seeds, or select checkpoints using test-set results. Use the same predeclared
seed list for every variant, report every seed, and summarize the arithmetic
mean and sample standard deviation without dropping failed or unfavorable runs.
If a run fails, retain it in the experiment record with its failure reason and
do not silently replace its seed.

Report compute measurements under a shared measurement protocol: the same
hardware, batch size, input size, precision, and warm-up/timed-iteration counts.
When those conditions cannot be held constant, label the measurements as
non-comparable instead of using them for a speed or memory claim.

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
