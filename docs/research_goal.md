# Stage 2: PSMAF-UNIV research goal

## 1. Identity and research position

**English name:** PSMAF-UNIV — *Pseudo-Semantic Guided Multi-scale Adaptive
Fusion for UNIV-based RGB-Infrared Task Adaptation*

**Chinese name:** 基于 UNIV 跨模态统一表征的伪语义引导多尺度任务适配模型

PSMAF-UNIV is an extension of UNIV, not a replacement for it and not merely a
detector attached to a pretrained encoder. UNIV remains the base framework that
learns a shared representation for visible and infrared inputs. The new work asks:

> How can UNIV-style cross-modal unified representations be adapted into
> multi-scale task features for object detection and semantic segmentation?

The contribution is a task-adaptation path that makes UNIV features useful at the
multiple spatial resolutions required by dense prediction. It consists of:

1. a **Multi-scale Task Adapter** that converts the unified representation into a
   task-ready hierarchy;
2. **Pseudo-Semantic Guidance** that transfers training-time semantic structure to
   the scale-specific features without requiring extra inference inputs;
3. **Multi-scale Adaptive Fusion** that selects and combines complementary scale
   information rather than applying a fixed sum or concatenation; and
4. bindings for **downstream detection and semantic segmentation** heads.

Throughout the project, “PSMAF-UNIV” denotes the complete combination of these
elements. “UNIV + Multi-scale Task Adapter” denotes the adapter without
pseudo-semantic guidance and adaptive fusion, and is a required intermediate
baseline.

## 2. Model boundary

The intended data path is:

```text
RGB or infrared image
        │
        ▼
UNIV shared encoder
        │ unified cross-modal representation
        ▼
Multi-scale Task Adapter
        │ scale-aligned task features
        ├──── Pseudo-Semantic Guidance (training supervision/modulation)
        ▼
Multi-scale Adaptive Fusion
        │ task pyramid (for example, strides 4/8/16/32)
        ├───────────────┬─────────────────┐
        ▼               ▼                 ▼
Faster/Mask R-CNN   segmentation head   later: YOLOv8
```

The boundary between the proposed representation and any task head must be a
head-neutral pyramid of NCHW feature tensors with documented channel counts and
strides. Detection- or segmentation-specific prediction layers are validation
heads, not part of the core representation contribution.

Pseudo-semantic signals should supervise or modulate adaptation during training.
Unless an experiment explicitly studies otherwise, teacher-derived signals must
not require a second backbone at inference time. This keeps gains attributable to
the learned task representation rather than hidden inference-time teacher compute.

## 3. Component responsibilities

### 3.1 Multi-scale Task Adapter (MTA)

The MTA exposes and transforms UNIV's native hierarchy into task features. It must:

* consume shared-encoder features rather than create independent RGB and infrared
  task backbones;
* produce a stable pyramid contract usable by both detection and segmentation;
* preserve spatial detail at shallow scales and semantic context at deep scales;
* define all resizing and channel projections explicitly; and
* remain independently testable as the `UNIV + MTA` comparison.

### 3.2 Pseudo-Semantic Guidance (PSG)

PSG turns UNIV's teacher attention, patch relations, or derived confidence maps
into training targets for the task pyramid. The implementation should document:

* the source of the pseudo semantics (for example, frozen-teacher tokens or
  attention relations);
* whether the target is hard, soft, or confidence weighted;
* how a token grid is aligned to every spatial scale;
* which adapter/fusion features receive the signal; and
* the loss definition and weight, including any stop-gradient boundary.

“Pseudo-semantic” must not be used as a vague synonym for attention. Each experiment
must state precisely how the signal is constructed and where it enters the model.

### 3.3 Multi-scale Adaptive Fusion (MAF)

MAF learns how much information to take from each scale or path. Its adaptive
weights may be spatial, channel-wise, or scale-wise, but they must be observable for
analysis and compared against a non-adaptive alternative. Resampling rules must be
resolution safe, and fusion must preserve the output pyramid contract.

### 3.4 Task bindings

The first detection binding will use a **Faster R-CNN- or Mask R-CNN-style** head and
feature-pyramid interface. This is the primary detector because it is closer to the
downstream setting of the original UNIV work and makes representation comparisons
less confounded by a substantially different detector design.

Semantic segmentation is a co-primary dense-prediction validation task and should
consume the same adapted pyramid. Existing UPerNet-style infrastructure can serve as
the initial binding, provided the core PSMAF modules remain head neutral.

**YOLOv8 is not the first detector or a primary baseline.** It will be introduced
only after the core comparisons are established, under the explicit label
**PSMAF-UNIV + YOLOv8 real-time detection extension**.

## 4. Required comparison order

The main experiment must first isolate the value added at each research step:

| Order | System | Purpose |
|---:|---|---|
| 1 | Original UNIV direct adaptation | Establish the base transfer result with no proposed multi-scale adapter. |
| 2 | UNIV + Multi-scale Task Adapter | Measure the benefit of task-oriented multi-scale features alone. |
| 3 | PSMAF-UNIV | Measure the additional effect of pseudo-semantic guidance and adaptive fusion. |

All three systems must use the same UNIV initialization, dataset split, input
resolution, augmentation policy, task head, training schedule, and evaluation code
where architecturally possible. Parameter count, FLOPs, peak memory, and inference
latency should be reported alongside accuracy so that added capacity is visible.

Only after this controlled sequence should the work compare alternative heads or
introduce the YOLOv8 extension.

## 5. Ablation matrix

The minimum component ablation holds the UNIV backbone and primary task head fixed:

| Experiment | MTA | PSG | Adaptive fusion | Question answered |
|---|:---:|:---:|:---:|---|
| Direct adaptation | — | — | — | How well does original UNIV transfer directly? |
| Adapter baseline | ✓ | — | — | Does a task-specific hierarchy help? |
| Guidance ablation | ✓ | ✓ | — | Does pseudo-semantic supervision improve the hierarchy? |
| Fusion ablation | ✓ | — | ✓ | Is learned cross-scale fusion useful without PSG? |
| PSMAF-UNIV | ✓ | ✓ | ✓ | Are the proposed components complementary? |

Supporting ablations should compare hard versus soft pseudo targets, guidance at
individual versus all scales, adaptive versus fixed fusion, frozen versus fine-tuned
UNIV, and inference with versus without any training-only guidance machinery.

## 6. Evaluation contract

### Detection

Use the Faster R-CNN/Mask R-CNN family first and report standard box AP metrics; if
Mask R-CNN annotations are available, report mask AP separately. Evaluate RGB and
infrared inputs separately as well as the intended cross-modal setting so that a
single dominant modality cannot hide weak unified adaptation.

### Semantic segmentation

Report mIoU and per-class IoU, with pixel accuracy as a supporting metric. Use the
same PSMAF pyramid contract as detection; differences should be confined to the task
head and task loss.

### Representation and efficiency analysis

In addition to downstream scores, report cross-modal feature alignment and visualize
pseudo-semantic/fusion weights at each scale. Always report trainable and total
parameters, FLOPs under a declared input size, latency, and whether the teacher is
present during training or inference.

Experiments must record seeds and publish dataset splits and configuration files.
Claims about a component require repeated runs or uncertainty estimates rather than
a single favorable checkpoint.

## 7. Implementation sequence and acceptance criteria

1. **Define direct adaptation.** Add a primary Faster/Mask R-CNN-style binding to
   the unchanged UNIV downstream features and reproduce a deterministic baseline.
2. **Create the feature contract.** Expose native encoder stages and test feature
   shape, channel, stride, and checkpoint-loading behavior across supported input
   sizes.
3. **Implement MTA.** Produce the task pyramid and validate `UNIV + MTA` with the
   identical head and training recipe.
4. **Implement PSG.** Generate device-safe, resolution-safe scale targets under
   no-gradient teacher inference and test alignment and loss propagation.
5. **Implement MAF.** Add inspectable adaptive weights, a fixed-fusion control, and
   numerical/shape tests.
6. **Run PSMAF-UNIV.** Complete the ordered comparison and component ablations on
   detection and segmentation.
7. **Add the real-time extension.** Bind the established pyramid to YOLOv8 and label
   all results as the PSMAF-UNIV + YOLOv8 extension.

The stage is complete when the three primary systems can be selected from explicit
configurations, share a controlled evaluation protocol, and produce enough logged
metadata to attribute improvements to MTA, PSG, and MAF rather than to a new head or
training recipe.

## 8. Non-goals and naming guardrails

* Do not rename original UNIV pretraining behavior as a new PSMAF component.
* Do not characterize a conventional FPN alone as the complete MTA/MAF contribution.
* Do not make YOLO the default detector or use YOLO-only results to establish the
  principal claim.
* Do not compare PSMAF-UNIV only against unrelated detectors; the three ordered UNIV
  comparisons are mandatory.
* Do not require ground-truth semantic labels to construct a signal described as
  pseudo-semantic.
* Do not report a two-backbone inference cost as though it were a single-backbone
  model.

This contract keeps the scientific claim focused: PSMAF-UNIV improves the conversion
of UNIV's cross-modal unified representation into multi-scale features for dense
downstream tasks.
