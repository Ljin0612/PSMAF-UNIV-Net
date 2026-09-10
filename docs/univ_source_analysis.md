# Original UNIV source analysis

## Scope and terminology

This report is a code-level audit of the source currently preserved under
`UNIV-main/`. It intentionally describes what the repository **implements**, rather
than assuming that every concept or symbol used in the paper has a separately named
Python class. In particular, there is no class named `UNIV`, `E_train`, `E_frozen`,
`PCCL`, `semantic_anchor`, or `M^A` in this snapshot. The mappings below are therefore
explicitly identified from the computation performed by the code.

The central implementation is a teacher/student adaptation of the ConvMAE ConvViT
encoder. The frozen RGB teacher supplies both visible patch representations and its
last-block self-attention. The trainable student sees both modalities and is aligned
to the teacher's RGB representation geometry. This is the UNIV cross-modal unified
representation mechanism in the supplied source; it is not a YOLO plug-in and there
is no YOLO integration in this repository.

### Audit method and source boundary

The findings were produced by tracing the checked-in Python, YAML, shell launchers,
and Markdown documentation. No conclusions depend on the downloadable MVIP data or
external checkpoint linked from the README, because neither artifact is committed.
Consequently, tensor shapes below are derived from the configured 224-pixel
pretraining and 512-pixel segmentation resolutions, while checkpoint mismatch lists
are structural expectations rather than output captured from the published binary.

For repeatability, these are the most useful code symbols at each step of the trace:

| Trace step | Symbol or configuration field |
|---|---|
| Paired sample discovery | `RGB_pair_IR_dataset.__init__` |
| Synchronized pair transforms | `CoDataAugmentation`, `CoTransCompose` |
| Student/teacher construction | `pretrain_mcmae.train` |
| Per-batch modality flow | `pretrain_mcmae.train_one_epoch` |
| Encoder/token/attention return | `MaskedAutoencoderConvViT.forward_encoder` and `.forward` |
| Attention pseudo-labels and PCCL | `attention_simi_guided_loss.threshold_attention_map` and `.__call__` |
| Pretraining serialization | the `save_dict` in `pretrain_mcmae.train` |
| Segmentation feature extraction | `ConvMAE.forward_features` |
| UNIV checkpoint branch selection | `mmcv_custom.checkpoint.load_checkpoint` |
| Absolute position interpolation | the `pos_embed` branch in `mmcv_custom.checkpoint.load_checkpoint` |

After excluding generated analysis documents and focusing on implementation and
configuration files, the repository-wide inventory found no ready-to-run Mask R-CNN
or Faster R-CNN detection pipeline in the uploaded UNIV source snapshot. This is a
claim about the absence of detector implementations and configurations, not the
literal absence of those terms in documentation. The detection discussion below
therefore does not infer a detector from ConvMAE's upstream ecosystem.

## 1. Source tree and responsibilities

```text
UNIV-main/
├── README.md                         # setup, checkpoint links, entry-point overview
├── run.sh                            # distributed pretraining launcher
├── pretrain_mcmae.py                 # complete UNIV pretraining loop
├── configs/mcmae.yaml                # pretraining data/model/loss/hyperparameters
├── datasets/
│   ├── datasets.py                   # paired visible/infrared Dataset
│   ├── co_data_augmentation.py       # paired transform recipe
│   └── transforms/co_transform.py    # synchronized crop/flip/resize/normalization
├── loss/
│   ├── cross_modality_loss.py        # attention-guided patch correspondence loss
│   └── RGB_distillation_loss.py      # alternative RGB relational distillation
├── models/
│   ├── backbone/mcmae/
│   │   ├── models_convmae.py         # pretraining ConvMAE encoder/unused MAE decoder
│   │   ├── vision_transformer.py     # patch embed, conv blocks, MHSA, attention return
│   │   └── pos_embed.py              # fixed 2-D sin/cos embeddings + interpolation helper
│   ├── backbone/vits/                # DINO-style ViT and adaptor; not used by current entry point
│   └── projector/projector.py        # legacy/pass-through wrappers; not used in pretraining
├── utils/
│   ├── utils.py                      # distributed, schedules, optimization, generic loading
│   └── load_pretrained_model.py      # generic backbone/head extraction helper
└── SEG/MCMAE_SEG/
    ├── backbone/convmae.py           # MMSeg ConvMAE feature-pyramid backbone
    ├── configs/convmae/upernet_msrs.py
    ├── configs/_base_/models/upernet.py
    ├── configs/_base_/datasets/{msrs,msrs_rgb}.py
    ├── tools/{train,test,flops}.py    # MMSeg entry points
    ├── run_{train,test}.sh            # launch examples
    └── mmcv_custom/                   # checkpoint, optimizer decay, AMP runner helpers
```

### Principal files

| Concern | Authoritative source | Notes |
|---|---|---|
| Pretraining entry point | `pretrain_mcmae.py` | Builds data, student/teacher, losses, optimizer, schedules, DDP, and checkpoints. |
| Pretraining configuration | `configs/mcmae.yaml` | ConvMAE initialization checkpoint, optional LoRA, MVIP roots, PCCL-related threshold/temperature. |
| UNIV/MCMAE model | `models/backbone/mcmae/models_convmae.py` | `MaskedAutoencoderConvViT`; public factory is `convmae_convvit_base_patch16`. |
| Transformer internals | `models/backbone/mcmae/vision_transformer.py` | The final `Block` can return its `[B,H,N,N]` attention tensor. |
| PCCL implementation | `loss/cross_modality_loss.py` | Named `attention_simi_guided_loss` in code. |
| RGB consistency loss | `loss/RGB_distillation_loss.py` | Matches teacher/student RGB patch-to-patch similarity matrices. |
| Paired data | `datasets/datasets.py` | Matches sorted files from `vi_images/training` and `images/training`. |
| Paired augmentation | `datasets/co_data_augmentation.py`, `datasets/transforms/co_transform.py` | Applies identical crop and flip geometry to the pair, then modality-specific normalization. |
| Downstream semantic segmentation | `SEG/MCMAE_SEG/tools/train.py`, `tools/test.py`, and `configs/convmae/upernet_msrs.py` | MMSeg EncoderDecoder with UPerNet and an auxiliary FCN head. |

No object detection implementation, Mask R-CNN, YOLO, or SegFormer configuration is
present. The only supplied downstream task is semantic segmentation. The text in
`SEGMENTATION.md` retains upstream ConvMAE/ADE20K wording, while the checked-in active
configuration and launcher target the nine-class MSRS dataset.

## 2. Original model architecture

### 2.1 Mapping the paper-level components to code

| Concept | Concrete implementation |
|---|---|
| Trainable encoder, \(E_{train}\) | `student`, an instance of `MaskedAutoencoderConvViT`. Unless LoRA is enabled all student parameters remain trainable. With the checked-in configuration, the base parameters are frozen and PEFT LoRA parameters are trainable. |
| Frozen encoder, \(E_{frozen}\) | `teacher`, the same ConvMAE architecture initialized from the same upstream checkpoint; every teacher parameter receives `requires_grad=False`. It is evaluated only on RGB images. |
| Visible semantic anchors | `rgb_teacher_embed` from the frozen teacher, shape `[B,196,768]` for 224×224 inputs. There is no separate pooling, prototype bank, anchor module, or projector in the active path. Each teacher patch token acts as an anchor. |
| Attention pseudo labels, \(M^A\) | The head-mean teacher attention followed by cumulative-mass thresholding in `threshold_attention_map`. |
| Cross-modal similarity, \(M^{IA}\) | Batched cosine similarity between normalized student IR tokens and teacher visible anchors, divided by temperature. |
| Visible similarity, \(M^{VA}\) | In the configured attention-guided RGB branch, the same computation using student RGB tokens and teacher anchors. The alternative `RGB_patch_simi_loss` instead compares teacher-teacher and student-student RGB Gram matrices. |
| PCCL | Binary-cross-entropy alignment between a student/anchor similarity matrix and \(M^A\), implemented by `attention_simi_guided_loss`; its default transpose term duplicates the same mean-reduced elementwise objective. |

The variable name `teacher` denotes the fixed visible reference and `student` denotes
the adaptable unified encoder. This is not an EMA teacher: no momentum update or
student-to-teacher copy occurs after initialization.

### 2.2 Patch and token generation

The pretraining factory instantiates a three-stage ConvViT base model with:

* stage 1: a 4×4 stride-4 convolutional patch embedding, producing
  `[B,256,56,56]`, followed by two convolutional `CBlock`s;
* stage 2: a 2×2 stride-2 embedding, producing `[B,384,28,28]`, followed by two
  `CBlock`s;
* stage 3: a 2×2 stride-2 embedding, producing `[B,768,14,14]`, flattened to 196
  patch tokens, projected by `patch_embed4`, and augmented with a fixed 2-D sin/cos
  positional embedding;
* eleven standard transformer blocks with 12 attention heads, followed by LayerNorm.

There is no class token. At the 224×224 pretraining resolution, each final token has
an effective 16×16 input stride. Although `random_masking` and MAE decoder code remain
from ConvMAE, the training loop always passes `mask_ratio=0`. The encoder currently
does not gather the stage-3 tokens by `ids_keep`; that line is commented out. The
stage-1 and stage-2 projected token tensors are computed/gathered but their residual
addition is also commented out. Consequently the active returned representation is
the full normalized stage-3 token sequence.

The model still declares `decoder_embed`, `mask_token`, decoder positional embedding,
eight decoder blocks, decoder norm, and pixel predictor. However, `forward()` returns
the encoder latent and optional attention directly; decoder and reconstruction-loss
calls are commented out. They do not participate in UNIV optimization.

### 2.3 Frozen attention and pseudo-label matrix

On RGB input, the teacher calls its final transformer block with
`return_attention=True`. `Attention.forward` computes softmax-normalized multi-head
self-attention, and `Block.forward` returns both updated tokens and attention. Thus
`teacher_attention_map` has shape `[B,12,196,196]`.

The loss averages over heads to obtain `[B,196,196]`. For every query-patch row it:

1. sorts attention weights descending;
2. computes their cumulative sum;
3. marks entries whose cumulative mass is at most the configured threshold (0.6);
4. scatters those binary decisions back to the original key-patch order.

This binary relation is the code-equivalent \(M^A\). A subtle boundary property is
that the first attention element that makes cumulative mass exceed 0.6 is excluded.
The in-place propagation across sorted positions is redundant for ordinary
non-negative softmax attention but ensures a prefix-shaped selection.

### 2.4 Semantic anchors and PCCL alignment

`rgb_teacher_embed` is L2-normalized per patch and serves directly as the set of
semantic anchors \(A\). Student RGB or IR embeddings are normalized the same way.
For the infrared branch the logits are:

```text
M^IA = normalize(E_train(IR)) @ normalize(E_frozen(RGB))^T / temperature
```

For the configured RGB branch they are:

```text
M^VA = normalize(E_train(RGB)) @ normalize(E_frozen(RGB))^T / temperature
```

When `ir_info=None`, the code applies `BCEWithLogitsLoss` once to the logits and
labels as-is and once to both tensors transposed, then averages the two results.
Under the default mean reduction these calls contain exactly the same elementwise
terms and are mathematically identical. The transpose computation therefore does
not add a real symmetric patch-correspondence constraint and should not be
interpreted as a separate bidirectional alignment objective. If PSMAF-UNIV later
needs true symmetric RGB↔IR alignment, it should explicitly define separate
directional similarity maps or losses. Total training loss is
`ir_alpha * L_IR + rgb_beta * L_RGB` (both coefficients are 1 by default).

An optional `ir_info` map can reduce selected pseudo-label rows by 0.7 before clamping
to zero. It depends on `utils.IR_info_richness.gray_value_rank`, but that module is
absent from this repository snapshot. The default configuration disables the feature,
although the unconditional top-level import still prevents a clean import unless the
missing file is supplied.

### 2.5 Student, teacher, and LoRA details

Both networks load `checkpoint['model']` from the configured upstream ConvMAE file
using `strict=False`. The teacher is frozen, but the code does not explicitly call
`teacher.eval()`: DDP/module training mode therefore remains active. The architecture
has no BatchNorm in the pretraining path, and no configured dropout, so this is mostly
a lifecycle caveat rather than evidence of a second training path.

When `model.use_lora` is true, all base student parameters are first frozen and PEFT
wraps configured modules (`decoder_embed`, `fc1`, `qkv`, `fc2`, `proj`,
`decoder_pred`, and `patch_embed4`) at rank 8, alpha 32, dropout 0.1. Some targets are
in the inactive decoder, so adapters there receive no training signal. Before each
checkpoint save, a deep copy is merged with `merge_and_unload()`. Saved `student`
weights are therefore ordinary merged model weights rather than a standalone PEFT
adapter checkpoint. The teacher has no LoRA wrapper.

`models/projector/projector.py` is not imported by the training entry point, and its
`PatchProjector.forward` is currently an identity. It should not be interpreted as an
active semantic-anchor head.

## 3. Original pretraining data flow

### 3.1 Paired input and augmentation

For each configured MVIP root, the dataset independently sorts filenames under:

* RGB: `vi_images/training/`
* infrared: `images/training/`

It concatenates all roots, zips the two lists, checks corresponding basenames, and
opens both files as three-channel RGB PIL images (so a grayscale IR file becomes
three repeated/converted channels). There is no explicit robustness check for extra
files beyond the final equal-length assertion.

The paired augmentation samples one random 40%–100% crop and one horizontal-flip
decision and applies the exact same geometry to RGB and IR. Both are resized to
224×224 and converted to tensors. RGB uses ImageNet mean/std; IR uses its own repeated
channel mean/std. This preserves patch correspondence across modalities.

### 3.2 One training iteration

```text
(paired RGB, IR)
       │
       ├── teacher(RGB, mask=0, return_attention=True)
       │      ├── visible anchors A: [B,196,768]
       │      └── last attention:   [B,12,196,196] ──threshold──> M^A
       │
       └── student(concat(RGB, IR), mask=0)
              └── [2B,196,768] ──batch split──> V_student, I_student

M^IA = sim(I_student, A) / T ──BCE with M^A──> L_IR
M^VA = sim(V_student, A) / T ──BCE with M^A──> L_RGB
                                              total = α L_IR + β L_RGB
```

There is **no random modality selection** in the supplied loop. RGB and IR are
concatenated along the batch dimension and processed together by the same student on
every iteration. The only relevant randomness is synchronized pair augmentation,
distributed sampling, and the masking helper (which still draws noise even at
`mask_ratio=0`, without changing the active token sequence).

Mixed precision is optional. Optimization uses AdamW by default, cosine learning-rate
and weight-decay schedules, gradient clipping, and DDP. Despite the inherited function
name `cancel_gradients_last_layer`, the model has no DINO-style `last_layer` parameter
in this active architecture, so the helper does not freeze a substantive output head.

### 3.3 Implementation cautions found during tracing

These are analysis findings, not changes made in this stage:

* `pretrain_mcmae.py` imports `utils.IR_info_richness`, which is missing.
* `RGB_patch_simi_loss.__call__` accepts two embeddings, but the training loop always
  passes four arguments. The default YAML selects `attention_simi_guided_loss`, so the
  configured path avoids this mismatch.
* The thresholded pseudo-label tensor is forced to the default CUDA device via
  `.cuda()` rather than retaining the input device; this can be wrong under local-rank
  DDP.
* Teacher forward is not enclosed in `torch.no_grad()`. Parameters are frozen, but an
  unnecessary teacher autograd graph may still be constructed.
* The pretraining architecture is size-specialized: masking reshapes are hard-coded
  to 14×14, 56×56, and 28×28.
* `MaskedAutoencoderConvViT.unpatchify()` references `self.patch_embed`, which this
  three-stage class does not define. It is inactive in the current forward path.

## 4. Original downstream flow

### 4.1 Transfer mechanism

The segmentation configuration constructs an MMSeg `EncoderDecoder` whose backbone
is a separately maintained `SEG/MCMAE_SEG/backbone/convmae.py::ConvMAE`. The launch
script supplies the UNIV checkpoint through `model.pretrained`. Backbone
`init_weights()` invokes the custom checkpoint loader with `strict=False`, and then
the entire segmentation model is fine-tuned using layer-wise learning-rate decay.
The code does not freeze the transferred backbone.

The downstream backbone reproduces the three ConvViT stages but exposes spatial
features rather than the pretraining model's token tuple. For a 512×512 input its
`forward_features()` returns:

| Tuple index | Source | Expected shape | Effective stride |
|---:|---|---|---:|
| 0 | stage 1 after two convolutional blocks | `[B,256,128,128]` | 4 |
| 1 | stage 2 after two convolutional blocks | `[B,384,64,64]` | 8 |
| 2 | final transformer tokens reshaped spatially | `[B,768,32,32]` | 16 |
| 3 | 3×3 stride-2 convolutional `fpn` | `[B,768,15,15]` | approximately 32 |

The last size is 15×15 rather than 16×16 because that convolution has no padding.
The constructor accepts `out_indices`, but `forward_features()` does not use it; all
four tensors are always returned.

The active UPerHead configuration expects the feature tuple in NCHW format with
channels `[256,384,768,768]` at indices `[0,1,2,3]`. It applies pyramid pooling at
scales `(1,2,3,6)` and produces nine MSRS logits. The auxiliary FCN head consumes
tuple index 2 (768 channels). Thus downstream transfer is explicitly multi-scale and
spatial, while pretraining returns only the final 196-token representation.

### 4.2 Tasks present and absent

The repository supplies semantic segmentation only: UPerNet decode head plus FCN
auxiliary head on MSRS. In the checked-in snapshot, `msrs.py` and `msrs_rgb.py`
appear to use the same data root, image directories, pipelines, and repeated-channel
normalization; the snapshot therefore provides no clear RGB-versus-IR normalization
difference between these configs. A downstream RGB/IR segmentation experiment must
verify the image directories, modality source, and normalization explicitly, and
must not assume that selecting `msrs_rgb.py` changes modality preprocessing. No
paired two-stream downstream loader is present—the active pipeline loads one `img`
per sample. There is no detection entry point, bounding-box dataset, Mask R-CNN,
YOLO head, or SegFormer head. Those should be treated as future validation adapters
rather than inferred parts of original UNIV.

## 5. Checkpoint formats and loading behavior

### 5.1 Upstream initialization and UNIV pretraining output

At pretraining startup, both student and teacher require an upstream ConvMAE
checkpoint with a top-level `model` key. It is loaded directly with `strict=False`;
there is no prefix cleanup or positional interpolation in this path.

Every UNIV epoch overwrites `checkpoint.pth`; configured frequency snapshots are also
written. The saved top-level keys are:

```text
student
teacher
optimizer
epoch
args
cross_modality_loss_fn
fp16_scaler                 # only when FP16 is enabled
```

`student` contains merged ordinary weights if LoRA training is enabled. `teacher` is
saved from the DDP wrapper when SyncBatchNorm handling takes that branch, otherwise
from the plain teacher; prefix behavior can consequently differ by architecture.
There is no resume call in `pretrain_mcmae.py`, even though generic resume utilities
exist elsewhere.

### 5.2 Segmentation preload selection

The custom MMSeg loader accepts top-level weights in this precedence order:
`state_dict`, `model`, `module`, `student`, then the whole dictionary. Therefore a
native UNIV checkpoint selects **student**, not teacher. If the first weight name
starts with `module.`, that prefix is stripped. It can also unwrap an `encoder.`
online-branch format. It does not explicitly recognize PEFT adapter-only files, but
native UNIV LoRA saves are already merged and therefore compatible in principle.

The loader does not explicitly delete decoder/reconstruction keys. Instead, because
the MMSeg backbone has no `decoder_*` or `mask_token` members, those source keys are
reported as unexpected and tolerated by `strict=False`. Conversely, downstream-only
parameters—most notably the spatial `fpn` convolution and normalization—and any
architecture-dependent relative-position parameters absent from pretraining are
reported missing and remain initialized by the downstream model.

Typical categories (the exact list depends on checkpoint provenance) are:

* **unexpected:** `mask_token`, `decoder_embed.*`, `decoder_pos_embed`,
  `decoder_blocks.*`, `decoder_norm.*`, `decoder_pred.*`, and pretraining-only
  `stage1_output_decode.*` / `stage2_output_decode.*`;
* **missing:** downstream `fpn.*`, downstream relative-position-bias tables when
  enabled but absent in the source, and possibly downstream-specific positional
  parameters;
* **shared and loadable:** `patch_embed1/2/3`, `patch_embed4`, `blocks1/2/3`, and
  compatible `pos_embed` weights.

This list is derived structurally because no checkpoint binary is committed, so an
exact runtime `IncompatibleKeys` listing cannot be reproduced from this repository
alone.

### 5.3 Position embedding compatibility

The segmentation loader bicubically interpolates `pos_embed` when the source and
destination patch grids differ (for example 14×14 pretraining to 32×32 segmentation).
It also contains interpolation logic for relative-position bias tables, removes
`relative_position_index` buffers, and handles Swin-style `absolute_pos_embed`.

There is a separate `models/backbone/mcmae/pos_embed.py::interpolate_pos_embed`
helper, but `pretrain_mcmae.py` does not call it. Position interpolation is therefore
available in the downstream custom loader, not in the direct upstream initialization
of student/teacher.

All relevant transfer loads use `strict=False`, so missing and unexpected keys are
diagnostic rather than fatal. The custom MMSeg loader logs mismatches. The generic
`utils/load_pretrained_model.py` can select an arbitrary named top-level branch and
extract `module.backbone`/`backbone` or head prefixes, but it is not used by either
the active pretraining or segmentation entry point.

## 6. Recommended extension points for PSMAF-UNIV

The following locations preserve the research position that PSMAF extends UNIV's
representation flow rather than treating UNIV as a frozen plug-in for a detector.

1. **Expose the native hierarchy before defining adapters.** Refactor the pretraining
   encoder to optionally return stage-1, stage-2, and stage-3 spatial features plus
   final tokens/attention. The calculations already exist in
   `forward_encoder`; today the earlier projected tensors are discarded. A stable
   feature contract could mirror the downstream `(C4,C8,C16,C32)` tuple without
   duplicating the backbone implementation.
2. **Insert pseudo-semantic guidance inside representation formation.** Candidate
   insertion points are after `blocks1`, after `blocks2`, and within/after `blocks3`.
   Teacher-derived \(M^A\) or softened attention can guide scale-specific adaptation
   there, before features reach any task neck or head.
3. **Build a PSMAF task pyramid on unified features.** Adapt RGB/IR-shared stage
   features into a documented, head-neutral list of NCHW tensors with explicit
   channels and strides. Keep UPerNet, Mask R-CNN, YOLO, and SegFormer bindings as
   thin downstream adapters consuming that contract.
4. **Retain the teacher only as training supervision.** The current frozen teacher is
   valuable for generating pseudo-semantic patch relations. It need not be shipped as
   an inference backbone; the learned student/PSMAF path should own inference.
5. **Generalize patch relation guidance across scales.** Convert the 14×14 relation
   or semantic confidence into scale-aligned maps carefully (preserving pair
   correspondence), then use it for gating, modulation, or cross-scale consistency.
   Avoid reducing the proposal to simply attaching an FPN after a frozen UNIV model.
6. **Separate base, adaptation, and validation checkpoints.** Define a canonical
   loader that accepts native `student` checkpoints, merged LoRA weights, and plain
   state dicts; filters decoder keys intentionally; interpolates positions; and emits
   a structured compatibility report. Save PSMAF module state and architecture
   metadata explicitly.
7. **Unify the duplicated backbone implementations.** Pretraining and MMSeg currently
   maintain different `ConvMAE` classes and different outputs. A shared encoder plus
   small pretraining/downstream wrappers will prevent silent architectural drift as
   PSMAF modules are inserted.
8. **Make supervision device- and resolution-safe before experiments.** Remove
   hard-coded CUDA/grid assumptions, place teacher inference under `no_grad`, define
   teacher evaluation behavior, and test pseudo-label and similarity shapes at each
   supported resolution.

### Proposed head-neutral boundary

A suitable future contract is conceptually:

```python
UnifiedRepresentation(
    tokens=...,                 # [B, N, C], UNIV-aligned final tokens
    attention=...,              # optional teacher/student relation metadata in training
    pyramid=(P4, P8, P16, P32), # PSMAF-adapted NCHW task features
    strides=(4, 8, 16, 32),
)
```

This would make the novel path “paired modalities → UNIV unified representation →
pseudo-semantic multi-scale adaptation” explicit. A validation head would consume
only `pyramid`, while PCCL/pseudo-semantic supervision remains attached to the unified
representation and adaptation internals.

## 7. Concise conclusions

* Original UNIV is implemented as a trainable shared RGB/IR ConvMAE student aligned
  to a frozen RGB ConvMAE teacher—not as a downstream detector plug-in.
* Frozen-teacher patch tokens are the effective semantic anchors; head-averaged,
  thresholded teacher self-attention is the pseudo-label relation matrix.
* PCCL applies BCE alignment to temperature-scaled patch/anchor cosine similarities
  for IR and, by default, RGB; its transposed mean-reduced term is equivalent to the
  untransposed term rather than a separate symmetric objective.
* Both modalities are always used per iteration; there is no random modality choice.
* The only supplied validation task is nine-class MSRS semantic segmentation via
  UPerNet/FCN, consuming four NCHW feature maps.
* Downstream loading prefers the saved student branch, uses `strict=False`, and
  interpolates positional embeddings; decoder keys are tolerated as unexpected, not
  deliberately filtered.
* PSMAF-UNIV should extend the shared encoder's multi-stage representation and its
  pseudo-semantic supervision, then expose head-neutral multi-scale task features.

## 8. Known limitations of Stage 1 analysis

* This report is a source-snapshot analysis.
* Some claims are based on the checked-in implementation and configuration files,
  not the full original experimental environment.
* A detection-head implementation may be absent from this snapshot even though the
  original UNIV paper reports Mask R-CNN downstream results.
* Paired RGB-IR downstream fusion is a planned PSMAF-UNIV extension, not a capability
  demonstrated by the original downstream source in this snapshot.
