# PSMAF-UNIV

**Pseudo-Semantic Guided Multi-scale Adaptive Fusion for UNIV-based
RGB-Infrared Task Adaptation**

中文名称：**基于 UNIV 跨模态统一表征的伪语义引导多尺度任务适配模型**。

PSMAF-UNIV studies how the cross-modal unified representation learned by UNIV can
be converted into multi-scale task features for object detection and semantic
segmentation. UNIV is the base representation framework; the proposed contribution
is the task-adaptation path built on top of, and jointly with, that representation.

## Research documentation

* [Stage 1: original UNIV source analysis](docs/univ_source_analysis.md)
* [Stage 2: PSMAF-UNIV research goal and experimental contract](docs/research_goal.md)

The primary detector will follow the Mask R-CNN/Faster R-CNN family to keep the
first comparison close to UNIV. YOLOv8 is reserved for a later real-time extension,
not the primary formulation or baseline.
