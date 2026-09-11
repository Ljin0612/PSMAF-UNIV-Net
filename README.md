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
* [Stage 3: source-layout decision](docs/source_layout.md)
* [Stage 3: modular PSMAF-UNIV design](docs/psmaf_univ_design.md)
* [Stage 2: runtime checkpoint and feature diagnostics](docs/stage2_diagnostics.md)

The primary detector will follow the Mask R-CNN/Faster R-CNN family to keep the
first comparison close to UNIV. YOLOv8 is reserved for a later real-time extension,
not the primary formulation or baseline.


## Repository layout

The untouched upstream snapshot remains in `UNIV-main/`; see the source-layout
decision before running it. New, importable components live in `psmaf_univ/`, while
`detection/`, `segmentation/`, and `pretraining/` separate task integrations. The
command entry points introduced in Stage 3 define interfaces only; they do not yet
launch full experiments.

Stage 2 provides executable JSON diagnostics for the original factory, checkpoint
compatibility, intermediate features, final semantic tokens, and self-attention.
Run that workflow before implementing downstream detection training.

## Environment setup

For general development, install `requirements.txt`. Its NumPy requirement is
selected by Python version so pip remains usable on Python 3.12 and newer:

```bash
python -m pip install -r requirements.txt
```

For the most faithful original UNIV and Stage 2 diagnostic environment, use
Python 3.10 with NumPy 1.23.5 via `requirements-univ.txt` or `environment.yml`:

```bash
python3.10 -m pip install -r requirements-univ.txt
# Alternatively: conda env create -f environment.yml
```

NumPy 1.24 removed legacy aliases such as `np.float`. On Python 3.12+, the
root requirements install NumPy 1.26.4 or newer, and the diagnostic tools use
the repository compatibility shim for the aliases needed by the original UNIV
source. See the Stage 2 diagnostics guide for details.
