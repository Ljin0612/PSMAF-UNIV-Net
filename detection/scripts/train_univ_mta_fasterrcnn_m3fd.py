#!/usr/bin/env python3
"""Run a short M3FD-IR UNIV/MTA Faster R-CNN training smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from psmaf_univ.m3fd_detection import M3FDDetectionDataset, detection_collate_fn
from psmaf_univ.univ_mta_detection_backbone import UNIVMTADetectionBackbone

EXPECTED_LOSSES = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
UNIV_IR_IMAGE_MEAN = (0.5338, 0.5338, 0.5338)
UNIV_IR_IMAGE_STD = (0.2519, 0.2519, 0.2519)
SUPPORTED_IMAGE_SIZES = (224, 320, 640)


def parse_image_size(value: str) -> int:
    """Parse one of the resolutions covered by the Stage 6 protocol."""
    image_size = int(value)
    if image_size not in SUPPORTED_IMAGE_SIZES:
        raise argparse.ArgumentTypeError("supported image sizes are 224, 320, and 640")
    return image_size


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def validate_detection_losses(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Validate the Faster R-CNN loss contract and return its finite total."""
    missing = [name for name in EXPECTED_LOSSES if name not in losses]
    if missing:
        raise RuntimeError(f"detection training forward omitted losses: {', '.join(missing)}")
    invalid = [name for name, value in losses.items() if value.ndim != 0 or not torch.isfinite(value)]
    if invalid:
        raise RuntimeError(f"detection losses must be finite scalar tensors: {', '.join(invalid)}")
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError("total detection loss is not finite")
    return total


def configure_univ_trainability(
    backbone: UNIVMTADetectionBackbone,
    freeze_univ: bool,
    unfreeze_last_n_blocks: int = 0,
    unfreeze_norm: bool = False,
) -> list[str]:
    """Configure late-layer UNIV fine-tuning and return unfrozen module names."""
    if unfreeze_last_n_blocks < 0:
        raise ValueError("unfreeze_last_n_blocks must be non-negative")
    encoder = backbone.encoder
    if not freeze_univ:
        encoder.requires_grad_(True)
        backbone.univ_grad_enabled = True
        names = [name for name, module in encoder.named_modules() if name and any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )]
        backbone.unfrozen_univ_module_names = tuple(names)
        return names

    encoder.requires_grad_(False)
    unfrozen = []
    blocks = getattr(encoder, "blocks3", None)
    if unfreeze_last_n_blocks:
        if blocks is None or not hasattr(blocks, "__len__"):
            raise ValueError("UNIV encoder has no indexable blocks3 module")
        if unfreeze_last_n_blocks > len(blocks):
            raise ValueError(
                f"cannot unfreeze {unfreeze_last_n_blocks} blocks3 blocks; encoder has {len(blocks)}"
            )
        start = len(blocks) - unfreeze_last_n_blocks
        for index in range(start, len(blocks)):
            blocks[index].requires_grad_(True)
            unfrozen.append(f"blocks3.{index}")
    if unfreeze_norm:
        norm = getattr(encoder, "norm", None)
        if norm is None:
            raise ValueError("UNIV encoder has no final norm module")
        norm.requires_grad_(True)
        unfrozen.append("norm")
    backbone.univ_grad_enabled = bool(unfrozen)
    backbone.unfrozen_univ_module_names = tuple(unfrozen)
    return unfrozen


def build_optimizer(model: torch.nn.Module, backbone: UNIVMTADetectionBackbone, lr: float, univ_lr: float):
    """Build SGD groups with a conservative LR for trainable UNIV weights."""
    univ_ids = {id(parameter) for parameter in backbone.encoder.parameters() if parameter.requires_grad}
    regular = [parameter for parameter in model.parameters()
               if parameter.requires_grad and id(parameter) not in univ_ids]
    univ = [parameter for parameter in backbone.encoder.parameters() if parameter.requires_grad]
    groups = [{"params": regular, "lr": lr}]
    if univ:
        groups.append({"params": univ, "lr": univ_lr, "name": "univ"})
    return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=0.0005)


def parameter_summary(model: torch.nn.Module, backbone: UNIVMTADetectionBackbone) -> dict:
    """Return auditable total and trainable parameter counts by component."""
    count = lambda parameters: sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return {
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": count(model.parameters()),
        "trainable_univ_parameter_count": count(backbone.encoder.parameters()),
        "trainable_adapter_parameter_count": count(backbone.adapter.parameters()),
        "trainable_detector_parameter_count": count(
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("backbone.encoder.") and not name.startswith("backbone.adapter.")
        ),
    }


def training_configuration_summary(args, model, backbone, unfrozen_univ_modules) -> dict:
    """Build the Stage 5 trainability section of the persisted summary."""
    return {
        "image_size": getattr(args, "image_size", 224),
        "input_token_grid_size": [getattr(args, "image_size", 224) // 16] * 2,
        "adapter_output_shapes": {
            "P3": [256] + [getattr(args, "image_size", 224) // 8] * 2,
            "P4": [256] + [getattr(args, "image_size", 224) // 16] * 2,
            "P5": [256] + [getattr(args, "image_size", 224) // 32] * 2,
        },
        "freeze_univ": args.freeze_univ,
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "unfreeze_norm": args.unfreeze_norm,
        "univ_lr": args.univ_lr,
        "unfrozen_univ_module_names": list(unfrozen_univ_modules),
        **parameter_summary(model, backbone),
    }


def build_detector(
    backbone: torch.nn.Module,
    image_size: int = 224,
    image_mean=UNIV_IR_IMAGE_MEAN,
    image_std=UNIV_IR_IMAGE_STD,
    box_score_thresh=None,
):
    """Build a three-level Faster R-CNN head without adding YOLO or fusion."""
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    anchors = AnchorGenerator(
        # RPNHead shares one predictor across levels, so every level must use
        # the same number of sizes/aspect ratios.
        sizes=((16, 32), (64, 128), (128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["P3", "P4", "P5"], output_size=7, sampling_ratio=2)
    detector_kwargs = {}
    if box_score_thresh is not None:
        detector_kwargs["box_score_thresh"] = box_score_thresh
    return FasterRCNN(
        backbone, num_classes=7, min_size=image_size, max_size=image_size,
        rpn_anchor_generator=anchors, box_roi_pool=roi_pooler,
        image_mean=list(image_mean), image_std=list(image_std),
        **detector_kwargs,
    )


def _device_targets(targets, device: torch.device):
    moved = []
    for target in targets:
        item = {key: value.to(device) for key, value in target.items()}
        # M3FD uses IDs 0..5, while torchvision reserves class 0 for background.
        item["labels"] = item["labels"] + 1
        moved.append(item)
    return moved


def evaluate_smoke(model, loader, device: torch.device) -> dict:
    model.eval()
    images, _targets = next(iter(loader))
    images = [image.to(device) for image in images]
    with torch.inference_mode():
        predictions = model(images)
    prediction = predictions[0]
    scores = prediction["scores"]
    return {
        "passed": True,
        "number_of_detections": int(prediction["boxes"].shape[0]),
        "box_tensor_shape": list(prediction["boxes"].shape),
        "score_tensor_shape": list(scores.shape),
        "finite_scores": bool(torch.isfinite(scores).all()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jinlei/database/M3FD_Detection"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/jinlei/checkpoints/UNIV/checkpoint0400.pth"))
    parser.add_argument("--checkpoint-key", choices=("student", "teacher"), default="student")
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    parser.add_argument("--min-load-fraction", type=float, default=0.5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=parse_image_size, default=224)
    parser.add_argument("--image-mean", type=float, nargs=3, default=list(UNIV_IR_IMAGE_MEAN))
    parser.add_argument("--image-std", type=float, nargs=3, default=list(UNIV_IR_IMAGE_STD))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage4_smoke"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-univ", type=parse_bool, default=True)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-norm", type=parse_bool, default=False)
    parser.add_argument("--univ-lr", type=float, default=1e-5)
    parser.add_argument("--max-train-steps", type=int, default=5)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_train_steps < 1:
        parser.error("epochs, batch-size, and max-train-steps must be positive")
    if args.unfreeze_last_n_blocks < 0 or args.lr <= 0 or args.univ_lr <= 0:
        parser.error("unfreeze-last-n-blocks must be non-negative and learning rates must be positive")

    train_data = M3FDDetectionDataset(args.data_root, args.split, args.image_size)
    val_data = M3FDDetectionDataset(args.data_root, args.val_split, args.image_size)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    # Explicitly force parsing of at least one real image and target before model construction.
    sample_image, sample_target = train_data[0]
    device = torch.device(args.device)
    backbone, load_report = UNIVMTADetectionBackbone.from_checkpoint(
        args.checkpoint, checkpoint_key=args.checkpoint_key,
        min_load_fraction=args.min_load_fraction, source_root=args.source_root,
        freeze_univ=args.freeze_univ,
        image_size=args.image_size,
    )
    unfrozen_univ_modules = configure_univ_trainability(
        backbone, args.freeze_univ, args.unfreeze_last_n_blocks, args.unfreeze_norm
    )
    model = build_detector(backbone, args.image_size, args.image_mean, args.image_std).to(device)
    optimizer = build_optimizer(model, backbone, args.lr, args.univ_lr)
    steps, last_losses = 0, {}
    model.train()
    for _epoch in range(args.epochs):
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = _device_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            total = validate_detection_losses(losses)
            total.backward()
            optimizer.step()
            steps += 1
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}
            if steps >= args.max_train_steps:
                break
        if steps >= args.max_train_steps:
            break

    evaluation = evaluate_smoke(model, val_loader, device)
    if not evaluation["finite_scores"]:
        raise RuntimeError("validation prediction scores contain non-finite values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage4_smoke_checkpoint.pth"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps}, checkpoint_path)
    summary = {
        "stage": 4, "stream": "IR", "train_samples": len(train_data), "val_samples": len(val_data),
        "sample_image_shape": list(sample_image.shape), "sample_box_count": int(sample_target["boxes"].shape[0]),
        "optimizer_steps": steps, "last_losses": last_losses, "checkpoint": str(checkpoint_path),
        "image_mean": args.image_mean, "image_std": args.image_std,
        "checkpoint_key": args.checkpoint_key,
        "pos_embed_resize_info": load_report.pos_embed_resize_info,
        "checkpoint_load": asdict(load_report), "evaluation": evaluation,
        **training_configuration_summary(args, model, backbone, unfrozen_univ_modules),
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
