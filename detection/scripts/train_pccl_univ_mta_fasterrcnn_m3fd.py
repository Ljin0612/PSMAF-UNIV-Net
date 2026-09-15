#!/usr/bin/env python3
"""Train Stage 7 paired PCCL adaptation while retaining IR-only detection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from detection.scripts.train_univ_mta_fasterrcnn_m3fd import (
    UNIV_IR_IMAGE_MEAN, UNIV_IR_IMAGE_STD, _device_targets, build_detector,
    configure_univ_trainability, parse_bool, parse_image_size, validate_detection_losses,
)
from psmaf_univ.m3fd_detection import PairedM3FDDetectionDataset, paired_detection_collate_fn
from psmaf_univ.pccl import pccl_objective, sample_patch_indices, sampled_attention_pseudo_labels
from psmaf_univ.univ_mta_detection_backbone import UNIVMTADetectionBackbone

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


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
    parser.add_argument("--image-size", type=parse_image_size, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--univ-lr", type=float, default=1e-5)
    parser.add_argument("--pccl-lr", type=float, default=None)
    parser.add_argument("--freeze-univ", type=parse_bool, default=False)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-norm", type=parse_bool, default=False)
    parser.add_argument("--pccl-enabled", type=parse_bool, default=True)
    parser.add_argument("--trainable-rgb-features", type=parse_bool, default=True)
    parser.add_argument("--lambda-pccl", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--pccl-num-patches", type=int, default=256)
    parser.add_argument("--pccl-feature-level", choices=("norm", "p4", "multiscale"), default="norm")
    parser.add_argument("--max-train-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage7_train"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/stage7_logs"))
    return parser


def _normalize_rgb(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(RGB_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(RGB_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _anchor_forward(anchor: nn.Module, rgb: torch.Tensor):
    with torch.no_grad():
        tokens, attention = anchor(_normalize_rgb(rgb), mask_ratio=0, return_last_attention=True)
    if attention is None:
        raise RuntimeError("frozen RGB anchor did not return its last self-attention")
    return tokens.detach(), attention.detach()


def validate_pccl_configuration(args, backbone: UNIVMTADetectionBackbone | None = None) -> None:
    """Reject PCCL configurations which can only train disposable projectors."""
    encoder_trainable = (not args.freeze_univ) if backbone is None else any(
        parameter.requires_grad for parameter in backbone.encoder.parameters())
    if args.pccl_enabled and args.pccl_feature_level == "norm" and not encoder_trainable:
        raise ValueError(
            "PCCL on frozen encoder tokens does not update inference-time parameters. Use "
            "--freeze-univ false, partial unfreeze, LoRA, or --pccl-feature-level p4/multiscale."
        )


def build_pccl_optimizer(model, backbone, projectors, *, lr: float, univ_lr: float, pccl_lr=None):
    """Use distinct detector/MTA, UNIV, and PCCL-projector learning rates."""
    univ_ids = {id(p) for p in backbone.encoder.parameters()}
    projector_ids = {id(p) for module in projectors for p in module.parameters()}
    regular = [p for p in model.parameters() if p.requires_grad and id(p) not in univ_ids]
    univ = [p for p in backbone.encoder.parameters() if p.requires_grad]
    projector_params = [p for module in projectors for p in module.parameters() if p.requires_grad]
    groups = []
    if regular:
        groups.append({"params": regular, "lr": lr, "name": "detector_mta"})
    if univ:
        groups.append({"params": univ, "lr": univ_lr, "name": "univ"})
    if projector_params:
        groups.append({"params": projector_params, "lr": lr if pccl_lr is None else pccl_lr,
                       "name": "pccl_projectors"})
    assert not projector_ids.intersection(univ_ids)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=0.0005)


def pccl_mta_tokens(pyramid, feature_level: str) -> torch.Tensor:
    """Convert inference-time MTA features to tokens aligned with UNIV P4."""
    p4 = pyramid["P4"]
    features = [p4]
    if feature_level == "multiscale":
        features = [torch.nn.functional.interpolate(pyramid[name], p4.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                    for name in ("P3", "P4", "P5")]
    feature = torch.cat(features, dim=1)
    return feature.flatten(2).transpose(1, 2)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.max_train_steps < 1
            or args.pccl_num_patches < 1 or args.temperature <= 0 or args.lambda_pccl < 0
            or args.lr <= 0 or args.univ_lr <= 0
            or (args.pccl_lr is not None and args.pccl_lr <= 0)
            or args.alpha < 0 or args.beta < 0 or not 0 <= args.gamma <= 1):
        parser.error("invalid positive PCCL/training parameter or gamma outside [0, 1]")

    train_data = PairedM3FDDetectionDataset(args.data_root, args.split, args.image_size)
    val_data = PairedM3FDDetectionDataset(args.data_root, args.val_split, args.image_size)
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=paired_detection_collate_fn)
    sample_ir, sample_rgb, sample_target = train_data[0]
    device = torch.device(args.device)
    backbone, load_report = UNIVMTADetectionBackbone.from_checkpoint(
        args.checkpoint, checkpoint_key=args.checkpoint_key, min_load_fraction=args.min_load_fraction,
        source_root=args.source_root, freeze_univ=args.freeze_univ, image_size=args.image_size)
    configure_univ_trainability(backbone, args.freeze_univ, args.unfreeze_last_n_blocks, args.unfreeze_norm)
    try:
        validate_pccl_configuration(args, backbone)
    except ValueError as error:
        parser.error(str(error))
    anchor_backbone, anchor_report = UNIVMTADetectionBackbone.from_checkpoint(
        args.checkpoint, checkpoint_key=args.checkpoint_key, min_load_fraction=args.min_load_fraction,
        source_root=args.source_root, freeze_univ=True, image_size=args.image_size)
    anchor = anchor_backbone.encoder.to(device).eval().requires_grad_(False)
    model = build_detector(backbone, args.image_size, UNIV_IR_IMAGE_MEAN, UNIV_IR_IMAGE_STD).to(device)
    semantic_dim = int(getattr(backbone.encoder.norm, "normalized_shape", (768,))[-1])
    ir_input_dim = semantic_dim if args.pccl_feature_level == "norm" else (
        backbone.out_channels * (3 if args.pccl_feature_level == "multiscale" else 1))
    ir_projector = nn.Linear(ir_input_dim, semantic_dim, bias=False).to(device)
    rgb_projector = nn.Linear(semantic_dim, semantic_dim, bias=False).to(device)
    if ir_input_dim == semantic_dim:
        nn.init.eye_(ir_projector.weight)
    nn.init.eye_(rgb_projector.weight)
    optimizer = build_pccl_optimizer(model, backbone, (ir_projector, rgb_projector), lr=args.lr,
                                     univ_lr=args.univ_lr, pccl_lr=args.pccl_lr)
    captured = {}
    hook = backbone.encoder.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("ir", out))
    adapter_hook = backbone.adapter.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("pyramid", out))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.log_dir / "metrics_history.jsonl"
    optimizer_steps = 0
    detection_last = pccl_last = total_last = None
    model.train()
    try:
        with history_path.open("w", encoding="utf-8") as history:
            for epoch in range(args.epochs):
                for ir_images, rgb_images, targets in loader:
                    if optimizer_steps >= args.max_train_steps:
                        break
                    ir = [image.to(device) for image in ir_images]
                    rgb = torch.stack([image.to(device) for image in rgb_images])
                    optimizer.zero_grad(set_to_none=True)
                    losses = model(ir, _device_targets(targets, device))
                    detection_loss = validate_detection_losses(losses)
                    pccl_loss = detection_loss.new_zeros(())
                    loss_ia = loss_va = pccl_loss
                    if args.pccl_enabled:
                        anchors, attention = _anchor_forward(anchor, rgb)
                        ir_tokens = (captured.get("ir") if args.pccl_feature_level == "norm" else
                                     pccl_mta_tokens(captured.get("pyramid"), args.pccl_feature_level))
                        if ir_tokens is None:
                            raise RuntimeError("IR norm tokens were not captured")
                        indices = sample_patch_indices(anchors.shape[1], args.pccl_num_patches, device=device)
                        anchors = anchors[:, indices]
                        ir_tokens = ir_projector(ir_tokens[:, indices])
                        rgb_tokens = rgb_projector(anchors) if args.trainable_rgb_features else None
                        pseudo = sampled_attention_pseudo_labels(attention, indices, args.gamma)
                        pccl_loss, loss_ia, loss_va = pccl_objective(
                            anchors, ir_tokens, rgb_tokens, pseudo, alpha=args.alpha,
                            beta=args.beta, temperature=args.temperature)
                    total = detection_loss + args.lambda_pccl * pccl_loss
                    if not torch.isfinite(total):
                        raise RuntimeError("Stage 7 total loss is not finite")
                    total.backward()
                    optimizer.step()
                    optimizer_steps += 1
                    detection_last, pccl_last, total_last = map(
                        lambda value: float(value.detach().cpu()), (detection_loss, pccl_loss, total))
                    history.write(json.dumps({"epoch": epoch + 1, "optimizer_step": optimizer_steps,
                        "detection_loss": detection_last, "pcc_loss": pccl_last,
                        "loss_ia": float(loss_ia.detach()), "loss_va": float(loss_va.detach()),
                        "total_loss": total_last, "elapsed_time": time.time()}) + "\n")
                    history.flush()
                if optimizer_steps >= args.max_train_steps:
                    break
    finally:
        hook.remove()
        adapter_hook.remove()

    checkpoint_path = args.output_dir / "last.pth"
    torch.save({"model": model.state_dict(), "model_state_dict": model.state_dict(),
        "pccl_projectors": {"ir": ir_projector.state_dict(), "rgb": rgb_projector.state_dict()},
        "optimizer": optimizer.state_dict(), "optimizer_steps": optimizer_steps,
        "image_size": args.image_size, "checkpoint_key": args.checkpoint_key,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, checkpoint_path)
    grid = args.image_size // 16
    summary = {"stage": 7, "dataset": "M3FD-paired", "evaluation_stream": "IR",
        "image_size": args.image_size, "pccl_enabled": args.pccl_enabled,
        "lambda_pccl": args.lambda_pccl, "alpha": args.alpha, "beta": args.beta,
        "gamma": args.gamma, "temperature": args.temperature,
        "pccl_feature_level": args.pccl_feature_level,
        "pccl_num_patches": args.pccl_num_patches, "pcc_loss_last": pccl_last,
        "detection_loss_last": detection_last, "total_loss_last": total_last,
        "checkpoint_key": args.checkpoint_key, "input_token_grid_size": [grid, grid],
        "pos_embed_resize_info": load_report.pos_embed_resize_info,
        "anchor_pos_embed_resize_info": anchor_report.pos_embed_resize_info,
        "adapter_output_shapes": {"P3": [256, grid * 2, grid * 2], "P4": [256, grid, grid],
                                  "P5": [256, grid // 2, grid // 2]},
        "train_samples": len(train_data), "val_samples": len(val_data),
        "optimizer_steps": optimizer_steps, "checkpoint": str(checkpoint_path),
        "metrics_history": str(history_path), "sample_ir_shape": list(sample_ir.shape),
        "sample_rgb_shape": list(sample_rgb.shape), "sample_box_count": int(sample_target["boxes"].shape[0]),
        "checkpoint_load": asdict(load_report)}
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
