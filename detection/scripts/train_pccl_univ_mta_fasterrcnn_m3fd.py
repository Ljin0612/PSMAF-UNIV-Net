#!/usr/bin/env python3
"""Train Stage 7 paired PCCL adaptation while retaining IR-only detection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from detection.scripts.train_univ_mta_fasterrcnn_m3fd import (
    UNIV_IR_IMAGE_MEAN, UNIV_IR_IMAGE_STD, _device_targets, build_detector,
    configure_univ_trainability, parse_bool, parse_image_size, validate_detection_losses,
    resolve_resume_progress,
)
from psmaf_univ.lora_adapter import attach_lora, is_lora_parameter, lora_state_dict
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
    parser.add_argument("--freeze-univ", type=parse_bool, default=True)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
    parser.add_argument("--unfreeze-norm", type=parse_bool, default=False)
    parser.add_argument("--pccl-enabled", type=parse_bool, default=True)
    parser.add_argument("--trainable-rgb-features", type=parse_bool, default=True)
    parser.add_argument("--lambda-pccl", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--pccl-num-patches", type=int, default=256)
    parser.add_argument("--pccl-feature-level", choices=("norm", "p4", "multiscale"), default="norm")
    parser.add_argument("--max-train-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage7_train"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/stage7_logs"))
    parser.add_argument("--lora-enabled", type=parse_bool, default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--save-interval-steps", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
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


def shared_encoder_tokens(backbone: UNIVMTADetectionBackbone, rgb: torch.Tensor) -> torch.Tensor:
    """Run visible images through the exact encoder used by the IR detector."""
    captured = {}
    handle = backbone.encoder.norm.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("norm", output))
    try:
        backbone.encoder(_normalize_rgb(rgb), mask_ratio=0, return_last_attention=True)
    finally:
        handle.remove()
    if "norm" not in captured:
        raise RuntimeError("shared UNIV encoder did not produce visible norm tokens")
    return captured["norm"]


def validate_pccl_configuration(args, backbone: UNIVMTADetectionBackbone | None = None) -> None:
    """Reject PCCL configurations which can only train disposable projectors."""
    encoder_trainable = (not args.freeze_univ) if backbone is None else any(
        parameter.requires_grad for parameter in backbone.encoder.parameters())
    if args.pccl_enabled and args.pccl_feature_level == "norm" and not encoder_trainable:
        raise ValueError(
            "PCCL on frozen encoder tokens does not update inference-time parameters. Use "
            "--freeze-univ false, partial unfreeze, LoRA, or --pccl-feature-level p4/multiscale."
        )


def build_pccl_optimizer(model, backbone, projectors, *, lr: float, univ_lr: float,
                         pccl_lr=None, lora_lr=None):
    """Use distinct detector/MTA, UNIV, and PCCL-projector learning rates."""
    named_univ = list(backbone.encoder.named_parameters())
    lora_ids = {id(p) for name, p in named_univ if is_lora_parameter(name)}
    univ_ids = {id(p) for _name, p in named_univ}
    projector_ids = {id(p) for module in projectors for p in module.parameters()}
    regular = [p for p in model.parameters() if p.requires_grad and id(p) not in univ_ids]
    lora = [p for name, p in named_univ if p.requires_grad and is_lora_parameter(name)]
    univ = [p for name, p in named_univ if p.requires_grad and not is_lora_parameter(name)]
    projector_params = [p for module in projectors for p in module.parameters() if p.requires_grad]
    groups = []
    if regular:
        groups.append({"params": regular, "lr": lr, "name": "detector_mta"})
    if univ:
        groups.append({"params": univ, "lr": univ_lr, "name": "univ"})
    if lora:
        groups.append({"params": lora, "lr": lora_lr if lora_lr is not None else univ_lr,
                       "name": "lora"})
    if projector_params:
        groups.append({"params": projector_params, "lr": lr if pccl_lr is None else pccl_lr,
                       "name": "pccl_projectors"})
    assert not projector_ids.intersection(univ_ids)
    assert not any(id(p) in lora_ids for p in univ)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=0.0005)


def _checkpoint_payload(model, backbone, projectors, optimizer, args, **progress) -> dict[str, Any]:
    state = model.state_dict()
    return {"model": state, "model_state_dict": state,
            "lora_state_dict": lora_state_dict(backbone.encoder),
            "pccl_projectors": {"ir": projectors[0].state_dict(), "rgb": projectors[1].state_dict()},
            "optimizer": optimizer.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "image_size": args.image_size, "checkpoint_key": args.checkpoint_key,
            "config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()}, **progress}


def save_pccl_checkpoint(path, model, backbone, projectors, optimizer, args, **progress):
    payload = _checkpoint_payload(model, backbone, projectors, optimizer, args, **progress)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


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
            or args.lora_rank < 1 or args.lora_alpha <= 0 or args.lora_lr <= 0
            or not 0 <= args.lora_dropout < 1 or args.save_interval_steps < 1
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
    if args.lora_enabled:
        if not args.freeze_univ or args.unfreeze_last_n_blocks or args.unfreeze_norm:
            parser.error("LoRA requires a frozen UNIV base and cannot be combined with block/norm unfreezing")
        lora_modules = attach_lora(backbone.encoder, rank=args.lora_rank, alpha=args.lora_alpha,
                                   dropout=args.lora_dropout)
        backbone.univ_grad_enabled = True
        backbone.unfrozen_univ_module_names = tuple(lora_modules)
    else:
        lora_modules = []
        configure_univ_trainability(backbone, args.freeze_univ, args.unfreeze_last_n_blocks,
                                    args.unfreeze_norm)
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
                                     univ_lr=args.univ_lr, pccl_lr=args.pccl_lr,
                                     lora_lr=args.lora_lr)
    captured = {}
    hook = backbone.encoder.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("ir", out))
    adapter_hook = backbone.adapter.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("pyramid", out))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.log_dir / "metrics_history.jsonl"
    optimizer_steps = global_step = completed_epochs = steps_in_current_epoch = 0
    detection_last = pccl_last = total_last = loss_ia_last = loss_va_last = None
    checkpoint_path = args.output_dir / "last.pth"
    resume_info = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload.get("model")) if isinstance(payload, Mapping) else None
        if state is None:
            raise ValueError("resume checkpoint does not contain a model state")
        model.load_state_dict(state, strict=True)
        projector_state = payload.get("pccl_projectors")
        if projector_state is None:
            raise ValueError("resume checkpoint does not contain PCCL projector state")
        ir_projector.load_state_dict(projector_state["ir"])
        rgb_projector.load_state_dict(projector_state["rgb"])
        optimizer_state = payload.get("optimizer_state_dict", payload.get("optimizer"))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        progress = resolve_resume_progress(payload, len(loader))
        global_step, optimizer_steps = progress["global_step"], progress["optimizer_steps"]
        completed_epochs = progress["completed_epochs"]
        steps_in_current_epoch = progress["steps_in_current_epoch"]
        resume_info = progress
    model.train()
    try:
        mode = "a" if args.resume is not None and history_path.exists() else "w"
        with history_path.open(mode, encoding="utf-8") as history:
            for epoch in range(completed_epochs, args.epochs):
                epoch_finished = True
                for batch_index, (ir_images, rgb_images, targets) in enumerate(loader):
                    if epoch == completed_epochs and batch_index < steps_in_current_epoch:
                        continue
                    if optimizer_steps >= args.max_train_steps:
                        epoch_finished = False
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
                        # This is the Stage 7.3 semantic fix: FV comes from the same
                        # trainable encoder object/weights as FI, not from FA.
                        rgb_tokens = (rgb_projector(shared_encoder_tokens(backbone, rgb)[:, indices])
                                      if args.trainable_rgb_features else None)
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
                    global_step += 1
                    steps_in_current_epoch = batch_index + 1
                    detection_last, pccl_last, total_last = map(
                        lambda value: float(value.detach().cpu()), (detection_loss, pccl_loss, total))
                    loss_ia_last = float(loss_ia.detach().cpu())
                    loss_va_last = float(loss_va.detach().cpu())
                    history.write(json.dumps({"epoch": epoch + 1, "optimizer_step": optimizer_steps,
                        "detection_loss": detection_last, "pcc_loss": pccl_last,
                        "loss_ia": float(loss_ia.detach()), "loss_va": float(loss_va.detach()),
                        "total_loss": total_last, "elapsed_time": time.time()}) + "\n")
                    history.flush()
                    if optimizer_steps % args.save_interval_steps == 0:
                        save_pccl_checkpoint(checkpoint_path, model, backbone,
                            (ir_projector, rgb_projector), optimizer, args, epoch=epoch,
                            completed_epochs=completed_epochs, global_step=global_step,
                            steps=global_step, optimizer_steps=optimizer_steps,
                            steps_in_current_epoch=steps_in_current_epoch,
                            batch_index_in_epoch=steps_in_current_epoch)
                    if optimizer_steps >= args.max_train_steps:
                        epoch_finished = batch_index + 1 == len(loader)
                        break
                if epoch_finished:
                    completed_epochs = epoch + 1
                    steps_in_current_epoch = 0
                    save_pccl_checkpoint(checkpoint_path, model, backbone,
                        (ir_projector, rgb_projector), optimizer, args, epoch=completed_epochs,
                        completed_epochs=completed_epochs, global_step=global_step,
                        steps=global_step, optimizer_steps=optimizer_steps,
                        steps_in_current_epoch=0, batch_index_in_epoch=0)
                if optimizer_steps >= args.max_train_steps:
                    break
    finally:
        hook.remove()
        adapter_hook.remove()

    # Always persist before any downstream evaluator can be invoked.
    save_pccl_checkpoint(checkpoint_path, model, backbone, (ir_projector, rgb_projector),
        optimizer, args, epoch=completed_epochs, completed_epochs=completed_epochs,
        global_step=global_step, steps=global_step, optimizer_steps=optimizer_steps,
        steps_in_current_epoch=steps_in_current_epoch,
        batch_index_in_epoch=steps_in_current_epoch)
    grid = args.image_size // 16
    summary = {"stage": 7, "dataset": "M3FD-paired", "evaluation_stream": "IR",
        "image_size": args.image_size, "pccl_enabled": args.pccl_enabled,
        "lambda_pccl": args.lambda_pccl, "alpha": args.alpha, "beta": args.beta,
        "gamma": args.gamma, "temperature": args.temperature,
        "pccl_feature_level": args.pccl_feature_level,
        "pccl_num_patches": args.pccl_num_patches, "pcc_loss_last": pccl_last,
        "loss_ia_last": loss_ia_last, "loss_va_last": loss_va_last,
        "detection_loss_last": detection_last, "total_loss_last": total_last,
        "lora_enabled": args.lora_enabled, "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
        "lora_lr": args.lora_lr,
        "checkpoint_key": args.checkpoint_key, "input_token_grid_size": [grid, grid],
        "pos_embed_resize_info": load_report.pos_embed_resize_info,
        "anchor_pos_embed_resize_info": anchor_report.pos_embed_resize_info,
        "adapter_output_shapes": {"P3": [256, grid * 2, grid * 2], "P4": [256, grid, grid],
                                  "P5": [256, grid // 2, grid // 2]},
        "train_samples": len(train_data), "val_samples": len(val_data),
        "optimizer_steps": optimizer_steps, "checkpoint": str(checkpoint_path),
        "global_step": global_step, "completed_epochs": completed_epochs,
        "resume_from": str(args.resume) if args.resume else None,
        "resume_mode": resume_info["resume_mode"] if resume_info else "fresh_start",
        "optimizer_groups": [{"name": group.get("name", "unnamed"), "lr": group["lr"]}
                             for group in optimizer.param_groups],
        "metrics_history": str(history_path), "sample_ir_shape": list(sample_ir.shape),
        "sample_rgb_shape": list(sample_rgb.shape), "sample_box_count": int(sample_target["boxes"].shape[0]),
        "checkpoint_load": asdict(load_report)}
    count = lambda parameters: sum(p.numel() for p in parameters if p.requires_grad)
    summary.update({
        "trainable_lora_parameter_count": sum(p.numel() for n, p in backbone.encoder.named_parameters()
                                                if p.requires_grad and is_lora_parameter(n)),
        "trainable_univ_base_parameter_count": sum(p.numel() for n, p in backbone.encoder.named_parameters()
                                                    if p.requires_grad and not is_lora_parameter(n)),
        "trainable_adapter_parameter_count": count(backbone.adapter.parameters()),
        "trainable_detector_parameter_count": count(p for n, p in model.named_parameters()
            if not n.startswith("backbone.encoder.") and not n.startswith("backbone.adapter.")),
        "trainable_projector_parameter_count": count(ir_projector.parameters()) + count(rgb_projector.parameters()),
    })
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
