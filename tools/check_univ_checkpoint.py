#!/usr/bin/env python3
"""Inspect a UNIV checkpoint and, optionally, test-load its encoder weights."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

# Direct script execution puts tools/, rather than the repository root, first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psmaf_univ.checkpoint_loader import resize_pos_embed_if_needed
from psmaf_univ.univ_diagnostics import build_original_univ


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state(value: Any) -> dict[str, torch.Tensor] | None:
    """Return only tensor entries from a state-like mapping."""
    if not isinstance(value, Mapping):
        return None
    tensors = {str(key): item for key, item in value.items() if isinstance(item, torch.Tensor)}
    return tensors or ({} if not value else None)


def _select_state(
    checkpoint: Any, requested_key: str
) -> tuple[dict[str, torch.Tensor], str]:
    """Select the requested branch, with compatibility fallbacks for base weights."""
    if isinstance(checkpoint, Mapping):
        requested = _tensor_state(checkpoint.get(requested_key))
        if requested is not None:
            return requested, requested_key
        for key in ("student", "model", "state_dict", "teacher"):
            state = _tensor_state(checkpoint.get(key))
            if state is not None:
                return state, key
        direct = _tensor_state(checkpoint)
        if direct is not None:
            return direct, "<root>"
    raise ValueError(
        f"checkpoint key {requested_key!r} is unavailable and no tensor state-dict branch was found"
    )


def _has(keys: list[str], *parts: str) -> bool:
    return any(all(part in key for part in parts) for key in keys)


def _shape_for_suffix(state: Mapping[str, torch.Tensor], suffix: str) -> list[int] | None:
    for key, tensor in state.items():
        if key == suffix or key.endswith("." + suffix):
            return list(tensor.shape)
    return None


def _finite_report(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    nan_keys: list[str] = []
    inf_keys: list[str] = []
    checked = 0
    for key, tensor in state.items():
        if tensor.is_floating_point() or tensor.is_complex():
            checked += 1
            if torch.isnan(tensor).any().item():
                nan_keys.append(key)
            if torch.isinf(tensor).any().item():
                inf_keys.append(key)
    return {
        "checked_tensor_count": checked,
        "has_nan": bool(nan_keys),
        "has_inf": bool(inf_keys),
        "nan_keys": nan_keys,
        "inf_keys": inf_keys,
    }


def _important(keys: list[str]) -> list[str]:
    markers = ("patch_embed", "pos_embed", "blocks", "attn", "mlp", "norm")
    return [key for key in keys if any(marker in key for marker in markers)]


def _load_against_model(
    state: Mapping[str, torch.Tensor], source_root: Path, device: str
) -> dict[str, Any]:
    """Construct the checked-in original encoder and safely test-load ``state``."""
    try:
        model = build_original_univ(source_root).to(device)
    except Exception as exc:  # Report missing optional upstream imports without claiming success.
        return {
            "status": "unavailable",
            "message": (
                "Could not build the original UNIV encoder. Install the dependency or restore "
                f"the constructor named below, then retry --build-model: {type(exc).__name__}: {exc}"
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    candidate = {key.removeprefix("module."): value for key, value in state.items()}
    model_state = model.state_dict()
    pos_info = resize_pos_embed_if_needed(candidate, model_state)
    resized = ["pos_embed"] if pos_info["resized"] else []
    skipped: list[str] = []
    if pos_info["source_shape"] != pos_info["target_shape"] and pos_info["skipped_reason"]:
        skipped.append("pos_embed")
    for key in list(candidate):
        if key in model_state and candidate[key].shape != model_state[key].shape:
            candidate.pop(key)
            skipped.append(key)
    incompatible = model.load_state_dict(candidate, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    return {
        "status": "loaded",
        "loaded_key_count": sum(key in model_state for key in candidate),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "resized_keys": resized,
        "skipped_shape_mismatch_keys": list(dict.fromkeys(skipped)),
        "pos_embed_resize_info": pos_info,
        "important_missing_keys": _important(missing),
        "important_unexpected_keys": _important(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def inspect_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    path = args.checkpoint_option or args.checkpoint
    if path is None:
        raise ValueError("a checkpoint is required (positional CHECKPOINT or --checkpoint CHECKPOINT)")
    path = path.expanduser().resolve()
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    state, selected = _select_state(checkpoint, args.checkpoint_key)
    student = _tensor_state(checkpoint.get("student")) if isinstance(checkpoint, Mapping) else None
    teacher = _tensor_state(checkpoint.get("teacher")) if isinstance(checkpoint, Mapping) else None
    keys = list(state)
    lora_keys = [key for key in keys if "lora_" in key.lower() or ".lora." in key.lower()]
    dtype_counts = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in state.values())
    n = args.max_keys

    report: dict[str, Any] = {
        "checkpoint_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "top_level_keys": [str(key) for key in checkpoint] if isinstance(checkpoint, Mapping) else [],
        "requested_checkpoint_branch": args.checkpoint_key,
        "selected_checkpoint_branch": selected,
        "student_exists": student is not None,
        "teacher_exists": teacher is not None,
        "student_tensor_count": len(student or {}),
        "teacher_tensor_count": len(teacher or {}),
        "selected_tensor_count": len(state),
        "lora_key_count": len(lora_keys),
        "sample_lora_keys": lora_keys[:n],
        "pos_embed_shape": _shape_for_suffix(state, "pos_embed"),
        "patch_embed_key_exists": _has(keys, "patch_embed"),
        "encoder_block_key_exists": _has(keys, "block"),
        "norm_key_exists": _has(keys, "norm"),
        "attention_qkv_key_exists": _has(keys, "attn", "qkv"),
        "attention_proj_key_exists": _has(keys, "attn", "proj"),
        "mlp_key_exists": _has(keys, "mlp"),
        "fc_key_exists": _has(keys, "fc"),
        "dtype_summary": dict(sorted(dtype_counts.items())),
        "nan_inf_check": _finite_report(state),
        "first_keys": keys[:n],
        "last_keys": keys[-n:] if n else [],
    }
    if args.print_keys:
        report["all_keys"] = keys
    if args.build_model:
        report["model_load"] = _load_against_model(state, args.source_root, args.device)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, help="checkpoint path")
    parser.add_argument("--checkpoint", dest="checkpoint_option", type=Path, help="checkpoint path")
    parser.add_argument("--checkpoint-key", default="student", help="preferred state-dict branch")
    parser.add_argument("--device", default="cpu", help="torch load/model device (default: cpu)")
    parser.add_argument("--build-model", action="store_true", help="test-load the original UNIV encoder")
    parser.add_argument("--json-out", type=Path, help="also write the complete JSON report here")
    parser.add_argument("--print-keys", action="store_true", help="include the complete selected key list")
    parser.add_argument("--max-keys", type=int, default=50, help="maximum keys in each sample (default: 50)")
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.checkpoint_option is not None:
        parser.error("use either positional CHECKPOINT or --checkpoint, not both")
    if args.checkpoint is None and args.checkpoint_option is None:
        parser.error("a checkpoint is required")
    if args.max_keys < 0:
        parser.error("--max-keys must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = inspect_checkpoint(args)
    output = json.dumps(report, indent=2)
    print(output)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
