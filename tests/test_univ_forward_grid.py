"""Regression tests for UNIV encoder token-grid derivation."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "UNIV-main/models/backbone/mcmae/models_convmae.py"


def _forward_encoder_ast() -> ast.FunctionDef:
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MaskedAutoencoderConvViT"
    )
    return next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_encoder"
    )


def _model_module():
    pytest.importorskip("torch")
    from psmaf_univ.compat import apply_numpy_legacy_aliases

    apply_numpy_legacy_aliases()
    source_root = str(ROOT / "UNIV-main")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return importlib.import_module("models.backbone.mcmae.models_convmae")


def test_forward_encoder_derives_token_count_from_returned_mask():
    function = _forward_encoder_ast()
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]

    assert any(
        any(isinstance(target, ast.Name) and target.id == "token_count" for target in node.targets)
        and ast.unparse(node.value) == "mask.shape[1]"
        for node in assignments
    )
    assert not any(isinstance(node, ast.Name) and node.id == "L" for node in ast.walk(function))


def test_forward_encoder_rejects_non_square_token_count_clearly():
    torch = pytest.importorskip("torch")
    model_class = _model_module().MaskedAutoencoderConvViT
    encoder = SimpleNamespace(
        random_masking=lambda _x, _ratio: (
            torch.zeros(1, 5, dtype=torch.long),
            torch.zeros(1, 15),
            torch.zeros(1, 15, dtype=torch.long),
        )
    )

    with pytest.raises(ValueError, match=r"square token grid; got 15 tokens"):
        model_class.forward_encoder(encoder, torch.zeros(1, 3, 16, 16), 0)


@pytest.mark.parametrize("image_size,token_count", [(224, 196), (320, 400)])
def test_supported_resolution_forward_uses_square_grid(image_size, token_count):
    torch = pytest.importorskip("torch")
    module = _model_module()
    model = module.MaskedAutoencoderConvViT(
        img_size=[image_size, image_size // 4, image_size // 8],
        patch_size=[4, 2, 2],
        embed_dim=[4, 8, 16],
        depth=[0, 0, 1],
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=0,
        decoder_num_heads=4,
        mlp_ratio=[1, 1, 1],
    ).eval()

    with torch.inference_mode():
        latent, attention = model(torch.zeros(1, 3, image_size, image_size), mask_ratio=0)

    assert latent.shape == (1, token_count, 16)
    assert attention is None
