import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from psmaf_univ import PSMAFUNIVModel
from psmaf_univ.checkpoint_loader import (
    extract_state_dict,
    load_univ_checkpoint,
    resize_pos_embed_if_needed,
)
from psmaf_univ.multiscale_task_adapter import MultiscaleTaskAdapter
from psmaf_univ.univ_encoder_wrapper import UNIVEncoderWrapper
from psmaf_univ.univ_diagnostics import capture_module_outputs, inspect_features, model_inventory


class Encoder(nn.Module):
    def forward(self, image):
        return (image, nn.functional.avg_pool2d(image, 2), nn.functional.avg_pool2d(image, 4))


def test_model_emits_multiscale_features():
    model = PSMAFUNIVModel(Encoder(), [3, 3, 3], out_channels=8)
    output = model(torch.randn(2, 3, 32, 32))
    assert [item.shape for item in output] == [(2, 8, 32, 32), (2, 8, 16, 16), (2, 8, 8, 8)]


class TokenEncoder(nn.Module):
    def __init__(self, token_count, spatial_shape=None):
        super().__init__()
        self.token_count = token_count
        self.spatial_shape = spatial_shape

    def forward(self, image):
        result = {"tokens": torch.randn(image.shape[0], self.token_count, 3)}
        if self.spatial_shape is not None:
            result["spatial_shape"] = self.spatial_shape
        return result


def test_wrapper_retains_structured_token_metadata_and_supports_legacy_mode():
    wrapper = UNIVEncoderWrapper(TokenEncoder(512, (16, 32)), return_dict=True)
    structured = wrapper(torch.randn(2, 3, 8, 8), modality="ir")[0]
    assert structured["tokens"].shape == (2, 512, 3)
    assert structured["spatial_shape"] == (16, 32)
    assert structured["modality"] == "ir"
    assert structured["source"] == "univ_encoder"
    assert structured["debug"] == {}
    assert wrapper(torch.randn(2, 3, 8, 8), return_dict=False)[0].shape == (2, 512, 3)


def test_model_forwards_rectangular_token_metadata():
    model = PSMAFUNIVModel(TokenEncoder(512, (16, 32)), [3], out_channels=8)
    assert model(torch.randn(2, 3, 8, 8))[0].shape == (2, 8, 16, 32)


def test_model_explains_how_to_supply_missing_token_shape():
    model = PSMAFUNIVModel(TokenEncoder(6), [3], out_channels=8)
    with pytest.raises(ValueError, match="pass rgb_spatial_shape"):
        model(torch.randn(2, 3, 8, 8))
    assert model(torch.randn(2, 3, 8, 8), rgb_spatial_shape=(2, 3))[0].shape == (2, 8, 2, 3)


def test_model_square_token_inference_is_explicit_opt_in():
    model = PSMAFUNIVModel(TokenEncoder(16), [3], out_channels=8, allow_square_infer=True)
    assert model(torch.randn(2, 3, 8, 8))[0].shape == (2, 8, 4, 4)


def test_checkpoint_branch_priority():
    tensor = torch.ones(1)
    assert extract_state_dict({"student": {"weight": tensor}})["weight"] is tensor


@pytest.mark.parametrize("branch", ["student", "model", "state_dict"])
def test_fake_checkpoint_branches_are_accepted(branch):
    tensor = torch.arange(3)
    extracted = extract_state_dict({branch: {"weight": tensor}})
    assert list(extracted) == ["weight"]
    assert extracted["weight"] is tensor


def test_invalid_fake_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="no recognizable state dictionary"):
        extract_state_dict({"epoch": 12})


def test_pos_embed_is_resized_and_special_token_is_preserved():
    cls_token = torch.full((1, 1, 8), 42.0)
    state = {"pos_embed": torch.cat((cls_token, torch.randn(1, 14 * 14, 8)), dim=1)}
    report = resize_pos_embed_if_needed(state, {"pos_embed": torch.empty(1, 1 + 32 * 32, 8)})

    assert state["pos_embed"].shape == (1, 1025, 8)
    assert torch.equal(state["pos_embed"][:, :1], cls_token)
    assert report["resized"] is True
    assert report["source_grid_size"] == (14, 14)
    assert report["target_grid_size"] == (32, 32)


def test_pos_embed_dimension_mismatch_is_removed():
    state = {"pos_embed": torch.randn(1, 196, 4)}
    report = resize_pos_embed_if_needed(state, {"pos_embed": torch.randn(1, 1024, 8)})
    assert "pos_embed" not in state
    assert report["skipped_reason"] == "embedding dimensions differ"


def test_checkpoint_load_report_records_resize(tmp_path):
    class PositionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Parameter(torch.zeros(1, 16, 4))

    path = tmp_path / "checkpoint.pt"
    torch.save({"student": {"pos_embed": torch.randn(1, 4, 4)}}, path)
    report = load_univ_checkpoint(PositionalModel(), path)

    assert report.resized_keys == ["pos_embed"]
    assert report.loaded_key_count == 1
    assert report.checkpoint_key == "student"
    assert report.missing_keys == []


def test_checkpoint_loader_strips_module_prefix_and_skips_bad_shapes(tmp_path):
    model = nn.Linear(3, 2)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": {"module.weight": torch.ones_like(model.weight), "module.bias": torch.ones(3)}},
        path,
    )

    with pytest.warns(UserWarning, match="bias"):
        report = load_univ_checkpoint(model, path)

    assert torch.equal(model.weight, torch.ones_like(model.weight))
    assert report.checkpoint_key == "model"
    assert report.loaded_key_count == 1
    assert report.skipped_shape_mismatch_keys == ["bias"]
    assert report.missing_keys == ["bias"]


def test_rectangular_bnc_features_require_and_use_spatial_shape():
    adapter = MultiscaleTaskAdapter([3], 5)
    tokens = torch.randn(2, 6, 3)
    assert adapter(tokens, spatial_shape=(2, 3))[0].shape == (2, 5, 2, 3)
    with pytest.raises(ValueError, match="require spatial_shape"):
        adapter(tokens)
    with pytest.raises(ValueError, match="does not match spatial shape"):
        adapter(tokens, grid_size=(3, 3))


def test_feature_dict_and_opt_in_square_inference():
    adapter = MultiscaleTaskAdapter([3], 5)
    assert adapter({"tensor": torch.randn(1, 6, 3), "grid_size": (3, 2)})[0].shape == (1, 5, 3, 2)
    assert adapter(torch.randn(1, 4, 3), allow_square_infer=True)[0].shape == (1, 5, 2, 2)


def test_rectangular_shapes_are_validated_for_each_adapter_level():
    adapter = MultiscaleTaskAdapter([3, 4], 2)
    outputs = adapter(
        [torch.randn(1, 6, 3), torch.randn(1, 10, 4)],
        spatial_shape=[(2, 3), (2, 5)],
    )
    assert [tuple(output.shape) for output in outputs] == [(1, 2, 2, 3), (1, 2, 2, 5)]
    with pytest.raises(ValueError, match="shape count"):
        adapter([torch.randn(1, 6, 3), torch.randn(1, 10, 4)], spatial_shape=[(2, 3)])


class DiagnosticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage = nn.Conv2d(3, 4, 1)

    def forward(self, image, mask_ratio=0, return_last_attention=False):
        feature = self.stage(image)
        latent = feature.flatten(2).transpose(1, 2)
        attention = torch.ones(image.shape[0], 2, latent.shape[1], latent.shape[1])
        return latent, attention if return_last_attention else None


def test_runtime_diagnostics_capture_features_and_attention():
    report = inspect_features(DiagnosticEncoder(), image_size=(2, 3), module_names=("stage",))
    assert report["activations"]["stage"]["shape"] == [1, 4, 2, 3]
    assert report["output"]["latent"]["shape"] == [1, 6, 4]
    assert report["output"]["attention"]["shape"] == [1, 2, 6, 6]
    assert report["attention_available"] is True


def test_diagnostic_hooks_reject_unknown_module_and_are_removed():
    model = DiagnosticEncoder()
    with pytest.raises(ValueError, match="unknown probe modules"):
        with capture_module_outputs(model, ("missing",)):
            pass
    assert model.stage._forward_hooks == {}


def test_model_inventory_reports_capabilities():
    report = model_inventory(DiagnosticEncoder())
    assert report["parameter_count"] == 16
    assert report["trainable_parameter_count"] == 16
    assert report["has_attention_api"] is False


def test_checkpoint_diagnostic_tool_accepts_a_fake_checkpoint(tmp_path):
    checkpoint = tmp_path / "fake.pt"
    report_path = tmp_path / "report.json"
    torch.save(
        {
            "student": {
                "module.pos_embed": torch.zeros(1, 5, 8),
                "module.patch_embed.proj.weight": torch.ones(2, 3, 1, 1),
                "module.blocks.0.attn.qkv.weight": torch.full((3, 3), 2.0),
            },
            "teacher": {"weight": torch.ones(1)},
        },
        checkpoint,
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "tools/check_univ_checkpoint.py"),
        str(checkpoint),
        "--json-out",
        str(report_path),
        "--max-keys",
        "2",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["selected_checkpoint_branch"] == "student"
    assert report["student_tensor_count"] == 3
    assert report["teacher_tensor_count"] == 1
    assert report["pos_embed_shape"] == [1, 5, 8]
    assert report["patch_embed_key_exists"] is True
    assert report["attention_qkv_key_exists"] is True
    assert report["nan_inf_check"]["has_nan"] is False
    assert report["sha256"]
