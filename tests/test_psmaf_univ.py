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
from psmaf_univ.univ_diagnostics import capture_module_outputs, inspect_features, model_inventory


class Encoder(nn.Module):
    def forward(self, image):
        return (image, nn.functional.avg_pool2d(image, 2), nn.functional.avg_pool2d(image, 4))


def test_model_emits_multiscale_features():
    model = PSMAFUNIVModel(Encoder(), [3, 3, 3], out_channels=8)
    output = model(torch.randn(2, 3, 32, 32))
    assert [item.shape for item in output] == [(2, 8, 32, 32), (2, 8, 16, 16), (2, 8, 8, 8)]


def test_checkpoint_branch_priority():
    tensor = torch.ones(1)
    assert extract_state_dict({"student": {"weight": tensor}})["weight"] is tensor


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
