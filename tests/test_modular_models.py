from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.metrics import evaluate_forecast_samples, evaluate_weighted_modes
from forecasting.adapters import build_adapter
from forecasting.config import load_config
from forecasting.interface import ForecastOutput
from forecasting.runner import save_prediction_artifact
from models.direct.coordinates import (
    bound_physical_displacement,
    integrate_normalized_displacements,
    normalized_to_physical_displacement,
    physical_coordinate_scale,
    physical_to_normalized_displacement,
)
from models.direct.features import ContinuousAISFeatures
from models.direct.losses import masked_trajectory_loss
from models.direct.model import DirectTrajectoryPredictor


def data_config():
    return SimpleNamespace(
        lat_min=55.5,
        lat_max=58.0,
        lon_min=10.3,
        lon_max=13.0,
        cadence_seconds=300,
    )


def test_continuous_features_are_local_global_and_angle_safe():
    context = torch.tensor(
        [[[0.4, 0.6, 0.5, 359 / 360], [0.41, 0.62, 0.5, 1 / 360]]]
    )
    features = ContinuousAISFeatures(delta_scale=10.0)(context)
    assert features.shape == (1, 2, 9)
    assert features[0, 1, 0].item() == pytest.approx(0.24)
    assert features[0, 1, 1].item() == pytest.approx(-0.18)
    assert features[0, 1, 2].item() == pytest.approx(0.2)
    assert features[0, 1, 3].item() == pytest.approx(0.1)
    assert torch.linalg.vector_norm(features[0, 0, 7:9] - features[0, 1, 7:9]) < 0.04


def test_b0_model_construction_and_forward():
    model = DirectTrajectoryPredictor(
        input_steps=3,
        output_steps=4,
        hidden_dim=8,
        encoder_layers=1,
        attention_heads=1,
        dropout=0.0,
    ).eval()
    result = model(torch.rand(2, 3, 4))
    assert result.trajectories.shape == (2, 1, 4, 2)
    assert result.logits.shape == (2, 1)
    assert torch.equal(result.probabilities, torch.ones(2, 1))
    assert not any("queries" in name or "score_head" in name for name, _ in model.named_parameters())


def test_b3a_model_construction_and_forward():
    model = DirectTrajectoryPredictor(
        input_steps=3,
        output_steps=4,
        hidden_dim=8,
        encoder_layers=1,
        attention_heads=1,
        dropout=0.0,
        vessel_history_dim=7,
        vessel_history_hidden_dim=4,
    ).eval()
    context = torch.rand(2, 3, 4)
    auxiliary = torch.zeros(2, 7)
    first = model(context, auxiliary).trajectories
    torch.testing.assert_close(first, model(context, auxiliary).trajectories)
    with pytest.raises(ValueError, match="Expected vessel-history"):
        model(context)


def test_canonical_configs_parse_and_instantiate():
    for name, history_dim in (("vessel_direct", 0), ("vessel_direct_b3a", 7)):
        config = load_config(name, "model")
        adapter = build_adapter(config)
        model = adapter.build_model(config, data_config())
        assert model.vessel_history_dim == history_dim


def test_config_validation_rejects_noncanonical_designs():
    config = load_config("vessel_direct", "model")
    config.num_modes = 2
    with pytest.raises(ValueError, match="deterministic"):
        build_adapter(config).validate(config, data_config())


def test_physical_decoder_is_isotropic_bounded_and_zero_safe():
    raw = torch.tensor([[20.0, 0.0], [0.0, 20.0], [0.0, 0.0]])
    bounded = bound_physical_displacement(raw, 2.5)
    assert torch.allclose(
        torch.linalg.vector_norm(bounded[:2], dim=-1), torch.full((2,), 2.5), atol=1e-6
    )
    assert torch.equal(bounded[2], torch.zeros(2))
    assert torch.all(torch.linalg.vector_norm(bounded, dim=-1) <= 2.5 + 1e-6)


def test_physical_normalized_roundtrip_and_integration():
    scale = physical_coordinate_scale(data_config())
    physical = np.array([[1.2, -0.7], [0.0, 2.5]])
    normalized = physical_to_normalized_displacement(physical, scale)
    np.testing.assert_allclose(
        normalized_to_physical_displacement(normalized, scale), physical
    )
    result = integrate_normalized_displacements(
        torch.tensor([[0.2, 0.3]]),
        torch.tensor([[[0.1, -0.1], [0.2, 0.05]]]),
    )
    torch.testing.assert_close(
        result, torch.tensor([[[0.3, 0.2], [0.5, 0.25]]])
    )


def test_masked_trajectory_loss_ignores_padding():
    model = DirectTrajectoryPredictor(
        input_steps=2,
        output_steps=2,
        hidden_dim=8,
        encoder_layers=1,
        attention_heads=1,
        dropout=0.0,
    )
    prediction = model(torch.rand(1, 2, 4))
    targets = prediction.trajectories[:, 0].detach().clone()
    targets[:, 1] = 1.0 - targets[:, 1]
    result = masked_trajectory_loss(
        prediction, targets, torch.tensor([[True, False]]), [1.0, 1.0]
    )
    assert result["loss"].item() == pytest.approx(0.0)


def test_cumulative_ade_uses_horizon_eligible_population():
    targets = np.array(
        [[[56.0, 12.0], [56.0, 12.0]], [[56.0, 12.0], [0.0, 0.0]]]
    )
    predictions = np.array(
        [[[[56.01, 12.0], [56.02, 12.0]], [[57.0, 12.0], [57.0, 12.0]]]]
    )
    masks = np.array([[1, 1], [1, 0]], dtype=bool)
    result = evaluate_forecast_samples(predictions, targets, masks, cadence_seconds=300)
    first = evaluate_forecast_samples(
        predictions[:, :1], targets[:1], masks[:1], cadence_seconds=300
    )
    assert result["ade_cumulative_nmi"][1] == pytest.approx(
        first["ade_cumulative_nmi"][1]
    )


def test_deterministic_energy_and_clipping_metrics():
    targets = np.array([[[56.0, 12.0], [56.1, 12.0]]])
    predictions = np.array([[[[56.05, 12.0], [56.2, 12.0]]]])
    result = evaluate_weighted_modes(
        predictions,
        np.ones((1, 1)),
        targets,
        np.ones((1, 2)),
        cadence_seconds=300,
        clip_masks=np.array([[[False, True]]]),
    )
    assert result["trajectory_energy_score_nmi"] == pytest.approx(
        result["min_ade_at_k_nmi"]
    )
    assert result["predicted_point_clipping_fraction"] == pytest.approx(0.5)


def test_saved_prediction_artifact_contains_vessel_ids(tmp_path):
    dataset = SimpleNamespace(
        targets=np.zeros((2, 1, 2)),
        masks=np.ones((2, 1), dtype=bool),
        trajectory_ids=np.array([10, 20]),
        vessel_ids=np.array([100, 200]),
    )
    path = tmp_path / "predictions.npz"
    output = ForecastOutput(
        trajectories=np.zeros((2, 1, 1, 2)),
        probabilities=np.ones((2, 1)),
        clip_mask=np.zeros((2, 1, 1), dtype=bool),
        trajectory_ids=np.array([10, 20]),
    )
    save_prediction_artifact(path, output, dataset, "direct")
    with np.load(path) as artifact:
        np.testing.assert_array_equal(artifact["vessel_ids"], dataset.vessel_ids)
