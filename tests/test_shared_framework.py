from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from forecasting.adapters import build_adapter
from forecasting.config import Config, load_config
from forecasting.data import (
    AutoregressiveTrajectoryDataset,
    DirectForecastDataset,
    autoregressive_collate,
    prepare_forecast_frame,
)
from forecasting.interface import ForecastOutput
from models.baselines.traisformer import TrAISformer
from scripts.compare import _forecast_errors
from scripts.plot_routes import representative_routes


def data_config():
    return SimpleNamespace(
        name="ct_dma",
        targets_are_normalized=True,
        cadence_seconds=300,
        lat_min=55.5,
        lat_max=58.0,
        lon_min=10.3,
        lon_max=13.0,
        sog_max=30.0,
    )


def tiny_traisformer_config(**overrides):
    values = {
        "name": "traisformer",
        "model_family": "traisformer",
        "dataset": "ct_dma",
        "seed": 123,
        "input_steps": 3,
        "output_steps": 2,
        "num_modes": 2,
        "batch_size": 2,
        "evaluation_batch_size": 2,
        "num_workers": 0,
        "normalize_input": False,
        "max_seqlen": 5,
        "training_sequence_len": 6,
        "crop_policy": "prefix",
        "lat_size": 5,
        "lon_size": 6,
        "sog_size": 3,
        "cog_size": 4,
        "n_lat_embd": 4,
        "n_lon_embd": 4,
        "n_sog_embd": 4,
        "n_cog_embd": 4,
        "mode": "pos",
        "sample_mode": "pos_vicinity",
        "top_k": 2,
        "r_vicinity": 4,
        "sample_predictions": True,
        "temperature": 1.0,
        "blur": True,
        "blur_learnable": False,
        "blur_loss_w": 1.0,
        "blur_n": 2,
        "n_head": 2,
        "n_layer": 1,
        "embd_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "learning_rate": 0.001,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "warmup_steps": 2,
    }
    values.update(overrides)
    return Config(values)


def raw_frame(length=8):
    frame = pd.DataFrame(
        [
            {
                "LAT": np.linspace(0.1, 0.2, length),
                "LON": np.linspace(0.3, 0.4, length),
                "SOG": np.full(length, 0.5),
                "COG": np.full(length, 0.25),
                "TIMESTAMP": np.arange(length) * 300,
                "VESSEL_MMSI": np.full(length, 100),
                "TRAJECTORY_ID": 200,
            }
        ]
    )
    frame.attrs["split"] = "valid"
    return frame


def test_all_canonical_models_use_registered_shared_adapters():
    assert build_adapter(load_config("traisformer", "model")).name == "traisformer"
    assert build_adapter(load_config("vessel_direct", "model")).name == "direct"
    assert build_adapter(load_config("vessel_direct_b3a", "model")).name == "direct"


def test_common_target_extraction_aligns_direct_and_autoregressive_views():
    raw = raw_frame()
    prepared = prepare_forecast_frame(raw, data_config(), 3, 2)
    direct = DirectForecastDataset(prepared, 3, 2)
    autoregressive = AutoregressiveTrajectoryDataset(
        prepared, tiny_traisformer_config(), data_config(), training=False
    )
    np.testing.assert_array_equal(direct.trajectory_ids, [200])
    assert autoregressive[0][2] == 200
    expected = np.stack(
        (raw.iloc[0]["LAT"][3:5], raw.iloc[0]["LON"][3:5]), axis=-1
    )
    np.testing.assert_allclose(direct.targets[0], expected)


def test_autoregressive_collate_masks_real_tokens_and_preserves_ids():
    first = (np.ones((3, 4), dtype=np.float32), 3, 10, 100, 1)
    second = (np.ones((2, 4), dtype=np.float32), 2, 20, 200, 2)
    sequence, mask, lengths, vessels, trajectories, starts = autoregressive_collate(
        [first, second]
    )
    assert sequence.shape == (2, 3, 4)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert lengths.tolist() == [3, 2]
    assert vessels.tolist() == [100, 200]
    assert trajectories.tolist() == [10, 20]
    assert starts.tolist() == [1, 2]


def test_traisformer_teacher_forcing_and_rollout_are_executable():
    config = tiny_traisformer_config(sample_predictions=False, num_modes=1)
    model = TrAISformer(config).eval()
    sequence = torch.rand(2, 4, 4) * 0.99
    mask = torch.tensor([[True] * 4, [True, True, True, False]])
    batch = (sequence, mask, None, None, None, None)
    loss = model.training_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    prediction = model.rollout(sequence[:, :3], output_steps=2)
    assert prediction.shape == (2, 2, 2)
    assert torch.all((prediction >= 0) & (prediction < 1))


def test_traisformer_config_fails_closed_on_repeated_deterministic_modes():
    config = tiny_traisformer_config(sample_predictions=False, num_modes=2)
    with pytest.raises(ValueError, match="Repeated deterministic"):
        build_adapter(config).validate(config, data_config())


def test_forecast_output_requires_aligned_normalized_probabilities():
    output = ForecastOutput(
        trajectories=np.zeros((2, 1, 3, 2)),
        probabilities=np.ones((2, 1)),
        clip_mask=np.zeros((2, 1, 3), dtype=bool),
        trajectory_ids=np.array([10, 20]),
    )
    output.validate(2, 3)
    output.probabilities[0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        output.validate(2, 3)


def test_comparison_can_use_probability_weighted_rollout_error():
    artifact = {
        "predictions": np.array([[[[0.0, 0.0]], [[0.04, 0.0]]]]),
        "probabilities": np.array([[0.25, 0.75]]),
        "targets": np.array([[[0.01, 0.0]]]),
    }
    expected = _forecast_errors(
        artifact, np.zeros(2), np.ones(2), "expected"
    )
    top1 = _forecast_errors(artifact, np.zeros(2), np.ones(2), "top1")
    assert expected.shape == top1.shape == (1, 1)
    assert expected[0, 0] < top1[0, 0]


def test_route_plot_uses_top_mode_or_probability_weighted_coordinates():
    artifact = {
        "predictions": np.array(
            [
                [
                    [[0.0, 0.2], [0.2, 0.4]],
                    [[0.4, 0.6], [0.6, 0.8]],
                ]
            ]
        ),
        "probabilities": np.array([[0.25, 0.75]]),
    }
    np.testing.assert_allclose(
        representative_routes(artifact, "top1"), artifact["predictions"][:, 1]
    )
    np.testing.assert_allclose(
        representative_routes(artifact, "expected"),
        np.array([[[0.3, 0.5], [0.5, 0.7]]]),
    )
