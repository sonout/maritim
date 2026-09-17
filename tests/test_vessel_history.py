import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from models.direct.model import DirectTrajectoryPredictor
from models.direct.vessel_history import (
    VesselHistoryNormalization,
    build_vessel_history_features,
    chronological_support_quintiles,
    prepare_vessel_history_context,
)


def history_inputs():
    source_context = np.zeros((3, 2, 4), dtype=np.float32)
    source_context[:, -1, :2] = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    source_context[:, -1, 3] = [0.0, 0.0, 0.25]
    source_targets = np.zeros((3, 4, 2), dtype=np.float32)
    source_targets[:, 1] = source_context[:, -1, :2] + [[0.01, 0.02]]
    source_targets[:, 3] = source_context[:, -1, :2] + [[0.03, 0.04]]
    return source_context, source_targets


def test_vessel_history_is_same_vessel_sector_and_strictly_historical():
    context, targets = history_inputs()
    features, diagnostics = build_vessel_history_features(
        source_context=context,
        source_targets=targets,
        source_masks=np.ones((3, 4), dtype=bool),
        source_vessel_ids=np.asarray([10, 20, 10]),
        source_trajectory_ids=np.asarray([1, 2, 3]),
        # Candidate 1 matches vessel/sector but its endpoint is in the future.
        source_endpoint_timestamps=np.asarray([110, 80, 70]),
        query_context=context[[0]],
        query_vessel_ids=np.asarray([10]),
        query_origin_timestamps=np.asarray([100]),
        physical_scale=np.asarray([100.0, 50.0]),
        descriptor_steps=(1, 3),
    )
    assert not diagnostics["support"][0]
    assert diagnostics["match_trajectory_ids"][0] == -1
    np.testing.assert_array_equal(features[0], np.zeros(7))


def test_vessel_history_descriptor_uses_relative_east_north_endpoints():
    context, targets = history_inputs()
    query = context[[0]].copy()
    query[:, -1, :2] = [0.21, 0.2]
    features, diagnostics = build_vessel_history_features(
        source_context=context,
        source_targets=targets,
        source_masks=np.ones((3, 4), dtype=bool),
        source_vessel_ids=np.asarray([10, 20, 10]),
        source_trajectory_ids=np.asarray([1, 2, 3]),
        source_endpoint_timestamps=np.asarray([50, 50, 50]),
        query_context=query,
        query_vessel_ids=np.asarray([20]),
        query_origin_timestamps=np.asarray([100]),
        physical_scale=np.asarray([100.0, 50.0]),
        descriptor_steps=(1, 3),
    )
    assert diagnostics["match_trajectory_ids"][0] == 2
    np.testing.assert_allclose(features[0, :4], [1.0, 1.0, 2.0, 3.0], atol=1e-6)
    assert features[0, -1] == 1


def test_vessel_history_selects_nearest_origin_within_same_mmsi_sector():
    context, targets = history_inputs()
    query = context[[0]].copy()
    query[:, -1, :2] = [0.21, 0.21]
    _, diagnostics = build_vessel_history_features(
        source_context=context,
        source_targets=targets,
        source_masks=np.ones((3, 4), dtype=bool),
        source_vessel_ids=np.asarray([10, 10, 10]),
        source_trajectory_ids=np.asarray([1, 2, 3]),
        source_endpoint_timestamps=np.asarray([50, 50, 50]),
        query_context=query,
        query_vessel_ids=np.asarray([10]),
        query_origin_timestamps=np.asarray([100]),
        physical_scale=np.asarray([100.0, 50.0]),
        descriptor_steps=(1, 3),
    )
    assert diagnostics["match_trajectory_ids"][0] == 2


def test_query_split_cannot_supply_its_own_vessel_history():
    source_context = np.zeros((1, 2, 4), dtype=np.float32)
    source_context[0, -1] = [0.2, 0.2, 0.0, 0.0]
    source_targets = np.repeat(source_context[:, -1:, :2], 72, axis=1)
    source_targets[:, 35] += [0.01, 0.02]
    source_targets[:, 71] += [0.03, 0.04]
    query_context = source_context.copy()
    query_context[0, -1, :2] = [0.21, 0.21]
    source = SimpleNamespace(
        context=source_context,
        targets=source_targets,
        masks=np.ones((1, 72), dtype=bool),
        vessel_ids=np.asarray([10]),
        trajectory_ids=np.asarray([1]),
    )
    query = SimpleNamespace(
        context=query_context,
        targets=source_targets.copy(),
        masks=np.ones((1, 72), dtype=bool),
        vessel_ids=np.asarray([10]),
        trajectory_ids=np.asarray([2]),
    )
    source_raw = pd.DataFrame(
        {"TRAJECTORY_ID": [1], "TIMESTAMP": [np.arange(74, dtype=np.int64)]}
    ).set_index("TRAJECTORY_ID")
    query_raw = pd.DataFrame(
        {"TRAJECTORY_ID": [2], "TIMESTAMP": [np.arange(100, 174, dtype=np.int64)]}
    ).set_index("TRAJECTORY_ID")
    prepared = prepare_vessel_history_context(
        source_dataset=source,
        query_dataset=query,
        source_raw_by_id=source_raw,
        query_raw_by_id=query_raw,
        input_steps=2,
        physical_scale=np.asarray([100.0, 50.0]),
    )
    assert prepared.diagnostics["match_trajectory_ids"][0] == 1
    assert 2 not in prepared.diagnostics["match_trajectory_ids"]


def test_vessel_history_normalization_keeps_unsupported_rows_zero():
    raw = np.asarray(
        [
            [1, 2, 3, 4, 5, np.log1p(2), 1],
            [2, 4, 6, 8, 10, np.log1p(3), 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    normalization = VesselHistoryNormalization.fit(raw)
    transformed = normalization.transform(raw)
    np.testing.assert_array_equal(transformed[-1], np.zeros(7))
    assert transformed[0, -1] == transformed[1, -1] == 1


def test_vessel_history_support_quintiles_preserve_chronology():
    result = chronological_support_quintiles(
        np.arange(10, dtype=np.int64),
        np.asarray([0, 0, 1, 0, 1, 0, 1, 1, 1, 1], dtype=bool),
    )
    assert [item["count"] for item in result] == [2] * 5
    assert result[0]["support_fraction"] == 0.0
    assert result[-1]["support_fraction"] == 1.0


def test_b3_model_requires_seven_history_values_and_is_deterministic():
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
    second = model(context, auxiliary).trajectories
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 1, 4, 2)
    with pytest.raises(ValueError, match="Expected vessel-history"):
        model(context)
