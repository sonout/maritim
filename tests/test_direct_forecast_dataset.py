import numpy as np
import pandas as pd
import pytest

from forecasting.data import DirectForecastDataset


def forecast_frame(rows=2, input_steps=3, output_steps=4):
    values = []
    for index in range(rows):
        values.append(
            {
                "LAT": np.linspace(0.1, 0.2, input_steps),
                "LON": np.linspace(0.3, 0.4, input_steps),
                "SOG": np.full(input_steps, 0.5),
                "COG": np.full(input_steps, 0.25),
                "target": np.full((output_steps, 2), 0.5),
                "mask": np.ones(output_steps),
                "VESSEL_MMSI": np.full(input_steps, 1000 + index),
                "TRAJECTORY_ID": 2000 + index,
            }
        )
    return pd.DataFrame(values)


def build(frame):
    return DirectForecastDataset(frame, input_steps=3, output_steps=4)


def test_direct_dataset_shapes():
    dataset = build(forecast_frame())
    assert dataset.context.shape == (2, 3, 4)
    assert dataset.targets.shape == (2, 4, 2)
    assert dataset.masks.shape == (2, 4)
    assert dataset.masks.dtype == np.bool_


def test_direct_dataset_returns_optional_auxiliary_context():
    auxiliary = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dataset = DirectForecastDataset(
        forecast_frame(), input_steps=3, output_steps=4,
        auxiliary_context=auxiliary,
    )
    assert len(dataset[0]) == 6
    np.testing.assert_allclose(dataset[1][-1].numpy(), auxiliary[1])


def test_direct_dataset_rejects_nonbinary_mask():
    frame = forecast_frame()
    frame.at[0, "mask"] = np.array([1, 0.5, 0, 0])
    with pytest.raises(ValueError, match="binary"):
        build(frame)


def test_direct_dataset_rejects_noncontiguous_mask():
    frame = forecast_frame()
    frame.at[0, "mask"] = np.array([1, 1, 0, 1])
    with pytest.raises(ValueError, match="contiguous prefix"):
        build(frame)


def test_direct_dataset_rejects_zero_future():
    frame = forecast_frame()
    frame.at[0, "mask"] = np.zeros(4)
    with pytest.raises(ValueError, match="valid future"):
        build(frame)


def test_direct_dataset_rejects_duplicate_trajectory_ids():
    frame = forecast_frame()
    frame.loc[1, "TRAJECTORY_ID"] = frame.loc[0, "TRAJECTORY_ID"]
    with pytest.raises(ValueError, match="unique"):
        build(frame)


def test_direct_dataset_rejects_mixed_mmsi():
    frame = forecast_frame()
    frame.at[0, "VESSEL_MMSI"] = np.array([1000, 1001, 1000])
    with pytest.raises(ValueError, match="mixed"):
        build(frame)


def test_direct_dataset_rejects_empty_mmsi():
    frame = forecast_frame()
    frame.at[0, "VESSEL_MMSI"] = np.array([])
    with pytest.raises(ValueError, match="empty"):
        build(frame)


def test_direct_dataset_rejects_out_of_range_coordinates():
    frame = forecast_frame()
    frame.at[0, "LAT"] = np.array([0.1, 1.1, 0.2])
    with pytest.raises(ValueError, match="normalized"):
        build(frame)
