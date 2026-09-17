"""Canonical trajectory loading, target extraction, and model datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import canonicalize_ct_dma, validate_canonical_trajectory_row


def load_raw_split(data_config, split):
    path = Path(data_config.preprocessed_folder) / getattr(data_config, f"{split}_file")
    frame = pd.read_parquet(path)
    if data_config.name == "ct_dma":
        return canonicalize_ct_dma(frame, split)
    frame = frame.copy()
    frame.attrs["split"] = split
    return frame


def prepare_forecast_frame(frame, data_config, input_steps, output_steps):
    """Extract one common LAT/LON target before truncating model history."""
    if frame.empty:
        raise ValueError("Cannot prepare an empty trajectory dataset.")
    if not bool(data_config.targets_are_normalized):
        raise ValueError(
            "Prepared raw-coordinate datasets need a normalization adapter before forecasting."
        )
    result = frame.copy()
    split = frame.attrs.get("split", "dataset")
    targets, masks = [], []
    for _, row in result.iterrows():
        length = validate_canonical_trajectory_row(
            row, split, data_config.cadence_seconds
        )
        target = np.zeros((output_steps, 2), dtype=np.float64)
        mask = np.zeros(output_steps, dtype=bool)
        target_length = max(0, min(output_steps, length - input_steps))
        if target_length:
            target[:target_length, 0] = np.asarray(row["LAT"])[
                input_steps : input_steps + target_length
            ]
            target[:target_length, 1] = np.asarray(row["LON"])[
                input_steps : input_steps + target_length
            ]
            mask[:target_length] = True
        valid = target[mask]
        if not np.isfinite(valid).all() or np.any((valid < -1e-12) | (valid > 1 + 1e-12)):
            raise ValueError(
                f"{split} trajectory {row['TRAJECTORY_ID']} has invalid normalized targets."
            )
        targets.append(target)
        masks.append(mask)
    result["target"] = targets
    result["mask"] = masks
    for column in ("LAT", "LON", "SOG", "COG", "TIMESTAMP", "VESSEL_MMSI"):
        result[column] = result[column].map(
            lambda values: np.asarray(values)[:input_steps]
        )
    result.attrs["split"] = split
    return result


class DirectForecastDataset(Dataset):
    """Dense shared evaluation target plus B0/B3A model inputs."""

    NORMALIZED_TOLERANCE = 1e-6
    CONTEXT_COLUMNS = ("LAT", "LON", "SOG", "COG")

    def __init__(self, frame, input_steps, output_steps, auxiliary_context=None):
        required = {
            *self.CONTEXT_COLUMNS,
            "target",
            "mask",
            "VESSEL_MMSI",
            "TRAJECTORY_ID",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Direct forecast columns missing: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Direct forecast dataset must not be empty.")
        input_steps = int(input_steps)
        output_steps = int(output_steps)
        if input_steps <= 0 or output_steps <= 0:
            raise ValueError("input_steps and output_steps must be positive.")
        self.context = np.stack(
            [
                np.stack(
                    [np.asarray(frame.iloc[row][name]) for name in self.CONTEXT_COLUMNS],
                    axis=-1,
                )
                for row in range(len(frame))
            ]
        ).astype(np.float32)
        self.targets = np.stack(frame["target"].values).astype(np.float32)
        raw_masks = np.stack(frame["mask"].values)
        expected_context = (len(frame), input_steps, len(self.CONTEXT_COLUMNS))
        expected_targets = (len(frame), output_steps, 2)
        expected_masks = (len(frame), output_steps)
        if self.context.shape != expected_context:
            raise ValueError(
                f"Direct forecast context shape {self.context.shape} does not match "
                f"{expected_context}."
            )
        if self.targets.shape != expected_targets:
            raise ValueError(
                f"Direct forecast target shape {self.targets.shape} does not match "
                f"{expected_targets}."
            )
        if raw_masks.shape != expected_masks:
            raise ValueError(
                f"Direct forecast mask shape {raw_masks.shape} does not match "
                f"{expected_masks}."
            )
        if not np.all((raw_masks == 0) | (raw_masks == 1)):
            raise ValueError("Direct forecast masks must be binary.")
        self.masks = raw_masks.astype(bool)
        if np.any(self.masks.sum(axis=1) == 0):
            raise ValueError("Every direct forecast example needs a valid future point.")
        if np.any(self.masks[:, 1:] & ~self.masks[:, :-1]):
            raise ValueError("Valid future entries must form one contiguous prefix.")
        vessel_ids = []
        for trajectory_id, values in zip(frame["TRAJECTORY_ID"], frame["VESSEL_MMSI"]):
            values = np.asarray(values)
            if not values.size:
                raise ValueError(
                    f"Trajectory {trajectory_id} has an empty VESSEL_MMSI array."
                )
            if np.any(values != values[0]):
                raise ValueError(
                    f"Trajectory {trajectory_id} contains mixed VESSEL_MMSI values."
                )
            vessel_ids.append(values[0])
        self.vessel_ids = np.asarray(vessel_ids, dtype=np.int64)
        self.trajectory_ids = frame["TRAJECTORY_ID"].to_numpy(dtype=np.int64)
        if len(np.unique(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("TRAJECTORY_ID values must be unique within the dataset.")
        valid_coordinates = np.concatenate(
            (self.context[..., :2].reshape(-1, 2), self.targets[self.masks]), axis=0
        )
        if not np.isfinite(self.context).all() or not np.isfinite(valid_coordinates).all():
            raise ValueError("Forecast inputs or targets contain non-finite values.")
        tolerance = self.NORMALIZED_TOLERANCE
        if np.any(
            (valid_coordinates < -tolerance) | (valid_coordinates > 1 + tolerance)
        ):
            raise ValueError(
                "Direct forecast LAT/LON values must be normalized to [0, 1]."
            )
        self.auxiliary_context = None
        if auxiliary_context is not None:
            auxiliary = np.asarray(auxiliary_context, dtype=np.float32)
            if auxiliary.ndim != 2 or auxiliary.shape[0] != len(frame):
                raise ValueError("Auxiliary context must have shape [N, features].")
            if auxiliary.shape[1] < 1:
                raise ValueError("Auxiliary context cannot be empty.")
            if not np.isfinite(auxiliary).all():
                raise ValueError("Auxiliary context contains non-finite values.")
            self.auxiliary_context = auxiliary

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        values = (
            torch.from_numpy(self.context[index]),
            torch.from_numpy(self.targets[index]),
            torch.from_numpy(self.masks[index]),
            torch.tensor(self.vessel_ids[index], dtype=torch.long),
            torch.tensor(self.trajectory_ids[index], dtype=torch.long),
        )
        if self.auxiliary_context is not None:
            values += (torch.from_numpy(self.auxiliary_context[index]),)
        return values


class AutoregressiveTrajectoryDataset(Dataset):
    """Four-channel sequence view used for TrAISformer teacher forcing/rollout."""

    def __init__(self, frame, config, data_config, *, training):
        required = {
            "LAT",
            "LON",
            "SOG",
            "COG",
            "TIMESTAMP",
            "VESSEL_MMSI",
            "TRAJECTORY_ID",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Canonical trajectory columns missing: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.split = frame.attrs.get("split", "dataset")
        self.config = config
        self.data_config = data_config
        self.training = bool(training)
        self.max_length = (
            int(config.training_sequence_len) if training else int(config.input_steps)
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        validate_canonical_trajectory_row(
            row, self.split, self.data_config.cadence_seconds
        )
        sequence = np.stack(
            [np.asarray(row[name]) for name in ("LAT", "LON", "SOG", "COG")], axis=-1
        ).astype(np.float32)
        if bool(self.config.normalize_input):
            sequence[:, 0] = (sequence[:, 0] - self.data_config.lat_min) / (
                self.data_config.lat_max - self.data_config.lat_min
            )
            sequence[:, 1] = (sequence[:, 1] - self.data_config.lon_min) / (
                self.data_config.lon_max - self.data_config.lon_min
            )
            sequence[:, 2] /= self.data_config.sog_max
            sequence[:, 3] /= 360.0
        elif np.any((sequence < -1e-12) | (sequence > 1 + 1e-12)):
            raise ValueError("TrAISformer input must be normalized to [0, 1].")
        sequence = np.clip(sequence, 0.0, 0.9999)
        if len(sequence) > self.max_length:
            if self.config.crop_policy != "prefix":
                raise ValueError("TrAISformer supports only deterministic prefix cropping.")
            sequence = sequence[: self.max_length]
        vessel = np.asarray(row["VESSEL_MMSI"])
        return (
            sequence,
            len(sequence),
            int(row["TRAJECTORY_ID"]),
            int(vessel[0]),
            int(np.asarray(row["TIMESTAMP"])[0]),
        )


def autoregressive_collate(batch):
    sequences, lengths, trajectory_ids, vessel_ids, start_times = zip(*batch)
    tensors = [torch.as_tensor(sequence, dtype=torch.float32) for sequence in sequences]
    padded = pad_sequence(tensors, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    mask = torch.arange(padded.shape[1])[None] < lengths[:, None]
    return (
        padded,
        mask,
        lengths,
        torch.tensor(vessel_ids, dtype=torch.long),
        torch.tensor(trajectory_ids, dtype=torch.long),
        torch.tensor(start_times, dtype=torch.long),
    )
