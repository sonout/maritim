"""Shared configuration, schema, and reproducibility utilities."""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT_DIR = Path(__file__).resolve().parents[1]

CT_DMA_LEGACY_TO_CANONICAL = {
    "TIMESTAMP": "HEADING",
    "A": "ROT",
    "B": "TIMESTAMP",
    "C": "VESSEL_MMSI",
    "MMSI": "TRAJECTORY_ID",
}


def seed_everything(seed=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return seed


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.__dict__.copy()


def load_config(name: str, config_type: str) -> Config:
    path = ROOT_DIR / "configs" / config_type / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    values: dict[str, Any] = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return Config(values)


def canonicalize_ct_dma(data: pd.DataFrame, split: str) -> pd.DataFrame:
    """Map the checked-in legacy parquet labels to semantic column names."""
    legacy = set(CT_DMA_LEGACY_TO_CANONICAL)
    canonical = set(CT_DMA_LEGACY_TO_CANONICAL.values())
    columns = set(data.columns)
    if legacy <= columns:
        result = data.rename(columns=CT_DMA_LEGACY_TO_CANONICAL).copy()
    elif canonical <= columns:
        result = data.copy()
    else:
        raise ValueError(
            f"{split}: neither legacy nor canonical ct_dma schema is complete."
        )
    result.attrs["split"] = split
    return result


def validate_canonical_trajectory_row(row, split, cadence_seconds):
    """Validate aligned canonical arrays and return their common length."""
    channels = ("LAT", "LON", "SOG", "COG", "TIMESTAMP", "VESSEL_MMSI")
    trajectory_id = row["TRAJECTORY_ID"]
    arrays = {name: np.asarray(row[name]) for name in channels}
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values()), 0) == 0:
        raise ValueError(
            f"{split} trajectory {trajectory_id}: channels must be aligned and non-empty; "
            f"got {lengths}."
        )
    for name, values in arrays.items():
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            raise ValueError(
                f"{split} trajectory {trajectory_id}: {name} contains non-finite values."
            )
    deltas = np.diff(arrays["TIMESTAMP"])
    if deltas.size and np.any(deltas != cadence_seconds):
        raise ValueError(
            f"{split} trajectory {trajectory_id}: timestamps do not match "
            f"{cadence_seconds}-second cadence."
        )
    vessel = arrays["VESSEL_MMSI"]
    if np.any(vessel != vessel[0]):
        raise ValueError(
            f"{split} trajectory {trajectory_id}: VESSEL_MMSI changes within row."
        )
    return next(iter(lengths.values()))
