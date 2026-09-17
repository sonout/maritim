"""Strictly chronological same-vessel route context for B3A."""

from dataclasses import dataclass

import numpy as np


HISTORY_FEATURE_NAMES = (
    "east_3h_nmi",
    "north_3h_nmi",
    "east_6h_nmi",
    "north_6h_nmi",
    "match_distance_nmi",
    "log1p_pool_size",
    "support_flag",
)


def trajectory_timestamps(raw_by_id, trajectory_ids, input_steps, endpoint_step):
    """Extract forecast origins and one future endpoint from untruncated rows."""
    origins = np.empty(len(trajectory_ids), dtype=np.int64)
    endpoints = np.full(len(trajectory_ids), -1, dtype=np.int64)
    for row, trajectory_id in enumerate(np.asarray(trajectory_ids, dtype=np.int64)):
        timestamps = np.asarray(
            raw_by_id.loc[int(trajectory_id), "TIMESTAMP"], dtype=np.int64
        )
        origins[row] = timestamps[int(input_steps) - 1]
        index = int(input_steps) + int(endpoint_step)
        if index < len(timestamps):
            endpoints[row] = timestamps[index]
    return origins, endpoints


@dataclass
class VesselHistoryNormalization:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, raw_features):
        values = np.asarray(raw_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(HISTORY_FEATURE_NAMES):
            raise ValueError("Vessel-history features must have shape (N, 7).")
        supported = values[:, -1] == 1
        if not supported.any():
            raise ValueError("Cannot normalize vessel history without supported examples.")
        continuous = values[supported, :-1]
        mean = continuous.mean(axis=0)
        scale = continuous.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, raw_features):
        values = np.asarray(raw_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(HISTORY_FEATURE_NAMES):
            raise ValueError("Vessel-history features must have shape (N, 7).")
        if self.mean.shape != (6,) or self.scale.shape != (6,) or np.any(self.scale <= 0):
            raise ValueError("Vessel-history normalization must contain six valid scales.")
        support = values[:, -1]
        if np.any((support != 0) & (support != 1)):
            raise ValueError("Vessel-history support flag must be binary.")
        output = np.zeros_like(values, dtype=np.float32)
        selected = support == 1
        output[selected, :-1] = (
            (values[selected, :-1] - self.mean) / self.scale
        ).astype(np.float32)
        output[selected, -1] = 1.0
        return output


@dataclass
class PreparedVesselHistory:
    """Normalized B3A context plus leakage-audit information."""

    raw_features: np.ndarray
    features: np.ndarray
    normalization: VesselHistoryNormalization
    diagnostics: dict
    query_origin_timestamps: np.ndarray


def build_vessel_history_features(
    *,
    source_context,
    source_targets,
    source_masks,
    source_vessel_ids,
    source_trajectory_ids,
    source_endpoint_timestamps,
    query_context,
    query_vessel_ids,
    query_origin_timestamps,
    physical_scale,
    descriptor_steps=(35, 71),
    heading_sectors=8,
):
    """Build one same-vessel, same-sector, strictly historical route descriptor."""
    source_context = np.asarray(source_context)
    source_targets = np.asarray(source_targets)
    source_masks = np.asarray(source_masks, dtype=bool)
    query_context = np.asarray(query_context)
    source_vessel_ids = np.asarray(source_vessel_ids)
    source_trajectory_ids = np.asarray(source_trajectory_ids, dtype=np.int64)
    source_endpoint_timestamps = np.asarray(source_endpoint_timestamps, dtype=np.int64)
    query_vessel_ids = np.asarray(query_vessel_ids)
    query_origin_timestamps = np.asarray(query_origin_timestamps, dtype=np.int64)
    steps = np.asarray(descriptor_steps, dtype=np.int64)
    scale = np.asarray(physical_scale, dtype=np.float64)
    if steps.shape != (2,) or np.any(steps < 0) or np.any(steps >= source_targets.shape[1]):
        raise ValueError("Vessel-history descriptor needs valid 3 h and 6 h steps.")
    if source_endpoint_timestamps.shape != (len(source_context),):
        raise ValueError("Source endpoint timestamps must have shape (N,).")
    if query_origin_timestamps.shape != (len(query_context),):
        raise ValueError("Query origin timestamps must have shape (Q,).")
    if scale.shape != (2,) or np.any(scale <= 0):
        raise ValueError("Physical coordinate scale must contain positive N/E values.")
    if heading_sectors <= 0:
        raise ValueError("heading_sectors must be positive.")

    source_origins = source_context[:, -1, :2].astype(np.float64) * scale
    query_origins = query_context[:, -1, :2].astype(np.float64) * scale
    source_sector = (
        np.floor(source_context[:, -1, 3] * heading_sectors).astype(np.int64)
        % heading_sectors
    )
    query_sector = (
        np.floor(query_context[:, -1, 3] * heading_sectors).astype(np.int64)
        % heading_sectors
    )
    complete = source_masks[:, steps].all(axis=1)
    displacement = (
        source_targets[:, steps] - source_context[:, None, -1, :2]
    ) * scale[None, None]

    features = np.zeros((len(query_context), len(HISTORY_FEATURE_NAMES)), dtype=np.float64)
    match_ids = np.full(len(query_context), -1, dtype=np.int64)
    pool_sizes = np.zeros(len(query_context), dtype=np.int64)
    for row in range(len(query_context)):
        eligible = (
            complete
            & (source_endpoint_timestamps >= 0)
            & (source_endpoint_timestamps < query_origin_timestamps[row])
            & (source_vessel_ids == query_vessel_ids[row])
            & (source_sector == query_sector[row])
        )
        indices = np.flatnonzero(eligible)
        pool_sizes[row] = len(indices)
        if not len(indices):
            continue
        distances = np.linalg.norm(source_origins[indices] - query_origins[row], axis=1)
        order = np.lexsort((source_trajectory_ids[indices], distances))
        selected = indices[order[0]]
        # Physical displacement order is North/East. The B3 descriptor is E/N.
        features[row] = (
            displacement[selected, 0, 1],
            displacement[selected, 0, 0],
            displacement[selected, 1, 1],
            displacement[selected, 1, 0],
            distances[order[0]],
            np.log1p(len(indices)),
            1.0,
        )
        match_ids[row] = source_trajectory_ids[selected]
    diagnostics = {
        "support": features[:, -1].astype(bool),
        "match_trajectory_ids": match_ids,
        "pool_sizes": pool_sizes,
    }
    return features, diagnostics


def prepare_vessel_history_context(
    *,
    source_dataset,
    query_dataset,
    source_raw_by_id,
    query_raw_by_id,
    input_steps,
    physical_scale,
    normalization=None,
):
    """Build frozen B3A context with an explicit source/query boundary.

    ``source_dataset`` is the only permitted historical pool. Training calls
    this with train/train; validation and test call it with train/query, so a
    query split can never provide history to itself. A route is eligible only
    after its complete six-hour future has occurred before the query origin.
    """
    _, source_endpoint_timestamps = trajectory_timestamps(
        source_raw_by_id,
        source_dataset.trajectory_ids,
        input_steps,
        endpoint_step=71,
    )
    query_origin_timestamps, _ = trajectory_timestamps(
        query_raw_by_id,
        query_dataset.trajectory_ids,
        input_steps,
        endpoint_step=71,
    )
    raw_features, diagnostics = build_vessel_history_features(
        source_context=source_dataset.context,
        source_targets=source_dataset.targets,
        source_masks=source_dataset.masks,
        source_vessel_ids=source_dataset.vessel_ids,
        source_trajectory_ids=source_dataset.trajectory_ids,
        source_endpoint_timestamps=source_endpoint_timestamps,
        query_context=query_dataset.context,
        query_vessel_ids=query_dataset.vessel_ids,
        query_origin_timestamps=query_origin_timestamps,
        physical_scale=physical_scale,
    )
    if normalization is None:
        normalization = VesselHistoryNormalization.fit(raw_features)
    return PreparedVesselHistory(
        raw_features=raw_features,
        features=normalization.transform(raw_features),
        normalization=normalization,
        diagnostics=diagnostics,
        query_origin_timestamps=query_origin_timestamps,
    )


def chronological_support_quintiles(origin_timestamps, support):
    timestamps = np.asarray(origin_timestamps, dtype=np.int64)
    support = np.asarray(support, dtype=bool)
    if timestamps.shape != support.shape or timestamps.ndim != 1:
        raise ValueError("Timestamps and support must be aligned 1D arrays.")
    order = np.argsort(timestamps, kind="stable")
    output = []
    for quintile, indices in enumerate(np.array_split(order, 5), start=1):
        output.append(
            {
                "quintile": quintile,
                "count": int(len(indices)),
                "support_count": int(support[indices].sum()),
                "support_fraction": float(support[indices].mean()),
                "earliest_origin_timestamp": int(timestamps[indices].min()),
                "latest_origin_timestamp": int(timestamps[indices].max()),
            }
        )
    return output
