"""Publication-oriented sample-forecast metrics in nautical miles."""

import numpy as np


KM_PER_NMI = 1.852


def haversine_nmi(actual_deg, predicted_deg):
    """Vectorized great-circle distance for arrays ending in (lat, lon)."""
    actual = np.deg2rad(np.asarray(actual_deg, dtype=np.float64))
    predicted = np.deg2rad(np.asarray(predicted_deg, dtype=np.float64))
    lat_error = predicted[..., 0] - actual[..., 0]
    lon_error = predicted[..., 1] - actual[..., 1]
    a = (
        np.sin(lat_error / 2.0) ** 2
        + np.cos(actual[..., 0])
        * np.cos(predicted[..., 0])
        * np.sin(lon_error / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)) / KM_PER_NMI


def _validate(predictions_deg, targets_deg, masks):
    predictions = np.asarray(predictions_deg, dtype=np.float64)
    targets = np.asarray(targets_deg, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape (samples, trajectories, steps, 2).")
    if targets.shape != predictions.shape[1:]:
        raise ValueError(
            f"Target shape {targets.shape} does not match {predictions.shape[1:]}."
        )
    if masks.shape != predictions.shape[1:3]:
        raise ValueError(f"Mask shape {masks.shape} does not match {predictions.shape[1:3]}.")
    if predictions.shape[0] < 1 or not np.isfinite(predictions).all():
        raise ValueError("Forecast samples must be non-empty and finite.")
    if not np.isfinite(targets[masks]).all():
        raise ValueError("Valid targets must be finite.")
    return predictions, targets, masks


def point_errors_nmi(predictions_deg, targets_deg):
    predictions = np.asarray(predictions_deg, dtype=np.float64)
    targets = np.asarray(targets_deg, dtype=np.float64)
    return haversine_nmi(targets[None, ...], predictions)


def energy_score_by_horizon(predictions_deg, targets_deg, masks):
    """Multivariate energy score from forecast samples at every horizon."""
    predictions, targets, masks = _validate(predictions_deg, targets_deg, masks)
    target_distance = point_errors_nmi(predictions, targets).mean(axis=0)
    scores = np.full(predictions.shape[2], np.nan, dtype=np.float64)
    spread = np.full_like(scores, np.nan)
    for horizon in range(predictions.shape[2]):
        valid = masks[:, horizon]
        if not valid.any():
            continue
        samples = predictions[:, valid, horizon, :]
        pair_distance = haversine_nmi(
            samples[:, None, :, :], samples[None, :, :, :]
        )
        mean_pair_distance = pair_distance.mean(axis=(0, 1))
        spread[horizon] = mean_pair_distance.mean()
        scores[horizon] = (
            target_distance[valid, horizon] - 0.5 * mean_pair_distance
        ).mean()
    return scores, spread


def central_interval_coverage(predictions_deg, targets_deg, masks, levels):
    """Empirical marginal and joint rectangular sample-interval coverage."""
    predictions, targets, masks = _validate(predictions_deg, targets_deg, masks)
    output = {}
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError("Coverage levels must lie strictly between zero and one.")
        tail = (1.0 - level) / 2.0
        lower = np.quantile(predictions, tail, axis=0)
        upper = np.quantile(predictions, 1.0 - tail, axis=0)
        inside = (targets >= lower) & (targets <= upper)
        lat = np.full(predictions.shape[2], np.nan)
        lon = np.full(predictions.shape[2], np.nan)
        joint = np.full(predictions.shape[2], np.nan)
        for horizon in range(predictions.shape[2]):
            valid = masks[:, horizon]
            if valid.any():
                lat[horizon] = inside[valid, horizon, 0].mean()
                lon[horizon] = inside[valid, horizon, 1].mean()
                joint[horizon] = inside[valid, horizon].all(axis=1).mean()
        output[str(level)] = {"latitude": lat, "longitude": lon, "joint": joint}
    return output


def evaluate_forecast_samples(
    predictions_deg,
    targets_deg,
    masks,
    *,
    cadence_seconds,
    coverage_levels=(0.5, 0.8, 0.9),
):
    """Return full curves and honest horizon summaries for an ensemble forecast."""
    predictions, targets, masks = _validate(predictions_deg, targets_deg, masks)
    distances = point_errors_nmi(predictions, targets)
    sample_mean_errors = distances.mean(axis=0)
    valid_counts = masks.sum(axis=0).astype(np.int64)
    positive = valid_counts > 0
    mean_error = np.full(predictions.shape[2], np.nan)
    p50 = np.full_like(mean_error, np.nan)
    p90 = np.full_like(mean_error, np.nan)
    oracle = np.full_like(mean_error, np.nan)
    for horizon in np.flatnonzero(positive):
        valid = masks[:, horizon]
        values = sample_mean_errors[valid, horizon]
        mean_error[horizon] = values.mean()
        p50[horizon] = np.quantile(values, 0.5)
        p90[horizon] = np.quantile(values, 0.9)
        oracle[horizon] = distances[:, valid, horizon].min(axis=0).mean()

    cumulative_ade = np.full_like(mean_error, np.nan)
    for horizon in range(predictions.shape[2]):
        eligible = masks[:, horizon]
        if eligible.any():
            prefix_mask = masks[eligible, : horizon + 1]
            prefix_errors = sample_mean_errors[eligible, : horizon + 1]
            per_trajectory = np.where(prefix_mask, prefix_errors, 0.0).sum(axis=1)
            per_trajectory /= prefix_mask.sum(axis=1)
            cumulative_ade[horizon] = per_trajectory.mean()

    energy, spread = energy_score_by_horizon(predictions, targets, masks)
    coverage = central_interval_coverage(
        predictions, targets, masks, coverage_levels
    ) if predictions.shape[0] > 1 else {}
    steps = np.arange(1, predictions.shape[2] + 1, dtype=np.int64)
    return {
        "horizon_steps": steps,
        "horizon_seconds": steps * int(cadence_seconds),
        "valid_counts": valid_counts,
        "fde_mean_nmi": mean_error,
        "ade_cumulative_nmi": cumulative_ade,
        "trajectory_error_p50_nmi": p50,
        "trajectory_error_p90_nmi": p90,
        "best_of_n_pointwise_nmi": oracle,
        "energy_score_nmi": energy,
        "ensemble_pairwise_spread_nmi": spread,
        "coverage": coverage,
        "sample_count": int(predictions.shape[0]),
    }


def evaluate_weighted_modes(
    predictions_deg,
    probabilities,
    targets_deg,
    masks,
    *,
    cadence_seconds,
    clip_masks=None,
):
    """Evaluate explicit trajectory modes without pretending they are iid samples.

    ``predictions_deg`` is mode-major ``(K, N, H, 2)`` and probabilities are
    trajectory-major ``(N, K)``. The oracle assigns a complete trajectory using
    masked ADE; top-1 uses only information available at deployment.
    """
    predictions, targets, masks = _validate(predictions_deg, targets_deg, masks)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected_probability_shape = (predictions.shape[1], predictions.shape[0])
    if probabilities.shape != expected_probability_shape:
        raise ValueError(
            f"Probability shape {probabilities.shape} does not match "
            f"{expected_probability_shape}."
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Mode probabilities must be finite and non-negative.")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Mode probabilities must sum to one for every trajectory.")

    mode_count, trajectory_count, horizon_count = predictions.shape[:3]
    distances = point_errors_nmi(predictions, targets)
    counts = masks.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every trajectory needs at least one valid horizon.")
    mode_ade = np.where(masks[None], distances, 0.0).sum(axis=2) / counts[None]
    oracle_mode = mode_ade.argmin(axis=0)
    top_mode = probabilities.argmax(axis=1)
    trajectory_index = np.arange(trajectory_count)
    top_distances = distances[top_mode, trajectory_index]
    oracle_distances = distances[oracle_mode, trajectory_index]

    valid_counts = masks.sum(axis=0).astype(np.int64)
    top1_fde = np.full(horizon_count, np.nan)
    top1_p50 = np.full_like(top1_fde, np.nan)
    top1_p90 = np.full_like(top1_fde, np.nan)
    expected_fde = np.full_like(top1_fde, np.nan)
    min_fde = np.full_like(top1_fde, np.nan)
    complete_oracle_fde = np.full_like(top1_fde, np.nan)
    top1_ade = np.full_like(top1_fde, np.nan)
    weighted_energy = np.full_like(top1_fde, np.nan)
    weighted_spread = np.full_like(top1_fde, np.nan)
    regret_mean = np.full_like(top1_fde, np.nan)
    regret_median = np.full_like(top1_fde, np.nan)
    regret_p90 = np.full_like(top1_fde, np.nan)

    ranked_modes = np.argsort(-probabilities, axis=1, kind="stable")
    top_m_values = []
    for candidate in (1, 2, 4, 8, mode_count):
        if candidate <= mode_count and candidate not in top_m_values:
            top_m_values.append(candidate)
    top_m_fde = {m: np.full(horizon_count, np.nan) for m in top_m_values}
    top_m_ade = {}
    top_m_point_errors = {}
    for top_m in top_m_values:
        selected = ranked_modes[:, :top_m].T
        selected_distances = distances[selected, trajectory_index[None, :]]
        top_m_point_errors[top_m] = selected_distances.min(axis=0)
        selected_ade = mode_ade[selected, trajectory_index[None, :]]
        top_m_ade[top_m] = float(selected_ade.min(axis=0).mean())

    for horizon in range(horizon_count):
        valid = masks[:, horizon]
        if not valid.any():
            continue
        top_values = top_distances[valid, horizon]
        top1_fde[horizon] = top_values.mean()
        top1_p50[horizon] = np.quantile(top_values, 0.5)
        top1_p90[horizon] = np.quantile(top_values, 0.9)
        weighted = (distances[:, valid, horizon].T * probabilities[valid]).sum(axis=1)
        expected_fde[horizon] = weighted.mean()
        pointwise_best = distances[:, valid, horizon].min(axis=0)
        min_fde[horizon] = pointwise_best.mean()
        complete_oracle_fde[horizon] = oracle_distances[valid, horizon].mean()

        prefix_mask = masks[valid, : horizon + 1]
        prefix_values = top_distances[valid, : horizon + 1]
        per_trajectory_ade = np.where(prefix_mask, prefix_values, 0.0).sum(axis=1)
        per_trajectory_ade /= prefix_mask.sum(axis=1)
        top1_ade[horizon] = per_trajectory_ade.mean()

        regret = np.maximum(top_values - pointwise_best, 0.0)
        regret_mean[horizon] = regret.mean()
        regret_median[horizon] = np.quantile(regret, 0.5)
        regret_p90[horizon] = np.quantile(regret, 0.9)
        for top_m in top_m_values:
            top_m_fde[top_m][horizon] = top_m_point_errors[top_m][valid, horizon].mean()

        horizon_modes = predictions[:, valid, horizon]
        pair_distance = haversine_nmi(horizon_modes[:, None], horizon_modes[None])
        valid_probabilities = probabilities[valid].T
        pair_weights = valid_probabilities[:, None, :] * valid_probabilities[None, :, :]
        per_trajectory_spread = (pair_distance * pair_weights).sum(axis=(0, 1))
        weighted_spread[horizon] = per_trajectory_spread.mean()
        weighted_energy[horizon] = (weighted - 0.5 * per_trajectory_spread).mean()

    coverage = _weighted_mode_coverage(
        predictions, probabilities, targets, masks, levels=(0.5, 0.8, 0.9)
    )

    winner_one_hot = np.eye(mode_count)[oracle_mode]
    brier = np.square(probabilities - winner_one_hot).sum(axis=1).mean()
    top1_accuracy = (top_mode == oracle_mode).mean()
    oracle_histogram = np.bincount(oracle_mode, minlength=mode_count).astype(np.int64)
    top1_histogram = np.bincount(top_mode, minlength=mode_count).astype(np.int64)
    oracle_frequency = oracle_histogram / oracle_histogram.sum()
    positive_frequency = oracle_frequency[oracle_frequency > 0]
    effective_oracle_query_count = np.exp(
        -(positive_frequency * np.log(positive_frequency)).sum()
    )
    probability_entropy = -(
        probabilities * np.log(probabilities.clip(min=1e-300))
    ).sum(axis=1)
    probability_effective_modes = np.exp(probability_entropy)

    confidence = probabilities.max(axis=1)
    calibration_bins = []
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        high = low + 0.1
        selected = (confidence >= low) & (
            (confidence <= high) if high >= 1.0 else (confidence < high)
        )
        if not selected.any():
            continue
        accuracy = (top_mode[selected] == oracle_mode[selected]).mean()
        mean_confidence = confidence[selected].mean()
        fraction = selected.mean()
        ece += fraction * abs(accuracy - mean_confidence)
        calibration_bins.append(
            {
                "lower": float(low),
                "upper": float(high),
                "count": int(selected.sum()),
                "accuracy": float(accuracy),
                "confidence": float(mean_confidence),
            }
        )

    output = {
        "horizon_steps": np.arange(1, horizon_count + 1, dtype=np.int64),
        "horizon_seconds": np.arange(1, horizon_count + 1) * int(cadence_seconds),
        "valid_counts": valid_counts,
        "top1_fde_nmi": top1_fde,
        "top1_trajectory_error_p50_nmi": top1_p50,
        "top1_trajectory_error_p90_nmi": top1_p90,
        "top1_ade_cumulative_nmi": top1_ade,
        "probability_weighted_fde_nmi": expected_fde,
        "weighted_energy_score_nmi": weighted_energy,
        "weighted_pairwise_spread_nmi": weighted_spread,
        "trajectory_energy_score_nmi": _trajectory_energy_score(
            predictions, probabilities, masks, mode_ade
        ),
        "weighted_coverage": coverage,
        "min_fde_at_k_nmi": min_fde,
        "complete_oracle_fde_at_k_nmi": complete_oracle_fde,
        "min_ade_at_k_nmi": float(mode_ade.min(axis=0).mean()),
        "mean_ranking_regret_nmi": regret_mean,
        "median_ranking_regret_nmi": regret_median,
        "p90_ranking_regret_nmi": regret_p90,
        "top1_mode_accuracy": float(top1_accuracy),
        "winner_selection_brier_score": float(brier),
        "winner_selection_ece": float(ece),
        "winner_selection_calibration_bins": calibration_bins,
        "mean_top1_probability": float(confidence.mean()),
        "mean_probability_entropy": float(probability_entropy.mean()),
        "mean_probability_effective_modes": float(probability_effective_modes.mean()),
        "median_probability_effective_modes": float(
            np.median(probability_effective_modes)
        ),
        "oracle_mode_histogram": oracle_histogram,
        "top1_mode_histogram": top1_histogram,
        "effective_oracle_query_count": float(effective_oracle_query_count),
        "oracle_mode": oracle_mode,
        "top1_mode": top_mode,
        "mode_count": int(mode_count),
        "top_m_values": np.asarray(top_m_values, dtype=np.int64),
        "min_fde_at_top_m_nmi": {str(m): top_m_fde[m] for m in top_m_values},
        "min_ade_at_top_m_nmi": {str(m): top_m_ade[m] for m in top_m_values},
    }
    for top_m in top_m_values:
        output[f"min_fde_at_top{top_m}_nmi"] = top_m_fde[top_m]
        output[f"min_ade_at_top{top_m}_nmi"] = top_m_ade[top_m]

    if clip_masks is not None:
        clip_masks = np.asarray(clip_masks, dtype=bool)
        if clip_masks.shape != (mode_count, trajectory_count, horizon_count):
            raise ValueError(
                f"Clip mask shape {clip_masks.shape} does not match "
                f"{(mode_count, trajectory_count, horizon_count)}."
            )
        output.update(
            {
                "predicted_point_clipping_fraction": float(clip_masks.mean()),
                "trajectory_clipping_fraction": float(
                    clip_masks.any(axis=(0, 2)).mean()
                ),
                "clipping_fraction_by_horizon": clip_masks.mean(axis=(0, 1)),
            }
        )
    return output


def _trajectory_energy_score(predictions, probabilities, masks, mode_ade):
    """Probability-weighted energy score using complete masked trajectories."""
    scores = np.empty(predictions.shape[1], dtype=np.float64)
    for trajectory in range(predictions.shape[1]):
        valid = masks[trajectory]
        paths = predictions[:, trajectory, valid]
        pair_distance = haversine_nmi(paths[:, None], paths[None]).mean(axis=-1)
        weights = probabilities[trajectory]
        expected_target_distance = np.dot(weights, mode_ade[:, trajectory])
        expected_pair_distance = np.sum(pair_distance * weights[:, None] * weights[None])
        scores[trajectory] = expected_target_distance - 0.5 * expected_pair_distance
    return float(scores.mean())


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, quantile, side="left"), len(values) - 1)
    return sorted_values[index]


def _weighted_mode_coverage(predictions, probabilities, targets, masks, levels):
    """Marginal/joint coverage of probability-weighted discrete mode intervals."""
    output = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        lat = np.full(predictions.shape[2], np.nan)
        lon = np.full(predictions.shape[2], np.nan)
        joint = np.full(predictions.shape[2], np.nan)
        for horizon in range(predictions.shape[2]):
            valid_indices = np.flatnonzero(masks[:, horizon])
            if not len(valid_indices):
                continue
            inside = []
            for trajectory in valid_indices:
                coordinates_inside = []
                for coordinate in range(2):
                    values = predictions[:, trajectory, horizon, coordinate]
                    weights = probabilities[trajectory]
                    lower = _weighted_quantile(values, weights, tail)
                    upper = _weighted_quantile(values, weights, 1.0 - tail)
                    coordinates_inside.append(
                        lower <= targets[trajectory, horizon, coordinate] <= upper
                    )
                inside.append(coordinates_inside)
            inside = np.asarray(inside, dtype=bool)
            lat[horizon] = inside[:, 0].mean()
            lon[horizon] = inside[:, 1].mean()
            joint[horizon] = inside.all(axis=1).mean()
        output[str(level)] = {"latitude": lat, "longitude": lon, "joint": joint}
    return output
