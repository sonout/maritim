"""Shared conversions between normalized ROI coordinates and physical motion."""

import math

import numpy as np
import torch


def physical_coordinate_scale(data_config):
    """Return North/East nautical miles per normalized latitude/longitude unit."""
    lat_range = float(data_config.lat_max) - float(data_config.lat_min)
    lon_range = float(data_config.lon_max) - float(data_config.lon_min)
    if lat_range <= 0 or lon_range <= 0:
        raise ValueError("Dataset latitude and longitude ranges must be positive.")
    mean_latitude = (float(data_config.lat_min) + float(data_config.lat_max)) / 2.0
    return np.asarray(
        [
            lat_range * 60.0,
            lon_range * math.cos(math.radians(mean_latitude)) * 60.0,
        ],
        dtype=np.float32,
    )


def max_step_distance_nmi(max_speed_knots, cadence_seconds):
    """Convert a speed bound and sampling cadence into distance per step."""
    speed = float(max_speed_knots)
    cadence = float(cadence_seconds)
    if speed <= 0 or cadence <= 0:
        raise ValueError("max_speed_knots and cadence_seconds must be positive.")
    return speed * cadence / 3600.0


def _scale_like(values, scale):
    if torch.is_tensor(values):
        return torch.as_tensor(scale, dtype=values.dtype, device=values.device)
    return np.asarray(scale, dtype=np.asarray(values).dtype)


def normalized_to_physical_displacement(displacement, scale):
    """Convert normalized ``[latitude, longitude]`` deltas to North/East nmi."""
    return displacement * _scale_like(displacement, scale)


def physical_to_normalized_displacement(displacement_nmi, scale):
    """Convert North/East nautical-mile deltas to normalized ROI increments."""
    return displacement_nmi / _scale_like(displacement_nmi, scale)


def bound_physical_displacement(raw, max_step_nmi, eps=None):
    """Bound each 2-D displacement magnitude without imposing axis-wise limits."""
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    if eps is None:
        eps = torch.finfo(raw.dtype).eps
    direction = raw / norm.clamp_min(eps)
    magnitude = torch.as_tensor(
        max_step_nmi, dtype=raw.dtype, device=raw.device
    ) * torch.tanh(norm)
    return direction * magnitude


def integrate_normalized_displacements(start, increments, integration_matrix=None):
    """Integrate normalized increments from a starting coordinate."""
    if integration_matrix is None:
        integrated = torch.cumsum(increments, dim=-2)
    else:
        matrix = torch.as_tensor(
            integration_matrix, dtype=increments.dtype, device=increments.device
        )
        integrated = torch.matmul(matrix, increments)
    return start.unsqueeze(-2) + integrated
