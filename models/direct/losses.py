"""Masked deterministic trajectory objective used by B0 and B3A."""

import torch
from torch.nn import functional as F


def masked_trajectory_loss(prediction, targets, masks, coordinate_scale):
    trajectories = prediction.trajectories
    if trajectories.ndim != 4 or trajectories.shape[1] != 1:
        raise ValueError("Deterministic predictions must have shape (batch, 1, steps, 2).")
    if targets.shape != trajectories[:, 0].shape:
        raise ValueError("Targets do not align with deterministic predictions.")
    if masks.shape != targets.shape[:2]:
        raise ValueError("Target masks do not align with targets.")
    valid_counts = masks.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("Every example needs at least one valid future point.")

    scale = torch.as_tensor(
        coordinate_scale, dtype=trajectories.dtype, device=trajectories.device
    )
    point_loss = F.smooth_l1_loss(
        trajectories[:, 0] * scale,
        targets * scale,
        reduction="none",
    ).sum(dim=-1)
    trajectory_loss = ((point_loss * masks).sum(dim=1) / valid_counts).mean()
    return {"loss": trajectory_loss, "trajectory_loss": trajectory_loss}
