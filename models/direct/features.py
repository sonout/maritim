"""Canonical continuous AIS representation used by B0 and B3A."""

import math

import torch
from torch import nn


class ContinuousAISFeatures(nn.Module):
    """Convert normalized ``[LAT, LON, SOG, COG]`` histories to nine features.

    The output order is ``[x_global, y_global, dx, dy, velocity_east,
    velocity_north, SOG, sin(COG), cos(COG)]``. This is the frozen B0/B3A
    representation; heading and ROT are deliberately not part of it.
    """

    output_dim = 9

    def __init__(self, delta_scale=100.0, include_global=True):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.include_global = bool(include_global)

    def forward(self, context):
        if context.ndim != 3 or context.shape[-1] != 4:
            raise ValueError(
                f"Expected AIS context (batch, steps, 4), got {context.shape}."
            )
        if not torch.isfinite(context).all():
            raise ValueError("AIS context contains non-finite values.")

        lat, lon, sog, cog = context.unbind(dim=-1)
        x_global = lon * 2.0 - 1.0
        y_global = lat * 2.0 - 1.0
        if not self.include_global:
            x_global = torch.zeros_like(x_global)
            y_global = torch.zeros_like(y_global)

        dx = torch.zeros_like(lon)
        dy = torch.zeros_like(lat)
        dx[:, 1:] = (lon[:, 1:] - lon[:, :-1]) * self.delta_scale
        dy[:, 1:] = (lat[:, 1:] - lat[:, :-1]) * self.delta_scale

        angle = cog * (2.0 * math.pi)
        sin_cog = torch.sin(angle)
        cos_cog = torch.cos(angle)
        return torch.stack(
            (
                x_global,
                y_global,
                dx,
                dy,
                sog * sin_cog,
                sog * cos_cog,
                sog,
                sin_cog,
                cos_cog,
            ),
            dim=-1,
        )
