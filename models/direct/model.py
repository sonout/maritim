"""Canonical deterministic whole-trajectory predictors for B0 and B3A."""

from dataclasses import dataclass

import torch
from torch import nn

from .coordinates import (
    bound_physical_displacement,
    integrate_normalized_displacements,
    physical_to_normalized_displacement,
)
from .encoders import TransformerEncoder
from .features import ContinuousAISFeatures


@dataclass
class TrajectoryPrediction:
    trajectories: torch.Tensor
    logits: torch.Tensor
    clip_mask: torch.Tensor | None = None

    @property
    def probabilities(self):
        return torch.softmax(self.logits, dim=-1)


def _mlp(input_dim, hidden_dim, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim * 2),
        nn.GELU(),
        nn.Linear(hidden_dim * 2, output_dim),
    )


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim,
        encoder_layers,
        attention_heads,
        dropout,
        input_steps,
        include_global=True,
        delta_scale=100.0,
    ):
        super().__init__()
        self.features = ContinuousAISFeatures(
            delta_scale=delta_scale, include_global=include_global
        )
        self.encoder = TransformerEncoder(
            input_dim=self.features.output_dim,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            heads=attention_heads,
            dropout=dropout,
            max_steps=input_steps,
        )

    def forward(self, context):
        return self.encoder(self.features(context))


class DirectTrajectoryPredictor(nn.Module):
    """Frozen deterministic K=1 design shared by B0 and B3A.

    B3A adds only a seven-value, missingness-aware vessel-history descriptor.
    Parameter names and layer shapes match the selected research checkpoints.
    """

    def __init__(
        self,
        *,
        input_steps=18,
        output_steps=72,
        hidden_dim=128,
        encoder_layers=3,
        attention_heads=4,
        dropout=0.1,
        max_step_nmi=2.5,
        physical_scale=(150.0, 90.0),
        include_global=True,
        delta_scale=100.0,
        vessel_history_dim=0,
        vessel_history_hidden_dim=32,
    ):
        super().__init__()
        if input_steps < 1 or output_steps < 1:
            raise ValueError("input_steps and output_steps must be positive.")
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.num_modes = 1
        self.vessel_history_dim = int(vessel_history_dim)
        if self.vessel_history_dim not in (0, 7):
            raise ValueError("vessel_history_dim must be 0 for B0 or 7 for B3A.")

        self.history_encoder = HistoryEncoder(
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            input_steps=input_steps,
            include_global=include_global,
            delta_scale=delta_scale,
        )
        if self.vessel_history_dim:
            if vessel_history_hidden_dim <= 0:
                raise ValueError("vessel_history_hidden_dim must be positive.")
            self.vessel_history_encoder = nn.Sequential(
                nn.Linear(self.vessel_history_dim, vessel_history_hidden_dim),
                nn.GELU(),
                nn.Linear(vessel_history_hidden_dim, vessel_history_hidden_dim),
            )
            history_output_dim = hidden_dim + vessel_history_hidden_dim + 1
        else:
            self.vessel_history_encoder = None
            history_output_dim = hidden_dim

        self.trajectory_head = _mlp(
            history_output_dim, hidden_dim, output_steps * 2, dropout
        )
        self.register_buffer("max_step_nmi", torch.tensor(float(max_step_nmi)))
        self.register_buffer(
            "physical_scale", torch.as_tensor(physical_scale, dtype=torch.float32)
        )
        # This is equivalent to cumsum and deterministic on pinned CUDA builds.
        self.register_buffer(
            "integration_matrix", torch.tril(torch.ones(output_steps, output_steps))
        )

    def forward(self, context, vessel_history=None):
        if context.ndim != 3 or context.shape[1:] != (self.input_steps, 4):
            raise ValueError(
                f"Expected context (batch, {self.input_steps}, 4), got "
                f"{tuple(context.shape)}."
            )
        history = self.history_encoder(context)
        if self.vessel_history_encoder is None:
            if vessel_history is not None:
                raise ValueError("B0 does not accept vessel-history context.")
        else:
            expected = (context.shape[0], self.vessel_history_dim)
            if vessel_history is None or tuple(vessel_history.shape) != expected:
                shape = None if vessel_history is None else tuple(vessel_history.shape)
                raise ValueError(f"Expected vessel-history context {expected}, got {shape}.")
            if not torch.isfinite(vessel_history).all():
                raise ValueError("Vessel-history context contains non-finite values.")
            support = vessel_history[:, -1:]
            if torch.any((support != 0) & (support != 1)):
                raise ValueError("Vessel-history support flag must be binary.")
            history = torch.cat(
                (history, self.vessel_history_encoder(vessel_history), support), dim=-1
            )

        raw_displacements = self.trajectory_head(history).view(
            -1, 1, self.output_steps, 2
        )
        displacement_nmi = bound_physical_displacement(
            raw_displacements, self.max_step_nmi
        )
        increments = physical_to_normalized_displacement(
            displacement_nmi, self.physical_scale
        )
        trajectories = integrate_normalized_displacements(
            context[:, None, -1, :2], increments, self.integration_matrix
        )
        clip_mask = ((trajectories < 0.0) | (trajectories > 0.9999)).any(dim=-1)
        trajectories = trajectories.clamp(0.0, 0.9999)
        logits = torch.zeros(
            context.shape[0], 1, device=context.device, dtype=trajectories.dtype
        )
        return TrajectoryPrediction(trajectories, logits, clip_mask)
