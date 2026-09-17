"""Canonical temporal encoder for deterministic vessel forecasting."""

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    """Encode the complete observed AIS history into one state vector."""

    def __init__(self, input_dim, hidden_dim, layers, heads, dropout, max_steps):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads.")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_steps, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.network = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, features):
        steps = features.shape[1]
        if steps > self.position.shape[1]:
            raise ValueError(
                f"Context has {steps} steps but encoder supports "
                f"{self.position.shape[1]}."
            )
        encoded = self.network(
            self.input_projection(features) + self.position[:, :steps]
        )
        return encoded[:, -1]
