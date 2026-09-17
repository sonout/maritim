"""Common types shared by every forecasting model adapter."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastOutput:
    """Model-independent prediction result in example-major ordering."""

    trajectories: np.ndarray  # [N, K, T, 2], normalized latitude/longitude
    probabilities: np.ndarray  # [N, K]
    clip_mask: np.ndarray  # [N, K, T]
    trajectory_ids: np.ndarray  # [N]

    def validate(self, examples, output_steps):
        expected_prefix = (int(examples),)
        if self.trajectories.ndim != 4 or self.trajectories.shape[:1] != expected_prefix:
            raise ValueError("Forecast trajectories must have shape [N, K, T, 2].")
        if self.trajectories.shape[2:] != (int(output_steps), 2):
            raise ValueError("Forecast trajectories have an invalid horizon or coordinate axis.")
        modes = self.trajectories.shape[1]
        if self.probabilities.shape != (examples, modes):
            raise ValueError("Forecast probabilities do not align with trajectories.")
        if self.clip_mask.shape != (examples, modes, output_steps):
            raise ValueError("Forecast clipping mask does not align with trajectories.")
        if self.trajectory_ids.shape != (examples,):
            raise ValueError("Forecast trajectory IDs do not align with examples.")
        if not np.isfinite(self.trajectories).all() or not np.isfinite(
            self.probabilities
        ).all():
            raise ValueError("Forecast output contains non-finite values.")
        if np.any(self.probabilities < 0) or not np.allclose(
            self.probabilities.sum(axis=1), 1.0
        ):
            raise ValueError("Forecast probabilities must be non-negative and sum to one.")
