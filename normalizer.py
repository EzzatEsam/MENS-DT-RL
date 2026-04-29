import numpy as np
from gymnasium import spaces


class ObservationNormalizer:
    """
    Handles normalization and denormalization of environment observations
    following the Attribute Normalization procedure described in §3 of the
    MENS-DT-RL paper.

    Attributes whose bounds are finite are normalized to the range [-1, 1].
    Attributes with infinite bounds are passed through unchanged.
    """

    def __init__(self, observation_space: spaces.Space) -> None:
        """
        Initialize the normalizer using the environment's observation space.

        Parameters
        ----------
        observation_space : spaces.Space
            The observation space of the environment (usually a spaces.Box).
        """
        # Per §3: only normalize when both low and high are finite.
        # We assume the space has .low and .high attributes (like Box).
        self.low = getattr(observation_space, "low", None)
        self.high = getattr(observation_space, "high", None)

        if self.low is not None and self.high is not None:
            self.finite_mask = np.isfinite(self.low) & np.isfinite(self.high)
            # Avoid division by zero for attributes where low == high (edge case).
            self.range = np.where(self.finite_mask, self.high - self.low, 1.0)
            self.range = np.where(self.range == 0.0, 1.0, self.range)
        else:
            self.finite_mask = None
            self.range = None

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations to [-1, 1] for finite-bounded attributes.

        Supports both single observations (1D) and batches of observations (2D).

        Parameters
        ----------
        obs : np.ndarray
            The original observation(s) from the environment.
            Shape (D,) for a single observation or (N, D) for a batch.

        Returns
        -------
        np.ndarray
            The normalized observation(s).
        """
        if self.finite_mask is None:
            return obs

        obs_norm = obs.copy().astype(float)

        # Handle both single observation (1D) and batch (2D)
        if obs_norm.ndim == 1:
            obs_norm[self.finite_mask] = (
                2.0 * (obs[self.finite_mask] - self.low[self.finite_mask])
                / self.range[self.finite_mask]
                - 1.0
            )
            obs_norm[self.finite_mask] = np.clip(obs_norm[self.finite_mask], -1.0, 1.0)
        else:
            obs_norm[:, self.finite_mask] = (
                2.0 * (obs[:, self.finite_mask] - self.low[self.finite_mask])
                / self.range[self.finite_mask]
                - 1.0
            )
            obs_norm[:, self.finite_mask] = np.clip(
                obs_norm[:, self.finite_mask], -1.0, 1.0
            )

        return obs_norm

    def denormalize(self, obs_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize observations from [-1, 1] back to their original scale.

        Supports both single observations (1D) and batches of observations (2D).

        Parameters
        ----------
        obs_norm : np.ndarray
            The normalized observation(s) in the range [-1, 1].
            Shape (D,) for a single observation or (N, D) for a batch.

        Returns
        -------
        np.ndarray
            The observation(s) in its original environment scale.
        """
        if self.finite_mask is None:
            return obs_norm

        obs = obs_norm.copy().astype(float)

        # Handle both single observation (1D) and batch (2D)
        if obs.ndim == 1:
            obs[self.finite_mask] = (
                (obs_norm[self.finite_mask] + 1.0) * self.range[self.finite_mask] / 2.0
                + self.low[self.finite_mask]
            )
        else:
            obs[:, self.finite_mask] = (
                (obs_norm[:, self.finite_mask] + 1.0)
                * self.range[self.finite_mask]
                / 2.0
                + self.low[self.finite_mask]
            )

        return obs

    def normalize_attribute(self, value: float, index: int) -> float:
        """
        Normalize a single attribute value by its index.

        Parameters
        ----------
        value : float
            The raw attribute value.
        index : int
            The index of the attribute in the observation space.

        Returns
        -------
        float
            The normalized value.
        """
        if self.finite_mask is None or not self.finite_mask[index]:
            return value

        val_norm = 2.0 * (value - self.low[index]) / self.range[index] - 1.0
        return float(np.clip(val_norm, -1.0, 1.0))

    def denormalize_attribute(self, value_norm: float, index: int) -> float:
        """
        Denormalize a single attribute value by its index.

        Parameters
        ----------
        value_norm : float
            The normalized attribute value in [-1, 1].
        index : int
            The index of the attribute in the observation space.

        Returns
        -------
        float
            The denormalized value.
        """
        if self.finite_mask is None or not self.finite_mask[index]:
            return value_norm

        val = (value_norm + 1.0) * self.range[index] / 2.0 + self.low[index]
        return float(val)
