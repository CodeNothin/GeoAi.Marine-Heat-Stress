"""Smooth random fields used for mesoscale ocean variability and fish patchiness."""
from __future__ import annotations

import numpy as np


class RandomField:
    """Stationary, smooth Gaussian-like field built from travelling cosine waves.

    Cheap to evaluate at arbitrary (x_km, y_km, t_days) points, so the simulator
    never needs to store full daily grids.
    """

    def __init__(self, rng, n_modes, length_km, sd, time_days=None):
        theta = rng.uniform(0.0, 2.0 * np.pi, n_modes)
        wavelength = length_km * np.exp(rng.normal(0.0, 0.3, n_modes))
        k = 2.0 * np.pi / wavelength
        self.kx = k * np.cos(theta)
        self.ky = k * np.sin(theta)
        if time_days is None:
            self.omega = np.zeros(n_modes)
        else:
            self.omega = 2.0 * np.pi / time_days * rng.normal(0.0, 1.0, n_modes)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, n_modes)
        self.amp = sd * np.sqrt(2.0 / n_modes)

    def __call__(self, x, y, t=0.0):
        x = np.asarray(x, float)[..., None]
        y = np.asarray(y, float)[..., None]
        t = np.asarray(t, float)[..., None]
        arg = self.kx * x + self.ky * y + self.omega * t + self.phase
        return self.amp * np.cos(arg).sum(axis=-1)
