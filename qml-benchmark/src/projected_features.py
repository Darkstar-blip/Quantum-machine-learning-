"""Projected feature map helpers for projected kernels.

Provides a simple Random Fourier Features (RFF) implementation to project
inputs into an explicit feature space approximating an RBF kernel. This is
used by the `04_projected_kernel.ipynb` notebook as a lightweight, fast
approximation to quantum-inspired projected kernel features.
"""
from typing import Optional
import numpy as np


def projected_phi(X: np.ndarray, n_components: int = 128, scale: float = 1.0, random_state: Optional[int] = None, **kwargs) -> np.ndarray:
    """Return random Fourier features for X.

    Args:
        X: array-like shape (n_samples, n_features)
        n_components: number of RFF components to produce
        scale: kernel lengthscale (larger -> smoother)
        random_state: optional seed

    Returns:
        Z: array shape (n_samples, n_components) of projected features
    """
    # Accept legacy kwargs used by notebooks (reps, shots) and map to n_components
    reps = kwargs.get("reps", None)
    shots = kwargs.get("shots", None)
    if reps is not None and (n_components is None or n_components == 128):
        # choose number of random features proportional to reps
        n_components = max(32, int(reps) * 64)
    if shots is not None and (n_components is None or n_components == 128):
        # shots is not directly applicable to RFF; use as an upper bound
        try:
            sc = int(shots)
            n_components = max(n_components, min(sc, 4096))
        except Exception:
            pass

    X = np.asarray(X)
    rng = np.random.RandomState(random_state)
    # Draw random Gaussian frequencies
    W = rng.normal(loc=0.0, scale=1.0 / float(scale), size=(X.shape[1], n_components))
    b = rng.uniform(0, 2 * np.pi, size=n_components)
    projection = X.dot(W) + b
    Z = np.sqrt(2.0 / n_components) * np.cos(projection)
    return Z
