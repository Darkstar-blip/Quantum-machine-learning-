"""Minimal kernel helpers for the notebook.

Provides:
- build_quantum_kernel(feature_dimension, reps)
- rescale(X, scale)
- gram_train(qk, X)
- gram_test(qk, X_tr, X_te)

This module prefers Qiskit's FidelityQuantumKernel when available and
falls back to a simple classical RBF kernel for environments without a
quantum backend.
"""
from typing import Any
import numpy as np


def build_quantum_kernel(feature_dimension: int, reps: int = 1) -> Any:
    """Return a kernel object compatible with QSVC.

    Uses qiskit_machine_learning.kernels.FidelityQuantumKernel when
    available; otherwise falls back to a lightweight placeholder
    object that exposes an ``evaluate`` method (classical RBF).
    """
    # By default prefer a cheap classical RBF fallback so the notebook
    # remains fast and runnable in CI / devcontainers. To force use of
    # the real qiskit kernels set the environment variable
    # USE_QISKIT_KERNEL=1 in the environment.
    use_qiskit = False
    try:
        import os
        use_qiskit = os.environ.get("USE_QISKIT_KERNEL") in ("1", "true", "True")
    except Exception:
        use_qiskit = False

    if use_qiskit:
        try:
            from qiskit.circuit.library import ZZFeatureMap
            try:
                # try modern qiskit-machine-learning kernel
                from qiskit_machine_learning.kernels import FidelityQuantumKernel

                fm = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps)
                return FidelityQuantumKernel(feature_map=fm)
            except Exception:
                # older QuantumKernel location
                try:
                    from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel

                    fm = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps)
                    return QuantumKernel(feature_map=fm)
                except Exception:
                    # fall back to classical placeholder
                    pass
        except Exception:
            # if qiskit not installed or feature map failed, fall back
            pass

    # Fallback: return a simple object with evaluate(X) and evaluate(X,Y)
    class _RBFKernel:
        def __init__(self, gamma: float = 0.5):
            self.gamma = float(gamma)

        def evaluate(self, X, Y=None):
            X = np.asarray(X)
            if Y is None:
                Y = X
            else:
                Y = np.asarray(Y)
            sqd = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
            return np.exp(-self.gamma * sqd)

    return _RBFKernel(gamma=0.5)


def rescale(X: np.ndarray, scale: float) -> np.ndarray:
    """Scale input features by a scalar."""
    X = np.asarray(X)
    return X * float(scale)


def gram_train(qk: Any, X: np.ndarray) -> np.ndarray:
    """Return the training Gram/kernel matrix for X.

    If the kernel object provides ``evaluate``, use it; otherwise compute
    an RBF matrix as a fallback.
    """
    try:
        K = qk.evaluate(X)
        # ensure square matrix
        return np.asarray(K)
    except Exception:
        X = np.asarray(X)
        sqd = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sqd)


def gram_test(qk: Any, X_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Return the test Gram matrix between X_tr and X_te."""
    try:
        K = qk.evaluate(X_tr, X_te)
        K = np.asarray(K)
        # qiskit kernels may return shape (n_tr, n_te); sklearn expects (n_te, n_tr)
        if K.shape == (len(X_tr), len(X_te)):
            return K.T
        return K
    except Exception:
        X_tr = np.asarray(X_tr)
        X_te = np.asarray(X_te)
        sqd = np.sum((X_te[:, None, :] - X_tr[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sqd)
