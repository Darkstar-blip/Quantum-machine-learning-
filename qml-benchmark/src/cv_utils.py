"""Cross-validation utilities used by the notebook.

This module provides simple helpers for creating a StratifiedKFold,
iterating parameter grids, writing JSON results to disk, and computing
mean/std summaries.
"""
import os
import json
from typing import Dict, List, Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid


def make_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    """Return a StratifiedKFold with shuffle and fixed seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def param_iter(grid: Dict[str, List[Any]]):
    """Return a ParameterGrid iterator for the given hyperparameter grid."""
    return ParameterGrid(grid)


def write_json(path: str, obj: Any):
    """Write an object as JSON to path, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def mean_std(xs: List[float]):
    """Return a dict with mean and std of a sequence (NaNs ignored)."""
    return {"mean": float(np.nanmean(xs)), "std": float(np.nanstd(xs))}
