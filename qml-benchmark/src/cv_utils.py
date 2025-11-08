"""Small cross-validation utilities used by the notebook."""
from typing import Dict, Iterable, Any
import itertools
import json
import numpy as np


def make_cv(n_splits: int = 5, seed: int = 42):
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def param_iter(grid: Dict[str, Iterable[Any]]):
    """Yield dicts from an hyperparameter grid (dict of lists)."""
    keys = list(grid.keys())
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield {k: v for k, v in zip(keys, vals)}


def write_json(path: str, obj: Any):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def cv_summary(values: Iterable[float]):
    vals = np.array(list(values), dtype=float)
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "n": int(np.sum(~np.isnan(vals)))}
