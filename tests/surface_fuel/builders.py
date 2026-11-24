"""Shared builders and measures for the surface_fuel tests.

Grids are generated with fixed seeds and measured, so tests use loose
tolerances to allow for chance. Large grids hold more patches, which leaves
chance less room.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from fastfuels_core.surface_fuel import build_distribution_grids

UNIFORM = {"type": "uniform", "low": 1, "high": 5}
NORMAL = {"type": "normal", "mean": 10, "std": 1}


def build(bands, correlations=None, shape=(20, 30), resolution=2.0, seed=1):
    """build_distribution_grids with a small grid and a fixed seed."""
    return build_distribution_grids(shape, resolution, bands, correlations, seed)


def uniform_bands(*keys, patch_size=16):
    """Uniform bands with the same patch size."""
    return {key: {"distribution": UNIFORM, "patch_size": patch_size} for key in keys}


def pair(a, b, coefficient):
    return {"bands": [a, b], "coefficient": coefficient}


def rank_correlation(a, b):
    return stats.spearmanr(a.ravel(), b.ravel()).statistic


def similarity(grid, lag):
    """Correlation between cells ``lag`` apart, along rows and columns."""
    first = np.concatenate([grid[:, :-lag].ravel(), grid[:-lag, :].ravel()])
    second = np.concatenate([grid[:, lag:].ravel(), grid[lag:, :].ravel()])
    return np.corrcoef(first, second)[0, 1]
