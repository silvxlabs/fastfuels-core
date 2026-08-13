"""Canopy fuel metrics (CBD, CBH, CH, CC, CFL) from a tree inventory.

The computation follows the FuelCalc profile method applied per output
cell instead of per plot: each tree's available canopy fuel is
distributed vertically over its crown into fixed-depth layers,
attributed horizontally to the cells its crown covers, and accumulated
into a per-cell vertical profile that is reduced to the requested bands.

Each stage is a public function so callers can run, test, or compose
them individually; :func:`compute_canopy_metrics` chains them to fill a
caller-provided georeferenced Dataset. The caller owns the output
lattice — nothing here creates grids or chooses resolutions.

All per-tree math is vectorized; trees are never iterated in Python.

Expected tree columns follow the v2 FastFuels inventory convention:
``x``, ``y``, ``height``, ``crown_ratio``, and, depending on options,
``dbh``, ``fia_species_code``, and caller-named biomass / crown radius
columns. Units: meters, centimeters (dbh), kilograms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

FT_TO_M = 0.3048


def available_canopy_fuel(
    trees: pd.DataFrame,
    *,
    fuel_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
) -> np.ndarray:
    """Per-tree available canopy fuel (kg).

    With ``fuel_column`` set, that column is returned as-is (precomputed
    fuel, e.g. from LiDAR regression); the fractions are ignored.
    Otherwise fuel is ``foliage_fraction * NSVB foliage +
    branchwood_fraction * fine branchwood``, where fine branchwood is
    the Brown (1978) fine share ``pb / (1 - pfol)`` applied to the NSVB
    total branch weight. Requires ``dbh``, ``height``, and
    ``fia_species_code`` columns.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(trees),)``, kg per tree.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def cumulative_fuel_fraction(species_code: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Cumulative fuel fraction at fractional crown height, by species.

    Evaluates the Reinhardt et al. (2006) cubic ``pw(ph) = B1*ph +
    B2*ph**2 + B3*ph**3`` for each (tree, height) pair, resolving each
    FIA species code to its FuelCalc vertical-distribution equation.
    Species absent from the FuelCalc table use the uniform distribution
    (``pw = ph``), as do hardwoods.

    Parameters
    ----------
    species_code : numpy.ndarray
        FIA species codes, shape ``(n_trees,)``.
    ph : numpy.ndarray
        Fractional heights within the crown, clipped to [0, 1]. Shape
        ``(n_trees,)`` or ``(n_trees, n_levels)``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``ph``.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def vertical_profile(
    trees: pd.DataFrame,
    fuel: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    n_layers: int | None = None,
    layer_depth: float = FT_TO_M,
    vertical_distribution: str = "reinhardt_2006",
    horizontal_distribution: str = "crown_projected",
    crown_radius_column: str | None = None,
) -> np.ndarray:
    """Accumulate per-tree fuel into a per-cell vertical profile grid.

    Each tree's fuel is split across profile layers spanning its crown
    (``vertical_distribution``: ``"reinhardt_2006"`` species cubics or
    ``"uniform"``) and across the cells its crown covers
    (``horizontal_distribution``: ``"crown_projected"`` splits by exact
    crown-disk / cell intersection area; ``"stem"`` assigns everything
    to the stem cell). Contributions scatter-add, so any batch of trees
    in any order accumulates to the same result.

    Parameters
    ----------
    trees : pandas.DataFrame
        Tree records in the grid's CRS.
    fuel : numpy.ndarray
        Per-tree available canopy fuel (kg), from
        :func:`available_canopy_fuel`.
    transform : tuple
        Rasterio-style affine (a, b, c, d, e, f) of the north-up output
        lattice.
    shape : tuple
        Output lattice shape ``(ny, nx)``.
    n_layers : int, optional
        Number of profile layers; defaults to covering the tallest tree.
    crown_radius_column : str, optional
        Per-tree max crown radius (m) column; defaults to the Purves
        allometric radius.

    Returns
    -------
    numpy.ndarray
        Bulk density profile (kg/m**3), shape ``(n_layers, ny, nx)``.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def cbd_running_mean(
    profile: np.ndarray,
    *,
    layer_depth: float = FT_TO_M,
    window: float | None = 3.0,
) -> np.ndarray:
    """Canopy bulk density: per-cell maximum running mean of the profile.

    ``window`` is the running-mean depth in meters (Reinhardt et al.
    2006 use 3.0 m; FuelCalc's guide states 5 ft in one place and no
    smoothing in another). ``window=None`` skips smoothing and returns
    the maximum single layer.

    Returns
    -------
    numpy.ndarray
        CBD (kg/m**3), shape ``(ny, nx)``.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def profile_threshold_heights(
    profile: np.ndarray,
    *,
    layer_depth: float = FT_TO_M,
    threshold: float = 0.012,
    relative_fraction: float | None = 0.1,
    smoothing_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Canopy base height and canopy height from a bulk-density threshold.

    Per cell, the effective threshold is ``min(relative_fraction *
    profile_max, threshold)`` (FuelCalc's rule; ``relative_fraction=None``
    uses the flat threshold alone). CBH is the bottom of the lowest
    layer at or above threshold; canopy height is the top of the highest.
    ``smoothing_window`` (m) optionally smooths the profile first
    (FFE-FVS uses 0.9144 m). Cells with no layer above threshold are NaN.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cbh, chm)`` in meters, each shape ``(ny, nx)``.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def canopy_fuel_load(
    profile: np.ndarray, *, layer_depth: float = FT_TO_M
) -> np.ndarray:
    """Canopy fuel load: vertical integral of the bulk-density profile.

    Returns
    -------
    numpy.ndarray
        CFL (kg/m**2), shape ``(ny, nx)``.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def canopy_cover(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    crown_radius_column: str | None = None,
) -> np.ndarray:
    """Per-cell projected canopy cover (%) from the crown-disk union.

    Overlapping crowns count once (a true union, not a sum of crown
    areas). Crown radii come from ``crown_radius_column`` or the Purves
    allometric radius.

    Returns
    -------
    numpy.ndarray
        Cover (%), shape ``(ny, nx)``.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")


def compute_canopy_metrics(
    trees: pd.DataFrame,
    dataset: xr.Dataset,
    *,
    fuel_column: str | None = None,
    crown_radius_column: str | None = None,
    foliage_fraction: float = 1.0,
    branchwood_fraction: float = 0.5,
    min_tree_height: float = 0.0,
    layer_depth: float = FT_TO_M,
    vertical_distribution: str = "reinhardt_2006",
    horizontal_distribution: str = "crown_projected",
    cbd_window: float | None = 3.0,
    cbh_threshold: float = 0.012,
    cbh_relative_fraction: float | None = 0.1,
    threshold_smoothing_window: float | None = None,
) -> xr.Dataset:
    """Fill a georeferenced Dataset with canopy fuel metrics.

    The caller provides ``dataset`` with one variable per requested band
    — any of ``cbd``, ``cbh``, ``chm``, ``cc``, ``cfl`` — on a north-up
    lattice with CRS and transform set via rioxarray. Only the variables
    present are computed. Tree coordinates must be in the dataset's CRS;
    only live trees should be passed. Trees shorter than
    ``min_tree_height`` (m) are excluded.

    Chains the public stages: :func:`available_canopy_fuel` →
    :func:`vertical_profile` → :func:`cbd_running_mean` /
    :func:`profile_threshold_heights` / :func:`canopy_fuel_load`, plus
    :func:`canopy_cover`. See each stage for parameter semantics.

    Returns
    -------
    xarray.Dataset
        ``dataset`` with its band variables filled in place.
    """
    raise NotImplementedError("fastfuels-core#95: first slice under construction.")
