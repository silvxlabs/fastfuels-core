"""Canopy base height and canopy height (m), from a bulk-density threshold.

Both are read off the same vertical profile with the same scan, so they
are produced together: CBH is the bottom of the lowest layer clearing
the threshold and canopy height the top of the highest. Callers writing
the pair into a LANDFIRE-keyed Dataset store the second under the band
key ``chm``; it is a threshold-derived canopy height, not a canopy
height model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastfuels_core.canopy_fuel.bulk_density import (
    FUELCALC_EDGE,
    profile_running_mean,
    window_in_layers,
)
from fastfuels_core.canopy_fuel.profile import FUELCALC_LAYER_DEPTH, stem_cells

# Canopy base height and canopy height each have a profile-threshold method
# (the pair read off one scan; see :func:`profile_threshold_heights`) and a
# per-tree alternative — the fuel-weighted mean crown base for cbh, a tree
# height percentile for chm.
THRESHOLD_METHOD = "bulk_density_threshold"
MEAN_CROWN_BASE_METHOD = "mean_crown_base"
HEIGHT_PERCENTILE_METHOD = "height_percentile"
CBH_METHODS = (THRESHOLD_METHOD, MEAN_CROWN_BASE_METHOD)
CHM_METHODS = (THRESHOLD_METHOD, HEIGHT_PERCENTILE_METHOD)


def validate_cbh_method(method: str) -> None:
    if method not in CBH_METHODS:
        raise ValueError(
            f"Unknown cbh method {method!r}; expected one of {list(CBH_METHODS)}."
        )


def validate_chm_method(method: str) -> None:
    if method not in CHM_METHODS:
        raise ValueError(
            f"Unknown chm method {method!r}; expected one of {list(CHM_METHODS)}."
        )


def mean_crown_base_height(
    trees: pd.DataFrame,
    fuel: np.ndarray,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Available-fuel-weighted mean per-tree crown base height (m), per cell.

    Each tree is binned to its stem cell (see
    :func:`~fastfuels_core.canopy_fuel.profile.stem_cells`) and its crown
    base height ``height * (1 - crown_ratio)`` is averaged over the cell,
    weighted by available canopy fuel so heavier crowns pull the mean.
    Cells with no fuel are NaN, matching the threshold method's no-canopy.

    This is a stand-scale summary; per-tree averaging is known to
    misrepresent crown-fire initiation in multi-storied stands, so it is a
    comparison method, not the default.
    """
    ny, nx = shape
    out = np.full(ny * nx, np.nan)
    if len(trees):
        row, col = stem_cells(trees, transform, shape)
        flat = row * nx + col
        height = trees["height"].to_numpy(dtype=np.float64)
        crown_ratio = trees["crown_ratio"].to_numpy(dtype=np.float64)
        crown_base = height * (1.0 - crown_ratio)
        weight = np.asarray(fuel, dtype=np.float64)
        num = np.bincount(flat, weights=weight * crown_base, minlength=ny * nx)
        den = np.bincount(flat, weights=weight, minlength=ny * nx)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
    return out.reshape(ny, nx)


def height_percentile(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    percentile: float = 99.0,
) -> np.ndarray:
    """Per-cell percentile of tree heights (m); a canopy-surface height.

    Each tree is binned to its stem cell and the given ``percentile`` of
    the heights in the cell is reported (99th by default, a robust stand
    top). Unlike the threshold canopy height, this reads the tree heights
    directly rather than where the bulk-density profile crosses a
    threshold, so it is the measure to compare against a lidar canopy
    height model. Cells with no trees are NaN.
    """
    ny, nx = shape
    out = np.full(ny * nx, np.nan)
    if len(trees):
        row, col = stem_cells(trees, transform, shape)
        flat = row * nx + col
        height = trees["height"].to_numpy(dtype=np.float64)
        grouped = pd.DataFrame({"cell": flat, "height": height}).groupby("cell")
        quantile = grouped["height"].quantile(percentile / 100.0)
        out[quantile.index.to_numpy()] = quantile.to_numpy()
    return out.reshape(ny, nx)


def mean_crown_length(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    """Per-cell mean of the per-tree crown length ``height * crown_ratio`` (m).

    A canopy depth for the load-over-depth CBD, the one Cruz et al. (2003)
    used. Each tree is binned to its stem cell and the crown lengths are
    averaged over the cell (an unweighted mean, one tree one vote). Cells
    with no trees are NaN.
    """
    ny, nx = shape
    out = np.full(ny * nx, np.nan)
    if len(trees):
        row, col = stem_cells(trees, transform, shape)
        flat = row * nx + col
        height = trees["height"].to_numpy(dtype=np.float64)
        crown_ratio = trees["crown_ratio"].to_numpy(dtype=np.float64)
        crown_length = height * crown_ratio
        num = np.bincount(flat, weights=crown_length, minlength=ny * nx)
        count = np.bincount(flat, minlength=ny * nx)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(count > 0, num / count, np.nan)
    return out.reshape(ny, nx)


def height_percentile_depth(
    trees: pd.DataFrame,
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    *,
    top_percentile: float = 90.0,
) -> np.ndarray:
    """Per-cell canopy depth from tree-height and crown-base percentiles (m).

    A canopy depth for the load-over-depth CBD: the ``top_percentile`` of
    tree heights in the cell (90th by default, a robust stand top) minus
    the median per-tree crown base height ``height * (1 - crown_ratio)``.
    Each tree is binned to its stem cell; cells with no trees are NaN.
    """
    ny, nx = shape
    out = np.full(ny * nx, np.nan)
    if len(trees):
        row, col = stem_cells(trees, transform, shape)
        flat = row * nx + col
        height = trees["height"].to_numpy(dtype=np.float64)
        crown_ratio = trees["crown_ratio"].to_numpy(dtype=np.float64)
        crown_base = height * (1.0 - crown_ratio)
        grouped = pd.DataFrame(
            {"cell": flat, "height": height, "base": crown_base}
        ).groupby("cell")
        top = grouped["height"].quantile(top_percentile / 100.0)
        base = grouped["base"].median()
        depth = top - base
        out[depth.index.to_numpy()] = depth.to_numpy()
    return out.reshape(ny, nx)


def _fuel_extent(
    profile: np.ndarray, layer_depth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bottom and top (m) of the layers holding any fuel, per cell."""
    n_layers = profile.shape[0]
    occupied = profile > 0.0
    lowest = occupied.argmax(axis=0)
    highest = n_layers - 1 - occupied[::-1].argmax(axis=0)
    return lowest * layer_depth, (highest + 1) * layer_depth


def profile_threshold_heights(
    profile: np.ndarray,
    *,
    layer_depth: float = FUELCALC_LAYER_DEPTH,
    threshold: float = 0.012,
    relative_fraction: float | None = 0.1,
    smoothing_window: float | None = 5 * FUELCALC_LAYER_DEPTH,
    smoothing_edge: str = FUELCALC_EDGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Canopy base height and canopy height from a bulk-density threshold.

    Per cell, the effective threshold is ``min(relative_fraction *
    profile_max, threshold)`` (FuelCalc's rule; ``relative_fraction=None``
    uses the flat threshold alone). ``smoothing_window`` (m) optionally
    smooths the profile first with a centered running mean, whose
    behaviour past the ends of the profile ``smoothing_edge`` selects —
    see :func:`~fastfuels_core.canopy_fuel.bulk_density.profile_running_mean`.
    The default is FuelCalc's, which zero-pads above the canopy against
    a full denominator; ``"truncate"`` is FFE-FVS's and reports canopy
    height a layer higher on the same profile, because dividing the
    topmost window by a short denominator inflates it over the
    threshold. Cells with no layer at or above threshold — including
    empty cells — are NaN.

    The pair spans the qualifying layers: CBH is the *bottom* of the
    lowest layer at or above threshold and canopy height the *top* of the
    highest, so ``[cbh, canopy_height]`` is exactly the union of those
    layers. That is deliberate and load-bearing, not an arbitrary
    rounding choice — it makes ``canopy_height - cbh`` the true depth of
    qualifying canopy (n layers for n qualifying layers), keeps
    ``canopy_fuel_load / (canopy_height - cbh)``
    equal to the mean bulk density over the canopy, and leaves the depth
    strictly positive when only one layer qualifies. Anchoring both ends
    the same way — FuelCalc labels every layer by its top
    (``NC_PTL.C:663-667``), and a midpoint would too — understates the
    depth by one layer and collapses it to zero in the single-layer case,
    which would divide by zero in a load-over-depth CBD.

    Both heights are then bounded by the layers that actually hold fuel:
    CBH may not fall below the lowest layer with positive density in the
    *unsmoothed* profile, and canopy height may not rise above the
    highest. Smoothing spreads density up to half a window past each end
    of the canopy, and that skirt can clear the threshold, so without the
    bounds a smoothed scan reports canopy where no crown reaches. They
    are FuelCalc's (``NC_PTL.C:677-695``) and, like it, are applied
    unconditionally — with ``smoothing_window=None`` every qualifying
    layer holds fuel by construction, so they are exactly inert.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cbh, canopy_height)`` in meters, each shape ``(ny, nx)``.
    """
    profile = np.asarray(profile, dtype=np.float64)
    n_layers = profile.shape[0]

    scanned = profile
    if smoothing_window is not None:
        scanned = profile_running_mean(
            profile,
            window_in_layers(smoothing_window, layer_depth),
            edge=smoothing_edge,
        )

    profile_max = scanned.max(axis=0)
    if relative_fraction is not None:
        effective = np.minimum(relative_fraction * profile_max, threshold)
    else:
        effective = np.full_like(profile_max, threshold)
    # An all-zero cell has effective threshold 0 under the relative rule;
    # require positive density so empty cells read as no-canopy.
    qualifies = (scanned >= effective) & (scanned > 0.0)

    any_qualifies = qualifies.any(axis=0)
    lowest = qualifies.argmax(axis=0)
    highest = n_layers - 1 - qualifies[::-1].argmax(axis=0)
    cbh = np.where(any_qualifies, lowest * layer_depth, np.nan)
    canopy_height = np.where(any_qualifies, (highest + 1) * layer_depth, np.nan)

    # Intersect the qualifying span with the unsmoothed extent of the
    # canopy. Cells with no fuel have no qualifying layer either, so
    # their NaN survives the maximum/minimum untouched.
    lowest_fuel, highest_fuel = _fuel_extent(profile, layer_depth)
    cbh = np.maximum(cbh, lowest_fuel)
    canopy_height = np.minimum(canopy_height, highest_fuel)
    # The two spans can be disjoint: truncating the smoothing window at
    # the profile ends inflates the outermost layers, so a layer holding
    # no fuel can clear the threshold when every layer that does hold
    # fuel falls short. FuelCalc clamps each end independently and lets
    # the pair cross; an empty intersection is no canopy, so say so.
    empty = cbh >= canopy_height
    cbh = np.where(empty, np.nan, cbh)
    canopy_height = np.where(empty, np.nan, canopy_height)
    return cbh, canopy_height
