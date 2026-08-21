"""Canopy bulk density (kg/m**3), reduced from the vertical profile."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FUELCALC_LAYER_DEPTH

SLAB_EDGE = "slab"
FUELCALC_EDGE = "fuelcalc"
TRUNCATE_EDGE = "truncate"
VALID_EDGES = (SLAB_EDGE, FUELCALC_EDGE, TRUNCATE_EDGE)

# CBD reduction methods. The maximum running mean (see
# :func:`cbd_running_mean`) is the effective-CBD convention; load over
# depth (see :func:`cbd_load_over_depth`) is the van Wagner average
# density, canopy fuel load divided by one of the canopy depths below.
MAX_RUNNING_MEAN_METHOD = "maximum_running_mean"
LOAD_OVER_DEPTH_METHOD = "load_over_depth"
CBD_METHODS = (MAX_RUNNING_MEAN_METHOD, LOAD_OVER_DEPTH_METHOD)

# Canopy depths that divide the fuel load under ``load_over_depth``. Two
# are per-tree cell statistics and live in :mod:`canopy_height`
# (:func:`~fastfuels_core.canopy_fuel.canopy_height.mean_crown_length`,
# :func:`~fastfuels_core.canopy_fuel.canopy_height.height_percentile_depth`);
# ``canopy_depth`` is the threshold ``chm - cbh`` and ``biomass_percentile``
# is :func:`biomass_percentile_depth` below.
CANOPY_DEPTH = "canopy_depth"
MEAN_CROWN_LENGTH_DEPTH = "mean_crown_length"
BIOMASS_PERCENTILE_DEPTH = "biomass_percentile"
HEIGHT_PERCENTILE_DEPTH = "height_percentile"
CBD_DEPTHS = (
    CANOPY_DEPTH,
    MEAN_CROWN_LENGTH_DEPTH,
    BIOMASS_PERCENTILE_DEPTH,
    HEIGHT_PERCENTILE_DEPTH,
)

# The central biomass fraction Albini (1996) spans for the depth: the
# 10th to the 90th percentile of cumulative canopy biomass by height.
BIOMASS_LOWER_FRACTION = 0.10
BIOMASS_UPPER_FRACTION = 0.90


def validate_cbd_method(method: str) -> None:
    if method not in CBD_METHODS:
        raise ValueError(
            f"Unknown cbd method {method!r}; expected one of {list(CBD_METHODS)}."
        )


def validate_cbd_depth(depth: str) -> None:
    if depth not in CBD_DEPTHS:
        raise ValueError(
            f"Unknown cbd depth {depth!r}; expected one of {list(CBD_DEPTHS)}."
        )


def window_in_layers(window: float, layer_depth: float) -> int:
    """Resolve a running-mean depth (m) to an odd number of layers.

    A centred window needs a centre layer, so the depth is rounded to
    the nearest layer count and an even count is widened by one. Every
    caller resolves through here, so the same ``window`` spans the same
    layers under every edge convention: 3.0 m over 0.3048 m layers is
    11 layers (3.35 m), not 10 under one rule and 11 under another.
    """
    n = max(1, int(round(window / layer_depth)))
    return n if n % 2 else n + 1


def profile_running_mean(
    profile: np.ndarray, window_layers: int, edge: str = SLAB_EDGE
) -> np.ndarray:
    """Running mean down the layer axis, by one of three edge conventions.

    A running mean is only defined once you say what lies past the ends
    of the profile, and the three answers in use disagree by enough to
    move a reported height by a layer, so the choice is named rather
    than assumed.

    ``"slab"``
        Fixed denominator throughout: layers outside the profile
        contribute zero at both ends. The mean is over a slab of fixed
        depth wherever it sits, which is what Reinhardt et al. (2006)'s
        "any 3-m deep layer" measures and what keeps CBD invariant to
        how high the canopy sits.
    ``"fuelcalc"``
        FuelCalc's ``_BulkDensity`` (``NC_PTL.C:626-636``): the window
        is clamped at the ground, and the denominator shrinks with it,
        but it is *not* clamped at the top, where the profile is
        zero-padded against a full denominator. Density therefore falls
        away above the canopy and is concentrated against the ground.
    ``"truncate"``
        Denominator is the number of layers actually averaged, at both
        ends. FFE-FVS smooths this way. It inflates the topmost layers,
        which can carry a height threshold a layer past where the same
        profile puts it under the other two.

    ``window_layers`` is FuelCalc's spread: the window spans
    ``window_layers // 2`` layers either side of the centre, so an even
    value behaves as the next odd one. :func:`window_in_layers` resolves a
    depth in metres to this count.

    Returns
    -------
    numpy.ndarray
        Same shape as ``profile``.
    """
    if edge not in VALID_EDGES:
        raise ValueError(f"Unknown edge {edge!r}; expected one of {list(VALID_EDGES)}.")
    n_layers = profile.shape[0]
    half = max(0, int(window_layers) // 2)
    if half == 0:
        return profile
    cumsum = np.concatenate(
        [np.zeros((1, *profile.shape[1:])), np.cumsum(profile, axis=0)], axis=0
    )
    k = np.arange(n_layers)
    lo = np.clip(k - half, 0, n_layers)
    hi = np.clip(k + half + 1, 0, n_layers)
    total = cumsum[hi] - cumsum[lo]
    if edge == TRUNCATE_EDGE:
        denominator = (hi - lo).astype(np.float64)
    elif edge == FUELCALC_EDGE:
        # Clamped at the ground and counted short there; unclamped above,
        # so the top of the profile divides by the full window.
        denominator = (k + half + 1 - lo).astype(np.float64)
    else:
        denominator = np.full(n_layers, 2 * half + 1, dtype=np.float64)
    return total / denominator.reshape(-1, *([1] * (profile.ndim - 1)))


def cbd_running_mean(
    profile: np.ndarray,
    *,
    layer_depth: float = FUELCALC_LAYER_DEPTH,
    window: float | None = 5 * FUELCALC_LAYER_DEPTH,
    edge: str = FUELCALC_EDGE,
) -> np.ndarray:
    """Canopy bulk density: per-cell maximum running mean of the profile.

    ``window`` is the running-mean depth in meters, resolved to an odd
    layer count by :func:`window_in_layers` (Reinhardt et al. 2006 use
    3.0 m; FuelCalc's guide states 5 ft in one place and no smoothing in
    another). ``window=None`` skips smoothing and returns the maximum
    single layer.

    ``edge`` says what lies past the ends of the profile; see
    :func:`profile_running_mean`. The default, ``"fuelcalc"``, is the
    ground-clamped mean FuelCalc reads its CBD off, so it reproduces
    FuelCalc. ``"slab"`` is the fixed-depth reading of Reinhardt et
    al.'s "any 3-m deep layer": the denominator is the window depth at
    every height, including against the ground, so the same slab of
    fuel reports the same bulk density wherever it sits.

    Returns
    -------
    numpy.ndarray
        CBD (kg/m**3), shape ``(ny, nx)``.
    """
    if window is None:
        return profile.max(axis=0)
    w = window_in_layers(window, layer_depth)
    return profile_running_mean(profile, w, edge=edge).max(axis=0)


def cbd_load_over_depth(fuel_load: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Canopy bulk density as canopy fuel load over a canopy depth.

    The van Wagner average-density convention: ``cfl / depth``
    (kg/m**2 over m, giving kg/m**3). Cells with no positive depth —
    empty cells, whose ``depth`` is NaN, and any degenerate cell whose
    depth came out non-positive — report zero density, matching the
    no-canopy convention CBD uses elsewhere.

    Returns
    -------
    numpy.ndarray
        CBD (kg/m**3), the shape of ``depth``.
    """
    fuel_load = np.asarray(fuel_load, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(depth > 0.0, fuel_load / depth, 0.0)


def biomass_percentile_depth(
    profile: np.ndarray,
    *,
    layer_depth: float = FUELCALC_LAYER_DEPTH,
    lower: float = BIOMASS_LOWER_FRACTION,
    upper: float = BIOMASS_UPPER_FRACTION,
) -> np.ndarray:
    """Height span holding the central canopy biomass, per cell (m).

    The depth between the heights where cumulative canopy biomass reaches
    ``lower`` and ``upper`` of the column total (10% and 90% by default,
    Albini's central 80%). Biomass accumulates linearly with height
    within a layer, so each crossing height is interpolated inside the
    layer that carries it rather than snapped to a layer edge. Cells with
    no fuel are NaN.

    Returns
    -------
    numpy.ndarray
        Depth (m), shape ``(ny, nx)``.
    """
    profile = np.asarray(profile, dtype=np.float64)
    total = profile.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = profile / total
    cumulative = np.cumsum(fraction, axis=0)
    below = cumulative - fraction
    rows, cols = np.indices(total.shape)

    def height_at(target):
        # The first layer whose cumulative fraction reaches the target;
        # cumulative is monotone, so argmax picks the crossing layer.
        layer = np.argmax(cumulative >= target, axis=0)
        into_layer = fraction[layer, rows, cols]
        start = below[layer, rows, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            offset = np.where(into_layer > 0, (target - start) / into_layer, 0.0)
        return (layer + offset) * layer_depth

    depth = height_at(upper) - height_at(lower)
    return np.where(total > 0.0, depth, np.nan)
