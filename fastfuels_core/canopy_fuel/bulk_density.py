"""Canopy bulk density (kg/m**3), reduced from the vertical profile."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FUELCALC_LAYER_DEPTH

SLAB_EDGE = "slab"
FUELCALC_EDGE = "fuelcalc"
TRUNCATE_EDGE = "truncate"
VALID_EDGES = (SLAB_EDGE, FUELCALC_EDGE, TRUNCATE_EDGE)

# CBD reduction methods. Only the maximum running mean is built (see
# :func:`cbd_running_mean`); ``load_over_depth`` is named in the FastFuels
# API schema but not yet implemented, so it is held apart from an unknown
# string to keep a recognized-but-unbuilt method (NotImplementedError)
# distinct from a typo (ValueError). See fastfuels-core#97.
MAX_RUNNING_MEAN_METHOD = "maximum_running_mean"
CBD_METHODS = (MAX_RUNNING_MEAN_METHOD,)
UNIMPLEMENTED_CBD_METHODS = ("load_over_depth",)


def validate_cbd_method(method: str) -> None:
    """Split an unbuilt CBD method arm from an unknown one.

    ``NotImplementedError`` for a method the API schema defines but this
    package has not built yet, ``ValueError`` for an unrecognized name.
    """
    if method in UNIMPLEMENTED_CBD_METHODS:
        raise NotImplementedError(
            f"cbd method {method!r} is defined in the FastFuels API schema "
            f"but is not yet implemented in fastfuels-core; implemented: "
            f"{list(CBD_METHODS)}."
        )
    if method not in CBD_METHODS:
        raise ValueError(
            f"Unknown cbd method {method!r}; expected one of {list(CBD_METHODS)}."
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
