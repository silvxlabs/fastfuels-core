"""Canopy bulk density (kg/m**3), reduced from the vertical profile."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FT_TO_M

SLAB_EDGE = "slab"
FUELCALC_EDGE = "fuelcalc"
TRUNCATE_EDGE = "truncate"
VALID_EDGES = (SLAB_EDGE, FUELCALC_EDGE, TRUNCATE_EDGE)


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
    value behaves as the next odd one.

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
    layer_depth: float = FT_TO_M,
    window: float | None = 5 * FT_TO_M,
    edge: str = FUELCALC_EDGE,
) -> np.ndarray:
    """Canopy bulk density: per-cell maximum running mean of the profile.

    ``window`` is the running-mean depth in meters (Reinhardt et al.
    2006 use 3.0 m; FuelCalc's guide states 5 ft in one place and no
    smoothing in another). ``window=None`` skips smoothing and returns
    the maximum single layer.

    The mean is over a slab of fixed depth, so the denominator is the
    window depth at every height, including against the ground. Layers
    outside the profile contribute zero at both ends: a profile
    shallower than the window is zero-padded above, and a canopy resting
    on layer 0 is diluted over the full window just as one higher up is.
    That is what Reinhardt et al.'s "any 3-m deep layer" measures, and
    it makes CBD invariant to how high the canopy sits — the same slab
    of fuel has the same bulk density wherever it is.

    ``edge`` selects that convention; see :func:`profile_running_mean`.
    FuelCalc reads its CBD off the same ground-clamped running mean it
    scans for the canopy heights, so ``edge="fuelcalc"`` is the setting
    that reproduces it.

    Returns
    -------
    numpy.ndarray
        CBD (kg/m**3), shape ``(ny, nx)``.
    """
    if window is None:
        return profile.max(axis=0)
    w = max(1, int(round(window / layer_depth)))
    n_layers = profile.shape[0]
    if n_layers < w:
        pad = np.zeros((w - n_layers, *profile.shape[1:]), dtype=profile.dtype)
        profile = np.concatenate([profile, pad], axis=0)
    # An even window has no centre layer; keep the historical forward
    # slab for "slab" so the reduction stays a pure fixed-depth maximum.
    if edge == SLAB_EDGE:
        cumsum = np.concatenate(
            [np.zeros((1, *profile.shape[1:])), np.cumsum(profile, axis=0)], axis=0
        )
        return ((cumsum[w:] - cumsum[:-w]) / w).max(axis=0)
    return profile_running_mean(profile, w, edge=edge).max(axis=0)
