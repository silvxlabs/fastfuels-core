"""Canopy bulk density (kg/m**3), reduced from the vertical profile."""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FT_TO_M


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

    The mean is over a slab of fixed depth, so the denominator is the
    window depth at every height, including against the ground. Layers
    outside the profile contribute zero at both ends: a profile
    shallower than the window is zero-padded above, and a canopy resting
    on layer 0 is diluted over the full window just as one higher up is.
    That is what Reinhardt et al.'s "any 3-m deep layer" measures, and
    it makes CBD invariant to how high the canopy sits — the same slab
    of fuel has the same bulk density wherever it is.

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
    cumsum = np.concatenate(
        [np.zeros((1, *profile.shape[1:])), np.cumsum(profile, axis=0)], axis=0
    )
    means = (cumsum[w:] - cumsum[:-w]) / w
    return means.max(axis=0)
