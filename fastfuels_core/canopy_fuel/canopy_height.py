"""Canopy base height and canopy height (m), from a bulk-density threshold.

Both are read off the same vertical profile with the same scan, so they
are produced together: CBH is the bottom of the lowest layer clearing
the threshold and canopy height the top of the highest.
"""

from __future__ import annotations

import numpy as np

from fastfuels_core.canopy_fuel.profile import FT_TO_M


def _centered_running_mean(profile: np.ndarray, window_layers: int) -> np.ndarray:
    """Running mean over the layer axis, truncated at the profile ends.

    The denominator is the number of layers actually averaged, so the
    outermost layers are means of a shorter window rather than being
    diluted by padding. That is FFE-FVS's smoothing, and the skirt it
    spreads past the canopy is what the extent bounds below undo.
    """
    n_layers = profile.shape[0]
    cumsum = np.concatenate(
        [np.zeros((1, *profile.shape[1:])), np.cumsum(profile, axis=0)], axis=0
    )
    k = np.arange(n_layers)
    lo = np.clip(k - (window_layers - 1) // 2, 0, n_layers)
    hi = np.clip(k + window_layers // 2 + 1, 0, n_layers)
    return (cumsum[hi] - cumsum[lo]) / (hi - lo).reshape(
        -1, *([1] * (profile.ndim - 1))
    )


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
    layer_depth: float = FT_TO_M,
    threshold: float = 0.012,
    relative_fraction: float | None = 0.1,
    smoothing_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Canopy base height and canopy height from a bulk-density threshold.

    Per cell, the effective threshold is ``min(relative_fraction *
    profile_max, threshold)`` (FuelCalc's rule; ``relative_fraction=None``
    uses the flat threshold alone). ``smoothing_window`` (m) optionally
    smooths the profile first with a centered running mean truncated at
    the profile ends (FFE-FVS uses 0.9144 m). Cells with no layer at or
    above threshold — including empty cells — are NaN.

    The pair spans the qualifying layers: CBH is the *bottom* of the
    lowest layer at or above threshold and canopy height the *top* of the
    highest, so ``[cbh, chm]`` is exactly the union of those layers. That
    is deliberate and load-bearing, not an arbitrary rounding choice —
    it makes ``chm - cbh`` the true depth of qualifying canopy (n layers
    for n qualifying layers), keeps ``canopy_fuel_load / (chm - cbh)``
    equal to the mean bulk density over the canopy, and leaves the depth
    strictly positive when only one layer qualifies. Anchoring both ends
    the same way — FuelCalc labels every layer by its top
    (``NC_PTL.C:663-667``), and a midpoint would too — understates the
    depth by one layer and collapses it to zero in the single-layer case,
    which would divide by zero in a load-over-depth CBD. The consequence
    is that our CBH sits one layer below FuelCalc's on the same profile;
    see ``tests/canopy_fuel/test_fuelcalc_parity.py``.

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
        ``(cbh, chm)`` in meters, each shape ``(ny, nx)``.
    """
    profile = np.asarray(profile, dtype=np.float64)
    n_layers = profile.shape[0]

    scanned = profile
    if smoothing_window is not None:
        scanned = _centered_running_mean(
            profile, max(1, int(round(smoothing_window / layer_depth)))
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
    chm = np.where(any_qualifies, (highest + 1) * layer_depth, np.nan)

    # Intersect the qualifying span with the unsmoothed extent of the
    # canopy. Cells with no fuel have no qualifying layer either, so
    # their NaN survives the maximum/minimum untouched.
    lowest_fuel, highest_fuel = _fuel_extent(profile, layer_depth)
    cbh = np.maximum(cbh, lowest_fuel)
    chm = np.minimum(chm, highest_fuel)
    # The two spans can be disjoint: truncating the smoothing window at
    # the profile ends inflates the outermost layers, so a layer holding
    # no fuel can clear the threshold when every layer that does hold
    # fuel falls short. FuelCalc clamps each end independently and lets
    # the pair cross; an empty intersection is no canopy, so say so.
    empty = cbh >= chm
    cbh = np.where(empty, np.nan, cbh)
    chm = np.where(empty, np.nan, chm)
    return cbh, chm
