"""Surface fuel grids filled from a distribution per band.

Each band's grid is filled with values from its requested distribution. Two
separate steps build it: ``patterns`` decides each cell's percentile,
which sets the spatial pattern and the correlations, and
``distributions`` turns percentiles into values.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from fastfuels_core.surface_fuel.correlation import correlation_matrix, mixing_factor
from fastfuels_core.surface_fuel.distributions import (
    build_distribution,
    is_number,
    percentiles_to_values,
)
from fastfuels_core.surface_fuel.patterns import band_percentiles

# Largest allowed characteristic patch size, in cells. Run time and memory
# grow quickly with it.
_MAX_PATCH_SIZE_CELLS = 500


def build_distribution_grids(
    shape: tuple[int, int],
    resolution: float,
    bands: dict,
    correlations: list[dict] | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Fill a grid per band with random values from that band's distribution.

    Parameters
    ----------
    shape : tuple of int
        Grid ``(height, width)``.
    resolution : float
        Cell size in the domain's CRS units, normally meters.
    bands : dict
        Band key -> ``{"distribution": {...}, "patch_size": float,
        "speckle": float}``.

        ``distribution`` holds ``"type"`` and that type's settings:

        - ``uniform``: ``low``, ``high``, with ``0 <= low < high``.
        - ``normal``: ``mean``, ``std``, both above 0.
        - ``lognormal``: ``mean``, ``std``, both above 0. These are the mean
          and std of the real values, not of their logs.
        - ``triangular``: ``low``, ``mode``, ``high``, with
          ``0 <= low <= mode <= high`` and ``low < high``.

        ``patch_size`` is the band's characteristic patch size: the distance
        at which similarity between cells falls to about 37%, in the same
        units as ``resolution``. It is optional and defaults to 0, which
        means independent cells with no patches. ``patch_size / resolution``
        can't exceed 500.

        ``speckle`` is the share of each cell's variation that is independent
        of its neighbors, from 0 to 1. It is optional and defaults to 0,
        which gives fully smooth patches. 1 gives no patches at all.
    correlations : list of dict, optional
        Pairs of ``{"bands": [a, b], "coefficient": x}``, where ``x`` is a
        rank (Spearman) correlation strictly between -1 and 1. See Notes for
        how unlisted pairs behave.
    seed : int, optional
        Non-negative integer that makes the result repeatable. ``None`` means
        the result isn't repeatable.

    Returns
    -------
    dict
        Band key -> float32 array of ``shape``.

    Raises
    ------
    ValueError
        A band, distribution, correlation, grid setting or seed is invalid,
        or the correlation coefficients contradict each other.

    Notes
    -----
    **Characteristic patch size** is the distance at which similarity between
    cells falls to about 37%, or to about ``(1 - speckle)`` times 37% with
    speckle. The fall-off is soft (Gaussian shape), so
    patches have no hard edges. A characteristic patch size that is large
    compared with the grid gives a smooth gradient across the full range of
    values, not one flat patch.

    **Speckle** adds independent per-cell variation, so patches aren't
    perfectly smooth. It limits similarity between neighboring cells to about
    ``1 - speckle`` when the characteristic patch size spans many cells, and to
    less for small ones. It has no effect on a band with no patch size, whose
    cells are already independent.

    **Distributions** are matched by ranking the cells, so patches and
    correlations only decide where values land. Each band's histogram closely
    matches the request whatever the patch size. The exception is ``normal``,
    where negative values are replaced with 0. That creates exact zeros and
    shifts the real mean and std. Many zeros also lower rank
    correlation slightly.

    **Correlations** are averages, not exact values for one grid. Chance can
    move a pair's correlation away from its target, including pairs with no
    relationship, and more so as patches get larger compared with the grid.

    Unlisted pairs are not forced to 0. Bands linked through another band
    become related, for example A-B and B-C at 0.6 give A-C near 0.36. Bands
    with no link get independent noise. List a pair with coefficient 0 to
    keep it unrelated.

    Coefficients that contradict each other slightly, as correlations measured
    from real data can, are nudged by at most 0.02 with a warning. Larger
    contradictions raise ValueError.

    Each band's patches follow its own characteristic patch size, so bands
    with different patch sizes come out less correlated than requested. With
    the same patch size, the correlation matches. Rough share of the requested
    correlation with no speckle: patch sizes differing 2x, about 80%; 4x,
    about 50%; 10x, about 20%. Speckle raises these slightly. A band with no
    patch size keeps only about ``sqrt(speckle)`` of its correlation with a
    patchy band, for example about 30% at a speckle of 0.1 and almost none at
    0.

    **Repeatability**: the same request with the same library versions gives
    identical output, whatever order the bands are listed in. Adding or
    removing a band, or changing the set of patch sizes, changes the noise and
    so the grids.
    """
    _validate_grid(shape, resolution)
    _validate_seed(seed)
    if not bands:
        raise ValueError("bands is empty.")

    keys = sorted(bands)
    distributions = [_band_distribution(key, bands[key]) for key in keys]
    sigmas = [_band_sigma(key, bands[key], resolution) for key in keys]
    speckles = [_band_speckle(key, bands[key]) for key in keys]
    factor = mixing_factor(correlation_matrix(keys, correlations))

    rng = np.random.default_rng(seed)
    percentiles = band_percentiles(shape, sigmas, speckles, factor, rng)

    return {
        key: percentiles_to_values(cell_percentiles, distribution)
        for key, cell_percentiles, distribution in zip(keys, percentiles, distributions)
    }


def _validate_grid(shape: tuple[int, int], resolution: float) -> None:
    """Raise ValueError unless shape and resolution describe a real grid."""
    if len(shape) != 2 or not all(
        isinstance(size, Integral) and size > 0 for size in shape
    ):
        raise ValueError(f"shape must be two positive integers, got {shape!r}.")

    if not is_number(resolution) or resolution <= 0:
        raise ValueError(f"resolution must be above 0, got {resolution!r}.")


def _validate_seed(seed: int | None) -> None:
    """Raise ValueError unless seed is None or a non-negative integer."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")


def _band_distribution(key, band: dict):
    """Validate a band's settings and return its SciPy distribution."""
    unknown = set(band) - {"distribution", "patch_size", "speckle"}
    if unknown:
        raise ValueError(f"Band {key!r} has unknown settings: {sorted(unknown)}.")

    spec = band.get("distribution")
    if not isinstance(spec, dict):
        raise ValueError(f"Band {key!r} needs a distribution dict, got {spec!r}.")

    return build_distribution(spec)


def _band_sigma(key, band: dict, resolution: float) -> float:
    """Validate a band's patch size and return its smoothing sigma in cells."""
    patch_size = band.get("patch_size", 0)
    if not is_number(patch_size) or patch_size < 0:
        raise ValueError(
            f"Band {key!r} patch_size must be 0 or above, got {patch_size!r}."
        )

    if patch_size / resolution > _MAX_PATCH_SIZE_CELLS:
        raise ValueError(
            f"Band {key!r} patch_size is {patch_size / resolution:g} cells, "
            f"above the limit of {_MAX_PATCH_SIZE_CELLS}."
        )

    return patch_size / 2 / resolution


def _band_speckle(key, band: dict) -> float:
    """Validate a band's speckle and return it."""
    speckle = band.get("speckle", 0)
    if not is_number(speckle) or not 0 <= speckle <= 1:
        raise ValueError(
            f"Band {key!r} speckle must be between 0 and 1, got {speckle!r}."
        )
    return speckle
