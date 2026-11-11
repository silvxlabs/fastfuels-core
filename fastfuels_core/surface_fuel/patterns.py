"""Spatial patterns: give each cell a percentile.

Noise is drawn per band, mixed across bands for correlation, smoothed with
a Gaussian filter for patches sized by each band's characteristic
patch size, blended with per-cell speckle, then ranked.
The percentiles set where values land, not what they are.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter


def band_percentiles(
    shape: tuple[int, int],
    sigmas: list[float],
    speckles: list[float],
    factor: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Give every cell of every band a percentile, setting the spatial pattern.

    The noise is mixed before smoothing, so each band's patch size follows
    its own sigma. Bands with different sigmas therefore come out less
    correlated than the matrix says.

    Parameters
    ----------
    shape : tuple of int
        Grid ``(height, width)``.
    sigmas : list of float
        Smoothing sigma per band, in cells, in matrix order. 0 means no
        smoothing.
    speckles : list of float
        Share of each band's variation that is independent per cell, from 0
        to 1, in matrix order. Only bands with a sigma use it.
    factor : numpy.ndarray
        Mixing factor from
        ``mixing_factor``.
    rng : numpy.random.Generator
        Source of the noise.

    Returns
    -------
    list of numpy.ndarray
        One array of ``shape`` per band, with values between 0 and 1.
    """
    height, width = shape
    padding = math.ceil(4 * max(sigmas))
    padded_shape = (height + 2 * padding, width + 2 * padding)
    window = (
        slice(padding, padding + height),
        slice(padding, padding + width),
    )

    noise = rng.standard_normal((len(sigmas), *padded_shape))
    mixed = np.tensordot(factor, noise, axes=1)
    del noise

    percentiles = []
    for band_noise, sigma, speckle in zip(mixed, sigmas, speckles):
        field = band_noise[window]
        if sigma > 0:
            smooth = gaussian_filter(band_noise, sigma)[window] / _kernel_std(sigma)
            field = math.sqrt(1 - speckle) * smooth + math.sqrt(speckle) * field
        percentiles.append(rank_percentiles(field))

    return percentiles


def _kernel_std(sigma: float) -> float:
    """Return the std of unit-variance noise after Gaussian smoothing."""
    radius = math.ceil(4 * sigma)
    impulse = np.zeros((2 * radius + 1, 2 * radius + 1))
    impulse[radius, radius] = 1
    return math.sqrt(np.sum(gaussian_filter(impulse, sigma) ** 2))


def rank_percentiles(field: np.ndarray) -> np.ndarray:
    """Replace each cell with its percentile among all cells in the grid."""
    ranks = stats.rankdata(field).reshape(field.shape)
    return (ranks - 0.5) / field.size
