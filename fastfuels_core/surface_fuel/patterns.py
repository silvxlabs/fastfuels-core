"""Spatial patterns: give each cell a percentile.

Noise is drawn per band, mixed across bands for correlation, smoothed with
a Gaussian filter for patches sized by each band's characteristic
patch size, blended with per-cell speckle, then ranked.
The percentiles set where values land, not what they are.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def band_percentiles(
    shape: tuple[int, int],
    sigmas: list[float],
    speckles: list[float],
    factor: np.ndarray,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Give every cell of every band a percentile, setting the spatial pattern.

    The noise is mixed before smoothing, so each band's patch size follows
    its own sigma. Bands with different sigmas therefore come out less
    correlated than the matrix says.

    Bands are yielded one at a time, so only one band's percentiles are held
    at once. The noise is float32 to halve its memory.

    Parameters
    ----------
    shape : tuple of int
        Grid ``(height, width)``.
    sigmas : list of float
        Smoothing sigma per band, in cells, in matrix order. 0 means no
        smoothing.
    speckles : list of float
        How much of each band's unsmoothed noise is blended back into its
        patches, from 0 to 1, in matrix order. Only bands with a sigma use it.
    factor : numpy.ndarray
        Lower-triangular mixing factor from ``mixing_factor``.
    rng : numpy.random.Generator
        Source of the noise.

    Yields
    ------
    numpy.ndarray
        One float64 array of ``shape`` per band, in matrix order, with values
        between 0 and 1.
    """
    height, width = shape
    padding = math.ceil(4 * max(sigmas))
    padded_shape = (height + 2 * padding, width + 2 * padding)
    window = (
        slice(padding, padding + height),
        slice(padding, padding + width),
    )

    noise = rng.standard_normal((len(sigmas), *padded_shape), dtype=np.float32)
    _mix_in_place(noise, factor)

    for band_noise, sigma, speckle in zip(noise, sigmas, speckles):
        field = band_noise[window]
        if sigma > 0:
            smooth = gaussian_filter(band_noise, sigma)[window] / _kernel_std(sigma)
            field = math.sqrt(1 - speckle) * smooth + math.sqrt(speckle) * field
        yield rank_percentiles(field)


def _mix_in_place(noise: np.ndarray, factor: np.ndarray) -> None:
    """Mix independent noise across bands, overwriting it.

    The factor is lower triangular, so each band needs only its own and
    earlier bands' noise. Working from the last band back leaves those
    untouched until they are mixed themselves, so no second copy is needed.
    """
    for i in reversed(range(len(noise))):
        noise[i] *= np.float32(factor[i, i])
        for j in range(i):
            if factor[i, j]:
                noise[i] += np.float32(factor[i, j]) * noise[j]


def _kernel_std(sigma: float) -> float:
    """Return the std of unit-variance noise after Gaussian smoothing."""
    radius = math.ceil(4 * sigma)
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1
    # The 2D kernel is the 1D kernel times itself, so its sum of squares is
    # the 1D sum of squares, squared.
    return float(np.sum(gaussian_filter1d(impulse, sigma) ** 2))


def rank_percentiles(field: np.ndarray) -> np.ndarray:
    """Replace each cell with its percentile among all cells in the grid."""
    percentiles = stats.rankdata(field).reshape(field.shape)
    percentiles -= 0.5
    percentiles /= field.size
    return percentiles


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the rank correlation between two grids of percentiles.

    Percentiles from ``rank_percentiles`` always have mean 0.5 and the same
    variance, so their correlation needs only one pass over the cells.
    """
    size = first.size
    mean_product = np.dot(first.ravel(), second.ravel()) / size
    variance = (size * size - 1) / (12 * size * size)
    return float((mean_product - 0.25) / variance)
