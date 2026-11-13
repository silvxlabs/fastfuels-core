"""The correlation matrix between bands, and the factor that mixes their noise.

Listed pairs are rank (Spearman) correlations. Each one is converted with
``2 * sin(pi * coefficient / 6)`` to the correlation the underlying normal
noise needs, so the final grids have the requested rank correlation.
Unlisted pairs are filled with the values the listed ones imply, and
coefficients that contradict each other slightly are nudged to fit.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from fastfuels_core.surface_fuel.distributions import is_number

# Filling in unlisted correlations stops once every listed pair matches within
# the tolerance, and gives up after this many passes over the pairs.
_FILL_TOLERANCE = 1e-9
_MAX_FILL_PASSES = 10000

# Eigenvalues slightly below 0 are rounding error, not a contradiction.
_EIGENVALUE_TOLERANCE = 1e-8

# Slightly contradictory coefficients are nudged until they fit together, as
# long as none moves more than this. Eigenvalues are raised to the floor.
_MAX_NUDGE = 0.02
_NUDGE_EIGENVALUE = 1e-4


def correlation_matrix(keys: list, correlations: list[dict] | None) -> np.ndarray:
    """Build the matrix of correlations between the bands' normal noise.

    Parameters
    ----------
    keys : list
        Band keys, in sorted order. They set the matrix's row order.
    correlations : list of dict or None
        Pairs of ``{"bands": [a, b], "coefficient": x}``, as rank correlations.

    Returns
    -------
    numpy.ndarray
        Square matrix with one row per band. Listed pairs hold their
        coefficient converted to the correlation of the underlying normal
        noise, and unlisted pairs hold the value the other bands imply, which
        is 0 for bands with no link.

    Raises
    ------
    ValueError
        A pair is invalid or repeated, or the coefficients contradict each
        other.
    """
    index = {key: i for i, key in enumerate(keys)}
    matrix = np.eye(len(keys))
    listed = np.eye(len(keys), dtype=bool)

    for pair in correlations or []:
        bands = pair.get("bands", ())
        if len(bands) != 2:
            raise ValueError(f"A correlation needs exactly two bands, got {bands!r}.")

        a, b = bands
        for band in (a, b):
            if band not in index:
                raise ValueError(f"Correlation names an unknown band: {band!r}")
        if a == b:
            raise ValueError(f"Band {a!r} can't be correlated with itself.")

        i, j = index[a], index[b]
        if listed[i, j]:
            raise ValueError(f"The pair {a!r}, {b!r} is listed twice.")

        coefficient = pair.get("coefficient")
        if not is_number(coefficient) or not -1 < coefficient < 1:
            raise ValueError(
                f"Coefficient for {a!r}, {b!r} must be between -1 and 1, "
                f"got {coefficient!r}."
            )

        matrix[i, j] = matrix[j, i] = 2 * math.sin(math.pi * coefficient / 6)
        listed[i, j] = listed[j, i] = True

    _fill_unlisted(matrix, listed)

    if np.linalg.eigvalsh(matrix)[0] < -_EIGENVALUE_TOLERANCE:
        return _nudge_coefficients(matrix)

    return matrix


def _fill_unlisted(matrix: np.ndarray, listed: np.ndarray) -> None:
    """Fill unlisted pairs in place with the values the other bands imply.

    This is maximum-determinant completion, found by iterative proportional
    fitting. It adjusts the matrix's inverse until the listed pairs match,
    which leaves every unlisted pair at the value the links imply. Listed
    pairs never change.

    Raises
    ------
    ValueError
        The values never settle.
    """
    if listed.all():
        return

    size = len(matrix)
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size) if listed[i, j]]
    precision = np.eye(size)

    try:
        for _ in range(_MAX_FILL_PASSES):
            largest_miss = 0.0
            for i, j in pairs:
                block = np.ix_([i, j], [i, j])
                current = np.linalg.inv(precision)
                miss = np.abs(current[block] - matrix[block]).max()
                largest_miss = max(largest_miss, miss)
                precision[block] += np.linalg.inv(matrix[block]) - np.linalg.inv(
                    current[block]
                )
            if largest_miss < _FILL_TOLERANCE:
                filled = np.linalg.inv(precision)
                matrix[~listed] = ((filled + filled.T) / 2)[~listed]
                return
    except np.linalg.LinAlgError:
        pass

    raise ValueError(
        "Unlisted correlations can't be filled in from the listed coefficients."
    )


def _nudge_coefficients(matrix: np.ndarray) -> np.ndarray:
    """Make a slightly contradictory matrix valid, warning about the change.

    Correlations measured from real data can miss fitting together by a tiny
    amount, especially when they are high or the data has many tied values.

    Raises
    ------
    ValueError
        A coefficient would have to move more than ``_MAX_NUDGE``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    raised = np.clip(eigenvalues, _NUDGE_EIGENVALUE, None)
    nudged = (eigenvectors * raised) @ eigenvectors.T
    scale = np.sqrt(np.diag(nudged))
    nudged = nudged / np.outer(scale, scale)

    change = 6 / np.pi * np.abs(np.arcsin(nudged / 2) - np.arcsin(matrix / 2))
    largest = change.max()
    if largest > _MAX_NUDGE:
        raise ValueError("The correlation coefficients contradict each other.")

    warnings.warn(
        f"Correlations were adjusted by up to {largest:.4f} so they fit together.",
        stacklevel=3,
    )
    return nudged


def mixing_factor(matrix: np.ndarray) -> np.ndarray:
    """Factor the correlation matrix, for mixing independent noise.

    Cholesky fails on a singular matrix, which a coefficient near 1 can
    produce, so that case falls back on the eigen-decomposition.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))
