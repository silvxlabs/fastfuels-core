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
# the tolerance, and gives up after this many steps.
_FILL_TOLERANCE = 1e-9
_MAX_FILL_STEPS = 100

# Near the solution a Newton step's predicted decrease falls below the
# objective's rounding error, so the line search can't judge it. Below this
# decrease, relative to the objective, the full step is taken.
_ROUNDING_DECREASE = 1e-12

# Eigenvalues slightly below 0 are rounding error, not a contradiction.
_EIGENVALUE_TOLERANCE = 1e-8

# Slightly contradictory coefficients are nudged until they fit together, as
# long as no listed one moves more than this. Eigenvalues are raised to the
# floor, and the search for nearby coefficients that fit gives up after this
# many passes.
_MAX_NUDGE = 0.02
_NUDGE_EIGENVALUE = 1e-4
_MAX_NUDGE_PASSES = 1000

_CANT_FILL = "Unlisted correlations can't be filled in from the listed coefficients."


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
    if correlations is not None and not isinstance(correlations, (list, tuple)):
        raise ValueError(f"correlations must be a list of pairs, got {correlations!r}.")

    index = {key: i for i, key in enumerate(keys)}
    matrix = np.eye(len(keys))
    listed = np.eye(len(keys), dtype=bool)

    for pair in correlations or []:
        if not isinstance(pair, dict):
            raise ValueError(f"Each correlation must be a dict, got {pair!r}.")
        unknown = set(pair) - {"bands", "coefficient"}
        if unknown:
            raise ValueError(f"Correlation has unknown settings: {sorted(unknown)}.")

        # A string has a length too, so "ab" would otherwise pass as a, b.
        bands = pair.get("bands", ())
        if not isinstance(bands, (list, tuple)) or len(bands) != 2:
            raise ValueError(f"A correlation needs exactly two bands, got {bands!r}.")

        a, b = bands
        for band in (a, b):
            if not isinstance(band, str) or band not in index:
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

    requested = matrix.copy()
    try:
        _fill_unlisted(matrix, listed)
    except ValueError:
        # Listed coefficients that contradict each other keep the fill from
        # settling.
        return _nudge_coefficients(requested, listed)

    if np.linalg.eigvalsh(matrix)[0] < -_EIGENVALUE_TOLERANCE:
        return _nudge_coefficients(requested, listed)

    return matrix


def _fill_unlisted(matrix: np.ndarray, listed: np.ndarray) -> None:
    """Fill unlisted pairs in place with the values the other bands imply.

    This is maximum-determinant completion, found by Newton's method on the
    matrix's inverse. Starting from the identity, it adjusts the inverse,
    keeping it 0 at unlisted pairs, until the listed pairs match, which leaves
    every unlisted pair at the value the links imply. Listed pairs never
    change.

    Raises
    ------
    ValueError
        The values never settle, as when the listed coefficients contradict
        each other.
    """
    if listed.all():
        return

    # Each listed pair, and each diagonal entry, is one adjustable entry of the
    # inverse. Diagonal entries count twice in the step, matching the gradient.
    rows, cols = np.nonzero(np.triu(listed))
    weight = np.where(rows == cols, 2, 1)
    target = np.where(listed, matrix, 0.0)
    precision = np.eye(len(matrix))
    objective = _fill_objective(precision, target)

    for _ in range(_MAX_FILL_STEPS):
        filled = np.linalg.inv(precision)
        miss = (target - filled)[rows, cols]
        if np.abs(miss).max() < _FILL_TOLERANCE:
            matrix[~listed] = ((filled + filled.T) / 2)[~listed]
            return

        gradient = 2 * miss
        hessian = 2 * (
            filled[np.ix_(rows, rows)] * filled[np.ix_(cols, cols)]
            + filled[np.ix_(rows, cols)] * filled[np.ix_(cols, rows)]
        )
        try:
            step = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        decrease = -gradient @ step

        # Shorten the step until the inverse stays valid and improves enough.
        # Once the decrease is too small to measure, Newton's method has
        # reached its quadratic phase (Boyd and Vandenberghe 2004, sec. 9.5),
        # and any step that keeps the inverse valid is taken.
        negligible = decrease < _ROUNDING_DECREASE * (1 + abs(objective))
        scale = 1.0
        while True:
            trial = precision.copy()
            trial[rows, cols] += scale * step * weight
            trial[cols, rows] = trial[rows, cols]
            trial_objective = _fill_objective(trial, target)
            if trial_objective <= objective - 0.25 * scale * decrease:
                break
            if negligible and np.isfinite(trial_objective):
                break
            scale /= 2
            if scale < 1e-10:
                raise ValueError(_CANT_FILL)
        precision, objective = trial, trial_objective

    raise ValueError(_CANT_FILL)


def _fill_objective(precision: np.ndarray, target: np.ndarray) -> float:
    """Return what the fill minimizes, or infinity for an invalid inverse."""
    try:
        lower = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError:
        return np.inf
    return -2 * np.log(np.diag(lower)).sum() + np.sum(precision * target)


def _nudge_coefficients(requested: np.ndarray, listed: np.ndarray) -> np.ndarray:
    """Make a slightly contradictory matrix valid, warning about the change.

    Correlations measured from real data can miss fitting together by a tiny
    amount, especially when they are high or the data has many tied values.

    Alternates between raising the matrix's eigenvalues to a floor and putting
    the listed coefficients back, which settles on valid coefficients close to
    the listed ones. Unlisted pairs are then filled in from the nudged values.

    Raises
    ------
    ValueError
        A listed coefficient would have to move more than ``_MAX_NUDGE``, or
        the unlisted pairs can't be filled in.
    """
    fitted = requested
    for _ in range(_MAX_NUDGE_PASSES):
        eigenvalues, eigenvectors = np.linalg.eigh(fitted)
        raised = np.clip(eigenvalues, _NUDGE_EIGENVALUE, None)
        valid = (eigenvectors * raised) @ eigenvectors.T
        refitted = np.where(listed, requested, valid)
        if np.abs(refitted - fitted).max() < _FILL_TOLERANCE:
            break
        fitted = refitted

    scale = np.sqrt(np.diag(valid))
    nudged = valid / np.outer(scale, scale)
    # The product above can differ across the diagonal in the last bit.
    nudged = (nudged + nudged.T) / 2

    change = 6 / np.pi * np.abs(np.arcsin(nudged / 2) - np.arcsin(requested / 2))
    largest = change[listed].max()
    if largest > _MAX_NUDGE:
        raise ValueError("The correlation coefficients contradict each other.")

    _fill_unlisted(nudged, listed)

    # Rounding moves coefficients by far less than this when nothing was nudged.
    if largest >= 5e-5:
        warnings.warn(
            f"Correlations were adjusted by up to {largest:.4f} so they fit "
            "together.",
            stacklevel=3,
        )
    return nudged


def mixing_factor(matrix: np.ndarray) -> np.ndarray:
    """Factor the correlation matrix, for mixing independent noise.

    The factor is always lower triangular, which lets the noise be mixed in
    place. Cholesky fails on a singular matrix, which a coefficient near 1
    can produce, so that case falls back on the eigen-decomposition, rotated
    to lower triangular.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))
        # root @ root.T is the matrix, and so is R.T @ R for root.T = Q @ R.
        return np.linalg.qr(root.T)[1].T
