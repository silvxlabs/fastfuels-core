"""Distribution types, and turning each cell's percentile into a value.

Distributions are plain dicts selected by ``"type"``. Values can't be
negative: ``normal`` replaces negative values with 0, and the other types
can't produce them.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy import stats

# Each distribution type and the settings it requires.
_DISTRIBUTION_SETTINGS = {
    "uniform": {"low", "high"},
    "normal": {"mean", "std"},
    "lognormal": {"mean", "std"},
    "triangular": {"low", "mode", "high"},
}


def is_number(value) -> bool:
    """Return True for a finite real number, which excludes booleans."""
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def build_distribution(spec: dict):
    """Validate a distribution dict and return its SciPy distribution.

    Parameters
    ----------
    spec : dict
        ``"type"`` plus that type's settings. ``lognormal``'s ``mean`` and
        ``std`` describe the real values, not their logs.

    Returns
    -------
    scipy.stats.rv_frozen
        The distribution, ready for ``ppf``.

    Raises
    ------
    ValueError
        The type is unknown, a setting is missing, extra or not a finite
        number, or the settings are out of range or could give negatives.
    """
    kind = spec.get("type")
    if kind not in _DISTRIBUTION_SETTINGS:
        raise ValueError(f"Unknown distribution type: {kind!r}")

    settings = {key: value for key, value in spec.items() if key != "type"}
    expected = _DISTRIBUTION_SETTINGS[kind]
    if set(settings) != expected:
        raise ValueError(
            f"A {kind} distribution takes {sorted(expected)}, "
            f"got {sorted(settings)}."
        )

    for name, value in settings.items():
        if not is_number(value):
            raise ValueError(f"{kind} {name} must be a finite number, got {value!r}.")

    if kind == "uniform":
        low, high = settings["low"], settings["high"]
        if not 0 <= low < high:
            raise ValueError(f"uniform needs 0 <= low < high, got {low}, {high}.")
        return stats.uniform(loc=low, scale=high - low)

    if kind == "triangular":
        low, mode, high = settings["low"], settings["mode"], settings["high"]
        if not (0 <= low <= mode <= high and low < high):
            raise ValueError(
                "triangular needs 0 <= low <= mode <= high and low < high, "
                f"got {low}, {mode}, {high}."
            )
        scale = high - low
        return stats.triang(c=(mode - low) / scale, loc=low, scale=scale)

    mean, std = settings["mean"], settings["std"]
    if mean <= 0 or std <= 0:
        raise ValueError(f"{kind} needs mean and std above 0, got {mean}, {std}.")

    if kind == "normal":
        return stats.norm(loc=mean, scale=std)

    sigma2 = math.log1p((std / mean) ** 2)
    mu = math.log(mean) - sigma2 / 2
    return stats.lognorm(s=math.sqrt(sigma2), scale=math.exp(mu))


def percentiles_to_values(percentiles: np.ndarray, distribution) -> np.ndarray:
    """Turn percentiles into float32 values, replacing negatives with 0."""
    return np.maximum(distribution.ppf(percentiles), 0).astype(np.float32)
