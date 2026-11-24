"""Tests for ``fastfuels_core.surface_fuel.distributions``.

What is pinned: each type validates its settings, ``lognormal``'s mean and
std describe the real values, and values are never negative.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.surface_fuel.distributions import (
    build_distribution,
    percentiles_to_values,
)


def test_lognormal_mean_and_std_describe_real_values():
    distribution = build_distribution({"type": "lognormal", "mean": 3, "std": 2})

    assert distribution.mean() == pytest.approx(3)
    assert distribution.std() == pytest.approx(2)


def test_values_are_float32_and_never_negative():
    distribution = build_distribution({"type": "normal", "mean": 1, "std": 1})

    values = percentiles_to_values(np.array([0.01, 0.5, 0.99]), distribution)

    assert values.dtype == np.float32
    assert values[0] == 0
    assert values[1] == pytest.approx(1)


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"type": "constant", "value": 1}, "Unknown distribution type"),
        ({"low": 1, "high": 2}, "Unknown distribution type"),
        ({"type": "uniform", "low": 1}, "takes"),
        ({"type": "normal", "mean": 1, "std": 1, "min": 0}, "takes"),
        ({"type": "lognormal", "mean": 1, "std": 1, "max": 5}, "takes"),
        ({"type": "uniform", "low": True, "high": 2}, "finite number"),
        ({"type": "uniform", "low": "1", "high": 2}, "finite number"),
        ({"type": "normal", "mean": float("nan"), "std": 1}, "finite number"),
        ({"type": "normal", "mean": 1, "std": float("inf")}, "finite number"),
        ({"type": "uniform", "low": -1, "high": 2}, "uniform needs"),
        ({"type": "uniform", "low": 2, "high": 2}, "uniform needs"),
        ({"type": "uniform", "low": 3, "high": 2}, "uniform needs"),
        ({"type": "normal", "mean": 0, "std": 1}, "mean and std"),
        ({"type": "normal", "mean": 1, "std": 0}, "mean and std"),
        ({"type": "lognormal", "mean": -1, "std": 1}, "mean and std"),
        ({"type": "lognormal", "mean": 1, "std": -1}, "mean and std"),
        (
            {"type": "triangular", "low": -1, "mode": 1, "high": 2},
            "triangular needs",
        ),
        (
            {"type": "triangular", "low": 1, "mode": 0.5, "high": 2},
            "triangular needs",
        ),
        (
            {"type": "triangular", "low": 1, "mode": 3, "high": 2},
            "triangular needs",
        ),
        (
            {"type": "triangular", "low": 1, "mode": 1, "high": 1},
            "triangular needs",
        ),
    ],
)
def test_bad_distribution(spec, message):
    with pytest.raises(ValueError, match=message):
        build_distribution(spec)
