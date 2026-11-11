"""Tests for ``fastfuels_core.surface_fuel.distribution_grids``.

These go through the public function end to end. The distribution types,
correlation matrix and patterns have their own test modules.

What is pinned: each band's values match its distribution, the same request
gives identical output, rank correlations match when patch sizes are equal,
linked bands become related, and bad bands, grids and seeds raise
ValueError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from fastfuels_core.surface_fuel.distribution_grids import _band_sigma
from tests.surface_fuel.builders import (
    NORMAL,
    UNIFORM,
    build,
    pair,
    rank_correlation,
    uniform_bands,
)

_LOGNORMAL_SIGMA = math.sqrt(math.log(1.25))
_DISTRIBUTIONS = [
    ({"type": "uniform", "low": 2, "high": 8}, stats.uniform(loc=2, scale=6)),
    ({"type": "normal", "mean": 10, "std": 2}, stats.norm(loc=10, scale=2)),
    (
        {"type": "lognormal", "mean": 2, "std": 1},
        stats.lognorm(
            s=_LOGNORMAL_SIGMA,
            scale=math.exp(math.log(2) - _LOGNORMAL_SIGMA**2 / 2),
        ),
    ),
    (
        {"type": "triangular", "low": 1, "mode": 2, "high": 6},
        stats.triang(c=0.2, loc=1, scale=5),
    ),
]


class TestDistributions:
    @pytest.mark.parametrize("patch_size", [0, 40])
    @pytest.mark.parametrize("spec, reference", _DISTRIBUTIONS)
    def test_grid_matches_distribution(self, spec, reference, patch_size):
        bands = {"a": {"distribution": spec, "patch_size": patch_size}}

        grid = build(bands, shape=(300, 300))["a"]

        assert grid.mean() == pytest.approx(reference.mean(), rel=0.02)
        assert grid.std() == pytest.approx(reference.std(), rel=0.02)
        assert np.percentile(grid, [10, 50, 90]) == pytest.approx(
            reference.ppf([0.1, 0.5, 0.9]), rel=0.02
        )

    def test_normal_replaces_negatives_with_zero(self):
        bands = {"a": {"distribution": {"type": "normal", "mean": 1, "std": 1}}}

        grid = build(bands, shape=(300, 300))["a"]

        assert grid.min() == 0
        assert (grid == 0).mean() == pytest.approx(0.159, abs=0.01)
        assert grid.mean() == pytest.approx(1.08, abs=0.01)
        assert grid.std() == pytest.approx(0.87, abs=0.01)


class TestOutput:
    def test_shape_and_dtype(self):
        grids = build(uniform_bands("a", "b"), shape=(20, 30))

        assert set(grids) == {"a", "b"}
        for grid in grids.values():
            assert grid.shape == (20, 30)
            assert grid.dtype == np.float32

    def test_same_seed_gives_identical_output(self):
        first = build(uniform_bands("a", "b"), [pair("a", "b", 0.5)], seed=7)
        second = build(uniform_bands("a", "b"), [pair("a", "b", 0.5)], seed=7)

        assert np.array_equal(first["a"], second["a"])
        assert np.array_equal(first["b"], second["b"])

    def test_different_seed_gives_different_output(self):
        first = build(uniform_bands("a"), seed=7)["a"]
        second = build(uniform_bands("a"), seed=8)["a"]

        assert not np.array_equal(first, second)

    def test_no_seed_is_allowed(self):
        assert build(uniform_bands("a"), seed=None)["a"].shape == (20, 30)

    def test_listing_order_does_not_change_output(self):
        forward = {"a": {"distribution": UNIFORM}, "b": {"distribution": NORMAL}}
        backward = {"b": {"distribution": NORMAL}, "a": {"distribution": UNIFORM}}

        first = build(forward, [pair("a", "b", 0.5)])
        second = build(backward, [pair("b", "a", 0.5)])

        assert np.array_equal(first["a"], second["a"])
        assert np.array_equal(first["b"], second["b"])

    def test_missing_patch_size_matches_zero(self):
        missing = build({"a": {"distribution": UNIFORM}})["a"]
        zero = build({"a": {"distribution": UNIFORM, "patch_size": 0}})["a"]

        assert np.array_equal(missing, zero)

    def test_missing_speckle_matches_zero(self):
        band = {"distribution": UNIFORM, "patch_size": 16}

        missing = build({"a": band})["a"]
        zero = build({"a": {**band, "speckle": 0}})["a"]

        assert np.array_equal(missing, zero)

    def test_largest_allowed_patch_size_is_accepted(self):
        assert _band_sigma("a", {"patch_size": 1000}, 2.0) == 250


class TestCorrelations:
    @pytest.mark.parametrize("coefficient", [0.7, -0.5])
    def test_two_bands_match_requested_coefficient(self, coefficient):
        grids = build(
            uniform_bands("a", "b"), [pair("a", "b", coefficient)], shape=(600, 600)
        )

        measured = rank_correlation(grids["a"], grids["b"])

        assert measured == pytest.approx(coefficient, abs=0.06)

    def test_three_bands_match_requested_coefficients(self):
        requested = {("a", "b"): 0.6, ("b", "c"): -0.4, ("a", "c"): 0.2}
        correlations = [pair(a, b, value) for (a, b), value in requested.items()]

        grids = build(uniform_bands("a", "b", "c"), correlations, shape=(600, 600))

        for (a, b), value in requested.items():
            measured = rank_correlation(grids[a], grids[b])
            assert measured == pytest.approx(value, abs=0.06)

    def test_linked_bands_become_related(self):
        correlations = [pair("a", "b", 0.6), pair("b", "c", 0.6)]

        grids = build(uniform_bands("a", "b", "c"), correlations, shape=(600, 600))

        assert rank_correlation(grids["a"], grids["b"]) == pytest.approx(0.6, abs=0.06)
        assert rank_correlation(grids["b"], grids["c"]) == pytest.approx(0.6, abs=0.06)
        assert rank_correlation(grids["a"], grids["c"]) == pytest.approx(0.36, abs=0.06)

    def test_bands_with_no_link_have_no_built_in_correlation(self):
        bands = {"a": {"distribution": UNIFORM}, "b": {"distribution": UNIFORM}}

        grids = build(bands, shape=(300, 300))

        assert abs(rank_correlation(grids["a"], grids["b"])) < 0.02

    def test_different_patch_sizes_lower_the_correlation(self):
        bands = {
            "a": {"distribution": UNIFORM, "patch_size": 20},
            "b": {"distribution": UNIFORM, "patch_size": 80},
        }

        grids = build(bands, [pair("a", "b", 0.8)], shape=(600, 600))

        assert 0.2 < rank_correlation(grids["a"], grids["b"]) < 0.65


class TestErrors:
    @pytest.mark.parametrize(
        "band, message",
        [
            ({}, "distribution dict"),
            ({"distribution": "uniform"}, "distribution dict"),
            ({"distribution": UNIFORM, "patch_sise": 10}, "unknown settings"),
            ({"distribution": UNIFORM, "patch_size": -1}, "patch_size must be"),
            (
                {"distribution": UNIFORM, "patch_size": float("nan")},
                "patch_size must be",
            ),
            ({"distribution": UNIFORM, "patch_size": 1002}, "above the limit"),
            ({"distribution": UNIFORM, "speckle": -0.1}, "speckle must be"),
            ({"distribution": UNIFORM, "speckle": 1.5}, "speckle must be"),
            ({"distribution": UNIFORM, "speckle": float("nan")}, "speckle must be"),
            ({"distribution": UNIFORM, "speckle": "0.1"}, "speckle must be"),
            ({"distribution": {"type": "constant", "value": 1}}, "Unknown"),
        ],
    )
    def test_bad_band(self, band, message):
        with pytest.raises(ValueError, match=message):
            build({"a": band})

    def test_empty_bands(self):
        with pytest.raises(ValueError, match="bands is empty"):
            build({})

    @pytest.mark.parametrize(
        "shape", [(20,), (20, 30, 40), (0, 20), (-1, 20), (1.5, 20)]
    )
    def test_bad_shape(self, shape):
        with pytest.raises(ValueError, match="shape must"):
            build(uniform_bands("a"), shape=shape)

    @pytest.mark.parametrize("resolution", [0, -2, float("nan"), float("inf")])
    def test_bad_resolution(self, resolution):
        with pytest.raises(ValueError, match="resolution must"):
            build(uniform_bands("a"), resolution=resolution)

    @pytest.mark.parametrize("seed", [-1, 1.5, "1", True])
    def test_bad_seed(self, seed):
        with pytest.raises(ValueError, match="seed must"):
            build(uniform_bands("a"), seed=seed)

    def test_bad_correlation(self):
        with pytest.raises(ValueError, match="unknown band"):
            build(uniform_bands("a", "b"), [pair("a", "missing", 0.5)])
