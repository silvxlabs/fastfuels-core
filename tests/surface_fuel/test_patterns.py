"""Tests for ``fastfuels_core.surface_fuel.patterns``.

Patterns are measured through
``build_distribution_grids`` with a
``normal`` distribution, so similarity is measured on normal values.

What is pinned: similarity falls off with distance as documented, patches
are smooth with no speckle, speckle limits similarity between neighbors and
full speckle removes patches, a larger patch size gives larger patches, no
patch size gives independent cells, and percentiles span 0 to 1 evenly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastfuels_core.surface_fuel.patterns import rank_percentiles
from tests.surface_fuel.builders import NORMAL, build, similarity


def test_similarity_at_patch_size_is_near_expected():
    bands = {"a": {"distribution": NORMAL, "patch_size": 40}}

    measured = [
        similarity(build(bands, shape=(1000, 1000), seed=seed)["a"].astype(float), 20)
        for seed in range(4)
    ]

    assert np.mean(measured) == pytest.approx(math.exp(-1), abs=0.04)


def test_no_speckle_gives_smooth_patches():
    bands = {"a": {"distribution": NORMAL, "patch_size": 40}}

    grid = build(bands, shape=(600, 600))["a"].astype(float)

    assert similarity(grid, 1) > 0.98


def test_speckle_limits_neighbor_similarity():
    bands = {"a": {"distribution": NORMAL, "patch_size": 40, "speckle": 0.1}}

    grid = build(bands, shape=(600, 600))["a"].astype(float)

    assert similarity(grid, 1) == pytest.approx(0.9, abs=0.03)


def test_full_speckle_gives_independent_cells():
    bands = {"a": {"distribution": NORMAL, "patch_size": 40, "speckle": 1}}

    grid = build(bands, shape=(300, 300))["a"].astype(float)

    assert abs(similarity(grid, 1)) < 0.02


def test_larger_patch_size_gives_larger_patches():
    short = {"a": {"distribution": NORMAL, "patch_size": 40}}
    long = {"a": {"distribution": NORMAL, "patch_size": 120}}

    short_grid = build(short, shape=(600, 600))["a"].astype(float)
    long_grid = build(long, shape=(600, 600))["a"].astype(float)

    assert similarity(long_grid, 20) > similarity(short_grid, 20) + 0.2


def test_no_patch_size_gives_independent_cells():
    bands = {"a": {"distribution": NORMAL}}

    grid = build(bands, shape=(300, 300))["a"].astype(float)

    assert abs(similarity(grid, 1)) < 0.02


def test_rank_percentiles_span_zero_to_one_evenly():
    field = np.array([[3.0, 1.0], [4.0, 2.0]])

    percentiles = rank_percentiles(field)

    assert percentiles == pytest.approx(np.array([[0.625, 0.125], [0.875, 0.375]]))
