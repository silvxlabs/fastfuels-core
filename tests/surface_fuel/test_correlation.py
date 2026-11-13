"""Tests for ``fastfuels_core.surface_fuel.correlation``.

What is pinned: listed pairs are kept, unlisted pairs follow from the
links, bands with no link get 0, slight contradictions are nudged with a
warning, and bad pairs or real contradictions raise ValueError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fastfuels_core.surface_fuel import correlation
from fastfuels_core.surface_fuel.correlation import correlation_matrix, mixing_factor
from tests.surface_fuel.builders import pair


def test_filled_values_follow_a_chain_of_links():
    correlations = [pair("a", "b", 0.6), pair("b", "c", 0.6), pair("c", "d", 0.6)]
    listed = 2 * math.sin(math.pi * 0.6 / 6)

    matrix = correlation_matrix(["a", "b", "c", "d"], correlations)

    assert matrix[0, 1] == pytest.approx(listed)
    assert matrix[0, 2] == pytest.approx(listed**2)
    assert matrix[0, 3] == pytest.approx(listed**3)
    assert matrix[3, 0] == matrix[0, 3]


def test_high_links_are_not_a_contradiction_when_a_pair_is_unlisted():
    correlations = [pair("a", "b", 0.9), pair("b", "c", 0.9)]

    matrix = correlation_matrix(["a", "b", "c"], correlations)

    assert matrix[0, 2] > 0.8


def test_listed_zero_is_kept():
    correlations = [pair("a", "b", 0.6), pair("b", "c", 0.6), pair("a", "c", 0)]

    matrix = correlation_matrix(["a", "b", "c"], correlations)

    assert matrix[0, 2] == 0


def test_bands_with_no_link_get_zero():
    correlations = [pair("a", "b", 0.6), pair("c", "d", 0.6)]

    matrix = correlation_matrix(["a", "b", "c", "d", "e"], correlations)

    assert matrix[0, 2] == pytest.approx(0, abs=1e-12)
    assert matrix[1, 3] == pytest.approx(0, abs=1e-12)
    assert np.array_equal(matrix[4], [0, 0, 0, 0, 1])


def test_singular_matrix_can_still_be_factored():
    matrix = np.ones((2, 2))

    factor = mixing_factor(matrix)

    assert factor @ factor.T == pytest.approx(matrix)


@pytest.mark.parametrize(
    "keys, correlations",
    [
        (
            "abcde",
            [
                pair("a", "b", 0.98),
                pair("b", "c", 0.98),
                pair("c", "d", 0.98),
                pair("d", "e", 0.98),
            ],
        ),
        (
            "abcde",
            [
                pair("a", "b", 0.9),
                pair("a", "c", 0.9),
                pair("a", "d", 0.9),
                pair("a", "e", 0.9),
            ],
        ),
        (
            "abcd",
            [
                pair("a", "b", 0.5),
                pair("b", "c", 0.5),
                pair("c", "d", 0.5),
                pair("d", "a", 0.5),
            ],
        ),
    ],
)
def test_strong_links_and_loops_can_be_filled(keys, correlations):
    matrix = correlation_matrix(list(keys), correlations)

    assert np.linalg.eigvalsh(matrix)[0] >= 0
    assert np.array_equal(matrix, matrix.T)


def test_slightly_contradictory_coefficients_are_nudged():
    keys = ["a", "b", "c", "d", "e"]
    requested = {
        ("a", "b"): -0.5,
        ("a", "c"): -0.43,
        ("a", "d"): -0.04,
        ("a", "e"): 0.77,
        ("b", "c"): 0.98,
        ("b", "d"): 0.86,
        ("b", "e"): -0.87,
        ("c", "d"): 0.9,
        ("c", "e"): -0.86,
        ("d", "e"): -0.64,
    }
    correlations = [pair(a, b, value) for (a, b), value in requested.items()]

    with pytest.warns(UserWarning, match="adjusted"):
        matrix = correlation_matrix(keys, correlations)

    measured = 6 / np.pi * np.arcsin(matrix / 2)
    assert np.linalg.eigvalsh(matrix)[0] >= 0
    for (a, b), value in requested.items():
        used = measured[keys.index(a), keys.index(b)]
        assert used == pytest.approx(value, abs=0.02)


@pytest.mark.parametrize(
    "correlations, message",
    [
        ([{"bands": ["a"], "coefficient": 0.5}], "exactly two"),
        ([{"bands": ["a", "b", "c"], "coefficient": 0.5}], "exactly two"),
        ([pair("a", "missing", 0.5)], "unknown band"),
        ([pair("a", "a", 0.5)], "with itself"),
        ([pair("a", "b", 0.5), pair("a", "b", 0.1)], "listed twice"),
        ([pair("a", "b", 0.5), pair("b", "a", 0.5)], "listed twice"),
        ([pair("a", "b", 1)], "between -1 and 1"),
        ([pair("a", "b", -1)], "between -1 and 1"),
        ([pair("a", "b", 1.5)], "between -1 and 1"),
        ([pair("a", "b", float("nan"))], "between -1 and 1"),
        ([{"bands": ["a", "b"]}], "between -1 and 1"),
    ],
)
def test_bad_correlation(correlations, message):
    with pytest.raises(ValueError, match=message):
        correlation_matrix(["a", "b", "c"], correlations)


def test_contradictory_coefficients():
    correlations = [pair("a", "b", 0.9), pair("b", "c", 0.9), pair("a", "c", 0)]

    with pytest.raises(ValueError, match="contradict"):
        correlation_matrix(["a", "b", "c"], correlations)


def test_unlisted_pairs_that_never_settle(monkeypatch):
    monkeypatch.setattr(correlation, "_MAX_FILL_PASSES", 0)
    correlations = [pair("a", "b", 0.6), pair("b", "c", 0.6)]

    with pytest.raises(ValueError, match="can't be filled"):
        correlation_matrix(["a", "b", "c"], correlations)


def test_impossible_loop_of_links():
    correlations = [
        pair("a", "b", 0.9),
        pair("b", "c", 0.9),
        pair("c", "d", 0.9),
        pair("d", "a", -0.9),
    ]

    with pytest.raises(ValueError, match="can't be filled"):
        correlation_matrix(["a", "b", "c", "d"], correlations)
