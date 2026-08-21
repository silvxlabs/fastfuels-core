"""Tests for :mod:`fastfuels_core.canopy_fuel.crown_radius`."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.canopy_fuel.crown_radius import max_crown_radius
from fastfuels_core.trees import Tree
from tests.canopy_fuel.builders import random_stand


class TestPurvesEquations:
    """The default arm, against the per-tree path it vectorizes."""

    def test_matches_the_per_tree_reference(self):
        trees = random_stand(500)
        reference = np.array(
            [
                Tree(
                    species_code=row.fia_species_code,
                    status_code=1,
                    diameter=row.dbh,
                    height=row.height,
                    crown_ratio=row.crown_ratio,
                ).max_crown_radius
                for row in trees.itertuples()
            ]
        )
        np.testing.assert_allclose(
            max_crown_radius(trees, equations="purves"), reference, rtol=1e-12
        )

    def test_returns_one_radius_per_tree(self):
        assert max_crown_radius(random_stand(50)).shape == (50,)

    def test_a_single_tree_still_returns_an_array(self):
        """Purves returns a scalar for one tree; callers index the result."""
        assert max_crown_radius(random_stand(1)).shape == (1,)

    def test_radii_are_positive(self):
        assert (max_crown_radius(random_stand(50)) > 0).all()


class TestColumnOverride:
    def test_named_column_is_returned_verbatim(self):
        trees = random_stand(10)
        trees["crad"] = np.arange(10, dtype=float) + 1.0
        np.testing.assert_array_equal(
            max_crown_radius(trees, crown_radius_column="crad"),
            trees["crad"].to_numpy(),
        )

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            max_crown_radius(random_stand(5), crown_radius_column="nope")


def test_unknown_equations_raises():
    with pytest.raises(ValueError, match="bogus"):
        max_crown_radius(random_stand(5), equations="bogus")
