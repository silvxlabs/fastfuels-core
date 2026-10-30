"""Every canopy_fuel stage that reads ``crown_base_height`` rejects bad values.

A crown base above the tree, below the ground, or missing once went
straight into the geometry and came back as dropped fuel, negative crown
lengths, or NaN cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastfuels_core.canopy_fuel.canopy_height import (
    crown_base_statistic,
    height_percentile_depth,
    mean_crown_length,
)
from fastfuels_core.canopy_fuel.crown_radius import max_crown_radius
from fastfuels_core.canopy_fuel.metrics import compute_canopy_metrics
from fastfuels_core.canopy_fuel.profile import vertical_profile
from tests.canopy_fuel.builders import SHAPE, TRANSFORM, band_template, single_tree

STAGES = {
    "vertical_profile": lambda t: vertical_profile(
        t, np.full(len(t), 9.0), TRANSFORM, SHAPE
    ),
    "crown_base_statistic": lambda t: crown_base_statistic(
        t, np.full(len(t), 9.0), TRANSFORM, SHAPE
    ),
    "mean_crown_length": lambda t: mean_crown_length(t, TRANSFORM, SHAPE),
    "height_percentile_depth": lambda t: height_percentile_depth(t, TRANSFORM, SHAPE),
    "max_crown_radius_purves": lambda t: max_crown_radius(t, equations="purves"),
    "compute_canopy_metrics": lambda t: compute_canopy_metrics(
        t,
        band_template(["cbd", "cbh", "chm", "cfl"]),
        fuel_column="acf",
        crown_class_adjustment="none",
    ),
}


def three_trees(crown_base_heights=(6.0, 6.0, 6.0)):
    """Three 12 m ponderosas in one cell with the given crown bases."""
    trees = pd.concat([single_tree(dbh=30.0, acf=9.0)] * 3, ignore_index=True)
    trees["crown_base_height"] = list(crown_base_heights)
    return trees


@pytest.mark.parametrize("stage", sorted(STAGES))
@pytest.mark.parametrize("bad", [12.5, -0.1, np.nan, np.inf])
def test_an_invalid_crown_base_raises(stage, bad):
    with pytest.raises(ValueError, match="1 of 3 trees have a crown_base_height"):
        STAGES[stage](three_trees((6.0, bad, 6.0)))


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_the_message_counts_every_offending_tree(stage):
    with pytest.raises(ValueError, match="2 of 3 trees"):
        STAGES[stage](three_trees((13.0, 6.0, -1.0)))


def test_a_nullable_missing_crown_base_raises():
    trees = three_trees()
    trees["crown_base_height"] = pd.array([6.0, pd.NA, 6.0], dtype="Float64")
    with pytest.raises(ValueError, match="1 of 3 trees"):
        STAGES["vertical_profile"](trees)


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_crown_bases_at_the_ground_and_the_top_are_accepted(stage):
    STAGES[stage](three_trees((0.0, 12.0, 6.0)))
