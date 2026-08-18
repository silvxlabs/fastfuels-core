"""Deterministic builders for the canopy_fuel test lattice.

Every module in this package works on the same 4x5, 30 m north-up
lattice anchored at (1000, 5000), so a hand-computed cell index means
the same thing everywhere. The builders are plain functions rather than
fixtures: they take arguments, hold no state, and reading a test should
not require looking up what a fixture name resolves to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr
from affine import Affine

TRANSFORM = (30.0, 0.0, 1000.0, 0.0, -30.0, 5000.0)
SHAPE = (4, 5)
CELL_AREA = 900.0

# The lattice spans x 1000-1150, y 4880-5000. Cell (row 2, col 1) is
# x 1030-1060, y 4910-4940, and (1045, 4915) is a point inside it.
CENTER_CELL = (2, 1)


def random_stand(n, seed=0):
    """A stand with plausible species, sizes and crown ratios, no CRS."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "fia_species_code": rng.choice([202, 122, 73, 17, 351, 15, 108], n),
            "dbh": rng.uniform(2.5, 90, n),
            "height": rng.uniform(2, 40, n),
            "crown_ratio": rng.uniform(0.1, 0.9, n),
        }
    )


def stand_on_lattice(n, seed=0):
    """A random stand with every stem strictly inside the test lattice."""
    trees = random_stand(n, seed)
    rng = np.random.default_rng(seed + 1)
    trees["x"] = rng.uniform(1000.1, 1149.9, n)
    trees["y"] = rng.uniform(4880.1, 4999.9, n)
    return trees


def interior_stand(n, seed=0):
    """A stand kept far enough from the edges that no crown overhangs.

    Mass conservation only holds for crowns fully inside the lattice, so
    tests of it need stems set back by more than the largest radius.
    """
    trees = stand_on_lattice(n, seed)
    trees["x"] = np.clip(trees["x"], 1010, 1140)
    trees["y"] = np.clip(trees["y"], 4890, 4990)
    return trees


def single_tree(**overrides):
    """One 12 m ponderosa in cell (2, 1), with columns named by overrides.

    Carries no ``dbh``: the tests that need allometry pass one, and the
    ones that must not touch allometry prove it by its absence.
    """
    tree = {
        "x": 1045.0,
        "y": 4915.0,
        "height": 12.0,
        "crown_ratio": 0.5,
        "fia_species_code": 122,
    }
    tree.update(overrides)
    return pd.DataFrame({k: [v] for k, v in tree.items()})


def column_profile(values):
    """A (n_layers, 1, 1) profile from a list of layer densities."""
    return np.asarray(values, dtype=float).reshape(-1, 1, 1)


def band_template(bands):
    """A griddle-style georeferenced template on the test lattice."""
    ds = xr.Dataset(
        {b: (("y", "x"), np.full(SHAPE, np.nan, dtype=np.float32)) for b in bands},
        coords={
            "y": 5000.0 - 30.0 * (np.arange(SHAPE[0]) + 0.5),
            "x": 1000.0 + 30.0 * (np.arange(SHAPE[1]) + 0.5),
        },
    )
    ds.rio.write_crs("EPSG:32611", inplace=True)
    ds.rio.write_transform(Affine(*TRANSFORM), inplace=True)
    return ds
