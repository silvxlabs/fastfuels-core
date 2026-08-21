"""Give fastfuels-core FuelCalc's treelist; check we get FuelCalc's numbers.

FuelCalc 1.7 ships a two-plot tutorial treelist. It was run on that
treelist, pre- and post-thinning, and the Stand Measurements block of
the four plot reports it wrote is the reference here. Every other
FuelCalc test in this package compares us to a *reading* of FuelCalc —
its published equations, or our transcription of its algorithm in
:mod:`tests.canopy_fuel.fuelcalc_reference`. This one compares us to
numbers the program itself printed.

There is one test per plot per treatment, each asserting the six values
that plot report shows. Where we do not reproduce a value the assertion
is on *our* number, with the reason beside it, so a change to either
side shows up as a failing test rather than a widened tolerance.

How the stand is built
----------------------
FuelCalc has no horizontal structure — it works from expansion factors
on one nominal acre — so the faithful lattice is a single cell with
every stem at the centre and ``horizontal_distribution="stem"``. The
cell is ten acres only so that expansion factors given to a tenth come
out as whole stems.

The thinning is ours
--------------------
FuelCalc thinned to a target tree density: 200 stems/acre, taking at
most 90% of each record, largest first, and never touching a tree of
8 in dbh or over. :class:`~fastfuels_core.treatments.DirectionalThinToTreeDensity`
does the same, so the post-thinning cases run end to end from this
package -- treelist in, thinned stand, canopy metrics out.
:func:`test_our_thinning_reproduces_the_fuelcalc_stand` checks the
thinned stand itself against the expansion factors FuelCalc reported,
before any canopy fuel is computed.

Where we differ, and why
------------------------
This package implements the published equations, not FuelCalc's build
of them, and on these stands the two disagree in exactly one place:
FuelCalc's compiled crown-proportion table carries a western larch P2
of ``0.745*exp(-0.0632d)`` where Brown 1978 Table 16 (p. 53) and
FuelCalc's own User Guide (Appendix D) both print
``0.745*exp(-0.0362d)``. We implement the published coefficient. Both
tutorial stands carry larch, and on the largest of them the two put the
fine branchwood fraction a factor of three apart, so the difference
reaches every stand total that larch touches.

Ten of the twenty-four reported values differ for that reason and no
other. :func:`test_the_larch_coefficient_accounts_for_the_deviations`
substitutes FuelCalc's coefficient -- in this module, never in the
package -- and every one of them lands on FuelCalc's number. Nothing
else about these four stands is unaccounted for.

Displayed precision
-------------------
FuelCalc's plot report rounds. Its GUI truncates the same values (canopy
cover 48.77003 prints as 48.77 in the report and 48.7 in the GUI). These
are the report's numbers, so comparisons round.
"""

from __future__ import annotations

import inspect
import math
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr
from affine import Affine

from fastfuels_core.allometry import brown
from fastfuels_core.canopy_fuel.metrics import compute_canopy_metrics
from fastfuels_core.treatments import (
    DirectionalThinToTreeDensity,
    ThinningDirection,
)

ACRE_M2 = 4046.8564224
FT_TO_M = 0.3048
LB_TO_KG = 0.45359237
CELL_ACRES = 10

SPECIES_CODES = {
    "PIPO": 122,
    "LAOC": 73,
    "PSME": 202,
    "PICO": 108,
    "ABLA": 19,
    "ABGR": 17,
}

# FuelCalc 1.7's option at every stage where this package offers more
# than one. These are also the module's defaults, which
# test_the_module_defaults_are_this_parameterization pins; the dict is
# kept as the explicit record of what FuelCalc does, so a default that
# drifts away from it fails there rather than quietly here.
FUELCALC_1_7 = dict(
    equations="brown_1978",
    crown_class_adjustment="reinhardt_2006",
    crown_class_column="crown_class",
    exclude_hardwoods=True,
    horizontal_distribution="stem",
    vertical_distribution="reinhardt_2006",
    layer_depth=FT_TO_M,  # one-foot layers
    cbd_window=5 * FT_TO_M,  # five-layer running mean
    cbd_window_edge="fuelcalc",
    threshold_smoothing_window=5 * FT_TO_M,
    threshold_smoothing_edge="fuelcalc",
    cbh_threshold=0.012,
    cbh_relative_fraction=0.1,
    cover_method="crown_overlap",
    crown_radius_equations="crookston_stage",
    min_tree_height=0.0,
)


def stems(plot: int, expansion_factors: str = "TPA_PRE") -> pd.DataFrame:
    """One tutorial plot's live trees, one row per stem.

    ``expansion_factors="TPA_POST"`` replays the post-thinning counts
    FuelCalc reported, for comparison against our own thinning.
    """
    path = files("tests.canopy_fuel").joinpath("data/fuelcalc_tutorial_treelist.csv")
    with path.open() as handle:
        trees = pd.read_csv(handle)

    trees = trees[trees["PLOT"] == plot]
    # FuelCalc zeroes the foliage and twig weight of a tree marked dead,
    # so it carries no canopy fuel and no crown area.
    trees = trees[trees["STATUS"] != "D"]
    trees = trees[trees[expansion_factors] > 0]

    side = math.sqrt(CELL_ACRES * ACRE_M2)
    counts = np.rint(trees[expansion_factors].to_numpy() * CELL_ACRES).astype(int)
    return pd.DataFrame(
        {
            "x": side / 2.0,
            "y": side / 2.0,
            "height": trees["HEIGHT_FT"].to_numpy() * FT_TO_M,
            "crown_ratio": 1.0
            - trees["CBH_FT"].to_numpy() / trees["HEIGHT_FT"].to_numpy(),
            "dbh": trees["DBH_IN"].to_numpy() * 2.54,
            "fia_species_code": trees["SPECIES"].map(SPECIES_CODES).to_numpy(),
            "crown_class": trees["CROWN_CLASS"].to_numpy(),
        }
    ).loc[np.repeat(np.arange(len(trees)), counts)]


# The tutorial's thinning: down to 200 stems/acre, largest first, at most
# 90% of any one record, and nothing 8 in dbh or over.
FUELCALC_THINNING = DirectionalThinToTreeDensity(
    target=200 * CELL_ACRES,
    direction=ThinningDirection.ABOVE,
    min_diameter=0.0,
    max_diameter=8.0 * 2.54,
    cut_efficiency=0.9,
)


def canopy_metrics(trees: pd.DataFrame) -> dict[str, float]:
    """Run the FuelCalc parameterization and report in FuelCalc's units."""
    side = math.sqrt(CELL_ACRES * ACRE_M2)
    dataset = xr.Dataset(
        {b: (("y", "x"), np.zeros((1, 1))) for b in ("cbd", "cbh", "chm", "cc", "cfl")},
        coords={"y": [side / 2.0], "x": [side / 2.0]},
    )
    dataset = dataset.rio.write_transform(
        Affine(side, 0.0, 0.0, 0.0, -side, side)
    ).rio.write_crs("EPSG:32611")
    out = compute_canopy_metrics(trees, dataset, **FUELCALC_1_7)
    return {
        "tree_density_tpa": len(trees) / CELL_ACRES,
        "canopy_cover_pct": float(out.cc.values[0, 0]),
        # FuelCalc labels a layer by its top; we anchor canopy base to
        # the layer bottom so chm - cbh is the depth of qualifying
        # canopy. The two conventions differ by exactly one layer.
        "canopy_base_height_ft": float(out.cbh.values[0, 0]) / FT_TO_M + 1.0,
        "stand_height_ft": float(out.chm.values[0, 0]) / FT_TO_M,
        "canopy_bulk_density_kg_m3": float(out.cbd.values[0, 0]),
        "canopy_fuel_load_ton_ac": float(out.cfl.values[0, 0])
        * ACRE_M2
        / LB_TO_KG
        / 2000.0,
    }


# Western larch P2. FuelCalc's compiled table carries exp(-0.0632d);
# Brown 1978 Table 16 (p. 53) and FuelCalc's own User Guide (Appendix D)
# both print exp(-0.0362d), which is what this package implements. Both
# tutorial plots carry larch, so the two coefficients separate our
# numbers from FuelCalc's; the last test below measures by how much.
FUELCALC_LARCH_P2 = (lambda dia, a, b: a * np.exp(b * dia), {"a": 0.745, "b": -0.0632})
LARCH = "Larch P2: FuelCalc's compiled coefficient, not the published one."


def test_plot_1_pre_treatment():
    got = canopy_metrics(stems(plot=1))

    assert got["tree_density_tpa"] == pytest.approx(280.0)
    assert got["canopy_cover_pct"] == pytest.approx(48.77, abs=0.005)
    assert got["stand_height_ft"] == pytest.approx(103.0)

    # FuelCalc reports 1 ft, 0.044 and 3.79. All three deviate by LARCH.
    assert got["canopy_base_height_ft"] == pytest.approx(2.0)
    assert got["canopy_bulk_density_kg_m3"] == pytest.approx(0.0447, abs=5e-5)
    assert got["canopy_fuel_load_ton_ac"] == pytest.approx(3.848, abs=5e-4)


def test_plot_1_post_thinning():
    thinned = FUELCALC_THINNING.apply(stems(plot=1), dia_column_name="dbh")
    got = canopy_metrics(thinned)

    assert got["tree_density_tpa"] == pytest.approx(200.0)
    assert got["canopy_cover_pct"] == pytest.approx(43.50, abs=0.005)
    assert got["canopy_base_height_ft"] == pytest.approx(2.0)
    assert got["stand_height_ft"] == pytest.approx(103.0)

    # FuelCalc reports 0.044 and 3.24. Both deviate by LARCH.
    assert got["canopy_bulk_density_kg_m3"] == pytest.approx(0.0447, abs=5e-5)
    assert got["canopy_fuel_load_ton_ac"] == pytest.approx(3.303, abs=5e-4)


def test_plot_2_pre_treatment():
    got = canopy_metrics(stems(plot=2))

    assert got["tree_density_tpa"] == pytest.approx(903.0)
    assert got["canopy_cover_pct"] == pytest.approx(40.99, abs=0.005)
    assert got["canopy_base_height_ft"] == pytest.approx(1.0)
    assert got["canopy_bulk_density_kg_m3"] == pytest.approx(0.046, abs=5e-4)

    # FuelCalc reports 123 ft and 2.76. Both deviate by LARCH: larch is
    # the tallest tree here, so its crown weight also sets where the
    # profile crosses the threshold near the canopy top.
    assert got["stand_height_ft"] == pytest.approx(125.0)
    assert got["canopy_fuel_load_ton_ac"] == pytest.approx(2.952, abs=5e-4)


def test_plot_2_post_thinning():
    thinned = FUELCALC_THINNING.apply(stems(plot=2), dia_column_name="dbh")
    got = canopy_metrics(thinned)

    assert got["tree_density_tpa"] == pytest.approx(200.0)
    assert got["canopy_cover_pct"] == pytest.approx(31.54, abs=0.005)
    assert got["canopy_base_height_ft"] == pytest.approx(3.0)

    # FuelCalc reports 0.023, 126 ft and 2.07. All three deviate by LARCH.
    assert got["canopy_bulk_density_kg_m3"] == pytest.approx(0.0236, abs=5e-5)
    assert got["stand_height_ft"] == pytest.approx(129.0)
    assert got["canopy_fuel_load_ton_ac"] == pytest.approx(2.254, abs=5e-4)


def test_the_module_defaults_are_this_parameterization():
    """compute_canopy_metrics defaults to FuelCalc at every stage.

    ``crown_class_column`` is the one setting with no default: it names
    a column in the caller's inventory, so the module cannot guess it,
    and it will not fall back to the table's Other/none factor because
    that is 0.5 for 50 of the 54 species.
    """
    defaults = inspect.signature(compute_canopy_metrics).parameters
    drifted = {
        name: (wanted, defaults[name].default)
        for name, wanted in FUELCALC_1_7.items()
        if name != "crown_class_column" and defaults[name].default != wanted
    }
    assert not drifted, f"defaults no longer match FuelCalc: {drifted}"


def test_our_thinning_reproduces_the_fuelcalc_stand():
    """The thinned stand itself, before any canopy fuel is computed.

    Plot 1 comes out identical. Plot 2 lands 0.1 stems/acre apart on two
    records: FuelCalc counts out its cut in tenths of a tree and takes
    ``int(tpa * cut_efficiency * 10)`` steps in single precision, so
    228 * 0.9 gives it 205.1 rather than 205.2, and the next record
    absorbs the difference. The stand totals are exact either way.
    """
    for plot, tolerance in ((1, 0.0), (2, 0.1)):
        thinned = FUELCALC_THINNING.apply(stems(plot), dia_column_name="dbh")
        ours = thinned.groupby("dbh").size() / CELL_ACRES
        fuelcalc = stems(plot, "TPA_POST").groupby("dbh").size() / CELL_ACRES
        assert ours.sum() == pytest.approx(200.0)
        assert fuelcalc.sum() == pytest.approx(200.0)
        difference = ours.subtract(fuelcalc, fill_value=0.0).abs()
        assert difference.max() <= tolerance + 1e-9, (
            f"plot {plot} thinned stand differs by diameter class:\n"
            f"{pd.DataFrame({'ours': ours, 'fuelcalc': fuelcalc})}"
        )


def test_the_larch_coefficient_accounts_for_the_deviations():
    """Substitute FuelCalc's larch P2 and every deviation closes.

    The substitution happens here, never in the package, which
    implements Brown's published coefficient. Running the same four
    stands both ways turns the attribution into a measurement: all ten
    values the tests above flag are this one coefficient, and with it
    the four plot reports come back exactly.
    """
    original = {i: brown.P2_EQUATIONS[i] for i in ("WL", "AL")}
    brown.P2_EQUATIONS.update({i: FUELCALC_LARCH_P2 for i in ("WL", "AL")})
    try:
        plot_1_pre = canopy_metrics(stems(plot=1))
        plot_2_pre = canopy_metrics(stems(plot=2))
        plot_1_post = canopy_metrics(
            FUELCALC_THINNING.apply(stems(plot=1), dia_column_name="dbh")
        )
        plot_2_post = canopy_metrics(
            FUELCALC_THINNING.apply(stems(plot=2), dia_column_name="dbh")
        )
        # FuelCalc's own post-thinning counts, to separate the thinning
        # from the canopy fuel chain in the one value that needs it.
        plot_2_post_their_stand = canopy_metrics(
            stems(plot=2, expansion_factors="TPA_POST")
        )
    finally:
        brown.P2_EQUATIONS.update(original)

    assert plot_1_pre["canopy_base_height_ft"] == pytest.approx(1.0)
    assert round(plot_1_pre["canopy_bulk_density_kg_m3"], 3) == 0.044
    assert round(plot_1_pre["canopy_fuel_load_ton_ac"], 2) == 3.79

    assert round(plot_1_post["canopy_bulk_density_kg_m3"], 3) == 0.044
    assert round(plot_1_post["canopy_fuel_load_ton_ac"], 2) == 3.24

    assert plot_2_pre["stand_height_ft"] == pytest.approx(123.0)
    assert round(plot_2_pre["canopy_fuel_load_ton_ac"], 2) == 2.76

    assert round(plot_2_post["canopy_bulk_density_kg_m3"], 3) == 0.023
    assert plot_2_post["stand_height_ft"] == pytest.approx(126.0)

    # Plot 2's post-thinning fuel load lands on the rounding boundary.
    # FuelCalc reports 2.07; its own thinned stand gives us 2.06508,
    # which rounds there, and ours gives 2.06497, which does not. The
    # 0.2 lb/acre between them is the tenth of a stem per acre that
    # test_our_thinning_reproduces_the_fuelcalc_stand describes.
    assert round(plot_2_post_their_stand["canopy_fuel_load_ton_ac"], 2) == 2.07
    assert plot_2_post["canopy_fuel_load_ton_ac"] == pytest.approx(2.065, abs=5e-4)
