"""Pin ``allometry.brown.crown_weight`` to Brown (1978) Table 1.

Brown, J.K. 1978. Weight and Density of Crowns of Rocky Mountain
Conifers. USDA For. Serv. Res. Pap. INT-197. Table 1, p. 10: "Live crown
weight equations for dominant and codominant trees greater than 1-inch
d.b.h." Weight is in pounds, diameter in inches.

Brown fits several predictors per species -- some use crown ratio,
tree height, or crown length alongside diameter. Only the
diameter-only forms are implemented, which are the ones FuelCalc uses
and the ones Gray & Reinhardt (2003) evaluated. :data:`TABLE_1` records
them as printed, transcribed by hand and repeated rather than imported,
so a typo in ``brown.py`` cannot hide by also being a typo here.

The companion module :mod:`test_brown_table16` pins the proportions
that split this weight into components.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.allometry import brown

# Brown letter -> {id, form, coefficients}, from the Equations column.
# "loglog" is w = EXP[a + b(lnd)]; "quadratic" is w = a + b(d^2),
# written by Brown with the constant last (e.g. WBP's 0.8371(d^2)-1.00).
TABLE_1 = {
    "GF": {"id": "GF", "form": "loglog", "a": 1.3094, "b": 1.6076},
    "L": {"id": "WL", "form": "loglog", "a": 0.4373, "b": 1.6786},
    "S": {"id": "ES", "form": "loglog", "a": 1.0404, "b": 1.7096},
    "AF": {"id": "SF", "form": "quadratic", "a": 7.345, "b": 1.255},
    "LP": {"id": "LP", "form": "loglog", "a": 0.1224, "b": 1.8820},
    "WP": {"id": "WP", "form": "loglog", "a": 0.7276, "b": 1.5497},
    "WBP": {"id": "WB", "form": "quadratic", "a": -1.00, "b": 0.8371},
    "C": {"id": "WC", "form": "loglog", "a": 0.8815, "b": 1.6389},
    "PP": {"id": "PP", "form": "loglog", "a": 0.2680, "b": 2.0740},
    "WH": {"id": "WH", "form": "loglog", "a": 0.7218, "b": 1.7502},
    # DF is the one two-part fit; see TestDouglasFir.
    "DF": {
        "id": "DF",
        "form": "piecewise",
        "a": 1.1368,
        "b": 1.5819,
        "c": 1.0237,
        "d": -20.74,
        "break_in": 17.0,
    },
}

DIAMETERS = np.round(np.arange(1.05, 45.0, 0.05), 4)


def printed(spec: dict, dia: np.ndarray) -> np.ndarray:
    """Evaluate a Table 1 equation exactly as printed."""
    if spec["form"] == "loglog":
        return np.exp(spec["a"] + spec["b"] * np.log(dia))
    if spec["form"] == "quadratic":
        return spec["a"] + spec["b"] * dia * dia
    return np.where(
        dia < spec["break_in"],
        np.exp(spec["a"] + spec["b"] * np.log(dia)),
        spec["c"] * dia * dia + spec["d"],
    )


@pytest.mark.parametrize("letter", sorted(TABLE_1))
def test_crown_weight_matches_table_1(letter):
    spec = TABLE_1[letter]
    ids = np.full(DIAMETERS.shape, spec["id"], dtype=object)
    # Brown's two quadratics dip below zero at the small end of the
    # range; brown.py floors there, so compare against the same floor.
    np.testing.assert_allclose(
        brown.crown_weight(ids, DIAMETERS),
        np.maximum(printed(spec, DIAMETERS), 0.0),
        atol=1e-12,
    )


def test_every_species_with_proportions_has_a_weight():
    """The eleven conifers Brown tabulates appear in both tables.

    Table 1 and Table 16 describe the same trees, so an Id with
    proportions but no crown weight would be a transcription gap. The
    Snell & Little hardwoods are the deliberate exception: their
    proportions come from PNW-GTR-151 and their weights are not
    implemented.
    """
    printed_ids = {spec["id"] for spec in TABLE_1.values()}
    assert set(brown.CROWN_WEIGHT_EQUATIONS) == printed_ids
    assert printed_ids <= set(brown.P1_EQUATIONS)


def test_hardwoods_raise():
    for eq_id in ("RA", "GC", "BM", "MA", "TO"):
        assert eq_id in brown.P1_EQUATIONS
        with pytest.raises(ValueError, match="Snell"):
            brown.crown_weight(np.array([eq_id], dtype=object), np.array([10.0]))


def test_quadratics_are_floored_at_zero():
    """WBP's fit is negative below 1.09 in, AF's is positive throughout.

    Brown fits WBP over 1-8 in, so just inside the low end the
    quadratic has not yet crossed zero. A negative crown weight would
    propagate as negative fuel.
    """
    dia = np.linspace(1.0, 1.2, 41)
    ids = np.full(dia.shape, "WB", dtype=object)
    weights = brown.crown_weight(ids, dia)
    assert (weights >= 0.0).all()
    assert weights[0] == 0.0  # printed value at 1.0 in is -0.163 lb
    assert weights[-1] > 0.0


class TestDouglasFir:
    """Table 1 gives Douglas-fir two diameter-only equations.

        w = EXP[1.1368 + 1.5819(lnd)],  for d <17 inches
        w = 1.0237d^2 - 20.74,          for d >=17 inches

    They are halves of one predictor, not alternatives, which the tests
    below establish from the numbers rather than from the layout.
    """

    def test_the_branches_meet_at_the_break(self):
        """Agreement at 17 in is what identifies them as one fit.

        Two independent fits to the same data would not be expected to
        agree to a fraction of a percent at one particular diameter.
        """
        d = 17.0
        low = np.exp(1.1368 + 1.5819 * np.log(d))
        high = 1.0237 * d * d - 20.74
        assert abs(low - high) / low < 0.002

    def test_both_branches_are_used(self):
        ids = np.array(["DF", "DF"], dtype=object)
        got = brown.crown_weight(ids, np.array([16.9, 17.1]))
        assert got[0] == pytest.approx(np.exp(1.1368 + 1.5819 * np.log(16.9)))
        assert got[1] == pytest.approx(1.0237 * 17.1**2 - 20.74)

    def test_the_only_step_down_is_the_break_itself(self):
        """Brown fitted the two branches separately, so they do not join
        exactly: crown weight steps down once, by 0.15%, at 17 in. That
        is a property of the published equations and is left alone --
        forcing continuity would mean shipping a curve Brown did not
        publish. Everywhere else the weight increases with diameter.
        """
        dia = np.round(np.arange(1.05, 45.0, 0.01), 4)
        ids = np.full(dia.shape, "DF", dtype=object)
        steps = np.diff(brown.crown_weight(ids, dia))
        down = np.flatnonzero(steps < 0)
        assert down.size == 1
        crossed = dia[down[0] + 1]
        assert crossed == pytest.approx(17.0, abs=0.011)
        weight_at_break = brown.crown_weight(
            np.array(["DF"], dtype=object), np.array([17.0])
        )[0]
        assert abs(steps[down[0]]) / weight_at_break < 0.002

    def test_the_upper_branch_matters(self):
        """Dropping it, as FuelCalc's table does, loses a third by 30 in."""
        d = 30.0
        low_only = np.exp(1.1368 + 1.5819 * np.log(d))
        got = brown.crown_weight(np.array(["DF"], dtype=object), np.array([d]))[0]
        assert got / low_only > 1.3
