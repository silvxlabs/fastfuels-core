"""Pin ``allometry.brown`` to Brown (1978) Table 16, the primary source.

Brown, J.K. 1978. Weight and Density of Crowns of Rocky Mountain
Conifers. USDA For. Serv. Res. Pap. INT-197. Table 16, p. 53:
"Accumulative proportions of foliage and branchwood by size classes for
live crowns of dominants greater than 1 inch d.b.h."

This module is the authority for the coefficients themselves.
:mod:`tests.canopy_fuel.test_fuelcalc_parity` covers the surrounding
algorithm against its canonical implementation; correctness of the
equations is settled here, against the page.

:data:`TABLE_16` is transcribed by hand from the page, including the
Conditions column. It deliberately repeats the coefficients rather than
importing them, so a typo in ``brown.py`` cannot hide by also being a
typo here. Brown's species letters differ from FuelCalc's two-letter
equation Ids; the mapping is recorded per row.

Two details of the printed page that are easy to get wrong:

- WP's P2 is ``0.914 - 0.0978*sqrt(d)``. The radical is small in print
  and text extraction renders it as ``0.0978/d``.
- Brown gives no Conditions entry for L (western larch), which is the
  evidence that its P2 curve never crosses P1 -- every species whose
  curves do cross gets an explicit high-diameter override.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.allometry import brown

# Equation forms as printed in the Function column.
FORMS = {
    "exp": lambda d, a, b: a * np.exp(b * d),
    "linear": lambda d, a, b: a + b * d,
    "reciprocal": lambda d, a, b: 1.0 / (a + b * d),
    "sqrt": lambda d, a, b: a + b * np.sqrt(d),
}

# Brown letter -> {id, P1, P2, conditions}. P1/P2 are (form, a, b).
# "high" is the Conditions-column override (break_in, P1, P2), absent
# for the species Brown prints no P1/P2 condition for.
TABLE_16 = {
    "GF": {
        "id": "GF",
        "P1": ("reciprocal", 1.592, 0.0529),
        "P2": ("reciprocal", 1.150, 0.0416),
        "high": (36.0, 0.286, 0.378),
    },
    "L": {
        "id": "WL",
        "P1": ("exp", 0.347, -0.0434),
        "P2": ("exp", 0.745, -0.0362),
    },
    "S": {
        "id": "ES",
        "P1": ("exp", 0.578, -0.0325),
        "P2": ("exp", 0.852, -0.0281),
        "high": (40.0, 0.158, 0.277),
    },
    "AF": {
        "id": "SF",
        "P1": ("exp", 0.597, -0.0425),
        "P2": ("exp", 0.864, -0.0373),
    },
    "LP": {
        "id": "LP",
        "P1": ("linear", 0.493, -0.0117),
        "P2": ("linear", 0.777, -0.0146),
    },
    "WP": {
        "id": "WP",
        "P1": ("exp", 0.550, -0.0345),
        "P2": ("sqrt", 0.914, -0.0978),
    },
    "WBP": {
        "id": "WB",
        "P1": ("exp", 0.512, -0.0374),
        "P2": ("exp", 0.864, -0.0585),
        "high": (20.0, 0.242, 0.268),
    },
    "C": {
        "id": "WC",
        "P1": ("exp", 0.617, -0.0233),
        "P2": ("exp", 0.756, -0.0241),
    },
    "PP": {
        "id": "PP",
        "P1": ("exp", 0.558, -0.0475),
        "P2": ("exp", 0.625, -0.0511),
    },
    "DF": {
        "id": "DF",
        "P1": ("exp", 0.484, -0.0210),
        "P2": ("exp", 0.729, -0.0233),
        "high": (36.0, 0.227, 0.315),
    },
    "WH": {
        "id": "WH",
        "P1": ("exp", 0.547, -0.0370),
        "P2": ("exp", 0.835, -0.0380),
        "high": (40.0, 0.125, 0.183),
    },
}


def evaluate(spec: tuple, dia: np.ndarray) -> np.ndarray:
    """Evaluate a printed Table 16 function."""
    form, a, b = spec
    return FORMS[form](dia, a, b)


def fitted_range(letter: str) -> np.ndarray:
    """Diameters where both sides use the fitted curve.

    Table 16 covers dominants over 1 inch. Above a species' Conditions
    break the curve is replaced by a constant, and past PP's crossing
    our P2 is overridden, so those regions are checked separately.
    """
    spec = TABLE_16[letter]
    top = spec.get("high", (60.0,))[0]
    if spec["id"] == "PP":
        top = min(top, brown.PP_CROSSOVER_IN)
    return np.arange(1.05, top, 0.05)


@pytest.mark.parametrize("letter", sorted(TABLE_16))
class TestPrintedCurves:
    """Every fitted P1/P2 curve, against the page."""

    def test_p1_matches(self, letter):
        spec = TABLE_16[letter]
        dia = fitted_range(letter)
        ids = np.full(dia.shape, spec["id"], dtype=object)
        np.testing.assert_allclose(
            brown.foliage_fraction(ids, dia),
            np.clip(evaluate(spec["P1"], dia), 0.0, brown.PROPORTION_MAX),
            atol=1e-12,
            err_msg=f"Brown species {letter} -> Id {spec['id']} P1",
        )

    def test_p2_matches(self, letter):
        spec = TABLE_16[letter]
        dia = fitted_range(letter)
        ids = np.full(dia.shape, spec["id"], dtype=object)
        np.testing.assert_allclose(
            brown.foliage_plus_fine_fraction(ids, dia),
            np.clip(evaluate(spec["P2"], dia), 0.0, brown.PROPORTION_MAX),
            atol=1e-12,
            err_msg=f"Brown species {letter} -> Id {spec['id']} P2",
        )

    def test_p2_stays_above_p1(self, letter):
        """An accumulative proportion cannot decrease with size class.

        P2 accumulates foliage plus 0-1/4 in branchwood, so P2 >= P1 for
        every diameter Brown's own Conditions leave to the fitted curve.
        This is the invariant FuelCalc's transposed larch coefficient
        breaks at 38.6 in.
        """
        spec = TABLE_16[letter]
        dia = fitted_range(letter)
        fine = evaluate(spec["P2"], dia) - evaluate(spec["P1"], dia)
        worst = dia[fine.argmin()]
        assert fine.min() > 0.0, (
            f"Brown species {letter}: printed P2 falls to or below P1 at "
            f"d={worst:.2f} in (fine fraction {fine.min():+.5f})"
        )


class TestConditionsColumn:
    """The Conditions column, and the species that have no entry."""

    @pytest.mark.parametrize(
        "letter", sorted(k for k, v in TABLE_16.items() if "high" in v)
    )
    def test_high_diameter_constants_match(self, letter):
        spec = TABLE_16[letter]
        assert brown.LARGE_DIAMETER_CONSTANTS[spec["id"]] == spec["high"]

    def test_no_constants_we_invented(self):
        """We must not carry an override Brown does not print."""
        printed = {v["id"] for v in TABLE_16.values() if "high" in v}
        assert set(brown.LARGE_DIAMETER_CONSTANTS) == printed

    def test_larch_needs_no_override(self):
        """Brown prints no larch Conditions, and none is needed.

        Brown gives a high-diameter override for exactly the species
        whose fitted P1 and P2 curves cross, so the absence of a larch
        entry is itself a check on the coefficients: read correctly,
        larch's P2 must stay above its P1 across the whole range.
        """
        assert "WL" not in brown.LARGE_DIAMETER_CONSTANTS
        dia = np.arange(1.05, 60.0, 0.01)
        spec = TABLE_16["L"]
        assert (evaluate(spec["P2"], dia) > evaluate(spec["P1"], dia)).all()


class TestPonderosaOverride:
    """Brown's PP condition, whose printed inequality is inverted.

    Table 16 reads "If d <=31 in, P2 = P1 + 0.01". Applied as printed it
    would supersede a curve Brown fits with R2 = 0.89 across almost the
    whole range, and contradict his Table 15 (p. 52), which gives
    ponderosa a fine fraction of 0.14 at d <= 1 in. Above the crossing
    it instead repairs a sign error, which is where we apply it.
    """

    def test_curves_cross_just_past_the_printed_break(self):
        a1, b1 = TABLE_16["PP"]["P1"][1:]
        a2, b2 = TABLE_16["PP"]["P2"][1:]
        crossing = np.log(a2 / a1) / (b1 - b2)
        assert crossing == pytest.approx(31.5, abs=0.05)
        assert brown.PP_CROSSOVER_IN <= crossing

    def test_fine_fraction_would_go_negative_untreated(self):
        spec = TABLE_16["PP"]
        fine = evaluate(spec["P2"], 32.0) - evaluate(spec["P1"], 32.0)
        assert fine < 0.0

    def test_override_holds_the_fine_fraction_positive(self):
        dia = np.arange(1.05, 60.0, 0.05)
        ids = np.full(dia.shape, "PP", dtype=object)
        fine = brown.foliage_plus_fine_fraction(ids, dia) - brown.foliage_fraction(
            ids, dia
        )
        assert (fine > 0.0).all()

    def test_printed_reading_would_contradict_table_15(self):
        """Table 15 gives ponderosa 0.14 fine at d <= 1 in, not 0.01.

        Read as printed, the override would force the fine fraction to
        0.01 for every tree up to 31 in. The fitted curves put it six
        times higher than that at the bottom of the range, and Brown's
        own small-tree table puts it fourteen times higher.
        """
        spec = TABLE_16["PP"]
        fitted = evaluate(spec["P2"], 1.0) - evaluate(spec["P1"], 1.0)
        assert fitted == pytest.approx(0.062, abs=0.001)
        assert fitted > 5 * 0.01
