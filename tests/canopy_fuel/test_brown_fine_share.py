"""The fine branchwood share, and the species the proportions reach.

``brown.fine_branchwood_share`` is the derived quantity available canopy
fuel is built from: the fraction of *branchwood* that is 0-1/4 in
material, ``(P2 - P1) / (1 - P1)``. P1 and P2 are accumulative
proportions of the whole crown, so this is not P2 and not P2 - P1, and
getting it wrong is silent -- the numbers stay in [0, 1] either way.

The coefficients themselves are pinned against the printed page in
:mod:`tests.canopy_fuel.test_brown_table16`, which is their authority.
This module covers only what that one does not: the composition above,
the clamps, and the Snell & Little (1983) hardwoods.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.allometry.brown import (
    fine_branchwood_share,
    foliage_fraction,
)


def share(eq_id, dia_in):
    ids = np.array([eq_id])
    return fine_branchwood_share(ids, ids, np.array([float(dia_in)]))[0]


class TestComposition:
    def test_it_is_the_fine_fraction_of_branchwood_not_of_the_crown(self):
        """PP at 10 in: 0.0428, not P2 / (1 - P1) which is about 0.57.

        Both readings are plausible-looking numbers in [0, 1], so this
        is the assertion that separates them.
        """
        p1 = 0.558 * np.exp(-0.475)
        p2 = 0.625 * np.exp(-0.511)
        assert share("PP", 10.0) == pytest.approx((p2 - p1) / (1 - p1), rel=1e-12)
        assert share("PP", 10.0) == pytest.approx(0.0428, abs=5e-4)


class TestClamps:
    def test_ponderosa_holds_at_a_hundredth_past_the_crossing(self):
        """Above ~31.5 in the fitted P1 and P2 cross; Brown pins the gap."""
        p1 = foliage_fraction(np.array(["PP"]), np.array([35.0]))[0]
        assert share("PP", 35.0) == pytest.approx(0.01 / (1 - p1), rel=1e-12)

    def test_lodgepole_reaches_zero_where_its_linear_fits_meet(self):
        assert share("LP", 60.0) == 0.0

    @pytest.mark.parametrize(
        "eq_id", ["PP", "GF", "DF", "LP", "WP", "WB", "ES", "WH", "AL", "RA"]
    )
    def test_the_share_stays_a_fraction_over_the_whole_range(self, eq_id):
        dia = np.linspace(1.0, 80.0, 200)
        ids = np.full(dia.shape, eq_id)
        got = fine_branchwood_share(ids, ids, dia)
        assert (got >= 0.0).all() and (got <= 1.0).all()


class TestSpeciesCoverage:
    def test_a_snell_and_little_hardwood_resolves(self):
        """GC is absent from the FuelCalc guide but printed in SL-83
        Table 3 (p. 6): ``f(1) = 1 / (1.6048 + 0.5630 d**0.6828)``."""
        got = foliage_fraction(np.array(["GC"]), np.array([10.0]))
        np.testing.assert_allclose(
            got, 1.0 / (1.6048 + 0.5630 * 10.0**0.6828), rtol=1e-12
        )

    def test_an_id_with_no_published_equations_raises(self):
        with pytest.raises(ValueError, match="PY"):
            foliage_fraction(np.array(["PY"]), np.array([8.0]))

    def test_quaking_aspen_raises(self):
        """QA once borrowed WB's P1 and WL's P2, a pairing no source gives."""
        with pytest.raises(ValueError, match="QA"):
            share("QA", 8.0)
