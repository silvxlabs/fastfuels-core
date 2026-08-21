"""Pin ``allometry.jenkins`` to Jenkins et al. (2003).

Jenkins, J.C., Chojnacky, D.C., Heath, L.S., Birdsey, R.A. 2003.
National-scale biomass estimators for United States tree species.
*Forest Science* 49(1): 12-35.

The coefficient tables are transcribed here by hand from the paper, so a
typo in ``REF_JENKINS.csv`` cannot hide by also being a typo in the test:

- **Table 4** (p. 20): total aboveground biomass ``bm = Exp(b0 + b1 ln dbh)``,
  dbh in cm, kg dry weight, one row per species group.
- **Table 6** (p. 24): component ratios ``ratio = Exp(b0 + b1 / dbh)`` of a
  component to total aboveground biomass, one set for hardwood and one for
  softwood. Branch is not fit directly: the paper takes "branch (bark and
  wood) biomass ... by difference", so branch is the aboveground residual
  after foliage, stem bark and stem wood. Coarse roots are belowground and
  are not part of that residual.

Groups 1-5 are the softwood groups and 6-9 the hardwood groups; the
woodland group (10) is priced with the hardwood component ratios in
``REF_JENKINS.csv``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuels_core.allometry import jenkins
from fastfuels_core.ref_data import REF_SPECIES
from fastfuels_core.trees import JenkinsBiomassEquations

# Table 4: species group -> (b0, b1) for total aboveground biomass.
TABLE_4 = {
    1: (-2.0336, 2.2592),  # Cedar/larch
    2: (-2.2304, 2.4435),  # Douglas-fir
    3: (-2.5384, 2.4814),  # True fir/hemlock
    4: (-2.5356, 2.4349),  # Pine  (b1 is POSITIVE; guards the sign fix)
    5: (-2.0773, 2.3323),  # Spruce
    6: (-2.2094, 2.3867),  # Aspen/alder/cottonwood/willow
    7: (-1.9123, 2.3651),  # Soft maple/birch
    8: (-2.4800, 2.4835),  # Mixed hardwood
    9: (-2.0127, 2.4342),  # Hard maple/oak/hickory/beech
    10: (-0.7152, 1.7029),  # Woodland juniper/oak/mesquite
}

# Table 6: component -> (b0, b1), one set per broad class.
TABLE_6 = {
    "softwood": {
        "foliage": (-2.9584, 4.4766),
        "stem_bark": (-2.0980, -1.1432),
        "stem_wood": (-0.3737, -1.8055),
    },
    "hardwood": {
        "foliage": (-4.0813, 5.8816),
        "stem_bark": (-2.0129, -1.6805),
        "stem_wood": (-0.3065, -5.4240),
    },
}

SOFTWOOD_GROUPS = {1, 2, 3, 4, 5}


def _class_for_group(group: int) -> str:
    return "softwood" if group in SOFTWOOD_GROUPS else "hardwood"


def _expected_agb(group: int, dbh: float) -> float:
    b0, b1 = TABLE_4[group]
    return float(np.exp(b0 + b1 * np.log(dbh)))


def _expected_ratio(group: int, component: str, dbh: float) -> float:
    b0, b1 = TABLE_6[_class_for_group(group)][component]
    return float(np.exp(b0 + b1 / dbh))


# A representative species per group, verified to route through REF_SPECIES.
SAMPLE_SPECIES = {
    122: 4,  # ponderosa pine
    202: 2,  # Douglas-fir
    746: 6,  # quaking aspen
    316: 7,  # red maple
    802: 9,  # white oak
    65: 10,  # Utah juniper (woodland)
}


class TestAbovegroundBiomass:
    def test_matches_table_4_every_group(self):
        # dbh above the sapling threshold, so this is Eq. 1 alone.
        dbh = 30.0
        for spcd in REF_SPECIES.index:
            group = int(REF_SPECIES.loc[spcd, "JENKINS_SPGRPCD"])
            got = jenkins.above_ground_biomass(np.array([spcd]), np.array([dbh]))[0]
            assert got == pytest.approx(_expected_agb(group, dbh))

    def test_pine_sign_regression(self):
        # The whole pine group collapsed to ~0 when b1 carried a stray minus
        # sign; a 24 cm ponderosa is on the order of 100 kg, not 1e-4 kg.
        agb = jenkins.above_ground_biomass(np.array([122]), np.array([24.0]))[0]
        assert 100.0 < agb < 400.0

    def test_sapling_adjustment_below_threshold(self):
        from fastfuels_core.ref_data import REF_JENKINS

        spcd, group, dbh = 122, 4, 10.0  # <= 12.7 cm
        factor = REF_JENKINS.loc[group, "JENKINS_SAPLING_ADJUSTMENT"]
        got = jenkins.above_ground_biomass(np.array([spcd]), np.array([dbh]))[0]
        assert got == pytest.approx(_expected_agb(group, dbh) * factor)

    def test_no_sapling_adjustment_above_threshold(self):
        spcd, group, dbh = 122, 4, 13.0  # > 12.7 cm
        got = jenkins.above_ground_biomass(np.array([spcd]), np.array([dbh]))[0]
        assert got == pytest.approx(_expected_agb(group, dbh))


class TestComponentBiomass:
    @pytest.mark.parametrize("spcd,group", SAMPLE_SPECIES.items())
    def test_foliage_matches_table_6(self, spcd, group):
        dbh = 30.0
        expected = _expected_agb(group, dbh) * _expected_ratio(group, "foliage", dbh)
        got = jenkins.foliage_biomass(np.array([spcd]), np.array([dbh]))[0]
        assert got == pytest.approx(expected)

    @pytest.mark.parametrize("spcd,group", SAMPLE_SPECIES.items())
    def test_branch_is_aboveground_residual(self, spcd, group):
        dbh = 30.0
        agb = _expected_agb(group, dbh)
        residual = 1.0
        for component in ("foliage", "stem_bark", "stem_wood"):
            residual -= _expected_ratio(group, component, dbh)
        got = jenkins.branch_biomass(np.array([spcd]), np.array([dbh]))[0]
        assert got == pytest.approx(agb * residual)

    def test_components_do_not_exceed_aboveground(self):
        # foliage + branch < aboveground for every species over a dbh sweep
        # (roots are belowground and excluded from the residual).
        spcd = np.array(REF_SPECIES.index)
        for dbh in (2.5, 10.0, 30.0, 80.0):
            d = np.full(spcd.shape, dbh)
            agb = jenkins.above_ground_biomass(spcd, d)
            foliage = jenkins.foliage_biomass(spcd, d)
            branch = jenkins.branch_biomass(spcd, d)
            assert np.all(branch > 0)
            assert np.all(foliage + branch < agb)


class TestVectorized:
    def test_matches_scalar_tree_adapter(self):
        for spcd in (122, 202, 316, 746):
            for dbh in (8.0, 24.0, 55.0):
                vector = jenkins.foliage_biomass(np.array([spcd]), np.array([dbh]))[0]
                scalar = JenkinsBiomassEquations(spcd, dbh).estimate_foliage_biomass()
                assert vector == pytest.approx(scalar)

    def test_batch_equals_elementwise(self):
        spcd = np.array([122, 202, 316, 746, 802])
        dbh = np.array([12.0, 20.0, 33.0, 8.0, 47.0])
        batch = jenkins.branch_biomass(spcd, dbh)
        one_by_one = np.array(
            [
                jenkins.branch_biomass(np.array([s]), np.array([d]))[0]
                for s, d in zip(spcd, dbh)
            ]
        )
        np.testing.assert_allclose(batch, one_by_one)


class TestErrors:
    def test_unknown_species_raises_valueerror(self):
        with pytest.raises(ValueError, match="not in the FIA reference table"):
            jenkins.above_ground_biomass(np.array([99999999]), np.array([30.0]))
