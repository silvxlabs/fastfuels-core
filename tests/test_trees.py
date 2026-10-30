# Core imports
from pathlib import Path

# Internal imports

from fastfuels_core.trees import (
    Tree,
    TreeSample,
    TreePopulation,
    JenkinsBiomassEquations,
    NSVBEquations,
    REF_SPECIES,
    REF_JENKINS,
    REF_TRY_DB_LEAF,
)
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile
from fastfuels_core.crown_profile_models.cone import ConeCrownProfile
from fastfuels_core.crown_profile_models.cylinder import CylinderCrownProfile
from fastfuels_core.crown_profile_models.paraboloid import ParaboloidCrownProfile
from fastfuels_core.crown_profile_models.ellipsoid import EllipsoidCrownProfile
from fastfuels_core.allometry.crown_ratio import estimate_crown_base_height
from tests.utils import make_random_tree

# External imports
import pytest
import numpy as np
import pandas as pd
from pandera.errors import SchemaError

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"


class TestBetaCrownProfileModel:

    def test_get_radius_at_height(self):
        """
        Tests that the radius at a given height is calculated correctly.
        """
        model = BetaCrownProfile(
            species_code=122, crown_base_height=10, crown_length=10
        )

        # Test that the bottom of the crown and the top of the crown both have
        # a radius of 0
        assert model.get_radius_at_height(10) == 0.0
        assert model.get_radius_at_height(20) == 0.0

        # Test with a single float within the range of the tree's height
        assert model.get_radius_at_height(15) > 0

        # Test with a single float outside the range of the tree's height
        assert model.get_radius_at_height(-1) == 0
        assert model.get_radius_at_height(21) == 0

        # Test with a numpy array of floats, all within the range of the tree's height
        height_values = np.array([12, 14, 16, 18])
        radii = model.get_radius_at_height(height_values)
        assert len(radii) == len(height_values)
        assert all(radii > 0)

        # Test with a numpy array of floats, some within and some outside the range of the tree's height
        height_values = np.array([-1, 10, 15, 20, 21])
        radii = model.get_radius_at_height(height_values)
        assert len(radii) == len(height_values)
        assert radii[0] == 0
        assert radii[1] == 0
        assert radii[2] > 0
        assert radii[3] == 0
        assert radii[4] == 0

    def test_get_normalized_height(self):
        """
        Tests that the normalized height is calculated correctly.
        """
        # Suppose that I have a tree with a height of 20m, a crown base height
        # of 10m, and a crown length of 10m
        model = BetaCrownProfile(
            species_code=122, crown_base_height=10, crown_length=10
        )
        assert model._get_normalized_height(0) < 0
        assert model._get_normalized_height(5) < 0
        assert model._get_normalized_height(10) == 0
        assert model._get_normalized_height(15) == 0.5
        assert model._get_normalized_height(20) == 1
        assert model._get_normalized_height(25) > 1

        # Suppose that I have a tree with a height of 100m, a crown base height
        # of 20m, and a crown length of 80m
        model = BetaCrownProfile(
            species_code=122, crown_base_height=20, crown_length=80
        )
        assert model._get_normalized_height(0) < 0
        assert model._get_normalized_height(10) < 0
        assert model._get_normalized_height(20) == 0
        assert model._get_normalized_height(40) == 0.25
        assert model._get_normalized_height(60) == 0.5
        assert model._get_normalized_height(80) == 0.75
        assert model._get_normalized_height(100) == 1
        assert model._get_normalized_height(120) > 1

        # Test with numpy
        heights = np.linspace(10, 20, 100)
        model = BetaCrownProfile(
            species_code=122, crown_length=10, crown_base_height=10
        )
        normalized_heights = model._get_normalized_height(heights).reshape(-1)
        assert len(normalized_heights) == 100
        assert normalized_heights[0] == 0
        assert normalized_heights[-1] == 1.0
        for i in range(1, 100):
            assert normalized_heights[i] > normalized_heights[i - 1]

    def test_get_radius_at_normalized_height(self):
        """
        Tests that the radius at a given normalized height is calculated correctly.
        """
        model = BetaCrownProfile(
            species_code=122, crown_base_height=10, crown_length=10
        )

        # Test that the bottom of the crown and the top of the crown both have
        # a radius of 0
        assert model._get_radius_at_normalized_height(0) == 0.0
        assert model._get_radius_at_normalized_height(1) == 0.0

        # Test with a single float within the range [0, 1]
        assert model._get_radius_at_normalized_height(0.5) > 0

        # Test with a single float outside the range [0, 1]
        assert model._get_radius_at_normalized_height(-1) == 0.0
        assert model._get_radius_at_normalized_height(2) == 0.0

        # Test with a numpy array of floats, all within the range [0, 1]
        z_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        radii = model._get_radius_at_normalized_height(z_values).reshape(-1)
        assert len(radii) == len(z_values)
        assert all(radii > 0)

        # Test with a numpy array of floats, some within and some outside the range [0, 1]
        z_values = np.array([-1, 0, 0.5, 1, 2])
        radii = model._get_radius_at_normalized_height(z_values).reshape(-1)
        assert len(radii) == len(z_values)
        assert radii[0] == 0.0
        assert radii[1] == 0.0
        assert radii[2] > 0
        assert radii[3] == 0.0
        assert radii[4] == 0.0

    def test_get_max_radius(self):
        """
        Tests that the maximum radius of the crown is calculated correctly.
        """
        # Test with species group 4, crown base height 10, and crown length 10
        model = BetaCrownProfile(
            species_code=122, crown_base_height=10, crown_length=10
        )
        max_crown_radius = model.get_max_radius()
        assert max_crown_radius > 0
        heights = np.linspace(10, 20, 1000)
        radii = model.get_radius_at_height(heights)
        assert np.all(max_crown_radius >= radii)

        # Test with species group 1, crown base height 5, and crown length 15
        model = BetaCrownProfile(species_code=202, crown_base_height=5, crown_length=15)
        max_crown_radius = model.get_max_radius()
        assert max_crown_radius > 0
        heights = np.linspace(5, 20, 1000)
        radii = model.get_radius_at_height(heights)
        assert np.all(max_crown_radius >= radii)

        # Test with species group 2, crown base height 20, and crown length 5
        model = BetaCrownProfile(species_code=747, crown_base_height=20, crown_length=5)
        max_crown_radius = model.get_max_radius()
        assert max_crown_radius > 0
        heights = np.linspace(20, 25, 1000)
        radii = model.get_radius_at_height(heights)
        assert np.all(max_crown_radius >= radii)


class TestCrownModelSpeedups:
    """De-pandas reference lookups must be value-exact, and crown_profile_model
    must be cached so it is built once per tree."""

    @pytest.mark.parametrize("spcd", [122, 202, 15, 93, 108, 17, 19, 746, 316])
    def test_beta_params_match_reference_table(self, spcd):
        model = BetaCrownProfile(spcd, crown_base_height=5.0, crown_length=15.0)
        group = REF_SPECIES.loc[spcd]["JENKINS_SPGRPCD"]
        row = REF_JENKINS.loc[group]
        assert model.a.item() == row["BETA_CANOPY_a"]
        assert model.b.item() == row["BETA_CANOPY_b"]
        assert model.c.item() == row["BETA_CANOPY_c"]
        assert model.beta_norm.item() == row["BETA_CANOPY_NORM"]

    @pytest.mark.parametrize("spcd", [122, 202, 15, 93, 108])
    def test_purves_trait_score_matches_reference_table(self, spcd):
        model = PurvesCrownProfile(spcd, dbh=25.0, height=18.0, crown_base_height=7.2)
        assert model.trait_score.item() == REF_SPECIES.loc[spcd]["PURVES_TRAIT_SCORE"]

    def test_crown_profile_model_is_cached(self):
        tree = make_random_tree(crown_profile_model="beta")
        first = tree.crown_profile_model
        assert tree.crown_profile_model is first  # same object, not rebuilt

    def test_max_crown_radius_reuses_cached_model(self):
        # max_crown_radius routes through the cached model; accessing it must not
        # invalidate or rebuild the cached crown_profile_model.
        tree = make_random_tree(crown_profile_model="beta")
        model = tree.crown_profile_model
        _ = tree.max_crown_radius
        assert tree.crown_profile_model is model


class TestJenkinsBiomassEquations:
    def test_init(self):
        # Test that the class can be initialized
        model = JenkinsBiomassEquations(species_code=122, diameter=24)
        assert model._species_group == 4

        # Test with a different species code
        model = JenkinsBiomassEquations(species_code=989, diameter=100)
        assert model._species_group == 8

        # Test with an invalid species code
        with pytest.raises(ValueError):
            JenkinsBiomassEquations(species_code=32480910, diameter=100)

        # Test with an invalid diameter
        with pytest.raises(ValueError):
            JenkinsBiomassEquations(species_code=122, diameter=-100)

        # Test initializing all species codes
        for species_code in REF_SPECIES.index:
            JenkinsBiomassEquations(species_code=int(species_code), diameter=100)

    def test_estimate_above_ground_biomass(self):
        # Test with a known species code and diameter
        model = JenkinsBiomassEquations(species_code=122, diameter=24)
        biomass = model._estimate_above_ground_biomass()
        assert biomass > 0, "Biomass should be a positive number"

        # Test each species code with a random diameter
        for species_code in REF_SPECIES.index:
            diameter = np.random.uniform(0, 100)
            model = JenkinsBiomassEquations(
                species_code=int(species_code), diameter=diameter
            )
            if model._species_group == -1:
                continue
            biomass = model._estimate_above_ground_biomass()
            assert (
                biomass > 0
            ), f"{species_code} and {diameter} should yield a positive biomass"

    def test_estimate_foliage_biomass(self):
        # Test with a known species code and diameter
        model = JenkinsBiomassEquations(species_code=122, diameter=24)
        biomass = model.estimate_foliage_biomass()
        assert biomass > 0, "Biomass should be a positive number"

        # Test each species code with a random diameter
        for species_code in REF_SPECIES.index:
            diameter = np.random.uniform(0, 100)
            model = JenkinsBiomassEquations(
                species_code=int(species_code), diameter=diameter
            )
            if model._species_group == -1:
                continue
            biomass = model.estimate_foliage_biomass()
            assert (
                biomass > 0
            ), f"{species_code} and {diameter} should yield a positive biomass"


class TestNSVBEquations:
    def test_init(self):
        # Test that the class can be initialized
        NSVBEquations(species_code=122, diameter=24, height=20)

        # Test with a different species code
        NSVBEquations(species_code=989, diameter=100, height=20)

        # Test with an invalid species code
        with pytest.raises(ValueError):
            NSVBEquations(species_code=32480910, diameter=100, height=20)

        # Test with an invalid diameter
        with pytest.raises(ValueError):
            NSVBEquations(species_code=122, diameter=-100, height=20)

    def test_estimate_foliage_biomass_example_1(self):
        # Compare to Example 1 of the GTR
        dia_in = 20
        dia_cm = dia_in * 2.54
        ht_ft = 110
        ht_m = ht_ft * 0.3048
        division = "240"
        model = NSVBEquations(
            species_code=202, diameter=dia_cm, height=ht_m, division=division
        )
        foliage_biomass = model.estimate_foliage_biomass()

        foliage_biomass_lb = foliage_biomass * 2.20462
        assert np.isclose(foliage_biomass_lb, 83.63478892024017)

    def test_estimate_foliage_biomass_example_2(self):
        # Compare to Example 2 of the GTR
        spcd = 316
        dia_in = 11.1
        dia_cm = dia_in * 2.54
        ht_ft = 38
        ht_m = ht_ft * 0.3048
        division = "M210"
        model = NSVBEquations(
            species_code=spcd, diameter=dia_cm, height=ht_m, division=division
        )
        foliage_biomass = model.estimate_foliage_biomass()

        foliage_biomass_lb = foliage_biomass * 2.20462
        assert np.isclose(foliage_biomass_lb, 22.807960563788)


@pytest.mark.parametrize("jenkins_species_group", REF_JENKINS.index)
def test_foliage_sav(jenkins_species_group):
    """
    tests that the foliage sav property is available for all jenkins species groups
    """
    # Get a random species code from the REF_SPECIES table where JENKINS_SPGRPCD equals the jenkins_species_group parameter
    spcd_spgrp_table = REF_SPECIES[
        REF_SPECIES["JENKINS_SPGRPCD"] == jenkins_species_group
    ].copy()
    random_spcd = np.random.choice(spcd_spgrp_table.index, 1)[0]

    tree = make_random_tree(species_code=random_spcd)
    assert tree.foliage_sav > 0


@pytest.mark.parametrize("spcd", REF_TRY_DB_LEAF.index.dropna())
def test_specific_leaf_area(spcd):
    """
    tests that the specific leaf area property is available for all SPCDs
    """

    tree = make_random_tree(species_code=spcd)
    assert tree.specific_leaf_area > 0


@pytest.mark.parametrize("spcd", REF_TRY_DB_LEAF.index.dropna())
def test_mean_leaf_angle(spcd):
    """
    tests that the mean leaf angle property is either available or set to nan for all SPCDs
    """

    tree = make_random_tree(species_code=spcd)
    assert not np.isinf(tree.mean_leaf_angle)


def test_tree():
    tree = Tree(
        species_code=122,
        status_code=1,
        diameter=5,
        height=20,
        crown_base_height=20 - 20 * 0.5,
    )
    assert isinstance(tree.biomass_allometry_model, NSVBEquations)


class TestTreeCrownBaseHeight:
    """crown_base_height sets the crown geometry; crown_ratio is data only."""

    @staticmethod
    def _tree(**kwargs):
        defaults = {"species_code": 122, "status_code": 1, "diameter": 25.0}
        return Tree(**{**defaults, "height": 15.0, **kwargs})

    # ── Signature ─────────────────────────────────────────────────────────

    def test_positional_fifth_argument_raises(self):
        with pytest.raises(TypeError):
            Tree(122, 1, 25.0, 15.0, 0.5)

    @pytest.mark.parametrize("cbh", [None, float("nan"), pd.NA])
    def test_neither_crown_base_height_nor_crown_ratio_raises(self, cbh):
        with pytest.raises(ValueError, match="crown_base_height, or a crown_ratio"):
            self._tree(crown_base_height=cbh)

    # ── crown_base_height validation ──────────────────────────────────────

    @pytest.mark.parametrize(
        "cbh", [-0.1, 15.1, float("nan"), float("inf"), float("-inf")]
    )
    def test_invalid_crown_base_height_raises(self, cbh):
        with pytest.raises(ValueError, match="crown_base_height"):
            self._tree(crown_base_height=cbh)

    @pytest.mark.parametrize("cbh", [None, pd.NA])
    def test_missing_crown_base_height_value_raises(self, cbh):
        with pytest.raises(ValueError, match="crown_base_height"):
            self._tree(crown_base_height=cbh)

    def test_nan_height_raises(self):
        with pytest.raises(ValueError, match="crown_base_height"):
            self._tree(height=float("nan"), crown_base_height=7.5)

    def test_float32_crown_base_height_above_height_raises(self):
        # float32(20.1) is 20.100000381 > 20.1; a float32 comparison hid it
        with pytest.raises(ValueError, match="crown_base_height"):
            self._tree(height=20.1, crown_base_height=np.float32(20.1))

    def test_float32_inputs_stored_as_python_floats(self):
        tree = self._tree(height=np.float32(20.1), crown_base_height=np.float32(20.1))
        assert type(tree.height) is float
        assert type(tree.crown_base_height) is float
        assert tree.crown_base_height <= tree.height

    def test_crown_base_height_equal_to_height_accepted(self):
        """A tree with no live crown: crown base at the top, zero crown length."""
        tree = self._tree(crown_base_height=15.0)
        assert tree.crown_base_height == 15.0
        assert tree.crown_length == 0.0

    def test_crown_base_height_zero_accepted(self):
        tree = self._tree(crown_base_height=0.0)
        assert tree.crown_base_height == 0.0
        assert tree.crown_length == 15.0

    def test_crown_base_height_stored_as_float(self):
        tree = self._tree(crown_base_height=np.float32(7.5))
        assert type(tree.crown_base_height) is float
        assert tree.crown_base_height == 7.5

    def test_crown_length_is_height_minus_crown_base_height(self):
        tree = self._tree(crown_base_height=4.2)
        assert tree.crown_length == tree.height - tree.crown_base_height

    @pytest.mark.parametrize("height, cr", [(12.0, 0.3), (20.1, 0.45), (30.0, 0.7)])
    def test_migration_reproduces_pre_018_crown_base_exactly(self, height, cr):
        # Pre-0.18 Tree: crown_length = height * crown_ratio and
        # crown_base_height = height - crown_length. height * (1 - cr) differs
        # from it in the last bit for these inputs.
        old = height - height * cr
        assert height * (1 - cr) != old
        tree = self._tree(height=height, crown_base_height=height - height * cr)
        assert tree.crown_base_height == old

    # ── crown_ratio validation ────────────────────────────────────────────

    @pytest.mark.parametrize("cr", [-0.1, 1.1, float("inf")])
    def test_invalid_crown_ratio_raises(self, cr):
        with pytest.raises(ValueError, match="crown_ratio"):
            self._tree(crown_base_height=7.5, crown_ratio=cr)

    @pytest.mark.parametrize("cr", [0.0, 0.5, 1.0])
    def test_valid_crown_ratio_accepted(self, cr):
        tree = self._tree(crown_base_height=7.5, crown_ratio=cr)
        assert tree.crown_ratio == cr

    def test_crown_ratio_defaults_to_none(self):
        tree = self._tree(crown_base_height=7.5)
        assert tree.crown_ratio is None

    def test_crown_ratio_none_accepted(self):
        tree = self._tree(crown_base_height=7.5, crown_ratio=None)
        assert tree.crown_ratio is None

    @pytest.mark.parametrize("cr", [float("nan"), np.nan, np.float32("nan"), pd.NA])
    def test_missing_crown_ratio_is_none(self, cr):
        tree = self._tree(crown_base_height=7.5, crown_ratio=cr)
        assert tree.crown_ratio is None

    # ── Crown base height estimated from crown_ratio ──────────────────────

    @pytest.mark.parametrize("pnw", [False, True])
    def test_estimate_matches_vectorized_exactly(self, pnw):
        rng = np.random.default_rng(0)
        n = 40
        spcd = rng.choice([202, 122, 747, 131, 316, 6156, 6051], n)
        dbh = rng.uniform(5.0, 80.0, n)
        height = rng.uniform(4.0, 45.0, n)
        cr = rng.uniform(0.0, 1.0, n)
        cr[:3] = [0.0, 1.0, 0.05]
        expected = estimate_crown_base_height(spcd, cr, dbh, height, pnw)
        got = [
            Tree(s, 1, d, h, crown_ratio=c, pacific_northwest=pnw).crown_base_height
            for s, d, h, c in zip(spcd, dbh, height, cr)
        ]
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize(
        "equation, species",
        [
            ("monleon_2004_eq1", [202, 15, 122, 101, 801, 64]),
            ("monleon_2004_eq2", [202, 263, 17, 21, 131, 805]),
            ("toney_reeves_2009", [122, 202, 65, 746, 93, 19]),
            ("randolph_2010", [131, 621, 61, 58, 68, 802]),
            ("westfall_2019", [12, 125, 129, 316, 202, 833]),
        ],
    )
    def test_crown_ratio_equation_matches_vectorized_exactly(self, equation, species):
        rng = np.random.default_rng(1)
        n = 30
        spcd = rng.choice(species, n)
        dbh = rng.uniform(5.0, 80.0, n)
        height = rng.uniform(4.0, 45.0, n)
        cr = rng.uniform(0.0, 1.0, n)
        cr[:4] = [0.0, 1.0, 0.95, 0.05]
        expected = estimate_crown_base_height(spcd, cr, dbh, height, equation=equation)
        assert np.isfinite(expected).all()
        got = [
            Tree(
                s, 1, d, h, crown_ratio=c, crown_ratio_equation=equation
            ).crown_base_height
            for s, d, h, c in zip(spcd, dbh, height, cr)
        ]
        np.testing.assert_array_equal(got, expected)

    def test_crown_ratio_equation_defaults_to_refit(self):
        tree = self._tree(crown_ratio=0.4)
        assert tree._crown_ratio_equation == "fastfuels_2026"
        assert (
            tree.crown_base_height
            == estimate_crown_base_height(
                122, 0.4, 25.0, 15.0, equation="fastfuels_2026"
            )[0]
        )

    def test_unknown_crown_ratio_equation_raises(self):
        with pytest.raises(ValueError, match="crown ratio equation"):
            self._tree(crown_base_height=6.0, crown_ratio=0.4, crown_ratio_equation="x")

    def test_species_outside_equation_raises(self):
        # Toney & Reeves 2009 has no loblolly pine model
        with pytest.raises(ValueError, match="toney_reeves_2009"):
            self._tree(
                species_code=131,
                crown_ratio=0.4,
                crown_ratio_equation="toney_reeves_2009",
            )

    def test_pacific_northwest_with_published_equation_raises(self):
        with pytest.raises(ValueError, match="pacific_northwest applies only"):
            self._tree(
                crown_ratio=0.4,
                pacific_northwest=True,
                crown_ratio_equation="monleon_2004_eq2",
            )

    def test_supplied_crown_base_height_takes_precedence(self):
        tree = self._tree(crown_base_height=6.0, crown_ratio=0.4)
        assert tree.crown_base_height == 6.0
        assert tree.crown_ratio == 0.4

    def test_pacific_northwest_ignored_when_crown_base_height_supplied(self):
        tree = self._tree(
            crown_base_height=6.0, crown_ratio=0.4, pacific_northwest=True
        )
        assert tree.crown_base_height == 6.0

    @pytest.mark.parametrize("cbh", [None, float("nan"), pd.NA])
    def test_missing_crown_base_height_is_estimated(self, cbh):
        tree = self._tree(crown_base_height=cbh, crown_ratio=0.4)
        expected = estimate_crown_base_height(122, 0.4, 25.0, 15.0)[0]
        assert tree.crown_base_height == expected
        assert type(tree.crown_base_height) is float

    def test_pacific_northwest_selects_regional_estimate(self):
        kwargs = {"species_code": 202, "diameter": 30.0, "crown_ratio": 0.4}
        national = self._tree(**kwargs).crown_base_height
        regional = self._tree(**kwargs, pacific_northwest=True).crown_base_height
        assert regional == estimate_crown_base_height(202, 0.4, 30.0, 15.0, True)[0]
        assert regional != national

    def test_unestimable_crown_base_height_raises(self):
        with pytest.raises(ValueError, match="could not be estimated"):
            self._tree(diameter=float("nan"), crown_ratio=0.4)

    def test_zero_crown_ratio_without_diameter_is_full_height(self):
        tree = self._tree(diameter=float("nan"), crown_ratio=0.0)
        assert tree.crown_base_height == 15.0

    # ── Geometry ignores crown_ratio ──────────────────────────────────────

    @pytest.mark.parametrize("profile_type", ["purves", "beta"])
    def test_crown_ratio_does_not_change_geometry(self, profile_type):
        trees = [
            self._tree(
                crown_base_height=6.0,
                crown_ratio=cr,
                crown_profile_model_type=profile_type,
            )
            for cr in (None, 0.0, 0.3, 1.0)
        ]
        z = np.linspace(0.0, 16.0, 33)
        for tree in trees[1:]:
            assert tree.crown_base_height == trees[0].crown_base_height
            assert tree.crown_length == trees[0].crown_length
            assert tree.max_crown_radius == trees[0].max_crown_radius
            np.testing.assert_array_equal(
                tree.get_crown_radius_at_height(z),
                trees[0].get_crown_radius_at_height(z),
            )

    # ── Row readers ───────────────────────────────────────────────────────

    @staticmethod
    def _frame(cr=(0.5, float("nan")), cbh=(7.5, 6.0)):
        import pandas as pd

        return pd.DataFrame(
            {
                "TREE_ID": [1, 2],
                "SPCD": [122, 202],
                "STATUSCD": [1, 1],
                "DIA": [25.0, 30.0],
                "HT": [15.0, 20.0],
                "CR": list(cr),
                "CBH": list(cbh),
                "X": [100.0, 200.0],
                "Y": [100.0, 200.0],
            }
        )

    def test_population_iteration_reads_cbh_and_maps_nan_cr_to_none(self):
        trees = list(TreePopulation(self._frame()))
        assert [t.crown_base_height for t in trees] == [7.5, 6.0]
        assert trees[0].crown_ratio == 0.5
        assert trees[1].crown_ratio is None
        assert [t.crown_length for t in trees] == [7.5, 14.0]

    def test_population_iteration_estimates_nan_cbh_from_cr(self):
        trees = list(TreePopulation(self._frame(cr=(0.5, 0.4), cbh=(7.5, np.nan))))
        assert trees[0].crown_base_height == 7.5
        expected = estimate_crown_base_height(202, 0.4, 30.0, 20.0)[0]
        assert trees[1].crown_base_height == expected

    @pytest.mark.parametrize("rows", ["iterrows", "itertuples"])
    def test_from_row_estimates_nan_cbh_from_cr(self, rows):
        df = self._frame(cr=(0.5, 0.4), cbh=(7.5, np.nan))
        if rows == "iterrows":
            trees = [Tree.from_row(row) for _, row in df.iterrows()]
        else:
            trees = [Tree.from_row(row) for row in df.itertuples()]
        expected = estimate_crown_base_height(202, 0.4, 30.0, 20.0)[0]
        assert trees[1].crown_base_height == expected

    def test_population_iteration_raises_on_nan_cbh(self):
        population = TreePopulation(self._frame(cbh=(7.5, float("nan"))))
        with pytest.raises(ValueError, match="crown_base_height"):
            list(population)

    @pytest.mark.parametrize("rows", ["iterrows", "itertuples"])
    def test_from_row_reads_cbh_and_maps_nan_cr_to_none(self, rows):
        df = self._frame()
        if rows == "iterrows":
            trees = [Tree.from_row(row) for _, row in df.iterrows()]
        else:
            trees = [Tree.from_row(row) for row in df.itertuples()]
        assert [t.crown_base_height for t in trees] == [7.5, 6.0]
        assert trees[0].crown_ratio == 0.5
        assert trees[1].crown_ratio is None

    def test_from_row_raises_on_nan_cbh(self):
        df = self._frame(cbh=(7.5, float("nan")))
        with pytest.raises(ValueError, match="crown_base_height"):
            [Tree.from_row(row) for _, row in df.iterrows()]

    @pytest.mark.parametrize("rows", ["iterrows", "itertuples"])
    def test_from_row_without_cbh_raises(self, rows):
        df = self._frame().drop(columns="CBH")
        row = next(df.iterrows())[1] if rows == "iterrows" else next(df.itertuples())
        with pytest.raises(AttributeError, match="CBH"):
            Tree.from_row(row)


class TestMaxCrownRadiusOverride:
    """Tests for the optional max_crown_radius parameter on Tree."""

    TREE_KWARGS = dict(
        species_code=122,
        status_code=1,
        diameter=25.0,
        height=15.0,
        crown_base_height=15.0 - 15.0 * 0.5,
    )

    # ── Backward compatibility ────────────────────────────────────────────

    def test_no_override_uses_allometric(self):
        """Without override, max_crown_radius comes from allometric model."""
        tree = Tree(**self.TREE_KWARGS)
        assert tree.max_crown_radius == tree.crown_profile_model.get_max_radius()
        assert tree._max_crown_radius_override is None
        assert tree._crown_radius_scale_factor == 1.0

    def test_none_override_same_as_no_override(self):
        """Explicit None behaves identically to omitting the parameter."""
        tree_default = Tree(**self.TREE_KWARGS)
        tree_none = Tree(**self.TREE_KWARGS, max_crown_radius=None)
        assert tree_default.max_crown_radius == tree_none.max_crown_radius

    # ── Override returns exact value ──────────────────────────────────────

    def test_override_returns_exact_value_purves(self):
        tree = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="purves", max_crown_radius=4.0
        )
        assert tree.max_crown_radius == 4.0

    def test_override_returns_exact_value_beta(self):
        tree = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="beta", max_crown_radius=4.0
        )
        assert tree.max_crown_radius == 4.0

    # ── Scale factor ──────────────────────────────────────────────────────

    def test_scale_factor_purves(self):
        tree = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="purves", max_crown_radius=4.0
        )
        allometric_max = tree.crown_profile_model.get_max_radius()
        assert tree._crown_radius_scale_factor == pytest.approx(4.0 / allometric_max)

    def test_scale_factor_beta(self):
        tree = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="beta", max_crown_radius=4.0
        )
        allometric_max = tree.crown_profile_model.get_max_radius()
        assert tree._crown_radius_scale_factor == pytest.approx(4.0 / allometric_max)

    # ── Shape preservation (Purves) ───────────────────────────────────────

    def test_shape_preserved_purves(self):
        """Normalized profile R(h)/max_R is identical with and without override."""
        t1 = Tree(**self.TREE_KWARGS, crown_profile_model_type="purves")
        t2 = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="purves", max_crown_radius=4.0
        )

        heights = np.linspace(t1.crown_base_height, t1.height, 50)
        for h in heights:
            r1 = t1.get_crown_radius_at_height(h)
            r2 = t2.get_crown_radius_at_height(h)
            if t1.max_crown_radius > 0 and t2.max_crown_radius > 0:
                assert r1 / t1.max_crown_radius == pytest.approx(
                    r2 / t2.max_crown_radius, abs=1e-10
                )

    # ── Shape preservation (Beta) ─────────────────────────────────────────

    def test_shape_preserved_beta(self):
        """Normalized profile R(h)/max_R is identical with and without override."""
        t1 = Tree(**self.TREE_KWARGS, crown_profile_model_type="beta")
        t2 = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="beta", max_crown_radius=4.0
        )

        heights = np.linspace(t1.crown_base_height, t1.height, 50)
        for h in heights:
            r1 = t1.get_crown_radius_at_height(h)
            r2 = t2.get_crown_radius_at_height(h)
            if t1.max_crown_radius > 0 and t2.max_crown_radius > 0:
                assert r1 / t1.max_crown_radius == pytest.approx(
                    r2 / t2.max_crown_radius, abs=1e-10
                )

    # ── Constant scale factor across all heights ──────────────────────────

    def test_constant_scale_factor_across_heights_purves(self):
        """r_scaled(h) / r_allometric(h) is constant for all h in crown."""
        t1 = Tree(**self.TREE_KWARGS, crown_profile_model_type="purves")
        t2 = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="purves", max_crown_radius=5.0
        )

        heights = np.linspace(t1.crown_base_height + 0.1, t1.height - 0.1, 30)
        ratios = []
        for h in heights:
            r1 = t1.get_crown_radius_at_height(h)
            r2 = t2.get_crown_radius_at_height(h)
            if r1 > 0:
                ratios.append(r2 / r1)
        assert len(ratios) > 0
        assert all(r == pytest.approx(ratios[0], abs=1e-10) for r in ratios)

    def test_constant_scale_factor_across_heights_beta(self):
        """r_scaled(h) / r_allometric(h) is constant for all h in crown."""
        t1 = Tree(**self.TREE_KWARGS, crown_profile_model_type="beta")
        t2 = Tree(
            **self.TREE_KWARGS, crown_profile_model_type="beta", max_crown_radius=5.0
        )

        heights = np.linspace(t1.crown_base_height + 0.1, t1.height - 0.1, 30)
        ratios = []
        for h in heights:
            r1 = t1.get_crown_radius_at_height(h)
            r2 = t2.get_crown_radius_at_height(h)
            if r1 > 0:
                ratios.append(r2 / r1)
        assert len(ratios) > 0
        assert all(r == pytest.approx(ratios[0], abs=1e-10) for r in ratios)

    # ── Zero outside crown ────────────────────────────────────────────────

    def test_zero_outside_crown_with_override(self):
        """Override doesn't affect regions outside the crown bounds."""
        tree = Tree(**self.TREE_KWARGS, max_crown_radius=10.0)
        assert tree.get_crown_radius_at_height(0.0) == 0.0
        assert tree.get_crown_radius_at_height(tree.crown_base_height - 1.0) == 0.0
        assert tree.get_crown_radius_at_height(tree.height + 1.0) == 0.0

    # ── from_row with MAX_CROWN_RADIUS ────────────────────────────────────

    def test_from_row_with_max_crown_radius(self):
        """from_row picks up MAX_CROWN_RADIUS column when present."""
        import pandas as pd

        row = pd.Series(
            {
                "SPCD": 122,
                "STATUSCD": 1,
                "DIA": 25.0,
                "HT": 15.0,
                "CR": 0.5,
                "CBH": 15.0 - 15.0 * 0.5,
                "X": 100.0,
                "Y": 200.0,
                "MAX_CROWN_RADIUS": 4.0,
            }
        )
        tree = Tree.from_row(row)
        assert tree.max_crown_radius == 4.0
        assert tree._max_crown_radius_override == 4.0

    def test_from_row_without_max_crown_radius(self):
        """from_row works normally when MAX_CROWN_RADIUS column is absent."""
        import pandas as pd

        row = pd.Series(
            {
                "SPCD": 122,
                "STATUSCD": 1,
                "DIA": 25.0,
                "HT": 15.0,
                "CR": 0.5,
                "CBH": 15.0 - 15.0 * 0.5,
                "X": 100.0,
                "Y": 200.0,
            }
        )
        tree = Tree.from_row(row)
        assert tree._max_crown_radius_override is None
        assert tree.max_crown_radius == tree.crown_profile_model.get_max_radius()

    def test_from_row_with_nan_max_crown_radius(self):
        """from_row treats NaN MAX_CROWN_RADIUS as absent."""
        import pandas as pd

        row = pd.Series(
            {
                "SPCD": 122,
                "STATUSCD": 1,
                "DIA": 25.0,
                "HT": 15.0,
                "CR": 0.5,
                "CBH": 15.0 - 15.0 * 0.5,
                "X": 100.0,
                "Y": 200.0,
                "MAX_CROWN_RADIUS": float("nan"),
            }
        )
        tree = Tree.from_row(row)
        assert tree._max_crown_radius_override is None

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_override_smaller_than_allometric(self):
        """Override works when smaller than allometric estimate (scale < 1)."""
        tree = Tree(**self.TREE_KWARGS, max_crown_radius=0.5)
        assert tree.max_crown_radius == 0.5
        assert tree._crown_radius_scale_factor < 1.0
        # All radii should be smaller than allometric
        tree_allom = Tree(**self.TREE_KWARGS)
        h = (tree.crown_base_height + tree.height) / 2
        assert tree.get_crown_radius_at_height(
            h
        ) < tree_allom.get_crown_radius_at_height(h)

    def test_override_equal_to_allometric(self):
        """Override matching allometric gives scale factor of 1.0."""
        tree_allom = Tree(**self.TREE_KWARGS)
        tree_override = Tree(
            **self.TREE_KWARGS, max_crown_radius=tree_allom.max_crown_radius
        )
        assert tree_override._crown_radius_scale_factor == pytest.approx(1.0)

    @pytest.mark.parametrize("profile_type", ["purves", "beta"])
    def test_override_with_multiple_species(self, profile_type):
        """Override works correctly across different species."""
        for spcd in [122, 202, 316, 747]:
            tree = Tree(
                species_code=spcd,
                status_code=1,
                diameter=20.0,
                height=12.0,
                crown_base_height=12.0 - 12.0 * 0.4,
                crown_profile_model_type=profile_type,
                max_crown_radius=3.0,
            )
            assert tree.max_crown_radius == 3.0
            mid_h = (tree.crown_base_height + tree.height) / 2
            assert tree.get_crown_radius_at_height(mid_h) > 0
            assert tree.get_crown_radius_at_height(mid_h) <= 3.0


class TestFromRowDataFrame:
    """Tests that from_row works correctly when iterating over a DataFrame,
    which is how treevox actually uses it."""

    def test_dataframe_with_all_optional_columns(self):
        """Iterate a DataFrame with both CROWN_FUEL_LOAD and MAX_CROWN_RADIUS."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "SPCD": [122, 202, 122],
                "STATUSCD": [1, 1, 1],
                "DIA": [25.0, 30.0, 15.0],
                "HT": [15.0, 20.0, 10.0],
                "CR": [0.5, 0.4, 0.6],
                "X": [100.0, 200.0, 300.0],
                "Y": [100.0, 200.0, 300.0],
                "CROWN_FUEL_LOAD": [14.0, 25.0, 5.0],
                "MAX_CROWN_RADIUS": [3.0, 4.5, 2.0],
            }
        )
        df["CBH"] = df["HT"] - df["HT"] * df["CR"]

        trees = [Tree.from_row(row) for _, row in df.iterrows()]

        assert len(trees) == 3
        for i, tree in enumerate(trees):
            assert tree.foliage_biomass == df["CROWN_FUEL_LOAD"].iloc[i]
            assert tree.max_crown_radius == df["MAX_CROWN_RADIUS"].iloc[i]
            assert tree.species_code == df["SPCD"].iloc[i]
            assert tree.height == df["HT"].iloc[i]

    def test_dataframe_without_optional_columns(self):
        """Iterate a DataFrame with only required columns (backward compat)."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "SPCD": [122, 202],
                "STATUSCD": [1, 1],
                "DIA": [25.0, 30.0],
                "HT": [15.0, 20.0],
                "CR": [0.5, 0.4],
                "X": [100.0, 200.0],
                "Y": [100.0, 200.0],
            }
        )
        df["CBH"] = df["HT"] - df["HT"] * df["CR"]

        trees = [Tree.from_row(row) for _, row in df.iterrows()]

        for tree in trees:
            assert tree._max_crown_radius_override is None
            assert tree._crown_fuel_load_override is None
            # Allometric values should be used
            assert (
                tree.foliage_biomass
                == tree.biomass_allometry_model.estimate_foliage_biomass()
            )
            assert tree.max_crown_radius == tree.crown_profile_model.get_max_radius()

    def test_dataframe_with_nan_optional_values(self):
        """NaN in optional columns should be treated as absent."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "SPCD": [122, 122],
                "STATUSCD": [1, 1],
                "DIA": [25.0, 25.0],
                "HT": [15.0, 15.0],
                "CR": [0.5, 0.5],
                "X": [100.0, 200.0],
                "Y": [100.0, 200.0],
                "CROWN_FUEL_LOAD": [14.0, float("nan")],
                "MAX_CROWN_RADIUS": [float("nan"), 3.0],
            }
        )
        df["CBH"] = df["HT"] - df["HT"] * df["CR"]

        trees = [Tree.from_row(row) for _, row in df.iterrows()]

        # Row 0: has CROWN_FUEL_LOAD but not MAX_CROWN_RADIUS
        assert trees[0].foliage_biomass == 14.0
        assert trees[0]._max_crown_radius_override is None
        assert (
            trees[0].max_crown_radius == trees[0].crown_profile_model.get_max_radius()
        )

        # Row 1: has MAX_CROWN_RADIUS but not CROWN_FUEL_LOAD
        assert trees[1]._crown_fuel_load_override is None
        assert (
            trees[1].foliage_biomass
            == trees[1].biomass_allometry_model.estimate_foliage_biomass()
        )
        assert trees[1].max_crown_radius == 3.0

    def test_dataframe_itertuples(self):
        """from_row also works with itertuples (namedtuple-like rows)."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "SPCD": [122],
                "STATUSCD": [1],
                "DIA": [25.0],
                "HT": [15.0],
                "CR": [0.5],
                "X": [100.0],
                "Y": [100.0],
                "CROWN_FUEL_LOAD": [10.0],
                "MAX_CROWN_RADIUS": [3.5],
            }
        )
        df["CBH"] = df["HT"] - df["HT"] * df["CR"]

        for row in df.itertuples():
            tree = Tree.from_row(row)
            assert tree.foliage_biomass == 10.0
            assert tree.max_crown_radius == 3.5


class TestCrownBaseHeightSchema:
    """CBH column and the frame-level CBH <= HT check on the tree schemas."""

    @staticmethod
    def _frame(cbh, ht=(15.0, 15.0)):
        import pandas as pd

        return pd.DataFrame(
            {
                "TREE_ID": [1, 2],
                "PLOT_ID": [1, 1],
                "SPCD": [122, 202],
                "STATUSCD": [1, 1],
                "DIA": [25.0, 30.0],
                "HT": list(ht),
                "CR": [0.5, 0.4],
                "CBH": list(cbh),
                "TPA": [0.01, 0.01],
            }
        )

    @pytest.mark.parametrize("cls", [TreeSample, TreePopulation])
    def test_rejects_cbh_above_ht(self, cls):
        with pytest.raises(SchemaError, match="CBH must be less than or equal to HT"):
            cls(self._frame(cbh=[7.5, 15.1]))

    @pytest.mark.parametrize("cls", [TreeSample, TreePopulation])
    @pytest.mark.parametrize(
        "cbh, ht",
        [
            ((15.0, 15.0), (15.0, 15.0)),  # no live crown: CBH == HT
            ((0.0, 0.0), (15.0, 15.0)),  # crown reaches the ground
            ((float("nan"), 7.5), (15.0, 15.0)),  # CBH unknown
            ((20.0, 7.5), (float("nan"), 15.0)),  # HT unknown
        ],
    )
    def test_accepts_valid_cbh(self, cls, cbh, ht):
        frame = self._frame(cbh=cbh, ht=ht)
        result = cls(frame)
        np.testing.assert_array_equal(result.data["CBH"].values, frame["CBH"].values)

    @pytest.mark.parametrize("cls", [TreeSample, TreePopulation])
    def test_rejects_negative_cbh(self, cls):
        with pytest.raises(SchemaError, match="greater_than_or_equal_to"):
            cls(self._frame(cbh=[-0.1, 7.5]))

    def test_tree_sample_requires_cbh(self):
        with pytest.raises(SchemaError, match="column 'CBH' not in dataframe"):
            TreeSample(self._frame(cbh=[7.5, 7.5]).drop(columns="CBH"))

    def test_tree_population_adds_missing_cbh_as_nan(self):
        population = TreePopulation(self._frame(cbh=[7.5, 7.5]).drop(columns="CBH"))
        assert "CBH" in population.data.columns
        assert population.data["CBH"].dtype == np.float64
        assert population.data["CBH"].isna().all()

    @pytest.mark.parametrize("lazy", [False, True])
    def test_expand_to_roi_carries_cbh(self, lazy):
        import geopandas as gpd
        import pandas as pd

        roi = gpd.read_file(TEST_DATA_PATH / "polygon.geojson")
        roi = roi.to_crs(roi.estimate_utm_crs())
        plots = gpd.read_file(TEST_DATA_PATH / "treemap_2016_plots_data.geojson")
        plots = plots.to_crs(plots.estimate_utm_crs())
        trees = pd.read_parquet(
            TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
        )
        # A distinct CBH per sample tree, so each output row can be traced back.
        fraction = np.linspace(0.05, 0.95, len(trees))
        trees["CBH"] = trees["HT"] * fraction
        assert trees["CBH"].dropna().is_unique

        result = TreeSample(trees).expand_to_roi(
            "inhomogeneous_poisson", roi, lazy=lazy, plots=plots, seed=42
        )
        expanded = result.compute() if lazy else result.data

        assert len(expanded) > 0
        assert "CBH" in expanded.columns
        assert expanded["CBH"].notna().any()
        expected = expanded["TREE_ID"].map(trees.set_index("TREE_ID")["CBH"])
        np.testing.assert_array_equal(expanded["CBH"].values, expected.values)


class TestGeometricCrownProfiles:
    """Tests for the geometric crown profile types on Tree (issue #76)."""

    TREE_KWARGS = dict(
        species_code=122,
        status_code=1,
        diameter=25.0,
        height=15.0,
        crown_base_height=15.0 - 15.0 * 0.5,
    )
    # crown_base_height = 15 - 15*0.5 = 7.5; height = 15; midpoint = 11.25
    HB, HT, MID = 7.5, 15.0, 11.25

    GEOMETRIC = [
        ("cone", ConeCrownProfile),
        ("cylinder", CylinderCrownProfile),
        ("paraboloid", ParaboloidCrownProfile),
        ("ellipsoid", EllipsoidCrownProfile),
    ]

    @pytest.mark.parametrize("profile_type,cls", GEOMETRIC)
    def test_factory_builds_expected_model(self, profile_type, cls):
        tree = Tree(
            **self.TREE_KWARGS,
            crown_profile_model_type=profile_type,
            max_crown_radius=3.0,
        )
        assert isinstance(tree.crown_profile_model, cls)

    @pytest.mark.parametrize("profile_type,_cls", GEOMETRIC)
    def test_requires_max_crown_radius(self, profile_type, _cls):
        with pytest.raises(ValueError):
            Tree(**self.TREE_KWARGS, crown_profile_model_type=profile_type)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            Tree(**self.TREE_KWARGS, crown_profile_model_type="pyramid")

    @pytest.mark.parametrize("profile_type,_cls", GEOMETRIC)
    def test_max_crown_radius_used_directly(self, profile_type, _cls):
        tree = Tree(
            **self.TREE_KWARGS,
            crown_profile_model_type=profile_type,
            max_crown_radius=3.0,
        )
        assert tree.max_crown_radius == pytest.approx(3.0)
        assert tree._crown_radius_scale_factor == pytest.approx(1.0)

    @pytest.mark.parametrize("profile_type,_cls", GEOMETRIC)
    def test_get_crown_radius_matches_profile(self, profile_type, _cls):
        tree = Tree(
            **self.TREE_KWARGS,
            crown_profile_model_type=profile_type,
            max_crown_radius=3.0,
        )
        z = np.linspace(0.0, 16.0, 40)
        assert np.allclose(
            tree.get_crown_radius_at_height(z),
            tree.crown_profile_model.get_radius_at_height(z),
        )

    def test_max_crown_diameter_height_defaults_to_midpoint(self):
        tree = Tree(
            **self.TREE_KWARGS,
            crown_profile_model_type="paraboloid",
            max_crown_radius=3.0,
        )
        assert tree.get_crown_radius_at_height(self.MID) == pytest.approx(3.0)

    def test_max_crown_diameter_height_moves_peak(self):
        hd = 9.0
        tree = Tree(
            **self.TREE_KWARGS,
            crown_profile_model_type="ellipsoid",
            max_crown_radius=3.0,
            max_crown_diameter_height=hd,
        )
        assert tree.get_crown_radius_at_height(hd) == pytest.approx(3.0)
        assert tree.get_crown_radius_at_height(self.MID) < 3.0

    def test_lanl_reproduction_recipe(self):
        # Decoupled, opt-in: derive Hd from the beta profile mode and pass it in.
        cbh = self.HT - self.HT * 0.5
        cl = self.HT * 0.5
        beta = BetaCrownProfile(
            species_code=122, crown_base_height=cbh, crown_length=cl
        )
        hd = beta.get_max_radius_height()
        assert cbh <= hd <= cbh + cl
        tree = Tree(
            **self.TREE_KWARGS,
            crown_profile_model_type="paraboloid",
            max_crown_radius=3.0,
            max_crown_diameter_height=hd,
        )
        assert tree.get_crown_radius_at_height(hd) == pytest.approx(3.0)
