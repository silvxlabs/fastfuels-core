# Core imports
from pathlib import Path

# Internal imports
from fastfuels_core.trees import (
    Tree,
    JenkinsBiomassEquations,
    NSVBEquations,
    REF_SPECIES,
)
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile

# External imports
import pytest
import numpy as np

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


def test_tree():
    tree = Tree(species_code=122, status_code=1, diameter=5, height=20, crown_ratio=0.5)
    assert isinstance(tree.biomass_allometry_model, NSVBEquations)
