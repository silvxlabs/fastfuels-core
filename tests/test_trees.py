# Internal imports
import numpy as np

from fastfuels_core.trees import Tree, BetaCrownProfile

# External imports
from pathlib import Path

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"


class TestBetaCrownProfileModel:

    def test_get_radius_at_height(self):
        """
        Tests that the radius at a given height is calculated correctly.
        """
        model = BetaCrownProfile(species_group=4, crown_base_height=10, crown_length=10)

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
        model = BetaCrownProfile(species_group=4, crown_base_height=10, crown_length=10)
        assert model._get_normalized_height(0) < 0
        assert model._get_normalized_height(5) < 0
        assert model._get_normalized_height(10) == 0
        assert model._get_normalized_height(15) == 0.5
        assert model._get_normalized_height(20) == 1
        assert model._get_normalized_height(25) > 1

        # Suppose that I have a tree with a height of 100m, a crown base height
        # of 20m, and a crown length of 80m
        model = BetaCrownProfile(species_group=4, crown_base_height=20, crown_length=80)
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
        model = BetaCrownProfile(species_group=4, crown_length=10, crown_base_height=10)
        normalized_heights = model._get_normalized_height(heights)
        assert len(normalized_heights) == 100
        assert normalized_heights[0] == 0
        assert normalized_heights[-1] == 1.0
        for i in range(1, 100):
            assert normalized_heights[i] > normalized_heights[i - 1]

    def test_get_radius_at_normalized_height(self):
        """
        Tests that the radius at a given normalized height is calculated correctly.
        """
        model = BetaCrownProfile(species_group=4, crown_base_height=10, crown_length=10)

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
        radii = model._get_radius_at_normalized_height(z_values)
        assert len(radii) == len(z_values)
        assert all(radii > 0)

        # Test with a numpy array of floats, some within and some outside the range [0, 1]
        z_values = np.array([-1, 0, 0.5, 1, 2])
        radii = model._get_radius_at_normalized_height(z_values)
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
        model = BetaCrownProfile(species_group=4, crown_base_height=10, crown_length=10)
        max_crown_radius = model.get_max_radius()
        assert max_crown_radius > 0
        heights = np.linspace(10, 20, 1000)
        radii = model.get_radius_at_height(heights)
        assert np.all(max_crown_radius >= radii)

        # Test with species group 1, crown base height 5, and crown length 15
        model = BetaCrownProfile(species_group=1, crown_base_height=5, crown_length=15)
        max_crown_radius = model.get_max_radius()
        assert max_crown_radius > 0
        heights = np.linspace(5, 20, 1000)
        radii = model.get_radius_at_height(heights)
        assert np.all(max_crown_radius >= radii)

        # Test with species group 2, crown base height 20, and crown length 5
        model = BetaCrownProfile(species_group=2, crown_base_height=20, crown_length=5)
        max_crown_radius = model.get_max_radius()
        assert max_crown_radius > 0
        heights = np.linspace(20, 25, 1000)
        radii = model.get_radius_at_height(heights)
        assert np.all(max_crown_radius >= radii)


def test_tree():
    tree = Tree(species_code=122, status_code=1, diameter=5, height=20, crown_ratio=0.5)
