# Core imports
from pathlib import Path
from fastfuels_core.crown_profile_models.beta import  BetaCrownProfile

# External imports
import pytest
import numpy as np


@pytest.fixture
def model():
    # Suppose that I have a tree with a crown base height of 10m and crown length of 10m
    return BetaCrownProfile(
        species_code = 122, crown_base_height = 10, crown_length = 10
        )

#generic way to create a custom model however we like for potential expansion of the tests
@pytest.fixture
def custom_model(scope="module"):
    def _model(sc, cbht, cl):
        return BetaCrownProfile(
            sc, cbht, cl
        )
    return _model



class TestBetaGetRadiusAtHeight:

    # test where radius should be 0
    @pytest.mark.parametrize("test_input", [5, 9, 10, 20, -1, 21])
    def test_grah_rad_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) == 0

    # test where radius should not be 0
    @pytest.mark.parametrize("test_input", [ 11, 12, 13, 14, 15, 16, 17, 18, 19])
    def test_grah_rad_not_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) > 0

class TestBetaGetMaxRadius:

    # test get_max_radius against a few points on tree to ensure it is the max radius
    # max radius of this default model occurs at 6
    @pytest.mark.parametrize("test_input", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    def test_get_max_radius(self, model, test_input):
        assert model.get_max_radius() > model.get_radius_at_height(test_input)
