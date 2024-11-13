# Core imports
from pathlib import Path
from fastfuels_core.crown_profile_models.purves import  PurvesCrownProfile

# External imports
import pytest
import numpy as np


@pytest.fixture
def model():
    # Suppose that I have a tree with a height of 10m, dbh of 15cm, and crown ratio of 0.4
    return PurvesCrownProfile(
        species_code = 122, height = 10, dbh = 15, crown_ratio = 0.4
        )

#generic way to create a custom model however we like for potential expansion of the tests
@pytest.fixture
def custom_model(scope="module"):
    def _model(sc, ht, dbh, cr):
        return PurvesCrownProfile(
            sc, ht, dbh, cr
        )
    return _model



class TestPurvesGetRadiusAtHeight:

    # test where radius should be 0
    @pytest.mark.parametrize("test_input", [5, 10, -1, 11])
    def test_grah_rad_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) == 0

    # test where radius should not be 0
    @pytest.mark.parametrize("test_input", [7, 8, 9])
    def test_grah_rad_not_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) > 0

class TestPurvesGetMaxRadius:

    # test get_max_radius against a few points on tree to ensure it is the max radius
    # max radius of this default model occurs at 6
    @pytest.mark.parametrize("test_input", [1,2,3,4,5,7,8,9,10])
    def test_get_max_radius(self, model, test_input):
        assert model.get_max_radius() > model.get_radius_at_height(test_input)
