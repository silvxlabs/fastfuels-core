# Core imports
from pathlib import Path

from dask.dataframe.partitionquantiles import tree_width

from tests.utils import make_tree_list
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

#generates a list of all the trees in the species code index
#the returned list is formatted in the particular way of PurvesCrownProfile models
@pytest.fixture()
def beta_tree_list():
    trees = make_tree_list("beta")
    tree_list =[]
    for t in trees:
        tree = BetaCrownProfile(
            species_code=t.species_code,
            crown_base_height=t.crown_base_height,
            crown_length=t.crown_length)
        tree_list.append(tree)
    return tree_list


#parameters that test edge cases as well as cases within tree
@pytest.mark.parametrize("test_input", [0,61,-1,1,5,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,55,60])
class TestBetaCrownProfile:

    #tests all trees in the species code list
    def test_all_trees(self, beta_tree_list, test_input):
        vtest_all = np.vectorize(self.test_tree)
        vtest_all(beta_tree_list, test_input)

    #tests individual trees and determines which test to run based on tree height and crown ratio
    def test_tree(self, tree, test_input):
        height = tree.crown_base_height + tree.crown_length
        #print(tree) #for debugging, ensuring we are testing EVERY tree
        if test_input > height:                                         #above tree
            self.test_get_radius_at_height_rad_0(tree, test_input)
        elif test_input < tree.crown_base_height:                        #below crown
            self.test_get_radius_at_height_rad_0(tree, test_input)
        else:
            self.test_get_radius_at_height_rad_not_0(tree, test_input)
            self.test_get_max_radius(tree, test_input)


    # test where radius should be 0
    def test_get_radius_at_height_rad_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) == 0

    # test where radius should not be 0
    def test_get_radius_at_height_rad_not_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) > 0

    # test that max radius is greater than or equal to any given point on tree
    def test_get_max_radius(self, model, test_input):
        assert model.get_max_radius() >= model.get_radius_at_height(test_input)




