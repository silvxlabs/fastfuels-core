# Core imports
from pathlib import Path

from dask.dataframe.partitionquantiles import tree_width
import numpy as np
from tests.utils import make_tree_list
from fastfuels_core.crown_profile_models.purves import  PurvesCrownProfile

# External imports
import pytest
import numpy as np

#generates a list of all the trees in the species code index
#the returned list is formatted in the particular way of PurvesCrownProfile models
@pytest.fixture()
def purves_tree_list():
    trees = make_tree_list("purves")
    tree_list =[]
    for t in trees:
        tree = PurvesCrownProfile(
            species_code=t.species_code,
            height=t.height,
            dbh=t.diameter,
            crown_ratio=t.crown_ratio)
        tree_list.append(tree)

    return tree_list

@pytest.fixture()
def vector_input():
    return np.array([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,61,-1])

#a test sample model we could use
@pytest.fixture
def _model():
    # Suppose that I have a tree with a height of 10m, dbh of 15cm, and crown ratio of 0.4
    return PurvesCrownProfile(
        species_code = 122, height = 10, dbh = 15, crown_ratio = 0.4
        )


# generic way to create a custom model however we like
@pytest.fixture
def custom_model(scope="module"):
    def _model(sc, ht, dbh, cr):
        return PurvesCrownProfile(
            sc, ht, dbh, cr
        )
    return _model

#parameters that test edge cases as well as cases within tree
@pytest.mark.parametrize("test_input", [0,61,-1,1,5,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,55,60])
class TestPurvesCrownProfile:

    #tests all trees in the species code list
    def test_all_trees(self, purves_tree_list, test_input):
        vtest_all = np.vectorize(self.test_tree)
        vtest_all(purves_tree_list, test_input)

    #tests individual trees and determines which test to run based on tree height and crown ratio
    def test_tree(self, tree, test_input):
        if test_input > tree.height:                                        #above tree
            self.test_get_radius_at_height_rad_0(tree, test_input)
        elif test_input < (tree.height - tree.height * tree.crown_ratio):   #below crown
            self.test_get_radius_at_height_rad_0(tree, test_input)
        else:
            self.test_get_radius_at_height_rad_not_0(tree, test_input)
            self.test_get_max_radius(tree, test_input)


    def test_radius_functions(self, purves_tree_list, test_input):

        for i, t in enumerate(purves_tree_list):
            #test scalar values
            assert t.get_max_radius() <= t._get_purves_max_crown_radius(str(t.species_code), t.dbh)


    def test_vector_scalar_returns(self, purves_tree_list, test_input, vector_input):
        """
        This test proves that when the crown_profile model objects are passed vectors, they return vectors
        vice versa for scalar

        """
        for tree in purves_tree_list:
            scalar_return = tree.get_radius_at_height(test_input)                        #test when we pass scalar we get scalar
            self.test_result_and_expected(type(scalar_return), type(0.0), test_input)
            print(type(scalar_return))

            vector_return = tree.get_radius_at_height(vector_input)                      #test when we pass vector we get vector
            self.test_result_and_expected(type(vector_return), type(np.array(0)), test_input)
            print(type(vector_return))

    #generic definition to make sure the result equals expected
    def test_result_and_expected(self, result, expected, test_input):
        assert result == expected



    # test where radius should be 0
    def test_get_radius_at_height_rad_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) == 0

    # test where radius should not be 0
    def test_get_radius_at_height_rad_not_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) > 0

    # test that max radius is greater than or equal to any given point on tree
    def test_get_max_radius(self, model, test_input):
        assert model.get_max_radius() >= model.get_radius_at_height(test_input)

class TestGetMaxRadius:
    def test_individual_tree(self):
        purves_model = PurvesCrownProfile(122, 25, 10, 0.7)
        max_crown_radius = purves_model.get_max_radius()
        assert isinstance(max_crown_radius, float)

    # TODO: Make me pass
    def test_tree_vector(self):
        import pandas as pd
        species_vector = np.array([122, 122, 124, 125])
        dbh_vector = np.array([25, 50, 75, 45])
        height_vector = np.array([10, 15, 20, 25])
        crown_ratio_vector = np.array([0.7, 0.6, 0.5, 0.4])
        tree_population_dict = {
            "species_code": species_vector,
            "dbh": dbh_vector,
            "height": height_vector,
            "crown_ratio": crown_ratio_vector
        }
        tree_population_df = pd.DataFrame(tree_population_dict)
        purves_model = PurvesCrownProfile(species_vector, dbh_vector, height_vector, crown_ratio_vector)
        max_crown_radius = purves_model.get_max_radius()
        assert isinstance(max_crown_radius, np.ndarray)
        assert len(max_crown_radius) == 4
