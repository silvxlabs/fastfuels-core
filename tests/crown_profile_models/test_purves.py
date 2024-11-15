# Core imports
from pathlib import Path

from dask.dataframe.partitionquantiles import tree_width

from tests.utils import make_tree_list
from fastfuels_core.crown_profile_models.purves import  PurvesCrownProfile

# External imports
import pytest
import numpy as np

#generates a list of all the trees in the species code index
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



@pytest.fixture
def model():
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


@pytest.mark.parametrize("test_input", [0,61,-1,5, 10, -1, 11])
class TestPurvesCrownProfile:

    def test_all_trees(self, purves_tree_list, test_input):
        for tree in purves_tree_list:
            self.test_tree(tree, test_input)

    def test_tree(self, tree, test_input):
        if test_input > tree.height:                        #above tree
            self.test_grah_rad_0(tree, test_input)
        elif test_input < (tree.height - tree.height * tree.crown_ratio):   #below crown
            self.test_grah_rad_0(tree, test_input)
        else:
            self.test_grah_rad_not_0(tree, test_input)
            self.test_get_max_radius(tree, test_input)


    # test where radius should be 0
    def test_grah_rad_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) == 0

    # test where radius should not be 0
    def test_grah_rad_not_0(self, model, test_input):
        assert model.get_radius_at_height(test_input) > 0

    # test that max radius is greater than or equal to any given point on tree
    def test_get_max_radius(self, model, test_input):
        assert model.get_max_radius() >= model.get_radius_at_height(test_input)

