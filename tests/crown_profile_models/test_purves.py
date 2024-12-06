# Core imports
from pathlib import Path

from dask.dataframe.partitionquantiles import tree_width
import numpy as np
from tests.utils import make_tree_list
from tests.utils import  make_random_tree
from tests.utils import  LIST_SPCDS
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
def vector_test_input():
    return np.array([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,61,-1])

@pytest.fixture()
def scalar_test_input():
    input_pool = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,61.0,-1.0])
    return input_pool[np.random.randint(input_pool.size)]

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


    def test_vector_scalar_returns(self, purves_tree_list, test_input, vector_test_input):
        """
        This test proves that when the crown_profile model objects are passed vectors, they return vectors
        vice versa for scalar

        """
        for tree in purves_tree_list:
            scalar_return = tree.get_radius_at_height(test_input)                        #test when we pass scalar we get scalar
            assert isinstance(scalar_return, float)
            print(type(scalar_return))

            vector_return = tree.get_radius_at_height(vector_test_input)                      #test when we pass vector we get vector
            assert isinstance(vector_return, np.ndarray)
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


vector_length = 100_000

@pytest.fixture(scope="module")
def list_scpds():
    return LIST_SPCDS

@pytest.fixture(scope="module")
def tree_list():
    return [make_random_tree() for _ in range(vector_length)]


@pytest.fixture(scope="module")
def dbh_vector(tree_list):
    return np.array([t.diameter for t in tree_list])


@pytest.fixture(scope="module")
def height_vector(tree_list):
    return np.array([t.height for t in tree_list])


@pytest.fixture(scope="module")
def crown_ratio_vector(tree_list):
    return np.array([t.crown_ratio for t in tree_list])


@pytest.fixture(scope="module")
def spcd_vector(tree_list):
    return np.array([t.species_code for t in tree_list])


class TestGetMaxRadius:

    @pytest.mark.parametrize("spcd", LIST_SPCDS)
    def test_individual_tree(self, spcd):
        """
        Tests the individual tree's crown profile using the species code `spcd`. The
        method generates a random tree based on the species code and evaluates the
        maximum crown radius using the Purves crown profile model. It verifies that
        the calculated maximum crown radius meets expected criteria:
        - The value should be a float (scalar)
        - The value should be greater than 0
        - The value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        """
        tree = make_random_tree(species_code=spcd)
        purves_model = PurvesCrownProfile(
            species_code=tree.species_code,
            dbh=tree.diameter,
            height=tree.height,
            crown_ratio=tree.crown_ratio,
        )
        max_crown_radius = purves_model.get_max_radius()

        # Value should be a float (scalar)
        assert isinstance(max_crown_radius, float)

        # Value should be greater than 0
        assert max_crown_radius > 0.0

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        assert max_crown_radius <= purves_model._get_purves_max_crown_radius(
            tree.species_code, tree.diameter
        )

    @pytest.mark.parametrize("library", ["numpy", "pandas"])
    def test_tree_vector_numpy(
        self,
        library,
        tree_list,
        dbh_vector,
        height_vector,
        crown_ratio_vector,
        spcd_vector,
    ):
        """
        Tests the tree vector functionality using the specified library,
        either "numpy" or "pandas". The test creates a `PurvesCrownProfile`
        model based on the provided tree attributes and checks the resulting
        maximum crown radius. The test ensures that the maximum crown radius
        is calculated correctly and conforms to expected constraints.
        """
        if library == "numpy":
            purves_model = PurvesCrownProfile(
                spcd_vector, dbh_vector, height_vector, crown_ratio_vector
            )
        elif library == "pandas":
            import pandas as pd

            tree_population_dict = {
                "species_code": spcd_vector,
                "dbh": dbh_vector,
                "height": height_vector,
                "crown_ratio": crown_ratio_vector,
            }
            tree_population_df = pd.DataFrame(tree_population_dict)
            purves_model = PurvesCrownProfile(
                tree_population_df["species_code"],
                tree_population_df["dbh"],
                tree_population_df["height"],
                tree_population_df["crown_ratio"],
            )
        else:
            raise ValueError(f"Unknown library: {library}")

        max_crown_radius = purves_model.get_max_radius()

        assert isinstance(max_crown_radius, np.ndarray)
        assert len(max_crown_radius) == vector_length
        assert np.all(max_crown_radius > 0.0)
        assert np.all(
            max_crown_radius
            <= purves_model._get_purves_max_crown_radius(spcd_vector, dbh_vector)
        )
class TestGetRadiusAtHeight:

    @pytest.mark.parametrize("spcd", LIST_SPCDS)
    def test_individual_tree(self, spcd, scalar_test_input, vector_test_input):
        """
        Tests the individual tree's crown profile using the species code `spcd`. The
        method generates a random tree based on the species code and evaluates the
        radius at random heights using the Purves crown profile model. It verifies that
        the calculated radius at height meets expected criteria:
        - The value should be a float (scalar) with scalar input
        - The value should be NDArray (vector) with vector input
        - The value should be greater than 0
        - The value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        """
        tree = make_random_tree(species_code=spcd)
        purves_model = PurvesCrownProfile(
            species_code=tree.species_code,
            dbh=tree.diameter,
            height=tree.height,
            crown_ratio=tree.crown_ratio,
        )

        # Test scalar input first
        scalar_crown_radius = purves_model.get_radius_at_height(scalar_test_input)

        # Value should be a float (scalar)
        assert isinstance(scalar_crown_radius, float)

        # Value should be greater than or equal to 0 (height could be larger or smaller than crown height)
        assert scalar_crown_radius >= 0.0

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        assert scalar_crown_radius <= purves_model._get_purves_max_crown_radius(
            tree.species_code, tree.diameter
        )

        # Now test vector input
        vector_crown_radius = purves_model.get_radius_at_height(vector_test_input)

        # Value should be NDArray (vector)
        assert isinstance(vector_crown_radius, np.ndarray)

        # Value should be greater than or equal to 0 (height could be larger or smaller than crown height)
        assert np.all(vector_crown_radius >= 0.0)

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        assert np.all(vector_crown_radius <= purves_model._get_purves_max_crown_radius(
            tree.species_code, tree.diameter
        ))

    @pytest.mark.parametrize("library", ["numpy", "pandas"])
    def test_tree_vector_numpy(
        self,
        library,
        tree_list,
        dbh_vector,
        height_vector,
        crown_ratio_vector,
        spcd_vector,
        scalar_test_input,
        vector_test_input
    ):
        """
        Tests the tree vector functionality using the specified library,
        either "numpy" or "pandas". The test creates a `PurvesCrownProfile`
        model based on the provided tree attributes and checks the resulting
        maximum crown radius. The test ensures that the maximum crown radius
        is calculated correctly and conforms to expected constraints.
        """
        if library == "numpy":
            purves_model = PurvesCrownProfile(
                spcd_vector, dbh_vector, height_vector, crown_ratio_vector
            )
        elif library == "pandas":
            import pandas as pd

            tree_population_dict = {
                "species_code": spcd_vector,
                "dbh": dbh_vector,
                "height": height_vector,
                "crown_ratio": crown_ratio_vector,
            }
            tree_population_df = pd.DataFrame(tree_population_dict)
            purves_model = PurvesCrownProfile(
                tree_population_df["species_code"],
                tree_population_df["dbh"],
                tree_population_df["height"],
                tree_population_df["crown_ratio"],
            )
        else:
            raise ValueError(f"Unknown library: {library}")

        # Test scalar test input
        scalar_crown_radius = purves_model.get_radius_at_height(scalar_test_input)

        assert isinstance(scalar_crown_radius, np.ndarray)
        assert len(scalar_crown_radius) == vector_length
        assert np.all(scalar_crown_radius >= 0.0)
        assert np.all(
            scalar_crown_radius
            <= purves_model._get_purves_max_crown_radius(spcd_vector, dbh_vector)
        )
        # Test vector test input
        vector_crown_radius = purves_model.get_radius_at_height(vector_test_input)

        assert isinstance(vector_crown_radius, np.ndarray)
        assert len(vector_crown_radius) == vector_length
        assert np.all(vector_crown_radius >= 0.0)
        assert np.all(
            vector_crown_radius
            <= purves_model._get_purves_max_crown_radius(spcd_vector, dbh_vector)
        )