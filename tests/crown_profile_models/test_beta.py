# Core imports
from tests.utils import make_random_tree
from tests.utils import LIST_SPCDS
from tests.utils import make_tree_list
from fastfuels_core.crown_profile_models.beta import BetaCrownProfile

# External imports
import pytest
import numpy as np


@pytest.fixture
def model():
    # Suppose that I have a tree with a crown base height of 10m and crown length of 10m
    return BetaCrownProfile(species_code=122, crown_base_height=10, crown_length=10)


# generic way to create a custom model however we like for potential expansion of the tests
@pytest.fixture
def custom_model(scope="module"):
    def _model(sc, cbht, cl):
        return BetaCrownProfile(sc, cbht, cl)

    return _model


# generates a list of all the trees in the species code index
# the returned list is formatted in the particular way of PurvesCrownProfile models
@pytest.fixture()
def beta_tree_list():
    trees = make_tree_list("beta")
    tree_list = []
    for t in trees:
        tree = BetaCrownProfile(
            species_code=t.species_code,
            crown_base_height=t.crown_base_height,
            crown_length=t.crown_length,
        )
        tree_list.append(tree)
    return tree_list


@pytest.fixture()
def vector_test_input():
    return np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            61,
            -1,
        ]
    )


@pytest.fixture()
def scalar_test_input():
    input_pool = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            35.0,
            40.0,
            45.0,
            50.0,
            55.0,
            60.0,
            61.0,
            -1.0,
        ]
    )
    return input_pool[np.random.randint(input_pool.size)]


vector_length = 100_000


@pytest.fixture(scope="module")
def list_scpds():
    return LIST_SPCDS


@pytest.fixture(scope="module")
def tree_list():
    return [make_random_tree() for _ in range(vector_length)]


@pytest.fixture(scope="module")
def crown_base_height_vector(tree_list):
    return np.array([t.crown_base_height for t in tree_list])


@pytest.fixture(scope="module")
def crown_length_vector(tree_list):
    return np.array([t.crown_length for t in tree_list])


@pytest.fixture(scope="module")
def spcd_vector(tree_list):
    return np.array([t.species_code for t in tree_list])


class TestGetMaxRadius:

    @pytest.mark.parametrize("spcd", LIST_SPCDS)
    def test_individual_tree(self, spcd):
        """
        Tests the individual tree's crown profile using the species code `spcd`. The
        method generates a random tree based on the species code and evaluates the
        maximum crown radius using the Beta crown profile model. It verifies that
        the calculated maximum crown radius meets expected criteria:
        - The value should be a float (scalar)
        - The value should be greater than 0
        - The value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        """
        tree = make_random_tree(species_code=spcd)
        beta_model = BetaCrownProfile(
            species_code=tree.species_code,
            crown_base_height=tree.crown_base_height,
            crown_length=tree.crown_length,
        )
        max_crown_radius = beta_model.get_max_radius()

        # Value should be a float (scalar)
        assert isinstance(max_crown_radius, float)

        # Value should be greater than 0
        assert max_crown_radius > 0.0

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        # The trunc function is used because in some instances where the radius ==,
        # The max_crown_radius has one more decimal point than get_beta_max_crown_radius, causing test failure
        assert np.trunc(max_crown_radius) <= beta_model.get_beta_max_crown_radius(
            (beta_model.crown_base_height + beta_model.crown_length),
            beta_model.crown_base_height,
            beta_model.a,
            beta_model.b,
            beta_model.c,
            beta_model.beta,
        )

    @pytest.mark.parametrize("library", ["numpy", "pandas"])
    def test_tree_vector_numpy(
        self,
        library,
        tree_list,
        crown_base_height_vector,
        crown_length_vector,
        spcd_vector,
    ):
        """
        Tests the tree vector functionality using the specified library,
        either "numpy" or "pandas". The test creates a `BetaCrownProfile`
        model based on the provided tree attributes and checks the resulting
        maximum crown radius. The test ensures that the maximum crown radius
        is calculated correctly and conforms to expected constraints.
        """
        if library == "numpy":
            beta_model = BetaCrownProfile(
                spcd_vector, crown_base_height_vector, crown_length_vector
            )
        elif library == "pandas":
            import pandas as pd

            tree_population_dict = {
                "species_code": spcd_vector,
                "crown_base_height": crown_base_height_vector,
                "crown_length": crown_length_vector,
            }
            tree_population_df = pd.DataFrame(tree_population_dict)
            beta_model = BetaCrownProfile(
                tree_population_df["species_code"],
                tree_population_df["crown_base_height"],
                tree_population_df["crown_length"],
            )
        else:
            raise ValueError(f"Unknown library: {library}")

        max_crown_radius = beta_model.get_max_radius()

        assert isinstance(max_crown_radius, np.ndarray)
        assert len(max_crown_radius) == vector_length
        assert np.all(max_crown_radius > 0.0)
        assert np.all(
            np.trunc(max_crown_radius)
            <= beta_model.get_beta_max_crown_radius(
                (beta_model.crown_base_height + beta_model.crown_length),
                beta_model.crown_base_height,
                beta_model.a,
                beta_model.b,
                beta_model.c,
                beta_model.beta,
            )
        )


class TestGetRadiusAtHeight:
    @pytest.mark.parametrize("spcd", LIST_SPCDS)
    def test_individual_tree(self, spcd, scalar_test_input, vector_test_input):
        """
        Tests the individual tree's crown profile using the species code `spcd`. The
        method generates a random tree based on the species code and evaluates the
        maximum crown radius using the Beta crown profile model. It verifies that
        the calculated maximum crown radius meets expected criteria:
        - The value should be a float (scalar)
        - The value should be greater than 0
        - The value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        """
        tree = make_random_tree(species_code=spcd)
        beta_model = BetaCrownProfile(
            species_code=tree.species_code,
            crown_base_height=tree.crown_base_height,
            crown_length=tree.crown_length,
        )
        # Test scalar input first
        scalar_crown_radius = beta_model.get_radius_at_height(scalar_test_input)

        # Value should be a float (scalar)
        assert isinstance(scalar_crown_radius, float)

        # Value should be greater than 0
        assert scalar_crown_radius >= 0.0

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        # The trunc function is used because in some instances where the radius ==,
        # The max_crown_radius has one more decimal point than get_beta_max_crown_radius, causing test failure
        assert scalar_crown_radius <= beta_model.get_beta_max_crown_radius(
            (beta_model.crown_base_height + beta_model.crown_length),
            beta_model.crown_base_height,
            beta_model.a,
            beta_model.b,
            beta_model.c,
            beta_model.beta,
        )

        # Now test vector input
        vector_crown_radius = beta_model.get_radius_at_height(vector_test_input)

        # Value should be a float (scalar)
        assert isinstance(vector_crown_radius, np.ndarray)

        # Value should be greater than 0
        assert np.all(vector_crown_radius >= 0.0)

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        # The trunc function is used because in some instances where the radius ==,
        # The max_crown_radius has one more decimal point than get_beta_max_crown_radius, causing test failure
        assert np.all(
            vector_crown_radius
            <= beta_model.get_beta_max_crown_radius(
                (beta_model.crown_base_height + beta_model.crown_length),
                beta_model.crown_base_height,
                beta_model.a,
                beta_model.b,
                beta_model.c,
                beta_model.beta,
            )
        )

    @pytest.mark.parametrize("library", ["numpy", "pandas"])
    def test_tree_vector_numpy(
        self,
        library,
        tree_list,
        crown_base_height_vector,
        crown_length_vector,
        spcd_vector,
        scalar_test_input,
        vector_test_input,
    ):
        """
        Tests the tree vector functionality using the specified library,
        either "numpy" or "pandas". The test creates a `BetaCrownProfile`
        model based on the provided tree attributes and checks the resulting
        maximum crown radius. The test ensures that the maximum crown radius
        is calculated correctly and conforms to expected constraints.
        """
        if library == "numpy":
            beta_model = BetaCrownProfile(
                spcd_vector, crown_base_height_vector, crown_length_vector
            )
        elif library == "pandas":
            import pandas as pd

            tree_population_dict = {
                "species_code": spcd_vector,
                "crown_base_height": crown_base_height_vector,
                "crown_length": crown_length_vector,
            }
            tree_population_df = pd.DataFrame(tree_population_dict)
            beta_model = BetaCrownProfile(
                tree_population_df["species_code"],
                tree_population_df["crown_base_height"],
                tree_population_df["crown_length"],
            )
        else:
            raise ValueError(f"Unknown library: {library}")

        # Test scalar input
        scalar_crown_radius = beta_model.get_radius_at_height(scalar_test_input)

        # Value should be a float (scalar)
        assert isinstance(scalar_crown_radius, np.ndarray)

        # Value should be greater than 0
        assert np.all(scalar_crown_radius >= 0.0)

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        # The trunc function is used because in some instances where the radius ==,
        # The max_crown_radius has one more decimal point than get_beta_max_crown_radius, causing test failure
        assert np.all(
            scalar_crown_radius
            <= beta_model.get_beta_max_crown_radius(
                (beta_model.crown_base_height + beta_model.crown_length),
                beta_model.crown_base_height,
                beta_model.a,
                beta_model.b,
                beta_model.c,
                beta_model.beta,
            )
        )

        # Now test vector input
        vector_crown_radius = beta_model.get_radius_at_height(
            np.atleast_2d(vector_test_input).T
        )

        # Value should be a float (scalar)
        assert isinstance(vector_crown_radius, np.ndarray)

        # Value should be greater than 0
        assert np.all(vector_crown_radius >= 0.0)

        # Value should be less than or equal to the maximum possible crown radius for a tree with Canopy Ratio of 1
        # The trunc function is used because in some instances where the radius ==,
        # The max_crown_radius has one more decimal point than get_beta_max_crown_radius, causing test failure
        assert np.all(
            vector_crown_radius
            <= beta_model.get_beta_max_crown_radius(
                (beta_model.crown_base_height + beta_model.crown_length),
                beta_model.crown_base_height,
                beta_model.a,
                beta_model.b,
                beta_model.c,
                beta_model.beta,
            )
        )
