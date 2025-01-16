"""
tests/crown_profile_models/test_purves.py
"""

# Core imports
from tests.utils import make_tree_list
from tests.utils import make_random_tree
from tests.utils import LIST_SPCDS
from fastfuels_core.crown_profile_models.purves import PurvesCrownProfile

# External imports
import pytest
import numpy as np
import pandas as pd

NUM_TREES = 100_000
NUM_HEIGHT_STEPS = 1_000


# generates a list of all the trees in the species code index
# the returned list is formatted in the particular way of PurvesCrownProfile models
@pytest.fixture()
def purves_tree_list():
    trees = make_tree_list("purves")
    tree_list = []
    for t in trees:
        tree = PurvesCrownProfile(
            species_code=t.species_code,
            height=t.height,
            dbh=t.diameter,
            crown_ratio=t.crown_ratio,
        )
        tree_list.append(tree)

    return tree_list


@pytest.fixture()
def vector_tree_heights() -> np.ndarray:
    """
    Creates a vector of test heights that spans the range of possible height values,
    plus edge cases.

    The heights array includes:
    - A negative value (-1) to test below-ground behavior
    - Regular sequence of heights across the crown's range
    - An extremely large value (100000) to test above-crown behavior
    """
    # Create evenly spaced sequence across a height range
    main_sequence = np.linspace(0, 100, NUM_HEIGHT_STEPS)

    # Add edge cases at beginning and end
    heights = np.concatenate([[-1], main_sequence, [100000]])  # type: ignore

    return heights


@pytest.fixture(scope="module")
def list_scpds():
    return LIST_SPCDS


@pytest.fixture(scope="module")
def tree_list():
    return [make_random_tree() for _ in range(NUM_TREES)]


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
    def test_scalar(self, spcd):
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
        assert max_crown_radius <= purves_model.max_theoretical_crown_radius

    @pytest.mark.parametrize("library", ["numpy", "pandas"])
    def test_vector(
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
        # Initialize the PurvesCrownProfile model with numpy arrays
        if library == "numpy":
            purves_model = PurvesCrownProfile(
                spcd_vector, dbh_vector, height_vector, crown_ratio_vector
            )

        # Initialize the PurvesCrownProfile model with pandas Series from a DataFrame
        elif library == "pandas":
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
        assert len(max_crown_radius) == NUM_TREES
        assert np.all(max_crown_radius > 0.0)
        assert np.all(
            max_crown_radius <= purves_model.max_theoretical_crown_radius.reshape(-1)
        )


class TestGetRadiusAtHeight:

    @pytest.mark.parametrize("spcd", LIST_SPCDS)
    def test_scalar_height_input(self, spcd):
        """
        Tests the crown profile radius calculation with a scalar height input.

        This tests that a single tree with a single height is evaluated to a single radius value.

        Verifies that:
        - Output is a float (scalar)
        - Output is non-negative
        - Output is less than or equal to the maximum crown radius
        """
        tree = make_random_tree(species_code=spcd)
        purves_model = PurvesCrownProfile(
            species_code=tree.species_code,
            dbh=tree.diameter,
            height=tree.height,
            crown_ratio=tree.crown_ratio,
        )
        random_tree_height = np.random.uniform(
            tree.crown_base_height, tree.crown_base_height + tree.crown_length
        )
        radius = purves_model.get_radius_at_height(random_tree_height)

        assert isinstance(radius, float)
        assert radius >= 0.0
        assert radius <= purves_model.get_max_radius()

    @pytest.mark.parametrize("spcd", LIST_SPCDS)
    def test_1d_vector_height_input(self, spcd, vector_tree_heights):
        """
        Tests the crown profile radius calculation with a 1D vector of heights.

        This tests that a single tree with multiple heights is evaluated to multiple radius values.

        Verifies that:
        - Output is a numpy array
        - Output has same shape as input
        - All values are non-negative
        - All values are less than or equal to the maximum crown radius
        - At least one value is greater than 0
        """
        tree = make_random_tree(species_code=spcd)
        purves_model = PurvesCrownProfile(
            species_code=tree.species_code,
            dbh=tree.diameter,
            height=tree.height,
            crown_ratio=tree.crown_ratio,
        )
        radii = purves_model.get_radius_at_height(vector_tree_heights)
        max_radius = purves_model.get_max_radius()

        assert isinstance(radii, np.ndarray)
        assert radii.shape == vector_tree_heights.shape
        assert np.all(radii >= 0.0)
        assert np.all(radii <= max_radius)
        assert np.any(radii > 0.0)

    @pytest.mark.parametrize("library", ["numpy", "pandas"])
    def test_2d_vector_height_input(
        self,
        library,
        tree_list,
        dbh_vector,
        height_vector,
        crown_ratio_vector,
        spcd_vector,
        vector_tree_heights,
    ):
        """
        Tests the crown profile radius calculation with a 2D vector of heights
        (multiple heights for multiple trees).
        Verifies that:
        - Output is a numpy array
        - Output has correct shape (n_trees x n_heights)
        - All values are non-negative
        - All values are less than or equal to their respective maximum crown radii
        - At least one value is greater than 0
        """
        # Initialize the PurvesCrownProfile model with numpy arrays
        if library == "numpy":
            purves_model = PurvesCrownProfile(
                spcd_vector, dbh_vector, height_vector, crown_ratio_vector
            )
        # Initialize the PurvesCrownProfile model with pandas Series from a DataFrame
        elif library == "pandas":
            tree_population_df = pd.DataFrame(
                {
                    "species_code": spcd_vector,
                    "dbh": dbh_vector,
                    "height": height_vector,
                    "crown_ratio": crown_ratio_vector,
                }
            )
            purves_model = PurvesCrownProfile(
                tree_population_df["species_code"],
                tree_population_df["dbh"],
                tree_population_df["height"],
                tree_population_df["crown_ratio"],
            )
        else:
            raise ValueError(f"Unknown library: {library}")

        radii = purves_model.get_radius_at_height(vector_tree_heights)
        max_radii = purves_model.get_max_radius()

        assert isinstance(radii, np.ndarray)
        assert radii.shape == (NUM_TREES, len(vector_tree_heights))
        assert np.all(radii >= 0.0)
        assert np.all(radii <= max_radii.reshape(-1, 1))
        assert np.any(radii > 0.0)
