# internal inputs
from fastfuels_core.disc_sampling import DiscSampler
from tests.utils import make_random_tree
from tests.utils import LIST_SPCDS


# external inputs
import pytest
import numpy as np
import pandas as pd

vector_length = 100_000


@pytest.fixture(scope="module")
def list_scpds():
    return LIST_SPCDS


@pytest.fixture(scope="module")
def tree_list():
    return [make_random_tree() for _ in range(vector_length)]


@pytest.fixture(scope="module")
def id_vector(tree_list):
    return np.array([np.random.randint(1) for t in tree_list])


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


@pytest.fixture(scope="module")
def stscd_vector(tree_list):
    return np.array([t.status_code for t in tree_list])


@pytest.fixture(scope="module")
def x_vector(tree_list):
    return np.array([t.x for t in tree_list])


@pytest.fixture(scope="module")
def y_vector(tree_list):
    return np.array([t.y for t in tree_list])


class TestDiscSampling:

    @pytest.mark.parametrize("crown_profile_model", ["beta", "purves"])
    def test_init(
        self,
        tree_list,
        id_vector,
        dbh_vector,
        height_vector,
        crown_ratio_vector,
        spcd_vector,
        stscd_vector,
        x_vector,
        y_vector,
        crown_profile_model,
    ):
        tree_population_dict = {
            "TREE_ID": id_vector,
            "SPCD": spcd_vector,
            "STATUSCD": stscd_vector,
            "DIA": dbh_vector,
            "HT": height_vector,
            "CR": crown_ratio_vector,
            "X": x_vector,
            "Y": y_vector,
        }
        tree_population_df = pd.DataFrame(tree_population_dict)

        disc_sampling_object = DiscSampler(
            tree_population_df, crown_profile_model=crown_profile_model
        )
        print(disc_sampling_object)
