# Internal imports
from fastfuels_core.trees import TreeCollection

# External imports
import pandas as pd
from pathlib import Path

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"


class TestTreeCollection:
    def test_from_fia(self):
        """
        Create a TreeCollection from FIA data stored in a parquet file and
        compare it to a previously saved TreeCollection.
        """
        fia_data_path = TEST_DATA_PATH / "trees_fia_format.parquet"
        fia_data = pd.read_parquet(fia_data_path)
        tree_collection = TreeCollection.from_fia_data(fia_data)

        test_data_path = TEST_DATA_PATH / "trees_test_data.parquet"
        test_data = pd.read_parquet(test_data_path)
        test_tree_collection = TreeCollection(test_data)

        assert tree_collection.data.equals(test_tree_collection.data)
