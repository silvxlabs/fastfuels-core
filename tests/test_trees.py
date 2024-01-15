# Internal imports

# External imports
from pathlib import Path

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"


class TestBetaCrownProfileModel:
    def test_get_normalized_height(self):
        """
        Tests that the normalized height is calculated correctly.
        """
        # Suppose that I have a tree with a height of 20m, a crown base height
        # of 10m, and a crown length of 10m
        height = 20
        crown_base_height = 10
        crown_length = 10
        model = BetaCrownProfileModel(1, 1, 1, crown_base_height, crown_length)

        # If I want the normalized height at 10m, I should get 0
