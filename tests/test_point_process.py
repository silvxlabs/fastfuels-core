# Core imports
from pathlib import Path

# Internal imports
from fastfuels_core.point_process import PointProcess, InhomogeneousPoissonProcess

# External imports
import pytest
import pandas as pd
import geopandas as gpd

TEST_PATH = Path(__file__).parent
TEST_DATA_PATH = TEST_PATH / "data"


class TestPointProcess:
    def test_init_poisson(self):
        point_process = PointProcess("inhomogeneous_poisson")
        assert isinstance(point_process, PointProcess)
        assert point_process.process_type == "inhomogeneous_poisson"
        assert point_process.process == InhomogeneousPoissonProcess

    def test_init_invalid_process_type(self):
        with pytest.raises(NotImplementedError):
            PointProcess("invalid_process_type")

    def test_run(self):
        point_process = PointProcess("inhomogeneous_poisson")
        with pytest.raises(TypeError):
            point_process.run(None, None)

    def test_run_invalid_process_type(self):
        point_process = PointProcess("inhomogeneous_poisson")
        point_process.process = None
        with pytest.raises(NotImplementedError):
            point_process.run(None, None)
