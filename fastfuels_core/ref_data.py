# Core imports
from importlib.resources import files

# External imports
import pandas as pd


DATA_PATH = files("fastfuels_core.data")
REF_SPECIES = pd.read_csv(  # type: ignore
    DATA_PATH / "REF_SPECIES.csv",
    index_col="SPCD",
    usecols=[
        "SPCD",
        "COMMON_NAME",
        "GENUS",
        "SPECIES",
        "SPECIES_SYMBOL",
        "WOODLAND",
        "SFTWD_HRDWD",
        "JENKINS_SPGRPCD",
        "PURVES_TRAIT_SCORE",
    ],
)
SPCD_PARAMS = REF_SPECIES.to_dict(orient="index")
REF_JENKINS = pd.read_csv(  # type: ignore
    DATA_PATH / "REF_JENKINS.csv",
    index_col="JENKINS_SPGRPCD",
)
