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

# Meta: This data is from the TRY database. Values are per species mean recomputed after removing
# measurements outside of 2.5 SD from mean. Species matched by genus and species to REF_SPECIES for
# SPCD and Jenkins group code.
REF_TRY_DB_LEAF = pd.read_csv(
    DATA_PATH / "REF_TRY_DB_LEAF.csv",
    index_col="SPCD",
    usecols=[
        "SPCD",
        "SLA_PETIOLE_EXCLUDED",
        "SLA_PETIOLE_INCLUDED",
        "SLA_PETIOLE_INDETERMINATE",
        "LEAF_ANGLE_DEG"
    ]
)
