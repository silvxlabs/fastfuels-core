# Core imports
import random
#from numpy import random as rnd

# Internal imports
from fastfuels_core.trees import Tree, SPCD_PARAMS

LIST_SPCDS = [k for k in SPCD_PARAMS.keys() if SPCD_PARAMS[k]["SPGRP"] != -1]


def make_random_tree(
    species_code=None,
    status_code=None,
    diameter=None,
    height=None,
    crown_ratio=None,
    x=None,
    y=None,
    crown_profile_model=None,
    biomass_allometry_model=None,
):
    if species_code is None:
        species_code = int(random.choice(LIST_SPCDS))
    if status_code is None:
        status_code = random.randint(1, 3)
    if height is None:
        height = random.uniform(1, 60)
    if diameter is None:
        diameter = random.uniform(1, 100.0)
    if crown_ratio is None:
        crown_ratio = random.uniform(0.1, 1)
    if x is None:
        x = random.uniform(-180, 180)
    if y is None:
        y = random.uniform(-90, 90)
    if crown_profile_model is None:
        crown_profile_model = "beta"
    if biomass_allometry_model is None:
        biomass_allometry_model = "jenkins"

    return Tree(
        species_code=species_code,
        status_code=status_code,
        diameter=diameter,
        height=height,
        crown_ratio=crown_ratio,
        x=x,
        y=y,
        crown_profile_model_type=crown_profile_model,
        biomass_allometry_model_type=biomass_allometry_model,
    )

#generate a randon list of trees out of SPCDS list
#can specify what crown_profile_model one wishes to use
def make_tree_list(
    _crown_profile_model=None,
):
    tree_list =[]
    spc_index = None
    for spc in LIST_SPCDS:
        tree = make_random_tree(crown_profile_model=_crown_profile_model, species_code=int(spc))
        tree_list.append(tree)

    return tree_list