# fastfuels-core
Core algorithms for FastFuels

## FastFuels Platform

The algorithms and methods in fastfuels-core serve as the foundation for the FastFuels Platform and the FastFuelsAPI.
The FastFuels API is a JSON API that connects various data sources to the algorithms in fastfuels-core in order to create, modify, and export fuel data for use in fire behavior models.

Documentation for the FastFuels Platform, including reference documentation for the API can be found at the following links:

1) https://docs.fastfuels.silvxlabs.com/
2) https://api.fastfuels.silvxlabs.com/redoc

## Crown ratio and crown base height

`Tree`, the tree DataFrames (`CBH` column) and `canopy_fuel` (`crown_base_height` column) take crown base height in meters, the height of the lowest live foliage, as crown geometry.
`CR` / `crown_ratio` is FIA's compacted crown ratio.
The uncompacted crown ratio model, a national refit to FIA trees with both `CR` and `UNCRCD` recorded, estimates crown base height from it, on arrays or on one `Tree`:

```python
from fastfuels_core.allometry.crown_ratio import estimate_crown_base_height
from fastfuels_core.trees import Tree

# Vectorized, e.g. over DataFrame columns
trees["CBH"] = estimate_crown_base_height(
    trees["SPCD"], trees["CR"], trees["DIA"], trees["HT"], pacific_northwest=False
)

# One tree: omit crown_base_height and it is estimated from crown_ratio
tree = Tree(202, 1, 30.0, 20.0, crown_ratio=0.4, pacific_northwest=False)
```

Both call the same function, so they give identical values.
A `crown_base_height` passed to `Tree` is used as is, and `crown_ratio` is then carried as data only.
A tree-frame row with a NaN `CBH` and a valid `CR` is estimated with `pacific_northwest=False`.
Set `pacific_northwest=True` for trees in `PACIFIC_NORTHWEST_STATES` (CA, OR, WA).
`crown_base_height = height - height * crown_ratio` reproduces the pre-0.18 behaviour.
