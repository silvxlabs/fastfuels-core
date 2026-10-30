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
`CR` / `crown_ratio` is FIA's compacted crown ratio and is carried as data only.
To convert, `fastfuels_core.allometry.crown_ratio.estimate_crown_base_height(species_code, compacted_crown_ratio, dbh_cm, height_m, pacific_northwest=False)` applies the uncompacted crown ratio model, a national refit to FIA trees with both `CR` and `UNCRCD` recorded.
Set `pacific_northwest=True` for trees in `PACIFIC_NORTHWEST_STATES` (CA, OR, WA).
`crown_base_height = height - height * crown_ratio` reproduces the pre-0.18 behaviour.
