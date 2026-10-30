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

`equation=` (and `Tree(..., crown_ratio_equation=...)`) selects the published regional equations instead of the refit; their coefficients are rows of `fastfuels_core/data/REF_CROWN_RATIO.csv`, stored as printed with the table and page in `TABLE`:

| `equation` | Region | Predictors | Coverage | Source |
|---|---|---|---|---|
| `fia_national_logit_2026` (default) | National, CA/OR/WA rows via `pacific_northwest` | CCR, ln DBH, ln height | Every FIA species (genus and softwood/hardwood fallbacks) | National FIA refit of the published form |
| `monleon_2004_eq1` | CA, OR, WA | CCR, ln DBH, ln height, height/DBH | 28 species, other *Pinus* and *Quercus* | Monleon, Azuma & Gedney 2004, *West. J. Appl. For.* 19(4):260–267, Table 2, [doi:10.1093/wjaf/19.4.260](https://doi.org/10.1093/wjaf/19.4.260) |
| `monleon_2004_eq2` | CA, OR, WA | CCR, ln DBH | 28 species, other *Pinus* and *Quercus* | Same paper, Table 4 (and Table 2 rows without height terms) |
| `toney_reeves_2009` | ID, MT, WY, NV, UT, CO, AZ, NM | CCR, ln diameter | 35 species, no fallback; 4 woodland species fitted on root-collar diameter give NaN | Toney & Reeves 2009, *West. J. Appl. For.* 24(2):76–82, Table 3, [doi:10.1093/wjaf/24.2.76](https://doi.org/10.1093/wjaf/24.2.76) |
| `randolph_2010` | AL, AR, FL, GA, KY, LA, NC, SC, TN, TX, VA | CCR, ln diameter; woodland junipers ln height | 24 species, 10 genera, forest and woodland junipers | Randolph 2010, *South. J. Appl. For.* 34(3):118–123, Table 3 and text p. 121, [doi:10.1093/sjaf/34.3.118](https://doi.org/10.1093/sjaf/34.3.118) |
| `westfall_2019` | 24 Northern Research Station states | CCR, ln DBH | 27 groups by FIA eastern species group | Westfall, Westfall & Randolph 2019, *For. Sci.* 65(5):593–601, Table 2, [doi:10.1093/forsci/fxz029](https://doi.org/10.1093/forsci/fxz029) (erratum [doi:10.1093/forsci/fxz067](https://doi.org/10.1093/forsci/fxz067) corrects an author name only) |

Each published equation applies its own paper's rule: UCR = CCR above CCR 0.90 (Monleon, Toney & Reeves, Randolph), or UCR = CCR only where the prediction falls below CCR (Westfall).
A species an equation does not cover gives NaN, and `pacific_northwest=True` with a published equation raises.
`fastfuels_core.allometry.crown_ratio` documents each equation's sample, caveats and not-recommended species (`RECOMMENDED` 0).
