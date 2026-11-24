# LANL Trees fixture

`lanl_trees.npz` holds crown bulk-density grids written by LANL Trees, used by
`tests/test_lanl_trees.py`.

**Source.** LANL Trees, https://github.com/lanl/Trees, commit `a9f32c1`
(LANL C22044, BSD-3). Each grid is the `treesrhof.dat` it writes for a tree
list read with `itrees = 2`: one species, one fuel bin, flat topography
(`topofile = 'flat'`, `aa1 = 1`), no litter or grass. The overlap cap
`trhofmax` is set to 1000 kg/m**3, so it never binds and each grid is a plain
sum of crowns. Inputs are float32, as LANL reads them.

**Cases.**

- Three single trees, `low_peak`, `mid_peak` and `high_peak` (height of maximum
  crown diameter low, middle and high in the crown), each at 0.5 m x 0.5 m,
  1 m x 1 m, 2 m x 1 m and 2 m x 2 m (horizontal x vertical). `low_peak`'s stem
  is at its cell's center; the other two are off center, one near a corner.
- `stand_1x1`: ten trees with overlapping crowns in a 24 m square at
  1 m x 1 m.

**Arrays.** `case` lists the case names. For each case `<name>`:

| name | shape, type | content |
|---|---|---|
| `<name>__trees` | (n, 7) float64 | per tree: `x`, `y` (m from the grid's south-west corner), height, crown base height, crown diameter, height of maximum crown diameter (m), crown bulk density (kg/m**3); `tree_columns` names them |
| `<name>__resolution` | (2,) float64 | horizontal and vertical cell size (m) |
| `<name>__bulk_density` | (nz, ny, nx) float32 | LANL's crown bulk density (kg/m**3); `z` from the ground, `y` northward |

`trees_commit` records the Trees commit.

**Rebuilding.** Build LANL Trees at commit `a9f32c1`, then run
`python scripts/build_lanl_trees_fixture.py path/to/trees.exe`, which rewrites
this file.
