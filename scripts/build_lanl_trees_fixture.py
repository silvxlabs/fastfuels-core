"""
Builds tests/data/lanl_trees/lanl_trees.npz, the LANL Trees fixture used by
tests/test_lanl_trees.py.

Runs a LANL Trees executable (https://github.com/lanl/Trees, built at commit
a9f32c1) on a fixed set of tree lists and stores the crown bulk-density grids it
writes (treesrhof.dat): single trees at four resolutions, with stems at and off
their cell centers, and a small stand of overlapping crowns. The overlap cap is
set high enough that it never binds, so every grid is a plain sum of crowns.

Usage (from the repository root):
    python scripts/build_lanl_trees_fixture.py path/to/trees.exe [output.npz]
"""

# Core imports
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# External imports
import numpy as np
from scipy.io import FortranFile

TREES_COMMIT = "a9f32c1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "lanl_trees"
    / "lanl_trees.npz"
)
# Columns of each case's tree array.
TREE_COLUMNS = (
    "x",
    "y",
    "height",
    "crown_base_height",
    "crown_diameter",
    "max_crown_diameter_height",
    "bulk_density",
)
# Well above any summed bulk density, so LANL's overlap cap never binds.
NO_CAP = 1000.0

# name: height, crown base height, crown diameter, height of maximum crown
# diameter (m), crown bulk density (kg/m**3), and the stem's position within
# its cell as a fraction of the cell along x and y.
SINGLE_TREES = {
    "low_peak": (18.3, 6.1, 5.6, 8.2, 0.21, (0.5, 0.5)),
    "mid_peak": (24.7, 9.4, 7.3, 17.0, 0.15, (0.13, 0.71)),
    "high_peak": (12.2, 2.3, 3.9, 10.9, 0.33, (0.94, 0.02)),
}
RESOLUTIONS = ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0))
STAND_RESOLUTION = (1.0, 1.0)
STAND_SIZE = 24.0  # m, square
STAND_TREES = 10
STAND_SEED = 137


def f32(value: float) -> float:
    """The float32 value LANL Trees computes with, as a Python float."""
    return float(np.float32(value))


def run_trees(exe: Path, trees: np.ndarray, nx, ny, nz, dx, dz) -> np.ndarray:
    """Run LANL Trees on ``trees`` and return treesrhof.dat as (nz, ny, nx)."""
    with tempfile.TemporaryDirectory(prefix="lanl_trees_") as tmp:
        work = Path(tmp)
        shutil.copy2(exe, work / "trees.exe")
        # itrees=2 columns: species x y height crown_base crown_diameter
        # max_crown_diameter_height bulk_density moisture size_scale.
        rows = [
            "\t".join(["1"] + [f"{np.float32(v):.9g}" for v in t] + ["1.0", "0.0005"])
            for t in trees
        ]
        (work / "treelist.txt").write_text("\n".join(rows) + "\n")
        (work / "fuellist").write_text(
            f"""&fuellist
    nx = {nx}
    ny = {ny}
    nz = {nz}
    dx = {dx}
    dy = {dx}
    dz = {dz}
    aa1 = 1
    singlefuel = 1
    lreduced = 0
    topofile = 'flat'
    itrees = 2
    tfuelbins = 1
    treefile = 'treelist.txt'
    ndatax = {nx * dx}
    ndatay = {ny * dx}
    datalocx = 0
    datalocy = 0
    trhofmax = {NO_CAP}
    ilitter = 0
    igrass = 0
/
"""
        )
        proc = subprocess.run(["./trees.exe"], cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout + proc.stderr)
        with open(work / "treesrhof.dat", "rb") as fh:
            rho = FortranFile(fh).read_reals(dtype="float32")
    return rho.reshape((nz, ny, nx))


def single_tree_cases():
    for name, (ht, hb, cd, hd, cbd, (fx, fy)) in SINGLE_TREES.items():
        for dx, dz in RESOLUTIONS:
            half = math.ceil(cd / 2.0 / dx) + 2  # cells from the stem to an edge
            n = 2 * half + 1
            nz = math.ceil(ht / dz) + 2
            tree = ((half + fx) * dx, (half + fy) * dx, ht, hb, cd, hd, cbd)
            yield f"{name}_{dx:g}x{dz:g}", [tree], n, n, nz, dx, dz


def stand_case():
    rng = np.random.default_rng(STAND_SEED)
    trees = []
    for _ in range(STAND_TREES):
        ht = rng.uniform(10.0, 25.0)
        hb = ht * rng.uniform(0.2, 0.5)
        length = ht - hb
        hd = hb + length * rng.uniform(0.1, 0.9)
        cd = rng.uniform(3.0, 7.0)
        x, y = rng.uniform(7.0, STAND_SIZE - 7.0, size=2)
        trees.append((x, y, ht, hb, cd, hd, rng.uniform(0.1, 0.4)))
    dx, dz = STAND_RESOLUTION
    n = int(STAND_SIZE / dx)
    nz = math.ceil(max(t[2] for t in trees) / dz) + 2
    return "stand_1x1", trees, n, n, nz, dx, dz


def main(exe: Path, output: Path) -> None:
    arrays = {"trees_commit": np.array(TREES_COMMIT), "tree_columns": TREE_COLUMNS}
    names = []
    for name, trees, nx, ny, nz, dx, dz in [*single_tree_cases(), stand_case()]:
        trees = np.array([[f32(v) for v in t] for t in trees])
        rho = run_trees(exe, trees, nx, ny, nz, dx, dz)
        if not rho.any():
            raise RuntimeError(f"LANL Trees wrote an empty grid for {name}.")
        names.append(name)
        arrays[f"{name}__trees"] = trees
        arrays[f"{name}__resolution"] = np.array([dx, dz])
        arrays[f"{name}__bulk_density"] = rho
        print(f"{name}: {len(trees)} trees, grid {nz}x{ny}x{nx}")
    arrays["case"] = np.array(names)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(
        Path(sys.argv[1]).resolve(),
        Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_OUTPUT,
    )
