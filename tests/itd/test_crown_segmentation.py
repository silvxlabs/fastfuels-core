from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_origin

from fastfuels_core.itd.crown_segmentation import _segment as _segment_frontier
from fastfuels_core.itd.crown_segmentation import dalponte2016
from fastfuels_core.itd.local_maxima_filter import fixed_window_filter
from tests.itd.reference_crown_segmentation import dalponte2016_reference
from tests.itd.reference_segment_full_scan import segment_full_scan

DEFAULTS = dict(
    min_height=2.0,
    max_height=None,
    min_relative_height=0.45,
    min_relative_crown_height=0.55,
    max_crown_radius=10.0,
)

CHUNK_SIZES = [4, 7, 16]


def _chm(values: np.ndarray, pixel_size: float = 1.0) -> xr.DataArray:
    chm = xr.DataArray(np.asarray(values, dtype=np.float64), dims=["y", "x"])
    chm.rio.write_crs("EPSG:32611", inplace=True)
    chm.rio.write_transform(
        from_origin(500000.0, 4000000.0, pixel_size, pixel_size), inplace=True
    )
    return chm


def _treetops(chm: xr.DataArray, cells: list[tuple[int, int]]) -> pd.DataFrame:
    """Treetops at the centers of the given (row, col) cells."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    xs, ys = rio.transform.xy(chm.rio.transform(), rows, cols)
    return pd.DataFrame(
        {
            "x": np.asarray(xs, dtype=np.float64),
            "y": np.asarray(ys, dtype=np.float64),
            "height": chm.values[rows, cols],
        }
    )


def _cone(shape, row, col, height, slope=2.0) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.clip(height - slope * np.hypot(yy - row, xx - col), 0.0, None)


def _dist(shape, row, col) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.hypot(yy - row, xx - col)


def _segment(chm, cells, **overrides) -> np.ndarray:
    """Run unchunked and at several chunk sizes; assert all agree."""
    params = {**DEFAULTS, **overrides}
    treetops = _treetops(chm, cells)
    expected = dalponte2016(chm, treetops, **params)
    assert isinstance(expected.data, np.ndarray)
    for size in CHUNK_SIZES:
        chunked = dalponte2016(chm.chunk(size), treetops, **params)
        assert isinstance(chunked.data, da.Array)
        assert chunked.data.chunks == chm.chunk(size).data.chunks
        np.testing.assert_array_equal(chunked.values, expected.values)

    reference = dalponte2016_reference(chm.values, cells, chm.rio.transform(), **params)
    np.testing.assert_array_equal(expected.values, reference)
    return expected.values


def _forest(seed: int, shape=(60, 70), n_trees=25):
    """Random cones with distinct heights, including some near chunk edges."""
    rng = np.random.default_rng(seed)
    chm = np.zeros(shape)
    cells: list[tuple[int, int]] = []
    while len(cells) < n_trees:
        r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        if any(abs(r - r2) + abs(c - c2) < 4 for r2, c2 in cells):
            continue
        cells.append((r, c))
    heights = rng.permutation(np.linspace(8.0, 30.0, n_trees))
    for (r, c), h in zip(cells, heights):
        slope = rng.uniform(1.0, 3.0)
        chm = np.maximum(chm, _cone(shape, r, c, h, slope))
    chm += rng.uniform(0.0, 0.3, shape)
    # A cell holding a treetop must be its crown's peak for a sensible scene.
    for r, c in cells:
        chm[r, c] = max(
            chm[r, c], chm[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2].max()
        )
    return _chm(chm), cells


def test_isolated_cone():
    shape = (25, 25)
    chm = _chm(_cone(shape, 12, 12, 20.0))
    labels = _segment(chm, [(12, 12)], min_relative_crown_height=0.0)
    # 20 - 2d >= 0.45 * 20  <=>  d <= 5.5
    expected = np.where(_dist(shape, 12, 12) <= 5.5, 1, 0)
    np.testing.assert_array_equal(labels, expected)


def test_touching_cones():
    shape = (21, 30)
    chm = _chm(np.maximum(_cone(shape, 10, 8, 20.0), _cone(shape, 10, 18, 16.0)))
    labels = _segment(
        chm, [(10, 8), (10, 18)], min_relative_height=0.3, min_relative_crown_height=0.0
    )
    values = chm.values
    cols = np.arange(shape[1])[None, :]
    # Both seeds are on row 10, so every cell in col 13 is reached by both
    # crowns in the same step and goes to the taller tree 1.
    np.testing.assert_array_equal(labels == 1, (values >= 0.3 * 20.0) & (cols <= 13))
    np.testing.assert_array_equal(labels == 2, (values >= 0.3 * 16.0) & (cols >= 14))
    assert set(np.unique(labels[10])) == {0, 1, 2}


def test_understory_below_min_relative_height_excluded():
    shape = (25, 25)
    values = _cone(shape, 12, 12, 20.0)
    crown = _dist(shape, 12, 12) <= 5.5
    understory = ~crown & (_dist(shape, 12, 12) <= 9.0)
    values[understory] = 5.0  # below 0.45 * 20 = 9
    chm = _chm(values)

    labels = _segment(chm, [(12, 12)], min_relative_crown_height=0.0)
    np.testing.assert_array_equal(labels, np.where(crown, 1, 0))

    # The understory joins once the threshold drops below it.
    labels = _segment(
        chm, [(12, 12)], min_relative_height=0.2, min_relative_crown_height=0.0
    )
    assert np.all(labels[understory] == 1)


def test_low_margin_cut_by_min_relative_crown_height():
    values = np.zeros((15, 15))
    values[3:12, 3:12] = 10.0  # margin: 0.5 of the crown mean
    values[5:10, 5:10] = 20.0  # crown core
    chm = _chm(values)
    core = values == 20.0

    labels = _segment(chm, [(7, 7)], min_relative_crown_height=0.55)
    np.testing.assert_array_equal(labels, np.where(core, 1, 0))

    labels = _segment(chm, [(7, 7)], min_relative_crown_height=0.45)
    np.testing.assert_array_equal(labels, np.where(values > 0, 1, 0))


@pytest.mark.parametrize("pixel_size", [1.0, 0.5])
def test_crown_clipped_at_max_crown_radius(pixel_size):
    shape = (31, 31)
    chm = _chm(np.full(shape, 10.0), pixel_size=pixel_size)
    labels = _segment(chm, [(15, 15)], max_crown_radius=3.0)
    expected = _dist(shape, 15, 15) * pixel_size <= 3.0
    np.testing.assert_array_equal(labels, np.where(expected, 1, 0))


@pytest.mark.parametrize(
    "heights, winner",
    [((10.0, 12.0), 2), ((12.0, 10.0), 1), ((11.0, 11.0), 1)],
)
def test_contested_cell_goes_to_taller_tree(heights, winner):
    # Col 2 is reached by both crowns in step 2.
    values = np.array([[heights[0], 9.0, 9.0, 9.0, heights[1]]])
    chm = _chm(values)
    labels = _segment(chm, [(0, 0), (0, 4)])
    assert labels[0, 2] == winner
    np.testing.assert_array_equal(labels[0, [0, 1, 3, 4]], [1, 1, 2, 2])


def test_ridge_from_distant_tall_tree_does_not_change_nearby_crown():
    shape = (21, 60)
    small = _cone(shape, 10, 45, 12.0, slope=1.0)
    # A ridge descending from a 30 m tree at col 5 all the way to the small
    # tree's crown.
    cols = np.arange(shape[1])
    ridge = np.zeros(shape)
    ridge[9:12, :] = np.clip(30.0 - 0.5 * np.abs(cols - 5), 0.0, None)
    tall = np.maximum(_cone(shape, 10, 5, 30.0, slope=1.0), ridge)

    chm = _chm(np.maximum(small, tall))

    params = dict(max_crown_radius=8.0, min_relative_height=0.3)
    alone = _segment(chm, [(10, 45)], **params)
    both = _segment(chm, [(10, 5), (10, 45)], **params)

    np.testing.assert_array_equal(both == 2, alone == 1)
    assert (alone == 1).sum() > 1
    assert np.all(_dist(shape, 10, 5)[both == 1] <= 8.0)


@pytest.mark.parametrize(
    "seed_height, overrides",
    [(1.95, {}), (30.0, {"max_height": 25.0}), (np.nan, {})],
)
def test_treetop_outside_height_range_is_one_cell(seed_height, overrides):
    # Every neighbour would qualify for a treetop inside the range.
    values = np.full((9, 9), 2.0 if seed_height == 1.95 else 24.0)
    values[4, 4] = seed_height
    chm = _chm(values)
    labels = _segment(chm, [(4, 4)], min_relative_crown_height=0.0, **overrides)
    expected = np.zeros((9, 9), dtype=np.int32)
    expected[4, 4] = 1
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize("fill", [0.0, np.nan])
def test_empty_chm(fill):
    chm = _chm(np.full((20, 20), fill))
    labels = _segment(chm, [])
    assert labels.dtype == np.int32
    assert not labels.any()


def test_no_treetops_on_forested_chm():
    chm, _ = _forest(0)
    labels = _segment(chm, [])
    assert not labels.any()


@pytest.mark.parametrize("seed", range(4))
def test_forest_matches_reference_and_chunked(seed):
    chm, cells = _forest(seed)
    labels = _segment(chm, cells, max_crown_radius=6.0)
    # Several crowns cross chunk boundaries at every tested chunk size.
    assert len(np.unique(labels)) > 10


def test_chunked_crowns_span_boundaries_and_chunk_edge_treetops():
    shape = (40, 40)
    # Treetops on both sides of the chunk boundaries at 8, 16, 24 and 32.
    cells = [(7, 7), (8, 13), (15, 16), (16, 24), (23, 31), (24, 8), (31, 23), (32, 32)]
    values = np.zeros(shape)
    for i, (r, c) in enumerate(cells):
        values = np.maximum(values, _cone(shape, r, c, 10.0 + i, slope=1.5))
    chm = _chm(values)
    params = {**DEFAULTS, "max_crown_radius": 3.5}
    labels = _segment(chm, cells, **params)

    spanning = 0
    for k in range(1, len(cells) + 1):
        rows, cols = np.nonzero(labels == k)
        if len(set(rows // 8)) > 1 or len(set(cols // 8)) > 1:
            spanning += 1
    assert spanning == len(cells)

    treetops = _treetops(chm, cells)
    for size in (8, (8, 5), (13, 6)):
        np.testing.assert_array_equal(
            dalponte2016(chm.chunk(size), treetops, **params).values, labels
        )


@pytest.mark.parametrize("seed", range(4))
def test_result_independent_of_treetop_order(seed):
    chm, cells = _forest(seed)
    treetops = _treetops(chm, cells)
    labels = dalponte2016(chm, treetops, **DEFAULTS).values

    order = np.random.default_rng(seed).permutation(len(cells))
    shuffled = dalponte2016(
        chm, treetops.iloc[order].reset_index(drop=True), **DEFAULTS
    ).values

    # Shuffled label j + 1 is original label order[j] + 1.
    relabel = np.zeros(len(cells) + 1, dtype=np.int32)
    relabel[1:] = order + 1
    np.testing.assert_array_equal(relabel[shuffled], labels)


def test_output_grid_dtype_and_seed_labels():
    chm, cells = _forest(1)
    treetops = _treetops(chm, cells)
    treetops.index = treetops.index + 100  # labels follow row position

    for chm_in in (chm, chm.chunk(16)):
        labels = dalponte2016(chm_in, treetops, **DEFAULTS)
        assert labels.dtype == np.int32
        assert labels.dims == chm.dims
        assert labels.shape == chm.shape
        assert labels.rio.transform() == chm.rio.transform()
        assert labels.rio.crs == chm.rio.crs
        values = labels.values
        for k, (r, c) in enumerate(cells, start=1):
            assert values[r, c] == k
        assert values.min() >= 0 and values.max() <= len(cells)


def test_accepts_itd_output():
    chm, _ = _forest(2)
    treetops = fixed_window_filter(chm, 2.0, 1.0, 3.0).compute()
    labels = dalponte2016(chm, treetops, **DEFAULTS).values
    assert set(np.unique(labels)) - {0} == set(range(1, len(treetops) + 1))


def test_float32_chunked_chm():
    chm, cells = _forest(3)
    chm32 = chm.astype(np.float32)
    treetops = _treetops(chm32, cells)
    np.testing.assert_array_equal(
        dalponte2016(chm32.chunk(7), treetops, **DEFAULTS).values,
        dalponte2016(chm32, treetops, **DEFAULTS).values,
    )


class TestValidation:
    def setup_method(self):
        self.chm = _chm(_cone((11, 11), 5, 5, 10.0))
        self.treetops = _treetops(self.chm, [(5, 5)])

    def test_chm_must_be_2d(self):
        chm3d = xr.DataArray(np.zeros((2, 11, 11)), dims=["band", "y", "x"])
        with pytest.raises(ValueError, match="2D"):
            dalponte2016(chm3d, self.treetops, **DEFAULTS)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"min_height": -1.0},
            {"min_height": np.nan},
            {"max_height": 1.0},
            {"max_height": np.inf},
            {"min_relative_height": -0.1},
            {"min_relative_height": 1.0},
            {"min_relative_crown_height": -0.1},
            {"min_relative_crown_height": 1.0},
            {"max_crown_radius": 0.0},
            {"max_crown_radius": -1.0},
            {"max_crown_radius": np.inf},
        ],
    )
    def test_parameter_ranges(self, overrides):
        with pytest.raises(ValueError):
            dalponte2016(self.chm, self.treetops, **{**DEFAULTS, **overrides})

    @pytest.mark.parametrize(
        "dx, dy", [(-6.0, 0.0), (6.0, 0.0), (0.0, 6.0), (0.0, -6.0)]
    )
    def test_treetop_outside_chm(self, dx, dy):
        treetops = self.treetops.copy()
        treetops["x"] += dx
        treetops["y"] += dy
        with pytest.raises(ValueError, match="outside"):
            dalponte2016(self.chm, treetops, **DEFAULTS)

    def test_two_treetops_in_same_cell(self):
        treetops = pd.concat([self.treetops, self.treetops + [0.2, 0.2, 0.0]])
        with pytest.raises(ValueError, match="same CHM cell"):
            dalponte2016(self.chm, treetops, **DEFAULTS)

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="missing"):
            dalponte2016(self.chm, self.treetops[["x"]], **DEFAULTS)

    def test_non_finite_coordinates(self):
        treetops = self.treetops.copy()
        treetops.loc[0, "x"] = np.nan
        with pytest.raises(ValueError, match="finite"):
            dalponte2016(self.chm, treetops, **DEFAULTS)


def _random_case(seed: int):
    """A random CHM, seeds and parameters for the full-scan comparison."""
    rng = np.random.default_rng(seed)
    nrows, ncols = (int(v) for v in rng.integers(1, 60, 2))
    yy, xx = np.mgrid[:nrows, :ncols]
    n_bumps = int(rng.integers(1, 40))
    rows, cols = rng.uniform(0, nrows, n_bumps), rng.uniform(0, ncols, n_bumps)
    heights = rng.uniform(3.0, 35.0, n_bumps)
    widths = rng.uniform(0.8, 6.0, n_bumps)
    bumps = heights[:, None, None] * np.exp(
        -((yy - rows[:, None, None]) ** 2 + (xx - cols[:, None, None]) ** 2)
        / (2 * widths[:, None, None] ** 2)
    )
    chm = bumps.max(axis=0) + rng.normal(0.0, rng.uniform(0.0, 3.0), (nrows, ncols))
    if rng.random() < 0.3:
        chm = np.round(chm)  # many exact ties
    if rng.random() < 0.3:
        chm[rng.random((nrows, ncols)) < rng.uniform(0.0, 0.2)] = np.nan
    if rng.random() < 0.3:
        chm = chm.astype(np.float32)

    n_seeds = min(nrows * ncols, int(rng.integers(0, nrows * ncols // 3 + 2)))
    flat = rng.choice(nrows * ncols, n_seeds, replace=False)
    seed_rows, seed_cols = np.divmod(flat, ncols)
    seed_labels = rng.choice(np.arange(1, 3 * n_seeds + 2), n_seeds, replace=False)

    size = rng.uniform(0.25, 3.0)
    shear = rng.uniform(-0.5, 0.5) * size if rng.random() < 0.3 else 0.0
    min_height = float(rng.uniform(0.0, 8.0))
    params = dict(
        min_height=min_height,
        max_height=(
            np.inf if rng.random() < 0.5 else min_height + float(rng.uniform(0, 30))
        ),
        min_relative_height=float(rng.uniform(0.0, 0.99)),
        min_relative_crown_height=float(rng.uniform(0.0, 0.99)),
        max_crown_radius=float(rng.uniform(0.1, 15.0)),
        transform=rio.Affine(size, shear, 0.0, shear, -size * rng.uniform(0.5, 2), 0.0),
    )
    return chm, seed_rows, seed_cols, seed_labels.astype(np.int32), params


@pytest.mark.parametrize("seed", range(300))
def test_frontier_growth_matches_full_scan(seed):
    chm, seed_rows, seed_cols, seed_labels, params = _random_case(seed)
    np.testing.assert_array_equal(
        _segment_frontier(chm, seed_rows, seed_cols, seed_labels, **params),
        segment_full_scan(chm, seed_rows, seed_cols, seed_labels, **params),
    )
