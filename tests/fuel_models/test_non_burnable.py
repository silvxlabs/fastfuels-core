"""Tests for :mod:`fastfuels_core.fuel_models.non_burnable`.

What is pinned: each targeted cell takes the most frequent burnable code in
its 5x5 window (ties to the larger code), wide features fill inward over
repeated passes from original burnable cells and earlier fills, burnable and
nodata cells never change, and the loop stops when a pass fills nothing.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fastfuels_core.fuel_models import non_burnable
from fastfuels_core.fuel_models.non_burnable import (
    _most_frequent_burnable,
    fill_non_burnable,
)


class TestMostFrequentBurnable:
    """Unit tests for the per-window rule of _most_frequent_burnable."""

    def test_skips_non_burnable_in_ranking(self):
        """Non-burnable codes are skipped even when most frequent."""
        windows = np.array([[99] * 20 + [101] * 5], dtype=np.int16)
        codes, found = _most_frequent_burnable(windows, [99])
        assert found[0] and codes[0] == 101

    def test_tie_goes_to_larger_code(self):
        """Tied burnable codes resolve to the larger code."""
        windows = np.array([[101] * 12 + [99] + [102] * 12], dtype=np.int16)
        codes, found = _most_frequent_burnable(windows, [99])
        assert found[0] and codes[0] == 102

    def test_not_found_when_window_has_no_burnable(self):
        """A window of only excluded codes reports no fill."""
        windows = np.array([[91] * 13 + [99] * 12], dtype=np.int16)
        _, found = _most_frequent_burnable(windows, [91, 99])
        assert not found[0]

    def test_rows_are_independent(self):
        """Each row is ranked on its own."""
        windows = np.array(
            [[101] * 20 + [102] * 5, [102] * 20 + [101] * 5], dtype=np.int16
        )
        codes, found = _most_frequent_burnable(windows, [99])
        assert found.all()
        np.testing.assert_array_equal(codes, [101, 102])


def _reference_targeted_fill(grid, non_burnable_keys, nodata=None):
    """Cell-by-cell statement of the targeted fill, used as a test oracle."""
    excluded = set(non_burnable_keys) | ({nodata} if nodata is not None else set())
    out = grid.copy()
    h, w = grid.shape
    while True:
        updates = {}
        for r, c in zip(*np.nonzero(np.isin(out, non_burnable_keys))):
            counts = {}
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    v = int(out[min(max(r + dr, 0), h - 1), min(max(c + dc, 0), w - 1)])
                    if v not in excluded:
                        counts[v] = counts.get(v, 0) + 1
            if counts:
                updates[(r, c)] = max(counts, key=lambda v: (counts[v], v))
        if not updates:
            return out
        for (r, c), v in updates.items():
            out[r, c] = v


class TestFillNonBurnable:
    """Unit tests for fill_non_burnable."""

    def test_no_non_burnable_returns_copy(self):
        """Grid with no non-burnable codes is returned unchanged."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        result = fill_non_burnable(grid, [91, 93, 99])
        np.testing.assert_array_equal(result, grid)
        assert result is not grid  # Returns a copy

    def test_replaces_single_non_burnable_cell(self):
        """A single non-burnable cell surrounded by burnable is replaced."""
        grid = np.full((10, 10), 102, dtype=np.int16)
        grid[5, 5] = 99  # Single bare ground cell
        result = fill_non_burnable(grid, [99])
        assert result[5, 5] == 102
        # All other cells unchanged
        grid[5, 5] = 102
        np.testing.assert_array_equal(result, grid)

    def test_only_targeted_codes_removed(self):
        """Only the specified non-burnable codes are removed; others are kept."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[3, 3] = 91  # Urban — targeted
        grid[7, 7] = 98  # Water — not targeted
        result = fill_non_burnable(grid, [91])
        assert result[3, 3] == 101  # Urban replaced
        assert result[7, 7] == 98  # Water preserved

    def test_replaces_with_most_frequent_neighbor(self):
        """Non-burnable cells are replaced by the most frequent burnable neighbor."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        # Put a block of 102 on the right side
        grid[:, 7:] = 102
        # Put a non-burnable cell in the 102 region
        grid[5, 8] = 99
        result = fill_non_burnable(grid, [99])
        assert result[5, 8] == 102  # Replaced by dominant neighbor

    def test_preserves_non_targeted_non_burnable(self):
        """Non-burnable codes not in the target list remain untouched."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[2, 2] = 92  # Snow/ice
        grid[4, 4] = 93  # Agriculture
        result = fill_non_burnable(grid, [93])
        assert result[2, 2] == 92  # Snow/ice preserved
        assert result[4, 4] == 101  # Agriculture replaced

    def test_large_non_burnable_patch(self):
        """A large patch of non-burnable codes is fully replaced."""
        grid = np.full((20, 20), 101, dtype=np.int16)
        grid[8:12, 8:12] = 99  # 4x4 bare ground block
        result = fill_non_burnable(grid, [99])
        assert not np.any(np.isin(result, [99]))

    def test_nodata_is_never_a_fill_source(self):
        """Non-burnable cells bordering nodata fill from burnable neighbours only."""
        grid = np.full((10, 10), 101, dtype=np.int16)
        grid[:, :6] = -9999
        grid[:, 6:8] = 99
        result = fill_non_burnable(grid, [99], nodata=-9999)
        np.testing.assert_array_equal(result[:, 6:8], 101)
        np.testing.assert_array_equal(result[:, :6], -9999)

    @pytest.mark.parametrize("nodata_cols", [0, 3])
    def test_stops_when_no_fill_source_exists(self, nodata_cols):
        """Non-burnable cells with no burnable neighbour are left as-is, promptly."""
        grid = np.full((5, 5), 99, dtype=np.int16)
        grid[:, :nodata_cols] = -9999
        real_filter = non_burnable._most_frequent_burnable
        calls = []

        def capped_filter(*args, **kwargs):
            calls.append(1)
            assert len(calls) <= 1, "filter kept running without progress"
            return real_filter(*args, **kwargs)

        with patch.object(non_burnable, "_most_frequent_burnable", capped_filter):
            result = fill_non_burnable(grid, [99], nodata=-9999)
        np.testing.assert_array_equal(result, grid)

    def test_burnable_cells_never_change(self):
        """Only targeted cells are written; a lone burnable cell outvoted by
        its neighbours keeps its code and still counts as a donor."""
        grid = np.full((20, 20), 101, dtype=np.int16)
        grid[6:14, :] = 91  # 8-cell road: needs two passes
        grid[5, 10] = 102  # lone burnable cell beside the road
        result = fill_non_burnable(grid, [91])
        keep = grid != 91
        np.testing.assert_array_equal(result[keep], grid[keep])
        assert not np.any(result == 91)

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_cell_by_cell_reference(self, seed):
        """Matches a cell-by-cell statement of the fill on random grids,
        including multi-pass fills, nodata, and unreachable cells."""
        rng = np.random.default_rng(seed)
        shape = (40, 53)
        coarse = rng.choice([101, 102, 121, 165], size=(10, 14))
        grid = np.kron(coarse, np.ones((4, 4), dtype=int))[: shape[0], : shape[1]]
        grid = grid.astype(np.int16)
        grid[5:17, 8:22] = 91
        grid[25:30, 30:45] = 99
        grid[rng.random(shape) < 0.08] = 93
        grid[rng.random(shape) < 0.03] = 98
        grid[:, :3] = -9999
        grid[0:6, 0:6] = 99
        keys = [91, 93, 99]

        expected = _reference_targeted_fill(grid, keys, -9999)
        actual = fill_non_burnable(grid, keys, -9999)
        np.testing.assert_array_equal(actual, expected)
