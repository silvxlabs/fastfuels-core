"""Fill non-burnable cells of a fuel model grid from their burnable neighbours.

Removes small non-burnable features (roads, water, bare ground) from a
categorical fuel model grid such as LANDFIRE FBFM13, FBFM40 or FCCS.
"""

from __future__ import annotations

import numpy as np


def remove_non_burnable(
    grid: np.ndarray,
    non_burnable_keys: list[int],
    nodata: float | None = None,
) -> np.ndarray:
    """Replace non-burnable codes with the most frequent burnable neighbour.

    Each targeted cell takes the most frequent burnable code in its 5x5
    window (edges extended by nearest value; ties go to the larger code).
    Wide features fill from the edges inward over repeated passes. Each pass
    reads the original burnable cells plus cells filled by earlier passes;
    burnable cells are never changed. Cells with no reachable burnable
    neighbour keep their code.

    Parameters
    ----------
    grid : numpy.ndarray
        2D grid of fuel model codes.
    non_burnable_keys : list[int]
        Codes to replace, e.g. ``[91, 93, 99]`` for FBFM40 NB1, NB3, NB9.
    nodata : float, optional
        Nodata value. Never used as a fill value and left untouched.

    Returns
    -------
    numpy.ndarray
        Copy of ``grid`` with the targeted codes replaced.
    """
    output = grid.copy()
    excluded = list(non_burnable_keys)
    if nodata is not None:
        excluded.append(nodata)

    rows, cols = np.nonzero(np.isin(grid, non_burnable_keys))
    offsets = np.arange(-2, 3)
    while rows.size:
        window_rows = np.clip(rows[:, None] + offsets, 0, grid.shape[0] - 1)
        window_cols = np.clip(cols[:, None] + offsets, 0, grid.shape[1] - 1)
        windows = output[window_rows[:, :, None], window_cols[:, None, :]]
        fill, found = _most_frequent_burnable(windows.reshape(rows.size, -1), excluded)
        if not found.any():
            break
        output[rows[found], cols[found]] = fill[found]
        rows, cols = rows[~found], cols[~found]

    return output


def _most_frequent_burnable(
    windows: np.ndarray, excluded: list
) -> tuple[np.ndarray, np.ndarray]:
    """Most frequent non-excluded code in each row of ``windows``.

    Ties go to the larger code. Returns the codes and a mask of rows that
    held any non-excluded code; codes for the other rows are meaningless.
    """
    best_code = np.zeros(windows.shape[0], dtype=windows.dtype)
    best_count = np.zeros(windows.shape[0], dtype=np.int64)
    candidates = np.unique(windows[~np.isin(windows, excluded)])
    # Ascending so that ">=" hands ties to the larger code.
    for code in candidates:
        count = np.count_nonzero(windows == code, axis=1)
        better = (count > 0) & (count >= best_count)
        best_count[better] = count[better]
        best_code[better] = code
    return best_code, best_count > 0
