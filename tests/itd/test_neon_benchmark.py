"""Regression benchmark: treetop detection on NeonTreeEvaluation crowns.

The fixture holds 187 1 m NEON CHM plots (40 m x 40 m) with hand-labelled
crown boxes; see ``data/README.md``.  A treetop matches a crown if its pixel
centre lies inside the crown's box; matching is one-to-one and maximises the
number of matches.

Two checks: an accuracy floor (F1 and treetops per crown), which the pre-#118
footprint fails (F1 0.517 and 0.506, 1.7x and 1.8x as many treetops as
crowns); and the exact treetop and match counts, which pin current behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_origin
from scipy.optimize import linear_sum_assignment

from fastfuels_core.itd.local_maxima_filter import (
    fixed_window_filter,
    variable_window_filter,
)

FIXTURE = Path(__file__).parent / "data" / "neon_tree_evaluation.npz"
MIN_HEIGHT = 3.0
PLOT_PIXELS = 40
# Zero-height gap between plots in the mosaic; wider than any window's
# half-width, so plots cannot see each other.
GAP_PIXELS = 10
F1_FLOOR = 0.55
DETECTION_RATIO_RANGE = (0.8, 1.2)
# Exact (treetops, matched) at MIN_HEIGHT.  Detection is deterministic, so any
# change here is a behaviour change: if it is intended, update these numbers.
EXPECTED_COUNTS = {"lmf_3px": (6600, 3656), "vwf_defaults": (5822, 3466)}


def load_fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


def mosaic(chm: np.ndarray) -> tuple[xr.DataArray, int]:
    """Tile the plots, row-major, into one CHM separated by zero gaps.

    Heights are non-negative, so a zero gap compares like the reflected
    boundary each plot would get on its own.
    """
    n_plots = len(chm)
    per_row = int(np.ceil(np.sqrt(n_plots)))
    tile = PLOT_PIXELS + GAP_PIXELS
    n_rows = int(np.ceil(n_plots / per_row))
    canvas = np.zeros((n_rows * tile, per_row * tile), dtype=np.float64)
    for i, plot in enumerate(chm):
        r, c = divmod(i, per_row)
        canvas[r * tile : r * tile + PLOT_PIXELS, c * tile : c * tile + PLOT_PIXELS] = (
            plot
        )
    chm_da = xr.DataArray(canvas, dims=["y", "x"])
    chm_da = chm_da.rio.write_crs("EPSG:32611")
    chm_da = chm_da.rio.write_transform(from_origin(0.0, canvas.shape[0], 1.0, 1.0))
    return chm_da, per_row


def count_matches(tops_xy: np.ndarray, boxes: np.ndarray) -> int:
    """One-to-one matches of treetops (x right, y down, metres) to boxes."""
    if len(tops_xy) == 0 or len(boxes) == 0:
        return 0
    x, y = tops_xy[:, :1], tops_xy[:, 1:]
    inside = (
        (x >= boxes[:, 0])
        & (x <= boxes[:, 2])
        & (y >= boxes[:, 1])
        & (y <= boxes[:, 3])
    )
    centre_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centre_y = (boxes[:, 1] + boxes[:, 3]) / 2
    distance = np.hypot(x - centre_x, y - centre_y)
    rows, cols = linear_sum_assignment(np.where(inside, distance, 1e6))
    return int(inside[rows, cols].sum())


def score(detect: Callable[[xr.DataArray], object], data: dict) -> dict[str, float]:
    """Pooled treetops, matches and F1 of ``detect`` over every plot."""
    chm_da, per_row = mosaic(data["chm"])
    treetops = detect(chm_da).compute()
    tile = PLOT_PIXELS + GAP_PIXELS
    # Mosaic metres from its top-left corner (y down).
    across = treetops["x"].to_numpy()
    down = chm_da.shape[0] - treetops["y"].to_numpy()
    plot_row, in_row = np.divmod(down, tile)
    plot_col, in_col = np.divmod(across, tile)
    plot = (plot_row * per_row + plot_col).astype(int)
    boxes = data["boxes_dm"].astype(np.float64) / 10.0

    n_crowns = len(boxes)
    n_treetops = len(treetops)
    matched = 0
    for i in range(len(data["chm"])):
        tops = np.c_[in_col[plot == i], in_row[plot == i]]
        matched += count_matches(tops, boxes[data["box_plot"] == i])
    return {
        "crowns": n_crowns,
        "treetops": n_treetops,
        "matched": matched,
        "f1": 2 * matched / (n_crowns + n_treetops),
    }


def detect_lmf(chm_da: xr.DataArray):
    return fixed_window_filter(chm_da, MIN_HEIGHT, 1.0, window_size_meters=3.0)


def detect_vwf(chm_da: xr.DataArray):
    return variable_window_filter(chm_da, MIN_HEIGHT, 1.0)


@pytest.fixture(scope="module")
def neon() -> dict[str, np.ndarray]:
    return load_fixture()


def test_fixture_shape(neon: dict[str, np.ndarray]):
    assert neon["chm"].shape == (187, PLOT_PIXELS, PLOT_PIXELS)
    assert len(neon["boxes_dm"]) == len(neon["box_plot"]) == 6200
    assert len(np.unique(neon["site"])) == 20
    assert (neon["chm"] >= 0).all()


DETECTORS = {"lmf_3px": detect_lmf, "vwf_defaults": detect_vwf}


@pytest.fixture(scope="module", params=list(DETECTORS))
def scored(request, neon: dict[str, np.ndarray]) -> tuple[str, dict[str, float]]:
    return request.param, score(DETECTORS[request.param], neon)


def test_detection_meets_benchmark_floors(scored):
    _, result = scored
    ratio = result["treetops"] / result["crowns"]
    assert result["f1"] >= F1_FLOOR, result
    assert DETECTION_RATIO_RANGE[0] <= ratio <= DETECTION_RATIO_RANGE[1], result


def test_detection_counts_are_unchanged(scored):
    name, result = scored
    assert (result["treetops"], result["matched"]) == EXPECTED_COUNTS[name], result
