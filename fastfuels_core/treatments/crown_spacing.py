# Core imports
from __future__ import annotations

# External imports
import numpy as np
from pandas import DataFrame
from scipy.sparse import coo_array
from scipy.spatial import cKDTree


class ThinToCrownSpacing:
    """
    Thinning treatment that leaves every retained crown at least
    ``spacing`` meters, edge to edge, from every other retained crown.

    Crowns are circles in plan view, centred on the stem with a radius
    read from a column, so two trees conflict when the distance between
    their stems, less both radii, is under ``spacing``. The class does
    no allometry: the caller fills the radius column, from a measured
    crown radius or from the same allometry the voxelizer uses.

    Trees are resolved greedily. Trees that may not be cut are kept
    first, then the eligible trees are visited from the largest
    ``rank_by`` value down, ties going to the tree that comes first in
    the frame, and each is kept only if it clears every tree already
    kept. The result is maximal: every eligible tree that was removed
    conflicts with one that was kept. There is no randomness.

    The ``eligible`` mask is what lets a treatment confined to a region
    respect the trees around it. Pass the whole neighbourhood, with the
    trees outside the region marked ineligible: they are never removed,
    and a tree inside the region is kept only if it clears them too.
    Two ineligible trees may be closer than ``spacing``; the treatment
    leaves them as they are.

    Parameters
    ----------
    spacing : float
        Minimum edge-to-edge distance between retained crowns, in
        meters. Must be positive.
    rank_by : str, optional
        Column that decides which tree survives a conflict: the larger
        value is kept. By default "dbh".

    Methods
    -------
    apply(trees, eligible=None, x_column="x", y_column="y",
          radius_column="crown_radius") -> DataFrame
        Apply the crown spacing thinning to the DataFrame of trees.
    """

    def __init__(self, spacing: float, rank_by: str = "dbh") -> None:
        if not (np.isfinite(spacing) and spacing > 0):
            raise ValueError(f"spacing must be a positive number; got {spacing}.")
        self.spacing = float(spacing)
        self.rank_by = rank_by

    def apply(
        self,
        trees: DataFrame,
        eligible: np.ndarray | None = None,
        x_column: str = "x",
        y_column: str = "y",
        radius_column: str = "crown_radius",
    ) -> DataFrame:
        """
        Apply the crown spacing thinning to the DataFrame of trees.

        Parameters
        ----------
        trees : DataFrame
            DataFrame containing tree data, with stem coordinates and
            crown radii in meters and the ``rank_by`` column.
        eligible : np.ndarray, optional
            Boolean mask, one entry per row of ``trees``, of the trees
            that may be removed. Ineligible trees are always kept. By
            default every tree is eligible.
        x_column, y_column : str, optional
            Names of the stem coordinate columns, by default "x" and
            "y".
        radius_column : str, optional
            Name of the crown radius column, by default "crown_radius".

        Returns
        -------
        DataFrame
            The retained trees, in their original order.

        Raises
        ------
        ValueError
            If ``eligible`` does not match ``trees``; if a coordinate or
            radius is missing, or a radius is negative; or if an
            eligible tree has no ``rank_by`` value.
        """
        df = trees.copy()
        n = len(df)

        if eligible is None:
            eligible = np.ones(n, dtype=bool)
        else:
            eligible = np.asarray(eligible)
            if eligible.dtype != bool or eligible.shape != (n,):
                raise ValueError(
                    f"eligible must be a boolean array with one entry per tree "
                    f"({n}); got dtype {eligible.dtype} and shape {eligible.shape}."
                )

        if not eligible.any():
            return df

        xy = df[[x_column, y_column]].to_numpy(dtype=float, na_value=np.nan)
        radius = df[radius_column].to_numpy(dtype=float, na_value=np.nan)
        rank = df[self.rank_by].to_numpy(dtype=float, na_value=np.nan)

        if np.isnan(xy).any():
            raise ValueError(
                f"Every tree needs a position; {x_column!r} or {y_column!r} "
                f"is missing on {np.isnan(xy).any(axis=1).sum()} trees."
            )
        if np.isnan(radius).any():
            raise ValueError(
                f"Every tree needs a crown radius; {radius_column!r} is "
                f"missing on {np.isnan(radius).sum()} trees."
            )
        if (radius < 0).any():
            raise ValueError(
                f"Crown radii must be non-negative; {radius_column!r} is "
                f"negative on {(radius < 0).sum()} trees."
            )
        if np.isnan(rank[eligible]).any():
            raise ValueError(
                f"Every tree eligible for removal needs a {self.rank_by!r} "
                f"value to rank it by; {np.isnan(rank[eligible]).sum()} "
                f"eligible trees have none."
            )

        # Every conflicting pair is within spacing + 2 * max(radius) stem
        # to stem; keep only the pairs whose crowns are actually too close.
        pairs = cKDTree(xy).query_pairs(
            self.spacing + 2 * radius.max(), output_type="ndarray"
        )
        i, j = pairs[:, 0], pairs[:, 1]
        gap = np.hypot(*(xy[i] - xy[j]).T) - radius[i] - radius[j]
        conflict = gap < self.spacing
        i, j = i[conflict], j[conflict]

        # The order trees are settled in: ineligible trees first, then
        # eligible trees from the largest rank down, ties in frame order.
        order = np.argsort(np.where(eligible, -rank, -np.inf), kind="stable")
        priority = np.empty(n, dtype=np.intp)
        priority[order] = np.arange(n)

        # A tree is kept only if no tree settled before it, and kept,
        # conflicts with it, so each tree needs only its conflicts with
        # trees that come before it. Ineligible trees need none.
        later = np.where(priority[i] > priority[j], i, j)
        earlier = i + j - later
        mine = eligible[later]
        earlier_conflicts = coo_array(
            (np.ones(mine.sum(), dtype=bool), (later[mine], earlier[mine])),
            shape=(n, n),
        ).tocsr()
        start, neighbours = earlier_conflicts.indptr, earlier_conflicts.indices

        # Ineligible trees are kept, and so are eligible trees with no
        # earlier conflict. The rest are settled in order.
        keep = ~eligible | (start[1:] == start[:-1])
        for k in order[~keep[order]]:
            if not keep[neighbours[start[k] : start[k + 1]]].any():
                keep[k] = True

        df = df[keep]

        assert isinstance(df, DataFrame), "Resulting object is not a DataFrame"

        return df
