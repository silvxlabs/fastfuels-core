"""Tests for :class:`fastfuels_core.treatments.ThinToClumpPattern`.

The stands are synthetic Thomas processes: clusters of trees around
Poisson parents, the clustered pattern ICO treatments start from. They
are small and the solver's tiles are patched down to 100 m, so a stand
spans several tiles and passes. Tests that check only mechanics also
patch the schedule down to a few sweeps.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from fastfuels_core.treatments import ThinToClumpPattern

# numba is optional (fastfuels-core[numba]); without it these tests skip.
numba = pytest.importorskip("numba")
from fastfuels_core.treatments import clump_pattern_solver  # noqa: E402

TEST_DATA_PATH = Path(__file__).parent.parent / "data"

#: Shares of the kept trees in clumps of 1, 2-4, 5-9, 10-15 and 16+.
SHARES = (0.15, 0.25, 0.25, 0.15, 0.20)


def thomas_stand(side, kappa=0.004, mu=10, sigma=5.0, seed=0) -> pd.DataFrame:
    """Thomas process on a side x side box: Poisson parents at kappa per
    m2, Poisson(mu) offspring each, displaced N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    n_parents = rng.poisson(kappa * side * side)
    px, py = rng.uniform(0, side, n_parents), rng.uniform(0, side, n_parents)
    k = rng.poisson(mu, n_parents)
    x = np.repeat(px, k) + rng.normal(0, sigma, k.sum())
    y = np.repeat(py, k) + rng.normal(0, sigma, k.sum())
    inside = (x >= 0) & (x < side) & (y >= 0) & (y < side)
    x, y = x[inside], y[inside]
    n = len(x)
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "dbh": np.clip(rng.lognormal(np.log(25), 0.5, n), 5, 120),
            "crown_ratio": rng.uniform(0.2, 0.8, n),
        }
    )


def box(side) -> shapely.Polygon:
    return shapely.box(0, 0, side, side)


def clump_sizes(df, distance=6.0) -> tuple[np.ndarray, np.ndarray]:
    """Clump label of each tree and size of each clump, by single linkage."""
    xy = df[["x", "y"]].to_numpy()
    pairs = cKDTree(xy).query_pairs(distance, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy))
    )
    _, labels = connected_components(graph, directed=False)
    return labels, np.bincount(labels)


def shares_of(df, distance=6.0) -> np.ndarray:
    labels, sizes = clump_sizes(df, distance)
    size = sizes[labels]
    bins = np.digitize(size, [2, 5, 10, 16])
    return np.bincount(bins, minlength=5) / len(df)


def open_space_of(kept, trees, side) -> float:
    """Share of the 3 m grid, over the 30 m cells that hold a tree, more
    than 9 m from a kept tree."""
    centres = (np.arange(int(np.ceil(side / 3))) + 0.5) * 3
    cx, cy = (a.ravel() for a in np.meshgrid(centres, centres, indexing="ij"))
    occupied = set(zip(trees["x"] // 30, trees["y"] // 30))
    region = np.array([c in occupied for c in zip(cx // 30, cy // 30)])
    d, _ = cKDTree(kept[["x", "y"]].to_numpy()).query(
        np.column_stack([cx[region], cy[region]])
    )
    return float((d > 9.0).mean())


@pytest.fixture
def small_tiles(monkeypatch):
    monkeypatch.setattr(clump_pattern_solver, "TILE_SIZE", 100.0)


@pytest.fixture
def quick(monkeypatch, small_tiles):
    """For tests of mechanics only: a few sweeps."""
    monkeypatch.setattr(clump_pattern_solver, "SWEEPS", 5)


def apply_quietly(treatment, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return treatment.apply(*args, **kwargs)


class TestTargets:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_density_shares_and_open_space_are_met(self, small_tiles, seed):
        df = thomas_stand(300, seed=seed)
        treatment = ThinToClumpPattern(100, SHARES)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = treatment.apply(df, box(300))
        report = treatment.report
        assert report["all_met"]

        # Checked from the kept trees, not just the report.
        assert report["stand_area_ha"] == pytest.approx(9.0)  # every cell treed
        assert len(out) / 9.0 == pytest.approx(100, rel=0.02)
        assert np.abs(shares_of(out) - SHARES).max() <= 0.02
        assert report["shares"]["achieved"] == pytest.approx(shares_of(out))
        got = open_space_of(out, df, 300)
        assert report["open_space"]["achieved"] == pytest.approx(got)
        assert abs(got - report["open_space"]["target"]) <= 0.02
        assert not report["open_space"]["infeasible"]

    def test_the_open_space_target_comes_from_the_reference_model(self):
        mean_size = 1 / sum(s / m for s, m in zip(SHARES, (1, 3, 7, 12, 17)))
        expected = 1.841 + 0.144 * np.log(mean_size) - 0.396 * np.log(100)
        assert clump_pattern_solver.reference_open_space(SHARES, 100) == pytest.approx(
            expected
        )
        assert clump_pattern_solver.reference_open_space(SHARES, 1e6) == 0.0

    def test_a_basal_area_target(self, small_tiles):
        df = thomas_stand(300)
        treatment = ThinToClumpPattern(10.0, SHARES, density_metric="ba")
        out = apply_quietly(treatment, df, box(300))
        ba = (np.pi * (out["dbh"] / 200) ** 2).sum() / 9.0
        assert ba == pytest.approx(10.0, rel=0.02)
        assert treatment.report["density"]["achieved"] == pytest.approx(ba)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"density": 100, "shares": SHARES},
            {"density": 300, "shares": (0.1, 0.1, 0.1, 0.2, 0.5), "open_space": None},
        ],
        ids=["guide-like", "dense, half in 16+"],
    )
    def test_the_open_top_bin_is_not_one_giant_clump(self, small_tiles, kwargs):
        """Clumps over 15 trees all count toward 16+, so nothing in the
        bins stops one clump from taking the whole share. On the dense
        stand the largest clump does reach 100-odd trees, but the share
        stays spread over many clumps."""
        dense = kwargs["density"] > 200
        df = thomas_stand(300, kappa=0.008 if dense else 0.004, mu=12 if dense else 10)
        treatment = ThinToClumpPattern(**kwargs)
        out = apply_quietly(treatment, df, box(300))
        report = treatment.report
        assert report["shares"]["met"][4]
        _, sizes = clump_sizes(out)
        big = sizes[sizes >= 16]
        assert report["clumps"]["largest"] == sizes.max()
        assert len(big) >= 8
        assert sizes.max() <= 0.2 * big.sum()

    def test_infeasible_shares_give_way_before_density(self):
        """No two trees on an 8 m grid are within 6 m, so every kept tree
        is a single: the shares can't be met, but density still is."""
        g = np.arange(4.0, 200.0, 8.0)
        gx, gy = np.meshgrid(g, g)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": gx.ravel(),
                "y": gy.ravel(),
                "dbh": rng.uniform(15, 50, gx.size),
                "crown_ratio": rng.uniform(0.2, 0.8, gx.size),
            }
        )
        treatment = ThinToClumpPattern(50, [0.2] * 5, open_space=None)
        with pytest.warns(UserWarning, match="share of bin 2-4"):
            out = treatment.apply(df, box(200))
        assert len(out) / 4.0 == pytest.approx(50, rel=0.02)
        assert treatment.report["density"]["met"]
        assert treatment.report["shares"]["achieved"][0] == 1.0
        assert not treatment.report["all_met"]

    def test_better_leave_trees_are_kept(self, small_tiles):
        df = thomas_stand(300)
        treatment = ThinToClumpPattern(100, SHARES)
        apply_quietly(treatment, df, box(300))
        score = treatment.report["mean_score"]
        assert score["kept"] > score["candidates"] + 0.05


class TestOpenSpace:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_basal_area_mode_meets_the_stand_level_band(self, small_tiles, seed):
        """
        Diameters grow from east to west, so tiles meet the same basal
        area with different numbers of trees. The reference model is a
        stand-level relationship, convex in TPH: evaluated at each tile's
        own TPH, as the prototype did, its targets average above the
        stand's, and this stand ended 3.6-4.3 points above the
        stand-level target on these seeds. One stand-level target lands
        inside the band.
        """
        df = thomas_stand(400, seed=3)
        df["dbh"] = np.clip(df["dbh"] * 1.2 ** (1 - df["x"] / 200), 5, 150)
        treatment = ThinToClumpPattern(14.0, SHARES, density_metric="ba", seed=seed)
        out = apply_quietly(treatment, df, box(400))

        tph = out.groupby([out["x"] // 100, out["y"] // 100]).size()
        assert tph.std() / tph.mean() > 0.1
        report = treatment.report
        assert report["density"]["met"]
        # The target is the model at the stand's TPH (within the density
        # tolerance, ln 1.02 * 0.396 < 0.01).
        stand_tph = len(out) / report["stand_area_ha"]
        expected = clump_pattern_solver.reference_open_space(SHARES, stand_tph)
        assert report["open_space"]["target"] == pytest.approx(expected, abs=0.01)
        assert report["open_space"]["met"]

    def test_an_infeasible_target_is_flagged(self, small_tiles):
        """The clusters leave open space even when every tree is kept."""
        df = thomas_stand(300)
        treatment = ThinToClumpPattern(100, SHARES, open_space=0.0)
        with pytest.warns(UserWarning, match="infeasible"):
            treatment.apply(df, box(300))
        report = treatment.report["open_space"]
        assert report["infeasible"]
        assert not report["met"]
        candidates = df[df["dbh"] >= 12.7]
        assert report["floor"] == pytest.approx(open_space_of(candidates, df, 300))
        assert report["floor"] > 0.02
        assert report["achieved"] >= report["floor"]

    def test_a_fixed_target(self, small_tiles):
        df = thomas_stand(300)
        treatment = ThinToClumpPattern(100, SHARES, open_space=0.3)
        apply_quietly(treatment, df, box(300))
        report = treatment.report["open_space"]
        assert report["mode"] == "fixed"
        assert report["target"] == 0.3
        assert report["met"]

    def test_it_can_be_turned_off(self, quick):
        treatment = ThinToClumpPattern(100, SHARES, open_space=None)
        apply_quietly(treatment, thomas_stand(200), box(200))
        assert treatment.report["open_space"] is None


class TestEligibility:
    def test_keep_and_ineligible_trees_are_untouched(self, quick):
        """Dead trees from the TreeMap fixture are left as snags by
        marking them ineligible; so are the trees east of x = 120."""
        df = thomas_stand(200)
        treemap = pd.read_parquet(
            TEST_DATA_PATH / "tree_collection_from_treemap_2016.parquet"
        )
        rows = np.random.default_rng(0).integers(0, len(treemap), len(df))
        df["dbh"] = treemap["DIA"].to_numpy()[rows]
        df["crown_ratio"] = treemap["CR"].to_numpy()[rows]
        df["live"] = treemap["STATUSCD"].to_numpy()[rows] == 1
        eligible = (df["live"] & (df["x"] < 120)).to_numpy()
        keep = eligible & (df["dbh"] > 45).to_numpy()
        keep[np.flatnonzero(~eligible)[:5]] = True  # no effect on ineligible trees
        assert (~df["live"]).any() and keep.sum() > 5

        treatment = ThinToClumpPattern(80, SHARES)
        out = apply_quietly(treatment, df, box(200), eligible=eligible, keep=keep)

        pd.testing.assert_frame_equal(
            out[~eligible[df.index.isin(out.index)]], df[~eligible]
        )
        assert df.index[keep].isin(out.index).all()
        small = eligible & ~keep & (df["dbh"] < 12.7).to_numpy()
        assert small.any() and not df.index[small].isin(out.index).any()
        trees = treatment.report["trees"]
        assert trees["ineligible"] == (~eligible).sum()
        assert trees["keep"] == (keep & eligible).sum()

    def test_keep_trees_count_toward_density(self, small_tiles):
        df = thomas_stand(200)
        keep = (df["dbh"] > 40).to_numpy()
        treatment = ThinToClumpPattern(100, SHARES, open_space=None)
        out = apply_quietly(treatment, df, box(200), keep=keep)
        assert keep.sum() > 50
        ha = treatment.report["stand_area_ha"]  # 3.91: one 30 m cell is treeless
        assert len(out) / ha == pytest.approx(100, rel=0.02)

    def test_min_dbh(self, quick):
        df = thomas_stand(200)
        out = apply_quietly(ThinToClumpPattern(100, SHARES, min_dbh=30), df, box(200))
        assert out["dbh"].min() >= 30

    def test_eligible_trees_outside_the_polygon_are_left_alone(self, small_tiles):
        df = thomas_stand(200)
        treatment = ThinToClumpPattern(100, SHARES)
        out = apply_quietly(treatment, df, shapely.box(0, 0, 100, 200))
        outside = (df["x"] > 100).to_numpy()
        assert df.index[outside].isin(out.index).all()
        assert treatment.report["trees"]["outside_stand"] == outside.sum()
        ha = treatment.report["stand_area_ha"]
        assert (len(out) - outside.sum()) / ha == pytest.approx(100, rel=0.02)


class TestTiles:
    def test_no_clump_crosses_a_tile_edge(self, quick):
        df = thomas_stand(300, kappa=0.006)
        treatment = ThinToClumpPattern(200, SHARES)
        out = apply_quietly(treatment, df, box(300))
        labels, _ = clump_sizes(out)
        tile = (out["x"] // 100).astype(int) * 3 + (out["y"] // 100).astype(int)
        tiles_per_clump = pd.Series(tile.to_numpy()).groupby(labels).nunique()
        assert (tiles_per_clump == 1).all()
        assert treatment.report["clumps"]["crossing_tiles"] == 0
        # Clumps do reach the edges: they are blocked only by earlier ones.
        near_edge = (out["x"] % 100 < 3) | (out["x"] % 100 > 97)
        assert near_edge.any()

    @pytest.mark.skipif(
        numba.config.NUMBA_NUM_THREADS < 2, reason="needs two numba threads"
    )
    def test_the_result_does_not_depend_on_the_thread_count(self, quick):
        df = thomas_stand(300)
        treatment = ThinToClumpPattern(100, SHARES, seed=3)
        before = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            one = apply_quietly(treatment, df, box(300))
            numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
            many = apply_quietly(treatment, df, box(300))
        finally:
            numba.set_num_threads(before)
        pd.testing.assert_frame_equal(one, many)

    def test_the_seed(self, quick):
        df = thomas_stand(200)
        a = apply_quietly(ThinToClumpPattern(100, SHARES, seed=1), df, box(200))
        b = apply_quietly(ThinToClumpPattern(100, SHARES, seed=1), df, box(200))
        c = apply_quietly(ThinToClumpPattern(100, SHARES, seed=2), df, box(200))
        pd.testing.assert_frame_equal(a, b)
        assert not a.index.equals(c.index)


class TestStandArea:
    def uniform(self, n, x1, y1, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "x": rng.uniform(0, x1, n),
                "y": rng.uniform(0, y1, n),
                "dbh": rng.uniform(15, 50, n),
                "crown_ratio": rng.uniform(0.2, 0.8, n),
            }
        )

    @pytest.mark.parametrize(
        "polygon",
        [shapely.box(0, 0, 200, 200), shapely.Point(100, 100).buffer(100)],
        ids=["box", "disk"],
    )
    def test_a_polygon_is_measured_from_its_geometry(self, quick, polygon):
        """Counting cell centres gave 3.8 ha for a 4 ha box."""
        df = self.uniform(4000, 200, 200)
        # A tree in every 30 m cell's piece of the polygon, so every cell
        # it touches is treed.
        corners = np.arange(0, 200, 30.0)
        cells = [shapely.box(i, j, i + 30, j + 30) for i in corners for j in corners]
        pieces = shapely.intersection(cells, polygon)
        points = shapely.point_on_surface(pieces[~shapely.is_empty(pieces)])
        extra = df.iloc[: len(points)].copy()
        extra["x"], extra["y"] = shapely.get_x(points), shapely.get_y(points)
        df = pd.concat([df, extra], ignore_index=True)

        treatment = ThinToClumpPattern(100, SHARES)
        apply_quietly(treatment, df, polygon)
        assert treatment.report["stand_area_ha"] == pytest.approx(
            polygon.area / 1e4, rel=1e-9
        )

    def test_only_treed_cells_count(self, small_tiles):
        df = self.uniform(2000, 90, 200)
        treatment = ThinToClumpPattern(100, SHARES)
        out = apply_quietly(treatment, df, shapely.box(0, 0, 200, 200))
        assert treatment.report["stand_area_ha"] == pytest.approx(1.8)
        assert len(out) / 1.8 == pytest.approx(100, rel=0.02)

    def test_an_area_is_used_as_given(self, small_tiles):
        df = self.uniform(2000, 200, 200)
        treatment = ThinToClumpPattern(50, SHARES)
        out = apply_quietly(treatment, df, 10.0)
        assert treatment.report["stand_area_ha"] == 10.0
        assert len(out) == pytest.approx(500, rel=0.02)

    def test_a_tree_on_the_far_edge_of_the_polygon(self, quick):
        df = self.uniform(1, 1, 1)
        df[["x", "y"]] = [30.0, 15.0]
        treatment = ThinToClumpPattern(100, SHARES)
        out = apply_quietly(treatment, df, shapely.box(0, 0, 30, 30))
        assert treatment.report["stand_area_ha"] == pytest.approx(0.09)
        assert len(out) == 1


class TestMaxClumpSize:
    DENSE = dict(density=300, shares=(0.1, 0.1, 0.1, 0.2, 0.5), open_space=None)

    @pytest.mark.parametrize("cap", [25, 40])
    def test_a_cap_holds_on_the_dense_stand(self, small_tiles, cap):
        """Uncapped, this stand's largest clump is 106 trees."""
        df = thomas_stand(300, kappa=0.008, mu=12)
        treatment = ThinToClumpPattern(**self.DENSE, max_clump_size=cap)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = treatment.apply(df, box(300))
        _, sizes = clump_sizes(out)
        assert sizes.max() <= cap
        report = treatment.report
        assert report["clumps"]["largest"] == sizes.max()
        assert report["clumps"]["max_size"] == cap
        assert report["clumps"]["max_size_met"]
        assert report["density"]["met"] and all(report["shares"]["met"])

    def test_a_cap_inside_a_bin(self, small_tiles):
        """With 12, the 10-15 bin holds clumps of 10-12."""
        shares = (0.2, 0.3, 0.3, 0.2, 0.0)
        treatment = ThinToClumpPattern(100, shares, max_clump_size=12)
        out = apply_quietly(treatment, thomas_stand(300), box(300))
        _, sizes = clump_sizes(out)
        assert sizes.max() <= 12
        assert all(treatment.report["shares"]["met"])

    def test_a_bin_above_the_cap_needs_a_share_of_0(self):
        with pytest.raises(ValueError, match="max_clump_size=12.*16\\+ bin"):
            ThinToClumpPattern(100, SHARES, max_clump_size=12)

    @pytest.mark.parametrize("bad", [0, -3, 2.5, True, "25"])
    def test_a_bad_cap_raises(self, bad):
        with pytest.raises(ValueError, match="max_clump_size"):
            ThinToClumpPattern(100, (1.0, 0, 0, 0, 0), max_clump_size=bad)

    def test_keep_trees_can_break_the_cap(self, quick):
        """Thirty keep trees 2 m apart are one clump the solver can't cut."""
        df = thomas_stand(200)
        row = pd.DataFrame(
            {"x": 5 + 2.0 * np.arange(30), "y": 150.0, "dbh": 40.0, "crown_ratio": 0.5}
        )
        df = pd.concat([df, row], ignore_index=True)
        keep = np.zeros(len(df), bool)
        keep[-30:] = True
        treatment = ThinToClumpPattern(100, SHARES, max_clump_size=25)
        with pytest.warns(UserWarning, match="cap of 25"):
            treatment.apply(df, box(200), keep=keep)
        assert treatment.report["clumps"]["max_size_met"] is False
        assert not treatment.report["all_met"]

    def test_no_cap_matches_a_cap_no_clump_can_reach(self, quick):
        """
        Without a cap the over-cap class stays empty, so the result is the
        same as with a cap larger than the stand: the cap changes nothing
        until a clump could exceed it.
        """
        df = thomas_stand(200)
        uncapped = ThinToClumpPattern(100, SHARES, max_clump_size=None)
        kept = apply_quietly(uncapped, df, box(200)).index
        capped = ThinToClumpPattern(100, SHARES, max_clump_size=len(df))
        assert apply_quietly(capped, df, box(200)).index.equals(kept)
        assert uncapped.report["clumps"]["max_size"] is None
        assert uncapped.report["clumps"]["max_size_met"] is None


class TestFrameHandling:
    def test_an_empty_stand_is_returned_unchanged(self):
        df = thomas_stand(100).iloc[:0]
        treatment = ThinToClumpPattern(100, SHARES)
        pd.testing.assert_frame_equal(treatment.apply(df, 1.0), df)
        assert treatment.report is None

    def test_no_eligible_tree_is_returned_unchanged(self):
        df = thomas_stand(100)
        eligible = np.zeros(len(df), bool)
        treatment = ThinToClumpPattern(100, SHARES)
        out = treatment.apply(df, 1.0, eligible)
        pd.testing.assert_frame_equal(out, df)
        assert out is not df
        assert treatment.report is None

    def test_all_keep_or_ineligible_is_returned_unchanged(self, quick):
        df = thomas_stand(200)
        eligible = (df["x"] < 100).to_numpy()
        treatment = ThinToClumpPattern(100, SHARES)
        out = apply_quietly(treatment, df, box(200), eligible=eligible, keep=eligible)
        pd.testing.assert_frame_equal(out, df)
        assert treatment.report["trees"]["kept"] == eligible.sum()

    def test_trees_below_min_dbh_are_cut_even_with_no_leave_trees(self, quick):
        df = thomas_stand(100)
        df["dbh"] = 10.0
        treatment = ThinToClumpPattern(100, SHARES)
        with pytest.warns(UserWarning, match="density"):
            out = treatment.apply(df, box(100))
        assert out.empty
        assert treatment.report["trees"]["candidates"] == 0

    def test_the_input_is_not_modified(self, quick):
        df = thomas_stand(200)
        before = df.copy()
        apply_quietly(ThinToClumpPattern(100, SHARES), df, box(200))
        pd.testing.assert_frame_equal(df, before)

    def test_survivors_keep_their_order_columns_and_index(self, quick):
        df = thomas_stand(200)
        df.index = df.index * 7 + 3
        df["species"] = "PIPO"
        out = apply_quietly(ThinToClumpPattern(100, SHARES), df, box(200))
        assert out.index.is_monotonic_increasing
        assert out.index.isin(df.index).all()
        assert list(out.columns) == list(df.columns)

    def test_other_column_names(self, small_tiles):
        df = thomas_stand(200).rename(
            columns={"x": "X", "y": "Y", "dbh": "DIA", "crown_ratio": "CR"}
        )
        treatment = ThinToClumpPattern(100, SHARES)
        out = apply_quietly(
            treatment,
            df,
            box(200),
            x_column="X",
            y_column="Y",
            dia_column="DIA",
            crown_ratio_column="CR",
        )
        ha = treatment.report["stand_area_ha"]
        assert len(out) / ha == pytest.approx(100, rel=0.02)


class TestValidation:
    @pytest.mark.parametrize(
        "shares, match",
        [
            ((0.5, 0.5), "five entries"),
            ((0.2, 0.2, 0.2, 0.2, 0.1), "sum to 1"),
            ((1.2, -0.2, 0.0, 0.0, 0.0), r"\[0, 1\]"),
            ((np.nan, 0.25, 0.25, 0.25, 0.25), r"\[0, 1\]"),
            ("abcde", "five numbers"),
        ],
    )
    def test_bad_shares_raise(self, shares, match):
        with pytest.raises(ValueError, match=match):
            ThinToClumpPattern(100, shares)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"density": 0},
            {"density": -5},
            {"density": np.nan},
            {"density": "100"},
            {"density_metric": "sdi"},
            {"open_space": "model"},
            {"open_space": 1.5},
            {"open_space": -0.1},
            {"clump_distance": 0},
            {"clump_compactness": -1},
            {"density_tolerance": -0.01},
            {"share_tolerance": np.inf},
            {"open_space_tolerance": -1},
            {"seed": -1},
            {"seed": 1.5},
            {"min_dbh": -1},
        ],
    )
    def test_bad_parameters_raise(self, kwargs):
        args = {"density": 100, "shares": SHARES} | kwargs
        name = next(iter(kwargs))
        with pytest.raises(ValueError, match=name):
            ThinToClumpPattern(**args)

    @pytest.mark.parametrize(
        "column, value", [("x", np.nan), ("dbh", np.nan), ("crown_ratio", np.nan)]
    )
    def test_a_missing_value_on_an_eligible_tree_raises(self, column, value):
        df = thomas_stand(100)
        df.loc[3, column] = value
        with pytest.raises(ValueError, match=column):
            ThinToClumpPattern(100, SHARES).apply(df, 1.0)

    def test_a_missing_value_on_an_ineligible_tree_is_fine(self, quick):
        df = thomas_stand(100)
        df.loc[3, ["x", "dbh", "crown_ratio"]] = np.nan
        eligible = np.ones(len(df), bool)
        eligible[3] = False
        out = apply_quietly(ThinToClumpPattern(100, SHARES), df, 1.0, eligible)
        assert 3 in out.index

    @pytest.mark.parametrize("column", ["x", "dbh", "crown_ratio"])
    def test_a_missing_column_raises(self, column):
        df = thomas_stand(100).drop(columns=column)
        with pytest.raises(ValueError, match=column):
            ThinToClumpPattern(100, SHARES).apply(df, 1.0)

    @pytest.mark.parametrize(
        "column, value, match",
        [("dbh", -1.0, "non-negative"), ("crown_ratio", 45.0, r"\[0, 1\]")],
    )
    def test_out_of_range_values_raise(self, column, value, match):
        df = thomas_stand(100)
        df.loc[3, column] = value
        with pytest.raises(ValueError, match=match):
            ThinToClumpPattern(100, SHARES).apply(df, 1.0)

    @pytest.mark.parametrize("name", ["eligible", "keep"])
    @pytest.mark.parametrize("bad", [np.array([1, 0]), np.array([True]), [True]])
    def test_a_malformed_mask_raises(self, name, bad):
        df = thomas_stand(100)
        with pytest.raises(ValueError, match=name):
            ThinToClumpPattern(100, SHARES).apply(df, 1.0, **{name: bad})

    @pytest.mark.parametrize(
        "area", [0.0, -1.0, np.nan, "4 ha", None, shapely.Polygon()]
    )
    def test_a_bad_stand_area_raises(self, area):
        with pytest.raises(ValueError, match="stand_area"):
            ThinToClumpPattern(100, SHARES).apply(thomas_stand(100), area)
