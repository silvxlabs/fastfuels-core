# NeonTreeEvaluation fixture

`neon_tree_evaluation.npz` is a subset of the NeonTreeEvaluation benchmark
data, used by `tests/itd/test_neon_benchmark.py`.

**Source.** Weinstein, B. G., Graves, S. J., Marconi, S., Singh, A., Zare, A.,
Stewart, D., Bohlman, S. A., & White, E. P. (2021). A benchmark dataset for
canopy crown detection and delineation in co-registered airborne RGB, LiDAR
and hyperspectral imagery from the National Ecological Observation Network.
*PLOS Computational Biology* 17(7): e1009180.
https://doi.org/10.1371/journal.pcbi.1009180

Data: Weinstein, B., Marconi, S., & White, E. (2022). Data for the
NeonTreeEvaluation Benchmark (Version 0.2.2) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5914554. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**What was extracted.** From `evaluation.zip`: the hand-labelled crown boxes
in `evaluation/RGB/benchmark_annotations.csv`, and for each annotated plot its
1 m canopy height model, `evaluation/CHM/<plot>_CHM.tif`. 189 annotated plots
have a CHM. Two of them, `SJER_062_2018` (40 x 3 pixels) and `TALL_043_2019`
(40 x 1), are truncated clips and were dropped, leaving 187 plots and 6,200
crowns from 20 NEON sites.

**Changes.** CHM nodata and negative heights are set to 0, and heights are
stored as float32. Nothing else is changed.

**Arrays.**

| name | shape, type | content |
|---|---|---|
| `chm` | (187, 40, 40) float32 | heights (m); row 0 is the plot's north edge |
| `boxes_dm` | (6200, 4) int16 | `xmin, ymin, xmax, ymax` in decimetres from the plot's top-left (north-west) corner, `y` pointing south |
| `box_plot` | (6200,) int16 | index into `plot` of each box |
| `plot` | (187,) str | plot name, as in the source files |
| `site` | (187,) str | four-letter NEON site code |

The boxes were drawn on 0.1 m RGB images of the same 40 m plots, so their
pixel coordinates are decimetres. RGB and CHM corners differ by up to
0.5 m, so F1 differences of about 0.003 at 1 m are within alignment noise.

**Rebuilding.** `python scripts/build_neon_itd_fixture.py` downloads the
annotations and CHMs from Zenodo with HTTP range requests, without
downloading the whole 3.9 GB archive, and rewrites this file.
