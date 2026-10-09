"""
Builds tests/itd/data/neon_tree_evaluation.npz, the NeonTreeEvaluation fixture
used by tests/itd/test_neon_benchmark.py.

Source: NeonTreeEvaluation benchmark data (Weinstein et al. 2021), Zenodo
record 5914554, v0.2.2, CC BY 4.0. Only the crown annotations
(evaluation/RGB/benchmark_annotations.csv) and the matching 1 m CHMs
(evaluation/CHM/<plot>_CHM.tif) are read. evaluation.zip is 3.9 GB, so its
members are fetched with HTTP range requests instead of downloading the
archive.

Usage (from the repository root; needs network access):
    python scripts/build_neon_itd_fixture.py [output.npz]
"""

# Core imports
import http.client
import io
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# External imports
import numpy as np
import pandas as pd
import rasterio

EVALUATION_ZIP = "https://zenodo.org/api/records/5914554/files/evaluation.zip/content"
ANNOTATIONS = "evaluation/RGB/benchmark_annotations.csv"
CHM_SHAPE = (40, 40)
# Zenodo intermittently times out (504); retry with backoff.
MAX_ATTEMPTS = 10
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "itd"
    / "data"
    / "neon_tree_evaluation.npz"
)


def fetch(request: urllib.request.Request, expected_bytes: int | None = None):
    """Open ``request`` and return (final URL, headers, body).

    Retries timeouts, 5xx, 429 and short reads; other 4xx errors are raised at
    once.  With ``expected_bytes``, the request is a range request and the
    server must answer 206 Partial Content.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if expected_bytes is not None and (
                    response.status != 206 or "Content-Range" not in response.headers
                ):
                    raise RuntimeError(
                        f"{response.url} ignored the Range header "
                        f"(status {response.status})"
                    )
                body = response.read()
                if expected_bytes is not None and len(body) != expected_bytes:
                    raise OSError(f"expected {expected_bytes} bytes, got {len(body)}")
                return response.url, response.headers, body
        except urllib.error.HTTPError as error:
            if 400 <= error.code < 500 and error.code != 429:
                raise
            if attempt == MAX_ATTEMPTS - 1:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == MAX_ATTEMPTS - 1:
                raise
        time.sleep(min(2**attempt, 60))


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP range requests."""

    def __init__(self, url: str):
        self.url, headers, _ = fetch(urllib.request.Request(url, method="HEAD"))
        self.size = int(headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}
        self.pos = base[whence] + offset
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        byte_range = f"bytes={self.pos}-{self.pos + n - 1}"
        request = urllib.request.Request(self.url, headers={"Range": byte_range})
        _, _, data = fetch(request, expected_bytes=n)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def site_code(plot: str) -> str:
    """NEON site of a plot name: ``SJER_005_2018`` or ``2018_SJER_3_...``."""
    parts = plot.split("_")
    return parts[1] if parts[0].isdigit() else parts[0]


def read_chm(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with rasterio.MemoryFile(archive.read(member)) as memfile:
        with memfile.open() as src:
            chm = src.read(1, masked=True).filled(0.0).astype(np.float32)
    chm[~np.isfinite(chm) | (chm < 0)] = 0.0
    return chm


def main(output: Path):
    archive = zipfile.ZipFile(
        io.BufferedReader(HttpRangeFile(EVALUATION_ZIP), buffer_size=1 << 18)
    )
    members = set(archive.namelist())
    annotations = pd.read_csv(io.BytesIO(archive.read(ANNOTATIONS)))
    annotations["plot"] = annotations["image_path"].str.removesuffix(".tif")

    plots, chms, boxes, box_plot, dropped = [], [], [], [], []
    for plot, crowns in sorted(annotations.groupby("plot")):
        member = f"evaluation/CHM/{plot}_CHM.tif"
        if member not in members:
            continue
        chm = read_chm(archive, member)
        if chm.shape != CHM_SHAPE:
            dropped.append(f"{plot} {chm.shape}")
            continue
        box_plot.append(np.full(len(crowns), len(plots), dtype=np.int16))
        boxes.append(crowns[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.int16))
        plots.append(plot)
        chms.append(chm)
        print(f"{plot}: {len(crowns)} crowns", flush=True)

    # Write next to the output and rename, so a failed run leaves it intact.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".npz.tmp", delete=False
    ) as tmp:
        np.savez_compressed(
            tmp,
            chm=np.stack(chms),
            boxes_dm=np.concatenate(boxes),
            box_plot=np.concatenate(box_plot),
            plot=np.array(plots),
            site=np.array([site_code(p) for p in plots]),
        )
    os.replace(tmp.name, output)
    print(f"Wrote {output}: {len(plots)} plots, {sum(map(len, boxes))} crowns")
    print("Dropped (truncated CHM):", ", ".join(dropped) or "none")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
