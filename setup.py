from __future__ import annotations
import json
import urllib.request
from setuptools import find_packages, setup


def read_file(fname):
    with open(fname, encoding="utf-8") as fd:
        return fd.read()


def get_requirements(fname):
    with open(fname, encoding="utf-8") as fd:
        reqs = [line.strip() for line in fd if line]
    return reqs


def get_version():
    """Get the version number."""
    with urllib.request.urlopen(
        "https://api.github.com/repos/silvxlabs/fastfuels-core/releases/latest"
    ) as response:
        data = json.loads(response.read().decode("utf-8"))
    version = data["tag_name"]
    return version[1:]  # Remove the leading "v" from the version number


NAME = "fastfuels-core"
DESCRIPTION = "Core algorithms for the FastFuels project"
LONG_DESCRIPTION = read_file("README.md")
VERSION = get_version()
LICENSE = "MIT"
URL = "https://github.com/silvxlabs/fastfuels-core"
PROJECT_URLS = {"Bug Tracker": f"{URL}/issues"}
INSTALL_REQUIRES = [
    "dask",
    "dask-expr",
    "dask-image",
    "fastparquet",
    "geopandas",
    "nsvb",
    "numba",
    "numpy",
    "pandas",
    "pandera",
    "rasterio",
    "rioxarray",
    "scipy",
    "shapely>=2",
    "xarray",
]

EXTRAS_REQUIRE = {
    "plotting": [
        "matplotlib",
        "pyvista",
    ],
    "test": [
        "pytest",
        "matplotlib",
    ],
}
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    project_urls=PROJECT_URLS,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["docs", "tests"]),
    package_data={"fastfuels_core": ["data/*"]},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.9",
)
