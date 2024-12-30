from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "dataforge/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="dataforge",
    version=version,
    description="MB-Fit-Data-Forge is a tool used to prepare input data for MB-Fit, starting from molecular trajectories",
    download_url="https://github.com/limresgrp/MB-Fit-Data-Forge.git",
    author="Daniele Angioletti",
    python_requires=">=3.8",
    packages=find_packages(include=["dataforge", "dataforge.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "dataforge-parse-traj = dataforge.scripts.parse_traj:main",
            "dataforge-build-nmers = dataforge.scripts.build_nmers:main",
            "dataforge-qchem = dataforge.scripts.qchem:main",
        ]
    },
    install_requires=[
        "numpy",
        "h5py",
        "ase",
        "tqdm",
        "pandas",
        "scikit-learn",
        "MDAnalysis",
    ],
    zip_safe=True,
)