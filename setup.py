# setup.py
from setuptools import setup, find_packages

setup(
    name="promoter_atlas",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy<2",
        "h5py",
        "matplotlib",
        "biopython",
        "bcbio-gff"
    ],
)