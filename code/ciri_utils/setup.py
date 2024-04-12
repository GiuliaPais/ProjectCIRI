from setuptools import setup, find_packages

setup(
    name="ciri_utils",
    version="0.1",
    packages=find_packages(),
    description="Utilities for the CIRI project",
    author="Giulia Pais",
    install_requires=[
        "pandas",
        "torch>=2.2.2",
        "torchvision>=0.17.2",
        "ray[tune]",
        "numpy",
        "scikit-learn",
        "tqdm"
    ],
)