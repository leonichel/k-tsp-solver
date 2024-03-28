from setuptools import setup, find_packages

version_string = "0.1.0"

setup(
    name="bl_segmentation",
    version=version_string,
    author="leonichel",
    description="",
    url="https://github.com/Keenwawa/brightloom-segmentation",
    package_data={},
    packages=find_packages("k_tsp_solver"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy"
        "tsplib95",
        "tqdm"
    ],
    python_requires=">=3.10",
)