from setuptools import setup, find_packages

setup(
    name="financial-analysis",
    version="0.0.1",
    packages=find_packages(where="src/main/python"),
    package_dir={"": "src/main/python"},
)
