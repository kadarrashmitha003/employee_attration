from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="TECHNOVA",
    version= "0.1",
    author= "Rashmitha",
    packages= find_packages(),
    install_requires = requirements,
)