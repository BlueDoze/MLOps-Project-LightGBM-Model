from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Mlops_Project_Reservation_Prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Da Silva Aguiar, Pedro"
)