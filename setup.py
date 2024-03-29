from setuptools import setup, find_packages

setup(
    name="HesScale",
    version="1.0.0",
    description="HesScale: Scalable Computation of Hessian Diagonals",
    url="https://github.com/mohmdelsayed",
    author="Mohamed Elsayed",
    author_email="mohamedelsayed@ualberta.ca",
    packages=find_packages(exclude=["tests*"]),
    install_requires=['backpack-for-pytorch==1.3.0'],
)
