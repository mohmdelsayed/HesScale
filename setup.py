from setuptools import setup, find_packages

setup(
    name="HesScale",
    version="1.0.0",  
    description="A scalable method of computing Hessian diagonals",
    url="https://github.com/mohmdelsayed",
    author="Mohamed Elsayed",
    author_email="mohamedelsayed@ualberta.ca",
    packages=find_packages(exclude=['tests*']),
)