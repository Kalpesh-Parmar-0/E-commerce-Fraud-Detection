from setuptools import find_packages, setup
from typing import List

HYPHON_E_DOT='-e .'
def get_requirements(filepath: str) -> List[str]:
    requriements = []

    with open (filepath) as file_obj:
        requriements = file_obj.readlines()
        requriements = [req.replace("\n", "") for req in requriements]

        if HYPHON_E_DOT in requriements:
            requriements = requriements.remove(HYPHON_E_DOT)
        return requriements

setup(
    name='E-Commerce-Fraud-Detection',
    version='0.0.1',
    author='Kalpesh',
    author_email='kalpesh2112004@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)