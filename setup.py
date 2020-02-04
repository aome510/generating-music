from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow>=1.15.2,<2',
    'tensorflow-model-analysis==0.13.0',
    'numpy>=1.14',
    'music21',
    'matplotlib',
    'six'
]

setup(
    name='',
    version='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)