from setuptools import setup, find_packages

setup(name='pset',
  version='0.1.0',
  install_requires = [
    'ml_utils @ git+https://github.com/tehzevo/ml-utils@master#egg=ml_utils',
    'tensorflow',
    'numpy'
  ],
  packages=find_packages())
