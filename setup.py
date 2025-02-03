from __future__ import print_function
from setuptools import setup

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

version = '1.0'

setup(name='presecolm',
      version=version,
      description='Experiment setup to evaluate the prediction of sensitive concepts in language models.',
      url='https://github.com/HammerLabML/PreSeCoLM',
      packages=['data_loader','models'],
      install_requires=INSTALL_REQUIRES,
      )
