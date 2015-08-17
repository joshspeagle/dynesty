#!/usr/bin/env python
import os
from glob import glob
from distutils.core import setup

url = "https://github.com/kbarbary/nestle"

setup(name="nestle", 
      version="0.1.0.dev",
      description="Nested sampling algorithms for evaluating Bayesian evidence",
      long_description=url,
      classifiers = ["Development Status :: 2 - Pre-Alpha",
                     "Programming Language :: Python :: 2",
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Intended Audience :: Science/Research"],
      py_modules=["nestle"],
      url=url,
      author="Kyle Barbary",
      author_email="kylebarbary@gmail.com")
