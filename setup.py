#!/usr/bin/env python
import os
from glob import glob
from distutils.core import setup

description = "nested sampling algorithms for computing Bayesian evidence"

classifiers = [
    "Development Status :: 3 - Alpha Development Status",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Intended Audience :: Science/Research"]

setup(name="nestle", 
      version="0.1.0.dev",
      description=description,
      long_description=description,
      license = "BSD",
      classifiers=classifiers,
      py_modules=["nestle"],
      url="https://github.com/kbarbary/nestle",
      author="Kyle Barbary",
      author_email="kylebarbary@gmail.com")
