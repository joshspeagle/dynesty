#!/usr/bin/env python
import re
from distutils.core import setup

# Synchronize version from code.
version = re.findall(r"__version__ = \"(.*?)\"", open("nestle.py").read())[0]

setup(name="nestle", 
      version=version,
      description=("Nested sampling algorithms for evaluating "
                   "Bayesian evidence"),
      long_description=("Package documentation: "
                        "http://kbarbary.github.io/nestle"),
      classifiers = ["Development Status :: 4 - Beta",
                     "Programming Language :: Python :: 2",
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Intended Audience :: Science/Research"],
      py_modules=["nestle"],
      url="http://github.com/kbarbary/nestle",
      author="Kyle Barbary",
      author_email="kylebarbary@gmail.com")
