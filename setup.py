#!/usr/bin/env python
import re
from setuptools import setup

# Synchronize version from code.
version = re.findall(r"__version__ = \"(.*?)\"", open("dynesty.py").read())[0]

setup(name="dynesty", 
      version=version,
      description=("Dynamic nested sampling for evaluating "
                   "Bayesian evidence and posteriors"),
      long_description=("Package documentation: "
                        "TO BE CONTINUED"),
      classifiers = ["Development Status :: 4 - Beta",
                     "Programming Language :: Python :: 2",
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Topic :: Scientific/Engineering",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Intended Audience :: Science/Research"],
      py_modules=["dynesty"],
      url="http://github.com/joshspeagle/dynesty",
      author="Josh Speagle",
      author_email="jspeagle@cfa.harvard.edu")
