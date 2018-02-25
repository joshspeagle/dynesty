#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'dynesty', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)

try:
    import pypandoc
    with open('README.md', 'r') as f:
        txt = f.read()
    txt = re.sub('<[^<]+>', '', txt)
    long_description = pypandoc.convert(txt, 'rst', 'md')
except ImportError:
    long_description = open('README.md').read()

    
setup(
    name="dynesty",
    url="https://github.com/joshspeagle/dynesty",
    version=__version__,
    author="Joshua S Speagle",
    author_email="jspeagle@cfa.harvard.edu",
    packages=["dynesty"],
    license="MIT",
    description=("A dynamic nested sampling package for computing Bayesian "
                 "posteriors and evidences."),
    long_description=long_description,
    package_data={"": ["README.md", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "six"],
    keywords=["nested sampling", "dynamic", "monte carlo", "bayesian",
              "inference", "modeling"],
    classifiers=["Development Status :: 4 - Beta",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.6",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering",
                 "Intended Audience :: Science/Research"]
)
