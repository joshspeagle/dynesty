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

    
setup(
    name="dynesty",
    url="https://github.com/joshspeagle/dynesty",
    version=__version__,
    author="Josh Speagle",
    author_email="jspeagle@cfa.harvard.edu",
    packages=["dynesty"],
    license="LICENSE",
    description="Dynamic Nested Sampling",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "six"],
)
