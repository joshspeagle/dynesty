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

    
setup(
    name="dynesty",
    url="https://github.com/joshspeagle/dynesty",
    version="0.8.0",
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
