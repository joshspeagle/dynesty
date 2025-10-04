#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dynesty is nested sampling package.
The main functionality of dynesty is performed by the
dynesty.NestedSampler and dynesty.DynamicNestedSampler
classes
"""
from importlib.metadata import version, PackageNotFoundError

from .dynesty import NestedSampler, DynamicNestedSampler
from . import bounding
from . import utils
from . import pool

try:
    __version__ = version("dynesty")
except PackageNotFoundError:
    # package is not installed
    pass
