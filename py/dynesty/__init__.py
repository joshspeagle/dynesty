#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dynesty is nested sampling package.
The main functionality of dynesty is performed by the
dynesty.NestedSampler and dynesty.DynamicNestedSampler
classes
"""
from .dynesty import NestedSampler, DynamicNestedSampler
from . import bounding
from . import utils

__version__ = "1.2.3"
