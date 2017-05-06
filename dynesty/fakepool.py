#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes for dealing with the serial case (i.e. no parallelism).

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np

__all__ = ["FakePool", "FakeFuture"]


class FakePool(object):
    """A fake Pool for serial function evaluations."""

    def __init__(self):
        pass

    def submit(self, fn, *args, **kwargs):
        return FakeFuture(fn, *args, **kwargs)

    def map(self, func, *iterables):
        return map(func, *iterables)

    def shutdown(self):
        pass


class FakeFuture(object):
    """A fake Future to mimic function calls."""

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def result(self):
        return self.fn(*self.args, **self.kwargs)

    def cancel(self):
        return True
