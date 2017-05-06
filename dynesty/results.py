#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for handling results.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np

__all__ = ["Results"]


class Results(dict):
    """Contains the full output of a run along with a set of helper
    functions for summarizing the output."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def summary(self):
        """Return a formatted string giving a quick summary
        of the results."""

        return ("nlive: {:d}\n"
                "niter: {:d}\n"
                "ncall: {:d}\n"
                "eff(%): {:6.3f}\n"
                "logz: {:6.3f} +/- {:6.3f}"
                .format(self.nlive, self.niter, self.ncall, self.eff,
                        self.logz[-1], self.logzerr[-1]))
