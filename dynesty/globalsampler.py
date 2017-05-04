#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampler classes for proposing new live points based on bounding the set of
live points globally. Includes:

SingleEllipsoidSampler : Uses a single ellipsoid to bound the set.
MultiEllipsoidSampler : Uses multiple ellipsoids to bound the set.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np

from .sampler import *
from .ellipsoid import *

__all__ = ["SingleEllipsoidSampler", "MultiEllipsoidSampler"]


class SingleEllipsoidSampler(Sampler):
    """Bounds live points in a single ellipsoid and samples uniformly
    from within that ellipsoid."""

    def set_options(self, options):
        self.enlarge = options.get('enlarge', 1.2)

    def update(self, pointvol):
        """Update bounding ellipsoid using the current set of live points."""
        self.empty_queue()
        self.ell = bounding_ellipsoid(self.live_points, pointvol=pointvol)
        self.ell.scale_to_vol(self.ell.vol * self.enlarge)
        self.fill_queue()

    def propose_point(self):
        while True:
            u = self.ell.sample(rstate=self.rstate)
            if self.check_unit_cube(u):
                break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl

    def new_point(self, loglstar):
        ncall = 0
        while True:
            u, v, logl = self.get_point_value()
            ncall += 1
            if logl >= loglstar:
                break

        return u, v, logl, ncall


class MultiEllipsoidSampler(Sampler):
    """Bounds live points in multiple ellipsoids and samples uniformly
    from within the volume spanned by their union."""

    def set_options(self, options):
        self.enlarge = options.get('enlarge', 1.2)
        self.vol_dec = options.get('vol_dec', 0.5)
        self.vol_check = options.get('vol_check', 2.0)

    def update(self, pointvol):
        self.empty_queue()
        self.mell = bounding_ellipsoids(self.live_points, pointvol=pointvol,
                                        vol_dec=self.vol_dec,
                                        vol_check=self.vol_check)
        self.mell.scale_to_vols(self.mell.vols * self.enlarge)
        self.fill_queue()

    def propose_point(self):
        while True:
            u, q = self.mell.sample(rstate=self.rstate, return_q=True)
            if self.check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping ellipsoids.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl

    def new_point(self, loglstar):
        ncall = 0
        while True:
            u, v, logl = self.get_point_value()
            ncall += 1
            if logl >= loglstar:
                break

        return u, v, logl, ncall
