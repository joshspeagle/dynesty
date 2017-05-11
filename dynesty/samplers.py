#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampler classes for proposing new live points. Includes:

    UnitCubeSampler:
        Samples from the unit N-cube with no constraints.

    SingleEllipsoidSampler:
        Uses a single ellipsoid to bound the set of live points.

    MultiEllipsoidSampler:
        Uses multiple ellipsoids to bound the set of live points.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np
import copy

from .sampler import *
from .ellipsoid import *

__all__ = ["SingleEllipsoidSampler", "MultiEllipsoidSampler"]


class SingleEllipsoidSampler(Sampler):
    """
    Bounds live points in a single ellipsoid and samples uniformly
    from within that ellipsoid.


    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


    Other Parameters
    ----------------

    enlarge : float, optional
        Enlarge the volume of the bounding ellipsoid by this fraction.
        Default is *1.2*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 update_interval, rstate, queue_size, pool, kwargs={}):
        self.kwargs = kwargs
        self.enlarge = kwargs.get('enlarge', 1.2)
        super(SingleEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)
        self.ell = Ellipsoid(np.zeros(self.npdim), np.identity(self.npdim))

    def update(self, pointvol):
        """Update bounding ellipsoid using the current set of live points."""

        self._empty_queue()
        self.ell.update(self.live_u, pointvol=pointvol)
        self.ell.scale_to_vol(self.ell.vol * self.enlarge)
        self._fill_queue()

        return copy.deepcopy(self.ell)

    def propose_point(self):
        """Propose a new live point."""

        while True:
            u = self.ell.sample(rstate=self.rstate)
            if self._check_unit_cube(u):
                break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl


class MultiEllipsoidSampler(Sampler):
    """
    Bounds live points in multiple ellipsoids and samples uniformly
    from within their union.


    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


    Other Parameters
    ----------------

    enlarge : float, optional
        Enlarge the volume of all bounding ellipsoids by this fraction.
        Default is *1.2*.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the proposed split. Default is *0.5*.

    vol_check : float, optional
        The factor used to when checking whether the volume of the original
        bounding ellipsoid is large enough to warrant more trial splits via
        `ell.vol > vol_check * nlive * pointvol`. Default is *2.0*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 update_interval, rstate, queue_size, pool, kwargs={}):
        self.kwargs = kwargs
        self.enlarge = kwargs.get('enlarge', 1.2)
        self.vol_dec = kwargs.get('vol_dec', 0.5)
        self.vol_check = kwargs.get('vol_check', 2.0)
        super(MultiEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)
        self.mell = MultiEllipsoid(ctrs=[np.zeros(self.npdim)],
                                   ams=[np.identity(self.npdim)])

    def update(self, pointvol):
        """Update bounding ellipsoids using the current set of live points."""

        self._empty_queue()
        self.mell.update(self.live_u, pointvol=pointvol,
                         vol_dec=self.vol_dec, vol_check=self.vol_check)
        self.mell.scale_to_vols(self.mell.vols * self.enlarge)
        self._fill_queue()

        return copy.deepcopy(self.mell)

    def propose_point(self):
        """Propose a new live point."""

        while True:
            u, q = self.mell.sample(rstate=self.rstate, return_q=True)
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping ellipsoids.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl
