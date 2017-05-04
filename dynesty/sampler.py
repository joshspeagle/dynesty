#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base sampler class containing various helpful functions.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math

import numpy as np

__all__ = ["Sampler"]


class Sampler:
    """
    A basic sampler object that proposes new live points.

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior. The input is a 1-d numpy array with length `ndim`, where
        each value is in the range [0, 1). The return value should also be a
        1-d numpy array with length `ndim`, where each value is a parameter.
        The return value is passed to the loglikelihood function. For example,
        for a 2 parameter model with flat priors in the range [0, 2), the
        function would be::

            def prior_transform(u):
                return 2.0 * u

    live_points : `~numpy.ndarray` with shape (nlive, ndim)
        Initial set of "live" points. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    options : dict
        A collection of extra arguments specific to each sampler.

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
        For the 'single' and 'multi' methods, enlarge the ellipsoid(s) by
        this fraction in volume. Default is *1.2*.

    vol_dec : float, optional
        For the 'multi' method, the required fractional reduction in volume
        after splitting an ellipsoid in order to to accept the split.
        Default is *0.5*.

    vol_check : float, optional
        For the 'multi' method, the factor used to when checking whether the
        volume of the original bounding ellipsoid is large enough to warrant
        more trial splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is *2.0*.

    """

    def __init__(self, loglikelihood, prior_transform, live_points, rstate,
                 options, queue_size, pool):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.live_points = live_points
        self.rstate = rstate
        self.set_options(options)
        self.queue_size = queue_size
        self.pool = pool
        self.queue = []
        self.nqueue = 0
        self.submitted = 0
        self.cancelled = 0
        self.unused = 0
        self.used = 0

    def check_unit_cube(self, point):
        """Check whether a point falls within the unit cube."""

        return np.all(point > 0.) and np.all(point < 1.)

    def empty_queue(self):
        """Dump all live point proposals currently on the queue."""

        while self.nqueue > 0:
            f = self.queue.pop()
            if f.cancel():
                self.cancelled += 1
            else:
                self.unused += 1
            self.nqueue -= 1

    def fill_queue(self):
        """Sequentially add new live point proposals to the queue."""

        while self.nqueue < self.queue_size:
            self.queue.append(self.pool.submit(self.propose_point))
            self.nqueue += 1
            self.submitted += 1

    def get_point_value(self):
        """Get a live point proposal sequentially from the filled queue.
        Afterwards, refill the queue."""

        f = self.queue.pop(0)
        self.nqueue -= 1
        u, v, logl = f.result()
        self.fill_queue()
        self.used += 1

        return u, v, logl
