#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dynesty: Bayesian evidence and posteriors using dynamic nested sampling

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np
import scipy.misc as misc

from .sampler import *
from .nestedsamplers import *
from .fakepool import *

__all__ = ["NestedSampler"]

_SAMPLERS = {'none': UnitCubeSampler,
             'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler}
_SAMPLING = ['uniform', 'randomwalk', 'slice', 'randomtrajectory']

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def NestedSampler(loglikelihood, prior_transform, ndim, nlive=100,
                  bound='multi', sample='uniform', update_interval=None,
                  npdim=None, rstate=None, queue_size=1, pool=None,
                  live_points=None, **kwargs):
    """
    Initializes and returns a chosen sampler that will perform nested sampling
    to evaluate Bayesian evidence and posteriors.

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

    ndim : int
        Number of parameters returned by prior and accepted by loglikelihood.

    nlive : int, optional
        Number of "live" points. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is *100*.

    bound : {'none', 'single', 'multi'}, optional
        Method used to approximately bound the prior using the current
        set of live points. Used to condition sampling methods used to
        propose new live points. Choices are no bound ('none'), a single
        bounding ellipsoid ('single'), and multiple bounding ellipsoids
        ('multi'). Default is 'multi'.

    sample : {'uniform', 'randomwalk', 'slice', 'randomtrajectory'}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Choices are uniform sampling
        ('uniform'), random walking away from a current live point
        ('randomwalk'), repeated slice sampling away from a current live
        point ('slice'), and initializing a random trajectory away from a
        current live point ('randomtrajectory').

    update_interval : int, optional
        Only update the proposal distribution every `update_interval`-th
        likelihood call. Update intervals larger than 1 can be more efficient
        when the likelihood function is very fast, particularly when
        using the multi-ellipsoid method. Default is `round(0.6 * nlive)`.

    npdim : int, optional
        Number of parameters accepted by prior. This might differ from `ndim`
        in the case where a parameter of loglikelihood is dependent upon
        multiple independently distributed parameters, some of which may be
        nuisance parameters.

    rstate : `~numpy.random.RandomState`, optional
        RandomState instance. If not given, the global random state of the
        `numpy.random` module will be used.

    queue_size : int, optional
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated. Default is *1* (no parallelism).

    pool : ThreadPoolExecutor, optional
        Use this pool of workers to propose live points in parallel. If
        `queue_size > 1` and `pool` is not specified, a `ValueError` will be
        thrown.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        A set of live points used to initialize the nested sampling run.
        Contains `live_u`, the coordinates on the unit cube, `live_v`, the
        transformed variables, and `live_logl`, the associated loglikelihoods.
        By default, if these are not provided the initial set of live points
        will be drawn uniformly from the unit `npdim`-cube.
        **WARNING: It is crucial that the initial set of live points have been
        sampled from the prior. Failure to provide a set of valid live points
        will result in biased results.**


    Other Parameters
    ----------------

    enlarge : float, optional
        For the 'single' and 'multi' bounding options, enlarge the volumes of
        the ellipsoid(s) by this fraction. Default is *1.2*.

    vol_dec : float, optional
        For the 'multi' bounding option, the required fractional reduction in
        volume after splitting an ellipsoid in order to to accept the split.
        Default is *0.5*.

    vol_check : float, optional
        For the 'multi' bounding option, the factor used when checking if
        the volume of the original bounding ellipsoid is large enough to
        warrant >2 splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is *2.0*.

    walks : int, optional
        For the 'randomwalk' sampling option, the minimum number of steps
        to take before proposing a new live point. Default is *25*.

    nrepeat : int, optional
        For the 'slice' sampling option, the number of times to repeat a
        slice sampling update, which consists of slicing through a set of
        `npdim` basis vectors in a random order. Default is *3*.


    Returns
    -------
    sampler : A child of `Sampler`
        An initialized instance of the chosen sampler specified via `method`.

    """

    # Initialize variables.
    if npdim is None:
        npdim = ndim

    if bound not in _SAMPLERS:
        raise ValueError("Unknown bounding method: '{0}'".format(bound))

    if sample not in _SAMPLING:
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    if nlive < 2 * ndim:
        warnings.warn("You really want to make `nlive >= 2 * ndim`!")

    if update_interval is None:
        update_interval = max(1, round(0.6 * nlive))
    else:
        update_interval = round(update_interval)
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")

    if rstate is None:
        rstate = np.random

    # Set up parallel (or serial) evaluation.
    if queue_size == 1:
        pool = FakePool()
    else:
        if pool is None:
            raise ValueError("Missing `pool`. Please provide a Pool.")

    # Initialize live points and calculate likelihoods.
    if live_points is None:
        live_u = rstate.rand(nlive, npdim)  # positions in unit cube
        live_v = np.empty((nlive, ndim), dtype=np.float64)  # real params
        for i in range(nlive):
            live_v[i, :] = prior_transform(live_u[i, :])
        live_logl = np.fromiter(pool.map(loglikelihood, live_v),
                                dtype=np.float64)  # log likelihood
        live_points = [live_u, live_v, live_logl]

    # Initialize our nested sampler.
    sampler = _SAMPLERS[bound](loglikelihood, prior_transform, npdim,
                               live_points, sample, update_interval,
                               rstate, queue_size, pool, kwargs)

    return sampler
