#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dynesty: using dynamic nested sampling routines
to evaluate Bayesian evidence and posteriors.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np
import scipy.misc as misc

from .sampler import *
from .samplers import *
from .fakepool import *

__all__ = ["NestedSampler"]

_SAMPLERS = {'single_ell': SingleEllipsoidSampler,
             'multi_ell': MultiEllipsoidSampler}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def NestedSampler(loglikelihood, prior_transform, ndim, nlive=100,
                  method='multi', update_interval=None, npdim=None,
                  rstate=None, queue_size=1, pool=None, **kwargs):
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

    method : {'single_ell', 'multi_ell'}, optional
        Method used to select new points. Choices are single-ellipsoidal
        ('single_ell') and multi-ellipsoidal ('multi_ell').
        Default is 'multi_ell'.

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

    queue_size: int, optional
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated. Default is *1* (no parallelism).

    pool: ThreadPoolExecutor, optional
        Use this pool of workers to propose live points in parallel. If
        `queue_size > 1` and `pool` is not specified, a `ValueError` will be
        thrown.


    Other Parameters
    ----------------

    enlarge : float, optional
        For the 'single_ell' and 'multi_ell' methods, enlarge the volumes of
        the ellipsoid(s) by this fraction. Default is *1.2*.

    vol_dec : float, optional
        For the 'multi_ell' method, the required fractional reduction in
        volume after splitting an ellipsoid in order to to accept the split.
        Default is *0.5*.

    vol_check : float, optional
        For the 'multi_ell' method, the factor used to when checking whether
        the volume of the original bounding ellipsoid is large enough to
        warrant more trial splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is *2.0*.


    Returns
    -------
    sampler : A child of `Sampler`
        An initialized instance of the chosen sampler specified via `method`.

    """

    # Initialize variables.
    if npdim is None:
        npdim = ndim

    if method not in _SAMPLERS:
        raise ValueError("Unknown method: '{:r}'".format(method))

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

    # Set up parallel evaluation.
    if queue_size == 1:
        pool = FakePool()
    else:
        if pool is None:
            raise ValueError("Missing `pool`. Please provide a Pool.")

    # Initialize live points and calculate likelihoods.
    live_u = rstate.rand(nlive, npdim)  # positions in unit cube
    live_v = np.empty((nlive, ndim), dtype=np.float64)  # real params
    for i in range(nlive):
        live_v[i, :] = prior_transform(live_u[i, :])
    live_logl = np.fromiter(pool.map(loglikelihood, live_v),
                            dtype=np.float64)  # log likelihood
    live_points = [live_u, live_v, live_logl]

    # Initialize our sampler.
    sampler = _SAMPLERS[method](loglikelihood, prior_transform, npdim,
                                live_points, update_interval, rstate,
                                queue_size, pool, kwargs)

    return sampler
