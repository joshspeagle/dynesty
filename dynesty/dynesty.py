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
from .sampling import *
from .nestedsamplers import *
from .dynamicsampler import *

__all__ = ["NestedSampler", "DynamicNestedSampler"]

_SAMPLERS = {'none': UnitCubeSampler,
             'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler,
             'balls': RadFriendsSampler,
             'cubes': SupFriendsSampler}
_SAMPLING = {'unif': sample_unif,
             'rwalk': sample_rwalk,
             'slice': sample_slice}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def NestedSampler(loglikelihood, prior_transform, ndim, nlive=250,
                  bound='multi', sample='unif', update_interval=0.6,
                  npdim=None, rstate=None, queue_size=1, pool=None,
                  live_points=None, **kwargs):
    """
    Initializes and returns a chosen sampler to evaluate Bayesian evidence
    and posteriors using nested sampling.

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
        number of iterations required to converge. Default is *250*.

    bound : {'none', 'single', 'multi', 'balls', 'cubes'}, optional
        Method used to approximately bound the prior using the current
        set of live points. Used to condition sampling methods used to
        propose new live points. Choices are no bound ('none'), a single
        bounding ellipsoid ('single'), multiple bounding ellipsoids
        ('multi'), balls centered on each live point ('balls'), and
        cubes centered on each live point ('cubes'). Default is 'multi'.

    sample : {'unif', 'rwalk', 'slice', 'rtraj'}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Choices are uniform
        ('unif'), random walks ('rwalk'), slices ('slice'), and random
        trajectories ('rtraj'). Default is 'uniform'.

    update_interval : int or float, optional
        If an integer is passed, only update the proposal distribution every
        `update_interval`-th likelihood call. If a float is passed, update the
        proposal after every `round(update_interval * nlive)`-th likelihood
        call. Larger update intervals larger can be more efficient
        when the likelihood function is quick to evaluate. Default is *0.6*.

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
        Enlarge the volumes of the specified bounding object(s) by this
        fraction. The preferred method is to determine this organically
        using bootstrapping. If `bootstrap > 0`, this defaults to *1.0*.
        If `bootstrap = 0`, this instead defaults to *1.25*.

    bootstrap : int, optional
        Compute this many bootstrap-resampled realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes.
        Default is *20*.

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
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    Returns
    -------
    NestedSampler : `NestedSampler` instance
        An initialized instance of the chosen sampler specified via `bound`.

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

    if isinstance(update_interval, float):
        update_interval = max(1, round(update_interval * nlive))

    if rstate is None:
        rstate = np.random

    # Set up parallel (or serial) evaluation.
    if queue_size < 1:
        raise ValueError("The queue must contain at least one element!")
    elif queue_size == 1:
        M = map
    else:
        if pool is not None:
            M = pool.map
        else:
            raise ValueError("Missing `pool`. Please provide a Pool.")

    # Initialize live points and calculate likelihoods.
    if live_points is None:
        live_u = rstate.rand(nlive, npdim)  # positions in unit cube
        live_v = M(prior_transform, live_u)  # real parameters
        live_logl = M(loglikelihood, live_v)  # log likelihood
        live_points = [live_u, live_v, live_logl]

    # Convert all `-np.inf` log-likelihoods to finite large numbers.
    # Necessary to keep estimators in our sampler from breaking.
    for i, logl in enumerate(live_points[2]):
        if not np.isfinite(logl):
            if np.sign(logl) < 0:
                live_points[2][i] = -1e300
            else:
                raise ValueError("The log-likelihood ({0}) of live point {1} "
                                 "located at u={2} v={3} is invalid."
                                 .format(logl, i, live_points[0][i],
                                         live_points[1][i]))

    # Initialize our nested sampler.
    sampler = _SAMPLERS[bound](loglikelihood, prior_transform, npdim,
                               live_points, sample, update_interval,
                               rstate, queue_size, pool, kwargs)

    return sampler


def DynamicNestedSampler(loglikelihood, prior_transform, ndim,
                         bound='multi', sample='unif', update_interval=0.6,
                         npdim=None, rstate=None, queue_size=1, pool=None,
                         **kwargs):
    """
    Initializes and returns a chosen sampler to evaluate Bayesian evidence
    and posteriors using nested sampling.

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

    bound : {'none', 'single', 'multi', 'balls', 'cubes'}, optional
        Method used to approximately bound the prior using the current
        set of live points. Used to condition sampling methods used to
        propose new live points. Choices are no bound ('none'), a single
        bounding ellipsoid ('single'), multiple bounding ellipsoids
        ('multi'), balls centered on each live point ('balls'), and
        cubes centered on each live point ('cubes'). Default is 'multi'.

    sample : {'unif', 'rwalk', 'slice', 'rtraj'}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds. Choices are uniform
        ('unif'), random walks ('rwalk'), slices ('slice'), and random
        trajectories ('rtraj'). Default is 'uniform'.

    update_interval : int or float, optional
        If an integer is passed, only update the proposal distribution every
        `update_interval`-th likelihood call. If a float is passed, update the
        proposal after every `round(update_interval * nlive)`-th likelihood
        call, changing depending on how many live points are currently
        active. Larger update intervals can be more efficient
        when the likelihood function is quick to evaluate. Default is *0.6*.

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

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the specified bounding object(s) by this
        fraction. The preferred method is to determine this organically
        using bootstrapping. If `bootstrap > 0`, this defaults to *1.0*.
        If `bootstrap = 0`, this instead defaults to *1.25*.

    bootstrap : int, optional
        Compute this many bootstrap-resampled realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes.
        Default is *20*.

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
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    Returns
    -------
    NestedSampler : `NestedSampler` instance
        An initialized instance of the chosen sampler specified via `bound`.

    """

    # Initialize variables.
    if npdim is None:
        npdim = ndim

    if bound not in _SAMPLERS:
        raise ValueError("Unknown bounding method: '{0}'".format(bound))

    if sample not in _SAMPLING:
        raise ValueError("Unknown sampling method: '{0}'".format(sample))

    if rstate is None:
        rstate = np.random

    # Set up parallel (or serial) evaluation.
    if queue_size < 1:
        raise ValueError("The queue must contain at least one element!")
    elif queue_size == 1:
        M = map
    else:
        if pool is not None:
            M = pool.map
        else:
            raise ValueError("Missing `pool`. Please provide a Pool.")

    # Initialize our nested sampler.
    sampler = DynamicSampler(loglikelihood, prior_transform, npdim,
                             bound, sample, update_interval,
                             rstate, queue_size, pool, kwargs)

    return sampler
