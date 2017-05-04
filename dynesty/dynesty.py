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

from .sampler import *
from .globalsampler import *
from .localsampler import *
from .fakepool import *

__all__ = ["sample", "Results"]

_SAMPLERS = {'single_ell': SingleEllipsoidSampler,
             'multi_ell': MultiEllipsoidSampler}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


class Results(dict):
    """
    Contains the output of a dynamic nested sampling run.

    Since this class is essentially a subclass of dict with attribute
    accessors, one can see which attributes are available using the
    `keys()` method.

    """

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
                "nsamples: {:d}\n"
                "logz: {:6.3f} +/- {:6.3f}\n"
                "h: {:6.3f}"
                .format(self.nlive, self.niter, self.ncall, self.eff,
                        len(self.samples), self.logz, self.logzerr,
                        self.h))


def sample(loglikelihood, prior_transform, ndim, nlive=100,
           method='multi', update_interval=None, npdim=None,
           maxiter=None, maxcall=None, dlogz=None, decline_factor=None,
           rstate=None, callback=None, queue_size=1, pool=None, **options):
    """
    Perform nested sampling to evaluate Bayesian evidence.

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
        Only update the new point selector every `update_interval`-th
        likelihood call. Update intervals larger than 1 can be more efficient
        when the likelihood function is very fast, particularly when
        using the multi-ellipsoid method. Default is `round(0.6 * nlive)`.

    npdim : int, optional
        Number of parameters accepted by prior. This might differ from `ndim`
        in the case where a parameter of loglikelihood is dependent upon
        multiple independently distributed parameters, some of which may be
        nuisance parameters.

    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is no limit (`sys.maxsize`).

    maxcall : int, optional
        Maximum number of likelihood evaluations. Iteration may stop earlier
        if termination condition is reached. Default is no limit
        (`sys.maxsize`).

    dlogz : float, optional
        If supplied, iteration will stop when the estimated contribution
        of the remaining prior volume to the total evidence falls below
        this threshold. Explicitly, the stopping criterion is
        `log(z + z_est) - log(z) < dlogz`, where `z` is the current evidence
        from all saved samples and `z_est` is the estimated contribution
        from the remaining volume. This option and decline_factor are
        mutually exclusive. Default is *0.5*.

    decline_factor : float, optional
        If supplied, iteration will stop when the sample weights
        (likelihood times prior volume) of newly saved samples has been
        declining for `decline_factor * nsamples` consecutive samples.
        A value of *1.0* works well for most cases. This option and `dlogz`
        are mutually exclusive. If not specified, the default `dlogz` criterion
        is used.

    rstate : `~numpy.random.RandomState`, optional
        RandomState instance. If not given, the global random state of the
        `numpy.random` module will be used.

    callback : function, optional
        Callback function to be called at each iteration. A single argument,
        a dictionary, is passed to the callback. The keys include 'it',
        the current iteration number, and 'logz', the current total
        log evidence of all saved points. To simply print these at each
        iteration, use the convience function `callback=nestle.print_progress`.

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


    Returns
    -------
    result : `Result`
        A dictionary-like object with attribute access: Attributes can be
        accessed with, for example, either `result['niter']` or
        `result.niter`. Attributes:

        niter : int
            Number of iterations.

        ncall : int
            Number of likelihood calls.

        logz : float
            Natural logarithm of the evidence (integral of posterior).

        logzerr : float
            Estimated numerical (sampling) error on `logz`.

        h : float
            Information. This is a measure of the "peakiness" of the
            likelihood function. A constant likelihood has zero information.

        samples : `~numpy.ndarray` with shape (nsamples, ndim)
            Parameter values of each sample.

        logvol : `~numpy.ndarray` with shape (nsamples,)
            Natural log of prior volume of corresponding to each sample.

        logl : `~numpy.ndarray` with shape (nsamples,)
            Natural log of the likelihood for each sample, as returned by
            user-supplied `loglikelihood` function.

        logwt : `~numpy.ndarray` with shape (nsamples,)
            Natural log of the weights corresponding to each sample defined as
            `logwt = logvol + logl - logz`.

    """

    # Initialize variables.
    if npdim is None:
        npdim = ndim

    if maxiter is None:
        maxiter = sys.maxsize

    if maxcall is None:
        maxcall = sys.maxsize

    if method not in _SAMPLERS:
        raise ValueError("Unknown method: '{:r}'".format(method))

    if nlive < 2 * ndim:
        warnings.warn("You really want to make `nlive >= 2 * ndim`!")

    if rstate is None:
        rstate = np.random

    # Establish stopping criterion.
    if dlogz is not None and decline_factor is not None:
        raise ValueError("Cannot specify two separate stopping criteria: "
                         "decline_factor and dlogz")
    elif dlogz is None and decline_factor is None:
        dlogz = 0.5

    if update_interval is None:
        update_interval = max(1, round(0.6 * nlive))
    else:
        update_interval = round(update_interval)
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")

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

    # Initialize our sampler.
    sampler = _SAMPLERS[method](loglikelihood, prior_transform, live_u,
                                rstate, options, queue_size, pool)

    # Initialize values for nested sampling loop.
    saved_u = []  # samples (unit cube)
    saved_v = []  # samples (transformed)
    saved_logl = []  # log(likelihood)
    saved_logvol = []  # log(volume)
    saved_logwt = []  # log(weight)
    h = 0.0  # Information, initially *0.*
    logz = -1e300  # log(evidence), initially *0.*
    logvol = math.log(1.0 - math.exp(-1.0/nlive))  # initially `1-e^(1/n)`
    ncall = nlive  # number of calls we already made

    # Initialize proposal distribution for our sampler.
    pointvol = 1./nlive
    sampler.update(pointvol)

    callback_info = {'it': 0,
                     'logz': logz,
                     'live_u': live_u,
                     'sampler': sampler}

    # The main nested sampling loop.
    ndecl = 0
    logwt_old = -np.inf
    it = 1  # iterations start from 1
    since_update = 0
    while it < maxiter:
        # Output callback if requested.
        if callback is not None:
            callback_info.update(it=it, logz=logz)
            callback(callback_info)

        # After `update_interval` interations have passed, update the sampler
        # using the current set of live points.
        if since_update >= update_interval:
            expected_vol = math.exp(-it / nlive)  # average volume
            pointvol = expected_vol / nlive  # volume per point
            sampler.update(pointvol)
            since_update = 0

        # Locate the "live" point with the lowest `logl` (the "worst" point).
        worst = np.argmin(live_logl)

        # Set our new worst likelihood constraint.
        ustar, vstar = live_u[worst], live_v[worst]  # position
        loglstar = live_logl[worst]  # likelihood

        # Set our new weight.
        logwt = logvol + loglstar

        # Sample a new live point from within the likelihood constraint
        # `logl > loglstar` using the proposal distribution from our sampler.
        u, v, logl, nc = sampler.new_point(loglstar)
        ncall += nc
        since_update += nc

        # Add the worst live point to samples. It is now a "dead" point.
        saved_u.append(np.array(ustar))
        saved_v.append(np.array(vstar))
        saved_logl.append(loglstar)
        saved_logvol.append(logvol)
        saved_logwt.append(logwt)

        # Update evidence `logz` and information `h` using our new dead point.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * loglstar +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Update the live point (previously our "worst" point).
        live_u[worst] = u
        live_v[worst] = v
        live_logl[worst] = logl

        # Apply expected shrinkage to `logvol` for the next live point.
        logvol -= 1.0 / nlive

        # Stopping criterion 1: estimated (fractional) remaining evidence
        # lies below some threshold set by `dlogz`.
        if dlogz is not None:
            logz_remain = np.max(live_logl) - it / nlive
            if np.logaddexp(logz, logz_remain) - logz < dlogz:
                break

        # Stopping criterion 2: `logwt` has been declining for longer
        # than `decline_factor`.
        if decline_factor is not None:
            if logwt < logwt_old:
                ndecl += 1
            else:
                ndecl = 0
            logwt_old = logwt
            if ndecl > decline_factor * nlive:
                break

        # Stopping criterion 3: number of `loglikelihood` calls
        # exceeds `maxcall`.
        if ncall > maxcall:
            break

        it += 1

    # Add remaining live points to our set of dead points.
    # After N samples have been taken out, the remaining volume is
    # `e^(-N / nlive)`. Thus, the remaining volume for each live point
    # is `e^(-N / nlive) / nlive`. The log of this for each live point is:
    # `log(e^(-N / nlive) / nlive) = -N / nlive - log(nlive)`.
    logvol = -len(saved_v) / nlive - math.log(nlive)
    for i in xrange(nlive):
        ustar, vstar = live_u[i], live_v[i]
        loglstar = live_logl[i]
        logwt = logvol + loglstar
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * loglstar +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        saved_u.append(np.array(ustar))
        saved_v.append(np.array(vstar))
        saved_logl.append(loglstar)
        saved_logvol.append(logvol)
        saved_logwt.append(logwt)

    # h should always be nonnegative (we take the sqrt below).
    # Numerical error makes it negative in pathological corner cases
    # such as flat likelihoods. Here we correct those cases to zero.
    if h < 0.0:
        if h > -SQRTEPS:
            h = 0.0
        else:
            raise RuntimeError("Negative h encountered (h={}). Please report "
                               "this as a likely bug.".format(h))

    # Compute our sampling efficiency.
    eff = 100. * it / (ncall - nlive)

    # Saving results.
    results = Results([('nlive', nlive),
                       ('niter', it),
                       ('ncall', ncall),
                       ('eff', eff),
                       ('logz', logz),
                       ('logzerr', math.sqrt(h / nlive)),
                       ('h', h),
                       ('samples_unit', np.array(saved_u)),
                       ('samples', np.array(saved_v)),
                       ('logwt', np.array(saved_logwt) - logz),
                       ('logvol', np.array(saved_logvol)),
                       ('logl', np.array(saved_logl))])

    return results
