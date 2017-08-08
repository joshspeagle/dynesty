#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic nested sampler class for adaptively proposing new live points.

"""

from __future__ import (print_function, division)
from builtins import range

import sys
import warnings
import math
import numpy as np
import copy
from scipy import optimize as opt
from numpy import linalg
from scipy import misc
import warnings

from .nestedsamplers import *
from .sampler import *
from .bounding import *
from .sampling import *
from .results import *
from .utils import *

__all__ = ["DynamicSampler", "weight_function", "stopping_function"]

_SAMPLERS = {'none': UnitCubeSampler,
             'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler,
             'balls': RadFriendsSampler,
             'cubes': SupFriendsSampler}
_SAMPLING = {'unif': sample_unif,
             'rwalk': sample_rwalk,
             'slice': sample_slice}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def weight_function(results, args, return_weights=False):
    """
    The default weight function utilized by `DynamicSampler` defined
    based on Higson et al. (2017). Parameters
    are passed to the function via `args`. This simply assigns
    each point a weight based on a weighted average of the
    posterior and evidence information content as::

        weight = pfrac * pweight + (1-pfrac) * zweight

    where the evidence weight is based on the estimated remaining prior
    volume and the posterior weight is simply the importance weight. The
    function returns a set of log-likelihood bounds set by the earliest/latest
    samples where `weight > maxfrac * max(weight)`, with left/right padding of
    `pad`.

    Parameters
    ----------
    results : `Results` instances
        `Results` instance.

    args : dictionary of keyword arguments, optional
        Arguments used to set the log-likelihood bounds used for sampling,
        as described above. Default values are `pfrac = 0.8`, `maxfrac = 0.8`,
        and `pad = 1`.

    return_weights : bool, optional
        Whether to return the individual weights (and their components) used
        to compute the log-likelihood bounds. Default is *False*.

    Returns
    -------
    logl_bounds : length-2 tuple
        Log-likelihood bounds `(logl_min, logl_max)` determined by the weights.

    weights : length-3 tuple, optional
        The individual weights `(pweight, zweight, weight)` used to determine
        `logl_bounds`.

    """

    # Initialize hyperparameters.
    pfrac = args.get('pfrac', 0.8)
    if not 0. <= pfrac <= 1.:
        raise ValueError("The provided `pfrac` {0} is not between 0. and 1."
                         .format(pfrac))
    maxfrac = args.get('maxfrac', 0.8)
    if not 0. <= maxfrac <= 1.:
        raise ValueError("The provided `maxfrac` {0} is not between 0. and 1."
                         .format(maxfrac))
    lpad = args.get('pad', 1)
    if lpad < 0:
        raise ValueError("`lpad` {0} is less than zero.".format(lpad))

    # Derive evidence weights.
    logz = results.logz  # final log(evidence)
    logz_remain = results.logl[-1] + results.logvol[-1]  # remainder
    logz_tot = np.logaddexp(logz[-1], logz_remain)  # estimated upper bound
    zin = np.exp(logz_tot) - np.exp(logz)  # remaining evidence
    zweight = zin / results.samples_n  # evidence weight
    zweight /= sum(zweight)  # normalize

    # Derive posterior weights.
    pweight = np.exp(results.logwt - results.logz[-1])  # importance weight
    pweight /= sum(pweight)  # normalize

    # Compute combined weights.
    weight = (1 - pfrac) * zweight + pfrac * pweight

    # Compute logl bounds
    nsamps = len(logz)
    bounds = np.arange(nsamps)[weight > maxfrac * max(weight)]
    bounds = (min(bounds) - lpad, min(max(bounds) + lpad, nsamps - 1))
    if bounds[0] < 0:
        logl_min = -np.inf
    else:
        logl_min = results.logl[bounds[0]]
    logl_max = results.logl[bounds[1]]

    if return_weights:
        return (logl_min, logl_max), (pweight, zweight, weight)
    else:
        return (logl_min, logl_max)


def stopping_function(results, args, M=map, return_vals=False):
    """
    The default stopping function utilized by `DynamicSampler`. Parameters
    are passed to the function via `args`. This assigns the run a stopping
    value based on `n_mc` realizations of the input run that represents
    a weighted average of the stopping values for the posterior and evidence::

        stop = pfrac * stop_post + (1-pfrac) * stop_evid

    The evidence stopping value is based on the estimated evidence error
    relative to a given threshold::

        stop_evid = evid_std / evid_thresh

    The posterior stopping value is based on the fractional variation
    in the Kullback-Leibler (KL) divergence relative to a given threshold::

        stop_post = (kld_std / kld_mean) / post_thresh

    The function returns the boolean `stop <= 1` used to decide when to stop
    the run.

    Parameters
    ----------
    results : `Results` instances
        `Results` instance.

    args : dictionary of keyword arguments, optional
        Arguments used to set the stopping values. In addition to the values
        outlined above, users can also choose the *type* of realizations used
        to compute quantities via the `'error'` keyword (choices are 'jitter'
        and 'simulate'). Default values are `pfrac = 1.0`, `evid_thresh = 0.1`,
        `post_thresh = 0.025`, `n_mc = 32`, and `error = 'simulate'`.

    return_vals : bool, optional
        Whether to return the stopping value (and its components). Default
        is *False*.

    Returns
    -------
    stop_flag : bool
        Boolean flag indicating whether we have passed the desired stopping
        criteria.

    stop_vals : length-3 tuple, optional
        The individual stopping values `(stop_post, stop_evid, weight)` used
        to determine the stopping criteria.

    """

    # Initialize hyperparameters.
    pfrac = args.get('pfrac', 1.0)
    if not 0. <= pfrac <= 1.:
        raise ValueError("The provided `pfrac` {0} is not between 0. and 1."
                         .format(pfrac))
    evid_thresh = args.get('evid_thresh', 0.1)
    if pfrac < 1. and evid_thresh < 0.:
        raise ValueError("The provided `evid_thresh` {0} is not non-negative "
                         "even though `1 - pfrac` is {1}."
                         .format(evid_thresh, 1. - pfrac))
    post_thresh = args.get('post_thresh', 0.025)
    if pfrac > 0. and post_thresh < 0.:
        raise ValueError("The provided `post_thresh` {0} is not non-negative "
                         "even though `pfrac` is {1}."
                         .format(post_thresh, pfrac))
    n_mc = args.get('n_mc', 32)
    if n_mc <= 1:
        raise ValueError("The number of realizations {0} must be greater "
                         "than 1.".format(n_mc))
    elif n_mc < 20:
        warnings.warn("Using a small number of realizations might result in "
                      "noisy stopping value estimates.")
    error = args.get('error', 'simulate')
    if error not in {'jitter', 'simulate'}:
        raise ValueError("The chosen `'error'` option {0} is not valid."
                         .format(noise))

    # Compute realizations of ln(evidence) and the KL divergence.
    rlist = [results for i in range(n_mc)]
    error_list = [error for i in range(n_mc)]
    return_options = [True for i in range(n_mc)]
    outputs = M(kld_error, rlist, error_list, return_options)
    kld_arr, lnz_arr = np.array([(kld[-1], res.logz[-1])
                                 for kld, res in outputs]).T

    # Evidence stopping value.
    lnz_std = np.std(lnz_arr)
    stop_evid = lnz_std / evid_thresh

    # Posterior stopping value.
    kld_mean, kld_std = np.mean(kld_arr), np.std(kld_arr)
    stop_post = (kld_std / kld_mean) / post_thresh

    # Effective stopping value.
    stop = pfrac * stop_post + (1. - pfrac) * stop_evid

    if return_vals:
        return stop <= 1., (stop_post, stop_evid, stop)
    else:
        return stop <= 1.


class DynamicSampler(object):
    """
    A dynamic nested sampler that allocates live points adaptively during
    a single run until a specified stopping criteria is reached.


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

    choose_method : function
        A function that takes information on the run and returns an
        associated bounding and sampling method.

    update_interval : int
        Only update the proposal distribution every
        `update_interval * nlive`-th likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: pool
        Use this pool of workers to propose live points in parallel.


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

    """

    def __init__(self, loglikelihood, prior_transform, npdim,
                 bound, method, update_interval, rstate, queue_size,
                 pool, kwargs):
        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.npdim = npdim

        # bounding/sampling
        self.bound = bound
        self.method = method
        self.update_interval = update_interval

        # extra arguments
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap', 20)
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        self.vol_dec = kwargs.get('vol_dec', 0.5)
        self.vol_check = kwargs.get('vol_check', 2.0)
        self.walks = self.kwargs.get('walks', 25)
        self.slices = self.kwargs.get('slices', 3)

        # random state
        self.rstate = rstate

        # parallelism
        self.queue_size = queue_size
        self.pool = pool
        if self.pool is None:
            self.M = map
        else:
            self.M = pool.map
        self.queue = []  # proposed live point queue
        self.nqueue = 0  # current size of the queue
        self.unused = 0  # total number of proposals unused
        self.used = 0  # total number of proposals used

        # sampling details
        self.it = 1  # number of iterations
        self.batch = 0  # number of batches allocated dynamically
        self.ncall = 0  # number of function calls
        self.prop = []  # initial states used to compute proposals
        self.eff = 1.  # sampling efficiency
        self.base = False  # base run complete

        # results
        self.saved_id = []  # live point labels
        self.saved_u = []  # unit cube samples
        self.saved_v = []  # transformed variable samples
        self.saved_logl = []  # loglikelihoods of samples
        self.saved_logvol = []  # expected log(volume)
        self.saved_logwt = []  # log(weights)
        self.saved_logz = []  # cumulative log(evidence)
        self.saved_logzvar = []  # cumulative error on log(evidence)
        self.saved_h = []  # cumulative information
        self.saved_nc = []  # number of calls at each iteration
        self.saved_propidx = []  # index of proposal dead point was drawn from
        self.saved_it = []  # iteration the live (now dead) point was proposed
        self.saved_n = []  # number of live points interior to dead point
        self.saved_piter = []  # active proposal at a specific iteration
        self.saved_scale = []  # scale factor at each iteration
        self.saved_batch = []  # live point batch ID
        self.saved_batch_nlive = []  # number of live points added in batch
        self.saved_batch_bounds = []  # loglikelihood bounds used in batch

        # results from our base run
        self.base_id = []
        self.base_u = []
        self.base_v = []
        self.base_logl = []
        self.base_logvol = []
        self.base_logwt = []
        self.base_logz = []
        self.base_logzvar = []
        self.base_h = []
        self.base_nc = []
        self.base_propidx = []
        self.base_it = []
        self.base_n = []
        self.base_piter = []
        self.base_scale = []

        # results from our most recent addition
        self.new_id = []
        self.new_u = []
        self.new_v = []
        self.new_logl = []
        self.new_nc = []
        self.new_it = []
        self.new_n = []
        self.new_propidx = []
        self.new_piter = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

    def reset(self):
        """Re-initialize the sampler."""

        # parallelism
        self.queue = []
        self.nqueue = 0
        self.unused = 0
        self.used = 0

        # sampling
        self.it = 1
        self.batch = 0
        self.ncall = 0
        self.prop = []
        self.eff = 1.
        self.base = False

        # results
        self.saved_id = []
        self.saved_u = []
        self.saved_v = []
        self.saved_logl = []
        self.saved_logvol = []
        self.saved_logwt = []
        self.saved_logz = []
        self.saved_logzvar = []
        self.saved_h = []
        self.saved_nc = []
        self.saved_propidx = []
        self.saved_it = []
        self.saved_n = []
        self.saved_piter = []
        self.saved_scale = []
        self.saved_batch = []
        self.saved_batch_nlive = []
        self.saved_batch_bounds = []

        # results from our base run
        self.base_id = []
        self.base_u = []
        self.base_v = []
        self.base_logl = []
        self.base_logvol = []
        self.base_logwt = []
        self.base_logz = []
        self.base_logzvar = []
        self.base_h = []
        self.base_nc = []
        self.base_propidx = []
        self.base_it = []
        self.base_n = []
        self.base_piter = []
        self.base_scale = []

        # results from our most recent addition
        self.new_id = []
        self.new_u = []
        self.new_v = []
        self.new_logl = []
        self.new_nc = []
        self.new_it = []
        self.new_n = []
        self.new_propidx = []
        self.new_piter = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

    @property
    def results(self):
        """The full results from the dynamic nested sampling run."""

        results = [('niter', self.it - 1),
                   ('ncall', np.array(self.saved_nc)),
                   ('eff', self.eff),
                   ('samples', np.array(self.saved_v)),
                   ('samples_id', np.array(self.saved_id)),
                   ('samples_batch', np.array(self.saved_batch, dtype='int')),
                   ('samples_it', np.array(self.saved_it)),
                   ('samples_u', np.array(self.saved_u)),
                   ('samples_n', np.array(self.saved_n)),
                   ('logwt', np.array(self.saved_logwt)),
                   ('logl', np.array(self.saved_logl)),
                   ('logvol', np.array(self.saved_logvol)),
                   ('logz', np.array(self.saved_logz)),
                   ('logzerr', np.sqrt(np.array(self.saved_logzvar))),
                   ('scale', np.array(self.saved_scale)),
                   ('h', np.array(self.saved_h)),
                   ('prop', copy.deepcopy(self.prop)),
                   ('prop_iter', np.array(self.saved_piter, dtype='int')),
                   ('samples_prop', np.array(self.saved_propidx, dtype='int')),
                   ('batch_nlive', np.array(self.saved_batch_nlive,
                                            dtype='int')),
                   ('batch_bounds', np.array(self.saved_batch_bounds))]

        return Results(results)

    def sample_initial(self, nlive=100, update_interval=None, maxiter=None,
                       maxcall=None, dlogz=0.01, live_points=None):
        """
        Generate a series of initial samples from a nested sampling
        run using a fixed number of live points. Instantiates a
        generator object that will be called by the user.

        Parameters
        ----------
        nlive : int, optional
            The number of live points to use for the baseline nested
            sampling run. Default is *100*.

        update_interval : int or float, optional
            If an integer is passed, only update the proposal distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the proposal after every `round(update_interval * nlive)`-th
            likelihood call. Larger update intervals can be more efficient
            when the likelihood function is quick to evaluate. If no value is
            provided, defaults to the value passed during initialization.

        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `log(z + z_est) - log(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. The default is
            *0.01*.

        live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
            A set of live points used to initialize the nested sampling run.
            Contains `live_u`, the coordinates on the unit cube, `live_v`, the
            transformed variables, and `live_logl`, the associated
            loglikelihoods. By default, if these are not provided the initial
            set of live points will be drawn from the unit `npdim`-cube.
            **WARNING: It is crucial that the initial set of live points have
            been sampled from the prior. Failure to provide a set of valid
            live points will result in biased results.**

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : double
            Ln(likelihood) of the sample.

        logvol : double
            Ln(volume) of the prior contained within the sample.

        logwt : double
            Ln(weight) of the sample.

        logz : double
            Cumulative ln(evidence) up to the sample (inclusive).

        logzvar : double
            Associated error on `logz`.

        h : double
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before a new proposed
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        n : int
            Number of live points at the current iteration.

        """

        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize

        self.reset()

        # Initialize the first set of live points.
        if live_points is None:
            self.nlive_init = nlive
            self.live_u = self.rstate.rand(self.nlive_init, self.npdim)
            self.live_v = self.M(self.prior_transform, self.live_u)
            self.live_logl = self.M(self.loglikelihood, self.live_v)
        else:
            self.nlive_init = len(live_points[0])

        # Convert all `-np.inf` log-likelihoods to finite large numbers.
        # Necessary to keep estimators in our sampler from breaking.
        for i, logl in enumerate(self.live_logl):
            if not np.isfinite(logl):
                if np.sign(logl) < 0:
                    self.live_logl[i] = -1e300
                else:
                    raise ValueError("The log-likelihood ({0}) of live point "
                                     "{1} located at u={2} v={3} is invalid."
                                     .format(logl, i, self.live_u[i],
                                             self.live_v[i]))

        live_points = [self.live_u, self.live_v, self.live_logl]
        self.live_init = [np.array(l) for l in live_points]
        self.ncall += self.nlive_init
        self.live_prop = np.zeros(self.nlive_init, dtype='int')
        self.live_it = np.zeros(self.nlive_init, dtype='int')

        # Initialize the sampler.
        if update_interval is None:
            update_interval = self.update_interval
        if isinstance(update_interval, float):
            update_interval = int(round(self.update_interval * nlive))

        bound = self.bound
        self.sampler = _SAMPLERS[bound](self.loglikelihood,
                                        self.prior_transform,
                                        self.npdim, self.live_init,
                                        self.method, update_interval,
                                        self.rstate, self.queue_size,
                                        self.pool, self.kwargs)
        self.prop = self.sampler.prop

        # Run the sampler internally as a generator.
        for i in range(1):
            for it, results in enumerate(self.sampler.sample(maxiter=maxiter,
                                         save_samples=False,
                                         maxcall=maxcall, dlogz=dlogz)):
                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt, logz,
                 logzvar, h, nc, worst_it, propidx, eff, delta_logz) = results

                # Save our base run (which we will use later).
                self.base_id.append(worst)
                self.base_u.append(ustar)
                self.base_v.append(vstar)
                self.base_logl.append(loglstar)
                self.base_logvol.append(logvol)
                self.base_logwt.append(logwt)
                self.base_logz.append(logz)
                self.base_logzvar.append(logzvar)
                self.base_h.append(h)
                self.base_nc.append(nc)
                self.base_it.append(worst_it)
                self.base_n.append(self.nlive_init)
                self.base_propidx.append(propidx)
                self.base_piter.append(self.sampler.nprop - 1)
                self.base_scale.append(self.sampler.scale)

                # Save a copy of the results.
                self.saved_id.append(worst)
                self.saved_u.append(ustar)
                self.saved_v.append(vstar)
                self.saved_logl.append(loglstar)
                self.saved_logvol.append(logvol)
                self.saved_logwt.append(logwt)
                self.saved_logz.append(logz)
                self.saved_logzvar.append(logzvar)
                self.saved_h.append(h)
                self.saved_nc.append(nc)
                self.saved_it.append(worst_it)
                self.saved_n.append(self.nlive_init)
                self.saved_propidx.append(propidx)
                self.saved_piter.append(self.sampler.nprop - 1)
                self.saved_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.ncall += nc
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, logvol, logwt, logz,
                       logzvar, h, nc, worst_it, propidx, self.eff, delta_logz)

            for it, results in enumerate(self.sampler.add_live_points()):
                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt, logz,
                 logzvar, h, nc, worst_it, propidx, eff, delta_logz) = results

                # Save our base run (which we will use later).
                self.base_id.append(worst)
                self.base_u.append(ustar)
                self.base_v.append(vstar)
                self.base_logl.append(loglstar)
                self.base_logvol.append(logvol)
                self.base_logwt.append(logwt)
                self.base_logz.append(logz)
                self.base_logzvar.append(logzvar)
                self.base_h.append(h)
                self.base_nc.append(nc)
                self.base_it.append(worst_it)
                self.base_n.append(self.nlive_init - it)
                self.base_propidx.append(propidx)
                self.base_piter.append(self.sampler.nprop - 1)
                self.base_scale.append(self.sampler.scale)

                # Save a copy of the results.
                self.saved_id.append(worst)
                self.saved_u.append(ustar)
                self.saved_v.append(vstar)
                self.saved_logl.append(loglstar)
                self.saved_logvol.append(logvol)
                self.saved_logwt.append(logwt)
                self.saved_logz.append(logz)
                self.saved_logzvar.append(logzvar)
                self.saved_h.append(h)
                self.saved_nc.append(nc)
                self.saved_it.append(worst_it)
                self.saved_n.append(self.nlive_init - it)
                self.saved_propidx.append(propidx)
                self.saved_piter.append(self.sampler.nprop - 1)
                self.saved_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, logvol, logwt, logz,
                       logzvar, h, nc, worst_it, propidx, self.eff, delta_logz)

        self.base = True  # baseline run complete
        self.saved_batch = np.zeros(len(self.saved_id), dtype='int')  # batch
        self.saved_batch_nlive.append(self.nlive_init)  # initial nlive
        self.saved_batch_bounds.append((-np.inf, np.inf))  # initial bounds

    def sample_batch(self, nlive_new=100, update_interval=None,
                     logl_bounds=None, maxiter=None, maxcall=None,
                     save_proposals=True):
        """
        Generate an additional series of nested samples to be added to
        the previous set of dead points. Instantiates a generator object
        that will be called by the user.

        Parameters
        ----------
        nlive_new : int
            Number of new live points to be added.

        update_interval : int or float, optional
            If an integer is passed, only update the proposal distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the proposal after every `round(update_interval * nlive)`-th
            likelihood call. Larger update intervals can be more efficient
            when the likelihood function is quick to evaluate. If no value is
            provided, defaults to the value passed during initialization.

        logl_bounds : tuple of 2 floats, optional
            The ln(likelihood) bounds used to bracket the run. If *None*,
            the default is to span the entire range covered by the
            original run.

        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        save_proposals : bool, optional
            Whether or not to save past proposal distributions used to bound
            the live points internally. Default is *True*.

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : double
            Ln(likelihood) of the sample.

        nc : int
            Number of likelihood calls performed before a new proposed
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        """

        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize

        # Internal sampler should always dump them to avoid wasting memory.
        self.sampler.save_proposals = save_proposals

        # Initialize values.
        h = 0.0  # Information, initially *0.*
        logz = -1.e300  # log(evidence), initially *0.*
        logvol = 0.  # initially contains the whole prior (volume=1.)

        # Grab results from base run.
        base_id = np.array(self.base_id)
        base_u = np.array(self.base_u)
        base_v = np.array(self.base_v)
        base_logl = np.array(self.base_logl)
        base_nc = np.array(self.base_nc)
        base_propidx = np.array(self.base_propidx)
        base_it = np.array(self.base_it)
        base_n = np.array(self.base_n)
        base_piter = np.array(self.base_piter)
        base_scale = np.array(self.base_scale)
        nbase = len(base_n)
        nblive = self.nlive_init

        # Reset "new" results.
        self.new_id = []
        self.new_u = []
        self.new_v = []
        self.new_logl = []
        self.new_nc = []
        self.new_it = []
        self.new_n = []
        self.new_propidx = []
        self.new_piter = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

        # Initialize ln(likelihood) bounds.
        if logl_bounds is None:
            logl_min, logl_max = -np.inf, max(base_logl[:-nblive])
        else:
            logl_min, logl_max = logl_bounds
        self.new_logl_min, self.new_logl_max = logl_min, logl_max

        psel = np.all(logl_min <= base_logl)
        vol = 1. - 1. / nblive
        if psel:
            # If the lower bound encompasses all live points, we want
            # to propose a new set of points from the unit cube.
            live_u = self.rstate.rand(nlive_new, self.npdim)  # unit cube
            live_v = self.M(self.prior_transform, live_u)  # real parameters
            live_logl = self.M(self.loglikelihood, live_v)  # log likelihood
            live_prop = np.zeros(nlive_new, dtype='int')  # unit cube
            live_it = np.zeros(nlive_new, dtype='int') + self.it
            live_nc = np.ones(nlive_new, dtype='int')
            self.ncall += nlive_new
        else:
            # Rewind our previous run until we arrive at the relevant
            # set of live points (and scale).
            live_u = np.empty((nblive, self.npdim))
            live_v = np.empty((nblive, self.npdim))
            live_logl = np.empty(nblive)
            live_u[base_id[-nblive:]] = base_u[-nblive:]
            live_v[base_id[-nblive:]] = base_v[-nblive:]
            live_logl[base_id[-nblive:]] = base_logl[-nblive:]
            for i in range(1, nbase - nblive):
                r = -(nblive + i)
                uidx = base_id[r]
                live_u[uidx] = base_u[r]
                live_v[uidx] = base_v[r]
                live_logl[uidx] = base_logl[r]
                if live_logl[uidx] <= logl_min:
                    break
            live_scale = base_scale[r]

            # Overwrite the live points and scale factor of our internal
            # sampler and trigger an update of our bound.
            self.sampler.nlive = nblive
            self.sampler.live_u = np.array(live_u)
            self.sampler.live_v = np.array(live_v)
            self.sampler.live_logl = np.array(live_logl)
            self.sampler.scale = live_scale
            vol = math.exp(- 1. * (nbase + r) / nblive)
            prop = self.sampler.update(vol / nblive)
            if save_proposals:
                self.sampler.prop.append(copy.deepcopy(prop))
            self.sampler.nprop += 1
            self.sampler.since_update = 0

            # Sample a new batch of `nlive_new` live points given our
            # `logl_min` constraint.
            live_u = np.empty((nlive_new, self.npdim))
            live_v = np.empty((nlive_new, self.npdim))
            live_logl = np.empty(nlive_new)
            live_prop = np.zeros(nlive_new, dtype='int')
            live_prop += self.sampler.nprop - 1
            live_it = np.empty(nlive_new, dtype='int')
            live_nc = np.empty(nlive_new, dtype='int')
            for i in range(nlive_new):
                (live_u[i], live_v[i], live_logl[i],
                 live_nc[i]) = self.sampler._new_point(logl_min)
                live_it[i] = self.it
                self.ncall += live_nc[i]

        # Overwrite the live points of our previous sampler with the
        # new batch of points and trigger an update (again).
        self.sampler.nlive = nlive_new
        self.sampler.live_u = np.array(live_u)
        self.sampler.live_v = np.array(live_v)
        self.sampler.live_logl = np.array(live_logl)
        self.sampler.live_prop = np.array(live_prop)
        self.sampler.live_it = np.array(live_it)
        prop = self.sampler.update(vol / nlive_new)
        if save_proposals:
                self.sampler.prop.append(copy.deepcopy(prop))
        self.sampler.nprop += 1
        self.sampler.since_update = 0

        # Copy over proposal reference.
        self.prop = self.sampler.prop

        # Update `update_interval`.
        if update_interval is None:
            update_interval = self.update_interval
        if isinstance(update_interval, float):
            update_interval = int(round(self.update_interval * nlive_new))
        self.sampler.update_interval = update_interval

        # Tell the sampler *not* to try and remove the previous addition of
        # live points (the internal results are garbage anyways).
        self.sampler.added_live = False

        # Run the sampler internally as a generator until we hit
        # the lower likelihood threshold. Afterwards, we add in our remaining
        # live points as if we had terminated the run. This allows us to
        # sample past the original bounds "for free".
        for i in range(1):
            for it, results in enumerate(self.sampler.sample(dlogz=0.,
                                         logl_max=logl_max,
                                         maxiter=maxiter-nlive_new-1,
                                         maxcall=maxcall-sum(live_nc),
                                         save_samples=False,
                                         save_proposals=save_proposals)):

                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt, logz,
                 logzvar, h, nc, worst_it, propidx, eff, delta_logz) = results

                # Save results.
                self.new_id.append(worst)
                self.new_u.append(ustar)
                self.new_v.append(vstar)
                self.new_logl.append(loglstar)
                self.new_nc.append(nc)
                self.new_it.append(worst_it)
                self.new_n.append(nlive_new)
                self.new_propidx.append(propidx)
                self.new_piter.append(self.sampler.nprop - 1)
                self.new_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.ncall += nc
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, nc,
                       worst_it, propidx, self.eff)

            for it, results in enumerate(self.sampler.add_live_points()):
                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt, logz,
                 logzvar, h, nc, worst_it, propidx, eff, delta_logz) = results

                # Save results.
                self.new_id.append(worst)
                self.new_u.append(ustar)
                self.new_v.append(vstar)
                self.new_logl.append(loglstar)
                self.new_nc.append(live_nc[worst])
                self.new_it.append(worst_it)
                self.new_n.append(nlive_new - it)
                self.new_propidx.append(propidx)
                self.new_piter.append(self.sampler.nprop - 1)
                self.new_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, live_nc[worst],
                       worst_it, propidx, self.eff)

    def combine_runs(self):
        """ Merge the most recent run into the previous (combined) run by
        "stepping through" both runs simultaneously."""

        # Make sure we have a run to add.
        if len(self.new_id) == 0:
            raise ValueError("No new samples are currently saved.")

        # Grab results from saved run.
        saved_id = np.array(self.saved_id)
        saved_u = np.array(self.saved_u)
        saved_v = np.array(self.saved_v)
        saved_logl = np.array(self.saved_logl)
        saved_nc = np.array(self.saved_nc)
        saved_propidx = np.array(self.saved_propidx)
        saved_it = np.array(self.saved_it)
        saved_n = np.array(self.saved_n)
        saved_piter = np.array(self.saved_piter)
        saved_scale = np.array(self.saved_scale)
        saved_batch = np.array(self.saved_batch)
        nsaved = len(saved_n)

        # Grab results from new run.
        new_id = np.array(self.new_id) + max(saved_id) + 1
        new_u = np.array(self.new_u)
        new_v = np.array(self.new_v)
        new_logl = np.array(self.new_logl)
        new_nc = np.array(self.new_nc)
        new_propidx = np.array(self.new_propidx)
        new_it = np.array(self.new_it)
        new_n = np.array(self.new_n)
        new_piter = np.array(self.new_piter)
        new_scale = np.array(self.new_scale)
        nnew = len(new_n)
        llmin, llmax = self.new_logl_min, self.new_logl_max

        # Reset saved results.
        self.saved_id = []
        self.saved_u = []
        self.saved_v = []
        self.saved_logl = []
        self.saved_logvol = []
        self.saved_logwt = []
        self.saved_logz = []
        self.saved_logzvar = []
        self.saved_h = []
        self.saved_nc = []
        self.saved_propidx = []
        self.saved_it = []
        self.saved_n = []
        self.saved_piter = []
        self.saved_scale = []
        self.saved_batch = []

        # Start our counters at the beginning of each set of dead points.
        idx_saved, idx_new = 0, 0  # start of our dead points
        logl_s, logl_n = saved_logl[idx_saved], new_logl[idx_new]
        nlive_s, nlive_n = saved_n[idx_saved], new_n[idx_new]

        # Iteratively walk through both set of samples to simulate
        # a combined run.
        ntot = nsaved + nnew
        logvol = 0.
        for i in range(ntot):
            if logl_s > self.new_logl_min:
                # If our saved samples are past the lower log-likelihood
                # bound, both runs are now "active" and should be used.
                nlive = nlive_s + nlive_n
            else:
                # If instead our collection of dead points are below
                # the bound, just use our collection of saved samples.
                nlive = nlive_s
            # Increment our position along depending on
            # which dead point (saved or new) is worse.
            if logl_s <= logl_n:
                self.saved_id.append(saved_id[idx_saved])
                self.saved_u.append(saved_u[idx_saved])
                self.saved_v.append(saved_v[idx_saved])
                self.saved_logl.append(saved_logl[idx_saved])
                self.saved_nc.append(saved_nc[idx_saved])
                self.saved_propidx.append(saved_propidx[idx_saved])
                self.saved_it.append(saved_it[idx_saved])
                self.saved_piter.append(saved_piter[idx_saved])
                self.saved_scale.append(saved_scale[idx_saved])
                self.saved_batch.append(saved_batch[idx_saved])
                idx_saved += 1
            else:
                self.saved_id.append(new_id[idx_new])
                self.saved_u.append(new_u[idx_new])
                self.saved_v.append(new_v[idx_new])
                self.saved_logl.append(new_logl[idx_new])
                self.saved_nc.append(new_nc[idx_new])
                self.saved_propidx.append(new_propidx[idx_new])
                self.saved_it.append(new_it[idx_new])
                self.saved_piter.append(new_piter[idx_new])
                self.saved_scale.append(new_scale[idx_new])
                self.saved_batch.append(self.batch + 1)
                idx_new += 1

            # Save the number of live points and expected ln(volume).
            logvol -= math.log((nlive + 1.) / nlive)
            self.saved_n.append(nlive)
            self.saved_logvol.append(logvol)

            # Attempt to step along our samples. If we're out of samples,
            # set values to defaults.
            try:
                logl_s = saved_logl[idx_saved]
                nlive_s = saved_n[idx_saved]
            except:
                logl_s = np.inf
                nlive_s = 0
            try:
                logl_n = new_logl[idx_new]
                nlive_n = new_n[idx_new]
            except:
                logl_n = np.inf
                nlive_n = 0

        # Compute the posterior quantities of interest.
        h = 0.
        logz = -1.e300
        loglstar = -1.e300
        logzvar = 0.
        logvols_pad = np.concatenate(([0.], self.saved_logvol))
        logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                                  axis=1, b=np.c_[np.ones(ntot),
                                                  -np.ones(ntot)])
        logdvols += math.log(0.5)
        dlvs = logvols_pad[:-1] - logvols_pad[1:]
        for i in range(ntot):
            loglstar_new = self.saved_logl[i]
            logdvol, dlv = logdvols[i], dlvs[i]
            logwt = np.logaddexp(loglstar_new, loglstar) + logdvol
            logz_new = np.logaddexp(logz, logwt)
            lzterm = (math.exp(loglstar - logz_new) * loglstar +
                      math.exp(loglstar_new - logz_new) * loglstar_new)
            h_new = (math.exp(logdvol) * lzterm +
                     math.exp(logz - logz_new) * (h + logz) -
                     logz_new)
            dh = h_new - h
            h = h_new
            logz = logz_new
            logzvar += dh * dlv
            loglstar = loglstar_new
            self.saved_logwt.append(logwt)
            self.saved_logz.append(logz)
            self.saved_logzvar.append(logzvar)
            self.saved_h.append(h)

        # Reset results.
        self.new_id = []
        self.new_u = []
        self.new_v = []
        self.new_logl = []
        self.new_nc = []
        self.new_it = []
        self.new_n = []
        self.new_propidx = []
        self.new_piter = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

        # Increment batch counter.
        self.batch += 1

        # Saved batch quantities.
        self.saved_batch_nlive.append(max(new_n))
        self.saved_batch_bounds.append((llmin, llmax))

    def run_nested(self, nlive_init=100, maxiter_init=None,
                   maxcall_init=None, dlogz_init=0.01,
                   nlive_batch=100, wt_function=None, wt_kwargs=None,
                   maxiter_batch=None, maxcall_batch=None,
                   maxiter=None, maxcall=None, maxbatch=None,
                   stop_function=None, stop_kwargs=None, use_stop=True,
                   save_proposals=True, print_progress=True, live_points=None):
        """
        Dynamically allocate (nested) samples to optimize a target
        weight function until a specified stopping criterion is reached.

        Parameters
        ----------
        nlive_init : int, optional
            The number of live points used during the initial ("baseline")
            nested sampling run. Default is *100*.

        maxiter_init : int, optional
            Maximum number of iterations for the initial baseline nested
            sampling run. Iteration may stop earlier if the
            termination condition is reached. Default is no limit.

        maxcall_init : int, optional
            Maximum number of likelihood evaluations for the initial
            baseline nested sampling run. Iteration may stop earlier
            if the termination condition is reached. Default is no limit.

        dlogz_init : float, optional
            The baseline run will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `log(z + z_est) - log(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. The default is
            *0.01*.

        nlive_batch : int, optional
            The number of live points used when adding additional samples
            from a nested sampling run within each batch. Default is *100*.

        wt_function : func, optional
            A cost function that takes a `Results` dictionary instance
            and returns a log-likelihood range over which a new batch of
            samples should be generated. The default function simply
            computes a weighted average of the posterior and evidence
            information content as::

                weight = (1 - pfrac) * zweight + pfrac * pweight

        wt_kwargs : dict, optional
            Extra arguments to be passed to the weight function.

        maxiter_batch : int, optional
            Maximum number of iterations for the nested
            sampling run within each batch. Iteration may stop earlier
            if the termination condition is reached. Default is no limit.

        maxcall_batch : int, optional
            Maximum number of likelihood evaluations for the nested
            sampling run within each batch. Iteration may stop earlier
            if the termination condition is reached. Default is no limit.

        maxiter : int, optional
            Maximum number of iterations allowed. Default is no limit.

        maxcall : int, optional
            Maximum number of likelihood evaluations allowed.
            Default is no limit.

        maxbatch : int, optional
            Maximum number of batches allowed. Default is no limit.

        stop_function : func, optional
            A function that takes a `Results` dictionary instance and
            returns a boolean indicating that we should terminate the run
            because we've collected enough samples.

        stop_kwargs : float, optional
            Extra arguments to be passed to the stopping function.

        use_stop : bool, optional
            Whether to evaluate our stopping function after each batch.
            Disabling this can improve performance if other stopping criteria
            are already specified. Default is *True*.

        save_proposals : bool, optional
            Whether or not to save past proposal distributions used to bound
            the live points internally during dynamic live point additions.
            Default is *True*.

        print_progress : bool, optional
            If *True*, outputs a simple summary of the current run that
            updates each iteration. Default is *True*.

        live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
            A set of live points used to initialize the nested sampling run.
            Contains `live_u`, the coordinates on the unit cube, `live_v`, the
            transformed variables, and `live_logl`, the associated
            loglikelihoods. By default, if these are not provided the initial
            set of live points will be drawn from the unit `npdim`-cube.
            **WARNING: It is crucial that the initial set of live points have
            been sampled from the prior. Failure to provide a set of valid
            live points will result in biased results.**

        """

        # Initialize values.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        if maxiter_batch is None:
            maxiter_batch = sys.maxsize
        if maxcall_batch is None:
            maxcall_batch = sys.maxsize
        if maxbatch is None:
            maxbatch = sys.maxsize
        if maxiter_init is None:
            maxiter_init = sys.maxsize
        if maxcall_init is None:
            maxcall_init = sys.maxsize
        if wt_function is None:
            wt_function = weight_function
        if wt_kwargs is None:
            wt_kwargs = dict()
        if stop_function is None:
            stop_function = stopping_function
        if stop_kwargs is None:
            stop_kwargs = dict()
        saveprop = save_proposals

        # Run the main dynamic nested sampling loop.
        ncall = self.ncall
        niter = self.it - 1
        maxcall_init = min(maxcall_init, maxcall)  # set max calls
        maxiter_init = min(maxiter_init, maxiter)  # set max iterations

        # Baseline run.
        if not self.base:
            for results in self.sample_initial(nlive=nlive_init,
                                               dlogz=dlogz_init,
                                               maxcall=maxcall_init,
                                               maxiter=maxiter_init,
                                               live_points=live_points):
                (worst, ustar, vstar, loglstar, logvol,
                 logwt, logz, logzvar, h, nc, worst_it,
                 propidx, eff, delta_logz) = results
                if delta_logz > 1e6:
                    delta_logz = np.inf
                ncall += nc
                niter += 1
                if print_progress:
                    logzerr = np.sqrt(logzvar)
                    sys.stderr.write("\riter: {:d} | batch: {:d} | nc: {:d} | "
                                     "ncall: {:d} | eff(%): {:6.3f} | "
                                     "logz: {:6.3f} +/- {:6.3f} | "
                                     "dlogz: {:6.3f} > {:6.3f}    "
                                     .format(niter, 0, nc, ncall, eff, logz,
                                             logzerr, delta_logz, dlogz_init))

        # Add points in batches.
        for n in range(self.batch, maxbatch):
            # Update stopping criteria.
            res = self.results
            mcall = min(maxcall - ncall, maxcall_batch)
            miter = min(maxiter - niter, maxiter_batch)
            if use_stop:
                stop, stop_vals = stop_function(res, stop_kwargs, M=self.M,
                                                return_vals=True)
                stop_post, stop_evid, stop_val = stop_vals
            else:
                stop = False
                stop_val = np.NaN

            # If we have either likelihood calls or iterations remaining,
            # run our batch.
            if mcall > 0 and miter > 0 and not stop:
                # Compute our sampling bounds using the provided
                # weight function.
                logl_bounds = wt_function(res, wt_kwargs)
                lnz, lnzerr = res.logz[-1], res.logzerr[-1]
                for results in self.sample_batch(nlive_new=nlive_batch,
                                                 logl_bounds=logl_bounds,
                                                 maxiter=miter,
                                                 maxcall=mcall,
                                                 save_proposals=saveprop):
                    (worst, ustar, vstar, loglstar, nc,
                     worst_it, propidx, eff) = results
                    ncall += nc
                    niter += 1
                    if print_progress:
                        sys.stderr.write("\riter: {:d} | batch: {:d} | "
                                         "nc: {:d} | ncall: {:d} | "
                                         "eff(%): {:6.3f} | "
                                         "loglstar: {:6.3f} < {:6.3f} "
                                         "< {:6.3f} | "
                                         "logz: {:6.3f} +/- {:6.3f} | "
                                         "stop: {:6.3f}    "
                                         .format(niter, n+1, nc, ncall,
                                                 eff, logl_bounds[0], loglstar,
                                                 logl_bounds[1], lnz, lnzerr,
                                                 stop_val))
                self.combine_runs()
            else:
                # We're done!
                break

        if print_progress:
            sys.stderr.write("\n")

    def add_batch(self, nlive_batch=100, wt_function=None, wt_kwargs=None,
                  maxiter=None, maxcall=None, save_proposals=True,
                  print_progress=True):
        """
        Allocate an additional batch of (nested) samples based on
        the set of previous samples using the specified weight function.

        Parameters
        ----------
        nbatch : int, optional
            The number of live points used when adding additional samples
            from a nested sampling run within each batch. Default is *100*.

        wt_function : func, optional
            A cost function that takes a `Results` dictionary instance
            and returns a log-likelihood range over which a new batch of
            samples should be generated. The default function simply
            computes a weighted average of the posterior and evidence
            information content as::

                weight = (1-pweight) * w_evid + pweight * w_post

        wt_kwargs : dict, optional
            Extra arguments to be passed to the weight function.

        maxiter : int, optional
            Maximum number of iterations allowed. Default is no limit.

        maxcall : int, optional
            Maximum number of likelihood evaluations allowed.
            Default is no limit.

        save_proposals : bool, optional
            Whether or not to save past proposal distributions used to bound
            the live points internally during dynamic live point additions.
            Default is *True*.

        print_progress : bool, optional
            If *True*, outputs a simple summary of the current run that
            updates each iteration. Default is *True*.

        """

        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        if wt_function is None:
            wt_function = weight_function
        if wt_kwargs is None:
            wt_kwargs = dict()

        # If we have either likelihood calls or iterations remaining,
        # add our new batch of live points.
        ncall, niter, n = self.ncall, self.it - 1, self.batch
        if maxcall > 0 and maxiter > 0:
            # Compute our sampling bounds using the provided
            # weight function.
            res = self.results
            lnz, lnzerr = res.logz[-1], res.logzerr[-1]
            logl_bounds = wt_function(res, wt_kwargs)
            for results in self.sample_batch(nlive_new=nlive_batch,
                                             logl_bounds=logl_bounds,
                                             maxiter=maxiter,
                                             maxcall=maxcall,
                                             save_proposals=save_proposals):
                (worst, ustar, vstar, loglstar, nc,
                 worst_it, propidx, eff) = results
                ncall += nc
                niter += 1
                if print_progress:
                    sys.stderr.write("\riter: {:d} | batch: {:d} | "
                                     "nc: {:d} | ncall: {:d} | "
                                     "eff(%): {:6.3f} | "
                                     "loglstar: {:6.3f} < {:6.3f} "
                                     "< {:6.3f} | "
                                     "logz: {:6.3f} +/- {:6.3f}    "
                                     .format(niter, n+1, nc, ncall,
                                             eff, logl_bounds[0], loglstar,
                                             logl_bounds[1], lnz, lnzerr))
            self.combine_runs()
