#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the dynamic nested sampler class :class:`DynamicSampler` used to
dynamically allocate nested samples. Note that :class:`DynamicSampler`
implicitly wraps a sampler from :mod:`~dynesty.nestedsamplers`. Also contains
the weight function :meth:`weight_function` and stopping function
:meth:`stopping_function`. These are used by default within
:class:`DynamicSampler` if corresponding functions are not provided
by the user.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import warnings
import math
import numpy as np
import copy
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp


from .nestedsamplers import (UnitCubeSampler, SingleEllipsoidSampler,
                             MultiEllipsoidSampler, RadFriendsSampler,
                             SupFriendsSampler)
from .sampling import (sample_unif, sample_rwalk, sample_rstagger,
                       sample_slice, sample_rslice, sample_hslice)
from .results import Results, print_fn
from .utils import kld_error

__all__ = ["DynamicSampler", "weight_function", "stopping_function",
           "_kld_error"]

_SAMPLERS = {'none': UnitCubeSampler,
             'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler,
             'balls': RadFriendsSampler,
             'cubes': SupFriendsSampler}
_SAMPLING = {'unif': sample_unif,
             'rwalk': sample_rwalk,
             'rstagger': sample_rstagger,
             'slice': sample_slice,
             'rslice': sample_rslice,
             'hslice': sample_hslice}

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def _kld_error(args):
    """ Internal `pool.map`-friendly wrapper for :meth:`kld_error` used by
    :meth:`stopping_function`."""

    # Extract arguments.
    results, error, approx = args

    return kld_error(results, error, rstate=np.random, return_new=True,
                     approx=approx)


def weight_function(results, args=None, return_weights=False):
    """
    The default weight function utilized by :class:`DynamicSampler`.
    Zipped parameters are passed to the function via :data:`args`.
    Assigns each point a weight based on a weighted average of the
    posterior and evidence information content::

        weight = pfrac * pweight + (1. - pfrac) * zweight

    where `pfrac` is the fractional importance placed on the posterior,
    the evidence weight `zweight` is based on the estimated remaining
    posterior mass, and the posterior weight `pweight` is the sample's
    importance weight.

    Returns a set of log-likelihood bounds set by the earliest/latest
    samples where `weight > maxfrac * max(weight)`, with additional
    left/right padding based on `pad`.

    Parameters
    ----------
    results : :class:`Results` instance
        :class:`Results` instance.

    args : dictionary of keyword arguments, optional
        Arguments used to set the log-likelihood bounds used for sampling,
        as described above. Default values are `pfrac = 0.8`, `maxfrac = 0.8`,
        and `pad = 1`.

    return_weights : bool, optional
        Whether to return the individual weights (and their components) used
        to compute the log-likelihood bounds. Default is `False`.

    Returns
    -------
    logl_bounds : tuple with shape (2,)
        Log-likelihood bounds `(logl_min, logl_max)` determined by the weights.

    weights : tuple with shape (3,), optional
        The individual weights `(pweight, zweight, weight)` used to determine
        `logl_bounds`.

    """

    # Initialize hyperparameters.
    if args is None:
        args = dict({})
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
    logz = results.logz  # final ln(evidence)
    logz_remain = results.logl[-1] + results.logvol[-1]  # remainder
    logz_tot = np.logaddexp(logz[-1], logz_remain)  # estimated upper bound
    lzones = np.ones_like(logz)
    logzin = logsumexp([lzones * logz_tot, logz], axis=0,
                       b=[lzones, -lzones])  # ln(remaining evidence)
    logzweight = logzin - np.log(results.samples_n)  # ln(evidence weight)
    logzweight -= logsumexp(logzweight)  # normalize
    zweight = np.exp(logzweight)  # convert to linear scale

    # Derive posterior weights.
    pweight = np.exp(results.logwt - results.logz[-1])  # importance weight
    pweight /= sum(pweight)  # normalize

    # Compute combined weights.
    weight = (1. - pfrac) * zweight + pfrac * pweight

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


def stopping_function(results, args=None, rstate=None, M=None,
                      return_vals=False):
    """
    The default stopping function utilized by :class:`DynamicSampler`.
    Zipped parameters are passed to the function via :data:`args`.
    Assigns the run a stopping value based on a weighted average of the
    stopping values for the posterior and evidence::

        stop = pfrac * stop_post + (1.- pfrac) * stop_evid

    The evidence stopping value is based on the estimated evidence error
    (i.e. standard deviation) relative to a given threshold::

        stop_evid = evid_std / evid_thresh

    The posterior stopping value is based on the fractional error (i.e.
    standard deviation / mean) in the Kullback-Leibler (KL) divergence
    relative to a given threshold::

        stop_post = (kld_std / kld_mean) / post_thresh

    Estimates of the mean and standard deviation are computed using `n_mc`
    realizations of the input using a provided `'error'` keyword (either
    `'jitter'` or `'simulate'`, which call related functions :meth:`jitter_run`
    and :meth:`simulate_run` in :mod:`dynesty.utils`, respectively, or
    `'sim_approx'`, which boosts `'jitter'` by a factor of two).

    Returns the boolean `stop <= 1`. If `True`, the :class:`DynamicSampler`
    will stop adding new samples to our results.

    Parameters
    ----------
    results : :class:`Results` instance
        :class:`Results` instance.

    args : dictionary of keyword arguments, optional
        Arguments used to set the stopping values. Default values are
        `pfrac = 1.0`, `evid_thresh = 0.1`, `post_thresh = 0.02`,
        `n_mc = 128`, `error = 'sim_approx'`, and `approx = True`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    M : `map` function, optional
        An alias to a `map`-like function. This allows users to pass
        functions from pools (e.g., `pool.map`) to compute realizations in
        parallel. By default the standard `map` function is used.

    return_vals : bool, optional
        Whether to return the stopping value (and its components). Default
        is `False`.

    Returns
    -------
    stop_flag : bool
        Boolean flag indicating whether we have passed the desired stopping
        criteria.

    stop_vals : tuple of shape (3,), optional
        The individual stopping values `(stop_post, stop_evid, stop)` used
        to determine the stopping criteria.

    """

    # Initialize values.
    if args is None:
        args = dict({})
    if rstate is None:
        rstate = np.random
    if M is None:
        M = map

    # Initialize hyperparameters.
    pfrac = args.get('pfrac', 1.0)
    if not 0. <= pfrac <= 1.:
        raise ValueError("The provided `pfrac` {0} is not between 0. and 1."
                         .format(pfrac))
    evid_thresh = args.get('evid_thresh', 0.1)
    if pfrac < 1. and evid_thresh < 0.:
        raise ValueError("The provided `evid_thresh` {0} is not non-negative "
                         "even though `1. - pfrac` is {1}."
                         .format(evid_thresh, 1. - pfrac))
    post_thresh = args.get('post_thresh', 0.02)
    if pfrac > 0. and post_thresh < 0.:
        raise ValueError("The provided `post_thresh` {0} is not non-negative "
                         "even though `pfrac` is {1}."
                         .format(post_thresh, pfrac))
    n_mc = args.get('n_mc', 128)
    if n_mc <= 1:
        raise ValueError("The number of realizations {0} must be greater "
                         "than 1.".format(n_mc))
    elif n_mc < 20:
        warnings.warn("Using a small number of realizations might result in "
                      "excessively noisy stopping value estimates.")
    error = args.get('error', 'sim_approx')
    if error not in {'jitter', 'simulate', 'sim_approx'}:
        raise ValueError("The chosen `'error'` option {0} is not valid."
                         .format(error))
    if error == 'sim_approx':
        error = 'jitter'
        boost = 2.
    else:
        boost = 1.
    approx = args.get('approx', True)

    # Compute realizations of ln(evidence) and the KL divergence.
    rlist = [results for i in range(n_mc)]
    error_list = [error for i in range(n_mc)]
    approx_list = [approx for i in range(n_mc)]
    args = zip(rlist, error_list, approx_list)
    outputs = list(M(_kld_error, args))
    kld_arr, lnz_arr = np.array([(kld[-1], res.logz[-1])
                                 for kld, res in outputs]).T

    # Evidence stopping value.
    lnz_std = np.std(lnz_arr)
    stop_evid = np.sqrt(boost) * lnz_std / evid_thresh

    # Posterior stopping value.
    kld_mean, kld_std = np.mean(kld_arr), np.std(kld_arr)
    stop_post = boost * (kld_std / kld_mean) / post_thresh

    # Effective stopping value.
    stop = pfrac * stop_post + (1. - pfrac) * stop_evid

    if return_vals:
        return stop <= 1., (stop_post, stop_evid, stop)
    else:
        return stop <= 1.


class DynamicSampler(object):
    """
    A dynamic nested sampler that allocates live points adaptively during
    a single run according to a specified weight function until a specified
    stopping criteria is reached.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int, optional
        Number of parameters accepted by `prior_transform`.

    bound : {`'none'`, `'single'`, `'multi'`, `'balls'`, `'cubes'`}, optional
        Method used to approximately bound the prior using the current
        set of live points. Conditions the sampling methods used to
        propose new live points.

    method : {`'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.RandomState`
        `~numpy.random.RandomState` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    kwargs : dict, optional
        A dictionary of additional parameters (described below).

    """

    def __init__(self, loglikelihood, prior_transform, npdim,
                 bound, method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs):
        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.npdim = npdim

        # bounding/sampling
        self.bounding = bound
        self.method = method
        self.update_interval = update_interval
        self.first_update = first_update

        # internal sampler object
        self.sampler = None

        # extra arguments
        self.kwargs = kwargs
        if kwargs.get('bootstrap') is None:
            if self.method == 'unif':
                self.bootstrap = 20
            else:
                self.bootstrap = 0
        else:
            self.bootstrap = kwargs.get('bootstrap')
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
        self.use_pool = use_pool  # provided flags for when to use the pool
        self.use_pool_ptform = use_pool.get('prior_transform', True)
        self.use_pool_logl = use_pool.get('loglikelihood', True)
        self.use_pool_evolve = use_pool.get('propose_point', True)
        self.use_pool_update = use_pool.get('update_bound', True)
        self.use_pool_stopfn = use_pool.get('stop_function', True)

        # sampling details
        self.it = 1  # number of iterations
        self.batch = 0  # number of batches allocated dynamically
        self.ncall = 0  # number of function calls
        self.bound = []  # initial states used to compute bounds
        self.eff = 1.  # sampling efficiency
        self.base = False  # base run complete

        # results
        self.saved_id = []  # live point labels
        self.saved_u = []  # unit cube samples
        self.saved_v = []  # transformed variable samples
        self.saved_logl = []  # loglikelihoods of samples
        self.saved_logvol = []  # expected ln(volume)
        self.saved_logwt = []  # ln(weights)
        self.saved_logz = []  # cumulative ln(evidence)
        self.saved_logzvar = []  # cumulative error on ln(evidence)
        self.saved_h = []  # cumulative information
        self.saved_nc = []  # number of calls at each iteration
        self.saved_boundidx = []  # index of bound dead point was drawn from
        self.saved_it = []  # iteration the live (now dead) point was proposed
        self.saved_n = []  # number of live points interior to dead point
        self.saved_bounditer = []  # active bound at a specific iteration
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
        self.base_boundidx = []
        self.base_it = []
        self.base_n = []
        self.base_bounditer = []
        self.base_scale = []

        # results from our most recent addition
        self.new_id = []
        self.new_u = []
        self.new_v = []
        self.new_logl = []
        self.new_nc = []
        self.new_it = []
        self.new_n = []
        self.new_boundidx = []
        self.new_bounditer = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf  # logl bounds

    def reset(self):
        """Re-initialize the sampler."""

        # sampling
        self.it = 1
        self.batch = 0
        self.ncall = 0
        self.bound = []
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
        self.saved_boundidx = []
        self.saved_it = []
        self.saved_n = []
        self.saved_bounditer = []
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
        self.base_boundidx = []
        self.base_it = []
        self.base_n = []
        self.base_bounditer = []
        self.base_scale = []

        # results from our most recent addition
        self.new_id = []
        self.new_u = []
        self.new_v = []
        self.new_logl = []
        self.new_nc = []
        self.new_it = []
        self.new_n = []
        self.new_boundidx = []
        self.new_bounditer = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

    @property
    def results(self):
        """Saved results from the dynamic nested sampling run. All saved
        bounds are also returned."""

        # Add all saved samples (and ancillary quantities) to the results.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = [('niter', self.it - 1),
                       ('ncall', np.array(self.saved_nc)),
                       ('eff', self.eff),
                       ('samples', np.array(self.saved_v)),
                       ('samples_id', np.array(self.saved_id)),
                       ('samples_batch', np.array(self.saved_batch,
                                                  dtype='int')),
                       ('samples_it', np.array(self.saved_it)),
                       ('samples_u', np.array(self.saved_u)),
                       ('samples_n', np.array(self.saved_n)),
                       ('logwt', np.array(self.saved_logwt)),
                       ('logl', np.array(self.saved_logl)),
                       ('logvol', np.array(self.saved_logvol)),
                       ('logz', np.array(self.saved_logz)),
                       ('logzerr', np.sqrt(np.array(self.saved_logzvar))),
                       ('information', np.array(self.saved_h)),
                       ('batch_nlive', np.array(self.saved_batch_nlive,
                                                dtype='int')),
                       ('batch_bounds', np.array(self.saved_batch_bounds))]

        # Add any saved bounds (and ancillary quantities) to the results.
        if self.sampler.save_bounds:
            results.append(('bound', copy.deepcopy(self.bound)))
            results.append(('bound_iter',
                            np.array(self.saved_bounditer, dtype='int')))
            results.append(('samples_bound',
                            np.array(self.saved_boundidx, dtype='int')))
            results.append(('scale', np.array(self.saved_scale)))

        return Results(results)

    def sample_initial(self, nlive=500, update_interval=None,
                       first_update=None, maxiter=None, maxcall=None,
                       logl_max=np.inf, dlogz=0.01, live_points=None):
        """
        Generate a series of initial samples from a nested sampling
        run using a fixed number of live points using an internal
        sampler from :mod:`~dynesty.nestedsamplers`. Instantiates a
        generator that will be called by the user.

        Parameters
        ----------
        nlive : int, optional
            The number of live points to use for the baseline nested
            sampling run. Default is `500`.

        update_interval : int or float, optional
            If an integer is passed, only update the bounding distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the bound after every `round(update_interval * nlive)`-th
            likelihood call. Larger update intervals can be more efficient
            when the likelihood function is quick to evaluate. If no value is
            provided, defaults to the value passed during initialization.

        first_update : dict, optional
            A dictionary containing parameters governing when the sampler will
            first update the bounding distribution from the unit cube
            (`'none'`) to the one specified by `sample`.

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
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. The default is
            `0.01`.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
            A set of live points used to initialize the nested sampling run.
            Contains `live_u`, the coordinates on the unit cube, `live_v`, the
            transformed variables, and `live_logl`, the associated
            loglikelihoods. By default, if these are not provided the initial
            set of live points will be drawn from the unit `npdim`-cube.
            **WARNING: It is crucial that the initial set of live points have
            been sampled from the prior. Failure to provide a set of valid
            live points will lead to incorrect results.**

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : float
            Ln(likelihood) of the sample.

        logvol : float
            Ln(prior volume) within the sample.

        logwt : float
            Ln(weight) of the sample.

        logz : float
            Cumulative ln(evidence) up to the sample (inclusive).

        logzvar : float
            Estimated cumulative variance on `logz` (inclusive).

        h : float
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        boundidx : int
            Index of the bound the dead point was originally drawn from.

        bounditer : int
            Index of the bound being used at the current iteration.

        eff : float
            The cumulative sampling efficiency (in percent).

        delta_logz : float
            The estimated remaining evidence expressed as the ln(ratio) of the
            current evidence.

        """

        # Initialize inputs.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        if nlive <= 2 * self.npdim:
            warnings.warn("Beware: `nlive_init <= 2 * ndim`!")

        # Reset saved results to avoid any possible conflicts.
        self.reset()

        # Initialize the first set of live points.
        if live_points is None:
            self.nlive_init = nlive
            self.live_u = self.rstate.rand(self.nlive_init, self.npdim)
            if self.use_pool_ptform:
                self.live_v = np.array(list(self.M(self.prior_transform,
                                                   np.array(self.live_u))))
            else:
                self.live_v = np.array(list(map(self.prior_transform,
                                                np.array(self.live_u))))
            if self.use_pool_logl:
                self.live_logl = np.array(list(self.M(self.loglikelihood,
                                                      np.array(self.live_v))))
            else:
                self.live_logl = np.array(list(map(self.loglikelihood,
                                                   np.array(self.live_v))))
        else:
            self.live_u, self.live_v, self.live_logl = live_points
            self.nlive_init = len(self.live_u)

        # Convert all `-np.inf` log-likelihoods to finite large numbers.
        # Necessary to keep estimators in our sampler from breaking.
        for i, logl in enumerate(self.live_logl):
            if not np.isfinite(logl):
                if np.sign(logl) < 0:
                    self.live_logl[i] = -1e300
                else:
                    raise ValueError("The log-likelihood ({0}) of live "
                                     "point {1} located at u={2} v={3} "
                                     " is invalid."
                                     .format(logl, i, self.live_u[i],
                                             self.live_v[i]))

        # (Re-)bundle live points.
        live_points = [self.live_u, self.live_v, self.live_logl]
        self.live_init = [np.array(l) for l in live_points]
        self.ncall += self.nlive_init
        self.live_bound = np.zeros(self.nlive_init, dtype='int')
        self.live_it = np.zeros(self.nlive_init, dtype='int')

        # Initialize the internal `sampler` object.
        if update_interval is None:
            update_interval = self.update_interval
        if isinstance(update_interval, float):
            update_interval = int(round(self.update_interval * nlive))
        bounding = self.bounding
        if bounding == 'none':
            update_interval = np.inf  # no need to update with no bounds
        if first_update is None:
            first_update = self.first_update
        self.sampler = _SAMPLERS[bounding](self.loglikelihood,
                                           self.prior_transform,
                                           self.npdim, self.live_init,
                                           self.method, update_interval,
                                           first_update,
                                           self.rstate, self.queue_size,
                                           self.pool, self.use_pool,
                                           self.kwargs)
        self.bound = self.sampler.bound

        # Run the sampler internally as a generator.
        for i in range(1):
            for it, results in enumerate(self.sampler.sample(maxiter=maxiter,
                                         save_samples=False,
                                         maxcall=maxcall, dlogz=dlogz)):
                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt,
                 logz, logzvar, h, nc, worst_it, boundidx, bounditer,
                 eff, delta_logz) = results

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
                self.base_boundidx.append(boundidx)
                self.base_bounditer.append(bounditer)
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
                self.saved_boundidx.append(boundidx)
                self.saved_bounditer.append(bounditer)
                self.saved_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.ncall += nc
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, logvol, logwt,
                       logz, logzvar, h, nc, worst_it, boundidx, bounditer,
                       self.eff, delta_logz)

            for it, results in enumerate(self.sampler.add_live_points()):
                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt,
                 logz, logzvar, h, nc, worst_it, boundidx, bounditer,
                 eff, delta_logz) = results

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
                self.base_boundidx.append(boundidx)
                self.base_bounditer.append(bounditer)
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
                self.saved_boundidx.append(boundidx)
                self.saved_bounditer.append(bounditer)
                self.saved_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, logvol, logwt,
                       logz, logzvar, h, nc, worst_it, boundidx, bounditer,
                       self.eff, delta_logz)

        self.base = True  # baseline run complete
        self.saved_batch = np.zeros(len(self.saved_id), dtype='int')  # batch
        self.saved_batch_nlive.append(self.nlive_init)  # initial nlive
        self.saved_batch_bounds.append((-np.inf, np.inf))  # initial bounds

    def sample_batch(self, nlive_new=500, update_interval=None,
                     logl_bounds=None, maxiter=None, maxcall=None,
                     save_bounds=True):
        """
        Generate an additional series of nested samples that will be combined
        with the previous set of dead points. Works by hacking the internal
        `sampler` object.
        Instantiates a generator that will be called by the user.

        Parameters
        ----------
        nlive_new : int
            Number of new live points to be added. Default is `500`.

        update_interval : int or float, optional
            If an integer is passed, only update the bounding distribution
            every `update_interval`-th likelihood call. If a float is passed,
            update the bound after every `round(update_interval * nlive)`-th
            likelihood call. Larger update intervals can be more efficient
            when the likelihood function is quick to evaluate. If no value is
            provided, defaults to the value passed during initialization.

        logl_bounds : tuple of size (2,), optional
            The ln(likelihood) bounds used to bracket the run. If `None`,
            the default bounds span the entire range covered by the
            original run.

        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        save_bounds : bool, optional
            Whether or not to save past distributions used to bound
            the live points internally. Default is `True`.

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample. **Negative values indicate the index
            of a new live point generated when initializing a new batch.**

        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : float
            Ln(likelihood) of the sample.

        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        boundidx : int
            Index of the bound the dead point was originally drawn from.

        bounditer : int
            Index of the bound being used at the current iteration.

        eff : float
            The cumulative sampling efficiency (in percent).

        """

        # Initialize default values.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        if nlive_new <= 2 * self.npdim:
            warnings.warn("Beware: `nlive_batch <= 2 * ndim`!")
        self.sampler.save_bounds = save_bounds

        # Initialize starting values.
        h = 0.0  # Information, initially *0.*
        logz = -1.e300  # ln(evidence), initially *0.*
        logvol = 0.  # initially contains the whole prior (volume=1.)

        # Grab results from base run.
        base_id = np.array(self.base_id)
        base_u = np.array(self.base_u)
        base_v = np.array(self.base_v)
        base_logl = np.array(self.base_logl)
        base_n = np.array(self.base_n)
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
        self.new_boundidx = []
        self.new_bounditer = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

        # Initialize ln(likelihood) bounds.
        if logl_bounds is None:
            logl_min, logl_max = -np.inf, max(base_logl[:-nblive])
        else:
            logl_min, logl_max = logl_bounds
        self.new_logl_min, self.new_logl_max = logl_min, logl_max

        # Check whether the lower bound encompasses all previous base samples.
        psel = np.all(logl_min <= base_logl)
        vol = 1. - 1. / nblive  # starting ln(prior volume)
        if psel:
            # If the lower bound encompasses all base samples, we want
            # to propose a new set of points from the unit cube.
            live_u = self.rstate.rand(nlive_new, self.npdim)
            if self.use_pool_ptform:
                live_v = np.array(list(self.M(self.prior_transform,
                                              np.array(live_u))))
            else:
                live_v = np.array(list(map(self.prior_transform,
                                           np.array(live_u))))
            if self.use_pool_logl:
                live_logl = np.array(list(self.M(self.loglikelihood,
                                                 np.array(live_v))))
            else:
                live_logl = np.array(list(map(self.loglikelihood,
                                              np.array(live_v))))
            # Convert all `-np.inf` log-likelihoods to finite large numbers.
            # Necessary to keep estimators in our sampler from breaking.
            for i, logl in enumerate(live_logl):
                if not np.isfinite(logl):
                    if np.sign(logl) < 0:
                        live_logl[i] = -1e300
                    else:
                        raise ValueError("The log-likelihood ({0}) of live "
                                         "point {1} located at u={2} v={3} "
                                         " is invalid."
                                         .format(logl, i, live_u[i],
                                                 live_v[i]))
            live_bound = np.zeros(nlive_new, dtype='int')
            live_it = np.zeros(nlive_new, dtype='int') + self.it
            live_nc = np.ones(nlive_new, dtype='int')
            self.ncall += nlive_new
            # Return live points in generator format.
            for i in range(nlive_new):
                yield (-i - 1, live_u[i], live_v[i], live_logl[i], live_nc[i],
                       live_it[i], 0, 0, self.eff)
        else:
            # If the lower bound doesn't encompass all base samples, we need
            # to "rewind" our previous base run until we arrive at the
            # relevant set of live points (and scale) at the bound.
            live_u = np.empty((nblive, self.npdim))
            live_v = np.empty((nblive, base_v.shape[1]))
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

            # Hack the internal sampler by overwriting the live points
            # and scale factor.
            self.sampler.nlive = nblive
            self.sampler.live_u = np.array(live_u)
            self.sampler.live_v = np.array(live_v)
            self.sampler.live_logl = np.array(live_logl)
            self.sampler.scale = live_scale

            # Trigger an update of the internal bounding distribution based
            # on the "new" set of live points.
            vol = math.exp(- 1. * (nbase + r) / nblive)
            loglmin = min(live_logl)
            if self.sampler._beyond_unit_bound(loglmin):
                bound = self.sampler.update(vol / nblive)
                if save_bounds:
                    self.sampler.bound.append(copy.deepcopy(bound))
                self.sampler.nbound += 1
                self.sampler.since_update = 0

            # Sample a new batch of `nlive_new` live points using the
            # internal sampler given the `logl_min` constraint.
            live_u = np.empty((nlive_new, self.npdim))
            live_v = np.empty((nlive_new, base_v.shape[1]))
            live_logl = np.empty(nlive_new)
            live_bound = np.zeros(nlive_new, dtype='int')
            if self.sampler._beyond_unit_bound(loglmin):
                live_bound += self.sampler.nbound - 1
            live_it = np.empty(nlive_new, dtype='int')
            live_nc = np.empty(nlive_new, dtype='int')
            for i in range(nlive_new):
                (live_u[i], live_v[i], live_logl[i],
                 live_nc[i]) = self.sampler._new_point(logl_min, math.log(vol))
                live_it[i] = self.it
                self.ncall += live_nc[i]
                # Return live points in generator format.
                yield (-i - 1, live_u[i], live_v[i], live_logl[i], live_nc[i],
                       live_it[i], live_bound[i], live_bound[i], self.eff)

        # Overwrite the previous set of live points in our internal sampler
        # with the new batch of points we just generated.
        self.sampler.nlive = nlive_new
        self.sampler.live_u = np.array(live_u)
        self.sampler.live_v = np.array(live_v)
        self.sampler.live_logl = np.array(live_logl)
        self.sampler.live_bound = np.array(live_bound)
        self.sampler.live_it = np.array(live_it)

        # Trigger an update of the internal bounding distribution (again).
        loglmin = min(live_logl)
        if self.sampler._beyond_unit_bound(loglmin):
            bound = self.sampler.update(vol / nlive_new)
            if save_bounds:
                self.sampler.bound.append(copy.deepcopy(bound))
            self.sampler.nbound += 1
            self.sampler.since_update = 0

        # Copy over bound reference.
        self.bound = self.sampler.bound

        # Update `update_interval` based on our new set of live points.
        if update_interval is None:
            update_interval = self.update_interval
        if isinstance(update_interval, float):
            update_interval = int(round(self.update_interval * nlive_new))
        if self.bounding == 'none':
            update_interval = np.inf  # no need to update with no bounds
        self.sampler.update_interval = update_interval

        # Update internal ln(prior volume)-based quantities used to set things
        # like `pointvol` that help to prevent constructing over-constrained
        # bounding distributions.
        if self.new_logl_min == -np.inf:
            bound_logvol = 0.
        else:
            vol_idx = np.argmin(abs(self.saved_logl - self.new_logl_min))
            bound_logvol = self.saved_logvol[vol_idx]
        bound_dlv = math.log((nlive_new + 1.) / nlive_new)
        self.sampler.saved_logvol[-1] = bound_logvol
        self.sampler.dlv = bound_dlv

        # Tell the sampler *not* to try and remove the previous addition of
        # live points. All the hacks above make the internal results
        # garbage anyways.
        self.sampler.added_live = False

        # Run the sampler internally as a generator until we hit
        # the lower likelihood threshold. Afterwards, we add in our remaining
        # live points *as if* we had terminated the run. This allows us to
        # sample past the original bounds "for free".
        for i in range(1):
            for it, results in enumerate(self.sampler.sample(dlogz=0.,
                                         logl_max=logl_max,
                                         maxiter=maxiter-nlive_new-1,
                                         maxcall=maxcall-sum(live_nc),
                                         save_samples=False,
                                         save_bounds=save_bounds)):

                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt,
                 logz, logzvar, h, nc, worst_it, boundidx, bounditer,
                 eff, delta_logz) = results

                # Save results.
                self.new_id.append(worst)
                self.new_u.append(ustar)
                self.new_v.append(vstar)
                self.new_logl.append(loglstar)
                self.new_nc.append(nc)
                self.new_it.append(worst_it)
                self.new_n.append(nlive_new)
                self.new_boundidx.append(boundidx)
                self.new_bounditer.append(bounditer)
                self.new_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.ncall += nc
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, nc,
                       worst_it, boundidx, bounditer, self.eff)

            for it, results in enumerate(self.sampler.add_live_points()):
                # Grab results.
                (worst, ustar, vstar, loglstar, logvol, logwt,
                 logz, logzvar, h, nc, worst_it, boundidx, bounditer,
                 eff, delta_logz) = results

                # Save results.
                self.new_id.append(worst)
                self.new_u.append(ustar)
                self.new_v.append(vstar)
                self.new_logl.append(loglstar)
                self.new_nc.append(live_nc[worst])
                self.new_it.append(worst_it)
                self.new_n.append(nlive_new - it)
                self.new_boundidx.append(boundidx)
                self.new_bounditer.append(bounditer)
                self.new_scale.append(self.sampler.scale)

                # Increment relevant counters.
                self.eff = 100. * self.it / self.ncall
                self.it += 1

                yield (worst, ustar, vstar, loglstar, live_nc[worst],
                       worst_it, boundidx, bounditer, self.eff)

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
        saved_boundidx = np.array(self.saved_boundidx)
        saved_it = np.array(self.saved_it)
        saved_n = np.array(self.saved_n)
        saved_bounditer = np.array(self.saved_bounditer)
        saved_scale = np.array(self.saved_scale)
        saved_batch = np.array(self.saved_batch)
        nsaved = len(saved_n)

        # Grab results from new run.
        new_id = np.array(self.new_id) + max(saved_id) + 1
        new_u = np.array(self.new_u)
        new_v = np.array(self.new_v)
        new_logl = np.array(self.new_logl)
        new_nc = np.array(self.new_nc)
        new_boundidx = np.array(self.new_boundidx)
        new_it = np.array(self.new_it)
        new_n = np.array(self.new_n)
        new_bounditer = np.array(self.new_bounditer)
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
        self.saved_boundidx = []
        self.saved_it = []
        self.saved_n = []
        self.saved_bounditer = []
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
                self.saved_boundidx.append(saved_boundidx[idx_saved])
                self.saved_it.append(saved_it[idx_saved])
                self.saved_bounditer.append(saved_bounditer[idx_saved])
                self.saved_scale.append(saved_scale[idx_saved])
                self.saved_batch.append(saved_batch[idx_saved])
                idx_saved += 1
            else:
                self.saved_id.append(new_id[idx_new])
                self.saved_u.append(new_u[idx_new])
                self.saved_v.append(new_v[idx_new])
                self.saved_logl.append(new_logl[idx_new])
                self.saved_nc.append(new_nc[idx_new])
                self.saved_boundidx.append(new_boundidx[idx_new])
                self.saved_it.append(new_it[idx_new])
                self.saved_bounditer.append(new_bounditer[idx_new])
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

        # Compute quantities of interest.
        h = 0.
        logz = -1.e300
        loglstar = -1.e300
        logzvar = 0.
        logvols_pad = np.concatenate(([0.], self.saved_logvol))
        logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                             axis=1, b=np.c_[np.ones(ntot), -np.ones(ntot)])
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
            logzvar += 2. * dh * dlv
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
        self.new_boundidx = []
        self.new_bounditer = []
        self.new_scale = []
        self.new_logl_min, self.new_logl_max = -np.inf, np.inf

        # Increment batch counter.
        self.batch += 1

        # Saved batch quantities.
        self.saved_batch_nlive.append(max(new_n))
        self.saved_batch_bounds.append((llmin, llmax))

    def run_nested(self, nlive_init=500, maxiter_init=None,
                   maxcall_init=None, dlogz_init=0.01, logl_max_init=np.inf,
                   nlive_batch=500, wt_function=None, wt_kwargs=None,
                   maxiter_batch=None, maxcall_batch=None,
                   maxiter=None, maxcall=None, maxbatch=None,
                   stop_function=None, stop_kwargs=None, use_stop=True,
                   save_bounds=True, print_progress=True, print_func=None,
                   live_points=None):
        """
        **The main dynamic nested sampling loop.** After an initial "baseline"
        run using a constant number of live points, dynamically allocates
        additional (nested) samples to optimize a specified weight function
        until a specified stopping criterion is reached.

        Parameters
        ----------
        nlive_init : int, optional
            The number of live points used during the initial ("baseline")
            nested sampling run. Default is `500`.

        maxiter_init : int, optional
            Maximum number of iterations for the initial baseline nested
            sampling run. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall_init : int, optional
            Maximum number of likelihood evaluations for the initial
            baseline nested sampling run. Iteration may stop earlier
            if the termination condition is reached. Default is `sys.maxsize`
            (no limit).

        dlogz_init : float, optional
            The baseline run will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. The default is
            `0.01`.

        logl_max_init : float, optional
            The baseline run will stop when the sampled ln(likelihood) exceeds
            this threshold. Default is no bound (`np.inf`).

        nlive_batch : int, optional
            The number of live points used when adding additional samples
            from a nested sampling run within each batch. Default is `500`.

        wt_function : func, optional
            A cost function that takes a :class:`Results` instance
            and returns a log-likelihood range over which a new batch of
            samples should be generated. The default function simply
            computes a weighted average of the posterior and evidence
            information content as::

                weight = pfrac * pweight + (1. - pfrac) * zweight

        wt_kwargs : dict, optional
            Extra arguments to be passed to the weight function.

        maxiter_batch : int, optional
            Maximum number of iterations for the nested
            sampling run within each batch. Iteration may stop earlier
            if the termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall_batch : int, optional
            Maximum number of likelihood evaluations for the nested
            sampling run within each batch. Iteration may stop earlier
            if the termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxiter : int, optional
            Maximum number of iterations allowed. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations allowed.
            Default is `sys.maxsize` (no limit).

        maxbatch : int, optional
            Maximum number of batches allowed. Default is `sys.maxsize`
            (no limit).

        stop_function : func, optional
            A function that takes a :class:`Results` instance and
            returns a boolean indicating that we should terminate the run
            because we've collected enough samples.

        stop_kwargs : float, optional
            Extra arguments to be passed to the stopping function.

        use_stop : bool, optional
            Whether to evaluate our stopping function after each batch.
            Disabling this can improve performance if other stopping criteria
            such as :data:`maxcall` are already specified. Default is `True`.

        save_bounds : bool, optional
            Whether or not to save distributions used to bound
            the live points internally during dynamic live point allocation.
            Default is `True`.

        print_progress : bool, optional
            Whether to output a simple summary of the current run that
            updates each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

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
        if print_func is None:
            print_func = print_fn

        # Run the main dynamic nested sampling loop.
        ncall = self.ncall
        niter = self.it - 1
        logl_bounds = (-np.inf, np.inf)
        maxcall_init = min(maxcall_init, maxcall)  # set max calls
        maxiter_init = min(maxiter_init, maxiter)  # set max iterations

        # Baseline run.
        if not self.base:
            for results in self.sample_initial(nlive=nlive_init,
                                               dlogz=dlogz_init,
                                               maxcall=maxcall_init,
                                               maxiter=maxiter_init,
                                               logl_max=logl_max_init,
                                               live_points=live_points):
                (worst, ustar, vstar, loglstar, logvol,
                 logwt, logz, logzvar, h, nc, worst_it,
                 boundidx, bounditer, eff, delta_logz) = results
                ncall += nc
                niter += 1

                # Print progress.
                if print_progress:
                    print_func(results, niter, ncall, nbatch=0,
                               dlogz=dlogz_init, logl_max=logl_max_init)

        # Add points in batches.
        for n in range(self.batch, maxbatch):
            # Update stopping criteria.
            res = self.results
            mcall = min(maxcall - ncall, maxcall_batch)
            miter = min(maxiter - niter, maxiter_batch)
            if mcall > 0 and miter > 0 and use_stop:
                if self.use_pool_stopfn:
                    M = self.M
                else:
                    M = map
                stop, stop_vals = stop_function(res, stop_kwargs,
                                                rstate=self.rstate, M=M,
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
                passback = self.add_batch(nlive=nlive_batch,
                                          wt_function=wt_function,
                                          wt_kwargs=wt_kwargs,
                                          maxiter=miter,
                                          maxcall=mcall,
                                          save_bounds=save_bounds,
                                          print_progress=print_progress,
                                          print_func=print_func,
                                          stop_val=stop_val)
                ncall, niter, logl_bounds, results = passback
            elif logl_bounds[1] != np.inf:
                # We ran at least one batch and now we're done!
                if print_progress:
                    print_func(results, niter, ncall, nbatch=n,
                               stop_val=stop_val,
                               logl_min=logl_bounds[0],
                               logl_max=logl_bounds[1])
                break
            else:
                # We didn't run a single batch but now we're done!
                break

        if print_progress:
            sys.stderr.write("\n")

    def add_batch(self, nlive=500, wt_function=None, wt_kwargs=None,
                  maxiter=None, maxcall=None, save_bounds=True,
                  print_progress=True, print_func=None, stop_val=None):
        """
        Allocate an additional batch of (nested) samples based on
        the combined set of previous samples using the specified
        weight function.

        Parameters
        ----------
        nlive : int, optional
            The number of live points used when adding additional samples
            in the batch. Default is `500`.

        wt_function : func, optional
            A cost function that takes a `Results` instance
            and returns a log-likelihood range over which a new batch of
            samples should be generated. The default function simply
            computes a weighted average of the posterior and evidence
            information content as::

                weight = pfrac * pweight + (1. - pfrac) * zweight

        wt_kwargs : dict, optional
            Extra arguments to be passed to the weight function.

        maxiter : int, optional
            Maximum number of iterations allowed. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations allowed.
            Default is `sys.maxsize` (no limit).

        save_bounds : bool, optional
            Whether or not to save distributions used to bound
            the live points internally during dynamic live point allocations.
            Default is `True`.

        print_progress : bool, optional
            Whether to output a simple summary of the current run that
            updates each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        stop_val : float, optional
            The value of the stopping criteria to be passed to
            :meth:`print_func`. Used internally within :meth:`run_nested` to
            keep track of progress.

        """

        # Initialize values.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        if wt_function is None:
            wt_function = weight_function
        if wt_kwargs is None:
            wt_kwargs = dict()
        if print_func is None:
            print_func = print_fn

        # If we have either likelihood calls or iterations remaining,
        # add our new batch of live points.
        ncall, niter, n = self.ncall, self.it - 1, self.batch
        if maxcall > 0 and maxiter > 0:
            # Compute our sampling bounds using the provided
            # weight function.
            res = self.results
            lnz, lnzerr = res.logz[-1], res.logzerr[-1]
            logl_bounds = wt_function(res, wt_kwargs)
            for results in self.sample_batch(nlive_new=nlive,
                                             logl_bounds=logl_bounds,
                                             maxiter=maxiter,
                                             maxcall=maxcall,
                                             save_bounds=save_bounds):
                (worst, ustar, vstar, loglstar, nc,
                 worst_it, boundidx, bounditer, eff) = results

                # When initializing a batch (i.e. when `worst < 0`),
                # don't increment our call counter or our current
                # number of iterations.
                if worst >= 0:
                    ncall += nc
                    niter += 1

                # Reorganize results.
                results = (worst, ustar, vstar, loglstar, np.nan, np.nan,
                           lnz, lnzerr**2, np.nan, nc, worst_it, boundidx,
                           bounditer, eff, np.nan)

                # Print progress.
                if print_progress:
                    print_func(results, niter, ncall, nbatch=n+1,
                               stop_val=stop_val,
                               logl_min=logl_bounds[0],
                               logl_max=logl_bounds[1])

            # Combine batch with previous runs.
            self.combine_runs()

        # Pass back info.
        return ncall, niter, logl_bounds, results
