#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A collection of useful functions.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import warnings
import math
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

import numpy as np
import copy

from .results import Results

__all__ = ["unitcheck", "resample_equal", "mean_and_cov", "quantile",
           "jitter_run", "resample_run", "simulate_run", "reweight_run",
           "unravel_run", "merge_runs", "kl_divergence", "kld_error",
           "_merge_two", "_get_nsamps_samples_n"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def unitcheck(u, nonbounded=None):
    """Check whether `u` is inside the unit cube. Given a masked array
    `nonbounded`, also allows periodic boundaries conditions to exceed
    the unit cube."""

    if nonbounded is None:
        # No periodic boundary conditions provided.
        return np.all(u > 0.) and np.all(u < 1.)
    else:
        # Alternating periodic and non-periodic boundary conditions.
        return (np.all(u[nonbounded] > 0.) and
                np.all(u[nonbounded] < 1.) and
                np.all(u[~nonbounded] > -0.5) and
                np.all(u[~nonbounded] < 1.5))


def reflect(u):
    """
    Iteratively reflect a number until it is contained in [0, 1].

    This is for priors with a reflective boundary condition, all numbers in the
    set `u = 2n +/- x` should be mapped to x.

    For the `+` case we just take `u % 1`.
    For the `-` case we take `1 - (u % 1)`.

    E.g., -0.9, 1.1, and 2.9 should all map to 0.9.

    Parameters
    ----------
    u: array-like
        The array of points to map to the unit cube

    Returns
    -------
    u: array-like
       The input array, modified in place.
    """
    idxs_even = np.mod(u, 2) < 1
    u[idxs_even] = np.mod(u[idxs_even], 1)
    u[~idxs_even] = 1 - np.mod(u[~idxs_even], 1)
    return u


def mean_and_cov(samples, weights):
    """
    Compute the weighted mean and covariance of the samples.

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples, ndim)
        2-D array containing data samples. This ordering is equivalent to
        using `rowvar=False` in `~numpy.cov`.

    weights : `~numpy.ndarray` with shape (nsamples,)
        1-D array of sample weights.

    Returns
    -------
    mean : `~numpy.ndarray` with shape (ndim,)
        Weighted sample mean vector.

    cov : `~numpy.ndarray` with shape (ndim, ndim)
        Weighted sample covariance matrix.

    Notes
    -----
    Implements the formulae found `here <https://goo.gl/emWFLR>`_.

    """

    # Compute the weighted mean.
    mean = np.average(samples, weights=weights, axis=0)

    # Compute the weighted covariance.
    dx = samples - mean
    wsum = np.sum(weights)
    w2sum = np.sum(weights**2)
    cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', weights, dx, dx)

    return mean, cov


def resample_equal(samples, weights, rstate=None):
    """
    Resample a new set of points from the weighted set of inputs
    such that they all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
    with `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Set of unequally weighted samples.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New set of samples with equal weights.

    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> utils.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in `Hol, Schon, and
    Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.

    """

    if rstate is None:
        rstate = np.random

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        # Guarantee that the weights will sum to 1.
        warnings.warn("Weights do not sum to 1 and have been renormalized.")
        weights = np.array(weights) / np.sum(weights)

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = np.zeros(nsamples, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def _get_nsamps_samples_n(res):
    """ Helper function for calculating the number of samples

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    Returns
    -------
    nsamps: int
        The total number of samples
    samples_n: array
        Number of live points at a given iteration

    """
    try:
        # Check if the number of live points explicitly changes.
        samples_n = res.samples_n
        nsamps = len(samples_n)
    except:
        # If the number of live points is constant, compute `samples_n`.
        niter = res.niter
        nlive = res.nlive
        nsamps = len(res.logvol)
        if nsamps == niter:
            samples_n = np.ones(niter, dtype='int') * nlive
        elif nsamps == (niter + nlive):
            samples_n = np.append(np.ones(niter, dtype='int') * nlive,
                                  np.arange(1, nlive + 1)[::-1])
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")
    return nsamps, samples_n


def jitter_run(res, rstate=None, approx=False):
    """
    Probes **statistical uncertainties** on a nested sampling run by
    explicitly generating a *realization* of the prior volume associated
    with each sample (dead point). Companion function to :meth:`resample_run`
    and :meth:`simulate_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        weights based on our "jittered" prior volume realizations.

    """

    if rstate is None:
        rstate = np.random

    # Initialize evolution of live points over the course of the run.
    nsamps, samples_n = _get_nsamps_samples_n(res)
    logl = res.logl

    # Simulate the prior volume shrinkage associated with our set of "dead"
    # points. At each iteration, if the number of live points is constant or
    # increasing, our prior volume compresses by the maximum value of a set
    # of `K_i` uniformly distributed random numbers (i.e. as `Beta(K_i, 1)`).
    # If instead the number of live points is decreasing, that means we're
    # instead  sampling down a set of uniform random variables
    # (i.e. uniform order statistics).
    nlive_flag = np.ones(nsamps, dtype='bool')
    nlive_start, bounds = [], []

    if not approx:
        # Find all instances where the number of live points is either constant
        # or increasing.
        nlive_flag[1:] = np.diff(samples_n) >= 0

        # For all the portions that are decreasing, find out where they start,
        # where they end, and how many live points are present at that given
        # iteration.

        if np.any(~nlive_flag):
            i = 0
            while i < nsamps:
                if not nlive_flag[i]:
                    bound = []
                    bound.append(i-1)
                    nlive_start.append(samples_n[i-1])
                    while i < nsamps and not nlive_flag[i]:
                        i += 1
                    bound.append(i)
                    bounds.append(bound)
                i += 1

    # The maximum out of a set of `K_i` uniformly distributed random variables
    # has a marginal distribution of `Beta(K_i, 1)`.
    t_arr = np.zeros(nsamps)
    t_arr[nlive_flag] = rstate.beta(a=samples_n[nlive_flag], b=1)

    # If we instead are sampling the set of uniform order statistics,
    # we note that the jth largest value is marginally distributed as
    # `Beta(j, K_i-j+1)`. The full joint distribution is::
    #
    #     X_(j) / X_N = (Y_1 + ... + Y_j) / (Y_1 + ... + Y_{K+1})
    #
    # where X_(j) is the prior volume of the live point with the `j`-th
    # *best* likelihood (i.e. prior volume shrinks as likelihood increases)
    # and the `Y_i`'s are i.i.d. exponentially distributed random variables.
    nunif = len(nlive_start)
    for i in range(nunif):
        nstart = nlive_start[i]
        bound = bounds[i]
        sn = samples_n[bound[0]:bound[1]]
        y_arr = rstate.exponential(scale=1.0, size=nstart+1)
        ycsum = y_arr.cumsum()
        ycsum /= ycsum[-1]
        uorder = ycsum[np.append(nstart, sn-1)]
        rorder = uorder[1:] / uorder[:-1]
        t_arr[bound[0]:bound[1]] = rorder

    # These are the "compression factors" at each iteration. Let's turn
    # these into associated ln(volumes).
    logvol = np.log(t_arr).cumsum()

    # Compute weights using quadratic estimator.
    h = 0.
    logz = -1.e300
    loglstar = -1.e300
    logzvar = 0.
    logvols_pad = np.concatenate(([0.], logvol))
    logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                         axis=1, b=np.c_[np.ones(nsamps), -np.ones(nsamps)])
    logdvols += math.log(0.5)
    dlvs = -np.diff(np.append(0., res.logvol))
    saved_logwt, saved_logz, saved_logzvar, saved_h = [], [], [], []
    for i in range(nsamps):
        loglstar_new = logl[i]
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
        saved_logwt.append(logwt)
        saved_logz.append(logz)
        saved_logzvar.append(logzvar)
        saved_h.append(h)

    # Copy results.
    new_res = Results([item for item in res.items()])

    # Overwrite items with our new estimates.
    new_res.logvol = np.array(logvol)
    new_res.logwt = np.array(saved_logwt)
    new_res.logz = np.array(saved_logz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_res.logzerr = np.sqrt(np.array(saved_logzvar))
    new_res.h = np.array(saved_h)

    return new_res


def resample_run(res, rstate=None, return_idx=False):
    """
    Probes **sampling uncertainties** on a nested sampling run using bootstrap
    resampling techniques to generate a *realization* of the (expected) prior
    volume(s) associated with each sample (dead point). This effectively
    splits a nested sampling run with `K` particles (live points) into a
    series of `K` "strands" (i.e. runs with a single live point) which are then
    bootstrapped to construct a new "resampled" run. Companion function to
    :meth:`jitter_run` and :meth:`simulate_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    return_idx : bool, optional
        Whether to return the list of resampled indices used to construct
        the new run. Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        samples and weights based on our "bootstrapped" samples and
        (expected) prior volumes.

    """

    if rstate is None:
        rstate = np.random

    # Check whether the final set of live points were added to the
    # run.
    nsamps = len(res.ncall)
    try:
        # Check if the number of live points explicitly changes.
        samples_n = res.samples_n
        samples_batch = res.samples_batch
        batch_bounds = res.batch_bounds
        added_final_live = True
    except:
        # If the number of live points is constant, compute `samples_n` and
        # set up the `added_final_live` flag.
        nlive = res.nlive
        niter = res.niter
        if nsamps == niter:
            samples_n = np.ones(niter, dtype='int') * nlive
            added_final_live = False
        elif nsamps == (niter + nlive):
            samples_n = np.append(np.ones(niter, dtype='int') * nlive,
                                  np.arange(1, nlive + 1)[::-1])
            added_final_live = True
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")
        samples_batch = np.zeros(len(samples_n), dtype='int')
        batch_bounds = np.array([(-np.inf, np.inf)])
    batch_llmin = batch_bounds[:, 0]

    # Identify unique particles that make up each strand.
    ids = np.unique(res.samples_id)

    # Split the set of strands into two groups: a "baseline" group that
    # contains points initially sampled from the prior, which gives information
    # on the evidence, and an "add-on" group, which gives additional
    # information conditioned on our baseline strands.
    base_ids = []
    addon_ids = []
    for i in ids:
        sbatch = samples_batch[res.samples_id == i]
        if np.any(batch_llmin[sbatch] == -np.inf):
            base_ids.append(i)
        else:
            addon_ids.append(i)
    nbase, nadd = len(base_ids), len(addon_ids)
    base_ids, addon_ids = np.array(base_ids), np.array(addon_ids)

    # Resample strands.
    if nbase > 0 and nadd > 0:
        live_idx = np.append(base_ids[rstate.randint(0, nbase, size=nbase)],
                             addon_ids[rstate.randint(0, nadd, size=nadd)])
    elif nbase > 0:
        live_idx = base_ids[rstate.randint(0, nbase, size=nbase)]
    elif nadd > 0:
        raise ValueError("The provided `Results` does not include any points "
                         "initially sampled from the prior!")
    else:
        raise ValueError("The provided `Results` does not appear to have "
                         "any particles!")

    # Find corresponding indices within the original run.
    samp_idx = np.arange(len(res.ncall))
    samp_idx = np.concatenate([samp_idx[res.samples_id == idx]
                               for idx in live_idx])

    # Derive new sample size.
    nsamps = len(samp_idx)

    # Sort the loglikelihoods (there will be duplicates).
    logls = res.logl[samp_idx]
    idx_sort = np.argsort(logls)
    samp_idx = samp_idx[idx_sort]
    logl = res.logl[samp_idx]

    if added_final_live:
        # Compute the effective number of live points for each sample.
        samp_n = np.zeros(nsamps, dtype='int')
        uidxs, uidxs_n = np.unique(live_idx, return_counts=True)
        for uidx, uidx_n in zip(uidxs, uidxs_n):
            sel = (res.samples_id == uidx)  # selection flag
            sbatch = samples_batch[sel][0]  # corresponding batch ID
            lower = batch_llmin[sbatch]  # lower bound
            upper = max(res.logl[sel])  # upper bound

            # Add number of live points between endpoints equal to number of
            # times the strand has been resampled.
            samp_n[(logl > lower) & (logl < upper)] += uidx_n

            # At the endpoint, divide up the final set of points into `uidx_n`
            # (roughly) equal chunks and have live points decrease across them.
            endsel = (logl == upper)
            endsel_n = np.count_nonzero(endsel)
            chunk = endsel_n / uidx_n  # define our chunk
            counters = np.array(np.arange(endsel_n) / chunk, dtype='int')
            nlive_end = counters[::-1] + 1  # decreasing number of live points
            samp_n[endsel] += nlive_end  # add live point sequence
    else:
        # If we didn't add the final set of live points, the run has a constant
        # number of live points and can simply be re-ordered.
        samp_n = samples_n[samp_idx]

    # Assign log(volume) to samples.
    logvol = np.cumsum(np.log(samp_n / (samp_n + 1.)))

    # Computing weights using quadratic estimator.
    h = 0.
    logz = -1.e300
    loglstar = -1.e300
    logzvar = 0.
    logvols_pad = np.concatenate(([0.], logvol))
    logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                         axis=1, b=np.c_[np.ones(nsamps), -np.ones(nsamps)])
    logdvols += math.log(0.5)
    dlvs = logvols_pad[:-1] - logvols_pad[1:]
    saved_logwt, saved_logz, saved_logzvar, saved_h = [], [], [], []
    for i in range(nsamps):
        loglstar_new = logl[i]
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
        saved_logwt.append(logwt)
        saved_logz.append(logz)
        saved_logzvar.append(logzvar)
        saved_h.append(h)

    # Compute sampling efficiency.
    eff = 100. * len(res.ncall[samp_idx]) / sum(res.ncall[samp_idx])

    # Copy results.
    new_res = Results([item for item in res.items()])

    # Overwrite items with our new estimates.
    new_res.niter = len(res.ncall[samp_idx])
    new_res.ncall = res.ncall[samp_idx]
    new_res.eff = eff
    new_res.samples = res.samples[samp_idx]
    new_res.samples_id = res.samples_id[samp_idx]
    new_res.samples_it = res.samples_it[samp_idx]
    new_res.samples_u = res.samples_u[samp_idx]
    new_res.samples_n = samp_n
    new_res.logwt = np.array(saved_logwt)
    new_res.logl = logl
    new_res.logvol = logvol
    new_res.logz = np.array(saved_logz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_res.logzerr = np.sqrt(np.array(saved_logzvar))
    new_res.h = np.array(saved_h)

    if return_idx:
        return new_res, samp_idx
    else:
        return new_res


def simulate_run(res, rstate=None, return_idx=False, approx=False):
    """
    Probes **combined uncertainties** (statistical and sampling) on a nested
    sampling run by wrapping :meth:`jitter_run` and :meth:`resample_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    return_idx : bool, optional
        Whether to return the list of resampled indices used to construct
        the new run. Default is `False`.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        samples and weights based on our "simulated" samples and
        prior volumes.

    """

    if rstate is None:
        rstate = np.random

    # Resample run.
    new_res, samp_idx = resample_run(res, rstate=rstate, return_idx=True)

    # Jitter run.
    new_res = jitter_run(new_res, rstate=rstate, approx=approx)

    if return_idx:
        return new_res, samp_idx
    else:
        return new_res


def reweight_run(res, logp_new, logp_old=None):
    """
    Reweight a given run based on a new target distribution.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    logp_new : `~numpy.ndarray` with shape (nsamps,)
        New target distribution evaluated at the location of the samples.

    logp_old : `~numpy.ndarray` with shape (nsamps,)
        Old target distribution evaluated at the location of the samples.
        If not provided, the `logl` values from `res` will be used.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        weights based on our reweighted samples.

    """

    # Extract info.
    if logp_old is None:
        logp_old = res['logl']
    logrwt = logp_new - logp_old  # ln(reweight)
    logvol = res['logvol']
    logl = res['logl']
    nsamps = len(logvol)

    # Compute weights using quadratic estimator.
    h = 0.
    logz = -1.e300
    loglstar = -1.e300
    logzvar = 0.
    logvols_pad = np.concatenate(([0.], logvol))
    logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                         axis=1, b=np.c_[np.ones(nsamps), -np.ones(nsamps)])
    logdvols += math.log(0.5)
    dlvs = -np.diff(np.append(0., logvol))
    saved_logwt, saved_logz, saved_logzvar, saved_h = [], [], [], []
    for i in range(nsamps):
        loglstar_new = logl[i]
        logdvol, dlv = logdvols[i], dlvs[i]
        logwt = np.logaddexp(loglstar_new, loglstar) + logdvol + logrwt[i]
        logz_new = np.logaddexp(logz, logwt)
        try:
            lzterm = (math.exp(loglstar - logz_new) * loglstar +
                      math.exp(loglstar_new - logz_new) * loglstar_new)
        except:
            lzterm = 0.
        h_new = (math.exp(logdvol) * lzterm +
                 math.exp(logz - logz_new) * (h + logz) -
                 logz_new)
        dh = h_new - h
        h = h_new
        logz = logz_new
        logzvar += dh * dlv
        loglstar = loglstar_new
        saved_logwt.append(logwt)
        saved_logz.append(logz)
        saved_logzvar.append(logzvar)
        saved_h.append(h)

    # Copy results.
    new_res = Results([item for item in res.items()])

    # Overwrite items with our new estimates.
    new_res.logwt = np.array(saved_logwt)
    new_res.logz = np.array(saved_logz)
    new_res.logzerr = np.sqrt(np.array(saved_logzvar))
    new_res.h = np.array(saved_h)

    return new_res


def unravel_run(res, save_proposals=True, print_progress=True):
    """
    Unravels a run with `K` live points into `K` "strands" (a nested sampling
    run with only 1 live point). **WARNING: the anciliary quantities provided
    with each unraveled "strand" are only valid if the point was initialized
    from the prior.**

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    save_proposals : bool, optional
        Whether to save a reference to the proposal distributions from the
        original run in each unraveled strand. Default is `True`.

    print_progress : bool, optional
        Whether to output the current progress to `~sys.stderr`.
        Default is `True`.

    Returns
    -------
    new_res : list of :class:`~dynesty.results.Results` instances
        A list of new :class:`~dynesty.results.Results` instances
        for each individual strand.

    """

    idxs = res.samples_id  # label for each live/dead point

    # Check if we added in the last set of dead points.
    added_live = True
    try:
        if len(idxs) != (res.niter + res.nlive):
            added_live = False
    except:
        pass

    # Recreate the nested sampling run for each strand.
    new_res = []
    nstrands = len(np.unique(idxs))
    for counter, idx in enumerate(np.unique(idxs)):
        # Select strand `idx`.
        strand = (idxs == idx)
        nsamps = sum(strand)
        logl = res.logl[strand]

        # Assign log(volume) to samples. With K=1 live point, the expected
        # shrinking in `logvol` at each iteration is `-log(2)` (i.e.
        # shrinking by 1/2). If the final set of live points were added,
        # the expected value of the final live point is a uniform
        # sample and so has an expected value of half the volume
        # of the final dead point.
        if added_live:
            niter = nsamps - 1
            logvol_dead = -math.log(2) * (1. + np.arange(niter))
            if niter > 0:
                logvol_live = logvol_dead[-1] + math.log(0.5)
                logvol = np.append(logvol_dead, logvol_live)
            else:  # point always live
                logvol = np.array([math.log(0.5)])
        else:
            niter = nsamps
            logvol = -math.log(2) * (1. + np.arange(niter))

        # Compute weights using quadratic estimator.
        h = 0.
        logz = -1.e300
        loglstar = -1.e300
        logzvar = 0.
        logvols_pad = np.concatenate(([0.], logvol))
        logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                             axis=1, b=np.c_[np.ones(nsamps), -np.ones(nsamps)])
        logdvols += math.log(0.5)
        dlvs = logvols_pad[:-1] - logvols_pad[1:]
        saved_logwt, saved_logz, saved_logzvar, saved_h = [], [], [], []
        for i in range(nsamps):
            loglstar_new = logl[i]
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
            saved_logwt.append(logwt)
            saved_logz.append(logz)
            saved_logzvar.append(logzvar)
            saved_h.append(h)

        # Compute sampling efficiency.
        eff = 100. * nsamps / sum(res.ncall[strand])

        # Save results.
        r = [('nlive', 1),
             ('niter', niter),
             ('ncall', res.ncall[strand]),
             ('eff', eff),
             ('samples', res.samples[strand]),
             ('samples_id', res.samples_id[strand]),
             ('samples_it', res.samples_it[strand]),
             ('samples_u', res.samples_u[strand]),
             ('logwt', np.array(saved_logwt)),
             ('logl', logl),
             ('logvol', logvol),
             ('logz', np.array(saved_logz)),
             ('logzerr', np.sqrt(np.array(saved_logzvar))),
             ('h', np.array(saved_h))]

        # Add proposal information (if available).
        if save_proposals:
            try:
                r.append(('prop', res.prop))
                r.append(('prop_iter', res.prop_iter[strand]))
                r.append(('samples_prop', res.samples_prop[strand]))
                r.append(('scale', res.scale[strand]))
            except:
                pass

        # Add on batch information (if available).
        try:
            r.append(('samples_batch', res.samples_batch[strand]))
            r.append(('batch_bounds', res.batch_bounds))
        except:
            pass

        # Append to list of strands.
        new_res.append(Results(r))

        # Print progress.
        if print_progress:
            sys.stderr.write('\rStrand: {0}/{1}     '
                             .format(counter + 1, nstrands))

    return new_res


def merge_runs(res_list, print_progress=True):
    """
    Merges a set of runs with differing (possibly variable) numbers of
    live points into one run.

    Parameters
    ----------
    res_list : list of :class:`~dynesty.results.Results` instances
        A list of :class:`~dynesty.results.Results` instances returned from
        previous runs.

    print_progress : bool, optional
        Whether to output the current progress to `~sys.stderr`.
        Default is `True`.

    Returns
    -------
    combined_res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance for the combined run.

    """

    ntot = len(res_list)
    counter = 0

    # Establish our set of baseline runs and "add-on" runs.
    rlist_base = []
    rlist_add = []
    for r in res_list:
        try:
            if np.any(r.samples_batch == 0):
                rlist_base.append(r)
            else:
                rlist_add.append(r)
        except:
            rlist_base.append(r)
    nbase, nadd = len(rlist_base), len(rlist_add)
    if nbase == 1 and nadd == 1:
        rlist_base = res_list
        rlist_add = []

    # Merge baseline runs while there are > 2 remaining results.
    if len(rlist_base) > 1:
        while len(rlist_base) > 2:
            rlist_new = []
            nruns = len(rlist_base)
            i = 0
            while i < nruns:
                try:
                    # Ignore posterior quantities while merging the runs.
                    r1, r2 = rlist_base[i], rlist_base[i+1]
                    res = _merge_two(r1, r2, compute_aux=False)
                    rlist_new.append(res)
                except:
                    # Append the odd run to the new list.
                    rlist_new.append(rlist_base[i])
                i += 2
                counter += 1
                # Print progress.
                if print_progress:
                    sys.stderr.write('\rMerge: {0}/{1}     '.format(counter,
                                                                    ntot))
            # Overwrite baseline set of results with merged results.
            rlist_base = copy.copy(rlist_new)

        # Compute posterior quantities after merging the final baseline runs.
        res = _merge_two(rlist_base[0], rlist_base[1], compute_aux=True)
    else:
        res = rlist_base[0]

    # Iteratively merge any remaining "add-on" results.
    nruns = len(rlist_add)
    for i, r in enumerate(rlist_add):
        if i < nruns - 1:
            res = _merge_two(res, r, compute_aux=False)
        else:
            res = _merge_two(res, r, compute_aux=True)
        counter += 1
        # Print progress.
        if print_progress:
            sys.stderr.write('\rMerge: {0}/{1}     '.format(counter, ntot))

    nsamps, samples_n = _get_nsamps_samples_n(res)
    nlive = max(samples_n)
    niter = res.niter
    standard_run = False

    # Check if we have a constant number of live points.
    try:
        nlive_test = np.ones(niter, dtype='int') * nlive
        if np.all(samples_n == nlive_test):
            standard_run = True
    except:
        pass

    # Check if we have a constant number of live points where we have
    # recycled the final set of live points.
    try:
        nlive_test = np.append(np.ones(niter - nlive, dtype='int') * nlive,
                               np.arange(1, nlive + 1)[::-1])
        if np.all(samples_n == nlive_test):
            standard_run = True
    except:
        pass

    # If the number of live points is consistent with a standard nested
    # sampling run, slightly modify the format to keep with previous usage.
    if standard_run:
        res.__delitem__('samples_n')
        res.nlive = nlive
        res.niter = niter - nlive

    return res


def kl_divergence(res1, res2):
    """
    Computes the `Kullback-Leibler (KL) divergence
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`_ *from* the
    discrete probability distribution defined by `res2` *to* the discrete
    probability distribution defined by `res1`.

    Parameters
    ----------
    res1 : :class:`~dynesty.results.Results` instance
        :class:`~dynesty.results.Results` instance for the distribution we are
        computing the KL divergence *to*. **Note that, by construction,
        the samples in `res1` *must* be a subset of the samples in `res2`.**

    res2 : :class:`~dynesty.results.Results` instance
        :class:`~dynesty.results.Results` instance for the distribution we
        are computing the KL divergence *from*. **Note that, by construction,
        the samples in `res2` *must* be a superset of the samples in `res1`.**

    Returns
    -------
    kld : `~numpy.ndarray` with shape (nsamps,)
        The cumulative KL divergence defined over `res1`.

    """

    # Define our importance weights.
    logp1, logp2 = res1.logwt - res1.logz[-1], res2.logwt - res2.logz[-1]

    # Define the positions where the discrete probability distributions exists.
    samples1, samples2 = res1.samples, res2.samples
    samples1_id, samples2_id = res1.samples_id, res2.samples_id
    nsamps1, nsamps2 = len(samples1), len(samples2)

    # Compute the KL divergence.
    if nsamps1 == nsamps2 and np.all(samples1_id == samples2_id):
        # If our runs have the same particles in the same order, compute
        # the KL divergence in one go.
        kld = np.exp(logp1) * (logp1 - logp2)
    else:
        # Otherwise, compute the components of the KL divergence one at a time.
        uidxs = np.unique(samples1_id)  # unique particle IDs
        count1, count2 = np.arange(nsamps1), np.arange(nsamps2)
        kld = np.zeros(nsamps1)
        for uidx in uidxs:

            # Select matching particles.
            sel1 = count1[samples1_id == uidx]
            sel2 = count2[samples2_id == uidx]

            # Select corresponding positions.
            pos1, pos2 = samples1[sel1], samples2[sel2]
            for s, p in zip(sel1, pos1):
                # Search for a matching position.
                pos_sel = sel2[np.all(np.isclose(pos2, p), axis=1)]
                npos = len(pos_sel)
                if npos > 1:
                    # If there are several possible matches, pick the
                    # one with the closet importance weight.
                    diff = logp1[s] - logp2[pos_sel]
                    # Compute the `s`-th term.
                    kld[s] = np.exp(logp1[s]) * diff[np.argmin(abs(diff))]
                elif npos == 1:
                    # If there is only one match, compute the result directly.
                    kld[s] = np.exp(logp1[s]) * (logp1[s] - logp2[pos_sel])
                else:
                    raise ValueError("Distribution from `res2` undefined at "
                                     "position {0}.".format(p))

    return np.cumsum(kld)


def kld_error(res, error='simulate', rstate=None, return_new=False,
              approx=False):
    """
    Computes the `Kullback-Leibler (KL) divergence
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`_ *from* the
    discrete probability distribution defined by `res` *to* the discrete
    probability distribution defined by a **realization** of `res`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        :class:`~dynesty.results.Results` instance for the distribution we
        are computing the KL divergence *from*.

    error : {`'jitter'`, `'resample'`, `'simulate'`}, optional
        The error method employed, corresponding to :meth:`jitter_run`,
        :meth:`resample_run`, and :meth:`simulate_run`, respectively.
        Default is `'simulate'`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    return_new : bool, optional
        Whether to return the realization of the run used to compute the
        KL divergence. Default is `False`.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    kld : `~numpy.ndarray` with shape (nsamps,)
        The cumulative KL divergence defined *from* `res` *to* a
        random realization of `res`.

    new_res : :class:`~dynesty.results.Results` instance, optional
        The :class:`~dynesty.results.Results` instance corresponding to
        the random realization we computed the KL divergence *to*.

    """

    # Define our original importance weights.
    logp2 = res.logwt - res.logz[-1]

    # Compute a random realization of our run.
    if error == 'jitter':
        new_res = jitter_run(res, rstate=rstate, approx=approx)
    elif error == 'resample':
        new_res, samp_idx = resample_run(res, rstate=rstate, return_idx=True)
        logp2 = logp2[samp_idx]  # re-order our original results to match
    elif error == 'simulate':
        new_res, samp_idx = resample_run(res, rstate=rstate, return_idx=True)
        new_res = jitter_run(new_res)
        logp2 = logp2[samp_idx]  # re-order our original results to match
    else:
        raise ValueError("Input `'error'` option '{0}' is not valid."
                         .format(error))

    # Define our new importance weights.
    logp1 = new_res.logwt - new_res.logz[-1]

    # Compute the KL divergence.
    kld = np.cumsum(np.exp(logp1) * (logp1 - logp2))

    if return_new:
        return kld, new_res
    else:
        return kld


def _merge_two(res1, res2, compute_aux=False):
    """
    Internal method used to merges two runs with differing (possibly variable)
    numbers of live points into one run.

    Parameters
    ----------
    res1 : :class:`~dynesty.results.Results` instance
        The "base" nested sampling run.

    res2 : :class:`~dynesty.results.Results` instance
        The "new" nested sampling run.

    compute_aux : bool, optional
        Whether to compute auxiliary quantities (evidences, etc.) associated
        with a given run. **WARNING: these are only valid if `res1` or `res2`
        was initialized from the prior *and* their sampling bounds overlap.**
        Default is `False`.

    Returns
    -------
    res : :class:`~dynesty.results.Results` instances
        :class:`~dynesty.results.Results` instance from the newly combined
        nested sampling run.

    """

    # Initialize the first ("base") run.
    base_id = res1.samples_id
    base_u = res1.samples_u
    base_v = res1.samples
    base_logl = res1.logl
    base_nc = res1.ncall
    base_it = res1.samples_it
    nbase = len(base_id)

    # Number of live points throughout the run.
    try:
        base_n = res1.samples_n
    except:
        niter, nlive = res1.niter, res1.nlive
        if nbase == niter:
            base_n = np.ones(niter, dtype='int') * nlive
        elif nbase == (niter + nlive):
            base_n = np.append(np.ones(niter, dtype='int') * nlive,
                               np.arange(1, nlive + 1)[::-1])
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points in `res1`.")

    # Proposal information (if available).
    try:
        base_prop = res1.prop
        base_propidx = res1.samples_prop
        base_piter = res1.prop_iter
        base_scale = res1.scale
        base_proposals = True
    except:
        base_proposals = False

    # Batch information (if available).
    try:
        base_batch = res1.samples_batch
        base_bounds = res1.batch_bounds
    except:
        base_batch = np.zeros(nbase, dtype='int')
        base_bounds = np.array([(-np.inf, np.inf)])

    # Initialize the second ("new") run.
    new_id = res2.samples_id
    new_u = res2.samples_u
    new_v = res2.samples
    new_logl = res2.logl
    new_nc = res2.ncall
    new_it = res2.samples_it
    nnew = len(new_id)

    # Number of live points throughout the run.
    try:
        new_n = res2.samples_n
    except:
        niter, nlive = res2.niter, res2.nlive
        if nnew == niter:
            new_n = np.ones(niter, dtype='int') * nlive
        elif nnew == (niter + nlive):
            new_n = np.append(np.ones(niter, dtype='int') * nlive,
                              np.arange(1, nlive + 1)[::-1])
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points in `res2`.")

    # Proposal information (if available).
    try:
        new_prop = res2.prop
        new_propidx = res2.samples_prop
        new_piter = res2.prop_iter
        new_scale = res2.scale
        new_proposals = True
    except:
        new_proposals = False

    # Batch information (if available).
    try:
        new_batch = res2.samples_batch
        new_bounds = res2.batch_bounds
    except:
        new_batch = np.zeros(nnew, dtype='int')
        new_bounds = np.array([(-np.inf, np.inf)])

    # Initialize our new combind run.
    combined_id = []
    combined_u = []
    combined_v = []
    combined_logl = []
    combined_logvol = []
    combined_logwt = []
    combined_logz = []
    combined_logzvar = []
    combined_h = []
    combined_nc = []
    combined_propidx = []
    combined_it = []
    combined_n = []
    combined_piter = []
    combined_scale = []
    combined_batch = []

    # Check if proposal info is the same and modify counters accordingly.
    if base_proposals and new_proposals:
        if base_prop == new_prop:
            prop = base_prop
            poffset = 0
        else:
            prop = np.concatenate((base_prop, new_prop))
            poffset = len(base_prop)

    # Check if batch info is the same and modify counters accordingly.
    if np.all(base_bounds == new_bounds):
        bounds = base_bounds
        boffset = 0
    else:
        bounds = np.concatenate((base_bounds, new_bounds))
        boffset = len(base_bounds)

    # Start our counters at the beginning of each set of dead points.
    idx_base, idx_new = 0, 0
    logl_b, logl_n = base_logl[idx_base], new_logl[idx_new]
    nlive_b, nlive_n = base_n[idx_base], new_n[idx_new]

    # Iteratively walk through both set of samples to simulate
    # a combined run.
    ntot = nbase + nnew
    llmin_b = np.min(base_bounds[base_batch])
    llmin_n = np.min(new_bounds[new_batch])
    logvol = 0.
    for i in range(ntot):
        if logl_b > llmin_n and logl_n > llmin_b:
            # If our samples from the both runs are past the each others'
            # lower log-likelihood bound, both runs are now "active".
            nlive = nlive_b + nlive_n
        elif logl_b <= llmin_n:
            # If instead our collection of dead points from the "base" run
            # are below the bound, just use those.
            nlive = nlive_b
        else:
            # Our collection of dead points from the "new" run
            # are below the bound, so just use those.
            nlive = nlive_n

        # Increment our position along depending on
        # which dead point (saved or new) is worse.
        if logl_b <= logl_n:
            combined_id.append(base_id[idx_base])
            combined_u.append(base_u[idx_base])
            combined_v.append(base_v[idx_base])
            combined_logl.append(base_logl[idx_base])
            combined_nc.append(base_nc[idx_base])
            combined_it.append(base_it[idx_base])
            combined_batch.append(base_batch[idx_base])
            if base_proposals and new_proposals:
                combined_propidx.append(base_propidx[idx_base])
                combined_piter.append(base_piter[idx_base])
                combined_scale.append(base_scale[idx_base])
            idx_base += 1
        else:
            combined_id.append(new_id[idx_new])
            combined_u.append(new_u[idx_new])
            combined_v.append(new_v[idx_new])
            combined_logl.append(new_logl[idx_new])
            combined_nc.append(new_nc[idx_new])
            combined_it.append(new_it[idx_new])
            combined_batch.append(new_batch[idx_new] + boffset)
            if base_proposals and new_proposals:
                combined_propidx.append(new_propidx[idx_new] + poffset)
                combined_piter.append(new_piter[idx_new] + poffset)
                combined_scale.append(new_scale[idx_new])
            idx_new += 1

        # Save the number of live points and expected ln(volume).
        logvol -= math.log((nlive + 1.) / nlive)
        combined_n.append(nlive)
        combined_logvol.append(logvol)

        # Attempt to step along our samples. If we're out of samples,
        # set values to defaults.
        try:
            logl_b = base_logl[idx_base]
            nlive_b = base_n[idx_base]
        except:
            logl_b = np.inf
            nlive_b = 0
        try:
            logl_n = new_logl[idx_new]
            nlive_n = new_n[idx_new]
        except:
            logl_n = np.inf
            nlive_n = 0

    # Compute sampling efficiency.
    eff = 100. * ntot / sum(combined_nc)

    # Save results.
    r = [('niter', ntot),
         ('ncall', np.array(combined_nc)),
         ('eff', eff),
         ('samples', np.array(combined_v)),
         ('samples_id', np.array(combined_id)),
         ('samples_it', np.array(combined_it)),
         ('samples_n', np.array(combined_n)),
         ('samples_u', np.array(combined_u)),
         ('samples_batch', np.array(combined_batch)),
         ('logl', np.array(combined_logl)),
         ('logvol', np.array(combined_logvol)),
         ('batch_bounds', np.array(bounds))]

    # Add proposal information (if available).
    if base_proposals and new_proposals:
        r.append(('prop', prop))
        r.append(('prop_iter', np.array(combined_piter)))
        r.append(('samples_prop', np.array(combined_propidx)))
        r.append(('scale', np.array(combined_scale)))

    # Compute the posterior quantities of interest if desired.
    if compute_aux:
        h = 0.
        logz = -1.e300
        loglstar = -1.e300
        logzvar = 0.
        logvols_pad = np.concatenate(([0.], combined_logvol))
        logdvols = logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                             axis=1, b=np.c_[np.ones(ntot), -np.ones(ntot)])
        logdvols += math.log(0.5)
        dlvs = logvols_pad[:-1] - logvols_pad[1:]
        for i in range(ntot):
            loglstar_new = combined_logl[i]
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
            combined_logwt.append(logwt)
            combined_logz.append(logz)
            combined_logzvar.append(logzvar)
            combined_h.append(h)

        # Compute batch information.
        combined_id = np.array(combined_id)
        batch_nlive = [len(np.unique(combined_id[combined_batch == i]))
                       for i in np.unique(combined_batch)]

        # Add to our results.
        r.append(('logwt', np.array(combined_logwt)))
        r.append(('logz', np.array(combined_logz)))
        r.append(('logzerr', np.sqrt(np.array(combined_logzvar))))
        r.append(('h', np.array(combined_h)))
        r.append(('batch_nlive', np.array(batch_nlive, dtype='int')))

    # Combine to form final results object.
    res = Results(r)

    return res
