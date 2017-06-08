#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Useful utilities.

"""

from __future__ import (print_function, division)
from builtins import range

import sys
import warnings
import math
import scipy.misc as misc
import numpy as np

from .results import *

__all__ = ["resample_equal", "mean_and_cov", "random_choice",
           "simulate_run", "resample_run", "sample_run",
           "unravel_run", "merge_runs"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def random_choice(a, p, rstate=np.random):
    """Replacement for numpy.random.choice (only in numpy 1.7+)."""

    if abs(np.sum(p) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("probabilities do not sum to 1")

    r = rstate.rand()
    i = 0
    t = p[i]
    while t < r:
        i += 1
        t += p[i]

    return i


def mean_and_cov(samples, weights):
    """
    Compute weighted sample mean and covariance.


    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples, ndim)
        2-D array containing data samples. This ordering is equivalent to
        using `rowvar=False` in `numpy.cov`.

    weights : `~numpy.ndarray` with shape (nsamples,)
        1-D array of sample weights.


    Returns
    -------
    mean : `~numpy.ndarray` with shape (ndim,)
        Weighted sample means.

    cov : `~numpy.ndarray` with shape (ndim, ndim)
        Weighted sample covariances.


    Notes
    -----
    Implements the formulae in the "weighted samples" section on
    <https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance>.

    """

    mean = np.average(samples, weights=weights, axis=0)

    dx = samples - mean
    wsum = np.sum(weights)
    w2sum = np.sum(weights**2)

    cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', weights, dx, dx)

    return mean, cov


def resample_equal(samples, weights, rstate=np.random):
    """
    Resample the samples so that the final samples all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times, with
    `floor` or `ceil` randomly selected (weighted by proximity).


    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Unequally weighted samples returned by the nested sampling algorithm.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.


    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New samples with equal weights.


    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> nestle.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])


    Notes
    -----
    Implements the systematic resampling method described in Hol, Schon, and
    Gustafsson (2006), which can be found at <doi:10.1109/NSSPW.2006.4378824>.
    This gives less "noisy" samples as compared to standard multinomial
    resampling techniques.

    """

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("Weights do not sum to 1.")

    nsamples = len(weights)

    # Make N subdivisions and choose positions with a consistent random offset.
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    idx = np.zeros(nsamples, dtype=np.int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


def simulate_run(res, rstate=np.random):
    """
    Probes uncertainties on a run with `K` live points by drawing
    a sample from the statistical distribution of prior volumes associated
    with the parameter samples collected over the course of a run. Uses the
    results to compute new realizations of the evidence and weights.

    This probes errors due to intrinsic *statistical* uncertainties.


    Parameters
    ----------
    res : `Results` instance
        The `Results` instance returned from a previous run.

    rstate: `~numpy.random` instance
        Random state.


    Returns
    -------
    new_res : `Results` instance
        A new `Results` instance based on our "simulated" weights.

    """

    # Initialize evolution of live points over the course of the run.
    logl = res.logl
    try:
        samples_n = res.samples_n
        nsamps = len(samples_n)
    except:
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

    # Simulate the prior volume shrinkage associated with our set of "dead"
    # points. At each iteration, if the number of live points is constant or
    # increasing, our prior volume compresses by the maximum value of a set
    # of `K_i` uniformly distributed random numbers. If instead the number
    # of live points is decreasing, that means we're instead sampling down
    # a set of uniform random variables (i.e. uniform order statistics).
    nlive_flag = np.ones(nsamps, dtype='bool')

    # Find all instances where the number of live points is either constant
    # or increasing.
    nlive_flag[1:] = np.diff(samples_n) >= 0

    # For all the portions that are decreasing, find out where they start,
    # where they end, and how many live points are present at that given
    # iteration.
    bounds = []
    nlive_start = []
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
    t_arr = np.empty(nsamps)
    t_arr[nlive_flag] = rstate.beta(a=samples_n[nlive_flag], b=1)

    # If we instead are sampling the set of uniform order statistics,
    # we note that the jth largest value is marginally distributed as
    # `Beta(j, K_i-j+1)`. The full joint distribution is::
    #
    # X_(j) / X_N = (Y_1 + ... + Y_j) / (Y_1 + ... + Y_{K+1})
    #
    # where X_(j) is the prior volume of the live point with the `j`-th
    # *lowest* likelihood and the `Y_i`'s are i.i.d. exponentially
    # distributed random variables.
    nunif = len(nlive_start)
    for i in range(nunif):
        nstart = nlive_start[i]
        bound = bounds[i]
        sn = samples_n[bound[0]:bound[1]]
        y_arr = rstate.exponential(scale=1.0, size=nstart+1)
        ycsum = y_arr.cumsum()
        ycsum /= ycsum[-1]
        uorder = ycsum[np.append(nstart, sn-1)]
        rorder = uorder[1:]/uorder[:-1]
        t_arr[bound[0]:bound[1]] = rorder

    # These are the "compression factors" at each iteration. Now let's turn
    # these into associated ln(volumes).
    logvol = np.log(t_arr).cumsum()

    # Compute weights using quadratic estimator.
    h = 0.
    logz = -1.e300
    loglstar = -1.e300
    logzvar = 0.
    logvols_pad = np.concatenate(([0.], logvol))
    logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                              axis=1, b=np.c_[np.ones(nsamps),
                                              -np.ones(nsamps)])
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
        h_new = _check_h(h_new)
        dh = h_new - h
        h = h_new
        logz = logz_new
        logzvar += dh * dlv
        loglstar = loglstar_new
        saved_logwt.append(logwt)
        saved_logz.append(logz)
        saved_logzvar.append(logzvar)
        saved_h.append(h)

    # Save results.
    new_res = Results([item for item in res.items()])
    new_res.logvol = np.array(logvol)
    new_res.logwt = np.array(saved_logwt)
    new_res.logz = np.array(saved_logz)
    new_res.logzerr = np.sqrt(np.array(saved_logzvar))
    new_res.h = np.array(saved_h)

    return new_res


def resample_run(res, rstate=np.random):
    """
    Probes uncertainties on a run with `K` live points by splitting the
    run into `K` strands, sampling from them with replacement, and combining
    the `K` resampled strands into a new run.

    This probes errors due to intrinsic *sampling* uncertainties.


    Parameters
    ----------
    res : `Results` instance
        The `Results` instance returned from a previous run.


    Returns
    -------
    new_res : `Results` instance
        A new `Results` instance based on our "resampled" weights.

    """

    # Check whether the final set of live points were added to the
    # run.
    nlive = res.nlive
    niter = res.niter
    nsamps = len(res.ncall)
    try:
        samples_n = res.samples_n  # check for dynamic run
    except:
        if nsamps == niter:
            samples_n = np.ones(niter, dtype='int') * nlive
        elif nsamps == (niter + nlive):
            samples_n = np.append(np.ones(niter, dtype='int') * nlive,
                                  np.arange(1, nlive + 1)[::-1])
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")

    # Resample strands.
    ids = np.unique(res.samples_id)
    nunique = len(ids)
    live_idx = rstate.randint(0, nunique, size=nunique)

    # Find corresponding indices within the original run.
    samp_idx = np.arange(len(res.ncall))
    samp_idx = np.concatenate([samp_idx[res.samples_id == ids[idx]]
                               for idx in live_idx])

    # Derive new sample size.
    nsamps = len(samp_idx)

    # Sort the loglikelihoods (there will be duplicates).
    logls = res.logl[samp_idx]
    idx_sort = np.argsort(logls)
    samp_idx = samp_idx[idx_sort]

    # Construct the new run.
    samp_n = samples_n[samp_idx]
    logl = res.logl[samp_idx]

    # Assign log(volume) to samples.
    logvol = np.cumsum(np.log(samp_n / (samp_n + 1.)))

    # Computing weights using quadratic estimator.
    h = 0.
    logz = -1.e300
    loglstar = -1.e300
    logzvar = 0.
    logvols_pad = np.concatenate(([0.], logvol))
    logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                              axis=1, b=np.c_[np.ones(nsamps),
                                              -np.ones(nsamps)])
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
        h_new = _check_h(h_new)
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
    eff = 100. * niter / sum(res.ncall[samp_idx])

    # Save results.
    new_res = Results([('nlive', nlive),
                       ('niter', niter),
                       ('ncall', res.ncall[samp_idx]),
                       ('eff', eff),
                       ('samples', res.samples[samp_idx]),
                       ('samples_id', res.samples_id[samp_idx]),
                       ('samples_it', res.samples_it[samp_idx]),
                       ('samples_u', res.samples_u[samp_idx]),
                       ('samples_n', samp_n),
                       ('logwt', np.array(saved_logwt)),
                       ('logl', logl),
                       ('logvol', logvol),
                       ('logz', np.array(saved_logz)),
                       ('logzerr', np.sqrt(np.array(saved_logzvar))),
                       ('h', np.array(saved_h))])

    return new_res


def sample_run(res, rstate=np.random):
    """
    Probes uncertainties on a run with `K` live points by (1) splitting the
    run into `K` strands, sampling from them with replacement, and combining
    the `K` resampled strands into a new run and then (2) sampling from the
    joint distribution of possible prior volumes.

    This probes errors due to both statistical *and* sampling uncertainties.


    Parameters
    ----------
    res : `Results` instance
        The `Results` instance returned from a previous run.


    Returns
    -------
    new_res : `Results` instance
        A new `Results` instance based on our "resampled" weights.

    """

    # Resample run.
    new_res = resample_run(res, rstate=np.random)

    # Simulate weights.
    new_res = simulate_run(new_res, rstate=np.random)

    return new_res


def unravel_run(res):
    """
    Unravels a run with `K` live points into `K` "strands" (a nested sampling
    run with only 1 live point). **Note that the anciliary quantities provided
    with the "unraveling" are only valid if the baseline nested sampling run
    used a with constant number of live points.**


    Parameters
    ----------
    res : `Results` instance
        The `Results` instance returned from a previous run.


    Returns
    -------
    new_res : list of `K` `Results` instances
        A list of `K` new `Results` instances for each individual strand.

    """

    idxs = res.samples_id  # label for each live/dead point

    # Check if we added in the set of dead points.
    if len(idxs) == (res.niter + res.nlive):
        added_live = True
    else:
        added_live = False

    # Recreate the nested sampling run for each strand.
    new_res = []
    for idx in np.unique(idxs):
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
        logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                                  axis=1, b=np.c_[np.ones(nsamps),
                                                  -np.ones(nsamps)])
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
            h_new = _check_h(h_new)
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
        new_res.append(Results([('nlive', 1),
                                ('niter', niter),
                                ('ncall', res.ncall[strand]),
                                ('eff', eff),
                                ('samples', res.samples[strand]),
                                ('samples_id', res.samples_id[strand]),
                                ('samples_it', res.samples_it[strand]),
                                ('samples_u', res.samples_u[strand]),
                                ('logwt', logwt),
                                ('logl', logl),
                                ('logvol', logvol),
                                ('logz', np.array(saved_logz)),
                                ('logzerr', np.sqrt(np.array(saved_logzvar))),
                                ('h', np.array(saved_h))]))

    return new_res


def merge_runs(res_list):
    """
    Merges a set of runs with `K_1`, `K_2`, ... live points into one run with
    `K_1 + K_2 + ...` live points. **Note that the anciliary quantities
    provided with the "merging" are only valid if the baseline nested
    sampling runs (1) used a with constant number of live points, (2) had the
    same stopping criteria and (3) consistently did/did not add the
    final set of live points.**


    Parameters
    ----------
    res_list : list of `Results` instances
        A list of `Results` instances returned from a previous runs.


    Returns
    -------
    combined_res : `Results` instances
        A single `Results` instance.

    """

    # Compute combined properties.
    nlive = sum([res.nlive for res in res_list])
    nsamps = sum([len(res.ncall) for res in res_list])

    # Check if we added in the final set of live points.
    if res_list[0].niter + res_list[0].nlive == len(res_list[0].ncall):
        added_live = True
        niter = nsamps - nlive
    else:
        added_live = False
        niter = nsamps

    # Sort the loglikelihoods.
    logls = np.concatenate([res.logl for res in res_list], axis=0)
    idx_sort = np.argsort(logls)
    logl = logls[idx_sort]

    # Assign log(volume) to samples.
    logvol = math.log(nlive / (1. + nlive)) * (1. + np.arange(niter))
    if added_live:
        logvol_live = logvol[-1]
        logvol_live += np.log(1. - (1. + np.arange(nlive)) / (nlive + 1.))
        logvol = np.append(logvol, logvol_live)

    # Compute weights using quadratic estimator.
    h = 0.
    logz = -1.e300
    loglstar = -1.e300
    logzvar = 0.
    logvols_pad = np.concatenate(([0.], logvol))
    logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                              axis=1, b=np.c_[np.ones(nsamps),
                                              -np.ones(nsamps)])
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
        h_new = _check_h(h_new)
        dh = h_new - h
        h = h_new
        logz = logz_new
        logzvar += dh * dlv
        loglstar = loglstar_new
        saved_logwt.append(logwt)
        saved_logz.append(logz)
        saved_logzvar.append(logzvar)
        saved_h.append(h)

    # Concatenating quantities.
    ncall = np.concatenate([res.ncall for res in res_list], axis=0)
    samples = np.concatenate([res.samples for res in res_list], axis=0)
    samples_id = np.concatenate([res.samples_id for res in res_list], axis=0)
    samples_it = np.concatenate([res.samples_it for res in res_list], axis=0)
    samples_u = np.concatenate([res.samples_u for res in res_list], axis=0)

    # Compute sampling efficiency.
    eff = 100. * nsamps / sum(ncall)

    # Save results.
    combined_res = Results([('nlive', nlive),
                            ('niter', niter),
                            ('ncall', ncall[idx_sort]),
                            ('eff', eff),
                            ('samples', samples[idx_sort]),
                            ('samples_id', samples_id[idx_sort]),
                            ('samples_it', samples_it[idx_sort]),
                            ('samples_u', samples_u[idx_sort]),
                            ('logwt', logwt),
                            ('logl', logl),
                            ('logvol', logvol),
                            ('logz', np.array(saved_logz)),
                            ('logzerr', np.sqrt(np.array(saved_logzvar))),
                            ('h', np.array(saved_h))])

    return combined_res


def _check_h(h):
    """Check whether information is non-negative
    to numerical precision. Numerical error can make it negative in
    pathological corner cases."""

    if h < 0.0:
        if h > -SQRTEPS:
            h = 0.0
        else:
            raise RuntimeError("Negative h encountered (h={}). Please "
                               "report this as a likely bug.".format(h))

    return h
