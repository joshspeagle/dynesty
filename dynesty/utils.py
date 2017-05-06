#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Useful utilities.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import scipy.misc as misc
import numpy as np

from .results import *

__all__ = ["resample_equal", "mean_and_cov", "random_choice",
           "simulate_weights"]

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

    mean = np.average(x, weights=weights, axis=0)

    dx = x - mean
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


def sample_run(res, rstate=np.random):
    """
    Draws a sample from the statistical distribution of prior volumes
    associated with the parameter samples collected over the course of a run
    and uses them to compute new realizations of the evidence and weights.


    Parameters
    ----------
    res : `Results` instance
        The `Results` instance returned from a previous run.


    Returns
    -------
    new_res : `Results` instance
        A new `Results` instance based on our "simulated" weights.

    """

    niter = res.niter
    nlive = res.nlive
    nsamps = niter + nlive

    # Simulate the prior volume shrinkage associated with each "dead" point.
    # This is the maximum value out of `K` uniformly distributed random
    # variables, which is a `Beta(K, 1)`-distributed random variable.
    # Since each sample is independent, we have `N` of these.
    t_arr = rstate.beta(a=nlive, b=1, size=niter)
    logvol_dead = np.log(t_arr).cumsum()

    # Simulate the prior volume associated with the leftover "live" points we
    # added to our final results. These are `K` uniformly distributed random
    # variables within the prior volume contained in the final dead point
    # `X_N`. The order statistics for these are jointly distributed as::
    #
    # X_(j) / X_N = (Y_1 + ... + Y_j) / (Y_1 + ... + Y_{K+1})
    #
    # where X_(j) is the prior volume of the live point with the `j`-th
    # *lowest* likelihood and the `Y_i`'s are i.i.d. exponentially
    # distributed random variables.
    y_arr = rstate.exponential(scale=1.0, size=nlive+1)
    log_unif_order = np.log(y_arr.cumsum()[::-1])[1:] - np.log(y_arr.sum())
    logvol_live = logvol_dead[-1] + log_unif_order

    # Combined the dead point and live point sequences.
    logvol = np.append(logvol_dead, logvol_live)

    # Compute weights using quadratic estimator.
    logvol_pad = np.concatenate(([0.], logvol, [-1.e300]))
    logdvol = misc.logsumexp(a=np.c_[logvol_pad[:-2], logvol_pad[2:]],
                             axis=1,
                             b=np.c_[np.ones(nsamps), -np.ones(nsamps)])
    logdvol += math.log(0.5)
    logwt = logdvol + res.logl

    # Compute cumulative quantities.
    h = 0.0  # Information, initially *0.*
    logz = -1.e300  # log(evidence), initially *0.*
    saved_logz, saved_h, saved_logzerr = [], [], []
    for i in xrange(nsamps):
        logz_new = np.logaddexp(logz, logwt[i])
        h = (math.exp(logwt[i] - logz_new) * res.logl[i] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        saved_logz.append(logz)
        saved_h.append(h)
        saved_logzerr.append(math.sqrt(h / nlive))

    # Save results.
    new_res = Results([('nlive', nlive),
                       ('niter', niter),
                       ('ncall', res.ncall),
                       ('eff', res.eff),
                       ('samples', res.samples),
                       ('samples_id', res.samples_id),
                       ('samples_u', res.samples_u),
                       ('logwt', logwt),
                       ('logl', res.logl),
                       ('logvol', logvol),
                       ('logz', np.array(saved_logz)),
                       ('logzerr', np.array(saved_logzerr)),
                       ('h', np.array(saved_h))])

    return new_res
