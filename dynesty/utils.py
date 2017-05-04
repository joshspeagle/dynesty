#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Useful utilities.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math

import numpy as np

__all__ = ["resample_equal", "mean_and_cov", "print_progress",
           "random_choice"]

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


def resample_equal(samples, weights, rstate=None):
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

    if rstate is None:
        rstate = np.random

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


def print_progress(info):
    """
    Callback function that prints a running total on a single line.

    Parameters
    ----------
    info : dict
        Dictionary containing keys 'it' and 'logz'.

    """

    print("\r\033[Kit={:6d} logz={:8f}".format(info['it'], info['logz']),
          end='')
    sys.stdout.flush()  # because flush keyword not in print() in py2.7
