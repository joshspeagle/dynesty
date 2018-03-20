#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for proposing new live points used by
:class:`~dynesty.sampler.Sampler` (and its children from
:mod:`~dynesty.nestedsamplers`) and
:class:`~dynesty.dynamicsampler.DynamicSampler`.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import warnings
import math
import numpy as np
from numpy import linalg
from scipy import misc


__all__ = ["sample_unif", "sample_rwalk",
           "sample_slice", "sample_rslice", "sample_hslice"]

EPS = float(np.finfo(np.float64).eps)
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def sample_unif(args):
    """
    Evaluate a new point sampled uniformly from a bounding proposal
    distribution. Parameters are zipped within `args` to utilize
    `pool.map`-style functions.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample.

    loglstar : float
        Ln(likelihood) bound. **Not applicable here.**

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. **Not applicable here.**

    scale : float
        Value used to scale the provided axes. **Not applicable here.**

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.
        **Not applicable here.**

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube. **For
        uniform sampling this is the same as the initial input position.**

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample. For uniform
        sampling this is `1` by construction.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`. **Not
        applicable for uniform sampling.**

    """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args

    v = prior_transform(np.array(u))
    logl = loglikelihood(np.array(v))
    nc = 1
    blob = None

    return u, v, logl, nc, blob


def sample_rwalk(args):
    """
    Return a new live point proposed by random walking away from an
    existing live point.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. For random walks new positions are
        proposed using the :class:`~dynesty.bounding.Ellipsoid` whose
        shape is defined by axes.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    n = len(u)
    walks = kwargs.get('walks', 25)  # number of steps
    accept = 0
    reject = 0
    fail = 0
    nc = 0
    ncall = 0

    drhat, dr, du, u_prop, logl_prop = np.nan, np.nan, np.nan, np.nan, np.nan
    while nc < walks or accept == 0:
        while True:

            # Check scale-factor.
            if scale == 0.:
                raise RuntimeError("The random walk sampling is stuck! "
                                   "Some useful output quantities:\n"
                                   "u: {0}\n"
                                   "drhat: {1}\n"
                                   "dr: {2}\n"
                                   "du: {3}\n"
                                   "u_prop: {4}\n"
                                   "loglstar: {5}\n"
                                   "logl_prop: {6}\n"
                                   "axes: {7}\n"
                                   "scale: {8}."
                                   .format(u, drhat, dr, du, u_prop,
                                           loglstar, logl_prop, axes, scale))

            # Propose a direction on the unit n-sphere.
            drhat = rstate.randn(n)
            drhat /= linalg.norm(drhat)

            # Scale based on dimensionality.
            dr = drhat * rstate.rand()**(1./n)

            # Transform to proposal distribution.
            du = np.dot(axes, dr)
            u_prop = u + scale * du

            # Check unit cube constraints.
            if np.all(u_prop > 0.) and np.all(u_prop < 1.):
                break
            else:
                fail += 1

            # Check if we're stuck generating bad numbers.
            if fail > 500 * walks:
                warnings.warn("Random number generation appears to be "
                              "extremely inefficient. Bounding distributions "
                              "might be sub-optimal.")

        # Check proposed point.
        v_prop = prior_transform(np.array(u_prop))
        logl_prop = loglikelihood(np.array(v_prop))
        if logl_prop >= loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
        else:
            reject += 1
        nc += 1
        ncall += 1

        # Check if we're stuck generating bad points.
        if nc > 50 * walks:
            scale *= math.exp(-1. / n)
            warnings.warn("Random walk proposals appear to be "
                          "extremely inefficient. Adjusting the "
                          "scale-factor accordingly.")
            nc, accept, reject = 0, 0, 0  # reset values

    blob = {'accept': accept, 'reject': reject, 'fail': fail, 'scale': scale}

    return u, v, logl, ncall, blob


def sample_slice(args):
    """
    Return a new live point proposed by a series of random slices
    away from an existing live point. Standard "Gibs-like" implementation where
    a single multivariate "slice" is a combination of `ndim` univariate slices
    through each axis.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. For slices new positions are
        proposed along the arthogonal basis defined by :data:`axes`.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    n = len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    fscale = []

    # Modifying axes and computing lengths.
    axes = scale * axes.T  # scale based on past tuning
    axlens = [linalg.norm(axis) for axis in axes]

    # Slice sampling loop.
    for it in range(slices):

        # Shuffle axis update order.
        idxs = np.arange(n)
        rstate.shuffle(idxs)

        # Slice sample along a random direction.
        for idx in idxs:

            # Select axis.
            axis = axes[idx]
            axlen = axlens[idx]

            # Define starting "window".
            r = rstate.rand()  # initial scale/offset
            u_l = u - r * axis  # left bound
            if np.all(u_l > 0.) and np.all(u_l < 1.):
                v_l = prior_transform(np.array(u_l))
                logl_l = loglikelihood(np.array(v_l))
            else:
                logl_l = -np.inf
            nc += 1
            u_r = u + (1 - r) * axis  # right bound
            if np.all(u_r > 0.) and np.all(u_r < 1.):
                v_r = prior_transform(np.array(u_r))
                logl_r = loglikelihood(np.array(v_r))
            else:
                logl_r = -np.inf
            nc += 1

            # "Stepping out" the left and right bounds.
            while logl_l >= loglstar:
                u_l -= axis
                if np.all(u_l > 0.) and np.all(u_l < 1.):
                    v_l = prior_transform(np.array(u_l))
                    logl_l = loglikelihood(np.array(v_l))
                else:
                    logl_l = -np.inf
                nc += 1
            while logl_r >= loglstar:
                u_r += axis
                if np.all(u_r > 0.) and np.all(u_r < 1.):
                    v_r = prior_transform(np.array(u_r))
                    logl_r = loglikelihood(np.array(v_r))
                else:
                    logl_r = -np.inf
                nc += 1

            # Sample within limits. If the sample is not valid, shrink
            # the limits until we hit the `loglstar` bound.
            while True:
                u_hat = u_r - u_l
                u_prop = u_l + rstate.rand() * u_hat  # scale from left
                if np.all(u_prop > 0.) and np.all(u_prop < 1.):
                    v_prop = prior_transform(np.array(u_prop))
                    logl_prop = loglikelihood(np.array(v_prop))
                else:
                    logl_prop = -np.inf
                nc += 1
                # If we succeed, move to the new position.
                if logl_prop >= loglstar:
                    window = np.linalg.norm(u_hat)  # length of window
                    fscale.append(window / axlen)
                    u = u_prop
                    break
                # If we fail, check if the new point is to the left/right of
                # our original point along our proposal axis and update
                # the bounds accordingly.
                else:
                    s = np.dot(u_prop - u, u_hat)  # check sign (+/-)
                    if s < 0:  # left
                        u_l = u_prop
                    elif s > 0:  # right
                        u_r = u_prop
                    else:
                        raise RuntimeError("Slice sampler has failed to find "
                                           "a valid point. Some useful "
                                           "output quantities:\n"
                                           "u: {0}\n"
                                           "u_left: {1}\n"
                                           "u_right: {2}\n"
                                           "u_hat: {3}\n"
                                           "u_prop: {4}\n"
                                           "loglstar: {5}\n"
                                           "logl_prop: {6}\n"
                                           "axes: {7}\n"
                                           "axlens: {8}\n"
                                           "s: {9}."
                                           .format(u, u_l, u_r, u_hat, u_prop,
                                                   loglstar, logl_prop,
                                                   axes, axlens, s))

    blob = {'fscale': np.mean(fscale)}

    return u_prop, v_prop, logl_prop, nc, blob


def sample_rslice(args):
    """
    Return a new live point proposed by a series of random slices
    away from an existing live point. Standard "random" implementation where
    each slice is along a random direction based on the provided axes.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new slice directions.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    n = len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    fscale = []

    # Slice sampling loop.
    for it in range(slices):

        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Scale based on past tuning.
        axis = drhat * scale
        axlen = scale

        # Define starting "window".
        r = rstate.rand()  # initial scale/offset
        u_l = u - r * axis  # left bound
        if np.all(u_l > 0.) and np.all(u_l < 1.):
            v_l = prior_transform(np.array(u_l))
            logl_l = loglikelihood(np.array(v_l))
        else:
            logl_l = -np.inf
        nc += 1
        u_r = u + (1 - r) * axis  # right bound
        if np.all(u_r > 0.) and np.all(u_r < 1.):
            v_r = prior_transform(np.array(u_r))
            logl_r = loglikelihood(np.array(v_r))
        else:
            logl_r = -np.inf
        nc += 1

        # "Stepping out" the left and right bounds.
        while logl_l >= loglstar:
            u_l -= axis
            if np.all(u_l > 0.) and np.all(u_l < 1.):
                v_l = prior_transform(np.array(u_l))
                logl_l = loglikelihood(np.array(v_l))
            else:
                logl_l = -np.inf
            nc += 1
        while logl_r >= loglstar:
            u_r += axis
            if np.all(u_r > 0.) and np.all(u_r < 1.):
                v_r = prior_transform(np.array(u_r))
                logl_r = loglikelihood(np.array(v_r))
            else:
                logl_r = -np.inf
            nc += 1

        # Sample within limits. If the sample is not valid, shrink
        # the limits until we hit the `loglstar` bound.
        while True:
            u_hat = u_r - u_l
            u_prop = u_l + rstate.rand() * u_hat  # scale from left
            if np.all(u_prop > 0.) and np.all(u_prop < 1.):
                v_prop = prior_transform(np.array(u_prop))
                logl_prop = loglikelihood(np.array(v_prop))
            else:
                logl_prop = -np.inf
            nc += 1
            # If we succeed, move to the new position.
            if logl_prop >= loglstar:
                window = np.linalg.norm(u_hat)  # length of window
                fscale.append(window / axlen)
                u = u_prop
                break
            # If we fail, check if the new point is to the left/right of
            # our original point along our proposal axis and update
            # the bounds accordingly.
            else:
                s = np.dot(u_prop - u, u_hat)  # check sign (+/-)
                if s < 0:  # left
                    u_l = u_prop
                elif s > 0:  # right
                    u_r = u_prop
                else:
                    raise RuntimeError("Slice sampler has failed to find "
                                       "a valid point. Some useful "
                                       "output quantities:\n"
                                       "u: {0}\n"
                                       "u_left: {1}\n"
                                       "u_right: {2}\n"
                                       "u_hat: {3}\n"
                                       "u_prop: {4}\n"
                                       "loglstar: {5}\n"
                                       "logl_prop: {6}\n"
                                       "axis: {7}\n"
                                       "axlen: {8}\n"
                                       "s: {9}."
                                       .format(u, u_l, u_r, u_hat, u_prop,
                                               loglstar, logl_prop,
                                               axis, axlen, s))

    blob = {'fscale': np.mean(fscale)}

    return u_prop, v_prop, logl_prop, nc, blob


def sample_hslice(args):
    """
    Return a new live point proposed by Hamiltonian Slice Sampling
    using a series of random trajectories away from an existing live point.
    Each trajectory is based on the provided axes and samples are determined
    by slice sampling those (periodic) trajectories in *time*.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new slice directions.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    n = len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0

    # Slice sampling loop.
    for it in range(slices):

        # Random Gaussian proposal.
        vel = rstate.randn(n)

        # Define our starting "window" in time over our trajectory.
        t_l = -1e10  # "left" (past) direction
        t_r = 1e10  # "right" (future) direction

        # Sample in time between `t_l` and `t_r`. If the sample is not valid,
        # shrink the limits until we hit the `loglstar` bound.
        while True:

            # Sample time t.
            t_prop = rstate.uniform(t_l, t_r)

            # Compute new position x(t)
            u_prop = np.abs(np.remainder(vel * t_prop + u + 1., 2.) - 1.)

            # Check unit cube.
            if np.all(u_prop > 0.) and np.all(u_prop < 1.):
                v_prop = prior_transform(np.array(u_prop))
                logl_prop = loglikelihood(np.array(v_prop))
            else:
                logl_prop = -np.inf
            nc += 1

            # Update bounds.
            if logl_prop >= loglstar:
                # If we succeed, move to the new position.
                u = u_prop
                break
            else:
                # If we fail, check if the new point is to the left/right of
                # our original point along our proposal axis and update
                # the bounds accordingly.
                if t_prop <= 0:  # left
                    t_l = t_prop
                elif t_prop > 0:  # right
                    t_r = t_prop
                else:
                    raise RuntimeError("Slice sampler has failed to find "
                                       "a valid point. Some useful "
                                       "output quantities:\n"
                                       "u: {0}\n"
                                       "u_left: {1}\n"
                                       "u_right: {2}\n"
                                       "u_hat: {3}\n"
                                       "u_prop: {4}\n"
                                       "loglstar: {5}\n"
                                       "logl_prop: {6}\n"
                                       "axis: {7}\n"
                                       "axlen: {8}\n"
                                       "s: {9}."
                                       .format(u, u_l, u_r, u_hat, u_prop,
                                               loglstar, logl_prop,
                                               axis, axlen, s))

    blob = None

    return u_prop, v_prop, logl_prop, nc, blob
