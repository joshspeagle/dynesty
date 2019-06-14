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

import warnings
import math
import numpy as np
from numpy import linalg

from .utils import unitcheck


__all__ = ["sample_unif", "sample_rwalk", "sample_rstagger",
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

    # Evaluate.
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

    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)

    # Setup.
    n = len(u)
    walks = kwargs.get('walks', 25)  # number of steps
    accept = 0
    reject = 0
    fail = 0
    nfail = 0
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

            # Wrap periodic parameters and reflect non-perioridc parameters.
            if nonperiodic is not None:
                u_prop[~nonperiodic] = np.mod(u_prop[~nonperiodic], 1)
                u_prop_ref = u_prop[nonperiodic]
                u_prop[nonperiodic] = np.minimum(np.maximum(u_prop_ref,
                                                            abs(u_prop_ref)),
                                                 2 - u_prop_ref)

            # Check unit cube constraints.
            if unitcheck(u_prop, nonperiodic):
                break
            else:
                fail += 1
                nfail += 1

            # Check if we're stuck generating bad numbers.
            if fail > 100 * walks:
                warnings.warn("Random number generation appears to be "
                              "extremely inefficient. Adjusting the "
                              "scale-factor accordingly.")
                fail = 0
                scale *= math.exp(-1. / n)

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

    blob = {'accept': accept, 'reject': reject, 'fail': nfail, 'scale': scale}

    return u, v, logl, ncall, blob


def sample_rstagger(args):
    """
    Return a new live point proposed by random "staggering" away from an
    existing live point. The difference between this and the random walk is
    the step size is exponentially adjusted to reach a target acceptance rate
    *during* each proposal (in addition to *between* proposals).

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

    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)

    # Setup.
    n = len(u)
    walks = kwargs.get('walks', 25)  # number of steps
    facc_target = kwargs.get('facc', 0.5)  # acceptance fraction
    accept = 0
    reject = 0
    fail = 0
    nfail = 0
    nc = 0
    ncall = 0
    stagger = 1.  # adaptive scale

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
            u_prop = u + scale * stagger * du

            # Wrap periodic parameters and reflect non-perioridc parameters.
            if nonperiodic is not None:
                u_prop[~nonperiodic] = np.mod(u_prop[~nonperiodic], 1)
                u_prop_ref = u_prop[nonperiodic]
                u_prop[nonperiodic] = np.minimum(np.maximum(u_prop_ref,
                                                            abs(u_prop_ref)),
                                                 2 - u_prop_ref)

            # Check unit cube constraints.
            if unitcheck(u_prop, nonperiodic):
                break
            else:
                fail += 1
                nfail += 1

            # Check if we're stuck generating bad numbers.
            if fail > 100 * walks:
                warnings.warn("Random number generation appears to be "
                              "extremely inefficient. Adjusting the "
                              "scale-factor accordingly.")
                fail = 0
                scale *= math.exp(-1. / n)

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

        # Adjust `stagger` to target an acceptance ratio of `facc_target`.
        facc = 1. * accept / (accept + reject)
        if facc > facc_target:
            stagger *= np.exp(1. / accept)
        if facc < facc_target:
            stagger /= math.exp(1. / reject)

        # Check if we're stuck generating bad points.
        if nc > 50 * walks:
            scale *= math.exp(-1. / n)
            warnings.warn("Random walk proposals appear to be "
                          "extremely inefficient. Adjusting the "
                          "scale-factor accordingly.")
            nc, accept, reject = 0, 0, 0  # reset values

    blob = {'accept': accept, 'reject': reject, 'fail': nfail, 'scale': scale}

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

    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)

    # Setup.
    n = len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    nexpand = 0
    ncontract = 0
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
            if unitcheck(u_l, nonperiodic):
                v_l = prior_transform(np.array(u_l))
                logl_l = loglikelihood(np.array(v_l))
            else:
                logl_l = -np.inf
            nc += 1
            nexpand += 1
            u_r = u + (1 - r) * axis  # right bound
            if unitcheck(u_r, nonperiodic):
                v_r = prior_transform(np.array(u_r))
                logl_r = loglikelihood(np.array(v_r))
            else:
                logl_r = -np.inf
            nc += 1
            nexpand += 1

            # "Stepping out" the left and right bounds.
            while logl_l >= loglstar:
                u_l -= axis
                if unitcheck(u_l, nonperiodic):
                    v_l = prior_transform(np.array(u_l))
                    logl_l = loglikelihood(np.array(v_l))
                else:
                    logl_l = -np.inf
                nc += 1
                nexpand += 1
            while logl_r >= loglstar:
                u_r += axis
                if unitcheck(u_r, nonperiodic):
                    v_r = prior_transform(np.array(u_r))
                    logl_r = loglikelihood(np.array(v_r))
                else:
                    logl_r = -np.inf
                nc += 1
                nexpand += 1

            # Sample within limits. If the sample is not valid, shrink
            # the limits until we hit the `loglstar` bound.
            while True:
                u_hat = u_r - u_l
                u_prop = u_l + rstate.rand() * u_hat  # scale from left
                if unitcheck(u_prop, nonperiodic):
                    v_prop = prior_transform(np.array(u_prop))
                    logl_prop = loglikelihood(np.array(v_prop))
                else:
                    logl_prop = -np.inf
                nc += 1
                ncontract += 1
                # If we succeed, move to the new position.
                if logl_prop >= loglstar:
                    window = linalg.norm(u_hat)  # length of window
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

    blob = {'fscale': np.mean(fscale),
            'nexpand': nexpand, 'ncontract': ncontract}

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

    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)

    # Setup.
    n = len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    nexpand = 0
    ncontract = 0
    fscale = []

    # Slice sampling loop.
    for it in range(slices):

        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Transform and scale based on past tuning.
        axis = np.dot(axes, drhat) * scale
        axlen = linalg.norm(axis)

        # Define starting "window".
        r = rstate.rand()  # initial scale/offset
        u_l = u - r * axis  # left bound
        if unitcheck(u_l, nonperiodic):
            v_l = prior_transform(np.array(u_l))
            logl_l = loglikelihood(np.array(v_l))
        else:
            logl_l = -np.inf
        nc += 1
        nexpand += 1
        u_r = u + (1 - r) * axis  # right bound
        if unitcheck(u_r, nonperiodic):
            v_r = prior_transform(np.array(u_r))
            logl_r = loglikelihood(np.array(v_r))
        else:
            logl_r = -np.inf
        nc += 1
        nexpand += 1

        # "Stepping out" the left and right bounds.
        while logl_l >= loglstar:
            u_l -= axis
            if unitcheck(u_l, nonperiodic):
                v_l = prior_transform(np.array(u_l))
                logl_l = loglikelihood(np.array(v_l))
            else:
                logl_l = -np.inf
            nc += 1
            nexpand += 1
        while logl_r >= loglstar:
            u_r += axis
            if unitcheck(u_r, nonperiodic):
                v_r = prior_transform(np.array(u_r))
                logl_r = loglikelihood(np.array(v_r))
            else:
                logl_r = -np.inf
            nc += 1
            nexpand += 1

        # Sample within limits. If the sample is not valid, shrink
        # the limits until we hit the `loglstar` bound.
        while True:
            u_hat = u_r - u_l
            u_prop = u_l + rstate.rand() * u_hat  # scale from left
            if unitcheck(u_prop, nonperiodic):
                v_prop = prior_transform(np.array(u_prop))
                logl_prop = loglikelihood(np.array(v_prop))
            else:
                logl_prop = -np.inf
            nc += 1
            ncontract += 1
            # If we succeed, move to the new position.
            if logl_prop >= loglstar:
                window = linalg.norm(u_hat)  # length of window
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

    blob = {'fscale': np.mean(fscale),
            'nexpand': nexpand, 'ncontract': ncontract}

    return u_prop, v_prop, logl_prop, nc, blob


def sample_hslice(args):
    """
    Return a new live point proposed by "Hamiltonian" Slice Sampling
    using a series of random trajectories away from an existing live point.
    Each trajectory is based on the provided axes and samples are determined
    by moving forwards/backwards in time until the trajectory hits an edge
    and approximately reflecting off the boundaries.
    Once a series of reflections has been established, we propose a new live
    point by slice sampling across the entire path.

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

    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)

    # Setup.
    n = len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    grad = kwargs.get('grad', None)  # gradient of log-likelihood
    max_move = kwargs.get('max_move', 100)  # limit for `ncall`
    compute_jac = kwargs.get('compute_jac', False)  # whether Jacobian needed
    jitter = 0.25  # 25% jitter
    nc = 0
    nmove = 0
    nreflect = 0
    ncontract = 0

    # Slice sampling loop.
    for it in range(slices):
        # Define the left, "inner", and right "nodes" for a given chord.
        # We will plan to slice sampling using these chords.
        nodes_l, nodes_m, nodes_r = [], [], []

        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Transform and scale based on past tuning.
        axis = np.dot(axes, drhat) * scale * 0.01

        # Create starting window.
        vel = np.array(axis)  # current velocity
        u_l = u - rstate.uniform(1. - jitter, 1. + jitter) * vel
        u_r = u + rstate.uniform(1. - jitter, 1. + jitter) * vel
        nodes_l.append(np.array(u_l))
        nodes_m.append(np.array(u))
        nodes_r.append(np.array(u_r))

        # Progress "right" (i.e. "forwards" in time).
        reverse, reflect = False, False
        u_r = np.array(u)
        ncall = 0
        while ncall <= max_move:

            # Iterate until we can bracket the edge of the distribution.
            nodes_l.append(np.array(u_r))
            u_out, u_in = None, []
            while True:
                # Step forward.
                u_r += rstate.uniform(1. - jitter, 1. + jitter) * vel
                # Evaluate point.
                if unitcheck(u_r, nonperiodic):
                    v_r = prior_transform(np.array(u_r))
                    logl_r = loglikelihood(np.array(v_r))
                    nc += 1
                    ncall += 1
                    nmove += 1
                else:
                    logl_r = -np.inf
                # Check if we satisfy the log-likelihood constraint
                # (i.e. are "in" or "out" of bounds).
                if logl_r < loglstar:
                    if reflect:
                        # If we are out of bounds and just reflected, we
                        # reverse direction and terminate immediately.
                        reverse = True
                        nodes_l.pop()  # remove since chord does not exist
                        break
                    else:
                        # If we're already in bounds, then we're safe.
                        u_out = np.array(u_r)
                        logl_out = logl_r
                    # Check if we could compute gradients assuming we
                    # terminated with the current `u_out`.
                    if np.isfinite(logl_out):
                        reverse = False
                    else:
                        reverse = True
                else:
                    reflect = False
                    u_in.append(np.array(u_r))
                # Check if we've bracketed the edge.
                if u_out is not None:
                    break
            # Define the rest of our chord.
            if len(nodes_l) == len(nodes_r) + 1:
                try:
                    u_in = u_in[rstate.choice(len(u_in))]  # pick point randomly
                except:
                    u_in = np.array(u)
                    pass
                nodes_m.append(np.array(u_in))
                nodes_r.append(np.array(u_out))
            # Check if we have turned around.
            if reverse:
                break

            # Reflect off the boundary.
            u_r, logl_r = u_out, logl_out
            if grad is None:
                # If the gradient is not provided, we will attempt to
                # approximate it numerically using 2nd-order methods.
                h = np.zeros(n)
                for i in range(n):
                    u_r_l, u_r_r = np.array(u_r), np.array(u_r)
                    # right side
                    u_r_r[i] += 1e-10
                    if unitcheck(u_r_r, nonperiodic):
                        v_r_r = prior_transform(np.array(u_r_r))
                        logl_r_r = loglikelihood(np.array(v_r_r))
                    else:
                        logl_r_r = -np.inf
                        reverse = True  # can't compute gradient
                    nc += 1
                    # left side
                    u_r_l[i] -= 1e-10
                    if unitcheck(u_r_l, nonperiodic):
                        v_r_l = prior_transform(np.array(u_r_l))
                        logl_r_l = loglikelihood(np.array(v_r_l))
                    else:
                        logl_r_l = -np.inf
                        reverse = True  # can't compute gradient
                    if reverse:
                        break  # give up because we have to turn around
                    nc += 1
                    # compute dlnl/du
                    h[i] = (logl_r_r - logl_r_l) / 2e-10
            else:
                # If the gradient is provided, evaluate it.
                h = grad(v_r)
                if compute_jac:
                    jac = []
                    # Evaluate and apply Jacobian dv/du if gradient
                    # is defined as d(lnL)/dv instead of d(lnL)/du.
                    for i in range(n):
                        u_r_l, u_r_r = np.array(u_r), np.array(u_r)
                        # right side
                        u_r_r[i] += 1e-10
                        if unitcheck(u_r_r, nonperiodic):
                            v_r_r = prior_transform(np.array(u_r_r))
                        else:
                            reverse = True  # can't compute Jacobian
                            v_r_r = np.array(v_r)  # assume no movement
                        # left side
                        u_r_l[i] -= 1e-10
                        if unitcheck(u_r_l, nonperiodic):
                            v_r_l = prior_transform(np.array(u_r_l))
                        else:
                            reverse = True  # can't compute Jacobian
                            v_r_r = np.array(v_r)  # assume no movement
                        if reverse:
                            break  # give up because we have to turn around
                        jac.append((v_r_r - v_r_l) / 2e-10)
                    jac = np.array(jac)
                    h = np.dot(jac, h)  # apply Jacobian
                nc += 1
            # Compute specular reflection off boundary.
            vel_ref = vel - 2 * h * np.dot(vel, h) / linalg.norm(h)**2
            dotprod = np.dot(vel_ref, vel)
            dotprod /= linalg.norm(vel_ref) * linalg.norm(vel)
            # Check angle of reflection.
            if dotprod < -0.99:
                # The reflection angle is sufficiently small that it might
                # as well be a reflection.
                reverse = True
                break
            else:
                # If the reflection angle is sufficiently large, we
                # proceed as normal to the new position.
                vel = vel_ref
                u_out = None
                reflect = True
                nreflect += 1

        # Progress "left" (i.e. "backwards" in time).
        reverse, reflect = False, False
        vel = -np.array(axis)  # current velocity
        u_l = np.array(u)
        ncall = 0
        while ncall <= max_move:

            # Iterate until we can bracket the edge of the distribution.
            # Use a doubling approach to try and locate the bounds faster.
            nodes_r.append(np.array(u_l))
            u_out, u_in = None, []
            while True:
                # Step forward.
                u_l += rstate.uniform(1. - jitter, 1. + jitter) * vel
                # Evaluate point.
                if unitcheck(u_l, nonperiodic):
                    v_l = prior_transform(np.array(u_l))
                    logl_l = loglikelihood(np.array(v_l))
                    nc += 1
                    ncall += 1
                    nmove += 1
                else:
                    logl_l = -np.inf
                # Check if we satisfy the log-likelihood constraint
                # (i.e. are "in" or "out" of bounds).
                if logl_l < loglstar:
                    if reflect:
                        # If we are out of bounds and just reflected, we
                        # reverse direction and terminate immediately.
                        reverse = True
                        nodes_r.pop()  # remove since chord does not exist
                        break
                    else:
                        # If we're already in bounds, then we're safe.
                        u_out = np.array(u_l)
                        logl_out = logl_l
                    # Check if we could compute gradients assuming we
                    # terminated with the current `u_out`.
                    if np.isfinite(logl_out):
                        reverse = False
                    else:
                        reverse = True
                else:
                    reflect = False
                    u_in.append(np.array(u_l))
                # Check if we've bracketed the edge.
                if u_out is not None:
                    break
            # Define the rest of our chord.
            if len(nodes_r) == len(nodes_l) + 1:
                try:
                    u_in = u_in[rstate.choice(len(u_in))]  # pick point randomly
                except:
                    u_in = np.array(u)
                    pass
                nodes_m.append(np.array(u_in))
                nodes_l.append(np.array(u_out))
            # Check if we have turned around.
            if reverse:
                break

            # Reflect off the boundary.
            u_l, logl_l = u_out, logl_out
            if grad is None:
                # If the gradient is not provided, we will attempt to
                # approximate it numerically using 2nd-order methods.
                h = np.zeros(n)
                for i in range(n):
                    u_l_l, u_l_r = np.array(u_l), np.array(u_l)
                    # right side
                    u_l_r[i] += 1e-10
                    if unitcheck(u_l_r, nonperiodic):
                        v_l_r = prior_transform(np.array(u_l_r))
                        logl_l_r = loglikelihood(np.array(v_l_r))
                    else:
                        logl_l_r = -np.inf
                        reverse = True  # can't compute gradient
                    nc += 1
                    # left side
                    u_l_l[i] -= 1e-10
                    if unitcheck(u_l_l, nonperiodic):
                        v_l_l = prior_transform(np.array(u_l_l))
                        logl_l_l = loglikelihood(np.array(v_l_l))
                    else:
                        logl_l_l = -np.inf
                        reverse = True  # can't compute gradient
                    if reverse:
                        break  # give up because we have to turn around
                    nc += 1
                    # compute dlnl/du
                    h[i] = (logl_l_r - logl_l_l) / 2e-10
            else:
                # If the gradient is provided, evaluate it.
                h = grad(v_l)
                if compute_jac:
                    jac = []
                    # Evaluate and apply Jacobian dv/du if gradient
                    # is defined as d(lnL)/dv instead of d(lnL)/du.
                    for i in range(n):
                        u_l_l, u_l_r = np.array(u_l), np.array(u_l)
                        # right side
                        u_l_r[i] += 1e-10
                        if unitcheck(u_l_r, nonperiodic):
                            v_l_r = prior_transform(np.array(u_l_r))
                        else:
                            reverse = True  # can't compute Jacobian
                            v_l_r = np.array(v_l)  # assume no movement
                        # left side
                        u_l_l[i] -= 1e-10
                        if unitcheck(u_l_l, nonperiodic):
                            v_l_l = prior_transform(np.array(u_l_l))
                        else:
                            reverse = True  # can't compute Jacobian
                            v_l_r = np.array(v_l)  # assume no movement
                        if reverse:
                            break  # give up because we have to turn around
                        jac.append((v_l_r - v_l_l) / 2e-10)
                    jac = np.array(jac)
                    h = np.dot(jac, h)  # apply Jacobian
                nc += 1
            # Compute specular reflection off boundary.
            vel_ref = vel - 2 * h * np.dot(vel, h) / linalg.norm(h)**2
            dotprod = np.dot(vel_ref, vel)
            dotprod /= linalg.norm(vel_ref) * linalg.norm(vel)
            # Check angle of reflection.
            if dotprod < -0.99:
                # The reflection angle is sufficiently small that it might
                # as well be a reflection.
                reverse = True
                break
            else:
                # If the reflection angle is sufficiently large, we
                # proceed as normal to the new position.
                vel = vel_ref
                u_out = None
                reflect = True
                nreflect += 1

        # Initialize lengths of chords.
        if len(nodes_l) > 1:
            # remove initial fallback chord
            nodes_l.pop(0)
            nodes_m.pop(0)
            nodes_r.pop(0)
        nodes_l, nodes_m, nodes_r = (np.array(nodes_l), np.array(nodes_m),
                                     np.array(nodes_r))
        Nchords = len(nodes_l)
        axlen = np.zeros(Nchords, dtype='float')
        for i, (nl, nm, nr) in enumerate(zip(nodes_l, nodes_m, nodes_r)):
            axlen[i] = linalg.norm(nr - nl)

        # Slice sample from all chords simultaneously. This is equivalent to
        # slice sampling in *time* along our trajectory.
        while True:
            # Select chord.
            axprob = axlen / np.sum(axlen)
            idx = rstate.choice(Nchords, p=axprob)
            # Define chord.
            u_l, u_m, u_r = nodes_l[idx], nodes_m[idx], nodes_r[idx]
            u_hat = u_r - u_l
            rprop = rstate.rand()
            u_prop = u_l + rprop * u_hat  # scale from left
            if unitcheck(u_prop, nonperiodic):
                v_prop = prior_transform(np.array(u_prop))
                logl_prop = loglikelihood(np.array(v_prop))
            else:
                logl_prop = -np.inf
            nc += 1
            ncontract += 1
            # If we succeed, move to the new position.
            if logl_prop >= loglstar:
                u = u_prop
                break
            # If we fail, check if the new point is to the left/right of
            # the point interior to the bounds (`u_m`) and update
            # the bounds accordingly.
            else:
                s = np.dot(u_prop - u_m, u_hat)  # check sign (+/-)
                if s < 0:  # left
                    nodes_l[idx] = u_prop
                    axlen[idx] *= 1 - rprop
                elif s > 0:  # right
                    nodes_r[idx] = u_prop
                    axlen[idx] *= rprop
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
                                       "logl_prop: {6}."
                                       .format(u, u_l, u_r, u_hat, u_prop,
                                               loglstar, logl_prop))

    blob = {'nmove': nmove, 'nreflect': nreflect, 'ncontract': ncontract}

    return u_prop, v_prop, logl_prop, nc, blob
