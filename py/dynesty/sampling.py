#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for proposing new live points used by
:class:`~dynesty.sampler.Sampler` (and its children from
:mod:`~dynesty.nestedsamplers`) and
:class:`~dynesty.dynamicsampler.DynamicSampler`.

"""

from collections import namedtuple
import warnings
import numpy as np
from numpy import linalg

from .utils import unitcheck, apply_reflect, get_random_generator
from .bounding import randsphere

__all__ = [
    "sample_unif", "sample_bound_unif", "sample_rwalk", "sample_slice",
    "sample_rslice"
]

SamplerArgument = namedtuple('SamplerArgument', [
    'u', 'loglstar', 'axes', 'scale', 'prior_transform', 'loglikelihood',
    'rseed', 'kwargs'
])


def sample_unif(args):
    """
    Evaluate a new point sampled uniformly from a bounding proposal
    distribution. Parameters are zipped within `args` to utilize
    `pool.map`-style functions.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
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
    u : `~numpy.ndarray` with shape (ndim,)
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

    # Evaluate.
    v = args.prior_transform(np.asarray(args.u))
    logl = args.loglikelihood(np.asarray(v))
    nc = 1
    blob = None

    return args.u, v, logl, nc, blob


def sample_bound_unif(args):
    """
    Return a new live point sampling uniformly within the
    boundary.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
        Initial sample (not used)

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. (not used)

    scale : float
        Value used to scale the provided axes. (not used)

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.
        This method requires keywords:
        bound (dynesty.bounding object)
        ndim (number of dimensions)
        n_cluster (number of dimensions that are clustered)
        nonbounded array

    Returns
    -------
    u : `~numpy.ndarray` with shape (ndim,)
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
    rstate = get_random_generator(args.rseed)
    bound = args.kwargs['bound']
    nonbounded = args.kwargs.get('nonbounded')
    n_cluster = args.kwargs.get('n_cluster')
    ndim = args.kwargs.get('ndim')
    nc = 0
    blob = None
    if nonbounded is not None:
        nonbounded = nonbounded[:n_cluster]
    ntries = 0
    threshold_warning = 10000
    threshold_warned = False
    while True:
        u = bound.samples(1, rstate=rstate).flatten()
        if not unitcheck(u, nonbounded):
            ntries += 1
            if ntries > threshold_warning and not threshold_warned:
                warnings.warn("Ellipsoid sampling is extremely inefficient",
                              category=RuntimeWarning)
                threshold_warned = True

            continue
        else:
            ntries = 0
        if n_cluster != ndim:
            u = np.concatenate((u, rstate.uniform(size=(ndim - n_cluster))))
        v = args.prior_transform(np.asarray(u))
        logl = args.loglikelihood(np.asarray(v))
        nc += 1
        if logl > args.loglstar:
            break
    return u, v, logl, nc, blob


def sample_rwalk(args):
    """
    Return a new live point proposed by random walking away from an
    existing live point.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
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
    u : `~numpy.ndarray` with shape (ndim,)
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
    rstate = get_random_generator(args.rseed)
    return generic_random_walk(args.u, args.loglstar, args.axes, args.scale,
                               args.prior_transform, args.loglikelihood,
                               rstate, args.kwargs)


def generic_random_walk(u, loglstar, axes, scale, prior_transform,
                        loglikelihood, rstate, kwargs):
    """
    Generic random walk step
    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
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
    u : `~numpy.ndarray` with shape (ndim,)
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

    # Periodicity.
    nonbounded = kwargs.get('nonbounded')
    periodic = kwargs.get('periodic')
    reflective = kwargs.get('reflective')

    # Setup.
    n = len(u)
    n_cluster = axes.shape[0]
    walks = kwargs.get('walks', 25)  # number of steps

    naccept = 0
    # Total number of accepted points with L>L*

    nreject = 0
    # Total number of points proposed to within the ellipsoid and cube
    # but rejected due to L<=L* condition

    ncall = 0
    # Total number of Likelihood calls (proposals evaluated)

    # Here we loop for exactly walks iterations.
    while ncall < walks:

        # This proposes a new point within the ellipsoid
        # This also potentially modifies the scale
        u_prop, fail = propose_ball_point(u,
                                          scale,
                                          axes,
                                          n,
                                          n_cluster,
                                          rstate=rstate,
                                          periodic=periodic,
                                          reflective=reflective,
                                          nonbounded=nonbounded)
        # If generation of points within an ellipsoid was
        # highly inefficient we adjust the scale
        if fail:
            nreject += 1
            ncall += 1
            continue

        # Check proposed point.
        v_prop = prior_transform(u_prop)
        logl_prop = loglikelihood(v_prop)
        ncall += 1

        if logl_prop > loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            naccept += 1
        else:
            nreject += 1
    if naccept == 0:
        # Technically we can find out the likelihood value
        # stored somewhere
        # But I'm currently recomputing it
        v = prior_transform(u)
        logl = loglikelihood(v)

    blob = {'accept': naccept, 'reject': nreject, 'scale': scale}

    return u, v, logl, ncall, blob


def propose_ball_point(u,
                       scale,
                       axes,
                       n,
                       n_cluster,
                       rstate=None,
                       periodic=None,
                       reflective=None,
                       nonbounded=None):
    """
    Here we are proposing points uniformly within an n-d ellipsoid.
    We are only trying once.
    We return the tuple with
    1) proposed point or None
    2) failure flag (if True, the generated point was outside bounds)
    """

    # starting point for clustered dimensions
    u_cluster = u[:n_cluster]

    # draw random point for non clustering parameters
    # we only need to generate them once
    u_non_cluster = rstate.random(n - n_cluster)
    u_prop = np.zeros(n)
    u_prop[n_cluster:] = u_non_cluster

    # Propose a direction on the unit n-sphere.
    dr = randsphere(n_cluster, rstate=rstate)
    # This generates uniform distribution within n-d ball

    # Transform to proposal distribution.
    du = np.dot(axes, dr)
    u_prop[:n_cluster] = u_cluster + scale * du

    # Wrap periodic parameters
    if periodic is not None:
        u_prop[periodic] = np.mod(u_prop[periodic], 1)

    # Reflect
    if reflective is not None:
        u_prop[reflective] = apply_reflect(u_prop[reflective])

    # Check unit cube constraints.
    if unitcheck(u_prop, nonbounded):
        return u_prop, False
    else:
        return None, True


def _slice_doubling_accept(x1, F, loglstar, L, R, fL, fR):
    """
    Acceptance test of slice sampling when doubling mode is used.
    This is an exact implementation of algorithm 6 of Neal 2003
    here w=1 and x0=0 as we are working in the
    coordinate system of F(A) = f(x0+A*w)

    Arguments are
    1) candidate location x1
    2) wrapped logl function (see generic_slice_step)
    3) threshold logl value
    4) left edge of the full interval
    5) right edge of the full interval
    6) value at left edge
    7) value at right edge
    """
    lhat, rhat = L, R
    f_lhat = fL
    f_rhat = fR
    D = False
    while rhat - lhat > 1.1:
        # Define slice and window.
        M = (lhat + rhat) / 2.
        # Propose new position.
        if (0 < M <= x1) or (x1 < M <= 0):
            D = True
        if x1 < M:
            rhat = M
            f_rhat = F(rhat)[1]
        else:
            lhat = M
            f_lhat = F(lhat)[1]
        if D and loglstar >= f_lhat and loglstar >= f_rhat:
            return False
    return True


def generic_slice_step(u, direction, nonperiodic, loglstar, loglikelihood,
                       prior_transform, doubling, rstate):
    """
    Do a slice generic slice sampling step along a specified dimension

    Arguments
    u: ndarray (ndim sized)
        Starting point in unit cube coordinates
        It MUST satisfy the logl>loglstar criterion
    direction: ndarray (ndim sized)
        Step direction vector
    nonperiodic: ndarray(bool)
        mask for nonperiodic variables
    loglstar: float
        the critical value of logl, so that new logl must be >loglstar
    loglikelihood: function
    prior_transform: function
    rstate: random state
    """
    nc, nexpand, ncontract = 0, 0, 0
    nexpand_threshold = 1000  # Threshold for warning the user
    n = len(u)
    rand0 = rstate.random()  # initial scale/offset
    dirlen = linalg.norm(direction)
    maxlen = np.sqrt(n) / 2.
    # maximum initial interval length (the diagonal of the cube)
    if dirlen > maxlen:
        # I stopped giving warnings, as it was too noisy
        dirnorm = dirlen / maxlen
    else:
        dirnorm = 1
    direction = direction / dirnorm

    #  The function that evaluates the logl at the location of
    # u0 + x*direction0
    def F(x):
        nonlocal nc
        u_new = u + x * direction
        if unitcheck(u_new, nonperiodic):
            logl = loglikelihood(prior_transform(u_new))
        else:
            logl = -np.inf
        nc += 1
        return u_new, logl

    # asymmetric step size on the left/right (see Neal 2003)
    nstep_l = -rand0
    nstep_r = (1 - rand0)

    logl_l = F(nstep_l)[1]
    logl_r = F(nstep_r)[1]
    expansion_warning = False
    if not doubling:
        # "Stepping out" the left and right bounds.
        while logl_l > loglstar:
            nstep_l -= 1
            logl_l = F(nstep_l)[1]
            nexpand += 1
        while logl_r > loglstar:
            nstep_r += 1
            logl_r = F(nstep_r)[1]
            nexpand += 1
        if nexpand > nexpand_threshold:
            expansion_warning = True
            warnings.warn('The slice sample interval was expanded more '
                          f'than {nexpand_threshold} times')

    else:
        # "Stepping out" the left and right bounds.
        K = 1
        while (logl_l > loglstar or logl_r > loglstar):
            V = rstate.random()
            if V < 0.5:
                nstep_l -= (nstep_r - nstep_l)
                logl_l = F(nstep_l)[1]
            else:
                nstep_r += (nstep_r - nstep_l)
                logl_r = F(nstep_r)[1]
            nexpand += K
            K *= 2
        L = nstep_l
        R = nstep_r
        fL = logl_l
        fR = logl_r

    # Sample within limits. If the sample is not valid, shrink
    # the limits until we hit the `loglstar` bound.

    while True:
        # Define slice and window.
        nstep_hat = nstep_r - nstep_l

        # Propose new position.
        nstep_prop = nstep_l + rstate.random() * nstep_hat  # scale from left
        u_prop, logl_prop = F(nstep_prop)
        ncontract += 1

        # If we succeed, move to the new position.
        # note that if we are using doubling mode we accept only
        # if _slice_doubling_accept() returns True
        if logl_prop > loglstar and (not doubling or _slice_doubling_accept(
                nstep_prop, F, loglstar, L, R, fL, fR)):
            break
        # If we fail, check if the new point is to the left/right of
        # our original point along our proposal axis and update
        # the bounds accordingly.
        else:
            if nstep_prop < 0:
                nstep_l = nstep_prop
            elif nstep_prop > 0:  # right
                nstep_r = nstep_prop
            else:
                # If `nstep_prop = 0` something has gone horribly wrong.
                raise RuntimeError("Slice sampler has failed to find "
                                   "a valid point. Some useful "
                                   "output quantities:\n"
                                   f"u: {u}\n"
                                   f"nstep_left: {nstep_l}\n"
                                   f"nstep_right: {nstep_r}\n"
                                   f"nstep_hat: {nstep_hat}\n"
                                   f"u_prop: {u_prop}\n"
                                   f"loglstar: {loglstar}\n"
                                   f"logl_prop: {logl_prop}\n"
                                   f"direction: {direction}\n")
    v_prop = prior_transform(u_prop)
    return u_prop, v_prop, logl_prop, nc, nexpand, ncontract, expansion_warning


def sample_slice(args):
    """
    Return a new live point proposed by a series of random slices
    away from an existing live point. Standard "Gibs-like" implementation where
    a single multivariate "slice" is a combination of `ndim` univariate slices
    through each axis.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
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
    u : `~numpy.ndarray` with shape (ndim,)
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
    (u, loglstar, axes, scale, prior_transform, loglikelihood,
     kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                args.prior_transform, args.loglikelihood, args.kwargs)
    rstate = get_random_generator(args.rseed)
    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)
    doubling = kwargs.get('slice_doubling', False)
    # Setup.
    n = len(u)
    assert axes.shape[0] == n
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    nexpand = 0
    ncontract = 0

    # Modifying axes and computing lengths.
    axes = scale * axes.T  # scale based on past tuning
    # Note we are transposing as axes[:,i] corresponds to i-th principal axis
    # of the ellipsoid
    expansion_warning_set = False
    # Slice sampling loop.
    for _ in range(slices):

        # Shuffle axis update order.
        idxs = np.arange(n)
        rstate.shuffle(idxs)

        # Slice sample along a random direction.
        for idx in idxs:

            # Select axis.
            axis = axes[idx]
            (u_prop, v_prop, logl_prop, nc1, nexpand1, ncontract1,
             expansion_warning) = generic_slice_step(u, axis, nonperiodic,
                                                     loglstar, loglikelihood,
                                                     prior_transform, doubling,
                                                     rstate)
            u = u_prop
            nc += nc1
            nexpand += nexpand1
            ncontract += ncontract1
            if expansion_warning and not doubling:
                # if we expanded the interval by more than
                # the threshold we set the warning and enable doubling
                expansion_warning_set = True
                doubling = True
                warnings.warn('Enabling doubling strategy of slice '
                              'sampling from Neal(2003)')
    blob = {
        'nexpand': nexpand,
        'ncontract': ncontract,
        'expansion_warning_set': expansion_warning_set
    }

    return u_prop, v_prop, logl_prop, nc, blob


def sample_rslice(args):
    """
    Return a new live point proposed by a series of random slices
    away from an existing live point. Standard "random" implementation where
    each slice is along a random direction based on the provided axes.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (ndim,)
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
    u : `~numpy.ndarray` with shape (ndim,)
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
    (u, loglstar, axes, scale, prior_transform, loglikelihood,
     kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                args.prior_transform, args.loglikelihood, args.kwargs)
    rstate = get_random_generator(args.rseed)
    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)
    doubling = kwargs.get('slice_doubling', False)

    # Setup.
    n = len(u)
    assert axes.shape[0] == n
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    nexpand = 0
    ncontract = 0
    expansion_warning_set = False

    # Slice sampling loop.
    for _ in range(slices):

        # Propose a direction on the unit n-sphere.
        drhat = rstate.standard_normal(size=n)
        drhat /= linalg.norm(drhat)

        # Transform and scale based on past tuning.
        direction = np.dot(axes, drhat) * scale

        (u_prop, v_prop, logl_prop, nc1, nexpand1, ncontract1,
         expansion_warning) = generic_slice_step(u, direction, nonperiodic,
                                                 loglstar, loglikelihood,
                                                 prior_transform, doubling,
                                                 rstate)
        u = u_prop
        nc += nc1
        nexpand += nexpand1
        ncontract += ncontract1
        if expansion_warning and not doubling:
            doubling = True
            expansion_warning_set = True
            warnings.warn('Enabling doubling strategy of slice '
                          'sampling from Neal(2003)')

    blob = {
        'nexpand': nexpand,
        'ncontract': ncontract,
        'expansion_warning_set': expansion_warning_set
    }

    return u_prop, v_prop, logl_prop, nc, blob
