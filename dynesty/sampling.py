#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampling functions for proposing new live points.

"""

from __future__ import (print_function, division)
from builtins import range

import sys
import warnings
import math
import numpy as np
from numpy import linalg
from scipy import misc


__all__ = ["sample_unif", "sample_rwalk", "sample_slice", "sample_rtraj"]

EPS = float(np.finfo(np.float64).eps)
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def sample_unif(args):
    """Return a new live point sampled uniformly from a bounding proposal
    distribution."""

    # Unzipping.
    (u, loglstar, axes, scale, rstate,
     prior_transform, loglikelihood, kwargs) = args

    v = prior_transform(u)
    logl = loglikelihood(v)
    nc = 1
    blob = None

    return u, v, logl, nc, blob


def sample_rwalk(args):
    """Return a new live point proposed via a random walk away from an
    existing live point."""

    # Unzipping.
    (u, loglstar, axes, scale, rstate,
     prior_transform, loglikelihood, kwargs) = args

    n = len(u)
    walks = kwargs.get('walks', 25)  # number of steps
    accept = 0
    reject = 0
    nc = 0
    while nc < walks or accept == 0:
        while True:
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
                reject += 1
        v_prop = prior_transform(u_prop)
        logl_prop = loglikelihood(v_prop)
        if logl_prop >= loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
        else:
            reject += 1
        nc += 1
    blob = {'accept': accept, 'reject': reject}

    return u, v, logl, nc, blob


def sample_slice(args):
    """Return a new live point proposed via a series of random slices
    away from an existing live point."""

    # Unzipping.
    (u, loglstar, axes, scale, rstate,
     prior_transform, loglikelihood, kwargs) = args

    n = len(u)
    slices = kwargs.get('slices', 3)  # number of slices
    nc = 0
    fscale = []

    # Slice sampling loop.
    for it in range(slices):

        # Shuffle the order that we update our axes.
        idxs = np.arange(n)
        rstate.shuffle(idxs)

        # Slice sample along each axis.
        for axis in axes[idxs]:

            # Define starting "window".
            r = rstate.rand()  # initial scale/offset
            u_l = u - scale * r * axis  # left bound
            if np.all(u_l > 0.) and np.all(u_l < 1.):
                v_l = prior_transform(u_l)
                logl_l = loglikelihood(v_l)
            else:
                logl_l = -np.inf
            nc += 1
            u_r = u + scale * (1 - r) * axis  # right bound
            if np.all(u_r > 0.) and np.all(u_r < 1.):
                v_r = prior_transform(u_r)
                logl_r = loglikelihood(v_r)
            else:
                logl_r = -np.inf
            nc += 1

            # "Stepping out" the left and right bounds.
            while logl_l >= loglstar:
                u_l -= scale * axis
                if np.all(u_l > 0.) and np.all(u_l < 1.):
                    v_l = prior_transform(u_l)
                    logl_l = loglikelihood(v_l)
                else:
                    logl_l = -np.inf
                nc += 1
            while logl_r >= loglstar:
                u_r += scale * axis
                if np.all(u_r > 0.) and np.all(u_r < 1.):
                    v_r = prior_transform(u_r)
                    logl_r = loglikelihood(v_r)
                else:
                    logl_r = -np.inf
                nc += 1

            # Sample within limits. If the sample is not valid, shrink
            # the limits until we hit the `loglstar` bound.
            while True:
                window = np.linalg.norm(u_r - u_l)  # size of window
                du = rstate.rand() * window  # scale from left
                u_prop = u_l + du * axis
                if np.all(u_prop > 0.) and np.all(u_prop < 1.):
                    v_prop = prior_transform(u_prop)
                    logl_prop = loglikelihood(v_prop)
                else:
                    logl_prop = -np.inf
                nc += 1
                # If we succeed, move to the new position.
                if logl_prop >= loglstar:
                    fscale.append(window / scale)
                    u = u_prop
                    break
                # If we fail, check if the new point is to the left/right of
                # our original point along our proposal axis and update
                # the bounds accordingly.
                else:
                    s = np.dot(u_prop - u, axis)  # check sign (+/-)
                    if s < 0:  # left
                        u_l = u_prop
                    elif s > 0:  # right
                        u_r = u_prop
                    else:
                        raise RuntimeError("Slice sampler somehow shrank "
                                           "to the original point! Please "
                                           "report this as a likely bug.")

    blob = {'fscale': np.median(fscale)}

    return u_prop, v_prop, logl_prop, nc, blob


def sample_rtraj(args):
    """Return a new live point proposed via a random trajectory
    away from an existing live point, where we "bounce" off the
    iso-likelihood contour based on the gradient."""

    # Unzipping.
    (u, loglstar, axes, scale, rstate,
     prior_transform, loglikelihood, kwargs) = args

    n = len(u)
    v = prior_transform(u)
    logl = loglikelihood(v)
    lgrad = kwargs.get('lgrad', None)  # gradient of likelihood
    steps = kwargs.get('steps', 25)  # number of steps

    # Define a random trajectory.
    vel = np.dot(axes, rstate.randn(n))  # velocity
    vel *= scale  # scale based on past tuning

    # Evolve the trajectory.
    cont = 0
    reflect = 0
    reverse = 0
    nc = 0
    ngrad = 0
    while cont + reflect + reverse <= steps:
        # Update the position.
        u_prop = u + vel
        unitcheck = np.all(u_prop + 1e-5 > 0.) and np.all(u_prop + 1e-5 < 1.)
        if unitcheck:
            v_prop = prior_transform(u_prop)
            logl_prop = loglikelihood(v_prop)
            nc += 1
        else:
            logl_prop = -np.inf
        # If the proposed position is within the likelihood bound, accept.
        if logl_prop >= loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            cont += 1
        # If it's not (but still within the unit cube), attempt to
        # reflect off the boundary.
        else:
            if unitcheck:
                if lgrad is None:
                    # Compute numerical approximation to the gradient.
                    dvs = prior_transform(u_prop + 1e-5) - v_prop
                    h = np.zeros(n)
                    for i in range(n):
                        vprime = v_prop
                        vprime[i] += dvs[i]
                        loglprime = loglikelihood(vprime)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            logdl = misc.logsumexp(a=[loglprime, logl_prop],
                                                   b=[1., -1.])
                            h[i] = math.exp(logdl - math.log(dvs[i]))
                    ngrad += 1
                else:
                    # Compute provided gradient.
                    h = self.lgrad(v_prop)
                    ngrad += 1
                # If our gradient is unstable, reverse course.
                nnorm = linalg.norm(h)
                if nnorm <= EPS or not np.isfinite(nnorm):
                    vel = -vel
                    reverse += 1
                # Otherwise, reflect off of the boundary.
                else:
                    nhat = h / linalg.norm(h)  # normal vector
                    vel_prop = vel - 2 * nhat * np.dot(vel, nhat)
                    u_prop = u_prop + vel_prop
                    unitcheck = (np.all(u_prop > 0.) and
                                 np.all(u_prop < 1.))
                    if unitcheck:
                        v_prop = prior_transform(u_prop)
                        logl_prop = loglikelihood(v_prop)
                        nc += 1
                    else:
                        logl_prop = -np.inf
                    # Accepted reflected point if within our constraint.
                    if logl_prop >= loglstar:
                        u = u_prop
                        v = v_prop
                        logl = logl_prop
                        vel = vel_prop
                        reflect += 1
                    # If our reflected point is out of bounds, reverse course.
                    else:
                        vel = -vel
                        reverse += 1
            # If we proposed outside the unit cube, reverse course.
            else:
                vel = -vel
                reverse += 1

    blob = {'cont': cont, 'reflect': reflect, 'reverse': reverse}

    if lgrad is None:
        nc += ngrad * n
    else:
        nc += ngrad

    return u, v, logl, nc, blob
