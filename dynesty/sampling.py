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


__all__ = ["sample_unif", "sample_rwalk", "sample_slice"]

EPS = float(np.finfo(np.float64).eps)
SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


def sample_unif(args):
    """Return a new live point sampled uniformly from a bounding proposal
    distribution."""

    # Unzipping.
    (u, loglstar, axes, scale,
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
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

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

        # Check if we're stuck.
        if nc > 10 * walks:
            raise RuntimeError("The random walk sampling appears to be stuck! "
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
    blob = {'accept': accept, 'reject': reject}

    return u, v, logl, nc, blob


def sample_slice(args):
    """Return a new live point proposed via a series of random slices
    away from an existing live point."""

    # Unzipping.
    (u, loglstar, axes, scale,
     prior_transform, loglikelihood, kwargs) = args
    rstate = np.random

    n = len(u)
    slices = kwargs.get('slices', 3)  # number of slices
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
                v_l = prior_transform(u_l)
                logl_l = loglikelihood(v_l)
            else:
                logl_l = -np.inf
            nc += 1
            u_r = u + (1 - r) * axis  # right bound
            if np.all(u_r > 0.) and np.all(u_r < 1.):
                v_r = prior_transform(u_r)
                logl_r = loglikelihood(v_r)
            else:
                logl_r = -np.inf
            nc += 1

            # "Stepping out" the left and right bounds.
            while logl_l >= loglstar:
                u_l -= axis
                if np.all(u_l > 0.) and np.all(u_l < 1.):
                    v_l = prior_transform(u_l)
                    logl_l = loglikelihood(v_l)
                else:
                    logl_l = -np.inf
                nc += 1
            while logl_r >= loglstar:
                u_r += axis
                if np.all(u_r > 0.) and np.all(u_r < 1.):
                    v_r = prior_transform(u_r)
                    logl_r = loglikelihood(v_r)
                else:
                    logl_r = -np.inf
                nc += 1

            # Sample within limits. If the sample is not valid, shrink
            # the limits until we hit the `loglstar` bound.
            while True:
                uhat = u_r - u_l
                u_prop = u_l + rstate.rand() * uhat  # scale from left
                if np.all(u_prop > 0.) and np.all(u_prop < 1.):
                    v_prop = prior_transform(u_prop)
                    logl_prop = loglikelihood(v_prop)
                else:
                    logl_prop = -np.inf
                nc += 1
                # If we succeed, move to the new position.
                if logl_prop >= loglstar:
                    window = np.linalg.norm(uhat)  # length of window
                    fscale.append(window / axlen)
                    u = u_prop
                    break
                # If we fail, check if the new point is to the left/right of
                # our original point along our proposal axis and update
                # the bounds accordingly.
                else:
                    s = np.dot(u_prop - u, uhat)  # check sign (+/-)
                    if s < 0:  # left
                        u_l = u_prop
                    elif s > 0:  # right
                        u_r = u_prop
                    else:
                        raise RuntimeError("Slice sampler has failed to find a "
                                           "valid point. Some useful "
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
