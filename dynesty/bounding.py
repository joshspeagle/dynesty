#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounding classes used when proposing new live points, along with relevant
helper functions. Includes:

    UnitCube:
        The unit N-cube (unconstrained draws from the prior).

    Ellipsoid:
        Bounding ellipsoid.

    MultiEllipsoid:
        A set of (possibly overlapping) bounding ellipsoids.

    RadFriends:
        A set of (possibly overlapping) balls centered on each live point.

    SupFriends:
        A set of (possibly overlapping) cubes centered on each live point.

The ellipsoid methods are based on results from Feroz et al. (2009)
<https://arxiv.org/abs/0809.3437>. The RadFriends and SupFriends methods are
based on results from Buchner (2014) <https://arxiv.org/abs/1407.5459>.

"""

from __future__ import (print_function, division)
from builtins import range

import sys
import warnings
import math
import numpy as np
from numpy import linalg
from scipy import special
from scipy import spatial

try:
    from scipy.cluster.vq import kmeans2
    HAVE_KMEANS = True
except ImportError:  # pragma: no cover
    HAVE_KMEANS = False

from .utils import random_choice

__all__ = ["UnitCube", "Ellipsoid", "MultiEllipsoid", "RadFriends",
           "SupFriends"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


class UnitCube(object):
    """
    A N-dimensional unit cube.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the unit cube.

    """

    def __init__(self, ndim):
        self.n = ndim  # dimension
        self.vol = 1.  # volume
        self.funit = 1.  # overlap with the unit cube

    def contains(self, x):
        """Checks if ellipsoid contains `x`."""

        return np.all(point > 0.) and np.all(point < 1.)

    def randoffset(self, rstate=np.random):
        """Draw a sample randomly offset from the center of the
        unit cube."""

        return self.sample(rstate=rstate) - 0.5

    def sample(self, rstate=np.random):
        """
        Draw a sample uniformly distributed within the unit cube.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the unit cube.

        """

        return rstate.rand(self.n)

    def samples(self, nsamples, rstate=np.random):
        """
        Draw `nsamples` samples randomly distributed within the unit cube.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the unit cube.

        """

        xs = np.array([self.sample(rstate=rstate) for i in range(nsamples)])

        return xs

    def update(self, points, pointvol=0., rstate=np.random, bootstrap=0,
               pool=None):
        """
        Filler function since the unit cube does not change.

        """

        pass


class Ellipsoid(object):
    """
    An N-ellipsoid.

    Defined by::

        (x - v)^T A (x - v) = 1

    where the vector `v` is the center of the ellipsoid and `A` is an `N x N`
    matrix. Assumes that `A` is symmetric positive definite.

    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape (N,)
        Coordinates of ellipsoid center. Note that the array is *not* copied.
        This array is never modified internally.

    am : `~numpy.ndarray` with shape (N, N)
        Matrix describing the axes. Watch out! This array is *not* copied
        and may be modified internally!

    """

    def __init__(self, ctr, am):
        self.n = len(ctr)  # dimension
        self.ctr = ctr  # center coordinates
        self.am = am  # precision matrix (inverse of covariance)

        # Volume of ellipsoid is the volume of an n-sphere divided
        # by the (determinant of the) Jacobian associated with the
        # transformation, which by definition is the precision matrix.
        self.vol = vol_prefactor(self.n) / np.sqrt(linalg.det(self.am))

        # The eigenvalues (l) of `a` are (a^-2, b^-2, ...) where
        # (a, b, ...) are the lengths of principle axes.
        # The eigenvectors (v) are the normalized principle axes.
        l, v = linalg.eigh(self.am)
        if np.all((l > 0.) & (np.isfinite(l))):
            self.axlens = 1. / np.sqrt(l)
        else:
            raise ValueError("The input precision matrix defining the "
                             "ellipsoid {0} is apparently singular with "
                             "l={1} and v={2}.".format(self.am, l, v))

        # Scaled eigenvectors are the axes, where `axes[:,i]` is the
        # i-th axis.  Multiplying this matrix by a vector will transform a
        # point in the unit n-sphere to a point in the ellipsoid.
        self.axes = np.dot(v, np.diag(self.axlens))

        # Amount by which volume was expanded after initialization (i.e.
        # cumulative factor from `scale_to_vol`).
        self.expand = 1.

    def scale_to_vol(self, vol):
        """Scale ellipoid to encompass a target volume."""

        f = (vol / self.vol) ** (1.0 / self.n)  # linear factor
        self.expand *= f
        self.am *= f**-2
        self.axlens *= f
        self.axes *= f
        self.vol = vol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis."""

        i = np.argmax(self.axlens)  # find the major axis
        v = self.axes[:, i]  # vector from center to major axis endpoint

        return self.ctr - v, self.ctr + v

    def distance(self, x):
        """Compute the normalized distance to `x` from the center of the
        ellipsoid."""

        d = x - self.ctr

        return np.sqrt(np.dot(np.dot(d, self.am), d))

    def contains(self, x):
        """Checks if ellipsoid contains `x`."""

        return self.distance(x) <= 1.0

    def randoffset(self, rstate=np.random):
        """Return an offset from ellipsoid center that is randomly
        distributed within the ellipsoid."""

        return np.dot(self.axes, randsphere(self.n, rstate=rstate))

    def sample(self, rstate=np.random):
        """
        Draw a sample uniformly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the ellipsoid.

        """

        return self.ctr + self.randoffset(rstate=rstate)

    def samples(self, nsamples, rstate=np.random):
        """
        Draw `nsamples` samples randomly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the ellipsoid.

        """

        xs = np.array([self.sample(rstate=rstate) for i in range(nsamples)])

        return xs

    def unitcube_overlap(self, ndraws=10000, rstate=np.random):
        """Using `ndraws` Monte Carlo draws, evaluate the amount of
        overlap with the unit cube."""

        samples = [self.sample(rstate=rstate) for i in range(ndraws)]
        nin = sum([np.all(x > 0.) and np.all(x < 1.) for x in samples])

        return 1. * nin / ndraws

    def update(self, points, pointvol=0., rstate=np.random, bootstrap=0,
               pool=None, mc_integrate=False):
        """
        Update the ellipsoid to bound the collection of points.

        """

        # Compute new bounding ellipsoid.
        ell = bounding_ellipsoid(points, pointvol=pointvol)
        self.n = ell.n
        self.ctr = ell.ctr
        self.am = ell.am
        self.vol = ell.vol
        self.axlens = ell.axlens
        self.axes = ell.axes
        self.expand = ell.expand

        # Use bootstrapping to determine volume expansion factor.
        if bootstrap > 0:

            # If possible, compute bootstraps in parallel using a pool.
            if pool is None:
                M = map
            else:
                M = pool.map
            ps = [points for it in range(bootstrap)]
            pvs = [pointvol for it in range(bootstrap)]
            args = zip(ps, pvs)
            expands = M(_ellipsoid_bootstrap_expand, args)

            # Conservatively set the expansion factor to be the maximum
            # factor derived from our set of bootstraps.
            expand = max(expands)

            # If our ellipsoid is over-constrained, expand it.
            if expand > 1.:
                v = self.vol * expand**self.n
                self.scale_to_vol(v)

        # Compute overlap with the unit cube.
        if mc_integrate:
            self.funit = self.unitcube_overlap()


class MultiEllipsoid(object):
    """
    A collection of M N-ellipsoids.

    Parameters
    ----------
    ells : list of `Ellipsoid` objects with length `M`, optional
        A set of `Ellipsoid` objects that make up the collection of
        N-ellipsoids. Will be used to initialize ellipsoids if provided.

    ctrs : `~numpy.ndarray` with shape (M, N), optional
        Collection of coordinates of ellipsoid centers. Note that the array
        is *not* copied. This array is never modified internally. Will be
        used to initialize ellipsoids if `ams` is also provided.

    ams : `~numpy.ndarray` with shape (M, N, N), optional
        Collection of matrices describing the axes of the ellipsoids. Watch
        out! This array is *not* copied and may be modified internally! Will
        be used to initialize ellipsoids if `ctrs` also provided.

    """

    def __init__(self, ells=None, ctrs=None, ams=None):
        if ells is not None:
            if (ctrs is None) and (ams is None):
                self.nells = len(ells)
                self.ells = ells
                self.ctrs = np.array([ell.ctr for ell in self.ells])
                self.ams = np.array([ell.am for ell in self.ells])
            else:
                raise ValueError("You cannot specific both `ells` and "
                                 "(`ctrs`, `ams`)!")
        else:
            if (ctrs is None) and (ams is None):
                raise ValueError("You must specify either `ells` or "
                                 "(`ctrs`, `ams`).")
            else:
                self.nells = len(ctrs)
                self.ctrs = ctrs
                self.ams = ams
                self.ells = [Ellipsoid(ctrs[i], ams[i])
                             for i in range(self.nells)]
        self.vols = np.array([ell.vol for ell in self.ells])
        self.expands = np.ones(self.nells)
        self.vol_tot = sum(self.vols)
        self.expand_tot = 1.

    def scale_to_vols(self, vols):
        """Scale ellipoids to encompass a corresponding set of
        target volume."""

        _ = [self.ells[i].scale_to_vol(vols[i]) for i in range(self.nells)]
        self.vols = np.array(vols)
        self.expands = np.array([self.ells[i].expand
                                 for i in range(self.nells)])
        vol_tot = sum(vols)
        self.expand_tot *= vol_tot / self.vol_tot
        self.vol_tot = vol_tot

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis of each ellipsoid."""

        i = np.argmax(self.axlens)  # find the major axis
        v = self.axes[:, i]  # vector from center to major axis endpoint

        return np.array([ell.major_axis_endpoints() for ell in self.ells])

    def within(self, x, j=None):
        """Checks which ellipsoids `x` falls within, skipping the `j`-th
        ellipsoid."""

        within = np.array([self.ells[i].contains(x) if i != j else True
                           for i in range(self.nells)], dtype='bool')
        idxs = np.arange(self.nells)[within]

        return idxs

    def overlap(self, x, j=None):
        """Checks how many ellipsoids `x` falls within, skipping the `j`-th
        ellipsoid."""

        q = len(self.within(x, j=j))

        return q

    def contains(self, x):
        """Checks if the set of ellipsoids contains `x`."""

        return self.overlap(x) > 0

    def sample(self, rstate=np.random, return_q=False):
        """
        Sample a point uniformly distributed within the union of ellipsoids.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of ellipsoids.
        idx : int
            The index of the ellipsoid `x` was sampled from.
        q : int, optional
            The number of ellipsoids `x` falls within.

        """

        # If there is only one ellipsoid, sample from it.
        if self.nells == 1:
            x = self.ells[0].sample(rstate=rstate)
            idx = 0
            q = 1
            if return_q:
                return x, idx, q
            else:
                return x, idx

        # Select an ellipsoid at random proportional to its volume.
        idx = random_choice(self.nells, self.vols / self.vol_tot,
                            rstate=rstate)

        # Select a point from the chosen ellipsoid.
        x = self.ells[idx].sample(rstate=rstate)

        # Check how many ellipsoids the point lies within, passing over
        # the `idx`-th ellipsoid `x` was sampled from.
        q = self.overlap(x, j=idx) + 1

        if return_q:
            # If `q` is being returned, assume the user wants to
            # explicitly apply the `1. / q` acceptance criterion to
            # properly sample from the union of ellipsoids.
            return x, idx, q
        else:
            # If `q` is not being returned, assume the user wants this
            # done internally.
            while rstate.rand() > (1. / q):
                idx = random_choice(self.nells, self.vols / self.vol_tot,
                                    rstate=rstate)
                x = self.ells[idx].sample(rstate=rstate)
                q = self.overlap(x, j=idx) + 1

            return x, idx

    def samples(self, nsamples, rstate=np.random):
        """
        Draw `nsamples` samples uniformly distributed within the union of
        ellipsoids.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of ellipsoids.

        """

        xs = np.array([self.sample(rstate=rstate)[0]
                       for i in range(nsamples)])

        return xs

    def monte_carlo_vol(self, ndraws=10000, rstate=np.random,
                        return_overlap=False):
        """Using `ndraws` Monte Carlo draws, evaluate the amount of
        overlap with the unit cube."""

        samples = [self.sample(rstate=rstate, return_q=True)
                   for i in range(ndraws)]
        qsum = sum([q for (x, idx, q) in samples])
        vol = 1. * ndraws / qsum * self.vol_tot

        if return_overlap:
            qin = sum([q * (np.all(x > 0.) and np.all(x < 1.))
                       for (x, idx, q) in samples])
            overlap = 1. * qin / qsum
            return vol, overlap
        else:
            return vol

    def update(self, points, pointvol=0., vol_dec=0.5, vol_check=2.,
               rstate=np.random, bootstrap=0, pool=None, mc_integrate=False):
        """
        Update the set of ellipsoids to bound the collection of points.

        """

        if not HAVE_KMEANS:
            raise ValueError("scipy.cluster.vq.kmeans2 is required to compute "
                             "ellipsoid decompositions.")  # pragma: no cover

        npoints, ndim = points.shape

        # Calculate the bounding ellipsoid for the points possibly
        # enlarged to a minimum volume.
        ell = bounding_ellipsoid(points, pointvol=pointvol)

        # Recursively split the bounding ellipsoid until the volume of each
        # split no longer decreases by a factor of `vol_dec`.
        ells = _bounding_ellipsoids(points, ell, pointvol=pointvol,
                                    vol_dec=vol_dec, vol_check=vol_check)

        # Update the set of ellipsoids.
        self.nells = len(ells)
        self.ells = ells
        self.ctrs = np.array([ell.ctr for ell in self.ells])
        self.ams = np.array([ell.am for ell in self.ells])
        self.vols = np.array([ell.vol for ell in self.ells])
        self.vol_tot = sum(self.vols)

        # Use bootstrapping to determine volume expansion factor.
        if bootstrap > 0:

            # If possible, compute bootstraps in parallel using a pool.
            if pool is None:
                M = map
            else:
                M = pool.map
            ps = [points for it in range(bootstrap)]
            pvs = [pointvol for it in range(bootstrap)]
            vds = [vol_dec for it in range(bootstrap)]
            vcs = [vol_check for it in range(bootstrap)]
            args = zip(ps, pvs, vds, vcs)
            expands = M(_ellipsoids_bootstrap_expand, args)

            # Conservatively set the expansion factor to be the maximum
            # factor derived from our set of bootstraps.
            expand = max(expands)

            # If our ellipsoids are overly constrained, expand them.
            if expand > 1.:
                vs = self.vols * expand**ndim
                self.scale_to_vols(vs)

        # Monte Carlo integrate the volume and overlap with the unit cube.
        if mc_integrate:
            self.vol, self.funit = self.monte_carlo_vol(return_overlap=True)


class RadFriends(object):
    """
    A collection of N-balls of fixed size centered on each live point.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the ball.

    radius : float
        Radius of the ball.

    """

    def __init__(self, ndim, radius):
        self.n = ndim
        self.radius = radius
        self.vol_ball = vol_prefactor(self.n) * self.radius**self.n
        self.expand = 1.

    def scale_to_vol(self, vol):
        """Scale ball to encompass a target volume."""

        f = (vol / self.vol_ball) ** (1.0 / self.n)  # linear factor
        self.expand *= f
        self.radius *= f
        self.vol_ball = vol

    def within(self, x, ctrs, kdtree=None):
        """Checks which balls `x` falls within. Uses a KDTree to
        accelerate the search if provided."""

        if kdtree is None:
            # If no KDTree is provided, execute a brute-force search over all
            # balls.
            nctrs = len(ctrs)
            within = np.array([linalg.norm(ctrs[i] - x) <= self.radius
                               for i in range(nctrs)], dtype='bool')
            idxs = np.arange(nctrs)[within]
        else:
            # If a KDTree is provided, find all points within r.
            idxs = kdtree.query_ball_point(x, self.radius, p=2.0, eps=0)

        return idxs

    def overlap(self, x, ctrs, kdtree=None):
        """Checks how many balls `x` falls within. Uses a KDTree to
        accelerate the search if provided."""

        q = len(self.within(x, ctrs, kdtree=kdtree))

        return q

    def contains(self, x, ctrs, kdtree=None):
        """Checks if the set of balls contains `x`. Uses a KDTree to
        accelerate the search if provided."""

        return self.overlap(x, ctrs, kdtree=kdtree) > 0

    def sample(self, ctrs, rstate=np.random, return_q=False, kdtree=None):
        """
        Sample a point uniformly distributed within the union of balls.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of balls.
        q : int, optional
            The number of balls `x` falls within.

        """

        nctrs = len(ctrs)  # number of balls

        # If there is only one ball, sample from it.
        if nctrs == 1:
            dx = self.radius * randsphere(self.n, rstate=rstate)
            x = ctrs[0] + dx
            if return_q:
                return x, 1
            else:
                return x

        # Select a ball at random.
        idx = rstate.randint(nctrs)

        # Select a point from the chosen ball.
        dx = self.radius * randsphere(self.n, rstate=rstate)
        x = ctrs[idx] + dx

        # Check how many balls the point lies within, passing over
        # the `idx`-th ball `x` was sampled from.
        q = self.overlap(x, ctrs, kdtree=kdtree)

        if return_q:
            # If `q` is being returned, assume the user wants to
            # explicitly apply the `1. / q` acceptance criterion to
            # properly sample from the union of balls.
            return x, q
        else:
            # If `q` is not being returned, assume the user wants this
            # done internally.
            while rstate.rand() > (1. / q):
                idx = rstate.randint(nctrs)
                dx = self.radius * randsphere(self.n, rstate=rstate)
                x = ctrs[idx] + dx
                q = self.overlap(x, ctrs, kdtree=kdtree)
            return x

    def samples(self, nsamples, ctrs, rstate=np.random, kdtree=None):
        """
        Draw `nsamples` samples uniformly distributed within the union of
        balls.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of balls.

        """

        xs = np.array([self.sample(ctrs, rstate=rstate, kdtree=kdtree)
                       for i in range(nsamples)])

        return xs

    def monte_carlo_vol(self, ctrs, ndraws=10000, rstate=np.random,
                        return_overlap=False, kdtree=None):
        """Using `ndraws` Monte Carlo draws, evaluate the amount of
        overlap with the unit cube."""

        samples = [self.sample(ctrs, rstate=rstate, return_q=True,
                               kdtree=kdtree)
                   for i in range(ndraws)]
        qsum = sum([q for (x, q) in samples])
        vol = 1. * ndraws / qsum * len(ctrs) * self.vol_ball

        if return_overlap:
            nin = sum([q * (np.all(x > 0.) and np.all(x < 1.))
                       for (x, q) in samples])
            overlap = 1. * qin / qsum
            return vol, overlap
        else:
            return vol

    def update(self, points, pointvol=0., rstate=np.random, bootstrap=0,
               pool=None, kdtree=None, mc_integrate=False):
        """
        Update the radius/volume of our balls.

        """

        # If possible, compute bootstraps in parallel using a pool.
        if pool is None:
            M = map
        else:
            M = pool.map

        if bootstrap == 0.:
            # Construct radius using leave-one-out if no bootstraps used.
            radii = _friends_leaveoneout_radius(points, 'balls')
        else:
            # Bootstrap radius using the set of live points.
            ps = [points for it in range(bootstrap)]
            ftypes = ['balls' for it in range(bootstrap)]
            args = zip(ps, ftypes)
            radii = M(_friends_bootstrap_radius, args)

        # Conservatively set radius to be maximum of the set.
        rmax = max(radii)
        self.radius = rmax
        self.vol_ball = vol_prefactor(self.n) * self.radius**self.n

        # Expand our ball to encompass a minimum volume.
        if pointvol > 0.:
            v = pointvol
            if self.vol_ball < v:
                self.scale_to_vol(v)

        # Monte Carlo integrate the volume and overlap with the unit cube.
        if mc_integrate:
            self.vol, self.funit = self.monte_carlo_vol(points, kdtree=kdtree,
                                                        return_overlap=True)


class SupFriends(object):
    """
    A collection of N-cubes of fixed size centered on each live point.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the cube.

    hside : float
        Half the side-length of the cube.

    """

    def __init__(self, ndim, hside):
        self.n = ndim
        self.hside = hside
        self.vol_cube = (2. * self.hside)**self.n
        self.expand = 1.

    def scale_to_vol(self, vol):
        """Scale ball to encompass a target volume."""

        f = (vol / self.vol_cube) ** (1.0 / self.n)  # linear factor
        self.expand *= f
        self.hside *= f
        self.vol_cube = vol

    def within(self, x, ctrs, kdtree=None):
        """Checks which cubes `x` falls within. Uses a KDTree to
        accelerate the search if provided."""

        if kdtree is None:
            # If no KDTree is provided, execute a brute-force search
            # over all cubes.
            nctrs = len(ctrs)
            within = np.array([max(abs(ctrs[i] - x)) <= self.hside
                               for i in range(nctrs)], dtype='bool')
            idxs = np.arange(nctrs)[within]
        else:
            # If a KDTree is provided, find all points within r (`hside`).
            idxs = kdtree.query_ball_point(x, self.hside, p=np.inf, eps=0)

        return idxs

    def overlap(self, x, ctrs, kdtree=None):
        """Checks how many cubes `x` falls within, skipping the `j`-th
        cube."""

        q = len(self.within(x, ctrs, kdtree=kdtree))

        return q

    def contains(self, x, ctrs, kdtree=None):
        """Checks if the set of cubes contains `x`."""

        return self.overlap(x, ctrs, kdtree=kdtree) > 0

    def sample(self, ctrs, rstate=np.random, return_q=False, kdtree=None):
        """
        Sample a point uniformly distributed within the union of cubes.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of cubes.
        q : int, optional
            The number of cubes `x` falls within.

        """

        nctrs = len(ctrs)  # number of cubes

        # If there is only one cube, sample from it.
        if nctrs == 1:
            dx = self.hside * (2. * rstate.rand(self.n) - 1.)
            x = ctrs[0] + dx
            if return_q:
                return x, 1
            else:
                return x

        # Select a cube at random.
        idx = rstate.randint(nctrs)

        # Select a point from the chosen cube.
        dx = self.hside * (2. * rstate.rand(self.n) - 1.)
        x = ctrs[idx] + dx

        # Check how many cubes the point lies within, passing over
        # the `idx`-th cube `x` was sampled from.
        q = self.overlap(x, ctrs, kdtree=kdtree)

        if return_q:
            # If `q` is being returned, assume the user wants to
            # explicitly apply the `1. / q` acceptance criterion to
            # properly sample from the union of balls.
            return x, q
        else:
            # If `q` is not being returned, assume the user wants this
            # done internally.
            while rstate.rand() > (1. / q):
                idx = rstate.randint(nctrs)
                dx = self.hside * (2. * rstate.rand(self.n) - 1.)
                x = ctrs[idx] + dx
                q = self.overlap(x, ctrs, kdtree=kdtree)
            return x

    def samples(self, nsamples, ctrs, rstate=np.random, kdtree=None):
        """
        Draw `nsamples` samples uniformly distributed within the union of
        cubes.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of cubes.

        """

        xs = np.array([self.sample(ctrs, rstate=rstate, kdtree=kdtree)
                       for i in range(nsamples)])

        return xs

    def monte_carlo_vol(self, ctrs, ndraws=10000, rstate=np.random,
                        return_overlap=False, kdtree=None):
        """Using `ndraws` Monte Carlo draws, evaluate the amount of
        overlap with the unit cube."""

        samples = [self.sample(ctrs, rstate=rstate, return_q=True,
                               kdtree=kdtree)
                   for i in range(ndraws)]
        qsum = sum([q for (x, q) in samples])
        vol = 1. * ndraws / qsum * len(ctrs) * self.vol_cube

        if return_overlap:
            qin = sum([q * (np.all(x > 0.) and np.all(x < 1.))
                       for (x, q) in samples])
            overlap = 1. * qin / qsum
            return vol, overlap
        else:
            return vol

    def update(self, points, pointvol=0., rstate=np.random, bootstrap=0,
               pool=None, kdtree=None, mc_integrate=False):
        """
        Update the half-side-lengths/volumes of our cubes.

        """

        # If possible, compute bootstraps in parallel using a pool.
        if pool is None:
            M = map
        else:
            M = pool.map

        if bootstrap == 0.:
            # Construct radius using leave-one-out if no bootstraps used.
            hsides = _friends_leaveoneout_radius(points, 'cubes')
        else:
            # Bootstrap radius using the set of live points.
            ps = [points for it in range(bootstrap)]
            ftypes = ['cubes' for it in range(bootstrap)]
            args = zip(ps, ftypes)
            hsides = M(_friends_bootstrap_radius, args)

        # Conservatively set radius to be maximum of the set.
        hsmax = max(hsides)
        self.hside = hsmax
        self.vol_cube = (2. * self.hside)**self.n

        # Expand our cube to encompass a minimum volume.
        if pointvol > 0.:
            v = pointvol
            if self.vol_cube < v:
                self.scale_to_vol(v)

        # Monte Carlo integrate the volume and overlap with the unit cube.
        if mc_integrate:
            self.vol, self.funit = self.monte_carlo_vol(points, kdtree=kdtree,
                                                        return_overlap=True)


##################
# HELPER FUNCTIONS
##################

def vol_prefactor(n, p=2.):
    """
    Volume constant for an n-dimensional sphere with an L^p norm::

    f = (2 * Gamma(1/p + 1))**n / Gamma(n/p + 1)

    By default the norm is p=2 (the standard Euclidean norm).

    """

    p *= 1.  # convert to float in case user inputs int
    f = (2 * special.gamma(1./p + 1.))**n / special.gamma(n/p + 1)

    return f


def randsphere(n, rstate=np.random):
    """Draw a random point within an n-dimensional unit sphere."""

    z = rstate.randn(n)  # initial n-dim vector

    return z / linalg.norm(z) * rstate.rand()**(1./n)


def make_eigvals_positive(am, targetprod):
    """For the symmetric square matrix `am`, increase any zero eigenvalues
    to fulfill the given target product of eigenvalues. Returns a
    (possibly) new matrix."""

    w, v = linalg.eigh(am)  # use eigh since a is symmetric
    mask = w < 1.e-10
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
        am = np.dot(np.dot(v, np.diag(w)), linalg.inv(v))  # re-form cov

    return am


def bounding_ellipsoid(points, pointvol=0.):
    """
    Calculate bounding ellipsoid containing a collection of points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    pointvol : float, optional
        The minimum volume occupied by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is *None*.

    Returns
    -------
    ellipsoid : Ellipsoid

    """

    npoints, ndim = points.shape

    # Check for valid `pointvol` value if provided.
    if pointvol < 0.:
        raise ValueError("You must specify a non-negative value "
                         "for `pointvol`.")

    # If there is only a single point, return an n-sphere with volume
    # `pointvol` centered at the point.
    if npoints == 1:
        if pointvol > 0.:
            ctr = points[0]
            r = (pointvol / vol_prefactor(ndim))**(1./ndim)
            am = (1. / r**2) * np.identity(ndim)
            return Ellipsoid(ctr, am)
        else:
            raise ValueError("Cannot compute a bounding ellipsoid to a "
                             "single point if `pointvol` is not specified.")

    # Calculate covariance of points.
    ctr = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)

    # When ndim = 1, np.cov returns a 0-d array. Make it a 1x1 2-d array.
    if ndim == 1:
        cov = np.atleast_2d(cov)

    # For a ball of uniformly distributed points, the sample covariance
    # will be smaller than the true covariance by a factor of 1/(n+2)
    # [see, e.g., goo.gl/UbsjYl]. Since we are assuming all points are
    # uniformly distributed within the unit cube, they are uniformly
    # distributed within any sub-volume within the cube. We expand
    # our sample covariance `cov` to compensate for this.
    cov *= (ndim + 2)

    # Ensure that `cov` is nonsingular to deal with pathological cases
    # where the ellipsoid has zero volume. This can occur when
    # `npoints <= ndim` or when enough points are linear combinations
    # of other points. When this happens, we expand the ellipsoid
    # in the zero dimensions to fulfill the volume expected from
    # `pointvol`.
    if pointvol > 0.:
        targetprod = (npoints * pointvol / vol_prefactor(ndim))**2
        cov = make_eigvals_positive(cov, targetprod)
    elif linalg.cond(cov) >= 1./sys.float_info.epsilon:
        raise ValueError("Cannot modify `a` to be non-singular to give "
                         "our ellipsoid non-zero volume if `pointvol` "
                         "is not specified.")

    # The matrix defining the ellipsoid.
    am = linalg.inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # Points should obey `(x-v)^T A (x-v) <= 1`, so we calculate this for
    # each point and then scale A up or down to make the
    # "outermost" point obey `(x-v)^T A (x-v) = 1`. This can be done
    # quickly using `einsum` and `tensordot` to iterate over all points.
    delta = points - ctr
    f = np.einsum('...i, ...i', np.tensordot(delta, am, axes=1), delta)
    fmax = np.max(f)

    # Due to round-off errors, we actually scale the ellipsoid so the outermost
    # point obeys `(x-v)^T A (x-v) < 1 - (a bit) < 1`.
    one_minus_a_bit = 1. - SQRTEPS

    if fmax > one_minus_a_bit:
        am *= one_minus_a_bit / fmax

    # Initialize our ellipsoid.
    ell = Ellipsoid(ctr, am)

    # Expand our ellipsoid to encompass a minimum volume.
    if pointvol > 0.:
        v = npoints * pointvol
        if ell.vol < v:
            ell.scale_to_vol(v)

    return ell


def _bounding_ellipsoids(points, ell, pointvol=0., vol_dec=0.5,
                         vol_check=2.):
    """
    Internal bounding ellipsoids method used when a bounding ellipsoid for
    the entire set has already been calculated.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        Coordinates of points.

    ell : Ellipsoid
        The bounding ellipsoid of the set of points.

    pointvol : float, optional
        Volume represented by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is *None*.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the split. Default is *0.5*.

    vol_check : float, optional
        The factor used to when checking whether the volume of the
        original bounding ellipsoid is large enough to warrant more
        trial splits via `ell.vol > vol_check * npoints * pointvol`.
        Default is *2.0*.

    Returns
    -------
    ells : list of Ellipsoids

    """

    npoints, ndim = points.shape

    # Starting cluster centers are initialized using the major-axis
    # endpoints of the original bounding ellipsoid.
    p1, p2 = ell.major_axis_endpoints()
    start_ctrs = np.vstack((p1, p2))  # shape is (k, ndim) = (2, ndim)

    # Split points into two clusters using k-means clustering with k=2.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k2_res = kmeans2(points, k=start_ctrs, iter=10, minit='matrix',
                             check_finite=False)
        centroids = k2_res[0]  # shape is (k, ndim) = (2, ndim)
        labels = k2_res[1]  # cluster identifier ; shape is (npoints,)

        # Get points in each cluster.
        points_k = [points[labels == k, :] for k in (0, 1)]

        # If either cluster has less than ndim+1 points, the bounding ellipsoid
        # will be ill-constrained. Reject the split and simply return the
        # original ellipsoid bounding all the points.
        if points_k[0].shape[0] < 2 * ndim or points_k[1].shape[0] < 2 * ndim:
            return [ell]

        # Bounding ellipsoid for each cluster, possibly enlarged
        # to a minimum volume.
        ells = [bounding_ellipsoid(points_j, pointvol=pointvol)
                for points_j in points_k]

        # If the total volume decreased by a factor of `vol_dec`, we accept
        # the split into subsets. We then recursively split each subset.
        if ells[0].vol + ells[1].vol < vol_dec * ell.vol:
            return (_bounding_ellipsoids(points_k[0], ells[0],
                                         pointvol=pointvol, vol_dec=vol_dec,
                                         vol_check=vol_check) +
                    _bounding_ellipsoids(points_k[1], ells[1],
                                         pointvol=pointvol, vol_dec=vol_dec,
                                         vol_check=vol_check))

        # Otherwise, see if the total ellipsoid volume is larger than the
        # minimum volume by a factor of `vol_check`. If it is, this indicates
        # that there may be more than 2 clusters and we should try to
        # subdivide further.
        if ell.vol > vol_check * npoints * pointvol:
            out = (_bounding_ellipsoids(points_k[0], ells[0],
                                        pointvol=pointvol, vol_dec=vol_dec,
                                        vol_check=vol_check) +
                   _bounding_ellipsoids(points_k[1], ells[1],
                                        pointvol=pointvol, vol_dec=vol_dec,
                                        vol_check=vol_check))

            # only accept split if volume decreased significantly
            if sum(e.vol for e in out) < vol_dec * ell.vol:
                return out
    except:
        pass

    # Otherwise, we are happy with the single bounding ellipsoid.
    return [ell]


def bounding_ellipsoids(points, pointvol=0., vol_dec=0.5, vol_check=2.):
    """
    Calculate a set of ellipsoids that bound the points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        Coordinates of points.

    pointvol : float, optional
        Volume represented by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is *None*.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the split. Default is *0.5*.

    vol_check : float, optional
        The factor used to when checking whether the volume of the
        original bounding ellipsoid is large enough to warrant more
        trial splits via `ell.vol > vol_check * npoints * pointvol`.
        Default is *2.0*.

    Returns
    -------
    ells : list of Ellipsoids

    """

    if not HAVE_KMEANS:
        raise ValueError("scipy.cluster.vq.kmeans2 is required to compute "
                         "ellipsoid decompositions.")  # pragma: no cover

    # Calculate the bounding ellipsoid for the points possibly
    # enlarged to a minimum volume.
    ell = bounding_ellipsoid(points, pointvol=pointvol)

    # Recursively split the bounding ellipsoid until the volume of each
    # split no longer decreases by a factor of `vol_dec`.
    ells = _bounding_ellipsoids(points, ell, pointvol=pointvol,
                                vol_dec=vol_dec, vol_check=vol_check)

    return MultiEllipsoid(ells=ells)


def _ellipsoid_bootstrap_expand(args):
    """Internal method used when trying to compute the appropriate
    radius expansion factor for a bounding ellipsoid using bootstrap
    resampling."""

    # Unzipping.
    points, pointvol = args
    rstate = np.random

    # Resampling.
    npoints, ndim = points.shape
    idxs = rstate.randint(npoints, size=npoints)  # resample
    idx_in = np.unique(idxs)  # selected objects
    sel = np.ones(npoints, dtype='bool')
    sel[idx_in] = False
    idx_out = np.arange(npoints)[sel]  # "missing" objects
    if len(idx_out) < 2:  # edge case
        idx_out = np.append(idx_out, [0, 1])
    points_in, points_out = points[idx_in], points[idx_out]

    # Compute bounding ellipsoid.
    ell = bounding_ellipsoid(points_in, pointvol=pointvol)

    # Compute normalized distances to missing points.
    dists = [ell.distance(p) for p in points_out]

    # Compute expansion factor.
    expand = max(1., max(dists))

    return expand


def _ellipsoids_bootstrap_expand(args):
    """Internal method used when trying to compute the appropriate
    radius expansion factor for a set of bounding ellipsoids using
    bootstrap resampling."""

    # Unzipping.
    points, pointvol, vol_dec, vol_check = args
    rstate = np.random

    # Resampling.
    npoints, ndim = points.shape
    idxs = rstate.randint(npoints, size=npoints)  # resample
    idx_in = np.unique(idxs)  # selected objects
    sel = np.ones(npoints, dtype='bool')
    sel[idx_in] = False
    idx_out = np.arange(npoints)[sel]  # "missing" objects
    if len(idx_out) < 2:  # edge case
        idx_out = np.append(idx_out, [0, 1])
    points_in, points_out = points[idx_in], points[idx_out]

    # Compute bounding ellipsoids.
    ell = bounding_ellipsoid(points_in, pointvol=pointvol)
    ells = _bounding_ellipsoids(points_in, ell, pointvol=pointvol,
                                vol_dec=vol_dec, vol_check=vol_check)

    # Compute normalized distances to missing points.
    dists = [min([el.distance(p) for el in ells]) for p in points_out]

    # Compute expansion factor.
    expand = max(1., max(dists))

    return expand


def _friends_bootstrap_radius(args):
    """Internal method used when trying to compute the appropriate
    radius for a set of balls/cubes centered on the collection of
    live points using bootstrap resampling."""

    # Unzipping.
    points, ftype = args
    rstate = np.random

    # Resampling.
    npoints, ndim = points.shape
    idxs = rstate.randint(npoints, size=npoints)  # resample
    idx_in = np.unique(idxs)  # selected objects
    sel = np.ones(npoints, dtype='bool')
    sel[idx_in] = False
    idx_out = np.arange(npoints)[sel]  # "missing" objects
    if len(idx_out) < 2:  # edge case
        idx_out = np.append(idx_out, [0, 1])
    points_in, points_out = points[idx_in], points[idx_out]

    # Construct KDTree to enable quick nearest-neighbor lookup for
    # our resampled objects.
    kdtree = spatial.KDTree(points_in)

    if ftype == 'balls':
        # Compute distances from our "missing" points its closest neighbor
        # among the resampled points using the Euclidean norm
        # (i.e. "radius" of n-sphere).
        dists, ids = kdtree.query(points_out, k=1, eps=0, p=2)
    elif ftype == 'cubes':
        # Compute distances from our "missing" points its closest neighbor
        # among the resampled points using the Euclidean norm
        # (i.e. "half-side-length" of n-cube).
        dists, ids = kdtree.query(points_out, k=1, eps=0, p=np.inf)

    # Conservative upper-bound on radius.
    dist = max(dists)

    return dist


def _friends_leaveoneout_radius(points, ftype):
    """Internal method used when trying to compute the appropriate
    radius for a set of balls/cubes centered on the collection of
    live points using leave-one-out (LOO) cross-validation."""

    # Construct KDTree to enable quick nearest-neighbor lookup for
    # our resampled objects.
    kdtree = spatial.KDTree(points)

    if ftype == 'balls':
        # Compute radius to two nearest neighbors (self + neighbor).
        dists, ids = kdtree.query(points, k=2, eps=0, p=2)
    elif ftype == 'cubes':
        # Compute half-side-length to two nearest neighbors (self + neighbor).
        dists, ids = kdtree.query(points, k=2, eps=0, p=np.inf)

    dist = dists[:, 1]  # distances to LOO nearest neighbor

    return dist
