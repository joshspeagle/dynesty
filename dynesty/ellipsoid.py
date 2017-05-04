#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ellipsoid and MultiEllipsoid classes along with helper functions for bounding
a set of points using one or more ellipsoids based on Feroz et al. (2009)
<https://arxiv.org/abs/0809.3437>.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np

try:
    from scipy.cluster.vq import kmeans2
    HAVE_KMEANS = True
except ImportError:  # pragma: no cover
    HAVE_KMEANS = False

from utils import random_choice

__all__ = ["Ellipsoid", "MultiEllipsoid", "vol_prefactor", "randsphere",
           "make_eigvals_positive", "bounding_ellipsoid",
           "bounding_ellipsoids"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


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
        self.vol = vol_prefactor(self.n) / np.sqrt(np.linalg.det(self.am))

        # The eigenvalues (l) of `a` are (a^-2, b^-2, ...) where
        # (a, b, ...) are the lengths of principle axes.
        # The eigenvectors (v) are the normalized principle axes.
        l, v = np.linalg.eigh(self.am)
        self.axlens = 1. / np.sqrt(l)

        # Scaled eigenvectors are the axes, where `axes[:,i]` is the
        # i-th axis.  Multiplying this matrix by a vector will transform a
        # point in the unit n-sphere to a point in the ellipsoid.
        self.axes = np.dot(v, np.diag(self.axlens))

    def scale_to_vol(self, vol):
        """Scale ellipoid to encompass a target volume."""

        f = (vol / self.vol) ** (1.0 / self.n)  # linear factor
        self.am *= f**-2
        self.axlens *= f
        self.axes *= f
        self.vol = vol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis."""

        i = np.argmax(self.axlens)  # find the major axis
        v = self.axes[:, i]  # vector from center to major axis endpoint

        return self.ctr - v, self.ctr + v

    def contains(self, x):
        """Checks if ellipsoid contains `x`."""

        d = x - self.ctr

        return np.dot(np.dot(d, self.am), d) <= 1.0

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

        xs = np.array([self.sample(rstate=rstate) for i in xrange(nsamples)])

        return xs


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
                self.vols = np.array([ell.vol for ell in self.ells])
                self.vol_tot = sum(self.vols)
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
                             for i in xrange(self.nells)]
                self.vols = [ell.vol for ell in self.ells]

    def scale_to_vols(self, vols):
        """Scale ellipoids to encompass a corresponding set of
        target volume."""

        _ = [self.ells[i].scale_to_vol(vols[i]) for i in xrange(self.nells)]
        self.vols = np.array(vols)
        self.vol_tot = sum(vols)

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis of each ellipsoid."""

        i = np.argmax(self.axlens)  # find the major axis
        v = self.axes[:, i]  # vector from center to major axis endpoint

        return np.array([ell.major_axis_endpoints() for ell in self.ells])

    def overlap(self, x, j=None):
        """Checks how many ellipsoids `x` falls within, skipping the `j`-th
        ellipsoid."""

        q = sum([self.ells[i].contains(x) for i in xrange(self.nells)
                 if i != j])

        return q

    def contains(self, x):
        """Checks if the set of ellipsoids contains `x`."""

        return self.overlap(x) > 0

    def sample(self, rstate=np.random, return_q=False):
        """
        Sample a point uniformly distributed within the set of ellipsoids.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of ellipsoids.
        q : int, optional
            The number of ellipsoids `x` falls within.

        """

        # If there is only one ellipsoid, sample from it.
        if self.nells == 1:
            x = self.ells[0].sample(rstate=rstate)
            if return_q:
                return x, 1
            else:
                return x

        # Select an ellipsoid at random proportional to its volume.
        idx = random_choice(self.nells, self.vols / self.vol_tot,
                            rstate=rstate)

        # Select a point from the chosen ellipsoid.
        x = self.ells[idx].sample(rstate=rstate)

        if return_q:
            # Check how many ellipsoids the point lies within, passing over
            # the `idx`-th ellipsoid `x` was sampled from.
            q = self.overlap(x, j=idx) + 1
            return x, q
        else:
            return x

    def samples(self, nsamples, rstate=np.random, return_qs=False):
        """
        Draw `nsamples` samples uniformly distributed within the set of
        ellipsoids.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of ellipsoids.
        qs : `~numpy.ndarray` with shape (nsamples,), optional
            The number of ellipsoids each sample falls within.

        """

        if return_qs:
            samples = np.array([self.sample(rstate=rstate, return_q=True)
                                for i in xrange(nsamples)])
            xs, qs = samples[:, 0], samples[:, 1]
            return xs, qs
        else:
            xs = np.array([self.sample(rstate=rstate, return_q=False)
                           for i in xrange(nsamples)])
            return xs


def vol_prefactor(n):
    """
    Volume constant for an n-dimensional sphere:

    for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
    for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)

    """

    if n % 2 == 0:
        f = 1.
        i = 2
        while i <= n:
            f *= (2. / i * math.pi)
            i += 2
    else:
        f = 2.
        i = 3
        while i <= n:
            f *= (2. / i * math.pi)
            i += 2

    return f


def randsphere(n, rstate=np.random):
    """Draw a random point within an n-dimensional unit sphere."""

    z = rstate.randn(n)  # initial n-dim vector

    return z * rstate.rand()**(1./n) / np.sqrt(np.sum(z**2))


def make_eigvals_positive(am, targetprod):
    """For the symmetric square matrix `am`, increase any zero eigenvalues
    to fulfill the given target product of eigenvalues. Returns a
    (possibly) new matrix."""

    w, v = np.linalg.eigh(am)  # use eigh since a is symmetric
    mask = w < 1.e-10
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
        am = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov

    return am


def bounding_ellipsoid(points, pointvol=None):
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
    if pointvol is not None and pointvol <= 0.:
        raise ValueError("You must specify a positive value for `pointvol`.")

    # If there is only a single point, return an n-sphere with volume
    # `pointvol` centered at the point.
    if npoints == 1:
        if pointvol is not None:
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
    if pointvol is not None:
        targetprod = (npoints * pointvol / vol_prefactor(ndim))**2
        cov = make_eigvals_positive(cov, targetprod)
    else:
        raise ValueError("Cannot modify `a` to be non-singular to give "
                         "our ellipsoid non-zero volume if `pointvol` "
                         "is not specified.")

    # The matrix defining the ellipsoid.
    am = np.linalg.inv(cov)

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
    if pointvol is not None:
        v = npoints * pointvol
        if ell.vol < v:
            ell.scale_to_vol(v)
    else:
        raise ValueError("Cannot expand ellipsoid if `pointvol` "
                         "is not specified.")

    return ell


def _bounding_ellipsoids(points, ell, pointvol=None, vol_dec=0.5,
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
    k2_res = kmeans2(points, k=start_ctrs, iter=10, minit='matrix')
    centroids = k2_res[0]  # shape is (k, ndim) = (2, ndim)
    labels = k2_res[1]  # cluster identifier ; shape is (npoints,)

    # Get points in each cluster.
    points_k = [points[labels == k, :] for k in (0, 1)]

    # If either cluster has less than ndim+1 points, the bounding ellipsoid
    # will be ill-constrained. Reject the split and simply return the
    # original ellipsoid bounding all the points.
    if points_k[0].shape[0] < 2 * ndim or points_k[1].shape[0] < 2 * ndim:
        return [ell]

    # Bounding ellipsoid for each cluster, possibly enlarged to minimum volume.
    ells = [bounding_ellipsoid(points_j, pointvol=pointvol)
            for points_j in points_k]

    # If the total volume decreased by a factor of `vol_dec`, we accept
    # the split into subsets. We then recursively split each subset.
    if ells[0].vol + ells[1].vol < vol_dec * ell.vol:
        return (_bounding_ellipsoids(points_k[0], ells[0], pointvol=pointvol,
                                     vol_dec=vol_dec, vol_check=vol_check) +
                _bounding_ellipsoids(points_k[1], ells[1], pointvol=pointvol,
                                     vol_dec=vol_dec, vol_check=vol_check))

    # Otherwise, see if the total ellipsoid volume is larger than the minimum
    # volume by a factor of `vol_check`. If it is, this indicates that there
    # may be more than 2 clusters and we should try to subdivide further.
    if ell.vol > vol_check * npoints * pointvol:
        out = (_bounding_ellipsoids(points_k[0], ells[0], pointvol=pointvol,
                                    vol_dec=vol_dec, vol_check=vol_check) +
               _bounding_ellipsoids(points_k[1], ells[1], pointvol=pointvol,
                                    vol_dec=vol_dec, vol_check=vol_check))

        # only accept split if volume decreased significantly
        if sum(e.vol for e in out) < vol_dec * ell.vol:
            return out

    # Otherwise, we are happy with the single bounding ellipsoid.
    return [ell]


def bounding_ellipsoids(points, pointvol=None, vol_dec=0.5, vol_check=2.):
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
