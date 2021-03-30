#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bounding classes used when proposing new live points, along with a number of
useful helper functions. Bounding objects include:

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

"""

from __future__ import (print_function, division)
from six.moves import range

import warnings
import math
import numpy as np
from numpy import linalg
from scipy import special
from scipy import spatial
from scipy import cluster
from scipy import linalg as lalg
from numpy import cov as mle_cov
from scipy.special import logsumexp
from .utils import unitcheck

__all__ = ["UnitCube", "Ellipsoid", "MultiEllipsoid",
           "RadFriends", "SupFriends",
           "vol_prefactor", "logvol_prefactor", "randsphere",
           "bounding_ellipsoid", "bounding_ellipsoids",
           "_bounding_ellipsoids", "_ellipsoid_bootstrap_expand",
           "_ellipsoids_bootstrap_expand", "_friends_bootstrap_radius",
           "_friends_leaveoneout_radius"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

# Try and import k-means clustering (used with 'multi').
try:
    from scipy.cluster.vq import kmeans2
    HAVE_KMEANS = True
except ImportError:
    HAVE_KMEANS = False


class UnitCube(object):
    """
    An N-dimensional unit cube.

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
        """Checks if unit cube contains the point `x`."""

        return unitcheck(x)

    def randoffset(self, rstate=None):
        """Draw a random offset from the center of the unit cube."""

        if rstate is None:
            rstate = np.random

        return self.sample(rstate=rstate) - 0.5

    def sample(self, rstate=None):
        """
        Draw a sample uniformly distributed within the unit cube.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the unit cube.

        """

        if rstate is None:
            rstate = np.random

        return rstate.rand(self.n)

    def samples(self, nsamples, rstate=None):
        """
        Draw `nsamples` samples randomly distributed within the unit cube.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the unit cube.

        """

        if rstate is None:
            rstate = np.random

        xs = np.array([self.sample(rstate=rstate) for i in range(nsamples)])

        return xs

    def update(self, points, pointvol=0., rstate=None, bootstrap=0,
               pool=None):
        """Filler function."""

        pass


class Ellipsoid(object):
    """
    An N-dimensional ellipsoid defined by::

        (x - v)^T A (x - v) = 1

    where the vector `v` is the center of the ellipsoid and `A` is a
    symmetric, positive-definite `N x N` matrix.

    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape (N,)
        Coordinates of ellipsoid center.

    cov : `~numpy.ndarray` with shape (N, N)
        Covariance matrix describing the axes.

    """

    def __init__(self, ctr, cov):
        self.n = len(ctr)  # dimension
        self.ctr = np.array(ctr)  # center coordinates
        self.cov = np.array(cov)  # covariance matrix
        self.am = lalg.pinvh(cov)  # precision matrix (inverse of covariance)
        self.axes = lalg.cholesky(cov, lower=True)  # transformation axes


        # The eigenvalues (l) of `a` are (a^-2, b^-2, ...) where
        # (a, b, ...) are the lengths of principle axes.
        # The eigenvectors (v) are the normalized principle axes.
        l, v = lalg.eigh(self.cov)
        if np.all((l > 0.) & (np.isfinite(l))):
            self.axlens = np.sqrt(l)
            # Volume of ellipsoid is the volume of an n-sphere 
            # is a product of squares of eigen values
            self.logvol = logvol_prefactor(self.n) + 0.5*np.log(l).sum()
        else:
            raise ValueError("The input precision matrix defining the "
                             "ellipsoid {0} is apparently singular with "
                             "l={1} and v={2}.".format(self.cov, l, v))

        # Scaled eigenvectors are the principle axes, where `paxes[:,i]` is the
        # i-th axis. Multiplying this matrix by a vector will transform a
        # point in the unit n-sphere to a point in the ellipsoid.
        self.paxes = np.dot(v, np.diag(self.axlens))

        # Amount by which volume was increased after initialization (i.e.
        # cumulative factor from `scale_to_vol`).
        self.expand = 1.

    def scale_to_logvol(self, logvol):
        """Scale ellipoid to a target volume."""

        f = np.exp((logvol - self.logvol) / self.n)  # linear factor
        self.expand *= f
        self.cov *= f**2
        self.am *= f**-2
        self.axlens *= f
        self.axes *= f
        self.logvol = logvol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis."""

        i = np.argmax(self.axlens)  # find the major axis
        v = self.paxes[:, i]  # vector from center to major axis endpoint

        return self.ctr - v, self.ctr + v

    def distance(self, x):
        """Compute the normalized distance to `x` from the center of the
        ellipsoid."""

        d = x - self.ctr

        return np.sqrt(np.dot(np.dot(d, self.am), d))

    def contains(self, x):
        """Checks if ellipsoid contains `x`."""

        return self.distance(x) <= 1.0

    def randoffset(self, rstate=None):
        """Return a random offset from the center of the ellipsoid."""

        if rstate is None:
            rstate = np.random

        return np.dot(self.axes, randsphere(self.n, rstate=rstate))

    def sample(self, rstate=None):
        """
        Draw a sample uniformly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the ellipsoid.

        """

        if rstate is None:
            rstate = np.random

        return self.ctr + self.randoffset(rstate=rstate)

    def samples(self, nsamples, rstate=None):
        """
        Draw `nsamples` samples uniformly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the ellipsoid.

        """

        if rstate is None:
            rstate = np.random

        xs = np.array([self.sample(rstate=rstate) for i in range(nsamples)])

        return xs

    def unitcube_overlap(self, ndraws=10000, rstate=None):
        """Using `ndraws` Monte Carlo draws, estimate the fraction of
        overlap between the ellipsoid and the unit cube."""

        if rstate is None:
            rstate = np.random

        samples = [self.sample(rstate=rstate) for i in range(ndraws)]
        nin = sum([unitcheck(x) for x in samples])

        return 1. * nin / ndraws

    def update(self, points, pointvol=0., rstate=None, bootstrap=0,
               pool=None, mc_integrate=False):
        """
        Update the ellipsoid to bound the collection of points.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        pointvol : float, optional
            The minimum volume associated with each point. Default is `0.`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoid. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            overlap of the final ellipsoid with the unit cube.
            Default is `False`.

        """

        if rstate is None:
            rstate = np.random

        # Compute new bounding ellipsoid.
        ell = bounding_ellipsoid(points, pointvol=pointvol)
        self.n = ell.n
        self.ctr = ell.ctr
        self.cov = ell.cov
        self.am = ell.am
        self.logvol = ell.logvol
        self.axlens = ell.axlens
        self.axes = ell.axes
        self.paxes = ell.paxes
        self.expand = ell.expand

        # Use bootstrapping to determine the volume expansion factor.
        if bootstrap > 0:

            # If provided, compute bootstraps in parallel using a pool.
            if pool is None:
                M = map
            else:
                M = pool.map
            ps = [points for it in range(bootstrap)]
            pvs = [pointvol for it in range(bootstrap)]
            args = zip(ps, pvs)
            expands = list(M(_ellipsoid_bootstrap_expand, args))

            # Conservatively set the expansion factor to be the maximum
            # factor derived from our set of bootstraps.
            expand = max(expands)

            # If our ellipsoid is over-constrained, expand it.
            if expand > 1.:
                lv = self.logvol + self.n *  np.log( expand)
                self.scale_to_logvol(lv)

        # Estimate the fractional overlap with the unit cube using
        # Monte Carlo integration.
        if mc_integrate:
            self.funit = self.unitcube_overlap()


class MultiEllipsoid(object):
    """
    A collection of M N-dimensional ellipsoids.

    Parameters
    ----------
    ells : list of `Ellipsoid` objects with length M, optional
        A set of `Ellipsoid` objects that make up the collection of
        N-ellipsoids. Used to initialize :class:`MultiEllipsoid` if provided.

    ctrs : `~numpy.ndarray` with shape (M, N), optional
        Collection of coordinates of ellipsoid centers. Used to initialize
        :class:`MultiEllipsoid` if :data:`ams` is also provided.

    covs : `~numpy.ndarray` with shape (M, N, N), optional
        Collection of matrices describing the axes of the ellipsoids. Used to
        initialize :class:`MultiEllipsoid` if :data:`ctrs` also provided.

    """

    def __init__(self, ells=None, ctrs=None, covs=None):
        if ells is not None:
            # Try to initialize quantities using provided `Ellipsoid` objects.
            if (ctrs is None) and (covs is None):
                self.nells = len(ells)
                self.ells = ells
                self.ctrs = np.array([ell.ctr for ell in self.ells])
                self.covs = np.array([ell.cov for ell in self.ells])
                self.ams = np.array([ell.am for ell in self.ells])
            else:
                raise ValueError("You cannot specific both `ells` and "
                                 "(`ctrs`, `covs`)!")
        else:
            # Try to initialize quantities using provided `ctrs` and `covs`.
            if (ctrs is None) and (covs is None):
                raise ValueError("You must specify either `ells` or "
                                 "(`ctrs`, `covs`).")
            else:
                self.nells = len(ctrs)
                self.ctrs = np.array(ctrs)
                self.covs = np.array(covs)
                self.ells = [Ellipsoid(ctrs[i], covs[i])
                             for i in range(self.nells)]
                self.ams = np.array([ell.am for ell in self.ells])

        # Compute quantities.
        self.logvols = np.array([ell.logvol for ell in self.ells])
        self.expands = np.ones(self.nells)
        self.logvol_tot = logsumexp(self.logvols)
        self.expand_tot = 1.

    def scale_to_logvol(self, logvols):
        """Scale ellipoids to a corresponding set of
        target volumes."""

        [self.ells[i].scale_to_logvol(logvols[i]) for i in range(self.nells)]
        self.logvols = np.array(logvols)
        self.expands = np.array([self.ells[i].expand
                                 for i in range(self.nells)])
        logvol_tot = logsumexp(logvols)
        self.expand_tot *= np.exp(logvol_tot - self.logvol_tot)
        self.logvol_tot = logvol_tot

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis of each ellipsoid."""

        return np.array([ell.major_axis_endpoints() for ell in self.ells])

    def within(self, x, j=None):
        """Checks which ellipsoid(s) `x` falls within, skipping the `j`-th
        ellipsoid."""

        # Loop through distance calculations if there aren't too many.
        idxs = np.where([self.ells[i].contains(x) if i != j else True
                         for i in range(self.nells)])[0]

        return idxs

    def overlap(self, x, j=None):
        """Checks how many ellipsoid(s) `x` falls within, skipping the `j`-th
        ellipsoid."""

        q = len(self.within(x, j=j))

        return q

    def contains(self, x):
        """Checks if the set of ellipsoids contains `x`."""

        return self.overlap(x) > 0

    def sample(self, rstate=None, return_q=False):
        """
        Sample a point uniformly distributed within the *union* of ellipsoids.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of ellipsoids.

        idx : int
            The index of the ellipsoid `x` was sampled from.

        q : int, optional
            The number of ellipsoids `x` falls within.

        """

        if rstate is None:
            rstate = np.random

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
        idx = rstate.choice(self.nells, p=np.exp(self.logvols-self.logvol_tot))

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
                idx = rstate.choice(self.nells, p=np.exp(self.logvols-self.logvol_tot))
                x = self.ells[idx].sample(rstate=rstate)
                q = self.overlap(x, j=idx) + 1

            return x, idx

    def samples(self, nsamples, rstate=None):
        """
        Draw `nsamples` samples uniformly distributed within the *union* of
        ellipsoids.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of ellipsoids.

        """

        if rstate is None:
            rstate = np.random

        xs = np.array([self.sample(rstate=rstate)[0]
                       for i in range(nsamples)])

        return xs

    def monte_carlo_logvol(self, ndraws=10000, rstate=None,
                        return_overlap=True):
        """Using `ndraws` Monte Carlo draws, estimate the log volume of the
        *union* of ellipsoids. If `return_overlap=True`, also returns the
        estimated fractional overlap with the unit cube."""

        if rstate is None:
            rstate = np.random

        # Estimate volume using Monte Carlo integration.
        samples = [self.sample(rstate=rstate, return_q=True)
                   for i in range(ndraws)]
        qsum = sum([q for (x, idx, q) in samples])
        logvol = np.log( ndraws *1. /qsum ) +  self.logvol_tot

        if return_overlap:
            # Estimate the fractional amount of overlap with the
            # unit cube using the same set of samples.
            qin = sum([q * unitcheck(x) for (x, idx, q) in samples])
            overlap = 1. * qin / qsum
            return logvol, overlap
        else:
            return logvol

    def update(self, points, pointvol=0., vol_dec=0.5, vol_check=2.,
               rstate=None, bootstrap=0, pool=None, mc_integrate=False):
        """
        Update the set of ellipsoids to bound the collection of points.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        pointvol : float, optional
            The minimum volume associated with each point. Default is `0.`.

        vol_dec : float, optional
            The required fractional reduction in volume after splitting
            an ellipsoid in order to to accept the split.
            Default is `0.5`.

        vol_check : float, optional
            The factor used when checking if the volume of the original
            bounding ellipsoid is large enough to warrant `> 2` splits
            via `ell.vol > vol_check * nlive * pointvol`.
            Default is `2.0`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoids. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            volume and fractional overlap of the final union of ellipsoids
            with the unit cube. Default is `False`.

        """

        if rstate is None:
            rstate = np.random

        if not HAVE_KMEANS:
            raise ValueError("scipy.cluster.vq.kmeans2 is required "
                             "to compute ellipsoid decompositions.")

        npoints, ndim = points.shape

        # Calculate the bounding ellipsoid for the points, possibly
        # enlarged to a minimum volume.
        firstell = bounding_ellipsoid(points, pointvol=pointvol)

        # Recursively split the bounding ellipsoid using `vol_check`
        # until the volume of each split no longer decreases by a
        # factor of `vol_dec`.
        ells = _bounding_ellipsoids(points, firstell, pointvol=pointvol,
                                    vol_dec=vol_dec, vol_check=vol_check)

        # Sanity check: all points must be contained in some ellipsoid
        if not all(any(ell.contains(p) for ell in ells) for p in points):
            # refuse to update
            raise RuntimeError('Rejecting invalid MultiEllipsoid region')

        # Update the set of ellipsoids.
        self.nells = len(ells)
        self.ells = ells
        self.ctrs = np.array([ell.ctr for ell in self.ells])
        self.covs = np.array([ell.cov for ell in self.ells])
        self.ams = np.array([ell.am for ell in self.ells])
        self.logvols = np.array([ell.logvol for ell in self.ells])
        self.logvol_tot = logsumexp(self.logvols)

        # Compute expansion factor.
        expands = np.array([ell.expand for ell in self.ells])
        logvols_orig = self.logvols - np.log(expands)
        logvol_tot_orig = logsumexp(logvols_orig)
        self.expand_tot = np.exp(self.logvol_tot - logvol_tot_orig)

        # Use bootstrapping to determine the volume expansion factor.
        if bootstrap > 0:

            # If provided, compute bootstraps in parallel using a pool.
            if pool is None:
                M = map
            else:
                M = pool.map
            ps = [points for it in range(bootstrap)]
            pvs = [pointvol for it in range(bootstrap)]
            vds = [vol_dec for it in range(bootstrap)]
            vcs = [vol_check for it in range(bootstrap)]
            args = zip(ps, pvs, vds, vcs)
            expands = list(M(_ellipsoids_bootstrap_expand, args))

            # Conservatively set the expansion factor to be the maximum
            # factor derived from our set of bootstraps.
            expand = max(expands)

            # If our ellipsoids are overly constrained, expand them.
            if expand > 1.:
                lvs = self.logvols + ndim * np.log( expand)
                self.scale_to_logvol(lvs)

        # Estimate the volume and fractional overlap with the unit cube
        # using Monte Carlo integration.
        if mc_integrate:
            self.logvol, self.funit = self.monte_carlo_logvol(return_overlap=True)


class RadFriends(object):
    """
    A collection of N-balls of identical size centered on each live point.

    Parameters
    ----------
    ndim : int
        The number of dimensions of each ball.

    cov : `~numpy.ndarray` with shape `(ndim, ndim)`, optional
        Covariance structure (correlation and size) of each ball.

    """

    def __init__(self, ndim, cov=None):
        self.n = ndim

        if cov is None:
            cov = np.identity(self.n)
        self.cov = cov
        self.am = lalg.pinvh(self.cov)
        self.axes = lalg.sqrtm(self.cov)
        self.axes_inv = lalg.pinvh(self.axes)

        detsign, detln = linalg.slogdet(self.am)
        # TODO check for finiteness
        self.logvol_ball = (logvol_prefactor(self.n) - 0.5 * detln)
        self.expand = 1.

    def scale_to_logvol(self, logvol):
        """Scale ball to encompass a target volume."""

        f = np.exp((logvol - self.logvol_ball) * (1.0 / self.n))  # linear factor
        self.expand *= f
        self.cov *= f**2
        self.am /= f**2
        self.axes *= f
        self.axes_inv /= f
        self.logvol_ball = logvol

    def within(self, x, ctrs):
        """Check which balls `x` falls within."""

        # Execute a brute-force search over all balls.
        idxs = np.where(lalg.norm(np.dot(ctrs - x, self.axes_inv), axis=1)
                        <= 1.)[0]

        return idxs

    def overlap(self, x, ctrs):
        """Check how many balls `x` falls within."""

        q = len(self.within(x, ctrs))

        return q

    def contains(self, x, ctrs):
        """Check if the set of balls contains `x`."""

        return self.overlap(x, ctrs) > 0

    def sample(self, ctrs, rstate=None, return_q=False):
        """
        Sample a point uniformly distributed within the *union* of balls.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of balls.

        q : int, optional
            The number of balls `x` falls within.

        """

        if rstate is None:
            rstate = np.random

        nctrs = len(ctrs)  # number of balls

        # If there is only one ball, sample from it.
        if nctrs == 1:
            ds = randsphere(self.n, rstate=rstate)
            dx = np.dot(ds, self.axes)
            x = ctrs[0] + dx
            if return_q:
                return x, 1
            else:
                return x

        # Select a ball at random.
        idx = rstate.randint(nctrs)

        # Select a point from the chosen ball.
        ds = randsphere(self.n, rstate=rstate)
        dx = np.dot(ds, self.axes)
        x = ctrs[idx] + dx

        # Check how many balls the point lies within, passing over
        # the `idx`-th ball `x` was sampled from.
        q = self.overlap(x, ctrs)

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
                ds = randsphere(self.n, rstate=rstate)
                dx = np.dot(ds, self.axes)
                x = ctrs[idx] + dx
                q = self.overlap(x, ctrs)
            return x

    def samples(self, nsamples, ctrs, rstate=None):
        """
        Draw `nsamples` samples uniformly distributed within the *union* of
        balls.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of balls.

        """

        if rstate is None:
            rstate = np.random

        xs = np.array([self.sample(ctrs, rstate=rstate)
                       for i in range(nsamples)])

        return xs

    def monte_carlo_logvol(self, ctrs, ndraws=10000, rstate=None,
                        return_overlap=True):
        """Using `ndraws` Monte Carlo draws, estimate the log volume of the
        *union* of balls. If `return_overlap=True`, also returns the
        estimated fractional overlap with the unit cube."""

        if rstate is None:
            rstate = np.random

        # Estimate volume using Monte Carlo integration.
        samples = [self.sample(ctrs, rstate=rstate, return_q=True)
                   for i in range(ndraws)]
        qsum = sum([q for (x, q) in samples])
        logvol = np.log(1. * ndraws / qsum * len(ctrs) ) + self.logvol_ball

        if return_overlap:
            # Estimate the fractional amount of overlap with the
            # unit cube using the same set of samples.
            qin = sum([q * unitcheck(x) for (x, q) in samples])
            overlap = 1. * qin / qsum
            return logvol, overlap
        else:
            return logvol

    def update(self, points, pointvol=0., rstate=None, bootstrap=0,
               pool=None, mc_integrate=False, use_clustering=True):
        """
        Update the radii of our balls.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        pointvol : float, optional
            The minimum volume associated with each point. Default is `0.`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoids. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            volume and fractional overlap of the final union of balls
            with the unit cube. Default is `False`.

        use_clustering : bool, optional
            Whether to use clustering to avoid issues with widely-seperated
            modes. Default is `True`.

        """

        if rstate is None:
            rstate = np.random

        # If possible, compute bootstraps in parallel using a pool.
        if pool is None:
            M = map
        else:
            M = pool.map

        # Get new covariance.
        if use_clustering:
            self.cov = self._get_covariance_from_clusters(points)
        else:
            self.cov = self._get_covariance_from_all_points(points)
        self.am = lalg.pinvh(self.cov)
        self.axes = lalg.sqrtm(self.cov)
        self.axes_inv = lalg.pinvh(self.axes)

        # Decorrelate and re-scale points.
        points_t = np.dot(points, self.axes_inv)

        if bootstrap == 0.:
            # Construct radius using leave-one-out if no bootstraps used.
            radii = _friends_leaveoneout_radius(points_t, 'balls')
        else:
            # Bootstrap radius using the set of live points.
            ps = [points_t for it in range(bootstrap)]
            ftypes = ['balls' for it in range(bootstrap)]
            args = zip(ps, ftypes)
            radii = list(M(_friends_bootstrap_radius, args))

        # Conservatively set radius to be maximum of the set.
        rmax = max(radii)

        # Re-scale axes.
        self.cov *= rmax**2
        self.am /= rmax**2
        self.axes *= rmax
        self.axes_inv /= rmax

        # Compute volume.
        detsign, detln = linalg.slogdet(self.am)
        # TODO check finite
        self.logvol_ball = (logvol_prefactor(self.n) - 0.5 * detln)
        self.expand = 1.

        # Expand our ball to encompass a minimum volume.
        if pointvol > 0.:
            lv =  np.log(pointvol)
            if self.logvol_ball < lv:
                self.scale_to_logvol(lv)

        # Estimate the volume and fractional overlap with the unit cube
        # using Monte Carlo integration.
        if mc_integrate:
            self.logvol, self.funit = self.monte_carlo_logvol(points,
                                                        return_overlap=True)

    def _get_covariance_from_all_points(self, points):
        """Compute covariance using all points."""

        return np.cov(points, rowvar=False)

    def _get_covariance_from_clusters(self, points):
        """Compute covariance from re-centered clusters."""

        # Compute pairwise distances.
        distances = spatial.distance.pdist(points, metric='mahalanobis',
                                           VI=self.am)

        # Identify conglomerates of points by constructing a linkage matrix.
        linkages = cluster.hierarchy.single(distances)

        # Cut when linkage between clusters exceed the radius.
        clusteridxs = cluster.hierarchy.fcluster(linkages, 1.0,
                                                 criterion='distance')
        nclusters = np.max(clusteridxs)
        if nclusters == 1:
            return self._get_covariance_from_all_points(points)
        else:
            i = 0
            overlapped_points = np.empty_like(points)
            for idx in np.unique(clusteridxs):
                group_points = points[clusteridxs == idx, :]
                group_mean = group_points.mean(axis=0).reshape((1, -1))
                j = i + len(group_points)
                overlapped_points[i:j, :] = group_points - group_mean
                i = j
            return self._get_covariance_from_all_points(overlapped_points)


class SupFriends(object):
    """
    A collection of N-cubes of identical size centered on each live point.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the cube.

    cov : `~numpy.ndarray` with shape `(ndim, ndim)`, optional
        Covariance structure (correlation and size) of each cube.

    """

    def __init__(self, ndim, cov=None):
        self.n = ndim

        if cov is None:
            cov = np.identity(self.n)
        self.cov = cov
        self.am = lalg.pinvh(self.cov)
        self.axes = lalg.sqrtm(self.cov)
        self.axes_inv = lalg.pinvh(self.axes)

        detsign, detln = linalg.slogdet(self.am)
        self.logvol_cube = self.n * np.log(2.) - 0.5 * detln
        self.expand = 1.

    def scale_to_logvol(self, logvol):
        """Scale cube to encompass a target volume."""

        f = np.exp((logvol - self.logvol_cube) * (1.0 / self.n))  # linear factor
        self.expand *= f
        self.cov *= f**2
        self.am /= f**2
        self.axes *= f
        self.axes_inv /= f
        self.logvol_cube = logvol

    def within(self, x, ctrs):
        """Checks which cubes `x` falls within."""

        # Execute a brute-force search over all cubes.
        idxs = np.where(np.max(np.abs(np.dot(ctrs - x, self.axes_inv)), axis=1)
                        <= 1.)[0]

        return idxs

    def overlap(self, x, ctrs):
        """Checks how many cubes `x` falls within, skipping the `j`-th
        cube."""

        q = len(self.within(x, ctrs))

        return q

    def contains(self, x, ctrs):
        """Checks if the set of cubes contains `x`."""

        return self.overlap(x, ctrs) > 0

    def sample(self, ctrs, rstate=None, return_q=False):
        """
        Sample a point uniformly distributed within the *union* of cubes.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of cubes.

        q : int, optional
            The number of cubes `x` falls within.

        """

        if rstate is None:
            rstate = np.random

        nctrs = len(ctrs)  # number of cubes

        # If there is only one cube, sample from it.
        if nctrs == 1:
            ds = (2. * rstate.rand(self.n) - 1.)
            dx = np.dot(ds, self.axes)
            x = ctrs[0] + dx
            if return_q:
                return x, 1
            else:
                return x

        # Select a cube at random.
        idx = rstate.randint(nctrs)

        # Select a point from the chosen cube.
        ds = (2. * rstate.rand(self.n) - 1.)
        dx = np.dot(ds, self.axes)
        x = ctrs[idx] + dx

        # Check how many cubes the point lies within, passing over
        # the `idx`-th cube `x` was sampled from.
        q = self.overlap(x, ctrs)

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
                ds = (2. * rstate.rand(self.n) - 1.)
                dx = np.dot(ds, self.axes)
                x = ctrs[idx] + dx
                q = self.overlap(x, ctrs)
            return x

    def samples(self, nsamples, ctrs, rstate=None):
        """
        Draw `nsamples` samples uniformly distributed within the *union* of
        cubes.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of cubes.

        """

        if rstate is None:
            rstate = np.random

        xs = np.array([self.sample(ctrs, rstate=rstate)
                       for i in range(nsamples)])

        return xs

    def monte_carlo_logvol(self, ctrs, ndraws=10000, rstate=None,
                        return_overlap=False):
        """Using `ndraws` Monte Carlo draws, estimate the log volume of the
        *union* of cubes. If `return_overlap=True`, also returns the
        estimated fractional overlap with the unit cube."""

        if rstate is None:
            rstate = np.random

        # Estimate the volume using Monte Carlo integration.
        samples = [self.sample(ctrs, rstate=rstate, return_q=True)
                   for i in range(ndraws)]
        qsum = sum([q for (x, q) in samples])
        logvol = np.log(1. * ndraws / qsum * len(ctrs)) + self.logvol_cube

        if return_overlap:
            # Estimate the fractional overlap with the unit cube using
            # the same set of samples.
            qin = sum([q * unitcheck(x) for (x, q) in samples])
            overlap = 1. * qin / qsum
            return logvol, overlap
        else:
            return logvol

    def update(self, points, pointvol=0., rstate=None, bootstrap=0,
               pool=None, mc_integrate=False, use_clustering=True):
        """
        Update the half-side-lengths of our cubes.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        pointvol : float, optional
            The minimum volume associated with each point. Default is `0.`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoids. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            volume and fractional overlap of the final union of cubes
            with the unit cube. Default is `False`.

        use_clustering : bool, optional
            Whether to use clustering to avoid issues with widely-seperated
            modes. Default is `True`.

        """

        if rstate is None:
            rstate = np.random

        # If possible, compute bootstraps in parallel using a pool.
        if pool is None:
            M = map
        else:
            M = pool.map

        # Get new covariance.
        if use_clustering:
            self.cov = self._get_covariance_from_clusters(points)
        else:
            self.cov = self._get_covariance_from_all_points(points)
        self.am = lalg.pinvh(self.cov)
        self.axes = lalg.sqrtm(self.cov)
        self.axes_inv = lalg.pinvh(self.axes)

        # Decorrelate and re-scale points.
        points_t = np.dot(points, self.axes_inv)

        if bootstrap == 0.:
            # Construct radius using leave-one-out if no bootstraps used.
            hsides = _friends_leaveoneout_radius(points_t, 'cubes')
        else:
            # Bootstrap radius using the set of live points.
            ps = [points_t for it in range(bootstrap)]
            ftypes = ['cubes' for it in range(bootstrap)]
            args = zip(ps, ftypes)
            hsides = list(M(_friends_bootstrap_radius, args))

        # Conservatively set half-side-length to be maximum of the set.
        hsmax = max(hsides)

        # Re-scale axes.
        self.cov *= hsmax**2
        self.am /= hsmax**2
        self.axes *= hsmax
        self.axes_inv /= hsmax

        detsign, detln = linalg.slogdet(self.am)
        self.logvol_cube = (self.n * np.log(2.) - 0.5 * detln)
        self.expand = 1.

        # Expand our cube to encompass a minimum volume.
        if pointvol > 0.:
            lv = np.log(pointvol)
            if self.logvol_cube < lv:
                self.scale_to_logvol(lv)

        # Estimate the volume and fractional overlap with the unit cube
        # using Monte Carlo integration.
        if mc_integrate:
            self.logvol, self.funit = self.monte_carlo_logvol(points,
                                                        return_overlap=True)

    def _get_covariance_from_all_points(self, points):
        """Compute covariance using all points."""

        return np.cov(points, rowvar=False)

    def _get_covariance_from_clusters(self, points):
        """Compute covariance from re-centered clusters."""

        # Compute pairwise distances.
        distances = spatial.distance.pdist(points, metric='mahalanobis',
                                           VI=self.am)

        # Identify conglomerates of points by constructing a linkage matrix.
        linkages = cluster.hierarchy.single(distances)

        # Cut when linkage between clusters exceed the radius.
        clusteridxs = cluster.hierarchy.fcluster(linkages, 1.0,
                                                 criterion='distance')
        nclusters = np.max(clusteridxs)
        if nclusters == 1:
            return self._get_covariance_from_all_points(points)
        else:
            i = 0
            overlapped_points = np.empty_like(points)
            for idx in np.unique(clusteridxs):
                group_points = points[clusteridxs == idx, :]
                group_mean = group_points.mean(axis=0).reshape((1, -1))
                j = i + len(group_points)
                overlapped_points[i:j, :] = group_points - group_mean
                i = j
            return self._get_covariance_from_all_points(overlapped_points)


##################
# HELPER FUNCTIONS
##################

def vol_prefactor(n, p=2.):
    """
    Returns the volume constant for an `n`-dimensional sphere with an
    :math:`L^p` norm. The constant is defined as::

        f = (2. * Gamma(1./p + 1))**n / Gamma(n/p + 1.)

    By default the `p=2.` norm is used (i.e. the standard Euclidean norm).

    """

    p *= 1.  # convert to float in case user inputs an integer
    f = (2 * special.gamma(1./p + 1.))**n / special.gamma(n/p + 1)

    return f


def logvol_prefactor(n, p=2.):
    """
    Returns the ln(volume constant) for an `n`-dimensional sphere with an
    :math:`L^p` norm. The constant is defined as::

        lnf = n * ln(2.) + n * LogGamma(1./p + 1) - LogGamma(n/p + 1.)

    By default the `p=2.` norm is used (i.e. the standard Euclidean norm).

    """

    p *= 1.  # convert to float in case user inputs an integer
    lnf = (n * np.log(2.) + n * special.gammaln(1./p + 1.) -
           special.gammaln(n/p + 1))

    return lnf


def randsphere(n, rstate=None):
    """Draw a point uniformly within an `n`-dimensional unit sphere."""

    if rstate is None:
        rstate = np.random

    z = rstate.randn(n)  # initial n-dim vector
    zhat = z / lalg.norm(z)  # normalize
    xhat = zhat * rstate.rand()**(1./n)  # scale

    return xhat


def bounding_ellipsoid(points, pointvol=0.):
    """
    Calculate the bounding ellipsoid containing a collection of points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    pointvol : float, optional
        The minimum volume occupied by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is `0.`.

    Returns
    -------
    ellipsoid : :class:`Ellipsoid`
        The bounding :class:`Ellipsoid` object.

    """

    npoints, ndim = points.shape
    ntries = 100  # number of times to try fixing covariance

    # Check for valid `pointvol` value if provided.
    if pointvol < 0.:
        raise ValueError("You must specify a non-negative value "
                         "for `pointvol`.")

    # If there is only a single point, return an n-sphere with volume
    # `pointvol` centered at the point.
    if npoints == 1:
        if pointvol > 0.:
            ctr = points[0]
            r = np.exp((np.log(pointvol) - logvol_prefactor(ndim)) / ndim)
            covar = r**2 * np.identity(ndim)
            return Ellipsoid(ctr, covar)
        else:
            raise ValueError("Cannot compute a bounding ellipsoid to a "
                             "single point if `pointvol` is not specified.")

    # Calculate covariance of points.
    ctr = np.mean(points, axis=0)
    cov = mle_cov(points, rowvar=False)

    # When ndim = 1, `np.cov` returns a 0-d array. Make it a 1x1 2-d array.
    if ndim == 1:
        cov = np.atleast_2d(cov)

    # For a ball of uniformly distributed points, the sample covariance
    # will be smaller than the true covariance by a factor of 1/(n+2)
    # [see, e.g., goo.gl/UbsjYl]. Since we are assuming all points are
    # uniformly distributed within the unit cube, they are uniformly
    # distributed within any sub-volume within the cube. We expand
    # our sample covariance `cov` to compensate for this.
    cov *= (ndim + 2)

    # Define the axes of our ellipsoid. Ensures that `cov` is
    # nonsingular to deal with pathological cases where the ellipsoid has
    # "zero" volume. This can occur when `npoints <= ndim` or when enough
    # points are linear combinations of other points.
    covar = np.array(cov)
    for trials in range(ntries):
        failed = False
        try:
            # Check if matrix is invertible.
            am = lalg.pinvh(covar)
            l, v = lalg.eigh(covar)  # compute eigenvalues/vectors

            # Check if direct Cholesky decomposition exists.
            lalg.cholesky(covar, lower=True)

            # Check if everything worked.
            if np.all((l > 0.) & (np.isfinite(l))):
                break
            else:
                failed = True
        except lalg.LinAlgError:
            # If the matrix remains singular/unstable,
            # suppress the off-diagonal elements.
            coeff = 0.9**(trials + 1 - ntries)
            covar = (1. - coeff) * cov + coeff * np.eye(ndim)
            failed =  True
    if failed:
        warnings.warn("Failed to guarantee the ellipsoid axes will be "
                      "non-singular. Defaulting to a sphere.")
        covar = np.eye(ndim)  # default to identity

    # Calculate expansion factor necessary to bound each point.
    # Points should obey `(x-v)^T A (x-v) <= 1`, so we calculate this for
    # each point and then scale A up or down to make the
    # "outermost" point obey `(x-v)^T A (x-v) = 1`. This can be done
    # quickly using `einsum` and `tensordot` to iterate over all points.
    delta = points - ctr
    f = np.einsum('...i, ...i', np.tensordot(delta, am, axes=1), delta)
    fmax = np.max(f)

    
    # Due to round-off errors, we actually scale the ellipsoid so the
    # outermost point obeys `(x-v)^T A (x-v) < 1 - (a bit) < 1`.
    ROUND_DELTA = 1e-5
    # numerical experiments show that round off errors can reach large
    # values if the matrix eigen values are very low
    one_minus_a_bit = 1. - ROUND_DELTA
    covar_mod = np.array(covar)
    if fmax > one_minus_a_bit:
        covar_mod *= fmax / one_minus_a_bit

    # Repeat the ellipsoid check above just in case this modification makes
    # us numerically unstable again. Just to be **ultra safe**.
    covar2 = np.array(covar_mod)
    for trials in range(ntries):
        failed = False
        try:
            # Check if matrix is invertible.
            am = lalg.pinvh(covar2)
            l, v = lalg.eigh(covar2)  # compute eigenvalues/vectors

            # Check if direct Cholesky decomposition exists.
            lalg.cholesky(covar2, lower=True)

            # Check if everything worked.
            if np.all((l > 0.) & (np.isfinite(l))):
                break
            else:
                failed =True
        except lalg.LinAlgError:
            failed =True
        if failed:
            # If the matrix remains singular/unstable,
            # suppress the off-diagonal elements.
            coeff = 0.9**(trials + 1 - ntries)
            covar2 = (1. - coeff) * covar_mod + coeff * np.eye(ndim)
    if failed:
        warnings.warn("Failed to guarantee the ellipsoid axes will be "
                      "non-singular. Defaulting to last working axes.")
        covar2 = np.array(covar)  # default to last safe version
    # this is a final check
    # if this fails the ellipsoid is broken already
    if np.einsum('...i, ...i', np.tensordot(delta, lalg.pinvh(covar2), axes=1), delta).max()>=1:
        raise RuntimeError("Failed to initialize the ellipsoid to contain all the points")
        
    # Initialize our ellipsoid with *safe* covariance matrix.
    ell = Ellipsoid(ctr, covar2)

    # Expand our ellipsoid to encompass a minimum volume.
    if pointvol > 0.:
        minvol = npoints * pointvol
        if ell.logvol < np.log(minvol):
            ell.scale_to_logvol(np.log(minvol))

    return ell


def _bounding_ellipsoids(points, ell, pointvol=0., vol_dec=0.5,
                         vol_check=2.):
    """
    Internal method used to compute a set of bounding ellipsoids when a
    bounding ellipsoid for the entire set has already been calculated.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    ell : Ellipsoid
        The bounding ellipsoid containing :data:`points`.

    pointvol : float, optional
        Volume represented by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is `0.`.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the split. Default is `0.5`.

    vol_check : float, optional
        The factor used to when checking whether the volume of the
        original bounding ellipsoid is large enough to warrant more
        trial splits via `ell.vol > vol_check * npoints * pointvol`.
        Default is `2.0`.

    Returns
    -------
    ells : list of :class:`Ellipsoid` objects
        List of :class:`Ellipsoid` objects used to bound the
        collection of points. Used to initialize the :class:`MultiEllipsoid`
        object returned in :meth:`bounding_ellipsoids`.

    """

    npoints, ndim = points.shape

    # Starting cluster centers are initialized using the major-axis
    # endpoints of the original bounding ellipsoid.
    p1, p2 = ell.major_axis_endpoints()
    start_ctrs = np.vstack((p1, p2))  # shape is (k, ndim) = (2, ndim)

    # Split points into two clusters using k-means clustering with k=2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k2_res = kmeans2(points, k=start_ctrs, iter=10, minit='matrix',
                         check_finite=False)
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
    if np.logaddexp(ells[0].logvol, ells[1].logvol) < np.log(vol_dec)+ ell.logvol:
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
    if ell.logvol > np.log(vol_check * npoints * pointvol):
        out = (_bounding_ellipsoids(points_k[0], ells[0],
                                    pointvol=pointvol, vol_dec=vol_dec,
                                    vol_check=vol_check) +
               _bounding_ellipsoids(points_k[1], ells[1],
                                    pointvol=pointvol, vol_dec=vol_dec,
                                    vol_check=vol_check))

        # Only accept the split if the volume decreased significantly.
        if logsumexp([e.logvol for e in out]) <np.log( vol_dec) + ell.logvol:
            return out

    # Otherwise, we are happy with the single bounding ellipsoid.
    return [ell]


def bounding_ellipsoids(points, pointvol=0., vol_dec=0.5, vol_check=2.):
    """
    Calculate a set of ellipsoids that bound the collection of points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    pointvol : float, optional
        Volume represented by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is `0.`.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the split. Default is `0.5`.

    vol_check : float, optional
        The factor used to when checking whether the volume of the
        original bounding ellipsoid is large enough to warrant more
        trial splits via `ell.vol > vol_check * npoints * pointvol`.
        Default is `2.0`.

    Returns
    -------
    mell : :class:`MultiEllipsoid` object
        The :class:`MultiEllipsoid` object used to bound the
        collection of points.

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
    """Internal method used to compute the expansion factor for a bounding
    ellipsoid based on bootstrapping."""

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
    """Internal method used to compute the expansion factor(s) for a collection
    of bounding ellipsoids using bootstrapping."""

    # Unzipping.
    points, pointvol, vol_dec, vol_check = args
    rstate = np.random

    # Resampling.
    npoints, ndim = points.shape
    idxs = rstate.randint(npoints, size=npoints)  # resample
    idx_in = np.unique(idxs)  # selected objects
    sel = np.ones(npoints, dtype='bool')
    sel[idx_in] = False
    idx_out = np.where(sel)[0]  # "missing" objects
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
    """Internal method used to compute the radius (half-side-length) for each
    ball (cube) used in :class:`RadFriends` (:class:`SupFriends`) using
    bootstrapping."""

    # Unzipping.
    points, ftype = args
    rstate = np.random

    # Resampling.
    npoints, ndim = points.shape
    idxs = rstate.randint(npoints, size=npoints)  # resample
    idx_in = np.unique(idxs)  # selected objects
    sel = np.ones(npoints, dtype='bool')
    sel[idx_in] = False
    idx_out = np.where(sel)[0]  # "missing" objects
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
    """Internal method used to compute the radius (half-side-length) for each
    ball (cube) used in :class:`RadFriends` (:class:`SupFriends`) using
    leave-one-out (LOO) cross-validation."""

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
