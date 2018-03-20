#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Childen of :class:`dynesty.sampler` used to proposing new live points.
Includes:

    UnitCubeSampler:
        Uses the unit cube to bound the set of live points (i.e. no bound).

    SingleEllipsoidSampler:
        Uses a single ellipsoid to bound the set of live points.

    MultiEllipsoidSampler:
        Uses multiple ellipsoids to bound the set of live points.

    RadFriendsSampler:
        Uses an N-sphere of fixed radius centered on each
        live point to bound the set of live points.

    SupFriendsSampler:
        Uses an N-cube of fixed length centered on each
        live point to bound the set of live points.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import warnings
import math
import numpy as np
import copy
from numpy import linalg
from scipy import spatial

from .sampler import *
from .bounding import *
from .sampling import *

__all__ = ["UnitCubeSampler", "SingleEllipsoidSampler",
           "MultiEllipsoidSampler", "RadFriendsSampler", "SupFriendsSampler"]

_SAMPLING = {'unif': sample_unif,
             'rwalk': sample_rwalk,
             'slice': sample_slice,
             'rslice': sample_rslice,
             'hslice': sample_hslice}


class UnitCubeSampler(Sampler):
    """
    Samples conditioned on the unit N-cube (i.e. with no bounds).

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.RandomState`
        `~numpy.random.RandomState` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    kwargs : dict, optional
        A dictionary of additional parameters (described below).

    Other Parameters
    ----------------
    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling options, the
        number of times to execute a "slice update" before proposing a new
        live point. Default is `5`. Note that `'slice'` cycles through
        **all dimensions** when executing a "slice update".

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_unif}
        self.update_proposal = self._UPDATE[method]

        # Initialize other arguments.
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'unif':
                self.bootstrap = 20
            else:
                self.bootstrap = 0
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)

        # Initialize sampler.
        super(UnitCubeSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.unitcube = UnitCube(self.npdim)
        self.bounding = 'none'
        self.method = method

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)

    def update(self, pointvol):
        """Update the unit cube bound."""

        return copy.deepcopy(self.unitcube)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly*
        within the unit cube."""

        u = self.unitcube.sample(rstate=self.rstate)
        ax = np.identity(self.npdim)

        return u, ax

    def propose_live(self):
        """Return a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim)

        return u, ax

    def update_unif(self, blob):
        """Filler function."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        self.scale = blob['scale']
        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc) * self.npdim
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class SingleEllipsoidSampler(Sampler):
    """
    Samples conditioned on a single ellipsoid used to bound the
    set of live points.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.RandomState`
        `~numpy.random.RandomState` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    kwargs : dict, optional
        A dictionary of additional parameters (described below).

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the ellipsoids by this fraction. The preferred
        method is to determine this organically using bootstrapping. If
        `bootstrap > 0`, this defaults to `1.0`. If `bootstrap = 0`,
        this instead defaults to `1.25`.

    bootstrap : int, optional
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes.
        Default is `20` for uniform sampling (`'unif'`) and `0` otherwise.

    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling options, the
        number of times to execute a "slice update" before proposing a new
        live point. Default is `5`. Note that `'slice'` cycles through
        **all dimensions** when executing a "slice update".

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_unif}
        self.update_proposal = self._UPDATE[method]

        # Initialize other arguments.
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'unif':
                self.bootstrap = 20
            else:
                self.bootstrap = 0
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)

        # Initialize sampler.
        super(SingleEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.ell = Ellipsoid(np.zeros(self.npdim), np.identity(self.npdim))
        self.bounding = 'single'
        self.method = method

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)

    def update(self, pointvol):
        """Update the bounding ellipsoid using the current set of
        live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the ellipsoid.
        self.ell.update(self.live_u, pointvol=pointvol, rstate=self.rstate,
                        bootstrap=self.bootstrap, pool=pool)
        if self.enlarge != 1.:
            self.ell.scale_to_vol(self.ell.vol * self.enlarge)

        return copy.deepcopy(self.ell)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly*
        within the ellipsoid."""

        while True:
            # Sample a point from the ellipsoid.
            u = self.ell.sample(rstate=self.rstate)

            # Check if `u` is within the unit cube.
            if self._check_unit_cube(u):
                break  # if it is, we're done!

        return u, self.ell.axes

    def propose_live(self):
        """Return a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Choose axes.
        if self.sampling == 'rwalk':
            ax = self.ell.axes
        elif self.sampling == 'slice':
            ax = self.ell.paxes
        else:
            ax = np.identity(self.npdim)

        return u, ax

    def update_unif(self, blob):
        """Filler function."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        self.scale = blob['scale']
        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc) * self.npdim
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class MultiEllipsoidSampler(Sampler):
    """
    Samples conditioned on the union of multiple (possibly overlapping)
    ellipsoids used to bound the set of live points.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.RandomState`
        `~numpy.random.RandomState` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    kwargs : dict, optional
        A dictionary of additional parameters (described below).

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the ellipsoids by this fraction. The preferred
        method is to determine this organically using bootstrapping. If
        `bootstrap > 0`, this defaults to `1.0`. If `bootstrap = 0`,
        this instead defaults to `1.25`.

    bootstrap : int, optional
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes.
        Default is `20` for uniform sampling (`'unif'`) and `0` otherwise.

    vol_dec : float, optional
        For the `'multi'` bounding option, the required fractional reduction
        in volume after splitting an ellipsoid in order to to accept the split.
        Default is `0.5`.

    vol_check : float, optional
        For the `'multi'` bounding option, the factor used when checking if
        the volume of the original bounding ellipsoid is large enough to
        warrant `> 2` splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is `2.0`.

    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling options, the
        number of times to execute a "slice update" before proposing a new
        live point. Default is `5`. Note that `'slice'` cycles through
        **all dimensions** when executing a "slice update".

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_unif}
        self.update_proposal = self._UPDATE[method]

        # Initialize other arguments.
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'unif':
                self.bootstrap = 20
            else:
                self.bootstrap = 0
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        self.vol_dec = kwargs.get('vol_dec', 0.5)
        self.vol_check = kwargs.get('vol_check', 2.0)

        # Initialize sampler.
        super(MultiEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.mell = MultiEllipsoid(ctrs=[np.zeros(self.npdim)],
                                   covs=[np.identity(self.npdim)])
        self.bounding = 'multi'
        self.method = method

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)

    def update(self, pointvol):
        """Update the bounding ellipsoids using the current set of
        live points."""

        # Check if we should use the pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the bounding ellipsoids.
        self.mell.update(self.live_u, pointvol=pointvol,
                         vol_dec=self.vol_dec, vol_check=self.vol_check,
                         rstate=self.rstate, bootstrap=self.bootstrap,
                         pool=pool)
        if self.enlarge != 1.:
            self.mell.scale_to_vols(self.mell.vols * self.enlarge)

        return copy.deepcopy(self.mell)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the union of ellipsoids."""

        while True:
            # Sample a point from the union of ellipsoids.
            # Returns the point `u`, ellipsoid index `idx`, and number of
            # overlapping ellipsoids `q` at position `u`.
            u, idx, q = self.mell.sample(rstate=self.rstate, return_q=True)

            # Check if the point is within the unit cube.
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping ellipsoids.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break  # if successful, we're done!

        return u, self.mell.ells[idx].axes

    def propose_live(self):
        """Return a live point/axes to be used by other sampling methods."""

        # Copy a random live point.
        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Check for ellipsoid overlap.
        ell_idxs = self.mell.within(u)
        nidx = len(ell_idxs)

        # Automatically trigger an update if we're not in any ellipsoid.
        if nidx == 0:
            try:
                # Expected ln(prior volume) at a given iteration.
                expected_vol = math.exp(self.saved_logvol[-1] - self.dlv)
            except:
                # Expected ln(prior volume) at the first iteration.
                expected_vol = math.exp(-self.dlv)
            pointvol = expected_vol / self.nlive  # minimum point volume

            # Update the bounding ellipsoids.
            bound = self.update(pointvol)
            if self.save_bounds:
                self.bound.append(bound)
            self.nbound += 1
            self.since_update = 0

            # Check for ellipsoid overlap (again).
            ell_idxs = self.mell.within(u)
            nidx = len(ell_idxs)

        # Pick a random ellipsoid that encompasses `u`.
        ell_idx = ell_idxs[self.rstate.randint(nidx)]

        # Choose axes.
        if self.sampling == 'rwalk':
            ax = self.mell.ells[ell_idx].axes
        elif self.sampling == 'slice':
            ax = self.mell.ells[ell_idx].paxes
        else:
            ax = np.identity(self.npdim)

        return u, ax

    def update_unif(self, blob):
        """Filler function."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        self.scale = blob['scale']
        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc) * self.npdim
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class RadFriendsSampler(Sampler):
    """
    Samples conditioned on the union of (possibly overlapping) N-spheres
    centered on the current set of live points.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.RandomState`
        `~numpy.random.RandomState` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    kwargs : dict, optional
        A dictionary of additional parameters (described below).

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the N-spheres by this fraction. The preferred
        method is to determine this organically using bootstrapping. If
        `bootstrap > 0`, this defaults to `1.0`. If `bootstrap = 0`,
        this instead defaults to `1.25`.

    bootstrap : int, optional
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes.
        Default is `20` for uniform sampling (`'unif'`) and `0` otherwise.

    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling options, the
        number of times to execute a "slice update" before proposing a new
        live point. Default is `5`. Note that `'slice'` cycles through
        **all dimensions** when executing a "slice update".

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_unif}
        self.update_proposal = self._UPDATE[method]

        # Initialize other arguments.
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'unif':
                self.bootstrap = 20
            else:
                self.bootstrap = 0
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)

        # Initialize sampler.
        super(RadFriendsSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.radfriends = RadFriends(self.npdim, 0.)
        self.bounding = 'balls'
        self.method = method

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)

    def update(self, pointvol):
        """Update the N-sphere radii using the current set of live points."""

        # Initialize a K-D Tree to assist nearest neighbor searches.
        kdtree = spatial.KDTree(self.live_u)

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the N-spheres.
        self.radfriends.update(self.live_u, pointvol=pointvol,
                               rstate=self.rstate, bootstrap=self.bootstrap,
                               pool=pool, kdtree=kdtree)
        if self.enlarge != 1.:
            self.radfriends.scale_to_vol(self.radfriends.vol_ball *
                                         self.enlarge)

        return copy.deepcopy(self.radfriends)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the union of N-spheres defined by our live points."""

        # Initialize a K-D Tree to assist nearest neighbor searches.
        kdtree = spatial.KDTree(self.live_u)

        while True:
            # Sample a point `u` from the union of N-spheres along with the
            # number of overlapping spheres `q` at point `u`.
            u, q = self.radfriends.sample(self.live_u, rstate=self.rstate,
                                          return_q=True, kdtree=kdtree)

            # Check if our sample is within the unit cube.
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping balls.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break  # if successful, we're done!

        # Define the axes of the N-sphere.
        ax = np.identity(self.npdim) * self.radfriends.radius

        return u, ax

    def propose_live(self):
        """Propose a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim) * self.radfriends.radius

        return u, ax

    def update_unif(self, blob):
        """Filler function."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        self.scale = blob['scale']
        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc) * self.npdim
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class SupFriendsSampler(Sampler):
    """
    Samples conditioned on the union of (possibly overlapping) N-cubes
    centered on the current set of live points.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.RandomState`
        `~numpy.random.RandomState` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    kwargs : dict, optional
        A dictionary of additional parameters (described below).

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the N-cubes by this fraction. The preferred
        method is to determine this organically using bootstrapping. If
        `bootstrap > 0`, this defaults to `1.0`. If `bootstrap = 0`,
        this instead defaults to `1.25`.

    bootstrap : int, optional
        Compute this many bootstrapped realizations of the bounding
        objects. Use the maximum distance found to the set of points left
        out during each iteration to enlarge the resulting volumes.
        Default is `20` for uniform sampling (`'unif'`) and `0` otherwise.

    walks : int, optional
        For the `'rwalk'` sampling option, the minimum number of steps
        (minimum 2) before proposing a new live point. Default is `25`.

    facc : float, optional
        The target acceptance fraction for the `'rwalk'` sampling option.
        Default is `0.5`. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the `'slice'`, `'rslice'`, and `'hslice'` sampling options, the
        number of times to execute a "slice update" before proposing a new
        live point. Default is `5`. Note that `'slice'` cycles through
        **all dimensions** when executing a "slice update".

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_unif}
        self.update_proposal = self._UPDATE[method]

        # Initialize other arguments.
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'unif':
                self.bootstrap = 20
            else:
                self.bootstrap = 0
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)

        # Initialize sampler.
        super(SupFriendsSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.supfriends = SupFriends(self.npdim, 0.)
        self.bounding = 'cubes'
        self.method = method

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)

    def update(self, pointvol):
        """Update the N-cube side-lengths using the current set of
        live points."""

        # Initialize a K-D Tree to assist nearest neighbor searches.
        kdtree = spatial.KDTree(self.live_u)

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the N-cubes.
        self.supfriends.update(self.live_u, pointvol=pointvol,
                               rstate=self.rstate, bootstrap=self.bootstrap,
                               pool=pool, kdtree=kdtree)
        if self.enlarge != 1.:
            self.supfriends.scale_to_vol(self.supfriends.vol_cube *
                                         self.enlarge)

        return copy.deepcopy(self.supfriends)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the collection of N-cubes defined by our live points."""

        # Initialize a K-D Tree to assist nearest neighbor searches.
        kdtree = spatial.KDTree(self.live_u)

        while True:
            # Sample a point `u` from the union of N-cubes along with the
            # number of overlapping cubes `q` at point `u`.
            u, q = self.supfriends.sample(self.live_u, rstate=self.rstate,
                                          return_q=True, kdtree=kdtree)

            # Check if our point is within the unit cube.
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping cubes.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break  # if successful, we're done!

        # Define the axes of our N-cube.
        ax = np.identity(self.npdim) * self.supfriends.hside

        return u, ax

    def propose_live(self):
        """Return a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim) * self.supfriends.hside

        return u, ax

    def update_unif(self, blob):
        """Filler function."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        self.scale = blob['scale']
        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc) * self.npdim
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale
