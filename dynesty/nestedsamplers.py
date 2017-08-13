#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampler classes for proposing new live points. Includes:

    UnitCubeSampler:
        Samples from the unit N-cube with no constraints.

    SingleEllipsoidSampler:
        Uses a single ellipsoid to bound new live points proposals.

    MultiEllipsoidSampler:
        Uses multiple ellipsoids to bound new live points proposals.

    RadFriendsSampler:
        Uses an n-sphere of fixed radius centered on each
        live point to bound new live point proposals. Based on the
        p=2 L^p norm (Euclidean norm).

    SupFriendsSampler:
        Uses an n-cube of fixed length centered on each
        live point to bound new live point proposals. Based on the
        p=inf L^p norm (sup norm).

"""

from __future__ import (print_function, division)
from builtins import range

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
             'slice': sample_slice}


class UnitCubeSampler(Sampler):
    """
    Samples with no bounds (i.e. within the entire unit N-cube).

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    method : {'unif', 'rwalk', 'slice'}
        A chosen method for sampling conditioned on the proposal.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags for where a pool should be used to
        execute operations in parallel.

    Other Parameters
    ----------------
    walks : int, optional
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live}
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice}
        self.propose_point = self._PROPOSE[method]
        self.update_proposal = self._UPDATE[method]
        self.sampling, self.evolve_point = method, _SAMPLING[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'slice' or method == 'rwalk':
                self.bootstrap = 0
            else:
                self.bootstrap = 20
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(UnitCubeSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.unitcube = UnitCube(self.npdim)
        self.bound = 'none'
        self.method = method

        # random walk
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # slice
        self.slices = self.kwargs.get('slices', 3)

    def update(self, pointvol):
        """Update the unit cube proposal."""

        return copy.deepcopy(self.unitcube)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly*
        within the unit cube."""

        u = self.unitcube.sample(rstate=self.rstate)
        ax = np.identity(self.npdim)

        return u, ax

    def propose_live(self):
        """Propose a live point/bound to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim)

        return u, ax

    def update_unif(self, blob):
        """Update the uniform proposal."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc)
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of slice compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class SingleEllipsoidSampler(Sampler):
    """
    Bounds live points in a single ellipsoid and samples conditioned
    on the ellipsoid.

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    method : {'unif', 'rwalk', 'slice'}
        A chosen method for sampling conditioned on the proposal.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: pool
        Use this pool of workers to execute operations in parallel.


    use_pool : dict, optional
        A dictionary containing flags for where a pool should be used to
        execute operations in parallel.

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the ellipsoid by this fraction. The preferred
        method is to determine this organically using bootstrapping. If
        `bootstrap > 0`, this defaults to *1.0*. If `bootstrap = 0`,
        this instead defaults to *1.25*.

    bootstrap : int, optional
        Compute this many bootstrap resampled realizations of the bounding
        ellipsoid. Use the maximum distance found to the set of points
        left out during each iteration to enlarge the resulting ellipsoids.
        Default is *20*.

    walks : int, optional
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live}
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice}
        self.propose_point = self._PROPOSE[method]
        self.update_proposal = self._UPDATE[method]
        self.sampling, self.evolve_point = method, _SAMPLING[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'slice' or method == 'rwalk':
                self.bootstrap = 0
            else:
                self.bootstrap = 20
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(SingleEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.ell = Ellipsoid(np.zeros(self.npdim), np.identity(self.npdim))
        self.bound = 'single'
        self.method = method

        # random walk
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # slice
        self.slices = self.kwargs.get('slices', 3)

    def update(self, pointvol):
        """Update bounding ellipsoid using the current set of live points."""

        self._empty_queue()
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None
        self.ell.update(self.live_u, pointvol=pointvol, rstate=self.rstate,
                        bootstrap=self.bootstrap, pool=pool)
        if self.enlarge != 1.:
            self.ell.scale_to_vol(self.ell.vol * self.enlarge)

        return copy.deepcopy(self.ell)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly*
        within the ellipsoid."""

        while True:
            u = self.ell.sample(rstate=self.rstate)
            if self._check_unit_cube(u):
                break

        return u, self.ell.axes

    def propose_live(self):
        """Propose a live point/bound to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        return u, self.ell.axes

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc)
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of slice compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class MultiEllipsoidSampler(Sampler):
    """
    Bounds live points in multiple ellipsoids and samples conditioned
    on their union.

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    method : {'unif', 'rwalk', 'slice'}
        A chosen method for sampling conditioned on the proposal.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: pool
        Use this pool of workers to execute operations in parallel.


    use_pool : dict, optional
        A dictionary containing flags for where a pool should be used to
        execute operations in parallel.

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the ellipsoids by this fraction. The preferred
        method is to determine this organically using bootstrapping. If
        `bootstrap > 0`, this defaults to *1.0*. If `bootstrap = 0`,
        this instead defaults to *1.25*.

    bootstrap : int, optional
        Compute this many bootstrap resampled realizations of the bounding
        ellipsoids. Use the maximum distance found to the set of points
        left out during each iteration to enlarge the resulting ellipsoids.
        Default is *20*.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the proposed split. Default is *0.5*.

    vol_check : float, optional
        The factor used to when checking whether the volume of the original
        bounding ellipsoid is large enough to warrant more trial splits via
        `ell.vol > vol_check * nlive * pointvol`. Default is *2.0*.

    walks : int, optional
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live}
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice}
        self.propose_point = self._PROPOSE[method]
        self.update_proposal = self._UPDATE[method]
        self.sampling, self.evolve_point = method, _SAMPLING[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'slice' or method == 'rwalk':
                self.bootstrap = 0
            else:
                self.bootstrap = 20
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        self.vol_dec = kwargs.get('vol_dec', 0.5)
        self.vol_check = kwargs.get('vol_check', 2.0)
        super(MultiEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.mell = MultiEllipsoid(ctrs=[np.zeros(self.npdim)],
                                   ams=[np.identity(self.npdim)])
        self.bound = 'multi'
        self.method = method

        # random walk
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # slice
        self.slices = self.kwargs.get('slices', 3)

    def update(self, pointvol):
        """Update bounding ellipsoids using the current set of live points."""

        self._empty_queue()
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None
        self.mell.update(self.live_u, pointvol=pointvol,
                         vol_dec=self.vol_dec, vol_check=self.vol_check,
                         rstate=self.rstate, bootstrap=self.bootstrap,
                         pool=pool)
        if self.enlarge != 1.:
            self.mell.scale_to_vols(self.mell.vols * self.enlarge)

        return copy.deepcopy(self.mell)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the ellipsoids."""

        while True:
            u, idx, q = self.mell.sample(rstate=self.rstate, return_q=True)
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping ellipsoids.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break

        return u, self.mell.ells[idx].axes

    def propose_live(self):
        """Propose a live point/bound to be used by other sampling methods."""

        # Copy a random live point.
        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ell_idxs = self.mell.within(u)  # check ellipsoid overlap
        nidx = len(ell_idxs)  # get number of overlapping ellipsoids

        # Automatically trigger an update if we're not in any ellipsoid.
        if nidx == 0:
            expected_vol = math.exp(-self.it / self.nlive)
            pointvol = expected_vol / self.nlive
            prop = self.update(pointvol)
            if self.save_proposals:
                self.prop.append(prop)
            self.nprop += 1
            self.since_update = 0
            ell_idxs = self.mell.within(u)
            nidx = len(ell_idxs)
        ell_idx = ell_idxs[self.rstate.randint(nidx)]  # pick one

        return u, self.mell.ells[ell_idx].axes

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc)
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of slice compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class RadFriendsSampler(Sampler):
    """
    Bounds new live point proposals using n-spheres centered on the
    current set of live points.

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    method : {'unif', 'rwalk', 'slice'}
        A chosen method for sampling conditioned on the proposal.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags for where a pool should be used to
        execute operations in parallel.

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the n-spheres by this factor by scaling the
        associated radius. The preferred method is to set the radius
        organically using bootstrapping. If `bootstrap > 0`, this defaults
        to *1.0*. If `bootstrap = 0`, this instead defaults to *1.25*.

    bootstrap : int, optional
        Compute this many bootstrap resampled realizations when determining
        the radius of the n-sphere used at each iteration. Default is *20*.
        If this is set to *0*, then the radius is constructed using a
        leave-one-out approach (which may be slower depending on the number
        of live points!).

    walks : int, optional
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live}
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice}
        self.propose_point = self._PROPOSE[method]
        self.update_proposal = self._UPDATE[method]
        self.sampling, self.evolve_point = method, _SAMPLING[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'slice' or method == 'rwalk':
                self.bootstrap = 0
            else:
                self.bootstrap = 20
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(RadFriendsSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.radfriends = RadFriends(self.npdim, 0.)
        self.bound = 'balls'
        self.method = method

        # random walk
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # slice
        self.slices = self.kwargs.get('slices', 3)

    def update(self, pointvol):
        """Update proposal radius using the current set of live points."""

        kdtree = spatial.KDTree(self.live_u)

        self._empty_queue()
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None
        self.radfriends.update(self.live_u, pointvol=pointvol,
                               rstate=self.rstate, bootstrap=self.bootstrap,
                               pool=pool, kdtree=kdtree)
        if self.enlarge != 1.:
            self.radfriends.scale_to_vol(self.radfriends.vol_ball *
                                         self.enlarge)

        return copy.deepcopy(self.radfriends)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the collection of n-spheres defined by our live points."""

        kdtree = spatial.KDTree(self.live_u)

        while True:
            u, q = self.radfriends.sample(self.live_u, rstate=self.rstate,
                                          return_q=True, kdtree=kdtree)
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping balls.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        ax = np.identity(self.npdim) * self.radfriends.radius

        return u, ax

    def propose_live(self):
        """Propose a live point/bound to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim) * self.radfriends.radius

        return u, ax

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc)
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of slice compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale


class SupFriendsSampler(Sampler):
    """
    Bounds new live point proposals using n-cubes centered on the
    current set of live points.


    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    method : {'unif', 'rwalk', 'slice'}
        A chosen method for sampling conditioned on the proposal.

    update_interval : int
        Only update the proposal distribution every `update_interval`-th
        likelihood call.

    rstate : `~numpy.random.RandomState`
        RandomState instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated.

    pool: pool
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags for where a pool should be used to
        execute operations in parallel.

    Other Parameters
    ----------------
    enlarge : float, optional
        Enlarge the volumes of the n-cubes by this factor by scaling the
        associated radius. The preferred method is to set the radius
        organically using bootstrapping. If `bootstrap > 0`, this defaults
        to *1.0*. If `bootstrap = 0`, this instead defaults to *1.25*.

    bootstrap : int, optional
        Compute this many bootstrap resampled realizations when determining
        the side-length of the n-cube at each iteration. Default is *20*.
        If this is set to *0*, then the radius is constructed using a
        leave-one-out approach (which may be slower depending on the number
        of live points!).

    walks : int, optional
        For the 'rwalk' sampling option, the minimum number of steps (minimum
        2) to take before proposing a new live point. Default is *25*.

    facc : float, optional
        The target acceptance fraction for the 'rwalk' sampling option.
        Default is *0.5*. Bounded to be between `[1. / walks, 1.]`.

    slices : int, optional
        For the 'slice' sampling option, the number of times to slice through
        **all dimensions** before proposing a new live point. Default is *3*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'slice': self.propose_live}
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'slice': self.update_slice}
        self.propose_point = self._PROPOSE[method]
        self.update_proposal = self._UPDATE[method]
        self.sampling, self.evolve_point = method, _SAMPLING[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap')
        if self.bootstrap is None:
            if method == 'slice' or method == 'rwalk':
                self.bootstrap = 0
            else:
                self.bootstrap = 20
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(SupFriendsSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, first_update,
                             rstate, queue_size, pool, use_pool)
        self.supfriends = SupFriends(self.npdim, 0.)
        self.bound = 'cubes'
        self.method = method

        # random walk
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # slice
        self.slices = self.kwargs.get('slices', 3)

    def update(self, pointvol):
        """Update proposal side-length using the current set of live points."""

        kdtree = spatial.KDTree(self.live_u)

        self._empty_queue()
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None
        self.supfriends.update(self.live_u, pointvol=pointvol,
                               rstate=self.rstate, bootstrap=self.bootstrap,
                               pool=pool, kdtree=kdtree)
        if self.enlarge != 1.:
            self.supfriends.scale_to_vol(self.supfriends.vol_cube *
                                         self.enlarge)

        return copy.deepcopy(self.supfriends)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the collection of n-cubes defined by our live points."""

        kdtree = spatial.KDTree(self.live_u)

        while True:
            u, q = self.supfriends.sample(self.live_u, rstate=self.rstate,
                                          return_q=True, kdtree=kdtree)
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping cubes.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        ax = np.identity(self.npdim) * self.supfriends.hside

        return u, ax

    def propose_live(self):
        """Propose a live point/bound to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim) * self.supfriends.hside

        return u, ax

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        norm = max(self.facc, 1. - self.facc)
        self.scale *= math.exp((facc - self.facc) / norm)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of slice compared to our initial guess."""

        fscale = blob['fscale']
        self.scale *= fscale
