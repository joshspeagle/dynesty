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
from scipy import optimize as opt
from numpy import linalg

from .sampler import *
from .ellipsoid import *

__all__ = ["UnitCubeSampler", "SingleEllipsoidSampler",
           "MultiEllipsoidSampler", "RadFriendsSampler", "SupFriendsSampler"]


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

    method : {'uniform', 'randomwalk'}
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

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


    Other Parameters
    ----------------
    walks : int, optional
        For 'randomwalk', the minimum number of steps to take before
        proposing a new live point. Default is *25*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, rstate, queue_size, pool,
                 kwargs={}):
        self._SAMPLE = {'uniform': self.propose_unif,
                        'randomwalk': self.propose_rwalk}
        self._UPDATE = {'uniform': self.update_unif,
                        'randomwalk': self.update_rwalk}
        self.propose_point = self._SAMPLE[method]
        self.update_proposal = self._UPDATE[method]
        self.scale = 1.
        self.kwargs = kwargs
        super(UnitCubeSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)

        # random walk
        self.walks = self.kwargs.get('walks', 25)

    def update(self, pointvol):
        """Filler function since bound does not change."""

        pass

    def propose_unif(self, loglstar):
        """Propose a new live point by sampling *uniformly*
        within the unit cube."""

        u = self.rstate.rand(self.npdim)
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl, 1, None

    def update_unif(self, blob):
        """Filler function since proposal does not change."""

        pass

    def propose_rwalk(self, loglstar):
        """Propose a new live point by starting a *random walk* away
        from an existing live point within the likelihood constraint."""

        # Copy a random live point.
        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Random walk away.
        accept = 0
        reject = 0
        nc = 0
        while nc < self.walks or accept == 0:
            while True:
                du = self.rstate.rand(self.npdim) - 0.5
                u_prop = u + self.scale * du
                if self._check_unit_cube(u_prop):
                    break
                else:
                    reject += 1
            v_prop = self.prior_transform(u_prop)
            logl_prop = self.loglikelihood(v_prop)
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

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        self.scale *= math.exp(2 * facc - 1)


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

    method : {'uniform', 'randomwalk'}
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

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


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
        For 'randomwalk', the minimum number of steps to take before
        proposing a new live point. Default is *25*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, rstate, queue_size, pool,
                 kwargs={}):
        self._SAMPLE = {'uniform': self.propose_unif,
                        'randomwalk': self.propose_rwalk}
        self._UPDATE = {'uniform': self.update_unif,
                        'randomwalk': self.update_rwalk}
        self.propose_point = self._SAMPLE[method]
        self.update_proposal = self._UPDATE[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap', 20)
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(SingleEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)
        self.ell = Ellipsoid(np.zeros(self.npdim), np.identity(self.npdim))

        # random walk
        self.walks = self.kwargs.get('walks', 25)

    def update(self, pointvol):
        """Update bounding ellipsoid using the current set of live points."""

        self._empty_queue()
        self.ell.update(self.live_u, pointvol=pointvol, rstate=self.rstate,
                        bootstrap=self.bootstrap)
        if self.enlarge != 1.:
            self.ell.scale_to_vol(self.ell.vol * self.enlarge)

        return copy.deepcopy(self.ell)

    def propose_unif(self, loglstar):
        """Propose a new live point by sampling *uniformly*
        within the ellipsoid."""

        while True:
            u = self.ell.sample(rstate=self.rstate)
            if self._check_unit_cube(u):
                break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl, 1, None

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def propose_rwalk(self, loglstar):
        """Propose a new live point by starting a *random walk* away
        from an existing live point within the likelihood constraint."""

        # Copy a random live point.
        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Random walk away.
        accept = 0
        reject = 0
        nc = 0
        while nc < self.walks or accept == 0:
            while True:
                du = self.ell.randoffset(rstate=self.rstate)
                u_prop = u + self.scale * du
                if self._check_unit_cube(u_prop):
                    break
                else:
                    reject += 1
            v_prop = self.prior_transform(u_prop)
            logl_prop = self.loglikelihood(v_prop)
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

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        self.scale *= math.exp(2 * facc - 1)


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

    method : {'uniform', 'randomwalk'}
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

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


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
        For 'randomwalk', the minimum number of steps to take before
        proposing a new live point. Default is *25*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, rstate, queue_size, pool,
                 kwargs={}):
        self._SAMPLE = {'uniform': self.propose_unif,
                        'randomwalk': self.propose_rwalk}
        self._UPDATE = {'uniform': self.update_unif,
                        'randomwalk': self.update_rwalk}
        self.propose_point = self._SAMPLE[method]
        self.update_proposal = self._UPDATE[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap', 20)
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        self.vol_dec = kwargs.get('vol_dec', 0.5)
        self.vol_check = kwargs.get('vol_check', 2.0)
        super(MultiEllipsoidSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)
        self.mell = MultiEllipsoid(ctrs=[np.zeros(self.npdim)],
                                   ams=[np.identity(self.npdim)])

        # random walk
        self.walks = self.kwargs.get('walks', 25)

    def update(self, pointvol):
        """Update bounding ellipsoids using the current set of live points."""

        self._empty_queue()
        self.mell.update(self.live_u, pointvol=pointvol,
                         vol_dec=self.vol_dec, vol_check=self.vol_check,
                         rstate=self.rstate, bootstrap=self.bootstrap)
        if self.enlarge != 1.:
            self.mell.scale_to_vols(self.mell.vols * self.enlarge)

        return copy.deepcopy(self.mell)

    def propose_unif(self, loglstar):
        """Propose a new live point by sampling *uniformly* within
        the ellipsoids."""

        while True:
            u, q = self.mell.sample(rstate=self.rstate, return_q=True)
            if self._check_unit_cube(u):
                # Accept the point with probability 1/q to account for
                # overlapping ellipsoids.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl, 1, None

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def propose_rwalk(self, loglstar):
        """Propose a new live point by starting a *random walk* away
        from an existing live point within the likelihood constraint."""

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
                self.prop_iter.append(self.it)
            self.since_update = 0
            ell_idxs = self.mell.within(u)
            nidx = len(ell_idxs)
        ell_idx = ell_idxs[self.rstate.randint(nidx)]  # pick one

        # Random walk away.
        accept = 0
        reject = 0
        nc = 0
        while nc < self.walks or accept == 0:
            while True:
                du = self.mell.ells[ell_idx].randoffset(rstate=self.rstate)
                u_prop = u + self.scale * du
                if self._check_unit_cube(u_prop):
                    break
                else:
                    reject += 1
            v_prop = self.prior_transform(u_prop)
            logl_prop = self.loglikelihood(v_prop)
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

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        self.scale *= math.exp(2 * facc - 1)


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

    method : {'uniform', 'randomwalk'}
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

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


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
        For 'randomwalk', the minimum number of steps to take before
        proposing a new live point. Default is *25*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, rstate, queue_size, pool,
                 kwargs={}):
        self._SAMPLE = {'uniform': self.propose_unif,
                        'randomwalk': self.propose_rwalk}
        self._UPDATE = {'uniform': self.update_unif,
                        'randomwalk': self.update_rwalk}
        self.propose_point = self._SAMPLE[method]
        self.update_proposal = self._UPDATE[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap', 20)
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(RadFriendsSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)
        self.radius = 0.

        # random walk
        self.walks = self.kwargs.get('walks', 25)

    def update(self, pointvol):
        """Update proposal radius using the current set of live points."""

        self._empty_queue()

        # Set initial radius
        self.radius = 0.

        # Bootstrap radius.
        points = self.live_u
        npoints = self.nlive
        for it in range(self.bootstrap):
            idxs = self.rstate.randint(npoints, size=npoints)  # resample
            idx_in = np.unique(idxs)  # selected objects
            sel = np.ones(npoints, dtype='bool')
            sel[idx_in] = False
            idx_out = np.arange(npoints)[sel]  # "missing" objects
            if len(idx_out) < 2:  # edge case
                idx_out = np.append(idx_out, [0, 1])

            # Find largest distance between resampled points and
            # "missing" points.
            points_in, points_out = points[idx_in], points[idx_out]
            radius = max([min([linalg.norm(pin - pout) for pin in points_in])
                          for pout in points_out])

            # Increase radius if needed.
            self.radius = max(self.radius, radius)

        # Construct radius using leave-one-out if no bootstraps used.
        if self.bootstrap == 0.:
            self.radius = max([min([linalg.norm(points[i] - points[j])
                                    for i in range(self.nlive) if i != j])
                               for j in range(self.nlive)])

        # Expand volume (proportional to (cR)^n)
        if self.enlarge != 1.:
            self.radius *= self.enlarge**(1. / self.npdim)

        return copy.deepcopy([self.radius, 2.])

    def propose_unif(self, loglstar):
        """Propose a new live point by sampling *uniformly* within
        the collection of n-spheres defined by our live points."""

        i = self.rstate.randint(self.nlive)
        u_init = self.live_u[i, :]

        while True:
            du = self.radius * randsphere(self.npdim, rstate=self.rstate)
            u = u_init + du  # propose point
            if self._check_unit_cube(u):
                # Find neighbors with r <= R (i.e. number of overlapping
                # n-spheres at the proposed `u`).
                q = sum([linalg.norm(pos - u) <= self.radius
                         for pos in self.live_u])

                # Accept the point with probability 1/q.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl, 1, None

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def propose_rwalk(self, loglstar):
        """Propose a new live point by starting a *random walk* away
        from an existing live point within the likelihood constraint."""

        # Copy a random live point.
        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Random walk away.
        accept = 0
        reject = 0
        nc = 0
        while nc < self.walks or accept == 0:
            while True:
                du = self.radius * randsphere(self.npdim, rstate=self.rstate)
                u_prop = u + self.scale * du
                if self._check_unit_cube(u_prop):
                    break
                else:
                    reject += 1
            v_prop = self.prior_transform(u_prop)
            logl_prop = self.loglikelihood(v_prop)
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

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        self.scale *= math.exp(2 * facc - 1)


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

    method : {'uniform', 'randomwalk'}
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

    pool: ThreadPoolExecutor
        Use this pool of workers to propose live points in parallel.


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
        For 'randomwalk', the minimum number of steps to take before
        proposing a new live point. Default is *25*.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, rstate, queue_size, pool,
                 kwargs={}):
        self._SAMPLE = {'uniform': self.propose_unif,
                        'randomwalk': self.propose_rwalk}
        self._UPDATE = {'uniform': self.update_unif,
                        'randomwalk': self.update_rwalk}
        self.propose_point = self._SAMPLE[method]
        self.update_proposal = self._UPDATE[method]
        self.kwargs = kwargs
        self.scale = 1.
        self.bootstrap = kwargs.get('bootstrap', 20)
        if self.bootstrap > 0:
            self.enlarge = kwargs.get('enlarge', 1.0)
        else:
            self.enlarge = kwargs.get('enlarge', 1.25)
        super(SupFriendsSampler,
              self).__init__(loglikelihood, prior_transform, npdim,
                             live_points, update_interval, rstate,
                             queue_size, pool)
        self.side = 0.

        # random walk
        self.walks = self.kwargs.get('walks', 25)

    def update(self, pointvol):
        """Update proposal side-length using the current set of live points."""

        self._empty_queue()

        # Set initial side-length
        self.side = 0.

        # Bootstrap side-length.
        points = self.live_u
        npoints = self.nlive
        for it in range(self.bootstrap):
            idxs = self.rstate.randint(npoints, size=npoints)  # resample
            idx_in = np.unique(idxs)  # selected objects
            sel = np.ones(npoints, dtype='bool')
            sel[idx_in] = False
            idx_out = np.arange(npoints)[sel]  # "missing" objects
            if len(idx_out) < 2:  # edge case
                idx_out = np.append(idx_out, [0, 1])

            # Find largest distance between resampled points and
            # "missing" points.
            points_in, points_out = points[idx_in], points[idx_out]
            side = max([min([max(pin - pout) for pin in points_in])
                        for pout in points_out])

            # Increase radius if needed.
            self.side = max(self.side, side)

        # Construct radius using leave-one-out if no bootstraps used.
        if self.bootstrap == 0.:
            self.side = max([min([max(points[i] - points[j])
                                  for i in range(self.nlive) if i != j])
                             for j in range(self.nlive)])

        # Expand volume (proportional to (cL)^n)
        if self.enlarge != 1.:
            self.side *= self.enlarge**(1. / self.npdim)

        return copy.deepcopy([self.side, np.inf])

    def propose_unif(self, loglstar):
        """Propose a new live point by sampling *uniformly* within
        the collection of n-cubes defined by our live points."""

        i = self.rstate.randint(self.nlive)
        u_init = self.live_u[i, :]

        while True:
            du = self.side * (self.rstate.rand(self.npdim) - 0.5)
            u = u_init + du  # propose point
            if self._check_unit_cube(u):
                # Find neighbors with l <= L (i.e. number of overlapping
                # n-cubes at the proposed `u`).
                q = sum([max(pos - u) <= self.side
                         for pos in self.live_u])

                # Accept the point with probability 1/q.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        v = self.prior_transform(u)
        logl = self.loglikelihood(v)

        return u, v, logl, 1, None

    def update_unif(self, blob):
        """Update our uniform proposal."""

        pass

    def propose_rwalk(self, loglstar):
        """Propose a new live point by starting a *random walk* away
        from an existing live point within the likelihood constraint."""

        # Copy a random live point.
        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Random walk away.
        accept = 0
        reject = 0
        nc = 0
        while nc < self.walks or accept == 0:
            while True:
                du = self.side * (self.rstate.rand(self.npdim) - 0.5)
                u_prop = u + self.scale * du
                if self._check_unit_cube(u_prop):
                    break
                else:
                    reject += 1
            v_prop = self.prior_transform(u_prop)
            logl_prop = self.loglikelihood(v_prop)
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

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps."""

        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        self.scale *= math.exp(2 * facc - 1)
