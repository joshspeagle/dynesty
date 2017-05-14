#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampler classes for proposing new live points. Includes:

    UnitCubeSampler:
        Samples from the unit N-cube with no constraints.

    SingleEllipsoidSampler:
        Uses a single ellipsoid to bound the set of live points.

    MultiEllipsoidSampler:
        Uses multiple ellipsoids to bound the set of live points.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import numpy as np
import copy
from scipy import optimize as opt

from .sampler import *
from .ellipsoid import *

__all__ = ["UnitCubeSampler", "SingleEllipsoidSampler",
           "MultiEllipsoidSampler"]


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

    method : {'uniform', 'randomwalk', 'randomtrajectory'}
        A chosen method for sampling conditioned on the single ellipsoid.

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
        the unit cube."""

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

    method : {'uniform', 'randomwalk', 'randomtrajectory'}
        A chosen method for sampling conditioned on the single ellipsoid.

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
        the ellipsoid."""

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

    method : {'uniform', 'randomwalk', 'randomtrajectory'}
        A chosen method for sampling conditioned on the collection of
        ellipsoids.

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
        """Propose a new live point by sampling *uniformly*
        the ellipsoid."""

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
