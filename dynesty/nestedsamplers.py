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

import math
import numpy as np
import copy

from .sampler import Sampler
from .bounding import (UnitCube, Ellipsoid, MultiEllipsoid,
                       RadFriends, SupFriends)
from .sampling import (sample_unif, sample_rwalk, sample_rstagger,
                       sample_slice, sample_rslice, sample_hslice)
from .utils import unitcheck

__all__ = ["UnitCubeSampler", "SingleEllipsoidSampler",
           "MultiEllipsoidSampler", "RadFriendsSampler", "SupFriendsSampler"]

_SAMPLING = {'unif': sample_unif,
             'rwalk': sample_rwalk,
             'rstagger': sample_rstagger,
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

    method : {`'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

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
        A dictionary of additional parameters.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'rstagger': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'rstagger': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_hslice}
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
        self.nonperiodic = self.kwargs.get('nonperiodic', None)

        # Gradient.
        self.grad = self.kwargs.get('grad', None)
        self.compute_jac = self.kwargs.get('compute_jac', False)

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)
        self.fmove = self.kwargs.get('fmove', 0.9)
        self.max_move = self.kwargs.get('max_move', 100)

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
        self.scale = min(self.scale, math.sqrt(self.npdim))

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        nexpand, ncontract = blob['nexpand'], blob['ncontract']
        self.scale *= nexpand / (2. * ncontract)

    def update_hslice(self, blob):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting."""

        nmove, nreflect = blob['nmove'], blob['nreflect']
        ncontract = blob.get('ncontract', 0)
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)


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

    method : {`'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

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
        A dictionary of additional parameters.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'rstagger': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'rstagger': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_hslice}
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
        self.nonperiodic = self.kwargs.get('nonperiodic', None)

        # Gradient.
        self.grad = self.kwargs.get('grad', None)
        self.compute_jac = self.kwargs.get('compute_jac', False)

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)
        self.fmove = self.kwargs.get('fmove', 0.9)
        self.max_move = self.kwargs.get('max_move', 100)

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
            if unitcheck(u, self.nonperiodic):
                break  # if it is, we're done!

        return u, self.ell.axes

    def propose_live(self):
        """Return a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]

        # Choose axes.
        if self.sampling in ['rwalk', 'rstagger', 'rslice']:
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
        self.scale = min(self.scale, math.sqrt(self.npdim))

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        nexpand, ncontract = blob['nexpand'], blob['ncontract']
        self.scale *= nexpand / (2. * ncontract)

    def update_hslice(self, blob):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting."""

        nmove, nreflect = blob['nmove'], blob['nreflect']
        ncontract = blob.get('ncontract', 0)
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)


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

    method : {`'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

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
        A dictionary of additional parameters.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'rstagger': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'rstagger': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_hslice}
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
        self.nonperiodic = self.kwargs.get('nonperiodic', None)

        # Gradient.
        self.grad = self.kwargs.get('grad', None)
        self.compute_jac = self.kwargs.get('compute_jac', False)

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)
        self.fmove = self.kwargs.get('fmove', 0.9)
        self.max_move = self.kwargs.get('max_move', 100)

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
            if unitcheck(u, self.nonperiodic):
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
        if self.sampling in ['rwalk', 'rstagger', 'rslice']:
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
        self.scale = min(self.scale, math.sqrt(self.npdim))

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        nexpand, ncontract = blob['nexpand'], blob['ncontract']
        self.scale *= nexpand / (2. * ncontract)

    def update_hslice(self, blob):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting."""

        nmove, nreflect = blob['nmove'], blob['nreflect']
        ncontract = blob.get('ncontract', 0)
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)


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

    method : {`'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

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
        A dictionary of additional parameters.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'rstagger': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'rstagger': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_hslice}
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
        self.radfriends = RadFriends(self.npdim)
        self.bounding = 'balls'
        self.method = method
        self.nonperiodic = self.kwargs.get('nonperiodic', None)

        # Gradient.
        self.grad = self.kwargs.get('grad', None)
        self.compute_jac = self.kwargs.get('compute_jac', False)

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)
        self.fmove = self.kwargs.get('fmove', 0.9)
        self.max_move = self.kwargs.get('max_move', 100)

    def update(self, pointvol):
        """Update the N-sphere radii using the current set of live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the N-spheres.
        self.radfriends.update(self.live_u, pointvol=pointvol,
                               rstate=self.rstate, bootstrap=self.bootstrap,
                               pool=pool)
        if self.enlarge != 1.:
            self.radfriends.scale_to_vol(self.radfriends.vol_ball *
                                         self.enlarge)

        return copy.deepcopy(self.radfriends)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the union of N-spheres defined by our live points."""

        while True:
            # Sample a point `u` from the union of N-spheres along with the
            # number of overlapping spheres `q` at point `u`.
            u, q = self.radfriends.sample(self.live_u, rstate=self.rstate,
                                          return_q=True)

            # Check if our sample is within the unit cube.
            if unitcheck(u, self.nonperiodic):
                # Accept the point with probability 1/q to account for
                # overlapping balls.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break  # if successful, we're done!

        # Define the axes of the N-sphere.
        ax = self.radfriends.axes

        return u, ax

    def propose_live(self):
        """Propose a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = self.radfriends.axes

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
        self.scale = min(self.scale, math.sqrt(self.npdim))

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        nexpand, ncontract = blob['nexpand'], blob['ncontract']
        self.scale *= nexpand / (2. * ncontract)

    def update_hslice(self, blob):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting."""

        nmove, nreflect = blob['nmove'], blob['nreflect']
        ncontract = blob.get('ncontract', 0)
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)


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

    method : {`'unif'`, `'rwalk'`, `'rstagger'`,
              `'slice'`, `'rslice'`, `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

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
        A dictionary of additional parameters.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 method, update_interval, first_update, rstate,
                 queue_size, pool, use_pool, kwargs={}):

        # Initialize method to propose a new starting point.
        self._PROPOSE = {'unif': self.propose_unif,
                         'rwalk': self.propose_live,
                         'rstagger': self.propose_live,
                         'slice': self.propose_live,
                         'rslice': self.propose_live,
                         'hslice': self.propose_live}
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {'unif': self.update_unif,
                        'rwalk': self.update_rwalk,
                        'rstagger': self.update_rwalk,
                        'slice': self.update_slice,
                        'rslice': self.update_slice,
                        'hslice': self.update_hslice}
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
        self.supfriends = SupFriends(self.npdim)
        self.bounding = 'cubes'
        self.method = method
        self.nonperiodic = self.kwargs.get('nonperiodic', None)

        # Gradient.
        self.grad = self.kwargs.get('grad', None)
        self.compute_jac = self.kwargs.get('compute_jac', False)

        # Initialize random walk parameters.
        self.walks = max(2, self.kwargs.get('walks', 25))
        self.facc = self.kwargs.get('facc', 0.5)
        self.facc = min(1., max(1. / self.walks, self.facc))

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)
        self.fmove = self.kwargs.get('fmove', 0.9)
        self.max_move = self.kwargs.get('max_move', 100)

    def update(self, pointvol):
        """Update the N-cube side-lengths using the current set of
        live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the N-cubes.
        self.supfriends.update(self.live_u, pointvol=pointvol,
                               rstate=self.rstate, bootstrap=self.bootstrap,
                               pool=pool)
        if self.enlarge != 1.:
            self.supfriends.scale_to_vol(self.supfriends.vol_cube *
                                         self.enlarge)

        return copy.deepcopy(self.supfriends)

    def propose_unif(self):
        """Propose a new live point by sampling *uniformly* within
        the collection of N-cubes defined by our live points."""

        while True:
            # Sample a point `u` from the union of N-cubes along with the
            # number of overlapping cubes `q` at point `u`.
            u, q = self.supfriends.sample(self.live_u, rstate=self.rstate,
                                          return_q=True)

            # Check if our point is within the unit cube.
            if unitcheck(u, self.nonperiodic):
                # Accept the point with probability 1/q to account for
                # overlapping cubes.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break  # if successful, we're done!

        # Define the axes of our N-cube.
        ax = self.supfriends.axes

        return u, ax

    def propose_live(self):
        """Return a live point/axes to be used by other sampling methods."""

        i = self.rstate.randint(self.nlive)
        u = self.live_u[i, :]
        ax = self.supfriends.axes

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
        self.scale = min(self.scale, math.sqrt(self.npdim))

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess."""

        nexpand, ncontract = blob['nexpand'], blob['ncontract']
        self.scale *= nexpand / (2. * ncontract)

    def update_hslice(self, blob):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting."""

        nmove, nreflect = blob['nmove'], blob['nreflect']
        ncontract = blob.get('ncontract', 0)
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)
