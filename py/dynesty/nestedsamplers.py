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

import math
import copy
import numpy as np

from .sampler import Sampler
from .bounding import (UnitCube, Ellipsoid, MultiEllipsoid, RadFriends,
                       SupFriends, rand_choice)
from .sampling import (sample_unif, sample_rwalk, sample_slice, sample_rslice,
                       sample_hslice)
from .utils import unitcheck, get_enlarge_bootstrap

__all__ = [
    "UnitCubeSampler", "SingleEllipsoidSampler", "MultiEllipsoidSampler",
    "RadFriendsSampler", "SupFriendsSampler"
]

_SAMPLING = {
    'unif': sample_unif,
    'rwalk': sample_rwalk,
    'slice': sample_slice,
    'rslice': sample_rslice,
    'hslice': sample_hslice
}


class SuperSampler(Sampler):
    """
    This is a class that provides common functionality to all the
    implemented samplers
    """

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0):
        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         npdim,
                         live_points,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim)
        # Initialize method to propose a new starting point.
        self._PROPOSE = {
            'unif': self.propose_unif,
            'rwalk': self.propose_live,
            'slice': self.propose_live,
            'rslice': self.propose_live,
            'hslice': self.propose_live,
            'user-defined': self.propose_live
        }

        if callable(method):
            _SAMPLING["user-defined"] = method
            method = "user-defined"
        self.propose_point = self._PROPOSE[method]

        # Initialize method to "evolve" a point to a new position.
        self.sampling, self.evolve_point = method, _SAMPLING[method]

        # Initialize heuristic used to update our sampling method.
        self._UPDATE = {
            'unif': self.update_unif,
            'rwalk': self.update_rwalk,
            'slice': self.update_slice,
            'rslice': self.update_slice,
            'hslice': self.update_hslice,
            'user-defined': self.update_user
        }
        # Initialize other arguments.
        self.scale = 1.

        self.kwargs = kwargs or {}
        # please use self.kwargs below

        self.custom_update = self.kwargs.get('update_func')
        self.update_proposal = self._UPDATE[method]
        self.enlarge, self.bootstrap = get_enlarge_bootstrap(
            method, self.kwargs.get('enlarge'), self.kwargs.get('bootstrap'))

        self.cite = self.kwargs.get('cite')

        self.method = method
        self.nonbounded = self.kwargs.get('nonbounded', None)

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

    def propose_unif(self, *args):
        pass

    def propose_live(self, *args):
        pass

    def update_unif(self, blob):
        """Filler function."""
        pass

    def update_rwalk(self, blob):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps.
        For rwalk the scale is important because it
        determines the speed of diffusion of points.
        I.e. if scale is too large, the proposal efficiency will be very low
        so it's likely that we'll only do one random walk step at the time,
        thus producing very correlated chain.
        """
        self.scale = blob['scale']
        accept, reject = blob['accept'], blob['reject']
        facc = (1. * accept) / (accept + reject)
        # Here we are now trying to solve the Eqn
        # f0 = F(s) where F is the function
        # providing the acceptance rate given logscale
        # and f0 is our target acceptance rate
        # in this case a Newton like update to s
        # is s_{k+1} = s_k - 1/F'(s_k) * (F_k - F_0)
        # We can speculate that F(s)~ C*exp(-Ns)
        # i.e. it's inversely proportional to volume
        # Then F'(s) = -N * F \approx N * F_0
        # Therefore s_{k+1} = s_k + 1/(N*F_0) * (F_k-F0)
        # See also Robbins-Munro recursion which we don't follow
        # here because our coefficients a_k do not obey \sum a_k^2 = \infty
        self.scale *= math.exp((facc - self.facc) / self.ncdim / self.facc)

    def update_slice(self, blob):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess.
        For slice sampling the scale is only 'advisory' in the sense that
        the right scale will just speed up sampling as we'll have to expand
        or contract less. It won't affect the quality of the samples much.
        """
        # see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4063214/
        # also 2002.06212
        # https://www.tandfonline.com/doi/full/10.1080/10618600.2013.791193
        # and https://github.com/joshspeagle/dynesty/issues/260
        nexpand, ncontract = max(blob['nexpand'], 1), blob['ncontract']
        mult = (nexpand * 2. / (nexpand + ncontract))
        # avoid drastic updates to the scale factor limiting to factor
        # of two
        mult = np.clip(mult, 0.5, 2)
        # Remember I can't apply the rule that scale < cube diagonal
        # because scale is multiplied by axes
        self.scale = self.scale * mult

    def update_hslice(self, blob):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting."""

        nmove, nreflect = blob['nmove'], blob['nreflect']
        ncontract = blob.get('ncontract', 0)
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)

    def update_user(self, blob):
        """Update the scale based on the user-defined update function."""

        if callable(self.custom_update):
            self.scale = self.custom_update(blob, self.scale)


class UnitCubeSampler(SuperSampler):
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

    method : {`'unif'`, `'rwalk'`,
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

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

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

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0):

        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         npdim,
                         live_points,
                         method,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim,
                         kwargs=kwargs or {})

        self.unitcube = UnitCube(self.ncdim)
        self.bounding = 'none'

    def update(self):
        """Update the unit cube bound."""

        return copy.deepcopy(self.unitcube)

    def propose_unif(self, *args):
        """Propose a new live point by sampling *uniformly*
        within the unit cube."""

        u = self.unitcube.sample(rstate=self.rstate)
        ax = np.identity(self.npdim)
        if self.npdim != self.ncdim:
            u = np.concatenate(
                [u, self.rstate.uniform(0, 1, self.npdim - self.ncdim)])

        return u, ax

    def propose_live(self, *args):
        """Return a live point/axes to be used by other sampling methods.
           If args is not empty, it contains the subset of indices of points to
           sample from."""

        if len(args) > 0:
            i = self.rstate.choice(args[0])
        else:
            i = self.rstate.integers(self.nlive)
        u = self.live_u[i, :]
        ax = np.identity(self.npdim)

        return u, ax


class SingleEllipsoidSampler(SuperSampler):
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

    method : {`'unif'`, `'rwalk'`, `'slice'`, `'rslice'`,
        `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

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

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0):

        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         npdim,
                         live_points,
                         method,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim,
                         kwargs=kwargs or {})

        self.ell = Ellipsoid(np.zeros(self.ncdim), np.identity(self.ncdim))
        self.bounding = 'single'

    def update(self):
        """Update the bounding ellipsoid using the current set of
        live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the ellipsoid.
        self.ell.update(self.live_u[:, :self.ncdim],
                        rstate=self.rstate,
                        bootstrap=self.bootstrap,
                        pool=pool)
        if self.enlarge != 1.:
            self.ell.scale_to_logvol(self.ell.logvol + np.log(self.enlarge))

        return copy.deepcopy(self.ell)

    def propose_unif(self, *args):
        """Propose a new live point by sampling *uniformly*
        within the ellipsoid."""

        while True:
            # Sample a point from the ellipsoid.
            u = self.ell.sample(rstate=self.rstate)

            # Check if `u` is within the unit cube.
            if unitcheck(u, self.nonbounded):
                break  # if it is, we're done!
        if self.npdim != self.ncdim:
            u = np.concatenate(
                [u, self.rstate.uniform(0, 1, self.npdim - self.ncdim)])
        return u, self.ell.axes

    def propose_live(self, *args):
        """Return a live point/axes to be used by other sampling methods.
           If args is not empty, it contains the subset of indices of points to
           sample from."""
        if len(args) > 0:
            i = self.rstate.choice(args[0])
        else:
            i = self.rstate.integers(self.nlive)
        u = self.live_u[i, :]

        # Choose axes.
        if self.sampling in ['rwalk', 'rslice']:
            ax = self.ell.axes
        elif self.sampling == 'slice':
            ax = self.ell.paxes
        else:
            ax = np.identity(self.ncdim)

        return u, ax


class MultiEllipsoidSampler(SuperSampler):
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

    method : {`'unif'`, `'rwalk'`, `'slice'`, `'rslice'`,
        `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

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

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0):
        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         npdim,
                         live_points,
                         method,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim,
                         kwargs=kwargs or {})

        self.mell = MultiEllipsoid(ctrs=[np.zeros(self.ncdim)],
                                   covs=[np.identity(self.ncdim)])
        self.bounding = 'multi'

    def update(self):
        """Update the bounding ellipsoids using the current set of
        live points."""

        # Check if we should use the pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the bounding ellipsoids.
        self.mell.update(self.live_u[:, :self.ncdim],
                         rstate=self.rstate,
                         bootstrap=self.bootstrap,
                         pool=pool)
        if self.enlarge != 1.:
            self.mell.scale_to_logvol(self.mell.logvols + np.log(self.enlarge))

        return copy.deepcopy(self.mell)

    def propose_unif(self, *args):
        """Propose a new live point by sampling *uniformly* within
        the union of ellipsoids."""

        if self.ncdim != self.npdim and self.nonbounded is not None:
            nonb = self.nonbounded[:self.ncdim]
        else:
            nonb = self.nonbounded
        while True:
            # Sample a point from the union of ellipsoids.
            # Returns the point `u`, ellipsoid index `idx`, and number of
            # overlapping ellipsoids `q` at position `u`.
            u, idx = self.mell.sample(rstate=self.rstate)
            # Check if the point is within the unit cube.
            if unitcheck(u, nonb):
                break  # if successful, we're done!
        if self.ncdim != self.npdim:
            u = np.concatenate(
                [u, self.rstate.uniform(0, 1, self.npdim - self.ncdim)])
        return u, self.mell.ells[idx].axes

    def propose_live(self, *args):
        """Return a live point/axes to be used by other sampling methods.
           If args is not empty, it contains the subset of indices of points to
           sample from."""

        if len(args) > 0:
            i = self.rstate.choice(args[0])
        else:
            i = self.rstate.integers(self.nlive)
        # Copy a random live point.
        u = self.live_u[i, :]
        u_fit = u[:self.ncdim]

        # Automatically trigger an update if we're not in any ellipsoid.
        if not self.mell.contains(u_fit):
            # Update the bounding ellipsoids.
            bound = self.update()
            if self.save_bounds:
                self.bound.append(bound)
            self.nbound += 1
            self.since_update = 0

            # Check for ellipsoid overlap (again).
            if not self.mell.contains(u_fit):
                raise RuntimeError('Update of the ellipsoid failed')

        if self.sampling in ['rwalk', 'rslice', 'slice']:
            # Pick a random ellipsoid (not necessarily the one that contains u)
            # This a crucial step as we must choose a random ellipsoid,
            # rather than the ellipsoid to which this point belongs.
            # because a non-random ellipsoid can break detailed balance
            # see #364
            # here we choose ellipsoid in proportion of its volume
            probs = np.exp(self.mell.logvols - self.mell.logvol_tot)
            ell_idx = rand_choice(probs, self.rstate)
            # Choose axes.
            if self.sampling == 'slice':
                ax = self.mell.ells[ell_idx].paxes
            else:
                ax = self.mell.ells[ell_idx].axes
        else:
            ax = np.identity(self.npdim)

        return u, ax


class RadFriendsSampler(SuperSampler):
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

    method : {`'unif'`, `'rwalk'`, `'slice'`, `'rslice'`,
        `'hslice'`}, optional
        Method used to sample uniformly within the likelihood constraint,
        conditioned on the provided bounds.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

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

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0):

        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         npdim,
                         live_points,
                         method,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim,
                         kwargs=kwargs or {})

        self.radfriends = RadFriends(self.ncdim)
        self.bounding = 'balls'

    def update(self):
        """Update the N-sphere radii using the current set of live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the N-spheres.
        self.radfriends.update(self.live_u[:, :self.ncdim],
                               rstate=self.rstate,
                               bootstrap=self.bootstrap,
                               pool=pool)
        if self.enlarge != 1.:
            self.radfriends.scale_to_logvol(self.radfriends.logvol_ball +
                                            np.log(self.enlarge))

        return copy.deepcopy(self.radfriends)

    def propose_unif(self, *args):
        """Propose a new live point by sampling *uniformly* within
        the union of N-spheres defined by our live points."""

        while True:
            # Sample a point `u` from the union of N-spheres along with the
            # number of overlapping spheres `q` at point `u`.
            u, q = self.radfriends.sample(self.live_u[:, :self.ncdim],
                                          rstate=self.rstate,
                                          return_q=True)

            # Check if our sample is within the unit cube.
            if unitcheck(u, self.nonbounded):
                # Accept the point with probability 1/q to account for
                # overlapping balls.
                if q == 1 or self.rstate.uniform() < 1.0 / q:
                    break  # if successful, we're done!

        # Define the axes of the N-sphere.
        ax = self.radfriends.axes

        u = np.concatenate(
            [u, self.rstate.uniform(0, 1, self.npdim - self.ncdim)])
        return u, ax

    def propose_live(self, *args):
        """Propose a live point/axes to be used by other sampling methods.
           If args is not empty, it contains the subset of indices of points to
           sample from."""

        if len(args) > 0:
            subset = args[0]
            i = self.rstate.choice(subset)
        else:
            i = self.rstate.integers(self.nlive)
        u = self.live_u[i, :]
        ax = self.radfriends.axes

        return u, ax


class SupFriendsSampler(SuperSampler):
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

    method : {`'unif'`, `'rwalk'`,
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

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

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

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0):

        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         npdim,
                         live_points,
                         method,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim,
                         kwargs=kwargs or {})

        self.supfriends = SupFriends(self.ncdim)
        self.bounding = 'cubes'

    def update(self):
        """Update the N-cube side-lengths using the current set of
        live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        # Update the N-cubes.
        self.supfriends.update(self.live_u[:, :self.ncdim],
                               rstate=self.rstate,
                               bootstrap=self.bootstrap,
                               pool=pool)
        if self.enlarge != 1.:
            self.supfriends.scale_to_logvol(self.supfriends.logvol_cube +
                                            np.log(self.enlarge))

        return copy.deepcopy(self.supfriends)

    def propose_unif(self, *args):
        """Propose a new live point by sampling *uniformly* within
        the collection of N-cubes defined by our live points."""

        while True:
            # Sample a point `u` from the union of N-cubes along with the
            # number of overlapping cubes `q` at point `u`.
            u, q = self.supfriends.sample(self.live_u[:, :self.ncdim],
                                          rstate=self.rstate,
                                          return_q=True)

            # Check if our point is within the unit cube.
            if unitcheck(u, self.nonbounded):
                # Accept the point with probability 1/q to account for
                # overlapping cubes.
                if q == 1 or self.rstate.uniform() < 1.0 / q:
                    break  # if successful, we're done!

        # Define the axes of our N-cube.
        ax = self.supfriends.axes
        if self.npdim != self.ncdim:
            u = np.concatenate(
                [u, self.rstate.uniform(0, 1, self.npdim - self.ncdim)])
        return u, ax

    def propose_live(self, *args):
        """Return a live point/axes to be used by other sampling methods.
           If args is not empty, it contains the subset of indices of points to
           sample from."""

        if len(args) > 0:
            i = self.rstate.choice(args[0])
        else:
            i = self.rstate.integers(self.nlive)
        u = self.live_u[i, :]
        ax = self.supfriends.axes

        return u, ax
