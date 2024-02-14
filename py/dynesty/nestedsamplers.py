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
from .sampling import (sample_rwalk, sample_slice, sample_rslice,
                       sample_hslice, sample_bound_unif)
from .utils import (get_enlarge_bootstrap, save_sampler, restore_sampler)

__all__ = ["SuperSampler"]

_SAMPLING = {
    'unif': sample_bound_unif,
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
                 ndim,
                 live_points,
                 method,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 pool,
                 use_pool,
                 kwargs=None,
                 ncdim=0,
                 blob=False,
                 logvol_init=0,
                 bounding=None):
        # Initialize sampler.
        super().__init__(loglikelihood,
                         prior_transform,
                         ndim,
                         live_points,
                         update_interval,
                         first_update,
                         rstate,
                         queue_size,
                         pool,
                         use_pool,
                         ncdim=ncdim,
                         blob=blob,
                         logvol_init=logvol_init)
        # Initialize method to propose a new starting point.
        self._PROPOSE = {
            'rwalk': self.propose_live,
            'slice': self.propose_live,
            'rslice': self.propose_live,
            'hslice': self.propose_live,
            'user-defined': self.propose_live
        }

        if callable(method):
            _SAMPLING["user-defined"] = method
            method = "user-defined"
        self.propose_point = self._PROPOSE.get(method, self.propose_live)

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
        self.update_proposal = self._UPDATE.get(method, self.update_user)
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
        self.rwalk_history = {'naccept': 0, 'nreject': 0}

        # Initialize slice parameters.
        self.slices = self.kwargs.get('slices', 5)
        self.fmove = self.kwargs.get('fmove', 0.9)
        self.max_move = self.kwargs.get('max_move', 100)
        self.slice_history = {'ncontract': 0, 'nexpand': 0}
        self.hslice_history = {'nmove': 0, 'nreflect': 0, 'ncontract': 0}

        if bounding not in ['none', 'single', 'multi', 'balls', 'cubes']:
            raise Exception('oops')
        self.bounding = bounding
        if bounding == 'none':
            self.bound = UnitCube(self.ncdim)
        elif bounding == 'single':
            self.bound = Ellipsoid(
                np.zeros(self.ncdim) + .5,
                np.identity(self.ncdim) * self.ncdim / 4)
            # this is ellipsoid in the center of the cube that contains
            # the whole cube
        elif bounding == 'multi':
            self.bound = MultiEllipsoid(
                ctrs=[np.zeros(self.ncdim) + .5],
                covs=[np.identity(self.ncdim) * self.ncdim / 4])
            # this is ellipsoid in the center of the cube that contains
            # the whole cube
        elif bounding == 'balls':
            self.bound = RadFriends(self.ncdim)
        elif bounding == 'cubes':
            self.bound = SupFriends(self.ncdim)

    def update_unif(self, blob, update=True):
        """Filler function."""
        pass

    def update_rwalk(self, blob, update=True):
        """Update the random walk proposal scale based on the current
        number of accepted/rejected steps.
        For rwalk the scale is important because it
        determines the speed of diffusion of points.
        I.e. if scale is too large, the proposal efficiency will be very low
        so it's likely that we'll only do one random walk step at the time,
        thus producing very correlated chain.
        The keyword update determines if we are just accumulating the number
        of steps or actually adjusting the scale
        """
        self.scale = blob['scale']
        hist = self.rwalk_history
        hist['naccept'] += blob['accept']
        hist['nreject'] += blob['reject']
        if not update:
            return
        accept, reject = hist['naccept'], hist['nreject']
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
        hist['naccept'] = 0
        hist['nreject'] = 0

    def update_slice(self, blob, update=True):
        """Update the slice proposal scale based on the relative
        size of the slices compared to our initial guess.
        For slice sampling the scale is only 'advisory' in the sense that
        the right scale will just speed up sampling as we'll have to expand
        or contract less. It won't affect the quality of the samples much.
        The keyword update determines if we are just accumulating the number
        of steps or actually adjusting the scale
        """
        # see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4063214/
        # also 2002.06212
        # https://www.tandfonline.com/doi/full/10.1080/10618600.2013.791193
        # and https://github.com/joshspeagle/dynesty/issues/260
        hist = self.slice_history

        hist['nexpand'] += blob['nexpand']
        hist['ncontract'] += blob['ncontract']
        if blob['expansion_warning_set']:
            self.kwargs['slice_doubling'] = True
        if not update:
            return
        nexpand, ncontract = max(hist['nexpand'], 1), hist['ncontract']
        mult = (nexpand * 2. / (nexpand + ncontract))
        # avoid drastic updates to the scale factor limiting to factor
        # of two
        mult = np.clip(mult, 0.5, 2)
        # Remember I can't apply the rule that scale < cube diagonal
        # because scale is multiplied by axes
        self.scale = self.scale * mult
        hist['nexpand'] = 0
        hist['ncontract'] = 0

    def update_hslice(self, blob, update=True):
        """Update the Hamiltonian slice proposal scale based
        on the relative amount of time spent moving vs reflecting.
        The keyword update determines if we are just accumulating the number
        of steps or actually adjusting the scale
        """
        hist = self.hslice_history
        hist['nmove'] += blob['nmove']
        hist['nreflect'] += blob['nreflect']
        hist['ncontract'] += blob.get('ncontract', 0)
        if not update:
            return
        nmove, nreflect = hist['nmove'], hist['nreflect']
        ncontract = hist['ncontract']
        fmove = (1. * nmove) / (nmove + nreflect + ncontract + 2)
        norm = max(self.fmove, 1. - self.fmove)
        self.scale *= math.exp((fmove - self.fmove) / norm)
        hist['nmove'] = 0
        hist['nreflect'] = 0
        hist['ncontract'] = 0

    def update_user(self, blob, update=True):
        """Update the scale based on the user-defined update function."""

        if callable(self.custom_update):
            self.scale = self.custom_update(blob, self.scale, update=update)

    def save(self, fname):
        """
        Save the state of the dynamic sampler in a file

        Parameters
        ----------
        fname: string
            Filename of the save file.

        """
        save_sampler(self, fname)

    @staticmethod
    def restore(fname, pool=None):
        """
        Restore the dynamic sampler from a file.
        It is assumed that the file was created using .save() method
        of DynamicNestedSampler or as a result of checkpointing during
        run_nested()

        Parameters
        ----------
        fname: string
            Filename of the save file.
        pool: object(optional)
            The multiprocessing pool-like object that supports map()
            calls that will be used in the restored object.

        """
        return restore_sampler(fname, pool=pool)

    def propose_live(self, *args):
        """Return a live point/axes to be used by other sampling methods.
           If args is not empty, it contains the subset of indices of points to
           sample from."""

        if len(args) > 0:
            i = self.rstate.choice(args[0])
        else:
            i = self.rstate.integers(self.nlive)
        u = self.live_u[i, :]
        if self.bounding in ['single', 'balls', 'cubes']:
            if self.sampling in ['rwalk', 'rslice', 'slice']:
                ax = self.bound.axes
            else:
                ax = np.identity(self.ncdim)
        elif self.bound == 'multi':
            u_fit = u[:self.ncdim]

            # Automatically trigger an update if we're not in any ellipsoid.
            if not self.bound.contains(u_fit):
                # Update the bounding ellipsoids.
                self.update_bound_if_needed(-np.inf, force=True)
                # Check for ellipsoid overlap (again).
                if not self.bound.contains(u_fit):
                    raise RuntimeError('Update of the ellipsoid failed')

            if self.sampling in ['rwalk', 'rslice', 'slice']:
                # Pick a random ellipsoid (not necessarily the one that
                # contains u)
                # This a crucial step as we must choose a random ellipsoid,
                # rather than the ellipsoid to which this point belongs.
                # because a non-random ellipsoid can break detailed balance
                # see #364
                # here we choose ellipsoid in proportion of its volume
                probs = np.exp(self.bound.logvols - self.bound.logvol_tot)
                ell_idx = rand_choice(probs, self.rstate)
                # Choose axes.
                ax = self.bound.ells[ell_idx].axes
            else:
                ax = np.identity(self.ndim)
        else:
            ax = np.identity(self.ncdim)

        return u, ax

    def update(self, subset=slice(None)):
        """Update the bounds using the current set of
        live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None

        if self.bounding in ['single', 'multi', 'balls', 'cubes']:
            # Update the ellipsoid.
            self.bound.update(self.live_u[subset, :self.ncdim],
                              rstate=self.rstate,
                              bootstrap=self.bootstrap,
                              pool=pool)
        if self.enlarge != 1.:
            if self.bounding in ['single', 'balls', 'cubes']:
                self.bound.scale_to_logvol(self.bound.logvol +
                                           np.log(self.enlarge))
            if self.bounding == 'multi':
                self.bound.scale_to_logvol(self.bound.logvols +
                                           np.log(self.enlarge))

        return copy.deepcopy(self.bound)
