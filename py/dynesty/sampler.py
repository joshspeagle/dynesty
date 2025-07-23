#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The base `Sampler` class containing various helpful functions. All other
samplers inherit this class either explicitly or implicitly.

"""

import sys
import warnings
import math
import copy
import numpy as np
from .results import Results, print_fn
from .sampling import UnitCubeSampler
from .utils import (get_seed_sequence, get_print_func, progress_integration,
                    IteratorResult, RunRecord, get_neff_from_logwt,
                    compute_integrals, DelayTimer, _LOWL_VAL,
                    get_random_generator)

from .bounding import (UnitCube, Ellipsoid, MultiEllipsoid, RadFriends,
                       SupFriends, Bound, BOUND_LIST)
from .utils import (save_sampler, restore_sampler)

__all__ = ["Sampler"]


def _get_bound(bounding, ndim):
    if isinstance(bounding, str):
        if bounding not in BOUND_LIST:
            raise ValueError('Unsupported bounding type')
    elif isinstance(bounding, Bound):
        pass
    else:
        raise ValueError('Unsupported bounding type')

    if bounding == 'none':
        bound = UnitCube(ndim)
    elif bounding == 'single':
        bound = Ellipsoid(ndim)
        # this is ellipsoid in the center of the cube that contains
        # the whole cube
    elif bounding == 'multi':
        bound = MultiEllipsoid(ndim)
        # this is ellipsoid in the center of the cube that contains
        # the whole cube
    elif bounding == 'balls':
        bound = RadFriends(ndim)
    elif bounding == 'cubes':
        bound = SupFriends(ndim)
    else:
        bound = bounding
    return bound


def _initialize_live_points(live_points,
                            prior_transform,
                            loglikelihood,
                            mapper,
                            nlive=None,
                            ndim=None,
                            rstate=None,
                            blob=False,
                            use_pool_ptform=None):
    """
    Initialize the first set of live points before starting the sampling

    Parameters
    ----------
    live_points: tuple of arrays or None
        This can be either none or
        tuple of 3 arrays (u, v, logl) or
        tuple of 4 arrays (u, v, logl, blobs), i.e.
        points location in cube coordinates,
        point slocation in original coordinates,
        logl values and optionally blobs associated

    prior_transform: function

    log_likelihood: function

    mapper: function
        The function supporting parallel calls like mapper(func, list)

    nlive: int
        Number of live-points

    ndim: int
        Number of dimensions

    rstate: :class: numpy.random.RandomGenerator

    blob: bool
        If true we also keep track of blobs returned by likelihood

    use_pool_ptform: bool or None
        The flag to perform prior transform using multiprocessing pool or not

    Returns
    -------
    (live_u, live_v, live_logl, blobs), logvol_init, ncalls : tuple
        The first tuple consist of:
        live_u Unit cube coordinates of points
        live_v Original coordinates.
        live_logl log-likelihood values of points
        blobs - Array of blobs associated with logl calls (or None)
        The other arguments are
        logvol_init Log(volume) associated with returned points.
               It will be zero, if all the log(l) values were finite
        ncalls Integer number of function calls
    """
    logvol_init = 0
    ncalls = 0
    if live_points is None:
        # If no live points are provided, propose them by randomly
        # sampling from the unit cube.
        n_attempts = 1000

        min_npoints = min(nlive, max(ndim + 1, min(nlive - 20, 100)))
        # the minimum number points we want with finite logl
        # we want want at least ndim+1, because we want
        # to be able to constraint the ellipsoid
        # Note that if nlive <ndim+ 1 this doesn't really make sense
        # but we should have warned the user earlier, so they are on their own
        # And the reason we have max(ndim+1, X ) is that we'd like to get at
        # least X points as otherwise the poisson estimate of the volume will
        # be too large.
        # The reason why X is min(nlive-20, 100) is that we want at least 100
        # to have reasonable volume accuracy of ~ 10%
        # and the reason for nlive-20 is because if nlive is 100, we don't want
        # all points with finite logl, because this leads to issues with
        # integrals and batch sampling in plateau edge tests
        # The formula probably should be simplified

        live_u = np.zeros((nlive, ndim))
        live_v = np.zeros((nlive, ndim))
        live_logl = np.zeros(nlive)
        ngoods = 0  # counter for how many finite logl we have found
        live_blobs = []
        iattempt = 0
        while True:
            iattempt += 1

            # simulate nlive points by uniform sampling
            cur_live_u = rstate.random(size=(nlive, ndim))
            if use_pool_ptform:
                cur_live_v = mapper(prior_transform, np.asarray(cur_live_u))
            else:
                cur_live_v = map(prior_transform, np.asarray(cur_live_u))
            cur_live_v = np.array(list(cur_live_v))
            cur_live_logl = loglikelihood.map(np.asarray(cur_live_v))
            if blob:
                cur_live_blobs = np.array([_.blob for _ in cur_live_logl])
            cur_live_logl = np.array([_.val for _ in cur_live_logl])
            ncalls += nlive

            # Convert all `-np.inf` log-likelihoods to finite large
            # numbers. Necessary to keep estimators in our sampler from
            # breaking.
            finite = np.isfinite(cur_live_logl)
            not_finite = ~finite
            neg_infinite = np.isneginf(cur_live_logl)
            if np.any(not_finite & (~neg_infinite)):
                raise ValueError("The log-likelihood of live "
                                 "point is invalid.")
            cur_live_logl[not_finite] = _LOWL_VAL

            # how many finite logl values we have
            cur_ngood = finite.sum()
            if cur_ngood > 0:
                # append them to our list
                nextra = min(nlive - ngoods, cur_ngood)
                assert nextra >= 0
                cur_ind = np.nonzero(finite)[0][:nextra]
                live_logl[ngoods:ngoods + nextra] = cur_live_logl[cur_ind]
                live_u[ngoods:ngoods + nextra] = cur_live_u[cur_ind]
                live_v[ngoods:ngoods + nextra] = cur_live_v[cur_ind]
                if blob:
                    live_blobs.extend(cur_live_blobs[cur_ind])
                ngoods += nextra

            # Check if we have more than the minimum required number
            # after that we will stop
            if ngoods >= min_npoints:
                # we need to fill the rest with points with
                # not finite logl
                nextra = nlive - ngoods
                if nextra > 0:
                    cur_ind = np.nonzero(not_finite)[0][:nextra]
                    assert len(cur_ind) == nextra
                    live_logl[ngoods:ngoods + nextra] = cur_live_logl[cur_ind]
                    live_u[ngoods:ngoods + nextra] = cur_live_u[cur_ind]
                    live_v[ngoods:ngoods + nextra] = cur_live_v[cur_ind]
                    if blob:
                        live_blobs.extend(cur_live_blobs[cur_ind])
                logvol_init = -np.log(iattempt)
                # The logic is the following:
                # if we have n live points and we sampled N attempts
                # and we have k points above LOWL_VAL
                # then the volume associated with pts above LOWL_VAL
                # can be estimated as k/(Nn)
                # the rest of the points have 1/Nn volume per pt
                # Since we quit with k points above LOWL_VAL and
                # (n-k)  LOWL points
                # The volume is k/(Nn) + (n-k)/(Nn) = 1/N
                break
            if iattempt == n_attempts:
                if ngoods == 0:
                    # If we found nothing after many attempts, raise the alarm.
                    raise RuntimeError(
                        f"After {n_attempts} attempts, we could not "
                        "find a single point "
                        "that have a valid log-likelihood! Please "
                        "check your prior transform and/or "
                        "log-likelihood.")
                else:
                    # If we found nothing after many attempts, raise the alarm.
                    warnings.warn(f"After {n_attempts} attempts, we cound not "
                                  f"find at least {min_npoints} points "
                                  "that have a valid log-likelihood! "
                                  "The initial sampling is very inefficient!")

    else:
        # If live points were provided, convert the log-likelihoods and
        # then run a quick safety check.
        live_u, live_v, live_logl = live_points[:3]
        if blob:
            live_blobs = live_points[4]
        live_logl = np.asarray(live_logl)
        for i, logl in enumerate(live_logl):
            if not np.isfinite(logl):
                if np.sign(logl) < 0:
                    live_logl[i] = _LOWL_VAL
                else:
                    raise ValueError("The log-likelihood ({0}) of live "
                                     "point {1} located at u={2} v={3} "
                                     " is invalid.".format(
                                         logl, i, live_u[i], live_v[i]))
        if np.all(live_logl == _LOWL_VAL):
            raise ValueError("Not a single provided live point has a "
                             "valid log-likelihood!")
    if np.ptp(live_logl) == 0:
        warnings.warn(
            'All the initial likelihood values are the same. '
            'You likely have a plateau in the likelihood. '
            'Nested sampling may not be the best sampler in this case.',
            RuntimeWarning)
    if not blob:
        live_blobs = None
    return (live_u, live_v, live_logl, live_blobs), logvol_init, ncalls


class Sampler:
    """
    The basic sampler object that performs the actual nested sampling.
    It is used for the initial sampling and also for each batch.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    ndim : int, optional
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 or 4 `~numpy.ndarray`
        Each with shape (nlive, ndim) for the first three arrays.
        If `blob=True`, a fourth array of blobs (arbitrary shape) may be
        included.

    sampling : {`'unif'`, `'rwalk'`, `'slice'`, `'rslice'`}
        Sampling Method used to sample uniformly within the likelihood
        constraint, conditioned on the provided bounds.

    bound_update_interval : int
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

    ncdim: int, optional
        The number of clustering dimensions. The first ncdim dimensions
        will be sampled using the sampling method, the remaining
        dimensions will
        just sample uniformly from the prior distribution.
        If this is `None` (default), this will default to ndim.

    logvol_init: float, optional
        The initial log of volume when starting sampling. This is relevant
        when the log(L) is finite only within a fraction of prior volume.

    """

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 ndim,
                 live_points,
                 sampling,
                 bounding,
                 ncdim=None,
                 rstate=None,
                 pool=None,
                 use_pool=None,
                 queue_size=None,
                 bound_update_interval=None,
                 first_bound_update=None,
                 bound_bootstrap=None,
                 bound_enlarge=None,
                 blob=False,
                 cite=None,
                 logvol_init=0):

        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.ncdim = ncdim or ndim
        self.blob = blob
        # live points
        self.live_u, self.live_v, self.live_logl = live_points[:3]
        if blob:
            self.live_blobs = live_points[4]
        else:
            self.live_blobs = None
        self.nlive = len(self.live_u)
        self.live_bound = np.zeros(self.nlive, dtype=int)
        self.live_it = np.zeros(self.nlive, dtype=int)

        # random state
        self.rstate = rstate or get_random_generator()
        self.sampling = sampling
        # This is the sampler that will be used to sample after we
        # are done with the unit cube sampling
        self.internal_sampler_next = sampling
        self.internal_sampler = UnitCubeSampler(ndim=ndim)

        # parallelism
        self.pool = pool  # provided pool
        if self.pool is None:
            self.mapper = map
        else:
            self.mapper = pool.map
        self.use_pool = use_pool or {
        }  # provided flags for when to use the pool
        self.use_pool_ptform = use_pool.get('prior_transform', True)
        self.use_pool_logl = use_pool.get('loglikelihood', True)
        self.use_pool_evolve = use_pool.get('propose_point', True)
        self.use_pool_update = use_pool.get('update_bound', True)

        if self.use_pool_evolve:
            self.queue_size = queue_size  # size of the queue
        else:
            self.queue_size = 1
        self.queue = []  # proposed live point queue
        self.nqueue = 0  # current size of the queue

        # sampling
        self.it = 1  # current iteration
        self.ncall = self.nlive  # number of function calls
        self.dlv = math.log((self.nlive + 1.) / self.nlive)  # shrinkage/iter
        self.added_live = False  # whether leftover live points were used
        self.eff = 0.  # overall sampling efficiency
        self.cite = ''  # Default empty
        self.save_bounds = True

        # bounding updates
        self.bound_update_interval = bound_update_interval
        self.first_bound_update = first_bound_update
        self.first_bound_update_ncall = first_bound_update.get(
            'min_ncall', 2 * self.nlive)
        self.first_bound_update_eff = first_bound_update.get('min_eff', 10.)
        self.logl_first_update = None
        self.ncall_at_last_update = 0

        self.unit_cube_sampling = True
        self.bound = UnitCube(self.ncdim)
        self.bound_list = [self.bound]  # bounding distributions
        self.nbound = 1  # total number of unique bounding distributions

        self.logvol_init = logvol_init

        self.plateau_mode = False
        self.plateau_counter = None
        self.plateau_logdvol = None

        # results
        self.saved_run = RunRecord()

        # self.cite = self.kwargs.get('cite')
        self.bound_bootstrap = bound_bootstrap
        self.bound_enlarge = bound_enlarge

        self.bounding = bounding
        self.bound_next = _get_bound(bounding, ndim)
        # the reason I do not set it as self.bound
        # because we start from unit cube

        self.cite = cite

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
        ax = self.bound.get_random_axes(self.rstate)
        u_fit = u[:self.ncdim]
        if self.bound.need_centers:
            self.bound.ctrs = self.live_u
        # Automatically trigger an update if we're not in any ellipsoid.
        if not self.bound.contains(u_fit):
            # Update the bounding ellipsoids.
            self.update_bound_if_needed(-np.inf, force=True)
            # Check for ellipsoid overlap (again).
            if not self.bound.contains(u_fit):
                raise RuntimeError('Update of the ellipsoid failed')

        return u, ax

    def update_bound(self, subset=slice(None)):
        """Update the bounds using the current set of
        live points."""

        # Check if we should use the provided pool for updating.
        if self.use_pool_update:
            pool = self.pool
        else:
            pool = None
        self.bound.update(self.live_u[subset, :self.ncdim],
                          rstate=self.rstate,
                          bootstrap=self.bound_bootstrap,
                          pool=pool)
        if self.bound_enlarge != 1.:
            self.bound.scale_to_logvol(self.bound.logvol +
                                       np.log(self.bound_enlarge))

        return copy.deepcopy(self.bound)

    def __setstate__(self, state):
        self.__dict__ = state
        self.pool = None
        self.mapper = map

    def __getstate__(self):
        """Get state information for pickling."""

        state = self.__dict__.copy()
        for k in ['mapper', 'pool']:
            if k in state:
                del state[k]
        return state

    def reset(self):
        """Re-initialize the sampler."""

        # (self.live_u, self.live_v, self.live_logl, self.live_blobs)
        live_points, logvol_init, init_ncalls = _initialize_live_points(
            None,
            self.prior_transform,
            self.loglikelihood,
            self.mapper,
            nlive=self.nlive,
            ndim=self.ndim,
            rstate=self.rstate,
            blob=self.blob,
            use_pool_ptform=self.use_pool_ptform)

        self.__init__(self.loglikelihood,
                      self.prior_transform,
                      self.ndim,
                      live_points,
                      self.sampling,
                      self.bounding,
                      ncdim=self.ncdim,
                      rstate=self.rstate,
                      pool=self.pool,
                      use_pool=self.use_pool,
                      queue_size=self.queue_size,
                      bound_update_interval=self.bound_update_interval,
                      first_bound_update=self.first_bound_update,
                      bound_bootstrap=self.bound_bootstrap,
                      bound_enlarge=self.bound_enlarge,
                      blob=self.blob,
                      cite=self.cite,
                      logvol_init=logvol_init)

    @property
    def results(self):
        """Saved results from the nested sampling run. If bounding
        distributions were saved, those are also returned."""

        d = {}
        for k in [
                'nc', 'v', 'id', 'it', 'u', 'logwt', 'logl', 'logvol', 'logz',
                'logzvar', 'h', 'blob', 'proposal_stats'
        ]:
            d[k] = np.array(self.saved_run[k])

        # Add all saved samples to the results.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = [('nlive', self.nlive), ('niter', self.it - 1),
                       ('ncall', d['nc']), ('eff', self.eff),
                       ('samples', d['v']), ('blob', d['blob'])]
            for k in ['id', 'it', 'u']:
                results.append(('samples_' + k, d[k]))
            for k in ['logwt', 'logl', 'logvol', 'logz']:
                results.append((k, d[k]))
            results.append(('logzerr', np.sqrt(d['logzvar'])))
            results.append(('information', d['h']))

        # Add any saved bounds (and ancillary quantities) to the results.
        if self.save_bounds:
            results.append(('bound', copy.deepcopy(self.bound_list)))
            results.append(
                ('bound_iter', np.array(self.saved_run['bounditer'],
                                        dtype=int)))
            results.append(('samples_bound',
                            np.array(self.saved_run['boundidx'], dtype=int)))
            results.append(('scale', np.array(self.saved_run['scale'])))

        return Results(results)

    @property
    def n_effective(self):
        """
        Estimate the effective number of posterior samples using the Kish
        Effective Sample Size (ESS) where `ESS = sum(wts)^2 / sum(wts^2)`.
        Note that this is `len(wts)` when `wts` are uniform and
        `1` if there is only one non-zero element in `wts`.

        """
        logwt = self.saved_run['logwt']
        if len(logwt) == 0 or np.isneginf(np.max(logwt)):
            # If there are no saved weights, or its -inf return 0.
            return 0
        else:
            return get_neff_from_logwt(np.asarray(logwt))

    @property
    def citations(self):
        """
        Return list of papers that should be cited given the specified
        configuration of the sampler.

        """

        return self.cite

    def update_bound_if_needed(self, loglstar, ncall=None, force=False):
        """
        Here we update the bound depending on the situation
        The arguments are the loglstar and number of calls
        if force is true we update the bound no matter what
        """
        if ncall is None:
            ncall = self.ncall
        if self.bound_update_interval is None:
            delta_bound = self.sampler.bound_update_interval * self.nlive
        else:
            delta_bound = self.bound_update_interval

        call_check_first = (ncall >= self.first_bound_update_ncall)
        call_check = (ncall >= delta_bound + self.ncall_at_last_update)
        efficiency_check = (self.eff < self.first_bound_update_eff)
        # there are three cases when we update the bound
        # * if we are still using uniform cube sampling and both efficiency is
        # lower than the threshold and the number of calls is larger than the
        # threshold
        # * if we are sampling from uniform cube and loglstar is larger than
        # the previously saved logl_first_update
        # * if we are not uniformly cube sampling and the ncall is larger
        # than the ncall of the previous update by the update_interval
        # * we are forced
        if ((self.unit_cube_sampling and efficiency_check and call_check_first)
                or (not self.unit_cube_sampling and call_check) or
            (self.unit_cube_sampling and self.logl_first_update is not None
             and loglstar > self.logl_first_update)) or force:
            if loglstar == _LOWL_VAL:
                # in the case we just started and we have some
                # LOWL_VAL points we don't want to use them for the
                # boundary
                subset = self.live_logl > loglstar
            else:
                subset = slice(None)
            if self.unit_cube_sampling:
                # done with unit cube
                # updating the bound and internal sampler
                self.unit_cube_sampling = False
                self.logl_first_update = loglstar
                self.bound = self.bound_next
                self.internal_sampler = self.internal_sampler_next
                # self.bound_next = None
                # self.internal_sampler_next = None
            self.update_bound(subset=subset)
            if self.save_bounds:
                self.bound_list.append(self.bound)
            self.nbound += 1
            self.ncall_at_last_update = ncall

    def _fill_queue(self, loglstar):
        """Sequentially add new live point proposals to the queue."""

        args = (np.nonzero(self.live_logl > loglstar)[0], )
        if len(args[0]) == 0:
            raise RuntimeError(
                'No live points are above loglstar. '
                'Do you have a likelihood plateau ? '
                'It is also possible that you are trying to sample '
                'excessively around the very peak of the posterior')

        point_queue = []
        axes_queue = []
        # Propose points using the provided sampling/bounding options.
        while self.nqueue < self.queue_size:
            point, axes = self.propose_live(*args)
            # these points are wasted for UniformBoundSampler
            point_queue.append(point)
            axes_queue.append(axes)
            self.nqueue += 1
        if self.queue_size > 1:
            seeds = get_seed_sequence(self.rstate, self.queue_size)
        else:
            seeds = [self.rstate]

        if self.use_pool_evolve:
            # Use the pool to propose ("evolve") a new live point.
            mapper = self.mapper
        else:
            # Propose ("evolve") a new live point using the default `map`
            # function.
            mapper = map

        args = self.internal_sampler.prepare_sampler(
            loglstar=loglstar,
            points=point_queue,
            axes=axes_queue,
            seeds=seeds,
            prior_transform=self.prior_transform,
            loglikelihood=self.loglikelihood,
            nested_sampler=self)
        self.queue = list(mapper(self.internal_sampler.sample, args))

    def _get_point_value(self, loglstar):
        """Grab the first live point proposal in the queue."""

        # If the queue is empty, refill it.
        if self.nqueue <= 0:
            self._fill_queue(loglstar)

        # Grab the earliest entry.
        ret = self.queue.pop(0)
        self.nqueue -= 1

        return ret

    def _new_point(self, loglstar):
        """Propose points until a new point that satisfies the log-likelihood
        constraint `loglstar` is found."""

        ncall = self.ncall
        # this is a global counter
        # we do not update directly the counter inside the class
        ncall_accum = 0
        sampling_history = []
        while True:
            # Get the next point from the queue
            ret = self._get_point_value(loglstar)
            logl = ret.logl
            cur_ncalls = ret.ncalls
            ncall_accum += cur_ncalls
            ncall += cur_ncalls
            u, v = ret.u, ret.v
            tuning_info = ret.tuning_info
            sampling_history.extend(ret.sampling_history)

            if tuning_info is not None and not self.unit_cube_sampling:
                # If our queue is empty, update any tuning parameters
                # associated
                # with our proposal (sampling) method.
                # If it's not empty we are just accumulating the
                # the history of evaluations
                self.internal_sampler.tune(tuning_info,
                                           update=self.nqueue <= 0)

            # the reason I'm not using self.ncall is that it's updated at
            # higher level
            # also on purpose this is placed in nqueue==0
            # because we only want update if we are planning to generate
            # new points
            if self.nqueue == 0:
                self.update_bound_if_needed(loglstar, ncall=ncall)

            # If we satisfy the log-likelihood constraint, we're done!
            if logl > loglstar:
                break

        return u, v, logl, ncall_accum, ret.proposal_stats

    def add_live_points(self):
        """Add the remaining set of live points to the current set of dead
        points. Instantiates a generator that will be called by
        the user. Returns the same outputs as :meth:`sample`."""

        # Check if the remaining live points have already been added
        # to the output set of samples.
        if self.added_live:
            raise ValueError("The remaining live points have already "
                             "been added to the list of samples!")
        else:
            self.added_live = True
        if len(self.saved_run['logz']) > 0:
            logz = self.saved_run['logz'][-1]
            logzvar = self.saved_run['logzvar'][-1]
            h = self.saved_run['h'][-1]
            loglstar = self.saved_run['logl'][-1]
            logvol = self.saved_run['logvol'][-1]
        else:
            # this is special case if we didn't do any running
            # just sampled uniformly and bailed out
            h = 0.  # information, initially *0.*
            logz = -1.e300  # ln(evidence), initially *0.*
            logzvar = 0.  # var[ln(evidence)], initially *0.*
            logvol = self.logvol_init
            # initially contains the whole prior (volume=1.)
            loglstar = -1.e300  # initial ln(likelihood)

        # After N samples have been taken out, the remaining volume is
        # `e^(-N / nlive)`. The remaining points are distributed uniformly
        # within the remaining volume so that the expected volume enclosed
        # by the `i`-th worst likelihood is
        # `e^(-N / nlive) * (nlive + 1 - i) / (nlive + 1)`.
        # The tricky bit here is what to do if we have a plateau that we
        # haven't fully exhausted
        # then we first use the old delta(V) till we are done with the plateau
        if not self.plateau_mode:
            logvols = np.log(1. - (np.arange(self.nlive) + 1.) /
                             (self.nlive + 1.))
            # Defining change in `logvol` used in `logzvar` approximation.
        else:
            # we first just use old delta(v)'s associated with each point
            # in the plateau
            logvols = np.log1p(-((1 + np.arange(self.plateau_counter)) *
                                 np.exp(self.plateau_logdvol - logvol)))
            # after we're done with it we just assign 1/(nrest+1) fraction of
            # the remaining volume to each leftover point
            nrest = self.nlive - self.plateau_counter
            logvols = np.concatenate([
                logvols,
                logvols[-1] + np.log1p(-(1 + np.arange(nrest)) / (nrest + 1))
            ])
        # IMPORTANT in those caclulations I keep logvol separate
        # and add it later to ensure the first dlv=0
        dlvs = -np.diff(logvols, prepend=0)
        logvols += logvol
        # Sorting remaining live points.
        lsort_idx = np.argsort(self.live_logl)
        loglmax = max(self.live_logl)

        # Grabbing relevant values from the last dead point.
        if not self.unit_cube_sampling:
            bounditer = self.nbound - 1
        else:
            bounditer = 0

        # Add contributions from the remaining live points in order
        # from the lowest to the highest log-likelihoods.
        for i in range(self.nlive):

            # Grab live point with `i`-th lowest log-likelihood along with
            # ancillary quantities.
            idx = lsort_idx[i]
            logvol, dlv = logvols[i], dlvs[i]
            # we are doing copies here, because live_u/live_v are
            # updated in place
            ustar = self.live_u[idx].copy()
            vstar = self.live_v[idx].copy()
            if self.blob:
                old_blob = self.live_blobs[idx].copy()
            else:
                old_blob = None
            loglstar_new = self.live_logl[idx]
            boundidx = self.live_bound[idx]
            point_it = self.live_it[idx]

            (logwt, logz, logzvar,
             h) = progress_integration(loglstar, loglstar_new, logz, logzvar,
                                       logvol, dlv, h)
            loglstar = loglstar_new
            delta_logz = np.logaddexp(0, loglmax + logvol - logz)

            # Save results.
            self.saved_run.append(
                dict(
                    id=idx,
                    u=ustar,
                    v=vstar,
                    logl=loglstar,
                    logvol=logvol,
                    logwt=logwt,
                    logz=logz,
                    logzvar=logzvar,
                    h=h,
                    nc=1,  # this is technically a lie
                    # as we didn't call the likelihood even once
                    # however because we lose track of ncs if we start
                    # from points that are not sampled from unit cube
                    # it can lead to sum(nc)!=ncall
                    boundidx=boundidx,
                    it=point_it,
                    bounditer=bounditer,
                    scale=self.internal_sampler.scale,
                    blob=old_blob))
            self.eff = 100. * (self.it + i) / self.ncall  # efficiency

            # Return our new "dead" point and ancillary quantities.
            yield IteratorResult(worst=idx,
                                 ustar=ustar,
                                 vstar=vstar,
                                 loglstar=loglstar,
                                 logvol=logvol,
                                 logwt=logwt,
                                 logz=logz,
                                 logzvar=logzvar,
                                 h=h,
                                 nc=1,
                                 blob=old_blob,
                                 worst_it=point_it,
                                 boundidx=boundidx,
                                 bounditer=bounditer,
                                 eff=self.eff,
                                 delta_logz=delta_logz)

    def _remove_live_points(self):
        """Remove the final set of live points if they were
        previously added to the current set of dead points."""

        if self.added_live:
            self.added_live = False
            for k in [
                    'id', 'u', 'v', 'logl', 'logvol', 'logwt', 'logz',
                    'logzvar', 'h', 'nc', 'boundidx', 'it', 'bounditer',
                    'scale', 'blob', 'proposal_stats'
            ]:
                del self.saved_run[k][-self.nlive:]
        else:
            raise ValueError("No live points were added to the "
                             "list of samples!")

    def sample(self,
               maxiter=None,
               maxcall=None,
               dlogz=0.01,
               logl_max=np.inf,
               add_live=True,
               save_bounds=True,
               resume=False):
        """
        **The main nested sampling loop.** Iteratively replace the worst live
        point with a sample drawn uniformly from the prior until the
        provided stopping criteria are reached. Instantiates a generator
        that will be called by the user.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. Default is `0.01`.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        add_live : bool, optional
            Whether or not to add the remaining set of live points to
            the list of samples when calculating `n_effective`.
            Default is `True`.

        save_bounds : bool, optional
            Whether or not to save past distributions used to bound
            the live points internally. Default is `True`.

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (ndim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : float
            Ln(likelihood) of the sample.

        logvol : float
            Ln(prior volume) within the sample.

        logwt : float
            Ln(weight) of the sample.

        logz : float
            Cumulative ln(evidence) up to the sample (inclusive).

        logzvar : float
            Estimated cumulative variance on `logz` (inclusive).

        h : float
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        boundidx : int
            Index of the bound the dead point was originally drawn from.

        bounditer : int
            Index of the bound being used at the current iteration.

        eff : float
            The cumulative sampling efficiency (in percent).

        delta_logz : float
            The estimated remaining evidence expressed as the ln(ratio) of the
            current evidence.

        """

        # Initialize quantities.
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        self.save_bounds = save_bounds
        ncall = 0
        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1 or len(self.saved_run['logl']) == 0:
            # Initialize values for nested sampling loop.
            h = 0.  # information, initially *0.*
            logz = -1.e300  # ln(evidence), initially *0.*
            logzvar = 0.  # var[ln(evidence)], initially *0.*
            logvol = self.logvol_init
            # initially contains the whole prior (volume=1.)
            loglstar = -1.e300  # initial ln(likelihood)
            delta_logz = 1.e300  # ln(ratio) of total/current evidence

        else:
            # Remove live points (if added) from previous run.
            if self.added_live and not resume:
                warnings.warn(
                    'Repeatedly running sample() or run_nested() '
                    '(when not just resuming an existing run is considered '
                    'deprecated and will be removed in the future',
                    DeprecationWarning)
                self._remove_live_points()

            # Get final state from previous run.
            h, logz, logzvar, logvol, loglstar = [
                self.saved_run[_][-1]
                for _ in ['h', 'logz', 'logzvar', 'logvol', 'logl']
            ]
            delta_logz = np.logaddexp(0,
                                      np.max(self.live_logl) + logvol - logz)

        nplateau = 0
        stop_iterations = False
        # The main nested sampling loop.
        for it in range(sys.maxsize):
            delta_logz = np.logaddexp(0,
                                      np.max(self.live_logl) + logvol - logz)

            # Stopping criterion 1: current number of iterations
            # exceeds `maxiter`.
            # Stopping criterion 2: current number of `loglikelihood`
            # calls exceeds `maxcall`.
            if it > maxiter or ncall > maxcall:
                stop_iterations = True
                if dlogz is not None and delta_logz > 10 * dlogz:
                    warnings.warn('The sampling was stopped short due to'
                                  ' maxiter/maxcall limit the delta(log(z))'
                                  ' criterion is not achieved; posterior may'
                                  ' be poorly sampled')

            # Stopping criterion 3: estimated (fractional) remaining evidence
            # lies below some threshold set by `dlogz`.
            if dlogz is not None and delta_logz < dlogz:
                stop_iterations = True

            # Stopping criterion 4: last dead point exceeded the upper
            # `logl_max` bound.
            if loglstar > logl_max:
                stop_iterations = True

            if np.ptp(self.live_logl) == 0:
                warnings.warn(
                    'We have reached the plateau in the likelihood we are'
                    ' stopping sampling')
                stop_iterations = True

            if stop_iterations:
                break

            worst = np.argmin(self.live_logl)  # index
            # Locate the "live" point with the lowest `logl`.
            worst_it = self.live_it[worst]  # when point was proposed
            boundidx = self.live_bound[worst]  # associated bound index

            if not self.plateau_mode:
                nplateau = (self.live_logl == self.live_logl[worst]).sum()
                if nplateau > 1:
                    self.plateau_mode = True
                    self.plateau_counter = nplateau
                    self.plateau_logdvol = np.log(1. /
                                                  (self.nlive + 1)) + logvol
                    # this is log (delta vol)

            if not self.plateau_mode:
                # Expected ln(volume) shrinkage.
                cur_dlv = self.dlv
            else:
                cur_dlv = -np.log1p(-np.exp(self.plateau_logdvol - logvol))
            assert cur_dlv > 0
            logvol -= cur_dlv

            # Set our new worst likelihood constraint.
            # Notice we are doing copies here because live_u and live_v
            # are updated in-place
            ustar = self.live_u[worst].copy()  # unit cube position
            vstar = self.live_v[worst].copy()  # transformed position
            loglstar_new = self.live_logl[worst]  # new likelihood
            if self.blob:
                old_blob = self.live_blobs[worst].copy()
            else:
                old_blob = None

            # Sample a new live point from within the likelihood constraint
            # `logl > loglstar` using the bounding distribution and sampling
            # method from our sampler.
            u, v, logl, nc, proposal_stats = self._new_point(loglstar_new)
            ncall += nc
            self.ncall += nc
            if self.blob:
                new_blob = logl.blob
            else:
                new_blob = None
            (logwt, logz, logzvar,
             h) = progress_integration(loglstar, loglstar_new, logz, logzvar,
                                       logvol, cur_dlv, h)
            loglstar = loglstar_new

            # Compute bound index at the current iteration.
            if not self.unit_cube_sampling:
                bounditer = self.nbound - 1
            else:
                bounditer = 0

            # Save the worst live point. It is now a "dead" point.
            self.saved_run.append(
                dict(id=worst,
                     u=ustar,
                     v=vstar,
                     logl=loglstar,
                     logvol=logvol,
                     logwt=logwt,
                     logz=logz,
                     logzvar=logzvar,
                     h=h,
                     nc=nc,
                     it=worst_it,
                     bounditer=bounditer,
                     scale=self.internal_sampler.scale,
                     blob=old_blob,
                     proposal_stats=proposal_stats))

            # Update the live point (previously our "worst" point).
            self.live_u[worst] = u
            self.live_v[worst] = v
            self.live_logl[worst] = logl
            self.live_bound[worst] = bounditer
            self.live_it[worst] = self.it
            if self.blob:
                self.live_blobs[worst] = new_blob
            # Compute our sampling efficiency.
            self.eff = 100. * self.it / self.ncall

            # Increment total number of iterations.
            self.it += 1

            if self.plateau_mode:
                self.plateau_counter -= 1
                if self.plateau_counter == 0:
                    self.plateau_mode = False
            # Return dead point and ancillary quantities.
            yield IteratorResult(worst=worst,
                                 ustar=ustar,
                                 vstar=vstar,
                                 loglstar=loglstar,
                                 logvol=logvol,
                                 logwt=logwt,
                                 logz=logz,
                                 logzvar=logzvar,
                                 h=h,
                                 nc=nc,
                                 blob=old_blob,
                                 worst_it=worst_it,
                                 boundidx=boundidx,
                                 bounditer=bounditer,
                                 eff=self.eff,
                                 delta_logz=delta_logz)

    def run_nested(self,
                   maxiter=None,
                   maxcall=None,
                   dlogz=None,
                   logl_max=np.inf,
                   add_live=True,
                   print_progress=True,
                   print_func=None,
                   save_bounds=True,
                   checkpoint_file=None,
                   checkpoint_every=60,
                   resume=False):
        """
        **A wrapper that executes the main nested sampling loop.**
        Iteratively replace the worst live point with a sample drawn
        uniformly from the prior until the provided stopping criteria
        are reached.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. If `add_live` is `True`,
            the default is `1e-3 * (nlive - 1) + 0.01`. Otherwise, the
            default is `0.01`.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        n_effective: int, optional
            Minimum number of effective posterior samples. If the estimated
            "effective sample size" (ESS) exceeds this number,
            sampling will terminate. Default is no ESS (`np.inf`).
            This option is deprecated and will be removed in a future release.

        add_live : bool, optional
            Whether or not to add the remaining set of live points to
            the list of samples at the end of each run. Default is `True`.

        print_progress : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        save_bounds : bool, optional
            Whether or not to save past bounding distributions used to bound
            the live points internally. Default is *True*.

        checkpoint_file: string, optional
            if not None The state of the sampler will be saved into this
            file every checkpoint_every seconds

        checkpoint_every: float, optional
            The number of seconds between checkpoints that will save
            the internal state of the sampler. The sampler will also be
            saved in the end of the run irrespective of checkpoint_every.
        """

        # Define our stopping criteria.
        if dlogz is None:
            if add_live:
                dlogz = 1e-3 * (self.nlive - 1.) + 0.01
            else:
                dlogz = 0.01
        if resume and self.added_live:
            warnings.warn('You are resuming a finished static run. '
                          'This will not do anything')
            # TODO I should create a separate STATE Enum
            # here like to rely on that rather than added_live
            return

        # Run the main nested sampling loop.
        pbar, print_func = get_print_func(print_func, print_progress)
        if checkpoint_file is not None:
            timer = DelayTimer(checkpoint_every)
        try:
            ncall = self.ncall
            for it, results in enumerate(
                    self.sample(maxiter=maxiter,
                                maxcall=maxcall,
                                dlogz=dlogz,
                                logl_max=logl_max,
                                save_bounds=save_bounds,
                                resume=resume,
                                add_live=add_live)):
                ncall += results.nc

                # Print progress.
                if print_progress:
                    i = self.it - 1
                    print_func(results,
                               i,
                               ncall,
                               dlogz=dlogz,
                               logl_max=logl_max)

                if checkpoint_file is not None and timer.is_time():
                    self.save(checkpoint_file)

            # Add remaining live points to samples.
            if add_live:
                it = self.it - 1
                for i, results in enumerate(self.add_live_points()):
                    ncall += results.nc

                    # Print progress.
                    if print_progress:
                        print_func(results,
                                   it,
                                   ncall,
                                   add_live_it=i + 1,
                                   dlogz=dlogz,
                                   logl_max=logl_max)

            # Here we recompute the integrals using the full run
            new_logwt, new_logz, new_logzvar, new_h = compute_integrals(
                logl=self.saved_run['logl'], logvol=self.saved_run['logvol'])
            self.saved_run['logwt'] = new_logwt.tolist()
            self.saved_run['logz'] = new_logz.tolist()
            self.saved_run['logzvar'] = new_logzvar.tolist()
            self.saved_run['h'] = new_h.tolist()
            if checkpoint_file is not None:
                # I don't check the time timer here
                self.save(checkpoint_file)

        finally:
            if pbar is not None:
                pbar.close()
            self.loglikelihood.history_save()

    def add_final_live(self, print_progress=True, print_func=None):
        """
        **A wrapper that executes the loop adding the final live points.**
        Adds the final set of live points to the pre-existing sequence of
        dead points from the current nested sampling run.

        Parameters
        ----------
        print_progress : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        """

        if print_func is None:
            print_func = print_fn

        # Add remaining live points to samples.
        pbar, print_func = get_print_func(print_func, print_progress)
        try:
            ncall = self.ncall
            it = self.it - 1
            for i, results in enumerate(self.add_live_points()):

                # Print progress.
                if print_progress:
                    print_func(results,
                               it,
                               ncall,
                               add_live_it=i + 1,
                               dlogz=0.01)
        finally:
            if pbar is not None:
                pbar.close()
