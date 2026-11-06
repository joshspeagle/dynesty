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
from .internal_samplers import UnitCubeSampler, SamplerHistoryItem
from .utils import (get_seed_sequence, get_print_func, progress_integration,
                    IteratorResult, RunRecord, get_neff_from_logwt,
                    compute_integrals, DelayTimer, _LOWL_VAL,
                    get_random_generator)

from .bounding import (UnitCube, Ellipsoid, MultiEllipsoid, RadFriends,
                       PeriodicBound,
                       SupFriends, Bound, BOUND_LIST)
from .utils import (save_sampler, restore_sampler)

__all__ = ["Sampler", "Proposer", "LivePoints"]


def _get_bound(bounding, ndim, ncdim=None, periodic=None):
    if isinstance(bounding, str):
        if bounding not in BOUND_LIST:
            raise ValueError('Unsupported bounding type')
    elif isinstance(bounding, Bound):
        pass
    else:
        raise ValueError('Unsupported bounding type')

    # the bound only sees the first ncdim (clustered) dimensions;
    # the remaining ones are sampled uniformly and need no wrapping
    ncdim = ncdim or ndim
    if bounding == 'none':
        bound = UnitCube(ncdim)
    elif bounding == 'single':
        bound = Ellipsoid(ncdim)
        # this is ellipsoid in the center of the cube that contains
        # the whole cube
    elif bounding == 'multi':
        bound = MultiEllipsoid(ncdim)
        # this is ellipsoid in the center of the cube that contains
        # the whole cube
    elif bounding == 'balls':
        bound = RadFriends(ncdim)
    elif bounding == 'cubes':
        bound = SupFriends(ncdim)
    else:
        bound = bounding
    if periodic is not None:
        periodic = [i for i in periodic if i < ncdim]
    if periodic and not isinstance(bound, UnitCube):
        bound = PeriodicBound(bound, periodic)
    return bound


def _initialize_live_points(live_points,
                            prior_transform,
                            loglikelihood,
                            mapper,
                            nlive=None,
                            ndim=None,
                            rstate=None,
                            blob=False,
                            use_pool_ptform=None,
                            use_pool_logl=None):
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

    loglikelihood: function

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
    (live_u, live_v, live_logl, live_blobs), logvol_init, ncalls : tuple
        The first tuple consist of:
        live_u Unit cube coordinates of points
        live_v Original coordinates.
        live_logl log-likelihood values of points
        live_blobs - Array of blobs associated with logl calls (or None)
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
        ngoods = 0  # counter for how many finite logl we have kept
        ngoods_total = 0  # counter for how many finite logl we have found
        live_blobs = []
        iattempt = 0
        while True:
            iattempt += 1

            # Generate nlive points by uniform sampling (similar to original)
            cur_live_u = rstate.random(size=(nlive, ndim))

            if use_pool_ptform:
                cur_live_v = mapper(prior_transform, np.asarray(cur_live_u))
            else:
                cur_live_v = map(prior_transform, np.asarray(cur_live_u))
            cur_live_v = np.array(list(cur_live_v))
            if use_pool_logl:
                cur_live_logl = list(
                    mapper(loglikelihood, np.asarray(cur_live_v)))
            else:
                cur_live_logl = list(map(loglikelihood,
                                         np.asarray(cur_live_v)))
            if blob:
                cur_live_blobs = np.array([_.blob for _ in cur_live_logl])
            cur_live_logl = np.array([_.val for _ in cur_live_logl])
            # Add evaluation history entries for initialization
            if loglikelihood.save_evaluation_history:
                evaluation_history = []
                for i in range(len(cur_live_u)):
                    evaluation_history.append(
                        SamplerHistoryItem(u=cur_live_u[i],
                                           v=cur_live_v[i],
                                           logl=cur_live_logl[i]))
                loglikelihood.append_evaluation_history(evaluation_history)

            ncalls += nlive
            finite = np.isfinite(cur_live_logl)
            not_finite = ~finite
            neg_infinite = np.isneginf(cur_live_logl)
            if np.any(not_finite & (~neg_infinite)):
                raise ValueError("The log-likelihood of live "
                                 "point is invalid.")

            # how many finite logl values we have
            cur_ngood = finite.sum()
            ngoods_total += cur_ngood
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
                cur_live_logl[not_finite] = _LOWL_VAL
                nextra = nlive - ngoods
                if nextra > 0:
                    cur_ind = np.nonzero(not_finite)[0][:nextra]
                    assert len(cur_ind) == nextra
                    live_logl[ngoods:ngoods + nextra] = cur_live_logl[cur_ind]
                    live_u[ngoods:ngoods + nextra] = cur_live_u[cur_ind]
                    live_v[ngoods:ngoods + nextra] = cur_live_v[cur_ind]
                    if blob:
                        live_blobs.extend(cur_live_blobs[cur_ind])
                logvol_init = np.log(ngoods_total / (iattempt * ngoods))
                # The logic is the following:
                # if we have n live points and we sampled N attempts
                # and we found K points above LOWL_VAL in total
                # then the volume associated with pts above LOWL_VAL
                # can be estimated as K/(Nn)
                # We quit with k<=K points above LOWL_VAL (finite points
                # of the last attempt that do not fit into n are dropped)
                # and (n-k) LOWL points.
                # The run treats the n live points as uniformly
                # distributed in volume V, so the finite-logl volume
                # is V k/n. Equating it to K/(Nn) gives V = K/(Nk),
                # which is 1/N if no finite points were dropped
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
                    warnings.warn(f"After {n_attempts} attempts, we could not "
                                  f"find at least {min_npoints} points "
                                  "that have a valid log-likelihood! "
                                  "The initial sampling is very inefficient!")

    else:
        # If live points were provided, convert the log-likelihoods and
        # then run a quick safety check.
        live_u, live_v, live_logl = live_points[:3]
        if blob:
            live_blobs = live_points[3]
        live_logl = np.asarray(live_logl)
        for i, logl in enumerate(live_logl):
            if not np.isfinite(logl):
                if np.sign(logl) < 0:
                    live_logl[i] = _LOWL_VAL
                else:
                    raise ValueError(
                        f"The log-likelihood ({logl}) of live point {i} "
                        f"located at u={live_u[i]} v={live_v[i]} is invalid.")
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


class LivePoints:
    """
    A set of live points: their positions in the unit cube (`u`) and in
    the original parameter space (`v`), their log-likelihoods (`logl`),
    blobs (or None), the iteration at which each was drawn (`it`) and
    the index of the bound each was drawn from (`bound`).
    """

    def __init__(self, u, v, logl, blobs=None, it=None, bound=None):
        self.u = u
        self.v = v
        self.logl = logl
        self.blobs = blobs
        self.it = np.zeros(len(u), dtype=int) if it is None else it
        self.bound = np.zeros(len(u), dtype=int) if bound is None else bound

    def __len__(self):
        return len(self.logl)

    def replace(self, i, u, v, logl, blob, it, bound):
        """Replace the i-th live point."""
        self.u[i] = u
        self.v[i] = v
        self.logl[i] = logl
        if self.blobs is not None:
            self.blobs[i] = blob
        self.it[i] = it
        self.bound[i] = bound


class Proposer:
    """
    Draws new points from the prior above a likelihood threshold, given a
    set of live points above that threshold. While the prior volume is
    large it samples the unit cube; once the expected efficiency of that,
    exp(logvol), drops below `first_update['min_eff']` percent, it fits a
    bound of type `bounding` to the live points and uses the internal
    sampler `sampling` within it, refitting the bound every
    `bound_update_interval` likelihood calls.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    ndim : int
        Number of parameters accepted by `prior_transform`.

    sampling : :class:`~dynesty.internal_samplers.InternalSampler` instance
        The sampler used to propose new points within the bound. Its
        tuning state is updated as points are drawn.

    bounding : str or :class:`~dynesty.bounding.Bound` instance
        The bounding distribution (`'none'`, `'single'`, `'multi'`,
        `'balls'`, `'cubes'` or a `Bound` instance).

    ncdim : int, optional
        The number of clustering dimensions. The first ncdim dimensions
        will be sampled using the sampling method, the remaining
        dimensions will
        just sample uniformly from the prior distribution.
        If this is `None` (default), this will default to ndim.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    pool : pool, optional
        Use this pool of workers to execute operations in parallel.

    use_pool : dict, optional
        A dictionary containing flags indicating where the provided `pool`
        should be used to execute operations in parallel.

    queue_size : int, optional
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    bound_update_interval : int, optional
        Only update the bounding distribution every
        `bound_update_interval`-th likelihood call.

    first_update : dict, optional
        `{'min_eff': percent}`: stop sampling the unit cube once its
        expected efficiency falls below this value. The default is 3.

    bound_bootstrap : int, optional
        The number of bootstrapped realizations used to enlarge the bounds.

    bound_enlarge : float, optional
        The factor by which the volume of the bounds is enlarged.
    """

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 ndim,
                 sampling,
                 bounding,
                 ncdim=None,
                 rstate=None,
                 pool=None,
                 use_pool=None,
                 queue_size=None,
                 bound_update_interval=None,
                 first_update=None,
                 bound_bootstrap=None,
                 bound_enlarge=None):
        # settings
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.ndim = ndim
        self.ncdim = ncdim or ndim
        self.bounding = bounding
        self.rstate = rstate or get_random_generator()
        self.pool = pool
        self.use_pool = use_pool or {}
        if self.use_pool.get('propose_point', True):
            self.queue_size = queue_size
        else:
            self.queue_size = 1
        self.bound_update_interval = bound_update_interval
        self.first_update = first_update or {}
        self.bound_bootstrap = bound_bootstrap
        self.bound_enlarge = bound_enlarge

        # state
        self.internal_sampler = sampling
        self.unit_cube_sampling = True
        self.bound = UnitCube(self.ncdim)
        self.bound_list = [self.bound]  # saved bounds
        self.nbound = 1  # number of bounds used (saved or not)
        self.save_bounds = True
        self.logl_first_update = None  # logl at which the bound was first used
        self.ncall = 0
        self.ncall_at_last_update = 0
        self.queue = []

    def copy(self, scale=None, **settings):
        """
        A new proposer with the same settings (except those given) that
        starts from the tuning state of this one's internal sampler
        (with the `scale` replaced, if given), but from the unit cube.
        """
        sampling = copy.deepcopy(self.internal_sampler)
        if scale is not None:
            sampling.scale = scale
        return Proposer(**{
            **dict(loglikelihood=self.loglikelihood,
                   prior_transform=self.prior_transform,
                   ndim=self.ndim,
                   sampling=sampling,
                   bounding=self.bounding,
                   ncdim=self.ncdim,
                   rstate=self.rstate,
                   pool=self.pool,
                   use_pool=self.use_pool,
                   queue_size=self.queue_size,
                   bound_update_interval=self.bound_update_interval,
                   first_update=self.first_update,
                   bound_bootstrap=self.bound_bootstrap,
                   bound_enlarge=self.bound_enlarge),
            **settings
        })

    def __getstate__(self):
        state = self.__dict__.copy()
        state['pool'] = None
        return state

    @property
    def mapper(self):
        return map if self.pool is None else self.pool.map

    @property
    def scale(self):
        """Scale of the proposals being made (1 in the unit cube)."""
        return 1 if self.unit_cube_sampling else self.internal_sampler.scale

    def draw(self, loglstar, live, logvol):
        """
        Draw a new point from the prior with logl > loglstar. `live` are the
        current live points (all above loglstar); they are used to fit the
        bound and as starting points of the internal sampler, but are not
        changed. `logvol` is the log of the prior volume above loglstar.

        Returns u, v, logl, the number of likelihood calls, the proposal
        statistics and the index of the bound the point was drawn from.
        """
        min_eff = self.first_update.get('min_eff', 3.)
        if self.unit_cube_sampling and logvol < math.log(min_eff / 100.):
            self.switch_to_bound(loglstar, live)
        ncall = 0
        while True:
            if not self.queue:
                self._fill_queue(loglstar, live)
            ret = self.queue.pop(0)
            ncall += ret.ncalls
            self.ncall += ret.ncalls
            if self.loglikelihood.save_evaluation_history:
                self.loglikelihood.append_evaluation_history(
                    ret.evaluation_history)
            if ret.tuning_info is not None and not self.unit_cube_sampling:
                # the tuning parameters are only updated once the queue
                # is empty, until then the history is accumulated
                self.internal_sampler.tune(ret.tuning_info,
                                           update=not self.queue)
            if (not self.queue and not self.unit_cube_sampling
                    and self.ncall >= self.ncall_at_last_update +
                    self.bound_update_interval):
                self.update_bound(loglstar, live)
            if ret.logl > loglstar:
                bound_index = 0 if self.unit_cube_sampling else self.nbound - 1
                return (ret.u, ret.v, ret.logl, ncall, ret.proposal_stats,
                        bound_index)

    def switch_to_bound(self, loglstar, live):
        """Stop sampling the unit cube: fit the bound to the live points
        and use the internal sampler from now on."""
        self.unit_cube_sampling = False
        self.logl_first_update = loglstar
        self.bound = _get_bound(
            self.bounding,
            self.ndim,
            ncdim=self.ncdim,
            periodic=self.internal_sampler.sampler_kwargs.get('periodic'))
        self.update_bound(loglstar, live)

    def update_bound(self, loglstar, live):
        """Refit the bound to the live points."""
        if loglstar == _LOWL_VAL:
            # in the beginning, we do not want to use the points
            # with not finite logl for the bound
            subset = live.logl > loglstar
        else:
            subset = slice(None)
        pool = self.pool if self.use_pool.get('update_bound', True) else None
        self.bound.update(live.u[subset, :self.ncdim],
                          rstate=self.rstate,
                          bootstrap=self.bound_bootstrap,
                          pool=pool)
        if self.bound_enlarge != 1.:
            self.bound.scale_to_logvol(self.bound.logvol +
                                       np.log(self.bound_enlarge))
        if self.save_bounds:
            self.bound_list.append(copy.deepcopy(self.bound))
        self.nbound += 1
        self.ncall_at_last_update = self.ncall

    def propose_live(self, loglstar, live):
        """Return a live point above loglstar and the bound axes as
        the starting point for the internal sampler."""
        i = self.rstate.choice(np.nonzero(live.logl > loglstar)[0])
        u = live.u[i, :].copy()  # samplers may modify it in place
        ax = self.bound.get_random_axes(self.rstate)
        if self.bound.need_centers:
            self.bound.ctrs = live.u[:, :self.ncdim]
        # Automatically trigger an update if we're not in any ellipsoid.
        if not self.bound.contains(u[:self.ncdim]):
            self.update_bound(-np.inf, live)
            if not self.bound.contains(u[:self.ncdim]):
                raise RuntimeError('Update of the ellipsoid failed')
        return u, ax

    def _fill_queue(self, loglstar, live):
        """Propose and evaluate `queue_size` new points."""
        if not np.any(live.logl > loglstar):
            raise RuntimeError(
                'No live points are above loglstar. '
                'Do you have a likelihood plateau ? '
                'It is also possible that you are trying to sample '
                'excessively around the very peak of the posterior')
        points, axes = zip(*[
            self.propose_live(loglstar, live) for _ in range(self.queue_size)
        ])
        if self.queue_size > 1:
            seeds = get_seed_sequence(self.rstate, self.queue_size)
        else:
            seeds = [self.rstate]
        if self.use_pool.get('propose_point', True):
            mapper = self.mapper
        else:
            mapper = map
        if self.unit_cube_sampling:
            sampler = UnitCubeSampler(ndim=self.ndim)
        else:
            sampler = self.internal_sampler
        args = sampler.prepare_sampler(loglstar=loglstar,
                                       points=points,
                                       axes=axes,
                                       seeds=seeds,
                                       prior_transform=self.prior_transform,
                                       loglikelihood=self.loglikelihood,
                                       nested_sampler=self)
        self.queue = list(mapper(sampler.sample, args))


class Sampler:
    """
    One nested sampling run: a set of live points that is repeatedly
    updated by replacing the worst point with a new one drawn by the
    proposer, and the record of the dead points.

    Parameters
    ----------
    proposer : :class:`Proposer`
        Draws the new points. The run changes its state (bounds, tuning).

    live : :class:`LivePoints`
        The starting live points.

    logvol_init : float, optional
        The log of the prior volume above the lowest starting live point
        (zero for a run started from the whole prior).

    ncall_init : int, optional
        The number of likelihood calls spent on the starting live points
        (only used for reporting the total number of calls).

    blob : bool, optional
        Whether the log-likelihood returns a blob in addition to the logl.

    cite : str, optional
        The citation information for the run.
    """

    def __init__(self,
                 proposer,
                 live,
                 logvol_init=0.,
                 ncall_init=0,
                 blob=False,
                 cite=None):
        self.proposer = proposer
        self.live = live
        self.logvol_init = logvol_init
        self.ncall_init = ncall_init
        self.blob = blob
        self.cite = cite

        # integration state, changed by sample()
        self.it = 1  # current iteration
        self.logvol = logvol_init  # ln(volume) above the worst live point
        self.logz = -1.e300
        self.h = 0.
        self.loglstar = -1.e300  # logl of the last dead point
        self.plateau_counter = 0  # >0 while going through a plateau
        self.plateau_logdvol = None
        self.saved_run = RunRecord()  # the dead points
        self.added_live = False  # whether the results include the live points

    @property
    def nlive(self):
        return len(self.live)

    @property
    def ncall(self):
        return self.ncall_init + self.proposer.ncall

    @property
    def eff(self):
        """Overall sampling efficiency in percent."""
        niter = self.it - 1 + (self.nlive if self.added_live else 0)
        return 100. * niter / max(self.ncall, 1)

    @property
    def loglikelihood(self):
        return self.proposer.loglikelihood

    def save(self, fname):
        """
        Save the state of the sampler in a file

        Parameters
        ----------
        fname : string
            Filename of the save file.

        """
        save_sampler(self, fname)

    @staticmethod
    def restore(fname, pool=None):
        """
        Restore the sampler from a file.
        It is assumed that the file was created using .save() method
        of a sampler or as a result of checkpointing during
        run_nested()

        Parameters
        ----------
        fname : string
            Filename of the save file.
        pool : object(optional)
            The multiprocessing pool-like object that supports map()
            calls that will be used in the restored object.

        """
        return restore_sampler(fname, pool=pool)

    def reset(self):
        """Re-initialize the sampler from new prior draws."""
        proposer = self.proposer
        live_points, logvol_init, ncall_init = _initialize_live_points(
            None,
            proposer.prior_transform,
            proposer.loglikelihood,
            proposer.mapper,
            nlive=self.nlive,
            ndim=proposer.ndim,
            rstate=proposer.rstate,
            blob=self.blob,
            use_pool_ptform=proposer.use_pool.get('prior_transform', True),
            use_pool_logl=proposer.use_pool.get('loglikelihood', True))
        self.__init__(proposer.copy(),
                      LivePoints(*live_points),
                      logvol_init=logvol_init,
                      ncall_init=ncall_init,
                      blob=self.blob,
                      cite=self.cite)

    def _final_points(self):
        """
        The records of the remaining live points, treated as dead points
        in the order of increasing logl. After N points have been removed,
        the remaining volume is X = exp(logvol) and the remaining points are
        distributed uniformly within it, so the expected volume enclosed by
        the i-th worst of them is X * (nlive + 1 - i) / (nlive + 1). If we
        are in a plateau, the remaining points in it first use the volume
        decrements of the plateau. Nothing in the sampler changes.
        """
        nlive = self.nlive
        if self.plateau_counter == 0:
            logvols = np.log(1. - (np.arange(nlive) + 1.) / (nlive + 1.))
        else:
            logvols = np.log1p(-((1 + np.arange(self.plateau_counter)) *
                                 np.exp(self.plateau_logdvol - self.logvol)))
            nrest = nlive - self.plateau_counter
            logvols = np.concatenate([
                logvols,
                logvols[-1] + np.log1p(-(1 + np.arange(nrest)) / (nrest + 1))
            ])
        # the volumes are kept relative until here, so the first dlv is 0
        dlvs = -np.diff(logvols, prepend=0)
        logvols += self.logvol
        bounditer = self.proposer.nbound - 1
        if self.proposer.unit_cube_sampling:
            bounditer = 0
        logz, h, loglstar = self.logz, self.h, self.loglstar
        records = []
        for i, (idx, logvol, dlv) in enumerate(
                zip(np.argsort(self.live.logl), logvols, dlvs)):
            logl = self.live.logl[idx]
            logwt, logz, h = progress_integration(loglstar, logl, logz,
                                                  logvol, dlv, h)
            loglstar = logl
            records.append(
                dict(
                    id=idx,
                    u=self.live.u[idx].copy(),
                    v=self.live.v[idx].copy(),
                    logl=logl,
                    logvol=logvol,
                    logwt=logwt,
                    logz=logz,
                    h=h,
                    n=nlive - i,
                    # technically we did not call the likelihood, but
                    # this keeps sum(nc) close to ncall
                    nc=1,
                    boundidx=self.live.bound[idx],
                    it=self.live.it[idx],
                    bounditer=bounditer,
                    scale=self.proposer.scale,
                    blob=(self.live.blobs[idx].copy()
                          if self.blob else None),
                    proposal_stats=None))
        return records

    def run_record(self):
        """The records of all dead points (including the final live points
        if they were added) as a dictionary of arrays."""
        final = self._final_points() if self.added_live else []
        return {
            k: np.array(self.saved_run[k] + [_[k] for _ in final])
            for k in self.saved_run.keys()
        }

    @property
    def results(self):
        """Saved results from the nested sampling run. If bounding
        distributions were saved, those are also returned."""

        d = self.run_record()
        # saved_run only has the running estimates from progress_integration
        # so we recompute the integrals using the full run
        d['logwt'], d['logz'], d['logzvar'], d['h'] = compute_integrals(
            logl=d['logl'], logvol=d['logvol'], logvol_init=self.logvol_init)

        # Add all saved samples to the results.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = [('nlive', self.nlive), ('niter', self.it - 1),
                       ('ncall', d['nc']), ('eff', self.eff),
                       ('samples', d['v']), ('blob', d['blob']),
                       ('proposal_stats', d['proposal_stats']),
                       ('logvol_init', self.logvol_init)]
            for k in ['id', 'it', 'u']:
                results.append(('samples_' + k, d[k]))
            for k in ['logwt', 'logl', 'logvol', 'logz']:
                results.append((k, d[k]))
            results.append(('logzerr', np.sqrt(d['logzvar'])))
            results.append(('information', d['h']))
        results.append(('scale', d['scale']))

        # Add any saved bounds (and ancillary quantities) to the results.
        if self.proposer.save_bounds:
            results.append(('bound', copy.deepcopy(self.proposer.bound_list)))
            results.append(('bound_iter', d['bounditer'].astype(int)))
            results.append(('samples_bound', d['boundidx'].astype(int)))

        return Results(results)

    @property
    def n_effective(self):
        """
        Estimate the effective number of posterior samples using the Kish
        Effective Sample Size (ESS) where `ESS = sum(wts)^2 / sum(wts^2)`.
        Note that this is `len(wts)` when `wts` are uniform and
        `1` if there is only one non-zero element in `wts`.

        """
        logwt = self.run_record()['logwt']
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

    def add_live_points(self):
        """Add the remaining set of live points to the current set of dead
        points. Instantiates a generator that will be called by
        the user. Yields the same `IteratorResult` objects as
        :meth:`sample`."""

        if self.added_live:
            raise ValueError("The remaining live points have already "
                             "been added to the list of samples!")
        self.added_live = True
        loglmax = np.max(self.live.logl)
        for i, rec in enumerate(self._final_points()):
            yield IteratorResult(worst=rec['id'],
                                 ustar=rec['u'],
                                 vstar=rec['v'],
                                 loglstar=rec['logl'],
                                 logvol=rec['logvol'],
                                 logwt=rec['logwt'],
                                 logz=rec['logz'],
                                 h=rec['h'],
                                 nc=1,
                                 blob=rec['blob'],
                                 worst_it=rec['it'],
                                 boundidx=rec['boundidx'],
                                 bounditer=rec['bounditer'],
                                 eff=100. * (self.it + i) / max(self.ncall, 1),
                                 delta_logz=np.logaddexp(
                                     0, loglmax + rec['logvol'] - rec['logz']),
                                 proposal_stats=None)

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
            Not used by this method. The remaining set of live points
            is added to the list of samples by :meth:`add_live_points`
            (called by :meth:`run_nested` if its `add_live` is `True`).

        save_bounds : bool, optional
            Whether or not to save past distributions used to bound
            the live points internally. Default is `True`.

        resume : bool, optional
            If `True`, continue a previously interrupted run (e.g. one
            restored from a checkpoint file) instead of starting a new one.
            Default is `False`.

        Yields
        ------
        result : `IteratorResult`
            A named tuple (`dynesty.utils.IteratorResult`) describing the
            new dead point, with the following fields:

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

            h : float
                Cumulative information up to the sample (inclusive).

            nc : int
                Number of likelihood calls performed before the new
                live point was accepted.

            worst_it : int
                Iteration when the live (now dead) point was originally
                proposed.

            boundidx : int
                Index of the bound the dead point was originally drawn from.

            bounditer : int
                Index of the bound being used at the current iteration.

            eff : float
                The cumulative sampling efficiency (in percent).

            delta_logz : float
                The estimated remaining evidence expressed as the ln(ratio)
                of the current evidence.

            blob : object
                The blob returned by the log-likelihood for the sample (if
                `blob=True`, otherwise `None`).

            proposal_stats : dict
                Statistics of the internal sampler proposal of the sample
                (e.g. the number of accepted/rejected steps).

        """
        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize
        self.proposer.save_bounds = save_bounds
        if self.added_live:
            # continuing a run after its live points were added
            if not resume:
                warnings.warn(
                    'Repeatedly running sample() or run_nested() '
                    '(when not just resuming an existing run is considered '
                    'deprecated and will be removed in the future',
                    DeprecationWarning)
            self.added_live = False
        live = self.live
        ncall = 0
        for it in range(sys.maxsize):
            delta_logz = np.logaddexp(0,
                                      np.max(live.logl) + self.logvol - self.logz)

            # Stopping criteria: the number of iterations or likelihood
            # calls, the remaining evidence, the logl_max bound or the
            # plateau in the likelihood.
            stop = False
            if it > maxiter or ncall > maxcall:
                stop = True
                if dlogz is not None and delta_logz > 10 * dlogz:
                    warnings.warn('The sampling was stopped short due to'
                                  ' maxiter/maxcall limit the delta(log(z))'
                                  ' criterion is not achieved; posterior may'
                                  ' be poorly sampled')
            if dlogz is not None and delta_logz < dlogz:
                stop = True
            if self.loglstar > logl_max:
                stop = True
            if np.ptp(live.logl) == 0:
                warnings.warn(
                    'We have reached the plateau in the likelihood we are'
                    ' stopping sampling')
                stop = True
            if stop:
                break

            # Locate the "live" point with the lowest `logl`.
            worst = np.argmin(live.logl)
            if self.plateau_counter == 0:
                nplateau = (live.logl == live.logl[worst]).sum()
                if nplateau > 1:
                    self.plateau_counter = nplateau
                    # log(delta vol) of each point in the plateau
                    self.plateau_logdvol = np.log(
                        1. / (self.nlive + 1)) + self.logvol
            if self.plateau_counter == 0:
                # Expected ln(volume) shrinkage.
                dlv = math.log((self.nlive + 1.) / self.nlive)
            else:
                dlv = -np.log1p(-np.exp(self.plateau_logdvol - self.logvol))
            assert dlv > 0
            self.logvol -= dlv

            # The worst point becomes a dead point; we copy as the live
            # points are updated in place
            loglstar_new = live.logl[worst]
            dead = dict(id=worst,
                        u=live.u[worst].copy(),
                        v=live.v[worst].copy(),
                        logl=loglstar_new,
                        boundidx=live.bound[worst],
                        it=live.it[worst],
                        n=self.nlive,
                        blob=live.blobs[worst].copy() if self.blob else None)

            # Sample a new live point within the likelihood constraint
            u, v, logl, nc, proposal_stats, bounditer = self.proposer.draw(
                loglstar_new, live, self.logvol)
            ncall += nc
            logwt, self.logz, self.h = progress_integration(
                self.loglstar, loglstar_new, self.logz, self.logvol, dlv,
                self.h)
            self.loglstar = loglstar_new
            dead.update(logvol=self.logvol,
                        logwt=logwt,
                        logz=self.logz,
                        h=self.h,
                        nc=nc,
                        bounditer=bounditer,
                        scale=self.proposer.scale,
                        proposal_stats=proposal_stats)
            self.saved_run.append(dead)
            live.replace(worst,
                         u,
                         v,
                         logl,
                         logl.blob if self.blob else None,
                         it=self.it,
                         bound=bounditer)
            self.it += 1
            if self.plateau_counter > 0:
                self.plateau_counter -= 1

            yield IteratorResult(worst=worst,
                                 ustar=dead['u'],
                                 vstar=dead['v'],
                                 loglstar=loglstar_new,
                                 logvol=self.logvol,
                                 logwt=logwt,
                                 logz=self.logz,
                                 h=self.h,
                                 nc=nc,
                                 blob=dead['blob'],
                                 worst_it=dead['it'],
                                 boundidx=dead['boundidx'],
                                 bounditer=bounditer,
                                 eff=self.eff,
                                 delta_logz=delta_logz,
                                 proposal_stats=proposal_stats)

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

        add_live : bool, optional
            Whether or not to add the remaining set of live points to
            the list of samples at the end of each run. Default is `True`.

        print_progress : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :func:`~dynesty.utils.print_fn` is used.

        save_bounds : bool, optional
            Whether or not to save past bounding distributions used to bound
            the live points internally. Default is *True*.

        checkpoint_file : string, optional
            if not None The state of the sampler will be saved into this
            file every checkpoint_every seconds

        checkpoint_every : float, optional
            The number of seconds between checkpoints that will save
            the internal state of the sampler. The sampler will also be
            saved in the end of the run irrespective of checkpoint_every.
            Default is `60`.

        resume : bool, optional
            If `True`, continue a previously interrupted run (e.g. one
            restored from a checkpoint file) instead of starting a new one.
            Default is `False`.
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
        pbar, print_func = get_print_func(print_func,
                                          print_progress,
                                          initial=self.it - 1)
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

            if checkpoint_file is not None:
                # I don't check the time timer here
                self.save(checkpoint_file)

        finally:
            if pbar is not None:
                pbar.close()
            self.loglikelihood.finalize_history()

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
            If not provided, the default :func:`~dynesty.utils.print_fn` is used.

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
