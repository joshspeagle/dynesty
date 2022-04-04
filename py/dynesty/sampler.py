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
from .bounding import UnitCube
from .sampling import sample_unif
from .utils import (get_seed_sequence, get_print_func, progress_integration,
                    IteratorResult, RunRecord, get_neff_from_logwt,
                    compute_integrals)

__all__ = ["Sampler"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))
MAXINT = 2**32 - 1


class Sampler:
    """
    The basic sampler object that performs the actual nested sampling.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int, optional
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

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 update_interval, first_update, rstate, queue_size, pool,
                 use_pool, ncdim):

        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.npdim = npdim
        self.ncdim = ncdim

        # live points
        self.live_u, self.live_v, self.live_logl = live_points
        self.nlive = len(self.live_u)
        self.live_bound = np.zeros(self.nlive, dtype='int')
        self.live_it = np.zeros(self.nlive, dtype='int')

        # bounding updates
        self.update_interval = update_interval
        self.ubound_ncall = first_update.get('min_ncall', 2 * self.nlive)
        self.ubound_eff = first_update.get('min_eff', 10.)
        self.logl_first_update = None

        # random state
        self.rstate = rstate

        # set to none just for qa
        self.scale = None
        self.method = None
        self.kwargs = {}

        # parallelism
        self.pool = pool  # provided pool
        if self.pool is None:
            self.M = map
        else:
            self.M = pool.map
        self.use_pool = use_pool  # provided flags for when to use the pool
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
        self.unused = 0  # total number of proposals unused
        self.used = 0  # total number of proposals used

        # sampling
        self.it = 1  # current iteration
        self.since_update = 0  # number of calls since the last update
        self.ncall = self.nlive  # number of function calls
        self.dlv = math.log((self.nlive + 1.) / self.nlive)  # shrinkage/iter
        self.bound = [UnitCube(self.ncdim)]  # bounding distributions
        self.nbound = 1  # total number of unique bounding distributions
        self.added_live = False  # whether leftover live points were used
        self.eff = 0.  # overall sampling efficiency
        self.cite = ''  # Default empty
        self.save_samples = True
        self.save_bounds = True
        # results
        self.saved_run = RunRecord()

    def propose_point(self, *args):
        raise RuntimeError('Should be overriden')

    def evolve_point(self, *args):
        raise RuntimeError('Should be overriden')

    def update_proposal(self, *args):
        raise RuntimeError('Should be overriden')

    def update(self):
        raise RuntimeError('Should be overriden')

    def __setstate__(self, state):
        self.__dict__ = state
        self.pool = None
        self.M = map

    def __getstate__(self):
        """Get state information for pickling."""

        state = self.__dict__.copy()
        for k in ['M', 'pool']:
            if k in state:
                del state[k]
        return state

    def reset(self):
        """Re-initialize the sampler."""

        # live points
        self.live_u = self.rstate.uniform(size=(self.nlive, self.npdim))
        if self.use_pool_ptform:
            # Use the pool to compute the prior transform.
            self.live_v = np.array(
                list(self.M(self.prior_transform, np.asarray(self.live_u))))
        else:
            # Compute the prior transform using the default `map` function.
            self.live_v = np.array(
                list(map(self.prior_transform, np.asarray(self.live_u))))
        self.live_logl = self.loglikelihood.map(np.asarray(self.live_v))

        self.live_bound = np.zeros(self.nlive, dtype='int')
        self.live_it = np.zeros(self.nlive, dtype='int')

        # parallelism
        self.queue = []
        self.nqueue = 0
        self.unused = 0
        self.used = 0

        # sampling
        self.it = 1
        self.since_update = 0
        self.ncall = self.nlive
        self.bound = [UnitCube(self.ncdim)]
        self.nbound = 1
        self.added_live = False

        # results
        self.saved_run = RunRecord()

    @property
    def results(self):
        """Saved results from the nested sampling run. If bounding
        distributions were saved, those are also returned."""

        d = {}
        for k in [
                'nc', 'v', 'id', 'it', 'u', 'logwt', 'logl', 'logvol', 'logz',
                'logzvar', 'h'
        ]:
            d[k] = np.array(self.saved_run.D[k])

        # Add all saved samples to the results.
        if self.save_samples:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = [('nlive', self.nlive), ('niter', self.it - 1),
                           ('ncall', d['nc']), ('eff', self.eff),
                           ('samples', d['v'])]
                for k in ['id', 'it', 'u']:
                    results.append(('samples_' + k, d[k]))
                for k in ['logwt', 'logl', 'logvol', 'logz']:
                    results.append((k, d[k]))
                results.append(('logzerr', np.sqrt(d['logzvar'])))
                results.append(('information', d['h']))
        else:
            raise ValueError("You didn't save any samples!")

        # Add any saved bounds (and ancillary quantities) to the results.
        if self.save_bounds:
            results.append(('bound', copy.deepcopy(self.bound)))
            results.append(('bound_iter',
                            np.array(self.saved_run.D['bounditer'],
                                     dtype='int')))
            results.append(('samples_bound',
                            np.array(self.saved_run.D['boundidx'],
                                     dtype='int')))
            results.append(('scale', np.array(self.saved_run.D['scale'])))

        return Results(results)

    @property
    def n_effective(self):
        """
        Estimate the effective number of posterior samples using the Kish
        Effective Sample Size (ESS) where `ESS = sum(wts)^2 / sum(wts^2)`.
        Note that this is `len(wts)` when `wts` are uniform and
        `1` if there is only one non-zero element in `wts`.

        """
        logwt = self.saved_run.D['logwt']
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

    def _beyond_unit_bound(self, loglstar):
        """Check whether we should update our bound beyond the initial
        unit cube."""

        if self.logl_first_update is None:
            # If we haven't already updated our bounds, check if we satisfy
            # the provided criteria for establishing the first bounding update.
            check = (self.ncall > self.ubound_ncall
                     and self.eff < self.ubound_eff)
            if check:
                # Save the log-likelihood where our first update took place.
                self.logl_first_update = loglstar
            return check
        else:
            # If we've already update our bounds, check if we've exceeded the
            # saved log-likelihood threshold. (This is useful when sampling
            # within `dynamicsampler`).
            return loglstar >= self.logl_first_update

    def _fill_queue(self, loglstar):
        """Sequentially add new live point proposals to the queue."""

        # Add/zip arguments to submit to the queue.
        point_queue = []
        axes_queue = []

        # All the samplers should have have a starting point
        # satisfying a strict logl>loglstar criterion
        # The slice sampler will just fail if it's not the case
        # therefore we provide those subsets of points to choose from.

        if self.method != 'unif':
            args = (np.nonzero(self.live_logl > loglstar)[0], )
            if len(args[0]) == 0:
                raise RuntimeError(
                    'No live points are above loglstar. '
                    'Do you have a likelihood plateau ? '
                    'It is also possible that you are trying to sample '
                    'excessively around the very peak of the posterior')
        else:
            args = ()
        while self.nqueue < self.queue_size:
            if self._beyond_unit_bound(loglstar):
                # Propose points using the provided sampling/bounding options.
                point, axes = self.propose_point(*args)
                evolve_point = self.evolve_point
            else:
                # Propose/evaluate points directly from the unit cube.
                point = self.rstate.uniform(size=self.npdim)
                axes = np.identity(self.ncdim)
                evolve_point = sample_unif
            point_queue.append(point)
            axes_queue.append(axes)
            self.nqueue += 1
        loglstars = [loglstar for i in range(self.queue_size)]
        scales = [self.scale for i in range(self.queue_size)]
        ptforms = [self.prior_transform for i in range(self.queue_size)]
        logls = [self.loglikelihood for i in range(self.queue_size)]
        kwargs = [self.kwargs for i in range(self.queue_size)]
        seeds = get_seed_sequence(self.rstate, self.queue_size)
        args = zip(point_queue, loglstars, axes_queue, scales, ptforms, logls,
                   seeds, kwargs)

        if self.use_pool_evolve:
            # Use the pool to propose ("evolve") a new live point.
            self.queue = list(self.M(evolve_point, args))
        else:
            # Propose ("evolve") a new live point using the default `map`
            # function.
            self.queue = list(map(evolve_point, args))

    def _get_point_value(self, loglstar):
        """Grab the first live point proposal in the queue."""

        # If the queue is empty, refill it.
        if self.nqueue <= 0:
            self._fill_queue(loglstar)

        # Grab the earliest entry.
        u, v, logl, nc, blob = self.queue.pop(0)

        self.used += 1  # add to the total number of used points
        self.nqueue -= 1

        return u, v, logl, nc, blob

    def _new_point(self, loglstar):
        """Propose points until a new point that satisfies the log-likelihood
        constraint `loglstar` is found."""

        ncall, nupdate = 0, 0
        while True:
            # Get the next point from the queue
            u, v, logl, nc, blob = self._get_point_value(loglstar)
            ncall += nc

            # Bounding checks.
            ucheck = ncall >= self.update_interval * (1 + nupdate)
            bcheck = self._beyond_unit_bound(loglstar)

            # If our queue is empty, update any tuning parameters associated
            # with our proposal (sampling) method.
            if blob is not None and self.nqueue <= 0 and bcheck:
                self.update_proposal(blob)

            # If we satisfy the log-likelihood constraint, we're done!
            if logl > loglstar:
                break

            # If there has been more than `update_interval` function calls
            # made *and* we satisfy the criteria for moving beyond sampling
            # from the unit cube, update the bound.
            if ucheck and bcheck:
                bound = self.update()
                if self.save_bounds:
                    self.bound.append(bound)
                self.nbound += 1
                nupdate += 1
                self.since_update = -ncall  # ncall will be added back later

        return u, v, logl, ncall

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

        # After N samples have been taken out, the remaining volume is
        # `e^(-N / nlive)`. The remaining points are distributed uniformly
        # within the remaining volume so that the expected volume enclosed
        # by the `i`-th worst likelihood is
        # `e^(-N / nlive) * (nlive + 1 - i) / (nlive + 1)`.
        logvols = self.saved_run.D['logvol'][-1]
        logvols += np.log(1. - (np.arange(self.nlive) + 1.) /
                          (self.nlive + 1.))
        logvols_pad = np.concatenate(
            ([self.saved_run.D['logvol'][-1]], logvols))

        # Defining change in `logvol` used in `logzvar` approximation.
        dlvs = logvols_pad[:-1] - logvols_pad[1:]

        # Sorting remaining live points.
        lsort_idx = np.argsort(self.live_logl)
        loglmax = max(self.live_logl)

        # Grabbing relevant values from the last dead point.
        logz = self.saved_run.D['logz'][-1]
        logzvar = self.saved_run.D['logzvar'][-1]
        h = self.saved_run.D['h'][-1]
        loglstar = self.saved_run.D['logl'][-1]
        if self._beyond_unit_bound(loglstar):
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
            loglstar_new = self.live_logl[idx]
            boundidx = self.live_bound[idx]
            point_it = self.live_it[idx]

            (logwt, logz, logzvar,
             h) = progress_integration(loglstar, loglstar_new, logz, logzvar,
                                       logvol, dlv, h)
            loglstar = loglstar_new
            logz_remain = loglmax + logvol  # remaining ln(evidence)
            delta_logz = np.logaddexp(logz, logz_remain) - logz  # dlogz

            # Save results.
            if self.save_samples:
                self.saved_run.append(
                    dict(id=idx,
                         u=ustar,
                         v=vstar,
                         logl=loglstar,
                         logvol=logvol,
                         logwt=logwt,
                         logz=logz,
                         logzvar=logzvar,
                         h=h,
                         nc=1,
                         boundidx=boundidx,
                         it=point_it,
                         bounditer=bounditer,
                         scale=self.scale))
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
            if self.save_samples:
                for k in [
                        'id', 'u', 'v', 'logl', 'logvol', 'logwt', 'logz',
                        'logzvar', 'h', 'nc', 'boundidx', 'it', 'bounditer',
                        'scale'
                ]:
                    del self.saved_run.D[k][-self.nlive:]
        else:
            raise ValueError("No live points were added to the "
                             "list of samples!")

    def sample(self,
               maxiter=None,
               maxcall=None,
               dlogz=0.01,
               logl_max=np.inf,
               n_effective=np.inf,
               add_live=True,
               save_bounds=True,
               save_samples=True):
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

        n_effective: int, optional
            Minimum number of effective posterior samples. If the estimated
            "effective sample size" (ESS) exceeds this number,
            sampling will terminate. Default is no ESS (`np.inf`).

        add_live : bool, optional
            Whether or not to add the remaining set of live points to
            the list of samples when calculating `n_effective`.
            Default is `True`.

        save_bounds : bool, optional
            Whether or not to save past distributions used to bound
            the live points internally. Default is `True`.

        save_samples : bool, optional
            Whether or not to save past samples from the nested sampling run
            (along with other ancillary quantities) internally.
            Default is `True`.

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (npdim,)
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
        self.save_samples = save_samples
        self.save_bounds = save_bounds
        ncall = 0

        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1 or len(self.saved_run.D['logl']) == 0:
            # Initialize values for nested sampling loop.
            h = 0.  # information, initially *0.*
            logz = -1.e300  # ln(evidence), initially *0.*
            logzvar = 0.  # var[ln(evidence)], initially *0.*
            logvol = 0.  # initially contains the whole prior (volume=1.)
            loglstar = -1.e300  # initial ln(likelihood)
            delta_logz = 1.e300  # ln(ratio) of total/current evidence

            # Check if we should initialize a different bounding distribution
            # instead of using the unit cube.
            if self._beyond_unit_bound(loglstar):
                bound = self.update()
                if self.save_bounds:
                    self.bound.append(bound)
                    self.nbound += 1
                self.since_update = 0
        else:
            # Remove live points (if added) from previous run.
            if self.added_live:
                self._remove_live_points()

            # Get final state from previous run.
            h = self.saved_run.D['h'][-1]  # information
            logz = self.saved_run.D['logz'][-1]  # ln(evidence)
            logzvar = self.saved_run.D['logzvar'][-1]  # var[ln(evidence)]
            logvol = self.saved_run.D['logvol'][-1]  # ln(volume)
            loglstar = min(self.live_logl)  # ln(likelihood)
            delta_logz = np.logaddexp(
                logz,
                np.max(self.live_logl) + logvol) - logz  # log-evidence ratio

        stop_iterations = False
        # The main nested sampling loop.
        for it in range(sys.maxsize):

            # Stopping criterion 1: current number of iterations
            # exceeds `maxiter`.
            if it > maxiter:
                stop_iterations = True

            # Stopping criterion 2: current number of `loglikelihood`
            # calls exceeds `maxcall`.
            if ncall > maxcall:
                stop_iterations = True

            # Stopping criterion 3: estimated (fractional) remaining evidence
            # lies below some threshold set by `dlogz`.
            logz_remain = np.max(self.live_logl) + logvol
            delta_logz = np.logaddexp(logz, logz_remain) - logz
            if dlogz is not None and delta_logz < dlogz:
                stop_iterations = True

            # Stopping criterion 4: last dead point exceeded the upper
            # `logl_max` bound.
            if loglstar > logl_max:
                stop_iterations = True

            # Stopping criterion 5: the number of effective posterior
            # samples has been achieved.
            if n_effective is not None:
                if self.n_effective > n_effective:
                    if add_live:
                        self.add_final_live(print_progress=False)
                        neff = self.n_effective
                        self._remove_live_points()
                        self.added_live = False
                    else:
                        neff = self.n_effective
                    if neff > n_effective:
                        stop_iterations = True

            if stop_iterations:
                if not self.save_samples:
                    # If dumping past states, save only the required quantities
                    # TODO I don't quite understand why we do this
                    add_info = dict(logz=logz,
                                    logzvar=logzvar,
                                    h=h,
                                    logvol=logvol,
                                    logl=loglstar)
                    self.saved_run.append(add_info)
                break

            # Expected ln(volume) shrinkage.
            logvol -= self.dlv

            # After `update_interval` interations have passed *and* we meet
            # the criteria for moving beyond sampling from the unit cube,
            # update the bound using the current set of live points.
            ucheck = self.since_update >= self.update_interval
            bcheck = self._beyond_unit_bound(loglstar)
            if ucheck and bcheck:
                bound = self.update()
                if self.save_bounds:
                    self.bound.append(bound)
                self.nbound += 1
                self.since_update = 0

            # Locate the "live" point with the lowest `logl`.
            worst = np.argmin(self.live_logl)  # index
            worst_it = self.live_it[worst]  # when point was proposed
            boundidx = self.live_bound[worst]  # associated bound index

            # Set our new worst likelihood constraint.
            # Notice we are doing copies here because live_u and live_v
            # are updated in-place
            ustar = self.live_u[worst].copy()  # unit cube position
            vstar = self.live_v[worst].copy()  # transformed position
            loglstar_new = self.live_logl[worst]  # new likelihood

            # Sample a new live point from within the likelihood constraint
            # `logl > loglstar` using the bounding distribution and sampling
            # method from our sampler.
            u, v, logl, nc = self._new_point(loglstar_new)
            ncall += nc
            self.ncall += nc
            self.since_update += nc

            (logwt, logz, logzvar,
             h) = progress_integration(loglstar, loglstar_new, logz, logzvar,
                                       logvol, self.dlv, h)
            loglstar = loglstar_new

            # Compute bound index at the current iteration.
            if self._beyond_unit_bound(loglstar):
                bounditer = self.nbound - 1
            else:
                bounditer = 0

            # Save the worst live point. It is now a "dead" point.
            if self.save_samples:
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
                         scale=self.scale))

            # Update the live point (previously our "worst" point).
            self.live_u[worst] = u
            self.live_v[worst] = v
            self.live_logl[worst] = logl
            self.live_bound[worst] = bounditer
            self.live_it[worst] = self.it

            # Compute our sampling efficiency.
            self.eff = 100. * self.it / self.ncall

            # Increment total number of iterations.
            self.it += 1

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
                   n_effective=None,
                   add_live=True,
                   print_progress=True,
                   print_func=None,
                   save_bounds=True):
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

        """

        # Define our stopping criteria.
        if dlogz is None:
            if add_live:
                dlogz = 1e-3 * (self.nlive - 1.) + 0.01
            else:
                dlogz = 0.01

        # Run the main nested sampling loop.
        pbar, print_func = get_print_func(print_func, print_progress)
        try:
            ncall = self.ncall
            for it, results in enumerate(
                    self.sample(maxiter=maxiter,
                                maxcall=maxcall,
                                dlogz=dlogz,
                                logl_max=logl_max,
                                save_bounds=save_bounds,
                                save_samples=True,
                                n_effective=n_effective,
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
                logl=self.saved_run.D['logl'],
                logvol=self.saved_run.D['logvol'])
            self.saved_run.D['logwt'] = new_logwt.tolist()
            self.saved_run.D['logz'] = new_logz.tolist()
            self.saved_run.D['logzvar'] = new_logzvar.tolist()
            self.saved_run.D['h'] = new_h.tolist()

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
