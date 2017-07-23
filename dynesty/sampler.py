#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base `Sampler` class containing various helpful functions. Also contains a
`Results` class for storing our results.

"""

from __future__ import (print_function, division)
from builtins import range

import sys
import warnings
import math
import copy
import scipy.misc as misc
import numpy as np

from .results import *
from .bounding import *

__all__ = ["Sampler"]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


class Sampler(object):
    """
    The basic sampler object that performs dynamic nested sampling.


    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior.

    npdim : int, optional
        Number of parameters accepted by prior.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

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
        Use this pool of workers to propose live points in parallel.

    """

    def __init__(self, loglikelihood, prior_transform, npdim, live_points,
                 update_interval, rstate, queue_size, pool):
        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.npdim = npdim

        # live points
        self.live_u, self.live_v, self.live_logl = live_points
        self.nlive = len(self.live_u)
        self.live_prop = np.zeros(self.nlive, dtype='int')
        self.live_it = np.zeros(self.nlive, dtype='int')

        # proposal updates
        self.update_interval = update_interval

        # random state
        self.rstate = rstate

        # parallelism
        self.queue_size = queue_size
        self.pool = pool
        if self.pool is None:
            self.M = map
        else:
            self.M = pool.map
        self.queue = []  # proposed live point queue
        self.nqueue = 0  # current size of the queue
        self.unused = 0  # total number of proposals unused
        self.used = 0  # total number of proposals used

        # sampling
        self.it = 1  # current iteration
        self.since_update = 0  # number of calls since the last update
        self.ncall = self.nlive  # number of function calls
        self.dlv = math.log((self.nlive + 1.) / self.nlive)  # shrinkage/iter
        self.prop = [UnitCube(self.npdim)]  # proposals
        self.nprop = 1  # total number of unique proposal distributions
        self.added_live = False  # whether leftover live points were used

        # results
        self.saved_id = []  # live point labels
        self.saved_u = []  # unit cube samples
        self.saved_v = []  # transformed variable samples
        self.saved_logl = []  # loglikelihoods of samples
        self.saved_logvol = []  # expected log(volume)
        self.saved_logwt = []  # log(weights)
        self.saved_logz = []  # cumulative log(evidence)
        self.saved_logzvar = []  # cumulative error on log(evidence)
        self.saved_h = []  # cumulative information
        self.saved_nc = []  # number of calls at each iteration
        self.saved_propidx = []  # index of proposal dead point was drawn from
        self.saved_it = []  # iteration the live (now dead) point was proposed
        self.saved_piter = []  # active proposal at a specific iteration
        self.saved_scale = []  # scale factor at each iteration

    def reset(self):
        """Re-initialize the sampler."""

        # live points
        self.live_u = self.rstate.rand(self.nlive, self.npdim)
        self.live_v = self.M(self.prior_transform, self.live_u)
        self.live_logl = self.M(self.loglikelihood, self.live_v)
        self.live_prop = np.zeros(self.nlive, dtype='int')
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
        self.prop = [UnitCube(self.npdim)]
        self.nprop = 1
        self.added_live = False

        # results
        self.saved_id = []
        self.saved_u = []
        self.saved_v = []
        self.saved_logl = []
        self.saved_logvol = []
        self.saved_logwt = []
        self.saved_logz = []
        self.saved_logzvar = []
        self.saved_h = []
        self.saved_nc = []
        self.saved_propidx = []
        self.saved_it = []
        self.saved_piter = []
        self.saved_scale = []

    @property
    def results(self):
        """The full results from the nested sampling run. If proposals
        were saved, those are also returned."""

        if self.save_samples:
            results = [('nlive', self.nlive),
                       ('niter', self.it - 1),
                       ('ncall', np.array(self.saved_nc)),
                       ('eff', self.eff),
                       ('samples', np.array(self.saved_v)),
                       ('samples_id', np.array(self.saved_id)),
                       ('samples_it', np.array(self.saved_it)),
                       ('samples_u', np.array(self.saved_u)),
                       ('logwt', np.array(self.saved_logwt)),
                       ('logl', np.array(self.saved_logl)),
                       ('logvol', np.array(self.saved_logvol)),
                       ('logz', np.array(self.saved_logz)),
                       ('logzerr', np.sqrt(np.array(self.saved_logzvar))),
                       ('h', np.array(self.saved_h))]
        else:
            raise ValueError("You didn't save any samples!")

        if self.save_proposals:
            results.append(('prop', copy.deepcopy(self.prop)))
            results.append(('prop_iter',
                            np.array(self.saved_piter, dtype='int')))
            results.append(('samples_prop',
                            np.array(self.saved_propidx, dtype='int')))
            results.append(('scale', np.array(self.saved_scale)))

        return Results(results)

    def _check_unit_cube(self, point):
        """Check whether a point falls within the unit cube."""

        return np.all(point > 0.) and np.all(point < 1.)

    def _check_h(self, h):
        """Check whether the information is non-negative
        to numerical precision. Numerical error can make it negative in
        pathological corner cases."""

        if h < 0.0:
            if h > -SQRTEPS:
                h = 0.0
            else:
                raise RuntimeError("Negative h encountered (h={}). Please "
                                   "report this as a likely bug.".format(h))

        return h

    def _empty_queue(self):
        """Dump all live point proposals currently on the queue."""

        while self.nqueue > 0:
            f = self.queue.pop()
            self.unused += 1
            self.nqueue -= 1

    def _fill_queue(self, loglstar):
        """Sequentially add new live point proposals to the queue."""

        point_queue = []
        axes_queue = []
        while self.nqueue < self.queue_size:
            point, axes = self.propose_point()
            point_queue.append(point)
            axes_queue.append(axes)
            self.nqueue += 1
        loglstars = [loglstar for i in range(self.queue_size)]
        scales = [self.scale for i in range(self.queue_size)]
        rstates = [self.rstate for i in range(self.queue_size)]
        ptforms = [self.prior_transform for i in range(self.queue_size)]
        logls = [self.loglikelihood for i in range(self.queue_size)]
        kwargs = [self.kwargs for i in range(self.queue_size)]

        args = zip(point_queue, loglstars, axes_queue,
                   scales, rstates, ptforms, logls, kwargs)
        self.queue = self.M(self.evolve_point, args)

    def _get_point_value(self, loglstar):
        """Get a live point proposal sequentially from the filled queue."""

        if self.nqueue == 0:
            self._fill_queue(loglstar)
        u, v, logl, nc, blob = self.queue.pop(0)
        self.nqueue -= 1
        self.used += 1

        return u, v, logl, nc, blob

    def _new_point(self, loglstar):
        """Propose points until a new point that satisfies the likelihood
        constraint is found."""

        ncall = 0
        while True:
            u, v, logl, nc, blob = self._get_point_value(loglstar)
            ncall += nc
            if self.nqueue == 0:
                self.update_proposal(blob)
            if logl >= loglstar:
                break

        return u, v, logl, ncall

    def add_live_points(self):
        """Add the remaining set of live points to the current set of dead
        points to avoid wasting samples. Instantiates a generator object
        that will be called by the user with the same outputs as `sample`."""

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
        logvols = self.saved_logvol[-1]
        logvols += np.log(1. - (np.arange(self.nlive)+1.) / (self.nlive+1.))
        logvols_pad = np.concatenate(([self.saved_logvol[-1]], logvols))
        logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-1], logvols_pad[1:]],
                                  axis=1, b=np.c_[np.ones(self.nlive),
                                                  -np.ones(self.nlive)])
        logdvols += math.log(0.5)

        # Getting changes in logvol to weight new contributions to `logzvar`.
        dlvs = logvols_pad[:-1] - logvols_pad[1:]

        # Add contributions from the remaining live points in order
        # from the lowest to the highest loglikelihoods.
        lsort_idx = np.argsort(self.live_logl)
        logz = self.saved_logz[-1]
        logzvar = self.saved_logzvar[-1]
        h = self.saved_h[-1]
        loglstar = self.saved_logl[-1]
        loglmax = max(self.live_logl)
        for i in range(self.nlive):
            idx = lsort_idx[i]
            logvol, logdvol, dlv = logvols[i], logdvols[i], dlvs[i]
            ustar = np.array(self.live_u[idx])
            vstar = np.array(self.live_v[idx])
            loglstar_new = self.live_logl[idx]
            propidx = self.live_prop[idx]
            point_it = self.live_it[idx]
            logwt = np.logaddexp(loglstar_new, loglstar) + logdvol
            logz_new = np.logaddexp(logz, logwt)
            lzterm = (math.exp(loglstar - logz_new) * loglstar +
                      math.exp(loglstar_new - logz_new) * loglstar_new)
            h_new = (math.exp(logdvol) * lzterm +
                     math.exp(logz - logz_new) * (h + logz) -
                     logz_new)
            h_new = self._check_h(h_new)
            dh = h_new - h
            h = h_new
            logz = logz_new
            logzvar += dh * dlv
            loglstar = loglstar_new
            logz_remain = loglmax + logvol
            delta_logz = np.logaddexp(logz, logz_remain) - logz
            if self.save_samples:
                self.saved_id.append(idx)
                self.saved_u.append(ustar)
                self.saved_v.append(vstar)
                self.saved_logl.append(loglstar)
                self.saved_logvol.append(logvol)
                self.saved_logwt.append(logwt)
                self.saved_logz.append(logz)
                self.saved_logzvar.append(logzvar)
                self.saved_h.append(h)
                self.saved_nc.append(1)
                self.saved_propidx.append(propidx)
                self.saved_it.append(point_it)
                self.saved_piter.append(self.nprop - 1)
                self.saved_scale.append(self.scale)
            self.eff = 100. * (self.it + i) / self.ncall
            yield (idx, ustar, vstar, loglstar, logvol, logwt,
                   logz, logzvar, h, 1, point_it, propidx, self.eff,
                   delta_logz)

    def _remove_live_points(self):
        """Remove the remaining set of live points previously added to the
        current set of dead points."""

        if self.added_live:
            self.added_live = False
            if self.save_samples:
                del self.saved_id[-self.nlive:]
                del self.saved_u[-self.nlive:]
                del self.saved_v[-self.nlive:]
                del self.saved_logl[-self.nlive:]
                del self.saved_logvol[-self.nlive:]
                del self.saved_logwt[-self.nlive:]
                del self.saved_logz[-self.nlive:]
                del self.saved_logzvar[-self.nlive:]
                del self.saved_h[-self.nlive:]
                del self.saved_nc[-self.nlive:]
                del self.saved_propidx[-self.nlive:]
                del self.saved_it[-self.nlive:]
                del self.saved_piter[-self.nlive:]
                del self.saved_scale[-self.nlive:]
        else:
            raise ValueError("No live points were added to the "
                             "list of samples!")

    def sample(self, maxiter=None, maxcall=None, dlogz=0.5,
               logl_max=np.inf, save_proposals=True, save_samples=True):
        """
        The main nested sampling loop. Iteratively replace the worst live
        point with a sample drawn uniformly from the prior until the
        provided stopping criteria are reached. Instantiates a generator
        object that will be called by the user.

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
            `log(z + z_est) - log(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. Default is *0.5*.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        save_proposals : bool, optional
            Whether or not to save past proposal distributions used to bound
            the live points internally. Default is *True*.

        save_samples : bool, optional
            Whether or not to save past samples from the nested sampling run
            (along with other ancillary quantities) internally.
            Default is *True*.

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : double
            Ln(likelihood) of the sample.

        logvol : double
            Ln(volume) of the prior contained within the sample.

        logwt : double
            Ln(weight) of the sample.

        logz : double
            Cumulative ln(evidence) up to the sample (inclusive).

        logzvar : double
            Associated error on `logz`.

        h : double
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before a new proposed
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        propidx : int
            Index of the proposal the dead point was drawn from.

        eff : float
            The cumulative sampling efficiency (as a *percentage*).

        """

        if maxcall is None:
            maxcall = sys.maxsize
        if maxiter is None:
            maxiter = sys.maxsize

        self.save_samples = save_samples
        self.save_proposals = save_proposals

        ncall = 0

        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1:
            # Initialize values for nested sampling loop.
            h = 0.0  # Information, initially *0.*
            logz = -1.e300  # ln(evidence), initially *0.*
            logzvar = 0.  # var[ln(evidence)], initially *0.*
            logvol = 0.  # initially contains the whole prior (volume=1.)
            loglstar = -1.e300  # initial ln(likelihood)
            delta_logz = 1.e300  # ln(ratio) of total/current evidence

            # Initialize proposal distribution.
            pointvol = 1. / self.nlive
            prop = self.update(pointvol)
            if self.save_proposals:
                self.prop.append(prop)
                self.nprop += 1
            self.since_update = 0
        else:
            # Remove live points (if added) from previous run.
            if self.added_live:
                self._remove_live_points()

            # Get final state from previous run.
            h = self.saved_h[-1]  # Information
            logz = self.saved_logz[-1]  # log(evidence)
            logzvar = self.saved_logzvar[-1]  # var[ln(evidence)]
            logvol = self.saved_logvol[-1]  # log(volume)
            loglstar = min(self.live_logl)  # log(likelihood)
            delta_logz = np.logaddexp(logz, np.max(self.live_logl) +
                                      logvol) - logz  # log-evidence ratio

        # The main nested sampling loop.
        for it in range(sys.maxsize):

            # Stopping criterion 1: current number of iterations
            # exceeds `maxiter`.
            if it > maxiter:
                # If dumping past states, save only the required quantities.
                if not self.save_samples:
                    self.saved_logz.append(logz)
                    self.saved_logzvar.append(logzvar)
                    self.saved_h.append(h)
                    self.saved_logvol.append(logvol)
                    self.saved_logl.append(loglstar)
                break

            # Stopping criterion 2: current number of `loglikelihood`
            # calls exceeds `maxcall`.
            if ncall > maxcall:
                if not self.save_samples:
                    self.saved_logz.append(logz)
                    self.saved_logzvar.append(logzvar)
                    self.saved_h.append(h)
                    self.saved_logvol.append(logvol)
                    self.saved_logl.append(loglstar)
                break

            # Stopping criterion 3: estimated (fractional) remaining evidence
            # lies below some threshold set by `dlogz`.
            logz_remain = np.max(self.live_logl) + logvol
            delta_logz = np.logaddexp(logz, logz_remain) - logz
            if dlogz is not None:
                if delta_logz < dlogz:
                    if not self.save_samples:
                        self.saved_logz.append(logz)
                        self.saved_logzvar.append(logzvar)
                        self.saved_h.append(h)
                        self.saved_logvol.append(logvol)
                        self.saved_logl.append(loglstar)
                    break

            # Stopping criterion 4: last dead point exceeded the upper
            # `logl_max` bound.
            if loglstar > logl_max:
                if not self.save_samples:
                    self.saved_logz.append(logz)
                    self.saved_logzvar.append(logzvar)
                    self.saved_h.append(h)
                    self.saved_logvol.append(logvol)
                    self.saved_logl.append(loglstar)
                break

            # Expected log(volume) shrinkage.
            logvol -= self.dlv

            # After `update_interval` interations have passed,
            # update the sampler using the current set of live points.
            if self.since_update >= self.update_interval:
                pointvol = math.exp(logvol) / self.nlive
                prop = self.update(pointvol)
                if self.save_proposals:
                    self.prop.append(prop)
                self.nprop += 1
                self.since_update = 0

            # Locate the "live" point with the lowest `logl`.
            worst = np.argmin(self.live_logl)  # index
            worst_it = self.live_it[worst]  # when point was proposed
            propidx = self.live_prop[worst]  # associated proposal index

            # Set our new worst likelihood constraint.
            ustar = np.array(self.live_u[worst])  # unit cube position
            vstar = np.array(self.live_v[worst])  # transformed position
            loglstar_new = self.live_logl[worst]  # new likelihood

            # Set our new weight using quadratic estimates (trapezoid rule).
            logdvol = misc.logsumexp(a=[logvol + self.dlv, logvol],
                                     b=[0.5, -0.5])  # log(dvol)
            logwt = np.logaddexp(loglstar_new, loglstar) + logdvol  # log(wt)

            # Sample a new live point from within the likelihood constraint
            # `logl > loglstar` using the proposal distribution
            # from our sampler.
            u, v, logl, nc = self._new_point(loglstar_new)
            ncall += nc
            self.ncall += nc
            self.since_update += nc

            # Update evidence `logz` and information `h`.
            logz_new = np.logaddexp(logz, logwt)
            lzterm = (math.exp(loglstar - logz_new) * loglstar +
                      math.exp(loglstar_new - logz_new) * loglstar_new)
            h_new = (math.exp(logdvol) * lzterm +
                     math.exp(logz - logz_new) * (h + logz) -
                     logz_new)
            h_new = self._check_h(h_new)
            dh = h_new - h
            h = h_new
            logz = logz_new
            logzvar += dh * self.dlv
            loglstar = loglstar_new

            # Save the worst live point. It is now a "dead" point.
            if self.save_samples:
                self.saved_id.append(worst)
                self.saved_u.append(ustar)
                self.saved_v.append(vstar)
                self.saved_logl.append(loglstar)
                self.saved_logvol.append(logvol)
                self.saved_logwt.append(logwt)
                self.saved_logz.append(logz)
                self.saved_logzvar.append(logzvar)
                self.saved_h.append(h)
                self.saved_nc.append(nc)
                self.saved_propidx.append(propidx)
                self.saved_it.append(worst_it)
                self.saved_piter.append(self.nprop - 1)
                self.saved_scale.append(self.scale)

            # Update the live point (previously our "worst" point).
            self.live_u[worst] = u
            self.live_v[worst] = v
            self.live_logl[worst] = logl
            self.live_prop[worst] = self.nprop - 1
            self.live_it[worst] = self.it

            # Compute our sampling efficiency.
            self.eff = 100. * self.it / self.ncall

            # Increment total number of iterations.
            self.it += 1

            # Return dead point and ancillary quantities.
            yield (worst, ustar, vstar, loglstar, logvol, logwt,
                   logz, logzvar, h, nc, worst_it, propidx, self.eff,
                   delta_logz)

    def run_nested(self, maxiter=None, maxcall=None, dlogz=None,
                   add_live=True, print_progress=True, save_proposals=True):
        """
        Iteratively replace the worst live point with a sample drawn
        uniformly from the prior until the provided stopping criteria
        are reached.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is no limit.

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is no limit.

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `log(z + z_est) - log(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. If `add_live` is *True*,
            the default is `ln(nlive + 1)`. Otherwise, the default is *0.5*.

        add_live : bool, optional
            If *True*, adds the remaining set of live points to the list of
            samples at the end of each run. Default is *True*.

        print_progress : bool, optional
            If *True*, outputs a simple summary of the current run that
            updates each iteration. Default is *True*.

        save_proposals : bool, optional
            Whether or not to save past proposal distributions used to bound
            the live points internally. Default is *True*.

        """

        # Run the main nested sampling loop.
        if dlogz is None:
            if add_live:
                dlogz = 0.005 * (self.nlive + 1.)
            else:
                dlogz = 0.01

        ncall = self.nlive
        for it, results in enumerate(self.sample(maxiter=maxiter,
                                     maxcall=maxcall, dlogz=dlogz,
                                     save_proposals=save_proposals,
                                     save_samples=True)):
            (worst, ustar, vstar, loglstar, logvol, logwt,
             logz, logzvar, h, nc, worst_it, propidx, eff,
             delta_logz) = results
            ncall += nc
            if delta_logz > 1e6:
                delta_logz = np.inf
            if logzvar >= 0.:
                logzerr = np.sqrt(logzvar)
            else:
                logzerr = np.nan
            if print_progress:
                i = self.it - 1
                sys.stderr.write("\riter: {:d} | "
                                 "nc: {:d} | "
                                 "ncall: {:d} | "
                                 "eff(%): {:6.3f} | "
                                 "logz: {:6.3f} +/- {:6.3f} | "
                                 "dlogz: {:6.3f} > {:6.3f}    "
                                 .format(i, nc, ncall, eff,
                                         logz, logzerr,
                                         delta_logz, dlogz))

        if add_live:
            it = self.it - 1
            # Add remaining live points to samples.
            for i, results in enumerate(self.add_live_points()):
                (worst, ustar, vstar, loglstar, logvol, logwt,
                 logz, logzvar, h, nc, worst_it, propidx, eff,
                 delta_logz) = results
                if delta_logz > 1e6:
                    delta_logz = np.inf
                if logzvar >= 0.:
                    logzerr = np.sqrt(logzvar)
                else:
                    logzerr = np.nan
                if print_progress:
                    sys.stderr.write("\riter: {:d}+{:d} | "
                                     "nc: {:d} | "
                                     "ncall: {:d} | "
                                     "eff(%): {:6.3f} | "
                                     "logz: {:6.3f} +/- {:6.3f} | "
                                     "dlogz: {:6.3f} < {:6.3f}    "
                                     .format(it, i + 1, nc, ncall, eff,
                                             logz, logzerr,
                                             delta_logz, dlogz))

        if print_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()
