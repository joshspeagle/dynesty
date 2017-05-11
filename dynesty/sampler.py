#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base `Sampler` class containing various helpful functions. Also contains a
`Results` class for storing our results.

"""

from __future__ import (print_function, division)

import sys
import warnings
import math
import scipy.misc as misc
import numpy as np

from .results import *

__all__ = ["Sampler"]


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

    pool: ThreadPoolExecutor
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

        # proposal updates
        self.update_interval = update_interval

        # random state
        self.rstate = rstate

        # parallelism
        self.queue_size = queue_size
        self.pool = pool
        self.queue = []
        self.nqueue = 0
        self.submitted = 0
        self.cancelled = 0
        self.unused = 0
        self.used = 0

        # sampling
        self.it = 1
        self.since_update = 0
        self.ncall = self.nlive
        self.dlv = 1. / self.nlive  # expected logvol shrinkage/iteration
        self.prop = []  # initial states used to compute proposals
        self.prop_iter = []  # iteration when proposal was computed

        # results
        self.saved_id = []  # live point labels
        self.saved_u = []  # unit cube samples
        self.saved_v = []  # transformed variable samples
        self.saved_logl = []  # loglikelihoods of samples
        self.saved_logvol = []  # expected log(volume)
        self.saved_logwt = []  # log(weights)
        self.saved_logz = []  # cumulative log(evidence)
        self.saved_logzerr = []  # cumulative error on log(evidence)
        self.saved_h = []  # cumulative information
        self.saved_nc = []  # number of calls at each iteration

    @property
    def results(self):
        """The full results from the nested sampling run."""

        results = Results([('nlive', self.nlive),
                           ('niter', self.it),
                           ('ncall', np.array(self.saved_nc)),
                           ('eff', self.eff),
                           ('samples', np.array(self.saved_v)),
                           ('samples_id', np.array(self.saved_id)),
                           ('samples_u', np.array(self.saved_u)),
                           ('logwt', np.array(self.saved_logwt)),
                           ('logl', np.array(self.saved_logl)),
                           ('logvol', np.array(self.saved_logvol)),
                           ('logz', np.array(self.saved_logz)),
                           ('logzerr', np.array(self.saved_logzerr)),
                           ('h', np.array(self.saved_h))])

        return results

    def get_proposal(self, it):
        """Given the iteration, returns the proposal distribution."""

        prop_iter = np.array(self.prop_iter)
        idx = np.arange(len(prop_iter))[it >= prop_iter][-1]

        return self.prop[idx]

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
            if f.cancel():
                self.cancelled += 1
            else:
                self.unused += 1
            self.nqueue -= 1

    def _fill_queue(self, loglstar):
        """Sequentially add new live point proposals to the queue."""

        while self.nqueue < self.queue_size:
            self.queue.append(self.pool.submit(self.propose_point, loglstar))
            self.nqueue += 1
            self.submitted += 1

    def _get_point_value(self, loglstar):
        """Get a live point proposal sequentially from the filled queue.
        Afterwards, refill the queue."""

        self._fill_queue(loglstar)
        f = self.queue.pop(0)
        self.nqueue -= 1
        u, v, logl = f.result()
        self.used += 1

        return u, v, logl

    def _new_point(self, loglstar):
        """Propose points until a new point that satisfies the likelihood
        constraint is found."""

        ncall = 0
        while True:
            u, v, logl = self._get_point_value(loglstar)
            ncall += 1
            if logl >= loglstar:
                break

        return u, v, logl, ncall

    def _add_live_points(self):
        """Add the remaining set of live points to the current set of dead
        points to avoid wasting samples."""

        # After N samples have been taken out, the remaining volume is
        # `e^(-N / nlive)`. The remaining points are distributed uniformly
        # within the remaining volume so that the expected volume enclosed
        # by the `i`-th worst likelihood is
        # `e^(-N / nlive) * (nlive + 1 - i) / (nlive + 1)`.
        logvols = self.saved_logvol[-1]
        logvols += np.log(1. - (np.arange(self.nlive)+1.) / (self.nlive+1.))
        logvols_pad = np.concatenate(([self.saved_logvol[-1]], logvols,
                                      [-1.e300]))
        logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-2], logvols_pad[2:]],
                                  axis=1, b=np.c_[np.ones(self.nlive),
                                                  -np.ones(self.nlive)])

        logdvols += math.log(0.5)

        # Add contributions from the remaining live points in order
        # from the lowest to the highest loglikelihoods.
        lsort_idx = np.argsort(self.live_logl)
        logz, h = self.saved_logz[-1], self.saved_h[-1]
        for i in xrange(self.nlive):
            idx = lsort_idx[i]
            logvol, logdvol = logvols[i], logdvols[i]
            ustar, vstar = self.live_u[idx], self.live_v[idx]
            loglstar = self.live_logl[idx]
            logwt = loglstar + logdvol
            logz_new = np.logaddexp(logz, logwt)
            h = (math.exp(logwt - logz_new) * loglstar +
                 math.exp(logz - logz_new) * (h + logz) -
                 logz_new)
            logz = logz_new
            logzerr = math.sqrt(h / self.nlive)
            self.saved_id.append(idx)
            self.saved_u.append(np.array(ustar))
            self.saved_v.append(np.array(vstar))
            self.saved_logl.append(loglstar)
            self.saved_logvol.append(logvol)
            self.saved_logwt.append(logwt)
            self.saved_logz.append(logz)
            self.saved_logzerr.append(logzerr)
            self.saved_h.append(h)
            self.saved_nc.append(1)

    def reset(self):
        """Re-initialize the sampler."""

        # live points
        self.live_u = self.rstate.rand(self.nlive, self.npdim)
        for i in range(self.nlive):
            self.live_v[i, :] = self.prior_transform(self.live_u[i, :])
        self.live_logl = np.fromiter(self.pool.map(self.loglikelihood,
                                     self.live_v), dtype=np.float64)

        # parallelism
        self.queue = []
        self.nqueue = 0
        self.submitted = 0
        self.cancelled = 0
        self.unused = 0
        self.used = 0

        # sampling
        self.it = 1
        self.since_update = 0
        self.ncall = self.nlive
        self.prop = []
        self.prop_iter = []

        # results
        self.saved_id = []
        self.saved_u = []
        self.saved_v = []
        self.saved_logl = []
        self.saved_logvol = []
        self.saved_logwt = []
        self.saved_logz = []
        self.saved_logzerr = []
        self.saved_h = []
        self.saved_nc = []

    def sample(self, maxiter=None, maxcall=None, dlogz=None):
        """
        The main nested sampling loop. Sample an additional number of live
        points based on the current collection of live points until the
        provided stopping criteria are reached.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is no limit.

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is no limit.

        dlogz : float, optional
            If supplied, iteration will stop when the estimated contribution
            of the remaining prior volume to the total evidence falls below
            this threshold. Explicitly, the stopping criterion is
            `log(z + z_est) - log(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. Default is *0.5*.

        """

        # Establish stopping criteria.
        if maxiter is None:
            maxiter = sys.maxsize

        if maxcall is None:
            maxcall = sys.maxsize

        if dlogz is None:
            dlogz = 0.5

        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1:
            # Initialize values for nested sampling loop.
            h = 0.0  # Information, initially *0.*
            logz = -1.e300  # log(evidence), initially *0.*
            logvol = 0.  # initially contains the whole prior (volume=1.)

            # Initialize proposal distribution.
            pointvol = 1. / self.nlive
            self.prop.append(self.update(pointvol))
            self.prop_iter.append(self.it)
            self.since_update = 0
        else:
            # Remove addition of leftover live points from previous run since
            # we'll be using them now.
            del self.saved_id[-self.nlive:]
            del self.saved_u[-self.nlive:]
            del self.saved_v[-self.nlive:]
            del self.saved_logl[-self.nlive:]
            del self.saved_logvol[-self.nlive:]
            del self.saved_logwt[-self.nlive:]
            del self.saved_logz[-self.nlive:]
            del self.saved_logzerr[-self.nlive:]
            del self.saved_h[-self.nlive:]
            del self.saved_nc[-self.nlive:]

            # Grab last dead point.
            h = self.saved_h[-1]  # Information
            logz = self.saved_logz[-1]  # log(evidence)
            logvol = self.saved_logvol[-1]  # log(volume)

        # The main nested sampling loop.
        ncall = 0  # current number of loglikelihood calls
        for it in xrange(maxiter):

            # Stopping criterion 1: number of `loglikelihood` calls
            # exceeds `maxcall`.
            if ncall > maxcall:
                break

            # Stopping criterion 2: estimated (fractional) remaining evidence
            # lies below some threshold set by `dlogz`.
            if dlogz is not None:
                logz_remain = np.max(self.live_logl) - self.it / self.nlive
                if np.logaddexp(logz, logz_remain) - logz < dlogz:
                    break

            # After `update_interval` interations have passed,
            # update the sampler using the current set of live points.
            if self.since_update >= self.update_interval:
                expected_vol = math.exp(-self.it / self.nlive)
                pointvol = expected_vol / self.nlive
                self.prop.append(self.update(pointvol))
                self.prop_iter.append(self.it)
                self.since_update = 0

            # Locate the "live" point with the lowest `logl`.
            worst = np.argmin(self.live_logl)

            # Set our new worst likelihood constraint.
            ustar, vstar = self.live_u[worst], self.live_v[worst]  # position
            loglstar = self.live_logl[worst]  # likelihood

            # Set our new weight using quadratic estimates for dvol.
            logvol -= self.dlv  # expected log(volume) shrinkage
            logdvol = misc.logsumexp(a=[logvol + self.dlv, logvol - self.dlv],
                                     b=[0.5, -0.5])  # log(dvol)
            logwt = loglstar + logdvol  # log(weight)

            # Sample a new live point from within the likelihood constraint
            # `logl > loglstar` using the proposal distribution
            # from our sampler.
            u, v, logl, nc = self._new_point(loglstar)
            ncall += nc
            self.ncall += nc
            self.since_update += nc
            self.saved_nc.append(nc)

            # Add the worst live point to samples. It is now a "dead" point.
            self.saved_id.append(worst)
            self.saved_u.append(np.array(ustar))
            self.saved_v.append(np.array(vstar))
            self.saved_logl.append(loglstar)
            self.saved_logvol.append(logvol)
            self.saved_logwt.append(logwt)

            # Update evidence `logz` and information `h` using our
            # new dead point.
            logz_new = np.logaddexp(logz, logwt)
            h = (math.exp(logwt - logz_new) * loglstar +
                 math.exp(logz - logz_new) * (h + logz) -
                 logz_new)
            h = self._check_h(h)
            logz = logz_new
            logzerr = math.sqrt(h / self.nlive)
            self.saved_logz.append(logz)
            self.saved_logzerr.append(logzerr)
            self.saved_h.append(h)

            # Update the live point (previously our "worst" point).
            self.live_u[worst] = u
            self.live_v[worst] = v
            self.live_logl[worst] = logl

            self.it += 1  # increment total number of iterations

        # Add remaining live points to samples.
        self._add_live_points()

        # Compute our sampling efficiency.
        self.eff = 100. * self.it / (self.ncall - self.nlive)
