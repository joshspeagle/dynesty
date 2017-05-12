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
        self.queue = []  # proposed live point queue
        self.nqueue = 0  # current size of the queue
        self.submitted = 0  # total number of submitted jobs
        self.cancelled = 0  # total number of cancelled jobs
        self.unused = 0  # total number of unused cores
        self.used = 0  # total number of used cores

        # sampling
        self.it = 1  # current iteration
        self.since_update = 0  # number of calls since the last update
        self.ncall = self.nlive  # number of function calls
        self.dlv = 1. / self.nlive  # expected logvol shrinkage/iteration
        self.prop = []  # initial states used to compute proposals
        self.prop_iter = []  # iteration when proposal was computed
        self.added_live = False  # whether leftover live points were used

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
        self.added_live = False

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

    @property
    def results(self):
        """The full results from the nested sampling run."""

        results = Results([('nlive', self.nlive),
                           ('niter', self.it - 1),
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

        if self.prop_iter:
            raise ValueError("No proposals are currently saved!")

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
        u, v, logl, nc, blob = f.result()
        self.used += 1

        return u, v, logl, nc, blob

    def _new_point(self, loglstar):
        """Propose points until a new point that satisfies the likelihood
        constraint is found."""

        ncall = 0
        while True:
            u, v, logl, nc, blob = self._get_point_value(loglstar)
            ncall += nc
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
            ustar = np.array(self.live_u[idx])
            vstar = np.array(self.live_v[idx])
            loglstar = self.live_logl[idx]
            logwt = loglstar + logdvol
            logz_new = np.logaddexp(logz, logwt)
            h = (math.exp(logwt - logz_new) * loglstar +
                 math.exp(logz - logz_new) * (h + logz) -
                 logz_new)
            logz = logz_new
            logzerr = math.sqrt(h / self.nlive)
            self.saved_id.append(idx)
            self.saved_u.append(ustar)
            self.saved_v.append(vstar)
            self.saved_logl.append(loglstar)
            self.saved_logvol.append(logvol)
            self.saved_logwt.append(logwt)
            self.saved_logz.append(logz)
            self.saved_logzerr.append(logzerr)
            self.saved_h.append(h)
            self.saved_nc.append(1)
            yield (idx, ustar, vstar, loglstar, logvol, logwt,
                   logz, logzerr, h, 1)

    def _remove_live_points(self):
        """Remove the remaining set of live points previously added to the
        current set of dead points."""

        if self.added_live:
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
            self.added_live = False
        else:
            raise ValueError("No live points were added to the "
                             "list of samples!")

    def sample(self, maxiter=None, maxcall=None, dlogz=None,
               save_proposals=True, save_samples=True):
        """
        The main nested sampling loop. Iteratively replace the worst live
        point with a sample drawn uniformly from the prior until the
        provided stopping criteria are reached. Instantiates a generator
        object that will be called by the user.

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
            contribution from the remaining volume. Default is *0.5*.

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

        logzerr : double
            Associated error on `logz`.

        h : double
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before a new proposed
            live point was accepted.

        """

        # Establish stopping criteria.
        if maxiter is None:
            maxiter = sys.maxsize

        if maxcall is None:
            maxcall = sys.maxsize

        if dlogz is None:
            dlogz = 0.5

        ncall = 0

        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1:
            # Initialize values for nested sampling loop.
            h = 0.0  # Information, initially *0.*
            logz = -1.e300  # log(evidence), initially *0.*
            logvol = 0.  # initially contains the whole prior (volume=1.)

            # Initialize proposal distribution.
            pointvol = 1. / self.nlive
            prop = self.update(pointvol)
            if save_proposals:
                self.prop.append(prop)
                self.prop_iter.append(self.it)
            self.since_update = 0
        else:
            # Remove live points added from previous run.
            if self.added_live:
                self._remove_live_points()

            # Get final state from previous run.
            h = self.saved_h[-1]  # Information
            logz = self.saved_logz[-1]  # log(evidence)
            logvol = self.saved_logvol[-1]  # log(volume)

        # The main nested sampling loop.
        for it in xrange(sys.maxsize):

            # Stopping criterion 1: current number of iterations
            # exceeds `maxiter`.
            if it > maxiter:
                # If dumping past states, save only the required quantities.
                if not save_samples:
                    self.saved_logz.append(logz)
                    self.saved_h.append(h)
                    self.saved_logvol.append(logvol)
                break

            # Stopping criterion 2: current number of `loglikelihood`
            # calls exceeds `maxcall`.
            if ncall > maxcall:
                if not save_samples:
                    self.saved_logz.append(logz)
                    self.saved_h.append(h)
                    self.saved_logvol.append(logvol)
                break

            # Stopping criterion 3: estimated (fractional) remaining evidence
            # lies below some threshold set by `dlogz`.
            if dlogz is not None:
                logz_remain = np.max(self.live_logl) - self.it / self.nlive
                if np.logaddexp(logz, logz_remain) - logz < dlogz:
                    if not save_samples:
                        self.saved_logz.append(logz)
                        self.saved_h.append(h)
                        self.saved_logvol.append(logvol)
                    break

            # After `update_interval` interations have passed,
            # update the sampler using the current set of live points.
            if self.since_update >= self.update_interval:
                expected_vol = math.exp(-self.it / self.nlive)
                pointvol = expected_vol / self.nlive
                prop = self.update(pointvol)
                if save_proposals:
                    self.prop.append(prop)
                    self.prop_iter.append(self.it)
                self.since_update = 0

            # Locate the "live" point with the lowest `logl`.
            worst = np.argmin(self.live_logl)

            # Set our new worst likelihood constraint.
            ustar = np.array(self.live_u[worst])  # unit cube position
            vstar = np.array(self.live_v[worst])  # transformed position
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

            # Update evidence `logz` and information `h`.
            logz_new = np.logaddexp(logz, logwt)
            h = (math.exp(logwt - logz_new) * loglstar +
                 math.exp(logz - logz_new) * (h + logz) -
                 logz_new)
            h = self._check_h(h)
            logz = logz_new
            logzerr = math.sqrt(h / self.nlive)

            # Save the worst live point. It is now a "dead" point.
            if save_samples:
                self.saved_id.append(worst)
                self.saved_u.append(ustar)
                self.saved_v.append(vstar)
                self.saved_logl.append(loglstar)
                self.saved_logvol.append(logvol)
                self.saved_logwt.append(logwt)
                self.saved_logz.append(logz)
                self.saved_logzerr.append(logzerr)
                self.saved_h.append(h)
                self.saved_nc.append(nc)

            # Update the live point (previously our "worst" point).
            self.live_u[worst] = u
            self.live_v[worst] = v
            self.live_logl[worst] = logl

            # Compute our sampling efficiency.
            self.eff = 100. * self.it / (self.ncall - self.nlive)

            # Increment total number of iterations.
            self.it += 1

            # Return dead point and ancillary quantities.
            yield (worst, ustar, vstar, loglstar, logvol, logwt,
                   logz, logzerr, h, nc)

    def run_nested(self, maxiter=None, maxcall=None, dlogz=None,
                   add_live=True, print_progress=True):
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
            contribution from the remaining volume. Default is *0.5*.

        add_live : bool, optional
            If *True*, adds the remaining set of live points to the list of
            samples at the end of each run. Default is *True*.

        print_progress : bool, optional
            If *True*, outputs a simple summary of the current run that
            updates each iteration. Default is *True*.

        """

        # Run the main nested sampling loop.
        ncall = self.nlive
        for it, results in enumerate(self.sample(maxiter=maxiter,
                                     maxcall=maxcall, dlogz=dlogz,
                                     save_proposals=True,
                                     save_samples=True)):
            if print_progress:
                (worst, ustar, vstar, loglstar, logvol, logwt,
                 logz, logzerr, h, nc) = results
                ncall += nc
                sys.stderr.write("\riter: {:d} | "
                                 "nc: {:d} | "
                                 "ncall: {:d} | "
                                 "logz: {:6.3f} +/- {:6.3f}"
                                 .format(it + 1, nc, ncall, logz, logzerr))

        if add_live:
            # Add remaining live points to samples.
            for i, results in enumerate(self.add_live_points()):
                if print_progress:
                    (worst, ustar, vstar, loglstar, logvol, logwt,
                     logz, logzerr, h, nc) = results
                    ncall += nc
                    sys.stderr.write("\riter: {:d}+{:d} | "
                                     "nc: {:d} | "
                                     "ncall: {:d} | "
                                     "logz: {:6.3f} +/- {:6.3f}"
                                     .format(it + 1, i + 1, nc, ncall,
                                             logz, logzerr))

        if print_progress:
            sys.stderr.write("\n")
