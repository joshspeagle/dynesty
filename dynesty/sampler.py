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

__all__ = ["Results", "Sampler"]


class Results(dict):
    """
    Contains the output of a dynamic nested sampling run.

    Since this class is essentially a subclass of dict with attribute
    accessors, one can see which attributes are available using the
    `keys()` method.

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def summary(self):
        """Return a formatted string giving a quick summary
        of the results."""

        return ("nlive: {:d}\n"
                "niter: {:d}\n"
                "ncall: {:d}\n"
                "eff(%): {:6.3f}\n"
                "nsamples: {:d}\n"
                "logz: {:6.3f} +/- {:6.3f}\n"
                "h: {:6.3f}"
                .format(self.nlive, self.niter, self.ncall, self.eff,
                        len(self.samples), self.logz, self.logzerr,
                        self.h))


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

        # results
        self.saved_id = []  # live point labels
        self.saved_u = []  # unit cube samples
        self.saved_v = []  # transformed variable samples
        self.saved_logl = []  # loglikelihoods of samples
        self.saved_logvol = []  # expected log(volume)
        self.saved_logwt = []  # log(weights)
        self.saved_logz = []  # cumulative log(evidence)
        self.saved_h = []  # cumulative information

    def check_unit_cube(self, point):
        """Check whether a point falls within the unit cube."""

        return np.all(point > 0.) and np.all(point < 1.)

    def empty_queue(self):
        """Dump all live point proposals currently on the queue."""

        while self.nqueue > 0:
            f = self.queue.pop()
            if f.cancel():
                self.cancelled += 1
            else:
                self.unused += 1
            self.nqueue -= 1

    def fill_queue(self):
        """Sequentially add new live point proposals to the queue."""

        while self.nqueue < self.queue_size:
            self.queue.append(self.pool.submit(self.propose_point))
            self.nqueue += 1
            self.submitted += 1

    def get_point_value(self):
        """Get a live point proposal sequentially from the filled queue.
        Afterwards, refill the queue."""

        f = self.queue.pop(0)
        self.nqueue -= 1
        u, v, logl = f.result()
        self.fill_queue()
        self.used += 1

        return u, v, logl

    def new_point(self, loglstar):
        """Propose points until a new point that satisfies the likelihood
        constraint is found."""

        ncall = 0
        while True:
            u, v, logl = self.get_point_value()
            ncall += 1
            if logl >= loglstar:
                break

        return u, v, logl, ncall

    def get_results(self):

        logz, h = self.saved_logz[-1], self.saved_h[-1]
        logzerr = math.sqrt(h / self.nlive)

        results = Results([('nlive', self.nlive),
                           ('niter', self.it),
                           ('ncall', self.ncall),
                           ('eff', self.eff),
                           ('logz', logz),
                           ('logzerr', logzerr),
                           ('h', h),
                           ('samples', np.array(self.saved_v)),
                           ('logwt', np.array(self.saved_logwt)),
                           ('logl', np.array(self.saved_logl))])

        return results

    def sample(self, maxiter=None, maxcall=None, dlogz=None,
               decline_factor=None):
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
            contribution from the remaining volume. This option and
            decline_factor are mutually exclusive. Default is *0.5*.

        decline_factor : float, optional
            If supplied, iteration will stop when the sample weights
            (likelihood times prior volume) of newly saved samples has been
            declining for `decline_factor * nsamples` consecutive samples.
            A value of *1.0* works well for most cases. This option and
            `dlogz` are mutually exclusive. If not specified, the default
            `dlogz` criterion is used.

        """

        # Establish stopping criteria.
        if maxiter is None:
            maxiter = sys.maxsize

        if maxcall is None:
            maxcall = sys.maxsize

        if dlogz is not None and decline_factor is not None:
            raise ValueError("Cannot specify two separate stopping criteria: "
                             "decline_factor and dlogz")
        elif dlogz is None and decline_factor is None:
            dlogz = 0.5

        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1:
            # Initialize values for nested sampling loop.
            h = 0.0  # Information, initially *0.*
            logz = -1.e300  # log(evidence), initially *0.*
            logvol = 0.  # initially contains the whole prior (volume=1.)
            logwt_old = -np.inf  # initially weight = 0.

            # Initialize proposal distribution.
            pointvol = 1. / self.nlive
            self.update(pointvol)
            self.since_update = 0
        else:
            # Remove final addition of leftover live points.
            del self.saved_id[-self.nlive:]
            del self.saved_u[-self.nlive:]
            del self.saved_v[-self.nlive:]
            del self.saved_logl[-self.nlive:]
            del self.saved_logvol[-self.nlive:]
            del self.saved_logwt[-self.nlive:]
            del self.saved_logz[-self.nlive:]
            del self.saved_h[-self.nlive:]

            # Grab last dead point.
            h = self.saved_h[-1]  # Information
            logz = self.saved_logz[-1]  # log(evidence)
            logvol = self.saved_logvol[-1]  # log(volume)
            logwt_old = self.saved_logwt[-1]  # log(weight)

        # The main nested sampling loop.
        ndecl = 0  # previous number of declining weights
        it = 0  # current iteration
        while True:

            # After `update_interval` interations have passed,
            # update the sampler using the current set of live points.
            if self.since_update >= self.update_interval:
                expected_vol = math.exp(-self.it / self.nlive)
                pointvol = expected_vol / self.nlive
                self.update(pointvol)
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
            u, v, logl, nc = self.new_point(loglstar)
            self.ncall += nc
            self.since_update += nc

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
            logz = logz_new
            self.saved_logz.append(logz)
            self.saved_h.append(h)

            # Update the live point (previously our "worst" point).
            self.live_u[worst] = u
            self.live_v[worst] = v
            self.live_logl[worst] = logl

            # Stopping criterion 1: estimated (fractional) remaining evidence
            # lies below some threshold set by `dlogz`.
            if dlogz is not None:
                logz_remain = np.max(self.live_logl) - self.it / self.nlive
                if np.logaddexp(logz, logz_remain) - logz < dlogz:
                    break

            # Stopping criterion 2: `logwt` has been declining for longer
            # than `decline_factor`.
            if decline_factor is not None:
                if logwt < logwt_old:
                    ndecl += 1
                else:
                    ndecl = 0
                logwt_old = logwt
                if ndecl > decline_factor * self.nlive:
                    break

            # Stopping criterion 3: number of `loglikelihood` calls
            # exceeds `maxcall`.
            if self.ncall > maxcall:
                break

            # Stopping criterion 4: number of iterations exceeds
            # `maxiter`.
            if it >= maxiter:
                break

            it += 1  # increment current number of iterations
            self.it += 1  # increment total number of iterations

        # Add remaining live points to our set of dead points.
        # After N samples have been taken out, the remaining volume is
        # `e^(-N / nlive)`. The remaining points are distributed uniformly
        # within the remaining volume so that the expected volume enclosed
        # by the `i`-th worst likelihood is
        # `e^(-N / nlive) * (nlive + 1 - i) / (nlive + 1)`.
        logvols = -(self.it - 1.) / self.nlive
        logvols += np.log(1. - (np.arange(self.nlive)+1.) / (self.nlive+1.))
        logvols_pad = np.concatenate(([-(self.it - 1.) / self.nlive],
                                      logvols, [-1e300]))
        logdvols = misc.logsumexp(a=np.c_[logvols_pad[:-2], logvols_pad[2:]],
                                  axis=1, b=np.c_[np.ones(self.nlive),
                                                  -np.ones(self.nlive)])

        logdvols += math.log(0.5)
        for i in xrange(self.nlive):
            logvol, logdvol = logvols[i], logdvols[i]
            ustar, vstar = self.live_u[i], self.live_v[i]
            loglstar = self.live_logl[i]
            logwt = loglstar + logdvol
            logz_new = np.logaddexp(logz, logwt)
            h = (math.exp(logwt - logz_new) * loglstar +
                 math.exp(logz - logz_new) * (h + logz) -
                 logz_new)
            logz = logz_new
            self.saved_id.append(i)
            self.saved_u.append(np.array(ustar))
            self.saved_v.append(np.array(vstar))
            self.saved_logl.append(loglstar)
            self.saved_logvol.append(logvol)
            self.saved_logwt.append(logwt)
            self.saved_logz.append(logz)
            self.saved_h.append(h)

        # h should always be nonnegative (we take the sqrt below).
        # Numerical error makes it negative in pathological corner cases
        # such as flat likelihoods. Here we correct those cases to zero.
        if h < 0.0:
            if h > -SQRTEPS:
                h = 0.0
            else:
                raise RuntimeError("Negative h encountered (h={}). Please "
                                   "report this as a likely bug.".format(h))

        # Compute our sampling efficiency.
        self.eff = 100. * self.it / (self.ncall - self.nlive)
