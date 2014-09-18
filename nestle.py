# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simple implementation of nested sampling routine to evaluate Bayesian
evidence."""

import math
import time
from sys import stdout

import numpy as np


class Result(dict):
    """Represents an optimization result.

    Notes
    -----
    This is a cut and paste from scipy, normally imported with `from
    scipy.optimize import Result`. However, it isn't available in
    scipy 0.9 (or possibly 0.10), so it is included here.
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

def randsphere(n):
    """Draw a random point within a n-dimensional unit sphere"""

    z = np.random.randn(n)
    return z * np.random.rand()**(1./n) / np.sqrt(np.sum(z**2))


def ellipsoid(X, expand=1.):
    """
    Calculate ellipsoid containing all samples X.

    Parameters
    ----------
    X : (nobj, ndim) ndarray
        Coordinates of points.
    expand : float, optional.
        Expand the ellipsoid by this linear factor. Default is 1, which
        corresponds to an ellipsoid that just barely encloses all the points.
        Note that volume is increased by a factor of ``(expand)**ndim``

    Returns
    -------
    vs : (ndim, ndim) ndarray
        Scaled eigenvectors (in columns): vs[:,i] is the i-th eigenvector.
    mean : (ndim,) ndarray
        Simple average of all samples.

    Notes
    -----
    For the 2-d case, to verify that the generated ellipse encloses all
    the points, the ellipse can be plotted using matplotlib on an existing
    Axes  ``ax`` as follows:

        from matplotlib.patches import Ellipse

        width = np.sqrt(np.sum(vs[:,1]**2)) * 2.
        height = np.sqrt(np.sum(vs[:,0]**2)) * 2.
        angle = math.atan(vs[1,1] / vs[0,1]) * 180./math.pi
        e = Ellipse(mean, width, height, angle)
        e.set_facecolor('None')
        ax.add_artist(e)

    To draw the vectors ``vs``:

        for i in [0,1]:
            plt.arrow(mean[0], mean[1], vs[0, i], vs[1, i])
    """

    X_avg = np.mean(X, axis=0)
    Xp = X - X_avg
    c = np.cov(Xp, rowvar=0)
    cinv = np.linalg.inv(c)
    w, v = np.linalg.eig(c)
    vs = np.dot(v, np.diag(np.sqrt(w)))  # scaled eigenvectors

    # Calculate 'k' factor
    k = np.empty(len(X), dtype=np.float)

    # The lines below should be equivalent to:
    # for i in range(len(k)):
    #     k[i] = np.dot(np.dot(Xp[i,:], cinv), Xp[i,:])
    tmp = np.tensordot(Xp, cinv, axes=1)
    for i in range(len(k)):
        k[i] = np.dot(tmp[i, :], Xp[i, :])

    k = np.max(k)

    return np.sqrt(k) * expand * vs, X_avg


def sample_ellipsoid(vs, mean, nsamples=1):
    """Chose sample(s) randomly distributed within an ellipsoid.

    Parameters
    ----------
    vs : (ndim, ndim) ndarray
        Scaled eigenvectors (in columns): vs[:,i] is the i-th eigenvector.
    mean : (ndim,) ndarray
        Simple average of all samples.

    Returns
    -------
    x : (nsamples, ndim) array, or (ndim,) array when nsamples == 1
        Coordinates within the ellipsoid.
    """

    ndim = len(mean)
    if nsamples == 1:
        return np.dot(vs, randsphere(ndim)) + mean

    x = np.empty((nsamples, ndim), dtype=np.float)
    for i in range(nsamples):
        x[i, :] = np.dot(vs, randsphere(ndim)) + mean
    return x


def nest(loglikelihood, prior, npar, nipar=None, nobj=50, maxiter=10000,
         verbose=False, verbose_name=''):
    """Simple nested sampling algorithm to evaluate Bayesian evidence.

    Parameters
    ----------
    loglikelihood : func
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `npar`.
    prior : func
        Function translating a unit cube to the parameter space according to
        the prior. The input is a 1-d numpy array with length `npar`, where
        each value is in the range [0, 1). The return value should also be a
        1-d numpy array with length `npar`, where each value is a parameter.
        The return value is passed to the loglikelihood function. For example,
        for a 2 parameter model with flat priors in the range [0, 2), the
        function would be

            def prior(u):
                return 2. * u

    npar : int
        Number of parameters returned by prior and accepted by loglikelihood.
    nipar : int
        Number of parameters accepted by prior. This might differ from npar
        in the case where a parameter of loglikelihood is dependent upon
        multiple independently distributed parameters, some of which may be
        nuisance parameters.
    nobj : int, optional
        Number of random samples. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is 50.
    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is 10000. The total number
        of likelihood evaluations will be ``nexplore * niter``.
    verbose : bool, optional
        Print a single line of running total iterations.
    verbose_name : str, optional
        Print this string at start of the iteration line printed when
        verbose=True.

    Returns
    -------
    results : dict
        Containing following keys:

        * ``niter`` (int) number of iterations.
        * ``ncalls`` (int) number of likelihood calls.
        * ``time`` (float) time in seconds.
        * ``logz`` (float) log of evidence.
        * ``logzerr`` (float) error on ``logz``.
        * ``loglmax`` (float) Maximum likelihood of any sample.
        * ``h`` (float) information.
        * ``samples`` (array, shape=(nsamples, npar)) parameter values
          of each sample.
        * ``weights`` (array, shape=(nsamples,)) Weight of each sample.
        * ``logprior`` (array, shape=(nsamples,)) log(Prior volume) of
          each sample.
        * ``logl`` (array, shape=(nsamples,)) log(Likelihood) of each sample.

    Notes
    -----
    This is an implementation of John Skilling's Nested Sampling algorithm,
    following the ellipsoidal sampling algorithm in Shaw et al (2007). Only a
    single ellipsoid is used.

    Sample Weights are ``likelihood * prior_vol`` where
    prior_vol is the fraction of the prior volume the sample represents.

    References
    ----------
    http://www.inference.phy.cam.ac.uk/bayesys/
    Shaw, Bridges, Hobson 2007, MNRAS, 378, 1365
    """

    if nipar is None:
        nipar = npar

    # Initialize objects and calculate likelihoods
    objects_u = np.random.random((nobj, nipar))  # position in unit cube
    objects_v = np.empty((nobj, npar), dtype=np.float)  # position in unit cube
    objects_logl = np.empty(nobj, dtype=np.float)  # log likelihood
    for i in range(nobj):
        objects_v[i, :] = prior(objects_u[i, :])
        objects_logl[i] = loglikelihood(objects_v[i, :])

    # Initialize values for nested sampling loop.
    samples_parvals = []  # stored objects for posterior results
    samples_logl = []
    samples_logprior = []
    samples_logwt = []
    loglstar = None  # ln(Likelihood constraint)
    h = 0.  # Information, initially 0.
    logz = -1.e300  # ln(Evidence Z, initially 0)
    # ln(width in prior mass), outermost width is 1 - e^(-1/n)
    logwidth = math.log(1. - math.exp(-1./nobj))
    loglcalls = nobj  # number of calls we already made

    # Nested sampling loop.
    ndecl = 0
    logwt_old = None
    time0 = time.time()
    for it in range(maxiter):
        if verbose:
            if logz > -1.e6:
                print "\r{0} iter={1:6d} logz={2:8f}".format(verbose_name, it,
                                                             logz),
            else:
                print "\r{0} iter={1:6d} logz=".format(verbose_name, it),
            stdout.flush()

        # worst object in collection and its weight (= width * likelihood)
        worst = np.argmin(objects_logl)
        logwt = logwidth + objects_logl[worst]

        # update evidence Z and information h.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * objects_logl[worst] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Add worst object to samples.
        samples_parvals.append(np.array(objects_v[worst]))
        samples_logwt.append(logwt)
        samples_logprior.append(logwidth)
        samples_logl.append(objects_logl[worst])

        # The new likelihood constraint is that of the worst object.
        loglstar = objects_logl[worst]

        # calculate the ellipsoid in parameter space that contains all the
        # samples (including the worst one).
        vs, mean = ellipsoid(objects_u, expand=1.06)

        # choose a point from within the ellipse until it has likelihood
        # better than loglstar
        while True:
            u = sample_ellipsoid(vs, mean)
            if np.any(u < 0.) or np.any(u > 1.):
                continue
            v = prior(u)
            logl = loglikelihood(v)
            loglcalls += 1

            # Accept if and only if within likelihood constraint.
            if logl > loglstar:
                objects_u[worst] = u
                objects_v[worst] = v
                objects_logl[worst] = logl
                break

        # Shrink interval
        logwidth -= 1./nobj

        # stop when the logwt has been declining for more than nobj* 2
        # or niter/4 consecutive iterations.
        if logwt < logwt_old:
            ndecl += 1
        else:
            ndecl = 0
        if ndecl > nobj * 2 and ndecl > it / 6:
            break
        logwt_old = logwt

    tottime = time.time() - time0
    if verbose:
        print 'calls={0:d} time={1:7.3f}s'.format(loglcalls, tottime)

    # Add remaining objects.
    # After N samples have been taken out, the remaining width is e^(-N/nobj)
    # The remaining width for each object is e^(-N/nobj) / nobj
    # The log of this for each object is:
    # log(e^(-N/nobj) / nobj) = -N/nobj - log(nobj)
    logwidth = -len(samples_parvals) / nobj - math.log(nobj)
    for i in range(nobj):
        logwt = logwidth + objects_logl[i]
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * objects_logl[i] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        samples_parvals.append(np.array(objects_v[i]))
        samples_logwt.append(logwt)
        samples_logl.append(objects_logl[i])
        samples_logprior.append(logwidth)

    return Result([
        ('niter', it + 1),
        ('ncall', loglcalls),
        ('time', tottime),
        ('logz', logz),
        ('logzerr', math.sqrt(h / nobj)),
        ('loglmax', np.max(objects_logl)),
        ('h', h),
        ('samples', np.array(samples_parvals)),  # (nsamp, npar)
        ('weights', np.exp(np.array(samples_logwt) - logz)),  # (nsamp,)
        ('logprior', np.array(samples_logprior)),  # (nsamp,)
        ('logl', np.array(samples_logl))  # (nsamp,)
        ])
