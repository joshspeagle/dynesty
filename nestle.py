# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simple implementation of nested sampling routine to evaluate Bayesian
evidence."""

import math
import time
from sys import stdout

import numpy as np
try:
    from scipy.cluster.vq import kmeans2
    HAVE_KMEANS = True
except ImportError:
    HAVE_KMEANS = False


class Ellipsoid(object):
    def __init__(self, ctr, cov, icov, vol):
        self.ctr = ctr    # center coordinates
        self.cov = cov    # covariance
        self.icov = icov  # cov^-1
        self.vol = vol    # volume

    def scale_to_vol(self, vol):
        """Expand ellipoid to satisfy a target volume."""
        n = len(self.ctr)
        f = (vol / self.vol) ** (1.0 / n)
        self.cov *= f
        self.icov /= f
        self.vol = vol

    def scale_to_min_vol(self, vol):
        """Expand ellipoid to satisfy a target volume."""
        if self.vol > vol:
            return
        self.scale_to_vol(vol)

    def scale_vol(self, f):
        """Increase volume by a factor f"""
        g = f**(1.0 / len(self.ctr))
        self.cov *= g
        self.icov /= g
        self.vol *= f

    def expand(self, f):
        """Expand the ellipsoid by a factor f, increasing the volume by f^n"""
        n = len(self.ctr)
        self.cov *= f
        self.icov /= f
        self.vol = self.vol * f**n


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


# only needed for multi-ellipsoid method
def choice(p):
    """replacement for numpy.random.choice (only in numpy 1.7+)"""

    r = np.random.random() * sum(p)
    i = 0
    t = p[i]
    while t < r:
        i += 1
        t += p[i]
    return i


def randsphere(n):
    """Draw a random point within a n-dimensional unit sphere"""

    z = np.random.randn(n)
    return z * np.random.rand()**(1./n) / np.sqrt(np.sum(z**2))


def ellipsoid_volume(scaled_cov):
    """
    Parameters
    ----------
    scaled_cov : (ndim, ndim) ndarray
        Scaled covariance matrix.

    Returns
    -------
    volume : float
    """
    vol = np.sqrt(np.linalg.det(scaled_cov))

    # proportionality constant depending on dimension
    # for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
    # for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
    ndim = len(scaled_cov)
    if ndim % 2 == 0:
        i = 2
        while i <= ndim:
            vol *= (2. / i * np.pi)
            i += 2
    else:
        vol *= 2.
        i = 3
        while i <= ndim:
            vol *= (2. / i * np.pi)
            i += 2

    return vol


def bounding_ellipsoid(x):
    """Calculate bounding ellipsoid containing all samples x.

    Parameters
    ----------
    x : (nobj, ndim) ndarray
        Coordinates of points.

    Returns
    -------
    ellipsoid : Ellipsoid
        Attributes are:
        * ``ctr`` ndarray of shape (ndim,)
        * ``cov`` ndarray of shape (ndim, ndim)
          (f * C) which is the covariance of the data points, C,
          times an enlargement factor, f, that ensures that the ellipse
          defined by ``x^T <dot> (fC)^{-1} <dot> x <= 1`` encloses
          all points in the input set.
        * ``icov`` Inverse of cov.
        * ``vol`` Ellipse volume.
    """

    ctr = np.mean(x, axis=0)
    delta = x - ctr
    cov = np.cov(delta, rowvar=0)
    icov = np.linalg.inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # The line below should be equilvalent to:
    #
    #     f = np.empty(len(x), dtype=np.float)
    #     for i in range(len(x)):
    #         f[i] = np.dot(np.dot(delta[i,:], icov), delta[i,:])
    #
    # but without the loop.
    f = np.einsum('...i, ...i', np.tensordot(delta, icov, axes=1), delta)

    fmax = np.max(f)
    cov *= fmax
    icov /= fmax
    vol = ellipsoid_volume(cov)

    return Ellipsoid(ctr, cov, icov, vol)


# only needed for multi-ellipsoid method
def bounding_ellipsoids(x, min_vol=None, ellipsoid=None):
    """Calculate a set of ellipses that bound the points.

    Parameters
    ----------
    x : (nobj, ndim) ndarray
        Coordinates of points.
    min_vol : float
        Minimum allowed volume of ellipses enclosing points.
    ellipsoid : Ellipsoid, optional
        If known, the bounding ellipsoid of the points `x`. If not supplied,
        it will be calculated. This option is used when the function is
        called recursively.

    Returns
    -------
    ellipsoids : list of 2-tuples
        Ellipsoids, each represented by a tuple: ``(scaled_cov, x_mean)``
    """

    ellipsoids = []
    nobj, ndim = x.shape

    # If we don't already have a bounding ellipse for the points,
    # calculate it, and enlarge it so that it has at least the minimum
    # volume.
    if ellipsoid is None:
        ellipsoid = bounding_ellipsoid(x) 
        if min_vol is not None and ellipsoid.vol < min_vol:
            ellipsoid.scale_to_vol(min_vol)

    # Split points into two clusters using k-means clustering with k=2
    # centroid = (2, ndim) ; label = (nobj,)
    # [Each entry in `label` is 0 or 1, corresponding to cluster number]
    centroid, label = kmeans2(x, 2, iter=10)



    # calculate bounding ellipsoid for each cluster
    cluster_x = [None, None]
    cluster_ellipsoids = [None, None]
    cluster_minvols = [None, None]
    for k in [0, 1]:
        cluster_x[k] = x[label == k, :] # points in this cluster
        cluster_ellipsoids[k] = bounding_ellipsoid(cluster_x[k])
        
        print "\nk =", k, "before scaling vol=", cluster_ellipsoids[k].vol
        # enlarge ellipse so that it is at least as large as the fractional
        # volume according to the number of points in the cluster
        if min_vol is not None:
            cluster_minvols[k] = min_vol * len(cluster_x[k]) / float(nobj)
            if cluster_ellipsoids[k].vol < cluster_minvols[k]:
                cluster_ellipsoids[k].scale_to_vol(cluster_minvols[k])

    # debug
    print "\nvol=", ellipsoid.vol, "minvol=", min_vol
    for k in [0, 1]:
        print "    k=", k, "len=", len(cluster_x[k]),
        print "centroid=", centroid[k],
        print "vol=", cluster_ellipsoids[k].vol,
        print "minvol=", cluster_minvols[k]

    # Reassign points between ellipsoids.
    while False:
        h = []
        for k in [0, 1]:
            # Calculate mahalanobis distance between ALL points and the
            # current cluster. The mahalanobis distance squared is given by:
            #     delta = u - v
            #     m = np.dot(np.dot(delta, VI), delta)
            # where, in this case,
            #     VI = (f * C)^-1 = (scaled_cov)^-1
            d = np.empty(len(x), dtype=np.float)
            delta = x - cluster_ellipsoids[k].ctr
            for i in range(len(x)):
                d[i] = np.dot(np.dot(delta[i,:], cluster_ellipsoids[k].icov),
                              delta[i,:])

            # Multiply by ellipse ratio:
            # h_k(point) = V_k(actual) / V_k(expected) * d_k(point)
            # TODO: d is M. distance *squared*. Should it not be squared?
            # TODO: should this even be done? We've already scaled up each
            #       ellipsoid to the minvolume  when we found the bounding
            #       ellipse.

            #if cluster_minvols[k] is not None:
            #    d *= (cluster_ellipsoids[k] / cluster_minvols[k])

            h.append(d)

        # reassign each point to the cluster that gives it the smallest h.
        # Here, we are creating a bool array, h[1] < h[0]
        #     True -> h smaller for #1 -> assign to cluster 1
        #     False -> h smaller for #0 -> assign to cluster 0
        # then the cast to int converts True->1, False->0
        newlabel = (h[1] < h[0]).astype(np.int)

        # If no points were reassigned, exit the loop.
        if np.all(newlabel == label):
            break

        # Otherwise, update the label of the points and recalculate the
        # ellipsoids
        label = newlabel
        for k in [0, 1]:
            cluster_x[k] = x[label == k, :] # points in this cluster
            cluster_ellipsoids[k] = bounding_ellipsoid(cluster_x[k])

            # enlarge ellipse so that it is at least as large as the fractional
            # volume according to the number of points in the cluster
            if min_vol is not None:
                cluster_minvols[k] = min_vol * len(cluster_x[k]) / float(nobj)
                if cluster_ellipsoids[k].vol < cluster_minvols[k]:
                    cluster_ellipsoids[k].scale_to_vol(cluster_minvols[k])

    # if V(E_1) + V(E_2) < V(E) or V(E) > 2V(S):
    # perform entire algorithm on each subset
    if (cluster_ellipsoids[0].vol + cluster_ellipsoids[1].vol < 0.5 * ellipsoid.vol or
        (min_vol is not None and ellipsoid.vol > 2. * min_vol)):
        for k in [0, 1]:
            ellipsoids.extend(
                bounding_ellipsoids(cluster_x[k],
                                    min_vol=cluster_minvols[k],
                                    ellipsoid=cluster_ellipsoids[k]))

    # Otherwise, the full ellipse is fine; just return that.
    else:
        ellipsoids.append(ellipsoid)

    return ellipsoids


def sample_ellipsoid(ellipsoid, nsamples=1):
    """Chose sample(s) randomly distributed within an ellipsoid.
    
    Parameters
    ----------
    scaled_cov : (ndim, ndim) ndarray
        Scaled covariance matrix.
    x_mean : (ndim,) ndarray
        Simple average of all samples.

    Returns
    -------
    x : (nsamples, ndim) array, or (ndim,) array when nsamples == 1
        Coordinates within the ellipsoid.
    """

    # Get scaled eigenvectors (in columns): vs[:,i] is the i-th eigenvector.
    w, v = np.linalg.eig(ellipsoid.cov)
    vs = np.dot(v, np.diag(np.sqrt(w)))

    ndim = len(ellipsoid.ctr)
    if nsamples == 1:
        return np.dot(vs, randsphere(ndim)) + ellipsoid.ctr

    x = np.empty((nsamples, ndim), dtype=np.float)
    for i in range(nsamples):
        x[i, :] = np.dot(vs, randsphere(ndim)) + ellipsoid.ctr
    return x


# only needed for multi-ellipsoid method
def sample_ellipsoids(ellipsoids, nsamples=1):
    """Chose sample(s) randomly distributed within a set of
    (possibly overlapping) ellipsoids.
    
    Parameters
    ----------
    ellipsoids : list of 2-tuples
        Ellipsoids, each represented by a tuple: ``(scaled_cov, x_mean)``

    Returns
    -------
    x : numpy.ndarray (nsamples, ndim) [or (ndim,) when nsamples == 1]
        Coordinates within the ellipsoid. 
    """

    # Select an ellipsoid at random, according to volumes
    v = np.array([e.vol for e in ellipsoids])
    i = choice(v)
    ellipsoid = ellipsoids[i]
    
    # Select a point from the ellipsoid
    x = sample_ellipsoid(ellipsoid)

    # How many ellipsoids is the sample in?
    n = 0
    for ellipsoid in ellipsoids:
        delta = x - ellipsoid.ctr
        n += np.dot(np.dot(delta, ellipsoid.icov), delta) < 1.

    # Only accept the point with probability 1/n 
    if n > 1 and np.random.random() > 1./n:
        return sample_ellipsoids(ellipsoids)
    
    return x


def nest(loglikelihood, prior, npar, nipar=None, nobj=100, maxiter=10000,
         method='single', enlarge=1.5, verbose=False, verbose_name=''):
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
                return 2.0 * u

    npar : int
        Number of parameters returned by prior and accepted by loglikelihood.
    nipar : int, optional
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
    method : {'single', 'multi'}, optional
        Method used to select new points. Choices are
        single-ellipsoidal ('single'), multi-ellipsoidal ('multi'). Default
        is 'single'.
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

    if method is 'multi' and not HAVE_KMEANS:
        raise ValueError("scipy.cluster.vq.kmeans2 required for 'multi' "
                         "method")

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

        expected_vol = math.exp(-it/nobj)

        # calculate the ellipsoid in parameter space that contains all the
        # samples (including the worst one).
        if method == 'single':
            ell = bounding_ellipsoid(objects_u)
            ell.scale_vol(enlarge)
        else:
            ell = bounding_ellipsoids(objects_u, expected_vol * enlarge)

        # choose a point from within the ellipse until it has likelihood
        # better than loglstar
        while True:
            if method == 'single':
                u = sample_ellipsoid(ell)
            else:
                u = sample_ellipsoids(ell)
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
