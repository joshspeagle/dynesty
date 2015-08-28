# License is MIT: see LICENSE.md.
"""Nestle: nested sampling routines to evaluate Bayesian evidence."""

from __future__ import print_function, division

import warnings
import math

import numpy as np
try:
    from scipy.cluster.vq import kmeans2
    HAVE_KMEANS = True
except ImportError:
    HAVE_KMEANS = False

__all__ = ["sample", "print_progress"]

EPS = float(np.finfo(np.float64).eps)

# -----------------------------------------------------------------------------
# Helpers

def vol_prefactor(n):
    """Volume constant for an n-dimensional sphere:

    for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
    for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
    """
    if n % 2 == 0:
        f = 1.
        i = 2
        while i <= n:
            f *= (2. / i * math.pi)
            i += 2
    else:
        f = 2.
        i = 3
        while i <= n:
            f *= (2. / i * math.pi)
            i += 2

    return f


def randsphere(n, rstate=np.random):
    """Draw a random point within an n-dimensional unit sphere"""

    z = rstate.randn(n)
    return z * rstate.rand()**(1./n) / np.sqrt(np.sum(z**2))


def random_choice(a, p, rstate=np.random):
    """replacement for numpy.random.choice (only in numpy 1.7+)"""

    if np.sum(p) - 1. > math.sqrt(EPS):  # same tol as in np.random.choice.
        raise ValueError("probabilities do not sum to 1")

    r = rstate.rand()
    i = 0
    t = p[i]
    while t < r:
        i += 1
        t += p[i]
    return i


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


def weightedcov(x, w):
    """Estimate a covariance matrix, given data with weights.

    Implements formula described here:
    https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance
    (see "weighted samples" section)

    Parameters
    ----------
    x : `~numpy.ndarray`
        2-D array containing data samples. Shape is (M, N) where N is the
        number of variables and M is the number of samples or observations.
    w : `~numpy.ndarray`
        1-D array of sample weights. Shape is (M,).

    Returns
    -------
    mean : `~numpy.ndarray`
        Weighted mean of samples.
    cov : `~numpy.ndarray`
        Weighted covariance matrix.
    """

    xmean = np.average(x, weights=w, axis=0)
    xd = x - xmean
    wsum = np.sum(w)
    w2sum = np.sum(w**2)

    cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', w, xd, xd)

    return xmean, cov


# -----------------------------------------------------------------------------
# Ellipsoid

class Ellipsoid(object):
    """An N-ellipsoid.

    Defined by::

        (x - v)^T A (x - v) = 1

    where the vector ``v`` is the center of the ellipse and ``A`` is an N x N
    matrix. Assumes that `A` is symmetric positive definite.

    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape ``(N,)``
        Coordinates of ellipse center. Note that the array is *not* copied.
        This array is never modified internally.
    a : `~numpy.ndarray` with shape ``(N, N)``
        Matrix describing the axes. Note that the array is *not* copied.
        This array can be modified internally.
    """

    def __init__(self, ctr, a):
        self.n = len(ctr)
        self.ctr = ctr    # center coordinates
        self.a = a        # ~ inverse of covariance of points contained
        self.vol = vol_prefactor(self.n) / np.sqrt(np.linalg.det(a))

        # eigenvalues (l) are a^-2, b^-2, ... (lengths of principle axes)
        # eigenvectors (v) are normalized principle axes
        l, v = np.linalg.eigh(a)
        self.axlens = 1. / np.sqrt(l)

        # Scaled eigenvectors are the axes: axes[:,i] is the i-th
        # axis.  Multiplying this matrix by a vector will transform a
        # point inthe unit n-sphere into a point in the ellipsoid.
        self.axes = np.dot(v, np.diag(self.axlens))

    def scale_to_vol(self, vol):
        """Scale ellipoid to satisfy a target volume."""
        f = (vol / self.vol) ** (1.0 / self.n)  # linear factor
        self.a *= f**-2
        self.axlens *= f
        self.axes *= f
        self.vol = vol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis"""
        i = np.argmax(self.axlens)  # which is the major axis?
        v = self.axes[:, i]  # vector to end of major axis
        return self.ctr - v, self.ctr + v

    def contains(self, x):
        """Does the ellipse contain the point?"""
        d = x - self.ctr
        return np.dot(np.dot(d, self.a), d) <= 1.0

    def sample(self, rstate=np.random):
        """Chose a sample randomly distributed within the ellipsoid.

        Returns
        -------
        x : 1-d array
            A single point within the ellipsoid.
        """
        return self.ctr + np.dot(self.axes, randsphere(self.n, rstate=rstate))

    def samples(self, nsamples, rstate=np.random):
        """Chose a sample randomly distributed within the ellipsoid.

        Returns
        -------
        x : (nsamples, ndim) array
            Coordinates within the ellipsoid.
        """

        x = np.empty((nsamples, self.n), dtype=np.float)
        for i in range(nsamples):
            x[i, :] = self.sample(rstate=rstate)
        return x

    def plot(self, ax, show_axes=False):
        """Plot the ellipse on a matplotlib Axes object. (Ellipsoid must be
        2-d).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes to draw on.
        show_axes : bool, optional
            If true, plot arrows giving the major and minor semi axes of the
            ellipse.

        Returns
        -------
        e : matplotlib.patches.Ellipse
        """

        try:
            from matplotlib.patches import Ellipse
        except ImportError:
            raise ImportError("matplotlib must be installed to use plot "
                              "method.")
    
        # angle of 0-th axis
        theta = 180. / math.pi * math.atan(self.axes[1, 0] / self.axes[0, 0])

        e = Ellipse(self.ctr, 2. * self.axlens[0], 2. * self.axlens[1], theta)
        e.set_facecolor('None')
        ax.add_artist(e)
        if show_axes:
            ax.arrow(self.ctr[0], self.ctr[1],
                     self.axes[0, 0], self.axes[1, 0])
            ax.arrow(self.ctr[0], self.ctr[1],
                     self.axes[0, 1], self.axes[1, 1])
        return e

    def __repr__(self):
        return "Ellipsoid(ctr={})".format(self.ctr)

# -----------------------------------------------------------------------------
# Functions for determining the ellipsoid or set of ellipsoids bounding a
# set of points.

def bounding_ellipsoid(x, pointvol=0.):
    """Calculate bounding ellipsoid containing a set of points x.

    Parameters
    ----------
    x : (npoints, ndim) ndarray
        Coordinates of points.
    pointvol : float, optional
        Volume represented by a single point. Sets a minimum scale for the
        ellipse in all dimensions.

    Returns
    -------
    ellipsoid : Ellipsoid
    """
    npoints, n = x.shape

    # If there is only a single point, return an N-sphere with volume `pointvol`
    # centered at the point.
    if npoints == 1:
        rpoint = (pointvol / vol_prefactor(n))**(1./n)
        return Ellipsoid(x[0], (1./rpoint**2) * np.identity(n))

    # Calculate covariance of points
    ctr = np.mean(x, axis=0)
    delta = x - ctr
    cov = np.cov(delta, rowvar=0)
    
    # when n = 1, np.cov returns a 0-d array. Make it a 1x1 2-d array.
    if n == 1:
        cov = np.atleast_2d(cov)

    # For a ball of uniformly distributed points, the covariance will be
    # smaller than r^2 by a factor of 1/(n+2) [see, e.g.,
    # http://mathoverflow.net/questions/35276/
    # covariance-of-points-distributed-in-a-n-ball]. In nested sampling,
    # we are supposing the points are uniformly distributed within
    # an ellipse, so the same factor holds. Expand `cov`
    # to compensate for that when defining the ellipse matrix:
    cov *= (n + 2)

    # Check if any eigenvalues are zero and if so, increase them. This
    # expands the ellipsoid in cases where it has zero volume, which happens
    # when npoints <= ndim or when enough points are linear combinations
    # of other points. (e.g., npoints = ndim+1 but one point is a linear
    # combination of others).
    w, v = np.linalg.eigh(cov)  # use eigh because cov will be symmetric.
    mask = w < 1.e-10  # TODO: should this just be zero?
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        # target product of all eigenvalues based on target vol = n*pointvol:
        targetprod = (npoints * pointvol / vol_prefactor(n))**2
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
        cov = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov

    # Matrix defining ellipse
    a = np.linalg.inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # Points should obey x^T A x <= 1, so we calculate x^T A x for
    # each point and then scale A up or down to make the
    # "outermost" point obey x^T A x = 1.
    # 
    # The line below should be equilvalent to:
    #
    #     f = np.empty(len(x), dtype=np.float)
    #     for i in range(len(x)):
    #         f[i] = np.dot(np.dot(delta[i,:], icov), delta[i,:])
    #
    f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta)
    fmax = np.max(f)

    # Due to round-off errors, we actually scale the ellipse so the outermost
    # point obeys x^T A x < 1 - (a bit), so that all the points will
    # *definitely* obey x^T A x < 1.
    one_minus_a_bit = 1. - 1e4 * n * EPS  # I'm guessing error scales with n.
                                          # 1e4 was determined from tests.
    if fmax > one_minus_a_bit:
        a *= one_minus_a_bit/fmax

    return Ellipsoid(ctr, a)

# TODO: should
def bounding_ellipsoids(x, pointvol=0., ell=None, debug=False):
    """Calculate a set of ellipses that bound the points.

    Parameters
    ----------
    x : (npoints, ndim) ndarray
        Coordinates of points.
    pointvol : float, optional
        Volume represented by a single point. Used when number of points
        per ellipsoid is less than number of dimensions in order to make
        volume non-zero.
    ell : Ellipsoid, optional
        If known, the bounding ellipsoid of the points `x`. If not supplied,
        it will be calculated. This option is used when the function calls
        itself recursively.

    Returns
    -------
    ells : list of Ellipsoid
        Ellipsoids.
    """

    ells = []
    npoints, ndim = x.shape

    # If we don't already have a bounding ellipse for the points,
    # calculate it, and enlarge it so that it has at least the minimum
    # volume.
    if ell is None:
        ell = bounding_ellipsoid(x, pointvol=pointvol) 
        minvol = npoints * pointvol
        if ell.vol < minvol:
            ell.scale_to_vol(minvol)

    # debug
    if debug:
        print("cluster at {} with {} points and vol={:.5f}:"
              .format(ell.ctr, len(x), ell.vol))

    # starting cluster centers for kmeans (k=2)
    p1, p2 = ell.major_axis_endpoints()  # returns two 1-d arrays
    start_ctrs = np.vstack((p1, p2)) # shape is (k, N) = (2, N)

    # Split points into two clusters using k-means clustering with k=2
    # centroid = (2, ndim) ; label = (npoints,)
    # [Each entry in `label` is 0 or 1, corresponding to cluster number]
    centroid, label = kmeans2(x, k=start_ctrs, iter=10, minit='matrix')
    cluster_x = [x[label == k, :] for k in [0, 1]]  # points in each cluster

    # If either cluster has less than ndim+1 points, the bounding ellipsoid
    # will be ill-constrained, so we reject the split and simply return the
    # ellipsoid bounding all the points.
    if cluster_x[0].shape[0] <= ndim or cluster_x[1].shape[0] <= ndim:
        return [ell]

    # calculate bounding ellipsoid for each cluster
    cluster_ells = [None, None]
    for k in [0, 1]:
        cluster_ells[k] = bounding_ellipsoid(cluster_x[k], pointvol=pointvol)
        
        if debug:
            print("    cluster {}: centroid={} len={} initvol={:.5f} "
                  .format(k, centroid[k], len(cluster_x[k]),
                          cluster_ells[k].vol), end='')

        # enlarge ellipse so that it is at least as large as the fractional
        # volume according to the number of points in the cluster
        minvol = len(cluster_x[k]) * pointvol
        if cluster_ells[k].vol < minvol:
            cluster_ells[k].scale_to_vol(minvol)

        if debug:
            print("minvol={:.5f} vol={:.5f}".format(minvol,
                                                    cluster_ells[k].vol))

    if debug:
        print()

    # If the total volume decreased by a significant amount,
    # then we will accept the split into subsets and try to perform the
    # algorithm on each subset.
    if  cluster_ells[0].vol + cluster_ells[1].vol < 0.5 * ell.vol:
        ells = []
        for k in [0, 1]:
            ells.extend(bounding_ellipsoids(cluster_x[k], pointvol=pointvol,
                                            ell=cluster_ells[k], debug=debug))
        return ells

    # Otherwise, see if the total ellipse volume is significantly greater
    # than expected. If it is, this indicates that there may be more than 2
    # clusters and we should try to subdivide further.
    if ell.vol > 2. * npoints * pointvol:
        ells = []
        for k in [0, 1]:
            ells.extend(bounding_ellipsoids(cluster_x[k], pointvol=pointvol,
                                            ell=cluster_ells[k], debug=debug))
        # only accept split if volume decreased significantly
        if sum(e.vol for e in ells) < 0.5*ell.vol:
            return ells

    # Otherwise, we are happy with the single bounding ellipse.
    return [ell]


def sample_ellipsoids(ells, rstate=np.random):
    """Chose sample(s) randomly distributed within a set of
    (possibly overlapping) ellipsoids.
    
    Parameters
    ----------
    ells : list of Ellipsoid

    Returns
    -------
    x : 1-d ndarray
        Coordinates within the ellipsoids. 
    """

    # Select an ellipsoid at random, according to volumes
    v = np.array([ell.vol for ell in ells])
    ell = ells[random_choice(len(ells), v/v.sum(), rstate=rstate)]
    
    # Select a point from the ellipsoid
    x = ell.sample(rstate=rstate)

    # How many ellipsoids is the sample in?
    n = 0
    for ell in ells:
        n += ell.contains(x)

    # Only accept the point with probability 1/n
    # (If rejected, sample again).
    if n == 1 or rstate.rand() < 1.0 / n:
        return x
    else:
        return sample_ellipsoids(ells, rstate=rstate)


def print_progress(it, logz):
    print("\riter={:6d} logz={:8f}".format(it, logz), end='', flush=True)


def sample(loglikelihood, prior, npar, nipar=None, npoints=100, maxiter=10000,
           method='single', enlarge=1.5, callback=None, rstate=None,
           plot_callback=None):
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
    npoints : int, optional
        Number of active points. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is 100.
    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is 10000.
    method : {'single', 'multi'}, optional
        Method used to select new points. Choices are
        single-ellipsoidal ('single'), multi-ellipsoidal ('multi'). Default
        is 'single'.
    enlarge : float, optional
        Enlarge the ellipsoid(s) by this fraction in volume. Default is 1.5.
    rstate : `~numpy.random.RandomState`, optional
        RandomState instance. If not given, the global random state of the
        ``np.random`` module will be used.
    callback : func, optional
        Callback function called at each iteration. Two arguments are 
        passed to the callback, the number of iterations and the logarithm
        of the current evidence (``logz``). To print the progress at
        each iteration, use ``callback=nestle.print_progress``.

    Returns
    -------
    result : dict
        Containing following keys:

        * ``niter`` (int) number of iterations.
        * ``ncall`` (int) number of likelihood calls.
        * ``logz`` (float) log of evidence.
        * ``logzerr`` (float) error on ``logz``.
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
    following the ellipsoidal sampling algorithm in Shaw et al (2007).

    Sample Weights are ``likelihood * prior_vol`` where
    prior_vol is the fraction of the prior volume the sample represents.

    References
    ----------
    http://www.inference.phy.cam.ac.uk/bayesys/
    Shaw, Bridges, Hobson 2007, MNRAS, 378, 1365
    Feroz, Hobson, Bridges 2009, MNRAS, 398, 1601
    """

    if nipar is None:
        nipar = npar

    if method is 'multi' and not HAVE_KMEANS:
        raise ValueError("scipy.cluster.vq.kmeans2 required for 'multi' "
                         "method")

    if npoints < npar**2:
        warnings.warn("You really want to make npoints >= npar**2.")

    if rstate is None:
        rstate = np.random

    # Initialize active points and calculate likelihoods
    active_u = rstate.rand(npoints, nipar)  # position in unit cube
    active_v = np.empty((npoints, npar), dtype=np.float64)  # real params
    active_logl = np.empty(npoints, dtype=np.float64)  # log likelihood
    for i in range(npoints):
        active_v[i, :] = prior(active_u[i, :])
        active_logl[i] = loglikelihood(active_v[i, :])

    # Initialize values for nested sampling loop.
    saved_v = []  # stored points for posterior results
    saved_logl = []
    saved_logprior = []
    saved_logwt = []
    loglstar = None  # ln(Likelihood constraint)
    h = 0.0  # Information, initially 0.
    logz = -1e300  # ln(Evidence Z), initially 0.
    # ln(width in prior mass), outermost width is 1 - e^(-1/n)
    logwidth = math.log(1.0 - math.exp(-1.0/npoints))
    ncall = npoints  # number of calls we already made

    # Nested sampling loop.
    ndecl = 0
    logwt_old = -np.inf
    for it in range(maxiter):
        if callback is not None:
            callback(it, logz)

        # worst object in collection and its weight (= width * likelihood)
        worst = np.argmin(active_logl)
        logwt = logwidth + active_logl[worst]

        # update evidence Z and information h.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * active_logl[worst] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Add worst object to samples.
        saved_v.append(np.array(active_v[worst]))
        saved_logwt.append(logwt)
        saved_logprior.append(logwidth)
        saved_logl.append(active_logl[worst])

        # The new likelihood constraint is that of the worst object.
        loglstar = active_logl[worst]

        expected_vol = math.exp(-it/npoints)
        pointvol = expected_vol / npoints

        # calculate the ellipsoid in parameter space that contains all the
        # samples (including the worst one).
        if method == 'single':
            ell = bounding_ellipsoid(active_u, pointvol=pointvol)
            ell.scale_to_vol(ell.vol * enlarge)
        else:
            ells = bounding_ellipsoids(active_u, pointvol=pointvol)
            for ell in ells:
                ell.scale_to_vol(ell.vol * enlarge)

        if plot_callback is not None:
            plot_callback(active_v, ells)



        # Choose a point from within the ellipse until it has likelihood
        # better than loglstar.
        while True:
            if method == 'single':
                u = ell.sample(rstate=rstate)
            else:
                u = sample_ellipsoids(ells, rstate=rstate)
            if np.any(u < 0.0) or np.any(u > 1.0):
                continue
            v = prior(u)
            logl = loglikelihood(v)
            ncall += 1

            # Accept if and only if within likelihood constraint.
            if logl > loglstar:
                active_u[worst] = u
                active_v[worst] = v
                active_logl[worst] = logl
                break

        # Shrink interval
        logwidth -= 1.0 / npoints

        # Stopping criterion: stop when the logwt has been declining
        # for more than npoints* 2 and niter/6 consecutive iterations.
        if logwt < logwt_old:
            ndecl += 1
        else:
            ndecl = 0
        if ndecl > npoints * 2 and ndecl > it // 6:
            break
        logwt_old = logwt

    # Add remaining active points.
    # After N samples have been taken out, the remaining width is
    # e^(-N/npoints). Thus, the remaining width for each active point
    # is e^(-N/npoints) / npoints. The log of this for each object is:
    # log(e^(-N/npoints) / npoints) = -N/npoints - log(npoints)
    logwidth = -len(saved_v) / npoints - math.log(npoints)
    for i in range(npoints):
        logwt = logwidth + active_logl[i]
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * active_logl[i] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        saved_v.append(np.array(active_v[i]))
        saved_logwt.append(logwt)
        saved_logl.append(active_logl[i])
        saved_logprior.append(logwidth)

    return Result([
        ('niter', it + 1),
        ('ncall', ncall),
        ('logz', logz),
        ('logzerr', math.sqrt(h / npoints)),
        ('h', h),
        ('samples', np.array(saved_v)),  # (nsamples, npar)
        ('weights', np.exp(np.array(saved_logwt) - logz)),  # (nsamples,)
        ('logprior', np.array(saved_logprior)),  # (nsamples,)
        ('logl', np.array(saved_logl))  # (nsamples,)
        ])


# -----------------------------------------------------------------------------
# Mahalanobis distance reassignment.  This code was originally in
# bounding_ellipsoids, implemented as described in Feroz, Hobson &
# Bridges (2009). However, I don't think it helps, so I took it out.

"""
    while True:
        h = []
        for k in [0, 1]:
            # Calculate mahalanobis distance between ALL points and the
            # current cluster. The mahalanobis distance squared is given by:
            #     delta = u - v
            #     m = np.dot(np.dot(delta, VI), delta)
            # where, in this case,
            #     VI = (f * C)^-1 = (scaled_cov)^-1
            d = np.empty(npoints, dtype=np.float)
            delta = x - cluster_ells[k].ctr
            for i in range(npoints):
                d[i] = np.dot(np.dot(delta[i,:], cluster_ells[k].a),
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
            cluster_ells[k] = bounding_ellipsoid(cluster_x[k],
                                                 pointvol=pointvol)

            # enlarge ellipse so that it is at least as large as the fractional
            # volume according to the number of points in the cluster
            minvol = len(cluster_x[k]) * pointvol
            if cluster_ells[k].vol < minvol:
                cluster_ells[k].scale_to_vol(minvol)
"""
