# License is MIT: see LICENSE.md.
"""Nestle: nested sampling routines to evaluate Bayesian evidence."""

from __future__ import print_function, division

import sys
import warnings
import math

import numpy as np
try:
    from scipy.cluster.vq import kmeans2
    HAVE_KMEANS = True
except ImportError:  # pragma: no cover
    HAVE_KMEANS = False

__all__ = ["sample", "print_progress", "mean_and_cov", "Result"]
__version__ = "0.1.1"

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

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

    if np.sum(p) - 1. > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("probabilities do not sum to 1")

    r = rstate.rand()
    i = 0
    t = p[i]
    while t < r:
        i += 1
        t += p[i]
    return i


class Result(dict):
    """Represents a sampling result.

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
        """Return a nicely formatted string giving summary."""
        return ("niter: {:d}\n"
                "ncall: {:d}\n"
                "nsamples: {:d}\n"
                "logz: {:6.3f} +/- {:6.3f}\n"
                "h: {:6.3f}"
                .format(self.niter, self.ncall, len(self.samples),
                        self.logz, self.logzerr, self.h))


def mean_and_cov(x, weights):
    """Compute weighted sample mean and covariance.

    Parameters
    ----------
    x : `~numpy.ndarray`
        2-D array containing data samples. Shape is (M, N) where N is the
        number of variables and M is the number of samples or observations.
        This is ordering is equivalent to using ``rowvar=0`` in numpy.cov.
    weights : `~numpy.ndarray`
        1-D array of sample weights. Shape is (M,).

    Returns
    -------
    mean : `~numpy.ndarray`
        Weighted average of samples, with shape (N,).
    cov : `~numpy.ndarray`
        The covariance matrix of the variables with shape (N, N).

    Notes
    -----
    Implements formula described here:
    https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance
    (see "weighted samples" section)
    """

    mean = np.average(x, weights=weights, axis=0)
    dx = x - mean
    wsum = np.sum(weights)
    w2sum = np.sum(weights**2)

    cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', weights, dx, dx)

    return mean, cov


def print_progress(info):
    """Callback function that prints a running total on a single line.

    Parameters
    ----------
    info : dict
        Dictionary containing keys ``'it'`` and ``'logz'``.
    """

    print("\rit={:6d} logz={:8f}".format(info['it'], info['logz']),
          end='')
    sys.stdout.flush()  # because flush keyword not in print() in py2.7

# -----------------------------------------------------------------------------
# Ellipsoid

class Ellipsoid(object):
    """An N-ellipsoid.

    Defined by::

        (x - v)^T A (x - v) = 1

    where the vector ``v`` is the center of the ellipse and ``A`` is an N x N
    matrix. Assumes that ``A`` is symmetric positive definite.

    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape ``(N,)``
        Coordinates of ellipse center. Note that the array is *not* copied.
        This array is never modified internally.
    a : `~numpy.ndarray` with shape ``(N, N)``
        Matrix describing the axes. Watch out! This array is *not* copied.
        but may be modified internally!
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
        # point in the unit n-sphere into a point in the ellipsoid.
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

    def randoffset(self, rstate=np.random):
        """Return an offset from ellipsoid center, randomly distributed
        within ellipsoid."""
        return np.dot(self.axes, randsphere(self.n, rstate=rstate))

    def sample(self, rstate=np.random):
        """Chose a sample randomly distributed within the ellipsoid.

        Returns
        -------
        x : 1-d array
            A single point within the ellipsoid.
        """
        return self.ctr + self.randoffset(rstate=rstate)

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

    def __repr__(self):
        return "Ellipsoid(ctr={})".format(self.ctr)


# -----------------------------------------------------------------------------
# Functions for determining the ellipsoid or set of ellipsoids bounding a
# set of points.

def make_eigvals_positive(a, targetprod):
    """For the symmetric square matrix ``a``, increase any zero eigenvalues
    to fulfill the given target product of eigenvalues.

    Returns a (possibly) new matrix."""

    w, v = np.linalg.eigh(a)  # Use eigh because we assume a is symmetric.
    mask = w < 1.e-10
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
        a = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov

    return a


def bounding_ellipsoid(x, pointvol=0., minvol=False):
    """Calculate bounding ellipsoid containing a set of points x.

    Parameters
    ----------
    x : (npoints, ndim) ndarray
        Coordinates of points.
    pointvol : float, optional
        Used to set a minimum bound on the ellipsoid volume when
        minvol is True.
    minvol : bool, optional
        If True, ensure that ellipsoid volume is at least len(x) * pointvol.

    Returns
    -------
    ellipsoid : Ellipsoid
    """
    npoints, ndim = x.shape

    # If there is only a single point, return an N-sphere with volume `pointvol`
    # centered at the point.
    if npoints == 1:
        r = (pointvol / vol_prefactor(ndim))**(1./ndim)
        return Ellipsoid(x[0], (1. / r**2) * np.identity(ndim))

    # Calculate covariance of points
    ctr = np.mean(x, axis=0)
    delta = x - ctr
    cov = np.cov(delta, rowvar=0)
    
    # when ndim = 1, np.cov returns a 0-d array. Make it a 1x1 2-d array.
    if ndim == 1:
        cov = np.atleast_2d(cov)

    # For a ball of uniformly distributed points, the covariance will be
    # smaller than r^2 by a factor of 1/(n+2) [see, e.g.,
    # http://mathoverflow.net/questions/35276/
    # covariance-of-points-distributed-in-a-n-ball]. In nested sampling,
    # we are supposing the points are uniformly distributed within
    # an ellipse, so the same factor holds. Expand `cov`
    # to compensate for that when defining the ellipse matrix:
    cov *= (ndim + 2)

    # Ensure that ``cov`` is nonsingular.
    # It can be singular when the ellipsoid has zero volume, which happens
    # when npoints <= ndim or when enough points are linear combinations
    # of other points. (e.g., npoints = ndim+1 but one point is a linear
    # combination of others). When this happens, we expand the ellipse
    # in the zero dimensions to fulfill the volume expected from
    # ``pointvol``.
    targetprod = (npoints * pointvol / vol_prefactor(ndim))**2
    cov = make_eigvals_positive(cov, targetprod)

    # The matrix defining the ellipsoid.
    a = np.linalg.inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # Points should obey x^T A x <= 1, so we calculate x^T A x for
    # each point and then scale A up or down to make the
    # "outermost" point obey x^T A x = 1.
    # 
    # fast way to compute delta[i] @ A @ delta[i] for all i.
    f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta)
    fmax = np.max(f)

    # Due to round-off errors, we actually scale the ellipse so the outermost
    # point obeys x^T A x < 1 - (a bit), so that all the points will
    # *definitely* obey x^T A x < 1.
    one_minus_a_bit = 1. - SQRTEPS

    if fmax > one_minus_a_bit:
        a *= one_minus_a_bit / fmax

    ell = Ellipsoid(ctr, a)

    if minvol:
        v = len(x) * pointvol
        if ell.vol < v:
            ell.scale_to_vol(v)

    return ell


def _bounding_ellipsoids(x, ell, pointvol=0.):
    """Internal bounding ellipsoids method for when a bounding ellipsoid for
    the entire set has already been calculated.

    Parameters
    ----------
    x : (npoints, ndim) ndarray
        Coordinates of points.
    ell : Ellipsoid, optional
        If known, the bounding ellipsoid of the points `x`. If not supplied,
        it will be calculated. This option is used when the function calls
        itself recursively.
    pointvol : float, optional
        Volume represented by a single point. Used when number of points
        per ellipsoid is less than number of dimensions in order to make
        volume non-zero.

    Returns
    -------
    ells : list of Ellipsoid
        Ellipsoids.
    """

    npoints, ndim = x.shape

    # starting cluster centers for kmeans (k=2)
    p1, p2 = ell.major_axis_endpoints()  # returns two 1-d arrays
    start_ctrs = np.vstack((p1, p2)) # shape is (k, N) = (2, N)

    # Split points into two clusters using k-means clustering with k=2
    # centroid = (2, ndim) ; label = (npoints,)
    # [Each entry in `label` is 0 or 1, corresponding to cluster number]
    centroid, label = kmeans2(x, k=start_ctrs, iter=10, minit='matrix')

    # Get points in each cluster.
    xs = [x[label == k, :] for k in (0, 1)]  # points in each cluster

    # If either cluster has less than ndim+1 points, the bounding ellipsoid
    # will be ill-constrained, so we reject the split and simply return the
    # ellipsoid bounding all the points.
    if xs[0].shape[0] < 2 * ndim or xs[1].shape[0] < 2 * ndim:
        return [ell]

    # Bounding ellipsoid for each cluster, enlarging to minimum volume.
    ells = [bounding_ellipsoid(xi, pointvol=pointvol, minvol=True)
            for xi in xs]

    # If the total volume decreased by a significant amount,
    # then we will accept the split into subsets and try to perform the
    # algorithm on each subset.
    if ells[0].vol + ells[1].vol < 0.5 * ell.vol:
        return (_bounding_ellipsoids(xs[0], ells[0], pointvol=pointvol) +
                _bounding_ellipsoids(xs[1], ells[1], pointvol=pointvol))

    # Otherwise, see if the total ellipse volume is significantly greater
    # than expected. If it is, this indicates that there may be more than 2
    # clusters and we should try to subdivide further.
    if ell.vol > 2. * npoints * pointvol:
        out = (_bounding_ellipsoids(xs[0], ells[0], pointvol=pointvol) +
               _bounding_ellipsoids(xs[1], ells[1], pointvol=pointvol))

        # only accept split if volume decreased significantly
        if sum(e.vol for e in out) < 0.5 * ell.vol:
            return out

    # Otherwise, we are happy with the single bounding ellipse.
    return [ell]


def bounding_ellipsoids(x, pointvol=0.):
    """Calculate a set of ellipses that bound the points.

    Parameters
    ----------
    x : (npoints, ndim) ndarray
        Coordinates of points.
    pointvol : float, optional
        Volume represented by a single point. Used when number of points
        per ellipsoid is less than number of dimensions in order to make
        volume non-zero.

    Returns
    -------
    ells : list of Ellipsoid
        Ellipsoids.
    """

    # Calculate a single bounding ellipsoid for the points, and enlarge it
    # so that it has at least the minimum volume.
    ell = bounding_ellipsoid(x, pointvol=pointvol, minvol=True)

    return _bounding_ellipsoids(x, ell, pointvol=pointvol)


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

    nells = len(ells)

    if nells == 1:
        return ells[0].sample(rstate=rstate)

    # Select an ellipsoid at random, according to volumes
    vols = np.array([ell.vol for ell in ells])
    i = random_choice(nells, vols / vols.sum(), rstate=rstate)
    
    # Select a point from the ellipsoid
    x = ells[i].sample(rstate=rstate)

    # How many ellipsoids is the sample in?
    n = 1
    for j in range(nells):
        if j == i:
            continue
        n += ells[j].contains(x)

    # Only accept the point with probability 1/n
    # (If rejected, sample again).
    if n == 1 or rstate.rand() < 1.0 / n:
        return x
    else:
        return sample_ellipsoids(ells, rstate=rstate)


# -----------------------------------------------------------------------------
# Sampler classes

class Sampler:
    """A sampler simply selects a new point obeying the likelihood bound,
    given some existing set of points."""

    def __init__(self, loglikelihood, prior_transform, points, rstate,
                 options):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.points = points
        self.rstate = rstate
        self.set_options(options)


class ClassicSampler(Sampler):
    """Picks an active point at random and evolves it with a
    Metropolis-Hastings style MCMC with fixed number of iterations."""

    def set_options(self, options):
        self.steps = options.get('steps', 20)

    def update(self, pointvol):
        """Calculate an ellipsoid to get the rough shape of the point
        distribution correct, but then scale it down to the volume
        corresponding to a single point."""

        self.ell = bounding_ellipsoid(self.points, pointvol=pointvol)
        self.ell.scale_to_vol(pointvol)

    def new_point(self, loglstar):
        # choose a point at random and copy it
        i = self.rstate.randint(len(self.points))
        u = self.points[i, :]

        # evolve it.
        scale = 1.
        accept = 0
        reject = 0
        ncall = 0
        while ncall < self.steps or accept == 0:
            while True:
                new_u = u + scale * self.ell.randoffset(rstate=self.rstate)
                if np.all(new_u > 0.) and np.all(new_u < 1.):
                    break
            new_v = self.prior_transform(new_u)
            new_logl = self.loglikelihood(new_v)
            if new_logl >= loglstar:
                u = new_u
                v = new_v
                logl = new_logl
                accept += 1
            else:
                reject += 1

            # adjust scale, aiming for acceptance ratio of 0.5.
            if accept > reject:
                scale *= math.exp(1. / accept)
            if accept < reject:
                scale /= math.exp(1. / reject)

            ncall += 1

        return u, v, logl, ncall


class SingleEllipsoidSampler(Sampler):
    """Bounds active points in a single ellipsoid and samples randomly
    from within that ellipsoid."""

    def set_options(self, options):
        self.enlarge = options.get('enlarge', 1.2)

    def update(self, pointvol):
        self.ell = bounding_ellipsoid(self.points, pointvol=pointvol,
                                      minvol=True)
        self.ell.scale_to_vol(self.ell.vol * self.enlarge)

    def new_point(self, loglstar):
        ncall = 0
        logl = -float('inf')
        while logl < loglstar:
            while True:
                u = self.ell.sample(rstate=self.rstate)
                if np.all(u > 0.) and np.all(u < 1.):
                    break
            v = self.prior_transform(u)
            logl = self.loglikelihood(v)
            ncall += 1

        return u, v, logl, ncall


class MultiEllipsoidSampler(Sampler):
    """Bounds active points in multiple ellipsoids and samples randomly
    from within joint distribution."""

    def set_options(self, options):
        self.enlarge = options.get('enlarge', 1.2)

    def update(self, pointvol):
        self.ells = bounding_ellipsoids(self.points, pointvol=pointvol)
        for ell in self.ells:
            ell.scale_to_vol(ell.vol * self.enlarge)

    def new_point(self, loglstar):
        ncall = 0
        logl = -float('inf')
        while logl < loglstar:
            while True:
                u = sample_ellipsoids(self.ells, rstate=self.rstate)
                if np.all(u > 0.) and np.all(u < 1.):
                    break
            v = self.prior_transform(u)
            logl = self.loglikelihood(v)
            ncall += 1

        return u, v, logl, ncall


# -----------------------------------------------------------------------------
# Main entry point

_SAMPLERS = {'classic': ClassicSampler,
             'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler}

def sample(loglikelihood, prior_transform, ndim, npoints=100,
           method='single', update_interval=None, npdim=None,
           maxiter=None, maxcall=None, dlogz=None, decline_factor=None,
           rstate=None, callback=None, **options):
    """Perform nested sampling to evaluate Bayesian evidence.

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length *ndim*.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior. The input is a 1-d numpy array with length *ndim*, where
        each value is in the range [0, 1). The return value should also be a
        1-d numpy array with length *ndim*, where each value is a parameter.
        The return value is passed to the loglikelihood function. For example,
        for a 2 parameter model with flat priors in the range [0, 2), the
        function would be::

            def prior_transform(u):
                return 2.0 * u

    ndim : int
        Number of parameters returned by prior and accepted by loglikelihood.

    npoints : int, optional
        Number of active points. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is 100.

    method : {'classic', 'single', 'multi'}, optional
        Method used to select new points. Choices are 'classic',
        single-ellipsoidal ('single'), multi-ellipsoidal ('multi'). Default
        is 'single'.

    update_interval : int, optional
        Only update the new point selector every ``update_interval``-th
        iteration. Update intervals larger than 1 can be more efficient
        when the likelihood function is very fast, particularly when
        using the multi-ellipsoid method. Default is round(0.2 * npoints).

    npdim : int, optional
        Number of parameters accepted by prior. This might differ from *ndim*
        in the case where a parameter of loglikelihood is dependent upon
        multiple independently distributed parameters, some of which may be
        nuisance parameters. 

    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is no limit.

    maxcall : int, optional
        Maximum number of likelihood evaluations. Iteration may stop earlier
        if termination condition is reached. Default is no limit.

    dlogz : float, optional
        If supplied, iteration will stop when the estimated contribution
        of the remaining prior volume to the total evidence falls below
        this threshold. Explicitly, the stopping criterion is
        ``log(z + z_est) - log(z) < dlogz`` where *z* is the current evidence
        from all saved samples, and *z_est* is the estimated contribution
        from the remaining volume. This option and decline_factor are
        mutually exclusive. If neither is specified, the default is
        ``dlogz=0.5``.

    decline_factor : float, optional
        If supplied, iteration will stop when the weight
        (likelihood times prior volume) of newly saved samples has been
        declining for ``decline_factor * nsamples`` consecutive samples.
        A value of 1.0 seems to work pretty well. This option and dlogz
        are mutually exclusive.

    rstate : `~numpy.random.RandomState`, optional
        RandomState instance. If not given, the global random state of the
        ``numpy.random`` module will be used.

    callback : function, optional
        Callback function to be called at each iteration. A single argument,
        a dictionary, is passed to the callback. The keys include ``'it'``,
        the current iteration number, and ``'logz'``, the current total
        log evidence of all saved points. To simply print these at each
        iteration, use the convience function
        ``callback=nestle.print_progress``. 


    Other Parameters
    ----------------
    steps : int, optional
        For the 'classic' method, the number of steps to take when selecting
        a new point. Default is 20.

    enlarge : float, optional
        For the 'single' and 'multi' methods, enlarge the ellipsoid(s) by
        this fraction in volume. Default is 1.2.


    Returns
    -------
    result : `Result`
        A dictionary-like object with attribute access: Attributes can be
        accessed with, for example, either ``result['niter']`` or
        ``result.niter``. Attributes:

        niter *(int)*
            Number of iterations.

        ncall *(int)*
            Number of likelihood calls.

        logz *(float)*
            Natural logarithm of evidence (integral of posterior).

        logzerr *(float)*
            Estimated numerical (sampling) error on *logz*.

        h *(float)*
            Information. This is a measure of the "peakiness" of the
            likelihood function. A constant likelihood has zero information.

        samples *(ndarray)*
            Parameter values of each sample. Shape is *(nsamples, ndim)*.

        logvol *(ndarray)*
            Natural log of prior volume of corresponding to each sample.
            Shape is *(nsamples,)*.

        logl *(ndarray)*
            Natural log of the likelihood for each sample, as returned by
            user-supplied *logl* function. Shape is *(nsamples,)*.

        weights *(ndarray)*
            Weight corresponding to each sample, normalized to unity.
            These are proportional to ``exp(logvol + logl)``. Shape is
            *(nsamples,)*.
    """

    if npdim is None:
        npdim = ndim

    if maxiter is None:
        maxiter = sys.maxsize

    if maxcall is None:
        maxcall = sys.maxsize

    if method == 'multi' and not HAVE_KMEANS:
        raise ValueError("scipy.cluster.vq.kmeans2 is required for the "
                         "'multi' method.")  # pragma: no cover

    if method not in _SAMPLERS:
        raise ValueError("Unknown method: {:r}".format(method))

    if npoints < 2 * ndim:
        warnings.warn("You really want to make npoints >= 2 * ndim!")

    if rstate is None:
        rstate = np.random

    # Stopping criterion.
    if dlogz is not None and decline_factor is not None:
        raise ValueError("Cannot specify two separate stopping criteria: "
                         "decline_factor and dlogz")
    elif dlogz is None and decline_factor is None:
        dlogz = 0.5

    if update_interval is None:
        update_interval = max(1, round(0.2 * npoints))
    else:
        update_interval = round(update_interval)
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")

    # Initialize active points and calculate likelihoods
    active_u = rstate.rand(npoints, npdim)  # position in unit cube
    active_v = np.empty((npoints, ndim), dtype=np.float64)  # real params
    active_logl = np.empty(npoints, dtype=np.float64)  # log likelihood
    for i in range(npoints):
        active_v[i, :] = prior_transform(active_u[i, :])
        active_logl[i] = loglikelihood(active_v[i, :])

    sampler = _SAMPLERS[method](loglikelihood, prior_transform, active_u,
                                rstate, options)

    # Initialize values for nested sampling loop.
    saved_v = []  # stored points for posterior results
    saved_logl = []
    saved_logvol = []
    saved_logwt = []
    h = 0.0  # Information, initially 0.
    logz = -1e300  # ln(Evidence Z), initially Z=0.
    logvol = math.log(1.0 - math.exp(-1.0/npoints))  # first point removed will
                                                     # have volume 1-e^(1/n)
    ncall = npoints  # number of calls we already made

    # Initialize sampler
    sampler.update(1./npoints)

    callback_info = {'it': 0,
                     'logz': logz,
                     'active_u': active_u,
                     'sampler': sampler}

    # Nested sampling loop.
    ndecl = 0
    logwt_old = -np.inf
    it = 0
    while it < maxiter:
        if callback is not None:
            callback_info.update(it=it, logz=logz)
            callback(callback_info)

        # worst object in collection and its weight (= volume * likelihood)
        worst = np.argmin(active_logl)
        logwt = logvol + active_logl[worst]

        # update evidence Z and information h.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * active_logl[worst] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Add worst object to samples.
        saved_v.append(np.array(active_v[worst]))
        saved_logwt.append(logwt)
        saved_logvol.append(logvol)
        saved_logl.append(active_logl[worst])

        # The new likelihood constraint is that of the worst object.
        loglstar = active_logl[worst]

        expected_vol = math.exp(-it / npoints)
        pointvol = expected_vol / npoints

        # Update the sampler based on the current active points.
        if it % update_interval == 0:
            sampler.update(pointvol)

        # Choose a new point from within the likelihood constraint
        # (having logl > loglstar).
        u, v, logl, nc = sampler.new_point(loglstar)

        # replace worst point with new point
        active_u[worst] = u
        active_v[worst] = v
        active_logl[worst] = logl
        ncall += nc

        # Shrink interval
        logvol -= 1.0 / npoints

        # Stopping criterion 1: estimated fractional remaining evidence
        # below some threshold.
        if dlogz is not None:
            logz_remain = np.max(active_logl) - it / npoints
            if np.logaddexp(logz, logz_remain) - logz < dlogz:
                break

        # Stopping criterion 2: logwt has been declining for a while.
        if decline_factor is not None:
            ndecl = ndecl + 1 if logwt < logwt_old else 0
            logwt_old = logwt
            if ndecl > decline_factor * npoints:
                break

        if ncall > maxcall:
            break

        it += 1

    # Add remaining active points.
    # After N samples have been taken out, the remaining volume is
    # e^(-N/npoints). Thus, the remaining volume for each active point
    # is e^(-N/npoints) / npoints. The log of this for each object is:
    # log(e^(-N/npoints) / npoints) = -N/npoints - log(npoints)
    logvol = -len(saved_v) / npoints - math.log(npoints)
    for i in range(npoints):
        logwt = logvol + active_logl[i]
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * active_logl[i] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        saved_v.append(np.array(active_v[i]))
        saved_logwt.append(logwt)
        saved_logl.append(active_logl[i])
        saved_logvol.append(logvol)

    # h should always be nonnegative (we take the sqrt below).
    # Numerical error makes it negative in pathological corner cases
    # such as flat likelihoods. Here we correct those cases to zero.
    if h < 0.0:
        if h > -SQRTEPS:
            h = 0.0
        else:
            raise RuntimeError("Negative h encountered (h={}). Please report "
                               "this as a likely bug.".format(h))

    return Result([
        ('niter', it + 1),
        ('ncall', ncall),
        ('logz', logz),
        ('logzerr', math.sqrt(h / npoints)),
        ('h', h),
        ('samples', np.array(saved_v)),
        ('weights', np.exp(np.array(saved_logwt) - logz)),
        ('logvol', np.array(saved_logvol)),
        ('logl', np.array(saved_logl))
        ])
