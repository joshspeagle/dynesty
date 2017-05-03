# License is MIT: see LICENSE.md.
"""dynesty: dynamic nested sampling routines
to evaluate Bayesian evidence and posteriors."""

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

__all__ = ["sample", "print_progress", "mean_and_cov", "resample_equal",
           "Result"]
__version__ = "0.2.0"

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
    """Draw a random point within an n-dimensional unit sphere."""

    z = rstate.randn(n)  # initial n-dim vector

    return z * rstate.rand()**(1./n) / np.sqrt(np.sum(z**2))


def random_choice(a, p, rstate=np.random):
    """Replacement for numpy.random.choice (only in numpy 1.7+)."""

    if abs(np.sum(p) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("probabilities do not sum to 1")

    r = rstate.rand()
    i = 0
    t = p[i]
    while t < r:
        i += 1
        t += p[i]

    return i


def resample_equal(samples, weights, rstate=None):
    """Resample the samples so that the final samples all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times, with
    `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Unequally weighted samples returned by the nested sampling algorithm.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New samples with equal weights.

    Examples
    --------

    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> nestle.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in Hol, Schon, and
    Gustafsson (2006), which can be found at <doi:10.1109/NSSPW.2006.4378824>.
    This gives less "noisy" samples as compared to standard multinomial
    resampling techniques.

    """

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("Weights do not sum to 1.")

    if rstate is None:
        rstate = np.random

    nsamples = len(weights)

    # Make N subdivisions and choose positions with a consistent random offset.
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    idx = np.zeros(nsamples, dtype=np.int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


class Results(dict):
    """Contains the output of a dynamic nested sampling run.

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
                "nsamples: {:d}\n"
                "logz: {:6.3f} +/- {:6.3f}\n"
                "h: {:6.3f}"
                .format(self.nlive, self.niter, self.ncall,
                        len(self.samples), self.logz, self.logzerr,
                        self.h))


def mean_and_cov(samples, weights):
    """Compute weighted sample mean and covariance.

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples, ndim)
        2-D array containing data samples. This ordering is equivalent to
        using `rowvar=False` in `numpy.cov`.

    weights : `~numpy.ndarray` with shape (nsamples,)
        1-D array of sample weights.

    Returns
    -------
    mean : `~numpy.ndarray` with shape (ndim,)
        Weighted sample means.

    cov : `~numpy.ndarray` with shape (ndim, ndim)
        Weighted sample covariances.

    Notes
    -----
    Implements the formulae in the "weighted samples" section on
    <https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance>.
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
        Dictionary containing keys 'it' and 'logz'.
    """

    print("\r\033[Kit={:6d} logz={:8f}".format(info['it'], info['logz']),
          end='')
    sys.stdout.flush()  # because flush keyword not in print() in py2.7


# -----------------------------------------------------------------------------
# Ellipsoid

class Ellipsoid(object):
    """An N-ellipsoid.

    Defined by::

        (x - v)^T A (x - v) = 1

    where the vector `v` is the center of the ellipsoid and `A` is an `N x N`
    matrix. Assumes that `A` is symmetric positive definite.

    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape (N,)
        Coordinates of ellipsoid center. Note that the array is *not* copied.
        This array is never modified internally.

    a : `~numpy.ndarray` with shape (N, N)
        Matrix describing the axes. Watch out! This array is *not* copied
        and may be modified internally!
    """

    def __init__(self, ctr, a):
        self.n = len(ctr)  # dimension
        self.ctr = ctr  # center coordinates
        self.a = a  # precision matrix (inverse of covariance)

        # Volume of ellipsoid is the volume of an n-sphere divided
        # by the (determinant of the) Jacobian associated with the
        # transformation, which by definition is the precision matrix.
        self.vol = vol_prefactor(self.n) / np.sqrt(np.linalg.det(a))

        # The eigenvalues (l) of `a` are (a^-2, b^-2, ...) where
        # (a, b, ...) are the lengths of principle axes.
        # The eigenvectors (v) are the normalized principle axes.
        l, v = np.linalg.eigh(a)
        self.axlens = 1. / np.sqrt(l)

        # Scaled eigenvectors are the axes, where `axes[:,i]` is the
        # i-th axis.  Multiplying this matrix by a vector will transform a
        # point in the unit n-sphere to a point in the ellipsoid.
        self.axes = np.dot(v, np.diag(self.axlens))

    def scale_to_vol(self, vol):
        """Scale ellipoid to encompass a target volume."""

        f = (vol / self.vol) ** (1.0 / self.n)  # linear factor
        self.a *= f**-2
        self.axlens *= f
        self.axes *= f
        self.vol = vol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis."""

        i = np.argmax(self.axlens)  # find the major axis
        v = self.axes[:, i]  # vector from center to major axis endpoint

        return self.ctr - v, self.ctr + v

    def contains(self, x):
        """Checks if ellipsoid contains `x`."""

        d = x - self.ctr

        return np.dot(np.dot(d, self.a), d) <= 1.0

    def randoffset(self, rstate=np.random):
        """Return an offset from ellipsoid center that is randomly
        distributed within the ellipsoid."""

        return np.dot(self.axes, randsphere(self.n, rstate=rstate))

    def sample(self, rstate=np.random):
        """Draw a sample randomly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the ellipsoid.
        """

        return self.ctr + self.randoffset(rstate=rstate)

    def samples(self, nsamples, rstate=np.random):
        """Draw `nsamples` samples randomly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the ellipsoid.
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
    """For the symmetric square matrix `a`, increase any zero eigenvalues
    to fulfill the given target product of eigenvalues.

    Returns a (possibly) new matrix."""

    w, v = np.linalg.eigh(a)  # use eigh since a is symmetric
    mask = w < 1.e-10
    if np.any(mask):
        nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
        nzeros = mask.sum()  # number of zero eigenvalues
        w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
        a = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov

    return a


def bounding_ellipsoid(points, pointvol=None):
    """Calculate bounding ellipsoid containing a collection of points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    pointvol : float, optional
        The minimum volume occupied by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is *None*.

    Returns
    -------
    ellipsoid : Ellipsoid
    """
    npoints, ndim = points.shape

    # Check for valid `pointvol` value if provided.
    if pointvol is not None and pointvol <= 0.:
        raise ValueError("You must specify a positive value for `pointvol`.")

    # If there is only a single point, return an n-sphere with volume
    # `pointvol` centered at the point.
    if npoints == 1:
        if pointvol is not None:
            ctr = points[0]
            r = (pointvol / vol_prefactor(ndim))**(1./ndim)
            a = (1. / r**2) * np.identity(ndim)
            return Ellipsoid(ctr, a)
        else:
            raise ValueError("Cannot compute a bounding ellipsoid to a "
                             "single point if `pointvol` is not specified.")

    # Calculate covariance of points.
    ctr = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)

    # When ndim = 1, np.cov returns a 0-d array. Make it a 1x1 2-d array.
    if ndim == 1:
        cov = np.atleast_2d(cov)

    # For a ball of uniformly distributed points, the sample covariance
    # will be smaller than the true covariance by a factor of 1/(n+2)
    # [see, e.g., goo.gl/UbsjYl]. Since we are assuming all points are
    # uniformly distributed within the unit cube, they are uniformly
    # distributed within any sub-volume within the cube. We expand
    # our sample covariance `cov` to compensate for this.
    cov *= (ndim + 2)

    # Ensure that `cov` is nonsingular to deal with pathological cases
    # where the ellipsoid has zero volume. This can occur when
    # `npoints <= ndim` or when enough points are linear combinations
    # of other points. When this happens, we expand the ellipsoid
    # in the zero dimensions to fulfill the volume expected from
    # `pointvol`.
    if pointvol is not None:
        targetprod = (npoints * pointvol / vol_prefactor(ndim))**2
        cov = make_eigvals_positive(cov, targetprod)
    else:
        raise ValueError("Cannot modify `a` to be non-singular to give "
                         "our ellipsoid non-zero volume if `pointvol` "
                         "is not specified.")

    # The matrix defining the ellipsoid.
    a = np.linalg.inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # Points should obey `(x-v)^T A (x-v) <= 1`, so we calculate this for
    # each point and then scale A up or down to make the
    # "outermost" point obey `(x-v)^T A (x-v) = 1`. This can be done
    # quickly using `einsum` and `tensordot` to iterate over all points.
    delta = points - ctr
    f = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta)
    fmax = np.max(f)

    # Due to round-off errors, we actually scale the ellipsoid so the outermost
    # point obeys `(x-v)^T A (x-v) < 1 - (a bit) < 1`.
    one_minus_a_bit = 1. - SQRTEPS

    if fmax > one_minus_a_bit:
        a *= one_minus_a_bit / fmax

    # Initialize our ellipsoid.
    ell = Ellipsoid(ctr, a)

    # Expand our ellipsoid to encompass a minimum volume.
    if pointvol is not None:
        v = npoints * pointvol
        if ell.vol < v:
            ell.scale_to_vol(v)
    else:
        raise ValueError("Cannot expand ellipsoid if `pointvol` "
                         "is not specified.")

    return ell


def _bounding_ellipsoids(points, ell, pointvol=None, vol_dec=0.5,
                         vol_check=2.):
    """Internal bounding ellipsoids method used when a bounding ellipsoid for
    the entire set has already been calculated.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        Coordinates of points.

    ell : Ellipsoid
        The bounding ellipsoid of the set of points.

    pointvol : float, optional
        Volume represented by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is *None*.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the split. Default is *0.5*.

    vol_check : float, optional
        The factor used to when checking whether the volume of the
        original bounding ellipsoid is large enough to warrant more
        trial splits via `ell.vol > vol_check * npoints * pointvol`.
        Default is *2.0*.

    Returns
    -------
    ells : list of Ellipsoid objects
    """

    npoints, ndim = points.shape

    # Starting cluster centers are initialized using the major-axis
    # endpoints of the original bounding ellipsoid.
    p1, p2 = ell.major_axis_endpoints()
    start_ctrs = np.vstack((p1, p2))  # shape is (k, ndim) = (2, ndim)

    # Split points into two clusters using k-means clustering with k=2.
    k2_res = kmeans2(points, k=start_ctrs, iter=10, minit='matrix')
    centroids = k2_res[0]  # shape is (k, ndim) = (2, ndim)
    labels = k2_res[1]  # cluster identifier ; shape is (npoints,)

    # Get points in each cluster.
    points_k = [points[labels == k, :] for k in (0, 1)]

    # If either cluster has less than ndim+1 points, the bounding ellipsoid
    # will be ill-constrained. Reject the split and simply return the
    # original ellipsoid bounding all the points.
    if points_k[0].shape[0] < 2 * ndim or points_k[1].shape[0] < 2 * ndim:
        return [ell]

    # Bounding ellipsoid for each cluster, possibly enlarged to minimum volume.
    ells = [bounding_ellipsoid(points_j, pointvol=pointvol)
            for points_j in points_k]

    # If the total volume decreased by a factor of `vol_dec`, we accept
    # the split into subsets. We then recursively split each subset.
    if ells[0].vol + ells[1].vol < vol_dec * ell.vol:
        return (_bounding_ellipsoids(points_k[0], ells[0], pointvol=pointvol,
                                     vol_dec=vol_dec, vol_check=vol_check) +
                _bounding_ellipsoids(points_k[1], ells[1], pointvol=pointvol,
                                     vol_dec=vol_dec, vol_check=vol_check))

    # Otherwise, see if the total ellipsoid volume is larger than the minimum
    # volume by a factor of `vol_check`. If it is, this indicates that there
    # may be more than 2 clusters and we should try to subdivide further.
    if ell.vol > vol_check * npoints * pointvol:
        out = (_bounding_ellipsoids(points_k[0], ells[0], pointvol=pointvol,
                                    vol_dec=vol_dec, vol_check=vol_check) +
               _bounding_ellipsoids(points_k[1], ells[1], pointvol=pointvol,
                                    vol_dec=vol_dec, vol_check=vol_check))

        # only accept split if volume decreased significantly
        if sum(e.vol for e in out) < vol_dec * ell.vol:
            return out

    # Otherwise, we are happy with the single bounding ellipsoid.
    return [ell]


def bounding_ellipsoids(points, pointvol=None, vol_dec=0.5, vol_check=2.):
    """Calculate a set of ellipsoids that bound the points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        Coordinates of points.

    pointvol : float, optional
        Volume represented by a single point. When provided,
        used to set a minimum bound on the ellipsoid volume
        as `npoints * pointvol`. Default is *None*.

    vol_dec : float, optional
        The required fractional reduction in volume after splitting an
        ellipsoid in order to to accept the split. Default is *0.5*.

    vol_check : float, optional
        The factor used to when checking whether the volume of the
        original bounding ellipsoid is large enough to warrant more
        trial splits via `ell.vol > vol_check * npoints * pointvol`.
        Default is *2.0*.

    Returns
    -------
    ells : list of Ellipsoid
        Ellipsoids.
    """

    # Calculate the bounding ellipsoid for the points possibly
    # enlarged to a minimum volume.
    ell = bounding_ellipsoid(points, pointvol=pointvol)

    # Recursively split the bounding ellipsoid until the volume of each
    # split no longer decreases by a factor of `vol_dec`.
    ells = _bounding_ellipsoids(points, ell, pointvol=pointvol,
                                vol_dec=vol_dec, vol_check=vol_check)
    nells = len(ells)

    return ells, nells


def check_ellipsoid_overlap(point, ells, nells):
    """Check how many ellipsoids in `ells` a point lies within."""

    return sum([ells[j].contains(point) for j in xrange(nells)])


def check_unit_cube(point):
    """Check whether a point falls within the unit cube."""

    return np.all(point > 0.) and np.all(point < 1.)


def sample_ellipsoids(ells, nells, rstate=np.random):
    """Sample a point uniformly distributed within a set of ellipsoids.

    Parameters
    ----------
    ells : list of Ellipsoid objects

    Returns
    -------
    point : `~numpy.ndarray` with shape (ndim,)
        Coordinates within the ellipsoids.
    """

    if nells == 1:
        point = ells[0].sample(rstate=rstate)
        q = 1
        return point, q

    # Select an ellipsoid at random proportional to its volume.
    vols = np.array([ell.vol for ell in ells])
    i = random_choice(nells, vols / vols.sum(), rstate=rstate)

    # Select a point from the ellipsoid.
    point = ells[i].sample(rstate=rstate)

    # Check how many ellipsoids the point lies within.
    q = check_ellipsoid_overlap(point, ells, nells)

    return point, q


# -----------------------------------------------------------------------------
# Classes for dealing with the serial case (i.e. no parallelism)


class FakePool(object):
    """A fake Pool for serial function evaluations."""

    def __init__(self):
        pass

    def submit(self, fn, *args, **kwargs):
        return FakeFuture(fn, *args, **kwargs)

    def map(self, func, *iterables):
        return map(func, *iterables)

    def shutdown(self):
        pass


class FakeFuture(object):
    """A fake Future to mimic function calls."""

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def result(self):
        return self.fn(*self.args, **self.kwargs)

    def cancel(self):
        return True


# -----------------------------------------------------------------------------
# Sampler classes


class Sampler:
    """A sampler simply selects a new point obeying the likelihood bound,
    given some existing set of points."""

    def __init__(self, loglikelihood, prior_transform, points, rstate,
                 options, queue_size, pool):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.points = points
        self.rstate = rstate
        self.set_options(options)
        self.queue_size = queue_size
        self.pool = pool
        self.queue = []
        self.nqueue = 0
        self.submitted = 0
        self.cancelled = 0
        self.unused = 0
        self.used = 0

    def empty_queue(self):
        """Dump all operations on the queue."""

        while self.nqueue > 0:
            x, v, f = self.queue.pop()
            if f.cancel():
                self.cancelled += 1
            else:
                self.unused += 1
            self.nqueue -= 1

    def fill_queue(self):
        """Sequentially add operations to the queue."""

        while self.nqueue < self.queue_size:
            x = self.propose_point()
            v = self.prior_transform(x)
            self.queue.append((x, v, self.pool.submit(self.loglikelihood, v)))
            self.nqueue += 1
            self.submitted += 1

    def get_point_value(self):
        """Get an evaluation sequentially from the filled queue. Afterwards,
        refill the queue."""

        x, v, f = self.queue.pop(0)
        self.nqueue -= 1
        r = f.result()
        self.fill_queue()
        self.used += 1

        return x, v, r


class SingleEllipsoidSampler(Sampler):
    """Bounds live points in a single ellipsoid and samples uniformly
    from within that ellipsoid."""

    def set_options(self, options):
        self.enlarge = options.get('enlarge', 1.2)

    def update(self, pointvol):
        self.empty_queue()
        self.ell = bounding_ellipsoid(self.points, pointvol=pointvol)
        self.ell.scale_to_vol(self.ell.vol * self.enlarge)
        self.fill_queue()

    def propose_point(self):
        while True:
            u = self.ell.sample(rstate=self.rstate)
            if check_unit_cube(u):
                break

        return u

    def new_point(self, loglstar):
        ncall = 0
        while True:
            u, v, logl = self.get_point_value()
            ncall += 1
            if logl >= loglstar:
                break

        return u, v, logl, ncall


class MultiEllipsoidSampler(Sampler):
    """Bounds live points in multiple ellipsoids and samples uniformly
    from within the volume spanned by their union."""

    def set_options(self, options):
        self.enlarge = options.get('enlarge', 1.2)
        self.vol_dec = options.get('vol_dec', 0.5)
        self.vol_check = options.get('vol_check', 2.0)

    def update(self, pointvol):
        self.empty_queue()
        ells, nells = bounding_ellipsoids(self.points, pointvol=pointvol,
                                          vol_dec=self.vol_dec,
                                          vol_check=self.vol_check)
        for ell in ells:
            ell.scale_to_vol(ell.vol * self.enlarge)
        self.ells, self.nells = ells, nells
        self.fill_queue()

    def propose_point(self):
        while True:
            u, q = sample_ellipsoids(self.ells, self.nells,
                                     rstate=self.rstate)
            if check_unit_cube(u):
                # Accept the point with probability 1/q to properly
                # sample from the union of overlapping ellipsoids.
                if q == 1 or self.rstate.rand() < 1.0 / q:
                    break
        return u

    def new_point(self, loglstar):
        ncall = 0
        while True:
            u, v, logl = self.get_point_value()
            ncall += 1
            if logl >= loglstar:
                break

        return u, v, logl, ncall


# -----------------------------------------------------------------------------
# Main entry point


_SAMPLERS = {'single': SingleEllipsoidSampler,
             'multi': MultiEllipsoidSampler}


def sample(loglikelihood, prior_transform, ndim, nlive=100,
           method='multi', update_interval=None, npdim=None,
           maxiter=None, maxcall=None, dlogz=None, decline_factor=None,
           rstate=None, callback=None, queue_size=1, pool=None, **options):
    """Perform nested sampling to evaluate Bayesian evidence.

    Parameters
    ----------
    loglikelihood : function
        Function returning log(likelihood) given parameters as a 1-d numpy
        array of length `ndim`.

    prior_transform : function
        Function translating a unit cube to the parameter space according to
        the prior. The input is a 1-d numpy array with length `ndim`, where
        each value is in the range [0, 1). The return value should also be a
        1-d numpy array with length `ndim`, where each value is a parameter.
        The return value is passed to the loglikelihood function. For example,
        for a 2 parameter model with flat priors in the range [0, 2), the
        function would be::

            def prior_transform(u):
                return 2.0 * u

    ndim : int
        Number of parameters returned by prior and accepted by loglikelihood.

    nlive : int, optional
        Number of "live" points. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is *100*.

    method : {'single', 'multi'}, optional
        Method used to select new points. Choices are single-ellipsoidal
        ('single') and multi-ellipsoidal ('multi'). Default is 'multi'.

    update_interval : int, optional
        Only update the new point selector every `update_interval`-th
        likelihood call. Update intervals larger than 1 can be more efficient
        when the likelihood function is very fast, particularly when
        using the multi-ellipsoid method. Default is `round(0.6 * nlive)`.

    npdim : int, optional
        Number of parameters accepted by prior. This might differ from `ndim`
        in the case where a parameter of loglikelihood is dependent upon
        multiple independently distributed parameters, some of which may be
        nuisance parameters.

    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is no limit (`sys.maxsize`).

    maxcall : int, optional
        Maximum number of likelihood evaluations. Iteration may stop earlier
        if termination condition is reached. Default is no limit
        (`sys.maxsize`).

    dlogz : float, optional
        If supplied, iteration will stop when the estimated contribution
        of the remaining prior volume to the total evidence falls below
        this threshold. Explicitly, the stopping criterion is
        `log(z + z_est) - log(z) < dlogz`, where `z` is the current evidence
        from all saved samples and `z_est` is the estimated contribution
        from the remaining volume. This option and decline_factor are
        mutually exclusive. Default is *0.5*.

    decline_factor : float, optional
        If supplied, iteration will stop when the sample weights
        (likelihood times prior volume) of newly saved samples has been
        declining for `decline_factor * nsamples` consecutive samples.
        A value of *1.0* works well for most cases. This option and `dlogz`
        are mutually exclusive. If not specified, the default `dlogz` criterion
        is used.

    rstate : `~numpy.random.RandomState`, optional
        RandomState instance. If not given, the global random state of the
        `numpy.random` module will be used.

    callback : function, optional
        Callback function to be called at each iteration. A single argument,
        a dictionary, is passed to the callback. The keys include 'it',
        the current iteration number, and 'logz', the current total
        log evidence of all saved points. To simply print these at each
        iteration, use the convience function `callback=nestle.print_progress`.

    queue_size: int, optional
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using at most this many threads. Each thread
        independently proposes new live points until the proposal distribution
        is updated. Default is *1* (no parallelism).

    pool: ThreadPoolExecutor, optional
        Use this pool of workers to propose live points in parallel. If
        `queue_size > 1` and `pool` is not specified, a `ValueError` will be
        thrown.


    Other Parameters
    ----------------

    enlarge : float, optional
        For the 'single' and 'multi' methods, enlarge the ellipsoid(s) by
        this fraction in volume. Default is *1.2*.

    vol_dec : float, optional
        For the 'multi' method, the required fractional reduction in volume
        after splitting an ellipsoid in order to to accept the split.
        Default is *0.5*.

    vol_check : float, optional
        For the 'multi' method, the factor used to when checking whether the
        volume of the original bounding ellipsoid is large enough to warrant
        more trial splits via `ell.vol > vol_check * nlive * pointvol`.
        Default is *2.0*.


    Returns
    -------
    result : `Result`
        A dictionary-like object with attribute access: Attributes can be
        accessed with, for example, either `result['niter']` or
        `result.niter`. Attributes:

        niter : int
            Number of iterations.

        ncall : int
            Number of likelihood calls.

        logz : float
            Natural logarithm of the evidence (integral of posterior).

        logzerr : float
            Estimated numerical (sampling) error on `logz`.

        h : float
            Information. This is a measure of the "peakiness" of the
            likelihood function. A constant likelihood has zero information.

        samples : `~numpy.ndarray` with shape (nsamples, ndim)
            Parameter values of each sample.

        logvol : `~numpy.ndarray` with shape (nsamples,)
            Natural log of prior volume of corresponding to each sample.

        logl : `~numpy.ndarray` with shape (nsamples,)
            Natural log of the likelihood for each sample, as returned by
            user-supplied `loglikelihood` function.

        logwt : `~numpy.ndarray` with shape (nsamples,)
            Natural log of the weights corresponding to each sample defined as
            `logwt = logvol + logl - logz`.
    """

    # Initialize variables.
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
        raise ValueError("Unknown method: '{:r}'".format(method))

    if nlive < 2 * ndim:
        warnings.warn("You really want to make `nlive >= 2 * ndim`!")

    if rstate is None:
        rstate = np.random

    # Establish stopping criterion.
    if dlogz is not None and decline_factor is not None:
        raise ValueError("Cannot specify two separate stopping criteria: "
                         "decline_factor and dlogz")
    elif dlogz is None and decline_factor is None:
        dlogz = 0.5

    if update_interval is None:
        update_interval = max(1, round(0.6 * nlive))
    else:
        update_interval = round(update_interval)
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")

    # Set up parallel evaluation.
    if queue_size == 1:
        pool = FakePool()
    else:
        if pool is None:
            raise ValueError("Missing `pool`. Please provide a Pool.")

    # Initialize live points and calculate likelihoods.
    live_u = rstate.rand(nlive, npdim)  # positions in unit cube
    live_v = np.empty((nlive, ndim), dtype=np.float64)  # real params
    for i in range(nlive):
        live_v[i, :] = prior_transform(live_u[i, :])
    live_logl = np.fromiter(pool.map(loglikelihood, live_v),
                            dtype=np.float64)  # log likelihood

    # Initialize our sampler.
    sampler = _SAMPLERS[method](loglikelihood, prior_transform, live_u,
                                rstate, options, queue_size, pool)

    # Initialize values for nested sampling loop.
    saved_u = []  # samples (unit cube)
    saved_v = []  # samples (transformed)
    saved_logl = []  # ln(likelihood)
    saved_logvol = []  # ln(volume)
    saved_logwt = []  # ln(weight)
    h = 0.0  # Information, initially *0.*
    logz = -1e300  # ln(evidence), initially *0.*
    logvol = math.log(1.0 - math.exp(-1.0/nlive))  # initially `1-e^(1/n)`
    ncall = nlive  # number of calls we already made

    # Initialize proposal distribution for our sampler.
    pointvol = 1./nlive
    sampler.update(pointvol)

    callback_info = {'it': 0,
                     'logz': logz,
                     'live_u': live_u,
                     'sampler': sampler}

    # The main nested sampling loop.
    ndecl = 0
    logwt_old = -np.inf
    it = 1  # iterations start from 1
    since_update = 0
    while it < maxiter:
        # Output callback if requested.
        if callback is not None:
            callback_info.update(it=it, logz=logz)
            callback(callback_info)

        # After `update_interval` interations have passed, update the sampler
        # using the current set of live points.
        if since_update >= update_interval:
            expected_vol = math.exp(-it / nlive)  # average volume
            pointvol = expected_vol / nlive  # volume per point
            sampler.update(pointvol)
            since_update = 0

        # Locate the "live" point with the lowest `logl` (the "worst" point).
        worst = np.argmin(live_logl)

        # Set our new worst likelihood constraint.
        ustar, vstar = live_u[worst], live_v[worst]  # position
        loglstar = live_logl[worst]  # likelihood

        # Set our new weight.
        logwt = logvol + loglstar

        # Sample a new live point from within the likelihood constraint
        # `logl > loglstar` using the proposal distribution from our sampler.
        u, v, logl, nc = sampler.new_point(loglstar)
        ncall += nc
        since_update += nc

        # Add the worst live point to samples. It is now a "dead" point.
        saved_u.append(np.array(ustar))
        saved_v.append(np.array(vstar))
        saved_logl.append(loglstar)
        saved_logvol.append(logvol)
        saved_logwt.append(logwt)

        # Update evidence `logz` and information `h` using our new dead point.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * loglstar +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Update the live point (previously our "worst" point).
        live_u[worst] = u
        live_v[worst] = v
        live_logl[worst] = logl

        # Apply expected shrinkage to `logvol` for the next live point.
        logvol -= 1.0 / nlive

        # Stopping criterion 1: estimated (fractional) remaining evidence
        # lies below some threshold set by `dlogz`.
        if dlogz is not None:
            logz_remain = np.max(live_logl) - it / nlive
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
            if ndecl > decline_factor * nlive:
                break

        # Stopping criterion 3: number of `loglikelihood` calls
        # exceeds `maxcall`.
        if ncall > maxcall:
            break

        it += 1

    # Add remaining live points to our set of dead points.
    # After N samples have been taken out, the remaining volume is
    # `e^(-N / nlive)`. Thus, the remaining volume for each live point
    # is `e^(-N / nlive) / nlive`. The log of this for each live point is:
    # `log(e^(-N / nlive) / nlive) = -N / nlive - log(nlive)`.
    logvol = -len(saved_v) / nlive - math.log(nlive)
    for i in xrange(nlive):
        ustar, vstar = live_u[i], live_v[i]
        loglstar = live_logl[i]
        logwt = logvol + loglstar
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * loglstar +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        saved_u.append(np.array(ustar))
        saved_v.append(np.array(vstar))
        saved_logl.append(loglstar)
        saved_logvol.append(logvol)
        saved_logwt.append(logwt)

    # h should always be nonnegative (we take the sqrt below).
    # Numerical error makes it negative in pathological corner cases
    # such as flat likelihoods. Here we correct those cases to zero.
    if h < 0.0:
        if h > -SQRTEPS:
            h = 0.0
        else:
            raise RuntimeError("Negative h encountered (h={}). Please report "
                               "this as a likely bug.".format(h))

    # Saving results.
    results = Results([('nlive', nlive),
                       ('niter', it + 1),
                       ('ncall', ncall),
                       ('logz', logz),
                       ('logzerr', math.sqrt(h / nlive)),
                       ('h', h),
                       ('samples_unit', np.array(saved_u)),
                       ('samples', np.array(saved_v)),
                       ('logwt', np.array(saved_logwt) - logz),
                       ('logvol', np.array(saved_logvol)),
                       ('logl', np.array(saved_logl))])

    return results
