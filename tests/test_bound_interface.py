import numpy as np
import pytest
from numpy import linalg
import numpy.testing as npt
import itertools
from utils import get_rstate, get_printing

import dynesty  # noqa
import dynesty.bounding as db
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 500
printing = get_printing()


class Box(db.Bound):
    """                                                                                                                               Dummy bounding class
    """

    def __init__(self, ndim):
        super().__init__(ndim)
        self.logvol = 0
        self.cen = np.zeros(ndim) + 0.5
        self.size = 0.5

    def contains(self, x):
        return (np.abs(x - self.cen) < self.size).all()

    def sample(self, rstate=None):
        return rstate.uniform(np.maximum(self.cen - self.size, 0),
                              np.minimum(self.cen + self.size, 1))

    def samples(self, nsamples, rstate=None):
        return np.array([self.sample(rstate) for i in range(nsamples)])

    def get_random_axes(self, rstate):
        return np.eye(self.ndim) * self.size

    def scale_to_logvol(self, logvol):
        self.size = np.exp(logvol / self.ndim)

    def update(self, points, rstate=None, bootstrap=0, pool=None):
        self.cen = points.mean(axis=0)
        expand = 2
        self.size = np.abs(points - self.cen).max() * expand
        self.logvol = np.log(self.size) * self.ndim


def bootstrap_tol(results, rstate):
    """ Compute the uncertainty of means/covs by doing bootstrapping """
    n = len(results['logz'])
    niter = 50
    pos = results.samples
    wts = results.importance_weights()
    means = []
    covs = []

    for i in range(niter):
        sub = rstate.uniform(size=n) < wts / wts.max()
        ind0 = np.nonzero(sub)[0]
        ind1 = rstate.choice(ind0, size=len(ind0), replace=True)
        mean = pos[ind1].mean(axis=0)
        cov = np.cov(pos[ind1].T)
        means.append(mean)
        covs.append(cov)
    return np.std(means, axis=0), np.std(covs, axis=0)


def check_results(results,
                  mean_truth,
                  cov_truth,
                  logz_truth,
                  mean_tol,
                  cov_tol,
                  logz_tol,
                  sig=4):
    """ Check if means and covariances match match expectations
    within the tolerances

    """
    results.summary()
    pos = results.samples
    wts = np.exp(results['logwt'] - results['logz'][-1])
    assert np.allclose(results.importance_weights(), wts)
    mean, cov = dyfunc.mean_and_cov(pos, wts)
    logz = results['logz'][-1]
    logzerr = results['logzerr'][-1]
    assert logzerr < 10  # check that it is not too large
    npt.assert_array_less(np.abs(mean - mean_truth), sig * mean_tol)
    npt.assert_array_less(np.abs(cov - cov_truth), sig * cov_tol)
    npt.assert_array_less(np.abs((logz_truth - logz)), sig * logz_tol)


# GAUSSIAN TEST


class Gaussian:

    def __init__(self, corr=.95, prior_win=10):
        self.ndim = 3
        self.mean = np.linspace(-1, 1, self.ndim)
        self.cov = np.identity(self.ndim)  # set covariance to identity matrix
        self.cov[self.cov ==
                 0] = corr  # set off-diagonal terms (strongly correlated)
        self.cov_inv = linalg.inv(self.cov)  # precision matrix
        self.lnorm = -0.5 * (np.log(2 * np.pi) * self.ndim +
                             np.log(linalg.det(self.cov)))
        self.prior_win = prior_win  # +/- on both sides
        self.logz_truth = self.ndim * (-np.log(2 * self.prior_win))

    # 3-D correlated multivariate normal log-likelihood
    def loglikelihood(self, x):
        """Multivariate normal log-likelihood."""

        ret = -0.5 * np.dot(
            (x - self.mean), np.dot(self.cov_inv,
                                    (x - self.mean))) + self.lnorm
        # notice here we overwrite the input array just to test
        # that this is not a problem
        x[:] = -np.ones(len(x))
        return ret

    # prior transform
    def prior_transform(self, u):
        """Flat prior between -10. and 10."""
        ret = self.prior_win * (2. * u - 1.)

        # notice here we overwrite the input array just to test
        # that this is not a problem
        u[:] = -np.ones(len(u))

        return ret


def check_results_gau(results, g, rstate, sig=4, logz_tol=None):
    if logz_tol is None:
        logz_tol = sig * results['logzerr'][-1]
    mean_tol, cov_tol = bootstrap_tol(results, rstate)
    # just check that resample_equal works
    dyfunc.resample_equal(results.samples,
                          np.exp(results['logwt'] - results['logz'][-1]))
    results.samples_equal()
    check_results(results,
                  g.mean,
                  g.cov,
                  g.logz_truth,
                  mean_tol,
                  cov_tol,
                  logz_tol,
                  sig=sig)


# try all combinations except
@pytest.mark.parametrize(
    "bound,sample",
    list(
        itertools.product(['single', 'multi', 'balls', 'cubes', 'none', 'box'],
                          ['unif', 'rwalk', 'slice', 'rslice'])))
def test_bounding_sample(bound, sample):
    # check various bounding methods

    rstate = get_rstate()
    if bound == 'none':
        if sample != 'unif':
            g = Gaussian(0.1)
        else:
            g = Gaussian(corr=0., prior_win=10)
            # make live easy if bound is none
            # but also not too easy so propose_point() is exercised
    else:
        g = Gaussian()
    ndim = g.ndim
    bound = {
        'none': db.UnitCube(ndim),
        'multi': db.MultiEllipsoid(ndim),
        'single': db.Ellipsoid(ndim),
        'balls': db.RadFriends(ndim),
        'cubes': db.SupFriends(ndim),
        'box': Box(ndim)
    }[bound]
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, g, rstate)
    print(sampler.citations)
