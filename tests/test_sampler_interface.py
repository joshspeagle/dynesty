import numpy as np
import pytest
from numpy import linalg
import numpy.testing as npt
import itertools

from utils import get_rstate, get_printing
import dynesty.sampling as dysa
import dynesty  # noqa
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 500
printing = get_printing()


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


class TestSampler(dysa.InternalSampler):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_sampler(self,
                        loglstar=None,
                        points=None,
                        axes=None,
                        seeds=None,
                        prior_transform=None,
                        loglikelihood=None,
                        nested_sampler=None):
        """

        """
        pass

    @staticmethod
    def sample(args):
        """

        """

        pass


class Gaussian:

    def __init__(self, corr=.95, prior_win=10, ndim=3):
        self.ndim = ndim
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

    # gradient (no jacobian)
    def grad_x(self, x):
        """Multivariate normal log-likelihood gradient."""
        return -np.dot(self.cov_inv, (x - self.mean))

    # gradient (with jacobian)
    def grad_u(self, x):
        """Multivariate normal log-likelihood gradient."""
        return -np.dot(self.cov_inv, x - self.mean) * 2 * self.prior_win


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


# try all combinations
@pytest.mark.parametrize(
    "bound,sample",
    list(
        itertools.product(['single', 'multi', 'balls', 'cubes', 'none'],
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
    sample = {
        'unif': dysa.UniformBoundSampler(),
        'rslice': dysa.RSliceSampler(),
        'rwalk': dysa.RWalkSampler(ncdim=g.ndim),
        'slice': dysa.SliceSampler()
    }[sample]
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


# try all combinations
@pytest.mark.parametrize("sample", ['rwalk', 'slice', 'rslice'])
@pytest.mark.parametrize("typ", [0, 1])
def test_walks_slices(sample, typ):
    # This tests if the walks= slices= options are used

    bound = 'single'
    g = Gaussian(0.1, ndim=2)
    res = []
    for i in range(2):
        # number of steps loop
        steps = {0: 10, 1: 20}[i]
        rstate = get_rstate()
        if typ == 0:
            cur_sample = {
                'rwalk': dysa.RWalkSampler(walks=steps),
                'slice': dysa.SliceSampler(slices=steps),
                'rslice': dysa.RSliceSampler(slices=steps),
            }[sample]
            kw = {}
        else:
            cur_sample = sample
            kw = {
                'rwalk': dict(walks=steps),
                'slice': dict(slices=steps),
                'rslice': dict(slices=steps),
            }[sample]
        sampler = dynesty.NestedSampler(g.loglikelihood,
                                        g.prior_transform,
                                        g.ndim,
                                        nlive=nlive,
                                        bound=bound,
                                        sample=cur_sample,
                                        rstate=rstate,
                                        **kw)
        sampler.run_nested(print_progress=printing)
        res.append(sampler.ncall)

    assert (res[1] > res[0])
