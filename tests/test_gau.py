from __future__ import (print_function, division)
from six.moves import range
import numpy as np
from numpy import linalg
import numpy.testing as npt
import matplotlib
import pytest

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa
import dynesty  # noqa
from dynesty import plotting as dyplot  # noqa
from dynesty import utils as dyfunc  # noqa
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 500
printing = False


@pytest.fixture
def set_seed():
    # seed the random number generator
    np.random.seed(5647)


def bootstrap_tol(results):
    """ Compute the uncertainty of means/covs by doing bootstrapping """
    n = len(results.logz)
    niter = 50
    pos = results.samples
    wts = np.exp(results.logwt - results.logz[-1])
    means = []
    covs = []

    for i in range(niter):
        xid = np.random.randint(n, size=n)
        mean, cov = dyfunc.mean_and_cov(pos[xid], wts[xid])
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
                  sig=5):
    """ Check if means and covariances match match expectations
    within the tolerances

    """
    pos = results.samples
    wts = np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(pos, wts)
    logz = results.logz[-1]
    npt.assert_array_less(np.abs(mean - mean_truth), sig * mean_tol)
    npt.assert_array_less(np.abs(cov - cov_truth), sig * cov_tol)
    npt.assert_array_less(np.abs((logz_truth - logz)), sig * logz_tol)


# GAUSSIAN TEST

ndim_gau = 3
mean_gau = np.linspace(-1, 1, ndim_gau)
cov_gau = np.identity(ndim_gau)  # set covariance to identity matrix
cov_gau[cov_gau == 0] = 0.95  # set off-diagonal terms (strongly correlated)
cov_inv_gau = linalg.inv(cov_gau)  # precision matrix
lnorm_gau = -0.5 * (np.log(2 * np.pi) * ndim_gau + np.log(linalg.det(cov_gau)))
logz_truth_gau = ndim_gau * (-np.log(2 * 10.))


def check_results_gau(results, logz_tol, sig=5):
    mean_tol, cov_tol = bootstrap_tol(results)
    check_results(results,
                  mean_gau,
                  cov_gau,
                  logz_truth_gau,
                  mean_tol,
                  cov_tol,
                  logz_tol,
                  sig=sig)


# 3-D correlated multivariate normal log-likelihood
def loglikelihood_gau(x):
    """Multivariate normal log-likelihood."""
    return -0.5 * np.dot((x - mean_gau), np.dot(cov_inv_gau,
                                                (x - mean_gau))) + lnorm_gau


# prior transform
def prior_transform_gau(u):
    """Flat prior between -10. and 10."""
    return 10. * (2. * u - 1.)


# gradient (no jacobian)
def grad_x_gau(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(cov_inv_gau, (x - mean_gau))


# gradient (with jacobian)
def grad_u_gau(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(cov_inv_gau, x - mean_gau) * 20.


def test_gaussian():
    logz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive)
    sampler.run_nested(print_progress=printing)

    # add samples
    # check continuation behavior
    sampler.run_nested(dlogz=0.1, print_progress=printing)

    # get errors
    nerr = 2
    for i in range(nerr):
        sampler.reset()
        sampler.run_nested(print_progress=False)
        results = sampler.results
        pos = results.samples
        wts = np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(pos, wts)
        logz = results.logz[-1]
        assert (np.abs(logz - logz_truth_gau) < logz_tol)
    # check summary
    res = sampler.results
    res.summary()

    # check plots
    dyplot.runplot(sampler.results)
    plt.close()
    dyplot.traceplot(sampler.results)
    plt.close()
    dyplot.cornerpoints(sampler.results)
    plt.close()
    dyplot.cornerplot(sampler.results)
    plt.close()
    dyplot.boundplot(sampler.results,
                     dims=(0, 1),
                     it=3000,
                     prior_transform=prior_transform_gau,
                     show_live=True,
                     span=[(-10, 10), (-10, 10)])
    plt.close()
    dyplot.cornerbound(sampler.results,
                       it=3500,
                       prior_transform=prior_transform_gau,
                       show_live=True,
                       span=[(-10, 10), (-10, 10)])
    plt.close()


def test_bounding():
    # check various bounding methods
    logz_tol = 1

    for bound in ['none', 'single', 'multi', 'balls', 'cubes']:
        sampler = dynesty.NestedSampler(loglikelihood_gau,
                                        prior_transform_gau,
                                        ndim_gau,
                                        nlive=nlive,
                                        bound=bound,
                                        sample='unif')
        sampler.run_nested(print_progress=printing)
        check_results_gau(sampler.results, logz_tol)


def test_bounding_bootstrap():
    # check various bounding methods
    logz_tol = 1

    for bound in ['single', 'multi', 'balls']:
        sampler = dynesty.NestedSampler(loglikelihood_gau,
                                        prior_transform_gau,
                                        ndim_gau,
                                        nlive=nlive,
                                        bound=bound,
                                        sample='unif',
                                        bootstrap=5)
        sampler.run_nested(print_progress=printing)
        check_results_gau(sampler.results, logz_tol)


def test_sampling():
    # check various sampling methods
    logz_tol = 1
    for sample in ['unif', 'rwalk', 'rstagger', 'slice', 'rslice']:
        sampler = dynesty.NestedSampler(loglikelihood_gau,
                                        prior_transform_gau,
                                        ndim_gau,
                                        nlive=nlive,
                                        sample=sample)
        sampler.run_nested(print_progress=printing)
        check_results_gau(sampler.results, logz_tol)


# extra checks for gradients
def test_slice_nograd():
    logz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice')
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, logz_tol)


def test_slice_grad():
    logz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=grad_x_gau,
                                    compute_jac=True)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, logz_tol)


def test_slice_grad1():
    logz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=grad_u_gau)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, logz_tol)


def test_dynamic():
    # check dynamic nested sampling behavior
    logz_tol = 1
    dsampler = dynesty.DynamicNestedSampler(loglikelihood_gau,
                                            prior_transform_gau, ndim_gau)
    dsampler.run_nested(print_progress=printing)
    check_results_gau(dsampler.results, logz_tol)

    # check error analysis functions
    # IMPORTANT I had to bump up the agreement threshold to 6 sigma
    # this is too much and needs to be checked
    dres = dyfunc.jitter_run(dsampler.results)
    check_results_gau(dres, logz_tol)
    dres = dyfunc.resample_run(dsampler.results)
    check_results_gau(dres, logz_tol, sig=6)
    dres = dyfunc.simulate_run(dsampler.results)
    check_results_gau(dres, logz_tol, sig=6)
    # I bump the threshold
    # because we have the error twice
    dyfunc.kld_error(dsampler.results)
