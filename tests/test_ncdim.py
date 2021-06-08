from __future__ import (print_function, division)
from six.moves import range
import numpy as np
from numpy import linalg
import numpy.testing as npt
import matplotlib

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
ntotdim = 5
mean_gau = np.linspace(-1, 1, ndim_gau)
cov_gau = np.identity(ndim_gau)  # set covariance to identity matrix
cov_gau[cov_gau == 0] = 0.95  # set off-diagonal terms (strongly correlated)
cov_inv_gau = linalg.inv(cov_gau)  # precision matrix
lnorm_gau = -0.5 * (np.log(2 * np.pi) * ndim_gau + np.log(linalg.det(cov_gau)))
prior_win = 10  # +/- 10 on both sides
logz_truth_gau = ndim_gau * (-np.log(2 * prior_win))

mean_vec = np.concatenate((mean_gau, np.zeros(ntotdim - ndim_gau)))
cov_true = np.zeros((ntotdim, ntotdim))
cov_true[:ndim_gau, :ndim_gau] = cov_gau
cov_true[-(ntotdim - ndim_gau):,
         -(ntotdim - ndim_gau):] = np.eye(ntotdim -
                                          ndim_gau) * prior_win**2 / 3


def check_results_gau(results, logz_tol, sig=5):
    mean_tol, cov_tol = bootstrap_tol(results)
    check_results(results,
                  mean_vec,
                  cov_true,
                  logz_truth_gau,
                  mean_tol,
                  cov_tol,
                  logz_tol,
                  sig=sig)


# 3-D correlated multivariate normal log-likelihood
def loglikelihood_gau(x0):
    """Multivariate normal log-likelihood."""
    x = x0[:ndim_gau]
    return -0.5 * np.dot((x - mean_gau), np.dot(cov_inv_gau,
                                                (x - mean_gau))) + lnorm_gau


# prior transform
def prior_transform_gau(u):
    """Flat prior between -10. and 10."""
    return prior_win * (2. * u - 1.)


def test_gaussian():
    logz_tol = 1
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ntotdim,
                                    nlive=nlive,
                                    ncdim=ndim_gau)
    sampler.run_nested(print_progress=printing)
    # check that jitter/resample/simulate_run work
    # for not dynamic sampler
    dyfunc.jitter_run(sampler.results)
    dyfunc.resample_run(sampler.results)
    dyfunc.simulate_run(sampler.results)

    # add samples
    # check continuation behavior
    sampler.run_nested(dlogz=0.1, print_progress=printing)

    # get errors
    nerr = 2
    result_list = []
    for i in range(nerr):
        sampler.reset()
        sampler.run_nested(print_progress=False)
        results = sampler.results
        result_list.append(results)
        pos = results.samples
        wts = np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(pos, wts)
        logz = results.logz[-1]
        assert (np.abs(logz - logz_truth_gau) < logz_tol)
    res_comb = dyfunc.merge_runs(result_list)
    assert (np.abs(res_comb.logz[-1] - logz_truth_gau) < logz_tol)
    # check summary
    res = sampler.results
    res.summary()


def test_dynamic():
    # check dynamic nested sampling behavior
    logz_tol = 1
    dsampler = dynesty.DynamicNestedSampler(loglikelihood_gau,
                                            prior_transform_gau,
                                            ntotdim,
                                            ncdim=ndim_gau)
    dsampler.run_nested(print_progress=printing)
    check_results_gau(dsampler.results, logz_tol)
