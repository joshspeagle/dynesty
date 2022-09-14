import numpy as np
from numpy import linalg
import numpy.testing as npt
import dynesty
import pytest
import itertools
from dynesty import utils as dyfunc
from utils import get_rstate, get_printing
"""
A rudimentary test that ncdim parameter works
"""

nlive = 500
printing = get_printing


def bootstrap_tol(results, rstate):
    """ Compute the uncertainty of means/covs by doing bootstrapping """
    n = len(results['logz'])
    niter = 50
    pos = results.samples
    wts = results.importance_weights()
    means = []
    covs = []

    for i in range(niter):
        xid = rstate.integers(n, size=n)
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
    wts = results.importance_weights()
    mean, cov = dyfunc.mean_and_cov(pos, wts)
    logz = results['logz'][-1]
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


def check_results_gau(results, rstate, logz_tol, sig=5):
    mean_tol, cov_tol = bootstrap_tol(results, rstate)
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
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ntotdim,
                                    nlive=nlive,
                                    ncdim=ndim_gau,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    # check that jitter/resample work
    # for not dynamic sampler
    dyfunc.jitter_run(sampler.results, rstate=rstate)
    dyfunc.resample_run(sampler.results, rstate=rstate)

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
        wts = results.importance_weights()
        mean, cov = dyfunc.mean_and_cov(pos, wts)
        logz = results['logz'][-1]
        assert (np.abs(logz - logz_truth_gau) < logz_tol)
    res_comb = dyfunc.merge_runs(result_list)
    assert (np.abs(res_comb['logz'][-1] - logz_truth_gau) < logz_tol)
    # check summary
    res = sampler.results
    res.summary()


def test_dynamic():
    # check dynamic nested sampling behavior
    logz_tol = 1
    rstate = get_rstate()
    dsampler = dynesty.DynamicNestedSampler(loglikelihood_gau,
                                            prior_transform_gau,
                                            ntotdim,
                                            ncdim=ndim_gau,
                                            rstate=rstate)
    dsampler.run_nested(print_progress=printing)
    check_results_gau(dsampler.results, rstate, logz_tol)


@pytest.mark.parametrize('bound,periodic',
                         itertools.product(['single', 'multi'], [False, True]))
def test_single_periodic(bound, periodic):
    # check single/multi ellipse bound with and without periodic vars
    logz_tol = 1
    rstate = get_rstate()
    if periodic:
        periodic = [0]
    else:
        periodic = None
    dsampler = dynesty.NestedSampler(loglikelihood_gau,
                                     prior_transform_gau,
                                     ntotdim,
                                     ncdim=ndim_gau,
                                     periodic=periodic,
                                     bound=bound,
                                     rstate=rstate)
    dsampler.run_nested(print_progress=printing)
    check_results_gau(dsampler.results, rstate, logz_tol)
