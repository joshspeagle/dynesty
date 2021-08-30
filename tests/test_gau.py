import numpy as np
import pytest
from numpy import linalg
import numpy.testing as npt
import itertools
from utils import get_rstate
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


def bootstrap_tol(results, rstate):
    """ Compute the uncertainty of means/covs by doing bootstrapping """
    n = len(results.logz)
    niter = 50
    pos = results.samples
    wts = np.exp(results.logwt - results.logz[-1])
    means = []
    covs = []

    for i in range(niter):
        # curpos = dyfunc.resample_equal(pos, wts)
        # xid = np.random.randint(len(curpos), size=len(curpos))
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
prior_win = 10  # +/- 10 on both sides
logz_truth_gau = ndim_gau * (-np.log(2 * prior_win))


def check_results_gau(results, rstate, sig=5, logz_tol=None):
    if logz_tol is None:
        logz_tol = sig * results.logzerr[-1]
    mean_tol, cov_tol = bootstrap_tol(results, rstate)
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
    return prior_win * (2. * u - 1.)


# gradient (no jacobian)
def grad_x_gau(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(cov_inv_gau, (x - mean_gau))


# gradient (with jacobian)
def grad_u_gau(x):
    """Multivariate normal log-likelihood gradient."""
    return -np.dot(cov_inv_gau, x - mean_gau) * 2 * prior_win


def test_gaussian():
    sig = 5
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    # check that jitter/resample/simulate_run work
    # for not dynamic sampler
    dyfunc.jitter_run(sampler.results, rstate=rstate)
    dyfunc.resample_run(sampler.results, rstate=rstate)
    dyfunc.simulate_run(sampler.results, rstate=rstate)

    # add samples
    # check continuation behavior
    sampler.run_nested(dlogz=0.1, print_progress=printing)

    # get errors
    nerr = 3
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
        assert (np.abs(logz - logz_truth_gau) < sig * results.logzerr[-1])
    res_comb = dyfunc.merge_runs(result_list)
    assert (np.abs(res_comb.logz[-1] - logz_truth_gau) <
            sig * results.logzerr[-1])
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


# try all combinations excepte none/unif
@pytest.mark.parametrize(
    "bound,sample",
    list(
        itertools.product(['single', 'multi', 'balls', 'cubes'],
                          ['unif', 'rwalk', 'slice', 'rslice'])) +
    list(itertools.product(['none'], ['rwalk', 'slice', 'rslice'])))
def test_bounding_sample(bound, sample):
    # check various bounding methods

    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, rstate)


@pytest.mark.parametrize("bound,sample",
                         itertools.product(
                             ['single', 'multi', 'balls', 'cubes'], ['unif']))
def test_bounding_bootstrap(bound, sample):
    # check various bounding methods

    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    bootstrap=5,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, rstate)


# extra checks for gradients
def test_slice_nograd():
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, rstate)


def test_slice_grad():
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=grad_x_gau,
                                    compute_jac=True,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, rstate)


def test_slice_grad1():
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglikelihood_gau,
                                    prior_transform_gau,
                                    ndim_gau,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=grad_u_gau,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, rstate)


def test_dynamic():
    # check dynamic nested sampling behavior
    rstate = get_rstate()
    dsampler = dynesty.DynamicNestedSampler(loglikelihood_gau,
                                            prior_transform_gau,
                                            ndim_gau,
                                            rstate=rstate)
    dsampler.run_nested(print_progress=printing)
    check_results_gau(dsampler.results, rstate)

    # check error analysis functions
    dres = dyfunc.jitter_run(dsampler.results, rstate=rstate)
    check_results_gau(dres, rstate)
    dres = dyfunc.resample_run(dsampler.results, rstate=rstate)
    check_results_gau(dres, rstate)
    dres = dyfunc.simulate_run(dsampler.results, rstate=rstate)
    check_results_gau(dres, rstate)

    dyfunc.kld_error(dsampler.results, rstate=rstate)
