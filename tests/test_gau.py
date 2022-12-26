import numpy as np
import pytest
from numpy import linalg
import numpy.testing as npt
import itertools
from utils import get_rstate, get_printing

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


def test_gaussian():
    sig = 4
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
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
    nerr = 3
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
        assert (np.abs(logz - g.logz_truth) < sig * results.logzerr[-1])
    res_comb = dyfunc.merge_runs([result_list[0]])
    res_comb = dyfunc.merge_runs(result_list)
    assert (np.abs(res_comb['logz'][-1] - g.logz_truth) <
            sig * results['logzerr'][-1])
    # check summary
    res = sampler.results
    res.summary()


def test_generator():
    # Test that we can use the sampler as a generator
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    for it in sampler.sample():
        pass
    for it in sampler.add_live_points():
        pass
    res = sampler.results
    check_results_gau(res, g, rstate)


def test_n_effective():
    # Test that n_effective controls the sample size
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    target_n_effective = 2

    current_n_effective = sampler.n_effective

    for _ in sampler.sample(add_live=False, n_effective=target_n_effective):
        previous_n_effective = current_n_effective
        current_n_effective = sampler.n_effective

    assert current_n_effective > target_n_effective > previous_n_effective


# try all combinations except
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


@pytest.mark.parametrize("bound,sample",
                         itertools.product(
                             ['single', 'multi', 'balls', 'cubes'], ['unif']))
def test_bounding_bootstrap(bound, sample):
    # check various bounding methods with bootstrap

    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    bootstrap=5,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, g, rstate)


def test_bounding_enlarge():
    # check if enlarging bound works

    rstate = get_rstate()
    g = Gaussian()
    bound = 'multi'
    sample = 'unif'
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    enlarge=1.5,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, g, rstate)


# extra checks for gradients
def test_hslice_nograd():
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    sample='hslice',
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, g, rstate)


@pytest.mark.parametrize("dyn", [False, True])
def test_hslice_grad(dyn):
    rstate = get_rstate()
    g = Gaussian()
    if dyn:
        CL = dynesty.DynamicNestedSampler
        kw = dict(dlogz_init=1, n_effective=100)
        # otherwise it's too slow
    else:
        CL = dynesty.NestedSampler
        kw = {}
    sampler = CL(g.loglikelihood,
                 g.prior_transform,
                 g.ndim,
                 nlive=nlive,
                 sample='hslice',
                 gradient=g.grad_x,
                 compute_jac=True,
                 rstate=rstate)
    sampler.run_nested(print_progress=printing, **kw)
    check_results_gau(sampler.results, g, rstate)


def test_hslice_grad1():
    rstate = get_rstate()
    g = Gaussian()
    sampler = dynesty.NestedSampler(g.loglikelihood,
                                    g.prior_transform,
                                    g.ndim,
                                    nlive=nlive,
                                    sample='hslice',
                                    gradient=g.grad_u,
                                    rstate=rstate)
    sampler.run_nested(print_progress=printing)
    check_results_gau(sampler.results, g, rstate)


def test_dynamic():
    # check dynamic nested sampling behavior
    rstate = get_rstate()
    g = Gaussian()
    dsampler = dynesty.DynamicNestedSampler(g.loglikelihood,
                                            g.prior_transform,
                                            g.ndim,
                                            rstate=rstate)
    dsampler.run_nested(print_progress=printing)
    # chechk explicit adding batches
    dsampler.add_batch(mode='auto')
    dsampler.add_batch(mode='weight')
    dsampler.add_batch(mode='full')
    dsampler.add_batch(logl_bounds=(-10, 0), mode='manual')
    dsampler.add_batch(logl_bounds=(-10000000, -1000), mode='manual')
    check_results_gau(dsampler.results, g, rstate)

    # check error analysis functions
    dres = dyfunc.jitter_run(dsampler.results, rstate=rstate)
    check_results_gau(dres, g, rstate)
    dres = dyfunc.resample_run(dsampler.results, rstate=rstate)
    check_results_gau(dres, g, rstate)

    dyfunc.kld_error(dsampler.results, rstate=rstate)


def test_ravel_unravel():
    """ Here I test that ravel/unravel preserves things correctly """
    rstate = get_rstate()
    g = Gaussian()

    dsampler = dynesty.DynamicNestedSampler(g.loglikelihood,
                                            g.prior_transform,
                                            g.ndim,
                                            bound='single',
                                            sample='unif',
                                            rstate=rstate,
                                            nlive=nlive)
    maxiter = 1800
    dsampler.run_nested(maxiter=maxiter,
                        use_stop=False,
                        nlive_batch=100,
                        print_progress=printing)
    dres = dsampler.results

    dres_list = dyfunc.unravel_run(dres)
    dres_merge = dyfunc.merge_runs(dres_list)
    assert np.abs(dres['logz'][-1] - dres_merge['logz'][-1]) < 0.01
