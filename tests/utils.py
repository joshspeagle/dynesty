import numpy as np
import numpy.testing as npt
import os
import dynesty.utils as dyfunc
'''
Here we setup a common seed for all the tests
But we also allow to set the seed through DYNESTY_TEST_RANDOMSEED
environment variable.
That allows to run long tests by looping over seed value to catch
potentially rare behaviour
'''


def get_rstate(seed=None):
    if seed is None:
        kw = 'DYNESTY_TEST_RANDOMSEED'
        if kw in os.environ:
            seed = int(os.environ[kw])
        else:
            seed = 56432
        # seed the random number generator
    return np.random.default_rng(seed)


def get_printing():
    kw = 'DYNESTY_TEST_PRINTING'
    if kw in os.environ:
        return int(os.environ[kw])
    else:
        return False


class NullContextManager(object):
    # https://stackoverflow.com/questions/45187286/how-do-i-write-a-null-no-op-contextmanager-in-python
    # this is to make it work for 3.6
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


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


def check_results_gau(results, g, rstate, sig=4, logz_tol=None):
    """ Check the results of sampling a Gaussian against the truth """
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
