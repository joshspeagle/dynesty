import numpy as np
import pytest
import dynesty
import multiprocessing as mp
from utils import get_rstate
"""
Run a series of basic tests to check whether anything huge is broken.

"""

nlive = 1000
printing = False

ndim = 2
gau_s = 0.01


def loglike_gau(x):
    return (-0.5 * np.log(2 * np.pi) * ndim - np.log(gau_s) * ndim -
            0.5 * np.sum((x - 0.5)**2) / gau_s**2)


def prior_transform_gau(x):
    return x


# EGGBOX
# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


LOGZ_TRUTH_GAU = 0
LOGZ_TRUTH_EGG = 235.856


def test_pool():
    # test pool on egg problem
    pool = mp.Pool(2)
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    pool=pool,
                                    queue_size=2,
                                    rstate=rstate)
    sampler.run_nested(dlogz=0.1, print_progress=printing)
    assert (abs(LOGZ_TRUTH_EGG - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


def test_pool_dynamic():
    # test pool in dynamic mode
    # here for speed I do a gaussian
    pool = mp.Pool(2)
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           ndim,
                                           nlive=nlive,
                                           pool=pool,
                                           queue_size=2,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing)
    assert (abs(LOGZ_TRUTH_GAU - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


POOL_KW = ['prior_transform', 'loglikelihood', 'propose_point', 'update_bound']


@pytest.mark.parametrize('func', POOL_KW)
def test_usepool(func):
    # test all the use_pool options, toggle them one by one
    rstate = get_rstate()
    pool = mp.Pool(2)
    use_pool = {}
    for k in POOL_KW:
        use_pool[k] = False
    use_pool[func] = True

    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate,
                                           use_pool=use_pool,
                                           pool=pool,
                                           queue_size=2)
    sampler.run_nested(maxiter=10000)
