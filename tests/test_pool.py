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

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


def test_pool():
    # test pool
    ndim = 2
    pool = mp.Pool(2)
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    pool=pool,
                                    queue_size=2,
                                    rstate=rstate)
    sampler.run_nested(dlogz=0.1, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


def test_pool_dynamic():
    # test pool
    ndim = 2
    pool = mp.Pool(2)
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike_egg,
                                           prior_transform_egg,
                                           ndim,
                                           nlive=nlive,
                                           bound='multi',
                                           sample='unif',
                                           pool=pool,
                                           queue_size=2,
                                           rstate=rstate)
    sampler.run_nested(print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


size = 3


def loglike_gau(x):
    return -0.5 * np.sum(x**2)


def prior_transform_gau(x):
    return (2 * x - 1) * size


@pytest.mark.parametrize(
    'func',
    ['prior_transform', 'loglikelihood', 'propose_point', 'update_bound'])
def test_usepool(func):
    ndim = 2
    rstate = get_rstate()
    pool = mp.Pool(2)

    use_pool = {func: True}
    sampler = dynesty.DynamicNestedSampler(loglike_gau,
                                           prior_transform_gau,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate,
                                           use_pool=use_pool,
                                           pool=pool,
                                           queue_size=20)
    sampler.run_nested(maxiter=10000)
