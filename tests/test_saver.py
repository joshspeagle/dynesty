import numpy as np
import dynesty
import os
import multiprocessing as mp
import pytest
"""
Run a series of basic tests to check whether saving likelikelihood evals 
are broken

"""

nlive = 100
printing = False


@pytest.fixture
def set_seed():
    # seed the random number generator
    np.random.seed(5647)


# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


def test_saving():
    # test saving
    ndim = 2
    fname = 'xx.h5'
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    save_history=True,
                                    history_filename=fname)
    sampler.run_nested(dlogz=1, print_progress=printing)
    assert (os.path.exists(fname))


def test_saving_pool():
    # test saving
    ndim = 2
    fname = 'xx1.h5'
    pool = mp.Pool(2)
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    save_history=True,
                                    history_filename=fname,
                                    pool=pool,
                                    queue_size=2)
    sampler.run_nested(dlogz=1, print_progress=printing)
    assert (os.path.exists(fname))
