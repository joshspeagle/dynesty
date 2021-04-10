import numpy as np
import dynesty
import multiprocessing as mp
"""
Run a series of basic tests to check whether anything huge is broken.

"""

# seed the random number generator
np.random.seed(5647)

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
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    pool=pool,
                                    queue_size=2)
    sampler.run_nested(dlogz=0.1, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


def test_pool2():
    # test pool
    ndim = 2
    pool = mp.Pool(2)
    sampler = dynesty.DynamicNestedSampler(loglike_egg,
                                           prior_transform_egg,
                                           ndim,
                                           nlive=nlive,
                                           bound='multi',
                                           sample='unif',
                                           pool=pool,
                                           queue_size=2)
    sampler.run_nested(dlogz_init=0.1, print_progress=printing)
    logz_truth = 235.856
    assert (abs(logz_truth - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])
