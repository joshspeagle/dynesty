import numpy as np
import dynesty
"""
Run a series of basic tests of the 2d eggbox
"""

# seed the random number generator
np.random.seed(56432)

nlive = 100


def loglike(x):
    return -0.5 * np.sum(x**2)


def prior_transform(x):
    return (2 * x - 1) * 10


def test_maxcall():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive)
    sampler.run_nested(dlogz=1, maxcall=1000)

    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive)
    sampler.run_nested(dlogz_init=1, maxcall=1000)
