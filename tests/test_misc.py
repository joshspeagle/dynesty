import numpy as np
import dynesty
"""
Run a series of basic tests changing various things like
maxcall options and potentially other things
"""

nlive = 100

size = 10  # box size


def loglike(x):
    return -0.5 * np.sum(x**2)


def loglike_inf(x):
    r2 = np.sum(x**2)
    if r2 > size * size or np.abs(x[0]) < 0.1:
        return -np.inf
    return -0.5 * r2


def prior_transform(x):
    return (2 * x - 1) * size


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


def test_inf():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    sampler = dynesty.NestedSampler(loglike_inf,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive)
    sampler.run_nested()

    sampler = dynesty.DynamicNestedSampler(loglike_inf,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive)
    sampler.run_nested(dlogz_init=1)
