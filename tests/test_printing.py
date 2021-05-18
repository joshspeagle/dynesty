import numpy as np
import dynesty
import pytest
"""
Run a series of basic tests testing printing output
"""

nlive = 100
printing = True


@pytest.fixture
def set_seed():
    # seed the random number generator
    np.random.seed(56432)


def loglike(x):
    return -0.5 * np.sum(x**2)


def prior_transform(x):
    return (2 * x - 1) * 10


def test_printing():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    sampler = dynesty.DynamicNestedSampler(
        loglike,
        prior_transform,
        ndim,
        nlive=nlive,
    )
    sampler.run_nested(dlogz_init=1, print_progress=printing)
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive)
    sampler.run_nested(dlogz=1, print_progress=printing)
