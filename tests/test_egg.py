import numpy as np
import dynesty
import pytest
import itertools
from utils import get_rstate, get_printing
"""
Run a series of basic tests of the 2d eggbox
"""

nlive = 1000
printing = get_printing()

# EGGBOX


# see 1306.2144
def loglike_egg(x):
    logl = ((2 + np.cos(x[0] / 2) * np.cos(x[1] / 2))**5)
    return logl


def prior_transform_egg(x):
    return x * 10 * np.pi


LOGZ_TRUTH = 235.856


@pytest.mark.parametrize(
    "bound,sample",
    itertools.product(['multi', 'balls', 'cubes'],
                      ['unif', 'rwalk', 'slice', 'rslice']))
def test_bounds(bound, sample):
    # stress test various boundaries
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound=bound,
                                    sample=sample,
                                    rstate=rstate)
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    assert (abs(LOGZ_TRUTH - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])


def test_ellipsoids_bootstrap():
    # stress test ellipsoid decompositions with bootstrap
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.NestedSampler(loglike_egg,
                                    prior_transform_egg,
                                    ndim,
                                    nlive=nlive,
                                    bound='multi',
                                    sample='unif',
                                    bootstrap=5,
                                    rstate=rstate)
    sampler.run_nested(dlogz=0.01, print_progress=printing)
    assert (abs(LOGZ_TRUTH - sampler.results.logz[-1]) <
            5. * sampler.results.logzerr[-1])
