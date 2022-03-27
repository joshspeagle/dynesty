import numpy as np
import pytest
from utils import get_rstate
"""
Run a series of basic tests testing printing output
"""

nlive = 100
printing = True


def loglike(x):
    return -0.5 * np.sum(x**2)


def prior_transform(x):
    return (2 * x - 1) * 10


@pytest.mark.parametrize('withtqdm', [False, True])
def test_printing(withtqdm):
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    if withtqdm:
        import dynesty.utils
    else:
        import dynesty.utils
        import tqdm
        dynesty.utils.tqdm = None
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing, maxiter=1000)
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(dlogz=1, print_progress=printing, maxiter=1000)
    if withtqdm:
        pass
    else:
        dynesty.utils.tqdm = tqdm


def prior_transform_wide(x):
    return (2 * x - 1) * 10000


def test_print_large():
    # test with very large logl values
    import dynesty
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform_wide,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(print_progress=printing, maxiter=1000)
