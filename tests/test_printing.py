import numpy as np
import dynesty
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


def test_printing():
    # hard test of dynamic sampler with high dlogz_init and small number
    # of live points
    ndim = 2
    rstate = get_rstate()
    sampler = dynesty.DynamicNestedSampler(loglike,
                                           prior_transform,
                                           ndim,
                                           nlive=nlive,
                                           rstate=rstate)
    sampler.run_nested(dlogz_init=1, print_progress=printing)
    sampler = dynesty.NestedSampler(loglike,
                                    prior_transform,
                                    ndim,
                                    nlive=nlive,
                                    rstate=rstate)
    sampler.run_nested(dlogz=1, print_progress=printing)
